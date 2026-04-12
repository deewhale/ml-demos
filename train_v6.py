import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

"""
V6 Agent Training Script -- PPO 全决策训练（无 Solver）
=====================================================
Key design:
  1. Agent (v6 model) 处理所有决策：combat + strategy（draft, path, choice, rest, shop）
  2. 无 TurnSolver — 所有战斗由 AI combat head 决策
  3. PPO 训练全部 transition（combat + strategy）
  4. 统一效果编码（CARD_DATA/RELIC_DATA/POTION_DATA 查表）

Usage:
  cd STSRLSOLVER_PATH
  uv run python3 REPO_ROOT/train_v6.py --from-scratch --test
  uv run python3 REPO_ROOT/train_v6.py --n-games 500 --workers 4
"""

import sys
import time
import random
import argparse
import numpy as np
import torch
from pathlib import Path
from multiprocessing import Pool

sys.path.insert(0, "STSRLSOLVER_PATH")
sys.path.insert(0, "REPO_ROOT")

from packages.engine.game import GameRunner, GamePhase
from packages.engine.content.cards import ALL_CARDS, CardType, CardRarity, CardTarget

import sts_agent_v6 as v6

BASE_DIR = Path(__file__).parent
STATS_LOG = BASE_DIR / "sts_v6_stats.log"
CHECKPOINT_DIR = BASE_DIR / "sts_models"

# 训练超参
N_GAMES_TOTAL = 1
N_RUNS_PER_UPDATE = 16
VALIDATION_INTERVAL = 50
VALIDATION_GAMES = 10

# Combat reward shaping — 逐卡即时反馈（主要逻辑在 agent.combat_act 内）
# train_v6 只处理 END TURN 后的 HP loss penalty 和 combat 结束奖励
COMBAT_WIN_REWARD = 5.0         # 赢得战斗
COMBAT_LOSE_PENALTY = -2.0      # 输掉战斗
HP_LOSS_PENALTY_SCALE = 0.1     # 敌人回合 HP 损失的惩罚系数


# ============================================================
# Room-type to path symbol mapping
# ============================================================

ROOM_TYPE_TO_SYMBOL = {
    "MONSTER":  "M",
    "ELITE":    "E",
    "REST":     "R",
    "SHOP":     "$",
    "EVENT":    "?",
    "TREASURE": "T",
    "BOSS":     "B",
}


def _enumerate_path_to_boss(start_node, current_map):
    """从起始节点沿确定性路径走到 Boss 层，返回房间类型符号序列。"""
    path_rooms = []
    current = start_node
    visited_ys = set()

    while current is not None:
        rt = current.room_type.name if current.room_type else "MONSTER"
        sym = ROOM_TYPE_TO_SYMBOL.get(rt, "M")
        path_rooms.append(sym)
        visited_ys.add(current.y)

        if not current.edges or rt == "BOSS":
            break

        best_next = None
        for edge in current.edges:
            if edge.dst_y not in visited_ys:
                if edge.is_boss:
                    path_rooms.append("B")
                    best_next = None
                    break
                if edge.dst_y < len(current_map) and edge.dst_x < len(current_map[edge.dst_y]):
                    best_next = current_map[edge.dst_y][edge.dst_x]
                    break

        if best_next is None:
            break
        current = best_next

    return path_rooms


# ============================================================
# Card data bridge — 从 ALL_CARDS 构建 agent card dict
# ============================================================

def card_id_to_agent_dict(card_id: str, upgraded: bool = False) -> dict:
    """从 simulator card ID 构建 v6 agent 期望的 card dict。

    包含完整字段：id, name, type, cost, damage, block, rarity, exhaust, ethereal, upgraded
    """
    base_id = card_id.rstrip("+") if card_id.endswith("+") else card_id
    upgraded = upgraded or card_id.endswith("+")

    card_def = ALL_CARDS.get(base_id)
    if card_def is None:
        return {
            "id": card_id,
            "name": card_id,
            "type": "SKILL",
            "cost": 1,
            "damage": 0,
            "block": 0,
            "rarity": "COMMON",
            "upgraded": upgraded,
        }

    ctype = card_def.card_type.value if hasattr(card_def.card_type, "value") else str(card_def.card_type)
    rarity = card_def.rarity.value if hasattr(card_def.rarity, "value") else str(card_def.rarity)
    damage = max(card_def.base_damage, 0) if card_def.base_damage > 0 else 0
    block = max(card_def.base_block, 0) if card_def.base_block > 0 else 0
    if upgraded:
        if card_def.base_damage > 0:
            damage = max(card_def.base_damage + card_def.upgrade_damage, 0)
        if card_def.base_block > 0:
            block = max(card_def.base_block + card_def.upgrade_block, 0)

    cost = card_def.cost if card_def.upgrade_cost is None or not upgraded else card_def.upgrade_cost
    if cost < 0:
        cost = 0

    return {
        "id": base_id,
        "name": card_def.name,
        "type": ctype,
        "cost": cost,
        "damage": damage,
        "block": block,
        "rarity": rarity,
        "exhaust": card_def.exhaust,
        "ethereal": card_def.ethereal,
        "upgraded": upgraded,
    }


# ============================================================
# Simulator state -> agent state bridge（V6 增强版）
# ============================================================

def build_agent_state(runner: GameRunner) -> dict:
    """将 simulator 状态转换为 v6 agent 期望的 game_state 格式。

    V6 增强点（相对 V5）：
    - 敌人 powers 完整传递 (power_id, amount)
    - 玩家 powers 完整传递
    - intent_damage / intent_hits 精确传递
    - 遗物带 counter 信息
    """
    rs = runner.run_state
    phase = runner.phase
    actions = runner.get_available_actions()
    action_type_names = set()
    for a in actions:
        if hasattr(a, "action_type"):
            action_type_names.add(a.action_type)
        else:
            action_type_names.add(type(a).__name__.lower())

    # Deck
    deck_cards = []
    for c in (rs.deck or []):
        cid = getattr(c, "id", str(c))
        upgraded = getattr(c, "upgraded", False)
        deck_cards.append(card_id_to_agent_dict(cid, upgraded))

    # Relics（含 counter）
    relics = []
    for r in (rs.relics or []):
        rid = getattr(r, "id", str(r))
        counter = getattr(r, "counter", 0) or 0
        relics.append({"id": rid, "name": rid, "counter": counter})

    # Potions
    potions = []
    for slot in getattr(rs, "potion_slots", []):
        pid = getattr(slot, "potion_id", None)
        if pid:
            potions.append({"id": pid, "name": pid})
        else:
            potions.append({"id": "Potion Slot"})

    # Screen type
    phase_name = phase.name if hasattr(phase, "name") else str(phase)
    screen_type_map = {
        "MAP_NAVIGATION": "MAP",
        "COMBAT": "NONE",
        "COMBAT_REWARDS": "COMBAT_REWARD",
        "BOSS_REWARDS": "BOSS_REWARD",
        "EVENT": "EVENT",
        "SHOP": "SHOP_SCREEN",
        "REST": "REST",
        "TREASURE": "CHEST",
        "NEOW": "EVENT",
        "RUN_COMPLETE": "GAME_OVER",
        "GRID_SELECT": "GRID",
        "HAND_SELECT": "HAND_SELECT",
        "CARD_SELECT": "GRID",
    }
    screen_type = screen_type_map.get(phase_name, phase_name)

    if "pick_card" in action_type_names or "skip_card" in action_type_names:
        screen_type = "CARD_REWARD"

    screen_state = {}
    available_commands = []

    # ----- Combat -----
    combat_state = {}
    combat = getattr(runner, "current_combat", None)
    if combat and hasattr(combat, "state"):
        cs = combat.state
        player = cs.player

        # 敌人编码（完整 power/intent 信息）
        monsters = []
        for e in cs.enemies:
            is_alive = e.is_alive() if callable(getattr(e, "is_alive", None)) else getattr(e, "is_alive", True)

            # 完整 power 列表
            enemy_powers = []
            statuses = getattr(e, "statuses", {})
            for k, v_amt in statuses.items():
                enemy_powers.append({"id": k, "name": k, "amount": v_amt})

            # Intent 详细信息
            move_damage = max(e.move_damage or 0, 0)
            move_hits = e.move_hits or 1
            # 尝试获取更精确的 intent 类型
            intent_str = getattr(e, "intent", None) or ""
            if not intent_str:
                if move_damage > 0:
                    intent_str = "ATTACK"
                else:
                    intent_str = "UNKNOWN"

            monsters.append({
                "id": getattr(e, "id", "?"),
                "name": getattr(e, "name", "?"),
                "current_hp": e.hp,
                "max_hp": e.max_hp,
                "block": e.block,
                "is_gone": not is_alive,
                "halfDead": getattr(e, "half_dead", False),
                "intent": intent_str,
                "move_adjusted_damage": move_damage,
                "move_hits": move_hits,
                "powers": enemy_powers,
            })

        # 手牌（含 is_playable / has_target）
        hand = []
        for i, card_str in enumerate(cs.hand):
            cid = str(card_str)
            upgraded = cid.endswith("+")
            cd = card_id_to_agent_dict(cid, upgraded)

            # has_target 来自卡牌定义
            base_id = cid.rstrip("+") if cid.endswith("+") else cid
            card_def = ALL_CARDS.get(base_id)
            if card_def is not None:
                cd["has_target"] = card_def.target in (CardTarget.ENEMY, CardTarget.SELF_AND_ENEMY)
            else:
                cd["has_target"] = False

            # is_playable 来自实际费用 vs 能量
            if cid in cs.card_costs:
                actual_cost = cs.card_costs[cid]
            elif card_def is not None:
                actual_cost = card_def.current_cost if upgraded and card_def.upgraded else card_def.cost
                if upgraded and card_def.upgrade_cost is not None:
                    actual_cost = card_def.upgrade_cost
                elif card_def.cost_for_turn is not None:
                    actual_cost = card_def.cost_for_turn
                else:
                    actual_cost = card_def.cost
            else:
                actual_cost = cd.get("cost", 1)

            if actual_cost == -1:
                cd["is_playable"] = cs.energy > 0
            elif actual_cost < -1 or actual_cost >= 999:
                cd["is_playable"] = False
            else:
                cd["is_playable"] = actual_cost <= cs.energy

            hand.append(cd)

        # 玩家 powers（完整传递）
        player_powers = []
        player_statuses = getattr(player, "statuses", {})
        for k, v_amt in player_statuses.items():
            player_powers.append({"id": k, "name": k, "amount": v_amt})

        combat_state = {
            "player": {
                "current_hp": player.hp,
                "max_hp": player.max_hp,
                "block": player.block,
                "energy": cs.energy,
                "powers": player_powers,
                # V6 增强：角色机制字段
                "stance": getattr(player, "stance", ""),
                "orbs": [{"id": getattr(o, "id", ""), "amount": getattr(o, "passive_amount", 0)}
                         for o in getattr(player, "orbs", [])],
                "orb_slots": getattr(player, "orb_slots", 0),
            },
            "monsters": monsters,
            "hand": hand,
            "draw_pile": [card_id_to_agent_dict(str(c)) for c in cs.draw_pile],
            "discard_pile": [card_id_to_agent_dict(str(c)) for c in cs.discard_pile],
            "turn": cs.turn,
        }
        available_commands.extend(["play", "end"])

    # ----- Card rewards -----
    if screen_type == "CARD_REWARD":
        obs = runner.get_observation()
        reward_obs = obs.get("reward", {})
        card_reward_list = reward_obs.get("card_rewards", []) if reward_obs else []

        cards_for_agent = []
        for cr in card_reward_list:
            if not cr.get("skipped") and cr.get("claimed_index") is None:
                for c in cr.get("cards", []):
                    cid = c.get("id", "")
                    upgraded = c.get("upgraded", False)
                    cards_for_agent.append(card_id_to_agent_dict(cid, upgraded))
                break
        screen_state["cards"] = cards_for_agent
        available_commands.append("choose")

    # ----- Map navigation -----
    if screen_type == "MAP":
        available_paths = runner.run_state.get_available_paths()
        current_map = runner.run_state.get_current_map()
        nodes = []
        full_paths = []

        for node in available_paths:
            rt = node.room_type.name if node.room_type else "MONSTER"
            symbol = ROOM_TYPE_TO_SYMBOL.get(rt, "M")
            nodes.append({
                "symbol": symbol,
                "x": node.x,
                "y": node.y,
                "room_type": rt,
            })
            if current_map:
                path_rooms = _enumerate_path_to_boss(node, current_map)
            else:
                path_rooms = [symbol]
            full_paths.append({"rooms": path_rooms, "start_node": symbol})

        screen_state["next_nodes"] = nodes
        screen_state["full_paths"] = full_paths
        available_commands.append("choose")

    # ----- Event -----
    if screen_type == "EVENT":
        obs = runner.get_observation()
        event_obs = obs.get("event", {})
        event_id = event_obs.get("event_id", "") or event_obs.get("id", "")
        if not event_id and phase_name == "NEOW":
            event_id = "Neow"
        screen_state["event_id"] = event_id
        if event_obs:
            options = []
            for ch in event_obs.get("choices", []):
                options.append({
                    "label": ch.get("label", str(ch.get("choice_index", "?"))),
                    "text": ch.get("label", ""),
                    "choice_index": ch.get("choice_index", 0),
                })
            screen_state["options"] = options
        available_commands.append("choose")

    # ----- Rest -----
    if screen_type == "REST":
        rest_options = ["rest", "smith"]
        for opt in ("dig", "lift", "toke", "recall", "ruby_key"):
            if opt in action_type_names:
                rest_options.append(opt)
        screen_state["rest_options"] = rest_options
        available_commands.append("choose")

    # ----- Shop -----
    if screen_type == "SHOP_SCREEN":
        obs = runner.get_observation()
        shop_obs = obs.get("shop", {}) if obs else {}
        if shop_obs:
            shop_cards = []
            for c in shop_obs.get("colored_cards", []):
                if not c.get("purchased", False):
                    shop_cards.append({
                        "id": c.get("id", ""),
                        "price": c.get("price", 999),
                        "upgraded": c.get("upgraded", False),
                    })
            for c in shop_obs.get("colorless_cards", []):
                if not c.get("purchased", False):
                    shop_cards.append({
                        "id": c.get("id", ""),
                        "price": c.get("price", 999),
                        "upgraded": c.get("upgraded", False),
                    })
            screen_state["cards"] = shop_cards

            shop_relics = []
            for r in shop_obs.get("relics", []):
                if not r.get("purchased", False):
                    shop_relics.append({
                        "id": r.get("id", ""),
                        "price": r.get("price", 999),
                    })
            screen_state["relics"] = shop_relics

            shop_potions = []
            for p in shop_obs.get("potions", []):
                if not p.get("purchased", False):
                    shop_potions.append({
                        "id": p.get("id", ""),
                        "price": p.get("price", 999),
                    })
            screen_state["potions"] = shop_potions

            screen_state["purge_available"] = shop_obs.get("purge_available", False)
            screen_state["purge_cost"] = shop_obs.get("purge_cost", 9999)

        available_commands.extend(["return", "leave"])

    # ----- Combat rewards (non-card) -----
    if screen_type == "COMBAT_REWARD":
        obs = runner.get_observation()
        reward_obs = obs.get("reward", {}) if obs else {}
        card_reward_list = reward_obs.get("card_rewards", []) if reward_obs else []
        reward_cards = []
        for cr in card_reward_list:
            if not cr.get("skipped") and cr.get("claimed_index") is None:
                for c in cr.get("cards", []):
                    cid = c.get("id", "")
                    upgraded = c.get("upgraded", False)
                    reward_cards.append(card_id_to_agent_dict(cid, upgraded))
                break
        screen_state["reward_cards"] = reward_cards
        available_commands.extend(["proceed", "choose"])

    # ----- Boss relic -----
    if screen_type == "BOSS_REWARD":
        obs = runner.get_observation()
        reward_obs = obs.get("reward", {}) if obs else {}
        boss_relics = reward_obs.get("boss_relics", {}) if reward_obs else {}
        if boss_relics:
            relics_list = [{"id": rid, "name": rid} for rid in boss_relics.get("relics", [])]
            screen_state["relics"] = relics_list
        available_commands.append("choose")

    # ----- Grid / Hand Select -----
    if screen_type in ("GRID", "HAND_SELECT") or "select_card" in action_type_names or "remove_card" in action_type_names:
        if screen_type not in ("GRID", "HAND_SELECT"):
            screen_type = "GRID"
        obs = runner.get_observation()
        select_obs = obs.get("card_select", obs.get("grid", {})) if obs else {}
        selectable = select_obs.get("cards", []) if select_obs else []
        if selectable:
            cards_for_agent = []
            for c in selectable:
                cid = c.get("id", "") if isinstance(c, dict) else str(c)
                upgraded = (c.get("upgraded", False) if isinstance(c, dict)
                            else str(c).endswith("+"))
                cards_for_agent.append(card_id_to_agent_dict(cid, upgraded))
            screen_state["cards"] = cards_for_agent
        elif not screen_state.get("cards"):
            screen_state["cards"] = deck_cards
        select_type = select_obs.get("type", "")
        if not select_type:
            if "remove_card" in action_type_names:
                select_type = "purge"
            elif "upgrade" in action_type_names:
                select_type = "upgrade"
        screen_state["select_type"] = select_type
        available_commands.extend(["choose", "confirm", "cancel"])

    # ----- Chest -----
    if screen_type == "CHEST":
        available_commands.extend(["choose", "proceed"])

    # ----- Game over -----
    if runner.game_over:
        screen_type = "GAME_OVER"
        available_commands = ["proceed"]

    if not available_commands:
        available_commands = ["choose"]

    gs = {
        "screen_type": screen_type,
        "current_hp": rs.current_hp,
        "max_hp": rs.max_hp,
        "gold": rs.gold,
        "floor": rs.floor,
        "act": rs.act,
        "ascension_level": rs.ascension,
        "deck": deck_cards,
        "relics": relics,
        "potions": potions,
        "combat_state": combat_state,
        "screen_state": screen_state,
    }

    return {
        "game_state": gs,
        "available_commands": list(set(available_commands)),
        "ready_for_command": True,
        "in_game": not runner.game_over,
    }


# ============================================================
# Command -> action bridge
# ============================================================

def cmd_to_action(cmd: str, actions):
    """将 agent 命令字符串翻译为 simulator GameAction。"""
    parts = cmd.strip().split()
    if not parts:
        return actions[0] if actions else None

    verb = parts[0].upper()
    idx = int(parts[1]) if len(parts) >= 2 and parts[1].isdigit() else 0

    if verb == "END":
        for a in actions:
            if hasattr(a, "action_type") and a.action_type == "end_turn":
                return a

    # POTION slot_idx [target_idx]
    if verb == "POTION" and len(parts) >= 2:
        potion_slot = int(parts[1])
        target_idx = int(parts[2]) if len(parts) >= 3 else -1
        for a in actions:
            if hasattr(a, "action_type") and a.action_type == "use_potion":
                if getattr(a, "potion_idx", -1) == potion_slot:
                    if target_idx < 0 or getattr(a, "target_idx", -1) == target_idx:
                        return a
        for a in actions:
            if hasattr(a, "action_type") and a.action_type == "use_potion":
                if getattr(a, "potion_idx", -1) == potion_slot:
                    return a

    if verb == "PLAY" and len(parts) >= 2:
        card_idx = int(parts[1]) - 1
        target_idx = int(parts[2]) if len(parts) >= 3 else -1
        for a in actions:
            if hasattr(a, "action_type") and a.action_type == "play_card":
                if getattr(a, "card_idx", None) == card_idx:
                    if target_idx < 0 or getattr(a, "target_idx", -1) == target_idx:
                        return a
        for a in actions:
            if hasattr(a, "action_type") and a.action_type == "play_card":
                return a

    if verb == "CHOOSE":
        choosable = [
            a for a in actions
            if hasattr(a, "action_type") and a.action_type in
               ("path_choice", "event_choice", "pick_card", "pick_boss_relic",
                "rest", "smith", "upgrade", "dig", "lift", "recall", "ruby_key",
                "take_relic", "sapphire_key", "claim_relic", "claim_gold",
                "buy_card", "buy_relic", "remove_card")
        ]
        if verb == "CHOOSE" and len(parts) >= 2 and not parts[1].isdigit():
            rest_word = parts[1].lower()
            for a in actions:
                if hasattr(a, "action_type") and a.action_type == rest_word:
                    return a

        if idx < len(choosable):
            return choosable[idx]
        if actions:
            return actions[0]

    if verb in ("PROCEED", "SKIP", "CONFIRM", "LEAVE", "RETURN"):
        targets = {
            "PROCEED": ("proceed_from_rewards", "leave_shop", "proceed"),
            "SKIP": ("skip_card",),
            "CONFIRM": ("confirm",),
            "LEAVE": ("leave_shop", "leave"),
            "RETURN": ("leave_shop", "leave"),
        }.get(verb, ())
        for a in actions:
            if hasattr(a, "action_type") and a.action_type in targets:
                return a
        for a in actions:
            if hasattr(a, "action_type") and "skip" in a.action_type:
                return a

    return actions[0] if actions else None


# ============================================================
# HP value helper for reward shaping
# ============================================================

def hp_value(hp, max_hp):
    """归一化 HP 价值 [0, 1]，用于奖励计算。"""
    if max_hp <= 0:
        return 0.0
    return float(float(hp) / float(max_hp)) ** 0.5


# ============================================================
# Agent state reset（每局游戏前调用）
# ============================================================

def reset_agent_for_game(agent):
    """重置 agent 的单局状态。"""
    agent.current_trajectory = []
    agent.in_combat = False
    agent.prev_combat_snap = None
    agent.current_floor = 0
    agent.combat_turns = 0
    agent.combat_hp_value_start = 0.0
    agent._stuck = 0
    agent._screen_repeat = 0
    agent._last_sig = ""
    agent._last_screen = ""
    if hasattr(agent, "cached_deck_profile"):
        agent.cached_deck_profile = None


# ============================================================
# Single game runner
# ============================================================

def run_one_game(agent, seed, ascension=0, training=True, verbose=False):
    """执行一局完整游戏。

    所有决策（combat + strategy）均由 agent 的 decide() 处理。
    training=True 时记录 transition 用于 PPO 更新。

    Returns: 游戏统计 dict
    """
    runner = GameRunner(
        seed=str(seed),
        ascension=ascension,
        character="Watcher",
        verbose=False,
    )

    steps = 0
    max_steps = 8000
    prev_floor = 0
    prev_in_combat = False
    was_in_combat = False
    n_strategy_decisions = 0
    n_combat_decisions = 0

    # 追踪 HP delta 用于策略层奖励
    pre_combat_hp_value = hp_value(runner.run_state.current_hp, runner.run_state.max_hp)

    # 追踪 END TURN 前后 HP（用于 HP loss penalty）
    pre_turn_hp = runner.run_state.current_hp

    while not runner.game_over and steps < max_steps:
        try:
            actions = runner.get_available_actions()
        except Exception as e:
            if verbose:
                print(f"  [ERR] get_available_actions: {e}")
            break

        if not actions:
            break

        phase = runner.phase

        # ---- Combat ----
        if phase == GamePhase.COMBAT:
            if not was_in_combat:
                was_in_combat = True
                pre_combat_hp_value = hp_value(
                    runner.run_state.current_hp,
                    runner.run_state.max_hp,
                )
                pre_turn_hp = runner.run_state.current_hp  # 追踪 END TURN 前后 HP

            # 药水决策（每次出牌前检查）
            potion_actions = [a for a in actions
                              if hasattr(a, "action_type") and a.action_type == "use_potion"]
            if potion_actions:
                try:
                    potion_state = build_agent_state(runner)
                    potion_cmd = agent.potion_decide(potion_state)
                    if potion_cmd and potion_cmd != "SKIP":
                        potion_action = cmd_to_action(potion_cmd, actions)
                        if potion_action is not None:
                            runner.take_action(potion_action)
                            steps += 1
                            continue
                except Exception as e:
                    if verbose:
                        print(f"  [WARN] potion_decide failed: {e}")

            # （逐卡 reward 已在 agent.combat_act 内计算，此处无需快照）

            # Agent 决策
            try:
                state = build_agent_state(runner)
            except Exception as e:
                if verbose:
                    print(f"  [ERR] build_agent_state (combat): {e}")
                runner.take_action(actions[0])
                steps += 1
                continue

            try:
                cmd = agent.decide(state)
                n_combat_decisions += 1
            except Exception as e:
                if verbose:
                    print(f"  [WARN] agent.decide (combat) failed: {e}")
                cmd = "END"

            action = cmd_to_action(cmd, actions)
            if action is None:
                action = actions[0]

            try:
                runner.take_action(action)
            except Exception as e:
                if verbose:
                    print(f"  [ERR] combat take_action: {e}")
                break

            # END TURN HP loss penalty — 敌人回合后的 HP 损失追加到 END TURN transition
            if training and agent.current_trajectory:
                last_t = agent.current_trajectory[-1]
                if last_t.decision_type == "combat":
                    cur_hp = runner.run_state.current_hp
                    hp_lost = max(pre_turn_hp - cur_hp, 0)
                    if hp_lost > 0:
                        last_t.reward -= hp_lost * HP_LOSS_PENALTY_SCALE
                    pre_turn_hp = cur_hp

            steps += 1
            continue

        # ---- Non-combat ----
        if was_in_combat and phase != GamePhase.COMBAT:
            was_in_combat = False
            prev_in_combat = True

        # Floor advancement
        current_floor = runner.run_state.floor
        if current_floor > prev_floor:
            agent.on_floor_cleared()
            prev_floor = current_floor

        # Combat end → 计算策略层延迟奖励
        if prev_in_combat:
            post_combat_hp_val = hp_value(
                runner.run_state.current_hp,
                runner.run_state.max_hp,
            )

            gs = {
                "game_state": {
                    "current_hp": runner.run_state.current_hp,
                    "max_hp": runner.run_state.max_hp,
                    "floor": runner.run_state.floor,
                    "screen_type": "COMBAT_REWARD",
                },
                "available_commands": [],
                "ready_for_command": True,
                "in_game": True,
            }
            won_combat = runner.run_state.current_hp > 0
            agent.on_combat_end(gs, won_combat)

            # 战斗结束奖励：win/lose + 策略层 HP delta
            if training and agent.current_trajectory:
                # Win/Lose bonus on last combat transition
                combat_end_bonus = COMBAT_WIN_REWARD if won_combat else COMBAT_LOSE_PENALTY
                for t in reversed(agent.current_trajectory):
                    if t.decision_type == "combat":
                        t.reward += combat_end_bonus
                        break

                # 策略层 HP delta reward（加到最后一条策略 transition）
                if hasattr(agent, 'compute_strategy_reward'):
                    combat_reward = agent.compute_strategy_reward(
                        pre_combat_hp_value, post_combat_hp_val
                    )
                    if agent.current_trajectory:
                        agent.current_trajectory[-1].reward += combat_reward

            pre_combat_hp_value = post_combat_hp_val
            prev_in_combat = False

        # 构建 agent 状态
        try:
            state = build_agent_state(runner)
        except Exception as e:
            if verbose:
                print(f"  [ERR] build_agent_state: {e}")
            runner.take_action(actions[0])
            steps += 1
            continue

        # Agent 策略决策
        try:
            cmd = agent.decide(state)
            n_strategy_decisions += 1
        except Exception as e:
            if verbose:
                print(f"  [WARN] agent.decide failed: {e}")
            cmd = "CHOOSE 0"

        action = cmd_to_action(cmd, actions)
        if action is None:
            action = actions[0]

        try:
            runner.take_action(action)
        except Exception as e:
            if verbose:
                print(f"  [ERR] take_action: {e}")
            break

        steps += 1

    # 终局回调
    if runner.game_over:
        final_state = {
            "game_state": {
                "current_hp": runner.run_state.current_hp,
                "max_hp": runner.run_state.max_hp,
                "floor": runner.run_state.floor,
                "screen_type": "GAME_OVER",
            },
            "available_commands": ["proceed"],
            "ready_for_command": True,
            "in_game": False,
        }
        agent.on_run_end(final_state)

    return {
        "won": runner.game_won,
        "floor": runner.run_state.floor,
        "hp": runner.run_state.current_hp,
        "max_hp": runner.run_state.max_hp,
        "steps": steps,
        "n_strategy_decisions": n_strategy_decisions,
        "n_combat_decisions": n_combat_decisions,
    }


# ============================================================
# Multiprocessing support
# ============================================================

_worker_agent = None


def _init_worker(model_state_dict):
    """每个 worker 进程初始化一个持久 agent。"""
    global _worker_agent
    import sts_agent_v6 as _v6
    _worker_agent = _v6.AgentV6(device="cpu")
    _worker_agent.model.load_state_dict(model_state_dict)
    _worker_agent.model.eval()
    _worker_agent.suppress_ppo = True  # workers 只收集，不做 PPO


def collect_one_game(args):
    """Worker 函数：跑一局游戏，返回 trajectories 和统计。"""
    seed, ascension = args
    global _worker_agent
    reset_agent_for_game(_worker_agent)
    _worker_agent.trajectory_buffer = []  # 清空上一局残留的 trajectories
    _worker_agent.suppress_ppo = True

    try:
        result = run_one_game(_worker_agent, seed, ascension, training=True, verbose=False)
    except Exception as e:
        return {
            "trajectories": [],
            "current_trajectory": [],
            "result": {"won": False, "floor": 0, "hp": 0, "max_hp": 80,
                        "steps": 0, "n_strategy_decisions": 0,
                        "n_combat_decisions": 0},
            "error": str(e),
        }

    return {
        "trajectories": list(_worker_agent.trajectory_buffer),  # 返回副本
        "current_trajectory": list(_worker_agent.current_trajectory) if _worker_agent.current_trajectory else [],
        "result": result,
    }


# ============================================================
# Validation
# ============================================================

def run_validation(agent, n_games=VALIDATION_GAMES, ascension=0, verbose=True):
    """跑验证局（不训练），返回统计摘要。"""
    was_suppress_ppo = getattr(agent, 'suppress_ppo', False)
    agent.suppress_ppo = True
    agent.model.eval()

    results = []
    val_start = time.time()

    for i in range(n_games):
        seed = random.randint(0, 999999)
        reset_agent_for_game(agent)

        try:
            result = run_one_game(agent, seed, ascension, training=False, verbose=False)
        except Exception as e:
            if verbose:
                print(f"  [VAL ERR] Game {i+1}: {e}")
            result = {"won": False, "floor": 0, "hp": 0, "max_hp": 80,
                      "steps": 0, "n_strategy_decisions": 0, "n_combat_decisions": 0}
        results.append(result)

    val_elapsed = time.time() - val_start

    wins = sum(1 for r in results if r["won"])
    floors = [r["floor"] for r in results]
    avg_floor = sum(floors) / max(len(floors), 1)
    max_floor = max(floors) if floors else 0

    if verbose:
        print(f"  [VAL] {wins}/{n_games} wins ({100*wins/max(n_games,1):.0f}%) "
              f"avg_floor={avg_floor:.1f} max_floor={max_floor} "
              f"({val_elapsed:.1f}s)")

    agent.suppress_ppo = was_suppress_ppo
    agent.model.train()

    return {
        "wins": wins,
        "n_games": n_games,
        "win_rate": wins / max(n_games, 1),
        "avg_floor": avg_floor,
        "max_floor": max_floor,
        "elapsed": val_elapsed,
    }


# ============================================================
# Logging
# ============================================================

def log(msg):
    """同时写 stats log 和 stdout。"""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        with open(STATS_LOG, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


# ============================================================
# Save/load checkpoint
# ============================================================

def save_checkpoint(agent, games_done, extra=None):
    """保存训练 checkpoint。"""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINT_DIR / "agent_v6.pt"
    data = {
        "model": agent.model.state_dict(),
        "optimizer": agent.optimizer.state_dict(),
        "runs": agent.runs,
        "games_done": games_done,
        "entropy_coeff": agent.entropy_coeff,
    }
    if extra:
        data.update(extra)
    try:
        torch.save(data, path)
        log(f"Checkpoint saved: games={games_done}, runs={agent.runs}")
    except Exception as e:
        log(f"Checkpoint save failed: {e}")


def load_checkpoint(agent, path=None, force=False):
    """加载训练 checkpoint（严格模式）。

    超过 30% 层加载失败时拒绝加载（除非 --force）。

    Returns:
        dict: {"games_done": int, "model_loaded": bool, ...}
    """
    if path is None:
        path = CHECKPOINT_DIR / "agent_v6.pt"
    else:
        path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint 不存在: {path}")

    try:
        d = torch.load(path, weights_only=False, map_location="cpu")
    except Exception as e:
        raise RuntimeError(f"Checkpoint 文件损坏或无法读取 ({path}): {e}")

    result = {
        "games_done": 0, "model_loaded": False, "optimizer_loaded": False,
        "layers_loaded": 0, "layers_total": 0, "layers_skipped": [],
        "path": str(path),
    }

    # 提取 state_dict
    if "model" in d:
        state_dict = d["model"]
    elif isinstance(d, dict) and any(k.startswith(("encoder", "draft", "combat", "choice", "path", "ctx")) for k in d.keys()):
        state_dict = d
    else:
        raise RuntimeError(f"Checkpoint 格式无法识别 ({path}): keys={list(d.keys())[:10]}")

    # 逐层安全加载
    model_state = agent.model.state_dict()
    loaded = 0
    skipped = 0
    skipped_keys = []
    total_keys = len(model_state)

    for key, param in state_dict.items():
        if key in model_state:
            if param.shape == model_state[key].shape:
                model_state[key] = param
                loaded += 1
            else:
                skipped_keys.append(f"{key} ({param.shape} vs {model_state[key].shape})")
                skipped += 1
        else:
            skipped_keys.append(f"{key} (不存在于当前模型)")
            skipped += 1

    # 兼容性检查
    if total_keys > 0 and loaded / total_keys < 0.7:
        msg = (f"Checkpoint 兼容性不足: 仅加载 {loaded}/{total_keys} 层 "
               f"({loaded/total_keys*100:.1f}%)，跳过: {skipped_keys[:10]}")
        if force:
            log(f"[PREFLIGHT] 警告: {msg}（--force 强制继续）")
        else:
            raise RuntimeError(f"{msg}\n使用 --force 强制加载")

    agent.model.load_state_dict(model_state)
    result["model_loaded"] = True
    result["layers_loaded"] = loaded
    result["layers_total"] = total_keys
    result["layers_skipped"] = skipped_keys

    if skipped_keys:
        log(f"模型加载: {loaded}/{total_keys} 层成功, {skipped} 层跳过")
        for sk in skipped_keys[:10]:
            log(f"  跳过: {sk}")
        if len(skipped_keys) > 10:
            log(f"  ... 共 {len(skipped_keys)} 层跳过")
    else:
        log(f"模型加载: {loaded}/{total_keys} 层全部成功")

    # 加载 optimizer
    if "optimizer" in d:
        try:
            agent.optimizer.load_state_dict(d["optimizer"])
            result["optimizer_loaded"] = True
            log("Optimizer 状态已加载")
        except Exception as e:
            log(f"[PREFLIGHT] Optimizer 状态跳过（架构变更），模型权重已加载: {e}")
            result["optimizer_loaded"] = False
    else:
        log("Checkpoint 中无 optimizer 状态，使用默认 optimizer")

    # 恢复元数据
    if "model" in d:
        agent.runs = d.get("runs", 0)
        agent.entropy_coeff = d.get("entropy_coeff", v6.ENTROPY_COEFF_START)
        result["games_done"] = d.get("games_done", 0)
        log(f"Checkpoint 元数据: games={result['games_done']}, runs={agent.runs}")
    else:
        log(f"Pretrain 权重加载自 {path}")

    return result


# ============================================================
# Main training loop
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="V6 PPO Training (Agent handles ALL decisions)")
    parser.add_argument("--n-games", type=int, default=N_GAMES_TOTAL,
                        help="Total training games")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of worker processes for game collection")
    parser.add_argument("--ascension", type=int, default=0,
                        help="Ascension level")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--from-scratch", action="store_true", default=False,
                        help="从随机权重开始训练（不加载任何 checkpoint）")
    parser.add_argument("--force", action="store_true", default=False,
                        help="强制加载兼容性不足的 checkpoint")
    parser.add_argument("--test", action="store_true", default=False,
                        help="测试模式：跑 1 局验证 pipeline 无崩溃")
    args = parser.parse_args()

    # ---- Test mode ----
    if args.test:
        log("=" * 60)
        log("V6 TEST MODE: 验证 pipeline")
        log("=" * 60)

        log("1. 创建 AgentV6...")
        agent = v6.AgentV6()
        agent.suppress_ppo = True
        log(f"   模型参数: {sum(p.numel() for p in agent.model.parameters()):,}")
        log(f"   Device: {v6.DEVICE}")

        log("2. 启动一局游戏...")
        seed = random.randint(0, 999999)
        reset_agent_for_game(agent)

        runner = GameRunner(
            seed=str(seed),
            ascension=0,
            character="Watcher",
            verbose=False,
        )
        log(f"   Seed={seed}, HP={runner.run_state.current_hp}/{runner.run_state.max_hp}")

        log("3. 运行前 50 步决策...")
        n_decisions = 0
        max_test_steps = 200
        steps = 0
        decision_log = []

        while not runner.game_over and steps < max_test_steps:
            try:
                actions = runner.get_available_actions()
            except Exception as e:
                log(f"   [ERR] get_available_actions: {e}")
                break
            if not actions:
                break

            try:
                state = build_agent_state(runner)
                cmd = agent.decide(state)
                n_decisions += 1

                phase_name = runner.phase.name if hasattr(runner.phase, "name") else str(runner.phase)
                if n_decisions <= 20:
                    decision_log.append(f"     [{n_decisions}] phase={phase_name} cmd={cmd}")

                action = cmd_to_action(cmd, actions)
                if action is None:
                    action = actions[0]
                runner.take_action(action)
            except Exception as e:
                log(f"   [ERR] step {steps}: {e}")
                import traceback
                log(traceback.format_exc())
                break

            steps += 1

        for dl in decision_log:
            log(dl)
        if n_decisions > 20:
            log(f"     ... ({n_decisions} decisions total)")

        log(f"4. 结果: floor={runner.run_state.floor}, "
            f"HP={runner.run_state.current_hp}/{runner.run_state.max_hp}, "
            f"steps={steps}, decisions={n_decisions}, "
            f"game_over={runner.game_over}")

        # 检查 trajectory 记录
        n_traj = len(agent.current_trajectory)
        types = {}
        for t in agent.current_trajectory:
            dt = t.decision_type
            types[dt] = types.get(dt, 0) + 1
        log(f"5. Trajectory: {n_traj} transitions, types={types}")

        log("=" * 60)
        log("TEST PASSED" if n_decisions > 0 else "TEST FAILED (0 decisions)")
        log("=" * 60)
        return

    # ---- Normal training mode ----
    n_games = args.n_games
    n_workers = args.workers
    ascension = args.ascension

    log("=" * 60)
    log(f"V6 PPO Training: {n_games} games, A{ascension}, workers={n_workers}")
    log(f"  Combat: AI combat head | Strategy: v6 agent")
    log(f"  PPO update every {N_RUNS_PER_UPDATE} games, validate every {VALIDATION_INTERVAL}")
    log(f"  Combat rewards: per-card immediate feedback, "
        f"win={COMBAT_WIN_REWARD}, lose={COMBAT_LOSE_PENALTY}, hp_loss_scale={HP_LOSS_PENALTY_SCALE}")
    log("=" * 60)

    # 创建 agent
    agent = v6.AgentV6()
    agent.suppress_ppo = False

    # ========== Preflight: checkpoint 加载 ==========
    games_done = 0
    ckpt_result = None
    default_ckpt = CHECKPOINT_DIR / "agent_v6.pt"

    if args.from_scratch:
        log("[PREFLIGHT] --from-scratch: 使用随机权重初始化")
    elif args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            log(f"[PREFLIGHT] 错误: 指定的 checkpoint 不存在: {ckpt_path}")
            sys.exit(1)
        try:
            ckpt_result = load_checkpoint(agent, args.checkpoint, force=args.force)
            games_done = ckpt_result["games_done"]
        except (RuntimeError, FileNotFoundError) as e:
            log(f"[PREFLIGHT] 错误: checkpoint 加载失败: {e}")
            sys.exit(1)
    else:
        if default_ckpt.exists():
            try:
                ckpt_result = load_checkpoint(agent, default_ckpt, force=args.force)
                games_done = ckpt_result["games_done"]
            except (RuntimeError, FileNotFoundError) as e:
                log(f"[PREFLIGHT] 错误: 默认 checkpoint ({default_ckpt}) 加载失败: {e}")
                log(f"[PREFLIGHT] 如需从随机权重开始，请使用 --from-scratch")
                sys.exit(1)
        else:
            log(f"[PREFLIGHT] 错误: 未找到 checkpoint ({default_ckpt})，"
                f"且未指定 --from-scratch")
            log(f"[PREFLIGHT] 如需从随机权重开始训练，请显式传入 --from-scratch")
            sys.exit(1)

    # Preflight 摘要
    if ckpt_result:
        model_status = (f"loaded from {ckpt_result['path']} "
                        f"({ckpt_result['layers_loaded']}/{ckpt_result['layers_total']} layers)")
        opt_status = "loaded" if ckpt_result["optimizer_loaded"] else "reset"
    else:
        model_status = "random init (--from-scratch)"
        opt_status = "default"
    log(f"[PREFLIGHT] Model: {model_status}")
    log(f"[PREFLIGHT] Optimizer: {opt_status}")
    log(f"[PREFLIGHT] Training: {n_games} games, {n_workers} workers")

    # 全参数训练（combat + strategy）
    if hasattr(agent, 'configure_full_training'):
        agent.configure_full_training()
    agent.model.train()

    n_batches = (n_games - games_done) // N_RUNS_PER_UPDATE
    total_start = time.time()

    # 追踪统计
    all_floors = []
    all_wins = 0
    batch_floors = []
    batch_wins = 0
    batch_strategy_decisions = []
    batch_combat_decisions = []

    if n_workers > 1:
        # ---- 并行收集 ----
        model_state = {k: v.cpu() for k, v in agent.model.state_dict().items()}
        pool = Pool(n_workers, initializer=_init_worker, initargs=(model_state,))
        log(f"Worker pool created: {n_workers} workers")

        try:
            for batch_idx in range(n_batches):
                batch_start = time.time()

                batch_args = [
                    (random.randint(0, 999999), ascension)
                    for _ in range(N_RUNS_PER_UPDATE)
                ]

                try:
                    results = pool.map(collect_one_game, batch_args)
                except Exception as e:
                    log(f"Pool error: {e}")
                    import traceback
                    log(traceback.format_exc())
                    # 回退到串行
                    results = []
                    for seed_asc in batch_args:
                        seed, asc = seed_asc
                        reset_agent_for_game(agent)
                        agent.suppress_ppo = True
                        try:
                            r = run_one_game(agent, seed, asc, training=True, verbose=False)
                        except Exception as e2:
                            log(f"Sequential fallback error: {e2}")
                            r = {"won": False, "floor": 0, "hp": 0, "max_hp": 80,
                                 "steps": 0, "n_strategy_decisions": 0,
                                 "n_combat_decisions": 0}
                        results.append({
                            "trajectories": agent.trajectory_buffer,
                            "current_trajectory": agent.current_trajectory,
                            "result": r,
                        })

                # 合并 worker trajectories
                for res in results:
                    for traj in res["trajectories"]:
                        agent.trajectory_buffer.append(traj)
                    if res["current_trajectory"]:
                        agent.trajectory_buffer.append(res["current_trajectory"])

                    r = res["result"]
                    all_floors.append(r["floor"])
                    batch_floors.append(r["floor"])
                    batch_strategy_decisions.append(r.get("n_strategy_decisions", 0))
                    batch_combat_decisions.append(r.get("n_combat_decisions", 0))
                    if r["won"]:
                        all_wins += 1
                        batch_wins += 1

                games_done += N_RUNS_PER_UPDATE

                # PPO update（全 transition）
                if agent.trajectory_buffer:
                    n_trajs = len(agent.trajectory_buffer)
                    log(f"PPO update: {n_trajs} trajectories")
                    try:
                        stats = v6.ppo_update(
                            agent.model, agent.optimizer,
                            agent.trajectory_buffer, agent.entropy_coeff,
                            strategy_only=False,
                        )
                        log(f"  policy_loss={stats.get('policy_loss',0):.4f} "
                            f"value_loss={stats.get('value_loss',0):.4f} "
                            f"entropy={stats.get('entropy',0):.4f} "
                            f"n={stats.get('n_transitions',0)}")
                    except Exception as e:
                        log(f"  PPO update FAILED: {e}")
                        import traceback
                        log(traceback.format_exc())
                    agent.trajectory_buffer = []

                    # Entropy annealing
                    agent.runs += N_RUNS_PER_UPDATE
                    agent.entropy_coeff = max(
                        v6.ENTROPY_COEFF_END,
                        v6.ENTROPY_COEFF_START - (v6.ENTROPY_COEFF_START - v6.ENTROPY_COEFF_END)
                        * agent.runs / v6.ENTROPY_ANNEAL_RUNS
                    )

                # 更新权重后重建 pool
                pool.terminate()
                pool.join()
                model_state = {k: v.cpu() for k, v in agent.model.state_dict().items()}
                pool = Pool(n_workers, initializer=_init_worker, initargs=(model_state,))

                # Batch 统计
                batch_elapsed = time.time() - batch_start
                avg_floor = sum(batch_floors) / max(len(batch_floors), 1)
                avg_strat = sum(batch_strategy_decisions) / max(len(batch_strategy_decisions), 1)
                avg_combat = sum(batch_combat_decisions) / max(len(batch_combat_decisions), 1)
                total_avg_floor = sum(all_floors) / max(len(all_floors), 1)

                log(f"[batch {batch_idx+1}/{n_batches}] "
                    f"games={games_done}/{n_games} | "
                    f"avg_floor={avg_floor:.1f} (total={total_avg_floor:.1f}) | "
                    f"wins={batch_wins}/{len(batch_floors)} | "
                    f"strat={avg_strat:.0f} combat={avg_combat:.0f} | "
                    f"ent={agent.entropy_coeff:.4f} | "
                    f"{batch_elapsed:.1f}s")

                batch_floors = []
                batch_wins = 0
                batch_strategy_decisions = []
                batch_combat_decisions = []

                # 定期 checkpoint
                if games_done % 100 == 0:
                    save_checkpoint(agent, games_done)

                # 定期 validation
                val_every_batches = max(VALIDATION_INTERVAL // N_RUNS_PER_UPDATE, 1)
                if (batch_idx + 1) % val_every_batches == 0:
                    log("--- Validation ---")
                    pool.terminate()
                    pool.join()

                    val_stats = run_validation(agent, n_games=VALIDATION_GAMES,
                                               ascension=ascension)
                    log(f"  VAL: {val_stats['wins']}/{val_stats['n_games']} wins "
                        f"({val_stats['win_rate']*100:.0f}%) "
                        f"avg_floor={val_stats['avg_floor']:.1f}")

                    agent.model.train()
                    model_state = {k: v.cpu() for k, v in agent.model.state_dict().items()}
                    pool = Pool(n_workers, initializer=_init_worker, initargs=(model_state,))

        finally:
            pool.terminate()
            pool.join()

    else:
        # ---- 串行模式 ----
        for batch_idx in range(n_batches):
            batch_start = time.time()

            for game_in_batch in range(N_RUNS_PER_UPDATE):
                seed = random.randint(0, 999999)
                reset_agent_for_game(agent)
                agent.suppress_ppo = True

                try:
                    result = run_one_game(agent, seed, ascension, training=True, verbose=False)
                except Exception as e:
                    log(f"Game crashed: {e}")
                    import traceback
                    log(traceback.format_exc())
                    result = {"won": False, "floor": 0, "hp": 0, "max_hp": 80,
                              "steps": 0, "n_strategy_decisions": 0,
                              "n_combat_decisions": 0}

                # 收集 trajectory
                if agent.current_trajectory:
                    agent.trajectory_buffer.append(agent.current_trajectory)

                all_floors.append(result["floor"])
                batch_floors.append(result["floor"])
                batch_strategy_decisions.append(result.get("n_strategy_decisions", 0))
                batch_combat_decisions.append(result.get("n_combat_decisions", 0))
                if result["won"]:
                    all_wins += 1
                    batch_wins += 1

            games_done += N_RUNS_PER_UPDATE

            # PPO update
            if agent.trajectory_buffer:
                n_trajs = len(agent.trajectory_buffer)
                log(f"PPO update: {n_trajs} trajectories")
                try:
                    stats = v6.ppo_update(
                        agent.model, agent.optimizer,
                        agent.trajectory_buffer, agent.entropy_coeff,
                        strategy_only=False,
                    )
                    log(f"  policy_loss={stats.get('policy_loss',0):.4f} "
                        f"value_loss={stats.get('value_loss',0):.4f} "
                        f"entropy={stats.get('entropy',0):.4f} "
                        f"n={stats.get('n_transitions',0)}")
                except Exception as e:
                    log(f"  PPO update FAILED: {e}")
                    import traceback
                    log(traceback.format_exc())
                agent.trajectory_buffer = []

                agent.runs += N_RUNS_PER_UPDATE
                agent.entropy_coeff = max(
                    v6.ENTROPY_COEFF_END,
                    v6.ENTROPY_COEFF_START - (v6.ENTROPY_COEFF_START - v6.ENTROPY_COEFF_END)
                    * agent.runs / v6.ENTROPY_ANNEAL_RUNS
                )

            # Batch 统计
            batch_elapsed = time.time() - batch_start
            avg_floor = sum(batch_floors) / max(len(batch_floors), 1)
            avg_strat = sum(batch_strategy_decisions) / max(len(batch_strategy_decisions), 1)
            avg_combat = sum(batch_combat_decisions) / max(len(batch_combat_decisions), 1)
            total_avg_floor = sum(all_floors) / max(len(all_floors), 1)

            log(f"[batch {batch_idx+1}/{n_batches}] "
                f"games={games_done}/{n_games} | "
                f"avg_floor={avg_floor:.1f} (total={total_avg_floor:.1f}) | "
                f"wins={batch_wins}/{len(batch_floors)} | "
                f"strat={avg_strat:.0f} combat={avg_combat:.0f} | "
                f"ent={agent.entropy_coeff:.4f} | "
                f"{batch_elapsed:.1f}s")

            batch_floors = []
            batch_wins = 0
            batch_strategy_decisions = []
            batch_combat_decisions = []

            if games_done % 100 == 0:
                save_checkpoint(agent, games_done)

            val_every_batches = max(VALIDATION_INTERVAL // N_RUNS_PER_UPDATE, 1)
            if (batch_idx + 1) % val_every_batches == 0:
                log("--- Validation ---")
                val_stats = run_validation(agent, n_games=VALIDATION_GAMES,
                                           ascension=ascension)
                log(f"  VAL: {val_stats['wins']}/{val_stats['n_games']} wins "
                    f"({val_stats['win_rate']*100:.0f}%) "
                    f"avg_floor={val_stats['avg_floor']:.1f}")
                agent.model.train()

    # 最终汇总
    total_elapsed = time.time() - total_start
    total_wins = all_wins
    total_games = len(all_floors)

    log("=" * 60)
    log(f"TRAINING COMPLETE")
    log(f"  {total_games} games in {total_elapsed:.0f}s "
        f"({total_elapsed/max(total_games,1):.1f}s/game)")
    log(f"  wins: {total_wins}/{total_games} ({100*total_wins/max(total_games,1):.0f}%)")
    if all_floors:
        log(f"  avg_floor: {sum(all_floors)/len(all_floors):.1f} "
            f"max_floor: {max(all_floors)}")

    # Floor distribution
    floor_dist = {}
    for f in all_floors:
        floor_dist[f] = floor_dist.get(f, 0) + 1
    log("Floor distribution:")
    for f in sorted(floor_dist.keys()):
        bar = "#" * floor_dist[f]
        log(f"  floor {f:2d}: {bar} ({floor_dist[f]})")

    # Act completion
    act1 = sum(1 for f in all_floors if f >= 17)
    act2 = sum(1 for f in all_floors if f >= 34)
    act3 = sum(1 for f in all_floors if f >= 51)
    log(f"  Act 1 cleared (floor>=17): {act1}/{total_games}")
    log(f"  Act 2 cleared (floor>=34): {act2}/{total_games}")
    log(f"  Act 3 cleared (floor>=51): {act3}/{total_games}")
    log("=" * 60)

    # Final validation
    log("--- Final Validation ---")
    val_stats = run_validation(agent, n_games=VALIDATION_GAMES, ascension=ascension)
    log(f"  FINAL VAL: {val_stats['wins']}/{val_stats['n_games']} wins "
        f"({val_stats['win_rate']*100:.0f}%) "
        f"avg_floor={val_stats['avg_floor']:.1f}")

    # 保存最终 checkpoint
    save_checkpoint(agent, games_done, extra={"final": True})


if __name__ == "__main__":
    main()

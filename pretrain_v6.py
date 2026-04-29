"""
V6 预训练脚本 — TurnSolver 作为 combat teacher + 人类 .run 数据 behavior cloning
=============================================================================
Phase 1: 用 TurnSolver 跑 N 局游戏，收集 combat 决策样本
         每个样本 = (encoded tokens, hand_encodings, monster_encodings, masks,
                      solver_card_idx, solver_target_idx)
Phase 2: 用 cross-entropy 训练 combat head（card + target 两阶段）
Phase 3 (可选): 从 .run 文件预训练 draft/choice/path heads

Usage:
  cd $STSRLSOLVER_PATH
  # 一步到位：收集 + 训练
  uv run python3 <repo>/pretrain_v6.py \
      --collect-games 200 --epochs 10 --save sts_models/pretrain_v6.pt

  # 分步执行
  uv run python3 <repo>/pretrain_v6.py \
      --collect-only --n-games 200 --data-file combat_data.pkl
  uv run python3 <repo>/pretrain_v6.py \
      --train-only --data-file combat_data.pkl --epochs 10 --save sts_models/pretrain_v6.pt
"""

import sys
import time
import random
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sts_paths import ensure_on_sys_path
ensure_on_sys_path()

from packages.engine.game import GameRunner, GamePhase
from packages.engine.content.cards import ALL_CARDS, CardType, CardRarity, CardTarget
from packages.training.turn_solver import TurnSolverAdapter
from packages.engine.state.combat import PlayCard, EndTurn, UsePotion

import sts_agent_v6 as v6

BASE_DIR = Path(__file__).parent
CHECKPOINT_DIR = BASE_DIR / "sts_models"
CHECKPOINT_DIR.mkdir(exist_ok=True)

DEVICE = v6.DEVICE
print(f"[pretrain_v6] Device: {DEVICE}")


# ============================================================
# 从 train_v6.py 复用：simulator state → agent state 桥接
# ============================================================

# 直接从 train_v6 导入 build_agent_state 和 card_id_to_agent_dict
# 但因为 train_v6 顶部有 multiprocessing.set_start_method，不能直接 import
# 所以我们自己内联关键函数

ROOM_TYPE_TO_SYMBOL = {
    "MONSTER":  "M",
    "ELITE":    "E",
    "REST":     "R",
    "SHOP":     "$",
    "EVENT":    "?",
    "TREASURE": "T",
    "BOSS":     "B",
}


def card_id_to_agent_dict(card_id: str, upgraded: bool = False) -> dict:
    """从 simulator card ID 构建 v6 agent 期望的 card dict。"""
    base_id = card_id.rstrip("+") if card_id.endswith("+") else card_id
    upgraded = upgraded or card_id.endswith("+")

    card_def = ALL_CARDS.get(base_id)
    if card_def is None:
        return {
            "id": card_id, "name": card_id, "type": "SKILL",
            "cost": 1, "damage": 0, "block": 0, "rarity": "COMMON",
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
        "id": base_id, "name": card_def.name, "type": ctype,
        "cost": cost, "damage": damage, "block": block, "rarity": rarity,
        "exhaust": card_def.exhaust, "ethereal": card_def.ethereal,
        "upgraded": upgraded,
    }


def build_agent_state(runner: GameRunner) -> dict:
    """将 simulator 状态转换为 v6 agent 期望的 game_state 格式。
    精简版 — 只需要 combat 相关字段用于预训练。
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
        "MAP_NAVIGATION": "MAP", "COMBAT": "NONE",
        "COMBAT_REWARDS": "COMBAT_REWARD", "BOSS_REWARDS": "BOSS_REWARD",
        "EVENT": "EVENT", "SHOP": "SHOP_SCREEN", "REST": "REST",
        "TREASURE": "CHEST", "NEOW": "EVENT",
        "RUN_COMPLETE": "GAME_OVER", "GRID_SELECT": "GRID",
        "HAND_SELECT": "HAND_SELECT", "CARD_SELECT": "GRID",
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

        monsters = []
        for e in cs.enemies:
            is_alive = e.is_alive() if callable(getattr(e, "is_alive", None)) else getattr(e, "is_alive", True)
            enemy_powers = []
            statuses = getattr(e, "statuses", {})
            for k, v_amt in statuses.items():
                enemy_powers.append({"id": k, "name": k, "amount": v_amt})

            move_damage = max(e.move_damage or 0, 0)
            move_hits = e.move_hits or 1
            intent_str = getattr(e, "intent", None) or ""
            if not intent_str:
                intent_str = "ATTACK" if move_damage > 0 else "UNKNOWN"

            monsters.append({
                "id": getattr(e, "id", "?"), "name": getattr(e, "name", "?"),
                "current_hp": e.hp, "max_hp": e.max_hp, "block": e.block,
                "is_gone": not is_alive,
                "halfDead": getattr(e, "half_dead", False),
                "intent": intent_str,
                "move_adjusted_damage": move_damage,
                "move_hits": move_hits,
                "powers": enemy_powers,
            })

        hand = []
        for i, card_str in enumerate(cs.hand):
            cid = str(card_str)
            upgraded = cid.endswith("+")
            cd = card_id_to_agent_dict(cid, upgraded)
            base_id = cid.rstrip("+") if cid.endswith("+") else cid
            card_def = ALL_CARDS.get(base_id)
            if card_def is not None:
                cd["has_target"] = card_def.target in (CardTarget.ENEMY, CardTarget.SELF_AND_ENEMY)
            else:
                cd["has_target"] = False

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

        player_powers = []
        player_statuses = getattr(player, "statuses", {})
        for k, v_amt in player_statuses.items():
            player_powers.append({"id": k, "name": k, "amount": v_amt})

        combat_state = {
            "player": {
                "current_hp": player.hp, "max_hp": player.max_hp,
                "block": player.block, "energy": cs.energy,
                "powers": player_powers,
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

    # ----- 非 combat 阶段使用简化处理 -----
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

    if screen_type == "MAP":
        available_paths = runner.run_state.get_available_paths()
        nodes = []
        for node in available_paths:
            rt = node.room_type.name if node.room_type else "MONSTER"
            symbol = ROOM_TYPE_TO_SYMBOL.get(rt, "M")
            nodes.append({"symbol": symbol, "x": node.x, "y": node.y, "room_type": rt})
        screen_state["next_nodes"] = nodes
        available_commands.append("choose")

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

    if screen_type == "REST":
        rest_options = ["rest", "smith"]
        for opt in ("dig", "lift", "toke", "recall", "ruby_key"):
            if opt in action_type_names:
                rest_options.append(opt)
        screen_state["rest_options"] = rest_options
        available_commands.append("choose")

    if runner.game_over:
        screen_type = "GAME_OVER"
        available_commands = ["proceed"]

    if not available_commands:
        available_commands = ["choose"]

    gs = {
        "screen_type": screen_type,
        "current_hp": rs.current_hp, "max_hp": rs.max_hp,
        "gold": rs.gold, "floor": rs.floor, "act": rs.act,
        "ascension_level": rs.ascension,
        "deck": deck_cards, "relics": relics, "potions": potions,
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
# TurnSolver 动作 → V6 模型动作空间映射
# ============================================================

def solver_action_to_model_indices(engine_action, hand, monsters):
    """将 TurnSolver 返回的 engine-level Action 映射到 V6 模型的 action space。

    返回 (card_idx, target_idx, is_end_turn)
    card_idx: 0..MAX_HAND-1 = 打某张手牌, MAX_HAND = END TURN
    target_idx: 0..MAX_MONSTERS-1 = 目标怪物索引（无目标时为 0）
    """
    if isinstance(engine_action, EndTurn):
        return v6.END_TURN_ACTION, 0, True

    if isinstance(engine_action, PlayCard):
        card_idx = engine_action.card_idx
        target_idx = max(engine_action.target_idx, 0)  # -1 → 0
        # 确保索引在范围内
        if card_idx >= v6.MAX_HAND:
            return None, None, False  # 溢出，跳过
        return card_idx, target_idx, False

    # UsePotion / 其他 action — 预训练不关心药水决策
    return None, None, False


# ============================================================
# Phase 1: 数据收集 — 用 TurnSolver 跑游戏，记录 combat 样本
# ============================================================

# TurnSolver 预算（预训练用高精度 solver）
PRETRAIN_SOLVER_BUDGETS = {
    "monster": (100.0,   10_000,  300_000),
    "elite":   (5_000.0, 100_000, 600_000),
    "boss":    (50_000.0, 300_000, 600_000),
}


def collect_combat_data(n_games: int, verbose: bool = True) -> list:
    """跑 n_games 局游戏，用 TurnSolver 决策 combat，收集训练数据。

    每个样本是一个 dict:
    {
        "tokens": list of (name, type_id, tensor),  # detached CPU tensors
        "hand_encodings": Tensor,      # [n_hand, CTX_DIM]
        "monster_encodings": Tensor,   # [n_monster, CTX_DIM]
        "card_mask": Tensor,           # [NUM_CARD_ACTIONS]
        "monster_mask": Tensor,        # [MAX_MONSTERS]
        "card_target": int,            # solver 选的手牌索引 (0..MAX_HAND)
        "target_target": int,          # solver 选的目标索引 (0..MAX_MONSTERS-1)
        "needs_target": bool,          # 是否需要目标选择
    }
    """
    all_samples = []

    # 创建一个临时 model 用于编码（不训练）
    model = v6.STSModelV6().to(DEVICE)
    model.eval()

    for game_i in range(n_games):
        seed = random.randint(0, 2**31 - 1)
        t0 = time.time()

        runner = GameRunner(
            seed=str(seed),
            ascension=0,
            character="Watcher",
            verbose=False,
        )

        turn_solver = TurnSolverAdapter(
            time_budget_ms=100.0,
            node_budget=10_000,
            multi_turn_depth=3,
            multi_turn_k=3,
            multi_turn_budget_ms=30_000.0,
            solver_budgets=PRETRAIN_SOLVER_BUDGETS,
        )

        steps = 0
        max_steps = 8000
        game_samples = 0
        was_in_combat = False

        while not runner.game_over and steps < max_steps:
            try:
                actions = runner.get_available_actions()
            except Exception:
                break
            if not actions:
                break

            phase = runner.phase

            if phase == GamePhase.COMBAT:
                if not was_in_combat:
                    was_in_combat = True
                    turn_solver.reset()

                # 1. 构建 V6 agent state
                try:
                    agent_state = build_agent_state(runner)
                except Exception:
                    runner.take_action(actions[0])
                    steps += 1
                    continue

                gs = agent_state.get("game_state", {})
                combat = gs.get("combat_state", {})
                hand = combat.get("hand", [])
                monsters = combat.get("monsters", [])
                available = agent_state.get("available_commands", [])

                if not hand and not combat:
                    runner.take_action(actions[0])
                    steps += 1
                    continue

                # 2. 用 TurnSolver 获取专家动作
                room_type = getattr(runner, "current_room_type", "monster")
                try:
                    solver_action_obj = turn_solver.pick_action(actions, runner, room_type)
                    if solver_action_obj is None:
                        solver_action_obj = actions[0]
                except Exception:
                    solver_action_obj = actions[0]

                # 从 solver 的 cached_plan 获取原始 engine action
                engine_action = None
                if turn_solver._cached_plan and turn_solver._cached_plan_index > 0:
                    plan_idx = turn_solver._cached_plan_index - 1
                    if plan_idx < len(turn_solver._cached_plan):
                        engine_action = turn_solver._cached_plan[plan_idx]

                # 如果 plan 里找不到，从 combat action 反推
                if engine_action is None:
                    at = getattr(solver_action_obj, "action_type", "")
                    if at == "end_turn":
                        engine_action = EndTurn()
                    elif at == "play_card":
                        ci = getattr(solver_action_obj, "card_idx", 0)
                        ti = getattr(solver_action_obj, "target_idx", -1)
                        engine_action = PlayCard(card_idx=ci, target_idx=ti)
                    elif at == "use_potion":
                        engine_action = UsePotion(
                            potion_idx=getattr(solver_action_obj, "potion_idx", 0),
                            target_idx=getattr(solver_action_obj, "target_idx", -1),
                        )

                if engine_action is None:
                    runner.take_action(solver_action_obj)
                    steps += 1
                    continue

                # 3. 映射到模型 action space
                card_target, target_target, is_end_turn = solver_action_to_model_indices(
                    engine_action, hand, monsters
                )

                if card_target is None:
                    # 不可映射的动作（药水等），跳过
                    runner.take_action(solver_action_obj)
                    steps += 1
                    continue

                # 4. 编码当前状态（与 combat_act 完全一致）
                try:
                    tokens = _build_tokens_standalone(model, agent_state, include_combat=True)

                    with torch.no_grad():
                        # 手牌编码
                        hand_encs = []
                        for c in hand[:v6.MAX_HAND]:
                            card_vec = v6.encode_card(c)
                            t = torch.FloatTensor(card_vec).to(DEVICE)
                            enc = model.ctx_transformer.token_proj.project(
                                "card", v6.TOKEN_HAND_CARD, t)
                            hand_encs.append(enc)

                        if hand_encs:
                            hand_encodings = torch.stack(hand_encs)
                        else:
                            hand_encodings = torch.zeros(0, v6.CTX_DIM, device=DEVICE)

                        # 怪物编码
                        monster_encs = []
                        alive_monsters = []
                        for m in monsters[:v6.MAX_MONSTERS]:
                            enemy_vec = v6.encode_enemy(m)
                            t = torch.FloatTensor(enemy_vec).to(DEVICE)
                            enc = model.ctx_transformer.token_proj.project(
                                "enemy", v6.TOKEN_MONSTER, t)
                            monster_encs.append(enc)
                            alive_monsters.append(
                                not m.get("is_gone", True) and m.get("current_hp", 0) > 0
                            )

                        if monster_encs:
                            monster_encodings = torch.stack(monster_encs)
                        else:
                            monster_encodings = torch.zeros(0, v6.CTX_DIM, device=DEVICE)

                    # Card mask
                    card_mask = torch.zeros(v6.NUM_CARD_ACTIONS, device=DEVICE)
                    for i, c in enumerate(hand[:v6.MAX_HAND]):
                        if c.get("is_playable", False):
                            card_mask[i] = 1.0
                    if "end" in available:
                        card_mask[v6.END_TURN_ACTION] = 1.0
                    if card_mask.sum() == 0:
                        card_mask[v6.END_TURN_ACTION] = 1.0

                    # Monster mask
                    monster_mask = torch.zeros(v6.MAX_MONSTERS, device=DEVICE)
                    for i, al in enumerate(alive_monsters):
                        if al:
                            monster_mask[i] = 1.0

                    # 判断 solver 选的卡是否需要目标
                    needs_target = False
                    if not is_end_turn and card_target < len(hand):
                        needs_target = hand[card_target].get("has_target", False)

                    # 验证 solver 选的卡确实可打出（mask 检查）
                    if card_mask[card_target] == 0:
                        # solver 选了一张不可打出的卡，跳过这个样本
                        runner.take_action(solver_action_obj)
                        steps += 1
                        continue

                    # 5. 保存样本（CPU tensors）
                    sample = {
                        "tokens": [(name, ttype, t.detach().cpu())
                                   for name, ttype, t in tokens],
                        "hand_encodings": hand_encodings.detach().cpu(),
                        "monster_encodings": monster_encodings.detach().cpu(),
                        "card_mask": card_mask.detach().cpu(),
                        "monster_mask": monster_mask.detach().cpu(),
                        "card_target": card_target,
                        "target_target": target_target,
                        "needs_target": needs_target,
                    }
                    all_samples.append(sample)
                    game_samples += 1

                except Exception as e:
                    if verbose and game_i < 3:
                        print(f"  [WARN] 编码失败: {e}")

                # 执行 solver 选的动作
                try:
                    runner.take_action(solver_action_obj)
                except Exception:
                    break
                steps += 1

            else:
                # 非 combat — 随机决策（预训练只关注 combat）
                if was_in_combat:
                    was_in_combat = False

                # 简单启发式：优先推进游戏
                try:
                    runner.take_action(actions[0])
                except Exception:
                    break
                steps += 1

        elapsed = time.time() - t0
        floor = runner.run_state.floor
        hp = runner.run_state.current_hp
        if verbose:
            print(f"  Game {game_i+1}/{n_games}: floor={floor}, hp={hp}, "
                  f"samples={game_samples}, time={elapsed:.1f}s, "
                  f"total_samples={len(all_samples)}")

    print(f"\n[收集完成] {n_games} 局游戏, 共 {len(all_samples)} 个 combat 样本")
    return all_samples


def _build_tokens_standalone(model, game_state, include_combat=False):
    """独立版 _build_tokens — 不依赖 AgentV6 实例"""
    gs = game_state.get("game_state", {})
    tokens = []

    # 逐卡 token
    deck = gs.get("deck", [])
    for card in deck[:v6.MAX_DECK_CARDS]:
        card_vec = v6.encode_card(card)
        tokens.append(("card", v6.TOKEN_DECK_CARD, torch.FloatTensor(card_vec).to(DEVICE)))

    # Run progress
    run_prog = v6.encode_run_progress(game_state)
    tokens.append(("run_progress", v6.TOKEN_RUN_PROGRESS, torch.FloatTensor(run_prog).to(DEVICE)))

    # 药水槽
    potions = gs.get("potions", [])
    for i in range(v6.MAX_POTION_SLOTS):
        potion_data = potions[i] if i < len(potions) else None
        potion_vec = v6.encode_potion_entity(potion_data)
        tokens.append(("potion", v6.TOKEN_POTION, torch.FloatTensor(potion_vec).to(DEVICE)))

    # 遗物汇总
    relics = gs.get("relics", [])
    if relics:
        relic_vecs = np.array([v6.encode_relic_entity(r) for r in relics], dtype=np.float32)
        relic_summary = relic_vecs.mean(axis=0)
    else:
        relic_summary = np.zeros(v6.ENTITY_DIM, dtype=np.float32)
    tokens.append(("relic", v6.TOKEN_RELIC_SUMMARY, torch.FloatTensor(relic_summary).to(DEVICE)))

    if include_combat:
        combat = gs.get("combat_state", {})
        if combat:
            player_vec = v6.encode_player_state(combat)
            tokens.append(("player", v6.TOKEN_PLAYER, torch.FloatTensor(player_vec).to(DEVICE)))

            for m in combat.get("monsters", [])[:v6.MAX_MONSTERS]:
                enemy_vec = v6.encode_enemy(m)
                tokens.append(("enemy", v6.TOKEN_MONSTER, torch.FloatTensor(enemy_vec).to(DEVICE)))

            for c in combat.get("hand", [])[:v6.MAX_HAND]:
                hand_vec = v6.encode_card(c)
                tokens.append(("card", v6.TOKEN_HAND_CARD, torch.FloatTensor(hand_vec).to(DEVICE)))

            draw_pile = combat.get("draw_pile", gs.get("deck", []))
            if draw_pile:
                draw_summary = v6.encode_pile_summary(draw_pile)
                tokens.append(("pile_summary", v6.TOKEN_DRAW_SUMMARY,
                               torch.FloatTensor(draw_summary).to(DEVICE)))

            discard = combat.get("discard_pile", [])
            if discard:
                disc_summary = v6.encode_pile_summary(discard)
                tokens.append(("pile_summary", v6.TOKEN_DISCARD_SUMMARY,
                               torch.FloatTensor(disc_summary).to(DEVICE)))

    return tokens


# ============================================================
# Phase 2: 训练 combat head — cross-entropy
# ============================================================

def train_combat_head(
    samples: list,
    model: v6.STSModelV6,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 3e-4,
    verbose: bool = True,
) -> dict:
    """用收集的 TurnSolver 样本训练 combat head。

    双阶段 loss:
    1. Card selection: cross-entropy(card_logits, solver_card)
    2. Target selection: cross-entropy(target_logits, solver_target)  仅对 needs_target 样本

    Returns:
        训练统计 dict
    """
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    n_total = len(samples)
    print(f"\n[训练] {n_total} 个样本, {epochs} epochs, batch_size={batch_size}")

    # 统计 needs_target 比例
    n_target = sum(1 for s in samples if s["needs_target"])
    print(f"  需要目标选择的样本: {n_target}/{n_total} ({100*n_target/max(n_total,1):.1f}%)")

    best_card_acc = 0.0
    stats_history = []

    for epoch in range(epochs):
        t0 = time.time()
        random.shuffle(samples)

        total_card_loss = 0.0
        total_target_loss = 0.0
        total_card_correct = 0
        total_target_correct = 0
        total_card_samples = 0
        total_target_samples = 0

        # 逐样本处理（因为 token 序列长度不同，无法真正 batch transformer）
        # 但我们可以累积梯度实现等效 batch
        optimizer.zero_grad()
        accum_count = 0

        for i, sample in enumerate(samples):
            # 将 tokens 移到 GPU
            tokens = [(name, ttype, t.to(DEVICE)) for name, ttype, t in sample["tokens"]]
            hand_enc = sample["hand_encodings"].to(DEVICE)
            monster_enc = sample["monster_encodings"].to(DEVICE)
            card_mask = sample["card_mask"].to(DEVICE)
            monster_mask = sample["monster_mask"].to(DEVICE)
            card_target = sample["card_target"]
            target_target = sample["target_target"]
            needs_target = sample["needs_target"]

            # Forward: shared_repr
            shared_repr = model.ctx_transformer(tokens)

            # Combat head forward
            card_logits, target_logits_raw, _ = model.combat_head(
                shared_repr, hand_enc, monster_enc, card_mask, monster_mask
            )

            # Card loss
            card_label = torch.tensor(card_target, dtype=torch.long, device=DEVICE)
            card_loss = F.cross_entropy(card_logits.unsqueeze(0), card_label.unsqueeze(0))

            # Target loss（仅对需要目标的样本）
            target_loss = torch.tensor(0.0, device=DEVICE)
            if needs_target and card_target < hand_enc.shape[0]:
                card_encoding = hand_enc[card_target]
                target_logits = model.combat_head.forward_with_card(
                    shared_repr, card_encoding, monster_enc, monster_mask
                )
                target_label = torch.tensor(target_target, dtype=torch.long, device=DEVICE)
                target_loss = F.cross_entropy(target_logits.unsqueeze(0), target_label.unsqueeze(0))

            loss = card_loss + target_loss
            # 梯度累积：除以 batch_size
            (loss / batch_size).backward()
            accum_count += 1

            # 统计
            total_card_loss += card_loss.item()
            total_card_samples += 1
            if card_logits.argmax().item() == card_target:
                total_card_correct += 1

            if needs_target:
                total_target_loss += target_loss.item()
                total_target_samples += 1
                if needs_target and card_target < hand_enc.shape[0]:
                    if target_logits.argmax().item() == target_target:
                        total_target_correct += 1

            # 每 batch_size 个样本更新一次
            if accum_count >= batch_size:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                optimizer.zero_grad()
                accum_count = 0

        # 处理剩余梯度
        if accum_count > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            optimizer.zero_grad()

        # 统计
        elapsed = time.time() - t0
        card_acc = total_card_correct / max(total_card_samples, 1)
        target_acc = total_target_correct / max(total_target_samples, 1)
        avg_card_loss = total_card_loss / max(total_card_samples, 1)
        avg_target_loss = total_target_loss / max(total_target_samples, 1)

        best_card_acc = max(best_card_acc, card_acc)

        stats = {
            "epoch": epoch + 1,
            "card_loss": avg_card_loss,
            "target_loss": avg_target_loss,
            "card_acc": card_acc,
            "target_acc": target_acc,
            "time": elapsed,
        }
        stats_history.append(stats)

        if verbose:
            print(f"  Epoch {epoch+1}/{epochs}: "
                  f"card_loss={avg_card_loss:.4f}, card_acc={card_acc:.1%}, "
                  f"target_loss={avg_target_loss:.4f}, target_acc={target_acc:.1%}, "
                  f"time={elapsed:.1f}s")

    return {
        "best_card_acc": best_card_acc,
        "final_card_acc": stats_history[-1]["card_acc"] if stats_history else 0,
        "final_target_acc": stats_history[-1]["target_acc"] if stats_history else 0,
        "epochs": stats_history,
    }


# ============================================================
# 保存 / 加载
# ============================================================

def save_checkpoint(model: v6.STSModelV6, path: str):
    """保存为与 train_v6.py 兼容的格式"""
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict()}, save_path)
    print(f"[保存] 模型已保存到 {save_path}")


def save_data(samples: list, path: str):
    """保存收集的数据到 pickle 文件"""
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(samples, f)
    print(f"[保存] {len(samples)} 个样本已保存到 {save_path}")


def load_data(path: str) -> list:
    """从 pickle 文件加载数据"""
    with open(path, "rb") as f:
        samples = pickle.load(f)
    print(f"[加载] 从 {path} 加载了 {len(samples)} 个样本")
    return samples


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="V6 预训练: TurnSolver combat teacher")

    # 模式选择
    parser.add_argument("--collect-only", action="store_true",
                        help="只收集数据，不训练")
    parser.add_argument("--train-only", action="store_true",
                        help="只训练，从已有数据文件加载")

    # 数据收集参数
    parser.add_argument("--collect-games", type=int, default=200,
                        help="一步模式的收集游戏数 (默认 200)")
    parser.add_argument("--n-games", type=int, default=200,
                        help="分步模式的收集游戏数 (默认 200)")

    # 训练参数
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数 (默认 10)")
    parser.add_argument("--batch-size", type=int, default=64, help="梯度累积 batch 大小 (默认 64)")
    parser.add_argument("--lr", type=float, default=3e-4, help="学习率 (默认 3e-4)")

    # IO 参数
    parser.add_argument("--data-file", type=str, default="combat_data.pkl",
                        help="数据文件路径 (默认 combat_data.pkl)")
    parser.add_argument("--save", type=str, default="sts_models/pretrain_v6.pt",
                        help="模型保存路径 (默认 sts_models/pretrain_v6.pt)")
    parser.add_argument("--load", type=str, default=None,
                        help="加载已有模型继续训练")

    args = parser.parse_args()

    # 决定工作模式
    if args.collect_only:
        # 只收集数据
        print(f"[模式] 仅收集数据: {args.n_games} 局游戏")
        samples = collect_combat_data(args.n_games, verbose=True)
        save_data(samples, args.data_file)
        return

    if args.train_only:
        # 只训练
        print(f"[模式] 仅训练: 从 {args.data_file} 加载数据")
        samples = load_data(args.data_file)
        model = v6.STSModelV6().to(DEVICE)
        if args.load:
            print(f"[加载] 从 {args.load} 加载模型权重")
            v6.load_model_safe(model, args.load)
        stats = train_combat_head(
            samples, model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )
        save_checkpoint(model, args.save)
        print(f"\n[结果] best_card_acc={stats['best_card_acc']:.1%}, "
              f"final_target_acc={stats['final_target_acc']:.1%}")
        return

    # 一步模式：收集 + 训练
    print(f"[模式] 收集 + 训练: {args.collect_games} 局游戏, {args.epochs} epochs")

    # Phase 1: 收集
    samples = collect_combat_data(args.collect_games, verbose=True)

    if not samples:
        print("[错误] 没有收集到任何样本!")
        return

    # 保存数据（备份）
    data_path = str(Path(args.save).parent / "pretrain_v6_data.pkl")
    save_data(samples, data_path)

    # Phase 2: 训练
    model = v6.STSModelV6().to(DEVICE)
    if args.load:
        print(f"[加载] 从 {args.load} 加载模型权重")
        v6.load_model_safe(model, args.load)

    stats = train_combat_head(
        samples, model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    # Phase 3: 保存
    save_checkpoint(model, args.save)

    print(f"\n{'='*60}")
    print(f"[预训练完成]")
    print(f"  游戏数: {args.collect_games}")
    print(f"  样本数: {len(samples)}")
    print(f"  最佳 card accuracy: {stats['best_card_acc']:.1%}")
    print(f"  最终 card accuracy: {stats['final_card_acc']:.1%}")
    print(f"  最终 target accuracy: {stats['final_target_acc']:.1%}")
    print(f"  模型已保存到: {args.save}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

"""V8 ↔ CommunicationMod state adapter（MVP）。

把 CommunicationMod 的游戏状态 JSON 反射成 V8 model 看得懂的形态：
    - V8State（v8/state.py）
    - available_actions list[str]（与 v8/action_space.py 字符串格式对齐）

并提供 model 选完 idx 后回写 mod command 的反向工具。

设计原则：
    - 不引入 bottled_ai 的整套 handler 体系，只复用 client.py 的 stdio 协议
    - action 字符串尽量与 v8/action_space._format_engine_action 同格式
      （让 hash → token id 与训练时一致）
    - 不强求 100% 覆盖 STS 所有 corner case；遇到不认识的 screen 返回空 list，
      由 driver 兜底（proceed / cancel）
    - 与训练时差异较大的字段（如 deck_strength）填 None，模型 forward 已兜底

非目标（MVP 不做）：
    - HAND_SELECT / GRID select / mass discard / scry 等复杂 sub-screen
    - 多选 reward（buy multiple shop items in one turn）
    - 战斗内 potion 使用

CommunicationMod JSON schema 参考：bottled_ai/rs/machine/state.py + handlers/*。
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from v8.state import V8State


# =============================================================================
# Phase 推断
# =============================================================================

# CommunicationMod screen_type → V8 phase
_SCREEN_TYPE_TO_PHASE = {
    "MAP": "MAP",
    "EVENT": "EVENT",  # NEOW 由 event_id 再细分
    "CARD_REWARD": "CARD_REWARDS",
    "COMBAT_REWARD": "CARD_REWARDS",  # combat reward 整体当 CARD_REWARDS 处理
    "BOSS_REWARD": "BOSS_REWARDS",
    "REST": "REST",
    "CHEST": "TREASURE",
    "SHOP_ROOM": "SHOP",
    "SHOP_SCREEN": "SHOP",
    # 卡牌选择子画面（事件 transform / shop purge / Astrolabe / 战斗内 Discovery 等）：
    # 训练时 StSRLSolver 在 runner 内部处理，对 RL 不暴露；实机走 mod 必须有 handler。
    "GRID": "GRID_SELECT",
    "HAND_SELECT": "HAND_SELECT",
    "GAME_OVER": "",
    "COMPLETE": "",
    "NONE": "",
}


def _infer_phase(mod_state: Dict[str, Any]) -> str:
    """从 mod_state 推断 V8 phase。

    优先级：
        1. combat_state 存在 → COMBAT
        2. screen_type=EVENT 且 event_id="Neow Event" → NEOW
        3. screen_type 查表
    """
    gs = mod_state.get("game_state", {}) or {}
    if gs.get("combat_state"):
        return "COMBAT"

    screen_type = gs.get("screen_type", "") or ""
    if screen_type == "EVENT":
        ss = gs.get("screen_state", {}) or {}
        if ss.get("event_id") == "Neow Event":
            return "NEOW"
        return "EVENT"
    return _SCREEN_TYPE_TO_PHASE.get(screen_type, "")


# =============================================================================
# Map 转换
# =============================================================================


def _build_map_nodes(
    map_list: List[Dict[str, Any]],
) -> Tuple[List[List[Dict[str, Any]]], int]:
    """mod map list → (map_nodes, max_floor)。

    mod schema:
        list of {"x": int, "y": int, "symbol": str, "children": [{"x": int, "y": int}]}
        y 是 floor（0-based），x 是层内位置

    V8 schema:
        map_nodes[floor_idx] = list of {"room_type": str, "x": int, "edges": [next_x]}
    """
    if not map_list:
        return ([], 0)

    max_y = max(int(r.get("y", 0)) for r in map_list)
    layers: List[List[Dict[str, Any]]] = [[] for _ in range(max_y + 1)]
    for r in map_list:
        y = int(r.get("y", 0))
        x = int(r.get("x", 0))
        symbol = r.get("symbol", "_") or "_"
        edges = [int(c.get("x", 0)) for c in (r.get("children") or [])]
        layers[y].append({"room_type": symbol, "x": x, "edges": edges})
    return (layers, max_y)


# =============================================================================
# Deck / relics / potions / monsters 转换
# =============================================================================


def _convert_deck(deck_json: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for c in deck_json or []:
        out.append({
            "name": c.get("id") or c.get("name") or "",
            "upgraded": int(c.get("upgrades", 0) or 0) > 0,
        })
    return out


def _convert_relics(relics_json: List[Dict[str, Any]]) -> List[str]:
    return [str(r.get("id") or r.get("name") or "") for r in (relics_json or [])]


def _convert_potions(potions_json: List[Dict[str, Any]]) -> List[str]:
    """mod 药水槽 → list[str]（空槽用 "" 占位，与 V8 一致）。"""
    out: List[str] = []
    for p in potions_json or []:
        pid = p.get("id") or ""
        # mod 用 "Potion Slot" 表示空槽
        if pid == "Potion Slot":
            out.append("")
        else:
            out.append(pid)
    return out


def _convert_monsters(monsters_json: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for m in monsters_json or []:
        if m.get("is_gone"):
            continue
        intent_dmg = int(m.get("move_adjusted_damage", -1) or -1)
        if intent_dmg < 0:
            intent_dmg = 0
        out.append({
            "name": m.get("id") or m.get("name") or "",
            "hp": int(m.get("current_hp", 0) or 0),
            "max_hp": int(m.get("max_hp", 0) or 0),
            "intent_dmg": intent_dmg,
            "intent_hits": int(m.get("move_hits", 0) or 0),
            "block": int(m.get("block", 0) or 0),
        })
    return out


def _convert_hand(hand_json: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for c in hand_json or []:
        out.append({
            "name": c.get("id") or c.get("name") or "",
            "upgraded": int(c.get("upgrades", 0) or 0) > 0,
            "cost": int(c.get("cost", 0) or 0),
        })
    return out


# =============================================================================
# 主：mod JSON → V8State
# =============================================================================


def mod_json_to_v8_state(mod_state: Dict[str, Any]) -> V8State:
    """从 CommunicationMod state dict 构造 V8State。

    Args:
        mod_state: mod 完整 JSON dict（含 "game_state" / "in_game" / ...）

    Returns:
        V8State，可直接喂给 V8Model.forward(state, available_actions)。
    """
    gs = mod_state.get("game_state", {}) or {}
    phase = _infer_phase(mod_state)

    hp = int(gs.get("current_hp", 0) or 0)
    max_hp = int(gs.get("max_hp", 0) or 0)
    floor = int(gs.get("floor", 0) or 0)
    act = int(gs.get("act", 1) or 1)
    gold = int(gs.get("gold", 0) or 0)

    deck = _convert_deck(gs.get("deck", []) or [])
    relics = _convert_relics(gs.get("relics", []) or [])
    potions = _convert_potions(gs.get("potions", []) or [])

    map_nodes, _ = _build_map_nodes(gs.get("map", []) or [])

    # 当前位置：mod 的 screen_state.current_node 才有；走完 map 后位置在 floor=y 的 x 上
    cur_pos: Optional[Dict[str, int]] = None
    ss = gs.get("screen_state", {}) or {}
    if "current_node" in ss and ss["current_node"]:
        cn = ss["current_node"]
        cur_pos = {"floor": int(cn.get("y", -1)), "x": int(cn.get("x", -1))}
    elif gs.get("floor", 0):
        # 不在 map screen 但有 floor → 从 game_state 反查上一次落点
        # mod 没直接给 last_node 字段，floor-1 不一定准；MVP 留空
        cur_pos = None

    combat_state = gs.get("combat_state") or {}
    in_combat = bool(combat_state)
    hand = _convert_hand(combat_state.get("hand", []) or []) if in_combat else []
    monsters = _convert_monsters(combat_state.get("monsters", []) or []) if in_combat else []
    energy = int((combat_state.get("player") or {}).get("energy", 0) or 0) if in_combat else 0

    # boss name: mod 没直接给 act_boss；从 boss key 找（部分 mod 版本有 game_state.boss）
    boss_name = ""
    if "boss" in gs:
        boss_name = str(gs.get("boss") or "")
    # 也可从 map 找 B symbol（保守不做，留空）

    state = V8State(
        hp=hp,
        max_hp=max_hp,
        floor=floor,
        act=act,
        gold=gold,
        potions=potions,
        deck=deck,
        relics=relics,
        map_nodes=map_nodes,
        current_position=cur_pos,
        deck_strength=None,
        in_combat=in_combat,
        hand=hand,
        draw_pile=[],   # MVP：不传 draw / discard / exhaust（model 不用）
        discard_pile=[],
        exhaust_pile=[],
        energy=energy,
        monsters=monsters,
        phase=phase,
        boss=boss_name,
    )
    return state


# =============================================================================
# Available actions 列表构造（与 V8 action 字符串格式对齐）
# =============================================================================


def mod_json_to_available_actions(
    mod_state: Dict[str, Any],
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """枚举当前 phase 的合法 V8 action 字符串 list + 对应 metadata。

    返回的 list 顺序固定（idx 稳定），metadata 与 action[i] 一一对应，
    存 mod 命令所需的额外信息（如 choice_list index、card name 等），
    避免 v8_idx_to_mod_command 重复解析。

    Returns:
        (action_strings, metas)
    """
    gs = mod_state.get("game_state", {}) or {}
    phase = _infer_phase(mod_state)
    actions: List[str] = []
    metas: List[Dict[str, Any]] = []

    if phase == "COMBAT":
        combat_state = gs.get("combat_state") or {}
        hand = combat_state.get("hand", []) or []
        monsters = combat_state.get("monsters", []) or []
        player_energy = int((combat_state.get("player") or {}).get("energy", 0) or 0)
        alive_monster_indices = [
            i for i, m in enumerate(monsters) if not m.get("is_gone")
        ]

        # 每张可打的手牌 × target（无目标的卡 target=-1）
        for ci, card in enumerate(hand):
            cost = int(card.get("cost", 0) or 0)
            playable = bool(card.get("is_playable", True))
            # 不可打的卡 / energy 不够 → skip
            if not playable or (cost >= 0 and cost > player_energy):
                continue
            if card.get("has_target"):
                for ti in alive_monster_indices:
                    actions.append(
                        f"COMBAT:play_card(card={ci},target={ti},potion=-1)"
                    )
                    metas.append({
                        "kind": "play_card",
                        "hand_idx": ci,
                        "monster_idx": ti,
                    })
            else:
                actions.append(
                    f"COMBAT:play_card(card={ci},target=-1,potion=-1)"
                )
                metas.append({
                    "kind": "play_card",
                    "hand_idx": ci,
                    "monster_idx": None,
                })

        # End turn 始终可选
        actions.append("COMBAT:end_turn(card=-1,target=-1,potion=-1)")
        metas.append({"kind": "end_turn"})
        return actions, metas

    # 元决策 phase 默认走 choice_list（mod 的 choose <idx> 体系）
    choice_list: List[str] = list(gs.get("choice_list", []) or [])
    ss = gs.get("screen_state", {}) or {}

    if phase == "NEOW":
        for i, text in enumerate(choice_list):
            actions.append(f"NEOW:choice={i}")
            metas.append({"kind": "choose", "choose_idx": i, "text": text})
        return actions, metas

    if phase == "MAP":
        # mod choice_list 里 map 节点是字符串如 "x=1 R" / "x=3 ?"；choose i 即可
        for i, text in enumerate(choice_list):
            # 节点 x 从 text 抽（用于诊断字符串，model 用 idx 即可）
            actions.append(f"MAP:node={i}")
            metas.append({"kind": "choose", "choose_idx": i, "text": text})
        return actions, metas

    if phase == "EVENT":
        event_id = ss.get("event_id", "?") or "?"
        for i, text in enumerate(choice_list):
            # 与 v8 action_space _format_engine_action 同格式：EVENT:<event_id>:<choice_name>:choice=<i>
            choice_name = re.sub(r"\s+", "_", str(text).strip().lower())
            actions.append(f"EVENT:{event_id}:{choice_name}:choice={i}")
            metas.append({"kind": "choose", "choose_idx": i, "text": text})
        return actions, metas

    if phase == "CARD_REWARDS":
        screen_type = gs.get("screen_type", "")
        if screen_type == "CARD_REWARD":
            # screen_state.cards 是候选卡
            cards = ss.get("cards", []) or []
            for i, c in enumerate(cards):
                cid = c.get("id") or c.get("name") or "?"
                actions.append(f"REWARD:card:{cid}:choice={i}")
                metas.append({
                    "kind": "choose",
                    "choose_idx": i,
                    "text": cid,
                })
            # skip 选项（mod 用 proceed 或 skip）
            if "skip" in choice_list:
                skip_idx = choice_list.index("skip")
                actions.append("REWARD:skip:choice=0")
                metas.append({
                    "kind": "choose",
                    "choose_idx": skip_idx,
                    "text": "skip",
                })
            return actions, metas

        # COMBAT_REWARD：choice_list 里有 gold / relic / card / potion 等
        for i, text in enumerate(choice_list):
            tlow = str(text).lower()
            if "card" in tlow:
                actions.append(f"REWARD:card:choice={i}")
            elif "relic" in tlow:
                actions.append(f"REWARD:relic:choice={i}")
            elif "potion" in tlow:
                actions.append(f"REWARD:potion:choice={i}")
            elif "gold" in tlow:
                actions.append(f"REWARD:gold:choice={i}")
            else:
                actions.append(f"REWARD:other:{tlow}:choice={i}")
            metas.append({"kind": "choose", "choose_idx": i, "text": text})
        # 加 proceed 作为"全拿完了"的退出动作
        actions.append("REWARD:proceed")
        metas.append({"kind": "proceed"})
        return actions, metas

    if phase == "BOSS_REWARDS":
        relics = ss.get("relics", []) or []
        for i, r in enumerate(relics):
            rid = r.get("id") or r.get("name") or "?"
            actions.append(f"BOSS:{rid}:relic={i}")
            metas.append({"kind": "choose", "choose_idx": i, "text": rid})
        # skip 作为最后选项
        actions.append("BOSS:skip:relic=-1")
        metas.append({"kind": "skip"})
        return actions, metas

    if phase == "REST":
        for i, text in enumerate(choice_list):
            tlow = str(text).lower()
            actions.append(f"REST:{tlow}(card=-1)")
            metas.append({"kind": "choose", "choose_idx": i, "text": text})
        return actions, metas

    if phase == "TREASURE":
        # mod chest screen: 通常单选 "open"；choice_list 可能空，要用 choose 0 / proceed
        if choice_list:
            for i, text in enumerate(choice_list):
                actions.append(f"TREASURE:{str(text).lower()}")
                metas.append({"kind": "choose", "choose_idx": i, "text": text})
        actions.append("TREASURE:skip")
        metas.append({"kind": "proceed"})
        return actions, metas

    if phase == "SHOP":
        screen_type = gs.get("screen_type", "")
        if screen_type == "SHOP_ROOM":
            # 入口：只有 "shop"（或类似）→ choose shop 进店
            for i, text in enumerate(choice_list):
                actions.append(f"SHOP:enter:item={i}")
                metas.append({"kind": "choose", "choose_idx": i, "text": text})
            return actions, metas
        # SHOP_SCREEN：买卡 / 买药水 / 买 relic / purge / leave
        # 不够金币的项依然能出现在 choice_list（mod 不过滤），我们用 gold 自筛
        gold = int(gs.get("gold", 0) or 0)
        ss_shop = gs.get("screen_state", {}) or {}
        # 收集所有 item 的价格：mod 把 cards / relics / potions / purge 分键
        # 每个 entry 一般有 {"price": int, "id": str/...}
        prices_by_text: Dict[str, int] = {}
        for key in ("cards", "relics", "potions"):
            for it in ss_shop.get(key, []) or []:
                name = str(it.get("id") or it.get("name") or "").strip()
                p = int(it.get("price", 0) or 0)
                if name:
                    prices_by_text[name.lower()] = p
        # purge_cost 用 mod 顶层字段
        purge_cost = int(ss_shop.get("purge_cost", 0) or 0)
        for i, text in enumerate(choice_list):
            tlow = str(text).lower()
            if "leave" in tlow:
                actions.append("SHOP:leave")
                metas.append({"kind": "leave"})
                continue
            # 价格判定：能买得起才暴露
            price = None
            if "purge" in tlow:
                price = purge_cost
            else:
                # text 里可能带后缀（如 "strike#-#100"），尝试关键词命中
                for k, p in prices_by_text.items():
                    if k and (k in tlow or tlow in k):
                        price = p
                        break
            if price is not None and price > gold:
                # 买不起：不暴露，避免 model 选了执行不了卡死
                continue
            if "purge" in tlow:
                actions.append(f"SHOP:remove_card:item={i}")
                metas.append({"kind": "choose", "choose_idx": i, "text": text})
            else:
                actions.append(f"SHOP:buy:{tlow}:item={i}")
                metas.append({"kind": "choose", "choose_idx": i, "text": text})
        # 保险：若所有买项都过滤掉了，且 leave 也不在 choice_list，强加一个 leave 兜底
        if not actions:
            actions.append("SHOP:leave")
            metas.append({"kind": "leave"})
        return actions, metas

    if phase == "GRID_SELECT":
        # GRID 子画面：transform / upgrade / purge / Astrolabe / Falling 选弃牌等
        # screen_state.cards 是候选卡牌，num_cards 是要选几张（多选时 driver 多次进来）
        cards = ss.get("cards", []) or []
        if not cards:
            # fallback: choice_list（mod 可能把卡名放这里）
            cards = [{"id": t, "name": t} for t in choice_list]
        for i, c in enumerate(cards):
            cid = c.get("id") or c.get("name") or "?"
            up = int(c.get("upgrades", 0) or 0)
            tag = f"{cid}+{up}" if up > 0 else cid
            actions.append(f"GRID:card:{tag}:choice={i}")
            metas.append({"kind": "choose", "choose_idx": i, "text": tag})
        # 若该画面允许 confirm（一般是已选够卡 / 可 0 选），暴露 confirm 出口
        available_cmds = mod_state.get("available_commands", []) or []
        if "confirm" in available_cmds:
            actions.append("GRID:confirm")
            metas.append({"kind": "confirm"})
        return actions, metas

    if phase == "HAND_SELECT":
        # HAND_SELECT：战斗中 mass-discard / 战斗内手牌选择（如 Sentinel 弃牌、
        # Gambling Chip）。screen_state.cards 是手牌候选；can_pick_zero / max_cards 控制
        ss_hs = gs.get("screen_state", {}) or {}
        cards = ss_hs.get("cards", []) or []
        can_pick_zero = bool(ss_hs.get("can_pick_zero", False))
        for i, c in enumerate(cards):
            cid = c.get("id") or c.get("name") or "?"
            up = int(c.get("upgrades", 0) or 0)
            tag = f"{cid}+{up}" if up > 0 else cid
            actions.append(f"HAND:card:{tag}:choice={i}")
            metas.append({"kind": "choose", "choose_idx": i, "text": tag})
        # confirm 暴露条件：can_pick_zero（可 0 选直接确认），或已选够卡（mod 会把
        # confirm 加进 available_commands）
        available_cmds = mod_state.get("available_commands", []) or []
        if can_pick_zero or "confirm" in available_cmds:
            actions.append("HAND:confirm")
            metas.append({"kind": "confirm"})
        return actions, metas

    # 未知 phase：空 list，driver 兜底
    return actions, metas


# =============================================================================
# idx → mod command
# =============================================================================


def v8_idx_to_mod_command(
    idx: int,
    metas: List[Dict[str, Any]],
    mod_state: Dict[str, Any],
) -> str:
    """模型选了 idx → 转成 CommunicationMod 命令字符串。

    Args:
        idx: model argmax 输出的 action 索引
        metas: mod_json_to_available_actions 返回的 metadata
        mod_state: mod 当前 JSON（用于 hand size 等防越界）

    Returns:
        命令字符串（如 "choose 0" / "play 3 1" / "end" / "proceed"）
    """
    if idx < 0 or idx >= len(metas):
        return "proceed"  # 兜底
    meta = metas[idx]
    kind = meta.get("kind", "")

    if kind == "choose":
        return f"choose {int(meta['choose_idx'])}"
    if kind == "proceed":
        return "proceed"
    if kind == "skip":
        return "skip"
    if kind == "leave":
        return "leave"
    if kind == "end_turn":
        return "end"
    if kind == "confirm":
        return "confirm"
    if kind == "play_card":
        # mod 命令：play <hand_idx_1based> [monster_idx]
        hi = int(meta["hand_idx"]) + 1  # 1-indexed
        mi = meta.get("monster_idx")
        if mi is None or int(mi) < 0:
            return f"play {hi}"
        return f"play {hi} {int(mi)}"

    return "proceed"


__all__ = [
    "mod_json_to_v8_state",
    "mod_json_to_available_actions",
    "v8_idx_to_mod_command",
]

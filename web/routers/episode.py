"""单局回放 router（P3）。

Endpoints:
  GET /api/episode/list[?beat_boss=1][&limit=50][&sort=reward][&order=desc][&ep_min=0][&ep_max=100]
      返回 episode 列表，可选过滤胜局，按指定列排序
  GET /api/episode/{ep}/timeline
      返回 floor-by-floor 时间轴，combat/event/deck/decisions JOIN i18n 后中文化
  GET /api/episode/{ep}/combat/{act}/{floor}/detail
      单场战斗回合级详情
"""
from __future__ import annotations

import json
import re
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from ..db import get_conn

# ---- event_id 别名映射（DB 里的 event_id 与 i18n_entries.en_id 不一致的情况） ----
# DB event_logs 使用 CamelCase 无空格 ID（如 FountainOfCleansing），
# 但 i18n_entries 可能存的是带空格/撇号/不同拼写的 display name。
EVENT_ID_ALIASES: dict[str, str] = {
    "WomanInBlue": "The Woman in Blue",
    "FountainOfCleansing": "Fountain of Cleansing",
    "WorldOfGoop": "World of Goop",
    "Transmogrifier": "Transmorgrifier",  # i18n 源里的拼写（多了个 r）
    "GremlinMatchGame": "Match and Keep!",
    "GremlinWheelGame": "Wheel of Change",
    "Nloth": "N'loth",
    "TheLab": "Lab",
    "WingStatue": "Golden Wing",
    "Sssserpent": "Sssserpent",  # i18n 源缺失，保留原 ID 兜底
}

# 提取 REST option 里的卡名：REST:upgrade:Strike_R(card=0) -> Strike_R
_RE_REST_UPGRADE = re.compile(r"^REST:upgrade:(.+?)\(card=\d+\)$")
# 提取 CARD_REWARDS option 里的卡名：REWARD:card:Metallicize:choice=2 -> Metallicize
_RE_REWARD_CARD = re.compile(r"^REWARD:card:(.+?):choice=\d+$")
# 提取 REST chosen_card 里的卡名（去掉 (card=N) 后缀）
_RE_CARD_SLOT = re.compile(r"^(.+?)\(card=\d+\)$")
# 提取 SHOP chosen 里的物品：SHOP:buy_colored_card:Anger:item=0
_RE_SHOP_BUY = re.compile(r"^SHOP:buy_(\w+):(.+?):item=\d+$")
# 提取 SHOP remove：SHOP:remove_card:Strike_R:item=0
_RE_SHOP_REMOVE = re.compile(r"^SHOP:remove_card:(.+?):item=\d+$")
# 提取 CARD_REWARDS 中的遗物奖励：REWARD:relic:Whetstone:choice=0 -> Whetstone
_RE_REWARD_RELIC = re.compile(r"^REWARD:relic:(.+?):choice=\d+$")
# 提取 BOSS relic：BOSS:Empty Cage:relic=0
_RE_BOSS_RELIC = re.compile(r"^BOSS:(.+?):relic=\d+$")
# 提取 option 里的 choice=<N> 数字（用于 Prayer Wheel 双屏拆分）
_RE_CHOICE_IDX = re.compile(r":choice=(\d+)$")

router = APIRouter()


_SORT_COLS = {
    "ep": "ep",
    "reward": "reward",
    "floor_reached": "floor_reached",
    "steps": "steps",
    "secs": "secs",
}


@router.get("/list")
def episode_list(
    ep_min: int | None = Query(None, description="最小 ep（含）"),
    ep_max: int | None = Query(None, description="最大 ep（含）"),
    beat_boss: int | None = Query(None, description="0 / 1 过滤；省略=不过滤"),
    limit: int = Query(50, ge=1, le=2000),
    sort: str = Query("ep", description="排序列"),
    order: str = Query("asc", description="asc / desc"),
):
    """episode 列表"""
    if sort not in _SORT_COLS:
        raise HTTPException(400, f"invalid sort: {sort}; allowed={list(_SORT_COLS)}")
    order = order.lower()
    if order not in ("asc", "desc"):
        raise HTTPException(400, f"invalid order: {order}")

    where: list[str] = []
    params: list[Any] = []
    if ep_min is not None:
        where.append("ep >= ?")
        params.append(ep_min)
    if ep_max is not None:
        where.append("ep <= ?")
        params.append(ep_max)
    if beat_boss is not None:
        if beat_boss not in (0, 1):
            raise HTTPException(400, "beat_boss must be 0 or 1")
        where.append("beat_boss = ?")
        params.append(beat_boss)
    where_sql = (" WHERE " + " AND ".join(where)) if where else ""
    sort_col = _SORT_COLS[sort]

    conn = get_conn()
    try:
        rows = conn.execute(
            f"""
            SELECT ep, steps, reward, floor_reached, beat_boss,
                   secs, search_calls, eval_deck_calls
            FROM episodes{where_sql}
            ORDER BY {sort_col} {order}, ep ASC
            LIMIT ?
            """,
            (*params, limit),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def _load_i18n_dict(conn, kind: str) -> dict[str, dict]:
    """整 kind dump 到 dict（en_id -> entry），避免每个 timeline item 单 query。

    含 meta_json 解析后的 rarity（cards 才有），方便终局牌组按稀有度分组。
    """
    rows = conn.execute(
        "SELECT en_id, en_name, zh_name, zh_desc, meta_json FROM i18n_entries WHERE kind = ?",
        (kind,),
    ).fetchall()
    out: dict[str, dict] = {}
    for r in rows:
        rarity = None
        meta_raw = r["meta_json"]
        if meta_raw:
            try:
                d = json.loads(meta_raw)
                if isinstance(d, dict):
                    rarity = d.get("rarity")
            except (ValueError, TypeError):
                rarity = None
        out[r["en_id"]] = {
            "en_id": r["en_id"],
            "en_name": r["en_name"],
            "zh_name": r["zh_name"],
            "zh_desc": r["zh_desc"],
            "rarity": rarity,
        }
    return out


def _zh_label(table: dict[str, dict], en_id: str) -> str:
    """en_id -> zh_name 兜底链：zh_name -> en_name -> en_id"""
    entry = table.get(en_id)
    if entry:
        return entry.get("zh_name") or entry.get("en_name") or en_id
    return en_id


def _lookup_event_zh(events_table: dict[str, dict], event_id: str) -> str | None:
    """event_id -> zh_name，先直查再走别名映射。

    event_logs 里的 event_id 是 CamelCase（如 FountainOfCleansing），
    i18n_entries 里可能存的是带空格的 display name（如 Fountain of Cleansing）。
    """
    entry = events_table.get(event_id)
    if entry:
        return entry.get("zh_name") or entry.get("en_name") or event_id
    alias = EVENT_ID_ALIASES.get(event_id)
    if alias:
        entry = events_table.get(alias)
        if entry:
            return entry.get("zh_name") or entry.get("en_name") or event_id
    return None


def _lookup_card_zh(cards_table: dict[str, dict], card_en_id: str) -> str | None:
    """card_en_id -> zh_name，带升级后缀（+1 / +）剥离。"""
    if not card_en_id:
        return None
    entry = cards_table.get(card_en_id)
    if entry:
        return entry.get("zh_name") or entry.get("en_name") or card_en_id
    # 尝试剥离升级后缀再查
    stripped = card_en_id.rstrip("+")
    if stripped.endswith("+1"):
        stripped = stripped[:-2]
    if stripped != card_en_id:
        entry = cards_table.get(stripped)
        if entry:
            return entry.get("zh_name") or entry.get("en_name") or stripped
    return None


def _extract_card_name(raw: str) -> str | None:
    """从 REST chosen_card（如 Defend_R(card=7)）中提取纯卡名。"""
    if not raw:
        return None
    m = _RE_CARD_SLOT.match(raw)
    return m.group(1) if m else raw


def _get_choice_idx(raw_label: str) -> int:
    """从 'REWARD:card:Inflame:choice=0' 中提取 choice 数字，未找到返回 -1。"""
    m = _RE_CHOICE_IDX.search(raw_label)
    return int(m.group(1)) if m else -1


def _build_decision(row: dict, cards_table: dict[str, dict],
                    relics_table: dict[str, dict],
                    potions_table: dict[str, dict] | None = None) -> dict[str, Any]:
    """把一行 meta_decisions 构造成结构化 decision 对象。

    根据 phase 分别处理：
    - CARD_REWARDS: 解析可选卡牌、标注选了哪张 / 跳过
    - REST: 休息或升级某张卡
    - SHOP: 买卡/买遗物/买药水/移除卡/离开
    - BOSS_REWARDS: 选遗物
    - 其它（MAP/NEOW/TREASURE）: 原始 label 透传
    """
    phase = row["phase"]
    decision_type = row["decision_type"]
    chosen = row["chosen"]
    chosen_card = row["chosen_card"]
    options_raw = row["options_json"]

    try:
        options = json.loads(options_raw) if options_raw else []
    except (ValueError, TypeError):
        options = []

    out: dict[str, Any] = {
        "step": row["step"],
        "phase": phase,
        "type": decision_type,
        "chosen": chosen,
    }

    if phase == "CARD_REWARDS":
        # 判断是卡牌奖励还是遗物奖励（同一 phase 下可能有 relic 子决策）
        m_relic = _RE_REWARD_RELIC.match(chosen) if chosen else None
        if m_relic:
            # 这是遗物奖励子步骤（如战胜 elite 后的 relic reward）
            relic_name = m_relic.group(1)
            out["type"] = "relic_reward"
            out["chosen_relic"] = relic_name
            out["chosen_relic_zh"] = _zh_label(relics_table, relic_name)
        else:
            # 提取选中的卡名 + 中文
            if chosen_card:
                out["chosen_card"] = chosen_card
                out["chosen_card_zh"] = _lookup_card_zh(cards_table, chosen_card)
        # 构造 options 列表（卡牌 + 遗物分别提取）
        card_options: list[dict[str, Any]] = []
        for opt in options:
            m = _RE_REWARD_CARD.match(opt)
            if m:
                card_name = m.group(1)
                card_options.append({
                    "card": card_name,
                    "zh": _lookup_card_zh(cards_table, card_name),
                    "_choice_idx": _get_choice_idx(opt),
                })
            else:
                m_r = _RE_REWARD_RELIC.match(opt)
                if m_r:
                    rname = m_r.group(1)
                    card_options.append({
                        "relic": rname,
                        "zh": _zh_label(relics_table, rname),
                        "_choice_idx": _get_choice_idx(opt),
                    })

        # Prayer Wheel 双屏拆分：screen 1 有 choice=0,1,2; screen 2 有 choice=100,101,102
        # 当 options 含 choice>=100 的卡牌时，只保留 chosen 所属 screen 的选项
        has_screen2 = any(o.get("_choice_idx", -1) >= 100 for o in card_options)
        if has_screen2:
            chosen_choice_idx = _get_choice_idx(chosen) if chosen else -1
            if chosen_choice_idx >= 100:
                card_options = [o for o in card_options if o.get("_choice_idx", -1) >= 100]
            else:
                card_options = [o for o in card_options if o.get("_choice_idx", -1) < 100]

        # 清理内部 _choice_idx 字段，不暴露给前端
        for o in card_options:
            o.pop("_choice_idx", None)

        out["options"] = card_options

    elif phase == "REST":
        # 提取升级的卡名（chosen_card 带 (card=N) 后缀）
        card_name = _extract_card_name(chosen_card)
        if card_name and decision_type == "upgrade":
            out["chosen_card"] = card_name
            out["chosen_card_zh"] = _lookup_card_zh(cards_table, card_name)
        # 构造可升级卡列表（去重，同名卡可能有多个 slot）
        upgrade_options: list[dict[str, Any]] = []
        seen_cards: set[str] = set()
        for opt in options:
            m = _RE_REST_UPGRADE.match(opt)
            if m:
                cname = m.group(1)
                if cname not in seen_cards:
                    seen_cards.add(cname)
                    upgrade_options.append({
                        "card": cname,
                        "zh": _lookup_card_zh(cards_table, cname),
                    })
        if upgrade_options:
            out["options"] = upgrade_options

    elif phase == "SHOP":
        # 解析购买 / 移除 / 离开
        m_buy = _RE_SHOP_BUY.match(chosen) if chosen else None
        m_remove = _RE_SHOP_REMOVE.match(chosen) if chosen else None
        if m_buy:
            item_type = m_buy.group(1)  # colored_card / colorless_card / relic / potion
            item_name = m_buy.group(2)
            out["item_type"] = item_type
            if "card" in item_type:
                out["chosen_card"] = item_name
                out["chosen_card_zh"] = _lookup_card_zh(cards_table, item_name)
            elif item_type == "relic":
                out["chosen_relic"] = item_name
                out["chosen_relic_zh"] = _zh_label(relics_table, item_name)
            elif item_type == "potion":
                out["chosen_potion"] = item_name
                out["chosen_potion_zh"] = _zh_label(potions_table, item_name) if potions_table else item_name
        elif m_remove:
            card_name = m_remove.group(1)
            out["chosen_card"] = card_name
            out["chosen_card_zh"] = _lookup_card_zh(cards_table, card_name)
        # 构造商店货架选项（排除 leave）
        shop_items: list[dict[str, Any]] = []
        for opt in options:
            m_s = _RE_SHOP_BUY.match(opt)
            if m_s:
                s_type = m_s.group(1)
                s_name = m_s.group(2)
                item: dict[str, Any] = {"raw_type": s_type, "name": s_name}
                if "card" in s_type:
                    item["card"] = s_name
                    item["zh"] = _lookup_card_zh(cards_table, s_name) or s_name
                elif s_type == "relic":
                    item["relic"] = s_name
                    item["zh"] = _zh_label(relics_table, s_name)
                elif s_type == "potion":
                    item["potion"] = s_name
                    item["zh"] = _zh_label(potions_table, s_name) if potions_table else s_name
                shop_items.append(item)
            else:
                m_sr = _RE_SHOP_REMOVE.match(opt)
                if m_sr:
                    s_name = m_sr.group(1)
                    shop_items.append({
                        "raw_type": "remove",
                        "card": s_name,
                        "zh": _lookup_card_zh(cards_table, s_name) or s_name,
                    })
        if shop_items:
            out["options"] = shop_items

    elif phase == "BOSS_REWARDS":
        # 解析选中的遗物
        m_boss = _RE_BOSS_RELIC.match(chosen)
        if m_boss:
            relic_name = m_boss.group(1)
            out["chosen_relic"] = relic_name
            out["chosen_relic_zh"] = _zh_label(relics_table, relic_name)
        # 构造可选遗物列表
        relic_options: list[dict[str, Any]] = []
        for opt in options:
            m = _RE_BOSS_RELIC.match(opt)
            if m:
                rname = m.group(1)
                relic_options.append({
                    "relic": rname,
                    "zh": _zh_label(relics_table, rname),
                })
        if relic_options:
            out["options"] = relic_options

    # MAP / NEOW / TREASURE 等：只保留 chosen 原始 label，不做额外解析
    return out


@router.get("/{ep}/timeline")
def episode_timeline(ep: int):
    """单局完整时间轴。

    返回结构：
        {summary: {...}, timeline: [{floor, act, room, hp_cur, hp_max,
                                     combat?, event?, deck?}]}
    """
    conn = get_conn()
    try:
        # ---- summary ----
        ep_row = conn.execute(
            """
            SELECT ep, steps, reward, floor_reached, beat_boss, secs,
                   search_calls, eval_deck_calls
            FROM episodes
            WHERE ep = ?
            """,
            (ep,),
        ).fetchone()
        if ep_row is None:
            raise HTTPException(404, f"episode not found: ep={ep}")
        summary = dict(ep_row)

        # ---- floor_events（主时间轴）按 (act, floor) 排，跨 Act 时序正确 ----
        floor_rows = conn.execute(
            """
            SELECT floor, act, abs_floor, hp_cur, hp_max, room
            FROM floor_events
            WHERE ep = ?
            ORDER BY act ASC, floor ASC
            """,
            (ep,),
        ).fetchall()

        # ---- combats ----（按 (act, floor) 索引）
        combat_rows = conn.execute(
            """
            SELECT floor, act, abs_floor, room, enemies_json, hp_before, hp_after,
                   turn_actions, exit_reason
            FROM combats
            WHERE ep = ?
            """,
            (ep,),
        ).fetchall()
        combats_by_key = {(r["act"], r["floor"]): r for r in combat_rows}

        # ---- event_logs ----
        event_rows = conn.execute(
            """
            SELECT act, floor, abs_floor, event_id, event_phase, choices_json, chosen_idx,
                   chosen_text, exit_reason
            FROM event_logs
            WHERE ep = ?
            """,
            (ep,),
        ).fetchall()
        events_by_key = {(r["act"], r["floor"]): r for r in event_rows}

        # ---- deck_snapshots ----
        deck_snap_rows = conn.execute(
            """
            SELECT act, floor, abs_floor, room, cards_raw, relics_raw, deck_size
            FROM deck_snapshots
            WHERE ep = ?
            """,
            (ep,),
        ).fetchall()
        decks_by_key = {(r["act"], r["floor"]): r for r in deck_snap_rows}

        # ---- deck_cards 展平（按 (act, floor) 分组）----
        deck_card_rows = conn.execute(
            """
            SELECT act, floor, abs_floor, card_en_id, upgraded, count
            FROM deck_cards
            WHERE ep = ?
            """,
            (ep,),
        ).fetchall()
        deck_cards_by_key: dict[tuple[int, int], list[dict]] = {}
        for r in deck_card_rows:
            deck_cards_by_key.setdefault((r["act"], r["floor"]), []).append(
                {
                    "card_en_id": r["card_en_id"],
                    "upgraded": bool(r["upgraded"]),
                    "count": r["count"],
                }
            )

        # ---- meta_decisions（元决策，按 (act, floor) 分组）----
        meta_rows = conn.execute(
            """
            SELECT step, act, phase, floor, abs_floor, chosen, chosen_card, decision_type, options_json
            FROM meta_decisions WHERE ep = ? ORDER BY step ASC
            """,
            (ep,),
        ).fetchall()
        # 按 (act, floor) 分组，避免跨 Act 同 floor 碰撞
        meta_by_act_floor: dict[tuple[int, int], list] = {}
        for row in meta_rows:
            key = (row["act"], row["floor"])
            meta_by_act_floor.setdefault(key, []).append(row)

        # ---- i18n lookup tables ----
        monsters = _load_i18n_dict(conn, "monster")
        events = _load_i18n_dict(conn, "event")
        cards = _load_i18n_dict(conn, "card")
        relics = _load_i18n_dict(conn, "relic")
        potions = _load_i18n_dict(conn, "potion")

        # ---- floor_events 的 "下一行" 索引，用来给 combat 推算更可信的战后 HP ----
        # combats 表的 hp_before/hp_after 在历史日志里大量出现等值（疑似训练侧 logging bug，
        # 见诊断报告）。这里用 floor_events 的相邻两行 hp_cur 作为「战前/战后真实 HP」补充。
        ordered_floors = [(fr["act"], fr["floor"], fr["hp_cur"], fr["hp_max"]) for fr in floor_rows]
        next_hp_by_key: dict[tuple[int, int], tuple[int, int]] = {}
        for i in range(len(ordered_floors) - 1):
            a, fl, _hp, _hp_max = ordered_floors[i]
            na, nf, nhp, nhpmax = ordered_floors[i + 1]
            next_hp_by_key[(a, fl)] = (nhp, nhpmax)

        # ---- 组装 timeline ----（floor_rows 已按 (act, floor) 排序）
        timeline: list[dict[str, Any]] = []
        for fr in floor_rows:
            act = fr["act"]
            floor = fr["floor"]
            key = (act, floor)
            item: dict[str, Any] = {
                "floor": floor,
                "act": act,
                "abs_floor": fr["abs_floor"],
                "hp_cur": fr["hp_cur"],
                "hp_max": fr["hp_max"],
                "room": fr["room"],
            }

            # combat block（按 (act, floor) 配对）
            if key in combats_by_key:
                c = combats_by_key[key]
                try:
                    enemies_en = json.loads(c["enemies_json"] or "[]")
                except Exception:
                    enemies_en = []
                # 更可信的「战前 / 战后真实 HP」用 floor_events 行（当前行 hp_cur 是
                # 战前；下一个 floor_events 行 hp_cur 是战后；末尾行没有 next 兜底为
                # combat 自身 hp_after，可能不准但至少不为空）
                floor_hp_before = fr["hp_cur"]
                next_hp = next_hp_by_key.get(key)
                if next_hp is not None:
                    floor_hp_after, _ = next_hp
                else:
                    floor_hp_after = c["hp_after"]
                item["combat"] = {
                    "room": c["room"],
                    "enemies_en": enemies_en,
                    "enemies_zh": [_zh_label(monsters, e) for e in enemies_en],
                    # 原始 combat 表字段（保留兼容）
                    "hp_before": c["hp_before"],
                    "hp_after": c["hp_after"],
                    # 基于 floor_events 的真实战前 / 战后 HP
                    "floor_hp_before": floor_hp_before,
                    "floor_hp_after": floor_hp_after,
                    "hp_max": fr["hp_max"],
                    "turn_actions": c["turn_actions"],
                    "exit_reason": c["exit_reason"],
                }

            # event block（用别名感知的 _lookup_event_zh 修复 i18n key 不匹配）
            if key in events_by_key:
                e = events_by_key[key]
                try:
                    choices = json.loads(e["choices_json"] or "[]")
                except Exception:
                    choices = []
                event_zh = (
                    _lookup_event_zh(events, e["event_id"])
                    if e["event_id"]
                    else None
                )
                item["event"] = {
                    "event_id": e["event_id"],
                    "zh_name": event_zh or e["event_id"],
                    "event_phase": e["event_phase"],
                    "choices": choices,
                    "chosen_idx": e["chosen_idx"],
                    "chosen_text": e["chosen_text"],
                    "exit_reason": e["exit_reason"],
                }

            # deck block
            if key in decks_by_key:
                d = decks_by_key[key]
                card_list = deck_cards_by_key.get(key, [])
                # 排序：基础顺序按 count desc, 再按 zh_name
                cards_out = [
                    {
                        "card_en_id": c["card_en_id"],
                        "upgraded": c["upgraded"],
                        "count": c["count"],
                        "zh_name": _zh_label(cards, c["card_en_id"]),
                        "display": _zh_label(cards, c["card_en_id"])
                        + ("+1" if c["upgraded"] else ""),
                        "rarity": (cards.get(c["card_en_id"]) or {}).get("rarity"),
                    }
                    for c in sorted(
                        card_list,
                        key=lambda x: (-x["count"], x["card_en_id"]),
                    )
                ]
                # relics raw 字符串切分
                relics_raw = d["relics_raw"] or ""
                relics_en = [
                    r.strip() for r in relics_raw.split(",") if r.strip()
                ]
                relics_zh = [_zh_label(relics, r) for r in relics_en]
                item["deck"] = {
                    "room": d["room"],
                    "deck_size": d["deck_size"],
                    "cards": cards_out,
                    "relics_en": relics_en,
                    "relics_zh": relics_zh,
                }

            # decisions block（该楼层的元决策列表，按 step 排序）
            floor_meta = meta_by_act_floor.get(key, [])
            if floor_meta:
                # 过滤掉 MAP 决策（MAP 是选路，不属于当前楼层的"内容决策"）
                decisions = [
                    _build_decision(dict(m), cards, relics, potions)
                    for m in floor_meta
                    if m["phase"] != "MAP"
                ]
                if decisions:
                    item["decisions"] = decisions

            timeline.append(item)

        # ---- endgame 终局总结 ----
        endgame = _build_endgame(
            summary=summary,
            timeline=timeline,
            floor_rows=floor_rows,
            combat_rows=combat_rows,
            decks_by_key=decks_by_key,
            deck_cards_by_key=deck_cards_by_key,
            cards=cards,
            relics=relics,
            monsters=monsters,
        )

        return {
            "ep": ep,
            "summary": summary,
            "timeline": timeline,
            "endgame": endgame,
        }
    finally:
        conn.close()


# 终局总结面板数据构造
# 稀有度分组顺序（中文 label 在前端做，这里只用英文 key）
_RARITY_GROUP_ORDER = ("RARE", "UNCOMMON", "COMMON", "BASIC", "SPECIAL", "CURSE")


def _build_endgame(
    *,
    summary: dict,
    timeline: list[dict],
    floor_rows: list,
    combat_rows: list,
    decks_by_key: dict,
    deck_cards_by_key: dict,
    cards: dict,
    relics: dict,
    monsters: dict,
) -> dict:
    """构造 endgame 终局总结字段。

    - won: episodes.beat_boss==1
    - final_floor / final_act：最后一条 floor_events 行
    - final_hp / final_hp_max：最后一条 floor_events 行 hp_cur/hp_max
    - total_reward / total_steps：summary 字段
    - exit_reason: 'victory'（won）/'defeat'（失败）
    - death_at: 失败时取最后一个有 combat 的 floor 信息；通关时 None
    - final_deck: 取最后一行 deck_snapshots 对应的 deck_cards，按 rarity 分组
    - final_relics: 同 deck_snapshots 的 relics_raw
    """
    won = bool(summary.get("beat_boss"))
    total_reward = summary.get("reward")
    total_steps = summary.get("steps")

    # 最终楼层信息：取 floor_rows 最后一行
    final_floor: int | None = None
    final_act: int | None = None
    final_hp: int | None = None
    final_hp_max: int | None = None
    if floor_rows:
        last_fr = floor_rows[-1]
        final_floor = last_fr["floor"]
        final_act = last_fr["act"]
        final_hp = last_fr["hp_cur"]
        final_hp_max = last_fr["hp_max"]

    # 死亡现场：取最后一条 combat（按 timeline 顺序）的敌人 + floor 信息
    death_at: dict | None = None
    if not won and combat_rows:
        # combats 表没保证排序，按 timeline 顺序找最后一个有 combat 的 item
        last_combat_item = None
        for it in timeline:
            if it.get("combat"):
                last_combat_item = it
        if last_combat_item is not None:
            c = last_combat_item["combat"]
            death_at = {
                "act": last_combat_item["act"],
                "floor": last_combat_item["floor"],
                "room": c.get("room"),
                "enemies_en": c.get("enemies_en") or [],
                "enemies_zh": c.get("enemies_zh") or [],
                "exit_reason": c.get("exit_reason"),
            }

    # 最终牌组：取 deck_snapshots 最后一条
    final_deck: dict[str, list[dict]] = {k: [] for k in _RARITY_GROUP_ORDER}
    final_deck["UNKNOWN"] = []  # 兜底
    final_relics_out: list[dict] = []
    deck_size_final: int | None = None
    if decks_by_key:
        # 按 (act, floor) 升序取最后一个 key
        last_key = max(decks_by_key.keys())
        d_row = decks_by_key[last_key]
        deck_size_final = d_row["deck_size"]
        card_list = deck_cards_by_key.get(last_key, [])
        for c in card_list:
            zh = _zh_label(cards, c["card_en_id"])
            display = zh + ("+1" if c["upgraded"] else "")
            rarity = (cards.get(c["card_en_id"]) or {}).get("rarity")
            group = rarity if rarity in _RARITY_GROUP_ORDER else "UNKNOWN"
            final_deck[group].append({
                "card_en_id": c["card_en_id"],
                "upgraded": c["upgraded"],
                "count": c["count"],
                "zh_name": zh,
                "display": display,
                "rarity": rarity,
            })
        # 每组内按 display 排序
        for g in final_deck:
            final_deck[g].sort(key=lambda x: (-x["count"], x["display"]))
        # 遗物 JOIN i18n
        relics_raw = d_row["relics_raw"] or ""
        for raw in relics_raw.split(","):
            raw = raw.strip()
            if not raw:
                continue
            zh = _zh_label(relics, raw)
            final_relics_out.append({"en_id": raw, "display": zh})

    return {
        "won": won,
        "final_floor": final_floor,
        "final_act": final_act,
        "final_hp": final_hp,
        "final_hp_max": final_hp_max,
        "total_reward": total_reward,
        "total_steps": total_steps,
        "exit_reason": "victory" if won else ("defeat" if death_at else "incomplete"),
        "death_at": death_at,
        "final_deck": final_deck,
        "final_relics": final_relics_out,
        "deck_size": deck_size_final,
    }


@router.get("/{ep}/combat/{act}/{floor}/detail")
def combat_detail(ep: int, act: int, floor: int):
    """单场战斗的回合级详情。

    返回每回合的手牌 / 玩家状态 / 敌人状态（hp / intent / buffs） + 每次出牌 /
    每个 action 的伤害与战损（player_delta / enemy_delta raw text）。

    需要先用 web/etl/import_trace.py 导入 trace 数据；
    旧 batch 没有 turn 级数据，会返回 turns=[] 但
    combat 元信息（enemies, hp_before/after, turn_actions）仍然可用。
    """
    conn = get_conn()
    try:
        # combat row（必要）
        combat_row = conn.execute(
            """
            SELECT ep, act, floor, abs_floor, room, enemies_json,
                   hp_before, hp_after, turn_actions, exit_reason
            FROM combats
            WHERE ep = ? AND act = ? AND floor = ?
            """,
            (ep, act, floor),
        ).fetchone()
        if combat_row is None:
            raise HTTPException(
                404,
                f"combat not found: ep={ep} act={act} floor={floor}",
            )

        # turns
        turn_rows = conn.execute(
            """
            SELECT turn_num, energy, energy_max, hp_cur, hp_max, block,
                   player_buffs_raw, hand_raw, hand_json, enemies_state_json
            FROM combat_turns
            WHERE ep = ? AND act = ? AND floor = ?
            ORDER BY turn_num ASC
            """,
            (ep, act, floor),
        ).fetchall()

        # plays（按 turn_num + action_idx）
        play_rows = conn.execute(
            """
            SELECT turn_num, action_idx, action_type, card_en_id, hand_slot,
                   target_idx, target_en_id, player_delta_raw, enemy_delta_raw
            FROM combat_card_plays
            WHERE ep = ? AND act = ? AND floor = ?
            ORDER BY turn_num ASC, action_idx ASC
            """,
            (ep, act, floor),
        ).fetchall()

        # i18n
        cards = _load_i18n_dict(conn, "card")
        monsters = _load_i18n_dict(conn, "monster")

        try:
            enemies_en = json.loads(combat_row["enemies_json"] or "[]")
        except Exception:
            enemies_en = []

        # 按 turn 分组 plays
        plays_by_turn: dict[int, list[dict]] = {}
        for p in play_rows:
            d = dict(p)
            if d["card_en_id"]:
                d["card_zh"] = _zh_label(cards, d["card_en_id"])
            if d["target_en_id"]:
                d["target_zh"] = _zh_label(monsters, d["target_en_id"])
            plays_by_turn.setdefault(d["turn_num"], []).append(d)

        turns_out: list[dict] = []
        for t in turn_rows:
            try:
                hand = json.loads(t["hand_json"] or "[]")
            except Exception:
                hand = []
            for h in hand:
                if h.get("card_en_id"):
                    h["card_zh"] = _zh_label(cards, h["card_en_id"])
            try:
                enemies_state = json.loads(t["enemies_state_json"] or "[]")
            except Exception:
                enemies_state = []
            for e in enemies_state:
                if e.get("en_id"):
                    e["zh_name"] = _zh_label(monsters, e["en_id"])
            turns_out.append({
                "turn_num": t["turn_num"],
                "energy": t["energy"],
                "energy_max": t["energy_max"],
                "hp_cur": t["hp_cur"],
                "hp_max": t["hp_max"],
                "block": t["block"],
                "player_buffs_raw": t["player_buffs_raw"],
                "hand_raw": t["hand_raw"],
                "hand": hand,
                "enemies_state": enemies_state,
                "plays": plays_by_turn.get(t["turn_num"], []),
            })

        return {
            "ep": ep,
            "act": act,
            "floor": floor,
            "abs_floor": combat_row["abs_floor"],
            "room": combat_row["room"],
            "enemies_en": enemies_en,
            "enemies_zh": [_zh_label(monsters, e) for e in enemies_en],
            "hp_before": combat_row["hp_before"],
            "hp_after": combat_row["hp_after"],
            "turn_actions": combat_row["turn_actions"],
            "exit_reason": combat_row["exit_reason"],
            "turns": turns_out,
        }
    finally:
        conn.close()

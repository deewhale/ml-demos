"""牌组聚合 router（P5）。

Endpoints:
  GET /api/deck/winning[?floor_min=0][&ep_min=0][&ep_max=100]
      胜局牌组中所有 (card_en_id, upgraded) 的累积频次，含中文 display
  GET /api/deck/card_frequency[?floor_min=10][&beat_boss=1][&ep_min=0][&ep_max=100]
      过滤条件下的卡牌出现频次
  GET /api/deck/by_episode/{ep}
      单局所有 deck snapshot（每个保存点全牌组）
"""
from __future__ import annotations

import json
from collections import Counter

from fastapi import APIRouter, HTTPException, Query

from ..db import get_conn

router = APIRouter()


def _card_frequency_sql(
    where_extra: list[str],
    params: list,
    ep_min: int | None = None,
    ep_max: int | None = None,
) -> tuple[str, list]:
    """组装 deck_cards JOIN episodes + i18n_entries 的频次查询。"""
    if ep_min is not None:
        where_extra.append("dc.ep >= ?")
        params.append(ep_min)
    if ep_max is not None:
        where_extra.append("dc.ep <= ?")
        params.append(ep_max)
    where_sql = " AND ".join(where_extra)
    sql = f"""
        SELECT
            dc.card_en_id,
            dc.upgraded,
            SUM(dc.count) AS freq,
            i.zh_name,
            i.en_name,
            i.meta_json
        FROM deck_cards dc
        JOIN episodes e ON dc.ep = e.ep
        LEFT JOIN i18n_entries i ON i.kind='card' AND i.en_id = dc.card_en_id
        WHERE {where_sql}
        GROUP BY dc.card_en_id, dc.upgraded
        ORDER BY freq DESC, dc.card_en_id ASC
    """
    return sql, params


def _extract_rarity(meta_json: str | None) -> str | None:
    """meta_json TEXT -> rarity 字段（出错返回 None）。"""
    if not meta_json:
        return None
    try:
        d = json.loads(meta_json)
        if isinstance(d, dict):
            return d.get("rarity")
    except (ValueError, TypeError):
        return None
    return None


def _row_to_card_freq(r) -> dict:
    upgraded = bool(r["upgraded"])
    zh_name = r["zh_name"] or r["en_name"] or r["card_en_id"]
    display = f"{zh_name}+1" if upgraded else zh_name
    return {
        "card_en_id": r["card_en_id"],
        "upgraded": upgraded,
        "freq": r["freq"],
        "zh_name": r["zh_name"],
        "en_name": r["en_name"],
        "display": display,
        "rarity": _extract_rarity(r["meta_json"]),
    }


@router.get("/winning")
def winning_decks(
    floor_min: int = Query(0, ge=0, description="只统计 abs_floor>=floor_min 的快照（累计绝对楼层 0-51）"),
    ep_min: int | None = Query(None, description="最小 ep（含）"),
    ep_max: int | None = Query(None, description="最大 ep（含）"),
):
    """胜局牌组聚合：beat_boss=1 的所有 episode，abs_floor>=floor_min 的 deck snapshot。

    `floor_min` 自 2026-05-26 起为绝对楼层（act1=0-17, act2=18-34, act3=35-51），
    不再是 act 内 floor。"""
    conn = get_conn()
    try:
        where = [
            "e.beat_boss = 1",
            "dc.abs_floor >= ?",
        ]
        params: list = [floor_min]
        sql, params2 = _card_frequency_sql(where, params, ep_min=ep_min, ep_max=ep_max)
        rows = conn.execute(sql, params2).fetchall()
        return [_row_to_card_freq(r) for r in rows]
    finally:
        conn.close()


@router.get("/card_frequency")
def card_frequency(
    floor_min: int = Query(0, ge=0, description="abs_floor>=floor_min（累计绝对楼层 0-51）"),
    beat_boss: int | None = Query(
        None, description="0 / 1 / 省略=不过滤"
    ),
    ep_min: int | None = Query(None, description="最小 ep（含）"),
    ep_max: int | None = Query(None, description="最大 ep（含）"),
):
    """通用版聚合：可选 beat_boss / floor_min / ep 范围过滤

    `floor_min` 语义同 `/winning`：绝对楼层 0-51。"""
    if beat_boss is not None and beat_boss not in (0, 1):
        raise HTTPException(400, "beat_boss must be 0 or 1")
    conn = get_conn()
    try:
        where = ["dc.abs_floor >= ?"]
        params: list = [floor_min]
        if beat_boss is not None:
            where.append("e.beat_boss = ?")
            params.append(beat_boss)
        sql, params2 = _card_frequency_sql(where, params, ep_min=ep_min, ep_max=ep_max)
        rows = conn.execute(sql, params2).fetchall()
        return [_row_to_card_freq(r) for r in rows]
    finally:
        conn.close()


@router.get("/relic_frequency")
def relic_frequency(
    ep_min: int | None = Query(None, description="最小 ep（含）"),
    ep_max: int | None = Query(None, description="最大 ep（含）"),
    beat_boss: int | None = Query(
        None, description="0 / 1 / 省略=不过滤"
    ),
):
    """遗物出现频次：从 deck_snapshots 的 relics_raw 聚合，取每局最后一个快照。"""
    if beat_boss is not None and beat_boss not in (0, 1):
        raise HTTPException(400, "beat_boss must be 0 or 1")
    conn = get_conn()
    try:
        # 取每局最后一个 deck_snapshot（按 act DESC, floor DESC 取第一行）
        # 用子查询取每 ep 的 max(act*100+floor) 对应的 relics_raw
        where: list[str] = []
        params: list = []
        if ep_min is not None:
            where.append("ds.ep >= ?")
            params.append(ep_min)
        if ep_max is not None:
            where.append("ds.ep <= ?")
            params.append(ep_max)
        if beat_boss is not None:
            where.append("e.beat_boss = ?")
            params.append(beat_boss)

        where_sql = (" AND " + " AND ".join(where)) if where else ""

        # 每 ep 取进度最深的 snapshot（act*100+floor 最大）
        rows = conn.execute(
            f"""
            SELECT ds.ep, ds.relics_raw
            FROM deck_snapshots ds
            JOIN episodes e ON ds.ep = e.ep
            WHERE ds.relics_raw IS NOT NULL AND ds.relics_raw != ''{where_sql}
              AND (ds.act * 100 + ds.floor) = (
                  SELECT MAX(ds2.act * 100 + ds2.floor)
                  FROM deck_snapshots ds2
                  WHERE ds2.ep = ds.ep AND ds2.relics_raw IS NOT NULL AND ds2.relics_raw != ''
              )
            ORDER BY ds.ep
            """,
            params,
        ).fetchall()

        # 聚合遗物频次
        relic_counter: Counter = Counter()
        total_eps = 0
        seen_eps: set = set()
        for r in rows:
            ep_val = r["ep"]
            if ep_val in seen_eps:
                continue
            seen_eps.add(ep_val)
            total_eps += 1
            relics_raw = r["relics_raw"] or ""
            for relic in relics_raw.split(","):
                relic = relic.strip()
                if relic:
                    relic_counter[relic] += 1

        # i18n 查询
        relic_i18n = conn.execute(
            "SELECT en_id, en_name, zh_name FROM i18n_entries WHERE kind='relic'"
        ).fetchall()
        relic_zh_map = {
            r["en_id"]: (r["zh_name"] or r["en_name"] or r["en_id"]) for r in relic_i18n
        }

        # 排序输出
        result = []
        for relic_en, freq in relic_counter.most_common():
            result.append({
                "relic_en_id": relic_en,
                "display": relic_zh_map.get(relic_en, relic_en),
                "freq": freq,
                "total_eps": total_eps,
            })
        return result
    finally:
        conn.close()


@router.get("/by_episode/{ep}")
def deck_by_episode(ep: int):
    """单局所有 deck_snapshots（各 floor），附中文化的卡牌。"""
    conn = get_conn()
    try:
        snap_rows = conn.execute(
            """
            SELECT act, floor, abs_floor, room, cards_raw, relics_raw, deck_size
            FROM deck_snapshots
            WHERE ep = ?
            ORDER BY act ASC, floor ASC
            """,
            (ep,),
        ).fetchall()
        if not snap_rows:
            return []

        # 一次性把 deck_cards 拉全（含 meta_json -> rarity）
        card_rows = conn.execute(
            """
            SELECT dc.act, dc.floor, dc.card_en_id, dc.upgraded, dc.count,
                   i.zh_name, i.en_name, i.meta_json
            FROM deck_cards dc
            LEFT JOIN i18n_entries i ON i.kind='card' AND i.en_id = dc.card_en_id
            WHERE dc.ep = ?
            ORDER BY dc.act ASC, dc.floor ASC, dc.count DESC
            """,
            (ep,),
        ).fetchall()
        by_key: dict[tuple[int, int], list[dict]] = {}
        for r in card_rows:
            upgraded = bool(r["upgraded"])
            zh = r["zh_name"] or r["en_name"] or r["card_en_id"]
            display = f"{zh}+1" if upgraded else zh
            by_key.setdefault((r["act"], r["floor"]), []).append(
                {
                    "card_en_id": r["card_en_id"],
                    "upgraded": upgraded,
                    "count": r["count"],
                    "zh_name": r["zh_name"],
                    "display": display,
                    "rarity": _extract_rarity(r["meta_json"]),
                }
            )

        # relics i18n
        relic_rows = conn.execute(
            "SELECT en_id, zh_name, en_name FROM i18n_entries WHERE kind='relic'"
        ).fetchall()
        relic_zh = {r["en_id"]: (r["zh_name"] or r["en_name"] or r["en_id"]) for r in relic_rows}

        out = []
        for s in snap_rows:
            relics_raw = s["relics_raw"] or ""
            relics_en = [r.strip() for r in relics_raw.split(",") if r.strip()]
            relics_zh_list = [relic_zh.get(r, r) for r in relics_en]
            out.append(
                {
                    "act": s["act"],
                    "floor": s["floor"],
                    "abs_floor": s["abs_floor"],
                    "room": s["room"],
                    "deck_size": s["deck_size"],
                    "cards": by_key.get((s["act"], s["floor"]), []),
                    "relics_en": relics_en,
                    "relics_zh": relics_zh_list,
                }
            )
        return out
    finally:
        conn.close()

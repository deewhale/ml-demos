"""英中 mapping lookup router（P4）。

Endpoints:
  GET  /api/lookup?kind=card&id=Strike_R          单查询，支持 +1 升级后缀
  POST /api/lookup/batch                          [{"kind":"card","id":"..."}] 批查
  GET  /api/lookup/all                            一次拉 5 类（前端启动缓存用）
  GET  /api/lookup/all?kind=card                  单类 dump（旧 P0 compat 行为）
  GET  /api/lookup/coverage                       每类 (total, with_zh) 统计

升级处理：
  - 卡 id 形如 'Strike_R+1' 表示升级版，display 拼 '打击+1'
  - regex strip '+1' 后缀，base lookup 用 stripped id，display 末尾追加 '+1'
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, HTTPException, Query

from ..db import get_conn

router = APIRouter()

# 5 类合法 kind
_VALID_KINDS = {"card", "relic", "monster", "event", "potion"}

# 升级后缀：'Strike_R+1' / 'Strike_R+'（两种习惯都接）
_UPGRADE_SUFFIX = re.compile(r"^(.+?)(?:\+1|\+)$")


def _strip_upgrade(raw_id: str) -> tuple[str, bool]:
    """剥离 '+1' / '+' 后缀。返回 (base_id, upgraded)."""
    if not raw_id:
        return raw_id, False
    m = _UPGRADE_SUFFIX.match(raw_id)
    if m:
        return m.group(1), True
    return raw_id, False


def _safe_parse_meta(meta_json: Any) -> Optional[Dict[str, Any]]:
    """meta_json TEXT → dict（坏数据返回 None）。"""
    if not meta_json:
        return None
    try:
        d = json.loads(meta_json)
        if isinstance(d, dict):
            return d
    except (ValueError, TypeError):
        return None
    return None


def _row_to_entry(row: Any) -> Dict[str, Any]:
    """sqlite Row → JSON dict（含 meta 解析后的常用字段：rarity / card_type / cost）。"""
    out: Dict[str, Any] = {
        "en_id": row["en_id"],
        "en_name": row["en_name"],
        "zh_name": row["zh_name"],
        "zh_desc": row["zh_desc"],
        "source": row["source"],
    }
    # meta_json 可能不在 row（部分 SELECT 没有取该列时），用 row.keys() 判一下
    try:
        meta = _safe_parse_meta(row["meta_json"])
    except (IndexError, KeyError):
        meta = None
    if meta:
        # 把常用字段平铺到顶层，方便前端直读
        for k in ("rarity", "card_type", "cost", "color"):
            if k in meta:
                out[k] = meta[k]
        out["meta"] = meta
    return out


def _lookup_one(conn, kind: str, raw_id: str) -> Dict[str, Any]:
    """单查询核心。带升级后缀处理。"""
    if kind not in _VALID_KINDS:
        return {"kind": kind, "en_id": raw_id, "display": raw_id, "found": False}
    base_id, upgraded = _strip_upgrade(raw_id)
    row = conn.execute(
        "SELECT en_id, en_name, zh_name, zh_desc, source, meta_json "
        "FROM i18n_entries WHERE kind=? AND en_id=?",
        (kind, base_id),
    ).fetchone()
    if row is None:
        # 找不到，display 兜底用 raw_id
        display = raw_id
        return {
            "kind": kind,
            "en_id": raw_id,
            "base_id": base_id,
            "upgraded": upgraded,
            "display": display,
            "found": False,
        }
    entry = _row_to_entry(row)
    label = entry["zh_name"] or entry["en_name"] or base_id
    display = f"{label}+1" if upgraded else label
    return {
        "kind": kind,
        "en_id": raw_id,
        "base_id": base_id,
        "upgraded": upgraded,
        "display": display,
        "found": True,
        **entry,
    }


@router.get("")
def lookup_single(
    kind: str = Query(..., description="card|relic|monster|event|potion"),
    id: str = Query(..., description="en_id, 支持 '+1' 升级后缀"),
):
    """单查询。"""
    if kind not in _VALID_KINDS:
        raise HTTPException(status_code=400, detail=f"invalid kind: {kind}")
    conn = get_conn()
    try:
        return _lookup_one(conn, kind, id)
    finally:
        conn.close()


@router.post("/batch")
def lookup_batch(items: List[Dict[str, str]] = Body(...)):
    """批量查询。body: [{"kind":"card","id":"Strike_R"}, ...]"""
    conn = get_conn()
    try:
        return [_lookup_one(conn, it.get("kind", ""), it.get("id", "")) for it in items]
    finally:
        conn.close()


@router.get("/all")
def lookup_all(kind: Optional[str] = Query(None, description="可省，省则返回 5 类合并")):
    """整 kind dump（前端缓存用）。

    - kind 省略 → 返回 {card: {en_id: entry}, relic: ..., ...}
    - kind 指定 → 返回 {en_id: entry}（dict，便于前端 O(1) 查）
    """
    conn = get_conn()
    try:
        if kind is None:
            # 全部 5 类（含 meta_json）
            rows = conn.execute(
                "SELECT kind, en_id, en_name, zh_name, zh_desc, source, meta_json "
                "FROM i18n_entries"
            ).fetchall()
            out: Dict[str, Dict[str, Dict[str, Any]]] = {k: {} for k in _VALID_KINDS}
            for r in rows:
                k = r["kind"]
                if k not in out:
                    continue
                out[k][r["en_id"]] = _row_to_entry(r)
            return out
        if kind not in _VALID_KINDS:
            raise HTTPException(status_code=400, detail=f"invalid kind: {kind}")
        rows = conn.execute(
            "SELECT en_id, en_name, zh_name, zh_desc, source, meta_json "
            "FROM i18n_entries WHERE kind=?",
            (kind,),
        ).fetchall()
        return {r["en_id"]: _row_to_entry(r) for r in rows}
    finally:
        conn.close()


@router.get("/coverage")
def lookup_coverage():
    """每类 (total, with_zh) 统计。"""
    conn = get_conn()
    try:
        rows = conn.execute(
            "SELECT kind, COUNT(*) AS total, "
            "SUM(CASE WHEN zh_name IS NOT NULL AND zh_name != '' THEN 1 ELSE 0 END) AS with_zh "
            "FROM i18n_entries GROUP BY kind ORDER BY kind"
        ).fetchall()
        result = {}
        for r in rows:
            total = r["total"] or 0
            with_zh = r["with_zh"] or 0
            ratio = (with_zh / total) if total else 0.0
            result[r["kind"]] = {"total": total, "with_zh": with_zh, "ratio": round(ratio, 4)}
        return result
    finally:
        conn.close()

"""把训练日志解析入库到 episodes / floor_events / combats / event_logs /
deck_snapshots / deck_cards 6 张表（每次幂等：先 DELETE 再 INSERT）。

用法：
    python -m web.etl.parse_log                       # 所有有 log_path 的源
    python -m web.etl.parse_log --source_key v15      # 单源
    python -m web.etl.parse_log --force                # 同义于默认（全量）

设计要点：
- 流式逐行读，不 read 整文件
- combat enter / exit、event enter / choice / exit 三种 marker 用 dict 缓存配对，
  exit 时合并写入；只有 enter 没 exit 也允许 partial insert（缺字段 NULL）
- 2026-05-20: act 维度修复。STS 日志里 floor 每个 Act 重置 0-17，五张表 PK 必须
  含 act。combat exit / deck / event exit 这些 marker 本身不带 act，靠 per-ep
  act 状态机回填（取该 ep 最近一次 [floor] / [combat enter] / [event enter] 的 act）
- 2026-05-25: v3 schema 重写——去掉所有 batch_id 列，PK 以 ep 为核心。
  幂等 DELETE 改用 ep IN (...) 集合。
- 单事务 + executemany 批量写入，最后一次 commit
- 每条入库源末打印各表写入行数 + 总耗时
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

from ..db import get_conn, init_db
from . import markers as M


def _abs_floor(floor: int, act: int) -> int:
    """跨 act 绝对楼层。公式: (act-1)*17 + floor

    与 web/etl/import_trace.py 中的同名 helper 等价，且与训练日志里 `[floor]`
    marker 的 floor 字段语义一致——保持 0-index 还是 1-index 取决于上游 marker
    自身（不在这里 +1）。
    """
    return (act - 1) * 17 + floor


def _ensure_abs_floor_columns(conn) -> None:
    """幂等给 6 张表加 abs_floor INTEGER 列。

    新建库由 schema.sql 直接含 abs_floor，不会重复加；旧库通过 PRAGMA table_info
    检测后 ALTER TABLE 补列。参考 web/etl/import_trace.py `_ensure_schema` 的模式。
    """
    cur = conn.cursor()
    targets = (
        "floor_events",
        "combats",
        "event_logs",
        "deck_snapshots",
        "deck_cards",
        "meta_decisions",
    )
    for tbl in targets:
        existing = {r[1] for r in cur.execute(f"PRAGMA table_info({tbl})").fetchall()}
        if "abs_floor" not in existing:
            cur.execute(f"ALTER TABLE {tbl} ADD COLUMN abs_floor INTEGER")
    conn.commit()


# --- 内存收集容器 ---

class _BatchAccum:
    """单源解析过程中的内存累积容器，最后一次性 INSERT。

    所有 list tuple 都是 (ep, act, floor, ...)。
    act 状态机：_current_act_by_ep 记录每个 ep 最近一次出现的 act（来自 [floor]
    或 [combat enter] 或 [event enter]），其他不带 act 的 marker 用此回填。
    """

    def __init__(self) -> None:
        self.episodes: dict[int, dict] = {}  # ep -> heartbeat dict
        # tuple 内顺序：(ep, act, floor, ...其他列...)
        self.floor_events: list[tuple] = []
        self.combats: list[tuple] = []
        self.events: list[tuple] = []
        self.deck_snapshots: list[tuple] = []
        self.deck_cards: list[tuple] = []
        # meta_decisions: (ep, step, act, phase, floor, chosen, chosen_card, decision_type, options_json)
        self.meta_decisions: list[tuple] = []

        # 配对缓存：key 用 (ep, act, floor)
        self._combat_enter: dict[tuple[int, int, int], dict] = {}
        self._event_enter: dict[tuple[int, int, int], dict] = {}
        self._event_choice: dict[tuple[int, int, int], dict] = {}
        # 用于 deck PK 去重（同 ep/act/floor/room 多次 [deck] 取最后一次）
        # 暂时不做覆盖（DB 端 INSERT OR REPLACE 兜底）

        # act 状态机：ep -> 最近见到的 act 值（默认 1）
        self._current_act_by_ep: dict[int, int] = {}

    # ---- act 状态机 helper ----
    def _set_act(self, ep: int, act: int) -> None:
        self._current_act_by_ep[ep] = act

    def _get_act(self, ep: int) -> int:
        """拿当前 ep 的 act。无记录则兜底 1（Act 起点）。"""
        return self._current_act_by_ep.get(ep, 1)

    # -- heartbeat: episode 维度，按 ep 覆盖 --
    def add_heartbeat(self, d: dict) -> None:
        # heartbeat 自带 act（最终 ep 状态），顺便更新状态机（兜底）
        self._set_act(d["ep"], d["act"])
        self.episodes[d["ep"]] = d

    # -- floor: 主要的 act 源头 --
    def add_floor(self, d: dict) -> None:
        ep = d["ep"]
        self._set_act(ep, d["act"])
        # tuple: (ep, act, floor, abs_floor, hp_cur, hp_max, room)
        self.floor_events.append(
            (ep, d["act"], d["floor"], _abs_floor(d["floor"], d["act"]),
             d["hp_cur"], d["hp_max"], d["room"])
        )

    # -- combat enter / exit --
    def add_combat_enter(self, d: dict) -> None:
        ep = d["ep"]
        self._set_act(ep, d["act"])
        self._combat_enter[(ep, d["act"], d["floor"])] = d

    def add_combat_exit(self, d: dict) -> None:
        ep = d["ep"]
        floor = d["floor"]
        # combat exit marker 本身不带 act，用状态机回填
        act = self._get_act(ep)
        key = (ep, act, floor)
        enter = self._combat_enter.pop(key, None)
        # 兼容：如果 act 在 enter / exit 之间被推进（极少见），回退试别的 act
        if enter is None:
            for fallback_key in list(self._combat_enter.keys()):
                if fallback_key[0] == ep and fallback_key[2] == floor:
                    enter = self._combat_enter.pop(fallback_key)
                    act = fallback_key[1]
                    break
        room = enter["room"] if enter else None
        if enter is not None:
            act = enter["act"]  # enter 自带 act，最准
        enemies_json = json.dumps(enter["enemies"], ensure_ascii=False) if enter else None
        hp_before = (enter or {}).get("hp_before", d.get("hp_before"))
        self.combats.append(
            (
                ep,
                act,
                floor,
                _abs_floor(floor, act),
                room,
                enemies_json,
                hp_before,
                d["hp_after"],
                d["turn_actions"],
                d["reason"],
            )
        )

    # -- event enter / choice / exit --
    def add_event_enter(self, d: dict) -> None:
        ep = d["ep"]
        self._set_act(ep, d["act"])
        self._event_enter[(ep, d["act"], d["floor"])] = d

    def add_event_choice(self, d: dict) -> None:
        ep = d["ep"]
        floor = d["floor"]
        # choice 也不带 act，回填
        act = self._get_act(ep)
        # 若 enter 已落，act 用 enter 的；否则用状态机
        for k in self._event_enter:
            if k[0] == ep and k[2] == floor:
                act = k[1]
                break
        self._event_choice[(ep, act, floor)] = d

    def add_event_exit(self, d: dict) -> None:
        ep = d["ep"]
        floor = d["floor"]
        act = self._get_act(ep)
        key = (ep, act, floor)
        enter = self._event_enter.pop(key, None)
        if enter is None:
            # 兜底：找同 (ep, floor) 不同 act 的 enter
            for fk in list(self._event_enter.keys()):
                if fk[0] == ep and fk[2] == floor:
                    enter = self._event_enter.pop(fk)
                    act = fk[1]
                    break
        choice = self._event_choice.pop(key, None)
        if choice is None:
            for ck in list(self._event_choice.keys()):
                if ck[0] == ep and ck[2] == floor:
                    choice = self._event_choice.pop(ck)
                    break
        if enter is not None:
            act = enter["act"]
        event_id = (enter or {}).get("event_id") or d.get("event_id")
        event_phase = (enter or {}).get("event_phase")
        choices_json = json.dumps((enter or {}).get("choices") or [], ensure_ascii=False)
        chosen_idx = (choice or {}).get("choice_idx")
        chosen_text = (choice or {}).get("choice_text")
        self.events.append(
            (
                ep,
                act,
                floor,
                _abs_floor(floor, act),
                event_id,
                event_phase,
                choices_json,
                chosen_idx,
                chosen_text,
                d["reason"],
            )
        )

    # -- deck snapshot --
    def add_deck(self, d: dict) -> None:
        ep, floor, room = d["ep"], d["floor"], d["room"]
        # deck marker 不带 act，用状态机回填
        act = self._get_act(ep)
        cards_raw = " ".join(
            f"{c['card_en_id']}{'+1' if c['upgraded'] else ''}*{c['count']}"
            for c in d["cards"]
        )
        relics_raw = ",".join(d["relics"])
        deck_size = sum(c["count"] for c in d["cards"])
        abs_fl = _abs_floor(floor, act)
        self.deck_snapshots.append(
            (ep, act, floor, abs_fl, room, cards_raw, relics_raw, deck_size)
        )
        for c in d["cards"]:
            up = 1 if c["upgraded"] else 0
            self.deck_cards.append(
                (ep, act, floor, abs_fl, c["card_en_id"], up, c["count"])
            )

    # -- meta decision（经 postprocess 后直接入列）--
    def add_meta(self, d: dict) -> None:
        ep = d["ep"]
        # [meta] log 行不带 act，用 _current_act_by_ep 状态机回填
        act = d.get("act") or self._get_act(ep)
        self.meta_decisions.append(
            (
                ep,
                d["step"],
                act,
                d["phase"],
                d["floor"],
                _abs_floor(d["floor"], act),
                d["chosen"],
                d["chosen_card"],
                d["decision_type"],
                d["options_json"],
            )
        )

    # -- 收尾：把孤儿 enter（无 exit）也写一行 --
    def flush_orphans(self) -> None:
        for (ep, act, floor), enter in self._combat_enter.items():
            enemies_json = json.dumps(enter["enemies"], ensure_ascii=False)
            self.combats.append(
                (
                    ep,
                    act,
                    floor,
                    _abs_floor(floor, act),
                    enter["room"],
                    enemies_json,
                    enter.get("hp_before"),
                    None,
                    None,
                    "orphan_no_exit",
                )
            )
        self._combat_enter.clear()
        for (ep, act, floor), enter in self._event_enter.items():
            choice = None
            for ck in list(self._event_choice.keys()):
                if ck[0] == ep and ck[2] == floor:
                    choice = self._event_choice.pop(ck)
                    break
            choices_json = json.dumps(enter.get("choices") or [], ensure_ascii=False)
            chosen_idx = (choice or {}).get("choice_idx")
            chosen_text = (choice or {}).get("choice_text")
            self.events.append(
                (
                    ep,
                    act,
                    floor,
                    _abs_floor(floor, act),
                    enter["event_id"],
                    enter["event_phase"],
                    choices_json,
                    chosen_idx,
                    chosen_text,
                    "orphan_no_exit",
                )
            )
        self._event_enter.clear()
        self._event_choice.clear()


class LogParser:
    """单源日志解析 + 入库主类。"""

    def __init__(
        self, conn: sqlite3.Connection, log_path: str
    ) -> None:
        self.conn = conn
        self.log_path = log_path

    def parse_all(self) -> dict[str, int]:
        """流式解析全部日志，写入 6 表（先 DELETE 后 INSERT）。返回各表行数。"""
        t0 = time.time()
        accum = _BatchAccum()
        path = Path(self.log_path)
        if not path.exists():
            return {"error": -1, "reason": "log_not_found"}

        n_lines = 0
        n_matched = 0
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                n_lines += 1
                kw = M.classify_line(line)
                if kw is None:
                    continue
                matched = False
                if kw == "[heartbeat]":
                    m = M.HEARTBEAT_RE.search(line)
                    if m:
                        accum.add_heartbeat(M.parse_heartbeat(m))
                        matched = True
                elif kw == "[floor]":
                    m = M.FLOOR_RE.search(line)
                    if m:
                        accum.add_floor(M.parse_floor(m))
                        matched = True
                elif kw == "[combat]":
                    m = M.COMBAT_ENTER_RE.search(line)
                    if m:
                        accum.add_combat_enter(M.parse_combat_enter(m))
                        matched = True
                    else:
                        m = M.COMBAT_EXIT_RE.search(line)
                        if m:
                            accum.add_combat_exit(M.parse_combat_exit(m))
                            matched = True
                elif kw == "[event]":
                    m = M.EVENT_ENTER_RE.search(line)
                    if m:
                        accum.add_event_enter(M.parse_event_enter(m))
                        matched = True
                    else:
                        m = M.EVENT_CHOICE_RE.search(line)
                        if m:
                            accum.add_event_choice(M.parse_event_choice(m))
                            matched = True
                        else:
                            m = M.EVENT_PHASE_TRANS_RE.search(line)
                            if m:
                                matched = True
                            else:
                                m = M.EVENT_EXIT_RE.search(line)
                                if m:
                                    accum.add_event_exit(M.parse_event_exit(m))
                                    matched = True
                elif kw == "[deck]":
                    m = M.DECK_RE.search(line)
                    if m:
                        accum.add_deck(M.parse_deck(m))
                        matched = True
                elif kw == "[meta]":
                    m = M.META_RE.search(line)
                    if m:
                        raw = M.parse_meta(m)
                        processed = M.postprocess_meta(raw)
                        if processed is not None:
                            accum.add_meta(processed)
                        matched = True
                if matched:
                    n_matched += 1

        # 收尾孤儿
        accum.flush_orphans()

        # ---- DB 写入：单事务 ----
        cur = self.conn.cursor()

        # 幂等：先 DELETE（按 ep 集合）
        all_eps = set(accum.episodes.keys())
        # 也从其他表收集 ep（以防 heartbeat 缺失但有 floor/combat 等数据）
        for t in accum.floor_events:
            all_eps.add(t[0])
        for t in accum.combats:
            all_eps.add(t[0])
        for t in accum.events:
            all_eps.add(t[0])
        for t in accum.deck_snapshots:
            all_eps.add(t[0])
        for t in accum.deck_cards:
            all_eps.add(t[0])
        for t in accum.meta_decisions:
            all_eps.add(t[0])

        if all_eps:
            placeholders = ",".join("?" * len(all_eps))
            ep_list = list(all_eps)
            cur.execute(f"DELETE FROM meta_decisions WHERE ep IN ({placeholders})", ep_list)
            cur.execute(f"DELETE FROM deck_cards WHERE ep IN ({placeholders})", ep_list)
            cur.execute(f"DELETE FROM deck_snapshots WHERE ep IN ({placeholders})", ep_list)
            cur.execute(f"DELETE FROM event_logs WHERE ep IN ({placeholders})", ep_list)
            cur.execute(f"DELETE FROM combats WHERE ep IN ({placeholders})", ep_list)
            cur.execute(f"DELETE FROM floor_events WHERE ep IN ({placeholders})", ep_list)
            cur.execute(f"DELETE FROM episodes WHERE ep IN ({placeholders})", ep_list)

        # episodes（取每个 ep 的最后一条 heartbeat，已经覆盖在 dict 里）
        # floor_reached 存绝对楼层：(act - 1) * 17 + floor + 1
        # env 输出 floor 是 0-indexed（每 Act 0-16），STS 实际楼层 1-indexed：
        # act=1 floor=0 → 绝对 1（起始），act=1 floor=16 → 绝对 17（boss 层）
        # act=2 floor=0 → 绝对 18，act=2 floor=5 → 绝对 23
        # act=3 floor=0 → 绝对 35，act=3 floor=16 → 绝对 51
        ep_rows = [
            (
                d["ep"],
                d["steps"],
                d["reward"],
                (d.get("act", 1) - 1) * 17 + d["floor"] + 1,
                1 if d["beat_boss"] else 0,
                d["secs"],
                d["search_calls"],
                d["eval_deck_calls"],
            )
            for d in accum.episodes.values()
        ]
        if ep_rows:
            cur.executemany(
                """
                INSERT INTO episodes
                    (ep, steps, reward, floor_reached, beat_boss, secs,
                     search_calls, eval_deck_calls)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                ep_rows,
            )

        # floor_events: (ep, act, floor, abs_floor, hp_cur, hp_max, room)
        if accum.floor_events:
            cur.executemany(
                """
                INSERT OR REPLACE INTO floor_events
                    (ep, act, floor, abs_floor, hp_cur, hp_max, room)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                accum.floor_events,
            )

        # combats: (ep, act, floor, abs_floor, room, enemies_json, hp_before, hp_after, turn_actions, exit_reason)
        if accum.combats:
            cur.executemany(
                """
                INSERT OR REPLACE INTO combats
                    (ep, act, floor, abs_floor, room, enemies_json,
                     hp_before, hp_after, turn_actions, exit_reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                accum.combats,
            )

        # event_logs: (ep, act, floor, abs_floor, event_id, event_phase, choices_json, chosen_idx, chosen_text, exit_reason)
        if accum.events:
            cur.executemany(
                """
                INSERT OR REPLACE INTO event_logs
                    (ep, act, floor, abs_floor, event_id, event_phase, choices_json,
                     chosen_idx, chosen_text, exit_reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                accum.events,
            )

        # deck_snapshots: (ep, act, floor, abs_floor, room, cards_raw, relics_raw, deck_size)
        if accum.deck_snapshots:
            cur.executemany(
                """
                INSERT OR REPLACE INTO deck_snapshots
                    (ep, act, floor, abs_floor, room, cards_raw, relics_raw, deck_size)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                accum.deck_snapshots,
            )

        # deck_cards: (ep, act, floor, abs_floor, card_en_id, upgraded, count)
        if accum.deck_cards:
            cur.executemany(
                """
                INSERT OR REPLACE INTO deck_cards
                    (ep, act, floor, abs_floor, card_en_id, upgraded, count)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                accum.deck_cards,
            )

        # meta_decisions: (ep, step, act, phase, floor, abs_floor, chosen, chosen_card, decision_type, options_json)
        if accum.meta_decisions:
            cur.executemany(
                """
                INSERT OR REPLACE INTO meta_decisions
                    (ep, step, act, phase, floor, abs_floor, chosen, chosen_card, decision_type, options_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                accum.meta_decisions,
            )

        self.conn.commit()
        elapsed = time.time() - t0

        return {
            "lines_read": n_lines,
            "lines_matched": n_matched,
            "episodes": len(ep_rows),
            "floor_events": len(accum.floor_events),
            "combats": len(accum.combats),
            "event_logs": len(accum.events),
            "deck_snapshots": len(accum.deck_snapshots),
            "deck_cards": len(accum.deck_cards),
            "meta_decisions": len(accum.meta_decisions),
            "elapsed_sec": round(elapsed, 2),
        }


def parse_all_batches(single_source: str | None = None) -> dict:
    """无参编程入口：解析所有源日志（或单源）入库，返回统计 dict。

    给 web.routers.admin 用，避免重复实现命令行解析。
    """
    init_db()
    conn = get_conn()
    try:
        _ensure_abs_floor_columns(conn)
        if single_source:
            rows = conn.execute(
                "SELECT source_key, log_path FROM etl_metadata WHERE source_key = ?",
                (single_source,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT source_key, log_path FROM etl_metadata WHERE log_path IS NOT NULL"
            ).fetchall()

        if not rows:
            print("[parse_log] no sources to process")
            return {"sources_processed": 0, "total_episodes": 0}

        total_ep = 0
        n_processed = 0
        for r in rows:
            source_key = r["source_key"]
            log_path = r["log_path"]
            if not log_path:
                print(f"[parse_log] SKIP {source_key}: no log_path")
                continue
            if not Path(log_path).exists():
                print(f"[parse_log] SKIP {source_key}: log file missing {log_path}")
                continue
            print(f"[parse_log] processing {source_key} ({log_path})...")
            try:
                stats = LogParser(conn, log_path).parse_all()
            except Exception as exc:
                print(f"[parse_log] ERROR {source_key}: {exc}", file=sys.stderr)
                raise
            print(f"[parse_log] DONE {source_key}: {stats}")
            n_processed += 1
            total_ep += int(stats.get("episodes", 0) or 0)
        return {"sources_processed": n_processed, "total_episodes": total_ep}
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="解析训练日志入库")
    parser.add_argument("--source_key", default=None, help="单源模式（如 v15），留空则跑所有")
    # 保留旧参数名兼容
    parser.add_argument("--batch_id", default=None, help="(deprecated) 同 --source_key")
    args = parser.parse_args()
    source = args.source_key or args.batch_id
    parse_all_batches(single_source=source)


if __name__ == "__main__":
    main()

"""Trace log → SQLite 导入器。

输入: /tmp/v8_trace_game_*.log（每行一个手跑 trace，含 step / combat / turn / hand / action / enemy 数据）
输出: 已有 episodes / floor_events / combats 表 + combat_turns / combat_card_plays

episodes 增加 seed / abs_floor_reached 列（ALTER TABLE 幂等）。

绝对楼层公式：abs_floor = floor + (act - 1) * 17
  act1 floor 1-16  → abs 1-16
  act1 boss=16     → abs 16（trace 里 boss room=boss floor=16）
  act2 floor 1-16  → abs 18-33
  act3 floor 1-16  → abs 35-50

用法：
  .venv/bin/python -m web.etl.import_trace /tmp/v8_trace_game_*.log --start_ep 1
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

# 把 web/ 上一层加进 sys.path，方便单独跑
_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent.parent))

from web.db import get_conn, init_db  # noqa: E402


# ============ 正则 ============

# 文件头：=== V8 trace game seed=N ascension=A ===
RE_HEADER = re.compile(r"=== V8 trace game seed=(\d+) ascension=(\d+) ===")
# step 标题: --- step N ---
RE_STEP = re.compile(r"^--- step (\d+) ---\s*$")
# combat enter:   >> COMBAT enter floor=N act=A room=monster enemies=E ...
RE_COMBAT_ENTER = re.compile(
    r"^\s*>> COMBAT enter floor=(\d+) act=(\d+) room=(\w+) enemies=(\d+)"
)
# combat exit:    << COMBAT exit floor=N hp=H/M actions=A won=True/False
RE_COMBAT_EXIT = re.compile(
    r"^\s*<< COMBAT exit floor=(\d+) hp=(\d+)/(\d+) actions=(\d+) won=(True|False)"
)
# turn start: --- turn N start: energy=E/M player[hp=H/M block=B (buffs[...])?] ---
RE_TURN_START = re.compile(
    r"^\s*--- turn (\d+) start: energy=(\d+)/(\d+) player\[hp=(\d+)/(\d+) block=(\d+)(?: buffs\[([^\]]*)\])?\] ---"
)
# hand line:    hand: [0]Strike_R(cost=?) [1]Bash(cost=?) ...
RE_HAND_LINE = re.compile(r"^\s*hand: (.+)$")
RE_HAND_ITEM = re.compile(r"\[(\d+)\]([^(]+)\(cost=([^)]*)\)")
# enemy line:    enemy#0 JawWorm hp=H/M block=B (buffs[...] )?intent[...]
RE_ENEMY = re.compile(
    r"^\s*enemy#(\d+) (\S+) hp=(\d+)/(\d+) block=(\d+)(?: buffs\[([^\]]*)\])? intent\[([^\]]*)\]"
)
# action: play_card[I]=CardName (target=[J]EnemyName)?
RE_ACTION_PLAY = re.compile(
    r"^\s*action: play_card\[(\d+)\]=(\S+?)(?:\s+target=\[(\d+)\](\S+))?\s*$"
)
RE_ACTION_END = re.compile(r"^\s*action: end_turn\s*$")
# player_d:   player_d: hp+0 block+5 energy:3->2 (buffs[...] )?
RE_PLAYER_D = re.compile(r"^\s*player_d: (.+)$")
# enemy_d:    enemy_d: JawWorm#0:hp-6 [Vulnerable+2]
RE_ENEMY_D = re.compile(r"^\s*enemy_d: (.+)$")
# final stats JSON 行 (single line JSON in trace tail)
RE_FINAL = re.compile(r"^final stats: (\{.*\})\s*$")
# scalar status line in step body
RE_SCALAR = re.compile(
    r"^\s*phase=(\S+) floor=(\d+) act=(\d+) hp=(\d+)/(\d+) gold=(\d+) deck_size=(\d+) relics=(\d+)"
)
# [deck] snapshot line（trace 扩展格式，多了 act 字段；cards 段空格分隔，relics 段逗号分隔）
# 例: [deck] ep=1 floor=7 act=1 room=monster cards=Strike_R*5 Defend_R*4 Bash*1 relics=Burning Blood
RE_DECK = re.compile(
    r"^\[deck\]\s+ep=(\d+)\s+floor=(\d+)\s+act=(\d+)\s+room=(\S+)"
    r"\s+cards=(.*?)\s+relics=(.*?)\s*$"
)
# 单卡 token: <base>+1*<count> 或 <base>*<count>
RE_DECK_CARD_TOKEN = re.compile(r"^(.+?)(\+1)?\*(\d+)$")


# ============ schema 演化 ============

def _ensure_schema(conn) -> None:
    """幂等给 episodes 补 seed / abs_floor_reached 列，combats / floor_events 补 abs_floor 列"""
    cur = conn.cursor()
    # episodes 加 seed (NULLABLE，旧数据不受影响)
    existing_cols = {r[1] for r in cur.execute("PRAGMA table_info(episodes)").fetchall()}
    if "seed" not in existing_cols:
        cur.execute("ALTER TABLE episodes ADD COLUMN seed INTEGER")
    if "abs_floor_reached" not in existing_cols:
        cur.execute("ALTER TABLE episodes ADD COLUMN abs_floor_reached INTEGER")
    # combats / floor_events / event_logs / deck_snapshots / deck_cards / meta_decisions
    # 6 张表 abs_floor 列：与 parse_log._ensure_abs_floor_columns 保持一致
    for tbl in ("combats", "floor_events", "event_logs",
                "deck_snapshots", "deck_cards", "meta_decisions"):
        existing = {r[1] for r in cur.execute(f"PRAGMA table_info({tbl})").fetchall()}
        if "abs_floor" not in existing:
            cur.execute(f"ALTER TABLE {tbl} ADD COLUMN abs_floor INTEGER")
    conn.commit()


# ============ parser ============

def _abs_floor(floor: int, act: int) -> int:
    """floor + (act-1)*17 — act1: 1..16, act2: 18..33, act3: 35..50"""
    return floor + (act - 1) * 17


def _parse_hand(text: str) -> list[dict]:
    """'[0]Strike_R(cost=?) [1]Bash(cost=?) ...' → [{slot,card_en_id,cost}, ...]"""
    out = []
    for m in RE_HAND_ITEM.finditer(text):
        out.append({"slot": int(m.group(1)), "card_en_id": m.group(2).strip(), "cost": m.group(3)})
    return out


def _parse_enemy_line(line: str) -> dict | None:
    m = RE_ENEMY.match(line)
    if not m:
        return None
    return {
        "idx": int(m.group(1)),
        "en_id": m.group(2),
        "hp": int(m.group(3)),
        "hp_max": int(m.group(4)),
        "block": int(m.group(5)),
        "buffs": m.group(6) or "",
        "intent_raw": m.group(7),
    }


def _parse_deck_cards(raw: str) -> list[dict]:
    """`Defend_R*4 Strike_R*4 Bash*1 Sentinel+1*1` → [{card_en_id, upgraded, count}, ...]

    `-` 占位（空牌组）返回空 list。
    """
    raw = raw.strip()
    if not raw or raw == "-":
        return []
    out: list[dict] = []
    for tok in raw.split():
        m = RE_DECK_CARD_TOKEN.match(tok)
        if not m:
            continue
        base = m.group(1)
        upgraded = m.group(2) is not None
        count = int(m.group(3))
        out.append({"card_en_id": base, "upgraded": upgraded, "count": count})
    return out


def _parse_deck_relics(raw: str) -> list[str]:
    """`Burning Blood,NeowsBlessing,Paper Frog` → list."""
    raw = raw.strip()
    if not raw or raw == "-":
        return []
    return [r.strip() for r in raw.split(",") if r.strip()]


class TraceParser:
    """单文件 trace → 内存中数据结构 (episode + combats + turns + plays)"""

    def __init__(self, path: Path, ep_id: int):
        self.path = path
        self.ep_id = ep_id          # ep 序号（按 game 文件名 1..10）

        # episode 级
        self.seed: int | None = None
        self.final_stats: dict | None = None

        # 中间状态
        self.cur_combat: dict | None = None   # 当前正在收集的 combat
        self.cur_turn: dict | None = None     # 当前 turn
        self.action_counter: int = 0          # 当前 turn 的 action_idx

        # 累积
        self.combats: list[dict] = []         # 每条含 turns + plays
        self.floor_events: list[dict] = []    # (act, floor, hp_cur, hp_max, room)
        self.seen_floors: set[tuple[int, int]] = set()
        # deck snapshots：dict key = (act, floor, room)，重复 key 取最后一次
        # value = {act, floor, room, cards: list[dict], relics: list[str]}
        self.deck_snapshots: dict[tuple[int, int, str], dict] = {}

        # 上一行的 phase scalar (用于 floor_events 兜底)
        self.last_scalar: dict | None = None

    def _flush_turn(self):
        if self.cur_turn is not None and self.cur_combat is not None:
            self.cur_combat["turns"].append(self.cur_turn)
        self.cur_turn = None
        self.action_counter = 0

    def _flush_combat(self):
        if self.cur_combat is not None:
            self._flush_turn()
            self.combats.append(self.cur_combat)
        self.cur_combat = None

    def parse(self) -> None:
        lines = self.path.read_text(encoding="utf-8", errors="replace").splitlines()
        pending_enemies: list[dict] = []  # 暂存 turn-start 后跟的 enemy 行
        in_turn_enemy_block = False

        for ln in lines:
            # 1. header
            m = RE_HEADER.match(ln)
            if m:
                self.seed = int(m.group(1))
                continue

            # 2. final stats
            m = RE_FINAL.match(ln)
            if m:
                try:
                    self.final_stats = json.loads(m.group(1))
                except json.JSONDecodeError:
                    self.final_stats = None
                continue

            # 3. step boundary（开新 step，flush 任何 in-flight turn-enemy block）
            if RE_STEP.match(ln):
                if self.cur_turn is not None and pending_enemies:
                    self.cur_turn["enemies_state"] = pending_enemies
                    pending_enemies = []
                in_turn_enemy_block = False
                continue

            # 4. scalar status (phase= floor= act= hp= ...)
            m = RE_SCALAR.match(ln)
            if m:
                self.last_scalar = {
                    "phase": m.group(1),
                    "floor": int(m.group(2)),
                    "act": int(m.group(3)),
                    "hp_cur": int(m.group(4)),
                    "hp_max": int(m.group(5)),
                }
                continue

            # 5. combat enter
            m = RE_COMBAT_ENTER.match(ln)
            if m:
                # 若上一个 combat 没正常 exit（罕见），先 flush
                self._flush_combat()
                floor = int(m.group(1)); act = int(m.group(2)); room = m.group(3); _n = int(m.group(4))
                hp_before = self.last_scalar["hp_cur"] if self.last_scalar else None
                self.cur_combat = {
                    "act": act, "floor": floor, "room": room,
                    "abs_floor": _abs_floor(floor, act),
                    "enemies_en": [],         # 初始敌人 en_id list（在后续 enemy 行收集）
                    "hp_before": hp_before, "hp_after": None,
                    "turn_actions": None, "exit_reason": None,
                    "turns": [], "won": None,
                }
                # 记录 floor_event（每个新 floor 一行）
                key = (act, floor)
                if key not in self.seen_floors:
                    self.floor_events.append({
                        "act": act, "floor": floor, "abs_floor": _abs_floor(floor, act),
                        "hp_cur": hp_before, "hp_max": self.last_scalar["hp_max"] if self.last_scalar else None,
                        "room": room,
                    })
                    self.seen_floors.add(key)
                pending_enemies = []
                in_turn_enemy_block = False
                continue

            # 6. combat exit
            m = RE_COMBAT_EXIT.match(ln)
            if m:
                if self.cur_combat is not None:
                    self.cur_combat["hp_after"] = int(m.group(2))
                    self.cur_combat["turn_actions"] = int(m.group(4))
                    self.cur_combat["won"] = (m.group(5) == "True")
                    self.cur_combat["exit_reason"] = "won" if self.cur_combat["won"] else "lost"
                self._flush_combat()
                pending_enemies = []
                in_turn_enemy_block = False
                continue

            # 7. turn start
            m = RE_TURN_START.match(ln)
            if m:
                # flush prev turn enemies first
                if self.cur_turn is not None and pending_enemies:
                    self.cur_turn["enemies_state"] = pending_enemies
                    pending_enemies = []
                self._flush_turn()
                self.cur_turn = {
                    "turn_num": int(m.group(1)),
                    "energy": int(m.group(2)),
                    "energy_max": int(m.group(3)),
                    "hp_cur": int(m.group(4)),
                    "hp_max": int(m.group(5)),
                    "block": int(m.group(6)),
                    "player_buffs_raw": m.group(7) or "",
                    "hand_raw": "",
                    "hand": [],
                    "enemies_state": [],
                    "plays": [],
                }
                in_turn_enemy_block = True
                continue

            # 8. hand line (inside turn start block)
            m = RE_HAND_LINE.match(ln)
            if m and self.cur_turn is not None:
                self.cur_turn["hand_raw"] = m.group(1)
                self.cur_turn["hand"] = _parse_hand(m.group(1))
                continue

            # 9. enemy line — multi-context:
            #    a) right after combat enter (collect initial enemies)
            #    b) right after turn start (collect per-turn enemy snapshot)
            em = _parse_enemy_line(ln)
            if em is not None:
                if in_turn_enemy_block and self.cur_turn is not None:
                    pending_enemies.append(em)
                elif self.cur_combat is not None and self.cur_turn is None:
                    # initial combat enemy roster
                    if em["en_id"] not in self.cur_combat["enemies_en"]:
                        self.cur_combat["enemies_en"].append(em["en_id"])
                # else: enemy 行也会出现在 end_turn 之后（更新后 snapshot），忽略
                continue

            # 10. action play_card
            m = RE_ACTION_PLAY.match(ln)
            if m and self.cur_turn is not None:
                # close enemies block (turn snapshot complete)
                if pending_enemies:
                    self.cur_turn["enemies_state"] = pending_enemies
                    pending_enemies = []
                in_turn_enemy_block = False
                self.cur_turn["plays"].append({
                    "action_idx": self.action_counter,
                    "action_type": "play_card",
                    "hand_slot": int(m.group(1)),
                    "card_en_id": m.group(2),
                    "target_idx": int(m.group(3)) if m.group(3) else None,
                    "target_en_id": m.group(4),
                    "player_delta_raw": None,
                    "enemy_delta_raw": None,
                })
                self.action_counter += 1
                continue

            # 11. action end_turn
            if RE_ACTION_END.match(ln) and self.cur_turn is not None:
                if pending_enemies:
                    self.cur_turn["enemies_state"] = pending_enemies
                    pending_enemies = []
                in_turn_enemy_block = False
                self.cur_turn["plays"].append({
                    "action_idx": self.action_counter,
                    "action_type": "end_turn",
                    "hand_slot": None,
                    "card_en_id": None,
                    "target_idx": None,
                    "target_en_id": None,
                    "player_delta_raw": None,
                    "enemy_delta_raw": None,
                })
                self.action_counter += 1
                continue

            # 12. player_d / enemy_d 附加到最近 play
            m = RE_PLAYER_D.match(ln)
            if m and self.cur_turn is not None and self.cur_turn["plays"]:
                self.cur_turn["plays"][-1]["player_delta_raw"] = m.group(1)
                continue
            m = RE_ENEMY_D.match(ln)
            if m and self.cur_turn is not None and self.cur_turn["plays"]:
                self.cur_turn["plays"][-1]["enemy_delta_raw"] = m.group(1)
                continue

            # 13. [deck] snapshot 行（同 (act, floor, room) 取最后一次）
            m = RE_DECK.match(ln)
            if m:
                _floor = int(m.group(2))
                _act = int(m.group(3))
                _room = m.group(4)
                _cards = _parse_deck_cards(m.group(5))
                _relics = _parse_deck_relics(m.group(6))
                self.deck_snapshots[(_act, _floor, _room)] = {
                    "act": _act,
                    "floor": _floor,
                    "room": _room,
                    "cards": _cards,
                    "relics": _relics,
                }
                continue

        # EOF flush
        if self.cur_turn is not None and pending_enemies:
            self.cur_turn["enemies_state"] = pending_enemies
        self._flush_combat()


# ============ DB writer ============

def _insert_episode(conn, parser: TraceParser) -> None:
    fs = parser.final_stats or {}
    final_floor = fs.get("final_floor")
    final_act = fs.get("final_act") or 1
    abs_final = _abs_floor(final_floor, final_act) if final_floor is not None else None
    beat_boss = 1 if fs.get("won_run") else 0
    conn.execute(
        """INSERT OR REPLACE INTO episodes
           (ep, steps, reward, floor_reached, beat_boss, secs,
            search_calls, eval_deck_calls, seed, abs_floor_reached)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (
            parser.ep_id,
            fs.get("steps"), None, final_floor, beat_boss,
            fs.get("elapsed_sec"), None, None,
            parser.seed, abs_final,
        ),
    )


def _insert_floor_events(conn, parser: TraceParser) -> None:
    for fe in parser.floor_events:
        conn.execute(
            """INSERT OR REPLACE INTO floor_events
               (ep, act, floor, hp_cur, hp_max, room, abs_floor)
               VALUES (?,?,?,?,?,?,?)""",
            (parser.ep_id, fe["act"], fe["floor"],
             fe["hp_cur"], fe["hp_max"], fe["room"], fe["abs_floor"]),
        )


def _insert_combats(conn, parser: TraceParser) -> None:
    for c in parser.combats:
        conn.execute(
            """INSERT OR REPLACE INTO combats
               (ep, act, floor, room, enemies_json, hp_before, hp_after,
                turn_actions, exit_reason, abs_floor)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (parser.ep_id, c["act"], c["floor"], c["room"],
             json.dumps(c["enemies_en"]), c["hp_before"], c["hp_after"],
             c["turn_actions"], c["exit_reason"], c["abs_floor"]),
        )

        # turns
        for t in c["turns"]:
            conn.execute(
                """INSERT OR REPLACE INTO combat_turns
                   (ep, act, floor, abs_floor, turn_num, energy, energy_max,
                    hp_cur, hp_max, block, player_buffs_raw, hand_raw, hand_json,
                    enemies_state_json)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (parser.ep_id, c["act"], c["floor"], c["abs_floor"],
                 t["turn_num"], t["energy"], t["energy_max"], t["hp_cur"], t["hp_max"],
                 t["block"], t["player_buffs_raw"], t["hand_raw"],
                 json.dumps(t["hand"], ensure_ascii=False),
                 json.dumps(t["enemies_state"], ensure_ascii=False)),
            )
            # plays
            for p in t["plays"]:
                conn.execute(
                    """INSERT OR REPLACE INTO combat_card_plays
                       (ep, act, floor, abs_floor, turn_num, action_idx,
                        action_type, card_en_id, hand_slot, target_idx, target_en_id,
                        player_delta_raw, enemy_delta_raw)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (parser.ep_id, c["act"], c["floor"], c["abs_floor"],
                     t["turn_num"], p["action_idx"], p["action_type"], p["card_en_id"],
                     p["hand_slot"], p["target_idx"], p["target_en_id"],
                     p["player_delta_raw"], p["enemy_delta_raw"]),
                )


def _insert_deck_snapshots(conn, parser: TraceParser) -> None:
    """deck_snapshots + deck_cards 双表写入。同 (ep, act, floor, room/card_en_id, upgraded) PK
    采用 INSERT OR REPLACE，幂等。"""
    for (act, floor, room), snap in parser.deck_snapshots.items():
        cards = snap["cards"]
        relics = snap["relics"]
        # cards_raw 用 markers.py 同款格式: "<id>[+1]*<count> ..."
        cards_raw = " ".join(
            f"{c['card_en_id']}{'+1' if c['upgraded'] else ''}*{c['count']}"
            for c in cards
        )
        relics_raw = ",".join(relics)
        deck_size = sum(c["count"] for c in cards)
        conn.execute(
            """INSERT OR REPLACE INTO deck_snapshots
               (ep, act, floor, abs_floor, room, cards_raw, relics_raw, deck_size)
               VALUES (?,?,?,?,?,?,?,?)""",
            (parser.ep_id, act, floor, _abs_floor(floor, act), room,
             cards_raw, relics_raw, deck_size),
        )
        # deck_cards：先按 (act, floor) 清掉该格旧记录，再插入；防止同 floor 多 room
        # snapshot（同 floor 战斗 room + reward 后 map room）互相污染
        # 但我们 PK 是 (ep, act, floor, card_en_id, upgraded) — 不带 room；
        # 多 room 同 floor 用 INSERT OR REPLACE 最后 room 胜出即可
        for c in cards:
            up = 1 if c["upgraded"] else 0
            conn.execute(
                """INSERT OR REPLACE INTO deck_cards
                   (ep, act, floor, abs_floor, card_en_id, upgraded, count)
                   VALUES (?,?,?,?,?,?,?)""",
                (parser.ep_id, act, floor, _abs_floor(floor, act),
                 c["card_en_id"], up, c["count"]),
            )


# ============ 入口 ============

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace_globs", nargs="+", help="trace 文件路径（支持 glob）")
    ap.add_argument("--start_ep", type=int, default=1, help="ep 起始编号")
    ap.add_argument("--purge", action="store_true",
                    help="先删 ep 范围内旧数据再导入（默认 INSERT OR REPLACE）")
    args = ap.parse_args()

    # 展开 glob，按文件名里的数字排序（1..10）
    paths: list[Path] = []
    for g in args.trace_globs:
        for p in glob.glob(g):
            paths.append(Path(p))
    def _key(p: Path):
        m = re.search(r"_(\d+)\.log$", p.name)
        return int(m.group(1)) if m else 0
    paths.sort(key=_key)
    if not paths:
        print("[import_trace] 没找到任何文件", file=sys.stderr)
        sys.exit(1)

    print(f"[import_trace] 待导入 {len(paths)} 个 trace, start_ep={args.start_ep}")

    # 确保基础 schema
    init_db()
    conn = get_conn()
    try:
        _ensure_schema(conn)

        if args.purge:
            start_ep = args.start_ep
            end_ep = start_ep + len(paths) - 1
            print(f"[import_trace] purge ep={start_ep}..{end_ep}")
            for tbl in ("combat_card_plays", "combat_turns", "combats", "event_logs",
                        "floor_events", "deck_cards", "deck_snapshots", "episodes"):
                conn.execute(
                    f"DELETE FROM {tbl} WHERE ep BETWEEN ? AND ?",
                    (start_ep, end_ep),
                )
            conn.commit()

        n_turns_total = 0
        n_plays_total = 0
        n_decks_total = 0
        for i, p in enumerate(paths):
            ep = args.start_ep + i
            print(f"  [{ep:2d}] parsing {p.name}")
            tp = TraceParser(p, ep_id=ep)
            tp.parse()

            n_combats = len(tp.combats)
            n_turns = sum(len(c["turns"]) for c in tp.combats)
            n_plays = sum(len(t["plays"]) for c in tp.combats for t in c["turns"])
            n_decks = len(tp.deck_snapshots)
            n_turns_total += n_turns
            n_plays_total += n_plays
            n_decks_total += n_decks
            fs = tp.final_stats or {}
            print(f"       seed={tp.seed} final_floor={fs.get('final_floor')} "
                  f"final_act={fs.get('final_act')} "
                  f"abs={_abs_floor(fs.get('final_floor') or 0, fs.get('final_act') or 1)} "
                  f"hp={fs.get('final_hp')} won={fs.get('won_run')} "
                  f"combats={n_combats} turns={n_turns} plays={n_plays} "
                  f"deck_snaps={n_decks}")

            _insert_episode(conn, tp)
            _insert_floor_events(conn, tp)
            _insert_combats(conn, tp)
            _insert_deck_snapshots(conn, tp)
            conn.commit()

        print(f"\n[import_trace] DONE: {len(paths)} eps, "
              f"{n_turns_total} turns, {n_plays_total} plays, "
              f"{n_decks_total} deck snapshots")
    finally:
        conn.close()


if __name__ == "__main__":
    main()

r"""V8 RL 训练日志 marker 正则 + 解析器。

每条 marker 一对 (compiled regex, parse function)。parse function 返回 dict，
key 为列名，已做类型转换（int/float/bool/list）。

实际样例（来自 /tmp/v8_ppo_batch_v8.log）：

    [heartbeat] ep=512 steps=41 secs=220.0 reward=38.333 floor=16 act=1
                beat_boss=False search_calls=39 eval_deck_calls=41
    [floor] ep=512 floor=0 act=1 hp=80/80 room=unknown
    [combat] enter ep=512 floor=1 act=1 room=monster enemies=['Cultist']
             deck_size=10 hp=80/80
    [combat] exit ep=512 reason=victory floor=1 hp_before=80 hp_after=80
             turn_actions=1
    [event] enter ep=512 floor=2 act=1 event_id=NoteForYourself
             event_phase=INITIAL choices=[0:[Take] Take a card, leave one behind, 1:[Leave]]
    [event] choice ep=512 floor=2 choice_idx=1 choice_text=[Leave]
             event_id=NoteForYourself event_phase=INITIAL
    [event] phase_transition ep=549 floor=2 event_id=GoldenIdol
             from=INITIAL to=SECONDARY
    [event] exit ep=512 floor=2 event_id=NoteForYourself
             reason=resolved->MAP_NAVIGATION max_hp_now=80 hp_now=80/80
             deck_size=11 relics_count=2
    [deck] ep=512 floor=7 room=elite cards=Defend_R*4 Strike_R*4 Bash*1
            Clash*1 Clothesline*1 Sentinel+1*1
            relics=Burning Blood,NeowsBlessing,Paper Frog
    [startup] output_dir=... num_episodes=640 ... resume_from=... start_episode=512

注意陷阱：
- event choices 内部含 `, ` 分隔多 choice，但 choice text 也可能含逗号；
  按前向 lookahead `(?=\d+:)` 切。
- combat exit 没有 act 字段（设计时省略），act 从 enter 缓存里补。
- deck cards 末尾跟 `relics=...`，relics 用逗号分隔（不是空格）。
- deck cards 单卡格式 `<base>+1*<count>` 或 `<base>*<count>`，可能含数字下标。
"""
from __future__ import annotations

import ast
import json
import re
from typing import Any


# ===== 公共工具 =====

def _to_int(s: str | None) -> int | None:
    if s is None or s == "":
        return None
    try:
        return int(s)
    except (TypeError, ValueError):
        return None


def _to_float(s: str | None) -> float | None:
    if s is None or s == "":
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _to_bool(s: str | None) -> bool:
    if s is None:
        return False
    return s.strip().lower() == "true"


# ===== [heartbeat] =====
# 旧 (2026-05-20): ... act=3 beat_boss=True search_calls=83 eval_deck_calls=138
# 新 (2026-05-29 阶段0): beat_boss 后追加 a1_boss_killed / a2_boss_killed / won_game 三个诚实字段。
#   ... beat_boss=True a1_boss_killed=True a2_boss_killed=True won_game=True search_calls=83 eval_deck_calls=138
# 正则把这三个字段设为可选（旧日志仍能解析），三者缺省时 beat_boss 既是 won_game 别名也兜底 a1/a2。
HEARTBEAT_RE = re.compile(
    r"\[heartbeat\]\s+"
    r"ep=(?P<ep>\d+)\s+"
    r"steps=(?P<steps>\d+)\s+"
    r"secs=(?P<secs>[\d.eE+\-]+)\s+"
    r"reward=(?P<reward>[\d.eE+\-]+)\s+"
    r"floor=(?P<floor>\d+)\s+"
    r"act=(?P<act>\d+)\s+"
    r"beat_boss=(?P<beat_boss>True|False)\s+"
    r"(?:a1_boss_killed=(?P<a1_boss_killed>True|False)\s+"
    r"a2_boss_killed=(?P<a2_boss_killed>True|False)\s+"
    r"won_game=(?P<won_game>True|False)\s+)?"
    r"search_calls=(?P<search_calls>\d+)\s+"
    r"eval_deck_calls=(?P<eval_deck_calls>\d+)"
)


def parse_heartbeat(m: re.Match) -> dict[str, Any]:
    beat_boss = _to_bool(m["beat_boss"])
    # 阶段0 新字段缺省（旧日志）时回落：won_game = beat_boss（旧别名语义），
    # a1/a2 也回落 beat_boss（旧日志拿不到更细粒度，至少不丢通关信号）。
    a1 = m.groupdict().get("a1_boss_killed")
    a2 = m.groupdict().get("a2_boss_killed")
    won = m.groupdict().get("won_game")
    return {
        "ep": int(m["ep"]),
        "steps": int(m["steps"]),
        "secs": float(m["secs"]),
        "reward": float(m["reward"]),
        "floor": int(m["floor"]),
        "act": int(m["act"]),
        "beat_boss": beat_boss,
        "a1_boss_killed": _to_bool(a1) if a1 is not None else beat_boss,
        "a2_boss_killed": _to_bool(a2) if a2 is not None else beat_boss,
        "won_game": _to_bool(won) if won is not None else beat_boss,
        "search_calls": int(m["search_calls"]),
        "eval_deck_calls": int(m["eval_deck_calls"]),
    }


# ===== [floor] =====
# 2026-05-20 03:30:46,067 INFO [floor] ep=512 floor=0 act=1 hp=80/80 room=unknown
FLOOR_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d+)\s+\w+\s+"
    r"\[floor\]\s+"
    r"ep=(?P<ep>\d+)\s+"
    r"floor=(?P<floor>\d+)\s+"
    r"act=(?P<act>\d+)\s+"
    r"hp=(?P<hp_cur>\d+)/(?P<hp_max>\d+)\s+"
    r"room=(?P<room>\S+)"
)


def parse_floor(m: re.Match) -> dict[str, Any]:
    return {
        "ts": m["ts"],
        "ep": int(m["ep"]),
        "floor": int(m["floor"]),
        "act": int(m["act"]),
        "hp_cur": int(m["hp_cur"]),
        "hp_max": int(m["hp_max"]),
        "room": m["room"],
    }


# ===== [combat] enter =====
# 2026-05-20 03:30:57,785 INFO [combat] enter ep=512 floor=1 act=1 room=monster enemies=['Cultist'] deck_size=10 hp=80/80
COMBAT_ENTER_RE = re.compile(
    r"\[combat\]\s+enter\s+"
    r"ep=(?P<ep>\d+)\s+"
    r"floor=(?P<floor>\d+)\s+"
    r"act=(?P<act>\d+)\s+"
    r"room=(?P<room>\S+)\s+"
    r"enemies=(?P<enemies>\[.*?\])\s+"
    r"deck_size=(?P<deck_size>\d+)\s+"
    r"hp=(?P<hp_cur>\d+)/(?P<hp_max>\d+)"
)

# enemies=['Cultist'] / ['Louse', 'Louse'] / ["Slime Boss"]
_ENEMY_TOKEN_RE = re.compile(r"['\"]([^'\"]+)['\"]")


def _parse_enemy_list(raw: str) -> list[str]:
    """解析 ['Cultist', 'Louse'] 这种 list literal 字符串"""
    return _ENEMY_TOKEN_RE.findall(raw)


def parse_combat_enter(m: re.Match) -> dict[str, Any]:
    return {
        "ep": int(m["ep"]),
        "floor": int(m["floor"]),
        "act": int(m["act"]),
        "room": m["room"],
        "enemies": _parse_enemy_list(m["enemies"]),
        "deck_size": int(m["deck_size"]),
        "hp_before": int(m["hp_cur"]),
        "hp_max": int(m["hp_max"]),
    }


# ===== [combat] exit =====
# 2026-05-20 03:30:57,795 INFO [combat] exit ep=512 reason=victory floor=1 hp_before=80 hp_after=80 turn_actions=1
COMBAT_EXIT_RE = re.compile(
    r"\[combat\]\s+exit\s+"
    r"ep=(?P<ep>\d+)\s+"
    r"reason=(?P<reason>\S+)\s+"
    r"floor=(?P<floor>\d+)\s+"
    r"hp_before=(?P<hp_before>\d+)\s+"
    r"hp_after=(?P<hp_after>\d+)\s+"
    r"turn_actions=(?P<turn_actions>\d+)"
)


def parse_combat_exit(m: re.Match) -> dict[str, Any]:
    return {
        "ep": int(m["ep"]),
        "reason": m["reason"],
        "floor": int(m["floor"]),
        "hp_before": int(m["hp_before"]),
        "hp_after": int(m["hp_after"]),
        "turn_actions": int(m["turn_actions"]),
    }


# ===== [event] enter =====
# 2026-05-20 03:31:04,039 INFO [event] enter ep=512 floor=2 act=1 event_id=NoteForYourself event_phase=INITIAL choices=[0:[Take] Take a card, leave one behind, 1:[Leave]]
# 注意 choices 可能为空 []
EVENT_ENTER_RE = re.compile(
    r"\[event\]\s+enter\s+"
    r"ep=(?P<ep>\d+)\s+"
    r"floor=(?P<floor>\d+)\s+"
    r"act=(?P<act>\d+)\s+"
    r"event_id=(?P<event_id>\S+)\s+"
    r"event_phase=(?P<event_phase>\S+)\s+"
    r"choices=(?P<choices>\[.*\])\s*$"
)


def _parse_event_choices(raw: str) -> list[dict[str, Any]]:
    """choices=[0:[Take] Take a card, leave one behind, 1:[Leave]] → 2 items.

    切分策略：去掉外层 [...]，按 ', ' 但前看必须是 `<digit>+:` 才算分隔；
    每段格式 `<idx>:<text>`。
    """
    inner = raw.strip()
    if inner.startswith("["):
        inner = inner[1:]
    if inner.endswith("]"):
        inner = inner[:-1]
    inner = inner.strip()
    if not inner:
        return []
    # 用前向 lookahead 切：分隔符必须是 `, ` 后面跟 `<num>:`
    parts = re.split(r",\s+(?=\d+:)", inner)
    out: list[dict[str, Any]] = []
    for p in parts:
        m = re.match(r"^(\d+):\s*(.*)$", p)
        if m:
            out.append({"idx": int(m.group(1)), "text": m.group(2).strip()})
    return out


def parse_event_enter(m: re.Match) -> dict[str, Any]:
    return {
        "ep": int(m["ep"]),
        "floor": int(m["floor"]),
        "act": int(m["act"]),
        "event_id": m["event_id"],
        "event_phase": m["event_phase"],
        "choices": _parse_event_choices(m["choices"]),
    }


# ===== [event] choice =====
# 2026-05-20 03:31:04,078 INFO [event] choice ep=512 floor=2 choice_idx=1 choice_text=[Leave] event_id=NoteForYourself event_phase=INITIAL
# choice_text 可能含空格，但末尾固定跟 ` event_id=...`，用 non-greedy + lookahead
EVENT_CHOICE_RE = re.compile(
    r"\[event\]\s+choice\s+"
    r"ep=(?P<ep>\d+)\s+"
    r"floor=(?P<floor>\d+)\s+"
    r"choice_idx=(?P<choice_idx>\d+)\s+"
    r"choice_text=(?P<choice_text>.*?)\s+"
    r"event_id=(?P<event_id>\S+)\s+"
    r"event_phase=(?P<event_phase>\S+)\s*$"
)


def parse_event_choice(m: re.Match) -> dict[str, Any]:
    return {
        "ep": int(m["ep"]),
        "floor": int(m["floor"]),
        "choice_idx": int(m["choice_idx"]),
        "choice_text": m["choice_text"],
        "event_id": m["event_id"],
        "event_phase": m["event_phase"],
    }


# ===== [event] phase_transition =====
# 2026-05-20 06:25:18,252 INFO [event] phase_transition ep=549 floor=2 event_id=GoldenIdol from=INITIAL to=SECONDARY
EVENT_PHASE_TRANS_RE = re.compile(
    r"\[event\]\s+phase_transition\s+"
    r"ep=(?P<ep>\d+)\s+"
    r"floor=(?P<floor>\d+)\s+"
    r"event_id=(?P<event_id>\S+)\s+"
    r"from=(?P<from_phase>\S+)\s+"
    r"to=(?P<to_phase>\S+)"
)


def parse_event_phase_transition(m: re.Match) -> dict[str, Any]:
    return {
        "ep": int(m["ep"]),
        "floor": int(m["floor"]),
        "event_id": m["event_id"],
        "from_phase": m["from_phase"],
        "to_phase": m["to_phase"],
    }


# ===== [event] exit =====
# 2026-05-20 03:31:04,078 INFO [event] exit ep=512 floor=2 event_id=NoteForYourself reason=resolved->MAP_NAVIGATION max_hp_now=80 hp_now=80/80 deck_size=11 relics_count=2
EVENT_EXIT_RE = re.compile(
    r"\[event\]\s+exit\s+"
    r"ep=(?P<ep>\d+)\s+"
    r"floor=(?P<floor>\d+)\s+"
    r"event_id=(?P<event_id>\S+)\s+"
    r"reason=(?P<reason>\S+)\s+"
    r"max_hp_now=(?P<max_hp>\d+)\s+"
    r"hp_now=(?P<hp_cur>\d+)/(?P<hp_max>\d+)\s+"
    r"deck_size=(?P<deck_size>\d+)\s+"
    r"relics_count=(?P<relics_count>\d+)"
)


def parse_event_exit(m: re.Match) -> dict[str, Any]:
    return {
        "ep": int(m["ep"]),
        "floor": int(m["floor"]),
        "event_id": m["event_id"],
        "reason": m["reason"],
        "max_hp": int(m["max_hp"]),
        "hp_cur": int(m["hp_cur"]),
        "hp_max": int(m["hp_max"]),
        "deck_size": int(m["deck_size"]),
        "relics_count": int(m["relics_count"]),
    }


# ===== [deck] =====
# 2026-05-20 03:31:29,852 INFO [deck] ep=512 floor=7 room=elite cards=Defend_R*4 Strike_R*4 Bash*1 ... relics=Burning Blood,NeowsBlessing,Paper Frog
# cards 段空格分隔多个 `<id>[+1]*<count>`，relics 段逗号分隔多个名字（含空格）
DECK_RE = re.compile(
    r"\[deck\]\s+"
    r"ep=(?P<ep>\d+)\s+"
    r"floor=(?P<floor>\d+)\s+"
    r"room=(?P<room>\S+)\s+"
    r"cards=(?P<cards>.*?)\s+relics=(?P<relics>.*?)\s*$"
)

# 单卡：<base>+1*<count> 或 <base>*<count>，base 可含字母数字下划线
_DECK_CARD_TOKEN_RE = re.compile(r"^(.+?)(\+1)?\*(\d+)$")


def _parse_deck_cards(raw: str) -> list[dict[str, Any]]:
    """`Defend_R*4 Strike_R*4 Bash*1 Sentinel+1*1` → list of dict."""
    out: list[dict[str, Any]] = []
    for tok in raw.strip().split():
        m = _DECK_CARD_TOKEN_RE.match(tok)
        if not m:
            continue
        base = m.group(1)
        upgraded = m.group(2) is not None
        count = int(m.group(3))
        out.append({"card_en_id": base, "upgraded": upgraded, "count": count})
    return out


def _parse_deck_relics(raw: str) -> list[str]:
    """`Burning Blood,NeowsBlessing,Paper Frog` → ['Burning Blood', ...]"""
    raw = raw.strip()
    if not raw:
        return []
    return [r.strip() for r in raw.split(",") if r.strip()]


def parse_deck(m: re.Match) -> dict[str, Any]:
    return {
        "ep": int(m["ep"]),
        "floor": int(m["floor"]),
        "room": m["room"],
        "cards": _parse_deck_cards(m["cards"]),
        "relics": _parse_deck_relics(m["relics"]),
    }


# ===== [meta] =====
# [meta] ep=513 step=42 phase=CARD_REWARDS floor=3 hp=72/80 gold=120 n_options=4 chosen_idx=2 chosen=REWARD:card:Inflame:choice=0 options=['REWARD:card:Inflame:choice=0', 'REWARD:card:Flex:choice=1', 'REWARD:card:Sentinel:choice=2', 'REWARD:skip_card']
META_RE = re.compile(
    r"\[meta\]\s+"
    r"ep=(?P<ep>\d+)\s+"
    r"step=(?P<step>\d+)\s+"
    r"phase=(?P<phase>\S+)\s+"
    r"floor=(?P<floor>\d+)\s+"
    r"hp=\d+/\d+\s+"
    r"gold=\d+\s+"
    r"n_options=(?P<n_options>\d+)\s+"
    r"chosen_idx=(?P<chosen_idx>-?\d+)\s+"
    r"chosen=(?P<chosen>.+?)\s+"
    r"options=(?P<options>\[.*)"
)


def parse_meta(m: re.Match) -> dict[str, Any]:
    return {
        "ep": int(m["ep"]),
        "step": int(m["step"]),
        "phase": m["phase"],
        "floor": int(m["floor"]),
        "n_options": int(m["n_options"]),
        "chosen_idx": int(m["chosen_idx"]),
        "chosen": m["chosen"],
        "options_raw": m["options"].rstrip(),
    }


def _options_to_json(raw: str) -> str:
    """把 Python list repr 转为 JSON array。ast.literal_eval 失败时保留原始字符串。"""
    try:
        parsed = ast.literal_eval(raw)
        return json.dumps(parsed, ensure_ascii=False)
    except Exception:
        return raw


# CARD_REWARDS chosen 里 "proceed" 行是"关闭奖励弹窗"确认，不是真正的卡牌选择
_CARD_REWARD_RE = re.compile(r"REWARD:card:(?P<card>[^:]+):choice=")
_REST_UPGRADE_RE = re.compile(r"REST:upgrade:(?P<card>.+)")


def postprocess_meta(d: dict) -> dict | None:
    """后处理 [meta] 解析结果。

    - 过滤 CARD_REWARDS 的 proceed 行
    - 提取 decision_type / chosen_card
    - 转换 options 到 JSON
    返回 None 表示此行应被丢弃。
    """
    phase = d["phase"]
    chosen = d["chosen"]

    # 过滤 CARD_REWARDS proceed（关闭奖励弹窗确认）
    if phase == "CARD_REWARDS" and "proceed" in chosen:
        return None

    # options → JSON
    options_json = _options_to_json(d["options_raw"])

    decision_type = None
    chosen_card = None

    if phase == "CARD_REWARDS":
        cm = _CARD_REWARD_RE.search(chosen)
        if cm:
            decision_type = "pick"
            chosen_card = cm.group("card")
        elif "skip_card" in chosen:
            decision_type = "skip"
    elif phase == "REST":
        if "REST:rest" in chosen:
            decision_type = "rest"
        else:
            um = _REST_UPGRADE_RE.search(chosen)
            if um:
                decision_type = "upgrade"
                chosen_card = um.group("card")
    elif phase == "SHOP":
        if "buy" in chosen.lower():
            decision_type = "buy"
        elif "remove" in chosen.lower():
            decision_type = "remove"
        elif "leave" in chosen.lower():
            decision_type = "leave"
    # MAP / NEOW / TREASURE / BOSS_REWARDS: decision_type 留 None

    return {
        "ep": d["ep"],
        "step": d["step"],
        "act": d.get("act"),  # 由 parse_log._BatchAccum 回填
        "phase": phase,
        "floor": d["floor"],
        "chosen": chosen,
        "chosen_card": chosen_card,
        "decision_type": decision_type,
        "options_json": options_json,
    }


# ===== marker 类型常量 + 派发表 =====
# 按"先频率高 / 先精确"顺序排列，每行一次性 try-match
MARKERS = [
    ("heartbeat", HEARTBEAT_RE, parse_heartbeat),
    ("floor", FLOOR_RE, parse_floor),
    ("combat_enter", COMBAT_ENTER_RE, parse_combat_enter),
    ("combat_exit", COMBAT_EXIT_RE, parse_combat_exit),
    ("event_enter", EVENT_ENTER_RE, parse_event_enter),
    ("event_choice", EVENT_CHOICE_RE, parse_event_choice),
    ("event_phase_transition", EVENT_PHASE_TRANS_RE, parse_event_phase_transition),
    ("event_exit", EVENT_EXIT_RE, parse_event_exit),
    ("deck", DECK_RE, parse_deck),
    ("meta", META_RE, parse_meta),
]


# 用于快速预筛：判断一行是否可能含某 marker（避免每行跑 9 个 regex）
_MARKER_PREFIX_KEYWORDS = {
    "[heartbeat]": ("heartbeat",),
    "[floor]": ("floor",),
    "[combat]": ("combat_enter", "combat_exit"),
    "[event]": (
        "event_enter",
        "event_choice",
        "event_phase_transition",
        "event_exit",
    ),
    "[deck]": ("deck",),
    "[meta]": ("meta",),
}


def classify_line(line: str) -> str | None:
    """根据 marker 前缀关键字快速判断当前行需要哪几条 regex try。

    返回 marker keyword（如 '[heartbeat]'）或 None。
    """
    # 找最早出现的 marker 关键字
    for kw in _MARKER_PREFIX_KEYWORDS:
        if kw in line:
            return kw
    return None

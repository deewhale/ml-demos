"""
V8 数据采集器 — V7Bot 子类，在 COMBAT 决策点写 JSONL。

设计原则
--------
1. 完全复用 V7Bot 的决策逻辑（不重写 _pick_action 主流程）。
2. 在父类决策结束后，仅当 phase==COMBAT 时，把当前 state + chosen action +
   solver scores + bottled_ai 评估尝试结果写一行 JSONL。
3. bottled_ai 评估是 best-effort：任何 Exception（含 NotImplementedError）
   都 catch，错误信息写到 `bottled_ai_error` 字段，**不阻塞游戏**。
4. 序列化非 JSON 兼容类型 → str() 强转（json.dumps default=str）。

JSONL schema 见模块顶部 `_SCHEMA_HINT`。
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, TextIO

# V7Bot 已经做了 ensure_on_sys_path，import 顺序保持一致
from v7_bot import V7Bot
from sts_paths import ensure_on_sys_path
ensure_on_sys_path()

from packages.engine.game import GamePhase  # noqa: E402

# 失败也无所谓的 v8 模块（catch 在 _try_bottled_assessment 里）
try:
    from v8_battle_state_adapter import BottledStateAdapter
    from v8_teacher_eval import (
        ComparatorAssessment,
        ComparatorAssessmentConfig,
    )
    from v8_teacher_ironclad import (
        POWERS_WE_LIKE,
        POWERS_WE_LIKE_LESS,
        POWERS_WE_DISLIKE,
        CARDS_THAT_EXIT_WRATH,
    )
    _BOTTLED_IMPORT_OK = True
    _BOTTLED_IMPORT_ERR: Optional[str] = None
except Exception as _e:  # noqa: BLE001
    _BOTTLED_IMPORT_OK = False
    _BOTTLED_IMPORT_ERR = f"{type(_e).__name__}: {_e}"

logger = logging.getLogger(__name__)


_SCHEMA_HINT = """
JSONL line schema (one decision per line):
{
  "seed": int, "floor": int, "act": int, "turn": int,
  "player": {"hp": int, "max_hp": int, "block": int, "energy": int, "powers": {str: int}},
  "enemies": [{"id": str, "hp": int, "max_hp": int, "block": int, "powers": {str: int}}],
  "hand": [str], "draw_pile_size": int, "discard_pile_size": int, "exhaust_pile_size": int,
  "available_actions": [str],
  "chosen_action": str,
  "solver_scores": [[str, float], ...],
  "bottled_ai_assessment": {...} or null,
  "bottled_ai_error": str or null
}
"""


def try_bottled_assessment(run_state, combat_state):
    """构造 bottled_ai 评估 dict（best-effort）。返回 (assessment, error)。

    被 V8CollectorBot 与 V8InferenceBot 共用：前者写 JSONL，后者
    在 inference 时构造 record 给 model.encode_state 消费。
    """
    if not _BOTTLED_IMPORT_OK:
        return None, f"import failed: {_BOTTLED_IMPORT_ERR}"

    try:
        relics_dict: Dict[str, int] = {}
        for r in getattr(run_state, "relics", []) or []:
            rid = getattr(r, "id", str(r))
            relics_dict[rid] = relics_dict.get(rid, 0) + 1

        try:
            potions_list: List[str] = list(run_state.get_potions())
        except Exception:  # noqa: BLE001
            potions_list = []

        adapter = BottledStateAdapter(
            combat_state,
            run_relics=relics_dict,
            run_potions=potions_list,
        )
        cfg = ComparatorAssessmentConfig(
            powers_we_like=POWERS_WE_LIKE,
            powers_we_like_less=POWERS_WE_LIKE_LESS,
            powers_we_dislike=POWERS_WE_DISLIKE,
            cards_that_exit_wrath=CARDS_THAT_EXIT_WRATH,
        )
        ca = ComparatorAssessment(adapter, adapter, cfg)

        assessment: Dict[str, Any] = {}
        for name in (
            "battle_won", "battle_lost", "incoming_damage", "energy",
            "intangible", "dead_monsters", "lowest_true_health_monster",
            "total_monster_health", "enemy_vulnerable", "enemy_weak",
            "player_powers_good", "player_powers_bad", "bad_cards_exhausted",
        ):
            fn = getattr(ca, name, None)
            if fn is None:
                continue
            try:
                val = fn()
                if isinstance(val, (bool, int, float, str)):
                    assessment[name] = val
                else:
                    assessment[name] = str(val)
            except Exception as inner:  # noqa: BLE001
                assessment[name] = f"<err: {type(inner).__name__}: {inner}>"

        return assessment, None
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


def build_combat_record(
    runner,
    actions,
    chosen_action,
    seed: Any = None,
) -> Dict[str, Any]:
    """从 runner 的当前 COMBAT 状态构造一条 record（与 JSONL schema 一致）。

    chosen_action 可为 None（inference 时还没选定）。
    """
    rs = runner.run_state
    cc = runner.current_combat
    state = cc.state

    player = state.player
    player_dict = {
        "hp": getattr(player, "hp", 0),
        "max_hp": getattr(player, "max_hp", 0),
        "block": getattr(player, "block", 0),
        "energy": getattr(state, "energy", 0),
        "powers": dict(getattr(player, "statuses", {}) or {}),
    }

    enemies_list = []
    for e in getattr(state, "enemies", []) or []:
        enemies_list.append({
            "id": getattr(e, "id", ""),
            "hp": getattr(e, "hp", 0),
            "max_hp": getattr(e, "max_hp", 0),
            "block": getattr(e, "block", 0),
            "powers": dict(getattr(e, "statuses", {}) or {}),
        })

    hand = list(getattr(state, "hand", []) or [])
    draw_pile = getattr(state, "draw_pile", []) or []
    discard_pile = getattr(state, "discard_pile", []) or []
    exhaust_pile = getattr(state, "exhaust_pile", []) or []

    turn = getattr(state, "turn", getattr(state, "_turn", 0))
    bottled_assessment, bottled_error = try_bottled_assessment(rs, state)

    rec = {
        "seed": seed,
        "floor": rs.floor,
        "act": rs.act,
        "turn": turn,
        "player": player_dict,
        "enemies": enemies_list,
        "hand": hand,
        "draw_pile_size": len(draw_pile),
        "discard_pile_size": len(discard_pile),
        "exhaust_pile_size": len(exhaust_pile),
        "available_actions": [_action_to_str(a) for a in actions],
        "chosen_action": _action_to_str(chosen_action) if chosen_action is not None else "",
        "solver_scores": [],
        "bottled_ai_assessment": bottled_assessment,
        "bottled_ai_error": bottled_error,
    }
    return rec


def _action_to_str(a: Any) -> str:
    """把 action 对象简化成可识别的字符串。"""
    if a is None:
        return "None"
    at = getattr(a, "action_type", None)
    if at == "play_card":
        return f"play_card[{getattr(a, 'card_idx', '?')}->{getattr(a, 'target_idx', '?')}]"
    if at == "use_potion":
        return f"use_potion[{getattr(a, 'potion_idx', '?')}->{getattr(a, 'target_idx', '?')}]"
    if at == "end_turn":
        return "end_turn"
    if at == "select_scry_discard":
        return f"select_scry_discard[{getattr(a, 'card_idx', '?')}]"
    if at is not None:
        return str(at)
    # PathAction 等
    return f"{type(a).__name__}({a!r})"


class V8CollectorBot(V7Bot):
    """V7Bot 子类，在每个 COMBAT 决策点写 JSONL。"""

    def __init__(
        self,
        *,
        output_jsonl: Optional[str] = None,
        solver_budgets=None,
        verbose: bool = False,
    ):
        super().__init__(solver_budgets=solver_budgets, verbose=verbose)
        self._output_jsonl_path = output_jsonl
        self._jsonl_fh: Optional[TextIO] = None
        # 统计
        self.records_written = 0
        self.bottled_error_count = 0
        self.first_bottled_error: Optional[str] = None
        self._current_seed: Any = None

        if output_jsonl is not None:
            self._jsonl_fh = open(output_jsonl, "w")

        if not _BOTTLED_IMPORT_OK:
            logger.warning(
                "V8CollectorBot: v8_* import failed — bottled_ai eval will be skipped: %s",
                _BOTTLED_IMPORT_ERR,
            )

    # ------------------------------------------------------------------
    # 包装父类 run，捕获 seed 用于写 JSONL，并保证关 fh
    # ------------------------------------------------------------------

    def run(self, seed, ascension: int = 0, character: str = "ironclad",
            max_actions: int = 20_000, log_path: Optional[str] = None) -> Dict[str, Any]:
        self._current_seed = seed
        try:
            return super().run(
                seed=seed,
                ascension=ascension,
                character=character,
                max_actions=max_actions,
                log_path=log_path,
            )
        finally:
            if self._jsonl_fh is not None:
                try:
                    self._jsonl_fh.close()
                except Exception:  # noqa: BLE001
                    pass
                self._jsonl_fh = None

    # ------------------------------------------------------------------
    # 重写 _pick_action：父类决策完后写 JSONL（仅 COMBAT phase）
    # ------------------------------------------------------------------

    def _pick_action(self, runner, actions):
        action = super()._pick_action(runner, actions)

        # 只在 COMBAT phase 且 fh 已开 + action 非 None 时写
        if (
            self._jsonl_fh is not None
            and runner.phase == GamePhase.COMBAT
            and action is not None
            and runner.current_combat is not None
        ):
            try:
                rec = self._build_record(runner, actions, action)
                self._jsonl_fh.write(json.dumps(rec, default=str) + "\n")
                self._jsonl_fh.flush()
                self.records_written += 1
            except Exception as e:  # noqa: BLE001
                logger.warning("V8 collector: failed to write record: %s", e)

        return action

    # ------------------------------------------------------------------
    # 构造一条 JSONL 记录
    # ------------------------------------------------------------------

    def _build_record(self, runner, actions, chosen_action) -> Dict[str, Any]:
        rec = build_combat_record(runner, actions, chosen_action, seed=self._current_seed)
        # 加上 solver_scores（仅 collector 有意义）
        rec["solver_scores"] = list(
            getattr(self._adapter, "last_solver_scores", []) or []
        )
        # 统计 bottled_ai 错误
        if rec.get("bottled_ai_error") is not None:
            self.bottled_error_count += 1
            if self.first_bottled_error is None:
                self.first_bottled_error = rec["bottled_ai_error"]
        return rec


__all__ = ["V8CollectorBot", "build_combat_record", "try_bottled_assessment"]

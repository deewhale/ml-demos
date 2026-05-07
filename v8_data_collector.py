"""
V8 数据采集器 — V8Bot 子类，在元决策点写 JSONL（不在 COMBAT 内每步采集）。

设计方向（重要变更）
--------------------
原版本在 COMBAT 内每个出牌步采集（~200 步/局，1 个胜负信号 → 严重稀疏）。
新版本反过来：

  * COMBAT 阶段交给父类（solver/启发式擅长稀疏信号）→ **不采集**。
  * 元决策（CARD_REWARD / MAP / REST / SHOP / EVENT / NEOW / TREASURE /
    BOSS_REWARDS）步数少（一局 ~30-50 步），且每一步都对最终胜率有强影响 →
    **采集这里**，给 V8 model 学。

老师标签
--------
V8 阶段：v8_strategy 是 stub，teacher_label 暂时也只是「父类返回的 idx」（瞎选
但记录下来，让 schema/pipeline 跑通）。后续接入真启发式 / 模型再升级。

bottled_ai_assessment 字段
--------------------------
原本只在 COMBAT 有意义（依赖 combat_state）。元决策阶段 combat_state == None，
此字段统一写 None，error="not_in_combat" — 不阻塞，留给后续用 run_state 维度
的 assessment 替换（目前 v8_teacher_eval 没覆盖元决策）。

JSONL schema 见 _SCHEMA_HINT。
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, TextIO

# V8Bot 已经做了 ensure_on_sys_path，import 顺序保持一致
from v8_bot import V8Bot
from sts_paths import ensure_on_sys_path
ensure_on_sys_path()

from packages.engine.game import (  # noqa: E402
    GamePhase,
    PathAction, NeowAction, RewardAction,
    EventAction, ShopAction, RestAction, TreasureAction, BossRewardAction,
)

import v8_strategy as strat  # noqa: E402

# 失败也无所谓的 v8 模块（catch 在 try_bottled_assessment 里）
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
JSONL line schema (一条 = 一个元决策点，COMBAT 内不写):
{
  "seed": int, "floor": int, "act": int,
  "phase": str,                      # CARD_REWARD / MAP / REST / SHOP / EVENT / NEOW / TREASURE / BOSS_REWARDS
  "state": {                         # run-level 状态（不依赖 combat_state）
    "hp": int, "max_hp": int, "gold": int,
    "deck": [str],                   # 全 deck 的 card id 列表
    "relics": [str],
    "potions": [str],
    "map_position": [int|null, int|null],  # (current_floor_idx, current_x) 没有就 null
    "options": [str]                 # 当前 phase 的可选项摘要（短字符串）
  },
  "available_actions": [str],        # 原始 action 对象的简短表示（与 options 一一对应或更细）
  "action": int,                     # 父类实际选了第几个 action（idx in available_actions）
  "teacher_label": int,              # v8_strategy 老师建议的 idx（当前 stub，与 action 几乎一致）
  "teacher_assessment": {...} or null,   # COMBAT 维度的 13 维评估（元决策阶段固定 null）
  "bottled_ai_error": str or null    # 元决策阶段恒为 "not_in_combat"
}
"""


# 我们采集的元决策 phase 集合（COMBAT 显式排除）
_META_PHASES = frozenset({
    GamePhase.MAP_NAVIGATION,
    GamePhase.NEOW,
    GamePhase.COMBAT_REWARDS,
    GamePhase.EVENT,
    GamePhase.SHOP,
    GamePhase.REST,
    GamePhase.TREASURE,
    GamePhase.BOSS_REWARDS,
})


# ------------------------------------------------------------------------
# 兼容保留：try_bottled_assessment / build_combat_record 仍被 v8_inference_bot
# 引用（COMBAT 推理时用）。本次只动 collector，不改 inference_bot —— 这两个
# 函数原样保留。
# ------------------------------------------------------------------------

def try_bottled_assessment(run_state, combat_state):
    """构造 bottled_ai 评估 dict（best-effort）。返回 (assessment, error)。

    仅在 COMBAT 维度有意义；元决策阶段调用方应直接传 None 跳过。
    """
    if not _BOTTLED_IMPORT_OK:
        return None, f"import failed: {_BOTTLED_IMPORT_ERR}"

    if combat_state is None:
        return None, "not_in_combat"

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
    """COMBAT 阶段的旧格式 record（v8_inference_bot 还在用）。

    本次重构未删除：v8_inference_bot 的 encode_state 依赖此 schema。后续若把
    inference 迁到元决策，再统一删。
    """
    rs = runner.run_state
    cc = runner.current_combat
    state = cc.state if cc is not None else None

    if state is None:
        # 极端兜底
        return {
            "seed": seed, "floor": rs.floor, "act": rs.act, "turn": 0,
            "player": {}, "enemies": [], "hand": [],
            "draw_pile_size": 0, "discard_pile_size": 0, "exhaust_pile_size": 0,
            "available_actions": [_action_to_str(a) for a in actions],
            "chosen_action": _action_to_str(chosen_action) if chosen_action is not None else "",
            "solver_scores": [],
            "bottled_ai_assessment": None,
            "bottled_ai_error": "no_combat_state",
        }

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
    """把 action 对象简化成可识别的字符串（多 phase 通用）。"""
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
    if isinstance(a, PathAction):
        return f"path[node={a.node_index}]"
    if isinstance(a, NeowAction):
        return f"neow[{a.choice_index}]"
    if isinstance(a, RewardAction):
        return f"reward[{a.reward_type}:{a.choice_index}]"
    if isinstance(a, EventAction):
        return f"event[{a.choice_index}]"
    if isinstance(a, ShopAction):
        return f"shop[{a.action_type}:{a.item_index}]"
    if isinstance(a, RestAction):
        return f"rest[{a.action_type}:{a.card_index}]"
    if isinstance(a, TreasureAction):
        return f"treasure[{a.action_type}]"
    if isinstance(a, BossRewardAction):
        return f"boss_reward[{a.relic_index}]"
    if at is not None:
        return str(at)
    return f"{type(a).__name__}({a!r})"


# ------------------------------------------------------------------------
# 老师标签计算：调用 v8_strategy 的同一套函数，但**不副作用**（只读取）。
# 返回值是 actions 列表里的 idx（如果找不到，返回 0 兜底）。
# ------------------------------------------------------------------------

def _teacher_label_for_phase(runner, actions) -> int:
    """根据 phase 用 v8_strategy 算出老师建议的 action idx。

    任意异常都吞掉，返回 0（与 V8Bot 本身的 fallback 一致）。
    """
    phase = runner.phase
    rs = runner.run_state

    try:
        if phase == GamePhase.MAP_NAVIGATION:
            target = strat.pick_path(rs)
            for i, a in enumerate(actions):
                if isinstance(a, PathAction) and a.node_index == target:
                    return i
            return 0

        if phase == GamePhase.NEOW:
            blessings = runner.neow_blessings or []
            target = strat.pick_neow(blessings) if blessings else 0
            for i, a in enumerate(actions):
                if isinstance(a, NeowAction) and a.choice_index == target:
                    return i
            return 0

        if phase == GamePhase.COMBAT_REWARDS:
            # reward phase 复杂，没法只看 idx；最简化做法：找第一个 RewardAction 的位置
            # 老师真正决策被 V8Bot._pick_reward_action 一步一步驱动，每一步都
            # 进 collector，这里我们退化为「与父类返回的 action 同步」——
            # 在调用方的 _build_meta_record 里直接把 teacher_label = chosen_idx。
            return -1  # sentinel: 让调用方填 chosen_idx

        if phase == GamePhase.EVENT:
            target = strat.pick_event_action(rs, runner.current_event_state, actions)
            if 0 <= target < len(actions):
                return target
            return 0

        if phase == GamePhase.SHOP:
            target_action = strat.pick_shop_action(rs, runner.current_shop, actions)
            for i, a in enumerate(actions):
                if a is target_action:
                    return i
            return 0

        if phase == GamePhase.REST:
            target_action = strat.pick_rest_action(rs, actions)
            for i, a in enumerate(actions):
                if a is target_action:
                    return i
            return 0

        if phase == GamePhase.TREASURE:
            for i, a in enumerate(actions):
                if isinstance(a, TreasureAction) and a.action_type == "take_relic":
                    return i
            return 0

        if phase == GamePhase.BOSS_REWARDS:
            choices = None
            if runner.current_rewards and runner.current_rewards.boss_relics:
                choices = runner.current_rewards.boss_relics
            if choices is None or not choices.relics:
                return 0
            target = strat.pick_boss_relic(rs, choices)
            for i, a in enumerate(actions):
                if isinstance(a, BossRewardAction) and a.relic_index == target:
                    return i
            return 0
    except Exception as e:  # noqa: BLE001
        logger.debug("teacher_label calc failed (phase=%s): %s", phase, e)
        return 0

    return 0


def _build_meta_state(runner) -> Dict[str, Any]:
    """构造元决策阶段的 state 摘要（不依赖 combat_state）。"""
    rs = runner.run_state

    # deck（card id 列表）
    deck_ids: List[str] = []
    for c in getattr(rs, "deck", []) or []:
        cid = getattr(c, "id", None)
        if cid is not None:
            deck_ids.append(cid)

    # relics
    relics_ids: List[str] = []
    for r in getattr(rs, "relics", []) or []:
        rid = getattr(r, "id", None) or str(r)
        relics_ids.append(rid)

    # potions
    try:
        potions = list(rs.get_potions())
    except Exception:  # noqa: BLE001
        potions = []

    # map position（best-effort，不一定都有）
    cur_floor = getattr(rs, "floor", None)
    cur_x = getattr(rs, "current_map_x", None)

    return {
        "hp": getattr(rs, "current_hp", 0),
        "max_hp": getattr(rs, "max_hp", 0),
        "gold": getattr(rs, "gold", 0),
        "deck": deck_ids,
        "relics": relics_ids,
        "potions": potions,
        "map_position": [cur_floor, cur_x],
    }


def _phase_options_summary(runner, actions) -> List[str]:
    """对当前 phase 的可选项做一份「人能读懂」的摘要。"""
    phase = runner.phase
    rs = runner.run_state

    if phase == GamePhase.MAP_NAVIGATION:
        try:
            paths = rs.get_available_paths()
        except Exception:  # noqa: BLE001
            paths = []
        out = []
        for p in paths:
            rt = getattr(p, "room_type", None)
            rt_str = rt.value if rt is not None and hasattr(rt, "value") else str(rt)
            out.append(f"path[{rt_str}]")
        return out

    if phase == GamePhase.NEOW:
        blessings = runner.neow_blessings or []
        out = []
        for b in blessings:
            bt = getattr(b, "blessing_type", None)
            bt_str = bt.value if bt is not None and hasattr(bt, "value") else str(bt)
            out.append(f"neow[{bt_str}]")
        return out

    if phase == GamePhase.COMBAT_REWARDS:
        rewards = runner.current_rewards
        out = []
        if rewards is not None:
            for cr in getattr(rewards, "card_rewards", []) or []:
                if getattr(cr, "is_resolved", False):
                    continue
                ids = [getattr(c, "id", "?") for c in getattr(cr, "cards", []) or []]
                out.append(f"card_reward[{','.join(ids)}]")
            if getattr(rewards, "gold", 0):
                out.append(f"gold[{rewards.gold}]")
            if getattr(rewards, "relic", None) is not None:
                out.append(f"relic[{getattr(rewards.relic, 'id', '?')}]")
        # 加上原始 action 摘要
        return out + [_action_to_str(a) for a in actions]

    if phase == GamePhase.EVENT:
        es = runner.current_event_state
        eid = getattr(es, "event_id", "?") if es is not None else "?"
        return [f"event[{eid}]"] + [_action_to_str(a) for a in actions]

    if phase == GamePhase.SHOP:
        shop = runner.current_shop
        out = []
        if shop is not None:
            try:
                for sc in shop.get_available_colored_cards():
                    out.append(f"shop_card[{getattr(sc.card, 'id', '?')}@{sc.price}]")
            except Exception:  # noqa: BLE001
                pass
        return out + [_action_to_str(a) for a in actions]

    if phase == GamePhase.REST:
        return [_action_to_str(a) for a in actions]

    if phase == GamePhase.TREASURE:
        return [_action_to_str(a) for a in actions]

    if phase == GamePhase.BOSS_REWARDS:
        out = []
        choices = None
        if runner.current_rewards and runner.current_rewards.boss_relics:
            choices = runner.current_rewards.boss_relics
        if choices is not None:
            for r in choices.relics:
                rid = getattr(r, "id", None) or str(r)
                out.append(f"boss_relic[{rid}]")
        return out

    return [_action_to_str(a) for a in actions]


def _build_meta_record(
    runner,
    actions,
    chosen_idx: int,
    teacher_label: int,
    seed: Any,
) -> Dict[str, Any]:
    """构造元决策点的 JSONL record。"""
    rs = runner.run_state
    return {
        "seed": seed,
        "floor": rs.floor,
        "act": rs.act,
        "phase": runner.phase.name,
        "state": _build_meta_state(runner),
        "available_actions": [_action_to_str(a) for a in actions],
        "action": chosen_idx,
        "teacher_label": teacher_label if teacher_label >= 0 else chosen_idx,
        "teacher_assessment": None,           # 元决策没有 13 维 combat assessment
        "bottled_ai_error": "not_in_combat",  # 显式标记，下游统计用
    }


class V8CollectorBot(V8Bot):
    """V8Bot 子类。在每个**元决策**点写 JSONL（COMBAT 内不写）。"""

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
        self.phase_counts: Dict[str, int] = {}
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
    # 重写 _pick_action：父类决策完后，仅当 phase 是元决策时写 JSONL
    # ------------------------------------------------------------------

    def _pick_action(self, runner, actions):
        # 1) 先记录决策前的 phase（父类调用过程中可能改 phase，要锁住）
        pre_phase = runner.phase

        # 2) 在元决策点先算老师建议（只读，不副作用）；COMBAT 跳过
        teacher_label = -1
        if pre_phase in _META_PHASES and self._jsonl_fh is not None:
            teacher_label = _teacher_label_for_phase(runner, actions)

        # 3) 让父类正常驱动游戏（COMBAT 走 solver，元决策走 v8_strategy.*）
        action = super()._pick_action(runner, actions)

        # 4) 仅在元决策 phase 写一条 record
        if (
            self._jsonl_fh is not None
            and pre_phase in _META_PHASES
            and action is not None
        ):
            try:
                # 找父类返回的 action 在 actions 里的 idx
                chosen_idx = -1
                for i, a in enumerate(actions):
                    if a is action:
                        chosen_idx = i
                        break
                if chosen_idx < 0:
                    # action 可能是新构造的同类对象（罕见）；按相等回退
                    for i, a in enumerate(actions):
                        if a == action:
                            chosen_idx = i
                            break
                if chosen_idx < 0:
                    chosen_idx = 0

                rec = _build_meta_record(
                    runner=runner,
                    actions=actions,
                    chosen_idx=chosen_idx,
                    teacher_label=teacher_label,
                    seed=self._current_seed,
                )
                # 修正 phase 字段：用决策前的 phase（避免 take_action 后变了）
                rec["phase"] = pre_phase.name
                self._jsonl_fh.write(json.dumps(rec, default=str) + "\n")
                self._jsonl_fh.flush()
                self.records_written += 1
                self.phase_counts[pre_phase.name] = (
                    self.phase_counts.get(pre_phase.name, 0) + 1
                )
                # 兼容统计：元决策固定 not_in_combat（不计入异常错误）
                # bottled_error_count 保留语义为「真实异常」，元决策不递增。
            except Exception as e:  # noqa: BLE001
                logger.warning("V8 collector: failed to write meta record: %s", e)

        return action


__all__ = ["V8CollectorBot", "build_combat_record", "try_bottled_assessment"]

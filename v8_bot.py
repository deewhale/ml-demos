"""
V8 主 bot —— 从 V7Bot infra 机械迁移而来，**不依赖 v7_strategy / v7_evaluator**。

设计要点
--------
- 这个文件只承担「跑一局 STS + phase dispatch + COMBAT solver 调用 + 奖励分类」
  这些 *基础设施* 工作。元决策的真正启发式（pick_path / pick_neow / ...）
  全部交给 v8_strategy（当前是 stub，后续接 V8 model 或新启发式）。
- COMBAT 仍然用 packages.training.turn_solver 的 TurnSolverAdapter。
- 不 import v7_evaluator（patch_turn_solver_eval 是 V7 时期对 solver
  评分函数的临时 monkey-patch，V8 暂不复用）。
- 类名 V8Bot；signal SIGALRM 仅在主线程可用，子进程 / 非主线程调用时会跳过。
"""

from __future__ import annotations

import json
import logging
import signal
import time
from typing import Any, Dict, List, Optional, TextIO

# 保证能 import StSRLSolver
from sts_paths import ensure_on_sys_path
ensure_on_sys_path()

from packages.engine.game import (
    GameRunner, GamePhase,
    PathAction, NeowAction, CombatAction, RewardAction,
    EventAction, ShopAction, RestAction, TreasureAction, BossRewardAction,
)
from packages.training.turn_solver import TurnSolverAdapter
from packages.engine.content.cards import ALL_CARDS

# V8 Phase 1.4: 启用 NN value head leaf evaluator（替代 StSRLSolver 默认启发式）
# 模块加载时 patch_solver() 自动执行，幂等
# NOTE: 30-seed search action 数据采集临时关闭 patch，确保用默认 StSRLSolver 评分
# import v8_evaluator  # noqa: F401  # side-effect: patch_solver() 模块加载时执行
# TODO RL 重构时移除：v8_evaluator 已 archive 到 archive/v8_bc_pivot/，BC 路线 NN value head 已废弃

# TODO RL 重构时移除：v8_strategy（启发式 BC teacher）已 archive 到 archive/v8_bc_pivot/。
# 元决策路线已切到 RL，下面 _pick_action 里所有 strat.* 调用都需要换成 RL policy。
# 暂时挂一个 stub 模块对象，让 v8_bot 仍可 import；任何 strat.* 调用都会抛
# NotImplementedError 提醒后续 RL agent 改写。
# import v8_strategy as strat
class _StratStub:
    def __getattr__(self, name):  # noqa: D401
        raise NotImplementedError(
            f"v8_strategy.{name} 已 archive (archive/v8_bc_pivot/v8_strategy.py)。"
            " 元决策路线切到 RL，请在 RL 重构时替换调用点。"
        )

strat = _StratStub()


# 必败阈值：solver 最佳 score 低于此值视为「必败」，走 defensive fallback
_LOSS_THRESHOLD = -500_000.0

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Phase 1.1: 战斗内 turn-level state collector
# ------------------------------------------------------------------
# 每个 COMBAT turn 开始时记 1 条 snapshot；战斗结束时写 1 条 outcome record。
# 用 (combat_id, turn) 作为去重 key，防止同一 turn 内多次 _pick_action 重复写。
# 跑完一个 seed 后 flush 到 jsonl 文件。
_COMBAT_STATE_LOG: List[Dict[str, Any]] = []
_CURRENT_COMBAT_ID: Optional[str] = None
_CURRENT_COMBAT_TURNS_SEEN: set = set()  # 当前 combat 已 snapshot 的 turn idx
_CURRENT_COMBAT_FIRST_HP: Optional[int] = None  # combat 开始时玩家 HP，用于 outcome


def _make_combat_id(seed: Any, floor: int, act: int, room_type: str) -> str:
    """combat_id 唯一标识 = seed_act_floor_roomtype。"""
    return f"{seed}_a{act}_f{floor}_{(room_type or 'unknown').lower()}"


def _record_combat_turn_snapshot(runner, seed: Any) -> None:
    """战斗内每个新 turn 开始时调用，snapshot state 到 _COMBAT_STATE_LOG。

    幂等：同一 (combat_id, turn) 只写一次。
    """
    global _CURRENT_COMBAT_ID, _CURRENT_COMBAT_TURNS_SEEN, _CURRENT_COMBAT_FIRST_HP

    cc = getattr(runner, "current_combat", None)
    if cc is None:
        return
    st = getattr(cc, "state", None)
    if st is None:
        return

    rs = runner.run_state
    try:
        room_type = runner.current_room_type
    except Exception:  # noqa: BLE001
        room_type = "unknown"
    room_type_str = (room_type or "unknown").lower() if isinstance(room_type, str) else \
        (getattr(room_type, "name", "unknown") or "unknown").lower()

    combat_id = _make_combat_id(seed, rs.floor, rs.act, room_type_str)

    # 新 combat：reset 计数
    if combat_id != _CURRENT_COMBAT_ID:
        _CURRENT_COMBAT_ID = combat_id
        _CURRENT_COMBAT_TURNS_SEEN = set()
        _CURRENT_COMBAT_FIRST_HP = rs.current_hp

    turn_idx = getattr(st, "turn", -1)
    if turn_idx in _CURRENT_COMBAT_TURNS_SEEN:
        return  # 同一 turn 已 snapshot 过
    _CURRENT_COMBAT_TURNS_SEEN.add(turn_idx)

    # ---- 提取 player ----
    player = getattr(st, "player", None)
    p_hp = getattr(player, "hp", rs.current_hp) if player else rs.current_hp
    p_max_hp = getattr(player, "max_hp", rs.max_hp) if player else rs.max_hp
    p_block = getattr(player, "block", 0) if player else 0
    p_powers = dict(getattr(player, "statuses", {})) if player else {}

    energy = getattr(st, "energy", 0)
    hand = list(getattr(st, "hand", []) or [])
    draw = list(getattr(st, "draw_pile", []) or [])
    discard = list(getattr(st, "discard_pile", []) or [])
    exhaust = list(getattr(st, "exhaust_pile", []) or [])

    # ---- 提取 monsters（含已死的也跳过）----
    enemies = []
    for e in getattr(st, "enemies", []) or []:
        if getattr(e, "hp", 0) <= 0:
            continue
        enemies.append({
            "id": getattr(e, "id", ""),
            "hp": e.hp,
            "max_hp": e.max_hp,
            "block": getattr(e, "block", 0),
            "intent_dmg": getattr(e, "move_damage", 0),
            "intent_hits": getattr(e, "move_hits", 0),
            "powers": dict(getattr(e, "statuses", {})),
        })

    snap = {
        "kind": "turn",
        "combat_id": combat_id,
        "seed": seed,
        "floor": rs.floor,
        "act": rs.act,
        "room_type": room_type_str,
        "turn": turn_idx,
        "player": {
            "hp": p_hp,
            "max_hp": p_max_hp,
            "block": p_block,
            "energy": energy,
            "powers": p_powers,
            "hand": hand,
            "hand_size": len(hand),
            "draw_size": len(draw),
            "discard_size": len(discard),
            "exhaust_size": len(exhaust),
            "deck_card_names": [c for c in rs.deck],
        },
        "monsters": enemies,
    }
    _COMBAT_STATE_LOG.append(snap)


def _record_combat_end(seed: Any, runner, prev_hp: int) -> None:
    """战斗结束侦测时调用：写 outcome record。

    outcome 判定：
    - 玩家 HP <= 0 → lost
    - 否则 → won（包括因杀光怪/Burning Elite 等正常退出）
    """
    global _CURRENT_COMBAT_ID, _CURRENT_COMBAT_TURNS_SEEN, _CURRENT_COMBAT_FIRST_HP

    if _CURRENT_COMBAT_ID is None:
        return  # 没记录过 turn，跳过

    rs = runner.run_state
    final_hp = rs.current_hp
    outcome = "lost" if final_hp <= 0 else "won"
    turns_taken = len(_CURRENT_COMBAT_TURNS_SEEN)

    _COMBAT_STATE_LOG.append({
        "kind": "outcome",
        "combat_id": _CURRENT_COMBAT_ID,
        "seed": seed,
        "combat_outcome": outcome,
        "final_player_hp": final_hp,
        "turns_taken": turns_taken,
    })

    # reset，下场战斗重新开始
    _CURRENT_COMBAT_ID = None
    _CURRENT_COMBAT_TURNS_SEEN = set()
    _CURRENT_COMBAT_FIRST_HP = None


def flush_combat_log(output_path: str) -> int:
    """把 in-memory _COMBAT_STATE_LOG 落盘到 jsonl，并清空。

    返回写入的 record 条数。
    """
    global _COMBAT_STATE_LOG
    n = len(_COMBAT_STATE_LOG)
    with open(output_path, "w") as f:
        for record in _COMBAT_STATE_LOG:
            f.write(json.dumps(record, default=str) + "\n")
    _COMBAT_STATE_LOG = []
    return n


def reset_combat_log() -> None:
    """开始新一轮采集前清空 in-memory log。"""
    global _COMBAT_STATE_LOG, _CURRENT_COMBAT_ID, _CURRENT_COMBAT_TURNS_SEEN, _CURRENT_COMBAT_FIRST_HP
    _COMBAT_STATE_LOG = []
    _CURRENT_COMBAT_ID = None
    _CURRENT_COMBAT_TURNS_SEEN = set()
    _CURRENT_COMBAT_FIRST_HP = None


# ------------------------------------------------------------------
# Phase 2.0（搜索作老师）: 战斗内 search-action 数据收集
# ------------------------------------------------------------------
# 每次 _pick_action COMBAT 分支调完搜索后，记录一条
# (combat_id, turn, available_actions, chosen_action, search_info)。
# 后续用作 BC 训练数据：让 model 模仿 StSRLSolver 搜索的最优动作。
_COMBAT_ACTION_LOG: List[Dict[str, Any]] = []


def _record_combat_search_action(
    seed: Any,
    runner,
    available_actions: List,
    chosen_action: Any,
    search_info: Optional[Dict[str, Any]],
    via: str,
) -> None:
    """战斗内每次 _pick_action 决策记录。

    via: "solver" / "defensive_fallback" / "timeout_fallback" / "first_action"
        ——表明 chosen_action 来自哪条路径（用于后续过滤训练样本）。
    """
    cc = getattr(runner, "current_combat", None)
    if cc is None:
        return
    st = getattr(cc, "state", None)
    if st is None:
        return

    rs = runner.run_state
    try:
        room_type = runner.current_room_type
    except Exception:  # noqa: BLE001
        room_type = "unknown"
    room_type_str = (room_type or "unknown").lower() if isinstance(room_type, str) else \
        (getattr(room_type, "name", "unknown") or "unknown").lower()

    combat_id = _make_combat_id(seed, rs.floor, rs.act, room_type_str)
    turn_idx = getattr(st, "turn", -1)

    # available_actions 字符串化
    avail_strs = [repr(a) for a in available_actions]
    chosen_str = repr(chosen_action) if chosen_action is not None else None

    # chosen_idx: chosen 在 available_actions 中的位置（identity 优先，其次 repr 比较）
    chosen_idx: Optional[int] = None
    if chosen_action is not None:
        for i, a in enumerate(available_actions):
            if a is chosen_action:
                chosen_idx = i
                break
        if chosen_idx is None:
            for i, s in enumerate(avail_strs):
                if s == chosen_str:
                    chosen_idx = i
                    break

    rec = {
        "kind": "combat_action",
        "combat_id": combat_id,
        "seed": seed,
        "floor": rs.floor,
        "act": rs.act,
        "room_type": room_type_str,
        "turn": turn_idx,
        "available_actions": avail_strs,
        "n_available": len(avail_strs),
        "chosen_action": chosen_str,
        "chosen_idx": chosen_idx,
        "via": via,
        "search_best_score": (search_info or {}).get("best_score"),
        "search_hard_timeout": (search_info or {}).get("hard_timeout"),
        "search_n_scores": (search_info or {}).get("n_scores"),
    }
    _COMBAT_ACTION_LOG.append(rec)


def flush_combat_action_log(output_path: str) -> int:
    """把 _COMBAT_ACTION_LOG 落盘到 jsonl（追加模式）并清空。返回写入条数。"""
    global _COMBAT_ACTION_LOG
    n = len(_COMBAT_ACTION_LOG)
    with open(output_path, "a") as f:
        for record in _COMBAT_ACTION_LOG:
            f.write(json.dumps(record, default=str) + "\n")
    _COMBAT_ACTION_LOG = []
    return n


def reset_combat_action_log() -> None:
    """开始新一轮采集前清空 search-action log。"""
    global _COMBAT_ACTION_LOG
    _COMBAT_ACTION_LOG = []


# Solver 预算（迁自 V7；V8 阶段 1.5 放宽 cap 让 elite/boss 战不再 hard-timeout）
# 元组语义：(time_budget_ms, node_budget, cap_ms)
# cap_ms 决定 v8_bot.run 里 SIGALRM 的 hard_timeout = cap_ms/1000 + 2 秒
SOLVER_BUDGETS = {
    "monster": (50.0,    5_000,    3_000),   # 50ms, cap 3s   (原 2s, +50%)
    "elite":   (500.0,   20_000,  12_000),   # 500ms, cap 12s (原 2s)
    "boss":    (2_000.0, 50_000,  25_000),   # 2s, cap 25s  (阶段 4 老师改进：让 boss 战多给 5s)
}


class V8Bot:
    """V8 主 bot：GameRunner + TurnSolver + v8_strategy（stub）调度。"""

    def __init__(self, solver_budgets=None, verbose: bool = False):
        self._budgets = solver_budgets or SOLVER_BUDGETS
        self._verbose = verbose
        self._adapter = TurnSolverAdapter(
            time_budget_ms=50.0,
            node_budget=5_000,
            solver_budgets=self._budgets,
        )

    def run(
        self,
        seed,
        ascension: int = 0,
        character: str = "ironclad",
        max_actions: int = 20_000,
        log_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """跑一局，返回 summary dict。"""
        t0 = time.monotonic()
        # Phase 1.1: 让 _pick_action / _record_combat_turn_snapshot 能拿到 seed
        self._current_seed = seed
        runner = GameRunner(
            seed=seed,
            ascension=ascension,
            character=character,
            skip_neow=False,    # V8 手动处理 Neow
            verbose=self._verbose,
        )
        self._adapter.reset()

        floor_hp: List[tuple] = []
        prev_floor = -1
        actions_taken = 0
        error_msg = None

        # ---------- per-floor logging ----------
        log_fh: Optional[TextIO] = open(log_path, "w") if log_path else None
        last_combat_sig = None  # (floor, room_type) of most recent combat-start we logged

        def _emit(rec):
            if log_fh is None:
                return
            try:
                log_fh.write(json.dumps(rec, default=str) + "\n")
                log_fh.flush()
            except Exception:  # noqa: BLE001
                pass

        def _room_type_str() -> str:
            try:
                rt = runner.get_current_room_type()
                return rt.name if rt is not None else "START"
            except Exception:  # noqa: BLE001
                return "UNKNOWN"

        def _snapshot_floor(tag: str):
            rs = runner.run_state
            rec = {
                "event": tag,
                "t": round(time.monotonic() - t0, 3),
                "seed": seed,
                "floor": rs.floor,
                "act": rs.act,
                "phase": runner.phase.name,
                "room_type": _room_type_str(),
                "hp": rs.current_hp,
                "max_hp": rs.max_hp,
                "gold": rs.gold,
                "deck_size": len(rs.deck),
                "relics": [r.id for r in rs.relics][:30],
                "actions_taken": actions_taken,
            }
            _emit(rec)

        def _snapshot_combat_start():
            nonlocal last_combat_sig
            cc = runner.current_combat
            rs = runner.run_state
            sig = (rs.floor, runner.current_room_type)
            if sig == last_combat_sig:
                return
            last_combat_sig = sig
            enemies = []
            if cc is not None:
                for e in cc.state.enemies:
                    enemies.append({
                        "id": getattr(e, "id", ""),
                        "name": getattr(e, "name", ""),
                        "hp": e.hp,
                        "max_hp": e.max_hp,
                        "type": getattr(e, "enemy_type", ""),
                    })
            _emit({
                "event": "combat_start",
                "t": round(time.monotonic() - t0, 3),
                "seed": seed,
                "floor": rs.floor,
                "act": rs.act,
                "room_type": runner.current_room_type,
                "hp": rs.current_hp,
                "max_hp": rs.max_hp,
                "enemies": enemies,
                "deck_size": len(rs.deck),
                "actions_taken": actions_taken,
            })

        prev_phase = None
        prev_hp = runner.run_state.current_hp
        # ---------------------------------------

        while not runner.game_over and actions_taken < max_actions:
            actions = runner.get_available_actions()
            if not actions:
                break

            cur_floor = runner.run_state.floor
            if cur_floor != prev_floor:
                floor_hp.append((cur_floor, runner.run_state.current_hp))
                prev_floor = cur_floor
                _snapshot_floor("floor_enter")

            # 侦测 combat 开始
            if (runner.phase.name == "COMBAT"
                    and prev_phase != "COMBAT"
                    and runner.current_combat is not None):
                _snapshot_combat_start()
                prev_hp = runner.run_state.current_hp
            # 侦测 combat 结束（离开 COMBAT phase）
            if prev_phase == "COMBAT" and runner.phase.name != "COMBAT":
                rs = runner.run_state
                _emit({
                    "event": "combat_end",
                    "t": round(time.monotonic() - t0, 3),
                    "seed": seed,
                    "floor": rs.floor,
                    "hp": rs.current_hp,
                    "hp_lost": prev_hp - rs.current_hp,
                    "actions_taken": actions_taken,
                })
                # Phase 1.1: 记 combat outcome
                try:
                    _record_combat_end(seed, runner, prev_hp)
                except Exception as _e:  # noqa: BLE001
                    logger.warning("V8 combat collector: _record_combat_end failed: %s", _e)
                prev_hp = rs.current_hp
            prev_phase = runner.phase.name

            try:
                action = self._pick_action(runner, actions)
            except Exception as e:  # noqa: BLE001
                error_msg = f"_pick_action crash: {type(e).__name__}: {e}"
                logger.exception("V8 _pick_action crashed")
                # fallback 第一个
                action = actions[0]

            if action is None:
                action = actions[0]

            ok = runner.take_action(action)
            if not ok:
                # 跳过非法 action，防止死循环
                logger.warning("take_action returned False at floor=%d phase=%s action=%s",
                               runner.run_state.floor, runner.phase.name, action)
                # 随便选一个有效 action
                action = actions[0]
                runner.take_action(action)
            actions_taken += 1

        # Phase 1.1: 游戏结束时若最后一战还没写 outcome（玩家在 COMBAT 中死亡），补一条
        if _CURRENT_COMBAT_ID is not None:
            try:
                _record_combat_end(seed, runner, prev_hp)
            except Exception as _e:  # noqa: BLE001
                logger.warning("V8 combat collector: end-of-run _record_combat_end failed: %s", _e)

        elapsed = time.monotonic() - t0
        stats = runner.get_run_statistics() if hasattr(runner, "get_run_statistics") else {}

        # final 记录
        if log_fh is not None:
            try:
                _emit({
                    "event": "game_end",
                    "t": round(elapsed, 3),
                    "seed": seed,
                    "floor": runner.run_state.floor,
                    "hp": runner.run_state.current_hp,
                    "won": bool(stats.get("game_won", False)),
                    "actions_taken": actions_taken,
                    "error": error_msg,
                })
                log_fh.close()
            except Exception:  # noqa: BLE001
                pass

        return {
            "seed": seed,
            "ascension": ascension,
            "character": character,
            "game_won": bool(stats.get("game_won", False)),
            "game_lost": bool(stats.get("game_lost", runner.game_over and not stats.get("game_won", False))),
            "final_floor": runner.run_state.floor,
            "final_act": stats.get("final_act", runner.run_state.act),
            "final_hp": runner.run_state.current_hp,
            "max_hp": runner.run_state.max_hp,
            "deck_size": len(runner.run_state.deck),
            "gold": runner.run_state.gold,
            "decisions": actions_taken,
            "elapsed_s": round(elapsed, 2),
            "floor_hp": floor_hp,
            "error": error_msg,
        }

    # ---------------------------------------------------------------
    # Defensive fallback —— solver 必败时的兜底（迁自 V7 修法 A）
    # ---------------------------------------------------------------

    def _defensive_fallback_if_losing(self, runner, actions):
        """如果 solver 认为必败（最佳 score < _LOSS_THRESHOLD）且玩家 HP>0，
        返回一个 defensive action（优先打 block 最大的可负担 skill，其次 0 费 block 牌）。
        否则返回 None，让主路径使用 solver 结果。
        """
        scores = getattr(self._adapter, "last_solver_scores", None)
        if not scores:
            return None
        # scores 是 List[Tuple[desc, score]]，取最大 score
        try:
            best = max(s for _, s in scores)
        except (TypeError, ValueError):
            return None
        if best >= _LOSS_THRESHOLD:
            return None

        engine = getattr(runner, "current_combat", None)
        if engine is None:
            return None
        state = engine.state
        player = getattr(state, "player", None)
        if player is None or getattr(player, "hp", 0) <= 0:
            return None

        hand = list(getattr(state, "hand", []) or [])
        energy = getattr(state, "energy", 0)

        # 找可负担（cost<=energy 或 cost==0）的、base_block>0 的牌
        best_affordable = None  # (block, cost, card_idx)
        best_free = None
        for i, card_id in enumerate(hand):
            data = ALL_CARDS.get(card_id)
            if data is None:
                continue
            block = getattr(data, "base_block", -1)
            if block is None or block <= 0:
                continue
            cost = getattr(data, "cost", 0) or 0
            # X-cost 等异常成本，保守跳过
            if cost < 0:
                continue
            if cost == 0:
                if best_free is None or block > best_free[0]:
                    best_free = (block, cost, i)
            if cost <= energy:
                if best_affordable is None or block > best_affordable[0]:
                    best_affordable = (block, cost, i)

        # 1) 能量允许：block 最高的 skill
        if best_affordable is not None:
            a = self._find_play(actions, best_affordable[2])
            if a is not None:
                return a
        # 2) 能量不够：block 最高的 0 费 skill
        if best_free is not None:
            a = self._find_play(actions, best_free[2])
            if a is not None:
                return a
        # 3) 都没有：end_turn
        for a in actions:
            if isinstance(a, CombatAction) and a.action_type == "end_turn":
                return a
        return None

    @staticmethod
    def _find_play(actions, card_idx):
        """在 actions 里找 play_card[card_idx] 这个动作。"""
        for a in actions:
            if isinstance(a, CombatAction) and a.action_type == "play_card" and a.card_idx == card_idx:
                return a
        return None

    # ---------------------------------------------------------------
    # Phase dispatch
    # ---------------------------------------------------------------

    def _pick_action(self, runner, actions):
        phase = runner.phase
        rs = runner.run_state

        if phase == GamePhase.COMBAT:
            room_type = runner.current_room_type  # "monster" / "elite" / "boss"

            # === Phase 1.1: 战斗内 turn-level state collector ===
            # 每个新 turn 开始时 snapshot 一次 (combat_id, turn) 去重
            try:
                # seed 在 run() 闭包里；这里通过 runner 拿不到，改从 cc.state 关联不到 seed。
                # 用 runner.run_state 的 act/floor + module-level _CURRENT_COMBAT_ID 接力。
                # seed 由 _record_combat_turn_snapshot 第二个参数传入，从 self._current_seed 取。
                _seed = getattr(self, "_current_seed", None)
                _record_combat_turn_snapshot(runner, _seed)
            except Exception as _e:  # noqa: BLE001
                logger.warning("V8 combat collector: _record_combat_turn_snapshot failed: %s", _e)

            # === case_study instrument: boss 战 turn-level log（写 stderr） ===
            # 仅 boss 房间记录，避免 spam。打印玩家 HP/block/energy/手牌、敌人 HP/intent
            _is_boss_room = (str(room_type or "").lower() == "boss")
            _boss_dbg_pre = None
            if _is_boss_room:
                try:
                    import sys as _sys
                    cc = runner.current_combat
                    rs = runner.run_state
                    if cc is not None:
                        st = cc.state
                        player = getattr(st, "player", None)
                        p_hp = getattr(player, "hp", rs.current_hp) if player else rs.current_hp
                        p_max_hp = getattr(player, "max_hp", rs.max_hp) if player else rs.max_hp
                        p_block = getattr(player, "block", 0) if player else 0
                        p_powers = dict(getattr(player, "statuses", {})) if player else {}
                        energy = getattr(st, "energy", 0)
                        hand = list(getattr(st, "hand", []) or [])
                        turn_idx = getattr(st, "turn", -1)
                        enemies = []
                        for e in getattr(st, "enemies", []) or []:
                            if getattr(e, "hp", 0) <= 0:
                                continue
                            enemies.append({
                                "id": getattr(e, "id", ""),
                                "hp": e.hp,
                                "max_hp": e.max_hp,
                                "block": getattr(e, "block", 0),
                                "intent_dmg": getattr(e, "move_damage", 0),
                                "intent_hits": getattr(e, "move_hits", 0),
                                "powers": dict(getattr(e, "statuses", {})),
                            })
                        _boss_dbg_pre = {
                            "floor": rs.floor, "turn": turn_idx,
                            "p_hp": p_hp, "p_max": p_max_hp,
                            "p_block": p_block, "energy": energy,
                            "p_powers": p_powers,
                            "hand": hand,
                            "enemies": enemies,
                        }
                        print(f"[BOSS_TURN] {json.dumps(_boss_dbg_pre, default=str)}",
                              file=_sys.stderr, flush=True)
                except Exception as _e:  # noqa: BLE001
                    print(f"[BOSS_TURN_ERR] {_e!r}", file=__import__('sys').stderr, flush=True)

            # === Solver timeout fix（迁自 V7） ===
            # TurnSolverAdapter._apply_room_type_budgets 只设置 inner solver,
            # 没设 MultiTurnSolver.time_budget_ms (默认 30s). 又因为 pick_action
            # 先调用 _get_turn_candidates(deadline=30s) 再 solve(30s), 单步 action
            # 最多 60s, elite+boss 组合爆炸可 >90s.
            # 修法: 把 multi_turn 外层预算强制拉到 cap_ms, 并用 SIGALRM 做硬超时保险.
            _rt_key = (room_type or "monster").lower()
            budgets = self._budgets.get(_rt_key)
            cap_ms = budgets[2] if budgets else 3000.0
            # 同步 multi_turn 的外层 deadline 与 cap_ms 一致
            try:
                self._adapter._multi_turn.time_budget_ms = float(cap_ms)
            except AttributeError:
                pass

            # 硬超时保险 (SIGALRM, 主线程), 预算 +2s buffer
            hard_timeout_s = max(1, int(cap_ms / 1000.0) + 2)
            a = None
            timed_out = False
            old_handler = None
            alarm_set = False

            def _on_alarm(signum, frame):
                raise TimeoutError(f"solver hard timeout {hard_timeout_s}s (room={_rt_key})")

            try:
                # signal.SIGALRM 仅主线程可用；子线程 / 非主线程调用会抛 ValueError，
                # 直接跳过硬超时保险（solver 自己的 deadline 仍生效）。
                try:
                    old_handler = signal.signal(signal.SIGALRM, _on_alarm)
                    signal.alarm(hard_timeout_s)
                    alarm_set = True
                except (ValueError, OSError):
                    alarm_set = False
                a = self._adapter.pick_action(actions, runner, room_type=room_type)
            except TimeoutError as te:
                logger.warning("V8 solver hard-timeout: %s, fallback to defensive/first action", te)
                timed_out = True
                a = None
            finally:
                if alarm_set:
                    signal.alarm(0)
                    if old_handler is not None:
                        signal.signal(signal.SIGALRM, old_handler)

            # ---- 收集 search 元信息（best_score / n_scores），供下面 record / 日志共用 ----
            _scores = getattr(self._adapter, "last_solver_scores", None)
            _best = None
            _n_scores = 0
            if _scores:
                _n_scores = len(_scores)
                try:
                    _best = max(s for _, s in _scores)
                except Exception:  # noqa: BLE001
                    _best = None
            _search_info = {
                "best_score": _best,
                "hard_timeout": bool(timed_out),
                "n_scores": _n_scores,
            }

            # solver 返回「必败」（最佳 score < _LOSS_THRESHOLD）
            # 且当前 HP > 0 时，走 defensive fallback 先苟活几回合
            fallback = self._defensive_fallback_if_losing(runner, actions)
            if fallback is not None:
                # Phase 2.0: 记录 search-action（即使最终走了 defensive_fallback，
                # 也把 solver 的 best_score 一并记下，便于后续过滤"必败"样本）
                try:
                    _seed_rec = getattr(self, "_current_seed", None)
                    _record_combat_search_action(
                        seed=_seed_rec, runner=runner,
                        available_actions=actions, chosen_action=fallback,
                        search_info=_search_info, via="defensive_fallback",
                    )
                except Exception as _e:  # noqa: BLE001
                    logger.warning("V8 combat collector: _record_combat_search_action failed: %s", _e)
                if _is_boss_room:
                    try:
                        import sys as _sys
                        print(f"[BOSS_ACTION] chosen={fallback!r} via=defensive_fallback best_score={_best}",
                              file=_sys.stderr, flush=True)
                    except Exception:
                        pass
                return fallback
            if a is None and timed_out:
                # 超时 fallback: 优先 EndTurn / 第一个合法动作
                _to_chosen = None
                for act in actions:
                    if getattr(act, "action_type", None) == "end_turn":
                        _to_chosen = act
                        break
                if _to_chosen is None:
                    _to_chosen = actions[0]
                # Phase 2.0: 记录 timeout fallback 样本
                try:
                    _seed_rec = getattr(self, "_current_seed", None)
                    _record_combat_search_action(
                        seed=_seed_rec, runner=runner,
                        available_actions=actions, chosen_action=_to_chosen,
                        search_info=_search_info, via="timeout_fallback",
                    )
                except Exception as _e:  # noqa: BLE001
                    logger.warning("V8 combat collector: _record_combat_search_action failed: %s", _e)
                if _is_boss_room:
                    import sys as _sys
                    print(f"[BOSS_ACTION] chosen={_to_chosen!r} via=timeout_fallback",
                          file=_sys.stderr, flush=True)
                return _to_chosen
            chosen = a if a is not None else actions[0]
            _via = "solver" if a is not None else "first_action"
            # Phase 2.0: 记录 solver 选的最佳 action（核心训练数据来源）
            try:
                _seed_rec = getattr(self, "_current_seed", None)
                _record_combat_search_action(
                    seed=_seed_rec, runner=runner,
                    available_actions=actions, chosen_action=chosen,
                    search_info=_search_info, via=_via,
                )
            except Exception as _e:  # noqa: BLE001
                logger.warning("V8 combat collector: _record_combat_search_action failed: %s", _e)
            if _is_boss_room:
                try:
                    import sys as _sys
                    print(f"[BOSS_ACTION] chosen={chosen!r} via=solver best_score={_best}",
                          file=_sys.stderr, flush=True)
                except Exception:
                    pass
            return chosen

        # 进入其他 phase 前 reset adapter cache
        self._adapter.reset()

        if phase == GamePhase.MAP_NAVIGATION:
            idx = strat.pick_path(rs)
            # 保证是有效的 PathAction
            for a in actions:
                if isinstance(a, PathAction) and a.node_index == idx:
                    return a
            return actions[0]

        if phase == GamePhase.NEOW:
            if runner.neow_blessings is None:
                # 触发初始化
                runner.get_available_actions()
            blessings = runner.neow_blessings or []
            idx = strat.pick_neow(blessings) if blessings else 0
            # 找对应 NeowAction
            for a in actions:
                if isinstance(a, NeowAction) and a.choice_index == idx:
                    return a
            return actions[0]

        if phase == GamePhase.COMBAT_REWARDS:
            return self._pick_reward_action(runner, actions)

        if phase == GamePhase.EVENT:
            idx = strat.pick_event_action(rs, runner.current_event_state, actions)
            if 0 <= idx < len(actions):
                return actions[idx]
            return actions[0]

        if phase == GamePhase.SHOP:
            return strat.pick_shop_action(rs, runner.current_shop, actions)

        if phase == GamePhase.REST:
            return strat.pick_rest_action(rs, actions)

        if phase == GamePhase.TREASURE:
            # 默认拿 relic；sapphire key 跳过（A0 一般用不上）
            for a in actions:
                if isinstance(a, TreasureAction) and a.action_type == "take_relic":
                    return a
            return actions[0]

        if phase == GamePhase.BOSS_REWARDS:
            # 从 BossRelicChoices 里挑
            choices = None
            if runner.current_rewards and runner.current_rewards.boss_relics:
                choices = runner.current_rewards.boss_relics
            if choices is None or not choices.relics:
                return actions[0]
            idx = strat.pick_boss_relic(rs, choices)
            for a in actions:
                if isinstance(a, BossRewardAction) and a.relic_index == idx:
                    return a
            return actions[0]

        return actions[0]

    # ---------------------------------------------------------------
    # Reward phase
    # ---------------------------------------------------------------

    def _pick_reward_action(self, runner, actions):
        rs = runner.run_state
        rewards = runner.current_rewards

        # 分类
        gold_a = None
        potion_claim = None
        potion_skip = None
        emerald_claim = None
        emerald_skip = None
        relic_a = None
        proceed_a = None
        card_actions: List = []      # (card_reward_idx, action)
        skip_card_actions: List = [] # (card_reward_idx, action)
        singing_actions: List = []

        for a in actions:
            if not isinstance(a, RewardAction):
                continue
            t = a.reward_type
            if t == "gold":
                gold_a = a
            elif t == "potion":
                potion_claim = a
            elif t == "skip_potion":
                potion_skip = a
            elif t == "emerald_key":
                emerald_claim = a
            elif t == "skip_emerald_key":
                emerald_skip = a
            elif t == "relic":
                relic_a = a
            elif t == "proceed":
                proceed_a = a
            elif t == "card":
                reward_idx = a.choice_index // 100
                card_actions.append((reward_idx, a))
            elif t == "skip_card":
                skip_card_actions.append((a.choice_index, a))
            elif t == "singing_bowl":
                singing_actions.append(a)

        # 优先级：gold -> relic -> emerald -> card -> potion -> proceed
        if gold_a is not None:
            return gold_a
        if relic_a is not None:
            return relic_a
        if emerald_claim is not None:
            return emerald_claim

        # 处理卡牌奖励
        if card_actions and rewards is not None:
            # 按 reward_idx 分组，对每组选最好的
            # 找第一个未解决的 card reward
            for i, cr in enumerate(rewards.card_rewards):
                if cr.is_resolved:
                    continue
                # 在 card_actions 里找对应 reward_idx = i 的
                my_card_actions = [(ri, a) for ri, a in card_actions if ri == i]
                if not my_card_actions:
                    # 对应的 skip
                    for ri, a in skip_card_actions:
                        if ri == i:
                            return a
                    continue
                pick_idx = strat.pick_card_reward(rs, cr)
                if pick_idx is None:
                    # skip
                    for ri, a in skip_card_actions:
                        if ri == i:
                            return a
                    # 没 skip action 就拿第一张
                    return my_card_actions[0][1]
                # 找对应 card_index = pick_idx 的 action
                for ri, a in my_card_actions:
                    card_idx_in_action = a.choice_index % 100
                    if card_idx_in_action == pick_idx:
                        return a
                # fallback
                return my_card_actions[0][1]

        # Potion：MVP 直接不拿（留给 boss 的原则下，我们其实也没真正留）
        # 简化：有 claim 就拿，满了就 skip
        if potion_claim is not None:
            return potion_claim
        if potion_skip is not None:
            return potion_skip

        if emerald_skip is not None:
            return emerald_skip

        if proceed_a is not None:
            return proceed_a

        return actions[0]


__all__ = ["V8Bot", "SOLVER_BUDGETS"]

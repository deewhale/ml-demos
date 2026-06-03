"""StSRLBackend：把 StSRLSolver Python 引擎包成 GameBackend。

第一个 GameBackend 实现。内部持有：
- GameRunner（生命周期 / 状态 / 动作）
- TurnSolverAdapter（战斗内搜索）

行为契约：与重构前 env.py 直接调用 GameRunner / TurnSolverAdapter 逐位一致
（stage1 纯重构）。state 构造 / room type 归一 / snapshot / combat turn 的具体
逻辑从 env.py 原样搬过来，未做任何数值/控制流改动。

pass-through 债见 protocol.py 顶部说明：run_state / current_combat /
current_event_state / event_handler / current_rewards / current_shop /
neow_blessings / last_combat_card_log / phase / get_available_actions 仍直接返回
引擎对象，stage2 接真机 / Rust 时中性化。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

# 保证能 import StSRLSolver
from sts_paths import ensure_on_sys_path

ensure_on_sys_path()

from packages.engine.game import (  # noqa: E402
    GameRunner,
    GamePhase,
    CombatAction,
)
from packages.training.turn_solver import TurnSolverAdapter  # noqa: E402

from v8.state import V8State
from v8.action_space import get_available_actions


logger = logging.getLogger(__name__)


# StSRLSolver GamePhase → V8 phase 名 映射（与 env 内一致；state 构造用）
_ENGINE_TO_V8_PHASE: Dict[Any, str] = {
    GamePhase.NEOW: "NEOW",
    GamePhase.MAP_NAVIGATION: "MAP",
    GamePhase.COMBAT: "COMBAT",
    GamePhase.COMBAT_REWARDS: "CARD_REWARDS",
    GamePhase.EVENT: "EVENT",
    GamePhase.SHOP: "SHOP",
    GamePhase.REST: "REST",
    GamePhase.TREASURE: "TREASURE",
    GamePhase.BOSS_REWARDS: "BOSS_REWARDS",
    GamePhase.RUN_COMPLETE: "RUN_COMPLETE",
}


# 房间名归一表（从 env._normalize_room_name 搬来，行为一致）
_ROOM_NAME_MAP = {
    "monster": "monster",
    "elite": "elite",
    "boss": "boss",
    "rest": "rest",
    "campfire": "rest",
    "shop": "shop",
    "event": "event",
    "question": "event",
    "?": "event",
    "treasure": "treasure",
    "t": "treasure",
    "true_victory": "boss",
}


def _safe_action_repr(action: Any) -> str:
    """把任意 GameAction 转成一行紧凑字符串（与 env._safe_action_repr 同逻辑）。"""
    if action is None:
        return "None"
    try:
        cls_name = type(action).__name__
        atype = getattr(action, "action_type", None)
        if atype is not None:
            parts = [str(atype)]
            cid = getattr(action, "card_id", None)
            if cid:
                parts.append(str(cid))
            tgt = getattr(action, "target_index", None)
            if tgt is not None and tgt != -1:
                parts.append(f"t{tgt}")
            return f"{cls_name}:{'|'.join(parts)}"
        dst = getattr(action, "dst_x", None)
        if dst is not None:
            return f"{cls_name}:dst={dst}"
        return cls_name
    except Exception:  # noqa: BLE001
        return type(action).__name__


def _build_state_from_runner(runner: GameRunner) -> V8State:
    """从 GameRunner 提取 V8State（从 env._build_state_from_runner 原样搬来）。"""
    rs = runner.run_state
    phase_name = _ENGINE_TO_V8_PHASE.get(runner.phase, runner.phase.name)

    hp = int(getattr(rs, "current_hp", 0) or 0)
    max_hp = int(getattr(rs, "max_hp", 0) or 0)
    floor = int(getattr(rs, "floor", 0) or 0)
    act = int(getattr(rs, "act", 1) or 1)
    gold = int(getattr(rs, "gold", 0) or 0)

    potions: List[str] = []
    for slot in getattr(rs, "potion_slots", []) or []:
        pid = getattr(slot, "potion_id", None)
        potions.append(pid if pid else "")

    deck: List[Dict[str, Any]] = []
    for c in getattr(rs, "deck", []) or []:
        deck.append({
            "name": getattr(c, "id", "") or "",
            "upgraded": bool(getattr(c, "upgraded", False)),
        })

    relics: List[str] = []
    for r in getattr(rs, "relics", []) or []:
        rid = getattr(r, "id", None) or str(r)
        relics.append(rid)

    map_nodes: List[List[Dict[str, Any]]] = []
    try:
        cur_map = rs.act_maps.get(act) if getattr(rs, "act_maps", None) else None
    except Exception:  # noqa: BLE001
        cur_map = None
    if cur_map is not None:
        for layer in cur_map:
            layer_repr: List[Dict[str, Any]] = []
            for node in layer:
                rt = getattr(node, "room_type", None)
                rt_str = rt.value if rt is not None and hasattr(rt, "value") else str(rt or "")
                edges_x = []
                for e in getattr(node, "edges", []) or []:
                    edges_x.append(int(getattr(e, "dst_x", -1)))
                layer_repr.append({
                    "room_type": rt_str,
                    "x": int(getattr(node, "x", -1)),
                    "edges": edges_x,
                })
            map_nodes.append(layer_repr)

    current_position: Optional[Dict[str, int]] = None
    mp = getattr(rs, "map_position", None)
    if mp is not None and not (mp.x == -1 and mp.y == -1):
        current_position = {"floor": int(mp.y), "x": int(mp.x)}

    boss_name = str(getattr(runner, "_boss_name", "") or "")

    return V8State(
        hp=hp,
        max_hp=max_hp,
        floor=floor,
        act=act,
        gold=gold,
        potions=potions,
        deck=deck,
        relics=relics,
        map_nodes=map_nodes,
        current_position=current_position,
        deck_strength=None,
        in_combat=(runner.phase == GamePhase.COMBAT),
        phase=phase_name,
        boss=boss_name,
    )


class StSRLBackend:
    """StSRLSolver Python 引擎的 GameBackend 实现。"""

    def __init__(
        self,
        *,
        character: str = "ironclad",
        ascension: int = 0,
        verbose: bool = False,
        solver_budgets: Optional[Dict[str, Tuple[float, int, int]]] = None,
        combat_net_wrapper: Optional[Any] = None,
    ):
        self.character = character
        self.ascension = ascension
        self.verbose = verbose
        self._solver_budgets = solver_budgets
        self._combat_net_wrapper = combat_net_wrapper

        self._runner: Optional[GameRunner] = None
        self._adapter: Optional[TurnSolverAdapter] = None

        # perf 计数（env 读取累加）
        self.combat_search_calls: int = 0

    # =====================================================================
    # 1. 生命周期
    # =====================================================================
    def reset(self, seed: int) -> None:
        """开新局：创建 GameRunner + TurnSolverAdapter（不跑 advance loop）。

        逻辑从 env.reset() 的 runner/adapter 创建段原样搬来，行为一致：
        阶段 0 随机桩已禁用（不传 combat_net），走纯手写启发。
        """
        self._runner = GameRunner(
            seed=seed,
            ascension=self.ascension,
            character=self.character,
            skip_neow=False,
            verbose=self.verbose,
        )
        adapter_kwargs: Dict[str, Any] = dict(
            time_budget_ms=50.0,
            node_budget=5_000,
            solver_budgets=self._solver_budgets,
        )
        # 阶段 0：随机桩已禁用（不传 combat_net）；待阶段 5 接真网络后恢复下面这行。
        # if self._combat_net_wrapper is not None:
        #     adapter_kwargs["combat_net"] = self._combat_net_wrapper
        self._adapter = TurnSolverAdapter(**adapter_kwargs)
        self._adapter.reset()
        self.combat_search_calls = 0

    @property
    def game_over(self) -> bool:
        return bool(self._runner.game_over) if self._runner is not None else True

    @property
    def game_won(self) -> bool:
        return bool(self._runner.game_won) if self._runner is not None else False

    @property
    def phase(self) -> Any:
        return self._runner.phase if self._runner is not None else None

    def force_terminate(self) -> None:
        """强制把 run 标 terminal（从 env._force_terminate_run 搬，行为一致）。"""
        assert self._runner is not None
        try:
            self._runner.game_over = True
            self._runner.game_won = False
            self._runner.phase = GamePhase.RUN_COMPLETE
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "StSRLBackend.force_terminate: set attrs failed: %s: %s",
                type(e).__name__, e,
            )

    def close(self) -> None:
        self._runner = None
        self._adapter = None

    # =====================================================================
    # 2. 状态
    # =====================================================================
    def build_v8_state(self) -> V8State:
        assert self._runner is not None
        return _build_state_from_runner(self._runner)

    @property
    def run_state(self) -> Any:
        return self._runner.run_state if self._runner is not None else None

    @property
    def current_room_type(self) -> Any:
        return getattr(self._runner, "current_room_type", None) if self._runner is not None else None

    def get_current_room_type(self) -> Any:
        if self._runner is None:
            return None
        getter = getattr(self._runner, "get_current_room_type", None)
        if getter is None:
            return None
        return getter()

    @property
    def current_combat(self) -> Any:
        return getattr(self._runner, "current_combat", None) if self._runner is not None else None

    @property
    def current_event_state(self) -> Any:
        return getattr(self._runner, "current_event_state", None) if self._runner is not None else None

    @property
    def event_handler(self) -> Any:
        return getattr(self._runner, "event_handler", None) if self._runner is not None else None

    @property
    def current_rewards(self) -> Any:
        return getattr(self._runner, "current_rewards", None) if self._runner is not None else None

    @property
    def current_shop(self) -> Any:
        return getattr(self._runner, "current_shop", None) if self._runner is not None else None

    @property
    def neow_blessings(self) -> Any:
        return getattr(self._runner, "neow_blessings", None) if self._runner is not None else None

    @property
    def last_combat_card_log(self) -> Any:
        return getattr(self._runner, "last_combat_card_log", None) if self._runner is not None else None

    @property
    def boss_name(self) -> str:
        if self._runner is None:
            return ""
        return str(getattr(self._runner, "_boss_name", "") or "")

    @property
    def seed_string(self) -> str:
        if self._runner is None:
            return ""
        return str(getattr(self._runner, "seed_string", "") or "")

    @property
    def seed_int(self) -> Optional[int]:
        if self._runner is None:
            return None
        return getattr(self._runner, "seed", None)

    def build_runner_snapshot(self) -> Dict[str, Any]:
        """floor/act/.../game_won 纯 dict（从 env._build_runner_snapshot 搬）。"""
        runner = self._runner
        if runner is None:
            return {
                "floor": 0,
                "act": 1,
                "hp": 0,
                "max_hp": 0,
                "gold": 0,
                "game_won": False,
                "deck_size": 0,
                "relics_size": 0,
            }
        rs = getattr(runner, "run_state", None)
        return {
            "floor": int(getattr(rs, "floor", 0) or 0) if rs else 0,
            "act": int(getattr(rs, "act", 1) or 1) if rs else 1,
            "hp": int(getattr(rs, "current_hp", 0) or 0) if rs else 0,
            "max_hp": int(getattr(rs, "max_hp", 0) or 0) if rs else 0,
            "gold": int(getattr(rs, "gold", 0) or 0) if rs else 0,
            "game_won": bool(getattr(runner, "game_won", False)),
            "deck_size": len(getattr(rs, "deck", []) or []) if rs else 0,
            "relics_size": len(getattr(rs, "relics", []) or []) if rs else 0,
        }

    # =====================================================================
    # 3. 动作
    # =====================================================================
    def get_available_actions(self) -> List[Any]:
        assert self._runner is not None
        return self._runner.get_available_actions()

    def take_action(self, action: Any) -> bool:
        assert self._runner is not None
        return self._runner.take_action(action)

    def get_available_action_labels(self, state: V8State) -> List[str]:
        """模型可见的 action 字符串描述（走 action_space，注入候选内容）。"""
        return get_available_actions(state, runner=self._runner)

    # =====================================================================
    # 4. 战斗
    # =====================================================================
    def run_combat_turn(
        self,
        *,
        solver_budgets: Dict[str, Tuple[float, int, int]],
    ) -> Dict[str, Any]:
        """战斗内走一个动作（从 env._run_combat_turn 搬，行为一致）。

        返回 {action_repr, turn, ok, fallback_used}：
            action_repr  : 实际执行动作的 _safe_action_repr（env 进 recent_actions）
            turn         : 战斗进行中读到的真实回合数（env 更新 _last_combat_turn）
            ok           : take_action 是否成功
            fallback_used: 是否触发越界 fallback
        无合法动作时返回 {"action_repr": "", "turn": 0, "ok": True, "no_actions": True}。
        """
        assert self._runner is not None
        assert self._adapter is not None

        actions = self._runner.get_available_actions()
        if not actions:
            return {"action_repr": "", "turn": 0, "ok": True, "no_actions": True,
                    "fallback_used": False}

        # 战斗进行中 current_combat 还在，读真实回合数供 env 缓存
        turn = 0
        try:
            cc = getattr(self._runner, "current_combat", None)
            if cc is not None:
                st = getattr(cc, "state", None)
                if st is not None:
                    turn = int(getattr(st, "turn", 0) or 0)
        except Exception:  # noqa: BLE001
            turn = 0

        room_type = self._runner.current_room_type or "monster"

        # 同步 multi_turn 的外层 deadline 与 cap_ms 一致（与 v8_bot 一致）
        rt_key = (room_type or "monster").lower() if isinstance(room_type, str) else "monster"
        budgets = solver_budgets.get(rt_key)
        cap_ms = budgets[2] if budgets else 3000.0
        try:
            self._adapter._multi_turn.time_budget_ms = float(cap_ms)
        except (AttributeError, Exception):  # noqa: BLE001
            pass

        action: Optional[Any] = None
        self.combat_search_calls += 1
        try:
            action = self._adapter.pick_action(actions, self._runner, room_type=room_type)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "StSRLBackend.run_combat_turn: adapter.pick_action raised %s: %s",
                type(e).__name__, e,
            )

        if action is None:
            action = next(
                (
                    a for a in actions
                    if isinstance(a, CombatAction) and a.action_type == "end_turn"
                ),
                actions[0],
            )

        action_repr = _safe_action_repr(action)
        ok = self._runner.take_action(action)
        fallback_used = False
        fallback_repr = ""
        if not ok:
            fallback_used = True
            fallback_repr = _safe_action_repr(actions[0])
            self._runner.take_action(actions[0])

        return {
            "action_repr": action_repr,
            "turn": turn,
            "ok": ok,
            "fallback_used": fallback_used,
            "fallback_repr": fallback_repr,
            "no_actions": False,
        }


__all__ = ["StSRLBackend", "_build_state_from_runner", "_safe_action_repr"]

"""RealGameBackend：把真实游戏（CommunicationMod）包成 GameBackend。

第二个 GameBackend 实现（stage1 的 StSRLBackend 之后）。让统一评估 harness
（v8/eval_harness.py）能在真机上跑同一套评估语义。

------------------------------------------------------------------------------
复用而非重写
------------------------------------------------------------------------------
- 状态/动作转换：复用 v8/mod_state_adapter.py（mod_json_to_v8_state /
  mod_json_to_available_actions / v8_idx_to_mod_command），与 tools/v8_play_real.py 同源。
- stdin/stdout 协议：复用 v8_play_real 的 ready/start/choose/play/end 命令 +
  逐行 JSON 读取。把 v8_play_real 主循环的 "读 state → 转 → 决策 → 发命令" 重构进
  本 backend 的 reset / take_action 形态，**不另写一套游戏逻辑**。

------------------------------------------------------------------------------
与 sim 后端的差异（如实保留，日后再说）
------------------------------------------------------------------------------
- **战斗**：真机里战斗内 model 直接出招（argmax），没有 StSRLSolver 的搜索。
  这是真机现状（v8_play_real 就是这么干的）。因此 run_combat_turn 在本 backend
  里**不参与**——COMBAT phase 被 adapter 当成普通决策 phase 枚举出 play_card / end_turn
  动作，由 harness 的 model argmax 选。日后战斗蒸馏再补搜索。
- 很多 pass-through 引擎对象属性（current_combat / current_event_state /
  event_handler / current_rewards / current_shop / neow_blessings /
  last_combat_card_log）真机没有等价引擎对象 → 返回 None。这些是 env 诊断日志
  / 战斗搜索专用；评估 harness 走的是 build_v8_state / get_available_action_labels /
  take_action / build_runner_snapshot 这条中性链路，不依赖它们。

------------------------------------------------------------------------------
IO 注入（便于离线测试）
------------------------------------------------------------------------------
默认走真实 stdin/stdout（连 mod）。构造时可注入 `io` 对象（含 send(str) /
recv()->Optional[dict]），离线 mock 单测用 fake io 喂预设 mod JSON 序列，
不需要真机。见 tools/test_real_game_backend_offline.py。
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any, Dict, List, Optional, Tuple

from v8.state import V8State
from v8.mod_state_adapter import (
    mod_json_to_available_actions,
    mod_json_to_v8_state,
    v8_idx_to_mod_command,
)

logger = logging.getLogger(__name__)


# =============================================================================
# IO 抽象：默认连真实 stdin/stdout；可注入 fake 做离线测试
# =============================================================================


class StdioModIO:
    """真机 IO：stdout 发命令（一行），stdin 收 game-state JSON（一行）。

    与 v8_play_real._send / _read_state 行为一致。stdout 是 mod 通信通道，
    日志必须走 stderr（本类不打印业务日志）。
    """

    def send(self, cmd: str) -> None:
        print(cmd, flush=True)

    def recv(self) -> Optional[Dict[str, Any]]:
        line = sys.stdin.readline()
        if not line:
            return None
        line = line.strip()
        if not line:
            return None
        try:
            return json.loads(line)
        except json.JSONDecodeError as e:  # noqa: BLE001
            logger.warning("[real] JSON decode error: %s; raw=%s", e, line[:200])
            return None


class ListModIO:
    """离线测试 IO：从预设 mod JSON 序列里逐条吐 recv，send 记录到 sent。

    每次 send 后，下一个 recv 返回 states 列表里的下一条；states 耗尽返回 None
    （模拟 stdin EOF）。
    """

    def __init__(self, states: List[Optional[Dict[str, Any]]]):
        self._states = list(states)
        self._idx = 0
        self.sent: List[str] = []

    def send(self, cmd: str) -> None:
        self.sent.append(cmd)

    def recv(self) -> Optional[Dict[str, Any]]:
        if self._idx >= len(self._states):
            return None
        st = self._states[self._idx]
        self._idx += 1
        return st


# =============================================================================
# RealGameBackend
# =============================================================================

# act boss 楼层（与 v8_play_real._BOSS_FLOORS 一致，IRONCLAD/SILENT/DEFECT/WATCHER 通用）
_BOSS_FLOORS = {1: 17, 2: 34, 3: 51, 4: 57}

# 同 screen + 同 cmd 连续 N 次 → 当死循环放弃（与 v8_play_real SAME_SCREEN_STALL_LIMIT 一致）
_SAME_SCREEN_STALL_LIMIT = 8


class RealGameBackend:
    """CommunicationMod 真机 GameBackend 实现。

    Args:
        character: 角色（mod start 命令用，大写如 IRONCLAD）。
        ascension: 进阶等级。
        seed_str: 可选 seed 字符串（mod start 命令第三参；空则随机）。
        io: IO 后端（默认 StdioModIO 连真机；测试注入 ListModIO）。
        max_steps: 单局安全上限（防死循环），超过即标 game_over（abort）。
    """

    def __init__(
        self,
        *,
        character: str = "IRONCLAD",
        ascension: int = 0,
        seed_str: str = "",
        io: Optional[Any] = None,
        max_steps: int = 2000,
    ):
        self.character = character
        self.ascension = ascension
        self.seed_str = seed_str
        self._io = io or StdioModIO()
        self.max_steps = max_steps

        # 与 GameBackend 协议对齐：真机不走搜索，恒 0。
        self.combat_search_calls: int = 0

        # 最新 mod 状态（每次 recv 更新）
        self._mod_state: Optional[Dict[str, Any]] = None
        # 当前 phase 的 (action_strings, metas)（build/get_available 缓存，take_action 用）
        self._cur_actions: List[str] = []
        self._cur_metas: List[Dict[str, Any]] = []

        self._game_over: bool = False
        self._game_won: bool = False
        self._step_count: int = 0
        # 进度追踪（snapshot / 指标用；与 v8_play_real metrics 同口径）
        self._max_floor_reached: int = 0
        self._handshaked: bool = False

        # stall 检测
        self._stall_screen: str = ""
        self._stall_cmd: str = ""
        self._stall_count: int = 0

    # ---- 内部：读到一帧"可决策"的 in-game 状态（跳过过渡帧 / 处理 GAME_OVER）----
    def _screen_type(self) -> str:
        gs = (self._mod_state or {}).get("game_state", {}) or {}
        return str(gs.get("screen_type", "") or "")

    def _floor_act_hp(self) -> Tuple[int, int, int, int]:
        gs = (self._mod_state or {}).get("game_state", {}) or {}
        floor = int(gs.get("floor", 0) or 0)
        act = int(gs.get("act", 1) or 1)
        hp = int(gs.get("current_hp", 0) or 0)
        max_hp = int(gs.get("max_hp", 0) or 0)
        return floor, act, hp, max_hp

    def _track_progress(self) -> None:
        floor, _, _, _ = self._floor_act_hp()
        if floor > self._max_floor_reached:
            self._max_floor_reached = floor

    def _check_game_over(self) -> bool:
        """当前 mod_state 是不是 GAME_OVER；是则设置 game_over/game_won 并返回 True。"""
        if self._screen_type() == "GAME_OVER":
            gs = (self._mod_state or {}).get("game_state", {}) or {}
            ss = gs.get("screen_state", {}) or {}
            self._game_over = True
            self._game_won = bool(ss.get("victory", False))
            return True
        return False

    def _advance_to_decision(self) -> None:
        """从当前已有 mod_state 出发，把过渡帧 / 无动作帧推到下一个可决策帧。

        处理：
            - in_game=False 过渡帧：等下一帧（best-effort，最多推几帧）。
            - GAME_OVER：设 game_over。
            - 无可选 action 的 screen：按 available_commands 兜底发 proceed/confirm/cancel，
              并做 stall 检测（与 v8_play_real 一致）。
        推进到 "有可选 action 的 in-game 决策帧" 或 game_over 即返回。
        """
        guard = 0
        guard_cap = self.max_steps * 4
        while guard < guard_cap:
            guard += 1
            st = self._mod_state
            if st is None:
                # stdin EOF / 序列耗尽 → 当作局结束（abort）
                self._game_over = True
                self._game_won = False
                return

            if not st.get("in_game"):
                # 过渡帧：再读一帧
                self._mod_state = self._io.recv()
                continue

            if self._check_game_over():
                return

            self._track_progress()

            # 枚举当前 phase 的动作
            try:
                actions, metas = mod_json_to_available_actions(st)
            except Exception as e:  # noqa: BLE001
                logger.warning("[real] adapter error %s: %s, sending proceed", type(e).__name__, e)
                self._send_and_recv("proceed")
                continue

            if actions:
                self._cur_actions = actions
                self._cur_metas = metas
                return

            # 无 action：按 available_commands 兜底
            cmds = st.get("available_commands", []) or []
            fallback = (
                "proceed" if "proceed" in cmds else
                "confirm" if "confirm" in cmds else
                "cancel" if "cancel" in cmds else "proceed"
            )
            screen = self._screen_type()
            if screen == self._stall_screen and fallback == self._stall_cmd:
                self._stall_count += 1
            else:
                self._stall_screen, self._stall_cmd, self._stall_count = screen, fallback, 1
            if self._stall_count >= _SAME_SCREEN_STALL_LIMIT:
                logger.warning("[real] STALL screen=%s cmd=%s x%d, abort game",
                               screen, fallback, self._stall_count)
                self._game_over = True
                self._game_won = False
                return
            self._send_and_recv(fallback)

    def _send_and_recv(self, cmd: str) -> None:
        """发一条命令，读下一帧 mod state（更新 _mod_state）。"""
        self._io.send(cmd)
        self._mod_state = self._io.recv()

    # =====================================================================
    # 1. 生命周期
    # =====================================================================
    def reset(self, seed: int) -> None:
        """开新局：mod 握手 + start 命令 + 推进到第一个决策帧。

        seed: harness 传的数值 seed。优先用构造时的 seed_str（用户显式指定）；
        否则不附 seed 让 mod 随机（真机无法可靠注入任意 int seed 为 STS seed token，
        故 evaluate 在真机上更像 "跑 N 局" 而非 "N 个可复现种子"——如实记录）。
        """
        self._game_over = False
        self._game_won = False
        self._step_count = 0
        self._max_floor_reached = 0
        self._cur_actions = []
        self._cur_metas = []
        self._stall_screen = self._stall_cmd = ""
        self._stall_count = 0

        # 握手：首次 reset 发 ready（mod 启动后只需一次；多局时后续 reset 跳过）
        if not self._handshaked:
            self._io.send("ready")
            self._handshaked = True

        # 读一帧，确认 out-of-game（主菜单），再发 start
        self._mod_state = self._io.recv()
        # 若拿到的是 in-game 帧（罕见，已在局中），直接进决策
        if self._mod_state is not None and self._mod_state.get("in_game"):
            self._advance_to_decision()
            return

        seed_part = f" {self.seed_str}" if self.seed_str else ""
        start_cmd = f"start {self.character} {self.ascension}{seed_part}".strip()
        self._send_and_recv(start_cmd)
        self._advance_to_decision()

    @property
    def game_over(self) -> bool:
        return self._game_over

    @property
    def game_won(self) -> bool:
        return self._game_won

    @property
    def phase(self) -> Any:
        """真机 phase：返回 adapter 推断的 V8 phase 名（str），非引擎枚举。"""
        if self._mod_state is None:
            return "RUN_COMPLETE" if self._game_over else None
        from v8.mod_state_adapter import _infer_phase  # 局部 import 避免循环
        return _infer_phase(self._mod_state)

    def force_terminate(self) -> None:
        self._game_over = True
        self._game_won = False

    def close(self) -> None:
        self._mod_state = None

    # =====================================================================
    # 2. 状态
    # =====================================================================
    def build_v8_state(self) -> V8State:
        if self._mod_state is None:
            # 兜底空 state（game_over 后可能被读）
            return V8State(
                hp=0, max_hp=0, floor=0, act=1, gold=0,
                potions=[], deck=[], relics=[], map_nodes=[],
                current_position=None, deck_strength=None,
                in_combat=False, phase="", boss="",
            )
        return mod_json_to_v8_state(self._mod_state)

    @property
    def run_state(self) -> Any:
        """真机无引擎 run_state；返回 mod game_state dict（诊断用，非引擎对象）。"""
        return (self._mod_state or {}).get("game_state", {}) or {}

    @property
    def current_room_type(self) -> Any:
        return None

    def get_current_room_type(self) -> Any:
        return None

    # 以下 pass-through 引擎对象真机无等价物，返回 None（评估链路不依赖）
    @property
    def current_combat(self) -> Any:
        return None

    @property
    def current_event_state(self) -> Any:
        return None

    @property
    def event_handler(self) -> Any:
        return None

    @property
    def current_rewards(self) -> Any:
        return None

    @property
    def current_shop(self) -> Any:
        return None

    @property
    def neow_blessings(self) -> Any:
        return None

    @property
    def last_combat_card_log(self) -> Any:
        return None

    @property
    def boss_name(self) -> str:
        gs = (self._mod_state or {}).get("game_state", {}) or {}
        return str(gs.get("boss", "") or "")

    @property
    def seed_string(self) -> str:
        return self.seed_str

    @property
    def seed_int(self) -> Optional[int]:
        return None

    def build_runner_snapshot(self) -> Dict[str, Any]:
        """floor/act/hp/.../game_won 纯 dict（与 StSRLBackend 同 schema，跨进程安全）。"""
        gs = (self._mod_state or {}).get("game_state", {}) or {}
        floor = int(gs.get("floor", 0) or 0)
        return {
            "floor": floor,
            "act": int(gs.get("act", 1) or 1),
            "hp": int(gs.get("current_hp", 0) or 0),
            "max_hp": int(gs.get("max_hp", 0) or 0),
            "gold": int(gs.get("gold", 0) or 0),
            "game_won": bool(self._game_won),
            "deck_size": len(gs.get("deck", []) or []),
            "relics_size": len(gs.get("relics", []) or []),
            # 真机额外：最大到达楼层（指标归因用，与 v8_play_real 一致）
            "max_floor_reached": self._max_floor_reached,
        }

    # =====================================================================
    # 3. 动作
    # =====================================================================
    def get_available_actions(self) -> List[Any]:
        """返回当前决策帧的 action 字符串 list（idx 与 take_action 对齐）。

        真机没有引擎 GameAction 对象，直接返回 action 字符串（中性）。
        """
        return list(self._cur_actions)

    def take_action(self, action: Any) -> bool:
        """执行一个动作。

        action 可以是 idx（int，harness 走这条）或 action 字符串（容错）。
        把 idx → mod 命令 → 发送 → 读下一帧 → 推进到下一个决策帧。
        """
        if self._game_over:
            return False
        if isinstance(action, int):
            idx = action
        else:
            try:
                idx = self._cur_actions.index(str(action))
            except ValueError:
                idx = 0
        if idx < 0 or idx >= len(self._cur_metas):
            idx = 0

        cmd = v8_idx_to_mod_command(idx, self._cur_metas, self._mod_state or {})

        # stall 检测：同 screen + 同 cmd 连续 N 次 → abort
        screen = self._screen_type()
        if screen == self._stall_screen and cmd == self._stall_cmd:
            self._stall_count += 1
        else:
            self._stall_screen, self._stall_cmd, self._stall_count = screen, cmd, 1
        if self._stall_count >= _SAME_SCREEN_STALL_LIMIT:
            logger.warning("[real] STALL screen=%s cmd=%s x%d, abort game",
                           screen, cmd, self._stall_count)
            self._game_over = True
            self._game_won = False
            return True

        self._step_count += 1
        if self._step_count > self.max_steps:
            logger.warning("[real] max_steps=%d hit, abort game", self.max_steps)
            self._game_over = True
            self._game_won = False
            return True

        self._send_and_recv(cmd)
        # 推进过渡帧 / 兜底无动作帧，直到下一个可决策帧或 game_over
        self._advance_to_decision()
        return True

    def get_available_action_labels(self, state: V8State) -> List[str]:  # noqa: ARG002
        """模型可见的 action 字符串描述（= get_available_actions，真机已是字符串）。"""
        return list(self._cur_actions)

    # =====================================================================
    # 4. 战斗（真机无搜索：model 直接出招，COMBAT 当普通决策 phase 处理）
    # =====================================================================
    def run_combat_turn(
        self,
        *,
        solver_budgets: Dict[str, Tuple[float, int, int]],  # noqa: ARG002
    ) -> Dict[str, Any]:
        """真机不走搜索式 combat turn。

        本 backend 的 COMBAT 由 adapter 枚举成普通决策动作（play_card / end_turn），
        harness model argmax 选 → take_action。故此方法不应被 harness 调用；
        保留以满足 GameBackend 协议，调用即 no-op（返回无动作）。
        """
        return {"action_repr": "", "turn": 0, "ok": True, "no_actions": True,
                "fallback_used": False}


__all__ = ["RealGameBackend", "StdioModIO", "ListModIO"]

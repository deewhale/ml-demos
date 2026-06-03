"""GameBackend 接口（V8 RL 引擎中间层）。

目的：把 env.py 与具体引擎（当前 StSRLSolver Python 引擎）解耦，让日后能插入
真实游戏（CommunicationMod，参考 v8/mod_state_adapter.py）/ Rust 引擎 / 其它后端，
而不改 env.py 的决策 / reward / 日志逻辑。

env 只跟本接口说话；不再直接 import / 触碰 GameRunner / TurnSolverAdapter。

------------------------------------------------------------------------------
接口四类（按 env 对引擎的依赖归类）
------------------------------------------------------------------------------

1. 生命周期 (lifecycle)
   - reset(seed)              : 开新局，推进到第一个元决策点前的引擎初始化
   - game_over / game_won     : 终态标志
   - phase                    : 当前引擎 phase（**pass-through 引擎 GamePhase 枚举**，
                                env 用 _ENGINE_TO_V8_PHASE / _META_PHASES 映射）
   - force_terminate()        : 强制把当前 run 标 terminal（loss）
   - close()                  : 释放后端资源

2. 状态 (state)
   - build_v8_state()         : 从引擎当前状态构造 V8State（中性化：返回 V8State，
                                不暴露引擎对象）
   - run_state                : **pass-through** 引擎 run_state（env 诊断日志 +
                                action_space 大量直接读 ~20 字段；stage2 中性化债）
   - current_room_type        : **pass-through** 引擎 current_room_type（combat 内才设）
   - get_current_room_type()  : 引擎地图 RoomType enum getter（可能不存在 → None）
   - current_combat           : **pass-through** 战斗对象（.state.enemies/.hand/.turn）
   - current_event_state      : **pass-through** 事件状态对象
   - event_handler            : **pass-through** 事件 handler（get_available_choices）
   - current_rewards          : **pass-through** 奖励对象（card/relic/boss_relics）
   - current_shop             : **pass-through** 商店对象
   - neow_blessings           : **pass-through** Neow 祝福 list
   - last_combat_card_log     : **pass-through** 上场战斗逐动作细账（CombatLogEntry list）
   - boss_name                : 当前 act boss 名（中性化：返回 str）
   - seed_string / seed_int   : 复现用 seed token（中性化：str / Optional[int]）
   - build_runner_snapshot()  : floor/act/hp/.../game_won 的纯 dict（中性化，跨进程回传）

3. 动作 (action)
   - get_available_actions()  : List[engine GameAction]（**pass-through 引擎动作对象**，
                                env 把 idx → 对象 → take_action；action_space 也直接
                                format 这些对象。stage2 若要中性化需定义中性 action 表示）
   - take_action(action)      : bool，执行一个引擎动作
   - get_available_action_labels(state): List[str] 模型可见的 action 字符串描述
                                （走 action_space.get_available_actions，注入候选内容）

4. 战斗 (combat)
   - run_combat_turn(...)     : 战斗内走一个动作（封装 TurnSolverAdapter.pick_action +
                                fallback + take_action）。返回本回合相关诊断信息。
   - combat_search_calls      : 累计 adapter.pick_action 调用次数（env perf 计数读取）

------------------------------------------------------------------------------
pass-through 债（stage2 中性化目标）
------------------------------------------------------------------------------
以下属性当前直接返回引擎对象（duck-typed），env 的诊断日志 / action_space 直接
introspection 其内部字段。这是接缝，stage2 接真机 / Rust 时要把它们换成中性数据结构：
    phase, run_state, current_room_type, get_current_room_type, current_combat,
    current_event_state, event_handler, current_rewards, current_shop,
    neow_blessings, last_combat_card_log, get_available_actions(返回引擎动作对象)

中性化的部分（已不暴露引擎对象）：
    reset / game_over / game_won / force_terminate / close / build_v8_state /
    boss_name / seed_string / seed_int / build_runner_snapshot / take_action /
    get_available_action_labels / run_combat_turn

本接口用 typing.Protocol（鸭子类型，不强制继承），让 StSRLBackend 及未来后端
只要实现这些成员即可，不引入运行时基类耦合。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

from v8.state import V8State


@runtime_checkable
class GameBackend(Protocol):
    """V8 引擎后端协议。env 通过它驱动一局游戏。"""

    # ---- perf 计数（env 读取累加）----
    combat_search_calls: int

    # =====================================================================
    # 1. 生命周期
    # =====================================================================
    def reset(self, seed: int) -> None:
        """开新局：初始化引擎到第一个元决策点前的状态（不推进 advance loop，
        advance 由 env 控制）。"""

    @property
    def game_over(self) -> bool:
        """当前 run 是否结束。"""

    @property
    def game_won(self) -> bool:
        """当前 run 是否通关。"""

    @property
    def phase(self) -> Any:
        """当前引擎 phase（pass-through 引擎 GamePhase 枚举）。"""

    def force_terminate(self) -> None:
        """强制把 run 标 terminal（loss）：game_over=True / game_won=False /
        phase=RUN_COMPLETE。"""

    def close(self) -> None:
        """释放后端资源。"""

    # =====================================================================
    # 2. 状态
    # =====================================================================
    def build_v8_state(self) -> V8State:
        """从引擎当前状态构造 V8State（元决策 phase 用）。"""

    @property
    def run_state(self) -> Any:
        """pass-through 引擎 run_state（~20 字段）。"""

    @property
    def current_room_type(self) -> Any:
        """pass-through 引擎 current_room_type。"""

    def get_current_room_type(self) -> Any:
        """引擎地图 RoomType enum（覆盖所有节点类型）；不可用时返回 None。"""

    @property
    def current_combat(self) -> Any:
        """pass-through 战斗对象（.state.enemies / .hand / .turn）。"""

    @property
    def current_event_state(self) -> Any:
        """pass-through 当前事件状态对象。"""

    @property
    def event_handler(self) -> Any:
        """pass-through 事件 handler（.get_available_choices）。"""

    @property
    def current_rewards(self) -> Any:
        """pass-through 奖励对象。"""

    @property
    def current_shop(self) -> Any:
        """pass-through 商店对象。"""

    @property
    def neow_blessings(self) -> Any:
        """pass-through Neow 祝福 list。"""

    @property
    def last_combat_card_log(self) -> Any:
        """pass-through 上场战斗逐动作细账（CombatLogEntry list）。"""

    @property
    def boss_name(self) -> str:
        """当前 act boss 名（中性化 str）。"""

    @property
    def seed_string(self) -> str:
        """引擎内部 seed token（复现用）。"""

    @property
    def seed_int(self) -> Optional[int]:
        """引擎内部数值 seed（复现用）。"""

    def build_runner_snapshot(self) -> Dict[str, Any]:
        """floor/act/hp/max_hp/gold/game_won/deck_size/relics_size 纯 dict
        （跨进程回传 trainer 用，全部基础类型）。"""

    # =====================================================================
    # 3. 动作
    # =====================================================================
    def get_available_actions(self) -> List[Any]:
        """当前合法引擎动作对象 list（pass-through；idx 与 take_action 对齐）。"""

    def take_action(self, action: Any) -> bool:
        """执行一个引擎动作，返回是否成功。"""

    def get_available_action_labels(self, state: V8State) -> List[str]:
        """模型可见的 action 字符串描述（与 get_available_actions idx 对齐）。"""

    # =====================================================================
    # 4. 战斗
    # =====================================================================
    def run_combat_turn(
        self,
        *,
        solver_budgets: Dict[str, Tuple[float, int, int]],
    ) -> Dict[str, Any]:
        """战斗内走一个动作（adapter.pick_action + fallback + take_action）。

        返回诊断 dict：{action_repr, turn, ok, fallback_used}。
        env 负责把 action_repr 进 recent_actions、把 turn 更新到 _last_combat_turn。
        """

"""V8 RL 环境（gym-like）。

按 docs/v8_implementation_design.md 组件 8（Episode Loop）设计：
- Episode 粒度 = 一局 STS（默认 act1）。
- Step 粒度 = 一个**元决策**（NEOW / MAP / EVENT / SHOP / REST / TREASURE /
  CARD_REWARDS / BOSS_REWARDS）。
- 战斗内由 env 内部跑 StSRLSolver TurnSolver 完成（**不暴露给 RL trajectory**）。
- 战斗结束触发 evaluate_deck（post-battle check），驱动 step reward。

设计原则对照（docs/v8_design_principles.md）：
- 战斗内是搜索主导，model 暂不参与（用户原话第 2 点：搜索 + model 联合，
  本 env 先把搜索部分接通；model prior 推理留给 trainer 接入预训练好的战斗 head）
- 元决策才写入 RL trajectory（PPO 学元决策）
- 不复活 v8_strategy 启发式（元决策 action 由外部 model 选 idx，env 只负责执行）
- evaluate_deck 在战斗结束后触发（用户原话"战斗之后"做 check）

实现状态：
- reset / step 完整接 GameRunner（不 stub）
- 战斗内：runner.take_action(CombatAction) by TurnSolver（与 v8_bot 同套搜索预算）
- post-battle 自动触发 evaluate_deck
- node_reward detect：?事件成功 / 商店买 relic / 休息使用 / 宝箱开 relic 简单识别

不在本文件内：
- PPO trainer（下一步）
- 预训练 script（下一步）
- model prior 接入（trainer 阶段把 model 喂给 TurnSolver 做剪枝；目前 env 用纯搜索）
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

# 保证能 import StSRLSolver
from sts_paths import ensure_on_sys_path

ensure_on_sys_path()

# StSRLSolver 引擎
from packages.engine.game import (  # noqa: E402
    GameRunner,
    GamePhase,
    PathAction,
    NeowAction,
    CombatAction,
    RewardAction,
    EventAction,
    ShopAction,
    RestAction,
    TreasureAction,
    BossRewardAction,
)
from packages.training.turn_solver import TurnSolverAdapter  # noqa: E402

# V8 内部模块
from v8.state import V8State
from v8.action_space import get_available_actions
from v8.deck_evaluator import evaluate_deck, clear_cache as clear_deck_cache
from v8.reward import (
    compute_step_reward,
    compute_final_reward,
    NODE_REWARD_EVENT_SUCCESS,
    NODE_REWARD_SHOP_RELIC,
    NODE_REWARD_REST_USE,
    NODE_REWARD_TREASURE,
)


logger = logging.getLogger(__name__)


# StSRLSolver GamePhase → V8 phase 名 映射
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


# 元决策 phase（写入 RL trajectory）
_META_PHASES = {
    GamePhase.NEOW,
    GamePhase.MAP_NAVIGATION,
    GamePhase.COMBAT_REWARDS,
    GamePhase.EVENT,
    GamePhase.SHOP,
    GamePhase.REST,
    GamePhase.TREASURE,
    GamePhase.BOSS_REWARDS,
}


# Solver 预算（与 v8_bot.SOLVER_BUDGETS 对齐，让战斗内搜索行为一致）
SOLVER_BUDGETS = {
    "monster": (50.0, 5_000, 3_000),
    "elite": (500.0, 20_000, 12_000),
    "boss": (2_000.0, 50_000, 25_000),
}


# 一局最多 step 上限（防卡死）
DEFAULT_MAX_STEPS_PER_EPISODE: int = 5_000


# =============================================================================
# 辅助：runner → V8State
# =============================================================================


def _build_state_from_runner(runner: GameRunner) -> V8State:
    """从 GameRunner 提取 V8State（元决策 phase 用，不含战斗内字段）。

    战斗内字段（hand / draw / discard / monsters / energy）只在 phase=COMBAT 时
    填充；元决策 phase 留空（避免假数据）。本 env 设计是战斗内由 env 内部处理，
    模型只在元决策 phase 看 state，所以 COMBAT phase 一般不会被 caller 看到。
    """
    rs = runner.run_state
    phase_name = _ENGINE_TO_V8_PHASE.get(runner.phase, runner.phase.name)

    # ----- 数字状态 -----
    hp = int(getattr(rs, "current_hp", 0) or 0)
    max_hp = int(getattr(rs, "max_hp", 0) or 0)
    floor = int(getattr(rs, "floor", 0) or 0)
    act = int(getattr(rs, "act", 1) or 1)
    gold = int(getattr(rs, "gold", 0) or 0)

    # ----- 药水（含空槽用 "" 占位）-----
    potions: List[str] = []
    for slot in getattr(rs, "potion_slots", []) or []:
        pid = getattr(slot, "potion_id", None)
        potions.append(pid if pid else "")

    # ----- 牌组 {"name": id, "upgraded": bool} -----
    deck: List[Dict[str, Any]] = []
    for c in getattr(rs, "deck", []) or []:
        deck.append({
            "name": getattr(c, "id", "") or "",
            "upgraded": bool(getattr(c, "upgraded", False)),
        })

    # ----- 遗物 -----
    relics: List[str] = []
    for r in getattr(rs, "relics", []) or []:
        rid = getattr(r, "id", None) or str(r)
        relics.append(rid)

    # ----- 完整 act 地图 -----
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

    # ----- 当前位置 -----
    current_position: Optional[Dict[str, int]] = None
    mp = getattr(rs, "map_position", None)
    if mp is not None and not (mp.x == -1 and mp.y == -1):
        current_position = {"floor": int(mp.y), "x": int(mp.x)}

    state = V8State(
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
        deck_strength=None,  # 由 env 在 reset/step 中按时机填
        in_combat=(runner.phase == GamePhase.COMBAT),
        phase=phase_name,
    )
    return state


# =============================================================================
# 节点 reward 简单 detect
# =============================================================================


def _detect_node_reward(
    runner: GameRunner,
    prev_phase: Optional[Any],
    prev_relics_count: int,
    prev_max_hp: int,
    next_phase: Optional[Any],
) -> float:
    """根据 phase 转换 + state 变化粗略判定 node_reward。

    简单规则（保守，宁少给不错给）：
    - 离开 EVENT 且 max_hp 变高 / relic 数变多 → +NODE_REWARD_EVENT_SUCCESS
    - 离开 SHOP 且 relic 数变多 → +NODE_REWARD_SHOP_RELIC
    - 离开 REST → +NODE_REWARD_REST_USE（rest / smith 都给小奖励）
    - 离开 TREASURE 且 relic 数变多 → +NODE_REWARD_TREASURE
    - 其他（COMBAT / CARD_REWARDS / MAP / NEOW / BOSS_REWARDS）→ 0
      （战斗 / 选卡的收益已经在 Δdeck_strength 里反映）
    """
    rs = runner.run_state
    new_relics_count = len(getattr(rs, "relics", []) or [])
    new_max_hp = int(getattr(rs, "max_hp", 0) or 0)

    # 没切 phase（同一 phase 内连续 step） → 不给 node reward
    if prev_phase is None or prev_phase == next_phase:
        return 0.0

    if prev_phase == GamePhase.EVENT:
        if new_max_hp > prev_max_hp or new_relics_count > prev_relics_count:
            return NODE_REWARD_EVENT_SUCCESS
        return 0.0

    if prev_phase == GamePhase.SHOP:
        if new_relics_count > prev_relics_count:
            return NODE_REWARD_SHOP_RELIC
        return 0.0

    if prev_phase == GamePhase.REST:
        return NODE_REWARD_REST_USE

    if prev_phase == GamePhase.TREASURE:
        if new_relics_count > prev_relics_count:
            return NODE_REWARD_TREASURE
        return 0.0

    return 0.0


# =============================================================================
# V8Env
# =============================================================================


class V8Env:
    """Gym-like RL 环境。

    用法：
        env = V8Env(character="ironclad", ascension=0)
        state = env.reset(seed=42)
        while True:
            actions = env.get_available_actions()
            idx = model.choose(state, actions)
            state, reward, done, info = env.step(idx)
            if done:
                break
        env.close()

    设计：
    - reset() 创建 GameRunner，推进到第一个元决策 phase（一般是 NEOW）
    - step(idx) 把 model 选的 idx 转成 GameAction，take_action，然后内部 loop
      推进直到下一个元决策 phase（或游戏结束）。战斗中由 TurnSolver 处理。
    - reward 按 v8/reward.py 的 compute_step_reward 计算（依赖 evaluate_deck 触发）
    """

    def __init__(
        self,
        character: str = "ironclad",
        ascension: int = 0,
        verbose: bool = False,
        max_steps_per_episode: int = DEFAULT_MAX_STEPS_PER_EPISODE,
        solver_budgets: Optional[Dict[str, Tuple[float, int, int]]] = None,
        combat_net_wrapper: Optional[Any] = None,
    ):
        """V8Env 构造器。

        新参数（Round 2）：
            combat_net_wrapper: 可选 V8CombatNetWrapper（让 model 在战斗 search
                里参与 leaf 评估）。None 时 fallback 到原纯搜索（hand-rolled heuristic）。
                设计原则：用户原话第 2 点"搜索+模型联合"，wrapper 是 hook。
        """
        self.character = character
        self.ascension = ascension
        self.verbose = verbose
        self.max_steps_per_episode = max_steps_per_episode
        self._solver_budgets = solver_budgets or SOLVER_BUDGETS
        self._combat_net_wrapper = combat_net_wrapper

        # 每局重置的运行时状态
        self._runner: Optional[GameRunner] = None
        self._adapter: Optional[TurnSolverAdapter] = None
        self._current_state: Optional[V8State] = None

        # reward 计算需要的历史
        self._prev_state: Optional[V8State] = None
        self._prev_deck_strength: Optional[Dict[str, float]] = None

        # 节点 reward detect 需要的快照（step 开始时记录）
        self._step_start_phase: Optional[Any] = None
        self._step_start_relics_count: int = 0
        self._step_start_max_hp: int = 0

        # 统计 / debug
        self._step_count: int = 0
        self._actions_taken: int = 0
        self._battles_finished: int = 0
        self._last_seed: Optional[int] = None
        self._last_act_for_cache: int = 1

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def reset(self, seed: int) -> V8State:
        """开始新局，推进到第一个元决策点（NEOW）。

        清空 deck_evaluator cache（act 切换 / 新局，旧 cache 失效）。
        触发首次 evaluate_deck（开局 deck 已知）。
        """
        # 清 cache（act 1 标准敌人，每新局重新评估）
        clear_deck_cache()

        self._runner = GameRunner(
            seed=seed,
            ascension=self.ascension,
            character=self.character,
            skip_neow=False,  # V8 让 model 决策 Neow
            verbose=self.verbose,
        )
        # Round 2：如果有 combat_net_wrapper，把它作为 search leaf evaluator
        # 接进 adapter（用户原话第 2 点：搜索 + 模型联合）
        adapter_kwargs: Dict[str, Any] = dict(
            time_budget_ms=50.0,
            node_budget=5_000,
            solver_budgets=self._solver_budgets,
        )
        if self._combat_net_wrapper is not None:
            adapter_kwargs["combat_net"] = self._combat_net_wrapper
        self._adapter = TurnSolverAdapter(**adapter_kwargs)
        self._adapter.reset()

        self._step_count = 0
        self._actions_taken = 0
        self._battles_finished = 0
        self._last_seed = seed
        self._last_act_for_cache = self._runner.run_state.act

        # 推进到第一个元决策 phase（NEOW 一般直接就是；保险起见 advance）
        self._advance_to_meta_decision()

        # 构造初始 V8State
        state = _build_state_from_runner(self._runner)

        # 开局 deck 已知 → 首次 evaluate_deck
        try:
            ds = evaluate_deck(
                deck=state.deck,
                relics=state.relics,
                hp=state.hp,
                max_hp=state.max_hp,
                act=state.act,
            )
            state.deck_strength = ds
            self._prev_deck_strength = ds
        except Exception as e:  # noqa: BLE001
            logger.warning("initial evaluate_deck failed: %s: %s", type(e).__name__, e)
            state.deck_strength = None
            self._prev_deck_strength = None

        self._current_state = state
        self._prev_state = state
        return state

    def step(
        self, action_idx: int
    ) -> Tuple[V8State, float, bool, Dict[str, Any]]:
        """执行 model 选的 action，推进游戏到下一个元决策点。

        返回：
            next_state: 下一个元决策点的 V8State（或终态）
            reward:     此 step 的 float reward（终态时含 final reward）
            done:       是否游戏结束
            info:       dict（debug：phase 变化 / 战斗发生 / 评估命中 等）
        """
        if self._runner is None or self._current_state is None:
            raise RuntimeError("V8Env: must call reset() before step()")

        info: Dict[str, Any] = {
            "step_count": self._step_count,
            "battle_happened": False,
            "phase_before": self._current_state.phase,
            "phase_after": "",
            "node_reward": 0.0,
            "deck_strength_evaluated": False,
            "error": None,
        }

        # 记录 step 开始快照（detect node_reward 用）
        rs = self._runner.run_state
        self._step_start_phase = self._runner.phase
        self._step_start_relics_count = len(getattr(rs, "relics", []) or [])
        self._step_start_max_hp = int(getattr(rs, "max_hp", 0) or 0)

        # 取当前可执行 actions（GameAction 对象 list）
        engine_actions = self._runner.get_available_actions()
        if not engine_actions:
            # 没合法 action：游戏卡死 / 结束
            done = True
            reward = compute_final_reward(
                game_won=bool(self._runner.game_won),
                final_floor=int(self._runner.run_state.floor),
            )
            info["phase_after"] = self._current_state.phase
            info["error"] = "no_available_actions"
            return self._current_state, reward, done, info

        # 把 idx 转成具体 action
        if action_idx < 0 or action_idx >= len(engine_actions):
            # idx 越界：fallback 到 0（与 v8_inference_bot 同样的 fallback 策略）
            logger.warning(
                "V8Env.step: action_idx=%d out of range [0,%d), fallback to 0",
                action_idx, len(engine_actions),
            )
            action_idx = 0
        chosen_action = engine_actions[action_idx]

        # 执行 action
        ok = self._runner.take_action(chosen_action)
        self._actions_taken += 1
        if not ok:
            logger.warning(
                "V8Env.step: take_action returned False at floor=%d phase=%s action=%s",
                rs.floor, self._runner.phase.name, chosen_action,
            )
            info["error"] = "take_action_failed"

        # 推进到下一个元决策 phase（中间所有 COMBAT 由 _advance 内部用 TurnSolver 处理）
        battle_happened = self._advance_to_meta_decision()
        info["battle_happened"] = battle_happened

        # 如果发生过战斗，post-battle 触发 evaluate_deck
        next_state = _build_state_from_runner(self._runner)

        # act 切换则清 cache（标准敌人换了）
        cur_act = next_state.act
        if cur_act != self._last_act_for_cache:
            clear_deck_cache()
            self._last_act_for_cache = cur_act

        # 触发 evaluate_deck 时机：
        #   1. 战斗刚结束（battle_happened=True）
        #   2. CARD_REWARDS phase 后 deck 可能变了（cache miss 自动重算）
        #   3. SHOP 后 deck 可能变了
        #   4. REST upgrade 后 deck 可能变了
        # 简化：每次 phase 切到元决策 phase（非 COMBAT）都调一次 evaluate_deck，
        # 内部 cache 命中重复 case，开销可控。
        next_deck_strength: Optional[Dict[str, float]] = self._prev_deck_strength
        if not self._runner.game_over:
            try:
                next_deck_strength = evaluate_deck(
                    deck=next_state.deck,
                    relics=next_state.relics,
                    hp=next_state.hp,
                    max_hp=next_state.max_hp,
                    act=next_state.act,
                )
                next_state.deck_strength = next_deck_strength
                info["deck_strength_evaluated"] = True
            except Exception as e:  # noqa: BLE001
                logger.warning("post-step evaluate_deck failed: %s: %s", type(e).__name__, e)
                next_state.deck_strength = self._prev_deck_strength

        # node_reward detect
        node_reward = _detect_node_reward(
            runner=self._runner,
            prev_phase=self._step_start_phase,
            prev_relics_count=self._step_start_relics_count,
            prev_max_hp=self._step_start_max_hp,
            next_phase=self._runner.phase,
        )
        info["node_reward"] = node_reward

        # 计算 step reward
        step_reward = compute_step_reward(
            prev_state=self._prev_state if self._prev_state is not None else next_state,
            next_state=next_state,
            prev_deck_strength=self._prev_deck_strength,
            next_deck_strength=next_deck_strength,
            node_reward=node_reward,
        )

        # done 判定
        done = bool(self._runner.game_over)
        if done:
            # 加上 final reward
            step_reward += compute_final_reward(
                game_won=bool(self._runner.game_won),
                final_floor=int(self._runner.run_state.floor),
            )

        # 步数 cap
        self._step_count += 1
        if not done and self._step_count >= self.max_steps_per_episode:
            logger.warning(
                "V8Env.step: hit max_steps_per_episode=%d, force done",
                self.max_steps_per_episode,
            )
            done = True
            info["error"] = "max_steps_exceeded"

        # 更新 prev 历史
        self._prev_state = next_state
        self._prev_deck_strength = next_deck_strength
        self._current_state = next_state

        info["phase_after"] = next_state.phase
        return next_state, float(step_reward), done, info

    def get_available_actions(self) -> List[str]:
        """当前 state 下的合法 action 字符串描述（pointer network 输入）。

        idx 与下次 step(idx) 严格对齐 runner.get_available_actions()。
        """
        if self._runner is None or self._current_state is None:
            return []
        return get_available_actions(self._current_state, runner=self._runner)

    def close(self) -> None:
        """清理资源：清 deck cache + 清 runner / adapter 引用。"""
        clear_deck_cache()
        self._runner = None
        self._adapter = None
        self._current_state = None
        self._prev_state = None
        self._prev_deck_strength = None

    @property
    def state(self) -> Optional[V8State]:
        """当前 V8State（外部只读访问，方便 debug）。"""
        return self._current_state

    @property
    def runner(self) -> Optional[GameRunner]:
        """底层 GameRunner（外部只读访问，方便 debug；trainer 不应直接改 runner 状态）。"""
        return self._runner

    # ---------------------------------------------------------------------
    # Internal: 内部 loop 推进到元决策 phase
    # ---------------------------------------------------------------------

    def _advance_to_meta_decision(self) -> bool:
        """推进 runner 直到下一个**元决策** phase（或 game_over）。

        中间所有 COMBAT phase 由 TurnSolver 一气呵成；不暴露给 RL trajectory。

        返回：本次推进过程中是否发生过 COMBAT（True 即战斗结束，触发 post-battle）。
        """
        assert self._runner is not None
        assert self._adapter is not None

        battle_happened = False
        guard = 0
        guard_cap = self.max_steps_per_episode * 4  # internal step 比 RL step 多

        while not self._runner.game_over and guard < guard_cap:
            guard += 1
            phase = self._runner.phase

            # 已到元决策 phase → 停下，让外部 model 选
            if phase in _META_PHASES:
                return battle_happened

            # 进入 COMBAT 内部 turn loop
            if phase == GamePhase.COMBAT:
                battle_happened = True
                self._run_combat_turn()
                # 战斗推完后回到 while 头继续判断 phase
                continue

            # RUN_COMPLETE：游戏结束
            if phase == GamePhase.RUN_COMPLETE:
                return battle_happened

            # 其他 phase（不应到这里）：取第一个合法 action 推进
            actions = self._runner.get_available_actions()
            if not actions:
                logger.warning(
                    "V8Env._advance: no actions available at phase=%s, abort",
                    phase.name,
                )
                return battle_happened
            self._runner.take_action(actions[0])
            self._actions_taken += 1

        if guard >= guard_cap:
            logger.warning(
                "V8Env._advance: hit guard_cap=%d, force return", guard_cap,
            )
        return battle_happened

    def _run_combat_turn(self) -> None:
        """战斗内：调 TurnSolver 选一个 CombatAction 执行。

        与 v8_bot._pick_action(COMBAT) 同套搜索预算。每次只走一个 action（因为
        runner.phase 在战斗里一直是 COMBAT，外层 while 会重复进来直到 COMBAT 结束）。

        简化版本（不接 SIGALRM 硬超时）：搜索内部已有 deadline，hard cap 由
        adapter 自身的 multi_turn deadline 控制。
        """
        assert self._runner is not None
        assert self._adapter is not None

        actions = self._runner.get_available_actions()
        if not actions:
            return

        room_type = self._runner.current_room_type or "monster"

        # 同步 multi_turn 的外层 deadline 与 cap_ms 一致（与 v8_bot 一致）
        rt_key = (room_type or "monster").lower() if isinstance(room_type, str) else "monster"
        budgets = self._solver_budgets.get(rt_key)
        cap_ms = budgets[2] if budgets else 3000.0
        try:
            self._adapter._multi_turn.time_budget_ms = float(cap_ms)
        except (AttributeError, Exception):  # noqa: BLE001
            pass

        # 直接调 adapter（不接 SIGALRM；smoke 阶段足够，hard timeout 在 RL trainer 阶段
        # 视情况再补，避免 env 本身带太多副作用）
        action: Optional[Any] = None
        try:
            action = self._adapter.pick_action(actions, self._runner, room_type=room_type)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "V8Env._run_combat_turn: adapter.pick_action raised %s: %s",
                type(e).__name__, e,
            )

        if action is None:
            # fallback：找 end_turn，否则第一个
            action = next(
                (
                    a for a in actions
                    if isinstance(a, CombatAction) and a.action_type == "end_turn"
                ),
                actions[0],
            )

        ok = self._runner.take_action(action)
        self._actions_taken += 1
        if not ok:
            # 失败：强制取第一个 fallback
            self._runner.take_action(actions[0])
            self._actions_taken += 1


__all__ = ["V8Env", "SOLVER_BUDGETS", "DEFAULT_MAX_STEPS_PER_EPISODE"]

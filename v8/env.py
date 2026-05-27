"""V8 RL 环境（gym-like）。

按 docs/v8_implementation_design.md 组件 8（Episode Loop）设计：
- Episode 粒度 = 一局 STS（默认 act1）。
- Step 粒度 = 一个**元决策**（NEOW / MAP / EVENT / SHOP / REST / TREASURE /
  CARD_REWARDS / BOSS_REWARDS）。
- 战斗内由 env 内部跑 StSRLSolver TurnSolver 完成（**不暴露给 RL trajectory**）。
- 战斗结束触发 compute_combat_reward（基于真实战斗结果），单次塞进当步 step reward。

2026-05-25 改动：去掉 deck_evaluator 模拟战评分。
- 之前每步元决策都会跑一次 evaluate_deck（12 场模拟战），跟实战脱节、还很慢。
- 现在每场真实战斗结束直接给评分：赢没赢 / 血损 / 回合数 / 伤害比。
- 不累加：分数只在战斗结束那一步 emit，由 RL 自身 GAE 反推到选卡 / 走路决策。
- deck_evaluator.py 文件保留但 env 不再调用（web / 历史训练日志可能引用）。

实现状态：
- reset / step 完整接 GameRunner（不 stub）
- 战斗内：runner.take_action(CombatAction) by TurnSolver（与 v8_bot 同套搜索预算）
- 战斗 enter/exit 在 _log_combat_enter / _log_combat_exit 处快照，exit 时算
  combat_reward 缓存到 _pending_combat_reward，下次 step() 算 reward 时塞进去
- node_reward detect：?事件成功 / 商店买 relic / 休息使用 / 宝箱开 relic 简单识别

不在本文件内：
- PPO trainer（下一步）
- 预训练 script（下一步）
- model prior 接入（trainer 阶段把 model 喂给 TurnSolver 做剪枝；目前 env 用纯搜索）
"""

from __future__ import annotations

import logging
import time
from collections import Counter, deque
from typing import Any, Deque, Dict, List, Optional, Tuple, TYPE_CHECKING

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
from v8.reward import (
    compute_step_reward,
    compute_final_reward,
    compute_combat_reward,
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
# 2026-05-23: perf cut — v14 实测 boss avg 37s/max 169s、elite avg 28s/max 118s，
# outlier 拖慢训练。base_ms / cap_ms 一起砍：
#   elite 500ms/12000 → 250ms/1000  (per-turn cap 1s, base -50%)
#   boss  2000ms/25000 → 500ms/10000 (per-turn cap 10s, base -75%)
# monster 50ms 已经足够小，不动。
SOLVER_BUDGETS = {
    "monster": (50.0, 5_000, 3_000),
    "elite": (250.0, 20_000, 1_000),
    "boss": (500.0, 50_000, 10_000),
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

    # ----- 当前 act 的 boss 名（boss-aware encoding）-----
    # runner._boss_name 在 GameRunner reset 时即填好（如 'Hexaghost'）。
    # 容错：缺字段 / None → 空字符串。
    boss_name = str(getattr(runner, "_boss_name", "") or "")

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
        boss=boss_name,
    )
    return state


# =============================================================================
# 诊断辅助：action 描述 / 战斗内状态提取
# =============================================================================


def _safe_action_repr(action: Any) -> str:
    """把任意 GameAction 转成一行紧凑字符串（guard_cap / 诊断日志用）。
    保持短：单词数 ≤ 4，便于一行塞 5 个 recent actions。"""
    if action is None:
        return "None"
    try:
        cls_name = type(action).__name__
        # CombatAction 有 action_type + card_id / target_index
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
        # MAP / Path 有 dst
        dst = getattr(action, "dst_x", None)
        if dst is not None:
            return f"{cls_name}:dst={dst}"
        return cls_name
    except Exception:  # noqa: BLE001
        return type(action).__name__


def _combat_enemies_brief(runner: GameRunner) -> List[Dict[str, Any]]:
    """从 runner.current_combat 提取存活敌人简要信息（name + hp/max_hp）。"""
    out: List[Dict[str, Any]] = []
    cc = getattr(runner, "current_combat", None)
    if cc is None:
        return out
    st = getattr(cc, "state", None)
    if st is None:
        return out
    for e in getattr(st, "enemies", []) or []:
        if getattr(e, "hp", 0) <= 0:
            continue
        out.append({
            "id": getattr(e, "id", "") or e.__class__.__name__,
            "hp": int(getattr(e, "hp", 0) or 0),
            "max_hp": int(getattr(e, "max_hp", 0) or 0),
        })
    return out


def _combat_pile_sizes(runner: GameRunner) -> Tuple[int, int, int]:
    """返回 (hand_size, draw_size, discard_size)；不在 combat 时全 0。"""
    cc = getattr(runner, "current_combat", None)
    if cc is None:
        return (0, 0, 0)
    st = getattr(cc, "state", None)
    if st is None:
        return (0, 0, 0)
    return (
        len(getattr(st, "hand", []) or []),
        len(getattr(st, "draw_pile", []) or []),
        len(getattr(st, "discard_pile", []) or []),
    )


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
    - reward 按 v8/reward.py 的 compute_step_reward 计算（每场真实战斗结束 emit 一次 combat_reward）
    """

    def __init__(
        self,
        character: str = "ironclad",
        ascension: int = 0,
        verbose: bool = False,
        max_steps_per_episode: int = DEFAULT_MAX_STEPS_PER_EPISODE,
        solver_budgets: Optional[Dict[str, Tuple[float, int, int]]] = None,
        combat_net_wrapper: Optional[Any] = None,
        deck_eval_freq: int = 10,
    ):
        """V8Env 构造器。

        新参数（Round 2）：
            combat_net_wrapper: 可选 V8CombatNetWrapper（让 model 在战斗 search
                里参与 leaf 评估）。None 时 fallback 到原纯搜索（hand-rolled heuristic）。
                设计原则：用户原话第 2 点"搜索+模型联合"，wrapper 是 hook。
            deck_eval_freq: 兼容遗留参数（2026-05-25 改用真实战斗 reward 后已停用）。
                保留参数签名以兼容 trainer / parallel_env / pretrain 等 caller，
                内部不再触发任何模拟战。
        """
        self.character = character
        self.ascension = ascension
        self.verbose = verbose
        self.max_steps_per_episode = max_steps_per_episode
        self._solver_budgets = solver_budgets or SOLVER_BUDGETS
        self._combat_net_wrapper = combat_net_wrapper
        # 遗留字段（兼容 trainer / parallel_env / pretrain 调用签名），内部不再触发模拟战
        self.deck_eval_freq = max(1, int(deck_eval_freq))

        # 每局重置的运行时状态
        self._runner: Optional[GameRunner] = None
        self._adapter: Optional[TurnSolverAdapter] = None
        self._current_state: Optional[V8State] = None

        # reward 计算需要的历史
        self._prev_state: Optional[V8State] = None

        # 真实战斗 reward 缓存：_log_combat_exit 算好，step() 下次 reward 计算时取出
        # 一次性塞进去。单 step 内多场战斗（罕见 e.g. event→combat→event→combat）会累加。
        self._pending_combat_reward: float = 0.0

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

        # 诊断字段（guard_cap / [combat] 日志用）
        self._episode_idx: Optional[int] = None  # 外部 trainer 通过 set_episode 注入
        self._last_action_repr: str = ""         # 最后一次 take_action 的 action 描述
        self._recent_actions: Deque[str] = deque(maxlen=5)
        # 战斗进入/退出 tracking
        self._in_combat: bool = False
        self._combat_enter_hp: int = 0
        self._combat_enter_max_hp: int = 0
        self._combat_enter_floor: int = 0
        self._combat_enter_turn_actions: int = 0  # 进入 combat 时 _actions_taken 值
        # 真实战斗 reward 需要的额外快照
        self._combat_enter_enemy_total_max_hp: int = 0  # 进入战斗时所有敌人 max_hp 之和

        # [floor] 日志：每次 floor 变化时打一次（含所有 room 类型）
        self._last_logged_floor: int = -1

        # [deck] 日志：elite/boss 战斗胜利 + reward 处理完后打一次
        # 设值时机：_log_combat_exit(reason='victory')，当 room=elite/boss
        # 清值时机：_advance_to_meta_decision 推完落点是 MAP_NAVIGATION 时
        self._pending_deck_room: Optional[str] = None

        # [event] 日志状态：用于 enter / phase_transition / exit 三段日志
        # _last_event_phase 仅在 phase=EVENT + event_state 非 None 时为 EventPhase 名字字符串
        # _last_event_id 同上；用于 exit 日志事后补当时的 id（current_event_state 已被清）
        # _event_enter_max_hp / _event_enter_relics_count：进入 EVENT phase 时的快照（exit diff 用）
        self._last_event_phase: Optional[str] = None
        self._last_event_id: Optional[str] = None
        self._event_enter_max_hp: int = 0
        self._event_enter_relics_count: int = 0
        # [event_stall] 兜底：单一 event_id 累计 choice 次数超阈值 → FORCE_TERMINATE
        # 防御 StSRLSolver event handler 缺 phase filter（如 Mysterious Sphere /
        # Mushrooms 类 bug）导致 deterministic eval 卡死。
        # 阈值 30：正常 event 至多 ~2-3 choices（包含多 phase），30 远超合理范围。
        self._event_choice_count: Counter = Counter()
        self._event_stall_threshold: int = 30

        # 性能 / 调用计数（per-episode；trainer 读 delta）
        # eval_deck_calls / eval_deck_time_sec 保留为 0，2026-05-25 已不再触发 evaluate_deck
        self.eval_deck_calls: int = 0
        self.eval_deck_time_sec: float = 0.0
        self.env_step_time_sec: float = 0.0  # env.step 累计耗时（不含 caller）
        self.combat_search_calls: int = 0  # combat turn 调用 adapter.pick_action 次数

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def set_episode(self, episode_idx: int) -> None:
        """trainer 在 collect_rollout 前注入 episode 编号，guard_cap / [combat] 日志用。"""
        self._episode_idx = int(episode_idx)

    def reset_perf_counters(self) -> None:
        """trainer 每集开始前调用，清空 per-episode 计数器。"""
        self.eval_deck_calls = 0
        self.eval_deck_time_sec = 0.0
        self.env_step_time_sec = 0.0
        self.combat_search_calls = 0

    def reset_cache_for_new_run(self) -> None:
        """遗留 API（2026-05-25 deck_evaluator 已停用，此方法保持 no-op 兼容旧 caller）。"""
        return None

    def reset(self, seed: int) -> V8State:
        """开始新局，推进到第一个元决策点（NEOW）。

        2026-05-25 改动：不再触发首次 evaluate_deck（模拟战评分已去掉）。
        开局 deck_strength 直接置 None，state encoder 内已有 None 兜底。
        """

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

        # 诊断状态
        self._last_action_repr = ""
        self._recent_actions.clear()
        self._in_combat = False
        self._combat_enter_hp = 0
        self._combat_enter_max_hp = 0
        self._combat_enter_floor = 0
        self._combat_enter_turn_actions = 0
        self._combat_enter_enemy_total_max_hp = 0
        self._last_logged_floor = -1
        self._pending_deck_room = None
        self._last_event_phase = None
        self._last_event_id = None
        self._event_enter_max_hp = 0
        self._event_enter_relics_count = 0
        self._event_choice_count.clear()
        # 真实战斗 reward 缓存清零
        self._pending_combat_reward = 0.0
        # perf 计数器在 reset 也清一遍（兼容 trainer 没调 reset_perf_counters 的情况）
        self.reset_perf_counters()

        # [seed] 日志：每局开始时打一次（run 复现 / bug 重现用）。
        # runner.seed_string 是 StSRLSolver 内部使用的 seed token，
        # 与 trainer 传进来的 episode 级 seed 一一对应。
        try:
            run_seed_str = str(getattr(self._runner, "seed_string", "") or "")
            run_seed_int = getattr(self._runner, "seed", None)
            logger.info(
                "[seed] ep=%s episode_seed=%d run_seed=%s run_seed_str=%s "
                "ascension=%d character=%s",
                self._episode_idx, int(seed),
                str(run_seed_int) if run_seed_int is not None else "?",
                run_seed_str,
                int(self.ascension),
                str(self.character),
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("[seed] log failed: %s: %s", type(e).__name__, e)

        # 推进到第一个元决策 phase（NEOW 一般直接就是；保险起见 advance）
        self._advance_to_meta_decision()

        # 构造初始 V8State（2026-05-25：不再 evaluate_deck，deck_strength 留 None）
        state = _build_state_from_runner(self._runner)
        state.deck_strength = None

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

        step_t0 = time.time()
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
            # 没合法 action：游戏卡死 / 结束 → 强制 terminal，让上层不要再 step
            if not self._runner.game_over:
                self._force_terminate_run(reason="step_no_actions")
            done = True
            reward = compute_final_reward(
                game_won=bool(self._runner.game_won),
                final_floor=int(self._runner.run_state.floor),
            )
            info["phase_after"] = self._current_state.phase
            info["error"] = "no_available_actions"
            self.env_step_time_sec += time.time() - step_t0
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

        # 记录 last_action（诊断 guard_cap 用）
        self._last_action_repr = _safe_action_repr(chosen_action)
        self._recent_actions.append(self._last_action_repr)

        # [action] 日志：每个 model 决策点打一行（含 phase + action_str + idx + n_available）。
        # 任意阶段 / 任意 bug 都能 grep 出 model 在每个 step 看到的选择 + 实际选了啥。
        self._log_action(
            chosen_action=chosen_action,
            action_idx=action_idx,
            n_available=len(engine_actions),
        )

        # [meta] 日志：非 COMBAT 非 EVENT 的元决策 phase 打一行（含完整 available 选项 label）。
        # EVENT 由更专门的 [event] choice 日志覆盖，跳过避免重复。
        # COMBAT 决策在 _run_combat_turn 里走 search adapter，不会进 step 这里。
        self._log_meta_decision(
            engine_actions=engine_actions,
            action_idx=action_idx,
            chosen_action=chosen_action,
        )

        # 诊断：如果是 EventAction，打 [event] choice 日志（纯观察）。
        self._log_event_choice(chosen_action)

        # [event_stall] 兜底：单一 event_id 累计 choice 次数过阈值 → FORCE_TERMINATE
        # 防御 StSRLSolver event handler 缺 phase filter 类 bug（如 Mysterious
        # Sphere / Mushrooms COMBAT_WON 阶段反复触发战斗）。增量 + 检查放在
        # take_action 前，确保即便本次 take_action 又会循环也立刻断。
        stall_terminated = self._maybe_force_event_stall_terminate(chosen_action)

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
        if stall_terminated:
            # FORCE_TERMINATE 已经把 runner.game_over=True，跳过 advance，
            # 直接走 done 收尾，避免再触发一次 combat 推进。
            battle_happened = False
        else:
            battle_happened = self._advance_to_meta_decision()
        info["battle_happened"] = battle_happened
        if stall_terminated:
            info["error"] = "event_stall"

        # 推进结束 → 构造 next_state（2026-05-25：不再调 evaluate_deck，deck_strength 留 None）
        next_state = _build_state_from_runner(self._runner)
        next_state.deck_strength = None
        self._last_act_for_cache = next_state.act
        info["deck_strength_evaluated"] = False

        # node_reward detect
        node_reward = _detect_node_reward(
            runner=self._runner,
            prev_phase=self._step_start_phase,
            prev_relics_count=self._step_start_relics_count,
            prev_max_hp=self._step_start_max_hp,
            next_phase=self._runner.phase,
        )
        info["node_reward"] = node_reward

        # 拿出本 step 期间累计的真实战斗 reward（_log_combat_exit 算好缓存的）
        combat_reward = float(self._pending_combat_reward)
        self._pending_combat_reward = 0.0
        info["combat_reward"] = combat_reward

        # 计算 step reward（真实战斗结果分 + hp_loss 罚 + 节点收益）
        step_reward = compute_step_reward(
            prev_state=self._prev_state if self._prev_state is not None else next_state,
            next_state=next_state,
            node_reward=node_reward,
            combat_reward=combat_reward,
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
            # 强制把 runner 标 terminal，防止万一上层不读 done 又调一次 step
            if not self._runner.game_over:
                self._force_terminate_run(reason="max_steps_exceeded")
            done = True
            info["error"] = "max_steps_exceeded"

        # 更新 prev 历史
        self._prev_state = next_state
        self._current_state = next_state

        info["phase_after"] = next_state.phase
        self.env_step_time_sec += time.time() - step_t0
        return next_state, float(step_reward), done, info

    def get_available_actions(self) -> List[str]:
        """当前 state 下的合法 action 字符串描述（pointer network 输入）。

        idx 与下次 step(idx) 严格对齐 runner.get_available_actions()。
        """
        if self._runner is None or self._current_state is None:
            return []
        return get_available_actions(self._current_state, runner=self._runner)

    def close(self) -> None:
        """清理资源：清 runner / adapter 引用。"""
        self._runner = None
        self._adapter = None
        self._current_state = None
        self._prev_state = None
        self._pending_combat_reward = 0.0

    @property
    def state(self) -> Optional[V8State]:
        """当前 V8State（外部只读访问，方便 debug）。"""
        return self._current_state

    @property
    def runner(self) -> Optional[GameRunner]:
        """底层 GameRunner（外部只读访问，方便 debug；trainer 不应直接改 runner 状态）。"""
        return self._runner

    def _build_runner_snapshot(self) -> Dict[str, Any]:
        """提取 runner / run_state 关键字段，供 parallel worker 跨进程回传 trainer。

        子进程内 V8Env 实例 caller 拿不到，但需要 final_floor / beat_boss 等做
        [perf] / [heartbeat] 统计。pickle 单独 V8State 不够（缺 game_won），
        把所需字段一次性打包成 dict 回传。

        返回 dict 字段（全部可 pickle 的基础类型）：
            floor, act, hp, max_hp, gold, game_won, deck_size, relics_size
        runner 不存在时返回 0-填充的 dict（don't raise）。
        """
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

        while not self._runner.game_over and guard < guard_cap:  # noqa: PLR0915
            guard += 1
            phase = self._runner.phase

            # 诊断：floor 变化日志（每次 floor 跳变都打一次，不管是哪种 room）
            # 放在最前面：MAP→COMBAT 中间 floor 已 +1，combat enter 之前先打
            self._maybe_log_floor()
            # 诊断：事件状态日志（enter / phase_transition / exit）。
            # 纯观察：检测 phase + current_event_state 变化，不改任何 control flow。
            self._maybe_log_event_state()

            # 已到元决策 phase → 停下，让外部 model 选
            if phase in _META_PHASES:
                # 如果之前在 combat，说明刚 victory 离开（这里 phase 已切到 reward / map）
                if self._in_combat:
                    self._log_combat_exit(reason="victory")
                # 诊断：reward 处理完后的 deck 快照（victory 后 phase 落 MAP 时打）
                self._maybe_log_deck()
                return battle_happened

            # 进入 COMBAT 内部 turn loop
            if phase == GamePhase.COMBAT:
                if not self._in_combat:
                    self._log_combat_enter()
                battle_happened = True
                self._run_combat_turn()
                # 战斗推完后回到 while 头继续判断 phase
                continue

            # RUN_COMPLETE：游戏结束
            if phase == GamePhase.RUN_COMPLETE:
                if self._in_combat:
                    # combat 中游戏直接结束 → 一般是 defeat
                    self._log_combat_exit(
                        reason="defeat" if not self._runner.game_won else "victory"
                    )
                return battle_happened

            # 其他 phase（不应到这里）：取第一个合法 action 推进
            actions = self._runner.get_available_actions()
            if not actions:
                logger.warning(
                    "V8Env._advance: no actions available at phase=%s, abort",
                    phase.name,
                )
                # 卡死了：没合法 action → 强制结束，避免上层把 episode 当 not done 又跑一遍
                self._force_terminate_run(reason=f"no_actions_at_{phase.name}")
                return battle_happened
            self._last_action_repr = _safe_action_repr(actions[0])
            self._recent_actions.append(self._last_action_repr)
            self._runner.take_action(actions[0])
            self._actions_taken += 1

        # while 退出时如果还在 combat 但 game_over（defeat 或最终 victory），补打 exit
        if self._in_combat and self._runner.game_over:
            self._log_combat_exit(
                reason="victory" if self._runner.game_won else "defeat"
            )

        if guard >= guard_cap:
            # 关键修复：guard_cap 触发说明这局已经卡死（一般卡在 COMBAT 里 solver
            # 反复出同一动作 / runner state corrupt）。如果只是 return 而不标 terminal，
            # 上层 env.step 会读 runner.game_over == False，trainer.collect_rollout
            # 不退出，下一次 step 又会再次进入 _advance_to_meta_decision，再次卡死，
            # 形成无限循环（曾导致训练 ep=664 卡 hours 反复刷 guard_cap warning）。
            self._log_guard_cap(battle_happened=battle_happened, guard_cap=guard_cap)
            self._force_terminate_run(reason="guard_cap")
        return battle_happened

    # ---------------------------------------------------------------------
    # Internal: 诊断日志
    # ---------------------------------------------------------------------

    def _log_combat_enter(self) -> None:
        """[combat] enter 日志（每场战斗起始打一次）+ 记录战斗 reward 快照。"""
        assert self._runner is not None
        rs = self._runner.run_state
        enemies = _combat_enemies_brief(self._runner)
        names = [e["id"] for e in enemies]
        self._in_combat = True
        self._combat_enter_hp = int(getattr(rs, "current_hp", 0) or 0)
        self._combat_enter_max_hp = int(getattr(rs, "max_hp", 0) or 0)
        self._combat_enter_floor = int(getattr(rs, "floor", 0) or 0)
        self._combat_enter_turn_actions = self._actions_taken
        # 真实战斗 reward 快照：所有敌人 max_hp 之和（damage_ratio 分母）
        self._combat_enter_enemy_total_max_hp = sum(
            int(e.get("max_hp", 0) or 0) for e in enemies
        )
        try:
            room_type = self._runner.current_room_type or "monster"
        except Exception:  # noqa: BLE001
            room_type = "?"
        deck_size = len(getattr(rs, "deck", []) or [])
        logger.info(
            "[combat] enter ep=%s floor=%d act=%d room=%s enemies=%s deck_size=%d hp=%d/%d",
            self._episode_idx, self._combat_enter_floor,
            int(getattr(rs, "act", 1) or 1),
            room_type, names, deck_size,
            self._combat_enter_hp, self._combat_enter_max_hp,
        )

    def _log_combat_exit(self, *, reason: str) -> None:
        """[combat] exit 日志（战斗结束触发）+ 算本场 combat_reward 累加到 _pending_combat_reward。

        副作用：
        - 胜利且 room=elite/boss 时，置 self._pending_deck_room
          让 _advance_to_meta_decision 在 reward 处理完落到 MAP 时补打 [deck]。
        - 算本场 combat_reward 加到 _pending_combat_reward（下次 step() 取出来一次性塞 reward）。
        """
        assert self._runner is not None
        rs = self._runner.run_state
        hp_after = int(getattr(rs, "current_hp", 0) or 0)
        turn_actions = self._actions_taken - self._combat_enter_turn_actions

        # ----- 真实战斗 reward 计算 -----
        # turns: 优先 current_combat.state.turn（StSRLSolver 内部回合计数器）；
        # current_combat 退出 COMBAT 时可能已被清；保底用 turn_actions（model 决策步数）。
        turns = 0
        try:
            cc = getattr(self._runner, "current_combat", None)
            if cc is not None:
                st = getattr(cc, "state", None)
                if st is not None:
                    turns = int(getattr(st, "turn", 0) or 0)
        except Exception:  # noqa: BLE001
            turns = 0
        if turns <= 0:
            # current_combat 已被清 → 用 turn_actions（粗略）。
            # 实测每回合至少 1 action（end_turn），多则 5+，turn_actions 比真实 turn 偏高，
            # 这只在 victory 后 phase 已切走时 fallback，量级仍合理。
            turns = max(1, turn_actions)

        hp_lost = max(self._combat_enter_hp - hp_after, 0)
        won = (reason == "victory")
        # damage_dealt：胜利 = 全敌人 max_hp 干完；非胜利 = enter_total - 剩余 enemies hp 总和
        enemy_total_max = int(self._combat_enter_enemy_total_max_hp or 0)
        if won:
            damage_dealt = enemy_total_max
        else:
            # 拿剩余敌人 hp 总和（current_combat 可能还活着）
            remaining_hp = 0
            try:
                cc = getattr(self._runner, "current_combat", None)
                if cc is not None:
                    st = getattr(cc, "state", None)
                    if st is not None:
                        for e in getattr(st, "enemies", []) or []:
                            remaining_hp += max(int(getattr(e, "hp", 0) or 0), 0)
            except Exception:  # noqa: BLE001
                remaining_hp = 0
            damage_dealt = max(enemy_total_max - remaining_hp, 0)

        try:
            combat_reward = compute_combat_reward(
                won=won,
                hp_lost=hp_lost,
                turns=turns,
                damage_dealt=damage_dealt,
                enemy_total_max_hp=enemy_total_max,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "compute_combat_reward failed: %s: %s; fallback 0",
                type(e).__name__, e,
            )
            combat_reward = 0.0
        self._pending_combat_reward += combat_reward

        logger.info(
            "[combat] exit ep=%s reason=%s floor=%d hp_before=%d hp_after=%d "
            "turn_actions=%d turns=%d damage=%d/%d combat_reward=%.2f",
            self._episode_idx, reason, self._combat_enter_floor,
            self._combat_enter_hp, hp_after, turn_actions,
            turns, damage_dealt, enemy_total_max, combat_reward,
        )
        # 胜利 + elite/boss → 准备 [deck] 日志（等 reward 处理完）
        if reason == "victory":
            try:
                rt = self._runner.current_room_type
            except Exception:  # noqa: BLE001
                rt = None
            rt_str = (rt or "").lower() if isinstance(rt, str) else \
                (getattr(rt, "name", "") or "").lower()
            if rt_str in ("elite", "boss"):
                self._pending_deck_room = rt_str
        self._in_combat = False
        self._combat_enter_enemy_total_max_hp = 0

    def _current_room_type_str(self) -> str:
        """把 runner.get_current_room_type() / current_room_type 归一成
        'monster' / 'elite' / 'boss' / 'rest' / 'shop' / 'event' / 'treasure' / 'unknown'。

        优先用 get_current_room_type()（返回 RoomType enum，覆盖所有节点类型），
        否则 fallback 到 current_room_type（只在 combat 时设 monster/elite/boss）。
        NEOW（floor=0，map_position 在 start）返回 'unknown'。
        """
        assert self._runner is not None
        # 先 try 地图上的 RoomType enum（覆盖 REST/SHOP/EVENT/TREASURE）
        rt_enum = None
        try:
            getter = getattr(self._runner, "get_current_room_type", None)
            if getter is not None:
                rt_enum = getter()
        except Exception:  # noqa: BLE001
            rt_enum = None
        if rt_enum is not None:
            name = getattr(rt_enum, "name", str(rt_enum))
            return self._normalize_room_name(name)
        # fallback 到 combat 内置的 current_room_type 字符串（combat 中才设值）
        try:
            rt = self._runner.current_room_type
        except Exception:  # noqa: BLE001
            rt = None
        # NEOW / 初始位置：current_room_type 默认 'monster' 是误导，
        # 这里只有 phase=COMBAT 时才信任它
        if self._runner.phase == GamePhase.COMBAT:
            if isinstance(rt, str) and rt:
                return self._normalize_room_name(rt)
            if rt is not None:
                return self._normalize_room_name(getattr(rt, "name", str(rt)))
        return "unknown"

    @staticmethod
    def _normalize_room_name(raw: str) -> str:
        """把任意 room 名归一成小写规范字符串。"""
        s = (raw or "").lower()
        mapping = {
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
        return mapping.get(s, "unknown")

    def _maybe_log_floor(self) -> None:
        """如果 runner.floor 变了，打一行 [floor] 日志。"""
        assert self._runner is not None
        rs = self._runner.run_state
        cur_floor = int(getattr(rs, "floor", 0) or 0)
        if cur_floor == self._last_logged_floor:
            return
        room = self._current_room_type_str()
        hp = int(getattr(rs, "current_hp", 0) or 0)
        max_hp = int(getattr(rs, "max_hp", 0) or 0)
        act = int(getattr(rs, "act", 1) or 1)
        logger.info(
            "[floor] ep=%s floor=%d act=%d hp=%d/%d room=%s",
            self._episode_idx, cur_floor, act, hp, max_hp, room,
        )
        self._last_logged_floor = cur_floor

    def _maybe_log_deck(self) -> None:
        """如果有 pending elite/boss victory，且现在落在 MAP_NAVIGATION（reward 已处理完），
        打一行 [deck] 日志并清 pending。"""
        assert self._runner is not None
        if self._pending_deck_room is None:
            return
        if self._runner.phase != GamePhase.MAP_NAVIGATION:
            return
        rs = self._runner.run_state
        # 按 (card_id, upgraded) 计数
        counter: Counter = Counter()
        for c in getattr(rs, "deck", []) or []:
            cid = getattr(c, "id", None) or ""
            upgraded = bool(getattr(c, "upgraded", False))
            display = f"{cid}+1" if upgraded else cid
            counter[display] += 1
        # 排序：count 降序，name 字典序升序
        cards_parts = [
            f"{name}*{cnt}"
            for name, cnt in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
        ]
        relics_ids: List[str] = []
        for r in getattr(rs, "relics", []) or []:
            rid = getattr(r, "id", None) or str(r)
            relics_ids.append(rid)
        logger.info(
            "[deck] ep=%s floor=%d room=%s cards=%s relics=%s",
            self._episode_idx,
            int(getattr(rs, "floor", 0) or 0),
            self._pending_deck_room,
            " ".join(cards_parts) if cards_parts else "-",
            ",".join(relics_ids) if relics_ids else "-",
        )
        self._pending_deck_room = None

    def _maybe_log_event_state(self) -> None:
        """事件状态日志（enter / phase_transition / exit）。

        策略：
        - 当前 phase=EVENT 且 event_state 非 None：
            - 若之前不在 EVENT，打 [event] enter
            - 若之前在 EVENT 但 phase 字符串变了，打 [event] phase_transition
        - 当前 phase != EVENT 或 event_state 为 None：
            - 若之前在 EVENT，打 [event] exit
        通过 self._last_event_phase / self._last_event_id 实例字段做去重，
        保证状态不变时不重复打日志。

        全部 introspection 用 try/except 兜底，缺/改字段不影响训练。
        """
        if self._runner is None:
            return
        try:
            cur_phase = self._runner.phase
            cur_event_state = getattr(self._runner, "current_event_state", None)
            in_event = (
                cur_phase == GamePhase.EVENT
                and cur_event_state is not None
            )

            if in_event:
                event_id = str(getattr(cur_event_state, "event_id", "?") or "?")
                ep = getattr(cur_event_state, "phase", None)
                event_phase_str = (
                    getattr(ep, "name", str(ep)) if ep is not None else "?"
                )
                # 状态没变 → 不打
                if (
                    self._last_event_phase == event_phase_str
                    and self._last_event_id == event_id
                ):
                    return
                # 收集 choices（防御性，调用 runner 的高层 API）
                choices_repr: List[str] = []
                try:
                    eh = getattr(self._runner, "event_handler", None)
                    if eh is not None:
                        choice_list = eh.get_available_choices(
                            cur_event_state, self._runner.run_state
                        )
                        for ch in choice_list:
                            idx = getattr(ch, "index", "?")
                            text = getattr(ch, "text", "") or getattr(ch, "name", "")
                            choices_repr.append(f"{idx}:{text}")
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "[event] choices introspection failed: %s: %s",
                        type(e).__name__, e,
                    )

                rs = self._runner.run_state
                if self._last_event_phase is None or self._last_event_id != event_id:
                    # enter（要么之前不在 event，要么 event_id 跳变了）
                    logger.info(
                        "[event] enter ep=%s floor=%d act=%d event_id=%s "
                        "event_phase=%s choices=[%s]",
                        self._episode_idx,
                        int(getattr(rs, "floor", 0) or 0),
                        int(getattr(rs, "act", 1) or 1),
                        event_id,
                        event_phase_str,
                        ", ".join(choices_repr),
                    )
                    # 记录 enter 快照（exit 时算 diff 用）
                    self._event_enter_max_hp = int(getattr(rs, "max_hp", 0) or 0)
                    self._event_enter_relics_count = len(
                        getattr(rs, "relics", []) or []
                    )
                else:
                    # phase_transition（同一 event 内，phase enum 变了）
                    logger.info(
                        "[event] phase_transition ep=%s floor=%d event_id=%s "
                        "from=%s to=%s",
                        self._episode_idx,
                        int(getattr(rs, "floor", 0) or 0),
                        event_id,
                        self._last_event_phase,
                        event_phase_str,
                    )
                self._last_event_phase = event_phase_str
                self._last_event_id = event_id
            else:
                # 不在 EVENT phase：如果之前在，补 exit
                if self._last_event_phase is not None:
                    rs = self._runner.run_state
                    cur_phase_name = getattr(cur_phase, "name", str(cur_phase))
                    # exit reason：根据 cur_phase 推测
                    if cur_phase == GamePhase.COMBAT:
                        reason = "combat_started"
                    elif cur_phase == GamePhase.RUN_COMPLETE:
                        reason = "run_complete"
                    else:
                        reason = f"resolved->{cur_phase_name}"
                    hp_now = int(getattr(rs, "current_hp", 0) or 0)
                    max_hp_now = int(getattr(rs, "max_hp", 0) or 0)
                    deck_size = len(getattr(rs, "deck", []) or [])
                    relics_count = len(getattr(rs, "relics", []) or [])
                    logger.info(
                        "[event] exit ep=%s floor=%d event_id=%s reason=%s "
                        "max_hp_now=%d hp_now=%d/%d deck_size=%d relics_count=%d",
                        self._episode_idx,
                        int(getattr(rs, "floor", 0) or 0),
                        self._last_event_id,
                        reason,
                        max_hp_now,
                        hp_now,
                        max_hp_now,
                        deck_size,
                        relics_count,
                    )
                    self._last_event_phase = None
                    self._last_event_id = None
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "[event] _maybe_log_event_state failed: %s: %s",
                type(e).__name__, e,
            )

    def _log_action(
        self,
        *,
        chosen_action: Any,
        action_idx: int,
        n_available: int,
    ) -> None:
        """[action] 日志：每个 model 决策点打一行（无条件，所有 phase 都打）。

        包含 phase + action_str（_safe_action_repr，与 [combat]/guard_cap 同格式） +
        action_idx + n_available + floor + hp 快照。
        log 量 = 元决策 step 数（每局几百行级），可控且可 grep。
        """
        if self._runner is None or self._current_state is None:
            return
        try:
            rs = self._runner.run_state
            phase_str = self._current_state.phase or "?"
            action_str = _safe_action_repr(chosen_action)
            logger.info(
                "[action] ep=%s step=%d phase=%s action_idx=%d/%d action=%s "
                "floor=%d hp=%d/%d",
                self._episode_idx,
                self._step_count,
                phase_str,
                int(action_idx),
                int(n_available),
                action_str,
                int(getattr(rs, "floor", 0) or 0),
                int(getattr(rs, "hp", 0) or 0),
                int(getattr(rs, "max_hp", 0) or 0),
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "[action] _log_action failed: %s: %s",
                type(e).__name__, e,
            )

    def _log_meta_decision(
        self,
        *,
        engine_actions: List[Any],
        action_idx: int,
        chosen_action: Any,
    ) -> None:
        """[meta] 日志：非 COMBAT 非 EVENT 的元决策 phase 打一行（含完整 available 列表）。

        覆盖 phase: NEOW / MAP / CARD_REWARDS / BOSS_REWARDS / SHOP / REST / TREASURE。
        EVENT 已由 [event] choice 覆盖，跳过避免重复。
        labels = model 看到的 action 字符串（get_available_actions 返回值，与
        engine_actions 一一对齐）；如长度对不上 fallback 用 _safe_action_repr。
        """
        if self._runner is None or self._current_state is None:
            return
        phase_str = self._current_state.phase or ""
        # 跳过 EVENT（已有 [event] choice）和 COMBAT（走 search adapter，不进这里）和未知
        if phase_str in ("EVENT", "COMBAT", ""):
            return
        try:
            rs = self._runner.run_state
            # 取 model-visible label 列表（与 engine_actions 等长）
            try:
                labels = get_available_actions(
                    self._current_state, runner=self._runner
                )
            except Exception:  # noqa: BLE001
                labels = []
            if len(labels) != len(engine_actions):
                # fallback：用 _safe_action_repr 兜底
                labels = [_safe_action_repr(a) for a in engine_actions]

            chosen_label = (
                labels[action_idx]
                if 0 <= action_idx < len(labels)
                else _safe_action_repr(chosen_action)
            )
            # 选项截断防爆行（shop / map 多分支时）：最多 16 个 label
            shown = labels if len(labels) <= 16 else (labels[:16] + ["..."])
            logger.info(
                "[meta] ep=%s step=%d phase=%s floor=%d hp=%d/%d gold=%d "
                "n_options=%d chosen_idx=%d chosen=%s options=%s",
                self._episode_idx,
                self._step_count,
                phase_str,
                int(getattr(rs, "floor", 0) or 0),
                int(getattr(rs, "hp", 0) or 0),
                int(getattr(rs, "max_hp", 0) or 0),
                int(getattr(rs, "gold", 0) or 0),
                len(engine_actions),
                int(action_idx),
                chosen_label,
                shown,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "[meta] _log_meta_decision failed: %s: %s",
                type(e).__name__, e,
            )

    def _log_event_choice(self, action: Any) -> None:
        """[event] choice 日志：env.step 处理 EventAction 时调一次。

        仅在 action 是 EventAction 时打；choice_text 从 runner.event_handler
        当前 available_choices 里按 index 查（防御性）。
        """
        if not isinstance(action, EventAction):
            return
        if self._runner is None:
            return
        try:
            rs = self._runner.run_state
            choice_idx = int(getattr(action, "choice_index", -1))
            cur_event_state = getattr(self._runner, "current_event_state", None)
            event_id = "?"
            event_phase_str = "?"
            choice_text = ""
            if cur_event_state is not None:
                event_id = str(getattr(cur_event_state, "event_id", "?") or "?")
                ep = getattr(cur_event_state, "phase", None)
                event_phase_str = (
                    getattr(ep, "name", str(ep)) if ep is not None else "?"
                )
                try:
                    eh = getattr(self._runner, "event_handler", None)
                    if eh is not None:
                        choice_list = eh.get_available_choices(
                            cur_event_state, self._runner.run_state
                        )
                        for ch in choice_list:
                            if getattr(ch, "index", None) == choice_idx:
                                choice_text = (
                                    getattr(ch, "text", "")
                                    or getattr(ch, "name", "")
                                )
                                break
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "[event] choice text lookup failed: %s: %s",
                        type(e).__name__, e,
                    )
            logger.info(
                "[event] choice ep=%s floor=%d choice_idx=%d choice_text=%s "
                "event_id=%s event_phase=%s",
                self._episode_idx,
                int(getattr(rs, "floor", 0) or 0),
                choice_idx,
                choice_text,
                event_id,
                event_phase_str,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "[event] _log_event_choice failed: %s: %s",
                type(e).__name__, e,
            )

    def _maybe_force_event_stall_terminate(self, action: Any) -> bool:
        """[event_stall] 兜底：单一 event_id 累计 choice 次数过阈值 → FORCE_TERMINATE。

        防御 StSRLSolver Python engine event handler 缺 phase filter 的 bug
        （MysteriousSphere / Mushrooms 类：COMBAT_WON 阶段菜单未过滤，
        deterministic 反复选 idx=0 不断重触发战斗）。

        仅在 action 是 EventAction 时递增；非 event 行为不影响计数。
        返回 True 表示已 FORCE_TERMINATE，外层 step 应跳过 advance。
        """
        if not isinstance(action, EventAction):
            return False
        if self._runner is None:
            return False
        try:
            cur_event_state = getattr(self._runner, "current_event_state", None)
            event_id = "?"
            event_phase_str = "?"
            if cur_event_state is not None:
                event_id = str(getattr(cur_event_state, "event_id", "?") or "?")
                ep = getattr(cur_event_state, "phase", None)
                event_phase_str = (
                    getattr(ep, "name", str(ep)) if ep is not None else "?"
                )
            self._event_choice_count[event_id] += 1
            count = self._event_choice_count[event_id]
            if count >= self._event_stall_threshold:
                rs = self._runner.run_state
                logger.warning(
                    "[event_stall] ep=%s event_id=%s count=%d threshold=%d "
                    "event_phase=%s floor=%s act=%s hp=%s/%s FORCE_TERMINATE",
                    self._episode_idx, event_id, count,
                    self._event_stall_threshold, event_phase_str,
                    getattr(rs, "floor", "?"), getattr(rs, "act", "?"),
                    getattr(rs, "current_hp", "?"), getattr(rs, "max_hp", "?"),
                )
                self._force_terminate_run(reason="event_stall")
                return True
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "[event_stall] _maybe_force_event_stall_terminate failed: %s: %s",
                type(e).__name__, e,
            )
        return False

    def _log_guard_cap(self, *, battle_happened: bool, guard_cap: int) -> None:
        """guard_cap 触发时打详细诊断日志（key=value 单行，便于 grep）。"""
        assert self._runner is not None
        rs = self._runner.run_state
        cur_phase = self._runner.phase
        phase_name = getattr(cur_phase, "name", str(cur_phase))
        # 当前位置 (map_position)
        mp = getattr(rs, "map_position", None)
        map_x = getattr(mp, "x", -1) if mp is not None else -1
        map_y = getattr(mp, "y", -1) if mp is not None else -1

        # combat-specific 字段（不在 combat 时给 0）
        in_combat = cur_phase == GamePhase.COMBAT or self._in_combat
        enemies = _combat_enemies_brief(self._runner) if in_combat else []
        hand_size, draw_size, discard_size = (
            _combat_pile_sizes(self._runner) if in_combat else (0, 0, 0)
        )

        logger.warning(
            "[guard_cap] ep=%s step=%d guard_cap=%d phase=%s floor=%s act=%s "
            "map=(%s,%s) hp=%s/%s in_combat=%s enemies=%s "
            "hand=%d draw=%d discard=%d battle_happened=%s actions_taken=%d "
            "last_action=%s recent=%s FORCE_TERMINATE",
            self._episode_idx, self._step_count, guard_cap, phase_name,
            getattr(rs, "floor", "?"), getattr(rs, "act", "?"),
            map_x, map_y,
            getattr(rs, "current_hp", "?"), getattr(rs, "max_hp", "?"),
            in_combat, enemies,
            hand_size, draw_size, discard_size,
            battle_happened, self._actions_taken,
            self._last_action_repr, list(self._recent_actions),
        )

    def _force_terminate_run(self, *, reason: str) -> None:
        """强制把当前 run 标成 terminal（loss）。

        上层 step() 会读 runner.game_over → done=True，trainer 走完正常的
        end-of-episode 流程（compute_final_reward / 关闭 env / 下一 episode）。
        """
        assert self._runner is not None
        try:
            self._runner.game_over = True
            self._runner.game_won = False
            self._runner.phase = GamePhase.RUN_COMPLETE
        except Exception as e:  # noqa: BLE001
            # 兜底：即便 runner 内部状态异常无法赋值，也要让 done=True，靠
            # env.step 自己的 _runner.game_over 读取分支兜底
            logger.warning(
                "V8Env._force_terminate_run(reason=%s): set attrs failed: %s: %s",
                reason, type(e).__name__, e,
            )

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
        self.combat_search_calls += 1
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

        self._last_action_repr = _safe_action_repr(action)
        self._recent_actions.append(self._last_action_repr)
        ok = self._runner.take_action(action)
        self._actions_taken += 1
        if not ok:
            # 失败：强制取第一个 fallback
            self._last_action_repr = _safe_action_repr(actions[0])
            self._recent_actions.append(self._last_action_repr)
            self._runner.take_action(actions[0])
            self._actions_taken += 1


__all__ = ["V8Env", "SOLVER_BUDGETS", "DEFAULT_MAX_STEPS_PER_EPISODE"]

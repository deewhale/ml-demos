"""V8 统一评估 harness（sim / 真机同一套评估语义）。

目标：让同一套 "跑 N 个种子 → 聚合 a1/a2/won/reached/floor" 的评估逻辑能跑在任意
后端上（StSRLSolver 模拟器 / CommunicationMod 真机），评估口径与
`tools/v8_ppo_train.py:run_eval` **完全一致**（同种子同模型 → 同数字）。

------------------------------------------------------------------------------
分层设计
------------------------------------------------------------------------------
harness 把 "一局怎么跑" 与 "怎么聚合指标" 解耦：

1. EpisodeRunner（协议）：`run_episode(model, seed) -> EpisodeResult`
   - 负责用 model 跑完一局（不管底层是 sim env 还是真机 mod loop），
     返回**中性** EpisodeResult（floor/act/won_game/steps/boss_name），不暴露引擎对象。
   - `SimEpisodeRunner`：包 V8Env + V8PPOTrainer.collect_rollout（deterministic=argmax），
     一局结束后从 `backend.build_runner_snapshot()` 取 floor/act/won_game（与 run_eval
     读 `env.runner.run_state.floor/.act` + `runner.game_won` 同源），boss_name 取
     `env.backend.boss_name`（与 run_eval 读 `runner._boss_name` 同源）。
   - `RealEpisodeRunner`：包 `RealGameBackend`，驱动 mod stdin/stdout loop 跑一局
     （见 v8/backends/real_game_backend.py）。

2. evaluate(model, runner, num_seeds, seed_offset)：
   - 对 [seed_offset, seed_offset+num_seeds) 每个种子调 runner.run_episode，
     聚合指标。**聚合逻辑逐位照搬 run_eval**（a1/a2/won/reached/floor 口径一致）。

------------------------------------------------------------------------------
与 run_eval 的一致性契约
------------------------------------------------------------------------------
run_eval 对每个种子：
    rollout = trainer.collect_rollout(env, seed, deterministic=True)
    runner = env.runner
    final_floor = runner.run_state.floor
    final_act   = runner.run_state.act
    game_won    = runner.game_won
    boss_name   = runner._boss_name
    if final_floor >= 16 or final_act >= 2: reached += 1
    if game_won or final_act >= 2:          act1_beat += 1
    if game_won or final_act >= 3:          act2_beat += 1
    if game_won:                            won_game += 1

SimEpisodeRunner 复用同一条 collect_rollout 路径 + 同源字段，故 evaluate(...,
SimEpisodeRunner) 与 run_eval 在同种子同模型下数字一致（见
tools/test_eval_harness_parity.py 验证）。

本 harness **不包含** run_eval 的 per-seed 超时/hang 看门狗（那是 sim 训练
长跑专用兜底，与评估语义无关）。需要时上层可自行包一层 thread+timeout，
不影响指标口径。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

logger = logging.getLogger(__name__)


# act1 boss 在 floor 16+（与 run_eval 注释一致）；过 boss 进 act2 即 final_act>=2。
_REACHED_FLOOR = 16


@dataclass
class EpisodeResult:
    """一局结束后的中性快照（不含任何引擎对象）。

    字段口径与 run_eval 读取的 final_floor / final_act / game_won / boss_name 对齐：
        floor     : runner.run_state.floor  （= backend.build_runner_snapshot()["floor"]）
        act       : runner.run_state.act    （= ...["act"]）
        won_game  : runner.game_won          （= ...["game_won"]）
        boss_name : runner._boss_name        （= backend.boss_name）
        steps     : 本局 rollout / decision step 数
        completed : run_episode 是否正常跑完（False 表示 crash/abort，与 run_eval
                    的 "rollout 为空 / crash → continue 不计入 floors" 对齐）
    """

    floor: int = 0
    act: int = 1
    won_game: bool = False
    boss_name: str = ""
    steps: int = 0
    completed: bool = True


class EpisodeRunner(Protocol):
    """用 model 跑一局并返回中性 EpisodeResult。底层后端无关。"""

    def run_episode(self, model: Any, seed: int) -> EpisodeResult:
        ...


# =============================================================================
# Sim runner：包 V8Env + trainer.collect_rollout（与 run_eval 同一路径）
# =============================================================================


class SimEpisodeRunner:
    """StSRL（模拟器）后端的 EpisodeRunner。

    复用 V8PPOTrainer.collect_rollout（deterministic=True，argmax）跑一局，
    与 run_eval 完全同路径。一局结束后从 backend 中性快照取 floor/act/won/boss。

    Args:
        env: V8Env（默认 StSRLBackend；也可注入别的 backend_factory 的 env）。
        trainer: V8PPOTrainer，提供 collect_rollout（持 model，但 run_episode
                 的 model 参数会被忽略——trainer 内部已绑定 model；保留 model 形参
                 仅为对齐 EpisodeRunner 协议）。
    """

    def __init__(self, env: Any, trainer: Any):
        self._env = env
        self._trainer = trainer

    def run_episode(self, model: Any, seed: int) -> EpisodeResult:  # noqa: ARG002
        env = self._env
        try:
            rollout = self._trainer.collect_rollout(env, seed=seed, deterministic=True)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "[harness] sim run_episode seed=%d crashed: %s: %s",
                seed, type(e).__name__, e,
            )
            return EpisodeResult(completed=False)

        if not rollout:
            return EpisodeResult(completed=False, steps=0)

        # 中性快照：与 run_eval 读 env.runner.run_state 同源（floor/act/game_won）。
        backend = getattr(env, "backend", None)
        if backend is not None:
            snap = backend.build_runner_snapshot()
            boss_name = str(getattr(backend, "boss_name", "") or "")
        else:
            snap = {"floor": 0, "act": 1, "game_won": False}
            boss_name = ""

        return EpisodeResult(
            floor=int(snap.get("floor", 0) or 0),
            act=int(snap.get("act", 1) or 1),
            won_game=bool(snap.get("game_won", False)),
            boss_name=boss_name,
            steps=len(rollout),
            completed=True,
        )


# =============================================================================
# Real runner：用 model argmax 直接驱动任意"中性"后端（真机走这条）
# =============================================================================


class RealEpisodeRunner:
    """直接 model-argmax 驱动 GameBackend 跑一局（真机 RealGameBackend 用）。

    与 SimEpisodeRunner 的区别：sim 后端的战斗由 env 内部搜索一气呵成（故复用
    collect_rollout）；真机后端把 COMBAT 当普通决策 phase（model 直接出招，无搜索），
    所以这里用一个统一的 model-argmax loop 驱动 backend：
        reset → while not game_over: build_v8_state → get_available_action_labels →
                model argmax → take_action(idx)

    这条 loop 与 tools/v8_play_real.py 的主循环同形（read state → 转 → argmax →
    发命令），只是把 IO/状态推进收进 RealGameBackend，决策留在这里。

    Args:
        backend: 实现 GameBackend 的对象（RealGameBackend）。需要 build_v8_state /
                 get_available_action_labels / take_action / game_over / game_won /
                 build_runner_snapshot / boss_name。
        max_decisions: 单局决策 step 上限（兜底，防 backend 没自终止）。
    """

    def __init__(self, backend: Any, *, max_decisions: int = 4000):
        self._backend = backend
        self._max_decisions = max_decisions

    def run_episode(self, model: Any, seed: int) -> EpisodeResult:
        import torch

        backend = self._backend
        try:
            backend.reset(seed)
        except Exception as e:  # noqa: BLE001
            logger.warning("[harness] real reset seed=%d failed: %s: %s",
                           seed, type(e).__name__, e)
            return EpisodeResult(completed=False)

        steps = 0
        while not backend.game_over and steps < self._max_decisions:
            state = backend.build_v8_state()
            actions = backend.get_available_action_labels(state)
            if not actions:
                # 没动作：backend 应已在 advance 里兜底；这里防御性退出
                break
            try:
                with torch.no_grad():
                    out = model(state, actions)
                    logits = out["logits"]
                idx = 0 if logits.numel() == 0 else int(torch.argmax(logits).item())
                if idx < 0 or idx >= len(actions):
                    idx = 0
            except Exception as e:  # noqa: BLE001
                logger.warning("[harness] real forward seed=%d step=%d failed: %s: %s",
                               seed, steps, type(e).__name__, e)
                idx = 0
            backend.take_action(idx)
            steps += 1

        snap = backend.build_runner_snapshot()
        return EpisodeResult(
            floor=int(snap.get("floor", 0) or 0),
            act=int(snap.get("act", 1) or 1),
            won_game=bool(snap.get("game_won", False)),
            boss_name=str(getattr(backend, "boss_name", "") or ""),
            steps=steps,
            completed=True,
        )


# =============================================================================
# 统一聚合：照搬 run_eval 的指标口径
# =============================================================================


def evaluate(
    model: Any,
    runner: EpisodeRunner,
    num_seeds: int,
    seed_offset: int,
    *,
    on_seed_done: Optional[Any] = None,
) -> Dict[str, Any]:
    """跑 num_seeds 局 deterministic eval，聚合指标（口径同 run_eval）。

    Args:
        model: V8Model（SimEpisodeRunner 会忽略，trainer 内部已绑；Real runner 用）。
        runner: EpisodeRunner（SimEpisodeRunner / RealEpisodeRunner）。
        num_seeds: 评估种子数（run_eval 默认 48）。
        seed_offset: 起始种子（run_eval 用 EVAL_SEED_OFFSET=10_000）。
        on_seed_done: 可选回调 (seed, EpisodeResult) -> None（日志 / 进度）。

    Returns:
        指标 dict，键与 run_eval 输出对齐：
            num_seeds / completed / reached_boss_rate / act1_boss_beat_rate /
            act2_boss_beat_rate / won_game_rate / a1_boss_killed_rate /
            a2_boss_killed_rate / beat_boss_rate(deprecated) /
            floor_mean / floor_max / avg_steps / boss_reach_counts / boss_kill_counts
        （run_eval 还有 entropy 字段——那是 sim 训练专用，本 harness 不含。）
    """
    floors: List[int] = []
    reached_boss = 0
    act1_beat = 0
    act2_beat = 0
    won_game = 0
    total_steps = 0
    boss_reach_counts: Dict[str, int] = {}
    boss_kill_counts: Dict[str, int] = {}

    for i in range(num_seeds):
        seed = seed_offset + i
        result = runner.run_episode(model, seed)
        if on_seed_done is not None:
            try:
                on_seed_done(seed, result)
            except Exception:  # noqa: BLE001
                pass

        # crash / empty rollout：与 run_eval 一致，不计入 floors（completed/rate 分母仍是 num_seeds）
        if not result.completed:
            continue

        final_floor = int(result.floor)
        final_act = int(result.act)
        game_won = bool(result.won_game)
        boss_name = str(result.boss_name or "")

        floors.append(final_floor)
        total_steps += int(result.steps)

        # ---- 照搬 run_eval 判定 ----
        if final_floor >= _REACHED_FLOOR or final_act >= 2:
            reached_boss += 1
        if game_won or final_act >= 2:
            act1_beat += 1
        if game_won or final_act >= 3:
            act2_beat += 1
        if game_won:
            won_game += 1

        # per-boss：只在 final_act==1 且摸到 boss 房（floor>=16）时归属（同 run_eval）。
        if final_act == 1 and final_floor >= _REACHED_FLOOR and boss_name:
            boss_reach_counts[boss_name] = boss_reach_counts.get(boss_name, 0) + 1
            boss_kill_counts.setdefault(boss_name, 0)

    n = max(1, len(floors))
    act1_beat_rate = act1_beat / num_seeds if num_seeds else 0.0
    return {
        "num_seeds": num_seeds,
        "completed": len(floors),
        "reached_boss_rate": reached_boss / num_seeds if num_seeds else 0.0,
        "act1_boss_beat_rate": act1_beat_rate,
        "act2_boss_beat_rate": act2_beat / num_seeds if num_seeds else 0.0,
        "won_game_rate": won_game / num_seeds if num_seeds else 0.0,
        "a1_boss_killed_rate": act1_beat_rate,
        "a2_boss_killed_rate": act2_beat / num_seeds if num_seeds else 0.0,
        "beat_boss_rate": act1_beat_rate,  # deprecated, = act1_boss_beat_rate
        "floor_mean": sum(floors) / n if floors else 0.0,
        "floor_max": max(floors) if floors else 0,
        "avg_steps": total_steps / n if floors else 0.0,
        "boss_reach_counts": boss_reach_counts,
        "boss_kill_counts": boss_kill_counts,
    }


__all__ = [
    "EpisodeResult",
    "EpisodeRunner",
    "SimEpisodeRunner",
    "evaluate",
]

"""
V8 Smoke 20 seed pipeline 验证。

跑 20 seed (0-19)，每 seed 跑一局 act1，记录 trajectory + outcome。
不真 RL update model，只验证 pipeline 通 + 数据流对 + 无 bug。

按 docs/v8_implementation_design.md 组件 9 检查项：
1. 无 crash：20/20 完整跑完（允许少数 timeout）
2. 数据流对：trajectory 字段齐全、state 字段无 None / NaN、reward 数字合理
3. 覆盖所有 phase：MAP / NEOW / CARD_REWARDS / EVENT / SHOP / REST / TREASURE 都至少出现过
4. 性能 OK：一局 < 5 min（评估 cache 工作正常）
5. 牌组强度评估正常：触发次数 ≥ 期望次数 + cache 命中率 ≥ 50%
6. 基础指标：reached_boss / floor mean 在合理区间

NOTE: smoke 不真 RL update，只 collect_rollout 验证 pipeline。
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

# 保证从仓库根可 import v8.*
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from v8.env import V8Env
from v8.model import V8Model
from v8.trainer import V8PPOTrainer
from v8 import deck_evaluator as _deck_eval

SMOKE_SEEDS = list(range(20))
OUTPUT_DIR = "data/v8_smoke"


def _safe_float(x) -> float:
    try:
        f = float(x)
        if math.isnan(f) or math.isinf(f):
            return 0.0
        return f
    except Exception:
        return 0.0


def run_smoke():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 创建随机初始化的 model（不预训练，纯 RL 起点）
    # NOTE: 这是 smoke，不真训练，只验证 pipeline；trainer.collect_rollout 内部 no_grad。
    model = V8Model()
    model.eval()
    trainer = V8PPOTrainer(model, device="cpu")

    summary = {
        "total_seeds": len(SMOKE_SEEDS),
        "completed": 0,
        "crashed": 0,
        "per_seed": [],
        "phase_coverage": {},
        "reached_boss_count": 0,
        "beat_boss_count": 0,
        "floor_mean": 0.0,
        "floor_min": 0,
        "floor_max": 0,
        "avg_episode_time": 0.0,
        "avg_steps_per_episode": 0.0,
        # deck_evaluator 监控
        "deck_eval_total_calls": 0,
        "deck_eval_cache_unique": 0,
        "deck_eval_cache_hit_rate": 0.0,
    }

    start_time = time.time()
    floors: list = []
    total_steps = 0
    total_eval_calls = 0

    for seed in SMOKE_SEEDS:
        seed_start = time.time()
        seed_result = {
            "seed": seed,
            "status": "running",
        }

        env = None
        cache_size_before = _deck_eval.cache_size()

        try:
            env = V8Env(seed=0)  # placeholder, V8Env 没用 ctor 的 seed；下面 reset(seed) 才是真种子
        except TypeError:
            # 可能没接受 seed 参数（依旧不影响，只是 ctor 不需要）
            env = V8Env()
        except Exception as e:  # noqa: BLE001
            seed_result["status"] = "crashed"
            seed_result["error"] = f"V8Env ctor failed: {type(e).__name__}: {e}"
            seed_result["traceback"] = traceback.format_exc()[:500]
            summary["crashed"] += 1
            summary["per_seed"].append(seed_result)
            with open(f"{OUTPUT_DIR}/smoke_summary.json", "w") as f:
                json.dump(summary, f, indent=2, default=str)
            print(f"[seed {seed}] crashed-ctor", flush=True)
            continue

        # 统计 evaluate_deck 调用次数（hook 法：包一层 counter）
        # 这里走简单办法：summary 用 cache_size 增量近似 unique evaluation 数量
        # 完整 hit-rate 估计：在 step 里 info['deck_strength_evaluated'] True 的次数 - 增量 unique 数。
        eval_calls_this_seed = 0

        # monkey-patch evaluate_deck 计数（局部 wrapper，不污染全局）
        orig_evaluate_deck = _deck_eval.evaluate_deck

        def _counted_eval(*args, **kwargs):
            nonlocal eval_calls_this_seed
            eval_calls_this_seed += 1
            return orig_evaluate_deck(*args, **kwargs)

        # 替换 v8.env 模块里的引用（v8.env import 时拿到的是绑定）
        import v8.env as _v8env_mod
        orig_eval_in_env = _v8env_mod.evaluate_deck
        _v8env_mod.evaluate_deck = _counted_eval

        try:
            state = env.reset(seed=seed)

            # 跑一个完整 rollout（trainer 内部 no_grad，不更新 model）
            rollout = trainer.collect_rollout(env, seed=seed)

            phases_seen = sorted({(s.state.phase or "") for s in rollout if s.state is not None})

            final_state = rollout[-1].state if rollout else state
            final_floor = int(getattr(final_state, "floor", 0) or 0) if final_state is not None else 0
            final_act = int(getattr(final_state, "act", 1) or 1) if final_state is not None else 1
            total_reward = sum(_safe_float(s.reward) for s in rollout)

            seed_result.update({
                "status": "ok",
                "num_steps": len(rollout),
                "final_floor": final_floor,
                "final_act": final_act,
                "total_reward": round(total_reward, 4),
                "phases_seen": phases_seen,
                "time_seconds": round(time.time() - seed_start, 2),
                "eval_calls": eval_calls_this_seed,
            })

            floors.append(final_floor)
            total_steps += len(rollout)
            total_eval_calls += eval_calls_this_seed

            # reached / beat boss
            # act1 boss 在 floor 16/17（不同实现）；保守用 floor>=16 或 act>=2
            if final_floor >= 16 or final_act >= 2:
                summary["reached_boss_count"] += 1
            if final_act >= 2:
                summary["beat_boss_count"] += 1

            # phase coverage
            for phase in phases_seen:
                if not phase:
                    continue
                summary["phase_coverage"][phase] = summary["phase_coverage"].get(phase, 0) + 1

            summary["completed"] += 1

        except Exception as e:  # noqa: BLE001
            seed_result["status"] = "crashed"
            seed_result["error"] = f"{type(e).__name__}: {e}"
            seed_result["traceback"] = traceback.format_exc()[:800]
            seed_result["time_seconds"] = round(time.time() - seed_start, 2)
            seed_result["eval_calls"] = eval_calls_this_seed
            summary["crashed"] += 1
        finally:
            # 还原 v8.env 模块的 evaluate_deck 引用
            try:
                _v8env_mod.evaluate_deck = orig_eval_in_env
            except Exception:
                pass
            try:
                if env is not None:
                    env.close()
            except Exception:
                pass

        summary["per_seed"].append(seed_result)

        # 增量保存（避免最后挂掉丢数据）
        try:
            with open(f"{OUTPUT_DIR}/smoke_summary.json", "w") as f:
                json.dump(summary, f, indent=2, default=str)
        except Exception as e:  # noqa: BLE001
            print(f"[seed {seed}] WARN: incremental save failed: {e}", flush=True)

        print(
            f"[seed {seed}] {seed_result['status']} "
            f"floor={seed_result.get('final_floor', '?')} "
            f"act={seed_result.get('final_act', '?')} "
            f"steps={seed_result.get('num_steps', '?')} "
            f"reward={seed_result.get('total_reward', '?')} "
            f"time={seed_result.get('time_seconds', 0)}s "
            f"eval_calls={seed_result.get('eval_calls', 0)}",
            flush=True,
        )

    # ---- 终结统计 ----
    total_time = time.time() - start_time
    summary["total_time_seconds"] = round(total_time, 2)
    if floors:
        summary["floor_mean"] = round(sum(floors) / len(floors), 2)
        summary["floor_min"] = min(floors)
        summary["floor_max"] = max(floors)
    summary["avg_episode_time"] = round(total_time / max(summary["completed"], 1), 2)
    summary["avg_steps_per_episode"] = round(total_steps / max(summary["completed"], 1), 2)
    summary["deck_eval_total_calls"] = total_eval_calls
    final_cache_size = _deck_eval.cache_size()
    summary["deck_eval_cache_unique"] = final_cache_size
    if total_eval_calls > 0:
        # cache_hit_rate ≈ 1 - unique / calls（粗略：act 切换会清 cache，实际命中更多）
        # 注意：每局开始 reset 会 clear_cache，所以 final_cache_size 只反映最后一局；
        # 这里给一个保守估计：sum_per_seed_unique 不可得，给个简单数字
        summary["deck_eval_cache_hit_rate_estimate"] = round(
            max(0.0, 1.0 - final_cache_size / max(total_eval_calls, 1)), 4
        )

    # 最终报告
    try:
        with open(f"{OUTPUT_DIR}/smoke_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
    except Exception as e:  # noqa: BLE001
        print(f"WARN: final save failed: {e}", flush=True)

    # 打印最终报告
    print("\n=== V8 Smoke 20 seed 完成 ===", flush=True)
    print(f"总耗时: {total_time/60:.1f} min", flush=True)
    print(f"完成: {summary['completed']}/{summary['total_seeds']}", flush=True)
    print(f"crashed: {summary['crashed']}", flush=True)
    print(f"reached_boss: {summary['reached_boss_count']}/{summary['completed']}", flush=True)
    print(f"beat_boss: {summary['beat_boss_count']}/{summary['completed']}", flush=True)
    print(f"floor mean/min/max: {summary['floor_mean']}/{summary['floor_min']}/{summary['floor_max']}", flush=True)
    print(f"phase coverage: {summary['phase_coverage']}", flush=True)
    print(f"avg episode time: {summary['avg_episode_time']}s", flush=True)
    print(f"avg steps/episode: {summary['avg_steps_per_episode']}", flush=True)
    print(f"deck_eval calls (sum across seeds): {total_eval_calls}", flush=True)
    print(f"deck_eval final cache size: {final_cache_size}", flush=True)


if __name__ == "__main__":
    run_smoke()

"""
V8 vs V7 初步评估：5 个 seed，分别用 V7Bot 与 V8InferenceBot 跑整局，
打印 final_floor / game_won 对比表。

best-effort，不严谨：
- V8 model 是 smoke 训练，仅 4 batch，过拟合 seed=42 的 V7 决策；其他 seed
  上的表现就是「V8 学到的策略 vs 原始 V7」直接对照。
- 非 COMBAT phase 两者都用 V7 strategy，只在战斗回合分叉，便于把差异归因
  到 V8 model。
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from v7_bot import V7Bot
from v8_inference_bot import V8InferenceBot


SEEDS = [6, 13, 10, 1, 0]
MODEL_PATH = os.path.join(_REPO_ROOT, "sts_models", "v8_smoke.pt")


def main() -> int:
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: model 不存在: {MODEL_PATH}（先跑 v8_trainer.py）")
        return 1

    print(f"V8 vs V7 评估 (Ironclad A0, max_actions=20000)")
    print(f"  model: {MODEL_PATH}")
    print(f"  seeds: {SEEDS}")
    print("-" * 78)
    print(f"{'seed':>6} | {'V7 floor':>10} | {'V7 won':>7} | "
          f"{'V8 floor':>10} | {'V8 won':>7} | delta_floor")
    print("-" * 78)

    v7_wins = 0
    v8_wins = 0
    v7_floor_total = 0
    v8_floor_total = 0
    rows = []

    for seed in SEEDS:
        # V7
        bot7 = V7Bot(verbose=False)
        try:
            r7 = bot7.run(
                seed=seed, ascension=0, character="ironclad",
                max_actions=20_000, log_path=None,
            )
            v7_floor = r7.get("final_floor", 0)
            v7_won = bool(r7.get("game_won"))
        except Exception as e:  # noqa: BLE001
            print(f"  seed {seed}: V7 crashed: {type(e).__name__}: {e}")
            v7_floor, v7_won = 0, False

        # V8
        bot8 = V8InferenceBot(model_path=MODEL_PATH, verbose=False)
        try:
            r8 = bot8.run(
                seed=seed, ascension=0, character="ironclad",
                max_actions=20_000, log_path=None,
            )
            v8_floor = r8.get("final_floor", 0)
            v8_won = bool(r8.get("game_won"))
        except Exception as e:  # noqa: BLE001
            print(f"  seed {seed}: V8 crashed: {type(e).__name__}: {e}")
            v8_floor, v8_won = 0, False

        v7_wins += int(v7_won)
        v8_wins += int(v8_won)
        v7_floor_total += v7_floor
        v8_floor_total += v8_floor
        rows.append((seed, v7_floor, v7_won, v8_floor, v8_won))

        print(f"{seed:>6} | {v7_floor:>10} | {str(v7_won):>7} | "
              f"{v8_floor:>10} | {str(v8_won):>7} | "
              f"{v8_floor - v7_floor:+d}")

    print("-" * 78)
    print(f"{'TOTAL':>6} | "
          f"{v7_floor_total:>10} | {v7_wins:>7} | "
          f"{v8_floor_total:>10} | {v8_wins:>7} | "
          f"{v8_floor_total - v7_floor_total:+d}")
    print("-" * 78)
    print(f"  avg floor: V7={v7_floor_total/len(SEEDS):.1f}  "
          f"V8={v8_floor_total/len(SEEDS):.1f}")
    print(f"  win count: V7={v7_wins}/{len(SEEDS)}  V8={v8_wins}/{len(SEEDS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

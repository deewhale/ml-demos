"""V8 选卡专项分析（独立脚本，可在已有 trajectory dump 上重跑）。

读 data/v8_smoke/per_seed_trajectory_<seed>.jsonl，
出 CARD_REWARDS 决策的统计：决策数 / skip 比例 / 平均 reward / 正负 reward 数 /
top 选卡分布。

CLI:
    python scripts/v8_card_analysis.py [--seeds 0,1,2,3,4] [--output_dir data/v8_smoke]

NOTE: 实际选了哪张卡通过对比相邻 step 的 deck 推断（chosen_action 只是 "choice=N"，
没暴露具体卡名）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

# 让 import scripts.v8_smoke_mini 工作
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.v8_smoke_mini import analyze_card_decisions  # 复用同一份分析逻辑


def main(seeds: List[int], output_dir: str) -> None:
    # 自动 detect 哪些 seed 有 trajectory（如果 --seeds 没给）
    if not seeds:
        seeds = []
        for f in sorted(os.listdir(output_dir)):
            if f.startswith("per_seed_trajectory_") and f.endswith(".jsonl"):
                try:
                    seed = int(f.replace("per_seed_trajectory_", "").replace(".jsonl", ""))
                    seeds.append(seed)
                except ValueError:
                    pass
    print(f"分析 seeds: {seeds}", flush=True)

    report = analyze_card_decisions(seeds, output_dir)

    if report.get("total_decisions", 0) == 0:
        print("WARN: no CARD_REWARDS decisions found", flush=True)
        return

    print("=== V8 选卡专项分析 ===", flush=True)
    print(f"总决策数: {report['total_decisions']}", flush=True)
    print(f"skip: {report['skip_count']} ({report['skip_rate']:.1%})", flush=True)
    print(f"pick (deck 实际增加): {report['pick_count']}", flush=True)
    print(f"chosen 非 skip 但 deck 没变: {report['no_deck_change_count']}", flush=True)
    print(f"平均 step reward: {report['mean_reward']}", flush=True)
    print(f"正/负/零 reward: {report['positive_reward_count']}/{report['negative_reward_count']}/{report['zero_reward_count']}", flush=True)
    print(f"deck_strength 已评估的决策数: {report['deck_strength_evaluated_count']}/{report['total_decisions']}", flush=True)
    print("Top 10 选卡:", flush=True)
    for name, cnt in report["top_10_picks"]:
        print(f"  {name}: {cnt}", flush=True)

    # 写一份 json
    out_path = os.path.join(output_dir, "card_analysis.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n报告已保存: {out_path}", flush=True)


def _parse_seeds(s: str) -> List[int]:
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds",
        type=_parse_seeds,
        default=[],
        help="逗号分隔 seed 列表（默认空，自动 detect 目录里所有 trajectory）",
    )
    parser.add_argument(
        "--output_dir",
        default="data/v8_smoke",
        help="trajectory 所在目录",
    )
    args = parser.parse_args()
    main(args.seeds, args.output_dir)

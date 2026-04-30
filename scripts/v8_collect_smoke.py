"""V8 smoke 采集：跑 1 局 ironclad seed=42，写 JSONL，打印统计。"""

from __future__ import annotations

import os
import sys

# 把 repo 根加进 sys.path，方便从 scripts/ 子目录 import
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from v8_data_collector import V8CollectorBot


def main() -> int:
    out_dir = os.path.join(_REPO_ROOT, "data", "v8_jsonl")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "smoke_seed42.jsonl")

    bot = V8CollectorBot(verbose=False, output_jsonl=out_path)
    result = bot.run(
        seed=42,
        ascension=0,
        character="ironclad",
        max_actions=20_000,
        log_path="/tmp/v8_collect.log",
    )

    print("=" * 60)
    print("V8 smoke collect 完成")
    print("=" * 60)
    print(f"  seed         : {result['seed']}")
    print(f"  character    : {result['character']}")
    print(f"  game_won     : {result['game_won']}")
    print(f"  final_floor  : {result['final_floor']}")
    print(f"  final_hp     : {result['final_hp']}/{result['max_hp']}")
    print(f"  decisions    : {result['decisions']}")
    print(f"  elapsed_s    : {result['elapsed_s']}")
    print(f"  jsonl_path   : {out_path}")
    print(f"  records_written       : {bot.records_written}")
    print(f"  bottled_error_count   : {bot.bottled_error_count}")
    print(f"  first_bottled_error   : {bot.first_bottled_error}")
    if result.get("error"):
        print(f"  game error   : {result['error']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

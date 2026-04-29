"""V7 smoke test: 跑 N 个 seeds，打印 floor / HP / 时间。"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import traceback

# 兼容历史 worktree 路径：可通过 V7_BOT_PATH 注入额外的 import 目录
_extra = os.environ.get("V7_BOT_PATH")
if _extra:
    sys.path.insert(0, _extra)

from v7_bot import V7Bot


DEFAULT_SEEDS = [42, 123, 7, 999, 2024, 31337, 8675309, 1, 2, 3]


def run_many(seeds, ascension: int = 0, verbose: bool = False, character: str = "watcher",
             log_dir: str = "."):
    bot = V7Bot(verbose=verbose)
    results = []
    for i, seed in enumerate(seeds):
        print(f"\n[{i+1}/{len(seeds)}] Seed={seed} (A{ascension}) {character.capitalize()} ...")
        t0 = time.monotonic()
        log_path = f"{log_dir}/v7_game_log_seed{seed}.jsonl"
        try:
            r = bot.run(seed=seed, ascension=ascension, character=character.lower(),
                        log_path=log_path)
            results.append(r)
            outcome = "WON" if r["game_won"] else "LOST"
            err = f"  [ERR: {r['error']}]" if r.get("error") else ""
            print(f"  -> {outcome}  floor={r['final_floor']}  hp={r['final_hp']}/{r['max_hp']}"
                  f"  deck={r['deck_size']}  time={r['elapsed_s']}s  decisions={r['decisions']}{err}")
        except Exception as e:  # noqa: BLE001
            print(f"  !! CRASH: {type(e).__name__}: {e}")
            traceback.print_exc()
            results.append({
                "seed": seed,
                "game_won": False,
                "final_floor": 0,
                "final_hp": 0,
                "max_hp": 72,
                "deck_size": 0,
                "elapsed_s": round(time.monotonic() - t0, 2),
                "error": f"CRASH: {e}",
            })

    print("\n" + "=" * 70)
    print(f"V7 smoke test — {len(seeds)} Watcher games @ A{ascension}")
    print("=" * 70)
    print(f"{'seed':>10}  {'floor':>5}  {'hp':>8}  {'won':>4}  {'time':>7}")
    for r in results:
        outcome = "Y" if r.get("game_won") else "."
        print(f"{str(r['seed']):>10}  {r['final_floor']:>5}  "
              f"{r.get('final_hp', 0):>3}/{r.get('max_hp', 72):<3}  "
              f"{outcome:>4}  {r.get('elapsed_s', 0):>6.1f}s")
    floors = [r["final_floor"] for r in results]
    wins = sum(1 for r in results if r.get("game_won"))
    floor16 = sum(1 for r in results if r["final_floor"] >= 17)  # Act 1 boss at floor 17 means passed F16
    print(f"\nWins:          {wins}/{len(results)}")
    print(f"Reached F>=17: {floor16}/{len(results)}  (Act 1 boss cleared)")
    if floors:
        print(f"Floor stats:   min={min(floors)}  avg={sum(floors)/len(floors):.1f}  max={max(floors)}")
    total_t = sum(r.get("elapsed_s", 0) for r in results)
    print(f"Total time:    {total_t:.1f}s  avg {total_t/max(len(results),1):.1f}s/game")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-games", "--n", dest="n_games", type=int, default=10,
                    help="number of seeds to run")
    ap.add_argument("--character", default="watcher")
    ap.add_argument("--ascension", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--seeds", nargs="+", type=int, default=None,
                    help="explicit seed list (overrides --seed-start)")
    ap.add_argument("--seed-start", type=int, default=None,
                    help="starting seed (generates seed-start, seed-start+1, ...)")
    ap.add_argument("--log-level", default="WARNING")
    ap.add_argument("--log-dir", default=".", help="directory for per-seed jsonl logs")
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.seeds:
        seeds = args.seeds
    elif args.seed_start is not None:
        seeds = [args.seed_start + i for i in range(args.n_games)]
    else:
        seeds = DEFAULT_SEEDS[: args.n_games]
    run_many(seeds, ascension=args.ascension, verbose=args.verbose, character=args.character,
             log_dir=args.log_dir)


if __name__ == "__main__":
    main()

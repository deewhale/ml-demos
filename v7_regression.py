"""V7 evaluator regression test.

改 evaluator 后跑 `python3 v7_regression.py` 看每个 test point 的 top-K action
和 score，快速验证不回退，不必每次跑 30 局（一次 30 min）。

方案：seed + floor + turn breakpoint。不 serialize CombatEngine（它带
RNG / 闭包 / 内部 ref，pickle 不稳定），而是用 V7Bot 跑到指定断点，在
那一刻把 `adapter.last_solver_scores` 拿出来做对比。

用法：
    # baseline 记录
    python3 v7_regression.py --record baseline.json

    # 改完 evaluator
    python3 v7_regression.py --compare baseline.json

    # 仅打印
    python3 v7_regression.py
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

sys.path.insert(0, "REPO_ROOT/")
sys.path.insert(0, "STSRLSOLVER_PATH")

from v7_bot import V7Bot, SOLVER_BUDGETS
from v7_evaluator import patch_turn_solver_eval
from packages.engine.game import GamePhase

# --------------------------------------------------------------------------
# 10 Test points: (name, seed, target_floor, turn_into_combat, expected_hint)
# turn_into_combat = 第几次进入该 floor 的 combat decision (0 = 第一个 decision)
# --------------------------------------------------------------------------

TEST_POINTS: List[Dict[str, Any]] = [
    {
        "name": "seed6_F5_SpikeSlimeL_T0",
        "seed": 6, "floor": 5, "turn": 0,
        "expected": "attack pressure over pure block (SpikeSlime_L 64 HP)",
    },
    {
        "name": "seed6_F1_JawWorm_T0",
        "seed": 6, "floor": 1, "turn": 0,
        "expected": "open F1 easy combat, vuln + big attack",
    },
    {
        "name": "seed13_F4_FiveSlimes_T0",
        "seed": 13, "floor": 4, "turn": 0,
        "expected": "5 small slimes, AoE opportunity",
    },
    {
        "name": "seed10_F6_3xSentry_T0",
        "seed": 10, "floor": 6, "turn": 0,
        "expected": "3 Sentry elite, kill one fast to reduce DPS",
    },
    {
        "name": "seed13_F6_Lagavulin_T0",
        "seed": 13, "floor": 6, "turn": 0,
        "expected": "sleeping Lagavulin, rush dmg while asleep",
    },
    {
        "name": "seed1_F6_GremlinNob_T0",
        "seed": 1, "floor": 6, "turn": 0,
        "expected": "Nob punishes skills; favor attacks, avoid skill spam",
    },
    {
        "name": "seed0_F1_Cultist_T0",
        "seed": 0, "floor": 1, "turn": 0,
        "expected": "F1 Cultist; kill before ritual stacks",
    },
    {
        "name": "seed3_F5_SlaverRed_T0",
        "seed": 3, "floor": 5, "turn": 0,
        "expected": "SlaverRed, entangle threat — kill fast",
    },
    {
        "name": "seed13_F8_Gremlins_T0",
        "seed": 13, "floor": 8, "turn": 0,
        "expected": "4 Gremlins, AoE or sequential kill",
    },
    {
        "name": "seed15_F13_Lagavulin_T0",
        "seed": 15, "floor": 13, "turn": 0,
        "expected": "Lagavulin elite after Act 1 progression",
    },
]


# --------------------------------------------------------------------------
# Core: run V7Bot and intercept at breakpoints
# --------------------------------------------------------------------------

class _Breakpoint(BaseException):
    """向上抛出中断信号（继承 BaseException 以绕过 bot 的 except Exception）."""


def run_to_breakpoint(seed: int, target_floor: int, turn_in_combat: int,
                      max_actions: int = 20_000) -> Dict[str, Any]:
    """跑 V7Bot 到 seed=seed, floor=target_floor 的 combat 第 turn_in_combat 个
    decision；在那一刻捕获 adapter.last_solver_scores 和 engine 状态摘要。

    返回 {"scores": [...], "engine_summary": {...}, "chosen": str}
    如果到不了断点（seed 先死），返回 {"error": "..."}
    """
    patch_turn_solver_eval()

    # 用低 budget 加速（断点场景不需要最佳解，快到断点就行）
    # 但断点那一刻的 solver 仍用完整 budget，保证可对比
    fast_budgets = {
        "monster": (30.0, 3_000, 1_000),
        "elite":   (300.0, 15_000, 1_500),
        "boss":    (1_000.0, 30_000, 2_000),
    }
    bot = V7Bot(solver_budgets=fast_budgets, verbose=False)

    # 为断点场景恢复完整 budget
    _rt_cap = {"monster": 2000, "elite": 2000, "boss": 3000}

    # Monkey patch V7Bot._pick_action
    orig_pick = bot._pick_action
    state: Dict[str, Any] = {
        "combat_decisions": 0,  # 本场 combat 内第几个 decision (reset on new combat)
        "last_floor": -1,
        "last_room_type": None,
        "captured": None,
    }

    def patched_pick(runner, actions):
        nonlocal state
        rs = runner.run_state
        cur_floor = rs.floor
        cur_phase = runner.phase

        # 过目标楼层 → 早退
        if cur_floor > target_floor:
            state["captured"] = {"error": f"passed target F{target_floor} (now F{cur_floor}) without hitting combat breakpoint"}
            raise _Breakpoint("passed target floor")

        # 跨 floor 或离开 combat 都重置 combat_decisions
        if cur_floor != state["last_floor"] or cur_phase != GamePhase.COMBAT:
            state["combat_decisions"] = 0
        state["last_floor"] = cur_floor

        # 只在 combat 断点拦截
        if cur_phase == GamePhase.COMBAT and cur_floor == target_floor \
                and state["combat_decisions"] == turn_in_combat:
            # 让 bot 用完整 budget 跑一遍；调用 orig_pick 并抓 scores
            # 为此先把 budget 暂时拉到默认 SOLVER_BUDGETS
            try:
                bot._budgets = SOLVER_BUDGETS
                # 同步到 adapter 上会在 orig_pick 里生效
                action = orig_pick(runner, actions)
            finally:
                bot._budgets = fast_budgets

            scores = list(getattr(bot._adapter, "last_solver_scores", []) or [])
            engine = getattr(runner, "current_combat", None)
            esum: Dict[str, Any] = {}
            if engine is not None:
                s = engine.state
                esum = {
                    "turn": s.turn,
                    "energy": s.energy,
                    "hp": s.player.hp,
                    "block": s.player.block,
                    "hand": list(s.hand),
                    "stance": getattr(s, "stance", None),
                    "enemies": [
                        {"id": getattr(e, "id", ""), "hp": e.hp, "max_hp": e.max_hp,
                         "statuses": dict(getattr(e, "statuses", {}))}
                        for e in s.enemies
                    ],
                    "player_statuses": dict(getattr(s.player, "statuses", {})),
                }
            state["captured"] = {
                "scores": scores,
                "engine": esum,
                "chosen_action": repr(action),
            }
            raise _Breakpoint("breakpoint hit")

        if cur_phase == GamePhase.COMBAT:
            state["combat_decisions"] += 1

        return orig_pick(runner, actions)

    bot._pick_action = patched_pick  # type: ignore

    try:
        bot.run(seed=seed, max_actions=max_actions)
        return {"error": f"did not reach F{target_floor} turn{turn_in_combat}"}
    except _Breakpoint:
        return state["captured"] or {"error": "breakpoint hit but no capture"}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def summarize(point: Dict[str, Any], result: Dict[str, Any], top_k: int = 5) -> str:
    lines = []
    lines.append(f"=== {point['name']} ===")
    lines.append(f"  seed={point['seed']} F{point['floor']} turn={point['turn']}")
    lines.append(f"  expected: {point['expected']}")
    if "error" in result:
        lines.append(f"  ERROR: {result['error']}")
        return "\n".join(lines)

    esum = result.get("engine", {})
    enemies = esum.get("enemies", [])
    enemy_str = ", ".join(f"{e['id']}({e['hp']}/{e['max_hp']})" for e in enemies)
    lines.append(f"  state: hp={esum.get('hp')} block={esum.get('block')} energy={esum.get('energy')} stance={esum.get('stance')}")
    lines.append(f"  hand: {esum.get('hand')}")
    lines.append(f"  enemies: {enemy_str}")

    scores = result.get("scores", [])
    # 按 score 降序排序
    scores_sorted = sorted(scores, key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0.0, reverse=True)
    lines.append(f"  top-{min(top_k, len(scores_sorted))} plans:")
    for desc, sc in scores_sorted[:top_k]:
        lines.append(f"    {sc:>12.2f}  {desc}")
    lines.append(f"  chosen: {result.get('chosen_action', '?')[:120]}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", type=str, help="save results to JSON for baseline")
    parser.add_argument("--compare", type=str, help="load baseline JSON and diff")
    parser.add_argument("--only", type=str, help="只跑 name 包含此子串的 test point")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    points = [p for p in TEST_POINTS if not args.only or args.only in p["name"]]
    all_results: Dict[str, Any] = {}
    t_start = time.monotonic()
    for i, p in enumerate(points):
        if not args.quiet:
            print(f"[{i+1}/{len(points)}] running {p['name']}...", flush=True)
        t0 = time.monotonic()
        res = run_to_breakpoint(p["seed"], p["floor"], p["turn"])
        res["_elapsed_s"] = round(time.monotonic() - t0, 2)
        all_results[p["name"]] = res
        if not args.quiet:
            print(summarize(p, res, top_k=args.top_k))
            print()

    print(f"total elapsed: {time.monotonic() - t_start:.1f}s")

    if args.record:
        with open(args.record, "w") as f:
            json.dump({"points": points, "results": all_results}, f, indent=2, default=str)
        print(f"saved baseline to {args.record}")

    if args.compare:
        with open(args.compare) as f:
            base = json.load(f)
        base_res = base.get("results", {})
        print("\n=== DIFF vs baseline ===")
        for p in points:
            name = p["name"]
            cur = all_results.get(name, {})
            old = base_res.get(name, {})
            cur_top = _top_desc(cur)
            old_top = _top_desc(old)
            flag = "SAME" if cur_top == old_top else "DIFF"
            print(f"  [{flag}] {name}")
            if cur_top != old_top:
                print(f"     OLD top: {old_top}")
                print(f"     NEW top: {cur_top}")


def _top_desc(result: Dict[str, Any]) -> str:
    scores = result.get("scores", [])
    if not scores:
        return "(none)"
    scores_sorted = sorted(scores, key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0.0, reverse=True)
    desc, sc = scores_sorted[0]
    return f"{sc:.1f} {desc}"


if __name__ == "__main__":
    main()

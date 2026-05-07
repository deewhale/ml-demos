"""V8 Round 2 smoke：验证 V8CombatNetWrapper 真被 search 调用。

设计原则对照（docs/v8_design_principles.md 第 2 点）：
- 战斗内深度搜索 + 模型联合
- 本 smoke 验证：跑 V8Env 一段 episode（含战斗），检查 wrapper.call_count > 0

期望：call_count > 0（说明 search leaf evaluator 真触达 model）
失败信号：call_count == 0（hook 没接通，需 debug TurnSolverAdapter combat_net 路径）

Usage:
    .venv/bin/python scripts/v8_round2_smoke.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# 让 v8 / sts_paths 可 import（脚本一般从 repo root 跑）
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from v8.combat_net_wrapper import V8CombatNetWrapper
from v8.env import V8Env
from v8.model import V8Model


def main() -> int:
    logging.basicConfig(level=logging.WARNING)

    print("=" * 60)
    print("V8 Round 2 smoke: model in combat search")
    print("=" * 60)

    print("[1/4] Instantiate V8Model...")
    model = V8Model()
    model.eval()

    print("[2/4] Wrap with V8CombatNetWrapper...")
    wrapper = V8CombatNetWrapper(model)
    print(f"      initial call_count = {wrapper.call_count}")

    print("[3/4] Build V8Env(combat_net_wrapper=wrapper) and reset...")
    env = V8Env(seed=0, combat_net_wrapper=wrapper) if False else V8Env(
        combat_net_wrapper=wrapper,
    )
    state = env.reset(seed=0)
    print(f"      initial state phase={state.phase} floor={state.floor}")

    print("[4/4] Step env up to 20 times (含战斗，看 wrapper 是否被调)...")
    n_steps = 0
    max_steps = 20
    for i in range(max_steps):
        actions = env.get_available_actions()
        if not actions:
            print(f"      step {i}: no actions, break")
            break
        try:
            next_state, reward, done, info = env.step(0)
        except Exception as e:  # noqa: BLE001
            print(f"      step {i}: exception {type(e).__name__}: {e}")
            break
        n_steps += 1
        if i < 5 or info.get("battle_happened"):
            print(
                f"      step {i}: phase {info.get('phase_before')} -> "
                f"{info.get('phase_after')} battle={info.get('battle_happened')} "
                f"reward={reward:.3f}"
            )
        if done:
            print(f"      step {i}: episode done")
            break

    env.close()

    print("=" * 60)
    print(f"steps_run = {n_steps}")
    print(f"wrapper.call_count = {wrapper.call_count}")
    print(f"wrapper.avg_value  = {wrapper.avg_value:.4f}")
    print("=" * 60)

    if wrapper.call_count > 0:
        print("PASS: combat_net hook is connected (search invoked predict).")
        return 0
    else:
        print("FAIL: combat_net hook NOT triggered. Debug TurnSolverAdapter wiring.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

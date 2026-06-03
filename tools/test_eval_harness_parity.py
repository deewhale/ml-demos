"""eval_harness 与 v8_ppo_train.run_eval 的一致性验证（硬性）。

证明统一评估 harness（v8/eval_harness.py）在 StSRLBackend 上跑出的指标，与现有
训练入口的 run_eval **完全一致**（同种子同模型 → 同数字），即 harness 没改变评估语义。

跑法（CPU，避免和训练抢 MPS）：
    .venv/bin/python tools/test_eval_harness_parity.py [--num_seeds 3] [--device cpu]

对比字段：reached_boss_rate / act1_boss_beat_rate / act2_boss_beat_rate /
won_game_rate / floor_mean / floor_max / avg_steps / boss_reach_counts。
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

_REPO = str(Path(__file__).resolve().parent.parent)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import torch  # noqa: E402

from v8.combat_net_wrapper import V8CombatNetWrapper  # noqa: E402
from v8.env import V8Env  # noqa: E402
from v8.eval_harness import SimEpisodeRunner, evaluate  # noqa: E402
from v8.model import V8Model  # noqa: E402
from v8.trainer import V8PPOTrainer  # noqa: E402
from tools.v8_ppo_train import load_combat_head, run_eval  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(message)s")

_COMBAT_HEAD = "sts_models/v8_combat_head_v1.pt"
_CKPT = "sts_models/v8_ppo_batch_v39/v8_ppo_final.pt"
_SEED_OFFSET = 10_000

# 对比的 rate / scalar 字段
_RATE_KEYS = [
    "reached_boss_rate",
    "act1_boss_beat_rate",
    "act2_boss_beat_rate",
    "won_game_rate",
    "a1_boss_killed_rate",
    "a2_boss_killed_rate",
    "beat_boss_rate",
    "floor_mean",
    "floor_max",
    "avg_steps",
    "completed",
]


def _build_model(device: torch.device) -> V8Model:
    model = V8Model()
    load_combat_head(model, _COMBAT_HEAD)
    # 注：parity 只需两条路径 model 权重一致（同 seed → 同结果），不需要训练好的 ckpt。
    # 旧 ckpt 可能与当前 V8Model schema 不兼容（state_fuse 维度变化），strict=False 兜底；
    # 不兼容时直接用 combat-head-only 权重，一致性验证依然成立。
    ckpt_path = os.path.join(_REPO, _CKPT)
    if os.path.exists(ckpt_path):
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            missing, unexpected = model.load_state_dict(sd, strict=False)
            print(f"[parity] loaded ckpt {_CKPT} (missing={len(missing)} unexpected={len(unexpected)})")
        except Exception as e:  # noqa: BLE001
            print(f"[parity] ckpt {_CKPT} schema 不兼容 ({type(e).__name__})，"
                  f"改用 combat-head-only 权重（一致性验证仍成立）")
    else:
        print(f"[parity] WARN ckpt {_CKPT} 不存在，用 combat-head-only 权重（仍可验一致性）")
    model.to(device)
    model.eval()
    return model


def _make_env(model: V8Model) -> V8Env:
    """与 run_eval 的 eval env 同构：V8Env + combat_net_wrapper(model)。"""
    return V8Env(combat_net_wrapper=V8CombatNetWrapper(model))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_seeds", type=int, default=3)
    ap.add_argument("--device", default="cpu", help="cpu（默认，避免抢训练 MPS）")
    args = ap.parse_args()

    device = torch.device(args.device)
    model = _build_model(device)

    # 同一个 model 实例，两条评估路径各自一份 env+trainer（trainer 内部绑同一 model）。
    # collect_rollout 按 seed 确定性 reset，故两条路径同种子 → 同结果。
    trainer_a = V8PPOTrainer(model=model, lr=3e-4, device=str(device))
    env_a = _make_env(model)
    print(f"[parity] run_eval: num_seeds={args.num_seeds} offset={_SEED_OFFSET} ...")
    ref = run_eval(trainer_a, env_a, num_seeds=args.num_seeds, seed_offset=_SEED_OFFSET)
    try:
        env_a.close()
    except Exception:  # noqa: BLE001
        pass

    trainer_b = V8PPOTrainer(model=model, lr=3e-4, device=str(device))
    env_b = _make_env(model)
    runner = SimEpisodeRunner(env_b, trainer_b)
    print(f"[parity] harness.evaluate(SimEpisodeRunner): num_seeds={args.num_seeds} ...")
    got = evaluate(model, runner, num_seeds=args.num_seeds, seed_offset=_SEED_OFFSET)
    try:
        env_b.close()
    except Exception:  # noqa: BLE001
        pass

    print("\n[parity] 字段对比 (run_eval vs harness):")
    ok = True
    for k in _RATE_KEYS:
        rv = ref.get(k)
        gv = got.get(k)
        match = (rv == gv) or (
            isinstance(rv, float) and isinstance(gv, float) and abs(rv - gv) < 1e-9
        )
        flag = "OK " if match else "MISMATCH"
        if not match:
            ok = False
        print(f"  [{flag}] {k:24s} run_eval={rv!r:>10} harness={gv!r:>10}")

    # boss_reach_counts dict 单独比
    rbr_ref = ref.get("boss_reach_counts", {})
    rbr_got = got.get("boss_reach_counts", {})
    bmatch = rbr_ref == rbr_got
    ok = ok and bmatch
    print(f"  [{'OK ' if bmatch else 'MISMATCH'}] boss_reach_counts        "
          f"run_eval={rbr_ref} harness={rbr_got}")

    print("\n[parity] RESULT:", "ALL MATCH ✓" if ok else "MISMATCH ✗")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

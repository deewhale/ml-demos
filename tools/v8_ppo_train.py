"""V8 阶段 B：PPO 训练入口。

按 docs/v8_implementation_design.md §10 + docs/v8_design_principles.md：
- Phase A 已训好 combat head（监督 BC，从 turn_records 学搜索的出牌顺序/联动/target）
- Phase B 在战斗 head 之上 RL：学元决策（路线 / 选卡 / 事件 / 商店 / 休息 / 宝箱 / Neow / Boss）
- 战斗内仍由 TurnSolver 主导，model 通过 V8CombatNetWrapper 参与 leaf 评估
- 元决策 trajectory 才进 PPO（V8Env 自动跳过战斗 phase）

CLI 用法：
    # Smoke：管线快速验证（不要给 user 跑这个当真训练）
    python tools/v8_ppo_train.py --smoke

    # 真训练（user 启动，~83h）
    python tools/v8_ppo_train.py --num_episodes 1000 --batch_size 32

实现要点：
- 每 batch_size=N 局收 rollout，concat 后做一次 PPO update
- 每 eval_frequency 局跑一次 eval（deterministic，no_grad）
- 每 checkpoint_frequency 局存一次 ckpt（含 optimizer state）+ 元数据 JSON
- smoke 模式覆盖少量 episode，验证 wrapper hook、PPO update、ckpt 写盘
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

# 保证从仓库根可 import v8.*
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from v8.combat_net_wrapper import V8CombatNetWrapper
from v8.env import V8Env
from v8.model import V8Model
from v8.trainer import RolloutStep, V8PPOTrainer

# 强制 stdout 行缓冲（unbuffered），保证 heartbeat 实时可见。
# python -u / PYTHONUNBUFFERED=1 也行，这里再加一道保险。
try:
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
except Exception:  # noqa: BLE001
    pass

# 用 StreamHandler 显式绑 stdout，避免落进默认 stderr buffer 区
_handler = logging.StreamHandler(stream=sys.stdout)
_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logging.basicConfig(level=logging.INFO, handlers=[_handler], force=True)
logger = logging.getLogger("v8_ppo_train")


# ============================================================
# 工具
# ============================================================


def select_device(preferred: str) -> torch.device:
    """选 device：优先用户传的；mps 不可用回 cpu。"""
    if preferred == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        logger.warning("MPS 不可用，fallback 到 CPU")
        return torch.device("cpu")
    return torch.device(preferred)


def load_combat_head(model: V8Model, ckpt_path: str) -> Dict[str, Any]:
    """加载 Phase A 训好的 combat head 权重到 model。

    Phase A 的 ckpt 是 {model_state_dict, metadata}（与 v8_bc_train.py 一致）。
    Phase A 训的就是 V8Model 全套权重，所以直接 load_state_dict 即可。
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"combat head checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict")
    if state_dict is None:
        raise RuntimeError(f"checkpoint {ckpt_path} 缺 model_state_dict")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    logger.info(
        "loaded combat head from %s (missing=%d, unexpected=%d)",
        ckpt_path, len(missing), len(unexpected),
    )
    if missing:
        logger.info("missing keys (前 5): %s", list(missing)[:5])
    if unexpected:
        logger.info("unexpected keys (前 5): %s", list(unexpected)[:5])
    return ckpt.get("metadata", {})


def save_metadata_json(
    json_path: str,
    *,
    num_episodes_so_far: int,
    eval_history: List[Dict[str, Any]],
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """checkpoint 旁边保存元数据 JSON。"""
    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "num_episodes_so_far": num_episodes_so_far,
        "eval_history": eval_history,
    }
    if extra:
        payload.update(extra)
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, default=str, ensure_ascii=False)


# ============================================================
# 评估
# ============================================================


def run_eval(
    trainer: V8PPOTrainer,
    env: V8Env,
    num_seeds: int,
    seed_offset: int,
) -> Dict[str, Any]:
    """跑 num_seeds 局 deterministic eval（no_grad）。

    指标：reached_boss_rate / beat_boss_rate / floor_mean / avg_episode_steps。
    """
    floors: List[int] = []
    reached_boss = 0
    beat_boss = 0
    total_steps = 0

    for i in range(num_seeds):
        seed = seed_offset + i
        try:
            rollout = trainer.collect_rollout(env, seed=seed, deterministic=True)
        except Exception as e:  # noqa: BLE001
            logger.warning("eval seed=%d crashed: %s: %s", seed, type(e).__name__, e)
            continue

        if not rollout:
            continue
        last = rollout[-1]
        # last.state 是这一步**之前**的 state；需要从 env 拿最新
        runner = env.runner
        if runner is not None:
            final_floor = int(getattr(runner.run_state, "floor", 0) or 0)
            final_act = int(getattr(runner.run_state, "act", 1) or 1)
            game_won = bool(runner.game_won)
        else:
            final_floor = int(getattr(last.state, "floor", 0) or 0)
            final_act = int(getattr(last.state, "act", 1) or 1)
            game_won = False

        floors.append(final_floor)
        total_steps += len(rollout)
        # act1 boss 在 floor 16+；过 boss 进 act2 即 final_act>=2 或 game_won
        if final_floor >= 16 or final_act >= 2:
            reached_boss += 1
        if game_won or final_act >= 2:
            beat_boss += 1
        try:
            env.close()
        except Exception:  # noqa: BLE001
            pass

    n = max(1, len(floors))
    return {
        "num_seeds": num_seeds,
        "completed": len(floors),
        "reached_boss_rate": reached_boss / num_seeds,
        "beat_boss_rate": beat_boss / num_seeds,
        "floor_mean": sum(floors) / n if floors else 0.0,
        "floor_max": max(floors) if floors else 0,
        "avg_steps": total_steps / n if floors else 0.0,
    }


# ============================================================
# Main
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V8 Phase B PPO 训练入口")
    parser.add_argument("--num_episodes", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32, help="每多少 episodes 做一次 PPO update")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="mps", help="mps / cpu / cuda")
    parser.add_argument(
        "--combat_head_checkpoint",
        type=str,
        default="sts_models/v8_combat_head_v1.pt",
    )
    parser.add_argument("--output_dir", type=str, default="sts_models/v8_ppo_rl")
    parser.add_argument("--eval_frequency", type=int, default=100)
    parser.add_argument("--eval_seeds", type=int, default=30)
    parser.add_argument("--checkpoint_frequency", type=int, default=500)
    parser.add_argument(
        "--max_steps_per_episode",
        type=int,
        default=None,
        help="V8Env 单局最多 RL step 数；None 用 env 默认（5000）。smoke 会自动覆盖成 30。",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="管线验证：覆盖小 episode/batch/freq + 写临时 output_dir + 限制 max_steps_per_episode",
    )
    return parser.parse_args()


def apply_smoke_overrides(args: argparse.Namespace) -> None:
    """smoke 模式：强制小 episode + 临时输出目录 + 卡每局 RL 步数。

    单局封顶 30 步（够覆盖几层 meta 决策 + 几场战斗），总耗时控制在 30min 内。
    """
    args.num_episodes = 10
    args.batch_size = 2
    args.eval_frequency = 5
    args.eval_seeds = 2  # smoke eval 也只跑很少几个种子
    args.checkpoint_frequency = 5
    if args.max_steps_per_episode is None:
        args.max_steps_per_episode = 30
    smoke_root = _REPO_ROOT / "sts_models" / "v8_ppo_smoke"
    smoke_root.mkdir(parents=True, exist_ok=True)
    args.output_dir = str(smoke_root / f"run_{int(time.time())}")
    logger.info(
        "smoke 模式：output_dir=%s max_steps_per_episode=%d",
        args.output_dir, args.max_steps_per_episode,
    )


def main() -> None:
    args = parse_args()
    if args.smoke:
        apply_smoke_overrides(args)

    device = select_device(args.device)
    logger.info("device=%s", device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Model + 加载 Phase A combat head
    model = V8Model()
    phase_a_meta = load_combat_head(model, args.combat_head_checkpoint)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "combat head loaded OK: ckpt=%s params=%d (%.2fM)",
        args.combat_head_checkpoint, n_params, n_params / 1e6,
    )

    # 2) Wrapper（让 model 在战斗 search leaf 评估时被调用）
    wrapper = V8CombatNetWrapper(model)

    # 3) Env
    env_kwargs: Dict[str, Any] = {"combat_net_wrapper": wrapper}
    if args.max_steps_per_episode is not None:
        env_kwargs["max_steps_per_episode"] = int(args.max_steps_per_episode)
    env = V8Env(**env_kwargs)

    # 4) Trainer
    trainer = V8PPOTrainer(model=model, lr=args.lr, device=str(device))

    # ---- 启动期 banner，保证用户看得到所有关键 config ----
    logger.info(
        "[startup] output_dir=%s num_episodes=%d batch_size=%d eval_freq=%d ckpt_freq=%d "
        "max_steps_per_episode=%s smoke=%s",
        args.output_dir, args.num_episodes, args.batch_size,
        args.eval_frequency, args.checkpoint_frequency,
        args.max_steps_per_episode, args.smoke,
    )

    # 5) 训练 loop
    eval_history: List[Dict[str, Any]] = []
    train_log: List[Dict[str, Any]] = []
    num_episodes_done = 0
    t_start = time.time()

    logger.info(
        "PPO 训练开始: num_episodes=%d batch_size=%d lr=%.2e eval_freq=%d ckpt_freq=%d",
        args.num_episodes, args.batch_size, args.lr,
        args.eval_frequency, args.checkpoint_frequency,
    )

    while num_episodes_done < args.num_episodes:
        # ---- 收 batch_size 个 rollout ----
        batch_rollouts: List[RolloutStep] = []
        batch_meta: List[Dict[str, Any]] = []
        batch_target = min(args.batch_size, args.num_episodes - num_episodes_done)
        prev_call_count = wrapper.call_count

        for k in range(batch_target):
            ep_idx = num_episodes_done + k
            ep_t0 = time.time()
            try:
                rollout = trainer.collect_rollout(env, seed=ep_idx, deterministic=False)
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "rollout ep=%d crashed: %s: %s (skip)",
                    ep_idx, type(e).__name__, e,
                )
                # crash 也打 heartbeat，方便定位卡顿点
                logger.info(
                    "[heartbeat] ep=%d CRASHED after %.1fs (%s)",
                    ep_idx, time.time() - ep_t0, type(e).__name__,
                )
                continue

            ep_reward = sum(s.reward for s in rollout)
            runner = env.runner
            final_floor = int(getattr(runner.run_state, "floor", 0) or 0) if runner else 0
            final_act = int(getattr(runner.run_state, "act", 1) or 1) if runner else 1
            beat_boss = bool(runner.game_won) if runner else False
            ep_secs = time.time() - ep_t0

            batch_rollouts.extend(rollout)
            batch_meta.append({
                "ep": ep_idx,
                "steps": len(rollout),
                "reward_sum": ep_reward,
                "final_floor": final_floor,
                "final_act": final_act,
                "beat_boss": beat_boss,
                "secs": ep_secs,
            })
            # ---- 每局 heartbeat：silent 跑步是不可接受的 ----
            logger.info(
                "[heartbeat] ep=%d steps=%d secs=%.1f reward=%.3f floor=%d act=%d beat_boss=%s",
                ep_idx, len(rollout), ep_secs, ep_reward,
                final_floor, final_act, beat_boss,
            )
            try:
                env.close()
            except Exception:  # noqa: BLE001
                pass

        if not batch_rollouts:
            logger.warning("batch 全部 crash，跳过 update")
            num_episodes_done += batch_target
            continue

        # ---- PPO update ----
        upd_t0 = time.time()
        metrics = trainer.update(batch_rollouts, num_epochs=4)
        upd_secs = time.time() - upd_t0

        # ---- batch 日志 ----
        ep_steps = [m["steps"] for m in batch_meta]
        ep_rewards = [m["reward_sum"] for m in batch_meta]
        ep_floors = [m["final_floor"] for m in batch_meta]
        n_beat = sum(1 for m in batch_meta if m["beat_boss"])
        wrapper_calls_in_batch = wrapper.call_count - prev_call_count

        num_episodes_done += batch_target
        log_entry = {
            "episodes_done": num_episodes_done,
            "batch_size": len(batch_meta),
            "mean_reward": sum(ep_rewards) / max(1, len(ep_rewards)),
            "mean_steps": sum(ep_steps) / max(1, len(ep_steps)),
            "mean_floor": sum(ep_floors) / max(1, len(ep_floors)),
            "beat_boss_in_batch": n_beat,
            "policy_loss": metrics.get("policy_loss", 0.0),
            "value_loss": metrics.get("value_loss", 0.0),
            "entropy": metrics.get("entropy", 0.0),
            "approx_kl": metrics.get("approx_kl", 0.0),
            "clip_frac": metrics.get("clip_frac", 0.0),
            "update_secs": upd_secs,
            "wrapper_calls": wrapper_calls_in_batch,
        }
        train_log.append(log_entry)
        logger.info(
            "[ep=%d] reward_mean=%.3f steps_mean=%.1f floor_mean=%.1f beat=%d/%d "
            "policy=%.4f value=%.4f entropy=%.4f kl=%.4f clip=%.3f wrapper_calls=%d upd=%.2fs",
            num_episodes_done, log_entry["mean_reward"], log_entry["mean_steps"],
            log_entry["mean_floor"], n_beat, len(batch_meta),
            log_entry["policy_loss"], log_entry["value_loss"], log_entry["entropy"],
            log_entry["approx_kl"], log_entry["clip_frac"],
            wrapper_calls_in_batch, upd_secs,
        )

        # ---- Eval ----
        if args.eval_frequency > 0 and num_episodes_done % args.eval_frequency == 0:
            eval_t0 = time.time()
            # eval 用大 seed offset，跟训练 seed 不冲突
            eval_metrics = run_eval(
                trainer, env,
                num_seeds=args.eval_seeds,
                seed_offset=10_000 + num_episodes_done,
            )
            eval_metrics["episodes_done"] = num_episodes_done
            eval_metrics["secs"] = time.time() - eval_t0
            eval_history.append(eval_metrics)
            logger.info(
                "[eval@ep=%d] reached_boss=%.2f beat_boss=%.2f floor_mean=%.1f (%.1fs)",
                num_episodes_done, eval_metrics["reached_boss_rate"],
                eval_metrics["beat_boss_rate"], eval_metrics["floor_mean"],
                eval_metrics["secs"],
            )

        # ---- Checkpoint ----
        if (
            args.checkpoint_frequency > 0
            and num_episodes_done % args.checkpoint_frequency == 0
        ):
            ckpt_path = output_dir / f"v8_ppo_ep{num_episodes_done}.pt"
            trainer.save_checkpoint(
                str(ckpt_path),
                metadata={
                    "phase": "B_ppo_rl",
                    "episodes_done": num_episodes_done,
                    "lr": args.lr,
                    "batch_size": args.batch_size,
                    "phase_a_meta": phase_a_meta,
                    "elapsed_sec": time.time() - t_start,
                },
            )
            save_metadata_json(
                str(output_dir / f"v8_ppo_ep{num_episodes_done}.json"),
                num_episodes_so_far=num_episodes_done,
                eval_history=eval_history,
                extra={"args": vars(args)},
            )
            logger.info("已保存 checkpoint: %s", ckpt_path)

    # ---- 最终 ckpt + summary ----
    final_ckpt = output_dir / "v8_ppo_final.pt"
    trainer.save_checkpoint(
        str(final_ckpt),
        metadata={
            "phase": "B_ppo_rl_final",
            "episodes_done": num_episodes_done,
            "phase_a_meta": phase_a_meta,
            "elapsed_sec": time.time() - t_start,
        },
    )
    summary = {
        "phase": "B_ppo_rl",
        "args": vars(args),
        "episodes_done": num_episodes_done,
        "elapsed_sec": time.time() - t_start,
        "train_log_tail": train_log[-10:],
        "eval_history": eval_history,
        "wrapper_total_calls": wrapper.call_count,
        "wrapper_avg_value": wrapper.avg_value,
    }
    with open(output_dir / "v8_ppo_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str, ensure_ascii=False)

    logger.info(
        "训练完成 episodes=%d 总耗时=%.1fs final_ckpt=%s wrapper_calls=%d",
        num_episodes_done, time.time() - t_start, final_ckpt, wrapper.call_count,
    )


if __name__ == "__main__":
    main()

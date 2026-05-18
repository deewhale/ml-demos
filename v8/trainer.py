"""V8 PPO Trainer。

按 docs/v8_implementation_design.md 组件 7（RL Trainer）+ 8（Episode Loop）设计：
- Episode 粒度 = 一局 STS（默认 act1），由 v8/env.py 提供。
- Step 粒度 = 一个**元决策**（NEOW / MAP / EVENT / SHOP / REST / TREASURE /
  CARD_REWARDS / BOSS_REWARDS）。战斗内由 env 内部跑搜索，不进 RL trajectory。
- PPO 算法：GAE advantage（γ=0.99, λ=0.95）+ clipped objective（ε=0.2）+ value
  loss + entropy bonus。

设计原则对照（docs/v8_design_principles.md）：
- 标准 RL 不引入新概念
- 战斗内 model 在 RL 阶段 frozen（只元决策 RL）—— 战斗能力靠预训练 + 搜索辅助
- 兼容预训练好的战斗 head（freeze 由 trainer 配置控制）

实现状态：
- collect_rollout：完整接 V8Env，跑一整局 episode
- compute_advantages：标准 GAE
- update：clipped PPO + value loss + entropy bonus
- save_checkpoint / load_checkpoint：标准 torch.save 流程

不在本文件内：
- 训练 driver（smoke 20 局 / 真训练 driver 是下一步）
- model prior 喂搜索（trainer 阶段把 model 喂给 TurnSolver；当前 env 用纯搜索）
- multiprocessing rollout（V6 风格，本期单进程跑通先）
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from v8.env import V8Env
from v8.model import V8Model
from v8.state import V8State


logger = logging.getLogger(__name__)


# ============================================================
# Rollout 数据结构
# ============================================================


@dataclass
class RolloutStep:
    """单个元决策 step 的 trajectory record。

    战斗内不进 trajectory（env 内部跑 TurnSolver）。这里记录的只是
    "model 真实做决策的 step"，对应 PPO 的一个 transition。
    """

    state: V8State                       # 决策时观测到的 state
    available_actions: List[str]         # 决策时可选 action 描述（pointer logits 长度）
    action_idx: int                      # 选中的 action 索引
    log_prob: float                      # 选中 action 的 log π_old(a|s)
    value: float                         # V_old(s)（critic baseline）
    reward: float                        # 此 step 收到的 reward（含 final reward 加成）
    done: bool                           # 此 step 后 episode 是否结束
    # 调试：phase 标识，便于看 trajectory 分布
    phase: str = ""


# ============================================================
# V8 PPO Trainer
# ============================================================


class V8PPOTrainer:
    """PPO trainer。

    用法：
        model = V8Model()
        trainer = V8PPOTrainer(model, device="cpu")
        env = V8Env()
        rollout = trainer.collect_rollout(env, seed=42)
        metrics = trainer.update(rollout, num_epochs=4)
        trainer.save_checkpoint("v8_ckpt.pt", metadata={"episode": 1})

    不做 multiprocessing：本期目标是单进程跑通 + 验证 forward/backward。
    多进程 rollout 收集是后续优化（参考 V6 train_v6.py 模式）。
    """

    def __init__(
        self,
        model: V8Model,
        lr: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=1e-4,
        )

        self.gamma = float(gamma)
        self.lam = float(lam)
        self.clip_eps = float(clip_eps)
        self.value_coef = float(value_coef)
        self.entropy_coef = float(entropy_coef)
        self.max_grad_norm = float(max_grad_norm)

        # 性能 / 调用计数（每次 collect_rollout / update 后填充，trainer driver 读）
        self.last_rollout_stats: Dict[str, float] = {}
        self.last_update_stats: Dict[str, float] = {}

    # ---------------------------------------------------------
    # Rollout collection
    # ---------------------------------------------------------

    @torch.no_grad()
    def collect_rollout(
        self,
        env: V8Env,
        seed: int,
        deterministic: bool = False,
    ) -> List[RolloutStep]:
        """跑一整局 episode（act1）收集 trajectory。

        每个元决策 step：
            1. env.get_available_actions() → list[str]
            2. model.forward(state, actions) → {logits, value}
            3. probs = softmax(logits); action_idx = sample/argmax
            4. log_prob = log(probs[action_idx])
            5. env.step(action_idx) → next_state, reward, done, info
            6. 记 RolloutStep
            7. 直到 done

        deterministic=False（训练时）：multinomial sample
        deterministic=True（eval 时）：argmax

        log_prob / value 在 collect 时存（"old" 版本，update 时算 ratio）。
        """
        was_training = self.model.training
        self.model.eval()

        rollout: List[RolloutStep] = []

        # ---- 性能 / 调用计数 ----
        # 让 env 在 reset 前清 per-episode 计数；reset 内部也会清，但显式更稳
        if hasattr(env, "reset_perf_counters"):
            env.reset_perf_counters()
        forward_time_sec = 0.0
        env_step_time_sec = 0.0

        state = env.reset(seed=seed)

        step_idx = 0
        while True:
            actions = env.get_available_actions()
            if not actions:
                # 没合法 action：env.step 会兜底 done，但这里也提前停
                logger.warning(
                    "collect_rollout: no available actions at step=%d phase=%s, break",
                    step_idx, state.phase,
                )
                break

            # forward
            fwd_t0 = time.time()
            out = self.model(state, actions)
            forward_time_sec += time.time() - fwd_t0
            logits = out["logits"]            # [n_options]
            value = out["value"]              # scalar tensor
            n_options = logits.shape[0]

            # 安全：env 给的 actions 长度可能跟 logits 不一致
            #     （model forward 在 COMBAT phase 用 hand×monsters；元决策用 actions 长度）
            # 元决策 phase 应该一致；如不一致 fallback 取 min
            valid_n = min(n_options, len(actions))
            if valid_n <= 0:
                logger.warning(
                    "collect_rollout: valid_n=0 at step=%d, break", step_idx,
                )
                break
            logits = logits[:valid_n]

            probs = F.softmax(logits, dim=-1)

            if deterministic:
                action_idx = int(torch.argmax(probs).item())
            else:
                # multinomial 在 MPS 上有兼容问题，统一回 CPU 采样
                probs_cpu = probs.detach().to("cpu")
                # 防 NaN / 全 0
                if torch.isnan(probs_cpu).any() or probs_cpu.sum().item() <= 0:
                    action_idx = 0
                else:
                    action_idx = int(torch.multinomial(probs_cpu, num_samples=1).item())

            # log_prob
            # log_softmax 数值更稳
            log_probs_all = F.log_softmax(logits, dim=-1)
            log_prob = float(log_probs_all[action_idx].item())
            value_scalar = float(value.item()) if value.dim() == 0 else float(value.flatten()[0].item())

            # 调用 env.step（这一步可能内部跑战斗 + post-battle evaluate）
            es_t0 = time.time()
            next_state, reward, done, info = env.step(action_idx)
            env_step_time_sec += time.time() - es_t0

            rollout.append(
                RolloutStep(
                    state=state,
                    available_actions=actions,
                    action_idx=action_idx,
                    log_prob=log_prob,
                    value=value_scalar,
                    reward=float(reward),
                    done=bool(done),
                    phase=state.phase or "",
                )
            )

            step_idx += 1
            state = next_state

            if done:
                break

        if was_training:
            self.model.train()

        # 暴露给 trainer driver 做 [perf] / [heartbeat] 日志
        self.last_rollout_stats = {
            "forward_time_sec": forward_time_sec,
            "env_step_time_sec": env_step_time_sec,
            "eval_deck_calls": float(getattr(env, "eval_deck_calls", 0)),
            "eval_deck_time_sec": float(getattr(env, "eval_deck_time_sec", 0.0)),
            "combat_search_calls": float(getattr(env, "combat_search_calls", 0)),
            "n_steps": float(len(rollout)),
        }
        return rollout

    # ---------------------------------------------------------
    # GAE advantage
    # ---------------------------------------------------------

    def compute_advantages(
        self,
        rollout: List[RolloutStep],
        last_value: float = 0.0,
    ) -> Dict[str, List[float]]:
        """标准 GAE。

        δ_t = r_t + γ V(s_{t+1}) (1 - done_t) - V(s_t)
        A_t = δ_t + γλ A_{t+1} (1 - done_t)
        return_t = A_t + V(s_t)

        Args:
            rollout: trainer.collect_rollout 输出
            last_value: V(s_T)（terminal 后 bootstrap）。一般 done=True
                        时设 0；本环境 episode 末必 done，传 0 即可。

        Returns:
            {
              "advantages": list[float] 长度 = len(rollout)
              "returns":    list[float] 长度 = len(rollout)
            }
        """
        n = len(rollout)
        advantages = [0.0] * n
        returns = [0.0] * n

        gae = 0.0
        next_value = float(last_value)

        for t in reversed(range(n)):
            step = rollout[t]
            done_t = 1.0 if step.done else 0.0
            # 注意：step.done 标识"这步之后游戏结束"，所以 V(s_{t+1}) 应该被屏蔽
            #       未结束时 next_value 是下一步 step.value
            mask = 1.0 - done_t
            delta = step.reward + self.gamma * next_value * mask - step.value
            gae = delta + self.gamma * self.lam * mask * gae
            advantages[t] = gae
            returns[t] = gae + step.value
            next_value = step.value

        return {"advantages": advantages, "returns": returns}

    # ---------------------------------------------------------
    # Update
    # ---------------------------------------------------------

    def update(
        self,
        rollout: List[RolloutStep],
        num_epochs: int = 4,
        normalize_advantage: bool = True,
    ) -> Dict[str, float]:
        """PPO update。

        步骤：
            1. compute_advantages → advantages + returns
            2. 多 epoch 在整个 rollout 上做 update：
               - new_logits, new_value = model(state, actions)
               - new_log_prob = log_softmax(new_logits)[action_idx]
               - ratio = exp(new_log_prob - old_log_prob)
               - clipped_ratio = clip(ratio, 1-eps, 1+eps)
               - policy_loss = -min(ratio * adv, clipped_ratio * adv).mean()
               - value_loss = (new_value - returns)^2.mean()
               - entropy = -(probs * log_probs).sum().mean()
               - loss = policy + value_coef·value - entropy_coef·entropy
               - loss.backward(); clip grad; optimizer.step()

        本期不做 minibatch（rollout 长度 ~10-50 步，整批跑也能 fit）。
        如果之后需要 minibatch，把整批跑改成 chunked 即可。
        """
        if not rollout:
            self.last_update_stats = {
                "update_total_sec": 0.0,
                "update_forward_sec": 0.0,
                "update_n_forwards": 0.0,
            }
            return {
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy": 0.0,
                "total_loss": 0.0,
                "approx_kl": 0.0,
                "clip_frac": 0.0,
                "n_steps": 0,
            }

        update_t0 = time.time()
        update_forward_sec = 0.0
        update_n_forwards = 0
        self.model.train()

        # 1. advantages / returns
        adv_ret = self.compute_advantages(rollout)
        advantages_list = adv_ret["advantages"]
        returns_list = adv_ret["returns"]

        if normalize_advantage and len(advantages_list) > 1:
            adv_arr = torch.tensor(advantages_list, dtype=torch.float32, device=self.device)
            adv_arr = (adv_arr - adv_arr.mean()) / (adv_arr.std() + 1e-8)
            advantages_norm: List[float] = adv_arr.tolist()
        else:
            advantages_norm = list(advantages_list)

        old_log_probs = [s.log_prob for s in rollout]

        metrics_accum: Dict[str, float] = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "total_loss": 0.0,
            "approx_kl": 0.0,
            "clip_frac": 0.0,
        }
        n_updates = 0

        for epoch in range(num_epochs):
            # 单 epoch：跑所有 step（loss 累加，再一次 backward）
            policy_losses = []
            value_losses = []
            entropies = []
            kls = []
            clip_fracs = []

            for t, step in enumerate(rollout):
                fwd_t0 = time.time()
                out = self.model(step.state, step.available_actions)
                update_forward_sec += time.time() - fwd_t0
                update_n_forwards += 1
                logits = out["logits"]
                value = out["value"]

                valid_n = min(logits.shape[0], len(step.available_actions))
                if valid_n <= 0:
                    continue
                logits = logits[:valid_n]
                if step.action_idx >= valid_n:
                    # 不正常：rollout 时 action_idx 应在 valid 范围内
                    logger.warning(
                        "update: action_idx=%d >= valid_n=%d, skip step",
                        step.action_idx, valid_n,
                    )
                    continue

                log_probs_all = F.log_softmax(logits, dim=-1)
                probs_all = F.softmax(logits, dim=-1)

                new_log_prob = log_probs_all[step.action_idx]
                # entropy = -Σ p_i log p_i
                entropy = -(probs_all * log_probs_all).sum()

                old_lp = torch.tensor(
                    old_log_probs[t], dtype=torch.float32, device=self.device
                )
                adv = torch.tensor(
                    advantages_norm[t], dtype=torch.float32, device=self.device
                )
                ret = torch.tensor(
                    returns_list[t], dtype=torch.float32, device=self.device
                )

                ratio = torch.exp(new_log_prob - old_lp)
                clipped_ratio = torch.clamp(
                    ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps
                )
                policy_loss = -torch.min(ratio * adv, clipped_ratio * adv)

                # value loss（MSE）
                value_pred = value if value.dim() == 0 else value.flatten()[0]
                value_loss = (value_pred - ret).pow(2)

                policy_losses.append(policy_loss)
                value_losses.append(value_loss)
                entropies.append(entropy)

                with torch.no_grad():
                    kl = (old_lp - new_log_prob).clamp(min=-10.0, max=10.0)
                    kls.append(float(kl.item()))
                    clipped = float(
                        ((ratio < 1.0 - self.clip_eps) | (ratio > 1.0 + self.clip_eps))
                        .float()
                        .item()
                    )
                    clip_fracs.append(clipped)

            if not policy_losses:
                # 整个 epoch 没收到合法 step（不应该）
                continue

            policy_loss_t = torch.stack(policy_losses).mean()
            value_loss_t = torch.stack(value_losses).mean()
            entropy_t = torch.stack(entropies).mean()

            total_loss = (
                policy_loss_t
                + self.value_coef * value_loss_t
                - self.entropy_coef * entropy_t
            )

            self.optimizer.zero_grad()
            total_loss.backward()
            if self.max_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
            self.optimizer.step()

            metrics_accum["policy_loss"] += float(policy_loss_t.item())
            metrics_accum["value_loss"] += float(value_loss_t.item())
            metrics_accum["entropy"] += float(entropy_t.item())
            metrics_accum["total_loss"] += float(total_loss.item())
            metrics_accum["approx_kl"] += sum(kls) / max(1, len(kls))
            metrics_accum["clip_frac"] += sum(clip_fracs) / max(1, len(clip_fracs))
            n_updates += 1

        if n_updates > 0:
            for k in list(metrics_accum.keys()):
                metrics_accum[k] /= n_updates
        metrics_accum["n_steps"] = float(len(rollout))

        self.last_update_stats = {
            "update_total_sec": time.time() - update_t0,
            "update_forward_sec": update_forward_sec,
            "update_n_forwards": float(update_n_forwards),
        }
        return metrics_accum

    # ---------------------------------------------------------
    # Checkpoint
    # ---------------------------------------------------------

    def save_checkpoint(self, path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """保存 model + optimizer + 元数据。"""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metadata": metadata or {},
            },
            path,
        )
        logger.info("V8PPOTrainer: saved checkpoint to %s", path)

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """加载 model + optimizer + 元数据。返回 metadata（方便恢复 episode 计数）。

        兼容性策略（2026-05-13 加 boss-aware encoding 后引入）：
        - model: 用 strict=False 加载，旧 ckpt 缺新加的 boss_proj.* keys 会被忽略
          并保留 module 内 random init（log missing/unexpected 数量）。
        - optimizer: 如果 param 数量变了（新 model 多了 boss_proj 参数），原 optimizer
          state_dict 跟 self.optimizer.param_groups 对不上，load 会 raise ValueError。
          捕获后保留 self.optimizer 的 fresh AdamW state（旧权重仍在，仅 Adam 一阶/
          二阶矩重新积累），不阻塞 resume。
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        missing, unexpected = self.model.load_state_dict(
            ckpt["model_state_dict"], strict=False
        )
        if missing or unexpected:
            logger.info(
                "V8PPOTrainer: model load_state_dict missing=%d unexpected=%d "
                "(missing keys preview: %s, unexpected preview: %s)",
                len(missing), len(unexpected),
                list(missing)[:5], list(unexpected)[:5],
            )
        try:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except (ValueError, KeyError) as e:
            logger.warning(
                "V8PPOTrainer: optimizer load_state_dict mismatch (%s: %s); "
                "keeping fresh optimizer state (model weights still loaded). "
                "Adam moments will rebuild over next batches.",
                type(e).__name__, e,
            )
        logger.info("V8PPOTrainer: loaded checkpoint from %s", path)
        return ckpt.get("metadata", {})


__all__ = ["V8PPOTrainer", "RolloutStep"]

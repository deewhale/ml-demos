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
        # ---- 阶段 3：自适应熵系数 / 探索度地板（防熵崩 → skip-all 锁死）----
        # v3 调温和（2026-05-31）：见下方 __init__ 注释「v3 调温和」段。初始值待调。
        adaptive_entropy: bool = True,
        target_entropy: float = 0.06,
        entropy_coef_min: float = 0.03,
        entropy_coef_max: float = 0.12,
        entropy_adjust_rate: float = 0.5,
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

        # ============================================================
        # 自适应熵系数 / 探索度地板（阶段 3）
        # ============================================================
        # 背景：上次训练熵从 ~0.71（batch_v15 起点）崩到 ~0.02，策略锁死 skip-all。
        #   旧实现 entropy_coef 固定 0.01，无 floor、无自适应，崩了救不回来。
        #
        # 设计（保持简单：一个带上下限的「比例控制器」，不上 SAC 那套
        #   target-entropy Lagrangian）：
        #   - 每次 PPO update 后读这批的平均策略熵 H。
        #   - H < target_entropy（探索不足）→ 乘性上调 entropy_coef（鼓励探索），
        #     但不超过 entropy_coef_max（cap，防探索过头学不动）。
        #   - H > target_entropy（探索足够）→ 乘性下调 entropy_coef（让策略收敛），
        #     但不低于 entropy_coef_min（floor，floor>0 → 永远保底一点探索，防熵崩死）。
        #   - 乘性步长用 (1 ± entropy_adjust_rate)，每 update 小步走、不震荡。
        #
        # 所有边界初始值待训练时按「熵轨迹守不守得住 + 学不学得动」两轴校验后调。
        #
        # ★ v3 调温和（2026-05-31）：v2（target=0.25 / floor=0.03 / cap=0.50 / rate=0.5）
        #   实测「硬顶没用还添乱」——redesign_v2 把 ent_coef 一路顶到 cap=0.5 仍拉不住
        #   探索度（熵磨到 0.027），且爬坡期被高 coef 引发过 KL 失稳（approx_kl 冲 0.18）。
        #   结论：**这套奖励下策略熵天生收敛得低**，追一个够不到的高 target（0.25）只会
        #   让控制器长期顶满 cap、扰动梯度、破坏正在爬坡的学习。
        #   改成「防真崩盘的温和安全网」（不再追高探索，只防掉到接近 0 的真崩）：
        #     1) target_entropy 0.25 → 0.06：只防熵掉到接近 0 的真崩盘，不再追够不到的高目标。
        #        熵稳在 ~0.04（略低于 target）时控制器轻微施压即可，不会狂顶。
        #     2) cap 0.50 → 0.12：温和上限。不再顶到 0.5（那会引发 KL 失稳）；
        #        最多到 0.12，给一点保底探索压力但不扰乱爬坡。
        #     3) floor 0.03：保持不变（永远保底一点探索）。
        #   响应公式（gap-proportional）和 rate=0.5 保留——只是 target/cap 收窄后，
        #   gap 变小 → 单步乘子变小 → 整体行为从「狂顶 cap」变成「温和停在 cap 附近」。
        #   高于 target 时仍对称地按 (1 - rate * (H - target)/target) 乘性下调，floor 兜底。
        #   所有参数仍可配，初始值待按熵轨迹 + 学习两轴校验后微调。
        self.adaptive_entropy = bool(adaptive_entropy)
        self.target_entropy = float(target_entropy)
        self.entropy_coef_min = float(entropy_coef_min)
        self.entropy_coef_max = float(entropy_coef_max)
        self.entropy_adjust_rate = float(entropy_adjust_rate)
        # entropy_coef 从构造值起步，但确保落在 [min, max] 区间内（防初值越界）
        self.entropy_coef = float(
            min(max(self.entropy_coef, self.entropy_coef_min), self.entropy_coef_max)
        )

        # 性能 / 调用计数（每次 collect_rollout / update 后填充，trainer driver 读）
        self.last_rollout_stats: Dict[str, float] = {}
        self.last_update_stats: Dict[str, float] = {}

    # ---------------------------------------------------------
    # 自适应熵系数控制器（阶段 3）
    # ---------------------------------------------------------

    def _update_entropy_coef(self, measured_entropy: float) -> None:
        """根据本批策略熵，按「与 gap 成比例」的控制器调 entropy_coef（带上下限）。

        逻辑（v2 加强，见 __init__ 注释）：
          gap_ratio = (target - H) / target，归一化的「熵离 target 多远」。
          - H < target（探索不足，gap_ratio>0）→ entropy_coef ×(1 + rate·gap_ratio)，
            上限 cap。gap 越大涨越猛，刹得住快速塌方。
          - H ≥ target（探索够，gap_ratio<0）→ entropy_coef ×(1 + rate·gap_ratio)
            = ×(1 - rate·|gap_ratio|)，下限 floor（floor>0 永远保底探索）。
          统一一条乘性式子，符号由 gap_ratio 决定，天然对称。

        每 update 调一次。gap_ratio 已用 target 归一并 clamp，单步乘子受控不爆。
        """
        if not self.adaptive_entropy:
            return
        if measured_entropy != measured_entropy:  # NaN 防御
            return
        target = self.target_entropy if self.target_entropy > 1e-8 else 1e-8
        gap_ratio = (target - measured_entropy) / target
        # clamp gap_ratio 到 [-1, 1]：熵=0 时 gap_ratio=1（最大上调），
        # 熵=2·target 时 gap_ratio=-1（最大下调），防极端熵值把单步乘子拉爆。
        gap_ratio = max(-1.0, min(1.0, gap_ratio))
        self.entropy_coef *= (1.0 + self.entropy_adjust_rate * gap_ratio)
        # clamp 到 [floor, cap]；floor>0 → 永远保底探索（核心防崩保险）
        self.entropy_coef = float(
            min(max(self.entropy_coef, self.entropy_coef_min), self.entropy_coef_max)
        )

    # ---------------------------------------------------------
    # Rollout collection
    # ---------------------------------------------------------

    @torch.no_grad()
    def collect_rollout_batched(
        self,
        parallel_env: Any,  # ParallelV8Env (duck typed to avoid circular import)
        seeds: List[int],
        deterministic: bool = False,
    ) -> List[List[RolloutStep]]:
        """并行收 N 局 episode 的 trajectory（每个 sub-env 一条）。

        与 collect_rollout 相比：
        - 主进程仍单 model forward（pointer-net 不同长度，不能 batch forward）
        - per step 流程：
            1) parallel_env.get_available_actions() → List[List[str]]  (n_envs)
            2) 主进程 per-active-env 跑 model.forward 选 action_idx
            3) parallel_env.step(action_idxs) → batched (states, rewards, dones, infos)
            4) 已 done 的 env 标 inactive；step idx 给 inactive env 填 0（被 worker
               忽略：worker 看 done state 不会再 step，但我们仍然要把 idx 占位以满足
               broadcast 长度约束）
            5) 等所有 N env 都 done → 收 batch 结束

        ⚠️ 已 done 的 env 在并行 batched step 中也仍然会被发 cmd（占位 0）。
        worker 端 V8Env 在已 done 状态下 step 行为：env 内 done 时 step 会返回 done=True
        + reward=0；这不会污染 trajectory（我们在 active mask 内不记录这步）。
        但还是有 cost。一种优化：把 active list 真正缩减后只 broadcast 子集；现行简单
        实现先保正确性。

        参数：
            parallel_env: ParallelV8Env 实例，n_envs == len(seeds)
            seeds: 每个 env 的 reset seed
            deterministic: True=argmax, False=multinomial sample

        返回：
            List[List[RolloutStep]]，外层 len=n_envs，内层是每 env trajectory
            （已 done 后不再 append）
        """
        n_envs = parallel_env.n_envs
        if len(seeds) != n_envs:
            raise ValueError(f"seeds len={len(seeds)} != n_envs={n_envs}")

        was_training = self.model.training
        self.model.eval()

        forward_time_sec = 0.0
        env_step_time_sec = 0.0

        # reset_perf_counters 让 per-rollout 子进程统计干净
        if hasattr(parallel_env, "reset_perf_counters"):
            parallel_env.reset_perf_counters()

        rollouts: List[List[RolloutStep]] = [[] for _ in range(n_envs)]
        active = [True] * n_envs
        states = parallel_env.reset(seeds)

        max_iter = 100_000  # 兜底，避免 batched 死循环
        it = 0
        while any(active) and it < max_iter:
            it += 1
            acts_lists = parallel_env.get_available_actions()
            assert len(acts_lists) == n_envs

            action_idxs: List[int] = []
            # 每个 active env 记下选了哪个 idx + log_prob + value 以便 step 后 push
            per_env_decision: List[Optional[Dict[str, Any]]] = [None] * n_envs

            for i in range(n_envs):
                if not active[i]:
                    # inactive env：发 placeholder 0；不记 decision
                    action_idxs.append(0)
                    continue

                actions = acts_lists[i]
                if not actions:
                    # 没合法 action：mark inactive，期望 env.step 兜底 done
                    logger.warning(
                        "collect_rollout_batched: env-%d no actions at phase=%s, mark done",
                        i, states[i].phase,
                    )
                    action_idxs.append(0)
                    active[i] = False
                    continue

                fwd_t0 = time.time()
                out = self.model(states[i], actions)
                forward_time_sec += time.time() - fwd_t0
                logits = out["logits"]
                value = out["value"]
                n_options = logits.shape[0]
                valid_n = min(n_options, len(actions))
                if valid_n <= 0:
                    action_idxs.append(0)
                    active[i] = False
                    continue
                logits = logits[:valid_n]

                probs = F.softmax(logits, dim=-1)
                if deterministic:
                    action_idx = int(torch.argmax(probs).item())
                else:
                    probs_cpu = probs.detach().to("cpu")
                    if torch.isnan(probs_cpu).any() or probs_cpu.sum().item() <= 0:
                        action_idx = 0
                    else:
                        action_idx = int(torch.multinomial(probs_cpu, num_samples=1).item())

                log_probs_all = F.log_softmax(logits, dim=-1)
                log_prob = float(log_probs_all[action_idx].item())
                value_scalar = (
                    float(value.item()) if value.dim() == 0
                    else float(value.flatten()[0].item())
                )

                action_idxs.append(action_idx)
                per_env_decision[i] = {
                    "actions": actions,
                    "action_idx": action_idx,
                    "log_prob": log_prob,
                    "value": value_scalar,
                    "state": states[i],
                }

            es_t0 = time.time()
            next_states, rewards, dones, infos = parallel_env.step(action_idxs)
            env_step_time_sec += time.time() - es_t0
            assert len(next_states) == n_envs

            # 把这步推到对应 env trajectory（只对原本 active 且有 decision 的 env）
            for i in range(n_envs):
                d = per_env_decision[i]
                if d is None:
                    continue
                rollouts[i].append(
                    RolloutStep(
                        state=d["state"],
                        available_actions=d["actions"],
                        action_idx=d["action_idx"],
                        log_prob=d["log_prob"],
                        value=d["value"],
                        reward=float(rewards[i]),
                        done=bool(dones[i]),
                        phase=d["state"].phase or "",
                    )
                )
                if dones[i]:
                    active[i] = False
                else:
                    states[i] = next_states[i]

        if it >= max_iter:
            logger.error(
                "collect_rollout_batched: hit max_iter=%d safety cap (still active=%s)",
                max_iter, [i for i, a in enumerate(active) if a],
            )

        if was_training:
            self.model.train()

        self.last_rollout_stats = {
            "forward_time_sec": forward_time_sec,
            "env_step_time_sec": env_step_time_sec,
            # eval_deck_calls / combat_search_calls 在 batched 模式下分散在子进程，
            # 主进程读不到（phase 2 简化：置 0；如需要可加 get_perf_counters cmd）
            "eval_deck_calls": 0.0,
            "eval_deck_time_sec": 0.0,
            "combat_search_calls": 0.0,
            "n_steps": float(sum(len(r) for r in rollouts)),
        }
        return rollouts

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

    @torch.no_grad()
    def policy_entropy(self, rollout: List[RolloutStep]) -> float:
        """计算一批 rollout step 上的平均策略熵 H = -Σ p log p。

        阶段 3：eval 路径用它把"当前策略熵"纳入 eval 输出（熵崩第一时间看见）。
        与 update() 里算 entropy 同口径（对每个决策 state 的 logits 算 softmax 熵，
        再对所有 step 求均值）。eval rollout 是 deterministic（argmax）采集的，但熵
        本身只依赖 logits 分布，与采样方式无关。
        """
        if not rollout:
            return 0.0
        was_training = self.model.training
        self.model.eval()
        ent_sum = 0.0
        ent_n = 0
        for step in rollout:
            out = self.model(step.state, step.available_actions)
            logits = out["logits"]
            valid_n = min(logits.shape[0], len(step.available_actions))
            if valid_n <= 0:
                continue
            logits = logits[:valid_n]
            log_probs = F.log_softmax(logits, dim=-1)
            probs = F.softmax(logits, dim=-1)
            ent = float(-(probs * log_probs).sum().item())
            ent_sum += ent
            ent_n += 1
        if was_training:
            self.model.train()
        return ent_sum / ent_n if ent_n > 0 else 0.0

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

        # ---- 阶段 3：用本批平均熵驱动自适应熵系数（带探索地板）----
        # 注意：先记录"本次 update 实际生效的 coef"（用于本批 loss 的那个），
        #   再根据本批熵调出"下批用的 coef"。这样 metrics 里的 entropy_coef 反映
        #   的是"刚跑完这批用的值"，driver 日志看得清楚。
        metrics_accum["entropy_coef"] = float(self.entropy_coef)
        self._update_entropy_coef(metrics_accum.get("entropy", 0.0))
        metrics_accum["entropy_coef_next"] = float(self.entropy_coef)
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

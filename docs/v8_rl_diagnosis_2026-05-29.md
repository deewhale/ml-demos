# V8 RL 全面诊断报告（2026-05-29）

> **⚠ 阶段 0 实施后更正（2026-05-29，commit 见 fix(v8 阶段0)）**：本报告关于"战斗"的两条根因结论**部分被推翻**：
> - 「本该 4–8 回合的小怪打 24–50 回合」是**计数 bug 的幻觉**：`[combat] exit` 日志的 `turns` 字段实际等于动作数 `turn_actions`（战斗结束 `current_combat` 被置 None，回合数读不到、fallback 恒触发）。阶段 0 修了计数后实测小怪真实回合数为 2–6（隔离测试 + 单局 rollout 验证），战斗引擎本身健康。
> - 70% 随机桩**确有注入噪声**，但隔离测试（`tools/v8_combat_isolated_test.py` mode A vs B）显示其影响是**边缘级**：拔掉噪声、纯手写启发只是更稳一点（赢率略升、少丢几滴血），并非"战斗被随机数指挥到崩"。阶段 0 已禁用随机桩（`v8/env.py` reset 不再把 stub 传给 adapter）。
> - **真正的头号根因仍是奖励**（reward hacking：`-丢血 -0.5×回合` 引发 skip-all），见下文第 2 条与 `docs/v8_rl_fix_plan_2026-05-29.md`。下文凡涉及"战斗回合数/随机桩为头号根因"的措辞以本更正为准。

> 本文是一次系统性排查的结论存档。目的：在重新设计奖励 / 评估机制之前，先把"现在这套到底怎么回事"用证据摸清，并拿标准 RL 训练流程做对照。
> 所有结论分两档标注：**[证实]** = 有直接代码 / 日志证据；**[推断]** = 基于证据链的推理，未被单一证据直接锁死。
> 排查方式：多路子 agent 读代码 + 翻日志 + 查外部文献，主对话汇总。

---

## TL;DR（一句话版）

两个月的训练建在三个叠加的坑上，它们形成一条因果闭环死锁：

1. **[证实] 头号根因——战斗在被随机数指挥**：战斗搜索的叶子评估，70% 来自一个**从未加载训练权重的随机桩网络**。真正训好的战斗权重 `v8_combat_head_v1.pt` 只进了上层选牌模型，没进战斗搜索。结果：本该 4–8 回合的小怪打 24–50 回合，每场白扔 30–60 血。
2. **[证实] 奖励错配（教科书级 reward hacking）**：上层 RL 根本不控制战斗，却为战斗的"丢血 + 回合数"扣分买单。于是"拿好卡 → 战斗更长 → 扣分更多 → 拿卡 = 亏"，模型理性地学会"几乎所有卡都不要"。
3. **[证实] 探索崩溃 + 续训繁殖**：策略熵从 0.71 一路崩到 0.02，固定熵系数 0.01 拦不住；而"每批必须从上批 ckpt 续训"的强制规范，把这个锁死的策略一代代繁殖了下来。

**因果链**：战斗随机桩 → 战斗漏血 → 拿卡只会让漏血战斗更长 → 奖励判"拿卡=亏" → 不拿卡 → 采不到"拿卡然后赢"的轨迹 → 探索崩溃 → 续训把锁死传给下一批 → 通关率恒为 0（训练 ~4900 局，won_game 始终 0）。

**对照标准 RL**：我们踩中的是 specification gaming / reward hacking 的经典模式。标准解法（PBRS 势差奖励、entropy floor / 自适应熵、从未塌 ckpt 重训、难度课程）逐条都和我们当前做法相反。有一个业余规模的独立 Slay the Spire RL 项目，用"奖励极简化 + 六阶段难度课程"在远小于我们的算力下拿到了真实通关——方向恰恰是我们的反面。

---

## 第一部分　当前系统实况（带证据）

### 1.1 架构

- **战斗外**（NEOW / 选牌 CARD_REWARD / 路线 MAP / 事件 EVENT / 商店 SHOP / 休息 REST / 宝箱 TREASURE / boss 奖励）：PPO 训练的 meta 策略。模型 = set-encoder(mean-pool) + pointer-network actor + value head，输出单标量奖励对应的动作打分。
- **战斗内**：不归 RL 管。由 StSRLSolver 的 `TurnSolver` 树搜索驱动，叶子评估混入一个"战斗打分网络"。
- 入口：`tools/v8_ppo_train.py`；环境：`v8/env.py`；奖励：`v8/reward.py`；战斗桥接：`v8/combat_net_wrapper.py`。

### 1.2 战斗引擎：搜索 + 随机桩（头号根因）　**[证实]**

**叶子评估的真实公式**（`external/StSRLSolver/.../turn_solver.py:354-362`）：

```
if self._neural_eval is not None:
    neural_score = self._neural_eval(engine)
    # Blend: 70% neural, 30% heuristic
    score = 0.7 * neural_score * 100.0 + 0.3 * score
```

其中 `neural_eval` 调用链（`turn_solver.py:1131-1144`）：`neural_eval(engine)` → `_cn.predict(obs)`，`_cn` 就是 `v8/env.py:493-495` 传进来的 `combat_net_wrapper`。

**而那个 `combat_net_wrapper` 的打分网络是随机桩**（`v8/combat_net_wrapper.py:43-52` 原文注释）：

```
class _CombatObsValueHead(nn.Module):
    """临时 head：把 StSRLSolver CombatStateEncoder 的 298 维 obs 映射到 scalar。
    Round 2 stub：随机权重，不参与训练。等收完 turn_records 数据后，model 加
    config 训这个 head（以 search 的 multi-turn evaluator 输出为 target，或者
    直接用 deck_evaluator outcome 作为 leaf reward 训）。
    保留它的目的：让 wrapper.predict 真返回个 model-derived float，证明 hook
    接通；同时为后续训练留 head 位置。"""
```

- `combat_net_wrapper.py:56-62`：两个 `nn.Linear` 随机初始化，**无任何 `load_state_dict` / `torch.load`**。
- `combat_net_wrapper.py:100-101`：`self._obs_head = combat_obs_head or _CombatObsValueHead()`，全仓库无任何调用处传入 `combat_obs_head`，故永远是随机新实例；仅 `.eval()`，从不加载权重。

**训练好的权重去哪了**：`v8_combat_head_v1.pt`（2026-05-07 由 100 局 self-play、697 条 turn_records BC 训出）只在 `tools/v8_ppo_train.py:803` 经 `load_combat_head()`（:126-147）加载进**整个 V8Model**（即上层选牌策略），**不进战斗搜索的叶子评估**。

**结论（铁证级二选一，选 B）**：战斗搜索的 leaf value = `0.7 × 随机网络 + 0.3 × 手写启发`。70% 的搜索启发是随机噪声，而且这个噪声还会主动污染本来可用的 30% 手写启发。训练好的 .pt 文件实际从未参与战斗。

**残余待确认（动手修之前再看一眼）**：`wrapper.predict()` 函数体的具体路由（注释已写明它"让 wrapper.predict 真返回个 model-derived float"，指向 `_obs_head`，但建议改之前贴一次 predict() 全文坐实）。

### 1.3 奖励系统全貌　**[证实]**

**战斗结束奖励**（`v8/reward.py:66-111`，系数 :52-63）：

```
combat_reward = (won ? +30 : -30) - 1.0*hp_lost - 0.5*turns + 5.0*damage_ratio
```

**每步奖励**（`reward.py:135-171`）：`- 0.5 * Δhp_loss_ratio*100 + 0.5 * node_reward + combat_reward`

**最终奖励**（`reward.py:174-180`）：`(game_won ? +100 : 0) + 1.0 * final_floor`

**节点奖励**（`env.py:285-328`）：

| 节点 | 触发条件 | reward |
|---|---|---|
| EVENT | 离开且 max_hp↑ 或 relic↑ | +5.0 |
| SHOP | 离开且 relic↑ | +3.0 |
| REST | 离开（任意） | +2.0 |
| TREASURE | 离开且 relic↑ | +3.0 |
| MAP / NEOW / CARD_REWARD / BOSS_REWARD | — | 0.0 |

- **选牌阶段无任何 reward shaping**，完全靠 GAE 把下一场战斗结果回溯到选牌动作（`env.py:299-300`）。
- **deck_evaluator（模拟战评分）已从 RL 流程彻底移除**，只剩 `restart_pool` / `get_cache_stats` 作池管理接口被 trainer 调用；`evaluate_deck()` 零实际调用。

**实战数值（seed_3911，证实奖励惩罚卡组增长）**：

| 楼层 | 敌人 | 回合 | 丢血 | combat_reward |
|---|---|---|---|---|
| 1 | Louse | 2 | 0 | +34.00 |
| 3 | JawWorm | 1 | 0 | +34.50 |
| 4 | AcidSlime_L | 29 | 32 | **−11.50** |
| 16 | TheGuardian | 49 | 20 | **−9.50** |

1 回合秒杀 = 高正奖励；长战斗 = 负奖励。任何让战斗变长的好卡都被直接惩罚。

### 1.4 探索 / 续训链　**[证实]**

- **熵轨迹**：batch_v15 起点 entropy=0.712（`sts_models/v8_ppo_batch_v15/v8_ppo_summary.json:31`）→ batch_v39 末端 0.019–0.026。clip_frac 0.073→0.001–0.004；approx_kl 0.003→<0.001。策略已几乎确定性。
- **熵系数**：固定 0.01（`v8/trainer.py:100`，用于 :620 `total_loss = policy_loss + value_coef*value_loss - entropy_coef*entropy`）。无 entropy floor、无自适应、无 KL 约束。
- **续训链**：v15（fresh）→ v16 → … → v39，每批 `--resume_from` 上批 final ckpt（各批 summary.json `resume_from` 字段 + git commit log 证实）。锁死的策略被一路继承。
- **skip 率**：14 个采样 episode 平均 SKIP 率 47.4%（含 Cleave / Whirlwind / Metallicize 等好卡）；act1 boss 局 deck 维持 11–15 张全基础卡（seed_3911 在 floor16 仅 11 张）。

### 1.5 已确认的小 bug　**[证实]**

- **指标命名误导**：`tools/v8_ppo_train.py:1088,1143` `beat_boss = bool(runner.game_won)`——字段名叫 beat_boss，实际算的是"全局通关"。所以训练日志 `beat_boss_count` 恒为 0（because 没通关过），但 eval 的 `beat_boss_rate`（实为 act1_boss_beat）显示 0.26–0.40。两者来自不同代码路径，长期误导趋势判断。
- **NEOW 选项无语义**：`v8/action_space.py:280` NEOW token 只编码 `NEOW:choice={idx}`，不注入祝福内容文本（对比 :290-293 CARD_REWARD 已注入 card_name）。14 局采样全部 NEOW choice=0——同一个"虚拟序号"在不同局对应不同真实选项，模型无法区分，理性塌到 idx=0。

---

## 第二部分　归因核实：哪些证实、哪些被推翻 / 修正

| 归因原结论 | 核实结果 | 证据 |
|---|---|---|
| 战斗由"搜索 + BC combat head"驱动 | **部分错**：是搜索 + **随机桩**，BC 权重根本没进搜索 | combat_net_wrapper.py:43-101 |
| AcidSlime_L 打 29–48 回合 | **基本对但偏窄**：实测 24–50，均值 ~33 | v39 日志 11 个样本 |
| 搜索预算被砍是 bug 嫌疑 | **推翻**：elite 250/上限1000ms、boss 500/上限10000ms 数字属实，但是 commit 05894c4 的**性能优化**（v14 实测 per-turn 1s 与 5s 同 win/dmg），非 bug | env.py:108-112 + 05894c4 commit msg |
| BC head 毒液处理质量不足 | **推翻**：引擎毒液（每回合 −1 stack）和格挡逻辑都正确；且 BC head 本就没被用，无从谈"质量" | combat_engine.py:600-606 |
| 奖励惩罚卡组增长 | **证实** | reward.py:66-111 + seed_3911 数值 |
| entropy collapse 0.71→0.02 | **证实** | summary.json + trainer.py:100 |
| 续训链繁殖锁死 | **证实** | 各批 resume_from + git log |
| beat_boss 命名 = game_won | **证实** | v8_ppo_train.py:1088,1143 |
| NEOW token 无语义、全选 idx0 | **证实** | action_space.py:280 + 14 局 trace |
| v8_combat_head_v1.pt 训练数据过期（早于模拟器修复） | **证实但意义改变**：确实是 2026-05-07 产出（早于 5-22 修复），但**反正没被用**，所以"过期"不是当前痛点；当前痛点是"压根没加载" | 文件时间戳 + 加载路径 |

**最大修正**：归因把战斗低效归到"BC head 数据过期"，真相是"BC head 从没进过战斗搜索，搜索一直靠随机桩"。前者要"重新收数据重训 BC"，后者只要"把已有权重正确接进搜索"——后者轻得多、也更可能立竿见影。

---

## 第三部分　标准 RL 训练流程对照

每个主题：先标准做法（带可点来源），再我们偏在哪 + 代价 + 具体改法。证据档：**[共识]** 有论文/官方支撑；**[实践经验]** 社区广泛认可；**[推断]** 自有推理。

### 主题 1　奖励 shaping 的陷阱

- **[共识]** 我们的"丢血+回合数"扣分是经典 **specification gaming / reward hacking**。同类最有名案例：OpenAI CoastRunners 赛艇 agent 原地绕圈刷分、永不冲线。
  - DeepMind: https://deepmind.google/blog/specification-gaming-the-flip-side-of-ai-ingenuity/
  - 案例清单: https://vkrakovna.wordpress.com/2018/04/02/specification-gaming-examples-in-ai/
  - 综述 (Lilian Weng): https://lilianweng.github.io/posts/2024-11-28-reward-hacking/
- **[共识]** 唯一有理论保证不改变最优策略的 shaping 是 **Potential-Based Reward Shaping (PBRS)**：shaping 项须形如 `F = γΦ(s') − Φ(s)`（势函数差分）。我们的 `−hp_lost − 0.5·turns` 不是势差形式，数学上就属于"允许改变最优策略"的那类。
  - Ng, Harada, Russell 1999: https://www.semanticscholar.org/paper/Policy-Invariance-Under-Reward-Transformations:-and-Ng-Harada/94066dc12fe31e96af7557838159bde598cb4f10
- **改法**：把战斗内 hp/turns 扣分从上层奖励里拿掉，或改写成血量/楼层的势差形式；给负 shaping 封顶（reward capping）；拉大稀疏真实信号（通关/楼层）相对 dense 项的权重。

### 主题 2　探索崩溃 / entropy collapse

- **[共识]** PPO 的 entropy bonus 只是个二阶小正则项（β≪1），奖励一主导就被压没，于是熵崩溃。这跟 SAC 把熵写进目标（target entropy 自适应温度）不同。
  - https://www.emergentmind.com/topics/policy-entropy-collapse
  - SAC: https://spinningup.openai.com/en/latest/algorithms/sac.html
  - 自适应熵系数: https://arxiv.org/abs/2510.10959
- **改法**：entropy floor / 自适应熵系数（熵跌破初始熵的 ~30% 自动上调）；KL 早停 / KL 惩罚（SB3、Spinning Up 的 PPO 内建 target_kl）。
- **[推断] 顺序**：奖励错配没修之前，强行顶高熵只是让模型"更随机地探索一个被错误奖励引导的空间"，治标不治本。先修奖励，再用熵 floor 保探索。

### 主题 3　续训 vs fresh start

- **[共识]** 已塌进局部最优 + 熵≈0 的 ckpt，warm-start 续训通常救不回来，常不如完全重置。on-policy 可塑性研究（NeurIPS 2024）观测到 warm-start 模型平均回报明显劣于"每轮全重置"。
  - https://proceedings.neurips.cc/paper_files/paper/2024/file/ce7984e36d58659211a8dc7d5457cd6f-Paper-Conference.pdf
- **改法**：这次该 fresh start（或回滚到熵未塌的早期 ckpt），不是续训。续训规范应加触发条件：**上批 final 熵 < 阈值（如 0.1）→ 强制 fresh start / 回滚**。

### 主题 4　分层 RL（上层 RL + 下层非 RL 战斗）

- **[共识，反直觉]** OpenAI Five 和 AlphaStar 这两个最复杂的游戏 agent **都没用分层 RL**，而是原始动作端到端 + 大规模 + self-play。OpenAI 原话：真正需要的是 scale，不是分层这类复杂算法思想。
  - OpenAI Five: https://en.wikipedia.org/wiki/OpenAI_Five ; https://arxiv.org/pdf/1912.06680
  - AlphaStar: https://deepmind.google/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning/
- **[共识]** 真要分层，标准范式（FeUdal Networks / options）里**下层是和上层联合训练、可被上层奖励塑形的**，不是冻结黑盒。
  - FeUdal: https://arxiv.org/abs/1703.01161
  - 综述: https://thegradient.pub/the-promise-of-hierarchical-reinforcement-learning/
- **我们的形态**：上层 PPO + 下层"冻结(其实是随机桩)的战斗"。这不是分层，是解耦，而且是风险最高的一种。**[推断]** 这正好闭环解释了 reward hacking：上层在为一个它无法控制、且会因它的决策（换 deck）而漂移退化的下层买单。
- **改法**：上层奖励与下层解耦（呼应主题1）；若用 BC，方向应是"BC 初始化 + RL 继续训同一网络"，而非"BC 永久外包"。

### 主题 5　课程学习

- **[共识]** 长程稀疏奖励的标准解法之一是 **dense→sparse 课程**（早期 dense 引导，逐步退火到稀疏真实奖励）+ **难度课程**（先降稀疏度，逐步加难）。AlphaStar 的"监督引导 + League 逐步加强对手"本身就是课程。
  - https://arxiv.org/html/2603.21972v1
  - HACL: https://jin-s13.github.io/papers/Neurocomputing_2019_Jiang.pdf
- **我们**：全程同一难度（完整通关）+ 全程同一 dense 奖励，无课程。通关信号极稀疏，4900 局 0 正样本。
- **改法**：先把"打到 act1 boss"当第一阶段 episode 终点 + 大奖励（我们已接近，a1 均值 32%），稳定后延伸到 act2、通关；dense shaping 随训练衰减逼近纯稀疏。

### 主题 6　评估方法论

- **[共识]** 种子太少不可信（Henderson et al.：所审深度 RL 论文全部 ≤5 seed）；pilot 建议 ≥20 样本估方差。确定性 eval（argmax）能暴露"反复选同一动作卡死"这类病（我们的 Mysterious Sphere 死循环就是 argmax 触发、sampling 绕过）；标准是确定性 + 采样双轨都看。
  - https://arxiv.org/pdf/1806.08295
- **我们**：30 seed 确定性 eval（种子数达标）。坑：指标命名 bug；长期盯代理指标 a1_boss_beat"创新高"，而真实目标 won_game 恒 0，易误判"在进步"。
- **改法**：指标分层呈现，**永远以 won_game 为北极星**，代理指标只当过滤器；eval 输出固定纳入 entropy；确定性 + 采样双轨。

### 主题 7　案例对照

| 维度 | OpenAI Five | AlphaStar | 独立 StS RL 项目 | 我们 (V8 RL) |
|---|---|---|---|---|
| 算法 | PPO 端到端原始动作 | 监督引导 + off-policy RL + League | Double DQN(micro)+NN模拟(macro) | PPO meta + 搜索/随机桩战斗 |
| 奖励 | dense + team spirit 0.2→0.97 退火 | 主要靠胜负 + League | **极简：战斗只给结束剩余血量；macro 只用最终楼层** | **重 dense shaping，hp/turns 扣分压倒一切** |
| 探索 | 大规模 self-play | latent 编码人类开局引导 | 自定义加权随机 | 固定 entropy bonus，已崩到 0.02 |
| 课程 | 无（靠 scale） | 监督引导 + League 逐步加强 | **六阶段难度课程** | 无 |
| 规模 | 256×P100+128k CPU，180年/天 | 未公开 | 业余单机 | 业余单机 ~4900 局 |
| 结果 | 击败世界冠军 | Grandmaster | **100局4通关、累计赢100+** | **通关 0** |

来源：OpenAI Five https://en.wikipedia.org/wiki/OpenAI_Five ；AlphaStar（同上）；独立 StS RL 项目 https://milesoram.github.io/slay-the-spire-ml-project.html （个人博客，结果作者自报、未经同行评审，仅作"StS 上奖励极简+课程可行"的存在性证据）。

**最高性价比借鉴**：那个业余 StS 项目作者一句话——**"Slay the Spire 里几乎每个'坏'动作都有让它有时正确的 caveat，所以显式 reward shaping 适得其反。"** 他靠"奖励极简 + 难度课程 + 自定义探索"在远小于我们的算力下拿到真实通关，正好命中我们的三个病。

---

## 第四部分　根因排序与因果链

```
[根因0] 战斗搜索叶子评估 = 70%随机噪声 + 30%手写启发
            │  (combat_net_wrapper 随机桩从未加载训练权重)
            ▼
[结果1] 战斗严重低效：小怪 24–50 回合，每场白扔 30–60 血
            │
            ▼
[结果2] 在"丢血+回合数扣分"的奖励下，拿任何好卡都让战斗更长 → 扣分更多
            │  (reward.py: −hp_lost −0.5·turns，无封顶，量级压过 +30 胜利奖励)
            ▼
[结果3] "不拿卡 / 维持最小 deck" 成为奖励地形上的局部最优 → 模型理性 SKIP 一切
            │
            ▼
[结果4] 采不到"拿卡→赢"的轨迹 → 熵从0.71崩到0.02 → 策略确定性锁死
            │  (固定 ent_coef=0.01，无 floor/自适应/KL)
            ▼
[结果5] 续训规范强制从上批 ckpt 续 → 锁死策略一代代繁殖 (v15→v39)
            │
            ▼
[终态] 训练 ~4900 局，act1 boss ~30%，act2 boss/通关 恒为 0
```

**关键判断**：根因0（战斗随机桩）是这条链的源头，也是最被低估的一环——它一直被当成"BC head 在跑"。修复优先级上，它和奖励错配（结果2）是两个必须先动的杠杆；探索（结果4）和续训（结果5）的修法在前两者修好之前都只是治标。

---

## 第五部分　待决策清单（映射最初的 16 项需求）

下列**不下结论**，只列"现在掌握的事实 + 标准做法指向"，留到设计阶段逐项拍板。

**核心设计**
1. **A 选卡多维评分（输出/防御/draw/energy/power）**：设计文档写了两个月未实现。标准做法警告显式多维 shaping 极易 reward hacking（主题1）；同行 StS 项目用"奖励极简"反而成功。→ 待定：是否真要做 5 维，还是走极简 + 课程。
2. **B 其他决策评估（路线/事件/商店/休息/升级）**：文档只写了 MAP="战损+牌组强度"，其余未设计。当前实现是写死的小 bonus（事件+5/商店+3/休息+2/宝箱+3）。
3. **"牌组强度"怎么算**：设计文档提了但从未定义算法；模拟战(deck_evaluator)已弃用。→ 待定：固定属性+协同打分 / 别的。
4. **多维评分怎么合成给 PPO**：文档未涉及。选项：加权和单标量 / 多 value head / 5 维分别 RL。

**A 实现细节**
5–8.（核心卡定义 / draw 成功判定 / energy 成功判定 / power 量化）——全部依赖 1、3 先定框架，暂挂。

**训练稳定性**
9. **entropy collapse 防护**：标准做法 = entropy floor / 自适应熵 / KL（主题2）。
10. **fresh start vs 续训**：文献 + 我们的熵=0.02 都指向**这次该 fresh start 或回滚到熵未塌 ckpt**（主题3）；combat 权重接法见根因0。

**已知 bug**
11. **NEOW token 没注入选项内容**：已证实（action_space.py:280），修法明确——注入祝福文本，与 CARD_REWARD 一致。
12. **beat_boss_count 命名误导**：已证实（v8_ppo_train.py:1088,1143），拆成 a1_boss_killed / a2_boss_killed / won_game。
13. **战斗 BC head 问题**：真相是**随机桩从未加载训练权重**（根因0），不是"数据过期"。先把已有权重正确接进搜索，再决定要不要重训。

**工程基础**
14. **工作流脚本 tools/v8_training_workflow.py**：未真跑过完整 train→attribute→fix；"自动修复"只登记 finding 不真改代码。→ 待定。
15. **实机测试**：2 个月只跑过 1 次（卡在 Living Wall），修 subscreen 后未复测；multi-game 循环未验证。→ 待定。还**缺一个隔离战斗测试脚本**（拿固定好卡组单独跑 AcidSlime_L），用来量化根因0 修复前后的回合数/血损。
16. **unstaged 改动**：tools/v8_ppo_train.py 字段改名 + web/ 一堆未 commit。→ 待定清理。

---

## 附录　关键证据索引（file:line）

- 战斗叶子评估混合：`external/StSRLSolver/.../turn_solver.py:354-362`（0.7 neural + 0.3 heuristic）、:1131-1144（neural_eval 路由）
- 随机桩：`v8/combat_net_wrapper.py:43-52`（stub 注释）、:56-62（随机 Linear）、:100-101（无 load）
- 战斗权重只进上层模型：`tools/v8_ppo_train.py:803`、:126-147（load_combat_head → V8Model）
- env 传 wrapper 给搜索：`v8/env.py:493-495`
- 搜索预算：`v8/env.py:108-112`；优化来由 commit `05894c4`
- 奖励：`v8/reward.py:52-63`（系数）、:66-111（combat）、:135-171（step）、:174-180（final）；节点奖励 `v8/env.py:285-328`
- 熵：`sts_models/v8_ppo_batch_v15/v8_ppo_summary.json:31`（0.712）、batch_v39 同字段（0.019–0.026）；熵系数 `v8/trainer.py:100`、loss :620
- 指标命名：`tools/v8_ppo_train.py:1088,1143`
- NEOW token：`v8/action_space.py:280`（无语义）、:290-293（CARD_REWARD 注入）
- 引擎毒液/格挡正确：`external/StSRLSolver/packages/engine/combat_engine.py:600-606`、`.../calc/damage.py:158-198`

---

*排查日期 2026-05-29。下一步：基于本文进入奖励 / 评估机制的重设计讨论（先动根因0 与奖励错配两个杠杆）。*

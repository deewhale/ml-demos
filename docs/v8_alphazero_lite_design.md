# V8 设计：AlphaZero-lite + bottled_ai teacher + 人类点评 fine-tune

**版本**：v1 草案
**日期**：2026-04-24
**目标**：用 ML 通关 A0 Ironclad（Phase 1），长期扩展全角色 A20 100 连胜
**信心**：**中低**。思路合理但风险点多。
**相关文档**：Notion 协作空间「STS机器学习」对应本文档，两者内容同步；本文档为 repo 内权威版本。

---

## 1. 目标与共识

### 1.1 长期愿景

**用 ML 打通 STS 全角色 A20 100 连胜**。
- 4 个角色（Ironclad / Silent / Defect / Watcher）
- 最高难度 A20
- 稳定性：100 连胜而非单次通关

### 1.2 当前阶段（Phase 1）目标

A0 Ironclad，30 seed 模拟器 win rate ≥ 20%。

### 1.3 核心约束

1. **不走纯 RL**（V3-V6 四次失败证据充分，样本效率是数学问题不是数据量问题）
2. **监督学习路线**：BC + MSE + DPO 全监督，无 policy gradient
3. **复用 V7 TurnSolver 搜索骨架 + V6 统一效果编码**
4. **模拟器训练**（StSRLSolver），真游戏只做最终验证
5. **Class-agnostic encoder + class-specific handler**

### 1.4 非目标

- 不做 MCTS / self-play policy improvement
- 不做真游戏部署（MVP 阶段）
- 不做 A1+ ascension（Phase 1）
- 不做多角色并行（先 Ironclad）

---

## 2. 与 V3-V7 的关系

| 版本 | 路线 | 结果 | V8 处理 |
|---|---|---|---|
| V3 DQN | 纯 RL | floor 7-12 plateau | 丢 |
| V4 PPO + Transformer | 纯 RL | floor 4.2（退化） | Transformer 设计可参考 |
| V5 TurnSolver + Agent | 搜索+RL | 信号稀疏 | TurnSolver 骨架复用 |
| V6 统一编码 + PPO | 纯 RL | floor 6.1 plateau, win 0% | **编码层复用，训练层丢** |
| V7 搜索+手调 lex | 规则 | avg floor ~10 | 骨架复用，evaluator 被 NN 替代 |

**关键复用资产**：
- `data/sts_data.py` — 57 维统一效果编码
- `sts_agent_v6.py` — Transformer encoder（CTX_DIM=128, 3 层 4 头）
- `tools/comm_bridge.py` — CommunicationMod 接口
- `v7_bot.py` + `v7_strategy.py` — 策略 handler
- `v7_regression.py` — 回归测试框架
- StSRLSolver `turn_solver.py` — `neural_eval` hook 已预留

**明确丢弃**：
- `train_v6.py` PPO 训练器
- `pretrain_v6.py`（有 grad accum 假 batch bug）
- V7 的 `v7_evaluator.py` weighted sum（被 NN 替代）

---

## 3. 总体架构

### 3.1 Runtime（部署）

```
StSRLSolver / CommMod
      ↓
comm_bridge → CombatEngine
      ↓
TurnSolver（V5 骨架，自适应 DFS/BFS/Beam）
      ↓
  [leaf eval]   value_net(state) → [0,1] 胜率
  [action prior] policy_net(state) → π(a|s) top-K 剪枝
      ↓ 混合 0.7·neural + 0.3·heuristic
      ↓
策略层 Handler（Draft / Path / Choice / Boss Relic，硬编码）
```

### 3.2 Training（离线）

```
bottled_ai teacher（移植到 StSRLSolver） → (state, action, outcome)
      ↓
State Featurizer（统一效果编码）
      ↓
┌──────────────────────┬──────────────────────┐
│  Policy Head (BC)    │  Value Head (MSE)    │
│  CE loss on action   │  MSE on floor/57     │
└──────────────────────┴──────────────────────┘
      ↓
v7_regression 10 点不退化 → 30 seed 评估
      ↓
用户点评台 → preference pairs → Claude API 扩展 → DPO fine-tune
```

---

## 4. bottled_ai Teacher 方案

### 4.1 事实

- **bottled_ai 是 100% Python**（不是 Java），基于 CommunicationMod
- **Ironclad bot "RequestedStrike" A20 胜率 20%**（远低于 Watcher 52%）
- A0 胜率**未实测**，预期显著更高（敌人弱、HP 不 scale）
- 仓库结构：`rs/ai/requested_strike/` 是 Ironclad，`rs/calculator/` 是自己的模拟器
- License：待查

### 4.2 数据生成策略

**不跑真游戏**：把 bottled_ai Ironclad 策略**移植到 StSRLSolver 模拟器**。

理由：
- 真游戏一局 2-5 分钟，10k 局 = 数百小时
- 模拟器一局 3-6 秒，多进程跑 10k 局可在 8-10 小时完成
- 真游戏只在最终评估用

具体步骤：
1. 抄 `rs/ai/requested_strike/config.py`（DESIRED_CARDS、removal priority）到 `v8_teacher_ironclad.py`
2. 抄 `rs/common/comparators/common_general_comparator.py`（42 维 lex comparator）到 `v8_teacher_eval.py`
3. 替换 V7 `v7_evaluator.py` 的 weighted-sum 为 bottled_ai 风格 lex
4. 加 hook 记录每个决策点 `(state_tokens, legal_actions, chosen_action, top_k_scores)` → JSONL
5. 跑 10k seed，过滤保留 winning games + floor ≥ 8 的 losing games（滤噪）

### 4.3 不确定项

- StSRLSolver ≠ bottled_ai `rs/calculator/`（intent / power 实现有差异）。需 parity 测试
- Teacher A0 胜率未知——如果只有 40%，V8 上限就被锁死
- Watcher stance / Deva / mantra 在 bottled_ai 里是核心，Ironclad 侧要看是否有同等质量

### 4.4 数据量

- 10k 局 × ~50 策略决策 + ~150 combat 决策 ≈ 2M samples
- Policy BC 训练规模合适
- Value 训练：每局一个 episode 级胜负 + 决策步级软标签 `floor/57`，密度够

---

## 5. 模型设计

### 5.1 Encoder（共享 backbone，继承 V6）

- 3 层 Transformer × 4 头，CTX_DIM = 128
- 输入：Token 序列（MAX_SEQ_LEN = 91）
- 每 token 类型有独立的实体编码（57-66 维）→ 投影到 128
- 输出 256 维 shared_repr

**class-agnostic**：不带 class_id embedding。角色差异通过初始遗物/起手牌隐式表达。扩 Watcher/Silent/Defect 只需扩 teacher 数据，encoder 无改动。

### 5.2 Value Head

```
shared_repr[256] → MLP[128→64→1] → sigmoid → win_prob ∈ [0,1]
```

**训练目标**：`floor/57` 软标签（决策步级，信号稠密）
**备用**：硬标签 win/loss（episode 级）

Loss：MSE

### 5.3 Policy Head（Combat Only）

**Combat：Autoregressive**
```
card_head：shared_repr[256] → MLP → 11 class（10 手牌 + end_turn）
target_head：shared_repr + card_emb → MLP → 4 class（4 enemies）
```

**Draft / Path / Choice**：MVP 阶段不做 NN policy，仍用 V7 handler 硬编码。

Loss：Cross-entropy on teacher action，with legal mask

### 5.4 联合训练

```
L = L_policy + 0.5 · L_value + 0.01 · entropy
```

### 5.5 模型规模

| 组件 | 参数量 |
|---|---|
| Entity projection (57→128) | 7k |
| Transformer (3×4×128) | 600k |
| Shared repr MLP (128→256) | 33k |
| Policy heads (card+target) | 50k |
| Value head | 20k |
| **合计** | **~710k** |

比 V4 的 152k 大 4-5 倍。比现代 LLM 小得多，适合 10k 局数据规模。

---

## 6. 设计考量：Build 深度 vs 长特征

### 6.1 用户疑问（2026-04-24）

> "RL 里长特征对局部感知不明显，但肉鸽的 build 深度又很高，有点想不清楚了"

这个 tension 是真实的：STS 通关需要理解 deck 全局协同（build = poison / strength / exhaust / block 等方向），但传统 RL 对长 context + 稀疏 reward 学习效率极差。

### 6.2 V8 如何化解这个 tension

**V8 不是 RL，tension 大部分消失**：

| 场景 | RL（V3-V6） | BC（V8） |
|---|---|---|
| 信用分配 | 需从最终胜负反推 200+ 决策 | Teacher 每步给明确答案，无需反推 |
| 长 context | 注意力学不透（噪声 > 信号） | 梯度直接推动，注意力可快速学到 |
| 稀疏奖励 | 只有 episode 末尾有信号 | 每步都有 label，密度 ×200 |

**BC 下 tension 仅剩一处：Value Net**
- Value 预测 win prob 需要从 deck 推断 build 质量
- 10k 局 × 100 steps = 1M value pair，对 710k 参数模型刚够用

### 6.3 如果 V8 仍学不会 build（三个补丁方案）

**补丁 A：硬编码 build 类型特征**
- 用 `data/sts_data.py` 的 effect 维度自动分类 deck 类型（strength/poison/exhaust/block）
- 作为辅助高层特征塞进 state
- **不硬编码游戏知识**（基于已有数据）
- 工作量：~1 天

**补丁 B：分层模型**
- Deck encoder（选卡时运行一次，产 deck embedding）
- Combat decoder（每回合用 deck embedding + combat state 决策）
- 专注学两种 task，减少注意力学习负担
- 工作量：~1 周（大改）

**补丁 C：数据扩增**
- 同 deck 不同 seed 跑多次
- 强迫模型学 deck 本身价值（不是 seed 运气）
- 工作量：~2 天（改 data loader）

**触发条件**：Stage B 训完后 value prediction AUC < 0.7 或 30 seed 评估 < V7 baseline。先不加补丁跑 MVP，出问题再按 A → C → B 顺序加。

---

## 7. Dense Card Scoring（辅助特征）

### 7.1 动机

V5 失败的核心是策略层（选卡 / 选路）只有局末胜负稀疏信号，跨 50+ 决策无法学习。V8 的主路线是 BC + value net 提供 dense supervision，但**仍可补一层"每张卡每场战斗的可解释分数"**作为辅助特征：

- **NN 不用从零学卡牌价值**——用历史分数初始化
- **可视化与调试**——训练完看分数排序是否符合直觉
- **冷启动**——MVP 阶段无 NN 时纯分数已能做基本选卡决策

这不是替代 NN value/policy，是补充。

### 7.2 评分原则

按**卡牌实际机制**计算贡献，不用统一比例。

| 卡牌效果类型 | 评分公式 |
|---|---|
| 直接攻击（Strike, Bash, Heavy Blade） | `damage_dealt`（含打在 block 上的） |
| 直接防御（Defend, Iron Wave 防御部分） | `damage_blocked`（cap at incoming）+ bonus if HP unchanged this turn |
| 力量增益（Inflame, Demon Form, Flex） | `Σ (Δstrength × attack_multiplier × hits)` 对此 buff 生效后的所有攻击 |
| 敏捷增益（Footwork, Watcher Pray） | `Σ (Δdex × block_card_count)` 对此 buff 生效后的所有防御 |
| 抽牌（Dark Embrace, Battle Trance, Skim） | `Σ score(extra_card_played)` 对因此**额外**抽到并打出的卡 |
| 易伤施加方（Bash, Sword Boomerang +Vuln 部分） | `0.5 × subsequent_damage_to_target_during_vuln` |
| 虚弱施加方（Flash of Steel +Weak 等） | `0.25 × incoming_damage_prevented_due_to_weak` |
| 脆弱施加方（罕见，敌方常用） | `0.25 × incoming_block_reduction_due_to_frail` |
| 能量获得（Berserk, Ball Lightning） | `score(next_card_played) × proportion` |
| 多段攻击（Twin Strike, Pummel） | 每段独立按直接攻击算分 |
| 消耗类（Feed, Reaper 回血） | 包含 max_hp gain 或 heal 折算 HP 价值 |
| 不可分类（Apotheosis, Madness） | 后处理：被改造卡的 score 增量归还原卡 |

### 7.3 长尾压制

Reaper infinite / Body Slam infinite / Whirlwind X 费爆发等局面会让单卡分数异常高，主导 deck-level 平均。需要：

- `log(1 + raw_score)` 压尾
- 或 winsorize at 95th percentile
- 防止单局 outlier 影响卡牌价值评估

### 7.4 累积与聚合

- 每场战斗结束，把所有出过的卡的分数记入 `card_score_log`
- 多局累积后，每张卡得到：
    - `historical_avg_score`（平均贡献）
    - `historical_play_count`（出现次数）
    - `historical_score_var`（方差，高方差 = Discovery 这种随机产卡的卡）
- deck-level 强度可由这些聚合（如 `mean(card_avg_scores)` weighted by play frequency）

### 7.5 NN 输入扩展

现有 token 序列每张卡的 token 维度增加 4-6 维：

| 字段 | 含义 |
|---|---|
| historical_avg_score | 这张卡在当前 deck 上下文中的平均贡献 |
| historical_play_count | 出现次数（低 count 时表示数据稀疏） |
| historical_score_var | 分数方差（识别随机产卡） |
| score_relative_rank | 在当前 deck 中的相对排名 |

NN encoder 把这些和 57 维效果向量拼接，**不替换原编码**。

### 7.6 冷启动用法

MVP 阶段（NN 还没训好）时，可以用纯 card score 排序做选卡：

```python
def cold_start_draft(deck, candidates):
    return max(candidates, key=lambda c: predicted_score(c, deck))
```

注意：score 受 deck 上下文影响，单看 raw score 排序不准。需要按 deck 类型条件化（strength deck 给 +str 加权，poison deck 给 poison 加权）。

### 7.7 已知 limitation

- **Synergy attribution 在复杂情况下不完美**：Shapley 值更准但计算昂贵
- **随机产卡（Discovery, Havoc）方差大**：需要更多样本才稳定
- **依赖战斗中能正确归因**：模拟器需要 expose 每张卡触发的具体效果链
- 这是**辅助信号**，不是替代 NN value/policy

### 7.8 实施工作量

| 步骤 | 工作量 | 依赖 |
|---|---|---|
| 1. 战斗后 hook：扫描每张卡的效果链 | 1-2 天 | StSRLSolver 的 effect log |
| 2. 评分公式实现（每类 1-2 行） | 1 天 | 步骤 1 |
| 3. 累积存储 + token 维度扩展 | 0.5 天 | 步骤 2 |
| 4. NN 训练接入新维度 | 0.5 天（已有 encoder） | 步骤 3 |

总计约 3-4 天，可与 Stage A（teacher 移植）并行。

### 7.9 与 §6（build 深度 tension）的关系

§6 提到三个补丁（A 硬编码 build 类型 / B 分层模型 / C 数据扩增）。本节是**补丁 A 的精细化版本**：
- 补丁 A 原方案是按 effect 维度自动分类 deck（strength/poison/exhaust/block）
- 本节进一步给每张卡打分，比"是不是 strength deck"更细粒度
- 两者可叠加使用

---

## 8. 训练 Pipeline

### 8.1 阶段划分

| Stage | 任务 | 预估工作量 | 预估机时 | DoD |
|---|---|---|---|---|
| A | bottled_ai Ironclad 移植 | 3-5 天 | — | 10 seed Ironclad 无 crash，avg floor ≈ bottled_ai A0 基线 |
| A' | 10k 局数据生成 | 1 天 | 10 小时 | JSONL ≥ 1M samples，过滤后 ≥ 500k |
| B | Policy+Value 联合训练 | 3-5 天 | 6-12 小时 | validation CE < 1.5，value AUC > 0.7 |
| C | V7 集成 + 30 seed 评估 | 2-3 天 | — | regression 10/10 通过，30 seed win ≥ 10% |
| D | 点评台 UI + DPO | 4-6 天（可并行 A-C） | — | 本地启动，10 case 可标完 |
| D' | DPO fine-tune | 3-4 天 | 2-4 小时 | win 相对 Stage C 涨 ≥ 3pp 或不退化 |
| E | 迭代 | 每轮 1 周 | — | 收敛或明显平台期 |

**关键路径**：A → A' → B → C（约 2 周）。D 可与 A-C 并行。

### 8.2 训练细节

- **真 minibatch**（修复 pretrain_v6.py 的 grad accum 假 batch bug）
- batch size 256
- 10 epoch × 2M samples ≈ 80k steps
- Optimizer：AdamW, LR=1e-4
- Scheduler：cosine decay

### 8.3 Regression 接入

V7 已有 `v7_regression.py`（10 决策点）。V8 加入 `neural_eval` 后：
- 运行同样 10 点，对比 `chosen_action`
- 变化 → 打印 diff，manual review（不自动 fail）
- top-3 都不含原 chosen → 标 regression，阻断

### 8.4 部署（Stage C）

- 实现 `V8Evaluator` 类，hook 到 `TurnSolver.neural_eval`（turn_solver.py:354）
- `policy_prior` 做 top-5 action 剪枝
- 混合策略：`score = 0.7 × neural × 100 + 0.3 × heuristic`（已预留）
- 保证 lethal 剪枝（`best_score >= 1e6`）不被破坏

---

## 9. 人类点评闭环（Stage D）

### 9.1 UI 技术栈

- FastAPI + 静态 HTML/CSS + vanilla JS
- 本地 localhost 运行
- 中文界面
- < 300 行前端代码

### 9.2 Active Learning 筛选

每局只 surface 关键决策（5-10 个点/局，10 局 = 50-100 点/批）：
- Policy uncertainty: max(π) < 0.4
- Value drop: 决策前后 value 跌 > 0.15
- HP catastrophe precursor: 下 3 回合 HP 跌 > 30%
- Rare rooms: Elite / Boss / Shop / 特殊事件

### 9.3 交互形式

**主：偏好对（Preference Pair）**
- 展示当前 state、AI 选的 action、2-3 个 top-k 替选
- 用户点选："AI 对" / "A 更好" / "B 更好" / "都不好"
- 可选填中文点评

**辅：中文点评**
- 用户写自然语言解释 → Claude API 解析为结构化信号

### 9.4 数据扩展（Label Propagation）

- Claude API 读用户偏好对 + 点评
- 扩展到同 state 其他 action 对、相似 state 的 action 对
- 扩展倍率：1 条原始 → 3-5 条扩展
- 扩展 pair 降权（loss_weight=0.3）
- 每轮抽样 20% 扩展 pair 让用户复核

### 9.5 门槛

攒到 **300-500 条原始偏好对**（≈ 1000-2000 条含扩展）再启动 DPO。

### 9.6 DPO 细节

- Reference model：冻结的 BC-trained policy
- Loss：`L_DPO = -log σ(β·(log π_θ(a_w|s) - log π_θ(a_l|s) - log π_ref(a_w|s) + log π_ref(a_l|s)))`
- β = 0.1
- 只 fine-tune policy head 最后两层，**冻结 encoder 和 value head**
- 3-5 epoch，LR=1e-5

---

## 10. 评估方案

### 10.1 三级指标

| 层级 | 频率 | 内容 | 断路器 |
|---|---|---|---|
| Regression | 每次改动 | v7_regression.py 10 点 | 退化即阻断 |
| Simulator | 每 checkpoint | 30 seed A0 Ironclad | 连续 2 次退化即回滚 |
| Live game | 重大版本 | 3-5 局真实游戏 | 崩溃即回滚 |

### 10.2 基线对比

| 版本 | Avg Floor | Win Rate |
|---|---|---|
| V6 PPO | 6.1 | 0% |
| V7 evaluator（当前） | ~10 | ~5% |
| **V8 Stage B 目标** | ≥ 10 | ≥ 10% |
| V8 Stage D 目标 | ≥ 12 | ≥ 20% |

**断路器**：Stage B 做不到 ≥ V7，不上 Stage C/D。回头检查 teacher 数据质量、模型容量、编码完整性。

---

## 11. 风险登记

| 风险 | 可能性 | 影响 | 缓解 |
|---|---|---|---|
| bottled_ai Ironclad A20 仅 20%，A0 上限未知 | 中 | 高 | Stage A 先跑 30 seed 实测 teacher A0 表现 |
| StSRLSolver ≠ bottled_ai simulator | 高 | 中 | Parity 测试（Stage A 必做） |
| NN leaf eval 破坏 TurnSolver lethal 剪枝 | 中 | 高 | 混合策略 0.7·neural + 0.3·heuristic |
| 模型学不到 build 逻辑 | 中 | 高 | 补丁 A/B/C（§6.3） |
| 57 维效果编码对 Ironclad 够用 | 低 | 中 | Ironclad 机制比 Watcher 简单 |
| pretrain_v6 假 batch bug 传染 v8_trainer | 已确认 | 中 | 从头写真 minibatch |
| DPO 300 条不够 | 中 | 中 | 先攒到 500-1000 再启动；Claude 扩展加码 |
| Claude label propagation 错标 | 中 | 低 | 降权 + 抽样复核 |
| MPS 推理延迟 | 低 | 中 | 实测。700k 参数 MPS 应 < 10ms/eval |
| bottled_ai license | 未查 | 中 | Stage A 第一步查 LICENSE |

---

## 12. 开放决策

- [ ] Teacher 是否 ensemble（bottled_ai + V7）？当前决策：**只 bottled_ai**
- [ ] Value target 软/硬标签？当前决策：**软 `floor/57`**
- [ ] UI 与训练串行还是并行？当前决策：**串行**（先看 BC 效果）
- [ ] 第二轮 DPO 的 reference model：初始 BC 还是上一轮 DPO 输出？待定
- [ ] bottled_ai license 允许什么程度复用？Stage A 第一步查
- [ ] Stage B 达 ≥ V7 的断路器阈值：win rate 多少才算通过？建议 ≥ 8%（V7 ~5%，考虑噪声）

---

## 13. 代码结构（实施时创建）

| 文件 | 作用 | 状态 |
|---|---|---|
| `v8_teacher_ironclad.py` | 移植 bottled_ai Ironclad config | Stage A |
| `v8_teacher_eval.py` | 移植 bottled_ai lex comparator | Stage A |
| `v8_data_generator.py` | 10k 局数据生成脚本 | Stage A' |
| `v8_model.py` | Policy + Value net 架构 | Stage B |
| `v8_trainer.py` | 真 minibatch 训练循环 | Stage B |
| `v8_evaluator.py` | 接入 TurnSolver.neural_eval | Stage C |
| `v8_ui_server.py` | FastAPI 点评台后端 | Stage D |
| `ui/v8_critique.html` | 点评台前端 | Stage D |
| `v8_dpo_trainer.py` | DPO fine-tune | Stage D' |
| `docs/v8_alphazero_lite_design.md` | 本文档 | 现行 |

---

## 14. 与 Notion 文档关系

Notion「STS机器学习」是协作者入口，面向上手 + 进度跟踪。
本文档（repo 内）是技术权威版本，面向实施细节。
两处内容变更应同步，避免漂移。

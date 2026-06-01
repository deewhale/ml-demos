# V8 实现设计

记录于 2026-05-06。

**前置阅读**：`docs/v8_design_principles.md`（用户 3 点设计原话 + 工作方法论）。本文档是基于设计原则的具体工程实现方案。

## 实施流程（强约束）

```
完整设计 → 你 review
        ↓
实现（写代码）
        ↓
Smoke 20 局（验证工程，2-3h）
        ↓
**强约束 A：Smoke 必须通过**
   - 无 crash / 数据流对 / 所有 phase 覆盖 / 性能 OK
   - 任一 fail → 停下查 bug，不进真训练
        ↓
你 review smoke 结果
        ↓
**强约束 B：真训练（83h）只能用户启动**
   - Claude 不主动启动真训练
   - 用户明确说"开始真训练"才跑
   - 每次重启 / 长跑都需要用户授权
```

每个组件含**自检**（对照设计原则）和**特殊情况说明**。

---

## 1. State Encoder（model 看什么）

**字段清单**（信息对齐人类玩家）：

| 类别 | 字段 |
|---|---|
| 数字状态 | HP / max_HP / floor / act / gold / potions（数量+类型）|
| 牌组 | 完整 deck（所有卡 + 升级状态）|
| 遗物 | 所有 relics |
| **完整 act 地图** | 整张 17 层图 / 节点类型 / 连接关系 / 当前位置标记 |
| 牌组强度 | 上次 evaluate 的 4 维数字（dmg_dealt / dmg_taken / turns / win_rate）|
| 战斗内额外（仅战斗 phase）| 手牌 / 抽牌 / 弃牌 / 消耗 / 能量 / 敌人 HP & intent |

**特殊情况**：
- Act 切换（floor 17 → 18）map 重置成新 act
- 战斗 phase 字段在元决策 phase 设为空 / mask（避免假数据）
- ? 节点未访问时只暴露 room type，不暴露具体事件名（信息对齐玩家）

**自检**：
- ✅ 包含牌组强度 4 维（用户设计核心）
- ✅ 不依赖 v8_strategy
- ✅ 信息对齐玩家（看完整 map，不超出玩家可见）
- ✅ 支持路线 long-term planning

---

## 2. Action Space（model 输出什么）

不同 phase 动作不同：

| Phase | Action |
|---|---|
| MAP | 从可选下层 path 节点中选 idx（变长 1-4）|
| NEOW | 从 4 个 blessing 中选 idx |
| CARD_REWARDS | 3 张候选 + skip = 4 选 1 |
| EVENT | 从事件选项中选 idx（变长）|
| SHOP | 从买卡 / 买药 / 删卡 / 买 relic / leave 中选 idx |
| REST | 选 rest 或 smith 哪张卡升级 |
| TREASURE | 开 / 不开 |
| COMBAT（每回合每步）| 从手牌选卡 + target，或 end_turn |

**架构**：Pointer network — 每个 phase 输入 (state, available_actions list)，输出从 list 选 idx + 概率分布。变长 actions 用 mask。

**特殊情况**：
- COMBAT 一个 turn 多步（出多张卡），每步独立调 model
- skip CARD_REWARDS 是有效 action（不强制选）
- 不合法 action（hand 里没那张卡 / 能量不够）由 environment mask 掉，model 不会看到

**自检**：
- ✅ 每个 phase 都用 pointer 架构（变长输入兼容）
- ✅ 不硬编码 STS 卡库（卡用 token embedding）
- ✅ 不复用 v8_strategy 启发式映射

---

## 3. Model 架构

**单个 model，双 head**（共享 encoder）：

```
Input: state (含 phase 标识)
        ↓
[Shared State Encoder]
        ↓
state_vec (~256 维)
        ↓
   ┌────┴────┐
   ↓         ↓
[战斗 Head]  [元决策 Head]
战斗 phase   非战斗 phase
   ↓         ↓
出牌 + target  从可选项选 idx
```

**Encoder 内部**：
- 卡 / 遗物 / 药水：token embedding + set encoder（mean-pool）
- 数字状态：MLP
- Map：节点 embedding + 简单图编码（每节点 + 邻接关系，类似 GNN 一层）
- 牌组强度 4 维：直接 concat
- Phase one-hot：concat
- 全部 concat → MLP → state_vec

**Head**：
- 战斗 head：state_vec + hand 卡 token + 敌人 token → pointer over hand × targets
- 元决策 head：state_vec + available_actions tokens → pointer over actions

**Value head**：state_vec → MLP → scalar（critic for PPO）

**参数量估计**：~500K-1M（单 Mac MPS 训得动）

**特殊情况**：
- 战斗内 model 推理时还会被搜索调用（model 给 priors 帮搜索剪枝）—— 见组件 6 预训练
- 战斗外 model 直接出动作（不需要搜索）

**自检**：
- ✅ 共享 encoder（用户最早说"一个 model"）
- ✅ 战斗 + 元决策共一个 model
- ✅ 不引入 v8_strategy 概念

---

## 4. Reward Function（分层评估）

按你 3 点原话**分层评估**，不是单 reward 反推：

### 4.1 路线 reward（每个 MAP 决策）

走到下一节点后立刻给 reward：
```
r_path = w1 × Δ牌组强度 - w2 × Δhp_loss + w3 × 节点收益
```

- `Δ牌组强度`：到达节点 + 完成该节点活动后，牌组强度 4 维数字加权和的提升
- `Δhp_loss`：到达过程中（含战斗）损失的 HP 比例
- `节点收益`：?事件给好东西、$节点买到好卡、R 升级 → 立即奖励

### 4.2 选卡 reward（每个 CARD_REWARDS 决策）

边际贡献：
```
r_card_reward = 牌组强度(deck + chosen_card) - 牌组强度(deck)
```

- 选 skip 时 reward = 0（保持 deck 不变）
- 这是你的 leave-one-out insight 直接落地

### 4.3 战斗内（不用 RL，用 imitation learning）

战斗内不给 step-level reward。Model 学搜索的出牌（搜索作老师），用 cross-entropy loss 直接监督学搜索选的 action。**这是预训练阶段的事**（见组件 6）。

战斗结束时给 outcome 反传到**元决策 reward**：
```
战斗结束后：r_path 中的 Δ牌组强度可能受这场战斗 deck 调整影响（如得到新卡），自然反映到下一次评估
```

### 4.4 整局最终 reward

```
r_final = game_won × 100 + final_floor × 1
```

- 通关大 bonus（model 知道赢是终极目标）
- final_floor 兜底（避免 0 winning data 时 model 完全没信号）

### 4.5 PPO 总 reward

每个元决策 step 的 reward = 局部 + γ × 未来 advantage（PPO/GAE 标准做法）。

**特殊情况**：
- 牌组强度评估几秒一次，可能 path 决策暂时没新评估 → 用最近一次缓存的
- 战斗内出 bug（搜索 timeout / 战败）时 r_path 仍正常给（hp_loss 大就负 reward）

**自检**：
- ✅ 路线用"战损 + 牌组强度"评估（用户原话第 1 点）
- ✅ 选卡用 leave-one-out 边际贡献（你的 insight）
- ✅ 战斗内 model 学搜索（用户原话第 2 点）
- ✅ 不是单 game_won 反推
- ✅ 不依赖 v8_strategy 启发式

---

## 5. 牌组强度评估接口

```python
def evaluate_deck(deck, relics, hp, max_hp, act) -> dict:
    """
    让 StSRLSolver 拿当前牌组打 3 个标准敌人，每个 3 次取平均。
    内部：cache by deck_hash，避免重算。
    """
    return {
        'damage_dealt': float,
        'damage_taken': float,
        'turns_to_win': float,
        'win_rate': float,
    }
```

**标准敌人**：
- act1：cultist（弱普通）+ Lagavulin（中精英）+ Hexaghost（boss）
- act2 / act3：用户后续训练时再扩展（先 act1 跑通）

**触发时机**：
- 每场战斗结束后调用（post-battle check，对应用户原话"战斗之后"）
- 选卡前评估候选每张：`evaluate_deck(deck + card)` ×3
- Cache by `(sorted_deck_tuple, sorted_relics_tuple, hp_bucket)`

**性能**：每次调 ~5s（3 敌 × 3 次 × 5s budget），act1 一局 ~10 次评估 = ~50s/局开销。

**特殊情况**：
- act 切换时 cache 清空（标准敌人换了）
- 评估失败（StSRLSolver 异常）回退到上次结果，不让训练挂

**自检**：
- ✅ 输出 4 维数字（你确认的设计）
- ✅ 用 search 实战 simulate（不是 static score）
- ✅ 自动反映运转 / 加费 / power 卡间接价值
- ✅ Cache 控制 overhead

---

## 6. 预训练（你说"可以但不确定有效"，加进去 smoke 阶段验证）

**只预训练战斗 head**，元决策 head 直接 RL。

**预训练流程**：
1. 让 StSRLSolver 搜索自己跑 act1 战斗 self-play
2. 起始 deck 用 Ironclad 标准 + RNG 加 5-15 张随机 act1 卡（保持 deck 多样性，**state 分布干净**）
3. 收 ~5000 场战斗的 (state, search_action) ≈ 50000 条数据
4. 训 model 战斗 head（cross-entropy on action） 5-10 epoch
5. 战斗 head freeze 一段时间，先 RL 训元决策 head

**Smoke 验证**：smoke 阶段对比"加预训练 vs 不加"两次跑 20 局，看哪个 reached_boss 更高。如果差不多，简化掉。

**特殊情况**：
- 预训练数据是搜索给的（不是启发式），符合"搜索作老师"原则
- state 分布用 RNG 多样化 deck 控制脏度

**自检**：
- ✅ 用搜索作老师（用户原话）
- ✅ 不复用 v8_combat_actions 脏数据（重新跑）
- ✅ smoke 验证有效性再决定要不要

---

## 7. RL Trainer

**算法**：PPO（成熟、稳定、单卡跑得动）

**核心组件**：
- Rollout buffer（收 N 个 episode trajectory）
- GAE advantage 估计（γ=0.99, λ=0.95）
- Clipped objective（ε=0.2）
- Value loss（MSE）
- Entropy bonus（防过早 deterministic）

**训练频率**：每 32 episodes update 一次 model

**特殊情况**：
- 元决策 step 数变长（一局 30 个左右）：rollout buffer 按 episode 存
- 战斗内 model 在 RL 阶段 frozen（只元决策 RL）—— 战斗能力靠预训练 + 搜索辅助

**自检**：
- ✅ 标准 RL 不引入新概念
- ✅ 兼容预训练好的战斗 head（可 freeze）
- ✅ PPO 是 policy-based，能处理变长 action space

---

## 8. Episode Loop

一局 STS run：

```
start
  ↓
NEOW（初始奖励）
  ↓
floor 1-17（act1）每层：
  - MAP 决策 → 走到下一节点
  - 节点活动：
    * COMBAT：搜索打（model 给 prior）→ 战斗结束触发 evaluate_deck
    * EVENT：model 选 → 触发奖励
    * SHOP：model 选 → 触发买
    * REST：model 选 → rest / smith
    * TREASURE：开
  - CARD_REWARDS（COMBAT 后）：model 选 → 触发 evaluate_deck
  ↓
floor 16 boss：搜索打
  ↓
end：game_won / game_lost
  ↓
计算 trajectory rewards + 加入 buffer
```

**特殊情况**：
- 战斗内是搜索主导，model 只观察 + 给 prior（不写入 RL trajectory）
- 元决策 phase 才写入 RL trajectory（PPO 学元决策）

**自检**：
- ✅ 整局完整 episode（act1 即可，act2/3 后续）
- ✅ 战斗内不参与 RL（用户原话：搜索辅助 model；model 在战斗外 RL）
- ✅ 评估机制嵌入 episode loop（post-battle / pre-card-reward）

---

## 9. Smoke 20 局流程

**目的**：验证 pipeline 通 + 无 bug，不期待性能突破。

**配置**：
- model 预训练完成（或随机初始化）+ 元决策 head 随机初始化
- 跑 20 个 seed（0-19）
- 每 seed 跑一局 act1，记录所有 trajectory + reward + outcome

**检查项**：
1. **无 crash**：20/20 完整跑完（允许少数 timeout）
2. **数据流对**：trajectory 字段齐全、state 字段无 None / NaN、reward 数字合理（不是全 0 或全 inf）
3. **覆盖所有 phase**：MAP / NEOW / CARD_REWARDS / EVENT / SHOP / REST / TREASURE 都至少出现过
4. **性能 OK**：一局 < 5 min（评估 cache 工作正常）
5. **牌组强度评估正常**：触发次数 ≥ 期望次数（一局 5-10 次）+ cache 命中率 ≥ 50%
6. **基础指标**：reached_boss ≥ 1/20（不强求，但完全 0 要查问题）+ floor mean 在合理区间

**对照实验**：
- 跑 1：纯 RL（无预训练）
- 跑 2：含战斗 head 预训练
- 比 reached_boss / floor mean 看预训练是否有用

**特殊情况**：
- 如果某项 fail → 停下查 bug，不进真训练
- 如果两次对照差不多 → 预训练简化掉省工程

**自检**：
- ✅ 不真训练只 sanity check
- ✅ 覆盖所有 phase 测全 pipeline
- ✅ 含预训练 yes/no 对比验证
- ✅ 不超 20 局浪费

---

## 10. 真训练规模

**⚠️ 强约束：真训练（83h）只能用户启动，Claude 不主动启动**

Smoke 通过 + 用户 review 通过 + 用户明确说 "开始真训练" → 才跑。

**参数（待 smoke 后调）**：
- num_episodes：1000-5000（看收敛）
- batch_size：32 episodes
- lr：3e-4 (Adam)
- gamma：0.99
- evaluate frequency：每 100 episodes 跑 30 seed eval
- checkpoint：每 500 episodes 存

**评估指标**：
- reached_boss / 30
- beat_boss / 30
- floor mean / median
- avg game length（不要 stall）

**停止条件**：
- 性能不再提升（连续 1000 episodes 无提升）
- 或 beat_boss ≥ 5/30（达到阶段性目标后停下复盘）

**特殊情况**：
- 训练耗时预估：1000 episodes × 5min/episode = 83h（一周左右）—— 单 Mac 限制
- 中间 checkpoint 让训练可恢复

**自检**：
- ✅ 评估指标符合阶段性目标（先过 act1 boss）
- ✅ 配置参数有 sanity 区间不离谱
- ✅ 评估频率 + checkpoint 防训练废掉

---

## 整体自检（对照防跑偏检查表）

设计原则：
- ✅ 路线评估用"战损 + 牌组强度"两个维度（组件 4.1）
- ✅ 战斗内 model 学搜索的出牌顺序 / 联动 / target（组件 6 预训练）
- ✅ 选卡依赖 post-battle 牌组多维评分（组件 4.2 + 5）
- ✅ 不是单 game_won 信号反推（组件 4 分层）
- ✅ 不重新引入 v8_strategy 启发式作为 BC teacher

工作方式：
- ✅ 完整设计 → smoke → 真训练（用户原则）
- ✅ 拆到具体子决策颗粒度（10 个组件每个有具体设计）
- ✅ 不让用户拍多选（每个组件给明确推荐）
- ✅ 不用脏数据 warm start（预训练用新 search self-play 数据）
- ✅ 特殊情况都有说明

---

## Open Questions（需要你确认）

1. **训练规模 1000-5000 episodes** 在你时间预算内吗？（83h 一周）
2. **act1 only 还是含 act2/3**？我的设计是先 act1 跑通，达到阶段性目标后再扩 act2/3
3. **算力**：单 Mac MPS 跑可行，但需要 model 一直占资源；你是否需要后台跑 + 不影响日常使用？

这些问题不影响设计本身，影响训练进度规划。

---

## 不在本设计内的（明确排除）

- v8_strategy.py 启发式（已 archive，不复活）
- v8_meta_5.pt 脏 model（已 archive，不复用）
- bottled_ai assessment（已 archive，不用）
- NN value head 替换 leaf eval（已 archive，不用）
- AlphaZero 完整版 MCTS（不在本期范围）

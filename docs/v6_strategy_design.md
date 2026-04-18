# V6 策略设计文档

## 0. 文档状态

- 版本：v1（初稿，2026-04-16）
- 目标：明确 V6 agent 的信息输入（encoding / observation space）和行为输出（action space / 决策边界），作为后续 reward 设计和训练迭代的基准
- 读者：待定
- 命名：待定（本文档暂命名 `v6_strategy_design`）

## 1. 背景

V6 架构在 2026-04-11 确认了几条核心原则（详见 `project_v6_design.md`）：

1. **统一效果编码**：卡牌、遗物、药水共用同一套约 30 个效果维度（damage / block / draw / energy / poison / vulnerable / weak / strength / dexterity 等），加基础属性共 40-45 维。
2. **Agent 自己打牌**：不再把 combat 外包给 Solver，combat 被视作模型学习卡牌机制的课堂。
3. **同一编码跨场景共享**：手牌、选牌奖励、牌组卡片用完全相同的 encoding，战斗经验直接迁移到策略决策。
4. **数据来源**：StSRLSolver 的 `cards.py` / `relics.py` / `potions.py` 静态查表，不依赖 CommunicationMod 透传。
5. **结构化 ID 查表**：不用 learned ID embedding（黑盒、不可解释）和 hash bucket（冲突）；用 `ID → 预定义效果向量` 的查表模式编码所有实体。卡牌已是这种做法（`CARD_DATA[id] → 57 维效果向量`），遗物、药水按相同模式。敌人靠槽位（C8）+ 当前意图区分，不强求种类 ID。

### 当前状态 vs 目标

- **目标**：Ascension 20（最高难度）稳定通关
- **当前状态**（2026-04-15 最新一轮 200 局 from-scratch）：
  - Avg floor：5.7（远低于 A0 通关所需的 floor 50（Act 3 Boss））
  - Win rate：0%
  - Value loss 改善（bug fix 后从 34 降到 4），但策略层仍有结构性问题
- **距离目标的差距**：不是"多训几轮"能补，需先检查策略设计是否**完备**

### 训练 Curriculum vs 设计目标

- **设计目标**：A20 通关。所有 observation / action / reward 设计按 A20 完备性要求展开
- **训练起点**：A0（最低难度）。用 A0 跑通基本闭环，验证策略可学性，再逐步升 ascension level
- **关键区分**：A0 是训练 curriculum 的起步，**不是设计目标**。设计上不能"先 A0 弄完整，A20 后续补"——所有完备性缺口（§2.11）都按 A20 标准列，不分阶段

### 为什么需要这份文档

前期发现策略侧多处设计缺陷（敌人 ID 缺失、遗物均值化、derived feature 缺失、END TURN 规则未兜底等）。这些缺陷使模型作为"玩家"等于瞎子。Reward 再怎么调、训练再多轮也无法突破。

**本文档的立场**：策略设计必须一开始就完备，不做"先 A0 跑通再补 A20"的阶段性妥协。每一项观察信号、每一个行为决策都按 A20 需求定义。**发现"这里和那里有缺陷"等于在已有断层上迭代**，不做。

## 2. 模型的信息（Observation / Encoding）

### 2.1 Token 结构

Transformer 的序列构造在 `_build_tokens`（sts_agent 对应模块 L1303-1375）。每 token 经 `TokenProjections` → `Linear` → CTX_DIM=128 → type embedding + 位置 embedding → 3 层 4 头 Transformer → `[CLS]` 取 `shared_repr` 256 维。

| 顺序 | Token | 数量 | 维度 | 备注 |
|---|---|---|---|---|
| 1 | `[CLS]` | 1 | — | 读出向量 |
| 2 | deck cards | 0-50 | TOKEN_DECK_CARD=8 | 战斗外卡组逐卡 token |
| 3 | run_progress | 1 | 13 | 楼层 / HP ratio / act / ascension / 房间提示等 |
| 4 | potion slots | 5 | 57 (UNIFIED_DIM) | 每槽独立 token |
| 5 | relic_summary | 1 | 57 | **所有遗物的均值向量** |
| 6 | player（战斗模式） | 1 | 64 | HP / block / energy / stance / power 聚合 |
| 7 | enemies（战斗模式） | 0-4 | 62 | 每敌人一个 token |
| 8 | hand cards（战斗模式） | 0-10 | 57 | 逐卡 token |
| 9 | draw pile summary（战斗模式） | 1 | 69 | 前 30 张均值 + 12 维统计 |
| 10 | discard pile summary（战斗模式） | 1 | 69 | 同上 |

**未进入 Transformer 序列的信息**：Draft / Path / Choice 的候选特征只在对应 head 内部与 `shared_repr` 拼接，不作为 token 输入。这意味着：

- 选路时看不到战斗 `shared_repr` 里的地图形状（地图特征也不在战斗序列里）
- 选牌时看不到其他候选卡
- 选择事件时看不到其他选项

### 2.2 玩家自身（A）

| 编号 | 项 | 状态 | 位置 | 编码方式 | 缺口 |
|---|---|---|---|---|---|
| A1 | HP / Max HP | ✓ | `encode_player_state` L329-331；`encode_run_progress` L448, 455 | player v[0]=hp/max_hp，v[1]=(1-ratio)²，v[2]=max_hp/200；run_progress v[2]=hp_ratio，v[8]=deficit² | — |
| A2 | Block | ✓ | `encode_player_state` L332 | v[3]=min(block/50, 2.0) | — |
| A3 | 能量 / 能量上限 | ⚠ | `encode_player_state` L333 | v[4]=energy/4.0（当前能量） | 没有"本回合起始能量上限"单独维度 |
| A4 | 金币 | ⚠ | `encode_run_progress` L449 | v[3]=min(gold/999, 1.0) | 战斗 token 里缺失，非战斗 OK |
| A5 | 楼层 | ✓ | `encode_run_progress` | 完整 | — |
| A6 | Stance | ✓ | L346-354 | v[53..56] 4 维 one-hot（Calm / Wrath / Divinity / None） | — |
| A7 | 玩家 Power / Buff / Debuff | ⚠ | L339-341，`encode_powers_categorical` L281-309 | 44 维：20 功能类别 × 2（值 + 计数）+ 4 hash fallback | 按类别聚合不区分 buff 种类；未覆盖 Energized / Equilibrium / Combust / EchoForm 等 |
| A8 | 药水槽 | ✓ | `_build_tokens` L1329-1333 | 每槽独立 token（57 维），5 个槽位 | — |

### 2.3 卡牌（B）

| 编号 | 项 | 状态 | 位置 | 编码方式 | 缺口 |
|---|---|---|---|---|---|
| B1 | 手牌 ID / cost / 效果 | ✓ | L1357-1359，`encode_card` L168-212 | 每张独立 token 57 维 UNIFIED_DIM | — |
| B2 | 是否打得起 | ⚠ | `encode_card` L208-210 | "借用 innate 位"（idx 12） | 战斗中 innate 信息丢失；仅 binary，没有"差多少能量"的渐进 |
| B3 | 是否升级 + 升级版效果 | ✓ | `data/sts_data.py` L283-369 | 按 `f"{base_id}+"` vs `base_id` 分别查表 | — |
| B4 | retain / ethereal / innate / exhaust | ⚠ | `data/sts_data.py` L321-338 | idx 10-14 | 战斗中手牌 innate 被 is_playable 覆盖 |
| B5 | 抽牌堆 | ⚠ | L1362-1366，`encode_pile_summary` | 单个摘要 token 69 维（前 57 维是前 30 张均值，后 12 维统计） | 逐卡信息丢失，30 张以上截断 |
| B6 | 弃牌堆 | ⚠ | L1369-1373 | 同上 | 同上 |
| B7 | 消耗堆 | ✗ | — | — | **完全没有 exhaust_pile token**，Dead Branch、Corruption 机制无信号 |
| B8 | 战斗外卡组全貌 | ✓ | L1319-1322 | 逐卡 token，MAX_DECK_CARDS=50 | 50 张以上截断，run_progress v[12] 给溢出信号 |

### 2.4 敌人（C）

| 编号 | 项 | 状态 | 位置 | 编码方式 | 缺口 |
|---|---|---|---|---|---|
| C1 | HP / MaxHP / Block | ✓ | `encode_enemy` L398-401 | 完整 | — |
| C2 | 意图类型 | ⚠ | L395, L416-419 | 7 类 one-hot（ATTACK / BUFF / DEBUFF / DEFEND / ESCAPE / UNKNOWN / SLEEP），用子串匹配 `if it in intent` | 子串匹配有歧义（`ATTACK_DEBUFF` 同时命中 ATTACK 和 DEBUFF）；STUN / MAGIC 未覆盖 |
| C3 | 意图伤害 | ✓ | v[14]=min(adj_damage/40, 2.0) | — | — |
| C4 | 意图段数 | ✓ | v[15]=min(hits/5, 2.0) | — | — |
| C5 | 多段标记 | ✓ | v[17] | — | — |
| C6 | 敌人 Power | ⚠ | 44 维同 A7 | 同 A7 | 同 A7，敌人独有 power（SporeCloud / CurlUp / Malleable / AngerNob 等）多数落入 hash fallback（升级为严重）敌人独有 power 多数落入 hash fallback，直接决定 C7 的区分能力——若 C6 修好，ID 之外的差异才能被捕捉 |
| C7 | 敌人 ID / 种类 | ✗ | — | **中等缺口**：敌人种类 ID（区分 Jaw Worm / Cultist 等）未编码。当前回合意图 + powers 已能区分大部分场景；A20 完备性下种类 ID 有助于预测下回合行为，但不是首要 |
| C8 | 敌人位置 / 槽位 | ✗ | `_build_tokens` L1352-1354 | **严重缺口**：每个敌人是独立 token，但相对顺序依赖 transformer 的绝对位置 embedding。前面有 deck/potion/relic 等 token，敌人 token 的绝对位置因卡组大小漂移（10 张牌时敌人 1 在 position 17，30 张时在 position 37）。模型学到的"position N = 第 K 个敌人"不稳定 | 修法：`encode_enemy` 加 slot_index 维度（4 维 one-hot 或单标量 0-3），让每个敌人 token 自带槽位标识 |

### 2.5 战斗上下文（D）

| 编号 | 项 | 状态 | 位置 | 编码方式 | 缺口 |
|---|---|---|---|---|---|
| D1 | 当前回合数 | ✓ | v[5]=min(turn/15, 1.0) | — | — |
| D2 | 房间类型 | ⚠ | `encode_run_progress` L457-458 | 只识别 ELITE，BOSS 靠 floor 推断 | 没有 room-type one-hot |
| D3 | 敌人组合 ID | ✗ | — | — | 无 encounter_id |
| D4 | **本回合历史** | ✗ | — | — | 无"本回合打过的牌 / 累计伤害 / 累计 block / 已用能量" |

### 2.6 地图（E）

| 编号 | 项 | 状态 | 位置 | 编码方式 | 缺口 |
|---|---|---|---|---|---|
| E1 | 当前节点坐标 | ✗ | — | — | 没有 x / y |
| E2 | 前方可选节点类型 | ⚠ | `choose_path` L2013-2046 | 仅在 choose_path 调用时临时构造 16 / 8 维 path features，**不进入 Transformer 上下文** | 战斗 / 选牌决策看不到地图形状 |
| E3 | 距离 Boss 步数 | ✓ | v[11]=floors_to_boss/15 | — | — |

### 2.7 遗物（F）

| 编号 | 项 | 状态 | 位置 | 编码方式 | 缺口 |
|---|---|---|---|---|---|
| F1 | 持有遗物 ID | ✗ | `_build_tokens` L1336-1342 | **严重缺口**：现状是单个 `relic_summary` token（所有遗物均值）。应改为**逐遗物 token + ID 查表**：`RELIC_DATA[id] → 57 维效果向量`，与卡牌共用同一套编码通路。模型必须能看到"我有什么遗物"——做 draft / shop / boss relic 决策时基于持有清单调整策略（有 Dead Branch → 选 Corruption / exhaust 流，有 Snecko Eye → 选高费牌） |
| F2 | 遗物 counter | ⚠ | `encode_relic_entity` L215-226 | **中等缺口**：counter 被写入 `gold_gain` 维度（借位）。大部分遗物没有有意义的 counter（被动效果反映在玩家 state 上）；少数有 counter 的（Preserved Insect、Pen Nib、Ink Bottle）才需要专门维度。优先级低于 F1，等 F1 改完后再决定 counter 是否独立维度 |

### 2.8 非战斗界面（G）

| 编号 | 项 | 状态 | 位置 | 编码方式 | 缺口 |
|---|---|---|---|---|---|
| G1 | Event 类型 + 选项 | ⚠ | `encode_choice_option` L589-667 | 每选项 40 维（type one-hot + 8 维文本 hash + 8 关键词 + 结构化字段） | 无 event_id 维度；文本 hash 8 维易冲突 |
| G2 | Shop 商品 | ✓ | `encode_shop_item` L670-706 | 每商品 40 维 + ENTITY_DIM 物品向量 | — |
| G3 | Campfire 选项 | ✓ | `decide` L2664-2681 | 40 维 CHOICE_OPTION_DIM | 无 recall；Dream Catcher 等稀有机制无显式处理 |
| G4 | Draft 选项 | ✓ | `pick_card` L1834-1896 | 每候选卡 57 维含升级位 | — |

### 2.9 全局（H）

| 编号 | 项 | 状态 | 位置 | 编码方式 | 缺口 |
|---|---|---|---|---|---|
| H1 | Ascension | ✓ | v[4]=min(lvl/20, 1.0) | — | — |
| H2 | Act | ✓ | v[1]=min(act/3, 1.0) | — | — |

### 2.10 Derived Feature 现状

**已有（9 项）**：

1. HP deficit²（player v[1]，run_progress v[8]）
2. Total threat（enemy v[16]）
3. `is_multi_attack`（enemy v[17]）
4. `is_boss_floor`（run_progress v[9]）
5. `boss_proximity`（run_progress v[11]）
6. `deck_overflow`（run_progress v[12]）
7. Pile 组成统计（12 维）
8. Path-level 聚合（16 维）
9. HP ratio 进 option encoding

**缺口（6 项）**：

1. 威胁-防御差（expected_damage vs block）——只在 reward 端用了，不进 state
2. 手牌剩余潜力 sum（总伤害 / 总 block / 可打几张）
3. Wrath 时实际伤害（stance × damage 乘积）
4. Vulnerable 下期望伤害
5. HP ratio × anxiety factor 作为 feature
6. 能量 - 费用差（"还能打几张"）

### 2.11 编码缺口汇总

分级标准：对**策略完备性（目标：A20 通关）**的影响。

**严重——策略完备性硬缺口，必须补**：

- **C8 敌人槽位漂移**：每个敌人 token 没有自带槽位标识，绝对位置因卡组大小漂移
- **C6 敌人 power hash fallback**：CurlUp / Ritual / AngerNob / Malleable / SporeCloud 等独有 power 未精确分类
- **F1 遗物均值化**：稀有遗物（Runic Pyramid / Snecko Eye / Dead Branch）信号消失，draft / shop 决策无依据
- **D4 本回合历史**：本回合已用能量、已打出牌、累计伤害/block 都没有，主动 END TURN 判断缺核心依据
- **E2 地图不进 Transformer**：选路时 shared_repr 看不到地图形状；战斗 / 选牌决策无法为未来路况做准备
- **§2.10 Derived feature 缺口**（威胁-防御差、手牌潜力 sum、stance×damage、vuln 下期望伤害、HP anxiety feature、能量-费用差）：主动 END TURN 和战术决策的依据全部缺失

**中等——影响决策质量但不阻塞完备性**：

- **C7 敌人种类 ID**：当前意图 + powers 已能区分大部分场景，A20 完备性下种类 ID 有助于预测下回合行为，但不是首要
- **F2 遗物 counter**：大部分遗物 counter 无意义，少数有 counter 的（Preserved Insect 等）需要专门维度，优先级低于 F1
- **A3 能量上限**：只编码当前能量，没有回合初始上限（影响 Cursed Key 等修改上限的场景）
- **B2 is_playable binary**：没有"差多少能量"的渐进信号
- **B7 消耗堆缺失**：Dead Branch / Corruption 机制无信号，但不是主流 build
- **D2 房间类型**：只有 ELITE，BOSS 靠 floor 推断，缺普通 / 商店 / 篝火 / 事件 one-hot
- **C2 意图子串匹配**：`ATTACK_DEBUFF` 会同时命中 ATTACK 和 DEBUFF
- **G1 Event 无 ID**：文本 hash 8 维易冲突

**轻微——边缘场景**：

- **B4 innate 位在战斗中被 is_playable 覆盖**：战斗内 innate 信息丢失
- **A7 power 按类别聚合**：同类别不同 buff 不区分（大多数 power 够用）

## 3. 模型的行为（Action Space / 决策逻辑）

### 3.1 战斗内行为

#### 3.1.1 决策点

- 每次游戏请求战斗决策时调用 `combat_act`
- 典型触发：回合开始、打完一张牌后、使用药水后

#### 3.1.2 动作空间

- `play card`：从手牌选一张打出（若该牌需目标，额外选敌人）
- `end turn`：结束本回合
- `use potion during combat`（是否支持、是否当前走 RL：待查）
- `discard potion during combat`（是否支持、是否当前走 RL：待查）

**Play card 的细分**：

- **单体目标选择**：需指定敌人下标（Strike / Bash 等单体牌）
- **AoE 不选目标**：Whirlwind / Cleave / 全体 AoE
- **自身目标**：Shiv / Tactician 等不需要敌人目标
- **Exhaust 选择**：Dual Wield 需选复制哪张手牌

**X-cost 牌**：费用=剩余能量（如 Whirlwind X），实际输出和打出时能量挂钩，是特殊动作。当前是否特殊处理：待查

**Dual-action 牌**：Discovery / Foresight 等打出后弹出额外选择界面，决策链超过一次 `combat_act` 调用。当前是否接入：待查

#### 3.1.3 决策边界（规则兜底 vs RL 学习）

**END TURN 的三种被动情况（规则应兜底）**：

1. 无能量
2. 无手牌
3. 手牌里没有能打得起的牌

这三种情况应通过 action mask 强制 END，不让模型学。**当前实现是否已 mask：待查**。

**END TURN 的主动情况（RL 学习）——本回合还能打但选择不打**：

- 格挡已溢出（敌人意图 < 当前 block，再打 Defend 浪费）
- 伤害已溢出（致死后再打无意义）
- 保留 retain 牌到下回合（Windmill Strike 等）
- 避免 Wrath 下伤害放大副作用
- 手牌全是 curse / status 等不想出的

**Play card 的 RL 学习内容**：选哪张牌（对谁），以及与 END TURN 的权衡。

**注意：当前 derived feature 缺口影响主动 END TURN 可学性**

主动 END TURN 的 5 个典型情境（格挡溢出 / 伤害溢出 / 保留 retain / 避 Wrath / 无用手牌），都需要模型看到：

- **威胁 - 防御差**：判断"再打一张 Defend 是否溢出"（§2.10 缺口 1）
- **手牌剩余潜力 sum**：判断"还能打多少伤害/block"（§2.10 缺口 2）
- **能量 - 费用差**："还能打几张"（§2.10 缺口 6）
- **Wrath 时 × 2 伤害**：避免 Wrath 下继续出攻击牌（§2.10 缺口 3）

这些 derived feature 目前全部缺失。**即使 reward 设计正确、训练量充足，模型也无法从原始 token 跨尺度跨位置自发学出这些判断**。这是当前策略层被动打光所有牌的根本原因之一。

### 3.2 战斗外行为

#### 3.2.1 Draft（选牌）

- **触发**：战斗结束奖励、草堆事件、升级 / 删牌选项
- **动作空间**：`pick_card(candidate_i)` 或 `skip`
- **决策边界**：RL 学习
- **当前编码**：G4 ✓

#### 3.2.2 Path（选路）

- **触发**：楼层结束后选下一个节点
- **动作空间**：`choose_path(node_i)`
- **决策边界**：RL 学习
- **当前编码**：E2 ⚠（路径特征不进 Transformer，仅作为 head 候选）

#### 3.2.3 Choice（事件）

- **触发**：? 房间、特殊交互
- **动作空间**：`make_choice(option_i)`
- **决策边界**：RL 学习
- **当前编码**：G1 ⚠（无 event_id，文本 hash 冲突）

#### 3.2.4 Rest（篝火）

- **触发**：R 节点
- **动作空间**：`rest` / `smith` / `dig`（Shovel） / `lift`（Girya） / `toke`（Peace Pipe）
- **决策边界**：RL 学习
- **当前编码**：G3 ✓

#### 3.2.5 Shop（商店）

- **触发**：$ 节点
- **动作空间**：每件商品买或不买、删牌、离开
- **决策边界**：RL 学习
- **当前编码**：G2 ✓

#### 3.2.6 Potion 决策

- **触发**：获得药水、战斗内可用药水（战斗内是否走 RL：待查）
- **动作空间**：`use` / `discard` / `keep`
- **决策边界**：RL 学习（前提是触发点已接入）
- **当前编码**：待查

#### 3.2.7 Boss Relic 选择

- **触发**：Act 结束拿 boss relic
- **动作空间**：从 3 个中选 1 或 skip
- **决策边界**：RL 学习
- **当前编码**：F1 ⚠（遗物均值化影响未来决策质量）

#### 3.2.8 Neow 祝福

- **触发**：游戏开始
- **动作空间**：从 Neow 给的选项中选 1
- **决策边界**：RL 学习
- **当前编码**：待查

### 3.3 行为决策边界总表

| 决策点 | 动作空间 | RL 学 or 规则兜底 | 依赖的编码项 |
|---|---|---|---|
| 战斗：play card | 选牌 + 选目标 | RL | A, B（手牌 / 堆）, C, D |
| 战斗：end turn（被动） | 强制 end | 规则兜底（现状待查） | A3（能量）, B1（手牌） |
| 战斗：end turn（主动） | 选择结束 | RL | A, B, C, D |
| 战斗：use / discard potion | 用 / 丢 | 待查（触发点是否接入） | A8 |
| Draft | 选牌 / skip | RL | G4 |
| Path | 选节点 | RL | E2（不进 Transformer） |
| Choice（事件） | 选选项 | RL | G1 |
| Rest（篝火） | rest / smith / dig / lift / toke | RL | G3, B8, F |
| Shop | 每件买 / 删牌 / 离开 | RL | G2, F, A4 |
| Boss Relic | 3 选 1 / skip | RL | F1 |
| Neow 祝福 | 从选项中选 1 | RL | 待查 |

## 4. 待定事项清单

以下是本文档写作时**未能从代码中确认**的事实，需要进一步验证：

- **战斗中是否支持 `use potion` / `discard potion`** 走 RL 决策（触发点是否已接入）
- **战斗中 END TURN 的 action mask 现状**：是否已对"无能量 / 无手牌 / 无可打牌"做规则兜底
- **Neow 祝福的编码现状**（是否有专门 encoder 还是走通用 choice）
- **Boss Relic 选择**：是否共用 draft head 还是独立路径
- **Shop 删牌**：是否共用 draft head
- **X-cost 牌（Whirlwind 等）和 Dual-action 牌（Discovery / Foresight 等）** 的动作空间处理
- **文档读者** / **文档命名**（本文档暂命名 `v6_strategy_design`）

## 5. 不在本文档范围

- Reward 设计（下一轮讨论）
- 网络架构细节（`shared_repr` / head 大小已在 code 里，不是本文档重点）
- 训练流程和超参
- Communication Mod 数据采集细节

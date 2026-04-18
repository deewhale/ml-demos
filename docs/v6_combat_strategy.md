# V6 Combat 策略设计文档

**日期**：2026-04-16

**状态**：
- 第一部分（编码层 / Observation）：本文档已完成
- 第二部分（动作空间 + 决策边界）：待后续补充
- 第三部分（Reward 设计）：待后续补充

**目标说明**：当前优先级是 **A0（非 Ascension 普通模式下能通关）**。本文档以"**模型不是瞎子**"为原则盘点 combat agent 的编码层现状 —— 玩家用肉眼能从游戏画面看到的信息，模型能否同样看到。精细化（意图歧义、位置稳定性、逐卡摘要等）延后到 A0 稳定通关之后再处理。

---

## 1. 设计原则

1. **模型视角 = 真实玩家视角**：玩家能看到的（HP、手牌、意图、地图、遗物 counter……）都要编码；玩家看不到的（下一张具体是什么、敌人 AI 种子、未来伤害 roll）不编码。
2. **不硬编码游戏知识**：沿用 V6 原则，卡牌/遗物/药水用统一效果维度（damage / block / draw / poison / vulnerable / weak 等 ~30 维），不做 card-id one-hot、不做"XX 遗物触发 XX 逻辑"的 if-else 特征。
3. **A0 通关优先**：先保证模型对关键信息不瞎，再谈粒度精细化。缺了会直接导致决策不可学（消耗堆、本回合历史、地图形状）的优先补；粒度粗但还能学（Power 类别聚合、意图子串匹配）的延后。
4. **战斗经验跨场景共享**：战斗中学到的卡牌理解要能迁移到选牌 / 商店，所以同一张卡在战斗手牌、draft 候选、deck overview 里用**完全相同的编码**。
5. **现状描述，不给方案**：本文档只盘点"现在编码了什么 / 缺什么"，具体修改方案在后续章节或待定项中讨论。

---

## 2. 玩家视角的信息类别清单

按以下八类组织 —— 这是玩家坐在屏幕前能看到的信息全集：

- **A. 玩家自身**：HP / Max HP、Block、能量（当前 + 上限）、金币、楼层、Stance、Power / Buff / Debuff、药水槽
- **B. 卡牌**：手牌（含 cost / 效果 / 是否打得起 / 升级 / retain / ethereal / innate）、抽牌堆、弃牌堆、消耗堆、战斗外卡组全貌
- **C. 敌人（逐个）**：HP / Max HP / Block、当前回合意图类型、意图伤害、意图段数、多段标记、Power / Buff / Debuff、ID / 种类、位置顺序
- **D. 战斗上下文**：当前回合数、房间类型、敌人组合 ID、本回合已打过的牌序列 / 累计伤害 / 累计格挡 / 已用能量
- **E. 地图**：当前节点坐标、前方可选节点类型、距离 Boss 步数
- **F. 遗物**：持有全部遗物 ID、触发状态 / counter
- **G. 非战斗界面**：Event 类型 + 选项、Shop 商品、Campfire 选项、Draft 选项（含升级状态）
- **H. 全局**：Ascension 等级、Act

---

## 3. 当前编码状态检查表

状态图例：**✓** 已覆盖 / **⚠** 部分覆盖或有缺陷 / **✗** 完全缺失

### A. 玩家自身

| 编号 | 项 | 状态 | 编码位置 | 编码方式 | 缺口 |
|---|---|---|---|---|---|
| A1 | HP / Max HP | ✓ | `encode_player_state` L329-331；`encode_run_progress` L448, L455 | v[0]=hp/max_hp；v[1]=(1-ratio)²；v[2]=max_hp/200；run_progress 另有 hp_ratio 和 deficit² | — |
| A2 | Block | ✓ | `encode_player_state` L332 | v[3]=min(block/50, 2.0) | — |
| A3 | 能量 | ⚠ | `encode_player_state` L333 | v[4]=energy/4.0 | 没有"本回合能量上限"单独维度，Mark of Pain / Sozu 等改能量上限的效果模型感知不到 |
| A4 | 金币 | ⚠ | `encode_run_progress` L449 | v[3]=min(gold/999, 1.0) | 战斗 token 内没有 gold 维度；商店决策时靠 run_progress 够用 |
| A5 | 楼层 | ✓ | `encode_run_progress` L446-469 | floor / act / is_boss_floor / floors_to_boss 都有 | — |
| A6 | Stance | ✓ | `encode_player_state` L346-354 | 4 维 one-hot（WRATH / CALM / DIVINITY / neutral） | — |
| A7 | Power / Buff / Debuff | ⚠ | `encode_powers_categorical` L281-309；Strength/Dex/Mantra 另有专用维度 L369-379 | 44 维 = 20 功能类别（amount + count）+ 4 hash fallback | (1) 同类多种 power（Strength / Vigor / DoubleDamage / PenNib）共享一维无法区分；(2) Energized / Equilibrium / Combust / EchoForm 等重要 power 未列入类别、走 hash fallback |
| A8 | 药水槽 | ✓ | `_build_tokens` L1329-1333 | 每个槽位独立 token（MAX_POTION_SLOTS=5），空槽全零 | — |

### B. 卡牌

| 编号 | 项 | 状态 | 编码位置 | 编码方式 | 缺口 |
|---|---|---|---|---|---|
| B1 | 手牌 ID / cost / damage / block / 效果 | ✓ | `_build_tokens` L1357-1359；`encode_card` L168-212；DIM_NAMES 见 `data/sts_data.py` L35-112 | 每张独立 57 维 token（UNIFIED_DIM） | — |
| B2 | 是否打得起 | ⚠ | `encode_card` L208-210 | **借用 innate 位（idx 12）**覆盖为 is_playable | 只有 binary，没有"差多少能量打得起"的渐进信号 |
| B3 | 升级 | ✓ | `encode_card` L178-186；`CARD_DATA` 预生成升级 / 未升级两套向量 | is_upgraded 单独一维（idx 9） | — |
| B4 | retain / ethereal / innate / exhaust | ⚠ | `data/sts_data.py` L321-338 预编码进 CARD_DATA | 各占一维 | 战斗中手牌的 innate 位被 is_playable 覆盖（仅对初始抽牌信号有影响） |
| B5 | 抽牌堆 | ⚠ | `_build_tokens` L1362-1366；`encode_pile_summary` L482-517 | 单个摘要 token 69 维（前 30 张均值 + 12 维组成统计） | 逐卡信息丢失，无法知道下一张具体是什么；30 张以上截断 |
| B6 | 弃牌堆 | ⚠ | 同 B5 结构 | 独立摘要 token | 同 B5；且 discard_pile summary 混合本回合和上回合打过的牌 |
| B7 | 消耗堆 | ✗ | —— | `_build_tokens` 只处理 draw_pile 和 discard_pile，**没有 exhaust_pile 处理** | Dead Branch、Corruption、Ragnarok / Apparitions 等依赖消耗堆的机制**完全不可感知** |
| B8 | 战斗外卡组全貌 | ✓ | `_build_tokens` L1319-1322 | 逐卡 token，MAX_DECK_CARDS=50 | — |

### C. 敌人

| 编号 | 项 | 状态 | 编码位置 | 编码方式 | 缺口 |
|---|---|---|---|---|---|
| C1 | HP / Max HP / Block | ✓ | `encode_enemy` L398-401 | 数值维度 | — |
| C2 | 意图类型 | ⚠ | `encode_enemy` L416-419 | 7 类 one-hot（ATTACK / BUFF / DEBUFF / DEFEND / ESCAPE / UNKNOWN / SLEEP），**用 "in" 子串匹配** | ATTACK_DEBUFF 会同时命中 ATTACK 和 DEBUFF 产生歧义；STUN / MAGIC 等 intent 未覆盖 |
| C3 | 意图伤害 | ✓ | `encode_enemy` L422 | v[14]=min(dmg/40, 2.0)，优先 `move_adjusted_damage` | — |
| C4 | 意图段数 | ✓ | `encode_enemy` L423 | v[15]=min(hits/5, 2.0) | — |
| C5 | 多段标记 | ✓ | `encode_enemy` L425 | v[17]=1.0 if hits > 1 | — |
| C6 | Power / Buff / Debuff | ⚠ | 同 A7，用 `encode_powers_categorical` | 44 维类别聚合 | 敌人独有 power（Ritual / CurlUp / AngerNob / TimeWarp / Reactive / Invincible / Anger / Malleable / Thievery 等）中只有 Ritual 明确分类，其余大部分 fallback 到 hash |
| C7 | 敌人 ID / 种类 | ✗ | —— | **不编码**。只有 v[5]=is_boss | Jaw Worm 和 Acid Slime 若 HP / intent / powers 相同则编码一致，无法区分同 HP 不同种类的敌人 |
| C8 | 位置顺序 | ⚠ | `_build_tokens` L1352-1354 按顺序生成 token | transformer 绝对位置 embedding | 位置信息依赖 absolute position embedding，但前面有不定数量的 deck / potion / relic token，位置 embedding 会因卡组大小漂移，敌人间相对顺序不稳 |

### D. 战斗上下文

| 编号 | 项 | 状态 | 编码位置 | 编码方式 | 缺口 |
|---|---|---|---|---|---|
| D1 | 回合数 | ✓ | `encode_player_state` L334 | v[5]=min(turn/15, 1.0) | — |
| D2 | 房间类型 | ⚠ | `encode_run_progress` L457-458 | 只有 is_elite_screen 一位；Boss 靠 is_boss_floor 间接推断 | 没有 room-type one-hot（normal / elite / boss） |
| D3 | 敌人组合 ID | ✗ | —— | **不编码** | agent 只能从若干敌人 token 的形态隐式识别，无法在 seed 间迁移 |
| D4 | 本回合历史（已打的牌 / 累计伤害 / 累计格挡 / 已用能量） | ✗ | —— | **完全没编码** | 没有"本回合已打的牌"token，也没有累计数值维度；discard_pile summary 混合本回合和上回合 |

### E. 地图

| 编号 | 项 | 状态 | 编码位置 | 编码方式 | 缺口 |
|---|---|---|---|---|---|
| E1 | 当前节点坐标 | ✗ | —— | 不编码，只有 floor，没有横坐标 | — |
| E2 | 前方可选节点类型 | ⚠ | `choose_path` 调用时临时构造：L2013-2046；`encode_path_node` L524-533；`encode_full_path` L536-586 | 16 维 path 特征 | **路径信息不进入 Transformer 上下文**，只作为 path_head 候选特征。Transformer shared_repr 看不到地图形状，combat 和 draft 决策都看不到"前面有几个精英"/"下一个休息站多远" |
| E3 | 距离 Boss | ✓ | `encode_run_progress` L461-469 | v[11]=floors_to_boss/15 | — |

### F. 遗物

| 编号 | 项 | 状态 | 编码位置 | 编码方式 | 缺口 |
|---|---|---|---|---|---|
| F1 | 持有全部遗物 ID | ⚠ | `_build_tokens` L1336-1342 | **单个** relic_summary token（所有遗物 57 维均值） | 不是逐遗物 token，所有遗物平均稀释，agent 分不清"有 Runic Pyramid"和"有 Bag of Preparation" |
| F2 遗物 counter | ⚠ | `encode_relic_entity` L215-226 | counter 写入 idx 54（借用 gold_gain 维度），min(counter/10, 2.0) | F1 均值化导致 counter 信息也被稀释 |

### G. 非战斗界面

| 编号 | 项 | 状态 | 编码位置 | 编码方式 | 缺口 |
|---|---|---|---|---|---|
| G1 | Event 类型 + 选项 | ⚠ | `encode_choice_option` L589-667 | 每选项 40 维（type one-hot + option_index + 8 维文本 hash + hp_ratio + 8 关键词 + 首数字 + text 长度 + 结构化 hp_gain / loss / gold / 卡片变更 / randomness） | 没有 event_id，不同事件同文本会撞；8 维文本 hash 容易冲突 |
| G2 | Shop 商品 | ✓ | `encode_shop_item` L670-706 | 每商品 40 维（价格、affordability、类型 one-hot、效果向量注入） | — |
| G3 | Campfire 选项 | ✓ | `decide` L2664-2681 | rest / smith / dig / lift / toke 按遗物动态构造 | — |
| G4 | Draft 选项 | ✓ | `pick_card` L1834-1896 | 每候选卡 57 维 UNIFIED_DIM（含升级位） | — |

### H. 全局

| 编号 | 项 | 状态 | 编码位置 | 编码方式 | 缺口 |
|---|---|---|---|---|---|
| H1 | Ascension 等级 | ✓ | `encode_run_progress` L450 | v[4]=min(ascension/20, 1.0) | — |
| H2 | Act | ✓ | `encode_run_progress` | v[1]=min(act/3, 1.0) | — |

---

## 4. Derived Feature 现状

### 4.1 已有的 derived features

- **HP deficit²**：`encode_run_progress` L455，对低血量做非线性放大
- **Total threat**：敌人 intent 伤害求和（player token 或 run_progress 层面）
- **is_multi_attack**：敌人 hits > 1 标记（`encode_enemy` v[17]）
- **is_boss_floor / boss_proximity / floors_to_boss**：楼层进度聚合
- **deck_overflow**：卡组大小超过 MAX_DECK_CARDS=50 的标记
- **Pile 组成统计**：`encode_pile_summary` 的 12 维（attack / skill / power 占比、平均 cost 等）
- **Path 聚合**：`encode_full_path` 对路径节点类型计数
- **hp_ratio 合成进 choice option**：事件选项内部带玩家当前血量比例，让模型判断"这个选项现在打不打得起"

### 4.2 缺失但会影响决策可学性的 derived features

- **威胁-防御差**（expected_damage - current_block）：玩家每回合都要算这个，模型现在只能从 intent 伤害和 block 两个独立维度里自己学差值
- **手牌剩余潜力 sum**（本回合还能打的总 damage / block）：现在靠逐卡 token 聚合，没有"本回合最多能输出 X / 格 Y"的 global 信号
- **Stance × damage 乘性**（Wrath 下实际 2× 伤害）：Stance 是 one-hot，damage 是卡牌维度，两者的乘性交互要模型自己学
- **Vulnerable 下敌人实际承受伤害**：现在敌人 Vulnerable 在 power 类别里，卡牌 damage 在卡牌维度里，乘性同样靠模型学
- **能量 - 手牌最小 cost**（是否还有牌能打）：现在只有能量数值和各卡 cost，没有"手上还有 X 张打得起"的聚合信号

---

## 5. Token 结构总览

Transformer 序列构造顺序（从 `_build_tokens`）：

```
CLS
 → deck_cards （MAX_DECK_CARDS=50 逐卡 token）
 → run_progress
 → potion_slots （MAX_POTION_SLOTS=5 逐槽 token）
 → relic_summary （单个均值 token）
 ↓（战斗模式追加）
 → player
 → enemies （逐个 token）
 → hand_cards （逐卡 token）
 → draw_pile_summary （单个摘要 token）
 → discard_pile_summary （单个摘要 token）
```

### 关键结论

- **Draft / Path / Choice head 的候选特征不进 Transformer**，只在 head 内部与 shared_repr 拼接
- **选路时 combat shared_repr 看不到地图形状**：地图只作为 path_head 的候选特征
- **选牌时 shared_repr 看不到候选卡本身**：候选卡只进 draft_head
- **只有 CLS 和 run_progress 起全局作用**；没有 combat-level global token（威胁-防御差、本回合累计、手牌剩余潜力等聚合信号无处安放）

---

## 6. A0 通关优先级分级

把所有 ⚠ 和 ✗ 按"是否影响 A0 可通关"分三档：

### 必修（缺了模型是瞎子）

- **B7** 消耗堆完全没编码
- **D4** 本回合历史完全没编码
- **F1** 遗物均值化稀释
- **E2** 地图信息不进 combat / draft context

### 选修（影响效果但不致命）

- **C7** 敌人 ID 不编码
- **B5 / B6** 抽 / 弃牌堆只有摘要
- **A7 / C6** Power 类别聚合粒度粗
- **G1** Event 文本 hash 容易撞

### 精细化（A0 后再处理）

- **A3** 能量上限单独维度
- **C2** 意图类型子串匹配歧义
- **C8** 敌人位置依赖绝对 position embedding
- **B2** is_playable 渐进信号
- **D2** 房间类型 one-hot
- **D3** 敌人组合 ID

---

## 7. 待定项

以下问题需要后续讨论后再定方案，本文档不给出答案：

- **待定**：地图信息如何进入 combat / draft context —— 整个地图 tokenize 不现实，可能是一个 map_summary global token 承载聚合特征，具体方案未定
- **待定**：本回合历史编码粒度 —— token 级"逐张打过的牌"vs global 级"累计 sum（伤害 / 格挡 / 已用能量）"，两者信号强度不同
- **待定**：遗物逐个 token 化的 MAX 数量 —— 后期遗物可能 15+ 个，MAX 定多少合理
- **待定**：derived feature 是加进现有 player / enemy token 扩展维度，还是引入新的 combat_global token
- **待定**：消耗堆编码粒度 —— 摘要 token vs 逐卡 token

---

## 8. 后续章节预告

本文档还有以下部分待补充：

- **第二部分：动作空间 + 决策边界** —— END TURN 分类、action mask 规则、卡牌-目标组合的表达
- **第三部分：Reward 设计** —— 每个动作的 reward 归属规则、即时（per-card）vs 长期（per-turn / per-combat / per-floor）信号分工
- **第四部分：已知不覆盖的场景** —— 例如 Dead Branch、Shop 决策的延后处理范围

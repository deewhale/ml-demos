# V6 Reward 设计文档

## 0. 文档状态

- 版本：v1（初稿，2026-04-16）
- 目标：明确 V6 agent 的 reward 公式、归属规则和已知设计问题
- 与 `v6_strategy_design.md` 配套：策略文档定义"模型看到什么、做什么"，本文档定义"模型每个动作得到什么反馈"
- 代码基准：`sts_agent_v6.py` @ HEAD `93b786b`（含 heal 被吞 + 致死一击 0 分两处 bug fix）

## 1. 背景与原则

Reward 是 PPO 最敏感、最易出 bug 的部分。过去几轮训练 floor 反复涨跌（从 scratch 200 局 avg_floor 5.7、win_rate 0%），根因多次落在 reward 公式上：baseline 减除过度、heal 被 Phase A 基础分覆盖、致死卡得 0 分、stuck 误检测等，最近几次 commit（`93b786b` / `dfaa6ce` / `881c262` / `edf84e1` / `d0fa2bb`）都在修 reward 通路。

独立成文的目的是把当前所有公式 / 归属规则 / magic number 的设计意图固化下来，让后续修改有对照，避免"改了上面忘了下面"的回归。

### 1.1 设计原则

1. **Causal credit assignment**：每张牌只为自己造成的结果负责（damage / block / kill 直接归属，不均摊到整回合所有牌）
2. **打牌始终正反馈**：好牌更正、差牌更弱，但绝不是负的（避免模型学"少打牌 / 直接 END TURN"）
3. **HP 焦虑放大**：**已回滚**（2026-04-18：疑似反向激励源，验证中）。原设计为 `anxiety_factor = 1 + (1 - hp_ratio) × 2`（满血 ×1，半血 ×2，濒死 ×3），现 anxiety_factor 固定为 1.0。怀疑低 HP 时 HP penalty × 3 放大，而 END TURN 不分摊 HP penalty，导致 agent 学"低 HP → END TURN"的反向激励。
4. **END TURN 不分摊掉血责任**：HP 损失只均摊给本回合实际打出的牌，END TURN action 不参与分摊
5. **Magic number 来源标注**：所有常数（0.05 / 0.1 / 0.3 / 1.0 / 0.5 等）都标明设计意图，不无理由

## 2. Reward 层次结构

简要列出，详细公式见 §3：

- **Combat 层**：每张牌即时反馈，覆盖伤害 / 格挡 / 击杀 / 掉血 / 回血 / 能量 / 抽牌 / effect source
- **Combat 结束层**：combat completion / clutch win bonus / 最后一击近似 / draft 回溯
- **策略层**：draft / path / rest / shop / boss relic / choice / potion
- **终局层**：win = +10，lose = −3

## 3. 详细公式（基于代码 HEAD `93b786b`）

### 3.1 Combat 层（Phase A flush，约 L1606-1695；Phase B 约 L1697-1785）

#### 3.1.1 基础分（覆盖式 → `+=`）

- **公式**：`t.reward += card_damage × 0.05 + card_block × 0.05 + card_kills × 1.0 + energy_bonus`（L1622）
- **归属**：`turn_buffer` 内每张 transition（包括 END TURN transition，但 END TURN 的 extra 里没有 `card_damage` 等字段，取默认 0，所以实际只影响打牌的）
- **触发条件**：Phase A flush（新回合开始）；以及 `on_combat_end` 内的相同逻辑（L2307-2312）
- **Magic number 意图**：
  - `0.05`：5 点伤害 ↔ 0.25 reward。和 Phase A 的 HP 焦虑惩罚基线（`hp_lost × 0.1`）形成 1:2 比例，表达"伤害是输出，掉血是惩罚，等幅掉血惩罚是等幅伤害的 2 倍"
  - `1.0`：杀死一个敌人给 1 分，是 combat reward 中最大的常驻信号
  - `energy_bonus`：由 Phase B 计算（L1734），传到 Phase A 累加
- **历史**：`93b786b` 前为 `=`（覆盖式），会吞掉 Phase B 写入的 `heal_reward`，此次 fix 改成 `+=`

#### 3.1.2 HP 掉血惩罚（带焦虑系数）

- **公式**：
  ```
  hp_lost = max(turn_hp_start - current_hp, 0)
  anxiety_factor = 1.0   # 已回滚（原公式：1 + (1 - hp_ratio) × 2）
  hp_penalty = hp_lost × 0.1 × anxiety_factor / n_playable
  t.reward -= hp_penalty   # 仅对 played_card_id 非空的 transition
  ```
- **归属**：仅分摊给本回合 `extra.played_card_id` 非空的 transition（L1632-1637）；END TURN transition 不参与分摊
- **触发条件**：Phase A flush，`hp_lost > 0 且 n_playable > 0`
- **Magic number 意图**：
  - `0.1`：比伤害系数 `0.05` 大一倍，表达"等幅自伤 > 等幅输出"的价值观（基线，未放大前）
  - `anxiety_factor`：**已回滚为常量 1.0**（2026-04-18）。原设计满血 ×1、半血 ×2、濒死 ×3，但与 "END TURN 不分摊 HP penalty" 规则叠加后，疑似导致低 HP 时 agent 倾向 END TURN（惩罚被放大 3 倍全压在出牌上）
  - 除以 `n_playable`：本回合打了 3 张牌各自承担 1/3；打 END TURN 前如果只打了 1 张，那张要承担全部

#### 3.1.3 Poison 跨回合 credit

- **公式**：
  ```
  # 本回合投毒的源 transition 在 _turn_effect_sources 里登记为 (buf_idx, "poison", amount)
  # Phase A flush 时把 buf_idx 转 trajectory_idx 存入 _poison_sources
  # 下一回合 Phase A flush 时结算：
  proportion = amt / total_sourced   # 多源按投毒量比例分配
  credit = _pending_poison_tick × 0.05 × proportion × 0.3
  current_trajectory[traj_idx].reward += credit
  ```
- **归属**：上一回合的投毒源 transition（trajectory 已写入部分）
- **触发条件**：Phase A flush，`_pending_poison_tick > 0 且 _poison_sources 非空`（L1640-1648）；`_pending_poison_tick` 在上一次 flush 末尾用新回合的 `sum(enemy_poison)` 记录（L1684-1686）
- **Magic number 意图**：
  - `0.05`：和伤害系数保持一致（毒 tick 也是伤害）
  - `0.3`：effect source 类 credit 统一折扣系数，表达"间接输出比直接输出弱一些"

#### 3.1.4 Strength credit

- **公式**：对 `buf_idx` 之后每张 `card_damage > 0` 的牌，`turn_buffer[buf_idx].reward += amount × 0.05 × 0.3`
- **归属**：同回合内给玩家加 Strength 的源 transition
- **触发条件**：Phase A flush（L1651-1657）；`on_combat_end` 相同逻辑（L2344-2347）
- **Magic number 意图**：`0.05` 伤害系数 × `0.3` 间接折扣。**注意：Strength 给 +1 strength → 后续每张攻击牌获得的间接 credit 是 `0.015 × 后续攻击牌数`，量级很小**

#### 3.1.5 Dexterity credit

- **公式**：对 `buf_idx` 之后每张 `card_block > 0` 的牌，`turn_buffer[buf_idx].reward += amount × 0.05 × 0.3`
- **归属 / 触发 / 系数**：同 Strength，只换成 block 牌

#### 3.1.6 Vulnerable credit

- **公式**：对 `buf_idx` 之后每张 `card_damage > 0` 的牌，`turn_buffer[buf_idx].reward += dmg × 0.5 × 0.05 × 0.3`
- **归属**：同回合施加 Vulnerable 的源 transition
- **Magic number 意图**：
  - `0.5`：Vulnerable 带来 50% 伤害加成
  - `0.05 × 0.3`：复用伤害系数 × 间接折扣
  - **有效倍率是 `dmg × 0.0075`**，在 `card_damage × 0.05` 的基础分前信号几乎被淹没（见 §5）

#### 3.1.7 Weak credit

- **公式**：`turn_buffer[buf_idx].reward += amount × 0.1 × 0.3`（不遍历后续牌，固定给一次）
- **归属**：同回合施加 Weak 的源 transition
- **Magic number 意图**：
  - `0.1`：与其他三种 effect 不一致（其他是 `× 0.05 × 0.3`，Weak 是 `× 0.1 × 0.3`）
  - 实际量级：给敌人 +2 weak 得 `2 × 0.1 × 0.3 = 0.06`，和 Strength +1 的潜在 credit 量级相近
  - **与其他 effect source credit 的系数不统一**（见 §5）

#### 3.1.8 抽牌回传（draw credit）

- **公式**：
  ```
  # Phase B 时记录：source_idx → 抽到的 card_id
  # Phase A flush 时遍历 turn_buffer，若 t.extra.drawn_by_idx 指向同 buffer 的 source_idx：
  credit = t.reward × 0.3
  turn_buffer[source_idx].reward += credit
  ```
  （`_apply_draw_credit`，L1486-1493）
- **归属**：本回合内抽到它的 source transition
- **触发条件**：仅同回合；跨回合抽的牌不追踪
- **Magic number 意图**：`0.3` = 间接贡献折扣（与 effect source credit 一致）
- **已知限制**：抽到的牌本身的 reward 已经包含 HP 惩罚，如果抽到的牌恰好成为"被 HP 惩罚分担的打牌"，credit 会把负值也传回源，抽牌源可能被负 reward 反向惩罚

#### 3.1.9 回血对称放大（Phase B）

- **公式**：
  ```
  player_hp_delta = new_snap.hp - old.hp
  if player_hp_delta > 0:
      prev_ratio = old.hp / old.max_hp
      anxiety_factor = 1 + (1 - prev_ratio) × 2
      heal_reward = player_hp_delta × 0.1 × anxiety_factor
      last_t.reward += heal_reward
      last_t.extra["card_heal"] = player_hp_delta
  ```
  （L1719-1727）
- **归属**：Phase B 结算的 `last_t`（上一张打出的牌）
- **触发条件**：Phase B（每次 `combat_act` 调用），且 `prev_combat_snap` 和 `new_snap` 都存在、`turn_buffer` 非空
- **Magic number 意图**：`0.1` 与 §3.1.2 的掉血惩罚系数相同（对称放大）；anxiety_factor 同理

#### 3.1.10a Block 吸收伤害 credit（新增）

- **公式**：`credit = absorbed × 0.05 × 0.3 × (blk_amt / total_block)`，其中 `absorbed = prev_combat_snap.block if hp_lost > 0 else 0`
- **归属**：本回合打过 `card_block > 0` 的 transition（按 block 比例分摊）
- **触发条件**：Phase A flush 和 `on_combat_end` flush，`hp_lost > 0` 且 `_block_sources` 非空
- **Magic number**：`0.05` 匹配 block 基础分，`0.3` 匹配 effect source credit ratio
- **已知限制**：`hp_lost == 0` 时 absorbed=0（保守）

#### 3.1.10b Calm 退出能量 credit（新增）

- **公式**：`credit = 2.0 × 0.1 × 0.3 = 0.06`（固定）
- **归属**：进入 Calm 的 transition（通过 `_calm_source` 追踪，支持跨回合 trajectory index）
- **触发条件**：Phase B 检测到 `old.stance == "Calm" && new.stance != "Calm"`；战斗结束时自动归 Neutral 的情形不发放（产生的 2 能量无意义）
- **Magic number**：`0.1` 与 `energy_bonus` 系数一致（每点能量 0.1 reward），`0.3` 间接折扣

#### 3.1.10c Wrath 挨打 credit（**已撤销**）

**已撤销**（临时回滚，因破坏 agent 当前唯一可行战术）。历史设计保留在 git 历史中供参考：曾对进入 Wrath 的牌回传 `credit = -hp_lost × 0.5 × 0.1 × anxiety × 0.3`，但会把 Eruption（agent 当前唯一成型的攻击战术）打成强负信号，导致 avg floor 从 ~7.4 退到 ~5.6。回滚后 Wrath 进入/退出不再产生 causal credit（只保留 stance transition 追踪供 diagnose 使用）。

#### 3.1.10 Energy bonus

- **公式**：
  ```
  expected_energy = _energy_before_play - _cost_of_played_card
  actual_energy = new_snap.energy
  energy_gained = max(actual_energy - expected_energy, 0)
  last_t.extra["energy_bonus"] = energy_gained × 0.1
  ```
  （L1730-1734）
- **归属**：Phase B 结算的 `last_t`（产能那张牌）
- **触发条件**：Phase B
- **Magic number 意图**：`0.1` = 每点能量生成值 0.1 reward，约等于一张小费牌的价值
- **生效时机**：`energy_bonus` 在 `extra` 里暂存，Phase A flush 时被 §3.1.1 的 `+=` 累加到 reward

### 3.2 Combat 结束层（`on_combat_end`，约 L2270-2469）

#### 3.2.1 Combat completion

- **公式**：`combat_completion_reward = (hp / max_hp) × 0.5`
- **归属**：`current_trajectory[-1].reward += combat_completion_reward`（flush 后最后一个 transition）
- **触发条件**：`won and hp > 0 and max_hp > 0`（L2379-2383）
- **Magic number 意图**：`0.5` 为满血通关 combat 的上限；半血 0.25；和 terminal win bonus `10.0` 比低两个数量级，不抢主信号

#### 3.2.2 Clutch win bonus（设计未定）

- **当前实现（`93b786b`）**：
  ```
  end_hp_ratio = clip(hp / max_hp, 0, 1)
  clutch_bonus = max(0, 1 - end_hp_ratio) × 0.5
  # 归属：倒序找首个 card_kills > 0 的 combat transition，
  #       fallback 首个 played_card_id 非空的 combat transition
  target_t.reward += clutch_bonus
  ```
  （L2388-2409）
- **归属**：致死卡 → fallback 最后一张打出的牌
- **Magic number 意图**：`0.5` 上限（残 HP 0 时最大 0.5）
- **已知问题**：致死卡不一定是关键卡——Defend 类、转防的牌、投毒铺伤的牌可能才是真正救命的；均摊到整场战斗也不对（稀释成噪音）。**状态：临时方案，待重新设计**

#### 3.2.3 Draft 回溯反馈

- **公式**：
  ```
  hp_delta = (hp_after - hp_before) / 20.0     # 20 HP = 1.0 reward unit
  # 难度：boss=3.0 / elite (max_hp > 100) =2.0 / normal=1.0
  decay = 0.7 ** n_combats_back
  retroactive_reward = hp_delta × difficulty × decay
  draft_transition.reward += retroactive_reward
  ```
  （L2411-2458）
- **归属**：`current_trajectory` 从后向前扫到的所有 draft transition，每跨过一个 combat 段 `n_combats_back += 1`
- **停止条件**：`n_combats_back > 5`（5 场战斗后衰减太小，提前剪枝）
- **Boss ID 判断**：`{"slime_boss", "hexaghost", "guardian", "automaton", "collector", "champ", "awakened", "time_eater", "donu", "deca", "heart", "corrupt_heart", "the_heart"}`（L2421-2423，子串匹配）
- **Magic number 意图**：
  - `20.0`：满血 20 HP 值 1.0 reward unit（和 terminal win `10.0` 比是 1/10 量级）
  - 难度系数：boss 打伤害 20 HP 给 draft −3.0，普通战只给 −1.0
  - `0.7 ** n_combats_back`：几何衰减（1.0 / 0.7 / 0.49 / 0.343 / 0.24）

#### 3.2.4 最后一击近似 Phase B（`93b786b` 新增，L2276-2304）

- **设计背景**：`on_combat_end` 被调用时 `game_state` 已不含 `combat_state`，`_combat_snapshot` 不可用。致死一击（击败所有敌人那张牌）若不补算，会缺 damage / kills / heal 三项信号（bug 之一：致死一击 0 分）
- **公式**（条件：`won and last_t.extra.played_card_id` 存在）：
  ```
  prev_enemy_hp_total = sum(max(0, h) for h in prev.enemy_hps)
  last_t.extra.card_damage += prev_enemy_hp_total     # 假设致死一击清零
  last_t.extra.card_kills += prev.n_alive
  hp_delta = hp - prev.hp
  if hp_delta > 0:
      anxiety = 1 + (1 - prev_hp/prev_max_hp) × 2
      last_t.reward += hp_delta × 0.1 × anxiety        # 和 Phase B heal 一致
  ```
- **已知损失**：
  - `card_block`：致死一击若同时产生 block（少见），无法估算
  - `energy_bonus`：致死一击产能的场景，无法估算
  - `poison_apply`：致死一击投毒后敌人死亡，无法登记为 effect source
  - `block / energy / poison_apply 在 Watcher 场景下罕见共存，放弃`（原注释）
- **失败场景**：`won=False` 时不补算 damage/kills（没有"清零"的假设），但 heal 仍补算

### 3.3 策略层

#### 3.3.1 Draft（`pick_card`，L1834-1896）

- **公式**（L1858-1883）：
  ```
  # 从 CARD_DATA 反归一化出真实 damage / block / cost
  dmg = damage_single + damage_aoe     # 各 × 40 还原
  blk = block_gain × 30                # 还原
  cost = round(cost × 4)（至少 1）
  base_reward = min((dmg + blk) / cost × 0.05, 0.3)
  skip → base_reward = 0
  ```
- **归属**：`draft` transition 即时
- **Magic number 意图**：
  - `0.05`：与 combat 伤害系数对齐
  - cap `0.3`：避免超大 (dmg+blk)/cost 牌（如 0 费 30 block）爆分，和 combat 一张好牌的量级基本对齐
  - fallback：`card` 对象本身带 `damage / block / cost`，查不到 CARD_DATA 时降级使用

#### 3.3.2 Draft（升级 / 删牌 / Boss relic）

- **公式**：`reward = 0.0`
  - `_pick_from_candidates`（L1898-1936）：reward = 0，由战斗结果回溯反馈（但此场景不会被 §3.2.3 draft 回溯扫到，因为 trajectory 已过）
  - `_pick_worst_candidate`（L1938-1969）：reward = 0（删牌场景，未接入回溯）
  - `_pick_boss_relic`（L1971-2011）：reward = 0
- **说明**：这些场景"暂时没设计"，依赖长期 terminal reward 通过 GAE 回传，信号稀疏

#### 3.3.3 Path（`choose_path`，L2013-2066）

- **公式**：`reward = 0.1`（常数，L2057）
- **归属**：每个 path transition
- **Magic number 意图**：给"走了一步"的存活信号；差异化完全依赖下游 combat / draft reward 回传
- **已知问题**：所有 path 选择同分，无法表达"选 R 比选 M 更安全"

#### 3.3.4 Rest / Campfire（`make_choice` 内，L2098-2102）

- **公式**：
  ```
  heal_pct = min((max_hp - current_hp) / max_hp, 1.0)
  reward = 0.1 × heal_pct
  ```
- **归属**：choice transition
- **Magic number 意图**：满血休息 0，半血 0.05，1 HP 休息 ≈ 0.1——缺血越多休息价值越大

#### 3.3.5 其他 choice（事件 / 非 rest 的 campfire 分支）

- **公式**：`reward = 0.1`（常数，L2104）
- **归属**：choice transition
- **已知问题**：无差异化，上下文完全依赖 choice head 的 logits 和 terminal reward 回传

#### 3.3.6 Potion / Shop

- **Potion**（`potion_decide`，L2189）：`reward = 0.0`
- **Shop**（`shop_decide`，L2259）：`reward = 0.0`
- **说明**：信号完全来自下游 combat reward 和 terminal reward

### 3.4 终局（`on_run_end`，L2480-2491）

#### 3.4.1 Win

- **公式**：`current_trajectory[-1].reward += 10.0`（L2488-2490）
- **归属**：trajectory 最后一个 transition（任意类型，通常是击杀 Act 3 boss 的那张牌 / 或选定 boss relic 的 choice）
- **Magic number 意图**：最大单点信号，`10.0` ≈ 100 点伤害的等效量

#### 3.4.2 Lose

- **公式**：`current_trajectory[-1].reward += -3.0`
- **归属**：同上，最后一个 transition
- **Magic number 意图**：比 win 小，避免"死亡惩罚过重导致保守"。死亡惩罚不对称设计

## 4. Credit Assignment 详解

### 4.1 抽牌链

- Phase B 检测新出现在手牌的 card_id，用 `_unique_draw_key` 去重，存 `_draw_credit_map[key] = source_idx`（L1751-1756）
- 该牌被打出时（下一次 `combat_act` 的 Phase D），`extra["drawn_by_idx"] = source_idx`（L1810-1812）
- Phase A flush 时 `_apply_draw_credit` 把该牌的 `t.reward × 0.3` 回传给 `turn_buffer[source_idx]`（L1487-1493）
- **跨回合限制**：`_draw_credit_map = {}` 在每次 flush 时清空（L1693），跨回合抽到的牌不追踪

### 4.2 能量链

- Phase B 记录 `_energy_before_play`、`_cost_of_played_card`
- 下一次 Phase B 比较 `actual_energy` vs `expected_energy`，差值 × 0.1 存入 `extra.energy_bonus`
- Phase A flush 通过 `+=` 累加到 `t.reward`
- **说明**：产能牌（如 Seek、Bloodletting）自己获得 energy_bonus，没有"传给后续使用能量的牌"——能量使用的 reward 由那张牌的 damage/block 直接反映

### 4.3 Poison 跨回合

- 同回合内：投毒那张牌在 Phase B 通过 enemy_poison 差值登记为 `_turn_effect_sources.append((buf_idx, "poison", amount))`（L1779-1782）
- Phase A flush 末尾：把 `_poison_sources` 里的 buf_idx 转 trajectory_idx（L1678-1681）并记录新回合的 `_pending_poison_tick = sum(enemy_poison)`（L1684-1686）
- 下一回合 Phase A flush：按 `_poison_sources` 的 amount 比例把 `_pending_poison_tick × 0.05 × 0.3` 分摊给投毒源（L1640-1648）
- `on_combat_end`：清空 `_pending_poison_tick` 和 `_poison_sources`（L2368-2369），跨战斗不保留

### 4.4 Strength / Dex / Vuln / Weak Effect Source Pool

- Phase B 通过 `new_snap - old` 差值检测 buff/debuff 的增长（L1761-1782）
- 记录 `_turn_effect_sources.append((buf_idx, effect_type, amount))`
- Phase A flush 同回合结算：从 `buf_idx + 1` 向后遍历 turn_buffer，按每种 effect 的规则把 credit 加到 `turn_buffer[buf_idx]`（L1651-1672；`on_combat_end` 内 L2342-2359）
- 每回合结束后 `_turn_effect_sources = []`（L1689；`on_combat_end` L2367）

### 4.5 Stance 副作用（部分实现 — 仅 Calm credit，Wrath 负 credit 已回滚）

**部分实现**：仅保留 Calm 退出能量 +credit 和 Block 吸收伤害 +credit；Wrath 挨打 −credit 已回滚（见 §3.1.10c）。

- **Calm 退出 +credit**：Phase B 检测 `old.stance != "Calm" && new.stance == "Calm"` 时，登记 `_calm_source = ("buffer", buf_idx)`；在 Calm 主动退出（如 Vault / Miracle / Worship 进 Divinity）时发放 `credit = 2 × 0.1 × 0.3 = 0.06`。
  - 战斗结束时 stance 自动归 Neutral，不发放（战斗结束后的 2 能量无意义）。
- **Wrath 挨打 −credit**：**已回滚**（见 §3.1.10c / §5）。
- **Block 吸收伤害 +credit**：见 §4.6。

Source 跨回合通过 `_promote_source_to_trajectory`（Phase A flush 时把 `("buffer", idx)` 升级为 `("trajectory", base_idx + idx)`）保持可追溯。

### 4.6 Block 吸收伤害 credit（已实现）

- Phase B：每张 `card_block > 0` 的 transition 登记为 `_block_sources.append((buf_idx, card_block))`。
- Phase A flush / `on_combat_end` flush：
  ```
  prev_block = prev_combat_snap.block  # 上回合末 block
  absorbed = prev_block if hp_lost > 0 else 0  # hp_lost==0 时无法区分被抵消 vs 自然衰减，保守置 0
  for (buf_idx, blk) in _block_sources:
      proportion = blk / total_block
      credit = absorbed × 0.05 × 0.3 × proportion
      turn_buffer[buf_idx].reward += credit
  ```
- Magic number：`0.05` 匹配 block 基础分；`0.3` 匹配 effect source credit ratio。
- 已知限制：若 `hp_lost == 0`（block 完全挡住攻击），absorbed 设为 0（under-count）。上游无明确的 enemy_attack 信号，无法区分"block 抵消攻击"vs"自然衰减 / 敌人未攻击"。

### 4.7 Clutch win bonus 归属

**待设计**：见 §3.2.2

## 5. 已知设计问题

1. **Clutch win bonus 归属未定**：见 §3.2.2。致死卡 ≠ 关键卡，fallback 到"最后一张打出的牌"也不合理
2. **END TURN 信号缺失**：当前 reward = 0，主动 END TURN（格挡溢出 / 伤害溢出 / 保留 retain / 避 Wrath / 无用手牌）学不到。依赖策略文档 §2.10 derived feature 完备后通过 V function 间接学
3. ~~**Stance 副作用未捕捉**：Wrath 下伤害放大、Divinity 进入、Calm 退出的 reward 影响没有体现~~ **部分解决**（见 §4.5 / §3.1.10a-b）：Block 吸收 / Calm 退出 causal credit 已实现；Wrath 挨打 −credit **已回滚**（见 §3.1.10c）。Divinity 进入仍未显式 credit（通常靠 Worship 触发，进入 Divinity 本身 = 3 能量大收益，由 energy_bonus 覆盖）
3a. **Wrath 过度循环未解决**——回滚负 credit 后，进入/退出 Wrath 无任何反馈信号，agent 可能继续滥用 Eruption；需要未来设计不破坏 Eruption 基础价值的信号（如仅在 Wrath 持续多回合且受大伤时给 −credit，或只对"无攻击牌时进 Wrath"归因）。
4. **Magic number 不一致**：
   - Strength / Dex credit：`amount × 0.05 × 0.3`
   - Vulnerable credit：`dmg × 0.5 × 0.05 × 0.3`
   - Weak credit：`amount × 0.1 × 0.3` ← 系数量级和其他三种不一致（用 0.1 而非 0.05）
5. **Vulnerable credit 量级被淹没**：`dmg × 0.0075` 相对攻击牌本身的 `dmg × 0.05` 几乎无信号（1/6.67 量级），施加 Vuln 的牌得到的回传信号远小于直接打伤害
6. **Draft 回溯惩罚量级偏大**：elite 战一次掉 20 HP 给 draft transition `−1.0 × 2.0 × 1.0 = −2.0` 单次惩罚，相对 terminal `win=+10` 量级偏大（1/5）；多场累加可能让 draft reward 淹没 terminal 信号
7. **`compute_strategy_reward` 是 dead code**（L1496-1498）：定义了但没有任何调用方，是 V5 遗留
8. **`on_combat_end` 最后一击近似有损失**：`block / energy / poison_apply` 在 `on_combat_end` 时无法获取，最后一张防御牌或投毒牌的这些效果未计入
9. **抽牌回传可能传负值**：被抽到的牌的 reward 已包含 HP 惩罚，若该牌是导致负 reward 的主因，抽牌源会被反向惩罚
10. **Poison source 跨战斗不保留**：`on_combat_end` 清空 `_pending_poison_tick`，如果一场战斗的最后一回合靠持续毒 tick 击杀敌人，投毒源拿不到最终 DOT credit
11. **HP anxiety factor 疑似反向激励源，已回滚验证**（2026-04-18）：原 `anxiety_factor = 1 + (1 - hp_ratio) × 2` 会在低 HP 时把 HP penalty 放大 3 倍，而 END TURN 不分摊 HP penalty——叠加后 agent 在低 HP 时学到"少出牌 → END TURN"的反向激励。已改为常量 1.0（涉及 Phase A flush / on_combat_end flush 的 HP penalty、Phase B heal、on_combat_end heal 共 4 处）。如回滚后 avg_floor 改善，确认此假设；否则再恢复

## 6. 与策略文档的对照

### 6.1 决策点 ↔ Reward 触发点

| 决策点 | Reward 触发位置 | 当前公式 | 状态 |
|---|---|---|---|
| Combat：play card | Combat 层 §3.1 | 多机制叠加（基础分 + HP 惩罚 + heal + energy + draw + effect source credit） | ✓ |
| Combat：end turn（主动） | — | 0 | ⚠ 信号缺失 |
| Combat：end turn（被动） | — | 0 + mask 兜底（待查） | ⚠ 依赖 mask |
| Combat：potion | §3.3.6 | 0 | ✗ 无信号 |
| Draft：pick | §3.3.1 + §3.2.3 | base + 战斗后回溯 | ✓ |
| Draft：upgrade / purge / boss relic | §3.3.2 | 0 | ✗ 无信号 |
| Path | §3.3.3 | 0.1 常数 | ⚠ 无差异化 |
| Rest / Campfire | §3.3.4 | 0.1 × heal_pct | ✓ |
| Event choice | §3.3.5 | 0.1 常数 | ⚠ 无差异化 |
| Shop | §3.3.6 | 0 | ✗ 无信号 |
| Boss Relic | §3.3.2 | 0 | ✗ 无信号 |
| Potion（战斗外） | 待查 | 待查 | 待查 |
| Neow 祝福 | 待查 | 待查 | 待查 |

### 6.2 编码项 ↔ Reward 反馈

引用策略文档 §2 的编码项，列出哪些有对应 reward 反馈、哪些没有：

- **A1 HP** → §3.1.2 HP penalty + §3.1.9 heal + §3.2.3 draft 回溯 + §3.2.1 combat completion + §3.2.2 clutch bonus ✓
- **A2 Block** → §3.1.1 `card_block × 0.05` ✓
- **A3 能量** → §3.1.10 energy_bonus ✓
- **A4 金币** → ✗ shop / rest dig 无 reward 反馈
- **A5 楼层** → terminal reward 通过 GAE 回传，间接 ✓
- **A6 Stance** → ✗ 无 reward 反馈（已知问题 §5.3）
- **A7 Power（玩家）** → 仅 Strength/Dex 有 effect source credit ✓ 部分；其他 buff（Energized / Combust / EchoForm）无 reward 反馈 ✗
- **A8 药水槽** → ✗ 无直接 reward；使用药水时 §3.3.6 reward = 0
- **B1 手牌效果** → §3.1.1 基础分 ✓
- **B2 is_playable** → combat mask 兜底，不直接进 reward ✓（通过 mask）
- **B3 升级** → §3.3.1 通过 CARD_DATA 反归一化体现升级数值 ✓
- **B4 retain / ethereal / innate / exhaust** → ⚠ reward 未显式识别，靠 `card_damage / card_block` 间接
- **B5-B7 抽 / 弃 / 消耗堆** → ⚠ 仅通过抽牌回传 §3.1.8 部分反馈；消耗堆完全无反馈
- **B8 战斗外卡组** → §3.2.3 draft 回溯间接 ✓
- **C1 敌人 HP** → §3.1.1 通过 `card_damage` 间接 ✓
- **C2 意图** → ✗ 无 reward 直接反馈；hp_penalty 通过"被意图打了多少"间接
- **C6 敌人 power** → Vuln / Weak / Poison 有 effect source credit ✓；CurlUp / Ritual / Malleable 等无
- **C7 敌人 ID** → §3.2.3 draft 回溯用 boss_id 判断难度 ✓（间接）
- **D1 回合数** → ✗ 无 reward 反馈
- **D2 房间类型** → §3.2.3 通过 elite/boss 难度乘子 ✓
- **D4 本回合历史** → ✗ 无 reward 反馈（和 END TURN 信号缺失联动）
- **E 地图** → ✗ path reward 常数 0.1，无差异化
- **F 遗物** → ✗ boss relic reward = 0
- **G1 Event** → §3.3.5 常数 0.1 ⚠
- **G2 Shop** → ✗ reward = 0
- **G3 Campfire** → §3.3.4 rest heal ✓；smith / dig / lift / toke 走 §3.3.5 常数
- **G4 Draft** → §3.3.1 ✓
- **H 全局** → terminal reward 间接 ✓

## 7. 待定事项清单

- Clutch win bonus 归属设计（§3.2.2 / §5.1）
- END TURN 信号设计：是否完全靠 V function，还是加 derived signal（§5.2）
- Stance 副作用 reward 设计（§5.3）
- Magic number 系数统一：Weak 的 `0.1` vs 其他的 `0.05`（§5.4）
- Vulnerable credit 量级放大（§5.5）
- Draft 回溯量级 vs terminal 量级平衡（§5.6）
- 清理 dead code `compute_strategy_reward`（§5.7）
- Shop / Boss relic / Potion 的 reward 设计（§3.3 所有 ✗ 项）
- 战斗外非 rest choice 的差异化 reward
- Poison 跨战斗 credit 是否保留（§5.10）
- Draft upgrade / purge / `_pick_from_candidates` 是否接入回溯机制
- 抽牌回传负值问题（§5.9）
- Potion 战斗外决策是否有独立 reward 通路（待查）
- Neow 祝福是否有 reward 通路（待查）

## 8. 不在本文档范围

- 训练超参（lr / gamma / lambda / clip_ratio / entropy_coeff）
- 网络架构（`shared_repr` / head 大小 / token projection）
- Communication Mod 数据采集细节
- 具体 bug fix 的历史 diff（见 git log）

# V7 架构设计（搜索 + 手调评估）

## 0. 文档状态

- 版本：v1 草案
- 日期：2026-04-17
- 目标：A0 Watcher 稳定过 floor 16（Act 1 Boss）为第一里程碑
- 信心：**中**。TurnSolver 骨架可用，评估函数需要替换/扩展。bottled_ai 的 Watcher 策略是 SOTA 参考。
- 路线：放弃 RL，走 bottled_ai 风格的"搜索 + 手调 heuristic"。

---

## 1. 路线切换的原因

### 1.1 RL 在 STS 上失败的证据

- **V3 DQN**：34 局训练，floor 6-16 plateau，已归档。
- **V4 PPO + Transformer**：2748 局训练，avg floor 4.2（基本等同随机）。
- **V5 Solver + PPO 策略**：策略层信号稀疏，未达预期。
- **V6 新奖励设计**：多轮 reward shaping（HP 焦虑、致死一击补丁、heal 对称放大），仍未收敛到稳定通关。
- 根因：STS 的 observation 空间巨大，reward 信号稀疏（一局 50+ 策略决策，胜败才给信号），PPO 样本效率远低于搜索型方法。

### 1.2 bottled_ai 的 52% 标杆

- A20 Watcher 52% 胜率，社区 SOTA（无 ML，纯搜索 + 手调规则）。
- 仓库：https://github.com/xaved88/bottled_ai
- 架构：`rs/calculator/executor.py` 做 play path 枚举，`rs/common/comparators/` 做 lexicographic 比较，策略层用 handler 硬编码规则。
- 作者自述："weighs about 40 different values against each other"——实际是 42 个 **lexicographic comparison**，不是加权和。

### 1.3 V5/V6 遗产的复用

- **可复用**：StSRLSolver 游戏引擎（含 Watcher stance/mantra/divinity 完整实现）、V5 的 TurnSolver 骨架、V6 的 encoding data tables。
- **可抛弃**：V4-V6 的 PPO trainer、Transformer backbone、reward shaping 逻辑。
- **新写**：heuristic evaluator（替换 TurnSolver 当前的 weighted-sum 评估为 lexicographic comparator）、策略层硬编码规则（draft/path/choice）。

---

## 2. V5 TurnSolver 现状评估

### 2.1 代码现状

文件：`STSRLSOLVER_PATH/packages/training/turn_solver.py`（1342 行）

**能跑的程度：8/10（可直接作为 V7 骨架）**

- **搜索算法已实现**（自适应切换）：
  - 小回合（<300 节点）：exact DFS + alpha-beta 剪枝
  - 中等（300-5000）：best-first search + 优先队列
  - 大回合（5000+）：beam search（宽度 20，保留 4 个 setup slot）
- **状态模拟可靠**：通过 `CombatEngine.copy() + execute_action()` 做真实模拟（包含 Watcher 的 stance/mantra/divinity）。
- **搜索空间**：每回合搜索所有合法 action（出牌 / 使用药水 / Scry 丢弃），包含 EndTurn。
- **Plan cache + tree reuse**：LRU 缓存 512 条计划，同状态秒回。
- **Neural leaf eval 钩子**：已预留（`neural_eval` 参数），当前未使用。
- **可跑脚本**：`REPO_ROOT/tools/test_solver_game.py`（Ironclad 端到端测试，打印 floor/HP/time；Watcher 未测）。

### 2.2 当前评估函数（`_score_terminal`，turn_solver.py:262-363）

- **不是空壳**，是 weighted-sum heuristic（但维度较浅）。
- 维度（权重来源 `training_config.SOLVER_SCORING`）：
  - 死亡 = `-1_000_000`；lethal = `+1_000_000 + hp*200`。
  - 模拟敌方回合（真打一次 enemy turn），测实际 HP 损失：`-1.0 * actual_hp_lost`。
  - 敌方击杀：`+10.0 * enemies_killed`。
  - 剩余敌方 HP 比例：`-1.0 * (now_ehp / orig_ehp)`。
  - 估计回合数击杀：`-1.0 * min(est_turns, 10)`。
  - 未消耗能量 + 可打牌：`-1.0 * energy`, `-0.5 * playable`。
  - **Stance**：Calm bonus（当前为 0，"let search decide"）；Wrath 来袭惩罚（当前为 0，同上）。
- 问题：对 Watcher 几乎没评估，没有 Mantra、Divinity 储备、Stance 切换的显式加分。

### 2.3 差距清单（要补什么才能当 V7 骨架）

| 项 | 现状 | V7 需要 |
|---|---|---|
| 搜索算法 | ✅ 成熟三档自适应 | 复用 |
| 评估函数（通用） | ⚠️ Weighted-sum，维度浅 | **替换为 lexicographic comparator**（参考 bottled_ai 42 维） |
| Watcher Divinity/Mantra 评估 | ❌ 几乎没有 | 新增：Mantra 累积、Divinity 预留、避免无意义出 Wrath |
| 策略层（draft/path/choice） | ❌ 完全不存在 | **全新写**：硬编码规则 handler |
| Character 默认 | Ironclad（test_solver_game.py）| 改 Watcher 默认 |
| 端到端 Watcher 跑通脚本 | ❌ 未测 | 写一个 test_solver_game_watcher.py |

---

## 3. bottled_ai 核心评估函数参考

### 3.1 架构要点（非加权和！）

bottled_ai **不用**加权和打分，而是 **lexicographic comparison**：

```python
# rs/common/comparators/common_general_comparator.py
class CommonGeneralComparator:
    def does_challenger_defeat_the_best(self, best, challenger, original) -> bool:
        for c in self.comparisons:
            v = c(best, challenger)  # Optional[bool]
            if v is not None:
                return v  # 第一个有差异的维度决定胜负
        return False
```

每个 comparison 返回 `Optional[bool]`：`None` 表示该维度打平（继续下一维），否则返回 challenger 是否更优。**严格优先级链**，不是加权求和。

### 3.2 42 维 comparison 清单（按优先级从高到低）

源：`rs/common/comparators/common_general_comparator.py` 的 `default_comparisons`

| # | Comparison | 含义 |
|---|---|---|
| 1 | `battle_not_lost` | 没死优先 |
| 2 | `battle_is_won` | 赢了优先 |
| 3 | `preserve_revive_options` | 保留复活手段（Fairy in a Bottle 等）|
| 4 | `most_optimal_winning_battle` | 赢的情况下：max HP > lesson learned kills > ritual dagger > incoming damage > ... > energy（11 层嵌套 lex）|
| 5 | `no_blasphemy` | 避免 Blasphemy 负面状态 |
| 6 | `most_free_early_draw` | 早期免费抽牌多 |
| 7 | `most_free_draw` | 免费抽牌多 |
| 8 | `most_lasting_intangible` | Intangible 多（>=1 才算，Watcher 的 Deva 类核心）|
| 9 | `least_incoming_damage_over_1` | 降低来袭伤害（阈值 2）|
| 10 | `most_great_player_powers` | "极好"玩家 power 数 |
| 11 | `most_dead_monsters` | 已死敌人多 |
| 12 | `most_tranquility` | Tranquility 状态（Watcher Calm 切换）|
| 13 | `most_enemy_talking_to_hand` | 敌人身上 "Talk to the Hand"（Watcher 核心牌）|
| 14 | `most_enemy_vulnerable` | 敌方 Vulnerable 多 |
| 15 | `most_enemy_weak` | 敌方 Weak 多 |
| 16 | `least_awkward_shivs` | Shiv 不要卡手 |
| 17 | `killed_with_lesson_learned` | Lesson Learned 击杀（学卡）|
| 18 | `most_powered_up_ritual_dagger` | Ritual Dagger 层数 |
| 19 | `kept_expensive_decreasing_cost_retain_cards` | 保留递减 cost 的 retain 卡 |
| 20 | `lowest_health_monster` | 最低血敌人再低 |
| 21 | `lowest_total_monster_health` | 总 HP 低 |
| 22 | `lowest_barricaded_block` | 敌方 barricade block 低 |
| 23 | `lowest_enemy_plated_armor` | 敌方 plated armor 低 |
| 24 | `most_orb_slots` | Orb slot（Defect）|
| 25 | `most_channeled_orbs` | 已 channel 的 orb |
| 26 | `most_draw_pay_early` | 早期付费抽牌 |
| 27 | `most_draw_pay` | 付费抽牌 |
| 28 | `most_good_player_powers` | "好"玩家 power 数 |
| 29 | `least_bad_player_powers` | "坏"玩家 power 少 |
| 30 | `most_less_good_player_powers` | "次好"玩家 power 数 |
| 31 | `least_enemy_artifacts` | 敌方 artifact 少（debuff 能过）|
| 32 | `most_bad_cards_exhausted` | 废牌被 exhaust |
| 33 | `most_powered_up_genetic_algorithm` | Genetic Algorithm 层数 |
| 34 | `most_cards_left_in_hand` | 手牌剩下多 |
| 35 | `least_incoming_damage` | 来袭伤害低（无阈值）|
| 36 | `most_ethereal_cards_saved_for_later` | Ethereal 卡留给下回合 |
| 37 | `most_powered_up_claws` | Claw 层数（Defect）|
| 38 | `stance_is_not_wrath` | **不在 Wrath**（避免下回合被翻倍打）|
| 39 | `stance_is_calm` | **在 Calm**（下回合 +2 能量）|
| 40 | `least_powered_down_steam_barrier` | Steam Barrier |
| 41 | `most_block_saved_for_next_turn` | Block 带到下回合（Barricade/Calipers）|
| 42 | `most_energy` | 剩余能量多 |

### 3.3 Power 分类（`powers_we_like` 等）

- **"great"**：`ECHO_FORM`, `ELECTRO`（pwnder_my_orbs 特化）
- **"good"**（48 个）：`ACCURACY`, `AFTER_IMAGE`, `BARRICADE`, `BERSERK`, `BLUR`, `BUFFER`, `CORRUPTION`, `DEMON_FORM`, `DEVA`, `DEVOTION`, `ECHO_FORM`, `ELECTRO`, `ENVENOM`, `ESTABLISHMENT`, `FEEL_NO_PAIN`, `FIRE_BREATHING`, `FOCUS`, `FORESIGHT`, `INFINITE_BLADES`, `INTANGIBLE_PLAYER`, `JUGGERNAUT`, `LIKE_WATER`, `MACHINE_LEARNING`, `MANTRA_INTERNAL`, `MASTER_REALITY`, `MENTAL_FORTRESS`, `METALLICIZE`, `NIRVANA`, `NOXIOUS_FUMES`, `OMEGA`, `PANACHE_INTERNAL`, `PLATED_ARMOR`, `REPAIR`, `RUSHDOWN`, `SADISTIC`, `STUDY`, `THORNS`, `THOUSAND_CUTS`, `TOOLS_OF_THE_TRADE` 等。
- **"less good"**：`ARTIFACT`, `DEXTERITY`, `ENERGIZED`, `STRENGTH`, `VIGOR`（可能被针对或浪费）。
- **"dislike"**：`DEBUFFS` 全集（Weak / Vulnerable / Frail / etc.）。

### 3.4 Watcher 特有评估（来自 `powers_we_like` + stance comparators）

- `MANTRA_INTERNAL`（Mantra 累积）算 good power。
- `DEVA`, `DEVOTION`, `ESTABLISHMENT`, `NIRVANA`, `MENTAL_FORTRESS`, `MASTER_REALITY`, `STUDY`, `RUSHDOWN`, `LIKE_WATER`, `FORESIGHT`：Watcher 核心 power 全入 good 集合。
- `most_tranquility`（#12）和 `stance_is_calm`（#39）：Calm 受青睐。
- `stance_is_not_wrath`（#38）：Wrath 结束回合 = 下回合双倍挨打，强烈避免。
- `cards_that_exit_wrath`：`EMPTY_BODY`, `EMPTY_FIST`, `EMPTY_MIND`, `FEAR_NO_EVIL`, `INNER_PEACE`, `TRANQUILITY`, `VIGILANCE`（用于从 Wrath 出来）。
- **Divinity 评估**：未找到显式 comparator，推测通过 "most energy" + "most great powers" 间接评估。**V7 应新增 Divinity 预留分**（MANTRA_INTERNAL 接近 10 时额外加分）。

### 3.5 Watcher 策略层（peaceful_pummeling）

源：`rs/ai/peaceful_pummeling/config.py`

**卡组构建（DESIRED_CARDS_FOR_DECK，部分）**：

```
blasphemy:1, talk to the hand:3, rushdown:1, tantrum:2, battle hymn:1,
mental fortress:2, vigilance:1, tranquility:2, wallop:1, flurry of blows:2,
empty body:2, indignation:1, crush joints:1, fear no evil:2, empty fist:1,
reach heaven:1, inner peace:1, cut through fate:1, eruption:1, crescendo:1,
halt:1, ritual dagger:1, ... perseverance:2, wheel kick:1, like water:1
```

**删牌优先级**：`conjure blade > vault > omniscience > meditate > defend > strike > bite`。

**升级优先级**：`Apotheosis > Blasphemy > Eruption`。

**Watcher 特殊约定**：**不用药水打普通战**（留给 boss）——peaceful_pummeling.py 明确排除 `PotionsEventFightHandler`。

### 3.6 搜索剪枝（11000 paths）

bottled_ai 的 `executor.py` 用 `play_path.py` 做 DFS，**不是剪枝成 11000**，而是 **生成了 ~11000 条 play path**，然后用 comparator 两两 PK 选出最优。无评估权重，只有严格优先级。

---

## 4. V7 架构

### 4.1 战斗层（search + heuristic）

**搜索算法**：直接复用 V5 `TurnSolver` 三档自适应（已验证可运行）。

**评估函数改造**（核心工作）：

- **保留**：`_SCORE_DEATH / _SCORE_LETHAL`（两个 terminal 极值，与 bottled_ai 的 `battle_not_lost / battle_is_won` 一致）。
- **替换 `_score_terminal` 的 weighted-sum 部分**为 **lexicographic comparator**：
  - 两候选 state PK：逐维比较，第一个有差异的决定胜负。
  - 实现：把 best-first / DFS / beam 的 `score` 升级为 tuple（按优先级排列），或者改写 `__lt__` 为 lex 比较。
  - **量级**：挑选 bottled_ai 42 维中与 StSRLSolver engine 已实现的匹配的 ~20-25 维作为 MVP（见 §5）。
- **Watcher 扩展**（bottled_ai 基础上再加）：
  - Mantra 累积分：越接近 10 越好（差一触发 Divinity）。
  - Divinity 预留分：手牌里有能触发 Mantra 的牌 + 当前 Mantra ≥7 时加分。
  - Stance：Calm +2 能量价值、Wrath 挨打翻倍（结束回合时）、Divinity 伤害 3x（出牌时）。

**Simulator 接口**：`StSRLSolver.packages.engine.CombatEngine`（已支持 Watcher 完整 stance 机制，见 `content/stances.py`）。

### 4.2 策略层（硬编码优先级规则）

全新写 `v7_strategy.py`，四个 handler：

**Draft（选牌）**：
- 优先级列表直接抄 bottled_ai Watcher（§3.5）。
- 逻辑：当前候选牌中，选 DESIRED 列表里"还差数量最多"的；候选都不在列表时选 Skip。
- 特殊：Snecko Eye / Runic Pyramid 激活时放宽 cost 要求。

**Path（选路）**：
- 简单启发式（A0 Watcher floor 16 够用）：
  1. HP ≤ 40%：优先 `?`（可能是治疗事件） > `R`（火堆） > `M` > `E`。
  2. HP 40-70%：`M` > `?` > `R` > `E`（累计遗物和金币）。
  3. HP ≥ 70%：`E`（精英给 relic） > `M` > `?` > `R`。
  4. 离 boss 1 层：强制 `R`（休息）。
- 简化版：只看下一层，不做多层 lookahead。

**Choice（事件 / 休息 / 商店）**：
- **休息**：HP < 60% → Rest；HP ≥ 60% 且有可升级卡 → Smith；无升级但有 Toke/Dig 等特殊火堆 → 按优先级。
- **商店**：
  - 预算 ≤ 75 gold：只看卡。
  - 75-150：卡 + 删牌。
  - ≥150：卡 + 删牌 + 遗物。
  - 优先级表硬编码（参考 bottled_ai）。
- **事件**：一张大表（event_name → best_choice），直接抄 bottled_ai `handlers/events/`。未知事件默认选"安全"选项（无扣血 / 无丢卡）。

**Boss Relic**：硬编码三选一优先级表（Watcher 偏好 `Runic Pyramid > Pandora's Box > Calling Bell > ...`）。

### 4.3 整体流程

```
Neow 祝福 → 硬编码选（默认 "Enemies have 1 HP in next combat"）
     ↓
地图选路 (path handler)
     ↓
进入房间：
  ├── Combat    → TurnSolver（新 lex comparator）每回合出牌 → EndTurn
  ├── Elite     → TurnSolver（更大预算）
  ├── Boss      → TurnSolver（最大预算 + 特殊 big_fight comparator）
  ├── Event     → choice handler
  ├── Rest      → choice handler（Rest/Smith/...）
  ├── Shop      → choice handler（买卡/删牌/离开）
  └── Unknown (?) → engine 已 roll，按上述分发
     ↓
战后：
  ├── Card reward → draft handler
  ├── Potion? → 硬编码（Watcher 留给 boss）
  └── Relic?  → 硬编码优先级
     ↓
下一层 (floor++)
     ↓
Boss floor (16/33/50): TurnSolver 大预算 → Boss Relic choice handler
     ↓
Heart (floor 57)
```

---

## 5. 第一里程碑（A0 Watcher floor 16）

### 5.1 最小实现（MVP）

**战斗层**（~1 周）：

1. 把 `TurnSolver._score_terminal` 的 weighted-sum 替换为 lex tuple（20 维 MVP）：
   - 生死 / 胜负
   - 模拟敌方回合后 HP（`least_incoming_damage`）
   - 已死敌人数
   - 敌方 Vulnerable / Weak
   - `stance_is_not_wrath` + `stance_is_calm`
   - Mantra 累积
   - Divinity 预留
   - Tranquility / Talk to the Hand stack
   - Intangible 层数
   - good powers 数、bad powers 数
   - Barricade block / plated armor（敌方）
   - 未消耗能量 + 可打牌
2. 加一个 `test_solver_game_watcher.py`（参考 `test_solver_game.py`），跑 10 seed × A0 Watcher。

**策略层**（~3-5 天）：

3. Draft：抄 peaceful_pummeling 的 DESIRED_CARDS_FOR_DECK + CARD_REMOVAL_PRIORITY。
4. Path：HP 阈值的 4 条规则（§4.2）。
5. Choice：Rest / Smith / 商店简单规则 + 已知事件小表（至少覆盖 Act 1 的 10-15 个事件）。
6. Boss relic：3 选 1 优先级表（Watcher 偏好）。

**集成**（~2 天）：

7. 改 `test_solver_game.py` 支持 Watcher 默认 + 接入 §3-6 的策略层。
8. 跑 50 seed 基线。

### 5.2 预期信号

- **成功标准**：A0 Watcher，50 seed 跑下来，floor 16 通过率 ≥ 30%。
- **理想**：≥ 50%（bottled_ai A20 是 52%，A0 应该显著高于此）。
- **时间预算**：7-10 天看到第一个数据点；不达标就 debug 评估函数。

---

## 6. 和 bottled_ai 的差异

### 6.1 我们做的

- 用 **StSRLSolver 引擎** 而不是 bottled_ai 的 `rs/calculator/`（它的 simulator，代码更重）。
  - 优点：StSRLSolver 已经是我们团队熟悉的代码，Watcher 已实现。
  - 风险：StSRLSolver 的 enemy intent / power 逻辑可能和 bottled_ai 不完全对齐，需要 parity 测试。
- 搜索算法用三档自适应（V5 现有），比 bottled_ai 的纯 DFS 更细。

### 6.2 我们省略的

- **Ascension 特化**：只做 A0，不处理 A20 的额外 debuff / enemy buff。
- **多 character 支持**：只做 Watcher（Ironclad / Silent / Defect 后续再说）。
- **精细事件库**：Act 1 事件小表 15 个左右，Act 2/3 后续补。
- **Scry / 特殊抽牌机制的微操**：MVP 交给 solver 搜索，不做专门规则。

### 6.3 复用 bottled_ai 的部分

- **卡组构建列表**（DESIRED_CARDS_FOR_DECK 等常量）：直接抄。
- **删牌 / 升级优先级**：直接抄。
- **Power 分类**（good/less_good/bad）：直接抄。
- **Comparator 列表 + 顺序**：抄 20 维 MVP，逐步加到 42 维。

---

## 7. 代码复用

| 组件 | 来源 | 去处 |
|---|---|---|
| 游戏引擎（combat / state / content） | StSRLSolver/packages/engine | 直接复用 |
| TurnSolver 搜索骨架 | StSRLSolver/packages/training/turn_solver.py | 复用，改评估函数 |
| encoding / data tables | V6 sts_agent_v6.py | 策略层查表时可借用（可选）|
| Reward 系统 / PPO trainer | V4-V6 | **丢弃** |
| Transformer backbone | V4-V6 | **丢弃** |
| Neural leaf eval 钩子 | TurnSolver（已预留 `neural_eval` 参数）| **保留占位**，远期给 AlphaZero-lite 用 |
| 新写 | — | `v7_evaluator.py`（lex comparator）, `v7_strategy.py`（handler）|

**预计代码量**：

- `v7_evaluator.py`：~500 行（20 维 comparator + assessment helper）。
- `v7_strategy.py`：~600 行（四 handler + 事件表）。
- TurnSolver 改造：~100 行 diff。
- 测试 / 脚本：~200 行。
- 合计新增：~1400 行。

---

## 8. 里程碑后的方向（长期）

### 8.1 第二里程碑：A0 Watcher 全通关（floor 57，Heart）

- 扩展事件库到 Act 2/3（约 +30 个事件）。
- Boss 评估加 big_fight comparator（参考 bottled_ai 的 `pmo_big_fight_comparator.py`）。
- 第三层特殊机制（Time Eater / Awakened One / Donu&Deca）的针对性调整。

### 8.2 第三里程碑：A20 Watcher

- Neow curse / ascension debuff 处理。
- Elite 血量 + 1/2/3 阶段 buff 对应。
- 预期工期 2-4 周，参考 bottled_ai 52% 作为天花板。

### 8.3 远期：AlphaZero-lite（value network 替换手调 heuristic）

**社区没人做过**。路径：

1. 手调 heuristic 跑出 10k+ 局高质量数据（胜率 ≥ 40% 的那些）。
2. 训一个小 value network（输入 BattleState，输出 [win prob, expected hp loss]）。
3. 接入 TurnSolver 的 `neural_eval` 钩子，替换 lex comparator 的打分部分。
4. 搜索仍是骨架（比纯 MCTS 成本低），但 leaf 评估用学到的 value——类似 AlphaZero 但不做 self-play policy improvement（手调策略层不变）。

**为什么值得做**：STS 是 perfect-info turn-based 带强随机性的游戏，搜索 + value network 是原生适配；纯 RL 样本效率差，纯手调评估达不到理论上限。这条中间路线社区未验证。

---

## 9. 待查项（不确定点）

- [ ] StSRLSolver 的 enemy intent 系统和 bottled_ai 是否对齐（可能需要 parity 测试）。
- [ ] Watcher 特殊 power（Mantra / Divinity）在 StSRLSolver 的 `CA` 等价物里怎么查询（需要读 `BattleState` / `PlayerState` API）。
- [ ] Scry 动作在 TurnSolver 当前已支持，但是否和 Watcher 的 Foresight / Third Eye 正确交互（需要实测）。
- [ ] `most_optimal_winning_battle`（bottled_ai #4）的 11 层 lex 链里的 `ritual_dagger` / `pen_nib_counter` 等是否需要全部实现（MVP 建议只保留 max HP + incoming damage）。
- [ ] A0 vs A20 的胜率 scaling（bottled_ai 52% 是 A20，A0 应该更高，但高多少未知）。

---

## 10. 不做什么

- **不做 RL**（V7 放弃 RL 路线）。
- **不做 self-play**（纯搜索 + 手调）。
- **不上 communication mod 实战**（MVP 阶段只跑 StSRLSolver 模拟）。
- **不做 Ironclad / Silent / Defect**（专注 Watcher）。
- **不做 multi-turn planning**（TurnSolver 只看一回合内 + 敌方回合反馈，不跨回合）。

# V8 卡牌/遗物行为测试集方案

- 日期：2026-06-04
- 状态：Phase 1（地基 + 小批量验证）已完成；Phase 2+ 待启动
- 关联：[设计 spec](superpowers/specs/2026-06-04-game-backend-test-harness-design.md)、[地基实施计划](superpowers/plans/2026-06-04-combat-probe-card-test-harness.md)

## 目的

独立验证任意游戏引擎（StSRLSolver / 未来 Rust 引擎 / 实机）的卡牌与遗物行为是否符合真实 STS。核心铁律：**标准答案只来自 wiki / 实机，绝不取被测引擎自己的数字**。新引擎过不了测试集就不进训练——这是防"再被烂模拟器坑 120 小时"的保险。

## 已验证的地基（Phase 1 ✅）

- `v8/backends/combat_probe.py`：`CombatProbe` 接口（构造受控战斗 + 打单卡 + 读中性结果 `CardPlayResult`）。正交于 `GameBackend`，只有模拟器后端实现。
- `v8/backends/stsrl_combat_probe.py`：`StSRLCombatProbe`，封装 StSRLSolver 战斗引擎。
- `tools/card_oracle.py`：独立 oracle 期望表（`CardExpectation`，每条带 `source` 出处）。
- `tools/test_combat_card_behavior.py`：后端无关对照跑器（换引擎只换 probe 工厂）。
- `tools/test_backend_conformance.py`：第 2 层接口合格性 + 初始状态合理性。
- `tools/test_combat_probe.py` / `tools/test_card_oracle.py`：probe / oracle 自身测试。

探针现有能力：单卡、单敌、预设玩家格挡 / 力量、读敌我状态层数（Vulnerable / Weakened / Strength）。

跑法（无 pytest，裸脚本）：`.venv/bin/python tools/test_combat_card_behavior.py` 等。

### 已覆盖 18 个场景，全绿
起始 3（Strike/Defend/Bash）+ 简单 8（Anger / Clothesline / Twin Strike / Thunderclap / Iron Wave / Pommel Strike / Shrug It Off / Inflame）+ Body Slam / Heavy Blade 区分性 7（含力量注入对照）。**含区分性场景**：Body Slam 力量加性非乘性 + 格挡不消耗、Heavy Blade 力量 ×3 倍率含负力量。证明 harness 有牙、且 StSRLSolver 对这些卡实现正确。

## 诚实现状

17 张卡测下来引擎**都对**（含最易写错的 Body Slam / Heavy Blade）。"40+ 张坏卡"在已测范围内**尚未现身**，可能在：
- 尚未覆盖的 ~90 张卡
- 战斗相关遗物（还没建 relic-probe）
- 多敌 AoE / 多次打出（Rampage）/ 牌库相关（Perfected Strike）等还没测的机制
- 状态交互边界
- 或早先已被 fork 修复的遗物 / 开战 effect bug（NeowsLament 计数、pre-battle buff）

要定位，必须全量覆盖。

## 覆盖目标

- **卡**：全部 Ironclad（约 77）+ 无色（约 30）≈ 107 张，每张 ≥ 1 个区分性场景
- **遗物**：影响战斗的遗物（需 relic-probe）
- **支撑（可选）**：状态（power）交互、敌人 intent / hp

## 探针能力：现有 vs 待扩（由卡倒逼）

| 能力 | 状态 | 触发的代表卡 |
|---|---|---|
| 单卡 / 单敌 / 预设 block+strength / 读状态 | ✅ 已有 | Strike / Bash / Body Slam / Heavy Blade |
| multi_enemy（多敌，验 AoE） | 待扩 | Thunderclap / Cleave / Whirlwind / Immolate |
| multi_play（同战斗连打验递增 / 战斗间重置 / 实例独立） | 待扩 | Rampage |
| deck_composition（控制战斗牌组构成） | 待扩 | Perfected Strike / Battle Trance / Headbutt |
| pile inspection（读抽 / 弃 / 消耗堆） | 待扩 | Anger（复制进弃牌堆）/ 抽牌类 / Exhaust 类 |
| x_cost（X 费按能量） | 待扩 | Whirlwind / Transmutation |
| status-key 归一化（oracle 用规范名而非引擎键名） | 待扩 | 跨引擎无关性 |
| relic-probe（设遗物 + 触发条件 + 观察，独立能力） | 待建 | Vajra(+1 Str) / Bag of Marbles(开战给敌 Vulnerable) / Burning Blood(战后回血) |

## Oracle 数据模型

`CardExpectation(card_id, label, setup, target_index, expect_enemy_hp_delta, expect_player_block_delta, expect_energy_delta, expect_enemy_status, expect_player_status, source)`。一张卡可多场景（用 `label` 区分）。每条必带 `source`（wiki 出处）。`card_id` 是引擎侧 id（驱动 probe 用）；期望值是引擎无关的真实机制值。

## 并行产出设计（fan-out）

1. **每张卡（或小批）一个调研 sub-agent** → 结构化输出：cost / 伤害 / 格挡 / 状态 / 抽牌 / 公式 + 易错点 + 需要的探针能力 + 区分性场景（带可断言数字）。来源 wiki / 社区，不碰模拟器。
2. **接入 + 跑 pass**：读 `IRONCLAD_CARDS` 映射真实 engine id → 写入 ORACLE → 跑 → 出判决。
3. **大规模 fan-out 用 Workflow**：pipeline 每张卡（调研 →（可选）二次核对）。~107 卡 + 遗物的联网调研是大 token 开销，分批跑。

## 鉴别"真坏卡"的方法论

- **区分性场景**：每张卡至少一个能区分对 / 错实现的场景（如 Heavy Blade 必须带力量才暴露 ×3 vs +1；简单卡全绿不足以证明引擎对复杂机制也对）。
- **注入对照**：前置状态（力量 / 格挡 / 牌库）注入后，先用一个已知正确的对照（如"打击 +3 力量 = 9"）确认注入真生效，避免假绿 / 假红。
- **状态断言先打印实际 statuses**：排除引擎键名差异（命名差异 ≠ bug；层数 / 数值不符才是 bug）。
- **高风险 / 出红卡 → 实机抓轨迹仲裁**：实机是最终标准答案。
- **纪律**：红灯 = 照出坏卡，如实记录进台账，**绝不准把期望值改成引擎数字让它变绿**。

## 分阶段路线

- **Phase 1 ✅**：地基 + 17 卡验证（done）
- **Phase 2**：扩探针（multi_enemy / multi_play / deck_composition / pile inspection / x_cost / status 归一化）+ 各自代表卡验证
- **Phase 3**：全量卡 oracle（mass fan-out ~90 张）
- **Phase 4**：relic-probe + 战斗相关遗物覆盖
- **Phase 5**：实机轨迹交叉验证 + Rust 后端接入（用本测试集验收）
- **贯穿**：中间层透传债随用到处增量抹平

## 退出标准

- 每张 Ironclad + 无色卡有 ≥ 1 区分性场景，绿或有记录的红
- 战斗相关遗物覆盖
- 一条命令可跑全套对照（裸脚本 `.venv/bin/python tools/test_*.py`）
- 产出一份"红灯清单" = 引擎待修坏卡台账

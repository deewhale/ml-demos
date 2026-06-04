# 设计：游戏后端测试 harness（GameBackend test harness）

- 日期：2026-06-04
- 状态：设计已确认，待写实施计划
- 分支：dev

## 背景与动机

V8 RL 训练长期被底层模拟器（StSRLSolver legacy Python 引擎）拖累：之前有 40+ 张卡的行为实现是错的，导致训练数据被污染、模型学到的是错误游戏规则而非真实 STS。历史上曾因 simulator bug 浪费 120+ 小时训练。

用户已在两个 commit 里搭好"游戏后端中间层"骨架：
- `9973049` refactor(v8): 引入 GameBackend 中间层 + StSRLBackend（stage1，行为不变）
- `cee4ab4` feat(v8 中间层 stage2): RealGameBackend + 统一评估 harness

中间层让训练逻辑（env.py）与底层引擎解耦，理论上可换别的模拟器（如上游已迁移的 Rust 引擎）或接实机游戏。但当前状态是"骨架在、未实测、抽象未收干净"。

本设计解决：**如何在换底层引擎之前，用一套独立测试集验证新引擎的行为正确性，避免再被烂模拟器坑。**

## 核心原则（最重要）

**测试集的标准答案（oracle）必须独立于被测模拟器本身。**

StSRLSolver 自带 6000+ 单元测试和 `JAVA_*_VALUES` 对标表，却没挡住那 40+ 张坏卡——因为那些测试是自指的（测试值照着同一份实现/反编译表写），且大多只测卡牌静态数值（卡面 cost/伤害对不对），没测出牌后的实际战斗效果。坏卡恰恰坏在执行/特效层（如计数器不回写、pre-battle buff 不触发、特效算错）。

因此：
- 期望值来源 = 联网查 STS wiki / 社区 + 本地开实机抓轨迹（最终仲裁）
- **绝不**取被测模拟器自己的数字
- 断言深度 = 机制级（伤害/格挡/buff/debuff/抽牌/能量/特殊机制），不只卡面数值

## 目标 / 非目标

### 目标
- 一套后端无关的卡牌行为测试集，任何模拟器后端接入前先过这套测试
- 全量覆盖训练角色（Ironclad）的卡池
- 把中间层抽象收口到"换 Rust/接实机无痛"的程度（增量进行）

### 非目标（本轮不做）
- 实现 Rust 后端本身（测试集就位后另起）
- 完整实机集成训练（实机抓轨迹仅作 oracle/交叉验证用）
- 复杂 sub-screen（HAND_SELECT / GRID 等）的完整实机支持

## 卡池范围

训练角色 = **Ironclad（铁甲战士）**。"全量"覆盖：
- Ironclad 卡（约 77 张），来源 `external/StSRLSolver/packages/engine/content/cards.py` 的 `IRONCLAD_CARDS`
- 无色卡（约 30 张），来源同文件 `COLORLESS_CARDS`
- 起始三卡：Strike_R / Defend_R / Bash

（注：之前探查误引 Watcher 对标表 Eruption/Vigilance，那是 Watcher 角色卡，与本项目无关。StSRLSolver 四角色卡池都有，按 character="ironclad" 选 IRONCLAD_CARDS。）

## 架构

### 一、把中间层做完

#### 1. 新增"战斗探针"能力接口 CombatProbe
- 单独的可选 capability 接口，**不塞进核心 GameBackend**——实机证明做不了受控单卡战斗（mod 协议不支持指定单卡、game-state JSON 缺 buff 字段），只有模拟器后端实现它
- 接口方法（中性签名，不漏引擎对象）：
  - `build_combat(player_hp, player_energy, hand, enemies, relics=...)` → 构造受控战斗
  - `play_card(hand_index, enemy_index)` → 打出指定单卡
  - 返回中性结果结构：敌人掉血 / 我方格挡 / 施加的 buff/debuff / 能量消耗 / 抽牌数等
- 底层 StSRLSolver 已有 `create_combat_from_enemies` + `play_card`（`external/StSRLSolver/packages/engine/combat_engine.py`），本层是中性壳封装，非从头造

#### 2. 增量抹平 11 处透传债
- 当前契约直接漏 Python 引擎对象（`run_state` / `current_combat` / `event_handler` / `current_event_state` / `current_rewards` / `current_shop` / `neow_blessings` / `last_combat_card_log` / `current_room_type` / `phase` / `get_available_actions` 返回引擎对象等）
- 目标：逐个改成中性 dataclass，使 env.py 与任何后端（含未来 Rust 后端）不依赖 Python 引擎对象
- **范围：增量抹平，随测试推进**——不一次性大改 env.py（现在在运作中，一次性改有回归风险）。每抹平一处，验证行为不变（用现有 eval_harness parity 测试兜底）

### 二、第 1 层：卡牌行为对照测试（核心）

- **期望效果表**（独立 oracle 数据文件）：每张卡一条，列出打出后的机制级期望结果。来源 = wiki/社区（联网查）+ 实机轨迹（仲裁）。分批查、分批人工核对。
- **后端无关测试跑器**：读期望表 → 用 CombatProbe 在受控场景打每张卡 → 断言结果对得上期望。换引擎只需换后端工厂，跑同一套表。
- 现在拿 StSRLSolver 跑 → 坏卡变红灯，证明测试有牙。

### 三、第 2 层：接口合格 + 状态合理 + 不卡死
- 便宜粗筛网：所有后端方法都实现、返回类型对、状态合法（血量/楼层/卡组在合理范围）、整局 reset→决策→结束不 stall、动作合法
- 后端无关，任何后端都跑

### 四、第 3 层（之后）：实机轨迹交叉验证
- 抓几条实机轨迹当最终仲裁，集成级抽查
- 实机缺 buff 字段、打不了受控单卡，故不当卡牌级主力，定位为"最终仲裁 + 边界补盲"

## 成本诚实声明

真正的重活是**手工建全量期望表**（~107 张卡，每张要从 wiki 核出机制级期望值）。这是独立 oracle 必付的代价——没有它，测试就退回"自己判自己及格"，照样漏坏卡。将用 sub-agent 联网分批查、分批核对。

## 风险

- 期望表人工核对成本高、易有疏漏 → 分批核、用实机轨迹交叉校验高风险卡
- 增量抹平透传债可能触碰运作中的 env.py → 每步用 eval_harness parity 测试守行为不变
- wiki/社区数值可能有版本差异（STS 不同 patch）→ 以实机抓的轨迹为最终仲裁
- StSRLSolver 是 legacy Python（上游已弃用迁 Rust），修它的坏卡 ROI 有限 → 测试集的价值在于"挡新引擎"，不一定要把 legacy 引擎的坏卡全修好

## 落地顺序（高层，细节见实施计划）

1. CombatProbe 接口定义 + StSRLBackend 实现（中性壳封装现有 combat_engine）
2. 第 2 层粗筛网（便宜、先立兜底）
3. 期望表 schema + 第 1 层测试跑器骨架（先跑通几张基础卡：Strike/Defend/Bash）
4. 全量期望表分批建（联网查 + 实机仲裁）+ 全量卡测试
5. 透传债随上述步骤增量抹平
6. （之后另起）第 3 层实机交叉验证；Rust 后端接入

## 后续（本设计之外）
- Rust 后端接入（用本测试集验证）
- 实机集成评估
- 复杂 sub-screen 实机支持

# V8 战斗动作空间地图（sts_lightspeed）

> Stage 1 产物（2026-06-09）。后续 Stage（env 战斗状态机 + 模型动作空间设计）的依据。
> 来源：`external/sts_lightspeed` 引擎源码逐行核实（非二手）。

## 一句话结论

在 sts_lightspeed 里，**战斗中真正会停下来等玩家做决策的交互状态只有两个**：
`PLAYER_NORMAL` 和 `CARD_SELECT`。其余所有"弃牌 / 消耗 / scry 式留弃 / 选目标"都被
归并进这两个状态（CARD_SELECT 再按 `cardSelectTask` 细分）。`InputState.h` 里列的
~28 个枚举值里，**只有 PLAYER_NORMAL / CARD_SELECT / EXECUTING_ACTIONS 真正被引擎
assign 成活动状态**，其余是死枚举（见下"证据"）。

## InputState 清单 + 各自决策形态

`include/combat/InputState.h` 列出的全部枚举值，逐个说明其在本引擎里的真实地位：

| InputState | 引擎里是否真用 | 玩家在此要做什么 | 合法动作形态 |
|---|---|---|---|
| `EXECUTING_ACTIONS` | ✅ 瞬态（不停） | 引擎正在结算队列里的卡/效果，**不轮到玩家** | 无（`get_battle_actions` 返回空） |
| `PLAYER_NORMAL` | ✅ **决策点** | 本回合主决策：出牌（选目标）/ 喝药 / 弃药 / 结束回合 | `CARD`(source_idx, target_idx) / `POTION`(slot, target) / `END_TURN` |
| `CARD_SELECT` | ✅ **决策点** | 某张卡/药水触发的"从某牌堆选 1 张或多张"子决策 | `SINGLE_CARD_SELECT`(select_idx) 或 `MULTI_CARD_SELECT`(idx 位掩码)，按 `cardSelectTask` 分 |
| `SCRY` | ⚠️ 仅 2 处比较、**从不被 assign** | （Watcher Scry，本引擎对 Ironclad 不触发） | — |
| `CHOOSE_STANCE_ACTION` | ⚠️ 仅 1 处比较、从不 assign | （姿态药水，Watcher 向） | — |
| `CHOOSE_DISCARD_CARDS` | ⚠️ 仅 1 处比较、从不 assign | 死枚举（弃牌实际走 CARD_SELECT） | — |
| `CHOOSE_TOOLBOX_COLORLESS_CARD` | ❌ 死枚举 | — | — |
| `CHOOSE_EXHAUST_POTION_CARDS` | ❌ 死枚举 | — | — |
| `CHOOSE_GAMBLING_CARDS` | ❌ 死枚举（Gamble 走 CARD_SELECT/GAMBLE task） | — | — |
| `CHOOSE_ENTROPIC_BREW_DISCARD_POTIONS` | ❌ 死枚举 | — | — |
| `SELECT_ENEMY_ACTIONS` / `FILL_RANDOM_POTIONS` / `SHUFFLE_INTO_DRAW_*` / `INITIAL_SHUFFLE` / `SHUFFLE_*_TO_DRAW` | ❌ 死枚举（随机事件，引擎内部消化） | — | — |
| `CREATE_RANDOM_CARD_IN_HAND_*` | ⚠️ 2 处比较、从不 assign | — | — |
| `SELECT_CARD_IN_HAND_EXHAUST` / `EXHAUST_RANDOM_CARD_IN_HAND` | ❌ 死枚举 | — | — |
| `GENERATE_NILRY_CARDS` / `SELECT_STRANGE_SPOON_PROC` / `SELECT_ENEMY_THE_SPECIMEN_APPLY_POISON` / `SELECT_WARPED_TONGS_CARD` / `CREATE_ENCHIRIDION_POWER` / `SELECT_CONFUSED_CARD_COST` | ❌/⚠️ 死枚举或单处比较 | — | — |

**证据**（`grep "inputState = InputState::" src/`）：整个引擎源码里被**赋值**为活动状态的
只有 `CARD_SELECT`（12 处）、`EXECUTING_ACTIONS`（3 处）；`PLAYER_NORMAL` 是默认/回合起始态。
其余枚举值最多只在少数 `==` 比较里出现，从不被设成活动状态 → 对 Ironclad 整局训练**无需
为它们单独建动作空间**。

## CARD_SELECT 的子任务（cardSelectTask）

`CARD_SELECT` 进一步按 `bc.cardSelectInfo.cardSelectTask` 分流（`include/combat/CardSelectInfo.h`）。
枚举入口：`Action::enumerateCardSelectActions` / `BattleScumSearcher2::enumerateCardSelectActions`
（`src/sim/search/Action.cpp` / `BattleScumSearcher2.cpp`）。各 task 从哪个牌堆选、几选：

| cardSelectTask | 从哪选 | 动作类型 | 触发卡/物示例 |
|---|---|---|---|
| ARMAMENTS | 手牌中可升级的 | SINGLE | Armaments |
| EXHAUST_ONE | 手牌任意 | SINGLE | （消耗 1 张手牌的效果） |
| FORETHOUGHT / WARCRY | 手牌任意 | SINGLE | Forethought / Warcry |
| DUAL_WIELD | 手牌中 ATTACK/POWER | SINGLE | Dual Wield |
| HEADBUTT / HOLOGRAM / LIQUID_MEMORIES_POTION | 弃牌堆 | SINGLE | Headbutt / Hologram / 药水 |
| EXHUME | 消耗堆（排除 Exhume 自身） | SINGLE | Exhume |
| SECRET_TECHNIQUE | 抽牌堆中 SKILL | SINGLE | Secret Technique |
| SECRET_WEAPON | 抽牌堆中 ATTACK | SINGLE | Secret Weapon |
| CODEX | 4 选 1（含 skip=idx3） | SINGLE | （Codex 类） |
| DISCOVERY | 3 选 1 | SINGLE | Discovery 类 |
| EXHAUST_MANY | 手牌多选（≤ pickCount） | MULTI | （多张消耗） |
| GAMBLE | 手牌多选（任意张） | MULTI | Gamble |
| MEDITATE / NIGHTMARE / RECYCLE / SETUP / SEEK | — | （执行未实现/无单独动作） | Watcher/特殊向，Ironclad 基本不遇 |

> MULTI（EXHAUST_MANY / GAMBLE）当前引擎枚举为单个占位 `MULTI_CARD_SELECT` 动作，
> 执行用 `bits` 还原；Python 侧手搓多选 dict 暂不支持（Stage1 标 TODO，需展开
> `selected_idxs` 位掩码再编进 `Action` 的低 10 位）。

## 引擎枚举入口（动作怎么表示）

- **总入口**：`BattleScumSearcher2::enumerateActionsForNode(node, bc)`
  （`src/sim/search/BattleScumSearcher2.cpp:175`）按 `bc.inputState` 分流：
  - `PLAYER_NORMAL` → `enumerateCardActions` + `enumeratePotionActions` + push `END_TURN`
  - `CARD_SELECT` → `enumerateCardSelectActions`
  - 其它 → assert（不该到这）。Stage1 绑定 `get_battle_actions` 直接复用它。
- **动作结构** `search::Action`（`include/sim/search/Action.h`）：一个 `uint32_t bits` 打包
  `actionType(3bit) | target/idx2(13bit) | source/idx1(16bit)`。
  - `ActionType`：`CARD=0 / POTION / SINGLE_CARD_SELECT / MULTI_CARD_SELECT / END_TURN`
  - `getSourceIdx()` = 手牌/药水槽 idx；`getTargetIdx()` = 敌人 idx；
    `getSelectIdx()` = 选牌 idx；`getSelectedIdxs()` = 多选的位掩码展开。
  - 合法性：`Action::isValidAction(bc)`（`src/sim/search/Action.cpp:226`）按 type +
    InputState + cardSelectTask 逐类校验。**Stage1 绑定不另写校验，全复用它。**
  - 执行：`Action::execute(bc)`（同文件 `:421`）执行后置 `inputState=EXECUTING_ACTIONS`
    再 `executeActions()` 消化到下一个停顿点（PLAYER_NORMAL / CARD_SELECT / 战斗结束）。

## Stage1 暴露给 Python 的绑定

（`bindings/slaythespire.cpp`，patch `0003`，详见 `external/sts_lightspeed_patches/README.md`）

- `get_input_state(bc) -> str`
- `get_battle_actions(bc) -> list[dict]`（dict 含 `type/bits/source_idx/target_idx/select_idx/card_select_task/label`）
- `execute_battle_action(bc, action_dict) -> bool`（优先用 `bits` 无损还原）
- 已有的 `make_encounter` / `make_test_combat` / `get_combat_snapshot` / `play_card` /
  `end_turn` 配合，构成"逐步驱动战斗"的完整 Python 接口。

验证：`/tmp/stage1_verify_battle_actions.py` 11/11 通过（进战斗→枚举→出牌→状态变化→
打完回合→end_turn；含 Armaments 触发 CARD_SELECT 子状态的多阶段验证）。

## 对后续 Stage 的设计提示

- **模型动作空间**：可设计成 (action_type, source_idx, target_idx/select_idx) 的离散头，
  或直接以 `get_battle_actions` 返回的合法动作列表做 pointer-net 打分（推荐后者，天然
  mask 非法动作、与现有 V8 pointer-net 一致）。
- **状态机**：env 每步先读 `get_input_state`；只在 PLAYER_NORMAL / CARD_SELECT 让模型决策，
  EXECUTING_ACTIONS / 空动作列表时不该轮到模型。CARD_SELECT 是同一回合内的子决策，
  不消耗"回合"。
- **MULTI（Gamble/EXHAUST_MANY）**：Stage1 用 `bits` 能执行引擎枚举出的占位多选；若要让
  模型自由选多张，需在后续 Stage 补多选 dict→Action 的位掩码重建（已标 TODO）。

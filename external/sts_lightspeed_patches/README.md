# sts_lightspeed 本地补丁

`external/sts_lightspeed` 的 clone 是 **gitignored** 的（不进本仓库 git），所以对引擎
源码（`.cpp` / `.h`）的修复**不会被本仓库追踪**。一旦重新 clone sts_lightspeed，这些
修复就会丢失。

本目录保存我们对 sts_lightspeed 的修复补丁（repo 相对路径格式，不含任何本机绝对路径），
**重新 clone 后必须重新 apply + 重编**，否则训练 / 验收会用回带 bug 的引擎。

## 如何应用

> **重要**：`0003` 是从工作树整体导出的（含全套 pybind11 绑定 **+ 0001 / 0002 两个 bug
> 修也已包含在内**）。所以 **fresh clone 只 apply `0003` 一个就够**，不要再 apply
> 0001 / 0002（会与 0003 里的同段改动冲突）。0001 / 0002 保留作单点 bug 的可读说明 + 备份。

```bash
cd external/sts_lightspeed
# 一步到位：0003 已含绑定 + 两个 bug 修
git apply ../sts_lightspeed_patches/0003-pybind-bindings-combat-fullgame-relic-monster.patch \
  || git apply --3way ../sts_lightspeed_patches/0003-pybind-bindings-combat-fullgame-relic-monster.patch
# 绑定依赖 pybind11 子模块
git submodule update --init --recursive
# 重编（用本仓库 venv 的 cmake；<repo> = ml-demos 仓库根）
<repo>/.venv/bin/cmake -S . -B build && <repo>/.venv/bin/cmake --build build --target slaythespire -j 10
# 验收（应 84/84 全过）
<repo>/.venv/bin/python tools/test_card_behavior.py
```

> 注：补丁是 `git diff` 格式（`a/...` `b/...` 相对仓库根），对 fresh clone 用
> `git apply` 即可；若 upstream 行号漂移导致 apply 失败，用 `git apply --3way` 或
> 手动按补丁内容定位修复点。

## 补丁清单

补丁来源：均为我们自己的卡牌行为测试集
（`tools/test_lightspeed_card_behavior.py` + `tools/card_oracle.py`）逮到的真 bug。

### 0001-dark-shackles-sign-fix.patch
- **文件**：`src/combat/BattleContext.cpp`（`CardId::DARK_SHACKLES` 分支）
- **Bug**：Dark Shackles 的力量符号**反了**——给敌人施加了 **+9 力量**（buff），
  而该卡应是「目标本回合**失去** 9 力量」（debuff，回合开始时还回去）。另外原实现
  只在敌人有 ARTIFACT 时才挂 `SHACKLED`（还原力量用的 power），逻辑也错。
- **修复**：改为 `DebuffEnemy<MS::STRENGTH>(t, -9)`（升级 -15）+ 恒挂
  `BuffEnemy<MS::SHACKLED>(t, +9)`（升级 +15），即「本回合 -X 力量、敌人回合开始还原」
  的正确 STS 语义。参考同文件 `CardId::DISARM`（传 -2，但 Disarm 是永久减力量，
  没有 SHACKLED 还原 —— 这是两张卡的区别）。
- **验收**：oracle 断言 `Strength: -9`（打牌瞬间可观测效果）。修前 +9 = 红灯，修后绿。

### 0002-trip-cost-fix.patch
- **文件**：`include/constants/Cards.h`（`getEnergyCost`）
- **Bug**：lightspeed 把 `CardId::TRIP` 的费用记成 **1**，应为 **0**。
- **同名异卡核实**：lightspeed 的 `CardId::TRIP` 只出现在无色卡池
  （`srcColorlessCardPool` / `baseColorlessPool` / `colorlessCardBlob`），效果是
  「施加 2 易伤」（升级版打全体），与无色 Trip 一致，**不是** Silent 的 Trip。
  我们 oracle 引用的也是无色 Trip（wiki 链接一致）。所以这是 lightspeed 的**真 bug**
  （无色 Trip 应 0 费），不是同名异卡 —— 改引擎，不改 oracle。
- **修复**：把 `CardId::TRIP` 从 `getEnergyCost` 的 `return 1` 组挪到 `return 0` 组。
- **验收**：oracle 断言 `expect_energy_delta=0` + `Vulnerable: 2`。修前费 1 → energy_delta=-1 = 红灯，修后绿。

### 0003-pybind-bindings-combat-fullgame-relic-monster.patch（**主补丁，fresh clone 只 apply 这个**）
- **文件**：`bindings/slaythespire.cpp`（+534 行绑定）、`include/constants/Cards.h`、
  `src/combat/BattleContext.cpp`（后两者即 0001 / 0002 的 bug 修，已合并进来）。
- **内容**：从工作树整体导出的全套 **pybind11 绑定**——这是把 V8 训练接到 lightspeed 的命脉，
  也是 gitignored clone 里**唯一没进本仓库 git** 的关键源码。绑定覆盖：
  - 战斗探针：`make_test_combat` / `get_state` / `step_choice` 等（供测试台 CombatProbe）；
  - 整局驱动：`play_battle` / `make_encounter`（供 LightspeedBackend 整局导航）；
  - 遗物 / 怪物字段：遗物状态、敌人 HP / 意图字段等（供 relic / monster probe + 训练状态编码）。
- **为什么是主补丁**：`0003` = 绑定 + 0001 + 0002 三者合一。重 clone 后 **只 apply 0003**
  即可拿到全部修复 + 绑定，**别再 apply 0001 / 0002**（会与 0003 内同段改动冲突）。
- **重建依赖**：apply 后必须 `git submodule update --init --recursive`（pybind11 子模块），
  否则编译找不到 pybind11 头。重编 + 验收命令见上方「如何应用」。

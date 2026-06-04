# CombatProbe + 卡牌行为测试 Harness 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 搭起一套后端无关的卡牌行为对照测试地基——加 `CombatProbe` 战斗探针接口 + StSRLSolver 实现 + 读独立 oracle 期望表的测试跑器，先在起始三卡（Strike/Defend/Bash）上跑通绿灯。

**Architecture:** `CombatProbe` 是正交于 `GameBackend` 的可选能力接口（只有模拟器后端实现，实机做不了受控单卡战斗），方法 `play_single_card(setup, hand_index, target_index)` 返回中性 `CardPlayResult`（不漏引擎对象）。测试跑器读 `card_oracle.ORACLE`（每张卡的机制级期望，标准答案来自 wiki/实机、绝不取被测引擎自己的数），用一个 probe 打每张卡断言结果。换引擎只换 probe 工厂、跑同一套表。

**Tech Stack:** Python 3.12（`.venv/bin/python`）、StSRLSolver legacy 引擎（`packages.engine.combat_engine`，经 `sts_paths.ensure_on_sys_path()` 上 path）、项目无 pytest——测试用裸 `def test_*()` + 文件末 `__main__` driver，跑法 `.venv/bin/python tools/test_xxx.py`。

**真实 API（已实跑验证，2026-06-04 探针）：**
- `create_combat_from_enemies(enemies, player_hp, player_max_hp, deck, energy=3, relics=None, ...)` 收 `Enemy` 对象（`create_enemy(id, ai_rng, ascension, hp_rng)` 造），非 EnemyCombatState
- 必须先 `engine.start_combat()`；它随机抽牌，故之后直接覆写 `engine.state.hand` 精确控制手牌
- `engine.play_card(hand_index, target_index)` → `{"success": bool, "card": id, "effects": [...]}`；effects 结构化：damage `{type,target,amount}` / block `{type,amount}` / debuff `{type,debuff,amount}`
- 读字段：`engine.state.player.hp/.block/.statuses`、`engine.state.energy`、`engine.state.enemies[i].hp/.statuses`；状态层数键是 STS 显示名（`"Vulnerable"`/`"Strength"`，虚弱是 `"Weakened"`）
- 卡 id：`"Strike_R"`（6 伤）、`"Defend_R"`（5 格挡）、`"Bash"`（8 伤 + Vulnerable 2，费 2）

---

### Task 1: CombatProbe 接口 + 中性结果类型

**Files:**
- Create: `v8/backends/combat_probe.py`
- Test: `tools/test_combat_probe.py`（本任务先建文件 + 写第一个测试）

- [ ] **Step 1: 写失败测试**

创建 `tools/test_combat_probe.py`：

```python
"""CombatProbe 接口与 StSRLCombatProbe 实现的测试。

跑法：.venv/bin/python tools/test_combat_probe.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = str(Path(__file__).resolve().parent.parent)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from v8.backends.combat_probe import CardPlayResult, CombatProbe, CombatSetup


def test_combat_setup_and_result_are_neutral():
    """CombatSetup / CardPlayResult 可构造，且全是中性 Python 类型。"""
    setup = CombatSetup(hand=("Strike_R",))
    assert setup.hand == ("Strike_R",)
    assert setup.enemy_id == "JawWorm"
    assert setup.player_hp == 80
    assert setup.player_energy == 3

    res = CardPlayResult(
        success=True,
        enemy_hp_delta=6,
        player_block_delta=0,
        energy_delta=1,
        enemy_statuses={},
        player_statuses={},
        effects=[{"type": "damage", "target": "JawWorm", "amount": 6}],
    )
    assert res.success is True
    assert res.enemy_hp_delta == 6
    assert isinstance(res.effects, list)


if __name__ == "__main__":
    test_combat_setup_and_result_are_neutral()
    print("ALL PASS")
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/python tools/test_combat_probe.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'v8.backends.combat_probe'`

- [ ] **Step 3: 实现 combat_probe.py**

创建 `v8/backends/combat_probe.py`：

```python
"""战斗探针能力接口（CombatProbe）。

正交于 GameBackend：GameBackend 是 env 驱动整局的契约，CombatProbe 是「构造受控
战斗、打单卡、读中性结果」的测试能力。只有模拟器后端（如 StSRLSolver）能实现它——
实机做不了受控单卡战斗。返回结构全为中性 Python 类型，不漏引擎对象。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Protocol, Tuple, runtime_checkable


@dataclass(frozen=True)
class CombatSetup:
    """受控战斗的初始条件。"""

    hand: Tuple[str, ...]            # 手牌卡 id（按顺序）
    enemy_id: str = "JawWorm"
    enemy_hp: int = 100
    player_hp: int = 80
    player_energy: int = 3
    relics: Tuple[str, ...] = ()


@dataclass(frozen=True)
class CardPlayResult:
    """打出单卡后的中性结果（不含任何引擎对象）。"""

    success: bool
    enemy_hp_delta: int              # 目标敌人掉血（前 - 后）
    player_block_delta: int          # 玩家格挡变化（后 - 前）
    energy_delta: int                # 消耗能量（前 - 后）
    enemy_statuses: Dict[str, int]   # 目标敌人出牌后的状态层数
    player_statuses: Dict[str, int]  # 玩家出牌后的状态层数
    effects: List[Dict[str, Any]]    # play_card 返回的结构化 effects（纯 dict）


@runtime_checkable
class CombatProbe(Protocol):
    """构造受控战斗 + 打单卡 + 读中性结果的能力接口。"""

    def play_single_card(
        self,
        setup: CombatSetup,
        hand_index: int,
        target_index: int,
    ) -> CardPlayResult:
        """在 setup 描述的受控战斗里打出 hand_index 处的卡（target_index<0 = 自我），
        返回中性 CardPlayResult。"""
        ...
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/bin/python tools/test_combat_probe.py`
Expected: PASS（打印 `ALL PASS`）

- [ ] **Step 5: 提交**

```bash
git add v8/backends/combat_probe.py tools/test_combat_probe.py
git commit -m "feat(中间层): CombatProbe 接口 + 中性结果类型"
```

---

### Task 2: StSRLCombatProbe 实现（封装战斗引擎）

**Files:**
- Create: `v8/backends/stsrl_combat_probe.py`
- Test: `tools/test_combat_probe.py:追加测试`

- [ ] **Step 1: 追加失败测试**

在 `tools/test_combat_probe.py` 的 import 段加：

```python
from v8.backends.stsrl_combat_probe import StSRLCombatProbe
```

在 `test_combat_setup_and_result_are_neutral` 之后、`if __name__` 之前插入：

```python
def test_stsrl_probe_strike_deals_6():
    probe = StSRLCombatProbe()
    res = probe.play_single_card(
        CombatSetup(hand=("Strike_R",)), hand_index=0, target_index=0
    )
    assert res.success is True
    assert res.enemy_hp_delta == 6
    assert res.energy_delta == 1


def test_stsrl_probe_defend_blocks_5():
    probe = StSRLCombatProbe()
    res = probe.play_single_card(
        CombatSetup(hand=("Defend_R",)), hand_index=0, target_index=-1
    )
    assert res.success is True
    assert res.player_block_delta == 5
    assert res.energy_delta == 1


def test_stsrl_probe_bash_damage_and_vulnerable():
    probe = StSRLCombatProbe()
    res = probe.play_single_card(
        CombatSetup(hand=("Bash",)), hand_index=0, target_index=0
    )
    assert res.success is True
    assert res.enemy_hp_delta == 8
    assert res.energy_delta == 2
    assert res.enemy_statuses.get("Vulnerable", 0) == 2


def test_stsrl_probe_satisfies_protocol():
    assert isinstance(StSRLCombatProbe(), CombatProbe)
```

并把 `__main__` 段改为：

```python
if __name__ == "__main__":
    test_combat_setup_and_result_are_neutral()
    test_stsrl_probe_strike_deals_6()
    test_stsrl_probe_defend_blocks_5()
    test_stsrl_probe_bash_damage_and_vulnerable()
    test_stsrl_probe_satisfies_protocol()
    print("ALL PASS")
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/python tools/test_combat_probe.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'v8.backends.stsrl_combat_probe'`

- [ ] **Step 3: 实现 stsrl_combat_probe.py**

创建 `v8/backends/stsrl_combat_probe.py`：

```python
"""StSRLCombatProbe：用 StSRLSolver 战斗引擎实现 CombatProbe。

封装 packages.engine.combat_engine 的 create_combat_from_enemies + play_card，
把结果转成中性 CardPlayResult。坑点（2026-06-04 探针实测）：
  - create_combat_from_enemies 收 Enemy 对象（create_enemy 造），非 EnemyCombatState
  - 必须先 start_combat()；它会随机抽牌，故之后直接覆写 state.hand 精确控制手牌
  - 能量在 state.energy，不在 player 上
  - 状态层数键名是 STS 显示名（"Vulnerable" / "Strength"；虚弱是 "Weakened"）
"""
from __future__ import annotations

from typing import Any, Dict

from sts_paths import ensure_on_sys_path

ensure_on_sys_path()

from packages.engine.combat_engine import create_combat_from_enemies  # noqa: E402
from packages.engine.content.enemies import create_enemy  # noqa: E402
from packages.engine.state.rng import Random as GameRandom  # noqa: E402

from v8.backends.combat_probe import CardPlayResult, CombatSetup  # noqa: E402

_ENEMY_SEED = 12345


class StSRLCombatProbe:
    """实现 CombatProbe（鸭子类型，无需显式继承 Protocol）。"""

    def play_single_card(
        self,
        setup: CombatSetup,
        hand_index: int,
        target_index: int,
    ) -> CardPlayResult:
        engine = self._build_engine(setup)
        read_idx = target_index if target_index >= 0 else 0
        enemy_hp_before = engine.state.enemies[read_idx].hp
        block_before = engine.state.player.block
        energy_before = engine.state.energy

        result: Dict[str, Any] = engine.play_card(hand_index, target_index)

        enemy_after = engine.state.enemies[read_idx]
        return CardPlayResult(
            success=bool(result.get("success", False)),
            enemy_hp_delta=enemy_hp_before - enemy_after.hp,
            player_block_delta=engine.state.player.block - block_before,
            energy_delta=energy_before - engine.state.energy,
            enemy_statuses=dict(enemy_after.statuses),
            player_statuses=dict(engine.state.player.statuses),
            effects=list(result.get("effects", []) or []),
        )

    @staticmethod
    def _build_engine(setup: CombatSetup):
        enemy = create_enemy(
            setup.enemy_id,
            GameRandom(_ENEMY_SEED),
            ascension=0,
            hp_rng=GameRandom(_ENEMY_SEED),
        )
        enemy.state.current_hp = setup.enemy_hp
        enemy.state.max_hp = setup.enemy_hp
        engine = create_combat_from_enemies(
            enemies=[enemy],
            player_hp=setup.player_hp,
            player_max_hp=setup.player_hp,
            deck=list(setup.hand),
            energy=setup.player_energy,
            relics=list(setup.relics),
        )
        engine.start_combat()
        engine.state.hand = list(setup.hand)
        return engine
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/bin/python tools/test_combat_probe.py`
Expected: PASS（打印 `ALL PASS`）。若 FAIL，对照本计划顶部「真实 API」段核对字段名/卡 id；探针已验证三卡数值，数值不符意味着引擎实现与 oracle 不一致——这正是测试要抓的，但起始三卡应当通过。

- [ ] **Step 5: 提交**

```bash
git add v8/backends/stsrl_combat_probe.py tools/test_combat_probe.py
git commit -m "feat(中间层): StSRLCombatProbe 实现 + 三卡数值验证"
```

---

### Task 3: 卡牌期望表（独立 oracle 骨架）

**Files:**
- Create: `tools/card_oracle.py`
- Test: `tools/test_card_oracle.py`

- [ ] **Step 1: 写失败测试**

创建 `tools/test_card_oracle.py`：

```python
"""card_oracle 期望表的结构测试。

跑法：.venv/bin/python tools/test_card_oracle.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = str(Path(__file__).resolve().parent.parent)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from tools.card_oracle import ORACLE, CardExpectation


def test_oracle_entries_well_formed():
    assert len(ORACLE) >= 3
    card_ids = {e.card_id for e in ORACLE}
    assert {"Strike_R", "Defend_R", "Bash"} <= card_ids
    for e in ORACLE:
        assert isinstance(e, CardExpectation)
        assert e.card_id, "card_id 不能为空"
        assert len(e.setup.hand) >= 1, f"{e.card_id} 手牌不能为空"
        assert e.card_id in e.setup.hand, f"{e.card_id} 必须在手牌里"
        assert e.source, f"{e.card_id} 必须标 oracle 来源（独立标准答案出处）"


if __name__ == "__main__":
    test_oracle_entries_well_formed()
    print("ALL PASS")
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/python tools/test_card_oracle.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'tools.card_oracle'`

- [ ] **Step 3: 实现 card_oracle.py**

创建 `tools/card_oracle.py`：

```python
"""卡牌行为期望表（独立 oracle）。

每条 CardExpectation 是一张卡的机制级期望结果，标准答案来源标在 source 字段
（STS wiki / 社区 / 实机轨迹），绝不取被测模拟器自己的数字。

当前为地基骨架，仅含起始三卡；全量 Ironclad + 无色卡分批补（另起计划）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from v8.backends.combat_probe import CombatSetup


@dataclass(frozen=True)
class CardExpectation:
    card_id: str
    setup: CombatSetup
    target_index: int                                    # 敌人 idx；<0 = 自我
    expect_enemy_hp_delta: Optional[int] = None
    expect_player_block_delta: Optional[int] = None
    expect_energy_delta: Optional[int] = None
    expect_enemy_status: Optional[Dict[str, int]] = None  # 子集匹配
    source: str = ""                                      # 标准答案出处


ORACLE: List[CardExpectation] = [
    CardExpectation(
        card_id="Strike_R",
        setup=CombatSetup(hand=("Strike_R",)),
        target_index=0,
        expect_enemy_hp_delta=6,
        expect_energy_delta=1,
        source="STS wiki: Strike deals 6 damage (cost 1)",
    ),
    CardExpectation(
        card_id="Defend_R",
        setup=CombatSetup(hand=("Defend_R",)),
        target_index=-1,
        expect_player_block_delta=5,
        expect_energy_delta=1,
        source="STS wiki: Defend gains 5 Block (cost 1)",
    ),
    CardExpectation(
        card_id="Bash",
        setup=CombatSetup(hand=("Bash",)),
        target_index=0,
        expect_enemy_hp_delta=8,
        expect_energy_delta=2,
        expect_enemy_status={"Vulnerable": 2},
        source="STS wiki: Bash deals 8 damage + 2 Vulnerable (cost 2)",
    ),
]
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/bin/python tools/test_card_oracle.py`
Expected: PASS（打印 `ALL PASS`）

- [ ] **Step 5: 提交**

```bash
git add tools/card_oracle.py tools/test_card_oracle.py
git commit -m "feat(中间层): 卡牌期望表 oracle 骨架（起始三卡）"
```

---

### Task 4: 第 1 层——卡牌行为对照测试跑器

**Files:**
- Create: `tools/test_combat_card_behavior.py`

- [ ] **Step 1: 写测试（即跑器本身，TDD 中它既是测试也是产物）**

创建 `tools/test_combat_card_behavior.py`：

```python
"""第 1 层：卡牌行为对照测试。

读 card_oracle.ORACLE，用一个 CombatProbe 在受控战斗里打每张卡，断言结果对得上
独立 oracle。后端无关：换引擎只换 probe 工厂。

跑法：.venv/bin/python tools/test_combat_card_behavior.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = str(Path(__file__).resolve().parent.parent)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from typing import List

from tools.card_oracle import ORACLE, CardExpectation
from v8.backends.combat_probe import CardPlayResult, CombatProbe
from v8.backends.stsrl_combat_probe import StSRLCombatProbe


def check_expectation(probe: CombatProbe, exp: CardExpectation) -> List[str]:
    """跑一条期望，返回失败描述 list（空 = 通过）。"""
    hand_index = list(exp.setup.hand).index(exp.card_id)
    res: CardPlayResult = probe.play_single_card(
        exp.setup, hand_index=hand_index, target_index=exp.target_index
    )
    fails: List[str] = []
    if not res.success:
        fails.append(f"play_card 未成功（effects={res.effects}）")
    if exp.expect_enemy_hp_delta is not None and res.enemy_hp_delta != exp.expect_enemy_hp_delta:
        fails.append(f"敌人掉血 {res.enemy_hp_delta} != 期望 {exp.expect_enemy_hp_delta}")
    if exp.expect_player_block_delta is not None and res.player_block_delta != exp.expect_player_block_delta:
        fails.append(f"玩家格挡 +{res.player_block_delta} != 期望 +{exp.expect_player_block_delta}")
    if exp.expect_energy_delta is not None and res.energy_delta != exp.expect_energy_delta:
        fails.append(f"耗能 {res.energy_delta} != 期望 {exp.expect_energy_delta}")
    if exp.expect_enemy_status is not None:
        for name, layers in exp.expect_enemy_status.items():
            got = res.enemy_statuses.get(name, 0)
            if got != layers:
                fails.append(f"敌人状态 {name}={got} != 期望 {layers}")
    return fails


def run_all(probe: CombatProbe) -> int:
    """跑全表，返回失败卡数。"""
    total = len(ORACLE)
    failed = 0
    for exp in ORACLE:
        fails = check_expectation(probe, exp)
        if fails:
            failed += 1
            print(f"[FAIL] {exp.card_id}")
            for f in fails:
                print(f"         - {f}")
            print(f"         (oracle 来源: {exp.source})")
        else:
            print(f"[ OK ] {exp.card_id}")
    print(f"\n卡牌行为测试：{total - failed}/{total} 通过，{failed} 失败")
    return failed


def test_starting_cards_behavior():
    """起始三卡行为对照（StSRLSolver 后端）应全绿。"""
    probe = StSRLCombatProbe()
    failed = run_all(probe)
    assert failed == 0, f"{failed} 张卡行为与 oracle 不符"


if __name__ == "__main__":
    test_starting_cards_behavior()
    print("ALL PASS")
```

- [ ] **Step 2: 跑测试**

Run: `.venv/bin/python tools/test_combat_card_behavior.py`
Expected: PASS — 三卡全 `[ OK ]`，打印 `卡牌行为测试：3/3 通过，0 失败` 和 `ALL PASS`。（三卡 probe 已在 Task 2 验证，此处证明「读 oracle → 跑 probe → 断言」整条 harness 通路成立。）

- [ ] **Step 3: 提交**

```bash
git add tools/test_combat_card_behavior.py
git commit -m "feat(中间层): 第1层卡牌行为对照测试跑器（三卡跑通）"
```

---

### Task 5: 第 2 层——后端接口合格性 + 初始状态合理性

**Files:**
- Create: `tools/test_backend_conformance.py`

- [ ] **Step 1: 写失败测试**

创建 `tools/test_backend_conformance.py`：

```python
"""第 2 层：后端接口合格性 + 初始状态合理性（便宜粗筛网）。

只验「reset 不崩 + 快照字段齐全且数值合理 + build_v8_state 非空」这类确定性事实，
不驱动整局（整局无 stall 验证由 eval_harness parity 兜底）。后端无关：任何实现
GameBackend 的后端都应过。

跑法：.venv/bin/python tools/test_backend_conformance.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = str(Path(__file__).resolve().parent.parent)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from v8.backends.stsrl_backend import StSRLBackend

_SNAPSHOT_KEYS = (
    "floor", "act", "hp", "max_hp", "gold",
    "game_won", "deck_size", "relics_size",
)


def _fresh_backend() -> StSRLBackend:
    backend = StSRLBackend(character="ironclad", ascension=0)
    backend.reset(seed=42)
    return backend


def test_reset_initial_flags():
    backend = _fresh_backend()
    assert backend.game_over is False
    assert backend.game_won is False


def test_snapshot_keys_and_ranges():
    backend = _fresh_backend()
    snap = backend.build_runner_snapshot()
    for key in _SNAPSHOT_KEYS:
        assert key in snap, f"snapshot 缺 key: {key}"
    assert snap["hp"] > 0, f"hp 应 > 0，实得 {snap['hp']}"
    assert snap["max_hp"] >= snap["hp"], "max_hp 应 >= hp"
    assert snap["floor"] >= 0, f"floor 应 >= 0，实得 {snap['floor']}"
    assert snap["act"] >= 1, f"act 应 >= 1，实得 {snap['act']}"
    assert snap["deck_size"] >= 10, f"Ironclad 起始 deck 应 >= 10，实得 {snap['deck_size']}"


def test_build_v8_state_non_none():
    backend = _fresh_backend()
    state = backend.build_v8_state()
    assert state is not None


if __name__ == "__main__":
    test_reset_initial_flags()
    test_snapshot_keys_and_ranges()
    test_build_v8_state_non_none()
    print("ALL PASS")
```

- [ ] **Step 2: 跑测试**

Run: `.venv/bin/python tools/test_backend_conformance.py`
Expected: PASS（打印 `ALL PASS`）。

排错提示（仅当 FAIL 时）：
- 若 `build_runner_snapshot` 缺某 key 或返回 0 值 → 可能 reset 后未推进到首决策点。读 `v8/backends/stsrl_backend.py` 的 `build_runner_snapshot` 实现，确认它读的是 `run_state` 哪些字段；`deck_size`/`hp` 应在 GameRunner 构造后即有效（Ironclad 起始 10 卡）。若确实因「reset 不 advance」导致字段为 0，把断言改成调用 `backend.build_v8_state()` 后再读快照，或在 test 里先驱动一步——但优先核对 snapshot 实现而非放宽断言。
- 构造签名若不符，读 `StSRLBackend.__init__` 真实参数对齐。

- [ ] **Step 3: 提交**

```bash
git add tools/test_backend_conformance.py
git commit -m "feat(中间层): 第2层后端接口合格性 + 初始状态合理性测试"
```

---

## 本计划之外（后续另起计划）

1. **全量期望表建表**（重活）：把 `card_oracle.ORACLE` 从 3 张扩到全量 Ironclad（~77）+ 无色卡（~30）。每张卡机制级期望值由 sub-agent 联网查 STS wiki/社区分批产出 + 人工核，高风险卡用实机轨迹仲裁。扩表后 `test_combat_card_behavior.py` 会把 StSRLSolver 的坏卡照成红灯——那正是目标。
2. **增量抹平透传债**：随上面建表过程，遇到哪处需要中性化就抹平哪处（spec 定的增量策略）。
3. **第 3 层实机轨迹交叉验证**：开 STS+Mod 抓轨迹当最终仲裁。
4. **多敌人/多卡序列场景**：当前 probe 是单敌人单卡；需要时扩 `CombatSetup` 支持多敌人、扩 probe 支持连打序列。

## 自查记录（写计划时）

- **Spec 覆盖**：CombatProbe 接口（Task 1）、StSRL 实现（Task 2）、独立 oracle 期望表（Task 3）、第1层跑器（Task 4）、第2层粗筛（Task 5）均落到任务；全量表 + 透传债增量抹平 + 第3层 明确列入「本计划之外」。
- **占位符**：无 TBD/TODO，每步含完整代码与确切命令。
- **类型一致**：`CombatSetup`/`CardPlayResult`/`CombatProbe`/`CardExpectation`/`StSRLCombatProbe.play_single_card(setup, hand_index, target_index)` 跨任务签名一致；卡 id（Strike_R/Defend_R/Bash）与状态键名（Vulnerable）与探针实测一致。

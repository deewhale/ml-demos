"""怪物（敌人）行为验收的金标准 oracle。

oracle = 独立的 wiki 标准答案（A0 难度），与 lightspeed 引擎实现完全解耦。
**纪律：oracle 是 wiki 独立标准答案，对不上如实记，绝不改 oracle 去迁就引擎。**

每条 MonsterExpectation 描述「开战瞬间快照」可验证的事实：
  - HP 范围（按 seed 随机，用 [min, max] 范围断言）
  - 首回合意图（是否攻击 + 攻击伤害量级；非攻击用 is_attack=False）
  - pre-battle 开战机制（开战 buff：metallicize / artifact / block / strength，历史 bug 重灾区）
  - 敌人数量（验召唤 / 多怪 / 双 boss）

assertion 字段（全部可选，None = 不验该项）：
  enemy_index   : 验第几个存活敌人（默认 0；多怪时指定）
  name_contains : 敌人 name 应包含的子串（大小写不敏感）
  hp_min/hp_max : HP 闭区间 [hp_min, hp_max]
  is_attack     : 首意图是否为攻击（True=攻击，False=非攻击）
  dmg_min/dmg_max: 首意图攻击伤害闭区间（仅 is_attack=True 时有意义）
  metallicize / artifact / block / strength / strength_up : 开战状态精确值
  alive_count   : 存活敌人数量（验多怪 / 召唤）

HP 范围按 A0 wiki：
  普通怪 HP 是一个小区间（引擎按 seed 随机），boss 多为固定值（范围退化成单点）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class MonsterExpectation:
    label: str  # 人类可读名（报告用）
    encounter: str  # lightspeed MonsterEncounter 枚举名
    seed: int  # 固定 seed（保证 HP / move roll 可复现）
    # ---- 单怪断言 ----
    enemy_index: int = 0
    name_contains: Optional[str] = None
    hp_min: Optional[int] = None
    hp_max: Optional[int] = None
    is_attack: Optional[bool] = None
    dmg_min: Optional[int] = None
    dmg_max: Optional[int] = None
    metallicize: Optional[int] = None
    artifact: Optional[int] = None
    block: Optional[int] = None
    strength: Optional[int] = None
    strength_up: Optional[int] = None
    # ---- 群体断言 ----
    alive_count: Optional[int] = None
    note: str = ""


# 固定 seed（任取一个能复现的值；不同 seed 只影响 HP roll，不影响 pre-battle 机制 / 意图分类）
_SEED = 42

MONSTER_ORACLE: list[MonsterExpectation] = [
    # ===================== Act1 普通 =====================
    # Jaw Worm：hp∈[40,44]，首意图 = 攻击约 11 伤（Chomp）
    MonsterExpectation(
        label="Jaw Worm",
        encounter="JAW_WORM",
        seed=_SEED,
        hp_min=40, hp_max=44,
        is_attack=True, dmg_min=10, dmg_max=12,
        note="A0 wiki: HP 40-44, 首招 Chomp 攻击 11",
    ),
    # Cultist：hp∈[48,54]，首意图 = Incantation（非攻击，buff 自己 Ritual）
    MonsterExpectation(
        label="Cultist",
        encounter="CULTIST",
        seed=_SEED,
        hp_min=48, hp_max=54,
        is_attack=False,
        note="A0 wiki: HP 48-54, 首招 Incantation 非攻击（给自己 Ritual）",
    ),
    # Red Louse：hp∈[10,15]（用 TWO_LOUSE 遭遇，验其中一个 Louse 的 HP 范围）
    MonsterExpectation(
        label="Red Louse",
        encounter="TWO_LOUSE",
        seed=_SEED,
        name_contains="LOUSE",
        hp_min=10, hp_max=15,
        note="A0 wiki: Louse HP 10-15（TWO_LOUSE 遭遇，验 idx0 那只）",
    ),
    # Fungi Beast：hp∈[22,28]
    MonsterExpectation(
        label="Fungi Beast",
        encounter="TWO_FUNGI_BEASTS",
        seed=_SEED,
        name_contains="FUNGI",
        hp_min=22, hp_max=28,
        note="A0 wiki: Fungi Beast HP 22-28（TWO_FUNGI_BEASTS 遭遇，验 idx0）",
    ),

    # ===================== Act1 精英（pre-battle 重点）=====================
    # Gremlin Nob：hp∈[82,86]，首意图 = Bellow（非攻击 buff，Enrage）
    MonsterExpectation(
        label="Gremlin Nob",
        encounter="GREMLIN_NOB",
        seed=_SEED,
        hp_min=82, hp_max=86,
        is_attack=False,
        note="A0 wiki: HP 82-86, 首招 Bellow 非攻击（给自己 Enrage）",
    ),
    # Lagavulin：hp∈[109,111]，开战 metallicize==8 且前几回合睡眠不攻击
    MonsterExpectation(
        label="Lagavulin",
        encounter="LAGAVULIN",
        seed=_SEED,
        hp_min=109, hp_max=111,
        is_attack=False,
        metallicize=8,
        note="A0 wiki: HP 109-111, 开战 Metallicize 8, 首几回合 Sleep 不攻击",
    ),
    # Sentries（三个）：每个 hp∈[38,42]，每个 artifact==1（开战 Artifact）
    MonsterExpectation(
        label="Sentry[0]",
        encounter="THREE_SENTRIES",
        seed=_SEED,
        enemy_index=0,
        hp_min=38, hp_max=42,
        artifact=1,
        alive_count=3,
        note="A0 wiki: 三个 Sentry, 每个 HP 38-42 + 开战 Artifact 1",
    ),
    MonsterExpectation(
        label="Sentry[1]",
        encounter="THREE_SENTRIES",
        seed=_SEED,
        enemy_index=1,
        hp_min=38, hp_max=42,
        artifact=1,
        note="A0 wiki: Sentry idx1 HP 38-42 + Artifact 1",
    ),
    MonsterExpectation(
        label="Sentry[2]",
        encounter="THREE_SENTRIES",
        seed=_SEED,
        enemy_index=2,
        hp_min=38, hp_max=42,
        artifact=1,
        note="A0 wiki: Sentry idx2 HP 38-42 + Artifact 1",
    ),

    # ===================== Act1 Boss =====================
    # Slime Boss：hp==140，首意图 = Goop Spray（非攻击，给 Slimed）
    MonsterExpectation(
        label="Slime Boss",
        encounter="SLIME_BOSS",
        seed=_SEED,
        hp_min=140, hp_max=140,
        is_attack=False,
        note="A0 wiki: HP 140, 首招 Goop Spray 非攻击（给 Slimed 状态卡）",
    ),
    # The Guardian：hp==240
    MonsterExpectation(
        label="The Guardian",
        encounter="THE_GUARDIAN",
        seed=_SEED,
        hp_min=240, hp_max=240,
        note="A0 wiki: HP 240",
    ),
    # Hexaghost：hp==250，首意图 = Activate（非攻击不动）
    MonsterExpectation(
        label="Hexaghost",
        encounter="HEXAGHOST",
        seed=_SEED,
        hp_min=250, hp_max=250,
        is_attack=False,
        note="A0 wiki: HP 250, 首招 Activate 非攻击（充能不出手）",
    ),

    # ===================== Act2/3 pre-battle 重点 =====================
    # Bronze Automaton：hp==300，artifact==3（开战）
    MonsterExpectation(
        label="Bronze Automaton",
        encounter="AUTOMATON",
        seed=_SEED,
        name_contains="AUTOMATON",
        hp_min=300, hp_max=300,
        artifact=3,
        note="A0 wiki: HP 300, 开战 Artifact 3（遭遇里有空 minion slot，验 Automaton 本体）",
    ),
    # Spheric Guardian：hp==20，开战高 block（约40）（Barricade + 开局护甲）
    MonsterExpectation(
        label="Spheric Guardian",
        encounter="SPHERIC_GUARDIAN",
        seed=_SEED,
        hp_min=20, hp_max=20,
        block=40,
        note="A0 wiki: HP 20, 开战 ~40 block（Barricade 不消散）",
    ),
    # Donu/Deca（双 Boss）：每个 hp==250，每个 artifact==2
    MonsterExpectation(
        label="Deca",
        encounter="DONU_AND_DECA",
        seed=_SEED,
        name_contains="DECA",
        hp_min=250, hp_max=250,
        artifact=2,
        alive_count=2,
        note="A0 wiki: Deca HP 250 + 开战 Artifact 2（双 boss）",
    ),
    MonsterExpectation(
        label="Donu",
        encounter="DONU_AND_DECA",
        seed=_SEED,
        name_contains="DONU",
        hp_min=250, hp_max=250,
        artifact=2,
        note="A0 wiki: Donu HP 250 + 开战 Artifact 2（双 boss）",
    ),
    # Awakened One：hp 约300，开战场上有 2 个 Cultist 小怪（敌人数==3：本体+2 Cultist）
    MonsterExpectation(
        label="Awakened One (boss body)",
        encounter="AWAKENED_ONE",
        seed=_SEED,
        name_contains="AWAKENED",
        hp_min=290, hp_max=320,
        alive_count=3,
        note="A0 wiki: Awakened One ~300 HP, 开战场上 2 个 Cultist 小怪（共 3 个敌人）",
    ),
    # Orb Walker：hp∈[90,96]，开战 strength==3
    MonsterExpectation(
        label="Orb Walker",
        encounter="ORB_WALKER",
        seed=_SEED,
        hp_min=90, hp_max=96,
        strength=3,
        note="A0 wiki: HP 90-96, 开战 Strength 3",
    ),
]

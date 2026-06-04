"""遗物行为期望表（独立 oracle，开战触发类 atBattleStart 优先）。

每条 RelicExpectation 是一个遗物在「开战瞬间」应造成的机制级结果。标准答案来源
标在 source 字段（STS wiki），**绝不取被测模拟器自己的数字**——lightspeed 对不上
如实记，不改 oracle 迁就引擎。

atBattleStart（开战触发）类历史上是巨型 bug 区（NeowsLament 永不递减、8 个 boss/elite
的 atBattleStart buff 全部失效），所以这批专门验「遗物到底装上没、开战效果有没有真触发」。

字段：
  - relic           : lightspeed RelicId 枚举名（字符串，运行时 getattr 取）
  - label           : 场景标签（同遗物多场景区分）
  - encounter       : MonsterEncounter 枚举名（字符串）
  - player_hp       : 开战玩家血量（<=0 表示用默认）
  - enemy_hps       : 每只怪 hp 覆写（按 idx）
  - energy          : 能量覆写（<0 = 不覆写，让遗物/init 决定，用于 Lantern）
  - expect_player   : 开战瞬间玩家状态期望（子集匹配；键见 canonical 列表）
  - expect_enemies  : 每只怪都应有的状态期望（子集匹配，全体性）
  - play_card       : 若非 None，开战后打这张牌（lightspeed CardId 枚举名）再断言伤害
  - play_target     : play_card 的目标怪 idx
  - expect_enemy_hp_delta : play_card 后目标怪应掉的血（用于 Akabeko / Vajra 伤害验证）
  - source          : 标准答案出处

canonical player 状态键（snapshot 暴露的）：
  hp / max_hp / block / energy / strength / dexterity / artifact /
  vulnerable / weak / frail / thorns / metallicize / regen / ritual / vigor
canonical enemy 状态键：
  hp / vulnerable / weak / strength / poison / metallicize / ritual
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class RelicExpectation:
    relic: str                                          # lightspeed RelicId 枚举名
    label: str = field(default="", kw_only=True)
    encounter: str = "JAW_WORM"                         # 单怪默认 JAW_WORM
    player_hp: int = 70
    enemy_hps: tuple = ()                               # 每只怪 hp 覆写
    energy: int = 3
    expect_player: Optional[Dict[str, int]] = None      # 玩家开战状态（子集匹配）
    expect_enemies: Optional[Dict[str, int]] = None     # 每只怪开战状态（子集匹配）
    play_card: Optional[str] = None                     # 开战后打的牌（CardId 枚举名）
    play_target: int = 0
    expect_enemy_hp_delta: Optional[int] = None         # play_card 后目标怪掉血
    source: str = ""


# 单怪 encounter = JAW_WORM；多怪验全体性用 THREE_LOUSE（3 怪）。
RELIC_ORACLE: List[RelicExpectation] = [
    RelicExpectation(
        relic="VAJRA",
        encounter="JAW_WORM",
        enemy_hps=(40,),
        expect_player={"strength": 1},
        source="STS wiki: Vajra — at combat start, gain 1 Strength",
    ),
    RelicExpectation(
        relic="BAG_OF_MARBLES",
        label="all_enemies_vulnerable",
        encounter="THREE_LOUSE",
        enemy_hps=(15, 15, 15),
        expect_enemies={"vulnerable": 1},
        source="STS wiki: Bag of Marbles — at combat start, apply 1 Vulnerable to ALL enemies",
    ),
    RelicExpectation(
        relic="ANCHOR",
        encounter="JAW_WORM",
        enemy_hps=(40,),
        expect_player={"block": 10},
        source="STS wiki: Anchor — first turn, start combat with 10 Block",
    ),
    RelicExpectation(
        relic="BRONZE_SCALES",
        encounter="JAW_WORM",
        enemy_hps=(40,),
        expect_player={"thorns": 3},
        source="STS wiki: Bronze Scales — at combat start, gain 3 Thorns",
    ),
    RelicExpectation(
        relic="ODDLY_SMOOTH_STONE",
        encounter="JAW_WORM",
        enemy_hps=(40,),
        expect_player={"dexterity": 1},
        source="STS wiki: Oddly Smooth Stone — at combat start, gain 1 Dexterity",
    ),
    RelicExpectation(
        relic="BLOOD_VIAL",
        encounter="JAW_WORM",
        player_hp=70,                                   # max 80，开战回 2 → 72
        enemy_hps=(40,),
        expect_player={"hp": 72},
        source="STS wiki: Blood Vial — at combat start, heal 2 HP (70/80 -> 72)",
    ),
    RelicExpectation(
        relic="RED_MASK",
        label="all_enemies_weak",
        encounter="THREE_LOUSE",
        enemy_hps=(15, 15, 15),
        expect_enemies={"weak": 1},
        source="STS wiki: Red Mask — at combat start, apply 1 Weak to ALL enemies",
    ),
    RelicExpectation(
        relic="LANTERN",
        encounter="JAW_WORM",
        enemy_hps=(40,),
        energy=-1,                                      # 不覆写能量，让 init(3)+Lantern(1)=4 生效
        expect_player={"energy": 4},
        source="STS wiki: Lantern — first turn, gain 1 extra Energy (base 3 -> 4)",
    ),
    RelicExpectation(
        relic="MUTAGENIC_STRENGTH",
        encounter="JAW_WORM",
        enemy_hps=(40,),
        expect_player={"strength": 3},
        source="STS wiki: Mutagenic Strength — at combat start, gain 3 Strength "
               "(lose 3 at end of turn; snapshot at start is 3)",
    ),
    # ---- play_card 伤害验证 ----
    RelicExpectation(
        relic="AKABEKO",
        label="first_strike_+8",
        encounter="JAW_WORM",
        enemy_hps=(40,),
        play_card="STRIKE_RED",
        play_target=0,
        expect_enemy_hp_delta=14,                       # Strike 6 + Vigor 8 = 14
        source="STS wiki: Akabeko — first attack each combat deals +8 (Strike 6 + 8 = 14)",
    ),
    RelicExpectation(
        relic="VAJRA",
        label="strike_damage_+1",
        encounter="JAW_WORM",
        enemy_hps=(40,),
        play_card="STRIKE_RED",
        play_target=0,
        expect_enemy_hp_delta=7,                        # Strike 6 + 1 Strength = 7
        source="STS wiki: Vajra — 1 Strength makes Strike deal 6+1 = 7",
    ),
]

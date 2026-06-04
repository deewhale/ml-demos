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
    expect_player_status: Optional[Dict[str, int]] = None  # 子集匹配（玩家自身状态）
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
    CardExpectation(
        card_id="Anger",
        setup=CombatSetup(hand=("Anger",)),
        target_index=0,
        expect_enemy_hp_delta=6,
        expect_energy_delta=0,
        source="STS wiki: Anger 0-cost, 6 damage (also adds a copy to discard — not asserted here)",
    ),
    CardExpectation(
        card_id="Clothesline",
        setup=CombatSetup(hand=("Clothesline",)),
        target_index=0,
        expect_enemy_hp_delta=12,
        expect_energy_delta=2,
        expect_enemy_status={"Weakened": 2},
        source="STS wiki: Clothesline 2-cost, 12 damage + 2 Weak",
    ),
    CardExpectation(
        card_id="Twin Strike",
        setup=CombatSetup(hand=("Twin Strike",)),
        target_index=0,
        expect_enemy_hp_delta=10,
        expect_energy_delta=1,
        source="STS wiki: Twin Strike 5 damage x2 = 10",
    ),
    CardExpectation(
        card_id="Thunderclap",
        setup=CombatSetup(hand=("Thunderclap",)),
        target_index=0,
        expect_enemy_hp_delta=4,
        expect_energy_delta=1,
        expect_enemy_status={"Vulnerable": 1},
        source="STS wiki: Thunderclap 4 damage + 1 Vulnerable to ALL (single-enemy slice here)",
    ),
    CardExpectation(
        card_id="Iron Wave",
        setup=CombatSetup(hand=("Iron Wave",)),
        target_index=0,
        expect_enemy_hp_delta=5,
        expect_player_block_delta=5,
        expect_energy_delta=1,
        source="STS wiki: Iron Wave 5 block + 5 damage",
    ),
    CardExpectation(
        card_id="Pommel Strike",
        setup=CombatSetup(hand=("Pommel Strike",)),
        target_index=0,
        expect_enemy_hp_delta=9,
        expect_energy_delta=1,
        source="STS wiki: Pommel Strike 9 damage + draw 1 (draw not asserted)",
    ),
    CardExpectation(
        card_id="Shrug It Off",
        setup=CombatSetup(hand=("Shrug It Off",)),
        target_index=-1,
        expect_enemy_hp_delta=0,
        expect_player_block_delta=8,
        expect_energy_delta=1,
        source="STS wiki: Shrug It Off 8 block + draw 1 (draw not asserted)",
    ),
    CardExpectation(
        card_id="Inflame",
        setup=CombatSetup(hand=("Inflame",)),
        target_index=-1,
        expect_enemy_hp_delta=0,
        expect_player_block_delta=0,
        expect_player_status={"Strength": 2},
        expect_energy_delta=1,
        source="STS wiki: Inflame power, +2 Strength",
    ),
]

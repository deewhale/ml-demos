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

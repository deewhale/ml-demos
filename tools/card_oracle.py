"""卡牌行为期望表（独立 oracle）。

每条 CardExpectation 是一张卡的机制级期望结果，标准答案来源标在 source 字段
（STS wiki / 社区 / 实机轨迹），绝不取被测模拟器自己的数字。

当前为地基骨架，仅含起始三卡；全量 Ironclad + 无色卡分批补（另起计划）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from v8.backends.combat_probe import CombatSetup


@dataclass(frozen=True)
class CardExpectation:
    card_id: str
    # 场景标签（同一张卡多场景时区分）；空则显示 card_id。kw_only 让它声明在
    # card_id 之后又不强制后续 setup/target_index 带默认值。
    label: str = field(default="", kw_only=True)
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
    # ---- 力量注入对照：证明 player_strength 注入真生效 ----
    CardExpectation(
        card_id="Strike_R",
        label="Strike_R (+3 Str control)",
        setup=CombatSetup(hand=("Strike_R",), player_strength=3),
        target_index=0,
        expect_enemy_hp_delta=9,   # 6 基础 + 3 力量
        expect_energy_delta=1,
        source="STS: Strike 6 + Strength 3 = 9（力量注入对照）",
    ),
    # ---- Body Slam：伤害=当前格挡，力量加性不是乘性，且不消耗格挡 ----
    CardExpectation(
        card_id="Body Slam",
        label="Body Slam (10 block)",
        setup=CombatSetup(hand=("Body Slam",), player_block=10, player_strength=0),
        target_index=0,
        expect_enemy_hp_delta=10,
        expect_player_block_delta=0,
        expect_energy_delta=1,
        source="STS wiki: Body Slam damage = current Block(10); block not consumed",
    ),
    CardExpectation(
        card_id="Body Slam",
        label="Body Slam (10 block +3 Str)",
        setup=CombatSetup(hand=("Body Slam",), player_block=10, player_strength=3),
        target_index=0,
        expect_enemy_hp_delta=13,
        expect_energy_delta=1,
        source="STS wiki: Body Slam = Block + Strength additive = 13",
    ),
    CardExpectation(
        card_id="Body Slam",
        label="Body Slam (0 block +5 Str)",
        setup=CombatSetup(hand=("Body Slam",), player_block=0, player_strength=5),
        target_index=0,
        expect_enemy_hp_delta=5,
        expect_energy_delta=1,
        source="STS wiki: Body Slam 0 block + 5 Str = 5",
    ),
    # ---- Heavy Blade：基础 14，力量×3 倍率，cost 2 ----
    CardExpectation(
        card_id="Heavy Blade",
        label="Heavy Blade (0 Str)",
        setup=CombatSetup(hand=("Heavy Blade",), player_strength=0),
        target_index=0,
        expect_enemy_hp_delta=14,
        expect_energy_delta=2,
        source="STS wiki: Heavy Blade base 14",
    ),
    CardExpectation(
        card_id="Heavy Blade",
        label="Heavy Blade (+3 Str)",
        setup=CombatSetup(hand=("Heavy Blade",), player_strength=3),
        target_index=0,
        expect_enemy_hp_delta=23,   # 14 + 3×3
        expect_energy_delta=2,
        source="STS wiki: Heavy Blade 14 + Strength×3 = 23",
    ),
    CardExpectation(
        card_id="Heavy Blade",
        label="Heavy Blade (-2 Str)",
        setup=CombatSetup(hand=("Heavy Blade",), player_strength=-2),
        target_index=0,
        expect_enemy_hp_delta=8,    # 14 + (-2)×3
        expect_energy_delta=2,
        source="STS wiki: Heavy Blade 14 + (-2)×3 = 8",
    ),
]

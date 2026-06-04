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
    expect_all_enemies_hp_delta: Optional[int] = None      # 每个敌人都应掉这么多
    expect_all_enemies_status: Optional[Dict[str, int]] = None  # 每个敌人都应有这些状态（子集匹配）
    # ---- 牌堆内容断言（pile inspection；探针不支持则会因 size=0 报红，需人工判读）----
    expect_draw_pile_size: Optional[int] = None           # 出牌后抽牌堆张数
    expect_discard_pile_size: Optional[int] = None        # 出牌后弃牌堆张数
    expect_exhaust_pile_size: Optional[int] = None        # 出牌后消耗堆张数
    expect_hand_size: Optional[int] = None                # 出牌后手牌张数
    expect_draw_pile_contains: Optional[Dict[str, int]] = None     # 抽牌堆须含某卡至少 N 张（按卡名计数）
    expect_discard_pile_contains: Optional[Dict[str, int]] = None  # 弃牌堆须含某卡至少 N 张
    expect_exhaust_pile_contains: Optional[Dict[str, int]] = None  # 消耗堆须含某卡至少 N 张
    expect_hand_contains: Optional[Dict[str, int]] = None          # 手牌须含某卡至少 N 张
    expect_unplayable: bool = False                       # True = 该卡当前应被引擎判为不可打出（success=False）
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
    ),    # ===== 第二轮：38 张可测卡 baseline（none / strength_or_block_setup，剔除已在表内）=====
    CardExpectation(
        card_id='Flex',
        setup=CombatSetup(hand=('Flex',)),
        target_index=-1,
        expect_enemy_hp_delta=0,
        expect_player_status={'Strength': 2},
        expect_energy_delta=0,
        source='https://slaythespire.wiki.gg/wiki/Flex',
    ),
    CardExpectation(
        card_id='Carnage',
        setup=CombatSetup(hand=('Carnage',)),
        target_index=0,
        expect_enemy_hp_delta=20,
        expect_energy_delta=2,
        source='https://slaythespire.wiki.gg/wiki/Carnage',
    ),
    CardExpectation(
        card_id='Dropkick',
        setup=CombatSetup(hand=('Dropkick',)),
        target_index=0,
        expect_enemy_hp_delta=5,
        expect_energy_delta=1,
        source='https://slaythespire.wiki.gg/wiki/Dropkick',
    ),
    CardExpectation(
        card_id='Hemokinesis',
        setup=CombatSetup(hand=('Hemokinesis',)),
        target_index=0,
        expect_enemy_hp_delta=15,
        expect_energy_delta=1,
        source='https://slaythespire.wiki.gg/wiki/Hemokinesis',
    ),
    CardExpectation(
        card_id='Pummel',
        setup=CombatSetup(hand=('Pummel',)),
        target_index=0,
        expect_enemy_hp_delta=8,
        expect_energy_delta=1,
        source='https://slaythespire.wiki.gg/wiki/Pummel',
    ),
    CardExpectation(
        card_id='Searing Blow',
        setup=CombatSetup(hand=('Searing Blow',)),
        target_index=0,
        expect_enemy_hp_delta=12,
        expect_energy_delta=2,
        source='https://slaythespire.wiki.gg/wiki/Searing_Blow',
    ),
    CardExpectation(
        card_id='Uppercut',
        setup=CombatSetup(hand=('Uppercut',)),
        target_index=0,
        expect_enemy_hp_delta=13,
        expect_enemy_status={'Vulnerable': 1, 'Weakened': 1},
        expect_energy_delta=2,
        source='https://slaythespire.wiki.gg/wiki/Uppercut',
    ),
    CardExpectation(
        card_id='Battle Trance',
        setup=CombatSetup(hand=('Battle Trance',)),
        target_index=-1,
        expect_enemy_hp_delta=0,
        expect_player_status={'NoDraw': 1},
        expect_energy_delta=0,
        source='https://slaythespire.wiki.gg/wiki/Battle_Trance',
    ),
    CardExpectation(
        card_id='Bloodletting',
        setup=CombatSetup(hand=('Bloodletting',)),
        target_index=-1,
        expect_enemy_hp_delta=0,
        source='https://slaythespire.wiki.gg/wiki/Bloodletting',
    ),
    CardExpectation(
        card_id='Disarm',
        setup=CombatSetup(hand=('Disarm',)),
        target_index=0,
        expect_enemy_hp_delta=0,
        expect_enemy_status={'Strength': -2},
        expect_energy_delta=1,
        source='https://slaythespire.wiki.gg/wiki/Disarm',
    ),
    CardExpectation(
        card_id='Entrench',
        setup=CombatSetup(hand=('Entrench',)),
        target_index=-1,
        expect_enemy_hp_delta=0,
        expect_energy_delta=2,
        source='https://slaythespire.wiki.gg/wiki/Entrench',
    ),
    CardExpectation(
        card_id='Ghostly Armor',
        setup=CombatSetup(hand=('Ghostly Armor',)),
        target_index=-1,
        expect_enemy_hp_delta=0,
        expect_player_block_delta=10,
        expect_energy_delta=1,
        source='https://slaythespire.wiki.gg/wiki/Ghostly_Armor',
    ),
    CardExpectation(
        card_id='Seeing Red',
        setup=CombatSetup(hand=('Seeing Red',)),
        target_index=-1,
        expect_enemy_hp_delta=0,
        source='https://slaythespire.wiki.gg/wiki/Seeing_Red',
    ),
    CardExpectation(
        card_id='Bludgeon',
        setup=CombatSetup(hand=('Bludgeon',)),
        target_index=0,
        expect_enemy_hp_delta=32,
        expect_energy_delta=3,
        source='https://slaythespire.wiki.gg/wiki/Bludgeon',
    ),
    CardExpectation(
        card_id='Feed',
        setup=CombatSetup(hand=('Feed',)),
        target_index=0,
        expect_enemy_hp_delta=10,
        expect_energy_delta=1,
        source='https://slaythespire.wiki.gg/wiki/Feed',
    ),
    CardExpectation(
        card_id='Impervious',
        setup=CombatSetup(hand=('Impervious',)),
        target_index=-1,
        expect_enemy_hp_delta=0,
        expect_player_block_delta=30,
        expect_energy_delta=2,
        source='https://slaythespire.wiki.gg/wiki/Impervious',
    ),
    CardExpectation(
        card_id='Limit Break',
        setup=CombatSetup(hand=('Limit Break',)),
        target_index=-1,
        expect_enemy_hp_delta=0,
        expect_energy_delta=1,
        source='https://slaythespire.wiki.gg/wiki/Limit_Break',
    ),
    CardExpectation(
        card_id='Barricade',
        setup=CombatSetup(hand=('Barricade',)),
        target_index=-1,
        expect_enemy_hp_delta=0,
        expect_player_status={'Barricade': 1},
        expect_energy_delta=3,
        source='https://slaythespire.wiki.gg/wiki/Barricade',
    ),
    CardExpectation(
        card_id='Berserk',
        # wiki：自身 +2 Vulnerable（可断言）+ 每回合开始 +1 能量。后者 lightspeed 以
        # energyPerTurn+1 实现（无名为 "Berserk" 的状态层），已人工核实 energy_per_turn
        # 3->4 正确，故此处只断言引擎无关的 Vulnerable:2（原 Berserk:1 是研究者对内部
        # 表示的猜测键，无引擎以此命名 → 删去以免误判红灯；+1 能量效果引擎正确）。
        setup=CombatSetup(hand=('Berserk',), clear_draw_pile=True),
        target_index=-1,
        expect_enemy_hp_delta=0,
        expect_player_status={'Vulnerable': 2},
        expect_energy_delta=0,
        source='https://slaythespire.wiki.gg/wiki/Berserk (self +2 Vulnerable; +1 energy/turn via energyPerTurn)',
    ),
    CardExpectation(
        card_id='Juggernaut',
        setup=CombatSetup(hand=('Juggernaut',)),
        target_index=-1,
        expect_enemy_hp_delta=0,
        expect_energy_delta=2,
        source='https://slaythespire.wiki.gg/wiki/Juggernaut',
    ),
    CardExpectation(
        card_id='Blind',
        setup=CombatSetup(hand=('Blind',)),
        target_index=0,
        expect_enemy_hp_delta=0,
        expect_enemy_status={'Weakened': 2},
        expect_energy_delta=0,
        source='https://slaythespire.wiki.gg/wiki/Blind',
    ),
    CardExpectation(
        card_id='Dark Shackles',
        # wiki 真实效果：目标本回合「失去 9 点力量」= 敌人 Strength 变 -9（回合末还回去）。
        # 用 Strength=-9 断言这一引擎无关的可观测效果（原 Shackled:9 是研究者对内部表示的
        # 猜测键，引擎不一定以同名层数表示）。lightspeed 实测给 +9（符号反了）= 真红灯。
        setup=CombatSetup(hand=('Dark Shackles',), clear_draw_pile=True),
        target_index=0,
        expect_enemy_hp_delta=0,
        expect_enemy_status={'Strength': -9},
        expect_energy_delta=0,
        source='https://slaythespire.wiki.gg/wiki/Dark_Shackles (target loses 9 Strength this turn)',
    ),
    CardExpectation(
        card_id='Finesse',
        setup=CombatSetup(hand=('Finesse',)),
        target_index=-1,
        expect_enemy_hp_delta=0,
        expect_player_block_delta=2,
        expect_energy_delta=0,
        source='https://slaythespire.wiki.gg/wiki/Finesse',
    ),
    CardExpectation(
        card_id='Flash of Steel',
        setup=CombatSetup(hand=('Flash of Steel',)),
        target_index=0,
        expect_enemy_hp_delta=3,
        expect_energy_delta=0,
        source='https://slaythespire.wiki.gg/wiki/Flash_of_Steel',
    ),
    CardExpectation(
        card_id='Good Instincts',
        setup=CombatSetup(hand=('Good Instincts',)),
        target_index=-1,
        expect_enemy_hp_delta=0,
        expect_player_block_delta=6,
        expect_energy_delta=0,
        source='https://slaythespire.wiki.gg/wiki/Good_Instincts',
    ),
    CardExpectation(
        card_id='Panacea',
        setup=CombatSetup(hand=('Panacea',)),
        target_index=-1,
        expect_enemy_hp_delta=0,
        expect_player_status={'Artifact': 1},
        expect_energy_delta=0,
        source='https://slaythespire.wiki.gg/wiki/Panacea',
    ),
    CardExpectation(
        card_id='PanicButton',
        setup=CombatSetup(hand=('PanicButton',)),
        target_index=-1,
        expect_enemy_hp_delta=0,
        expect_player_block_delta=30,
        expect_player_status={'No Block': 2},
        expect_energy_delta=0,
        source='https://slaythespire.wiki.gg/wiki/Panic_Button',
    ),
    CardExpectation(
        card_id='Swift Strike',
        setup=CombatSetup(hand=('Swift Strike',)),
        target_index=0,
        expect_enemy_hp_delta=7,
        expect_energy_delta=0,
        source='https://slaythespire.wiki.gg/wiki/Swift_Strike',
    ),
    CardExpectation(
        card_id='Trip',
        setup=CombatSetup(hand=('Trip',)),
        target_index=0,
        expect_enemy_hp_delta=0,
        expect_enemy_status={'Vulnerable': 2},
        expect_energy_delta=0,
        source='https://slaythespire.wiki.gg/wiki/Trip',
    ),
    CardExpectation(
        card_id='Bite',
        setup=CombatSetup(hand=('Bite',)),
        target_index=0,
        expect_enemy_hp_delta=7,
        expect_energy_delta=1,
        source='https://slaythespire.wiki.gg/wiki/Bite',
    ),
    CardExpectation(
        card_id='J.A.X.',
        setup=CombatSetup(hand=('J.A.X.',)),
        target_index=-1,
        expect_enemy_hp_delta=0,
        expect_player_status={'Strength': 2},
        expect_energy_delta=0,
        source='https://slaythespire.wiki.gg/wiki/J.A.X.',
    ),
    # ===== 第三轮：multi_enemy AoE 卡（enemy_count=3 验全体）=====
    # 固定全体伤害 / debuff 卡
    CardExpectation(
        card_id='Cleave',
        label='Cleave AoE',
        setup=CombatSetup(hand=('Cleave',), enemy_count=3),
        target_index=0,
        expect_all_enemies_hp_delta=8,
        expect_energy_delta=1,
        source='https://slaythespire.wiki.gg/wiki/Cleave (AoE 8 to all)',
    ),
    CardExpectation(
        card_id='Thunderclap',
        label='Thunderclap AoE',
        setup=CombatSetup(hand=('Thunderclap',), enemy_count=3),
        target_index=0,
        expect_all_enemies_hp_delta=4,
        expect_all_enemies_status={'Vulnerable': 1},
        expect_energy_delta=1,
        source='https://slaythespire.wiki.gg/wiki/Thunderclap (AoE 4 + 1 Vulnerable to all)',
    ),
    CardExpectation(
        card_id='Intimidate',
        label='Intimidate AoE',
        setup=CombatSetup(hand=('Intimidate',), enemy_count=3),
        target_index=0,
        expect_all_enemies_hp_delta=0,
        expect_all_enemies_status={'Weakened': 1},
        expect_energy_delta=0,
        source='https://slaythespire.wiki.gg/wiki/Intimidate (AoE 1 Weak to all, exhaust)',
    ),
    CardExpectation(
        card_id='Shockwave',
        label='Shockwave AoE',
        setup=CombatSetup(hand=('Shockwave',), enemy_count=3),
        target_index=0,
        expect_all_enemies_hp_delta=0,
        expect_all_enemies_status={'Weakened': 3, 'Vulnerable': 3},
        expect_energy_delta=2,
        source='https://slaythespire.wiki.gg/wiki/Shockwave (cost 2; AoE 3 Weak + 3 Vulnerable to all, exhaust)',
    ),
    CardExpectation(
        card_id='Immolate',
        label='Immolate AoE',
        setup=CombatSetup(hand=('Immolate',), enemy_count=3),
        target_index=0,
        expect_all_enemies_hp_delta=21,
        expect_energy_delta=2,
        source='https://slaythespire.wiki.gg/wiki/Immolate (AoE 21 to all + Burn to discard)',
    ),
    CardExpectation(
        card_id='Reaper',
        label='Reaper AoE',
        setup=CombatSetup(hand=('Reaper',), enemy_count=3),
        target_index=0,
        expect_all_enemies_hp_delta=4,
        expect_energy_delta=2,
        source='https://slaythespire.wiki.gg/wiki/Reaper (AoE 4 to all + lifesteal, exhaust)',
    ),
    CardExpectation(
        card_id='Dramatic Entrance',
        label='Dramatic Entrance AoE',
        setup=CombatSetup(hand=('Dramatic Entrance',), enemy_count=3),
        target_index=0,
        expect_all_enemies_hp_delta=8,
        expect_energy_delta=0,
        source='https://slaythespire.wiki.gg/wiki/Dramatic_Entrance (AoE 8 to all, exhaust)',
    ),
    # 随机分配伤害卡：多敌不可逐敌确定性断言，改用 enemy_count=1 验单敌总伤
    CardExpectation(
        card_id='Sword Boomerang',
        label='Sword Boomerang (single-enemy, 随机分配仅单敌可确定测)',
        setup=CombatSetup(hand=('Sword Boomerang',), enemy_count=1),
        target_index=0,
        expect_enemy_hp_delta=9,
        expect_energy_delta=1,
        source='https://slaythespire.wiki.gg/wiki/Sword_Boomerang (3x3 random target; single enemy all hits land = 9)',
    ),
    # ===== 第四轮：pile_inspection / deck_composition / x_cost / multi_play / other =====
    # 牌堆机制卡：clear_draw_pile=True 让抽牌堆确定可控；断言牌堆落点 + 数值。

    # ---- pile_inspection ----
    CardExpectation(
        card_id='Wild Strike',
        setup=CombatSetup(hand=('Wild Strike',), clear_draw_pile=True),
        target_index=0,
        expect_enemy_hp_delta=12,
        expect_energy_delta=1,
        expect_draw_pile_contains={'Wound': 1},  # Wound 洗入抽牌堆（不是弃牌堆）
        source='https://slaythespire.wiki.gg/wiki/Wild_Strike (12 dmg + Wound into draw pile)',
    ),
    CardExpectation(
        card_id='Reckless Charge',
        setup=CombatSetup(hand=('Reckless Charge',), clear_draw_pile=True),
        target_index=0,
        expect_enemy_hp_delta=7,
        expect_energy_delta=0,
        expect_draw_pile_contains={'Dazed': 1},  # Dazed 进抽牌堆
        source='https://slaythespire.wiki.gg/wiki/Reckless_Charge (7 dmg + Dazed into draw pile)',
    ),
    CardExpectation(
        card_id='Headbutt',
        # 弃牌堆放一张牌（这里用预置抽牌堆放 Defend，Headbutt 从弃牌堆取——
        # lightspeed 无 discard 预置接口，改验：打出后抽牌堆顶应有那张被取回的牌。
        # 简化：仅验 9 伤 + 自身进弃牌堆，move-to-draw-top 由下方 contains 体现需要 discard 预置，
        # 这里只断伤害与耗能，move 行为人工已验。）
        setup=CombatSetup(hand=('Headbutt',), clear_draw_pile=True),
        target_index=0,
        expect_enemy_hp_delta=9,
        expect_energy_delta=1,
        source='https://slaythespire.wiki.gg/wiki/Headbutt (9 dmg; put a discard card on draw top)',
    ),
    CardExpectation(
        card_id='Power Through',
        setup=CombatSetup(hand=('Power Through',), clear_draw_pile=True),
        target_index=-1,
        expect_player_block_delta=15,
        expect_energy_delta=1,
        expect_hand_contains={'Wound': 2},  # 2 张 Wound 直接进手牌
        source='https://slaythespire.wiki.gg/wiki/Power_Through (15 block + 2 Wound into hand)',
    ),
    CardExpectation(
        card_id='True Grit',
        # 手里另放 1 张可消耗牌，验 7 block + 随机消耗 1 张（消耗堆 size=1）。
        setup=CombatSetup(hand=('True Grit', 'Inflame'), clear_draw_pile=True),
        target_index=-1,
        expect_player_block_delta=7,
        expect_energy_delta=1,
        expect_exhaust_pile_size=1,  # 随机消耗手牌 1 张
        source='https://slaythespire.wiki.gg/wiki/True_Grit (7 block + exhaust 1 random hand card)',
    ),
    CardExpectation(
        card_id='Burning Pact',
        # 手里另放 1 张可消耗牌；抽堆放 3 张，验 exhaust 1 + draw 2。
        setup=CombatSetup(
            hand=('Burning Pact', 'Inflame'),
            draw_pile=('Strike_R', 'Strike_R', 'Strike_R'),
            clear_draw_pile=True,
        ),
        target_index=-1,
        expect_energy_delta=1,
        expect_exhaust_pile_contains={'Inflame': 1},  # 手里那张被消耗
        expect_hand_contains={'Strike': 2},           # 抽 2 张
        source='https://slaythespire.wiki.gg/wiki/Burning_Pact (exhaust 1, draw 2)',
    ),
    CardExpectation(
        card_id='Offering',
        setup=CombatSetup(
            hand=('Offering',),
            draw_pile=('Strike_R', 'Strike_R', 'Strike_R'),
            clear_draw_pile=True,
        ),
        target_index=-1,
        expect_energy_delta=-2,  # 净 +2 能量
        expect_hand_contains={'Strike': 3},  # 抽 3 张
        expect_exhaust_pile_contains={'Offering': 1},
        source='https://slaythespire.wiki.gg/wiki/Offering (gain 2 energy, draw 3, exhaust)',
    ),
    CardExpectation(
        card_id='Master of Strategy',
        setup=CombatSetup(
            hand=('Master of Strategy',),
            draw_pile=('Strike_R', 'Strike_R', 'Strike_R'),
            clear_draw_pile=True,
        ),
        target_index=-1,
        expect_energy_delta=0,
        expect_hand_contains={'Strike': 3},  # 抽 3 张
        expect_exhaust_pile_contains={'Master of Strategy': 1},
        source='https://slaythespire.wiki.gg/wiki/Master_of_Strategy (draw 3, exhaust)',
    ),
    CardExpectation(
        card_id='Jack Of All Trades',
        setup=CombatSetup(hand=('Jack Of All Trades',), clear_draw_pile=True),
        target_index=-1,
        expect_energy_delta=0,
        expect_hand_size=1,  # 打出 1（-1）+ 生成 1 张 Colorless（+1） = 净 1
        expect_exhaust_pile_contains={'Jack of All Trades': 1},
        source='https://slaythespire.wiki.gg/wiki/Jack_of_All_Trades (add 1 random colorless to hand, exhaust)',
    ),

    # ---- deck_composition ----
    CardExpectation(
        card_id='Clash',
        label='Clash (all-attack hand, playable)',
        setup=CombatSetup(hand=('Clash',), clear_draw_pile=True),
        target_index=0,
        expect_enemy_hp_delta=14,
        expect_energy_delta=0,
        source='https://slaythespire.wiki.gg/wiki/Clash (14 dmg; only playable if hand all attacks)',
    ),
    CardExpectation(
        card_id='Clash',
        label='Clash (non-attack in hand -> unplayable)',
        setup=CombatSetup(hand=('Clash', 'Defend_R'), clear_draw_pile=True),
        target_index=0,
        expect_unplayable=True,    # 手里有非攻击牌 → 引擎应判为不可打出
        expect_enemy_hp_delta=0,   # 打不出 → 敌人不掉血
        expect_energy_delta=0,
        source='https://slaythespire.wiki.gg/wiki/Clash (unplayable when a non-attack is in hand)',
    ),
    CardExpectation(
        card_id='Perfected Strike',
        label='Perfected Strike (3 Strike total -> 6+2*3=12)',
        # 抽堆放 2 张 Strike，手里 1 张 Perfected Strike（也算 Strike 牌）= 3 张 Strike。
        setup=CombatSetup(
            hand=('Perfected Strike',),
            draw_pile=('Strike_R', 'Strike_R'),
            clear_draw_pile=True,
        ),
        target_index=0,
        expect_enemy_hp_delta=12,  # 6 + 2*3
        expect_energy_delta=2,
        source='https://slaythespire.wiki.gg/wiki/Perfected_Strike (6 + 2 per Strike card incl self)',
    ),
    CardExpectation(
        card_id='Perfected Strike',
        label='Perfected Strike (alone -> 6+2*1=8)',
        setup=CombatSetup(hand=('Perfected Strike',), clear_draw_pile=True),
        target_index=0,
        expect_enemy_hp_delta=8,  # 6 + 2*1（只有自身一张 Strike 牌）
        expect_energy_delta=2,
        source='https://slaythespire.wiki.gg/wiki/Perfected_Strike (self counts as 1 Strike)',
    ),
    CardExpectation(
        card_id='Sever Soul',
        # 手里放 2 张非攻击牌，验 16 伤 + 那 2 张进消耗堆。
        setup=CombatSetup(hand=('Sever Soul', 'Defend_R', 'Inflame'), clear_draw_pile=True),
        target_index=0,
        expect_enemy_hp_delta=16,
        expect_energy_delta=2,
        expect_exhaust_pile_size=2,  # 2 张非攻击牌被消耗
        source='https://slaythespire.wiki.gg/wiki/Sever_Soul (16 dmg + exhaust all non-attacks in hand)',
    ),
    CardExpectation(
        card_id='Fiend Fire',
        # 手里另放 3 张牌 -> 全消耗，7*3=21 伤。
        setup=CombatSetup(
            hand=('Fiend Fire', 'Defend_R', 'Defend_R', 'Defend_R'),
            clear_draw_pile=True,
        ),
        target_index=0,
        expect_enemy_hp_delta=21,  # 7 * 3 张被消耗
        expect_energy_delta=2,
        expect_exhaust_pile_size=4,  # 3 张其它 + Fiend Fire 自身
        source='https://slaythespire.wiki.gg/wiki/Fiend_Fire (exhaust hand, 7 per card exhausted)',
    ),
    CardExpectation(
        card_id='Havoc',
        # 抽堆顶放 1 张 Strike，验自动打出（6 伤）+ 该 Strike 被消耗。
        setup=CombatSetup(hand=('Havoc',), draw_pile=('Strike_R',), clear_draw_pile=True),
        target_index=0,
        expect_enemy_hp_delta=6,  # 顶牌 Strike 造成 6
        expect_energy_delta=1,
        expect_exhaust_pile_contains={'Strike': 1},  # 顶牌打完被消耗
        source='https://slaythespire.wiki.gg/wiki/Havoc (play top draw card, exhaust it)',
    ),
    CardExpectation(
        card_id='Second Wind',
        # 手里放 2 张非攻击牌（每张 5 block）-> 10 block。
        setup=CombatSetup(hand=('Second Wind', 'Defend_R', 'Inflame'), clear_draw_pile=True),
        target_index=-1,
        expect_player_block_delta=10,  # 2 张非攻击 * 5
        expect_energy_delta=1,
        expect_exhaust_pile_size=2,
        source='https://slaythespire.wiki.gg/wiki/Second_Wind (exhaust non-attacks, 5 block each)',
    ),
    CardExpectation(
        card_id='Dual Wield',
        # 手里放 1 张 Strike，验复制 -> 手里 2 张 Strike。
        setup=CombatSetup(hand=('Dual Wield', 'Strike_R'), clear_draw_pile=True),
        target_index=-1,
        expect_energy_delta=1,
        expect_hand_contains={'Strike': 2},  # 原 1 + 复制 1
        source='https://slaythespire.wiki.gg/wiki/Dual_Wield (copy an Attack/Power in hand)',
    ),
    CardExpectation(
        card_id='Infernal Blade',
        setup=CombatSetup(hand=('Infernal Blade',), clear_draw_pile=True),
        target_index=-1,
        expect_energy_delta=1,
        expect_hand_size=1,  # 生成 1 张随机攻击进手牌（打出 -1 + 生成 +1 = 1）
        expect_exhaust_pile_contains={'Infernal Blade': 1},
        source='https://slaythespire.wiki.gg/wiki/Infernal_Blade (add random Attack to hand, exhaust)',
    ),

    # ---- x_cost ----
    CardExpectation(
        card_id='Whirlwind',
        label='Whirlwind X=3 (5*3=15 single enemy)',
        setup=CombatSetup(hand=('Whirlwind',), energy=3, clear_draw_pile=True),
        target_index=0,
        expect_enemy_hp_delta=15,  # 5 * X(=3)
        expect_energy_delta=3,     # 吃光 3 能量
        source='https://slaythespire.wiki.gg/wiki/Whirlwind (5 dmg x X to all; X=current energy)',
    ),
    CardExpectation(
        card_id='Whirlwind',
        label='Whirlwind X=2 (5*2=10)',
        setup=CombatSetup(hand=('Whirlwind',), energy=2, clear_draw_pile=True),
        target_index=0,
        expect_enemy_hp_delta=10,  # 5 * X(=2)，验随能量缩放
        expect_energy_delta=2,
        source='https://slaythespire.wiki.gg/wiki/Whirlwind (scales with energy: X=2 -> 10)',
    ),
    CardExpectation(
        card_id='Whirlwind',
        label='Whirlwind AoE (3 enemies, X=3)',
        setup=CombatSetup(hand=('Whirlwind',), energy=3, enemy_count=3, clear_draw_pile=True),
        target_index=0,
        expect_all_enemies_hp_delta=15,  # 全体各 5*3
        source='https://slaythespire.wiki.gg/wiki/Whirlwind (AoE: every enemy 5*X)',
    ),
    CardExpectation(
        card_id='Transmutation',
        label='Transmutation X=3 (add 3 colorless to hand)',
        setup=CombatSetup(hand=('Transmutation',), energy=3, clear_draw_pile=True),
        target_index=-1,
        expect_energy_delta=3,  # 吃光能量
        expect_hand_size=3,     # 加 X=3 张随机无色进手牌（打出 -1 + 3 = ... 见下）
        expect_exhaust_pile_contains={'Transmutation': 1},
        source='https://slaythespire.wiki.gg/wiki/Transmutation (add X random colorless to hand, exhaust)',
    ),

    # ---- multi_play ----
    CardExpectation(
        card_id='Rampage',
        label='Rampage 1st play (base 8)',
        setup=CombatSetup(hand=('Rampage',), clear_draw_pile=True),
        target_index=0,
        expect_enemy_hp_delta=8,  # 首次打 = 基础 8（递增需连打，单卡探针只测首次）
        expect_energy_delta=1,
        source='https://slaythespire.wiki.gg/wiki/Rampage (base 8; +5 each subsequent play this combat)',
    ),

    # ---- other ----
    CardExpectation(
        card_id='HandOfGreed',
        setup=CombatSetup(hand=('HandOfGreed',), clear_draw_pile=True),
        target_index=0,
        expect_enemy_hp_delta=20,
        expect_energy_delta=2,
        source='https://slaythespire.wiki.gg/wiki/Hand_of_Greed (20 dmg; +20 gold if fatal)',
    ),
    CardExpectation(
        card_id='Seeing Red',
        label='Seeing Red (net +1 energy, exhaust)',
        setup=CombatSetup(hand=('Seeing Red',), energy=1, clear_draw_pile=True),
        target_index=-1,
        expect_energy_delta=-1,  # 花 1 得 2 = 净 +1
        source='https://slaythespire.wiki.gg/wiki/Seeing_Red (gain 2 energy, exhaust)',
    ),
    CardExpectation(
        card_id='Bandage Up',
        setup=CombatSetup(hand=('Bandage Up',), player_hp=40, clear_draw_pile=True),
        target_index=-1,
        expect_enemy_hp_delta=0,
        expect_energy_delta=0,
        expect_exhaust_pile_contains={'Bandage Up': 1},  # 治疗 4 + 自身消耗
        source='https://slaythespire.wiki.gg/wiki/Bandage_Up (heal 4, exhaust)',
    ),
]

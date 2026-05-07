"""
V8 Stage A.2 — Ironclad 教师评估器（移植自 bottled_ai）。

移植自 bottled_ai (MIT 协议, https://github.com/xaved88/bottled_ai)
源 commit: 7b47a877d626cdb4fbeeb423c46472b15dd2ac00
源路径:
    rs/common/comparators/core/assessment.py        → ComparatorAssessment
    rs/common/comparators/core/comparisons.py       → 48 个比较函数
    rs/common/comparators/common_general_comparator.py → CommonGeneralComparator

修改要点
--------
1. **去掉 BattleState/CardId/PowerId/MemoryItem enum 依赖**：
   原代码读 `state.player.powers.get(PowerId.VULNERABLE, 0)`，移植后读
   `state.player.powers.get("VULNERABLE", 0)`（PowerSet 适配两种风格）。
2. **职业剪枝**：明显是 Watcher / Defect / Silent 专属的 comparison /
   assessment 方法（涉及 stance / orb / shiv / mantra / panache / ritual
   dagger / genetic algorithm / claw 等），改为 `_disabled_<name>(...)`
   或返回 0/None，不进入 Ironclad 默认 comparison list。
3. **memory 字段**：触达 `state.memory_general[...]` / `state.memory_by_card`
   会被 adapter 直接 raise NotImplementedError —— 我们在调用前就
   stub 掉，**不要** silent fallback。
4. **adapter 字段**：见 v8_battle_state_adapter.py「已实现/未实现」清单。

本文件不 import bottled_ai；移植完成后必须自包含。
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional

from v8_battle_state_adapter import BottledStateAdapter
from v8_teacher_ironclad import (
    POWERS_WE_LIKE,
    POWERS_WE_LIKE_LESS,
    POWERS_WE_DISLIKE,
    CARDS_THAT_EXIT_WRATH,
)


# ============================================================================
# § 1 配置类
# ============================================================================

class ComparatorAssessmentConfig:
    """对应 bottled_ai 的 ComparatorAssessmentConfig。"""

    def __init__(
        self,
        powers_we_like: List[str],
        powers_we_like_less: List[str],
        powers_we_dislike: List[str],
        powers_we_love: Optional[List[str]] = None,
        cards_that_exit_wrath: Optional[List[str]] = None,
    ):
        self.powers_we_love: List[str] = [] if powers_we_love is None else powers_we_love
        self.powers_we_like: List[str] = powers_we_like
        self.powers_we_like_less: List[str] = powers_we_like_less
        self.powers_we_dislike: List[str] = powers_we_dislike
        self.cards_that_exit_wrath: List[str] = (
            [] if cards_that_exit_wrath is None else cards_that_exit_wrath
        )


def _power_count(power_set: Any, desired: List[str]) -> int:
    """对应 bottled_ai 的 get_power_count。"""
    total = 0
    for name in desired:
        total += power_set.get(name, 0)
    return total


# ============================================================================
# § 2 ComparatorAssessment（CA）
# ----------------------------------------------------------------------------
# 字段名约定与 bottled_ai 完全一致，方便 diff。Watcher/Defect 专属字段
# 用 _disabled_* 前缀替换并立即返回中性值（0 / False）。
# ============================================================================

# 卡 ID 常量（bottled_ai 源里是 CardId.X，移植后用我们的 PascalCase 字符串）
_CARD_SHIV = "Shiv"
_CARD_RITUAL_DAGGER = "RitualDagger"          # Ironclad 不用，留 stub
_CARD_GENETIC_ALGORITHM = "GeneticAlgorithm"  # Defect，stub
_CARD_STEAM_BARRIER = "SteamBarrier"          # Defect，stub
_CARD_TRANQUILITY = "Tranquility"             # Watcher，stub
_CARD_CRESCENDO = "Crescendo"                 # Watcher，stub
_CARD_SANDS_OF_TIME = "SandsOfTime"

# Power 常量
_POW_UNAWAKENED = "UNAWAKENED"
_POW_VULNERABLE = "VULNERABLE"
_POW_WEAKENED = "WEAKENED"
_POW_BLOCK_RETURN = "BLOCK_RETURN"
_POW_INTANGIBLE_PLAYER = "INTANGIBLE_PLAYER"
_POW_BARRICADE = "BARRICADE"
_POW_ARTIFACT = "ARTIFACT"
_POW_PLATED_ARMOR = "PLATED_ARMOR"
_POW_STRENGTH = "STRENGTH"
_POW_ANGER_NOB = "ANGER_NOB"
_POW_REPAIR = "REPAIR"            # Ironclad 也不常用，但保留
_POW_BIAS = "BIAS"
_POW_BLASPHEMER = "BLASPHEMER"    # Watcher
_POW_TIME_WARP = "TIME_WARP"

# Relic 常量（仅评估器用到的几个）
_RELIC_PEN_NIB = "PEN_NIB"
_RELIC_NUNCHAKU = "NUNCHAKU"
_RELIC_INK_BOTTLE = "INK_BOTTLE"
_RELIC_LIZARD_TAIL = "LIZARD_TAIL"

# Potion 常量
_POTION_FAIRY = "FAIRY_IN_A_BOTTLE"

# CardType 字符串（StSRLSolver CardType enum 的 name 大概率是 "CURSE"/"STATUS"）
_CARD_TYPE_CURSE = "CURSE"
_CARD_TYPE_STATUS = "STATUS"


def _card_type_name(card: Any) -> str:
    """从 CardAdapter.type 拿到大写名字，兼容 enum 与字符串。"""
    t = card.type
    name = getattr(t, "name", None)
    if name is not None:
        return name
    return str(t).upper()


class ComparatorAssessment:
    """对应 bottled_ai 的 ComparatorAssessment。"""

    def __init__(
        self,
        state: BottledStateAdapter,
        original: BottledStateAdapter,
        config: ComparatorAssessmentConfig,
    ):
        self.state: BottledStateAdapter = state
        self.original: BottledStateAdapter = original
        self.cached_values: dict = {}
        self.config: ComparatorAssessmentConfig = config

    def _cache(self, name: str, fn: Callable[[], Any]) -> Any:
        if self.cached_values.get(name) is None:
            self.cached_values[name] = fn()
        return self.cached_values[name]

    # ---- 战斗胜负 ----------------------------------------------------------

    def battle_won(self) -> bool:
        def _calc():
            alive_monsters = False
            unawakened_present = False
            for mon in self.state.monsters:
                if mon.current_hp > 0:
                    alive_monsters = True
                if mon.powers.get(_POW_UNAWAKENED, 0):
                    unawakened_present = True
            return (not alive_monsters) and (not unawakened_present)
        return self._cache("bw", _calc)

    def battle_lost(self) -> bool:
        return self._cache("bl", lambda: self.state.player.current_hp <= 0)

    # ---- 伤害 / 玩家 -------------------------------------------------------

    def incoming_damage(self) -> int:
        return self._cache(
            "id",
            lambda: self.original.player.current_hp - self.state.player.current_hp,
        )

    def player_max_hp(self) -> int:
        return self._cache("pmhp", lambda: self.state.player.max_hp)

    def energy(self) -> int:
        return self._cache("e", lambda: self.state.player.energy)

    def intangible(self) -> int:
        return self._cache(
            "i", lambda: self.state.player.powers.get(_POW_INTANGIBLE_PLAYER, 0)
        )

    # ---- 怪物 --------------------------------------------------------------

    def dead_monsters(self) -> int:
        return self._cache(
            "dm",
            lambda: len([1 for m in self.state.monsters if m.current_hp <= 0]),
        )

    def dead_edge_monsters(self) -> int:
        def _calc():
            if self.battle_won():
                return 0
            mons = self.state.monsters
            # bottled_ai 假设至少 3 个怪；少于 3 个时退化为只看首尾
            if len(mons) == 0:
                return 0
            first_dead = mons[0].current_hp <= 0
            if len(mons) >= 3:
                last_dead = mons[2].current_hp <= 0
            else:
                last_dead = mons[-1].current_hp <= 0
            return int(first_dead or last_dead)
        return self._cache("dem", _calc)

    def monsters_vulnerable_hp(self) -> List[int]:
        def _calc():
            out = [
                m.current_hp - min(m.powers.get(_POW_VULNERABLE, 0) * 5, 3)
                for m in self.state.monsters if m.current_hp > 0
            ]
            return out or [0]
        return self._cache("mvhp", _calc)

    def lowest_health_monster(self) -> int:
        return self._cache(
            "lhm",
            lambda: 0 if self.battle_won() else min(self.monsters_vulnerable_hp()),
        )

    def lowest_true_health_monster(self) -> int:
        return self._cache(
            "lowest_true_health_monster",
            lambda: (
                0 if self.battle_won()
                else min(m.current_hp for m in self.state.monsters)
            ),
        )

    def lowest_health_edge_monster(self) -> int:
        def _calc():
            if self.battle_won():
                return 0
            mvhp = self.monsters_vulnerable_hp()
            return min(mvhp[0], mvhp[-1])
        return self._cache("lowest_health_edge_monster", _calc)

    def total_monster_health(self) -> int:
        def _calc():
            if self.battle_won():
                return 0
            return (
                sum(self.monsters_vulnerable_hp())
                - self.state.total_random_damage_dealt
                - self.state.total_random_poison_added
            )
        return self._cache("tmh", _calc)

    def total_monster_health_percent(self) -> float:
        def _calc():
            if self.battle_won():
                return 0.0
            denom = sum(m.max_hp for m in self.state.monsters)
            if denom == 0:
                return 0.0
            return float(sum(m.current_hp for m in self.state.monsters)) / float(denom)
        return self._cache("total_monster_health_percent", _calc)

    # ---- draw 计数 ---------------------------------------------------------

    def draw_free_early(self) -> int:
        return self._cache("dfe", lambda: self.state.draw_free_early)

    def draw_free(self) -> int:
        return self._cache(
            "df", lambda: self.state.draw_free + self.state.draw_free_early
        )

    def draw_pay_early(self) -> int:
        return self._cache("dpe", lambda: self.state.draw_pay_early)

    def draw_pay(self) -> int:
        return self._cache(
            "dp", lambda: self.state.draw_pay + self.state.draw_pay_early
        )

    # ---- 敌方状态 ----------------------------------------------------------

    def enemy_vulnerable(self) -> int:
        def _calc():
            mons = self.state.monsters
            if not mons:
                return 0
            return min(max(m.powers.get(_POW_VULNERABLE, 0) for m in mons), 4)
        return self._cache("ev", _calc)

    def enemy_weak(self) -> int:
        def _calc():
            mons = self.state.monsters
            if not mons:
                return 0
            return min(max(m.powers.get(_POW_WEAKENED, 0) for m in mons), 4)
        return self._cache("ew", _calc)

    def enemy_talking_to_hand(self) -> int:
        def _calc():
            mons = self.state.monsters
            if not mons:
                return 0
            return min(max(m.powers.get(_POW_BLOCK_RETURN, 0) for m in mons), 10)
        return self._cache("eh", _calc)

    def enemy_artifacts(self) -> int:
        return self._cache(
            "enemy_artifacts",
            lambda: sum(m.powers.get(_POW_ARTIFACT, 0) for m in self.state.monsters),
        )

    def enemy_plated_armor(self) -> int:
        return self._cache(
            "enemy_plated_armor",
            lambda: sum(
                m.powers.get(_POW_PLATED_ARMOR, 0) for m in self.state.monsters
                if m.powers.get(_POW_PLATED_ARMOR, 0) != 0
            ),
        )

    def barricaded_block(self) -> int:
        return self._cache(
            "barricaded_block",
            lambda: sum(
                m.block for m in self.state.monsters
                if m.powers.get(_POW_BARRICADE, 0) != 0
            ),
        )

    def nob_adjusted_scaling_damage(self) -> int:
        def _calc():
            anger_strength_up = sum(
                m.powers.get(_POW_STRENGTH, 0) for m in self.state.monsters
                if m.powers.get(_POW_ANGER_NOB, 0)
            )
            gremlin_nob_hp = sum(
                m.current_hp for m in self.state.monsters
                if m.powers.get(_POW_ANGER_NOB, 0)
            )
            return (
                self.original.player.current_hp - self.state.player.current_hp
                + (int(gremlin_nob_hp / 15) * anger_strength_up)
            )
        return self._cache("nob_adjusted_incoming_damage", _calc)

    # ---- 玩家 power 分组 ---------------------------------------------------

    def player_powers_great(self) -> int:
        return self._cache(
            "player_powers_great",
            lambda: _power_count(self.state.player.powers, self.config.powers_we_love),
        )

    def player_powers_good(self) -> int:
        return self._cache(
            "player_powers_good",
            lambda: _power_count(self.state.player.powers, self.config.powers_we_like),
        )

    def player_powers_less_good(self) -> int:
        return self._cache(
            "powers_less_good",
            lambda: _power_count(
                self.state.player.powers, self.config.powers_we_like_less
            ),
        )

    def player_powers_bad(self) -> int:
        return self._cache(
            "player_powers_bad",
            lambda: _power_count(
                self.state.player.powers, self.config.powers_we_dislike
            ),
        )

    def player_bias(self) -> int:
        return self._cache(
            "player_bias", lambda: self.state.player.powers.get(_POW_BIAS, 0)
        )

    # ---- 卡堆相关 ----------------------------------------------------------

    def bad_cards_exhausted(self) -> int:
        return self._cache(
            "bad_cards_exhausted",
            lambda: len([
                1 for c in self.state.exhaust_pile
                if _card_type_name(c) in (_CARD_TYPE_CURSE, _CARD_TYPE_STATUS)
            ]),
        )

    def ethereal_saved_for_later(self) -> int:
        return self._cache(
            "ethereal_saved_for_later",
            lambda: len([
                1 for c in self.state.discard_pile
                if c.ethereal
                and _card_type_name(c) not in (_CARD_TYPE_CURSE, _CARD_TYPE_STATUS)
            ]),
        )

    def cards_left_in_hand(self) -> int:
        return self._cache("cards_left_in_hand", lambda: len(self.state.hand))

    def count_expensive_cheapening_retain_cards(self) -> int:
        return self._cache(
            "cdst",
            lambda: len([
                1 for c in self.state.hand
                if c.id == _CARD_SANDS_OF_TIME and c.cost > 0
            ]),
        )

    # ---- relic 计数 --------------------------------------------------------

    def pen_nib_counter(self) -> int:
        return self._cache(
            "penc", lambda: self.state.relics.get(_RELIC_PEN_NIB, -1)
        )

    def nunchaku_counter(self) -> int:
        return self._cache(
            "nunc", lambda: self.state.relics.get(_RELIC_NUNCHAKU, -1)
        )

    def ink_bottle_counter(self) -> int:
        return self._cache(
            "ink_bottle_counter",
            lambda: self.state.relics.get(_RELIC_INK_BOTTLE, -1),
        )

    # ---- repair（Ironclad 也可能装 self-forming clay 之类，保留）-----------

    def repair_count(self) -> int:
        missing_hp = self.state.player.max_hp - self.state.player.current_hp
        if missing_hp >= 1:
            return self._cache(
                "repair_count",
                lambda: self.state.player.powers.get(_POW_REPAIR, 0),
            )
        return 0

    # ---- inconvenient time warp（Time Eater 行为）-------------------------

    def inconvenient_time_warp_count(self) -> int:
        def _calc():
            for m in self.state.monsters:
                tw = m.powers.get(_POW_TIME_WARP, 0)
                if tw == 10 or tw == 11:
                    return True
            return False
        return self._cache("nitwc", _calc)

    # ---- 复活选项（Lizard Tail / Fairy in a Bottle）------------------------

    def revive_option_count(self) -> int:
        def _calc():
            fairy_count = 0
            lizard_count = 0
            # potions 是 list（bottled_ai 风格）—— 数 fairy 的数量
            for p in self.state.potions:
                if p == _POTION_FAIRY:
                    fairy_count += 1
            # lizard tail：relics 字典里若有此 relic 且未消耗（!= -2）
            if self.state.relics.get(_RELIC_LIZARD_TAIL):
                if self.state.relics[_RELIC_LIZARD_TAIL] != -2:
                    lizard_count += 1
            return fairy_count + lizard_count
        return self._cache("roc", _calc)

    # ---- 下回合保留 block --------------------------------------------------

    def block_for_next_turn(self) -> int:
        return self._cache(
            "bfnt", lambda: self.state.saved_block_for_next_turn
        )

    # ============================================================================
    # § 2.X Watcher / Defect / Silent 专属 — Ironclad 不会触达 -----------------
    # ============================================================================

    # Defect 专属
    def orb_slot_count(self) -> int:                  # Defect 专属
        return 0

    def channeled_orb_count(self) -> int:             # Defect 专属
        return 0

    def power_up_genetic_algorithm(self) -> int:      # Defect 专属
        return 0

    def power_down_steam_barrier(self) -> int:        # Defect 专属
        return 0

    # Silent 专属
    def awkward_shivs(self) -> int:                   # Silent 专属
        # Ironclad 拿不到 Shiv，但 bottled_ai 的 default_comparisons 包含
        # least_awkward_shivs，所以保留实现（结果总是 0）。
        return self._cache(
            "awkward_shivs",
            lambda: (
                len([1 for c in self.state.hand if c.id == _CARD_SHIV])
                + len([1 for c in self.state.discard_pile if c.id == _CARD_SHIV])
            ),
        )

    def power_up_ritual_dagger(self) -> int:          # Silent 专属
        return 0

    # Watcher 专属
    def stance_is_calm(self) -> int:                  # Watcher 专属
        return 0

    def stance_is_not_wrath(self) -> int:             # Watcher 专属
        return 0

    def played_blasphemy(self) -> int:                # Watcher 专属
        return self._cache(
            "we_played_blasphemy_without_permission",
            lambda: 1 if self.state.player.powers.get(_POW_BLASPHEMER, 0) else 0,
        )

    def most_kills_with_lesson_learned(self) -> int:  # Watcher 专属
        return 0

    def count_tranquility(self) -> int:               # Watcher 专属
        return self._cache(
            "tinh",
            lambda: len([1 for c in self.state.hand if c.id == _CARD_TRANQUILITY]),
        )

    def count_crescendo(self) -> int:                 # Watcher 专属
        return self._cache(
            "cinh",
            lambda: len([1 for c in self.state.hand if c.id == _CARD_CRESCENDO]),
        )

    # 通用 / 跨职业未启用
    def powered_up_claws(self) -> int:                # Defect 专属（Claw 卡）
        return 0


# 别名：bottled_ai comparisons.py 用 `as CA`
CA = ComparatorAssessment


# ============================================================================
# § 3 Comparison 函数（48 个）
# ----------------------------------------------------------------------------
# 全部签名 (best: CA, challenger: CA) -> Optional[bool]
# 返回值含义：
#   True  → challenger 比 best 更好
#   False → best 比 challenger 更好（或同等）
#   None  → 此维度无差异，留给下一个 comparison 决定
#
# Ironclad 不用的 comparison 不进 default_comparisons 列表，但函数仍 port
# 一份（前缀 `_disabled_`）以便日后扩展时容易找回原型。
# ============================================================================


def battle_not_lost(best: CA, challenger: CA) -> Optional[bool]:
    return None if best.battle_lost() == challenger.battle_lost() \
        else (not challenger.battle_lost())


def battle_is_won(best: CA, challenger: CA) -> Optional[bool]:
    return None if best.battle_won() == challenger.battle_won() \
        else challenger.battle_won()


def most_optimal_winning_battle(best: CA, challenger: CA) -> Optional[bool]:
    if not best.battle_won() or not challenger.battle_won():
        return None

    if best.player_max_hp() != challenger.player_max_hp():
        return challenger.player_max_hp() > best.player_max_hp()

    if best.most_kills_with_lesson_learned() != challenger.most_kills_with_lesson_learned():
        return challenger.most_kills_with_lesson_learned() > best.most_kills_with_lesson_learned()

    if best.power_up_ritual_dagger() != challenger.power_up_ritual_dagger():
        return challenger.power_up_ritual_dagger() > best.power_up_ritual_dagger()

    if best.power_up_genetic_algorithm() != challenger.power_up_genetic_algorithm():
        return challenger.power_up_genetic_algorithm() > best.power_up_genetic_algorithm()

    if best.incoming_damage() != challenger.incoming_damage():
        return challenger.incoming_damage() < best.incoming_damage()

    if best.repair_count() != challenger.repair_count():
        return challenger.repair_count() > best.repair_count()

    if best.pen_nib_counter() != challenger.pen_nib_counter():
        return challenger.pen_nib_counter() > best.pen_nib_counter()

    if best.nunchaku_counter() != challenger.nunchaku_counter():
        return challenger.nunchaku_counter() > best.nunchaku_counter()

    if best.ink_bottle_counter() != challenger.ink_bottle_counter():
        return challenger.ink_bottle_counter() > best.ink_bottle_counter()

    if best.cards_left_in_hand() != challenger.cards_left_in_hand():
        return challenger.cards_left_in_hand() > best.cards_left_in_hand()

    if best.energy() != challenger.energy():
        return challenger.energy() > best.energy()

    return False


def most_free_early_draw(best: CA, challenger: CA) -> Optional[bool]:
    return None if best.draw_free_early() == challenger.draw_free_early() \
        else challenger.draw_free_early() > best.draw_free_early()


def most_free_draw(best: CA, challenger: CA) -> Optional[bool]:
    return None if best.draw_free() == challenger.draw_free() \
        else challenger.draw_free() > best.draw_free()


def most_lasting_intangible(best: CA, challenger: CA) -> Optional[bool]:
    return None if max(1, best.intangible()) == max(1, challenger.intangible()) \
        else challenger.intangible() > best.intangible()


def least_incoming_damage_over_1(best: CA, challenger: CA) -> Optional[bool]:
    return None if max(2, best.incoming_damage()) == max(2, challenger.incoming_damage()) \
        else challenger.incoming_damage() < best.incoming_damage()


def most_dead_monsters(best: CA, challenger: CA) -> Optional[bool]:
    return None if best.dead_monsters() == challenger.dead_monsters() \
        else challenger.dead_monsters() > best.dead_monsters()


def most_dead_edge_monsters(best: CA, challenger: CA) -> Optional[bool]:
    return None if best.dead_edge_monsters() == challenger.dead_edge_monsters() \
        else challenger.dead_edge_monsters() > best.dead_edge_monsters()


def most_enemy_vulnerable(best: CA, challenger: CA) -> Optional[bool]:
    return None if max(1, best.enemy_vulnerable()) == max(1, challenger.enemy_vulnerable()) \
        else challenger.enemy_vulnerable() > best.enemy_vulnerable()


def most_enemy_weak(best: CA, challenger: CA) -> Optional[bool]:
    return None if max(1, best.enemy_weak()) == max(1, challenger.enemy_weak()) \
        else challenger.enemy_weak() > best.enemy_weak()


def most_enemy_talking_to_hand(best: CA, challenger: CA) -> Optional[bool]:
    return None if best.enemy_talking_to_hand() == challenger.enemy_talking_to_hand() \
        else challenger.enemy_talking_to_hand() > best.enemy_talking_to_hand()


def lowest_health_monster(best: CA, challenger: CA) -> Optional[bool]:
    return None if best.lowest_health_monster() == challenger.lowest_health_monster() \
        else challenger.lowest_health_monster() < best.lowest_health_monster()


def highest_health_monster(best: CA, challenger: CA) -> Optional[bool]:
    return None if best.lowest_true_health_monster() == challenger.lowest_true_health_monster() \
        else challenger.lowest_true_health_monster() > best.lowest_true_health_monster()


def lowest_health_edge_monster(best: CA, challenger: CA) -> Optional[bool]:
    return None if best.lowest_health_edge_monster() == challenger.lowest_health_edge_monster() \
        else challenger.lowest_health_edge_monster() < best.lowest_health_edge_monster()


def lowest_total_monster_health(best: CA, challenger: CA) -> Optional[bool]:
    return None if best.total_monster_health() == challenger.total_monster_health() \
        else challenger.total_monster_health() < best.total_monster_health()


def lowest_barricaded_block(best: CA, challenger: CA) -> Optional[bool]:
    return None if best.barricaded_block() == challenger.barricaded_block() \
        else challenger.barricaded_block() < best.barricaded_block()


def most_draw_pay_early(best: CA, challenger: CA) -> Optional[bool]:
    return None if best.draw_pay_early() == challenger.draw_pay_early() \
        else challenger.draw_pay_early() > best.draw_pay_early()


def most_draw_pay(best: CA, challenger: CA) -> Optional[bool]:
    return None if best.draw_pay() == challenger.draw_pay() \
        else challenger.draw_pay() > best.draw_pay()


def most_good_player_powers(best: CA, challenger: CA) -> Optional[bool]:
    return None if best.player_powers_good() == challenger.player_powers_good() \
        else challenger.player_powers_good() > best.player_powers_good()


def most_great_player_powers(best: CA, challenger: CA) -> Optional[bool]:
    return None if best.player_powers_great() == challenger.player_powers_great() \
        else challenger.player_powers_great() > best.player_powers_great()


def most_less_good_player_powers(best: CA, challenger: CA) -> Optional[bool]:
    return None if best.player_powers_less_good() == challenger.player_powers_less_good() \
        else challenger.player_powers_less_good() > best.player_powers_less_good()


def least_bad_player_powers(best: CA, challenger: CA) -> Optional[bool]:
    return None if best.player_powers_bad() == challenger.player_powers_bad() \
        else challenger.player_powers_bad() < best.player_powers_bad()


def most_bad_cards_exhausted(best: CA, challenger: CA) -> Optional[bool]:
    return None if best.bad_cards_exhausted() == challenger.bad_cards_exhausted() \
        else challenger.bad_cards_exhausted() > best.bad_cards_exhausted()


def most_energy(best: CA, challenger: CA) -> Optional[bool]:
    return None if best.energy() == challenger.energy() \
        else challenger.energy() > best.energy()


def least_incoming_damage(best: CA, challenger: CA) -> Optional[bool]:
    return None if best.incoming_damage() == challenger.incoming_damage() \
        else challenger.incoming_damage() < best.incoming_damage()


def most_ethereal_cards_saved_for_later(best: CA, challenger: CA) -> Optional[bool]:
    return None if best.ethereal_saved_for_later() == challenger.ethereal_saved_for_later() \
        else challenger.ethereal_saved_for_later() > best.ethereal_saved_for_later()


def least_enemy_artifacts(best: CA, challenger: CA) -> Optional[bool]:
    return None if best.enemy_artifacts() == challenger.enemy_artifacts() \
        else challenger.enemy_artifacts() < best.enemy_artifacts()


def least_nob_adjusted_scaling_damage(best: CA, challenger: CA) -> Optional[bool]:
    return None if best.nob_adjusted_scaling_damage() == challenger.nob_adjusted_scaling_damage() \
        else challenger.nob_adjusted_scaling_damage() < best.nob_adjusted_scaling_damage()


def most_cards_left_in_hand(best: CA, challenger: CA) -> Optional[bool]:
    return None if best.cards_left_in_hand() == challenger.cards_left_in_hand() \
        else challenger.cards_left_in_hand() > best.cards_left_in_hand()


def lowest_enemy_plated_armor(best: CA, challenger: CA) -> Optional[bool]:
    return None if best.enemy_plated_armor() == challenger.enemy_plated_armor() \
        else challenger.enemy_plated_armor() < best.enemy_plated_armor()


def avoid_inconvenient_time_warp(best: CA, challenger: CA) -> Optional[bool]:
    return None if best.inconvenient_time_warp_count() == challenger.inconvenient_time_warp_count() \
        else challenger.inconvenient_time_warp_count() < best.inconvenient_time_warp_count()


def preserve_revive_options(best: CA, challenger: CA) -> Optional[bool]:
    return None if best.revive_option_count() == challenger.revive_option_count() \
        else challenger.revive_option_count() > best.revive_option_count()


def most_block_saved_for_next_turn(best: CA, challenger: CA) -> Optional[bool]:
    return None if best.block_for_next_turn() == challenger.block_for_next_turn() \
        else challenger.block_for_next_turn() > best.block_for_next_turn()


def kept_expensive_decreasing_cost_retain_cards(best: CA, challenger: CA) -> Optional[bool]:
    return None if best.count_expensive_cheapening_retain_cards() == challenger.count_expensive_cheapening_retain_cards() \
        else challenger.count_expensive_cheapening_retain_cards() > best.count_expensive_cheapening_retain_cards()


# -----------------------------------------------------------------------------
# 以下函数对 Ironclad 是恒等比较（assessment 已经返回 0），保留实现以便
# 日后扩展到其他职业时不用重新搬。它们不会被剪出 default_comparisons —
# 进了也只会一致返回 None。
# -----------------------------------------------------------------------------

def least_awkward_shivs(best: CA, challenger: CA) -> Optional[bool]:
    return None if best.awkward_shivs() == challenger.awkward_shivs() \
        else challenger.awkward_shivs() < best.awkward_shivs()


def killed_with_lesson_learned(best: CA, challenger: CA) -> Optional[bool]:
    # Watcher 专属。Ironclad 评估时两边都为 0 → 永远返回 None。
    return None if best.most_kills_with_lesson_learned() == challenger.most_kills_with_lesson_learned() \
        else challenger.most_kills_with_lesson_learned() > best.most_kills_with_lesson_learned()


def most_powered_up_ritual_dagger(best: CA, challenger: CA) -> Optional[bool]:
    return None if best.power_up_ritual_dagger() == challenger.power_up_ritual_dagger() \
        else challenger.power_up_ritual_dagger() > best.power_up_ritual_dagger()


def most_powered_up_genetic_algorithm(best: CA, challenger: CA) -> Optional[bool]:
    return None if best.power_up_genetic_algorithm() == challenger.power_up_genetic_algorithm() \
        else challenger.power_up_genetic_algorithm() > best.power_up_genetic_algorithm()


def least_powered_down_steam_barrier(best: CA, challenger: CA) -> Optional[bool]:
    return None if best.power_down_steam_barrier() == challenger.power_down_steam_barrier() \
        else challenger.power_down_steam_barrier() < best.power_down_steam_barrier()


def most_powered_up_claws(best: CA, challenger: CA) -> Optional[bool]:
    return None if best.powered_up_claws() == challenger.powered_up_claws() \
        else challenger.powered_up_claws() > best.powered_up_claws()


def stance_is_not_wrath(best: CA, challenger: CA) -> Optional[bool]:
    return None if best.stance_is_not_wrath() == challenger.stance_is_not_wrath() \
        else challenger.stance_is_not_wrath() < best.stance_is_not_wrath()


def stance_is_calm(best: CA, challenger: CA) -> Optional[bool]:
    return None if best.stance_is_calm() == challenger.stance_is_calm() \
        else challenger.stance_is_calm() > best.stance_is_calm()


def no_blasphemy(best: CA, challenger: CA) -> Optional[bool]:
    return None if best.played_blasphemy() == challenger.played_blasphemy() \
        else challenger.played_blasphemy() < best.played_blasphemy()


def most_tranquility(best: CA, challenger: CA) -> Optional[bool]:
    return None if best.count_tranquility() == challenger.count_tranquility() \
        else challenger.count_tranquility() > best.count_tranquility()


def most_crescendo(best: CA, challenger: CA) -> Optional[bool]:
    return None if best.count_crescendo() == challenger.count_crescendo() \
        else challenger.count_crescendo() > best.count_crescendo()


def most_orb_slots(best: CA, challenger: CA) -> Optional[bool]:
    return None if best.orb_slot_count() == challenger.orb_slot_count() \
        else challenger.orb_slot_count() > best.orb_slot_count()


def most_channeled_orbs(best: CA, challenger: CA) -> Optional[bool]:
    return None if best.channeled_orb_count() == challenger.channeled_orb_count() \
        else challenger.channeled_orb_count() > best.channeled_orb_count()


# ============================================================================
# § 4 默认 comparison 列表
# ----------------------------------------------------------------------------
# 与 bottled_ai 的 default_comparisons 顺序完全一致，便于 diff 与回归。
# ============================================================================

default_comparisons: List[Callable[[CA, CA], Optional[bool]]] = [
    battle_not_lost,
    battle_is_won,
    preserve_revive_options,
    most_optimal_winning_battle,
    no_blasphemy,
    most_free_early_draw,
    most_free_draw,
    most_lasting_intangible,
    least_incoming_damage_over_1,
    most_great_player_powers,
    most_dead_monsters,
    most_tranquility,
    most_enemy_talking_to_hand,
    most_enemy_vulnerable,
    most_enemy_weak,
    least_awkward_shivs,
    killed_with_lesson_learned,
    most_powered_up_ritual_dagger,
    kept_expensive_decreasing_cost_retain_cards,
    lowest_health_monster,
    lowest_total_monster_health,
    lowest_barricaded_block,
    lowest_enemy_plated_armor,
    most_orb_slots,
    most_channeled_orbs,
    most_draw_pay_early,
    most_draw_pay,
    most_good_player_powers,
    least_bad_player_powers,
    most_less_good_player_powers,
    least_enemy_artifacts,
    most_bad_cards_exhausted,
    most_powered_up_genetic_algorithm,
    most_cards_left_in_hand,
    least_incoming_damage,
    most_ethereal_cards_saved_for_later,
    most_powered_up_claws,
    stance_is_not_wrath,
    stance_is_calm,
    least_powered_down_steam_barrier,
    most_block_saved_for_next_turn,
    most_energy,
]


# ============================================================================
# § 5 CommonGeneralComparator
# ============================================================================

class CommonGeneralComparator:
    """对应 bottled_ai 的 CommonGeneralComparator。"""

    def __init__(
        self,
        comparisons: Optional[List[Callable[[CA, CA], Optional[bool]]]] = None,
        assessment_config: Optional[ComparatorAssessmentConfig] = None,
    ):
        self.comparisons = default_comparisons if comparisons is None else comparisons
        if assessment_config is None:
            assessment_config = ComparatorAssessmentConfig(
                powers_we_like=POWERS_WE_LIKE,
                powers_we_like_less=POWERS_WE_LIKE_LESS,
                powers_we_dislike=POWERS_WE_DISLIKE,
                cards_that_exit_wrath=CARDS_THAT_EXIT_WRATH,
            )
        self.assessment_config: ComparatorAssessmentConfig = assessment_config

    def does_challenger_defeat_the_best(
        self,
        best_state: BottledStateAdapter,
        challenger_state: BottledStateAdapter,
        original: BottledStateAdapter,
    ) -> bool:
        best = CA(best_state, original, self.assessment_config)
        challenger = CA(challenger_state, original, self.assessment_config)

        for c in self.comparisons:
            v = c(best, challenger)
            if v is not None:
                return v
        return False


__all__ = [
    "ComparatorAssessmentConfig",
    "ComparatorAssessment",
    "CA",
    "default_comparisons",
    "CommonGeneralComparator",
]

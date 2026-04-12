"""
STS 数据提取模块 — 从 StSRLSolver 读取卡牌/遗物/药水定义，生成统一效果向量。

用途：为 RL agent 提供结构化的实体编码，替代纯 bloom filter hash。
每个实体映射到一个 UNIFIED_DIM 维的 float32 向量，包含标准化的数值属性和效果标志。

公共 API:
    encode_entity(entity_type, entity_id, upgraded=False) → np.ndarray
    CARD_DATA, RELIC_DATA, POTION_DATA — 预计算的向量字典
    UNIFIED_DIM — 向量总维度
    DIM_NAMES — 维度名列表（调试用）
"""

import sys
import numpy as np
from typing import Dict, Optional, Tuple

# ============================================================================
# 将 StSRLSolver 加入 sys.path，以便 import 其内容模块
# ============================================================================
_SOLVER_ROOT = "STSRLSOLVER_PATH"
if _SOLVER_ROOT not in sys.path:
    sys.path.insert(0, _SOLVER_ROOT)

from packages.engine.content.cards import (
    ALL_CARDS, Card, CardType, CardRarity, CardTarget, CardColor,
    resolve_card_id, CARD_ID_ALIASES,
)
from packages.engine.content.relics import ALL_RELICS, Relic, RelicTier
from packages.engine.content.potions import ALL_POTIONS, Potion, PotionRarity, PotionTargetType

# ============================================================================
# 统一效果向量维度定义（~40 维）
# ============================================================================
DIM_NAMES = [
    # --- 基础属性 (0-9) ---
    "cost",                # 0: 能量费用（/4 归一化）
    "is_attack",           # 1: 攻击牌
    "is_skill",            # 2: 技能牌
    "is_power",            # 3: 能力牌
    "is_status",           # 4: 状态牌
    "is_curse",            # 5: 诅咒牌
    "rarity_common",       # 6: 普通稀有度
    "rarity_uncommon",     # 7: 非普通稀有度
    "rarity_rare",         # 8: 稀有稀有度
    "is_upgraded",         # 9: 是否升级

    # --- 旗标属性 (10-14) ---
    "exhaust",             # 10: 消耗
    "ethereal",            # 11: 虚无
    "innate",              # 12: 固有
    "retain",              # 13: 保留
    "x_cost",              # 14: X 费用（cost == -1）

    # --- 攻击 (15-19) ---
    "damage_single",       # 15: 单体伤害（/40 归一化）
    "damage_aoe",          # 16: 群体伤害（/40 归一化）
    "multi_hit",           # 17: 多段攻击次数（/10 归一化）
    "poison_apply",        # 18: 施加中毒层数（/12 归一化）
    "execute_threshold",   # 19: 斩杀阈值（/50 归一化）

    # --- 防御 (20-23) ---
    "block_gain",          # 20: 格挡值（/30 归一化）
    "hp_heal",             # 21: 治疗量（/20 归一化）
    "max_hp_change",       # 22: 最大生命变化（/10 归一化）
    "damage_reduction",    # 23: 减伤（保留）

    # --- 减益 (24-26) ---
    "vulnerable_apply",    # 24: 施加易伤（/5 归一化）
    "weak_apply",          # 25: 施加虚弱（/5 归一化）
    "frail_apply",         # 26: 施加脆弱（/5 归一化）

    # --- 增益 (27-35) ---
    "strength_gain",       # 27: 力量获取（/5 归一化）
    "dexterity_gain",      # 28: 敏捷获取（/5 归一化）
    "plated_armor",        # 29: 多层护甲（/5 归一化）
    "metallicize",         # 30: 金属化（/10 归一化）
    "thorns",              # 31: 荆棘（/5 归一化）
    "artifact",            # 32: 人工制品（/3 归一化）
    "intangible",          # 33: 无实体（/3 归一化）
    "ritual",              # 34: 仪式（每回合力量）（/3 归一化）
    "regen",               # 35: 再生（/10 归一化）

    # --- 循环/资源 (36-42) ---
    "card_draw",           # 36: 抽牌（/5 归一化）
    "energy_gain",         # 37: 能量获取（/3 归一化）
    "cost_reduction",      # 38: 费用减免
    "exhaust_other",       # 39: 消耗其他牌
    "card_generation",     # 40: 生成牌到手牌/抽牌堆
    "scry",                # 41: 预见（/5 归一化）
    "discard",             # 42: 弃牌（/5 归一化）

    # --- 自损 (43-44) ---
    "self_damage",         # 43: 自伤（/10 归一化）
    "hp_cost",             # 44: 生命消耗

    # --- 架势/Orb (45-53) ---
    "stance_wrath",        # 45: 进入愤怒架势
    "stance_calm",         # 46: 进入冷静架势
    "stance_exit",         # 47: 退出架势
    "orb_lightning",       # 48: 引导闪电球（/3 归一化）
    "orb_frost",           # 49: 引导冰霜球（/3 归一化）
    "orb_dark",            # 50: 引导暗球（/2 归一化）
    "focus_gain",          # 51: 专注获取（/4 归一化）
    "mantra",              # 52: 真言获取（/5 归一化）
    "orb_slots",           # 53: 球槽增加（/3 归一化）

    # --- 资源/其他 (54-56) ---
    "gold_gain",           # 54: 金币获取（/50 归一化）
    "is_aoe",              # 55: 是否群体（冗余标记，便于索引）
    "energy_bonus_permanent",  # 56: 永久能量加成（遗物用）
]

UNIFIED_DIM = len(DIM_NAMES)

# 维度名 → 索引的快速查找
_DIM_INDEX = {name: i for i, name in enumerate(DIM_NAMES)}


# ============================================================================
# magic_number 消歧义：根据 effects 列表判断 base_magic 对应哪个维度
# ============================================================================

# 优先级从高到低的关键词映射
_MAGIC_DISPATCH = [
    # (关键词子串, 目标维度名, 归一化除数)
    ("poison", "poison_apply", 12.0),
    ("mark", "poison_apply", 12.0),  # Pressure Points 的 mark 类似 poison
    ("vulnerable", "vulnerable_apply", 5.0),
    ("weak", "weak_apply", 5.0),
    ("frail", "frail_apply", 5.0),
    ("scry", "scry", 5.0),
    ("draw", "card_draw", 5.0),
    ("mantra", "mantra", 5.0),
    ("focus", "focus_gain", 4.0),
    ("strength", "strength_gain", 5.0),
    ("dexterity", "dexterity_gain", 5.0),
    ("plated_armor", "plated_armor", 5.0),
    ("artifact", "artifact", 3.0),
    ("intangible", "intangible", 3.0),
    ("damage_x_times", "multi_hit", 10.0),
    ("hits_x_times", "multi_hit", 10.0),
    ("x_times", "multi_hit", 10.0),
    ("twice", "multi_hit", 10.0),
    ("channel_lightning", "orb_lightning", 3.0),
    ("channel_frost", "orb_frost", 3.0),
    ("channel_dark", "orb_dark", 2.0),
    ("channel_random", "orb_lightning", 3.0),  # 近似
    ("channel_plasma", "energy_gain", 3.0),
    ("orb_slot", "orb_slots", 3.0),
    ("increase_orb", "orb_slots", 3.0),
    ("energy", "energy_gain", 3.0),
    ("heal", "hp_heal", 20.0),
    ("block", "block_gain", 30.0),
    ("shiv", "card_generation", 5.0),
    ("lockon", "vulnerable_apply", 5.0),  # Lock-On 类似 vulnerable
    ("choke", "damage_single", 40.0),
    ("ritual", "ritual", 3.0),
    ("regen", "regen", 10.0),
    ("prevent_next_hp_loss", "intangible", 3.0),
    ("gold", "gold_gain", 50.0),
    ("reduce_cost", "cost_reduction", 1.0),
    ("cost_reduces", "cost_reduction", 1.0),
    ("exhaust", "exhaust_other", 1.0),
    ("discard", "discard", 5.0),
    ("if_fatal", "execute_threshold", 50.0),
]


def _resolve_magic(card: Card, upgraded: bool = False) -> Tuple[Optional[str], float]:
    """
    根据卡牌效果列表判断 base_magic 应该映射到哪个维度。
    返回 (维度名, 归一化后的值)。如果无法判断或 base_magic < 0，返回 (None, 0)。
    """
    mag = card.base_magic
    if mag < 0:
        return None, 0.0
    if upgraded:
        mag += card.upgrade_magic

    effects_str = " ".join(card.effects).lower()

    for keyword, dim_name, norm in _MAGIC_DISPATCH:
        if keyword in effects_str:
            return dim_name, mag / norm

    # 没匹配到 → 尝试根据卡牌类型推测
    if card.card_type == CardType.ATTACK:
        return "multi_hit", mag / 10.0
    return None, 0.0


# ============================================================================
# 卡牌效果 → 维度的额外映射（非 magic_number 的效果）
# ============================================================================

def _parse_card_effects(card: Card, vec: np.ndarray, upgraded: bool = False):
    """
    扫描卡牌 effects 列表，设置对应维度。
    这里处理的是 magic_number 之外的效果标志。
    """
    effects_str = " ".join(card.effects).lower()

    # 架势
    if card.enter_stance:
        stance = card.enter_stance.lower()
        if stance == "wrath":
            vec[_DIM_INDEX["stance_wrath"]] = 1.0
        elif stance == "calm":
            vec[_DIM_INDEX["stance_calm"]] = 1.0
    if card.exit_stance:
        vec[_DIM_INDEX["stance_exit"]] = 1.0

    # AOE 标记
    if card.target == CardTarget.ALL_ENEMY:
        vec[_DIM_INDEX["is_aoe"]] = 1.0

    # 效果关键词扫描（设置标志位，不覆盖已有数值）
    effect_flags = {
        "exhaust_all": ("exhaust_other", 1.0),
        "exhaust_non_attack": ("exhaust_other", 1.0),
        "exhaust_card": ("exhaust_other", 1.0),
        "exhaust_random": ("exhaust_other", 0.5),
        "skills_cost_0": ("cost_reduction", 1.0),
        "cost_0": ("cost_reduction", 0.5),
        "reduce_cost": ("cost_reduction", 0.5),
        "add_": ("card_generation", 0.5),
        "shuffle_": ("card_generation", 0.3),
        "discover": ("card_generation", 0.5),
        "draw_1": ("card_draw", 0.2),
        "draw_2": ("card_draw", 0.4),
        "draw_cards": ("card_draw", 0.6),
        "draw_until": ("card_draw", 1.0),
        "lose_hp": ("self_damage", 0.3),
        "self_damage": ("self_damage", 0.5),
        "gain_energy": ("energy_gain", 0.33),
        "gain_1_energy": ("energy_gain", 0.33),
        "gain_2_energy": ("energy_gain", 0.67),
        "double_energy": ("energy_gain", 1.0),
        "block_not_lost": ("damage_reduction", 1.0),
        "end_turn": ("self_damage", 0.2),  # 结束回合视为轻微负面
        "play_twice": ("card_generation", 0.7),
        "play_attacks_twice": ("card_generation", 0.8),
        "damage_twice": ("multi_hit", 0.2),
        "damage_per_enemy": ("damage_aoe", 0.3),
        "damage_all": ("is_aoe", 1.0),
        "return_exhausted": ("card_generation", 0.5),
        "upgrade_all": ("card_generation", 0.3),
        "unplayable": ("self_damage", 0.5),  # 不可打出的牌视为负面
        "double_damage": ("damage_single", 0.5),
        "double_block": ("block_gain", 0.5),
        "double_strength": ("strength_gain", 1.0),
        "gain_strength_each_turn": ("ritual", 0.5),
        "take_extra_turn": ("energy_gain", 1.0),
        "enter_divinity": ("mantra", 1.0),
        "channel_lightning": ("orb_lightning", 0.33),
        "channel_frost": ("orb_frost", 0.33),
        "channel_dark": ("orb_dark", 0.5),
        "evoke": ("orb_lightning", 0.2),
        "block_return": ("block_gain", 0.3),
        "gain_block_per": ("block_gain", 0.3),
        "thorns": ("thorns", 0.3),
        "intangible": ("intangible", 0.5),
    }

    for keyword, (dim_name, value) in effect_flags.items():
        if keyword in effects_str:
            idx = _DIM_INDEX[dim_name]
            vec[idx] = max(vec[idx], value)  # 取最大值，不覆盖更高的数值


# ============================================================================
# 加载函数
# ============================================================================

def _load_cards() -> Dict[str, np.ndarray]:
    """
    加载所有卡牌，生成统一向量。
    每张牌生成两个 key：base_id（未升级）和 base_id+（升级版）。
    """
    result = {}

    for card_id, card in ALL_CARDS.items():
        for upgraded in [False, True]:
            key = card_id if not upgraded else f"{card_id}+"
            vec = np.zeros(UNIFIED_DIM, dtype=np.float32)

            # --- 基础属性 ---
            cost = card.cost
            if upgraded and card.upgrade_cost is not None:
                cost = card.upgrade_cost
            if cost == -1:  # X 费
                vec[_DIM_INDEX["x_cost"]] = 1.0
                vec[_DIM_INDEX["cost"]] = 0.0
            else:
                vec[_DIM_INDEX["cost"]] = cost / 4.0

            # 类型 one-hot
            type_map = {
                CardType.ATTACK: "is_attack",
                CardType.SKILL: "is_skill",
                CardType.POWER: "is_power",
                CardType.STATUS: "is_status",
                CardType.CURSE: "is_curse",
            }
            if card.card_type in type_map:
                vec[_DIM_INDEX[type_map[card.card_type]]] = 1.0

            # 稀有度
            rarity_map = {
                CardRarity.COMMON: "rarity_common",
                CardRarity.UNCOMMON: "rarity_uncommon",
                CardRarity.RARE: "rarity_rare",
            }
            if card.rarity in rarity_map:
                vec[_DIM_INDEX[rarity_map[card.rarity]]] = 1.0

            vec[_DIM_INDEX["is_upgraded"]] = 1.0 if upgraded else 0.0

            # 旗标
            exhaust = card.exhaust
            ethereal = card.ethereal
            innate = card.innate
            retain = card.retain
            if upgraded:
                if card.upgrade_exhaust is not None:
                    exhaust = card.upgrade_exhaust
                if card.upgrade_ethereal is not None:
                    ethereal = card.upgrade_ethereal
                if card.upgrade_innate is not None:
                    innate = card.upgrade_innate
                if card.upgrade_retain is not None:
                    retain = card.upgrade_retain

            vec[_DIM_INDEX["exhaust"]] = 1.0 if exhaust else 0.0
            vec[_DIM_INDEX["ethereal"]] = 1.0 if ethereal else 0.0
            vec[_DIM_INDEX["innate"]] = 1.0 if innate else 0.0
            vec[_DIM_INDEX["retain"]] = 1.0 if retain else 0.0

            # --- 数值属性 ---
            dmg = card.base_damage
            blk = card.base_block
            if upgraded:
                if dmg >= 0:
                    dmg += card.upgrade_damage
                if blk >= 0:
                    blk += card.upgrade_block

            if dmg >= 0:
                if card.target == CardTarget.ALL_ENEMY:
                    vec[_DIM_INDEX["damage_aoe"]] = dmg / 40.0
                    vec[_DIM_INDEX["is_aoe"]] = 1.0
                else:
                    vec[_DIM_INDEX["damage_single"]] = dmg / 40.0
            if blk >= 0:
                vec[_DIM_INDEX["block_gain"]] = blk / 30.0

            # --- magic_number 消歧义 ---
            dim_name, mag_val = _resolve_magic(card, upgraded)
            if dim_name is not None:
                idx = _DIM_INDEX[dim_name]
                vec[idx] = max(vec[idx], mag_val)

            # --- 效果扫描 ---
            _parse_card_effects(card, vec, upgraded)

            result[key] = vec

    return result


def _load_relics() -> Dict[str, np.ndarray]:
    """加载所有遗物，生成统一向量。"""
    result = {}

    for relic_id, relic in ALL_RELICS.items():
        vec = np.zeros(UNIFIED_DIM, dtype=np.float32)

        # 直接数值属性
        vec[_DIM_INDEX["energy_bonus_permanent"]] = relic.energy_bonus / 3.0
        vec[_DIM_INDEX["max_hp_change"]] = relic.max_hp_bonus / 10.0
        vec[_DIM_INDEX["card_draw"]] = relic.card_draw_bonus / 5.0
        vec[_DIM_INDEX["orb_slots"]] = relic.orb_slots / 3.0
        vec[_DIM_INDEX["damage_reduction"]] = relic.block_loss_reduction / 15.0
        vec[_DIM_INDEX["damage_single"]] = relic.damage_bonus_flat / 10.0

        # 负面标记
        if relic.prevents_healing:
            vec[_DIM_INDEX["hp_heal"]] = -0.5
        if relic.prevents_gold_gain:
            vec[_DIM_INDEX["gold_gain"]] = -0.5

        # 稀有度编码
        tier_map = {
            RelicTier.COMMON: "rarity_common",
            RelicTier.UNCOMMON: "rarity_uncommon",
            RelicTier.RARE: "rarity_rare",
        }
        if relic.tier in tier_map:
            vec[_DIM_INDEX[tier_map[relic.tier]]] = 1.0
        elif relic.tier == RelicTier.BOSS:
            vec[_DIM_INDEX["rarity_rare"]] = 1.0  # Boss 级遗物视为稀有

        # 效果文本关键词扫描
        effects_str = " ".join(relic.effects).lower()

        relic_keywords = {
            "heal": ("hp_heal", 0.3),
            "block": ("block_gain", 0.3),
            "strength": ("strength_gain", 0.3),
            "dexterity": ("dexterity_gain", 0.3),
            "vulnerable": ("vulnerable_apply", 0.3),
            "weak": ("weak_apply", 0.3),
            "thorns": ("thorns", 0.5),
            "draw": ("card_draw", 0.3),
            "energy": ("energy_gain", 0.3),
            "poison": ("poison_apply", 0.3),
            "focus": ("focus_gain", 0.3),
            "mantra": ("mantra", 0.3),
            "artifact": ("artifact", 0.3),
            "gold": ("gold_gain", 0.3),
            "plated": ("plated_armor", 0.3),
            "intangible": ("intangible", 0.3),
            "metallicize": ("metallicize", 0.3),
            "regen": ("regen", 0.3),
        }

        for keyword, (dim_name, value) in relic_keywords.items():
            if keyword in effects_str:
                idx = _DIM_INDEX[dim_name]
                vec[idx] = max(vec[idx], value)

        # 遗物 potion_slots 特殊处理：映射到一个近似维度
        if relic.potion_slots > 0:
            vec[_DIM_INDEX["card_generation"]] = relic.potion_slots / 3.0

        result[relic_id] = vec

    return result


def _load_potions() -> Dict[str, np.ndarray]:
    """加载所有药水，生成统一向量。"""
    result = {}

    for potion_id, potion in ALL_POTIONS.items():
        vec = np.zeros(UNIFIED_DIM, dtype=np.float32)

        desc = potion.description.lower()
        potency = potion.potency

        # 稀有度
        pot_rarity_map = {
            PotionRarity.COMMON: "rarity_common",
            PotionRarity.UNCOMMON: "rarity_uncommon",
            PotionRarity.RARE: "rarity_rare",
        }
        if potion.rarity in pot_rarity_map:
            vec[_DIM_INDEX[pot_rarity_map[potion.rarity]]] = 1.0

        # AOE / 投掷标记
        if potion.target_type == PotionTargetType.ALL_ENEMIES:
            vec[_DIM_INDEX["is_aoe"]] = 1.0

        # 基于描述关键词的效果映射
        potion_effects = [
            ("damage", "damage_single", 40.0),
            ("deal", "damage_single", 40.0),
            ("block", "block_gain", 30.0),
            ("strength", "strength_gain", 5.0),
            ("dexterity", "dexterity_gain", 5.0),
            ("draw", "card_draw", 5.0),
            ("energy", "energy_gain", 3.0),
            ("heal", "hp_heal", 20.0),
            ("max hp", "max_hp_change", 10.0),
            ("poison", "poison_apply", 12.0),
            ("vulnerable", "vulnerable_apply", 5.0),
            ("weak", "weak_apply", 5.0),
            ("thorns", "thorns", 5.0),
            ("plated armor", "plated_armor", 5.0),
            ("artifact", "artifact", 3.0),
            ("intangible", "intangible", 3.0),
            ("focus", "focus_gain", 4.0),
            ("orb slot", "orb_slots", 3.0),
            ("ritual", "ritual", 3.0),
            ("regenerat", "regen", 10.0),
        ]

        for keyword, dim_name, norm in potion_effects:
            if keyword in desc:
                idx = _DIM_INDEX[dim_name]
                val = potency / norm if potency > 0 else 0.3  # 无 potency 的给固定标记
                vec[idx] = max(vec[idx], val)

        # 特殊药水处理
        if "card" in desc and ("add" in desc or "random" in desc):
            vec[_DIM_INDEX["card_generation"]] = 0.5

        # 如果 AOE 伤害，移到 damage_aoe
        if potion.target_type == PotionTargetType.ALL_ENEMIES and vec[_DIM_INDEX["damage_single"]] > 0:
            vec[_DIM_INDEX["damage_aoe"]] = vec[_DIM_INDEX["damage_single"]]
            vec[_DIM_INDEX["damage_single"]] = 0.0

        # 逃跑药水和赌博药水等特殊药水：至少给 card_generation 标记
        if potency == 0 and np.sum(np.abs(vec)) < 0.01:
            # 极少数无法匹配的药水，给一个最小标记
            special_mechanics = " ".join(potion.special_mechanics).lower() if potion.special_mechanics else ""
            if "exhaust" in desc or "exhaust" in special_mechanics:
                vec[_DIM_INDEX["exhaust_other"]] = 0.3
            elif "discard" in desc:
                vec[_DIM_INDEX["discard"]] = 0.3
            elif "escape" in desc or "flee" in desc:
                vec[_DIM_INDEX["hp_heal"]] = 0.3  # 逃跑近似于保命

        result[potion_id] = vec

    return result


# ============================================================================
# 未知实体的 fallback 编码（用于 CommunicationMod 的原始字段）
# ============================================================================

def encode_from_raw(
    card_type: str = "UNKNOWN",
    cost: int = 1,
    damage: int = -1,
    block: int = -1,
    magic_number: int = -1,
    is_upgraded: bool = False,
    exhaust: bool = False,
    ethereal: bool = False,
) -> np.ndarray:
    """
    Fallback：当牌不在数据库中时，用 CommunicationMod 提供的原始字段编码。
    只能编码数值属性，无法推断效果类型。
    """
    vec = np.zeros(UNIFIED_DIM, dtype=np.float32)

    if cost == -1:
        vec[_DIM_INDEX["x_cost"]] = 1.0
    else:
        vec[_DIM_INDEX["cost"]] = cost / 4.0

    type_map = {
        "ATTACK": "is_attack",
        "SKILL": "is_skill",
        "POWER": "is_power",
        "STATUS": "is_status",
        "CURSE": "is_curse",
    }
    if card_type.upper() in type_map:
        vec[_DIM_INDEX[type_map[card_type.upper()]]] = 1.0

    vec[_DIM_INDEX["is_upgraded"]] = 1.0 if is_upgraded else 0.0
    vec[_DIM_INDEX["exhaust"]] = 1.0 if exhaust else 0.0
    vec[_DIM_INDEX["ethereal"]] = 1.0 if ethereal else 0.0

    if damage >= 0:
        vec[_DIM_INDEX["damage_single"]] = damage / 40.0
    if block >= 0:
        vec[_DIM_INDEX["block_gain"]] = block / 30.0

    return vec


# ============================================================================
# 公共 API
# ============================================================================

# 模块加载时预计算所有向量
CARD_DATA: Dict[str, np.ndarray] = _load_cards()
RELIC_DATA: Dict[str, np.ndarray] = _load_relics()
POTION_DATA: Dict[str, np.ndarray] = _load_potions()


def encode_entity(
    entity_type: str,
    entity_id: str,
    upgraded: bool = False,
) -> np.ndarray:
    """
    主入口：获取实体的统一效果向量。

    Args:
        entity_type: "card", "relic", 或 "potion"
        entity_id: 实体 ID（卡牌支持别名解析）
        upgraded: 是否升级（仅对卡牌有效）

    Returns:
        UNIFIED_DIM 维的 float32 向量
    """
    if entity_type == "card":
        # 尝试别名解析
        resolved = resolve_card_id(entity_id)
        key = f"{resolved}+" if upgraded else resolved
        if key in CARD_DATA:
            return CARD_DATA[key].copy()
        # Fallback：返回零向量（调用方可用 encode_from_raw 代替）
        return np.zeros(UNIFIED_DIM, dtype=np.float32)

    elif entity_type == "relic":
        if entity_id in RELIC_DATA:
            return RELIC_DATA[entity_id].copy()
        return np.zeros(UNIFIED_DIM, dtype=np.float32)

    elif entity_type == "potion":
        if entity_id in POTION_DATA:
            return POTION_DATA[entity_id].copy()
        return np.zeros(UNIFIED_DIM, dtype=np.float32)

    else:
        raise ValueError(f"未知实体类型: {entity_type}（应为 card/relic/potion）")


# ============================================================================
# 验证脚本
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("STS 数据提取模块 — 验证")
    print("=" * 60)

    # --- 统计 ---
    n_base_cards = len(ALL_CARDS)
    n_card_vectors = len(CARD_DATA)  # 含升级版
    n_relics = len(RELIC_DATA)
    n_potions = len(POTION_DATA)
    print(f"\n向量维度: {UNIFIED_DIM}")
    print(f"卡牌: {n_base_cards} 张（{n_card_vectors} 个向量，含升级版）")
    print(f"遗物: {n_relics} 个")
    print(f"药水: {n_potions} 个")

    # --- 验证非零向量 ---
    zero_cards = [k for k, v in CARD_DATA.items() if np.sum(np.abs(v)) < 1e-6]
    if zero_cards:
        print(f"\n⚠ 零向量卡牌 ({len(zero_cards)} 张): {zero_cards[:10]}")
    else:
        print(f"\n✓ 所有 {n_card_vectors} 个卡牌向量均非零")

    # --- 示例卡牌 ---
    examples = [
        ("Strike_R", False, "Strike（基础攻击）"),
        ("Bash", False, "Bash（攻击 + 易伤）"),
        ("Bash", True, "Bash+（升级版）"),
        ("Battle Trance", False, "Battle Trance（抽牌）"),
        ("Corruption", False, "Corruption（能力，复杂效果）"),
        ("Demon Form", False, "Demon Form（能力，力量成长）"),
    ]

    print("\n" + "-" * 60)
    print("示例卡牌编码:")
    print("-" * 60)

    for card_id, upg, desc in examples:
        vec = encode_entity("card", card_id, upgraded=upg)
        nonzero = [(DIM_NAMES[i], f"{vec[i]:.2f}") for i in range(UNIFIED_DIM) if abs(vec[i]) > 1e-4]
        print(f"\n  {desc} (id={card_id}, upgraded={upg}):")
        for name, val in nonzero:
            print(f"    {name}: {val}")

    # --- magic_number 消歧义验证 ---
    print("\n" + "-" * 60)
    print("magic_number 消歧义验证:")
    print("-" * 60)

    test_cases = [
        ("Bash", "vulnerable_apply", "Bash → vulnerable"),
        ("Battle Trance", "card_draw", "Battle Trance → draw"),
        ("Demon Form", "ritual", "Demon Form → ritual (strength_each_turn)"),
        ("CutThroughFate", "scry", "Cut Through Fate → scry"),
        ("PathToVictory", "poison_apply", "Pressure Points → mark/poison"),
    ]

    for card_id, expected_dim, desc in test_cases:
        vec = encode_entity("card", card_id)
        val = vec[_DIM_INDEX[expected_dim]]
        status = "✓" if val > 0 else "✗"
        print(f"  {status} {desc}: {expected_dim}={val:.2f}")

    # --- 示例遗物 ---
    print("\n" + "-" * 60)
    print("示例遗物编码:")
    print("-" * 60)

    relic_examples = ["Burning Blood", "Vajra", "Akabeko", "Lantern"]
    for relic_id in relic_examples:
        if relic_id in RELIC_DATA:
            vec = RELIC_DATA[relic_id]
            nonzero = [(DIM_NAMES[i], f"{vec[i]:.2f}") for i in range(UNIFIED_DIM) if abs(vec[i]) > 1e-4]
            print(f"\n  {relic_id}:")
            for name, val in nonzero:
                print(f"    {name}: {val}")

    # --- 示例药水 ---
    print("\n" + "-" * 60)
    print("示例药水编码:")
    print("-" * 60)

    potion_examples = ["Fire Potion", "Block Potion", "Strength Potion", "Weak Potion"]
    for pot_id in potion_examples:
        if pot_id in POTION_DATA:
            vec = POTION_DATA[pot_id]
            nonzero = [(DIM_NAMES[i], f"{vec[i]:.2f}") for i in range(UNIFIED_DIM) if abs(vec[i]) > 1e-4]
            print(f"\n  {pot_id}:")
            for name, val in nonzero:
                print(f"    {name}: {val}")

    print("\n" + "=" * 60)
    print("验证完成")
    print("=" * 60)

"""
V8 Stage A.1 — Ironclad 教师策略常量。

移植自 bottled_ai (MIT 协议, https://github.com/xaved88/bottled_ai)
源 commit: 7b47a877d626cdb4fbeeb423c46472b15dd2ac00
源路径:   rs/ai/requested_strike/config.py

本模块仅包含纯数据常量（删牌优先级、目标 deck、升级偏好、药水偏好）
以及 bottled_ai 风格 card id（lowercase，含空格）→ StSRLSolver 风格 card id
（PascalCase，下划线）的对照表，供 v8_teacher_eval.py 使用。

本文件不 import bottled_ai；移植完成后必须自包含。
"""

from __future__ import annotations

from typing import Dict, List


# ============================================================================
# § 1 bottled_ai 原始常量（逐字搬运，未做任何改动）
# ============================================================================

CARD_REMOVAL_PRIORITY_LIST: List[str] = [
    "defend",
    "strike",
    "defend+",
    "strike+",
]

DESIRED_CARDS_FOR_DECK: Dict[str, int] = {
    # ---- 原 23 张（bottled_ai 移植）----
    "perfected strike": 5,
    "offering": 1,
    "battle trance": 2,
    "reaper": 2,
    "twin strike": 2,
    "shockwave": 2,
    "thunderclap": 2,
    "dropkick": 2,
    "pommel strike": 2,
    "shrug it off": 2,
    "impervious": 2,
    "ghostly armor": 1,
    "flame barrier": 1,
    "blind": 1,
    "apotheosis": 1,
    "handofgreed": 1,
    "master of strategy": 1,
    "flash of steel": 1,
    "trip": 1,
    "dark shackles": 1,
    "swift strike": 1,
    "dramatic entrance": 1,
    "finesse": 1,
    # ---- 扩展：Ironclad 主流 common / uncommon / rare ----
    # Attacks
    "body slam": 2,
    "sword boomerang": 2,
    "anger": 2,
    "cleave": 1,
    "hemokinesis": 1,
    "iron wave": 2,
    "rampage": 1,
    "carnage": 1,
    "headbutt": 1,
    "heavy blade": 2,
    "uppercut": 1,
    "blood for blood": 1,
    "dropkick+": 1,            # 升级版需求位（实际通过 upgrade）
    "feed": 1,
    "fiend fire": 1,
    "immolate": 1,
    "reaper+": 1,
    "bludgeon": 1,
    "whirlwind": 2,
    # Skills
    "bash": 1,
    "true grit": 1,
    "spot weakness": 1,
    "sentinel": 1,
    "burning pact": 1,
    "second wind": 1,
    "bloodletting": 1,
    "intimidate": 1,
    "warcry": 1,
    "double tap": 1,
    "limit break": 1,
    "exhume": 1,
    # Powers
    "berserk": 1,
    "combust": 1,
    "dark embrace": 1,
    "evolve": 1,
    "feel no pain": 1,
    "fire breathing": 1,
    "inflame": 2,
    "metallicize": 1,
    "rupture": 1,
    "brutality": 1,
    "juggernaut": 1,
    "demon form": 1,
    "barricade": 1,
    "corruption": 1,
}

HIGH_PRIORITY_UPGRADES: List[str] = [
    "Apotheosis",
    "Perfected Strike",
]

DESIRED_POTIONS: List[str] = [
    "fruit juice",
    "fairy in a bottle",
    "cultist potion",
    "power potion",
    "potion of capacity",
    "heart of iron",
    "duplication potion",
    "distilled chaos",
    "blessing of the forge",
    "attack potion",
    "dexterity potion",
    "ambrosia",
    "fear potion",
    "essence of steel",
    "strength potion",
    "regen potion",
    "blood potion",
    "entropic brew",
    "liquid bronze",
    "energy potion",
    "skill potion",
    "ancient potion",
    "weak potion",
    "gambler's brew",
    "poison potion",
    "colorless potion",
    "flex potion",
    "swift potion",
    "bottled miracle",
    "fire potion",
    "explosive potion",
    "speed potion",
    "block potion",
    "stance potion",
    "smoke bomb",
    "elixir potion",
    "liquid memories",
    "snecko oil",
]


# ============================================================================
# § 2 卡名映射：bottled_ai 风格（lowercase, 空格） → StSRLSolver 风格
# ----------------------------------------------------------------------------
# StSRLSolver / data/sts_data.py 使用 PascalCase + 下划线 + Ironclad 后缀
# （示例：Strike_R, Defend_R, Bash, TwinStrike, PerfectedStrike）。
# bottled_ai 的 CardId enum value 是小写 + 空格（示例：'twin strike'）。
#
# 此处我们只为 Stage A.1 实际可能用到的 Ironclad 卡建表。
# 找不到对应 StSRLSolver id 的条目 → 显式列在 UNMAPPED 注释里，并用
# `None` 占位；调用方应跳过 None 并视为「该卡在我们的环境里不存在」。
#
# 注意：本表未经实际 ALL_CARDS 校验（StSRLSolver 不在本机），
#      使用前请用一个 unit test 跑 `assert v in ALL_CARDS or v is None`。
# ============================================================================

BOTTLED_TO_LOCAL_CARD_ID: Dict[str, str | None] = {
    # --- Ironclad 起始/基础 ---
    "strike":            "Strike_R",       # 红 Ironclad strike
    "defend":            "Defend_R",
    "strike+":           "Strike_R+",
    "defend+":           "Defend_R+",
    "bash":              "Bash",

    # --- DESIRED_CARDS_FOR_DECK 中的 Ironclad 卡 ---
    "perfected strike":  "PerfectedStrike",
    "offering":          "Offering",
    "battle trance":     "BattleTrance",
    "reaper":            "Reaper",
    "twin strike":       "TwinStrike",
    "shockwave":         "Shockwave",
    "thunderclap":       "Thunderclap",
    "dropkick":          "Dropkick",
    "pommel strike":     "PommelStrike",
    "shrug it off":      "ShrugItOff",
    "impervious":        "Impervious",
    "ghostly armor":     "GhostlyArmor",
    "flame barrier":     "FlameBarrier",

    # --- 通用 / Colorless ---
    "blind":             "Blind",
    "apotheosis":        "Apotheosis",
    "handofgreed":       "HandOfGreed",
    "master of strategy": "MasterOfStrategy",
    "flash of steel":    "FlashOfSteel",
    "trip":              "Trip",
    "dark shackles":     "DarkShackles",
    "swift strike":      "SwiftStrike",
    "dramatic entrance": "DramaticEntrance",
    "finesse":           "Finesse",
}

# UNMAPPED（调用方应当跳过）：
#   目前对照表覆盖了 DESIRED_CARDS_FOR_DECK 中的全部 23 张卡。
#   若日后 StSRLSolver 校验发现某 id 不在 ALL_CARDS，应将其在此表里改成 None
#   并在下方注释中记录，例如：
#     "handofgreed": None,  # 未在 sts_data 中找到，跳过
#
# 已知"待 ALL_CARDS 实测确认"的高风险条目（PascalCase 命名拼法可能不一致）：
#   - HandOfGreed   （也可能写作 "HandOfGreed_C"）
#   - MasterOfStrategy
#   - DramaticEntrance
#   - Trip          （Silent 卡，但 bottled_ai 把它列为 colorless 期望卡）

CARDS_THAT_EXIT_WRATH: List[str] = [
    # 移植自 bottled_ai/rs/common/comparators/common_general_comparator.py
    # 这些是 Watcher 卡，Ironclad 教师不会用到，仅为 ComparatorAssessment
    # 的 stance_is_not_wrath 方法保留接口。Ironclad 战斗中此列表实际为空效果。
    "EmptyBody", "EmptyFist", "EmptyMind",
    "FearNoEvil", "InnerPeace", "Tranquility", "Vigilance",
]


# ============================================================================
# § 3 Power 分组（Ironclad 教师评估时用）
# ----------------------------------------------------------------------------
# 这里用 PowerId 字符串而非 enum，避免依赖 bottled_ai。
# 字符串风格须与 StSRLSolver 的 power id 对齐 —— **未经实测确认**，
# 待 v8_battle_state_adapter 接通真实 state 后做一次 unit test。
# ============================================================================

POWERS_WE_LIKE: List[str] = [
    # 通用 / 全职业都喜欢
    "ACCURACY", "AFTER_IMAGE", "BARRICADE", "BERSERK", "BLUR",
    "BUFFER", "COLLECT", "CORRUPTION", "DARK_EMBRACE", "DEMON_FORM",
    "DEVA", "DEVOTION", "ECHO_FORM", "ELECTRO", "ENVENOM",
    "ESTABLISHMENT", "EVOLVE", "FAKE_ALPHA_BETA", "FEEL_NO_PAIN",
    "FIRE_BREATHING", "FOCUS", "FORESIGHT", "HEATSINK",
    "INFINITE_BLADES", "INTANGIBLE_PLAYER", "JUGGERNAUT", "LIKE_WATER",
    "LOOP", "MACHINE_LEARNING", "MANTRA_INTERNAL", "MASTER_REALITY",
    "MAYHEM", "MENTAL_FORTRESS", "METALLICIZE", "NIRVANA",
    "NOXIOUS_FUMES", "OMEGA", "PANACHE_INTERNAL", "PHANTASMAL",
    "PLATED_ARMOR", "REPAIR", "RUSHDOWN", "SADISTIC", "SIMMERING_RAGE",
    "STUDY", "THORNS", "THOUSAND_CUTS", "TOOLS_OF_THE_TRADE",
    "BATTLE_HYMN",
]

POWERS_WE_LIKE_LESS: List[str] = [
    "ARTIFACT", "DEXTERITY", "ENERGIZED", "FREE_ATTACK_POWER",
    "STRENGTH", "VIGOR",
]

POWERS_WE_DISLIKE: List[str] = [
    # = bottled_ai DEBUFFS（PowerId enum 名）
    "BIAS", "BLOCK_RETURN", "CHOKED", "CONFUSED", "CONSTRICTED",
    "DRAW_REDUCTION", "ENTANGLED", "FASTING", "FRAIL", "LOCK_ON",
    "MARK", "NO_DRAW", "POISON", "VULNERABLE", "WEAKENED",
    "WRAITH_FORM_POWER",
]


# ============================================================================
# § 4 卡 tier 表（pick_card_reward 用 tier 评分）
# ----------------------------------------------------------------------------
# id 为 StSRLSolver PascalCase（与 ALL_CARDS 对齐）。
# 评分: S=4, A=3, B=2, C=1, 未列名=0（视为 skip 候选）
# 升级版（id 末尾带 +）权重 +0.5（在 pick_card_reward 内部加）
# 覆盖 Ironclad 主流卡（无脑通用版本，未做 archetype 区分）
# ============================================================================

CARD_TIER: Dict[str, str] = {
    # ===== S 级（核心 / 必拿） =====
    "Apotheosis": "S",
    "PerfectedStrike": "S",
    "Bludgeon": "S",
    "DemonForm": "S",
    "Corruption": "S",
    "Limit Break": "S",   # 占位（StSRLSolver id 可能为 LimitBreak）
    "LimitBreak": "S",
    "Barricade": "S",
    "Impervious": "S",
    "Offering": "S",
    "DoubleTap": "S",
    "Reaper": "S",
    "FiendFire": "S",
    "Immolate": "S",
    "Whirlwind": "S",

    # ===== A 级（强力 / 高优先级） =====
    "BattleTrance": "A",
    "Inflame": "A",
    "FeelNoPain": "A",
    "Metallicize": "A",
    "FireBreathing": "A",
    "DarkEmbrace": "A",
    "Berserk": "A",
    "Combust": "A",
    "Evolve": "A",
    "Rupture": "A",
    "Juggernaut": "A",
    "Brutality": "A",
    "TwinStrike": "A",
    "Dropkick": "A",
    "PommelStrike": "A",
    "ShrugItOff": "A",
    "Shockwave": "A",
    "GhostlyArmor": "A",
    "BloodForBlood": "A",
    "SeeingRed": "A",
    "Bloodletting": "A",
    "Sentinel": "A",
    "SecondWind": "A",
    "Carnage": "A",
    "HeavyBlade": "A",
    "Feed": "A",
    "SpotWeakness": "A",
    "Exhume": "A",
    "BurningPact": "A",

    # ===== B 级（中等 / 视情况） =====
    "BodySlam": "B",
    "SwordBoomerang": "B",
    "Anger": "B",
    "Cleave": "B",
    "Hemokinesis": "B",
    "IronWave": "B",
    "Rampage": "B",
    "Headbutt": "B",
    "Uppercut": "B",
    "TrueGrit": "B",
    "Thunderclap": "B",
    "FlameBarrier": "B",
    "Bash": "B",
    "Intimidate": "B",
    "Warcry": "B",
    "Pummel": "B",
    "Disarm": "B",
    "Armaments": "B",
    "Flex": "B",
    "Havoc": "B",
    "PowerThrough": "B",
    "Rage": "B",
    "Entrench": "B",

    # ===== C 级（较弱，但好过空过） =====
    "Clothesline": "C",
    "Clash": "C",
    "WildStrike": "C",
    "PerfectedStrike+": "S",   # 升级版核心
    "Bash+": "B",
    "SearingBlow": "C",
    "Reckless Charge": "C",
    "RecklessCharge": "C",
    "Thunderclap+": "B",
    "Dropkick+": "A",
    "BattleTrance+": "A",

    # ===== Curse / Status 永远不要 =====
    # （这里不列，未列名默认 0 分→ skip）
}


# ============================================================================
# § 5 Boss relic tier（pick_boss_relic 用）
# ----------------------------------------------------------------------------
# 数据来自 bottled_ai/rs/ai/requested_strike/handlers/boss_relic_handler.py。
# id 用 lowercase（StSRLSolver 的 boss_relics 选项里 relic.name 通常 PascalCase
# 或带空格的人类可读形式；pick_boss_relic 会做大小写不敏感匹配）。
# ============================================================================

# bottled_ai 顺序保留（高优先 → 低优先）。注释里标记需依据状态调整的项。
BOSS_RELIC_PRIORITY: List[str] = [
    "sozu",
    "runic dome",
    "philosopher's stone",
    "ectoplasm",
    "velvet choker",
    "cursed key",
    "fusion hammer",
    "snecko eye",
    "mark of pain",         # 已有 energy relic 时移除
    "busted crown",         # act 1 或已有 energy relic 时移除
    "coffee dripper",       # act 1 或已有 energy relic 时移除
    "slaver's collar",
    "runic cube",
    "runic pyramid",
    "black blood",
    "calling bell",
    "empty cage",
    "black star",
    "sacred bark",
]

# 提供 energy 的 relic（与上表配合做去重逻辑）
ENERGY_RELICS: List[str] = [
    "sozu",
    "runic dome",
    "philosopher's stone",
    "ectoplasm",
    "velvet choker",
    "cursed key",
    "fusion hammer",
    "mark of pain",
    "busted crown",
    "coffee dripper",
    "nuclear battery",
]


# ============================================================================
# § 6 Event 决策映射（pick_event_action 用）
# ----------------------------------------------------------------------------
# 移植自 bottled_ai/rs/common/handlers/common_event_handler.py +
# bottled_ai/rs/ai/requested_strike/handlers/event_handler.py
#
# 数据结构: event_id (lowercase, 空格 / 标点保留) → callable 或 dict-style 决策。
# 我们这里用一个轻量 schema:
#   - "fixed": <choice_idx>   恒选某 idx
#   - "by_hp": [(threshold, idx), ...]  按 HP%（>= threshold）选；最后 fallback
# pick_event_action 内部根据 schema 推 idx；找不到 event_id 则 fallback 0。
#
# event_id 用 StSRLSolver 的 EventState.event_id（PascalCase 字符串）做主 key,
# bottled_ai 用 enum 名（GOLDEN_IDOL 等）。这里两套都列, 取并集。
# ============================================================================

EVENT_CHOICE_MAP: Dict[str, dict] = {
    # ---- ACT 1 ----
    "Big Fish":            {"by_hp": [(31, 1)], "fallback": 0},   # >30 拿 max hp; 否则 heal
    "The Cleric":          {"fixed": 0},                          # 简化：拿第一个（heal/purify）
    "Dead Adventurer":     {"fixed": 1},                          # 逃
    "Golden Idol":         {"fixed": 1},                          # 简化：默认 leave
    "Hypnotizing Colored Mushrooms": {"by_hp": [(40, 0)], "fallback": 1},
    "Living Wall":         {"fixed": 2},                          # 升级
    "Scrap Ooze":          {"fixed": 0},                          # yolo
    "Shining Light":       {"by_hp": [(70, 0)], "fallback": 1},
    "The Ssssserpent":     {"fixed": 1},
    "World of Goop":       {"by_hp": [(80, 0)], "fallback": 1},
    "Wing Statue":         {"by_hp": [(70, 0)], "fallback": 1},
    "Face Trader":         {"by_hp": [(75, 0)], "fallback": 2},
    # ---- ACT 1, 2, 3 ----
    "A Note For Yourself":          {"fixed": 1},
    "Bonfire Spirits":              {"fixed": 0},   # purge
    "The Divine Fountain":          {"fixed": 0},
    "Duplicator":                   {"fixed": 1},
    "Golden Shrine":                {"fixed": 0},
    "Lab":                          {"fixed": 0},
    "Match and Keep!":              {"fixed": 0},
    "Ominous Forge":                {"fixed": 1},
    "Purifier":                     {"fixed": 0},
    "Transmogrifier":               {"fixed": 1},
    "Upgrade Shrine":               {"fixed": 0},
    "We Meet Again!":               {"fixed": 0},
    "The Woman in Blue":            {"fixed": 0},
    # ---- ACT 2 ----
    "Ancient Writing":              {"fixed": 1},
    "Augmenter":                    {"fixed": 2},
    "The Colosseum":                {"fixed": 0},
    "Council of Ghosts":            {"fixed": 1},   # refuse
    "Cursed Tome":                  {"fixed": 1},
    "Forgotten Altar":              {"fixed": 0},
    "The Joust":                    {"fixed": 0},
    "Knowing Skull":                {"fixed": 3},   # leave
    "The Library":                  {"fixed": 0},
    "Masked Bandits":               {"by_hp": [(65, 1)], "fallback": 0},
    "The Mausoleum":                {"fixed": 1},   # leave
    "The Nest":                     {"fixed": 0},
    "N'loth":                       {"fixed": 2},   # leave
    "Old Beggar":                   {"fixed": 0},
    "Pleading Vagrant":             {"fixed": 1},
    "Vampires":                     {"fixed": 1},   # refuse
    "Designer In-Spire":            {"fixed": 0},
    # ---- ACT 3 ----
    "Falling":                      {"fixed": 0},   # 简化（理想是按 cards_desired 排）
    "Mind Bloom":                   {"fixed": 0},
    "Mysterious Sphere":            {"by_hp": [(70, 0)], "fallback": 1},
    "Secret Portal":                {"fixed": 1},
    "Sensory Stone":                {"fixed": 0},
    "Tomb of Lord Red Mask":        {"fixed": 0},
    "Winding Halls":                {"fixed": 2},
}


# ============================================================================
# § 7 Neow blessing 优先级（pick_neow 用）
# ----------------------------------------------------------------------------
# 移植自 bottled_ai NeowHandler.desired_choices（顺序保留）
# bottled_ai 用 description 字符串（lowercase）匹配；我们这里同时支持
# StSRLSolver 的 NeowBlessingType enum value 与 bottled_ai 描述串。
# pick_neow 内部根据 blessing_type / description 双路匹配。
# ============================================================================

# 描述字符串关键字（lowercase，做 substring 匹配）→ 优先级 idx（小=先选）
NEOW_PRIORITY_KEYWORDS: List[str] = [
    "upgrade a card",
    "obtain a random common relic",
    "obtain 100 gold",
    "choose a card to obtain",
    "obtain 3 random potions",
    "choose a colorless card to obtain",
    "max hp +8",
    "obtain a random rare card",
    "enemies in your next three combats",   # 1 hp enemies
    "remove a card from your deck",
    "transform a card",
    "lose your starting relic",             # boss swap
]

# StSRLSolver NeowBlessingType.value → 优先级 idx；用于 enum 路径
# (lower = better)
NEOW_PRIORITY_BY_TYPE: Dict[str, int] = {
    "upgrade_card":               0,
    "random_common_relic":        1,
    "hundred_gold":               2,
    "three_cards":                3,
    "three_potions":              4,
    "random_colorless_rare":      5,
    "ten_percent_hp_bonus":       6,
    "one_random_rare_card":       7,
    "three_enemy_kill":           8,
    "remove_card":                9,
    "transform_card":            10,
    "boss_swap":                 11,
    "remove_two":                12,
    "transform_two":             13,
    "random_rare_relic":         14,
}


# ============================================================================
# § 8 Shop 偏好（pick_shop_action 用）
# ----------------------------------------------------------------------------
# 直接搬 bottled_ai/rs/ai/requested_strike/handlers/shop_purchase_handler.py
# 的 self.relics 和 self.cards
# ============================================================================

SHOP_RELIC_PRIORITY: List[str] = [
    "Bag of Marbles",
    "Pen Nib",
    "Strike Dummy",
    "Paper Phrog",
    "Preserved Insect",
    "Red Skull",
    "Meat on the Bone",
    "Eternal Feather",
    "Regal Pillow",
    "Lee's Waffle",
    "Meal Ticket",
    "Strawberry",
    "Toy Ornithopter",
    "Pantograph",
    "Pear",
    "Orichalcum",
    "Anchor",
    "Horn Cleat",
    "Self-Forming Clay",
    "Thread and Needle",
    "Lantern",
    "Happy Flower",
    "Bag of Preparation",
    "Centennial Puzzle",
]

# Shop 里指定要买的卡（id 用 StSRLSolver PascalCase）
SHOP_CARD_PRIORITY: List[str] = [
    "Offering",
    "BattleTrance",
    "Shockwave",
]


__all__ = [
    "CARD_REMOVAL_PRIORITY_LIST",
    "DESIRED_CARDS_FOR_DECK",
    "HIGH_PRIORITY_UPGRADES",
    "DESIRED_POTIONS",
    "BOTTLED_TO_LOCAL_CARD_ID",
    "CARDS_THAT_EXIT_WRATH",
    "POWERS_WE_LIKE",
    "POWERS_WE_LIKE_LESS",
    "POWERS_WE_DISLIKE",
    # 新增：
    "CARD_TIER",
    "BOSS_RELIC_PRIORITY",
    "ENERGY_RELICS",
    "EVENT_CHOICE_MAP",
    "NEOW_PRIORITY_KEYWORDS",
    "NEOW_PRIORITY_BY_TYPE",
    "SHOP_RELIC_PRIORITY",
    "SHOP_CARD_PRIORITY",
]

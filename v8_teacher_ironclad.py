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
]

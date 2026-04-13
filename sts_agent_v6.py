"""
杀戮尖塔 ML Agent v6 — 统一效果编码 + PPO + Shared Transformer
================================================================
V6 核心变化：编码层重写。
  - 用 data/sts_data.py 的 CARD_DATA/RELIC_DATA/POTION_DATA 替代手写 CARD_EFFECTS/hash/learned embedding
  - 卡牌/遗物/药水统一为 UNIFIED_DIM (~57) 维效果向量
  - Power 编码改为 ~20 功能类别，每类 2 维 → 44 维
  - 模型架构不变：ContextTransformer(3层4头) + CombatHead + DraftHead + PathHead + ChoiceHead + PPO

Architecture:
  Context Transformer (3 layers, 4 heads, dim=128)
    [CLS] + run_progress + deck_cards/player/enemies/hand/piles/potions/relics → shared_repr (256-dim)

  Combat Head (Autoregressive)
    shared_repr → card_query @ hand_encodings → pick card
    shared_repr + card_encoding → target_scorer @ monster_encodings → pick target

  Draft/Path/Choice Heads
    shared_repr + option_features → score → softmax → choice

  PPO Training
    GAE(γ=0.999, λ=0.95), 4 epochs, minibatch=64
"""

import sys
import re
import json
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
from collections import defaultdict, Counter

# 从 data 层导入统一编码
from data.sts_data import (
    CARD_DATA, RELIC_DATA, POTION_DATA,
    UNIFIED_DIM, DIM_NAMES, encode_from_raw,
    _DIM_INDEX,
)


# ============================================================
# Section 1: Constants and Configuration
# ============================================================

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "sts_models"
MODEL_DIR.mkdir(exist_ok=True)
LOG_FILE = BASE_DIR / "sts_agent_v6.log"
MODEL_PATH = MODEL_DIR / "agent_v6.pt"

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 统一实体编码维度（来自 data/sts_data.py）
ENTITY_DIM = UNIFIED_DIM  # ~57 dims

# Context Transformer
CTX_DIM = 128
CTX_HEADS = 4
CTX_LAYERS = 3
CTX_FF_DIM = 256
SHARED_REPR_DIM = 256

# Token types
TOKEN_CLS = 0
TOKEN_RUN_PROGRESS = 1
TOKEN_PLAYER = 2
TOKEN_MONSTER = 3
TOKEN_HAND_CARD = 4
TOKEN_DRAW_SUMMARY = 5
TOKEN_DISCARD_SUMMARY = 6
TOKEN_POTION = 7
TOKEN_DECK_CARD = 8
TOKEN_RELIC_SUMMARY = 9
NUM_TOKEN_TYPES = 10

# Player state encoding: 6 基础 + 3 卡组信息 + 44 power + 11 角色机制 = 64
POWER_CATEGORY_DIM = 44   # 20 categories × 2 + 4 hash fallback
PLAYER_BASE_DIM = 9       # hp_ratio, hp_deficit², max_hp/200, block/50, energy/4, turn/15, hand_size/10, draw_pile/30, discard_pile/30
PLAYER_MECHANIC_DIM = 11  # stance one-hot(4) + orb_info(6) + mantra(1)
PLAYER_DIM = PLAYER_BASE_DIM + POWER_CATEGORY_DIM + PLAYER_MECHANIC_DIM  # 64

# Enemy encoding: 4 基础 + 3 状态标记 + 11 intent + 44 power = 62
ENEMY_BASE_DIM = 4        # hp_ratio, hp_abs/200, max_hp/200, block/50
ENEMY_STATUS_DIM = 3      # alive, is_boss, halfDead
ENEMY_INTENT_DIM = 11     # intent one-hot(7) + intent_damage/40 + intent_hits/5 + total_threat/60 + is_multi_attack
ENEMY_DIM = ENEMY_BASE_DIM + ENEMY_STATUS_DIM + ENEMY_INTENT_DIM + POWER_CATEGORY_DIM  # 62

# Pile summary = ENTITY_DIM + 12 (composition stats)
PILE_SUMMARY_DIM = 12
PILE_TOTAL_DIM = ENTITY_DIM + PILE_SUMMARY_DIM

# Run progress
RUN_PROGRESS_DIM = 13

# Limits
MAX_HAND = 10
MAX_MONSTERS = 4
MAX_DECK_CARDS = 50
MAX_POTION_SLOTS = 5
MAX_SEQ_LEN = 1 + MAX_DECK_CARDS + 1 + MAX_POTION_SLOTS + 1 + 1 + MAX_MONSTERS + MAX_HAND + 2 + 1  # ~76

# Combat
NUM_CARD_ACTIONS = MAX_HAND + 1
END_TURN_ACTION = MAX_HAND

# Choice head option encoding
CHOICE_OPTION_DIM = 40

# Path
PATH_FEATURE_DIM = 16

# PPO hyperparameters
GAMMA = 0.98
GAE_LAMBDA = 0.95
CLIP_RATIO = 0.2
VALUE_COEFF = 0.5
ENTROPY_COEFF_START = 0.05
ENTROPY_COEFF_END = 0.02
ENTROPY_ANNEAL_RUNS = 5000
LR = 2.5e-4
MAX_GRAD_NORM = 0.5
PPO_EPOCHS = 4
MINIBATCH_SIZE = 64
N_RUNS_PER_UPDATE = 16


# ============================================================
# Section 2: Communication and Logging
# ============================================================

def send(msg):
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()

def receive():
    line = sys.stdin.readline().strip()
    if not line:
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        log(f"JSON parse fail: {line[:200]}")
        return None

def log(msg):
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")


# ============================================================
# Section 3: Unified Entity Encoding
# ============================================================

# 特殊维度索引（从 DIM_NAMES 查找）
IS_PLAYABLE_IDX = None  # 在战斗中覆盖 is_attack 位来标记可打出状态
# 我们借用 is_upgraded 之后的空间 — 实际上直接用 DIM_NAMES 索引更安全
# 但 is_playable 不在 DIM_NAMES 中，我们用一个约定：
# 战斗中手牌编码时，将 innate 位 (idx 12) 临时覆盖为 is_playable
_INNATE_IDX = _DIM_INDEX["innate"]  # idx 12, 战斗中重用为 is_playable
_COUNTER_IDX = _DIM_INDEX.get("gold_gain", 54)  # 遗物 counter 借用一个低信息维度


def encode_card(card: dict) -> np.ndarray:
    """统一效果编码，ENTITY_DIM 维

    优先从 CARD_DATA 查表，未知卡用 encode_from_raw 兜底
    """
    card_id = card.get("id", card.get("name", ""))
    if not card_id:
        return np.zeros(ENTITY_DIM, dtype=np.float32)

    # 处理升级标记
    upgraded = card.get("upgraded", False)
    if not upgraded and card.get("upgrades", 0) > 0:
        upgraded = True
    if not upgraded and card_id.endswith("+"):
        upgraded = True

    # 标准化 card_id（去掉尾部 +，由 upgraded 标记决定）
    base_id = card_id.rstrip("+").strip()
    key = f"{base_id}+" if upgraded else base_id

    if key in CARD_DATA:
        vec = CARD_DATA[key].copy()
    elif base_id in CARD_DATA:
        vec = CARD_DATA[base_id].copy()
    else:
        # Fallback：从 card dict 的原始字段编码
        card_type = card.get("type", "UNKNOWN")
        cost = int(card.get("cost", 1) or 1)
        damage = int(card.get("damage", -1) or -1)
        block = int(card.get("block", -1) or -1)
        vec = encode_from_raw(
            card_type=card_type,
            cost=cost,
            damage=damage,
            block=block,
            is_upgraded=upgraded,
            exhaust=card.get("exhaust", False),
            ethereal=card.get("ethereal", False),
        )

    # 战斗中覆盖 is_playable（借用 innate 位）
    if "is_playable" in card:
        vec[_INNATE_IDX] = 1.0 if card["is_playable"] else 0.0

    return vec


def encode_relic_entity(relic: dict) -> np.ndarray:
    """遗物编码 → ENTITY_DIM 维向量"""
    relic_id = relic.get("id", relic.get("name", ""))
    if relic_id in RELIC_DATA:
        vec = RELIC_DATA[relic_id].copy()
    else:
        vec = np.zeros(ENTITY_DIM, dtype=np.float32)
    # 叠加 counter 信息
    counter = relic.get("counter", 0) or 0
    if counter > 0:
        vec[_COUNTER_IDX] = min(counter / 10.0, 2.0)
    return vec


def encode_potion_entity(potion_data: dict) -> np.ndarray:
    """药水编码 → ENTITY_DIM 维向量

    空槽返回全零。
    """
    if not potion_data:
        return np.zeros(ENTITY_DIM, dtype=np.float32)
    potion_id = potion_data.get("id", "")
    if not potion_id or potion_id == "Potion Slot":
        return np.zeros(ENTITY_DIM, dtype=np.float32)
    if potion_id in POTION_DATA:
        return POTION_DATA[potion_id].copy()
    return np.zeros(ENTITY_DIM, dtype=np.float32)


# ============================================================
# Section 4: Power Encoding (Functional Categories)
# ============================================================
# 20 功能类别，每类 2 维（total_amount/norm, count） = 40 dims
# + 4 dims hash fallback = 44 dims total

POWER_CATEGORIES = [
    # (类别名, {power_id: ...}, 归一化除数)
    ("damage_increase", {"Strength", "Vigor", "DoubleDamage", "PenNib"}, 10.0),
    ("damage_decrease", {"Weakened", "Weak"}, 5.0),
    ("block_increase", {"Dexterity"}, 10.0),
    ("block_decrease", {"Frail"}, 5.0),
    ("damage_taken_increase", {"Vulnerable"}, 5.0),
    ("damage_taken_decrease", {"Intangible", "IntangiblePlayer", "Flight"}, 3.0),
    ("dot_poison", {"Poison", "Constricted"}, 20.0),
    ("scaling_per_turn", {"Ritual", "Demon Form", "DemonForm", "DevaForm"}, 5.0),
    ("block_retention", {"Barricade", "Blur", "Metallicize", "Plated Armor", "PlatedArmor"}, 10.0),
    ("on_card_draw", {"Evolve", "DarkEmbrace", "Dark Embrace"}, 3.0),
    ("on_exhaust", {"Feel No Pain", "FeelNoPain"}, 5.0),
    ("retaliation", {"Thorns", "Flame Barrier", "FlameBarrier", "After Image", "AfterImage"}, 5.0),
    ("energy_mod", {"Berserk", "Energized"}, 3.0),
    ("status_lock", {"Entangle", "Entangled", "No Draw", "NoDraw", "Confused", "Hex"}, 1.0),
    ("artifact", {"Artifact"}, 5.0),
    ("buffer", {"Buffer"}, 3.0),
    ("corruption", {"Corruption"}, 1.0),
    ("orb_focus", {"Focus"}, 5.0),
    ("stance_mod", {"Wrath", "Calm", "Mantra"}, 1.0),
    ("other", set(), 5.0),  # fallback
]

# 构建 power_id → category_idx 的快速查找表
_POWER_TO_CATEGORY = {}
for cat_idx, (_, power_ids, _) in enumerate(POWER_CATEGORIES):
    for pid in power_ids:
        _POWER_TO_CATEGORY[pid] = cat_idx


def encode_powers_categorical(powers_list: list) -> np.ndarray:
    """将 power 列表编码为 44 维向量（20类 × 2 + 4 hash fallback）"""
    vec = np.zeros(POWER_CATEGORY_DIM, dtype=np.float32)
    if not powers_list:
        return vec

    # 20 categories × 2 dims
    for p in powers_list:
        pid = p.get("id", p.get("name", ""))
        amt = p.get("amount", 1)
        if not pid:
            continue

        cat_idx = _POWER_TO_CATEGORY.get(pid, -1)
        if cat_idx >= 0:
            _, _, norm = POWER_CATEGORIES[cat_idx]
            base = cat_idx * 2
            vec[base] += float(amt) / norm       # 累积量 / 归一化
            vec[base + 1] += 1.0                  # 该类别中的 power 数量
        else:
            # 未分类的 power → hash fallback（最后 4 维）
            hash_offset = 40
            # 简单 hash：用 power name 的字符和确定 bucket
            h = sum(ord(c) for c in pid) % 4
            vec[hash_offset + h] += float(min(abs(amt) / 5.0, 2.0))

    # clip 所有值到合理范围
    np.clip(vec, -5.0, 5.0, out=vec)
    return vec


# ============================================================
# Section 5: Player and Enemy State Encoding
# ============================================================

def encode_player_state(combat: dict) -> np.ndarray:
    """编码玩家战斗状态 → PLAYER_DIM (~64) 维向量"""
    player = combat.get("player", {})
    v = np.zeros(PLAYER_DIM, dtype=np.float32)

    max_hp = max(player.get("max_hp", 1), 1)
    hp = player.get("current_hp", 0)
    turn = combat.get("turn", 0)
    hand = combat.get("hand", [])
    draw_pile = combat.get("draw_pile", [])
    discard_pile = combat.get("discard_pile", [])

    # 基础维度 [0:9]
    v[0] = float(hp / max_hp)                              # hp_ratio
    v[1] = float((1.0 - hp / max_hp) ** 2)                 # hp_deficit²
    v[2] = float(max_hp / 200.0)                            # max_hp / 200
    v[3] = float(min(player.get("block", 0) / 50.0, 2.0))  # block / 50
    v[4] = float(player.get("energy", 0) / 4.0)            # energy / 4
    v[5] = float(min(turn / 15.0, 1.0))                    # turn / 15
    v[6] = float(len(hand) / 10.0)                          # hand_size / 10
    v[7] = float(len(draw_pile) / 30.0)                     # draw_pile_size / 30
    v[8] = float(len(discard_pile) / 30.0)                  # discard_pile_size / 30

    # Power 编码 [9:53] — 44 维
    powers = player.get("powers", [])
    v[9:9 + POWER_CATEGORY_DIM] = encode_powers_categorical(powers)

    # 角色机制 [53:64] — 11 维
    mech_offset = 9 + POWER_CATEGORY_DIM
    # Stance one-hot [0:4]
    stance = (player.get("stance", "") or "").upper()
    if stance == "WRATH":
        v[mech_offset] = 1.0
    elif stance == "CALM":
        v[mech_offset + 1] = 1.0
    elif stance == "DIVINITY":
        v[mech_offset + 2] = 1.0
    else:
        v[mech_offset + 3] = 1.0  # neutral / none

    # Orb info [4:10]
    orbs = player.get("orbs", [])
    orb_slots = player.get("orb_slots", 0)
    v[mech_offset + 4] = orb_slots / 5.0
    lightning = sum(1 for o in orbs if o.get("id", "").upper() == "LIGHTNING")
    frost = sum(1 for o in orbs if o.get("id", "").upper() == "FROST")
    dark = sum(1 for o in orbs if o.get("id", "").upper() == "DARK")
    plasma = sum(1 for o in orbs if o.get("id", "").upper() == "PLASMA")
    v[mech_offset + 5] = lightning / 5.0
    v[mech_offset + 6] = frost / 5.0
    v[mech_offset + 7] = dark / 5.0
    v[mech_offset + 8] = plasma / 5.0

    # Focus（从 powers 中提取，但也作为独立维度）
    for p in powers:
        if p.get("id") == "Focus":
            v[mech_offset + 9] = p.get("amount", 0) / 10.0
            break

    # Mantra [10]
    for p in powers:
        if p.get("id") == "Mantra":
            v[mech_offset + 10] = p.get("amount", 0) / 10.0
            break

    return v


def encode_enemy(monster: dict) -> np.ndarray:
    """编码敌人状态 → ENEMY_DIM (~62) 维向量"""
    v = np.zeros(ENEMY_DIM, dtype=np.float32)
    alive = not monster.get("is_gone", True) and monster.get("current_hp", 0) > 0
    if not alive:
        return v

    max_hp_m = max(monster.get("max_hp", 1), 1)
    hp_m = monster.get("current_hp", 0)
    dmg = float(monster.get("move_adjusted_damage", monster.get("move_base_damage", 0)) or 0)
    hits = float(monster.get("move_hits", 1) or 1)
    intent = (monster.get("intent", "") or monster.get("move_id", "") or "UNKNOWN").upper()

    # 基础 [0:4]
    v[0] = float(hp_m / max_hp_m)           # hp_ratio
    v[1] = float(hp_m / 200.0)              # hp_abs / 200
    v[2] = float(max_hp_m / 200.0)          # max_hp / 200
    v[3] = float(min(monster.get("block", 0) / 50.0, 2.0))  # block / 50

    # 状态标记 [4:7]
    v[4] = 1.0  # alive
    # Boss 判断：名称或 type 字段
    monster_id = monster.get("id", monster.get("name", "")).lower()
    is_boss = any(b in monster_id for b in [
        "slime_boss", "hexaghost", "guardian", "automaton",
        "collector", "champ", "awakened", "time_eater",
        "donu", "deca", "heart", "corrupt_heart",
    ])
    v[5] = 1.0 if is_boss else 0.0
    v[6] = 1.0 if monster.get("half_dead", False) else 0.0

    # Intent one-hot [7:14] + intent details [14:18]
    intent_types = ["ATTACK", "BUFF", "DEBUFF", "DEFEND", "ESCAPE", "UNKNOWN", "SLEEP"]
    for i, it in enumerate(intent_types):
        if it in intent:
            v[7 + i] = 1.0

    # Intent damage details
    v[14] = float(min(dmg / 40.0, 2.0))       # intent_damage / 40
    v[15] = float(min(hits / 5.0, 2.0))        # intent_hits / 5
    v[16] = float(min(dmg * hits / 60.0, 2.0)) # total_threat / 60
    v[17] = 1.0 if hits > 1 else 0.0           # is_multi_attack

    # Power 编码 [18:62] — 44 维
    powers = monster.get("powers", [])
    v[18:18 + POWER_CATEGORY_DIM] = encode_powers_categorical(powers)

    return v


# ============================================================
# Section 6: Run Progress Encoding
# ============================================================

def encode_run_progress(game_state: dict) -> np.ndarray:
    """编码整局进度 → RUN_PROGRESS_DIM (13) 维向量"""
    gs = game_state.get("game_state", {})
    v = np.zeros(RUN_PROGRESS_DIM, dtype=np.float32)
    floor = gs.get("floor", 0)
    hp = gs.get("current_hp", 1)
    max_hp = max(gs.get("max_hp", 1), 1)

    v[0] = float(min(floor / 50.0, 1.0))
    v[1] = float(min(gs.get("act", 1) / 3.0, 1.0))
    v[2] = float(hp / max_hp)  # hp_ratio
    v[3] = float(min(gs.get("gold", 0) / 999.0, 1.0))
    v[4] = float(min(gs.get("ascension_level", 0) / 20.0, 1.0))
    v[5] = float(min(len(gs.get("deck", [])) / 30.0, 1.5))
    v[6] = float(len([p for p in gs.get("potions", [])
                       if p.get("id") and p.get("id") != "Potion Slot"]) / 5.0)
    v[7] = float(min(len(gs.get("relics", [])) / 15.0, 1.0))
    v[8] = float((1.0 - hp / max_hp) ** 2)  # hp_deficit_nonlinear
    v[9] = 1.0 if floor in (16, 33, 50) else 0.0  # is_boss_floor
    screen = gs.get("screen_type", "")
    v[10] = 1.0 if "ELITE" in str(screen).upper() else 0.0

    # boss_proximity
    if floor <= 16:
        floors_to_boss = float(16 - floor)
    elif floor <= 33:
        floors_to_boss = float(33 - floor)
    elif floor <= 50:
        floors_to_boss = float(50 - floor)
    else:
        floors_to_boss = 0.0
    v[11] = float(floors_to_boss / 15.0)

    # 卡组截断溢出信号
    deck_len = len(gs.get("deck", []))
    v[12] = max(0.0, (deck_len - MAX_DECK_CARDS) / 10.0)

    return v


# ============================================================
# Section 7: Pile Summary Encoding
# ============================================================

def encode_pile_summary(cards: list) -> np.ndarray:
    """编码牌堆摘要 → ENTITY_DIM + 12 维向量

    前 ENTITY_DIM 维：牌堆中所有卡的效果向量均值
    后 12 维：牌堆组成统计
    """
    combined = np.zeros(PILE_TOTAL_DIM, dtype=np.float32)
    if not cards:
        return combined

    # 效果向量均值
    card_vecs = np.array([encode_card(c) for c in cards[:30]], dtype=np.float32)
    combined[:ENTITY_DIM] = card_vecs.mean(axis=0)

    # 组成统计
    n = len(cards)
    attacks = sum(1 for c in cards if c.get("type", "") == "ATTACK")
    skills = sum(1 for c in cards if c.get("type", "") == "SKILL")
    powers = sum(1 for c in cards if c.get("type", "") == "POWER")

    comp = combined[ENTITY_DIM:]
    comp[0] = n / 30.0                    # pile size
    comp[1] = attacks / max(n, 1)          # 攻击牌比例
    comp[2] = skills / max(n, 1)           # 技能牌比例
    comp[3] = powers / max(n, 1)           # 能力牌比例
    comp[4] = sum(c.get("damage", 0) or 0 for c in cards) / max(n * 20.0, 1)
    comp[5] = sum(c.get("block", 0) or 0 for c in cards) / max(n * 15.0, 1)
    comp[6] = sum(1 for c in cards if (c.get("cost", 1) or 1) == 0) / max(n, 1)
    comp[7] = sum(1 for c in cards if c.get("exhaust", False)) / max(n, 1)
    avg_cost = sum(c.get("cost", 1) or 1 for c in cards) / max(n, 1)
    comp[8] = avg_cost / 4.0
    comp[9] = sum(1 for c in cards if c.get("type", "") in ("CURSE", "STATUS")) / max(n, 1)
    comp[10] = sum(1 for c in cards if c.get("upgrades", 0) > 0 or c.get("upgraded", False)) / max(n, 1)
    comp[11] = 0.0  # reserved

    return combined


# ============================================================
# Section 8: Path and Choice Encoding
# ============================================================

def encode_path_node(node, hp_ratio=1.0):
    """编码单个地图节点 → PATH_NODE_DIM (8) 维"""
    v = np.zeros(8, dtype=np.float32)
    sym = node.get("symbol", "M")
    sym_map = {"M": 0, "E": 1, "R": 2, "$": 3, "?": 4, "T": 5}
    idx = sym_map.get(sym, 0)
    v[idx] = 1.0
    v[6] = float(node.get("x", 0) / 6.0)
    v[7] = 0.0
    return v


def encode_full_path(path_rooms, hp_ratio=1.0, deck_size=10, gold=0, act=1):
    """编码从当前位置到 Boss 的一条完整路径 → PATH_FEATURE_DIM (16) 维"""
    v = np.zeros(PATH_FEATURE_DIM, dtype=np.float32)
    n = len(path_rooms)
    if n == 0:
        return v

    counts = {"M": 0, "E": 0, "R": 0, "$": 0, "?": 0, "T": 0, "B": 0}
    for r in path_rooms:
        sym = r[0] if r else "M"
        if sym in counts:
            counts[sym] += 1
        else:
            counts["M"] += 1

    v[0] = counts["M"] / max(n, 1)
    v[1] = counts["E"] / max(n, 1)
    v[2] = counts["R"] / max(n, 1)
    v[3] = counts["$"] / max(n, 1)
    v[4] = counts["?"] / max(n, 1)
    v[5] = counts["T"] / max(n, 1)
    v[6] = min(n / 15.0, 1.0)

    first_rest = n
    for i, r in enumerate(path_rooms):
        if r in ("R", "REST"):
            first_rest = i
            break
    v[7] = min(first_rest / 10.0, 1.0)

    max_consec_fight = 0
    current_consec = 0
    for r in path_rooms:
        if r in ("M", "E", "MONSTER", "ELITE"):
            current_consec += 1
            max_consec_fight = max(max_consec_fight, current_consec)
        elif r in ("R", "REST"):
            current_consec = 0
    v[8] = min(max_consec_fight / 8.0, 1.0)

    elite_positions = [i for i, r in enumerate(path_rooms) if r in ("E", "ELITE")]
    if elite_positions:
        avg_elite_pos = sum(elite_positions) / len(elite_positions)
        v[9] = avg_elite_pos / max(n, 1)

    v[10] = hp_ratio
    v[11] = min(deck_size / 30.0, 1.5)
    v[12] = min(gold / 300.0, 1.0)
    v[13] = min(act / 3.0, 1.0)
    # v[14], v[15] reserved
    return v


def encode_choice_option(choice_type, option_text="", option_index=0, max_options=1, **kwargs):
    """编码选择选项 → CHOICE_OPTION_DIM (40) 维向量"""
    v = np.zeros(CHOICE_OPTION_DIM, dtype=np.float32)
    type_map = {
        "event": 0, "rest": 1, "boss_relic": 2, "potion_use": 3,
        "potion_discard": 4, "shop": 5, "reward": 6, "other": 7,
        "neow": 0,
    }
    idx = type_map.get(choice_type, 7)
    v[idx] = 1.0
    v[8] = float(option_index / max(max_options, 1))

    # 简单文本 hash（替代 bloom filter）
    text_lower = option_text.lower() if option_text else ""
    if text_lower:
        for i in range(8):
            h = sum(ord(c) * (i + 1) for c in text_lower) % 256
            v[9 + i] = h / 256.0

    v[17] = float(kwargs.get("hp_ratio", 0.0))
    v[18] = float(kwargs.get("threat_level", 0.0))
    v[19] = float(kwargs.get("extra", 0.0))

    # 关键词检测 [20:28]
    keywords = ["hp", "gold", "relic", "card", "upgrade", "curse", "remove", "max_hp"]
    for ki, kw in enumerate(keywords):
        if kw in text_lower:
            v[20 + ki] = 1.0

    # 数字提取 [28]
    numbers = re.findall(r'\d+', text_lower)
    if numbers:
        v[28] = min(float(numbers[0]) / 100.0, 2.0)
    v[29] = min(len(text_lower) / 100.0, 1.0)

    # 结构化效果维度 [30:39]
    hp_gain = 0
    hp_loss = 0
    if ("heal" in text_lower or "restore" in text_lower
            or ("gain" in text_lower and "hp" in text_lower)):
        nums = re.findall(r'(\d+)\s*(?:hp|health|hit point)', text_lower)
        if nums:
            hp_gain = int(nums[0])
    if "lose" in text_lower and "hp" in text_lower:
        nums = re.findall(r'(\d+)\s*(?:hp|health|hit point)', text_lower)
        if nums:
            hp_loss = int(nums[0])
    v[30] = hp_gain / 30.0
    v[31] = hp_loss / 30.0

    gold_gain = 0
    gold_loss = 0
    if "gain" in text_lower and "gold" in text_lower:
        nums = re.findall(r'(\d+)\s*gold', text_lower)
        if nums:
            gold_gain = int(nums[0])
    if "lose" in text_lower and "gold" in text_lower:
        nums = re.findall(r'(\d+)\s*gold', text_lower)
        if nums:
            gold_loss = int(nums[0])
    v[32] = gold_gain / 200.0
    v[33] = gold_loss / 200.0

    max_hp_change = 0
    if "max hp" in text_lower or "max health" in text_lower:
        nums = re.findall(r'(\d+)\s*max', text_lower)
        if nums:
            max_hp_change = int(nums[0])
        if "lose" in text_lower or "decrease" in text_lower:
            max_hp_change = -max_hp_change
    v[34] = max_hp_change / 20.0

    v[35] = 1.0 if ("add" in text_lower or "obtain" in text_lower or "receive" in text_lower) and "card" in text_lower else 0.0
    v[36] = 1.0 if ("remove" in text_lower or "delete" in text_lower) and "card" in text_lower else 0.0
    v[37] = 1.0 if "upgrade" in text_lower and "card" in text_lower else 0.0
    v[38] = 1.0 if any(w in text_lower for w in ("random", "unknown", "chance", "might", "maybe", "?")) else 0.0
    v[39] = 0.0  # reserved

    return v


def encode_shop_item(item, player_gold=0):
    """编码商店物品 → CHOICE_OPTION_DIM (40) 维向量"""
    feat = np.zeros(CHOICE_OPTION_DIM, dtype=np.float32)

    item_id = item.get("id", "")
    price = item.get("price", 0)
    item_type = item.get("type", "")

    # 价格编码 [0:3]
    feat[0] = price / 500.0
    feat[1] = 1.0 if player_gold >= price else 0.0
    feat[2] = (player_gold - price) / 500.0 if player_gold >= price else 0.0

    # 物品类型 one-hot [3:8]
    type_map = {"card": 3, "relic": 4, "potion": 5, "remove": 6, "leave": 7}
    if item_type in type_map:
        feat[type_map[item_type]] = 1.0

    # 物品效果 [8:ENTITY_DIM 截断到可用空间]
    if item_type == "card":
        card_vec = encode_card({"id": item_id})
        # 将 ENTITY_DIM 向量压缩到 [8:39] 的 31 个位置
        end_idx = min(8 + ENTITY_DIM, CHOICE_OPTION_DIM - 1)
        feat[8:end_idx] = card_vec[:end_idx - 8]
    elif item_type == "relic":
        relic_vec = encode_relic_entity({"id": item_id})
        end_idx = min(8 + ENTITY_DIM, CHOICE_OPTION_DIM - 1)
        feat[8:end_idx] = relic_vec[:end_idx - 8]
    elif item_type == "potion":
        pot_vec = encode_potion_entity({"id": item_id})
        end_idx = min(8 + ENTITY_DIM, CHOICE_OPTION_DIM - 1)
        feat[8:end_idx] = pot_vec[:end_idx - 8]
    elif item_type == "remove":
        feat[8] = 0.5  # 删牌标记

    feat[39] = 0.0  # reserved
    return feat


# ============================================================
# Section 9: Context Transformer
# ============================================================

class TokenProjections(nn.Module):
    """每种 token type 有独立的线性投影 + type embedding"""
    def __init__(self):
        super().__init__()
        self.projections = nn.ModuleDict({
            "cls": nn.Linear(CTX_DIM, CTX_DIM),
            "run_progress": nn.Linear(RUN_PROGRESS_DIM, CTX_DIM),
            "player": nn.Linear(PLAYER_DIM, CTX_DIM),
            "enemy": nn.Linear(ENEMY_DIM, CTX_DIM),
            "card": nn.Linear(ENTITY_DIM, CTX_DIM),
            "pile_summary": nn.Linear(PILE_TOTAL_DIM, CTX_DIM),
            "potion": nn.Linear(ENTITY_DIM, CTX_DIM),
            "relic": nn.Linear(ENTITY_DIM, CTX_DIM),
            "path": nn.Linear(PATH_FEATURE_DIM, CTX_DIM),
            "choice": nn.Linear(CHOICE_OPTION_DIM, CTX_DIM),
        })
        self.type_embeddings = nn.Embedding(NUM_TOKEN_TYPES, CTX_DIM)
        self.cls_token = nn.Parameter(torch.randn(CTX_DIM))

    def project(self, token_type_name, token_type_id, raw_features):
        proj = self.projections[token_type_name](raw_features)
        type_emb = self.type_embeddings(torch.tensor(token_type_id, device=raw_features.device))
        return proj + type_emb


class ContextTransformer(nn.Module):
    """[CLS] + tokens → shared_repr (256-dim)"""
    def __init__(self):
        super().__init__()
        self.token_proj = TokenProjections()
        self.pos_encoding = nn.Embedding(MAX_SEQ_LEN, CTX_DIM)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=CTX_DIM, nhead=CTX_HEADS,
            dim_feedforward=CTX_FF_DIM, dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=CTX_LAYERS)
        self.output_proj = nn.Sequential(
            nn.Linear(CTX_DIM, SHARED_REPR_DIM),
            nn.ReLU(),
        )

    def forward(self, tokens, attention_mask=None):
        device = next(self.parameters()).device

        seq = []
        cls_input = self.token_proj.cls_token.unsqueeze(0).to(device)
        cls_proj = self.token_proj.project("cls", TOKEN_CLS, cls_input).squeeze(0)
        seq.append(cls_proj)

        for name, type_id, feat in tokens:
            proj = self.token_proj.project(name, type_id, feat.to(device))
            seq.append(proj)

        seq_tensor = torch.stack(seq).unsqueeze(0)
        seq_len = seq_tensor.shape[1]

        positions = torch.arange(seq_len, device=device)
        seq_tensor = seq_tensor + self.pos_encoding(positions).unsqueeze(0)

        if attention_mask is not None:
            src_key_padding_mask = ~attention_mask.unsqueeze(0)
        else:
            src_key_padding_mask = None

        out = self.transformer(seq_tensor, src_key_padding_mask=src_key_padding_mask)
        cls_out = out[0, 0]
        return self.output_proj(cls_out)


# ============================================================
# Section 10: Task-Specific Heads
# ============================================================

class CombatHead(nn.Module):
    """战斗决策头：card_query dot-product scoring + target scoring"""
    def __init__(self):
        super().__init__()
        self.card_query = nn.Sequential(
            nn.Linear(SHARED_REPR_DIM, 192),
            nn.ReLU(),
            nn.Linear(192, CTX_DIM),
        )
        self.end_turn_key = nn.Parameter(torch.randn(CTX_DIM) * 0.01)  # 与手牌编码同量级的小随机初始化
        self.target_query = nn.Sequential(
            nn.Linear(SHARED_REPR_DIM + CTX_DIM, 192),
            nn.ReLU(),
            nn.Linear(192, CTX_DIM),
        )
        self.value_head = nn.Sequential(
            nn.Linear(SHARED_REPR_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, shared_repr, hand_encodings, monster_encodings,
                card_mask, monster_mask):
        query = self.card_query(shared_repr)
        n_hand = hand_encodings.shape[0]
        card_scores = torch.zeros(NUM_CARD_ACTIONS, device=shared_repr.device)
        for i in range(n_hand):
            card_scores[i] = torch.dot(query, hand_encodings[i])
        card_scores[END_TURN_ACTION] = torch.dot(query, self.end_turn_key)
        card_logits = card_scores.clone()
        card_logits[card_mask == 0] = -1e9
        value = self.value_head(shared_repr).squeeze(-1)
        target_logits = torch.zeros(MAX_MONSTERS, device=shared_repr.device)
        if monster_encodings.shape[0] > 0:
            target_q = self.target_query(torch.cat([shared_repr, query]))
            for j in range(min(monster_encodings.shape[0], MAX_MONSTERS)):
                target_logits[j] = torch.dot(target_q, monster_encodings[j])
        target_logits[monster_mask == 0] = -1e9
        return card_logits, target_logits, value

    def forward_with_card(self, shared_repr, card_encoding, monster_encodings, monster_mask):
        target_input = torch.cat([shared_repr, card_encoding])
        target_q = self.target_query(target_input)
        target_logits = torch.zeros(MAX_MONSTERS, device=shared_repr.device)
        for j in range(min(monster_encodings.shape[0], MAX_MONSTERS)):
            target_logits[j] = torch.dot(target_q, monster_encodings[j])
        target_logits[monster_mask == 0] = -1e9
        return target_logits


class DraftHead(nn.Module):
    """选牌决策头：shared_repr + 候选卡效果向量 → 分数"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(SHARED_REPR_DIM + ENTITY_DIM, 192),
            nn.ReLU(),
            nn.Linear(192, 96),
            nn.ReLU(),
            nn.Linear(96, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(SHARED_REPR_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, shared_repr, candidate_features_list):
        scores = []
        for feat in candidate_features_list:
            inp = torch.cat([shared_repr, feat.to(shared_repr.device)])
            scores.append(self.net(inp).squeeze(-1))
        logits = torch.stack(scores)
        value = self.value_head(shared_repr).squeeze(-1)
        return logits, value


class PathHead(nn.Module):
    """选路决策头"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(SHARED_REPR_DIM + PATH_FEATURE_DIM, 192),
            nn.ReLU(),
            nn.Linear(192, 96),
            nn.ReLU(),
            nn.Linear(96, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(SHARED_REPR_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, shared_repr, node_features_list):
        scores = []
        for feat in node_features_list:
            inp = torch.cat([shared_repr, feat.to(shared_repr.device)])
            scores.append(self.net(inp).squeeze(-1))
        logits = torch.stack(scores)
        value = self.value_head(shared_repr).squeeze(-1)
        return logits, value


class ChoiceHead(nn.Module):
    """通用选择决策头（事件/休息/Boss遗物/药水/商店）"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(SHARED_REPR_DIM + CHOICE_OPTION_DIM, 192),
            nn.ReLU(),
            nn.Linear(192, 96),
            nn.ReLU(),
            nn.Linear(96, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(SHARED_REPR_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, shared_repr, option_encodings_list):
        scores = []
        for enc in option_encodings_list:
            inp = torch.cat([shared_repr, enc.to(shared_repr.device)])
            scores.append(self.net(inp).squeeze(-1))
        logits = torch.stack(scores)
        value = self.value_head(shared_repr).squeeze(-1)
        return logits, value


# ============================================================
# Section 11: Full Model
# ============================================================

class STSModelV6(nn.Module):
    def __init__(self):
        super().__init__()
        self.ctx_transformer = ContextTransformer()
        self.combat_head = CombatHead()
        self.draft_head = DraftHead()
        self.path_head = PathHead()
        self.choice_head = ChoiceHead()

    def to_device(self):
        return self.to(DEVICE)


def load_model_safe(model, path, load_optimizer=False, optimizer=None):
    """加载模型权重，跳过维度不匹配的层（架构变更后的向后兼容）"""
    checkpoint = torch.load(path, weights_only=False, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    model_state = model.state_dict()

    loaded = 0
    skipped = 0
    skipped_keys = []
    for key, param in state_dict.items():
        if key in model_state:
            if param.shape == model_state[key].shape:
                model_state[key] = param
                loaded += 1
            elif 'pos_encoding.weight' in key:
                min_len = min(param.shape[0], model_state[key].shape[0])
                model_state[key][:min_len] = param[:min_len]
                loaded += 1
                log(f"部分加载 {key}: {param.shape} -> {model_state[key].shape} (前 {min_len} 位置)")
            else:
                skipped_keys.append(key)
                log(f"跳过 {key}: 形状不匹配 {param.shape} vs {model_state[key].shape}")
                skipped += 1
        else:
            skipped_keys.append(key)
            log(f"跳过 {key}: 新模型中不存在")
            skipped += 1

    model.load_state_dict(model_state)
    log(f"安全加载完成: {loaded} 层加载, {skipped} 层跳过")
    if skipped_keys:
        log(f"跳过的层: {skipped_keys[:10]}{'...' if len(skipped_keys) > 10 else ''}")

    if load_optimizer and optimizer is not None and "optimizer" in checkpoint:
        if skipped == 0:
            try:
                optimizer.load_state_dict(checkpoint["optimizer"])
                log("Optimizer 状态已加载")
            except Exception as e:
                log(f"Optimizer 加载失败（维度变化）: {e}")
        else:
            log("存在维度不匹配层，跳过 optimizer 加载")

    meta = {
        "runs": checkpoint.get("runs", 0),
        "entropy_coeff": checkpoint.get("entropy_coeff", ENTROPY_COEFF_START),
    }
    return meta


# ============================================================
# Section 12: Trajectory Buffer and GAE
# ============================================================

class Transition:
    __slots__ = ['decision_type', 'state_data', 'action', 'log_prob',
                 'value', 'reward', 'done', 'advantage', 'returns',
                 'extra']

    def __init__(self, decision_type, state_data, action, log_prob, value, reward, done=False, extra=None):
        self.decision_type = decision_type
        self.state_data = state_data
        self.action = action
        self.log_prob = log_prob
        self.value = value
        self.reward = reward
        self.done = done
        self.advantage = 0.0
        self.returns = 0.0
        self.extra = extra or {}


def compute_gae(trajectory, gamma=GAMMA, lam=GAE_LAMBDA):
    T = len(trajectory)
    if T == 0:
        return
    last_gae = 0.0
    for t in reversed(range(T)):
        if t == T - 1 or trajectory[t].done:
            next_value = 0.0
        else:
            next_value = trajectory[t + 1].value
        delta = trajectory[t].reward + gamma * next_value * (1.0 - float(trajectory[t].done)) - trajectory[t].value
        last_gae = delta + gamma * lam * (1.0 - float(trajectory[t].done)) * last_gae
        trajectory[t].advantage = last_gae
        trajectory[t].returns = last_gae + trajectory[t].value


# ============================================================
# Section 13: PPO Update
# ============================================================

def ppo_update(model, optimizer, trajectories, entropy_coeff,
               strategy_only=False, strategy_gamma=0.99, strategy_params=None):
    gamma = strategy_gamma if strategy_only else GAMMA
    for traj in trajectories:
        compute_gae(traj, gamma=gamma)

    all_transitions = []
    for traj in trajectories:
        if strategy_only:
            all_transitions.extend([t for t in traj if t.decision_type != "combat"])
        else:
            all_transitions.extend(traj)

    if not all_transitions:
        return {}

    # 按决策类型分组归一化 advantage，避免 combat 和 strategy 量纲差异
    combat_trans = [t for t in all_transitions if t.decision_type == "combat"]
    strategy_trans = [t for t in all_transitions if t.decision_type != "combat"]

    for group in [combat_trans, strategy_trans]:
        if len(group) > 1:
            advs = [t.advantage for t in group]
            mean_adv = sum(advs) / len(advs)
            std_adv = (sum((a - mean_adv) ** 2 for a in advs) / len(advs)) ** 0.5 + 1e-8
            for t in group:
                t.advantage = (t.advantage - mean_adv) / std_adv

    if strategy_only and strategy_params is not None:
        update_params = list(strategy_params)
    else:
        update_params = list(model.parameters())

    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    n_updates = 0

    for epoch in range(PPO_EPOCHS):
        random.shuffle(all_transitions)

        for batch_start in range(0, len(all_transitions), MINIBATCH_SIZE):
            batch = all_transitions[batch_start:batch_start + MINIBATCH_SIZE]

            policy_losses = []
            value_losses = []
            entropies = []

            for t in batch:
                new_log_prob, new_value, entropy = evaluate_transition(model, t)
                ratio = torch.exp(new_log_prob - t.log_prob)
                adv = torch.tensor(float(t.advantage), dtype=torch.float32, device=DEVICE)
                clipped = torch.clamp(ratio, 1 - CLIP_RATIO, 1 + CLIP_RATIO)
                policy_losses.append(-torch.min(ratio * adv, clipped * adv))
                # Clipped value loss
                returns_t = torch.tensor(float(t.returns), dtype=torch.float32, device=DEVICE)
                old_value_t = torch.tensor(float(t.value), dtype=torch.float32, device=DEVICE)
                value_pred_clipped = old_value_t + torch.clamp(new_value - old_value_t, -CLIP_RATIO, CLIP_RATIO)
                value_loss_unclipped = (new_value - returns_t) ** 2
                value_loss_clipped = (value_pred_clipped - returns_t) ** 2
                value_losses.append(0.5 * torch.max(value_loss_unclipped, value_loss_clipped))
                entropies.append(entropy)

            policy_loss = torch.stack(policy_losses).mean()
            value_loss = torch.stack(value_losses).mean()
            entropy_loss = -torch.stack(entropies).mean()

            loss = policy_loss + VALUE_COEFF * value_loss + entropy_coeff * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(update_params, MAX_GRAD_NORM)
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += -entropy_loss.item()
            n_updates += 1

    return {
        "policy_loss": total_policy_loss / max(n_updates, 1),
        "value_loss": total_value_loss / max(n_updates, 1),
        "entropy": total_entropy / max(n_updates, 1),
        "n_transitions": len(all_transitions),
        "n_updates": n_updates,
    }


def evaluate_transition(model, transition):
    """重新计算 transition 的 log_prob, value, entropy（PPO 多 epoch 需要）"""
    sd = transition.state_data
    dt = transition.decision_type

    if dt == "combat":
        shared_repr = rebuild_shared_repr(model, sd)
        hand_enc = sd["hand_encodings"].to(DEVICE)
        monster_enc = sd["monster_encodings"].to(DEVICE)
        card_mask = sd["card_mask"].to(DEVICE)
        monster_mask = sd["monster_mask"].to(DEVICE)

        card_logits, target_logits, value = model.combat_head(
            shared_repr, hand_enc, monster_enc, card_mask, monster_mask
        )

        card_action, target_action = transition.action
        card_dist = torch.distributions.Categorical(logits=card_logits)
        card_log_prob = card_dist.log_prob(torch.tensor(card_action, device=DEVICE))
        card_entropy = card_dist.entropy()

        if sd.get("needs_target", False) and card_action != END_TURN_ACTION:
            if card_action < hand_enc.shape[0]:
                card_enc_t = hand_enc[card_action]
            else:
                card_enc_t = torch.zeros(CTX_DIM, device=DEVICE)
            target_logits = model.combat_head.forward_with_card(
                shared_repr, card_enc_t, monster_enc, monster_mask
            )
            target_dist = torch.distributions.Categorical(logits=target_logits)
            target_log_prob = target_dist.log_prob(torch.tensor(target_action, device=DEVICE))
            target_entropy = target_dist.entropy()
        else:
            target_log_prob = torch.tensor(0.0, device=DEVICE)
            target_entropy = torch.tensor(0.0, device=DEVICE)

        return card_log_prob + target_log_prob, value, card_entropy + target_entropy

    elif dt == "draft":
        shared_repr = rebuild_shared_repr(model, sd)
        candidate_feats = sd["candidate_features"]
        logits, value = model.draft_head(shared_repr, candidate_feats)
        dist = torch.distributions.Categorical(logits=logits)
        action = transition.action[0]
        return dist.log_prob(torch.tensor(action, device=DEVICE)), value, dist.entropy()

    elif dt == "path":
        shared_repr = rebuild_shared_repr(model, sd)
        node_feats = sd["node_features"]
        padded_feats = []
        for f in node_feats:
            if isinstance(f, torch.Tensor) and f.shape[-1] < PATH_FEATURE_DIM:
                pad = torch.zeros(PATH_FEATURE_DIM - f.shape[-1], device=f.device)
                padded_feats.append(torch.cat([f, pad]))
            else:
                padded_feats.append(f)
        logits, value = model.path_head(shared_repr, padded_feats)
        dist = torch.distributions.Categorical(logits=logits)
        action = transition.action[0]
        return dist.log_prob(torch.tensor(action, device=DEVICE)), value, dist.entropy()

    elif dt == "choice":
        shared_repr = rebuild_shared_repr(model, sd)
        option_encs = sd["option_encodings"]
        logits, value = model.choice_head(shared_repr, option_encs)
        dist = torch.distributions.Categorical(logits=logits)
        action = transition.action[0]
        return dist.log_prob(torch.tensor(action, device=DEVICE)), value, dist.entropy()

    else:
        return torch.tensor(0.0, device=DEVICE), torch.tensor(0.0, device=DEVICE), torch.tensor(0.0, device=DEVICE)


def _detach_tokens(tokens):
    """将 tokens 从计算图中分离，避免 PPO 多 epoch 反向传播报错"""
    return [(name, ttype, t.detach().cpu() if isinstance(t, torch.Tensor) else t)
            for name, ttype, t in tokens]


def rebuild_shared_repr(model, state_data):
    tokens = state_data["tokens"]
    return model.ctx_transformer(tokens)


# ============================================================
# Section 14: Reward Functions
# ============================================================

def hp_value(hp, max_hp):
    """非线性 HP 价值 — 低 HP 指数级更珍贵"""
    ratio = float(hp) / float(max(max_hp, 1))
    return float(ratio ** 0.5)


FLOOR_MILESTONES = {
    3: 0.25, 6: 1.5, 10: 3.0, 13: 4.0, 15: 6.0,
    16: 2.0, 17: 12.0,
    25: 9.0,
    33: 2.0, 34: 22.0,
    50: 2.0, 51: 22.0,
    55: 50.0,
}


def floor_milestone_reward(floor):
    return float(FLOOR_MILESTONES.get(int(floor), 0.0))


# STEP_COST 已移除 — 逐卡即时反馈取代固定惩罚


# ============================================================
# Section 15: Agent (Communication Mod Interface)
# ============================================================

class AgentV6:
    def __init__(self, device=None):
        global DEVICE
        if device is not None:
            DEVICE = torch.device(device)
        self.model = STSModelV6().to_device()
        self.strategy_only = False
        self.strategy_gamma = 0.99
        self.optimizer = optim.AdamW(self.model.parameters(), lr=LR, weight_decay=1e-4)

        # 训练状态
        self.runs = 0
        self.entropy_coeff = ENTROPY_COEFF_START
        self.trajectory_buffer = []
        self.current_trajectory = []
        self.strategy_trajectory = []

        # 游戏状态追踪
        self.in_combat = False
        self.prev_combat_snap = None
        self.current_floor = 0
        self.combat_hp_value_start = 0.0
        self.combat_hp_start = 0  # 绝对 HP，用于选牌回溯反馈
        self.combat_turns = 0

        # 逐牌 reward 追踪
        self.turn_buffer = []  # 当回合出牌 transitions 暂存
        self.turn_hp_start = 0  # 回合开始时的 HP
        self._hand_before_play = Counter()   # 出牌前手牌
        self._energy_before_play = 0         # 出牌前能量
        self._cost_of_played_card = 0        # 打出的牌的费用
        self._draw_credit_map = {}           # 抽到的牌 → turn_buffer 中源牌的 index

        # Anti-stuck
        self._last_sig = ""
        self._stuck = 0
        self._last_screen = ""
        self._screen_repeat = 0

        self._load()

    # --- Save / Load ---
    def _save(self):
        try:
            torch.save({
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "runs": self.runs,
                "entropy_coeff": self.entropy_coeff,
            }, MODEL_PATH)
            log(f"Saved v6 (runs={self.runs}, ent={self.entropy_coeff:.4f})")
        except Exception as e:
            log(f"Save failed: {e}")

    def _load(self):
        if MODEL_PATH.exists():
            try:
                d = torch.load(MODEL_PATH, weights_only=False, map_location=DEVICE)
                self.model.load_state_dict(d["model"])
                self.optimizer.load_state_dict(d["optimizer"])
                self.runs = d.get("runs", 0)
                self.entropy_coeff = d.get("entropy_coeff", ENTROPY_COEFF_START)
                log(f"Loaded v6: {self.runs} runs, ent={self.entropy_coeff:.4f}")
                return
            except Exception as e:
                log(f"Load v6 failed (strict), trying safe load: {e}")
                try:
                    meta = load_model_safe(self.model, MODEL_PATH,
                                           load_optimizer=True, optimizer=self.optimizer)
                    self.runs = meta["runs"]
                    self.entropy_coeff = meta["entropy_coeff"]
                    log(f"Safe loaded v6: {self.runs} runs, ent={self.entropy_coeff:.4f}")
                    return
                except Exception as e2:
                    log(f"Safe load also failed: {e2}")

    # --- State Encoding ---
    def _build_tokens(self, game_state, include_combat=False):
        """构建 token 序列用于 Transformer 输入

        策略模式（draft/path/choice）：
            deck_cards + run_progress + potions + relic_summary

        战斗模式：
            deck_cards + run_progress + potions + relic_summary + player + enemies + hand + piles
        """
        gs = game_state.get("game_state", {})
        tokens = []

        if self.strategy_only:
            include_combat = False

        # 逐卡 token：每张牌作为独立 token 输入
        deck = gs.get("deck", [])
        for card in deck[:MAX_DECK_CARDS]:
            card_vec = encode_card(card)
            tokens.append(("card", TOKEN_DECK_CARD, torch.FloatTensor(card_vec).to(DEVICE)))

        # Run progress
        run_prog = encode_run_progress(game_state)
        tokens.append(("run_progress", TOKEN_RUN_PROGRESS, torch.FloatTensor(run_prog).to(DEVICE)))

        # 药水槽 tokens
        potions = gs.get("potions", [])
        for i in range(MAX_POTION_SLOTS):
            potion_data = potions[i] if i < len(potions) else None
            potion_vec = encode_potion_entity(potion_data)
            tokens.append(("potion", TOKEN_POTION, torch.FloatTensor(potion_vec).to(DEVICE)))

        # 遗物汇总 token（所有遗物编码的均值）
        relics = gs.get("relics", [])
        if relics:
            relic_vecs = np.array([encode_relic_entity(r) for r in relics], dtype=np.float32)
            relic_summary = relic_vecs.mean(axis=0)
        else:
            relic_summary = np.zeros(ENTITY_DIM, dtype=np.float32)
        tokens.append(("relic", TOKEN_RELIC_SUMMARY, torch.FloatTensor(relic_summary).to(DEVICE)))

        if include_combat:
            combat = gs.get("combat_state", {})
            if combat:
                # Player
                player_vec = encode_player_state(combat)
                tokens.append(("player", TOKEN_PLAYER, torch.FloatTensor(player_vec).to(DEVICE)))

                # Monsters
                for m in combat.get("monsters", [])[:MAX_MONSTERS]:
                    enemy_vec = encode_enemy(m)
                    tokens.append(("enemy", TOKEN_MONSTER, torch.FloatTensor(enemy_vec).to(DEVICE)))

                # Hand cards
                for c in combat.get("hand", [])[:MAX_HAND]:
                    hand_vec = encode_card(c)
                    tokens.append(("card", TOKEN_HAND_CARD, torch.FloatTensor(hand_vec).to(DEVICE)))

                # Draw pile summary
                draw_pile = combat.get("draw_pile", gs.get("deck", []))
                if draw_pile:
                    draw_summary = encode_pile_summary(draw_pile)
                    tokens.append(("pile_summary", TOKEN_DRAW_SUMMARY,
                                   torch.FloatTensor(draw_summary).to(DEVICE)))

                # Discard pile summary
                discard = combat.get("discard_pile", [])
                if discard:
                    disc_summary = encode_pile_summary(discard)
                    tokens.append(("pile_summary", TOKEN_DISCARD_SUMMARY,
                                   torch.FloatTensor(disc_summary).to(DEVICE)))

        return tokens

    def _combat_snapshot(self, game_state):
        """完整快照 — 追踪敌人 HP / 玩家 block / buff / debuff 用于逐卡 reward"""
        gs = game_state.get("game_state", {})
        combat = gs.get("combat_state", {})
        if not combat:
            return None
        player = combat.get("player", {})
        monsters = combat.get("monsters", [])

        # 敌人状态
        enemy_hps = []
        enemy_debuffs = 0  # Vulnerable + Weak stacks on enemies
        for m in monsters:
            is_gone = m.get("is_gone", True)
            ehp = m.get("current_hp", 0) if not is_gone else 0
            enemy_hps.append(ehp)
            for p in m.get("powers", []):
                pid = p.get("id", "")
                if pid in ("Vulnerable", "Weak", "Poison"):
                    enemy_debuffs += abs(p.get("amount", 0))

        # 玩家 buff（Strength, Dexterity 等有益状态）
        player_buffs = 0
        for p in player.get("powers", []):
            pid = p.get("id", "")
            if pid in ("Strength", "Dexterity", "Mantra", "Vigor"):
                player_buffs += max(p.get("amount", 0), 0)

        # 敌人 intent（用于评估 block 价值）
        expected_damage = 0
        enemy_intending_attack = False
        for m in monsters:
            if m.get("is_gone", True):
                continue
            intent = m.get("intent", "")
            dmg = m.get("move_adjusted_damage", 0) or 0
            hits = m.get("move_hits", 1) or 1
            if dmg > 0 or "ATTACK" in str(intent).upper():
                enemy_intending_attack = True
                expected_damage += dmg * hits

        return {
            "hp": player.get("current_hp", 0),
            "max_hp": max(player.get("max_hp", 1), 1),
            "block": player.get("block", 0),
            "energy": player.get("energy", 0),
            "enemy_hps": enemy_hps,
            "enemy_debuffs": enemy_debuffs,
            "player_buffs": player_buffs,
            "expected_damage": expected_damage,
            "enemy_intending_attack": enemy_intending_attack,
            "n_alive": sum(1 for h in enemy_hps if h > 0),
        }

    # --- Combat reward helpers ---

    @staticmethod
    def _get_hand_id_counter(hand):
        """手牌 ID 计数器（处理同名牌）"""
        return Counter(c.get("id", "unknown") for c in hand)

    @staticmethod
    def _unique_draw_key(card_id, existing_map):
        """为重复 ID 生成唯一 key"""
        for i in range(100):
            key = f"{card_id}#{i}"
            if key not in existing_map:
                return key
        return f"{card_id}#99"

    @staticmethod
    def _find_draw_key(card_id, draw_credit_map):
        """在 map 中查找匹配的 card_id"""
        for key in list(draw_credit_map.keys()):
            if key.startswith(card_id + "#"):
                return key
        return None

    @staticmethod
    def _apply_draw_credit(turn_buffer):
        """抽牌回传：被抽到的牌的 reward × 0.3 回传给抽牌源"""
        for t in turn_buffer:
            source_idx = t.extra.get("drawn_by_idx", None) if t.extra else None
            if source_idx is not None and 0 <= source_idx < len(turn_buffer):
                credit = t.reward * 0.3
                turn_buffer[source_idx].reward += credit

    # --- Strategy-only (V5 compatible) Methods ---
    def compute_strategy_reward(self, old_hp_val, new_hp_val):
        """计算策略层 transition 的 reward"""
        return float(new_hp_val - old_hp_val)

    def get_strategy_parameters(self):
        """返回 encoder + draft/path/choice head 参数（排除 combat head）"""
        params = []
        params.extend(self.model.ctx_transformer.parameters())
        params.extend(self.model.draft_head.parameters())
        params.extend(self.model.path_head.parameters())
        params.extend(self.model.choice_head.parameters())
        return params

    def configure_strategy_only(self):
        """切换到策略层模式：冻结 combat head，重建 optimizer"""
        self.strategy_only = True
        for p in self.model.combat_head.parameters():
            p.requires_grad = False
        self.optimizer = optim.AdamW(self.get_strategy_parameters(), lr=LR, weight_decay=1e-4)

    def configure_full_training(self):
        """切换到全训练模式：解冻 combat head"""
        self.strategy_only = False
        for p in self.model.combat_head.parameters():
            p.requires_grad = True
        self.optimizer = optim.AdamW(self.model.parameters(), lr=LR, weight_decay=1e-4)

    # --- Decision Methods ---
    def combat_act(self, game_state):
        gs = game_state.get("game_state", {})
        combat = gs.get("combat_state", {})
        hand = combat.get("hand", [])
        monsters = combat.get("monsters", [])
        available = game_state.get("available_commands", [])

        tokens = self._build_tokens(game_state, include_combat=True)

        with torch.no_grad():
            shared_repr = self.model.ctx_transformer(tokens)

            # 手牌编码：通过 token projection 得到 CTX_DIM 向量
            hand_encs = []
            for c in hand[:MAX_HAND]:
                card_vec = encode_card(c)
                t = torch.FloatTensor(card_vec).to(DEVICE)
                enc = self.model.ctx_transformer.token_proj.project("card", TOKEN_HAND_CARD, t)
                hand_encs.append(enc)

            if hand_encs:
                hand_encodings = torch.stack(hand_encs)
            else:
                hand_encodings = torch.zeros(0, CTX_DIM, device=DEVICE)

            # 怪物编码
            monster_encs = []
            alive_monsters = []
            for m in monsters[:MAX_MONSTERS]:
                enemy_vec = encode_enemy(m)
                t = torch.FloatTensor(enemy_vec).to(DEVICE)
                enc = self.model.ctx_transformer.token_proj.project("enemy", TOKEN_MONSTER, t)
                monster_encs.append(enc)
                alive_monsters.append(not m.get("is_gone", True) and m.get("current_hp", 0) > 0)

            if monster_encs:
                monster_encodings = torch.stack(monster_encs)
            else:
                monster_encodings = torch.zeros(0, CTX_DIM, device=DEVICE)

            card_mask = torch.zeros(NUM_CARD_ACTIONS, device=DEVICE)
            for i, c in enumerate(hand[:MAX_HAND]):
                if c.get("is_playable", False):
                    card_mask[i] = 1.0
            if "end" in available:
                card_mask[END_TURN_ACTION] = 1.0
            if card_mask.sum() == 0:
                card_mask[END_TURN_ACTION] = 1.0

            monster_mask = torch.zeros(MAX_MONSTERS, device=DEVICE)
            for i, al in enumerate(alive_monsters):
                if al:
                    monster_mask[i] = 1.0

            card_logits, target_logits, value = self.model.combat_head(
                shared_repr, hand_encodings, monster_encodings, card_mask, monster_mask
            )

        card_dist = torch.distributions.Categorical(logits=card_logits)
        card_action = card_dist.sample().item()
        card_log_prob = card_dist.log_prob(torch.tensor(card_action, device=DEVICE))

        needs_target = False
        target_action = 0
        target_log_prob = torch.tensor(0.0, device=DEVICE)

        if card_action < len(hand) and hand[card_action].get("has_target", False):
            needs_target = True
            with torch.no_grad():
                if card_action < hand_encodings.shape[0]:
                    card_enc = hand_encodings[card_action]
                else:
                    card_enc = torch.zeros(CTX_DIM, device=DEVICE)
                target_logits = self.model.combat_head.forward_with_card(
                    shared_repr, card_enc, monster_encodings, monster_mask
                )
            target_dist = torch.distributions.Categorical(logits=target_logits)
            target_action = target_dist.sample().item()
            target_log_prob = target_dist.log_prob(torch.tensor(target_action, device=DEVICE))

        total_log_prob = (card_log_prob + target_log_prob).item()

        # --- 逐牌 reward：Phase A - 回合切换时 flush turn_buffer ---
        gs_combat = gs.get("combat_state", {})
        current_turn = gs_combat.get("turn", self.combat_turns)
        if self.turn_buffer and current_turn > self.combat_turns:
            # 新回合开始，结算上一回合 reward
            player_now = gs_combat.get("player", {})
            current_hp = player_now.get("current_hp", gs.get("current_hp", 0))
            hp_lost = max(self.turn_hp_start - current_hp, 0)

            # 逐牌算分
            for t in self.turn_buffer:
                card_damage = t.extra.get("card_damage", 0) if t.extra else 0
                card_block = t.extra.get("card_block", 0) if t.extra else 0
                card_kills = t.extra.get("card_kills", 0) if t.extra else 0
                energy_bonus = t.extra.get("energy_bonus", 0) if t.extra else 0
                t.reward = card_damage * 0.05 + card_block * 0.05 + card_kills * 1.0 + energy_bonus

            # 掉血均摊
            n = len(self.turn_buffer)
            if n > 0:
                hp_penalty = hp_lost * 0.1 / n
                for t in self.turn_buffer:
                    t.reward -= hp_penalty

            # 抽牌回传
            self._apply_draw_credit(self.turn_buffer)

            self.current_trajectory.extend(self.turn_buffer)
            self.turn_buffer = []
            self._draw_credit_map = {}
            self.turn_hp_start = current_hp
        self.combat_turns = current_turn

        # --- Phase B - 计算上一张牌的效果（snapshot 差值）---
        new_snap = self._combat_snapshot(game_state)
        if self.prev_combat_snap and new_snap and self.turn_buffer:
            old = self.prev_combat_snap
            last_t = self.turn_buffer[-1]

            # 伤害
            total_hp_before = sum(old["enemy_hps"])
            total_hp_after = sum(new_snap["enemy_hps"])
            damage_dealt = min(max(total_hp_before - total_hp_after, 0), total_hp_before)
            if last_t.extra is None:
                last_t.extra = {}
            last_t.extra["card_damage"] = damage_dealt

            # 格挡
            block_gained = max(new_snap["block"] - old["block"], 0)
            last_t.extra["card_block"] = block_gained

            # 击杀
            kills = max(old["n_alive"] - new_snap["n_alive"], 0)
            last_t.extra["card_kills"] = kills

            # 能量生成
            expected_energy = self._energy_before_play - self._cost_of_played_card
            actual_energy = new_snap["energy"]
            energy_gained = max(actual_energy - expected_energy, 0)
            if energy_gained > 0:
                last_t.extra["energy_bonus"] = energy_gained * 0.1

            # 抽牌检测
            current_hand = self._get_hand_id_counter(gs_combat.get("hand", []))
            expected_hand = self._hand_before_play.copy()
            played_id = last_t.extra.get("played_card_id")
            if played_id and expected_hand[played_id] > 0:
                expected_hand[played_id] -= 1
                if expected_hand[played_id] <= 0:
                    del expected_hand[played_id]
            # 新牌 = 当前手牌 - 预期手牌
            drawn = dict(current_hand)
            for cid, count in expected_hand.items():
                drawn[cid] = drawn.get(cid, 0) - count
                if drawn[cid] <= 0:
                    if cid in drawn:
                        del drawn[cid]
            source_idx = len(self.turn_buffer) - 1
            for cid, count in drawn.items():
                if count > 0:
                    for _ in range(count):
                        key = self._unique_draw_key(cid, self._draw_credit_map)
                        self._draw_credit_map[key] = source_idx

        if new_snap:
            self.prev_combat_snap = new_snap

        # --- Phase C - 记录出牌前状态 ---
        self._hand_before_play = self._get_hand_id_counter(hand)
        player_info = gs_combat.get("player", {})
        self._energy_before_play = player_info.get("energy", 0)
        if card_action < len(hand):
            cost = hand[card_action].get("cost", 0)
            self._cost_of_played_card = max(cost if cost is not None else 0, 0)
        else:
            self._cost_of_played_card = 0

        # --- Phase D - 创建 transition ---
        state_data = {
            "tokens": _detach_tokens(tokens),
            "hand_encodings": hand_encodings.detach().cpu(),
            "monster_encodings": monster_encodings.detach().cpu(),
            "card_mask": card_mask.detach().cpu(),
            "monster_mask": monster_mask.detach().cpu(),
            "needs_target": needs_target,
        }
        extra = {}
        if card_action < len(hand):
            extra["played_card_id"] = hand[card_action].get("id", "unknown")
            # 检查这张牌是否被别的牌抽到的
            draw_key = self._find_draw_key(hand[card_action].get("id", ""), self._draw_credit_map)
            if draw_key is not None:
                extra["drawn_by_idx"] = self._draw_credit_map.pop(draw_key)

        transition = Transition(
            "combat", state_data, (card_action, target_action),
            total_log_prob, value.item(), 0.0, extra=extra
        )
        self.turn_buffer.append(transition)

        if card_action == END_TURN_ACTION:
            log(f"  END")
            return "END"

        cmd = f"PLAY {card_action + 1}"
        if needs_target:
            cmd += f" {target_action}"

        if card_action < len(hand):
            c = hand[card_action]
            target_str = f" -> monster {target_action}" if needs_target else ""
            log(f"  [{c.get('name','?')}] cost={c.get('cost','?')}{target_str}")
        return cmd

    def pick_card(self, game_state):
        """卡牌奖励选择（draft）"""
        gs = game_state.get("game_state", {})
        screen = gs.get("screen_state", {})
        cards = screen.get("cards", [])
        if not cards:
            return "SKIP"

        tokens = self._build_tokens(game_state, include_combat=False)

        candidate_feats = []
        for c in cards:
            card_vec = encode_card(c)
            candidate_feats.append(torch.FloatTensor(card_vec).to(DEVICE))
        candidate_feats.append(torch.zeros(ENTITY_DIM, device=DEVICE))  # skip option

        with torch.no_grad():
            shared_repr = self.model.ctx_transformer(tokens)
            logits, value = self.model.draft_head(shared_repr, candidate_feats)

        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample().item()
        log_prob = dist.log_prob(torch.tensor(action, device=DEVICE)).item()

        # 选牌基础分：按牌面数值给分，上限 0.3
        if action < len(cards):
            card = cards[action]
            dmg = max(card.get("damage", 0) or 0, 0)
            blk = max(card.get("block", 0) or 0, 0)
            cost = max(card.get("cost", 1) or 1, 1)
            base_reward = min((dmg + blk) / cost * 0.05, 0.3)
        else:
            base_reward = 0.0
        reward = base_reward

        extra = {"card_id": cards[action].get("id", "unknown")} if action < len(cards) else {"card_id": "skip"}
        state_data = {"tokens": _detach_tokens(tokens), "candidate_features": [f.detach().cpu() for f in candidate_feats]}
        self.current_trajectory.append(Transition(
            "draft", state_data, (action,), log_prob, value.item(), reward, extra=extra
        ))

        if action == len(cards):
            log(f"  SKIP card")
            return "SKIP"

        log(f"  Pick [{cards[action].get('name')}]")
        return f"CHOOSE {action}"

    def _pick_from_candidates(self, game_state, cards, allow_skip=True):
        """用 draft head 从候选卡中选择一张（升级/删牌等场景复用）"""
        if not cards:
            return "CHOOSE 0"

        tokens = self._build_tokens(game_state, include_combat=False)

        candidate_feats = []
        for c in cards:
            card_vec = encode_card(c)
            candidate_feats.append(torch.FloatTensor(card_vec).to(DEVICE))

        if allow_skip:
            candidate_feats.append(torch.zeros(ENTITY_DIM, device=DEVICE))

        with torch.no_grad():
            shared_repr = self.model.ctx_transformer(tokens)
            logits, value = self.model.draft_head(shared_repr, candidate_feats)

        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample().item()
        log_prob = dist.log_prob(torch.tensor(action, device=DEVICE)).item()

        reward = 0.0  # 由战斗结果回溯反馈
        extra = {"card_id": cards[action].get("id", "unknown")} if action < len(cards) else {"card_id": "skip"}
        state_data = {"tokens": _detach_tokens(tokens), "candidate_features": [f.detach().cpu() for f in candidate_feats]}
        self.current_trajectory.append(Transition(
            "draft", state_data, (action,), log_prob, value.item(), reward, extra=extra
        ))

        if allow_skip and action == len(cards):
            log(f"  SKIP (from candidates)")
            return "SKIP" if "skip" in game_state.get("available_commands", []) else "CANCEL"

        if action >= len(cards):
            action = 0  # safety fallback

        log(f"  Pick candidate [{cards[action].get('name', cards[action].get('id', '?'))}]")
        return f"CHOOSE {action}"

    def _pick_worst_candidate(self, game_state, cards):
        """选择得分最低的卡牌（删牌/purge 场景）"""
        if not cards:
            return "CHOOSE 0"

        tokens = self._build_tokens(game_state, include_combat=False)

        candidate_feats = []
        for c in cards:
            card_vec = encode_card(c)
            candidate_feats.append(torch.FloatTensor(card_vec).to(DEVICE))

        with torch.no_grad():
            shared_repr = self.model.ctx_transformer(tokens)
            logits, value = self.model.draft_head(shared_repr, candidate_feats)

        action = logits.argmin().item()

        dist = torch.distributions.Categorical(logits=logits)
        log_prob = dist.log_prob(torch.tensor(action, device=DEVICE)).item()

        reward = 0.0  # 由战斗结果回溯反馈
        extra = {"card_id": cards[action].get("id", "unknown") if action < len(cards) else "purge_skip"}
        state_data = {"tokens": _detach_tokens(tokens), "candidate_features": [f.detach().cpu() for f in candidate_feats]}
        self.current_trajectory.append(Transition(
            "draft", state_data, (action,), log_prob, value.item(), reward, extra=extra
        ))

        if action >= len(cards):
            action = 0
        log(f"  Purge worst [{cards[action].get('name', cards[action].get('id', '?'))}]")
        return f"CHOOSE {action}"

    def _pick_boss_relic(self, game_state, relics):
        """用 ENTITY_DIM 编码 + choice head 评估 Boss 遗物选择"""
        tokens = self._build_tokens(game_state, include_combat=False)

        gs = game_state.get("game_state", {})
        hp_ratio = float(gs.get("current_hp", 1) / max(gs.get("max_hp", 1), 1))

        option_encs = []
        for i, r in enumerate(relics):
            base_enc = encode_choice_option(
                "boss_relic",
                option_text=r.get("name", r.get("id", f"relic_{i}")),
                option_index=i,
                max_options=len(relics),
                hp_ratio=hp_ratio,
            )
            # 将 relic 效果向量注入 choice 编码的可用空间
            relic_vec = encode_relic_entity(r)
            # 覆盖 [10:min(10+ENTITY_DIM, 39)]
            end_idx = min(10 + ENTITY_DIM, CHOICE_OPTION_DIM - 1)
            base_enc[10:end_idx] = relic_vec[:end_idx - 10]

            option_encs.append(torch.FloatTensor(base_enc).to(DEVICE))

        with torch.no_grad():
            shared_repr = self.model.ctx_transformer(tokens)
            logits, value = self.model.choice_head(shared_repr, option_encs)

        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample().item()
        log_prob = dist.log_prob(torch.tensor(action, device=DEVICE)).item()

        reward = 0.0
        state_data = {"tokens": _detach_tokens(tokens), "option_encodings": [e.detach().cpu() for e in option_encs]}
        self.current_trajectory.append(Transition(
            "choice", state_data, (action,), log_prob, value.item(), reward
        ))

        chosen_name = relics[action].get("name", relics[action].get("id", f"relic_{action}"))
        log(f"  Boss relic: [{chosen_name}]")
        return f"CHOOSE {action}"

    def choose_path(self, game_state):
        gs = game_state.get("game_state", {})
        screen = gs.get("screen_state", {})

        full_paths = screen.get("full_paths", [])
        nodes = screen.get("next_nodes", [])

        if not full_paths and not nodes:
            available = game_state.get("available_commands", [])
            return "CHOOSE 0" if "choose" in available else "PROCEED"

        tokens = self._build_tokens(game_state, include_combat=False)

        hp_ratio = float(gs.get("current_hp", 1) / max(gs.get("max_hp", 1), 1))
        deck_size = len(gs.get("deck", []))
        gold = gs.get("gold", 0)
        act = gs.get("act", 1)

        node_feats = []
        if full_paths:
            for path in full_paths:
                rooms = path.get("rooms", [])
                feat = encode_full_path(rooms, hp_ratio, deck_size, gold, act)
                node_feats.append(torch.FloatTensor(feat).to(DEVICE))
        else:
            for node in nodes:
                feat_short = encode_path_node(node)
                feat_full = np.zeros(PATH_FEATURE_DIM, dtype=np.float32)
                feat_full[:len(feat_short)] = feat_short
                feat_full[10] = hp_ratio
                feat_full[11] = min(deck_size / 30.0, 1.5)
                feat_full[12] = min(gold / 300.0, 1.0)
                feat_full[13] = min(act / 3.0, 1.0)
                node_feats.append(torch.FloatTensor(feat_full).to(DEVICE))

        with torch.no_grad():
            shared_repr = self.model.ctx_transformer(tokens)
            logits, value = self.model.path_head(shared_repr, node_feats)

        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample().item()
        log_prob = dist.log_prob(torch.tensor(action, device=DEVICE)).item()

        # 路径选择即时奖励：给非零信号，楼层奖励提供长期信号
        reward = 0.1

        state_data = {"tokens": _detach_tokens(tokens), "node_features": [f.detach().cpu() for f in node_feats]}
        self.current_trajectory.append(Transition(
            "path", state_data, (action,), log_prob, value.item(), reward
        ))

        label = full_paths[action].get("start_node", "?") if full_paths else nodes[action].get("symbol", "?")
        log(f"  Path [{label}]")
        return f"CHOOSE {action}"

    def make_choice(self, game_state, choice_type, options_info):
        if not options_info:
            return "CHOOSE 0"

        tokens = self._build_tokens(game_state, include_combat=False)

        gs = game_state.get("game_state", {})
        hp_ratio = float(gs.get("current_hp", 1) / max(gs.get("max_hp", 1), 1))

        option_encs = []
        for i, opt in enumerate(options_info):
            enc = encode_choice_option(
                choice_type,
                option_text=opt.get("text", ""),
                option_index=i,
                max_options=len(options_info),
                hp_ratio=hp_ratio,
                **{k: v for k, v in opt.items() if k not in ("text", "command")}
            )
            option_encs.append(torch.FloatTensor(enc).to(DEVICE))

        with torch.no_grad():
            shared_repr = self.model.ctx_transformer(tokens)
            logits, value = self.model.choice_head(shared_repr, option_encs)

        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample().item()
        log_prob = dist.log_prob(torch.tensor(action, device=DEVICE)).item()

        # 事件/休息/商店即时奖励：休息按回复比例，其余给基础信号
        if choice_type == "rest" or choice_type == "campfire":
            current_hp = gs.get("current_hp", 1)
            max_hp = max(gs.get("max_hp", 1), 1)
            heal_pct = min((max_hp - current_hp) / max_hp, 1.0)
            reward = 0.1 * heal_pct  # 缺血越多休息越有价值
        else:
            reward = 0.1

        state_data = {"tokens": _detach_tokens(tokens), "option_encodings": [e.detach().cpu() for e in option_encs]}
        self.current_trajectory.append(Transition(
            "choice", state_data, (action,), log_prob, value.item(), reward
        ))

        chosen = options_info[action]
        log(f"  Choice: {chosen.get('text', f'option {action}')}")
        return chosen.get("command", f"CHOOSE {action}")

    def potion_decide(self, game_state):
        """战斗中药水决策"""
        gs = game_state.get("game_state", {})
        potions = gs.get("potions", [])
        combat_state = gs.get("combat_state", {})

        available = []
        for i, p in enumerate(potions):
            pid = p.get("id", "")
            if pid and pid != "Potion Slot":
                available.append((i, p))

        if not available:
            return "SKIP"

        tokens = self._build_tokens(game_state, include_combat=True)
        hp_ratio = float(gs.get("current_hp", 1) / max(gs.get("max_hp", 1), 1))

        option_encs = []
        options_info = []
        for slot_idx, p in available:
            pid = p.get("id", "")
            enc = encode_choice_option(
                "potion_use",
                option_text=pid,
                option_index=len(options_info),
                max_options=len(available) + 1,
                hp_ratio=hp_ratio,
            )
            # 注入药水效果向量
            potion_vec = encode_potion_entity(p)
            end_idx = min(30 + ENTITY_DIM, CHOICE_OPTION_DIM - 1)
            enc[30:end_idx] = potion_vec[:end_idx - 30]
            option_encs.append(torch.FloatTensor(enc).to(DEVICE))

            # 确定目标
            target_idx = -1
            potion_vec_check = potion_vec
            # 如果药水有伤害或 debuff 效果，瞄准最强怪物
            if potion_vec_check[_DIM_INDEX.get("damage_single", 15)] > 0 or \
               potion_vec_check[_DIM_INDEX.get("vulnerable_apply", 24)] > 0 or \
               potion_vec_check[_DIM_INDEX.get("weak_apply", 25)] > 0 or \
               potion_vec_check[_DIM_INDEX.get("poison_apply", 18)] > 0:
                monsters = combat_state.get("monsters", [])
                best_hp = -1
                for mi, m in enumerate(monsters):
                    if not m.get("is_gone", False) and m.get("current_hp", 0) > best_hp:
                        best_hp = m.get("current_hp", 0)
                        target_idx = mi

            options_info.append({
                "text": pid,
                "command": f"POTION {slot_idx}" + (f" {target_idx}" if target_idx >= 0 else ""),
            })

        # 跳过选项
        skip_enc = encode_choice_option(
            "potion_use",
            option_text="skip",
            option_index=len(options_info),
            max_options=len(available) + 1,
            hp_ratio=hp_ratio,
        )
        option_encs.append(torch.FloatTensor(skip_enc).to(DEVICE))
        options_info.append({"text": "skip", "command": "SKIP"})

        with torch.no_grad():
            shared_repr = self.model.ctx_transformer(tokens)
            logits, value = self.model.choice_head(shared_repr, option_encs)

        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample().item()
        log_prob = dist.log_prob(torch.tensor(action, device=DEVICE)).item()

        reward = 0.0
        state_data = {"tokens": _detach_tokens(tokens), "option_encodings": [e.detach().cpu() for e in option_encs]}
        self.current_trajectory.append(Transition(
            "choice", state_data, (action,), log_prob, value.item(), reward
        ))

        chosen = options_info[action]
        cmd = chosen.get("command", "SKIP")
        log(f"  Potion: {chosen.get('text', 'skip')}")
        return cmd

    def shop_decide(self, game_state):
        """商店决策：用 choice head 评估所有商店选项"""
        gs = game_state.get("game_state", game_state)
        gold = gs.get("gold", 0)
        screen_state = gs.get("screen_state", {})
        available = game_state.get("available_commands", [])

        tokens = self._build_tokens(game_state, include_combat=False)

        shop_options = []

        # 商店卡牌
        cards = screen_state.get("cards", [])
        for i, c in enumerate(cards):
            item = {"id": c.get("id", ""), "price": c.get("price", 999), "type": "card"}
            enc = encode_shop_item(item, player_gold=gold)
            shop_options.append((enc, f"CHOOSE {i}", f"buy card {c.get('id', '?')}"))

        # 商店遗物
        relics = screen_state.get("relics", [])
        for i, r in enumerate(relics):
            item = {"id": r.get("id", ""), "price": r.get("price", 999), "type": "relic"}
            enc = encode_shop_item(item, player_gold=gold)
            shop_options.append((enc, f"CHOOSE {len(cards) + i}", f"buy relic {r.get('id', '?')}"))

        # 商店药水
        shop_potions = screen_state.get("potions", [])
        for i, p in enumerate(shop_potions):
            item = {"id": p.get("id", ""), "price": p.get("price", 999), "type": "potion"}
            enc = encode_shop_item(item, player_gold=gold)
            shop_options.append((enc, f"CHOOSE {len(cards) + len(relics) + i}", f"buy potion {p.get('id', '?')}"))

        # 删牌选项
        purge_available = screen_state.get("purge_available", False)
        purge_cost = screen_state.get("purge_cost", 9999)
        if purge_available:
            item = {"id": "card_remove", "price": purge_cost, "type": "remove"}
            enc = encode_shop_item(item, player_gold=gold)
            shop_options.append((enc, "PURGE", "remove card"))

        # 离开选项
        leave_item = {"id": "leave_shop", "price": 0, "type": "leave"}
        leave_enc = encode_shop_item(leave_item, player_gold=gold)
        leave_cmd = "RETURN" if "return" in [a.lower() for a in available] else "LEAVE"
        shop_options.append((leave_enc, leave_cmd, "leave shop"))

        if not shop_options:
            return leave_cmd

        option_encs = [torch.FloatTensor(enc).to(DEVICE) for enc, _, _ in shop_options]

        with torch.no_grad():
            shared_repr = self.model.ctx_transformer(tokens)
            logits, value = self.model.choice_head(shared_repr, option_encs)

        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample().item()
        log_prob = dist.log_prob(torch.tensor(action, device=DEVICE)).item()

        reward = 0.0
        state_data = {"tokens": _detach_tokens(tokens), "option_encodings": [e.detach().cpu() for e in option_encs]}
        self.current_trajectory.append(Transition(
            "choice", state_data, (action,), log_prob, value.item(), reward
        ))

        chosen_enc, chosen_cmd, chosen_desc = shop_options[action]
        log(f"  Shop: {chosen_desc} (gold={gold})")
        return chosen_cmd

    # --- Event handlers ---
    def on_combat_end(self, game_state, won):
        gs = game_state.get("game_state", {})
        hp = gs.get("current_hp", 0)

        # --- flush 最后一回合的 turn_buffer ---
        if self.turn_buffer:
            # 先算最后一张牌的效果
            new_snap = self._combat_snapshot(game_state)
            if self.prev_combat_snap and new_snap and self.turn_buffer:
                old = self.prev_combat_snap
                last_t = self.turn_buffer[-1]
                if last_t.extra is None:
                    last_t.extra = {}
                total_hp_before = sum(old["enemy_hps"])
                total_hp_after = sum(new_snap["enemy_hps"])
                damage_dealt = min(max(total_hp_before - total_hp_after, 0), total_hp_before)
                last_t.extra["card_damage"] = damage_dealt
                block_gained = max(new_snap["block"] - old["block"], 0)
                last_t.extra["card_block"] = block_gained
                kills = max(old["n_alive"] - new_snap["n_alive"], 0)
                last_t.extra["card_kills"] = kills

            # 逐牌算分
            for t in self.turn_buffer:
                card_damage = t.extra.get("card_damage", 0) if t.extra else 0
                card_block = t.extra.get("card_block", 0) if t.extra else 0
                card_kills = t.extra.get("card_kills", 0) if t.extra else 0
                energy_bonus = t.extra.get("energy_bonus", 0) if t.extra else 0
                t.reward = card_damage * 0.05 + card_block * 0.05 + card_kills * 1.0 + energy_bonus

            # 掉血均摊
            hp_lost = max(self.turn_hp_start - hp, 0)
            n = len(self.turn_buffer)
            if n > 0:
                hp_penalty = hp_lost * 0.1 / n
                for t in self.turn_buffer:
                    t.reward -= hp_penalty

            self._apply_draw_credit(self.turn_buffer)
            self.current_trajectory.extend(self.turn_buffer)
            self.turn_buffer = []

        if not won and hp == 0 and self.current_trajectory:
            for t in reversed(self.current_trajectory):
                if t.decision_type == "combat":
                    t.done = True
                    break

        # --- 选牌回溯反馈：根据战斗 HP 变化更新之前的 draft 决策 ---
        hp_after = gs.get("current_hp", 0)
        hp_before = self.combat_hp_start
        if hp_before > 0 and self.current_trajectory:
            # 判断敌人难度
            monsters = gs.get("combat_state", {}).get("monsters", [])
            if not monsters:
                monsters = []
            difficulty = 1.0
            # Boss 检测：monster id 包含已知 boss 名
            _BOSS_IDS = {"slime_boss", "hexaghost", "guardian", "automaton", "collector",
                         "champ", "awakened", "time_eater", "donu", "deca",
                         "heart", "corrupt_heart", "the_heart"}
            for m in monsters:
                mid = m.get("id", "").lower().replace(" ", "_")
                if any(bid in mid for bid in _BOSS_IDS):
                    difficulty = 3.0
                    break
            # Elite 检测（如果不是 boss）：任意怪物 max_hp > 100
            if difficulty < 3.0:
                for m in monsters:
                    if m.get("max_hp", 0) > 100:
                        difficulty = 2.0
                        break

            hp_delta = (hp_after - hp_before) / 20.0  # 归一化：20HP = 1.0 reward unit
            n_combats_back = 0
            in_combat_chunk = True  # 当前位置是 combat 区域
            for i in range(len(self.current_trajectory) - 1, -1, -1):
                t = self.current_trajectory[i]
                if t.decision_type == "combat":
                    if not in_combat_chunk:
                        # 进入了新的 combat 区域 → 跨过了一个 combat 边界
                        n_combats_back += 1
                        in_combat_chunk = True
                    continue
                else:
                    in_combat_chunk = False

                if t.decision_type == "draft":
                    if n_combats_back > 5:
                        break  # 衰减太小，停止扫描
                    decay = 0.7 ** n_combats_back
                    retroactive_reward = hp_delta * difficulty * decay
                    t.reward += retroactive_reward
                    card_id = t.extra.get("card_id", "?") if t.extra else "?"
                    log(f"Draft retroactive: card={card_id} reward={retroactive_reward:.3f} "
                        f"difficulty={difficulty} decay={decay:.3f}")

        self.in_combat = False
        self.prev_combat_snap = None
        self.combat_hp_value_start = 0.0
        self.combat_hp_start = 0
        self.combat_turns = 0
        self.turn_buffer = []
        self.turn_hp_start = 0
        self._hand_before_play = Counter()
        self._energy_before_play = 0
        self._cost_of_played_card = 0
        self._draw_credit_map = {}
        log(f"{'WIN' if won else 'LOSE'} combat")

    def on_floor_cleared(self):
        reward = floor_milestone_reward(self.current_floor)
        if reward > 0.0 and self.current_trajectory:
            self.current_trajectory[-1].reward += reward
        self.current_floor += 1

    def on_run_end(self, game_state):
        gs = game_state.get("game_state", {})
        hp = gs.get("current_hp", 0)
        floor = gs.get("floor", 0)
        won = hp > 0
        if not self.strategy_only:
            self.runs += 1

        terminal_r = 50.0 if won else 0.0
        if self.current_trajectory:
            self.current_trajectory[-1].reward += terminal_r
            self.current_trajectory[-1].done = True

        import time
        stats_line = (f"{time.strftime('%m-%d %H:%M')} | "
                      f"run={self.runs:4d} | {'WIN ' if won else 'LOSE'} | "
                      f"floor={floor:2d} | HP={hp:3d}/{gs.get('max_hp',0):3d} | "
                      f"traj={len(self.current_trajectory)} | ent={self.entropy_coeff:.4f}")
        log(f"END {stats_line}")
        with open(BASE_DIR / "sts_v6_stats.log", "a") as f:
            f.write(stats_line + "\n")

        if self.current_trajectory:
            self.trajectory_buffer.append(self.current_trajectory)
        self.current_trajectory = []
        self.strategy_trajectory = []
        self.current_floor = 0
        self.combat_hp_value_start = 0.0
        self.combat_hp_start = 0
        self.combat_turns = 0
        self.turn_buffer = []
        self.turn_hp_start = 0
        self._hand_before_play = Counter()
        self._energy_before_play = 0
        self._cost_of_played_card = 0
        self._draw_credit_map = {}

        if len(self.trajectory_buffer) >= N_RUNS_PER_UPDATE and not getattr(self, 'suppress_ppo', False):
            log(f"PPO update: {len(self.trajectory_buffer)} runs")
            if self.strategy_only:
                strategy_params = self.get_strategy_parameters()
                stats = ppo_update(
                    self.model, self.optimizer, self.trajectory_buffer, self.entropy_coeff,
                    strategy_only=True, strategy_gamma=self.strategy_gamma,
                    strategy_params=strategy_params
                )
            else:
                stats = ppo_update(self.model, self.optimizer, self.trajectory_buffer, self.entropy_coeff)
            log(f"  policy_loss={stats.get('policy_loss',0):.4f} "
                f"value_loss={stats.get('value_loss',0):.4f} "
                f"entropy={stats.get('entropy',0):.4f} "
                f"n={stats.get('n_transitions',0)}")
            self.trajectory_buffer = []

            self.entropy_coeff = max(
                ENTROPY_COEFF_END,
                ENTROPY_COEFF_START - (ENTROPY_COEFF_START - ENTROPY_COEFF_END) * self.runs / ENTROPY_ANNEAL_RUNS
            )

        if not self.strategy_only:
            self._save()

    # --- Main Decision ---
    def decide(self, game_state):
        available = game_state.get("available_commands", [])
        gs = game_state.get("game_state", {})
        screen = gs.get("screen_type", "")
        screen_state = gs.get("screen_state", {})
        in_game = game_state.get("in_game", False)

        log(f"screen: {screen} | cmds: {available}")

        # Anti-stuck — 战斗中禁用（combat_state 存在说明在战斗），只对非战斗画面生效
        in_combat_now = bool(gs.get("combat_state"))
        if not in_combat_now:
            sig = f"{screen}|{'|'.join(sorted(available))}"
            if sig == self._last_sig:
                self._stuck += 1
            else:
                self._stuck = 0
            self._last_sig = sig

            if screen == self._last_screen:
                self._screen_repeat += 1
            else:
                self._screen_repeat = 0
            self._last_screen = screen

            if self._stuck > 3 or self._screen_repeat > 10:
                log(f"Stuck (sig={self._stuck}, screen={self._screen_repeat})")
                self._stuck = 0
                self._screen_repeat = 0
                for c, s in [("proceed","PROCEED"),("skip","SKIP"),("confirm","CONFIRM"),
                             ("cancel","CANCEL"),("leave","LEAVE"),("return","RETURN"),
                             ("choose","CHOOSE 0"),("end","END")]:
                    if c in available:
                        return s
                return "STATE"
        else:
            # 战斗中重置 stuck 计数，避免离开战斗时误触发
            self._stuck = 0
            self._screen_repeat = 0
            self._last_sig = ""
            self._last_screen = screen

        # Combat end detection
        if self.in_combat and "play" not in available and "end" not in available:
            won = gs.get("current_hp", 0) > 0
            self.on_combat_end(game_state, won)

        # Main menu
        if not in_game:
            if "start" in available:
                log("New game IRONCLAD A0")
                self.current_trajectory = []
                self.current_floor = 0
                self.combat_hp_value_start = 0.0
                self.combat_hp_start = 0
                self.combat_turns = 0
                self.turn_buffer = []
                self.turn_hp_start = 0
                self._hand_before_play = Counter()
                self._energy_before_play = 0
                self._cost_of_played_card = 0
                self._draw_credit_map = {}
                return "START ironclad 0"
            return "STATE"

        # Combat
        if "play" in available or ("end" in available and "choose" not in available):
            if not self.in_combat:
                self.in_combat = True
                hp = gs.get("current_hp", 0)
                max_hp = max(gs.get("max_hp", 1), 1)
                self.combat_hp_value_start = hp_value(hp, max_hp)
                self.combat_hp_start = hp  # 绝对 HP，用于选牌反馈
                self.combat_turns = 0
                # 逐牌 reward 初始化
                self.turn_buffer = []
                self.turn_hp_start = hp
                self._hand_before_play = Counter()
                self._energy_before_play = 0
                self._cost_of_played_card = 0
                self._draw_credit_map = {}
            combat_st = gs.get("combat_state", {})
            self.combat_turns = combat_st.get("turn", self.combat_turns)
            if self.prev_combat_snap is None:
                self.prev_combat_snap = self._combat_snapshot(game_state)
            if "play" in available:
                return self.combat_act(game_state)
            return "END"

        # Card Reward
        if screen == "CARD_REWARD":
            return self.pick_card(game_state)

        # Combat Reward
        if screen == "COMBAT_REWARD":
            if not self.strategy_only:
                self.on_floor_cleared()
            reward_cards = screen_state.get("reward_cards", [])
            if reward_cards and "choose" in available:
                result = self._pick_from_candidates(game_state, reward_cards, allow_skip=True)
                if result == "SKIP":
                    if "proceed" in available:
                        return "PROCEED"
                    return "SKIP"
                return result
            if "proceed" in available:
                return "PROCEED"
            if "choose" in available:
                return "CHOOSE 0"

        # Map
        if screen == "MAP":
            return self.choose_path(game_state)

        # Rest Site
        if screen == "REST":
            hp_ratio = float(gs.get("current_hp", 1) / max(gs.get("max_hp", 1), 1))
            options = [
                {"text": "rest", "command": "CHOOSE rest" if "choose" in available else "REST",
                 "extra": hp_ratio},
                {"text": "smith", "command": "CHOOSE smith" if "choose" in available else "SMITH",
                 "extra": 1.0 - hp_ratio},
            ]
            player_relics = {r.get("id", r.get("name", "")) for r in gs.get("relics", [])}
            rest_opts_from_sim = screen_state.get("rest_options", [])
            if "Shovel" in player_relics or "dig" in rest_opts_from_sim:
                options.append({"text": "dig", "command": "CHOOSE dig", "extra": 0.3})
            if "Girya" in player_relics or "lift" in rest_opts_from_sim:
                options.append({"text": "lift", "command": "CHOOSE lift", "extra": 0.5})
            if "Peace Pipe" in player_relics or "toke" in rest_opts_from_sim:
                options.append({"text": "toke", "command": "CHOOSE toke", "extra": 0.6})
            log(f"  Rest options: {[o['text'] for o in options]}")
            return self.make_choice(game_state, "rest", options)

        # Neow Blessing
        floor = gs.get("floor", -1)
        event_id = screen_state.get("event_id", "")
        is_neow = ("Neow" in event_id) or (floor == 0 and "choose" in available and screen in ("NONE", "EVENT", ""))
        if is_neow and "choose" in available:
            options = screen_state.get("options", [])
            if not options:
                options = screen_state.get("choices", [])
            if options:
                options_info = []
                for i, opt in enumerate(options):
                    if isinstance(opt, dict):
                        text = opt.get("label", opt.get("text", str(i)))
                    else:
                        text = str(opt)
                    options_info.append({"text": text, "command": f"CHOOSE {i}"})
                log(f"  Neow blessing: {len(options_info)} options")
                return self.make_choice(game_state, "neow", options_info)
            return "CHOOSE 0"

        # Event
        if screen == "EVENT" and "choose" in available:
            options = screen_state.get("options", [])
            if options:
                options_info = []
                for i, opt in enumerate(options):
                    text = opt.get("label", opt.get("text", str(i)))
                    options_info.append({"text": text, "command": f"CHOOSE {i}"})
                return self.make_choice(game_state, "event", options_info)
            return "CHOOSE 0"

        # Shop
        if screen == "SHOP_SCREEN":
            return self.shop_decide(game_state)

        # Grid / Hand Select
        if screen in ("GRID", "HAND_SELECT"):
            if "confirm" in available:
                return "CONFIRM"
            if "choose" in available:
                cards = screen_state.get("cards", [])
                if not cards:
                    cards = screen_state.get("selected_cards", [])
                if cards and len(cards) > 1:
                    select_type = screen_state.get("select_type", "")
                    is_purge = select_type == "purge" or "remove" in select_type
                    if is_purge:
                        return self._pick_worst_candidate(game_state, cards)
                    return self._pick_from_candidates(game_state, cards, allow_skip=False)
                return "CHOOSE 0"
            if "cancel" in available:
                return "CANCEL"

        # Boss Relic
        if screen == "BOSS_REWARD" and "choose" in available:
            relics = screen_state.get("relics", [])
            if relics:
                return self._pick_boss_relic(game_state, relics)
            return "CHOOSE 0"

        # Chest
        if screen == "CHEST":
            if "choose" in available:
                return "CHOOSE 0"
            if "proceed" in available:
                return "PROCEED"

        # Game Over
        if screen in ("GAME_OVER", "DEATH", "VICTORY", "COMPLETE"):
            self.on_run_end(game_state)
            for c in ["proceed", "confirm", "return", "leave"]:
                if c in available:
                    return c.upper()

        # Fallback
        for c, s in [("proceed","PROCEED"),("confirm","CONFIRM"),("skip","SKIP"),
                     ("cancel","CANCEL"),("leave","LEAVE"),("return","RETURN"),
                     ("choose","CHOOSE 0"),("end","END")]:
            if c in available:
                return s

        return "STATE"


# ============================================================
# Section 16: Main Loop
# ============================================================

def main():
    log("=" * 60)
    log("Agent v6 -- Unified Effect Encoding + PPO + Transformer")
    agent = AgentV6()

    send("ready")
    log("Waiting for game...")
    try:
        while True:
            state = receive()
            if state is None:
                break
            if "error" in state:
                err = state["error"]
                log(f"ERROR: {err}")
                if state.get("ready_for_command", False):
                    avail = state.get("available_commands", [])
                    for c, s in [("choose","CHOOSE 0"),("confirm","CONFIRM"),
                                 ("proceed","PROCEED"),("skip","SKIP"),
                                 ("cancel","CANCEL"),("return","RETURN"),
                                 ("end","END"),("leave","LEAVE")]:
                        if c in avail:
                            log(f"  Recovery: {s}")
                            send(s)
                            break
                continue
            if not state.get("ready_for_command", False):
                continue
            try:
                cmd = agent.decide(state)
            except Exception as e:
                log(f"decide error: {e}")
                import traceback
                log(traceback.format_exc())
                avail = state.get("available_commands", [])
                cmd = "STATE"
                for c, s in [("choose","CHOOSE 0"),("confirm","CONFIRM"),
                             ("proceed","PROCEED"),("skip","SKIP"),
                             ("end","END"),("leave","LEAVE")]:
                    if c in avail:
                        cmd = s
                        break
            log(f"  -> {cmd}")
            send(cmd)
    except KeyboardInterrupt:
        pass
    finally:
        agent._save()
        log("Exit")


# ============================================================
# Section 17: Validation
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("STS Agent V6 -- Validation")
    print("=" * 60)

    # 1. 创建模型
    model = STSModelV6()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel created successfully")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  ENTITY_DIM (unified): {ENTITY_DIM}")
    print(f"  PLAYER_DIM: {PLAYER_DIM}")
    print(f"  ENEMY_DIM: {ENEMY_DIM}")
    print(f"  PILE_TOTAL_DIM: {PILE_TOTAL_DIM}")
    print(f"  CTX_DIM: {CTX_DIM}")
    print(f"  SHARED_REPR_DIM: {SHARED_REPR_DIM}")

    # 2. 构建 dummy game state
    dummy_game_state = {
        "game_state": {
            "floor": 5,
            "act": 1,
            "current_hp": 60,
            "max_hp": 80,
            "gold": 100,
            "ascension_level": 0,
            "screen_type": "NONE",
            "screen_state": {},
            "deck": [
                {"id": "Strike_R", "name": "Strike", "type": "ATTACK", "cost": 1, "damage": 6},
                {"id": "Defend_R", "name": "Defend", "type": "SKILL", "cost": 1, "block": 5},
                {"id": "Bash", "name": "Bash", "type": "ATTACK", "cost": 2, "damage": 8},
            ],
            "potions": [
                {"id": "Fire Potion", "name": "Fire Potion"},
                {"id": "Potion Slot"},
                {"id": "Potion Slot"},
            ],
            "relics": [
                {"id": "Burning Blood", "name": "Burning Blood", "counter": 0},
            ],
            "combat_state": {
                "turn": 1,
                "player": {
                    "current_hp": 60,
                    "max_hp": 80,
                    "block": 5,
                    "energy": 3,
                    "powers": [
                        {"id": "Strength", "amount": 2},
                        {"id": "Vulnerable", "amount": 1},
                    ],
                    "stance": "",
                    "orbs": [],
                    "orb_slots": 0,
                },
                "monsters": [
                    {
                        "id": "JawWorm",
                        "name": "Jaw Worm",
                        "current_hp": 30,
                        "max_hp": 44,
                        "block": 0,
                        "intent": "ATTACK",
                        "move_adjusted_damage": 11,
                        "move_hits": 1,
                        "is_gone": False,
                        "powers": [{"id": "Strength", "amount": 3}],
                    },
                ],
                "hand": [
                    {"id": "Strike_R", "name": "Strike", "type": "ATTACK", "cost": 1, "damage": 6, "is_playable": True, "has_target": True},
                    {"id": "Defend_R", "name": "Defend", "type": "SKILL", "cost": 1, "block": 5, "is_playable": True, "has_target": False},
                    {"id": "Bash", "name": "Bash", "type": "ATTACK", "cost": 2, "damage": 8, "is_playable": True, "has_target": True},
                ],
                "draw_pile": [
                    {"id": "Strike_R", "type": "ATTACK", "cost": 1},
                    {"id": "Defend_R", "type": "SKILL", "cost": 1},
                ],
                "discard_pile": [
                    {"id": "Strike_R", "type": "ATTACK", "cost": 1},
                ],
            },
        },
        "available_commands": ["play", "end"],
        "ready_for_command": True,
        "in_game": True,
    }

    # 3. 测试编码函数
    print("\n--- Encoding Tests ---")

    card_vec = encode_card({"id": "Bash", "type": "ATTACK", "cost": 2, "damage": 8})
    print(f"  encode_card('Bash'): shape={card_vec.shape}, nonzero={np.count_nonzero(card_vec)}")

    relic_vec = encode_relic_entity({"id": "Burning Blood"})
    print(f"  encode_relic('Burning Blood'): shape={relic_vec.shape}, nonzero={np.count_nonzero(relic_vec)}")

    potion_vec = encode_potion_entity({"id": "Fire Potion"})
    print(f"  encode_potion('Fire Potion'): shape={potion_vec.shape}, nonzero={np.count_nonzero(potion_vec)}")

    player_vec = encode_player_state(dummy_game_state["game_state"]["combat_state"])
    print(f"  encode_player_state: shape={player_vec.shape}, nonzero={np.count_nonzero(player_vec)}")

    enemy_vec = encode_enemy(dummy_game_state["game_state"]["combat_state"]["monsters"][0])
    print(f"  encode_enemy: shape={enemy_vec.shape}, nonzero={np.count_nonzero(enemy_vec)}")

    run_prog = encode_run_progress(dummy_game_state)
    print(f"  encode_run_progress: shape={run_prog.shape}, nonzero={np.count_nonzero(run_prog)}")

    # 4. 测试 forward pass
    print("\n--- Forward Pass Test ---")
    model = model.to(DEVICE)

    # 创建 agent 实例（不加载模型文件）
    # 手动构建 tokens 测试
    agent = AgentV6.__new__(AgentV6)
    agent.model = model
    agent.strategy_only = False

    tokens = agent._build_tokens(dummy_game_state, include_combat=True)
    print(f"  Token count (combat): {len(tokens)}")

    with torch.no_grad():
        shared_repr = model.ctx_transformer(tokens)
    print(f"  shared_repr shape: {shared_repr.shape}")
    print(f"  shared_repr norm: {shared_repr.norm().item():.4f}")

    # 测试 combat head forward
    hand = dummy_game_state["game_state"]["combat_state"]["hand"]
    hand_encs = []
    for c in hand:
        cv = encode_card(c)
        t = torch.FloatTensor(cv).to(DEVICE)
        enc = model.ctx_transformer.token_proj.project("card", TOKEN_HAND_CARD, t)
        hand_encs.append(enc)
    hand_encodings = torch.stack(hand_encs)

    monsters = dummy_game_state["game_state"]["combat_state"]["monsters"]
    monster_encs = []
    for m in monsters:
        ev = encode_enemy(m)
        t = torch.FloatTensor(ev).to(DEVICE)
        enc = model.ctx_transformer.token_proj.project("enemy", TOKEN_MONSTER, t)
        monster_encs.append(enc)
    monster_encodings = torch.stack(monster_encs)

    card_mask = torch.ones(NUM_CARD_ACTIONS, device=DEVICE)
    card_mask[3:MAX_HAND] = 0
    monster_mask = torch.ones(MAX_MONSTERS, device=DEVICE)
    monster_mask[1:] = 0

    with torch.no_grad():
        card_logits, target_logits, value = model.combat_head(
            shared_repr, hand_encodings, monster_encodings, card_mask, monster_mask
        )
    print(f"  card_logits: {card_logits[:4].tolist()}")
    print(f"  target_logits: {target_logits[:2].tolist()}")
    print(f"  value: {value.item():.4f}")

    # 测试 draft head
    candidate_feats = [
        torch.FloatTensor(encode_card({"id": "Inflame"})).to(DEVICE),
        torch.FloatTensor(encode_card({"id": "Shrug It Off"})).to(DEVICE),
        torch.zeros(ENTITY_DIM, device=DEVICE),  # skip
    ]

    tokens_strategy = agent._build_tokens(dummy_game_state, include_combat=False)
    with torch.no_grad():
        shared_repr_s = model.ctx_transformer(tokens_strategy)
        draft_logits, draft_value = model.draft_head(shared_repr_s, candidate_feats)
    print(f"  draft_logits: {draft_logits.tolist()}")
    print(f"  draft_value: {draft_value.item():.4f}")

    print("\n" + "=" * 60)
    print("All validation tests passed!")
    print("=" * 60)

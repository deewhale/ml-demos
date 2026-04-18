"""
V7 策略层 handler (Phase 1 MVP)

四个 handler：
- DraftHandler: 选卡（卡牌奖励）
- PathHandler: 选路（地图）
- ChoiceHandler: 事件 / 休息 / 商店 / Neow / Boss relic
- (商店/事件复用 ChoiceHandler)

所有优先级列表抄自 bottled_ai/peaceful_pummeling。
ID 规范用 StSRLSolver 的 card id（resolve_card_id 可处理 Java alias）。
"""

from __future__ import annotations

import logging
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from packages.engine.game import GameRunner, GameAction

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Watcher 卡组构建优先级（抄 bottled_ai peaceful_pummeling config.py）
# ---------------------------------------------------------------------------
# key: StSRLSolver 内部 card id（小写字母 + 下划线/驼峰）
# value: 期望数量（超过这个数量就不再加）

DESIRED_CARDS = {
    "Blasphemy": 1,
    "TalkToTheHand": 3,
    "Adaptation": 1,      # Rushdown Java ID
    "Tantrum": 2,
    "BattleHymn": 1,
    "MentalFortress": 2,
    "Vigilance": 1,
    "ClearTheMind": 2,    # Tranquility Java ID
    "Wallop": 1,
    "FlurryOfBlows": 2,
    "EmptyBody": 2,
    "Indignation": 1,
    "CrushJoints": 1,
    "FearNoEvil": 2,
    "EmptyFist": 1,
    "ReachHeaven": 1,
    "InnerPeace": 1,
    "CutThroughFate": 1,
    "Eruption": 1,
    "Crescendo": 1,
    "Halt": 1,
    "RitualDagger": 1,
    "DeceiveReality": 1,
    "CarveReality": 1,
    "SpiritShield": 1,
    "SandsOfTime": 1,
    "Ragnarok": 1,
    "Perseverance": 2,
    "WheelKick": 1,
    "LikeWater": 1,
    "Apparition": 1,
}

# 删牌优先级（越靠前越优先删）
CARD_REMOVAL_PRIORITY = [
    "ConjureBlade",
    "Vault",
    "Omniscience",
    "Meditate",
    "Defend_P",
    "Strike_P",
    "Bite",
]

# 升级优先级（仅用于 Smith / Neow 升级卡）
HIGH_PRIORITY_UPGRADES = frozenset({
    "Apotheosis",
    "Blasphemy",
    "Eruption",
})


# ---------------------------------------------------------------------------
# Draft: 选卡奖励
# ---------------------------------------------------------------------------

def _base_card_id(card_id: str) -> str:
    """去掉 '+' 升级后缀，返回 base id。"""
    return card_id.rstrip("+")


def _deck_counts(run_state) -> dict:
    """统计当前 deck 中每张卡（base id）的数量"""
    counts: dict = {}
    for c in run_state.deck:
        base = _base_card_id(c.id)
        counts[base] = counts.get(base, 0) + 1
    return counts


def pick_card_reward(run_state, card_reward) -> Optional[int]:
    """从 card_reward.cards 里选择最想要的卡的 index。
    返回 None 表示 skip。
    """
    if card_reward is None or not card_reward.cards:
        return None

    counts = _deck_counts(run_state)

    best_idx = None
    best_gap = 0  # 当前缺多少张
    for i, card in enumerate(card_reward.cards):
        base = _base_card_id(card.id)
        desired = DESIRED_CARDS.get(base, 0)
        have = counts.get(base, 0)
        gap = desired - have
        if gap > best_gap:
            best_gap = gap
            best_idx = i

    # 如果候选都不在 DESIRED（或者都已满额），skip
    return best_idx  # 可能是 None


# ---------------------------------------------------------------------------
# Path: 地图选路（基于 HP% 的简单启发）
# ---------------------------------------------------------------------------

def pick_path(run_state) -> int:
    """返回 available_paths 的 index。"""
    paths = run_state.get_available_paths()
    if not paths:
        return 0

    hp_pct = run_state.current_hp / max(run_state.max_hp, 1)
    current_floor = run_state.floor
    # Act 1 boss 在 floor 16；act 转换前一层要休息
    # 简化：只看当前房间类型

    # 优先级（从高到低），按 hp 档位不同
    if hp_pct <= 0.40:
        priority = ["R", "?", "$", "M", "E"]  # 血低：休息优先
    elif hp_pct <= 0.70:
        priority = ["M", "?", "$", "R", "E"]  # 中血：累积遗物金币
    else:
        priority = ["E", "M", "?", "$", "R"]  # 高血：精英给 relic

    # 低血（<55%）避 ELITE：即使落在"中血"档，精英也排到最后
    # 低血下的优先级：R > ? > M > $ > E
    avoid_elite = hp_pct < 0.55
    if avoid_elite:
        priority = ["R", "?", "M", "$", "E"]

    # 离 boss 一层时（floor 15 时）强制 R（如可选）
    # Act 1 boss 在 floor 17（走完 15 层 + boss）——保险起见仅作加分项
    near_boss = current_floor in (15, 32, 49)

    def path_priority(path_node) -> tuple:
        """返回 (-优先级, idx)，越小越好。"""
        room_char = path_node.room_type.value if path_node.room_type else "M"
        if near_boss and room_char == "R":
            return (-1000, 0)  # 强制最高优先
        # 低血下再额外给 E 一个大惩罚（避免并列情况下还误选 E）
        if avoid_elite and room_char == "E":
            return (9999, 0)
        try:
            rank = priority.index(room_char)
        except ValueError:
            rank = len(priority)
        return (rank, 0)

    # 带 index 比较
    best_idx = 0
    best_key = path_priority(paths[0])
    for i in range(1, len(paths)):
        k = path_priority(paths[i])
        if k < best_key:
            best_key = k
            best_idx = i
    return best_idx


# ---------------------------------------------------------------------------
# Rest site: Rest / Smith / Toke / Dig / Lift
# ---------------------------------------------------------------------------

def pick_rest_action(run_state, actions):
    """actions: list of RestAction。返回一个 action。"""
    # 分类
    rest_act = None
    upgrades = []  # (action, card)
    toke_acts = []
    dig_act = None
    lift_act = None
    ruby_act = None

    for a in actions:
        t = a.action_type
        if t == "rest":
            rest_act = a
        elif t == "upgrade":
            # 拿到对应 card
            upg = run_state.get_upgradeable_cards()
            card = None
            for idx, c in upg:
                if idx == a.card_index:
                    card = c
                    break
            upgrades.append((a, card))
        elif t == "toke":
            toke_acts.append(a)
        elif t == "dig":
            dig_act = a
        elif t == "lift":
            lift_act = a
        elif t == "ruby_key":
            ruby_act = a

    hp_pct = run_state.current_hp / max(run_state.max_hp, 1)

    # Ruby key (act 3) 有就拿
    if ruby_act is not None:
        return ruby_act

    # 低血强制休息（60% 阈值）：不要 smith / dig / toke，先回血
    if hp_pct < 0.60 and rest_act is not None:
        return rest_act

    # Smith：按 HIGH_PRIORITY_UPGRADES 找高优升级
    if upgrades:
        # 先找 high-priority
        for a, c in upgrades:
            if c is not None and _base_card_id(c.id) in HIGH_PRIORITY_UPGRADES:
                return a
        # 否则升级 deck 中数量最多的 DESIRED 卡
        counts = _deck_counts(run_state)
        best_a = None
        best_score = -1
        for a, c in upgrades:
            if c is None:
                continue
            base = _base_card_id(c.id)
            # 越 DESIRED 分越高
            desired_score = DESIRED_CARDS.get(base, 0)
            dup_bonus = counts.get(base, 0)
            score = desired_score * 10 + dup_bonus
            if score > best_score:
                best_score = score
                best_a = a
        if best_a is not None and best_score >= 1:
            return best_a

    # Toke：删除最烂的卡
    if toke_acts:
        # 找优先级最高（越靠前越优先）的 toke 对象
        best_toke = None
        best_rank = 1e9
        for a in toke_acts:
            # 找对应 card
            for idx, c in run_state.get_removable_cards():
                if idx == a.card_index:
                    base = _base_card_id(c.id)
                    try:
                        rank = CARD_REMOVAL_PRIORITY.index(base)
                    except ValueError:
                        rank = 500  # 不在列表里，不优先删
                    if rank < best_rank:
                        best_rank = rank
                        best_toke = a
                    break
        if best_toke is not None and best_rank < 500:
            return best_toke

    # Dig 不怎么有用，但有就用
    if dig_act is not None:
        return dig_act

    # Lift：有就用（长期 strength）
    if lift_act is not None:
        return lift_act

    # Fallback：休息（哪怕满血也好）
    if rest_act is not None:
        return rest_act

    return actions[0]


# ---------------------------------------------------------------------------
# Shop
# ---------------------------------------------------------------------------

def pick_shop_action(run_state, shop_state, actions):
    """简单策略：
    - 钱少（<75）：不买任何东西，直接 leave
    - 钱 75~150：考虑删牌（若有可删）
    - 钱 >=150：考虑买 DESIRED 卡
    - 永远不买 potion（Watcher 留 potion 给 boss——MVP 直接不主动买）
    """
    gold = run_state.gold

    # 分类 actions
    leave = None
    buy_colored = []
    buy_colorless = []
    buy_relic = []
    buy_potion = []
    remove = []

    for a in actions:
        t = a.action_type
        if t == "leave":
            leave = a
        elif t == "buy_colored_card":
            buy_colored.append(a)
        elif t == "buy_colorless_card":
            buy_colorless.append(a)
        elif t == "buy_relic":
            buy_relic.append(a)
        elif t == "buy_potion":
            buy_potion.append(a)
        elif t == "remove_card":
            remove.append(a)

    # 钱太少，走人
    if gold < 75:
        return leave if leave is not None else actions[0]

    counts = _deck_counts(run_state)

    # 先考虑买 DESIRED 卡
    if gold >= 75:
        best_card_action = None
        best_gap = 0
        best_price = 10**9
        for a in buy_colored:
            slot = a.item_index
            for sc in shop_state.get_available_colored_cards():
                if sc.slot_index == slot and sc.price <= gold:
                    base = _base_card_id(sc.card.id)
                    desired = DESIRED_CARDS.get(base, 0)
                    gap = desired - counts.get(base, 0)
                    if gap > best_gap or (gap == best_gap and sc.price < best_price):
                        best_gap = gap
                        best_price = sc.price
                        best_card_action = a
        if best_card_action is not None and best_gap > 0:
            return best_card_action

    # 考虑删牌
    if gold >= 75 and remove:
        # 找最烂的卡来删
        best_remove = None
        best_rank = 1e9
        for a in remove:
            # item_index 是 deck 里 idx
            removable = run_state.get_removable_cards()
            for idx, c in removable:
                if idx == a.item_index:
                    base = _base_card_id(c.id)
                    try:
                        rank = CARD_REMOVAL_PRIORITY.index(base)
                    except ValueError:
                        rank = 500
                    if rank < best_rank:
                        best_rank = rank
                        best_remove = a
                    break
        if best_remove is not None and best_rank < 500:
            return best_remove

    # 考虑 relic（>=150 才敢买，不挑了直接买第一个）
    if gold >= 150 and buy_relic:
        return buy_relic[0]

    return leave if leave is not None else actions[0]


# ---------------------------------------------------------------------------
# Event: 简单默认（选 index 0，通常是"走人"/安全选项）
# ---------------------------------------------------------------------------

def pick_event_action(run_state, event_state, actions) -> int:
    """Return the action index (into `actions`).

    修法 B：HP-aware event choice —— 如果 HP < 50% max_hp，
    避开带 COMBAT outcome 的选项（Mushrooms/Dead Adventurer/Big Fish box/etc.）。
    默认退回 index 0（通常 safe/leave）。
    """
    if event_state is None or not actions:
        return 0

    # HP-aware：仅在低血时启用
    try:
        hp = run_state.current_hp
        max_hp = run_state.max_hp
    except AttributeError:
        return 0
    if max_hp <= 0 or hp / max_hp >= 0.5:
        return 0

    # 拿事件定义，检查每个 choice 是否会引发 COMBAT
    try:
        from packages.engine.content.events import (
            ALL_EVENTS, OutcomeType,
        )
    except ImportError:
        return 0

    event_id = getattr(event_state, "event_id", None)
    if event_id is None:
        return 0
    event = ALL_EVENTS.get(event_id)
    if event is None:
        # 兜底再试一次常见 alias
        event = ALL_EVENTS.get(event_id.replace(" ", ""))
    if event is None or not event.choices:
        return 0

    def _has_combat(choice) -> bool:
        for out in getattr(choice, "outcomes", []) or []:
            if getattr(out, "type", None) == OutcomeType.COMBAT:
                return True
        # 也检查 description 的文本 fallback（万一 outcome 没标）
        desc = (getattr(choice, "description", "") or "").lower()
        keywords = ("fight", "attack", "combat")
        return any(k in desc for k in keywords)

    # 把 actions 的 choice_index 和 Event.choices 对上
    # Event.choices 里的 EventChoice.index 是原始 button index
    idx_to_choice = {c.index: c for c in event.choices}

    non_combat_idx = None
    for i, a in enumerate(actions):
        ci = getattr(a, "choice_index", None)
        if ci is None:
            continue
        choice = idx_to_choice.get(ci)
        if choice is None:
            # 无法判定的 choice，认为保守可选
            non_combat_idx = i if non_combat_idx is None else non_combat_idx
            continue
        if not _has_combat(choice):
            non_combat_idx = i
            break

    if non_combat_idx is not None:
        logger.info(
            "pick_event_action: HP %d/%d (<50%%) at event=%s, picking non-combat idx=%d",
            hp, max_hp, event_id, non_combat_idx,
        )
        return non_combat_idx
    return 0


# ---------------------------------------------------------------------------
# Boss relic
# ---------------------------------------------------------------------------

BOSS_RELIC_PRIORITY = [
    "Runic Pyramid", "RunicPyramid",
    "Pandora's Box", "PandorasBox",
    "Calling Bell", "CallingBell",
    "Tiny House", "TinyHouse",
    "Velvet Choker", "VelvetChoker",
    "Sozu",
    "Snecko Eye", "SneckoEye",
    "Busted Crown", "BustedCrown",
    "Coffee Dripper", "CoffeeDripper",
    "Fusion Hammer", "FusionHammer",
    "Philosopher's Stone", "PhilosophersStone",
    "Ectoplasm",
    "Cursed Key", "CursedKey",
    "Mark of Pain", "MarkOfPain",
]


def pick_boss_relic(run_state, boss_relic_choices) -> int:
    """从 BossRelicChoices.relics 里选一个 index。"""
    relics = boss_relic_choices.relics
    best_idx = 0
    best_rank = 1e9
    for i, relic in enumerate(relics):
        rid = relic.id if hasattr(relic, "id") else str(relic)
        name = getattr(relic, "name", rid)
        try:
            rank = min(
                BOSS_RELIC_PRIORITY.index(rid) if rid in BOSS_RELIC_PRIORITY else 1000,
                BOSS_RELIC_PRIORITY.index(name) if name in BOSS_RELIC_PRIORITY else 1000,
            )
        except ValueError:
            rank = 1000
        if rank < best_rank:
            best_rank = rank
            best_idx = i
    return best_idx


# ---------------------------------------------------------------------------
# Neow
# ---------------------------------------------------------------------------

# 偏好顺序（直接拿好处，不要带副作用的）
NEOW_PRIORITY = [
    "random_rare_relic",       # 稀有遗物，最强
    "ten_percent_hp_bonus",    # +10% HP
    "three_enemy_kill",        # 首战 1HP 敌人
    "random_common_relic",
    "transform_two",           # 转化两张
    "remove_two",              # 删两张
    "transform_card",
    "remove_card",
    "upgrade_card",
    "one_random_rare_card",
    "hundred_gold",
    "three_cards",
    "three_potions",
    "boss_swap",
    "random_colorless_rare",
]


def pick_neow(blessings) -> int:
    """Pick a Neow blessing index."""
    best_idx = 0
    best_rank = 1e9
    for i, b in enumerate(blessings):
        btype = b.blessing_type.value if hasattr(b.blessing_type, "value") else str(b.blessing_type)
        try:
            rank = NEOW_PRIORITY.index(btype)
        except ValueError:
            rank = 1000
        if rank < best_rank:
            best_rank = rank
            best_idx = i
    return best_idx

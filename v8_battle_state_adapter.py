"""
V8 Stage A.2 — StSRLSolver state → bottled_ai BattleState 鸭子接口适配器。

bottled_ai 的 ComparatorAssessment / comparisons.* 全部直接读
`BattleState`、`Player`、`Monster` 等具体类的属性。我们把 StSRLSolver
runner.state（CombatState）包一层，对外暴露同样的属性名。

StSRLSolver 真实 schema（2026-04 验证，对照
external/StSRLSolver/packages/engine/state/combat.py）：

    EntityState:                # state.player
        hp, max_hp, block, statuses: Dict[str, int]
        — **没有 energy**

    CombatState:                # runner.current_combat.state
        player: EntityState
        energy: int             # 玩家能量在这里，不在 player 上
        max_energy: int
        stance: str = "Neutral"
        hand: List[str]         # 卡片是 ID 字符串，不是对象
        draw_pile: List[str]
        discard_pile: List[str]
        exhaust_pile: List[str]
        enemies: List[EnemyCombatState]
        potions: List[str]      # 包括 "" 空槽
        relics: List[str]       # combat 内可见的 relic id list
        relic_counters: Dict[str, int]
        combat_over: bool
        player_won: bool
        turn: int
        ...

    EnemyCombatState(EntityState):
        id, name, enemy_type, move_id, move_damage, move_hits, move_block,
        statuses, ...

设计原则
--------
1. **不猜字段**：所有未知/未实现字段用 `_NOT_AVAILABLE` 哨兵 + raise，
   错误信息带 "对照 packages/engine/state/combat.py"。
2. **薄包装**：所有适配器只做属性映射 / card_id → CardType 查询，不做计算。
3. **Power 名归一化**：bottled_ai 用 PowerId.X enum，StSRLSolver 用 PascalCase
   字符串（如 "Vulnerable"）；PowerSet 把两边都映射到 .get() 结果。

──────────────────────────────────────────────
已实现 — Ironclad 战斗评估必需
──────────────────────────────────────────────
PlayerAdapter:
    current_hp, max_hp, block, energy（从 combat_state 注入）, powers, relics
MonsterAdapter:
    current_hp, max_hp, block, powers
CardAdapter:
    id（字符串），type（从 ALL_CARDS 查 CardType），cost, ethereal
PowerSet:
    .get(power_id, default), __contains__, .keys(),
    支持 enum.value/enum.name/PascalCase 多种 lookup
BottledStateAdapter:
    player, monsters, hand, draw_pile, discard_pile, exhaust_pile,
    relics（dict）, potions（list），及 bottled_ai sim 计数器（默认 0）

──────────────────────────────────────────────
未实现 — 触达即 raise
──────────────────────────────────────────────
- BottledStateAdapter.memory_general[MemoryItem.*]
  StSRLSolver 没有 bottled_ai 的 ResetSchedule/MemoryItem 系统。
  Watcher stance / claws 等强依赖此机制，Ironclad 几乎用不到。
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional


# ============================================================================
# § 0 PowerId 名称归一化
# ============================================================================
#
# bottled_ai 用 PowerId.X enum，X 通常是大写下划线（VULNERABLE, INTANGIBLE_PLAYER）。
# StSRLSolver statuses dict 的 key 是 PascalCase 字符串（"Vulnerable",
# "Plated Armor"）。我们做一个最小映射表，PowerSet.get 同时尝试 raw / 映射后。

_POWER_ID_TO_STS: Dict[str, str] = {
    "VULNERABLE": "Vulnerable",
    "WEAKENED": "Weakened",
    "FRAIL": "Frail",
    "STRENGTH": "Strength",
    "DEXTERITY": "Dexterity",
    "ARTIFACT": "Artifact",
    "PLATED_ARMOR": "Plated Armor",
    "METALLICIZE": "Metallicize",
    "THORNS": "Thorns",
    "POISON": "Poison",
    "INTANGIBLE_PLAYER": "Intangible",
    "INTANGIBLE": "Intangible",
    "BARRICADE": "Barricade",
    "BLOCK_RETURN": "Block Return",
    "ANGER_NOB": "Enrage",  # bottled_ai naming difference
    "REPAIR": "Repair",
    "BIAS": "Bias",
    "BLASPHEMER": "Blasphemer",
    "TIME_WARP": "Time Warp",
    "UNAWAKENED": "Unawakened",
}


def _normalize_power_key(key: Any) -> List[str]:
    """返回所有可能的 lookup 名（按优先级）。"""
    candidates: List[str] = []
    # 直接
    if isinstance(key, str):
        candidates.append(key)
        mapped = _POWER_ID_TO_STS.get(key)
        if mapped:
            candidates.append(mapped)
    else:
        # enum 风格
        name = getattr(key, "name", None)
        if name is not None:
            candidates.append(name)
            mapped = _POWER_ID_TO_STS.get(name)
            if mapped:
                candidates.append(mapped)
        value = getattr(key, "value", None)
        if isinstance(value, str):
            candidates.append(value)
            mapped = _POWER_ID_TO_STS.get(value)
            if mapped:
                candidates.append(mapped)
    return candidates


# ============================================================================
# § 1 PowerSet — 模拟 bottled_ai Powers (dict[PowerId, int]) 的 .get / 'in'
# ============================================================================

class PowerSet:
    """包装 StSRLSolver 的 statuses dict（key 为 PascalCase 字符串）。"""

    def __init__(self, raw: Optional[Dict[Any, int]]):
        self._raw: Dict[Any, int] = raw if raw is not None else {}

    def _lookup(self, key: Any) -> Optional[int]:
        for candidate in _normalize_power_key(key):
            if candidate in self._raw:
                return self._raw[candidate]
        return None

    def get(self, key: Any, default: int = 0) -> int:
        v = self._lookup(key)
        return default if v is None else v

    def __contains__(self, key: Any) -> bool:
        return self._lookup(key) is not None

    def __getitem__(self, key: Any) -> int:
        v = self._lookup(key)
        if v is None:
            raise KeyError(key)
        return v

    def keys(self) -> Iterator[Any]:
        return iter(self._raw.keys())

    def __iter__(self) -> Iterator[Any]:
        return iter(self._raw.keys())


# ============================================================================
# § 2 PlayerAdapter / MonsterAdapter
# ============================================================================

def _require(obj: Any, attr: str, owner: str) -> Any:
    """读 attr，缺失就显式 raise NotImplementedError。"""
    if not hasattr(obj, attr):
        raise NotImplementedError(
            f"TODO: {owner} 缺字段 .{attr}；"
            f"对照 external/StSRLSolver/packages/engine/state/combat.py 实际字段名。"
            f" 已知字段: hp/max_hp/block/statuses（EntityState）, "
            f"energy/max_energy/hand/draw_pile/discard_pile/exhaust_pile/"
            f"enemies/potions/relics/combat_over/player_won/turn（CombatState）。"
        )
    return getattr(obj, attr)


def _optional(obj: Any, attr: str, default: Any) -> Any:
    """读 attr，缺失返回 default。仅用于已知 StSRLSolver 也可能不暴露的字段。"""
    return getattr(obj, attr, default)


class PlayerAdapter:
    """
    包装 StSRLSolver 的 EntityState（state.player），加上注入的 energy 与 relics。

    bottled_ai 期望字段：current_hp, max_hp, block, energy, powers, relics
    StSRLSolver EntityState 字段：hp, max_hp, block, statuses
    （**energy 在 CombatState 上，不在 player 上 → 由调用方注入**）
    """

    def __init__(
        self,
        raw_player: Any,
        energy: int,
        relics: Optional[Dict[Any, int]] = None,
    ):
        self._raw = raw_player
        self.current_hp: int = _require(raw_player, "hp", "state.player")
        self.max_hp: int = _require(raw_player, "max_hp", "state.player")
        self.block: int = _require(raw_player, "block", "state.player")
        self.energy: int = energy
        self.powers: PowerSet = PowerSet(_optional(raw_player, "statuses", {}))
        self.relics: Dict[Any, int] = relics if relics is not None else {}


class MonsterAdapter:
    """
    包装 StSRLSolver 的 EnemyCombatState。

    bottled_ai 期望字段：current_hp, max_hp, block, powers
    StSRLSolver EnemyCombatState 字段：hp, max_hp, block, statuses (+ id, ...)
    """

    def __init__(self, raw_monster: Any):
        self._raw = raw_monster
        self.current_hp: int = _require(raw_monster, "hp", "state.enemies[i]")
        self.max_hp: int = _require(raw_monster, "max_hp", "state.enemies[i]")
        self.block: int = _optional(raw_monster, "block", 0)
        self.powers: PowerSet = PowerSet(_optional(raw_monster, "statuses", {}))


# ============================================================================
# § 3 CardAdapter — 包装一张牌（StSRLSolver 中是 ID 字符串）
# ============================================================================

class _CardTypeStub:
    """对应 bottled_ai CardType enum 的最小占位，只暴露 .name。"""

    __slots__ = ("name",)

    def __init__(self, name: str):
        self.name = name

    def __repr__(self) -> str:
        return f"_CardTypeStub({self.name!r})"


def _resolve_card_type(card_id: str) -> _CardTypeStub:
    """
    从 card_id 字符串查 ALL_CARDS 拿 CardType。
    查不到（如 ascension curse "AscendersBane"）默认 SKILL 不影响 Ironclad 评估。
    Curse/Status 命中是 bad_cards_exhausted 的关键，要保证查得准。
    """
    base = card_id.rstrip("+")
    try:
        # 局部 import 避免循环依赖；ALL_CARDS 在 packages/engine/content/cards.py
        from packages.engine.content.cards import ALL_CARDS  # type: ignore
    except Exception:
        return _CardTypeStub("SKILL")

    card_def = ALL_CARDS.get(base)
    if card_def is None:
        # 曲线兜底：常见 curse 关键词
        lower = base.lower()
        if "curse" in lower or base in {"AscendersBane"}:
            return _CardTypeStub("CURSE")
        if base in {"Slimed", "Burn", "Dazed", "Wound", "VoidCard"}:
            return _CardTypeStub("STATUS")
        return _CardTypeStub("SKILL")

    ct = getattr(card_def, "card_type", None)
    if ct is None:
        return _CardTypeStub("SKILL")
    name = getattr(ct, "name", None) or str(ct).split(".")[-1].upper()
    return _CardTypeStub(name)


def _resolve_card_cost(card_id: str, card_costs_override: Dict[str, int]) -> int:
    """优先读 combat 内 card_costs 缓存，再读 ALL_CARDS.cost。"""
    if card_id in card_costs_override:
        return card_costs_override[card_id]
    base = card_id.rstrip("+")
    if base in card_costs_override:
        return card_costs_override[base]
    try:
        from packages.engine.content.cards import ALL_CARDS  # type: ignore
        card_def = ALL_CARDS.get(base)
        if card_def is not None:
            return int(getattr(card_def, "cost", 1))
    except Exception:
        pass
    return 1


def _resolve_card_ethereal(card_id: str) -> bool:
    base = card_id.rstrip("+")
    try:
        from packages.engine.content.cards import ALL_CARDS  # type: ignore
        card_def = ALL_CARDS.get(base)
        if card_def is not None:
            return bool(getattr(card_def, "ethereal", False))
    except Exception:
        pass
    return False


class CardAdapter:
    """
    包装 StSRLSolver 的 card 表示（字符串 ID 或对象）。

    bottled_ai 用到的字段：
        c.id     — 字符串
        c.type   — 有 .name 的 enum-like，bottled_ai 比较 .name in {"CURSE","STATUS"}
        c.cost   — int
        c.ethereal — bool
    """

    def __init__(
        self,
        raw_card: Any,
        card_costs_override: Optional[Dict[str, int]] = None,
    ):
        self._raw = raw_card
        if card_costs_override is None:
            card_costs_override = {}

        if isinstance(raw_card, str):
            self.id: Any = raw_card
            self.type: Any = _resolve_card_type(raw_card)
            self.cost: int = _resolve_card_cost(raw_card, card_costs_override)
            self.ethereal: bool = _resolve_card_ethereal(raw_card)
            return

        # 兜底：保留对象路径（少见，但保留以防上游传 dataclass）
        self.id = _require(raw_card, "id", "card")
        if hasattr(raw_card, "type"):
            self.type = raw_card.type
        elif hasattr(raw_card, "card_type"):
            self.type = raw_card.card_type
        else:
            raise NotImplementedError(
                "TODO: card 既无 .type 也无 .card_type，对照 ALL_CARDS schema 修正。"
            )
        self.cost = _optional(raw_card, "cost", 0)
        self.ethereal = _optional(raw_card, "ethereal", False)


def _wrap_pile(
    pile: Optional[List[Any]],
    card_costs_override: Optional[Dict[str, int]] = None,
) -> List[CardAdapter]:
    if pile is None:
        return []
    return [CardAdapter(c, card_costs_override) for c in pile]


# ============================================================================
# § 4 BottledStateAdapter — 顶层包装
# ============================================================================

class _MemoryShim:
    """
    bottled_ai 用 self.state.memory_general[MemoryItem.X] 拿 Watcher stance 等。
    StSRLSolver 没有这套机制 → 触达就 raise，强制调用方在 v8_teacher_eval 里 stub。
    """

    def __init__(self, owner_label: str):
        self._owner = owner_label

    def __getitem__(self, key: Any) -> Any:
        raise NotImplementedError(
            f"TODO: {self._owner}[{key!r}] — StSRLSolver 没有 bottled_ai 的 "
            f"ResetSchedule/MemoryItem 系统。Ironclad 评估若触达此分支，说明"
            f" comparison 实际是 Watcher/Defect 专属，应在 v8_teacher_eval 里 stub。"
        )

    def get(self, key: Any, default: Any = None) -> Any:
        raise NotImplementedError(
            f"TODO: {self._owner}.get({key!r}) — 同上。"
        )

    def __contains__(self, key: Any) -> bool:
        return False


class BottledStateAdapter:
    """
    包装 StSRLSolver 的 CombatState（runner.current_combat.state），
    向 bottled_ai 评估代码暴露 BattleState 接口。

    用法：
        adapter = BottledStateAdapter(combat_state, run_relics=..., run_potions=...)
        ca = ComparatorAssessment(adapter, original_adapter, config)
    """

    def __init__(
        self,
        raw_state: Any,
        run_relics: Optional[Dict[Any, int]] = None,
        run_potions: Optional[List[Any]] = None,
    ):
        self._raw = raw_state

        # ---- relics / potions（优先使用注入的 run-level；否则从 combat 自带 list）----
        if run_relics is not None:
            relics_dict: Dict[Any, int] = dict(run_relics)
        else:
            relics_dict = {}
            combat_relics = _optional(raw_state, "relics", []) or []
            relic_counters = _optional(raw_state, "relic_counters", {}) or {}
            for rid in combat_relics:
                # 优先用 counter，没 counter 用 1（"持有"）
                relics_dict[rid] = int(relic_counters.get(rid, 1))
        self.relics: Dict[Any, int] = relics_dict

        if run_potions is not None:
            self.potions: List[Any] = list(run_potions)
        else:
            # CombatState.potions 是 List[str]，"" 表示空槽 → 过滤
            raw_potions = _optional(raw_state, "potions", []) or []
            self.potions = [p for p in raw_potions if p]

        # ---- player ----
        raw_player = _require(raw_state, "player", "state")
        # energy 在 CombatState 上而不是 player 上
        energy = _require(raw_state, "energy", "state")
        self.player: PlayerAdapter = PlayerAdapter(
            raw_player, energy=energy, relics=self.relics
        )

        # ---- monsters ----
        raw_enemies = _require(raw_state, "enemies", "state")
        self.monsters: List[MonsterAdapter] = [MonsterAdapter(m) for m in raw_enemies]

        # ---- piles（List[str]）----
        card_costs_override = _optional(raw_state, "card_costs", {}) or {}
        self.hand: List[CardAdapter] = _wrap_pile(
            _require(raw_state, "hand", "state"), card_costs_override
        )
        self.draw_pile: List[CardAdapter] = _wrap_pile(
            _optional(raw_state, "draw_pile", []), card_costs_override
        )
        self.discard_pile: List[CardAdapter] = _wrap_pile(
            _optional(raw_state, "discard_pile", []), card_costs_override
        )
        self.exhaust_pile: List[CardAdapter] = _wrap_pile(
            _optional(raw_state, "exhaust_pile", []), card_costs_override
        )

        # ---- orb 系统（Defect 专属，Ironclad 永远空）----
        self.orb_slots: int = _optional(raw_state, "orb_slots", 0)
        self.orbs: List[Any] = _optional(raw_state, "orbs", [])

        # ---- bottled_ai 内部计数器（StSRLSolver 不维护，全 0 不影响 Ironclad）----
        self.saved_block_for_next_turn: int = 0
        self.total_random_damage_dealt: int = 0
        self.total_random_poison_added: int = 0
        self.draw_free_early: int = 0
        self.draw_free: int = 0
        self.draw_pay_early: int = 0
        self.draw_pay: int = 0

        # ---- memory（Watcher/Defect 专属）----
        self.memory_general = _MemoryShim("state.memory_general")
        self.memory_by_card = _MemoryShim("state.memory_by_card")


__all__ = [
    "PowerSet",
    "PlayerAdapter",
    "MonsterAdapter",
    "CardAdapter",
    "BottledStateAdapter",
]

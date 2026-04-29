"""
V8 Stage A.2 — StSRLSolver state → bottled_ai BattleState 鸭子接口适配器。

bottled_ai 的 ComparatorAssessment / comparisons.* 全部直接读
`BattleState`、`Player`、`Monster` 等具体类的属性。我们把 StSRLSolver
runner.state 包一层，对外暴露同样的属性名，让移植后的评估代码不用改。

设计原则
--------
1. **不猜字段**：StSRLSolver 实际字段名待与 packages/engine/state/combat.py
   对齐。现在仅根据本仓库 v7_evaluator.py 的现有用法（已知字段：
   `state.player.hp / max_hp / block / energy / statuses`、
   `state.enemies`、`state.stance`）做最佳推测；其余未知字段一律
   `raise NotImplementedError("TODO: <说明>")`，**绝不静默 fallback**。
2. **薄包装**：所有适配器只做属性映射，不做计算。计算在 v8_teacher_eval。
3. **Power 名归一化**：bottled_ai 用 PowerId enum；我们用大写字符串
   （见 v8_teacher_ironclad.POWERS_WE_LIKE）。`PowerSet` 在 lookup
   时把 enum/字符串都接受，以兼容两边。

──────────────────────────────────────────────
已实现 (Implemented) — Ironclad 战斗评估必需
──────────────────────────────────────────────
PlayerAdapter:
    current_hp, max_hp, block, energy, powers, relics
MonsterAdapter:
    current_hp, max_hp, block, powers
PowerSet:
    .get(power_id, default), __contains__, .keys(),
    支持 enum.value 风格 lookup（bottled_ai 传 PowerId.X 的那种）
BottledStateAdapter:
    player, monsters, hand, draw_pile, discard_pile, exhaust_pile,
    relics, potions, orb_slots, orbs,
    saved_block_for_next_turn, total_random_damage_dealt,
    total_random_poison_added,
    draw_free_early, draw_free, draw_pay_early, draw_pay,
    memory_general, memory_by_card

──────────────────────────────────────────────
未实现 (NotImplementedError) — 调用即抛
──────────────────────────────────────────────
- BottledStateAdapter.memory_general[MemoryItem.*]
    StSRLSolver 没有 bottled_ai 风格的 ResetSchedule / MemoryItem 内存系统。
    Watcher stance / ritual dagger / claws 等强依赖此机制，Ironclad 评估
    几乎全部用不到，所以我们让 memory_general/memory_by_card 在被读到
    Watcher/Defect 专属 key 时显式 raise。

- 任何 PowerId 字符串与 StSRLSolver power 实际命名约定不一致的情况。
  PowerSet 在拿不到时返回 default（保留 .get 语义），不会 raise；
  但 PowerSet 的 keys() 对 enum-style key 的迭代行为依赖原始 statuses
  字典的键格式 —— 待 adapter 接通后用 unit test 锁定。
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional


# ============================================================================
# § 1 PowerSet — 模拟 bottled_ai Powers (dict[PowerId, int]) 的 .get / 'in'
# ============================================================================

class PowerSet:
    """
    包装 StSRLSolver 的 statuses dict（key 类型未知，可能是字符串或 enum）。

    bottled_ai 调用形如 `state.player.powers.get(PowerId.VULNERABLE, 0)`。
    我们让 .get() 同时尝试 (key, key.value if hasattr value, key.name) 三种。
    """

    def __init__(self, raw: Optional[Dict[Any, int]]):
        # raw 可能为 None（StSRLSolver 早期 turn 没初始化 statuses 时）
        self._raw: Dict[Any, int] = raw if raw is not None else {}

    def _lookup(self, key: Any) -> Optional[int]:
        # 直接命中
        if key in self._raw:
            return self._raw[key]
        # bottled_ai 传的是 PowerId enum；StSRLSolver 可能用字符串
        # 尝试 key.name（"VULNERABLE"）和 key.value（"vulnerable"）
        name = getattr(key, "name", None)
        if name is not None and name in self._raw:
            return self._raw[name]
        value = getattr(key, "value", None)
        if value is not None and value in self._raw:
            return self._raw[value]
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
            f"对照 StSRLSolver packages/engine/state/combat.py 实际字段名后修正。"
        )
    return getattr(obj, attr)


def _optional(obj: Any, attr: str, default: Any) -> Any:
    """读 attr，缺失返回 default。**仅用于已知 StSRLSolver 也可能不暴露的字段**。"""
    return getattr(obj, attr, default)


class PlayerAdapter:
    """
    包装 StSRLSolver 的 state.player。

    bottled_ai 期望字段：current_hp, max_hp, block, energy, powers, relics
    StSRLSolver 已知字段：hp, max_hp, block, energy, statuses（参见 v7_evaluator.py）

    注意：StSRLSolver 没有把 player 单独持 relics 字段——relics 是 run-level
    数据。这里 player.relics 转发到 state.relics（外层 BottledStateAdapter
    构造时注入），保持与 bottled_ai Player.relics 接口一致。
    """

    def __init__(self, raw_player: Any, run_relics: Optional[Dict[Any, int]] = None):
        self._raw = raw_player
        self.current_hp: int = _require(raw_player, "hp", "state.player")
        self.max_hp: int = _require(raw_player, "max_hp", "state.player")
        self.block: int = _require(raw_player, "block", "state.player")
        self.energy: int = _require(raw_player, "energy", "state.player")
        self.powers: PowerSet = PowerSet(_optional(raw_player, "statuses", {}))
        # bottled_ai 的 Player 也持 relics —— 我们把 run-level relics 透传
        self.relics: Dict[Any, int] = run_relics if run_relics is not None else {}


class MonsterAdapter:
    """
    包装 StSRLSolver 的 state.enemies[i]。

    bottled_ai 期望字段：current_hp, max_hp, block, powers
    StSRLSolver 已知字段：hp, max_hp, block, statuses
    """

    def __init__(self, raw_monster: Any):
        self._raw = raw_monster
        self.current_hp: int = _require(raw_monster, "hp", "state.enemies[i]")
        self.max_hp: int = _require(raw_monster, "max_hp", "state.enemies[i]")
        # block 在 StSRLSolver 某些 enemy 类型下可能没有 → 默认 0
        self.block: int = _optional(raw_monster, "block", 0)
        self.powers: PowerSet = PowerSet(_optional(raw_monster, "statuses", {}))


# ============================================================================
# § 3 CardAdapter — 包装单张手牌
# ============================================================================

class CardAdapter:
    """
    包装 StSRLSolver 的 card 对象（来自 hand / discard_pile / exhaust_pile）。

    bottled_ai 的 ComparatorAssessment 用到的字段：
        c.id     — CardId enum（我们这里用字符串）
        c.type   — CardType enum
        c.cost   — int
        c.ethereal — bool

    StSRLSolver 实际字段名待校验。最大概率：id/card_type/cost/ethereal。
    """

    def __init__(self, raw_card: Any):
        self._raw = raw_card
        self.id: Any = _require(raw_card, "id", "card")
        # bottled_ai 用 c.type 而非 c.card_type → 提供两种属性
        if hasattr(raw_card, "type"):
            self.type: Any = raw_card.type
        elif hasattr(raw_card, "card_type"):
            self.type = raw_card.card_type
        else:
            raise NotImplementedError(
                "TODO: card 既无 .type 也无 .card_type，对照 StSRLSolver Card 类修正。"
            )
        self.cost: int = _optional(raw_card, "cost", 0)
        self.ethereal: bool = _optional(raw_card, "ethereal", False)


def _wrap_pile(pile: Optional[List[Any]]) -> List[CardAdapter]:
    if pile is None:
        return []
    return [CardAdapter(c) for c in pile]


# ============================================================================
# § 4 BottledStateAdapter — 顶层包装
# ============================================================================

class _MemoryShim:
    """
    bottled_ai 用 self.state.memory_general[MemoryItem.X] 拿 Watcher stance、
    ritual dagger 等。StSRLSolver 没有这套机制 → 任何读取一律 raise。

    我们故意让 .get() 也 raise，因为 bottled_ai 内部用的是 [] 而非 .get；
    若有 .get 调用混进来，照样要暴露问题。
    """

    def __init__(self, owner_label: str):
        self._owner = owner_label

    def __getitem__(self, key: Any) -> Any:
        raise NotImplementedError(
            f"TODO: {self._owner}[{key!r}] — StSRLSolver 没有 bottled_ai 的 "
            f"ResetSchedule/MemoryItem 内存系统。Ironclad 评估若触达此分支，"
            f"说明该 comparison 实际是 Watcher/Defect 专属，应在 v8_teacher_eval "
            f"里 stub 掉。"
        )

    def get(self, key: Any, default: Any = None) -> Any:
        # 故意不返回 default —— 见类 docstring
        raise NotImplementedError(
            f"TODO: {self._owner}.get({key!r}) — 同上，需要在调用处显式 stub。"
        )

    def __contains__(self, key: Any) -> bool:
        return False


class BottledStateAdapter:
    """
    包装 StSRLSolver 的 runner.state，向 bottled_ai 评估代码暴露 BattleState 接口。

    用法：
        adapter = BottledStateAdapter(runner.state)
        ca = ComparatorAssessment(adapter, original_adapter, config)
    """

    def __init__(self, raw_state: Any, run_relics: Optional[Dict[Any, int]] = None,
                 run_potions: Optional[List[Any]] = None):
        self._raw = raw_state

        # ---- player ----
        raw_player = _require(raw_state, "player", "state")
        # run_relics/run_potions 由调用方注入：StSRLSolver 把 relics/potions
        # 放在 run_state 而非 combat state，adapter 自己拿不到。
        self.relics: Dict[Any, int] = run_relics if run_relics is not None else {}
        self.potions: List[Any] = run_potions if run_potions is not None else []
        self.player: PlayerAdapter = PlayerAdapter(raw_player, run_relics=self.relics)

        # ---- monsters ----
        raw_enemies = _require(raw_state, "enemies", "state")
        # bottled_ai 的 ComparatorAssessment 假设 monsters 总是 list；
        # 死亡的 monster 仍保留在 list 中 (current_hp <= 0)。
        # StSRLSolver v7_evaluator.py 也是按 enemies list 遍历，行为一致。
        self.monsters: List[MonsterAdapter] = [MonsterAdapter(m) for m in raw_enemies]

        # ---- piles ----
        # StSRLSolver 字段名假设：state.hand / state.discard_pile /
        # state.exhaust_pile / state.draw_pile（与 bottled_ai 一致）。
        # 若实际不一致，_require 会 raise 出来。
        self.hand: List[CardAdapter] = _wrap_pile(_require(raw_state, "hand", "state"))
        self.draw_pile: List[CardAdapter] = _wrap_pile(
            _optional(raw_state, "draw_pile", [])
        )
        self.discard_pile: List[CardAdapter] = _wrap_pile(
            _optional(raw_state, "discard_pile", [])
        )
        self.exhaust_pile: List[CardAdapter] = _wrap_pile(
            _optional(raw_state, "exhaust_pile", [])
        )

        # ---- orb 系统 ----
        # Defect 专属。Ironclad 永远是空 list / 0。给默认值不算 fallback，
        # 因为对非 Defect 角色这就是真值。
        self.orb_slots: int = _optional(raw_state, "orb_slots", 0)
        self.orbs: List[Any] = _optional(raw_state, "orbs", [])

        # ---- 各种 bottled_ai 计数器 ----
        # 这些是 bottled_ai 内部的 simulation 计数器，StSRLSolver 不维护。
        # 默认 0：comparator 看到 best=0, challenger=0 会跳过该比较，
        # 不会影响 Ironclad 的实际决策路径。
        self.saved_block_for_next_turn: int = _optional(
            raw_state, "saved_block_for_next_turn", 0)
        self.total_random_damage_dealt: int = _optional(
            raw_state, "total_random_damage_dealt", 0)
        self.total_random_poison_added: int = _optional(
            raw_state, "total_random_poison_added", 0)
        self.draw_free_early: int = _optional(raw_state, "draw_free_early", 0)
        self.draw_free: int = _optional(raw_state, "draw_free", 0)
        self.draw_pay_early: int = _optional(raw_state, "draw_pay_early", 0)
        self.draw_pay: int = _optional(raw_state, "draw_pay", 0)

        # ---- memory（Watcher/Defect 专属，触达即 raise）----
        self.memory_general = _MemoryShim("state.memory_general")
        # memory_by_card 是嵌套 dict[CardId][ResetSchedule][...] —— 用同一个 shim
        self.memory_by_card = _MemoryShim("state.memory_by_card")


__all__ = [
    "PowerSet",
    "PlayerAdapter",
    "MonsterAdapter",
    "CardAdapter",
    "BottledStateAdapter",
]

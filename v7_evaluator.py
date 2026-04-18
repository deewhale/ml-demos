"""
V7 战斗评估函数 (Run 13: Weighted-sum baseline 重建)

Run 12 的 lex-tuple-as-float 实验失败 (avg 5.56，低于 Run 11 的 9.7)。
本文件回退到 Run 6 baseline 的 weighted sum 架构（avg 10.2），并参考
bottled_ai powers_we_like 把维度扩展到 ~15 维。

硬约束：TurnSolver 用 `best_score >= _SCORE_LETHAL(1e6)` 做 lethal 剪枝
（turn_solver.py 行 415）。因此非 lethal 分支的总分必须严格 < 1e6。
权重设计下 max total ≈ 100k，留 1 个数量级安全边界。

特殊 early-return：
  - player.hp <= 0 → _SCORE_DEATH (-1e6)
  - 无活敌人    → _SCORE_LETHAL + hp * 200 (≥ 1e6)

核心 15 个权重（按 Run 6 报告 + bottled_ai 校准）：
  W_HP_LOST               = 2000   # 玩家挨打每 HP
  W_ENEMIES_KILLED        = 5000   # 每杀一只
  W_VULNERABLE_ON_ENEMY   = 1000   # 敌方 Vulnerable 每层
  W_WEAK_ON_ENEMY         = 1000   # 敌方 Weakened 每层
  W_ENEMY_TOTAL_HP        = -2     # 敌方总 HP
  W_ENEMY_MIN_HP          = -5     # 最低血敌人
  W_GOOD_POWER_ON_SELF    = 500    # 玩家 good power 层数
  W_BAD_DEBUFF_ON_SELF    = -1000  # 玩家 Weak/Vuln/Frail 层数
  W_STANCE_NOT_WRATH      = 500    # 不在 Wrath 结束回合
  W_STANCE_CALM           = 300    # 在 Calm
  W_ENERGY                = 200    # 剩余能量
  W_BLOCK                 = 50     # 玩家 block
  W_HAND_SIZE             = 10     # 手牌数
  W_DAMAGE_DEALT          = 0      # 不用（防 flail 刷 HP）

  非加权：W_LETHAL = _SCORE_LETHAL + hp*200, W_DEATH = _SCORE_DEATH

用法：
    from v7_evaluator import patch_turn_solver_eval
    patch_turn_solver_eval()
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from packages.engine.combat_engine import CombatEngine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Power 分类（抄 bottled_ai powers_we_like）
# ---------------------------------------------------------------------------

GOOD_POWERS = frozenset({
    # Watcher 核心
    "Deva Form", "Devotion", "Establishment", "Nirvana",
    "MentalFortress", "Mental Fortress",
    "Master Reality", "MasterReality",
    "Study", "Rushdown", "LikeWater", "Like Water",
    "Foresight", "BattleHymn", "Battle Hymn",
    # 通用 good
    "DemonForm", "Demon Form", "Metallicize", "Plated Armor", "PlatedArmor",
    "Thorns", "Barricade", "FeelNoPain", "Feel No Pain",
    "Buffer", "Corruption", "Blur", "AfterImage", "After Image",
    "Juggernaut", "Berserk", "Fire Breathing", "FireBreathing",
    "Accuracy", "InfiniteBlades", "Infinite Blades",
    "NoxiousFumes", "Noxious Fumes", "Envenom",
    "Mantra", "MantraInternal",
    "IntangiblePlayer", "Intangible",
    "Echo Form", "EchoForm", "Electro",
    "Machine Learning", "MachineLearning",
    "Tools of the Trade", "ToolsOfTheTrade",
    "Sadistic", "Repair", "Panache", "PanacheInternal",
    "Strength", "Dexterity",
})

BAD_POWERS_ON_SELF = frozenset({
    "Weakened", "Weak", "Frail", "Vulnerable",
    "Entangled", "NoDraw", "No Draw",
    "Confused", "DrawReduction", "Draw Reduction",
    "Hex", "Bias", "LoseDexterity",
})


# ---------------------------------------------------------------------------
# 权重（Run 13 weighted sum）
# ---------------------------------------------------------------------------

W_HP_LOST = 2000.0
W_ENEMIES_KILLED = 5000.0
W_DAMAGE_DEALT = 0.0   # 不用

W_VULNERABLE_ON_ENEMY = 1000.0
W_WEAK_ON_ENEMY = 1000.0
W_ENEMY_TOTAL_HP = -2.0
W_ENEMY_MIN_HP = -5.0

W_GOOD_POWER_ON_SELF = 500.0    # 按 stack 总和
W_BAD_DEBUFF_ON_SELF = -1000.0  # 按 stack 总和

W_STANCE_NOT_WRATH = 500.0
W_STANCE_CALM = 300.0

W_ENERGY = 200.0
W_BLOCK = 50.0
W_HAND_SIZE = 10.0

# max 大致估算：
#   100 hp_loss prevented * 2000 = 2e5（但 hp_lost 只会 ≤ original hp，通常 ≤100）
#   5 kills * 5000 = 2.5e4
#   30 vuln * 1000 = 3e4 / 30 weak * 1000 = 3e4
#   敌方 HP 项是惩罚（负）
#   总计非 lethal 正值峰值 ≈ 1e5，严格 < _SCORE_LETHAL (1e6). ✓


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _sum_status(statuses: dict, names: frozenset) -> int:
    """返回 statuses 中属于 names 的 stack 总和"""
    total = 0
    for n, amt in statuses.items():
        if amt > 0 and n in names:
            total += amt
    return total


def _alive(e) -> bool:
    if getattr(e, "is_dead", None) is True:
        return False
    if e.hp <= 0:
        return False
    if getattr(e, "is_escaping", False):
        return False
    return True


# ---------------------------------------------------------------------------
# 核心评估
# ---------------------------------------------------------------------------

def v7_score_terminal(self, engine, original) -> float:
    """V7 评估函数（Run 13：15 维 weighted sum，重建 Run 6 baseline）。

    签名与 TurnSolver 原版一致。self 是 TurnSolver 实例（未使用）。
    engine 是 terminal 状态，original 是本回合开始的快照。

    关键：和原版 score_terminal 一样，**模拟敌方回合** 得到真实 hp_lost。
    不然评估函数就失去了"block 阻挡伤害"的反馈，bot 会不 block。
    """
    from packages.training.turn_solver import _SCORE_DEATH, _SCORE_LETHAL
    from packages.engine.state.combat import EndTurn

    state = engine.state
    player = state.player

    # 死亡 early-return（lex 最高优先级）
    if player.hp <= 0:
        return _SCORE_DEATH

    # 活敌列表（engine 的 terminal，玩家回合尚未结束）
    living_enemies = [e for e in state.enemies if _alive(e)]

    # 清场 early-return（lex 第二优先级）
    if not living_enemies:
        return _SCORE_LETHAL + player.hp * 200

    # 模拟敌方回合 — 得到真实 HP 损失（与原版 TurnSolver 一致）
    try:
        sim = engine.copy()
        sim.execute_action(EndTurn())
        hp_after = sim.state.player.hp
    except Exception as e:  # noqa: BLE001
        logger.warning("V7: enemy turn simulation failed: %s", e)
        hp_after = player.hp  # fallback：不模拟

    # 模拟后又死了 → 同样 DEATH（相当于下回合必死）
    if hp_after <= 0:
        return _SCORE_DEATH + player.hp  # tie-break 用当前 hp

    actual_hp_lost = max(0, player.hp - hp_after)

    # 以下：weighted sum，保证 < 1e6
    score = 0.0

    # 1. HP 变化（惩罚挨打）— 用模拟后的真实值
    score += -W_HP_LOST * actual_hp_lost

    # 2. 击杀数（本回合死掉的）
    orig_enemies = original.state.enemies
    orig_alive_count = sum(1 for e in orig_enemies if _alive(e))
    now_alive_count = len(living_enemies)
    enemies_killed = max(0, orig_alive_count - now_alive_count)
    score += W_ENEMIES_KILLED * enemies_killed

    # 3. 敌方 debuff（Vulnerable / Weakened）
    total_vuln = sum(e.statuses.get("Vulnerable", 0) for e in living_enemies)
    total_weak = sum(e.statuses.get("Weakened", 0) for e in living_enemies)
    score += W_VULNERABLE_ON_ENEMY * total_vuln
    score += W_WEAK_ON_ENEMY * total_weak

    # 4. 敌方 HP（越低越好）
    total_ehp = sum(max(0, e.hp) for e in living_enemies)
    min_ehp = min((max(0, e.hp) for e in living_enemies), default=0)
    score += W_ENEMY_TOTAL_HP * total_ehp
    score += W_ENEMY_MIN_HP * min_ehp

    # 5. 玩家 powers
    good_stacks = _sum_status(player.statuses, GOOD_POWERS)
    bad_stacks = _sum_status(player.statuses, BAD_POWERS_ON_SELF)
    score += W_GOOD_POWER_ON_SELF * good_stacks
    score += W_BAD_DEBUFF_ON_SELF * bad_stacks

    # 6. Stance
    stance = state.stance
    if stance != "Wrath":
        score += W_STANCE_NOT_WRATH
    if stance == "Calm":
        score += W_STANCE_CALM

    # 7. 资源
    score += W_ENERGY * state.energy
    score += W_BLOCK * player.block
    # 手牌数
    hand = getattr(state, "hand", None)
    if hand is not None:
        score += W_HAND_SIZE * len(hand)

    return score


# ---------------------------------------------------------------------------
# 性能补丁
# ---------------------------------------------------------------------------

def _patch_power_resolution_cache():
    """cProfile 发现 resolve_power_id 在 DFS 中被调用 3M+ 次，加 lru_cache 压到 <1%。"""
    from functools import lru_cache
    from packages.engine.content import powers as _pm

    _orig_norm = _pm.normalize_power_id
    if not getattr(_orig_norm, "_v7_cached", False):
        cached_norm = lru_cache(maxsize=4096)(_orig_norm)
        cached_norm._v7_cached = True  # type: ignore
        _pm.normalize_power_id = cached_norm

    _orig_build = _pm._build_normalized_power_index
    if not getattr(_orig_build, "_v7_cached", False):
        _cached_index_holder = {}
        def cached_build_index():
            if "v" not in _cached_index_holder:
                _cached_index_holder["v"] = _orig_build()
            return _cached_index_holder["v"]
        cached_build_index._v7_cached = True  # type: ignore
        _pm._build_normalized_power_index = cached_build_index

    _orig_resolve = _pm.resolve_power_id
    if not getattr(_orig_resolve, "_v7_cached", False):
        cached_resolve = lru_cache(maxsize=2048)(_orig_resolve)
        cached_resolve._v7_cached = True  # type: ignore
        _pm.resolve_power_id = cached_resolve

    logger.info("V7: patched power-resolution functions with lru_cache")


_patched = False


def patch_turn_solver_eval():
    """用 v7_score_terminal 替换 TurnSolver._score_terminal + 性能缓存补丁。幂等。"""
    global _patched
    if _patched:
        return
    from packages.training.turn_solver import TurnSolver
    TurnSolver._score_terminal = v7_score_terminal
    _patch_power_resolution_cache()
    _patched = True
    logger.info("TurnSolver._score_terminal patched with V7 evaluator (Run 13 weighted sum)")

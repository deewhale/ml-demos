"""V8 Reward function（分层评估）。

按 docs/v8_implementation_design.md 组件 4 设计：
- 每个元决策 step 给 step reward = w1·Δ牌组强度 - w2·Δhp_loss + w3·节点收益
- Episode 结束给 final reward = won·100 + final_floor·1
- 战斗内不给 RL step reward（战斗内由搜索作老师，预训练阶段处理）

设计原则对照（docs/v8_design_principles.md）：
- 路线评估融合"战损 + 牌组强度"两个维度（用户原话第 1 点）
- 选卡通过 Δ牌组强度 反映 leave-one-out 边际贡献（用户原话第 3 点）
- 不是单 game_won 信号反推
- 不依赖 v8_strategy 启发式

不在本文件内：
- evaluate_deck（v8/deck_evaluator.py 已实现）
- env step 内部 reward 调用（v8/env.py 串起来）
"""

from __future__ import annotations

from typing import Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from v8.state import V8State


# ============================================================
# Reward weights（design doc 默认值，后续可调）
# ============================================================

# 牌组强度变化权重（per-step Δstrength * W）
W_DECK_STRENGTH_DELTA: float = 1.0

# HP 损失惩罚权重（hp_loss 比例 ∈ [0,1] 乘 W * 100 让数量级与 strength 对齐）
W_HP_LOSS: float = 0.5

# 节点本身收益权重（事件 / 商店 / 休息特殊收益）
W_NODE_REWARD: float = 0.5

# 通关大 bonus（model 知道赢是终极目标）
GAME_WON_BONUS: float = 100.0

# 每过一层小 bonus（避免 0 winning data 时 model 完全没信号）
FLOOR_REACHED_BONUS: float = 1.0


# ============================================================
# 牌组强度 4 维 → scalar
# ============================================================

# 4 维内部权重（compute_deck_strength_score）：
#   damage_dealt × +1.0   输出多 = 好
#   damage_taken × -1.0   战损低 = 好
#   turns_to_win × -0.5   速度快 = 好
#   win_rate     × +50.0  胜率高 = 关键 indicator
W_STRENGTH_DAMAGE_DEALT: float = 1.0
W_STRENGTH_DAMAGE_TAKEN: float = -1.0
W_STRENGTH_TURNS_TO_WIN: float = -0.5
W_STRENGTH_WIN_RATE: float = 50.0


def compute_deck_strength_score(strength: Optional[Dict[str, float]]) -> float:
    """4 维牌组强度 → 加权 scalar。

    None / 缺字段时返回 0（视为未评估，不给 reward 信号）。
    """
    if strength is None:
        return 0.0
    return (
        float(strength.get("damage_dealt", 0.0)) * W_STRENGTH_DAMAGE_DEALT
        + float(strength.get("damage_taken", 0.0)) * W_STRENGTH_DAMAGE_TAKEN
        + float(strength.get("turns_to_win", 0.0)) * W_STRENGTH_TURNS_TO_WIN
        + float(strength.get("win_rate", 0.0)) * W_STRENGTH_WIN_RATE
    )


# ============================================================
# 节点收益默认值（env step 内部 detect 后传进来）
# ============================================================

# env detect 后传给 compute_step_reward 的 node_reward 默认值（参考量级）：
#   ?事件成功（拿到好东西）      : +5
#   $买到 relic                   : +3
#   R 升级 / 删卡                 : +2
#   宝箱开 relic                  : +3
#   普通战斗                       : 0   （已经在 deck strength 提升里反映）
#   战斗后 CARD_REWARDS 选 / skip : 0   （Δstrength 反映）
NODE_REWARD_EVENT_SUCCESS: float = 5.0
NODE_REWARD_SHOP_RELIC: float = 3.0
NODE_REWARD_REST_USE: float = 2.0
NODE_REWARD_TREASURE: float = 3.0


# ============================================================
# 主入口：step / final reward
# ============================================================

def compute_step_reward(
    prev_state: "V8State",
    next_state: "V8State",
    prev_deck_strength: Optional[Dict[str, float]],
    next_deck_strength: Optional[Dict[str, float]],
    node_reward: float = 0.0,
) -> float:
    """每个元决策 step 的 reward。

    按 design doc 4.1 / 4.2：
        r = W_DECK·Δstrength - W_HP·Δhp_loss + W_NODE·node_reward

    参数：
        prev_state / next_state: 前一个 / 当前元决策点 V8State（hp / max_hp 用于算 hp_loss）
        prev_deck_strength / next_deck_strength: 前后两次 evaluate_deck 结果（None 视为未评估）
        node_reward: env 内 detect 出的节点本身特殊收益（默认 0）

    返回：
        float reward（不是 NaN / inf）
    """
    # Δ牌组强度（scalar）
    delta_strength = (
        compute_deck_strength_score(next_deck_strength)
        - compute_deck_strength_score(prev_deck_strength)
    )

    # Δhp_loss 比例 ∈ [-1, 1]：
    #   正值 = 这一步丢血（next_hp_pct < prev_hp_pct）
    #   负值 = 这一步回血（rest / 事件回血）
    prev_max = max(int(getattr(prev_state, "max_hp", 0) or 0), 1)
    next_max = max(int(getattr(next_state, "max_hp", 0) or 0), 1)
    prev_hp_pct = float(getattr(prev_state, "hp", 0) or 0) / prev_max
    next_hp_pct = float(getattr(next_state, "hp", 0) or 0) / next_max
    delta_hp_loss = prev_hp_pct - next_hp_pct  # 丢血为正

    # × 100 让 hp_loss 项与 strength 项数量级接近（典型 strength 量级 50-200）
    return (
        W_DECK_STRENGTH_DELTA * delta_strength
        - W_HP_LOSS * delta_hp_loss * 100.0
        + W_NODE_REWARD * float(node_reward)
    )


def compute_final_reward(game_won: bool, final_floor: int) -> float:
    """Episode 结束时 final reward。

    通关大 bonus + final_floor 兜底（避免 0 winning data 时 model 完全没信号）。
    """
    won_bonus = GAME_WON_BONUS if game_won else 0.0
    return won_bonus + FLOOR_REACHED_BONUS * float(int(final_floor or 0))


__all__ = [
    "compute_deck_strength_score",
    "compute_step_reward",
    "compute_final_reward",
    # weights (导出方便 trainer / unit test 引用)
    "W_DECK_STRENGTH_DELTA",
    "W_HP_LOSS",
    "W_NODE_REWARD",
    "GAME_WON_BONUS",
    "FLOOR_REACHED_BONUS",
    "W_STRENGTH_DAMAGE_DEALT",
    "W_STRENGTH_DAMAGE_TAKEN",
    "W_STRENGTH_TURNS_TO_WIN",
    "W_STRENGTH_WIN_RATE",
    "NODE_REWARD_EVENT_SUCCESS",
    "NODE_REWARD_SHOP_RELIC",
    "NODE_REWARD_REST_USE",
    "NODE_REWARD_TREASURE",
]

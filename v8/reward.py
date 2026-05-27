"""V8 Reward function（按真实战斗结果评分）。

设计：
- 每个元决策 step 给 step reward =
    - W_HP_LOSS * Δhp_loss     （hp 损失惩罚，丢血为负）
    - + W_NODE_REWARD * node_reward （事件/商店/休息/宝箱节点收益）
    - + combat_reward          （本 step 发生过的真实战斗结果分，env 算好喂进来）
- Episode 结束给 final reward = won·100 + final_floor·1

关键变化（2026-05-25, batch_v38 起）：
- 去掉模拟战评分（W_DECK_STRENGTH_DELTA / W_STRENGTH_WIN_RATE 那套）。
  之前每选一张卡都跑 12 场模拟战算评分，跟实战脱节。
- 改成每场真实战斗结束直接给评分（赢没赢 / 血损 / 回合数 / 伤害比）。
- 不累加：每场战斗的分数只在那一步 emit 完就结束，不进 episode 级 buffer。
  RL 自身通过 GAE 把战斗结果反推到选卡 / 走路决策。
- 旧的 deck_evaluator 文件保留但不调用（保留是因为 web / 历史训练日志可能引用）。

不在本文件内：
- env step 内部 reward 调用串起来（v8/env.py 在 _log_combat_exit 算 combat_reward
  并塞到下一次 step() 输出）
"""

from __future__ import annotations

from typing import Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from v8.state import V8State


# ============================================================
# Reward weights（step / final）
# ============================================================

# HP 损失惩罚权重（hp_loss 比例 ∈ [0,1] 乘 W * 100 让数量级与 combat reward 对齐）
W_HP_LOSS: float = 0.5

# 节点本身收益权重（事件 / 商店 / 休息特殊收益）
W_NODE_REWARD: float = 0.5

# 通关大 bonus（model 知道赢是终极目标）
GAME_WON_BONUS: float = 100.0

# 每过一层小 bonus（避免 0 winning data 时 model 完全没信号）
FLOOR_REACHED_BONUS: float = 1.0


# ============================================================
# 单场真实战斗的 reward 权重
# ============================================================

# 胜负主导项（赢一场 +30，输直接 -W_LOSE）
W_COMBAT_WIN: float = 30.0
W_COMBAT_LOSE: float = 30.0   # 输的惩罚（绝对值，符号在公式里取负）

# 每场战斗丢失 hp 的惩罚（绝对 hp 数 × W）
W_HP_LOSS_PER_BATTLE: float = 1.0

# 战斗回合数惩罚（回合多 = 浪费 / 拖战）
W_TURNS: float = 0.5

# 输出效率奖励（damage_dealt / enemy_total_max_hp ∈ [0, 1+] × W）
W_DAMAGE_RATIO: float = 5.0


def compute_combat_reward(
    *,
    won: bool,
    hp_lost: int,
    turns: int,
    damage_dealt: int,
    enemy_total_max_hp: int,
) -> float:
    """单场真实战斗结束时的奖励分。

    每场战斗只算一次、不累加进 episode buffer，env 算完直接塞进当步 step_reward。

    公式：
        combat_reward = (won? +W_WIN : -W_LOSE)
                       - W_HP_LOSS_PER_BATTLE * hp_lost
                       - W_TURNS * turns
                       + W_DAMAGE_RATIO * (damage_dealt / enemy_total_max_hp)

    参数：
        won:                这场战斗有没赢（player 没死且 enemies 全死）
        hp_lost:            本场损失的 HP（绝对值 ≥ 0）
        turns:              本场战斗回合数（≥ 1）
        damage_dealt:       对敌人造成的总伤害（绝对值 ≥ 0）
        enemy_total_max_hp: 进入战斗时所有敌人 max_hp 之和（用于算输出效率比）

    返回：
        float（不是 NaN / inf）
    """
    win_term = W_COMBAT_WIN if won else -W_COMBAT_LOSE

    hp_lost = max(int(hp_lost or 0), 0)
    turns = max(int(turns or 0), 0)
    damage_dealt = max(int(damage_dealt or 0), 0)
    enemy_max = max(int(enemy_total_max_hp or 0), 1)  # 防除零

    damage_ratio = float(damage_dealt) / float(enemy_max)
    # 限制 ratio 上限（防 overkill 异常拉高，比如某些 reset relic 触发的大伤害）
    if damage_ratio > 2.0:
        damage_ratio = 2.0

    return (
        win_term
        - W_HP_LOSS_PER_BATTLE * float(hp_lost)
        - W_TURNS * float(turns)
        + W_DAMAGE_RATIO * damage_ratio
    )


# ============================================================
# 节点收益默认值（env detect 后传给 compute_step_reward）
# ============================================================

# env detect 后传给 compute_step_reward 的 node_reward 默认值：
#   ?事件成功（拿到好东西）      : +5
#   $买到 relic                   : +3
#   R 升级 / 删卡                 : +2
#   宝箱开 relic                  : +3
#   普通战斗                       : 0   （战斗 reward 由 combat_reward 直接给）
#   战斗后 CARD_REWARDS 选 / skip : 0   （让 RL 通过下场战斗结果反推）
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
    node_reward: float = 0.0,
    combat_reward: float = 0.0,
) -> float:
    """每个元决策 step 的 reward。

    公式：
        r = - W_HP_LOSS * Δhp_loss_ratio * 100
            + W_NODE_REWARD * node_reward
            + combat_reward

    参数：
        prev_state / next_state: 前一个 / 当前元决策点 V8State（hp / max_hp 用于算 hp_loss）
        node_reward: env 内 detect 出的节点本身特殊收益（默认 0）
        combat_reward: 本 step 期间发生过的真实战斗结果分（env 在 _log_combat_exit
            算好缓存，下一次 step 时取出来一次性塞进去）。0 表示本 step 没战斗。

    返回：
        float reward（不是 NaN / inf）
    """
    # Δhp_loss 比例 ∈ [-1, 1]：
    #   正值 = 这一步丢血（next_hp_pct < prev_hp_pct）
    #   负值 = 这一步回血（rest / 事件回血）
    prev_max = max(int(getattr(prev_state, "max_hp", 0) or 0), 1)
    next_max = max(int(getattr(next_state, "max_hp", 0) or 0), 1)
    prev_hp_pct = float(getattr(prev_state, "hp", 0) or 0) / prev_max
    next_hp_pct = float(getattr(next_state, "hp", 0) or 0) / next_max
    delta_hp_loss = prev_hp_pct - next_hp_pct  # 丢血为正

    # × 100 让 hp_loss 项数量级与 combat reward 接近
    return (
        - W_HP_LOSS * delta_hp_loss * 100.0
        + W_NODE_REWARD * float(node_reward)
        + float(combat_reward)
    )


def compute_final_reward(game_won: bool, final_floor: int) -> float:
    """Episode 结束时 final reward。

    通关大 bonus + final_floor 兜底（避免 0 winning data 时 model 完全没信号）。
    """
    won_bonus = GAME_WON_BONUS if game_won else 0.0
    return won_bonus + FLOOR_REACHED_BONUS * float(int(final_floor or 0))


__all__ = [
    "compute_step_reward",
    "compute_final_reward",
    "compute_combat_reward",
    # weights (导出方便 trainer / unit test 引用)
    "W_HP_LOSS",
    "W_NODE_REWARD",
    "GAME_WON_BONUS",
    "FLOOR_REACHED_BONUS",
    "W_COMBAT_WIN",
    "W_COMBAT_LOSE",
    "W_HP_LOSS_PER_BATTLE",
    "W_TURNS",
    "W_DAMAGE_RATIO",
    "NODE_REWARD_EVENT_SUCCESS",
    "NODE_REWARD_SHOP_RELIC",
    "NODE_REWARD_REST_USE",
    "NODE_REWARD_TREASURE",
]

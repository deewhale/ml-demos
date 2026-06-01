"""V8 Reward function。

设计来源：docs/v8_rl_fix_plan_2026-05-29.md「核心设计 / 阶段 2」。

奖励重对齐（2026-06-01 起，本批改动）—— **删 strength_reward（刷分元凶），
改「真实进度 / 过 boss / 通关」为绝对主轴**：

归因结论：旧奖励里 strength_reward（8×Δdeck_strength）占总奖励 ~73%，每打一仗
按伤害发大奖，与「赢」无关 → 死亡局也净赚 +485。把稀疏的真实进度信号淹没 100 倍。

本批改动：
- **彻底删掉 strength_reward**：`compute_strength_growth_reward` 不再进 step reward。
  （card_scorer.update_from_combat / deck_strength() 在 env 里**保留继续调用**——
  deck_strength 数值还要算、还要打日志，只是不进奖励；留着给下一步当模型特征。）
- **删掉 combat_reward 里的 `W_DAMAGE_RATIO × damage_ratio`** 项（刷战斗味、奖打架）。
- **战斗胜负调成小信号**：赢 +W_COMBAT_WIN(=2) / 输 -W_COMBAT_LOSE(=5)。保留这个小
  信号让模型在意单场胜负，但绝不大到变成「奖打架」。
- **进度（主轴，稠密）**：每到达一个新楼层 +W_FLOOR_PROGRESS(=3)，由 env 在检测到
  floor 变大时给。这是主信号。
- **过 act boss**：+BOSS_BEAT_BONUS(=25)。
- **通关**：+GAME_WON_BONUS(=100)。
- **避免和 final 重复计**：final_reward 里去掉 `FLOOR_REACHED_BONUS×final_floor`
  （改由 per-floor step reward 承担），final 只留通关 bonus。
- **存活 terminal 信号（小）**：episode 结束 +W_HP_TERMINAL(=5)×final_hp_ratio，
  让「活着到第 N 层」> 「死在第 N 层」。
- **「健康到达 act boss」高效奖励（v5 起，+W_BOSS_HP(=15)×boss_arrival_hp_ratio）**：
  到 boss 战时按**进入那一刻**的血量比给奖（血越满奖越多）。量的是「走到 boss 面前
  还剩多少血」，不是「boss 战少挨打」→ 奖励「拿好卡前面打得干净」，天然反 skip-all
  （好牌组到 boss 血多 → 奖高；skip-all 残血到 boss → 奖低、吃亏）。

权衡：走到第 10 层就死 ≈ 30 量级（主要来自 floor）；通关 ≈ 300+ 量级。
进度 / 过 boss / 通关在总奖励里占绝对主导，任何残留战斗 / node 小信号都不接近其量级。

历史（已删除的旧设计，留档防回退）：
- ≤2026-05-31 曾有 strength_reward（8×Δdeck_strength）+ combat 输出效率 reward
  —— 本批删掉（刷分、死亡局净赚 +485）。
- batch_v38 起曾用 real-combat hp/回合 reward（commit 8b9485a）—— 已删。
- 更早（≤v37）用 deck_evaluator 模拟战评分 —— 早已停用，文件保留供 web / 历史日志引用。

不在本文件内：
- per-card 评分 + 牌组总分聚合：v8/card_scorer.py（保留计算，不进奖励）
- env step 内部 reward 串起来 + 进度奖励触发：v8/env.py（_maybe_log_floor 给
  per-floor 进度奖励、_log_combat_exit 给 combat + 过 boss 奖励、reset 给存活 terminal）
"""

from __future__ import annotations

from typing import Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from v8.state import V8State


# ============================================================
# Reward weights（step / final / 进度）
# ============================================================

# 节点本身收益权重（事件 / 商店 / 休息特殊收益）
W_NODE_REWARD: float = 0.5

# 通关大 bonus（model 知道赢是终极目标）—— 进度奖励顶层（+++）
GAME_WON_BONUS: float = 100.0

# 进度奖励主轴（稠密）：每到达一个新楼层 +W_FLOOR_PROGRESS。
# 由 env 在 _maybe_log_floor 检测到 floor 变大时给，是整个奖励的主信号。
# 一个走到第 10 层的局光这项就攒 ~30，远大于残留的战斗 / node 小信号。
W_FLOOR_PROGRESS: float = 3.0

# 过每个 act boss 的中层进度奖励（++，介于过层 +3 与通关 +100 之间）。
# 对齐「到新楼层 + / 过每个 act boss ++ / 通关 +++」。
BOSS_BEAT_BONUS: float = 25.0

# 存活 terminal 信号（小）：episode 结束 + W_HP_TERMINAL × final_hp_ratio。
# 让「活着到第 N 层」> 「死在第 N 层」，但权重小到不和进度量级竞争。
W_HP_TERMINAL: float = 5.0

# 「健康到达 act boss」高效奖励（v5 起）：到达 / 进入每个 act boss 战时，按**进入
# boss 战那一刻**的 hp_ratio 给 + W_BOSS_HP × hp_ratio。即到 boss 时血越满奖越多。
#
# **为什么不会逼 skip-all（防 skip-all 设计核心）**：
#   这个奖励量的是「走到 boss 面前时还剩多少血」，不是「boss 战少挨打」。
#   - 好牌组 → 前面每场打得干净 → 到 boss 时血更多 → 这个奖更高
#     ⇒ 奖励「拿能打的好卡」，跟 skip-all 反向。
#   - skip-all 的小弱牌组 → 前面被磨 → 到 boss 时血更少 → 这个奖更低
#     ⇒ skip-all 在这个信号下吃亏、不占便宜。
#
# 权重取中等（15）：让「健康到达」成有意义信号（满血到 boss ≈ +15，介于过层 +3
# 与过 boss +25 之间），但不盖过通关（+100）/ 过 boss（+25）主轴。
W_BOSS_HP: float = 15.0


# ============================================================
# 单场真实战斗的 reward 权重（小信号，不压进度主轴）
# ============================================================

# 胜负小信号（赢 +W_COMBAT_WIN，输 -W_COMBAT_LOSE）。
# 量级要求（关键）：调到很小，让模型在意单场胜负但绝不变成「奖打架」。
# 输的惩罚 (5) > 赢的奖励 (1)：输≈死≈断进度本身就该罚得重一点。
# W_COMBAT_WIN 压到 1.0：一个走到 floor10 的局约打 7-10 仗，combat 累计 ~7-10，
# 远小于同局 floor 进度 (10×3=30)；即便极端多仗也压不过进度主轴 —— 满足
# 「任何残留战斗信号不接近进度量级」的对齐要求。
W_COMBAT_WIN: float = 1.0
W_COMBAT_LOSE: float = 5.0   # 输的惩罚（绝对值，符号在公式里取负）


def compute_combat_reward(*, won: bool) -> float:
    """单场真实战斗结束时的即时小信号（只剩胜负，刷分项已删）。

    每场战斗只算一次、不累加进 episode buffer，env 算完直接塞进当步 step_reward。
    进度奖励（过层 / 过 boss / 通关）**不在这里**，由 env / final 单独算。

    公式（奖励重对齐，2026-06-01 起）：
        combat_reward = (won? +W_COMBAT_WIN : -W_COMBAT_LOSE)

    **删除项**（reward hacking 源头）：
        - W_DAMAGE_RATIO * damage_ratio     （刷战斗味、奖打架——本批删）
        - W_HP_LOSS_PER_BATTLE * hp_lost    （丢血罚，更早已删）
        - W_TURNS * turns                   （回合罚，更早已删）

    参数：
        won: 这场战斗有没赢（player 没死且 enemies 全死）

    返回：
        float（不是 NaN / inf）
    """
    return W_COMBAT_WIN if won else -W_COMBAT_LOSE


def compute_floor_progress_reward(num_new_floors: int = 1) -> float:
    """到达新楼层的进度奖励（主轴）= num_new_floors × W_FLOOR_PROGRESS。

    env 在 _maybe_log_floor 检测到 floor 变大时给。通常一次 +1 层；
    极端情况（一次跳多层）按层数线性给，确保「越深奖励越高」单调。
    """
    return W_FLOOR_PROGRESS * float(max(int(num_new_floors or 0), 0))


def compute_boss_beat_reward() -> float:
    """过一个 act boss 的进度奖励（env 在 boss 战斗胜利时给一次）。"""
    return BOSS_BEAT_BONUS


def compute_boss_hp_reward(boss_arrival_hp_ratio: float) -> float:
    """「健康到达 act boss」奖励 = W_BOSS_HP × boss_arrival_hp_ratio。

    env 在 boss 战结束时给一次，用的是**进入 boss 战那一刻**的 hp_ratio
    （current_hp / max_hp ∈ [0,1]，即「走到 boss 面前还剩多少血」），
    而不是 boss 战内的丢血——所以奖励「健康地走到 boss」而非「boss 战少挨打」，
    天然反 skip-all（好牌组前面打得干净 → 到 boss 血多 → 奖高）。

    boss_arrival_hp_ratio ∈ [0, 1]。残血到 boss → 接近 0；满血到 boss → 1×W。
    """
    r = float(boss_arrival_hp_ratio or 0.0)
    r = max(0.0, min(1.0, r))  # clamp 到 [0,1]
    return W_BOSS_HP * r


def compute_hp_terminal_reward(final_hp_ratio: float) -> float:
    """episode 结束时的存活 terminal 信号 = W_HP_TERMINAL × final_hp_ratio。

    final_hp_ratio ∈ [0, 1]（current_hp / max_hp）。死亡局 = 0，满血通关 = 1×W。
    小权重，只为「活着到第 N 层 > 死在第 N 层」破平局，不和进度量级竞争。
    """
    r = float(final_hp_ratio or 0.0)
    r = max(0.0, min(1.0, r))  # clamp 到 [0,1]
    return W_HP_TERMINAL * r


# ============================================================
# 节点收益默认值（env detect 后传给 compute_step_reward）
# ============================================================

# env detect 后传给 compute_step_reward 的 node_reward 默认值：
#   ?事件成功（拿到好东西）      : +5
#   $买到 relic                   : +3
#   R 升级 / 删卡                 : +2
#   宝箱开 relic                  : +3
#   普通战斗                       : 0   （战斗 reward 由 combat_reward 直接给）
#   战斗后 CARD_REWARDS 选 / skip : 0   （让 RL 通过下场战斗结果 + 牌组实力增长反推）
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
    floor_reward: float = 0.0,
) -> float:
    """每个元决策 step 的 reward。

    公式（奖励重对齐，2026-06-01 起）：
        r = W_NODE_REWARD * node_reward
            + combat_reward
            + floor_reward

    其中：
        combat_reward: env 在战斗结束时一次性塞进来的**单场胜负小信号 + 过 act boss
            进度**（compute_combat_reward + compute_boss_beat_reward）。
        floor_reward: env 在本 step 期间检测到 floor 变大时累计的**进度主轴奖励**
            （compute_floor_progress_reward）。

    **删除项**（reward hacking 源头）：
        - 牌组实力增长（strength_reward = 8×Δdeck_strength）—— 刷分元凶，本批删
        - 单场输出效率（W_DAMAGE_RATIO×damage_ratio）—— 奖打架，本批删
        - 每点丢血罚 / hp/回合罚 —— 更早已删

    参数：
        prev_state / next_state: 前一个 / 当前元决策点 V8State（保留签名兼容 caller；
            hp 不进 step reward，存活信号改 terminal 给，见 compute_hp_terminal_reward）。
        node_reward: env 内 detect 出的节点本身特殊收益（默认 0）
        combat_reward: 本 step 期间的真实战斗胜负小信号 + 过 boss 进度（默认 0）
        floor_reward: 本 step 期间到达的新楼层进度奖励（默认 0，进度主轴）

    返回：
        float reward（不是 NaN / inf）
    """
    return (
        W_NODE_REWARD * float(node_reward)
        + float(combat_reward)
        + float(floor_reward)
    )


def compute_final_reward(game_won: bool, final_hp_ratio: float = 0.0) -> float:
    """Episode 结束时 final reward。

    进度奖励顶层（+++）：通关大 bonus + 存活 terminal 小信号。
    **不再含 FLOOR_REACHED_BONUS×final_floor**——逐层进度已由 per-floor step reward
    （compute_floor_progress_reward）承担，避免和 final 重复计。
    过 act boss 的 ++ 在战斗结束时单独给（compute_boss_beat_reward），不在这里。

    参数：
        game_won: 是否通关。
        final_hp_ratio: episode 结束时 current_hp / max_hp ∈ [0,1]，给小的存活信号。
    """
    won_bonus = GAME_WON_BONUS if game_won else 0.0
    return won_bonus + compute_hp_terminal_reward(final_hp_ratio)


__all__ = [
    "compute_step_reward",
    "compute_final_reward",
    "compute_combat_reward",
    "compute_floor_progress_reward",
    "compute_boss_beat_reward",
    "compute_boss_hp_reward",
    "compute_hp_terminal_reward",
    # weights (导出方便 trainer / unit test 引用)
    "W_NODE_REWARD",
    "GAME_WON_BONUS",
    "W_FLOOR_PROGRESS",
    "BOSS_BEAT_BONUS",
    "W_HP_TERMINAL",
    "W_BOSS_HP",
    "W_COMBAT_WIN",
    "W_COMBAT_LOSE",
    "NODE_REWARD_EVENT_SUCCESS",
    "NODE_REWARD_SHOP_RELIC",
    "NODE_REWARD_REST_USE",
    "NODE_REWARD_TREASURE",
]

"""V8 Reward function。

设计来源：docs/v8_rl_fix_plan_2026-05-29.md「核心设计 / 阶段 2」。

阶段 2 底座（2026-05-29 起，本批改动）—— 删 hp/回合战斗罚，换「牌组实力增长 + 真实进度」：
- **删掉**单场战斗的 hp/回合惩罚（`-W_HP_LOSS_PER_BATTLE*hp_lost - W_TURNS*turns`）。
  这两项是 reward hacking 的源头（诊断：模型 skip 所有卡保命 → 牌组永远 11 张烂牌 → 通关恒 0）。
- **删掉**每元决策 step 的「每点丢血惩罚」（`-W_HP_LOSS*Δhp_loss*100`）。HP 退为路线状态参考，
  不再作独立惩罚（其奖励影响只通过「死亡 → 进度低」体现，不变相惩罚拿卡）。
- 战斗本身保留一个**小**的胜负 + 输出效率信号（赢/打出伤害是真本事，不惩罚拿卡）。
- **新增**「牌组实力增长」奖励：每场战斗结束、card_scorer 重算牌组总分后，
  奖励 += Δdeck_strength × W_STRENGTH_GROWTH（牌里的卡这场证明了价值 → 实力涨 → 给正奖励，
  经 GAE 回溯到选卡 / 路线决策）。由 env 在 _log_combat_exit 算好、塞进 combat_reward。
- **进度奖励**：到新楼层 + / 过 act boss ++ / 通关 +++。真实、不可伪造、对齐北极星。

历史（已删除的旧设计，留档防回退）：
- batch_v38 起曾用 real-combat hp/回合 reward（commit 8b9485a）—— 本批删掉。
- 更早（≤v37）用 deck_evaluator 模拟战评分 —— 早已停用，文件保留供 web / 历史日志引用。

不在本文件内：
- per-card 评分 + 牌组总分聚合：v8/card_scorer.py
- env step 内部 reward 串起来 + 牌组实力增量计算：v8/env.py（_log_combat_exit 算
  combat_reward + Δdeck_strength，塞进下一次 step() 输出）
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

# 通关大 bonus（model 知道赢是终极目标）—— 进度奖励顶层
GAME_WON_BONUS: float = 100.0

# 每过一层小 bonus（稠密兜底，避免 0 winning data 时 model 完全没信号）
FLOOR_REACHED_BONUS: float = 1.0

# 过每个 act boss 的中层进度奖励（介于过层 +1 与通关 +100 之间）。
# 对齐计划「到新楼层 + / 过每个 act boss ++ / 通关 +++」。
# 初始值待训练时调（见两轴校验：要比单场战斗胜负信号显著大，才能把「过 boss」
# 这个稀疏 ground-truth 拉出来）。
BOSS_BEAT_BONUS: float = 25.0


# ============================================================
# 单场真实战斗的 reward 权重（已删 hp/回合罚）
# ============================================================

# 胜负信号（保留但调小：赢一场 +W_WIN，输 -W_LOSE）。
# 调小理由（计划「避免重新引入会惩罚拿卡的项」）：胜负本身不该压过牌组实力增长 /
# 进度；保留小幅胜负 + 输出效率作为「这场打得怎么样」的即时信号即可。
# 初始值待训练时调。
W_COMBAT_WIN: float = 10.0
W_COMBAT_LOSE: float = 10.0   # 输的惩罚（绝对值，符号在公式里取负）

# 输出效率奖励（damage_dealt / enemy_total_max_hp ∈ [0, 上限] × W）。
# 这是「打出伤害」的奖励，不惩罚拿卡 → 保留。
W_DAMAGE_RATIO: float = 5.0

# 牌组实力增长奖励系数（Δdeck_strength × W）。
# card_scorer 的 deck_strength 已对伤害 / 格挡做过 /50 归一，单场一张主力攻击卡
# Δ 约 ~1（量纲）；× 8 让「牌里的卡这场证明了价值」的增量奖励落在和单场胜负
# (±10) 同量级、又不压过「过 boss +25 / 通关 +100」的进度奖励。
# 初始值待训练时调（两轴校验：要能把「选到日后强的卡」推出来，又不空刷）。
W_STRENGTH_GROWTH: float = 8.0


def compute_combat_reward(
    *,
    won: bool,
    damage_dealt: int,
    enemy_total_max_hp: int,
) -> float:
    """单场真实战斗结束时的即时奖励分（已删 hp/回合罚）。

    每场战斗只算一次、不累加进 episode buffer，env 算完直接塞进当步 step_reward。
    牌组实力增长奖励 + 进度奖励**不在这里**，由 env 单独算（见 _log_combat_exit）。

    公式（阶段 2 底座，2026-05-29 起）：
        combat_reward = (won? +W_WIN : -W_LOSE)
                       + W_DAMAGE_RATIO * (damage_dealt / enemy_total_max_hp)

    **删除项**（reward hacking 源头 + 变相惩罚拿卡）：
        - W_HP_LOSS_PER_BATTLE * hp_lost   （丢血罚）
        - W_TURNS * turns                  （回合罚）

    参数：
        won:                这场战斗有没赢（player 没死且 enemies 全死）
        damage_dealt:       对敌人造成的总伤害（绝对值 ≥ 0）
        enemy_total_max_hp: 进入战斗时所有敌人 max_hp 之和（用于算输出效率比）

    返回：
        float（不是 NaN / inf）
    """
    win_term = W_COMBAT_WIN if won else -W_COMBAT_LOSE

    damage_dealt = max(int(damage_dealt or 0), 0)
    enemy_max = max(int(enemy_total_max_hp or 0), 1)  # 防除零

    damage_ratio = float(damage_dealt) / float(enemy_max)
    # 限制 ratio 上限（防 overkill 异常拉高，比如某些 reset relic 触发的大伤害）
    if damage_ratio > 2.0:
        damage_ratio = 2.0

    return win_term + W_DAMAGE_RATIO * damage_ratio


def compute_strength_growth_reward(delta_strength: float) -> float:
    """牌组实力增长奖励 = Δdeck_strength × W_STRENGTH_GROWTH。

    delta_strength: 本场战斗后牌组总分 − 上次记录的牌组总分（card_scorer.deck_strength 之差）。
        > 0：牌里的卡这场证明了价值（打了伤害 / 加了格挡）→ 正奖励。
        = 0：没新出力（如没打仗、或这局这套牌已稳定）→ 0，不空刷。
        < 0 几乎不会发生（累计分单调不减；牌组缩水如删卡才可能负，也合理：删掉的卡分没了）。

    经 GAE 回溯归给选卡 / 路线决策。
    """
    return W_STRENGTH_GROWTH * float(delta_strength)


def compute_boss_beat_reward() -> float:
    """过一个 act boss 的进度奖励（env 在 boss 战斗胜利时给一次）。"""
    return BOSS_BEAT_BONUS


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
) -> float:
    """每个元决策 step 的 reward。

    公式（阶段 2 底座，2026-05-29 起——**删了每点丢血惩罚**）：
        r = W_NODE_REWARD * node_reward
            + combat_reward

    其中 combat_reward 由 env 在战斗结束时一次性塞进来，已含：
        - 单场胜负 + 输出效率（compute_combat_reward）
        - 牌组实力增长（compute_strength_growth_reward）
        - 过 act boss 进度（compute_boss_beat_reward）

    **删除项**（HP 退为路线状态参考，不作独立惩罚）：
        - W_HP_LOSS * Δhp_loss_ratio * 100   （每点丢血罚 → reward hacking 源头之一）

    参数：
        prev_state / next_state: 前一个 / 当前元决策点 V8State（保留签名兼容 caller；
            本批起 hp 不再进 reward，仅留作未来路线状态扩展位）。
        node_reward: env 内 detect 出的节点本身特殊收益（默认 0）
        combat_reward: 本 step 期间发生过的真实战斗结果分 + 牌组实力增长 + 过 boss 进度
            （env 在 _log_combat_exit 算好缓存，下一次 step 时取出来一次性塞进去）。
            0 表示本 step 没战斗。

    返回：
        float reward（不是 NaN / inf）
    """
    return W_NODE_REWARD * float(node_reward) + float(combat_reward)


def compute_final_reward(game_won: bool, final_floor: int) -> float:
    """Episode 结束时 final reward。

    进度奖励顶层：通关大 bonus + final_floor 兜底（避免 0 winning data 时 model 完全没信号）。
    过 act boss 的 ++ 在战斗结束时单独给（compute_boss_beat_reward），不在这里。
    """
    won_bonus = GAME_WON_BONUS if game_won else 0.0
    return won_bonus + FLOOR_REACHED_BONUS * float(int(final_floor or 0))


__all__ = [
    "compute_step_reward",
    "compute_final_reward",
    "compute_combat_reward",
    "compute_strength_growth_reward",
    "compute_boss_beat_reward",
    # weights (导出方便 trainer / unit test 引用)
    "W_NODE_REWARD",
    "GAME_WON_BONUS",
    "FLOOR_REACHED_BONUS",
    "BOSS_BEAT_BONUS",
    "W_COMBAT_WIN",
    "W_COMBAT_LOSE",
    "W_DAMAGE_RATIO",
    "W_STRENGTH_GROWTH",
    "NODE_REWARD_EVENT_SUCCESS",
    "NODE_REWARD_SHOP_RELIC",
    "NODE_REWARD_REST_USE",
    "NODE_REWARD_TREASURE",
]

"""V8 阶段 2 底座：按局滚动的 per-card 评分 + 牌组实力（总分）。

设计来源：docs/v8_rl_fix_plan_2026-05-29.md「核心设计」。

一句话：删掉 hp/回合战斗 shaping，换成「牌组实力增长 + 真实进度」作奖励；
牌组实力由**战后从真实战斗里量的每张卡分**聚合而来。本模块负责前半截——
把每场真实战斗的 per-card 细账累加成「本局每张卡的分」，再聚合成牌组总分。

本轮（阶段 2 底座）只实现两个**锚维**（直接从战斗细账量、最实）：
- **输出 output** = 这张卡这局实打出的真实伤害累计（effects type=damage 的 amount 之和）。
- **防御 defense** = 这张卡这局加的格挡累计（effects type=block 的 amount 之和）。

花哨三维（运转 draw / 加费 energy / 能力 power）本轮**只留接口占位、权重为 0**，
不实现自举逻辑（那是下一轮，要单独两轴校验）。见 `_bootstrap_dims_placeholder`。

关键性质（防作弊，对齐计划「防钻空子红线」）：
- **按局算、不跨局攒**：一张卡价值高度依赖当前 relic / 血量 / 牌组组合，
  每局从零长 → 模型学到的是「这张卡跟我这套配着好」，不是绝对好坏。
  → 通过 `reset()` 在 env.reset 时清空实现。
- **牌组实力 = 全部卡综合分之和（总分，不是平均分）**：拿到任何净正分的卡都让
  实力上升，**不存在「为保平均分而 skip」的动机**（第一层防 skip-all 保险）。
- 分只来自真实伤害 / 格挡：不打仗不更新（`update_from_combat` 只在战斗结束被调），
  打了仗才更新，且增量来自细账里的真实 amount，凭空刷不出来。

量纲：伤害 / 格挡各除以一个温和常数（`OUTPUT_NORM` / `DEFENSE_NORM`）做归一，
避免「一场打几百伤害」把综合分炸到三位数。归一后一张主力攻击卡单场综合分约
个位数量级，和 reward.py 里 W_STRENGTH_GROWTH 配合后增量奖励落在合理区间。
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional


# ============================================================
# 维度权重（综合分 = Σ 维度归一值 × 权重）
# ============================================================

# 锚维（直接量、最实）——给合理正权重，初始两维同权。
W_OUTPUT: float = 1.0       # 输出维权重
W_DEFENSE: float = 1.0      # 防御维权重

# 自举三维占位权重（阶段 2 后续实现，本轮恒 0，不参与综合分）。
W_DRAW: float = 0.0         # 运转维（抽到的牌的分之和）—— 待实现
W_ENERGY: float = 0.0       # 加费维（能量 × 促成多打的高分卡价值）—— 待实现
W_POWER: float = 0.0        # 能力维（buff 层数 × 下游攻击额外收益估算）—— 待实现

# 归一常数（温和，量纲稳定用）：单场一张主力攻击卡伤害几十 → 除 50 落到 ~1。
OUTPUT_NORM: float = 50.0   # 伤害归一除数（与 v8/model.py deck_strength 编码 /50 对齐）
DEFENSE_NORM: float = 50.0  # 格挡归一除数


def canonical_card_key(name: str, upgraded: bool) -> str:
    """把 (name, upgraded) 归一成单一 card key。

    与 StSRLSolver 战斗日志的 card id 对齐：引擎里升级卡的 hand id 是 `name + "+"`
    （见 game.py `card.id + "+" if card.upgraded`），细账里的 effects 也挂在这个 id 上。
    牌组侧 `_build_state_from_runner` 给的是 {"name": base_id, "upgraded": bool}，
    这里统一拼成同一个 key，保证「升级版算不同 id」且两边能对上。
    """
    name = str(name or "")
    return name + "+" if upgraded else name


def card_key_from_log(log_card_id: str) -> str:
    """战斗细账里的 card id 已经是 canonical 形式（升级卡带 "+"），原样返回（兜空）。"""
    return str(log_card_id or "")


def card_key_from_deck_entry(entry: Dict[str, Any]) -> str:
    """牌组条目 {"name","upgraded"} → canonical key。"""
    return canonical_card_key(
        str(entry.get("name", "") or ""),
        bool(entry.get("upgraded", False)),
    )


class CardScorer:
    """按局滚动的 per-card 评分器。

    生命周期：env 每局 reset() 时 new 一个（或调 self.reset()）；每场战斗结束
    env 调 update_from_combat(card_log) 累加锚维；任意时刻可 deck_strength(deck)
    取当前牌组总分。
    """

    def __init__(self) -> None:
        # card_key -> 本局累计原始量（未归一）
        self._output: Dict[str, float] = {}   # 累计真实伤害
        self._defense: Dict[str, float] = {}  # 累计格挡
        # 自举三维原始量占位（本轮恒 0，保留 dict 方便后续轮次接管）。
        self._draw: Dict[str, float] = {}
        self._energy: Dict[str, float] = {}
        self._power: Dict[str, float] = {}

    # ---------------------------------------------------------------------
    # 生命周期
    # ---------------------------------------------------------------------

    def reset(self) -> None:
        """每局重置：清空所有 per-card 累计（不跨局攒分）。"""
        self._output.clear()
        self._defense.clear()
        self._draw.clear()
        self._energy.clear()
        self._power.clear()

    # ---------------------------------------------------------------------
    # 战后累加（锚维）
    # ---------------------------------------------------------------------

    def update_from_combat(self, card_log: Iterable[Dict[str, Any]]) -> None:
        """从一场战斗的 per-card 细账累加锚维（输出 / 防御）。

        card_log: env._last_combat_card_log 形态——List[dict]，每条
            {"turn", "event_type", "data"}，play_card 条目的 data 含
            {"card": card_id, "target", "effects": [{"type": "damage"/"block"/..., "amount": N}, ...]}。

        只读 event_type=="play_card" 条目；伤害 / 格挡按 card key 累加。
        其余 event_type（turn_start / enemy_move / combat_start ...）跳过。

        不打仗不调用 → 不更新；打了仗才更新，增量全来自细账真实 amount。
        """
        if not card_log:
            return
        for entry in card_log:
            if not isinstance(entry, dict):
                continue
            if entry.get("event_type") != "play_card":
                continue
            data = entry.get("data") or {}
            card_id = card_key_from_log(data.get("card", ""))
            if not card_id:
                continue
            for eff in data.get("effects", []) or []:
                if not isinstance(eff, dict):
                    continue
                t = eff.get("type")
                if t == "damage":
                    self._output[card_id] = self._output.get(card_id, 0.0) + float(
                        eff.get("amount", 0) or 0
                    )
                elif t == "block":
                    self._defense[card_id] = self._defense.get(card_id, 0.0) + float(
                        eff.get("amount", 0) or 0
                    )
                # 自举三维（draw / energy / power）本轮**不累加**——占位待实现。
                # 阶段2后续：自举维，待实现。届时在这里按
                #   draw  : 抽到的牌（eff["cards"]）的当前综合分之和累加到 _draw
                #   energy: eff["amount"] × 后续促成打出的高分卡价值
                #   power : eff["amount"] 层数 × 受影响攻击的估算额外收益
                # 累加，并把对应 W_* 权重从 0 调成保守正值（每维单独两轴校验后并入）。

    # ---------------------------------------------------------------------
    # 自举三维占位（阶段2后续：自举维，待实现）
    # ---------------------------------------------------------------------

    def _bootstrap_dims_placeholder(self, card_id: str) -> Dict[str, float]:
        """运转 / 加费 / 能力 三维的归一值。

        **阶段2后续：自举维，待实现。** 本轮恒返回 0（且 W_DRAW/W_ENERGY/W_POWER 也为 0），
        故不影响综合分。保留此接口 + 三个 _draw/_energy/_power dict，让下一轮接管时
        只需在 update_from_combat 里累加 + 在这里做归一 + 把权重从 0 调正，不动外层调用。
        """
        return {"draw": 0.0, "energy": 0.0, "power": 0.0}

    # ---------------------------------------------------------------------
    # 综合分 / 牌组实力
    # ---------------------------------------------------------------------

    def card_dims(self, card_id: str) -> Dict[str, float]:
        """单张卡的各维归一值（debug / 两轴校验用）。"""
        output_n = self._output.get(card_id, 0.0) / OUTPUT_NORM
        defense_n = self._defense.get(card_id, 0.0) / DEFENSE_NORM
        boot = self._bootstrap_dims_placeholder(card_id)
        return {
            "output": output_n,
            "defense": defense_n,
            "draw": boot["draw"],
            "energy": boot["energy"],
            "power": boot["power"],
        }

    def card_score(self, card_id: str) -> float:
        """单张卡综合分 = 各维归一值的加权和。

        本轮 = W_OUTPUT*output_n + W_DEFENSE*defense_n（三维占位权重 0，不参与）。
        没出过力（不在任何细账里）的卡 → 各维 0 → 综合分 0（不正不负）。
        """
        d = self.card_dims(card_id)
        return (
            W_OUTPUT * d["output"]
            + W_DEFENSE * d["defense"]
            + W_DRAW * d["draw"]
            + W_ENERGY * d["energy"]
            + W_POWER * d["power"]
        )

    def deck_strength(self, deck: Optional[List[Dict[str, Any]]]) -> float:
        """当前牌组实力 = 牌组里所有卡综合分之和（**总分，不是平均**）。

        deck: V8State.deck 形态——List[{"name","upgraded"}]（_build_state_from_runner 产出）。
        同名重复卡各算一次（牌组里有 2 张 Strike 就加 2 次 Strike 的综合分）。

        关键（防 skip-all）：用总分 → 加任意净正分卡都涨；不会因「拉低均分」被罚。
        没出过力的卡综合分 0 → 不拉总分（既不奖也不罚单纯持有烂牌）。
        """
        if not deck:
            return 0.0
        total = 0.0
        for entry in deck:
            if not isinstance(entry, dict):
                continue
            key = card_key_from_deck_entry(entry)
            total += self.card_score(key)
        return float(total)

    # ---------------------------------------------------------------------
    # debug / 两轴校验
    # ---------------------------------------------------------------------

    def scored_cards(self) -> Dict[str, float]:
        """本局所有「出过力」的卡 → 综合分（debug / 校验用）。"""
        keys = set(self._output) | set(self._defense) \
            | set(self._draw) | set(self._energy) | set(self._power)
        return {k: self.card_score(k) for k in sorted(keys)}


__all__ = [
    "CardScorer",
    "canonical_card_key",
    "card_key_from_log",
    "card_key_from_deck_entry",
    "W_OUTPUT",
    "W_DEFENSE",
    "W_DRAW",
    "W_ENERGY",
    "W_POWER",
    "OUTPUT_NORM",
    "DEFENSE_NORM",
]

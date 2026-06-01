"""V8 阶段 2：按局滚动的 per-card 评分 + 牌组实力（总分）。

设计来源：docs/v8_rl_fix_plan_2026-05-29.md「核心设计」。

一句话：删掉 hp/回合战斗 shaping，换成「牌组实力增长 + 真实进度」作奖励；
牌组实力由**战后从真实战斗里量的每张卡分**聚合而来。本模块负责前半截——
把每场真实战斗的 per-card 细账累加成「本局每张卡的分」，再聚合成牌组总分。

维度分两类：

**锚维（直接从战斗细账量、最实，是一切的基准）**
- **输出 output** = 这张卡这局实打出的真实伤害累计（effects type=damage 的 amount 之和）。
- **防御 defense** = 这张卡这局加的格挡累计（effects type=block 的 amount 之和）。

**协同三维（阶段 2 本轮新增，引用锚维、绝不引用综合分 → 无循环/递归）**
- **运转 draw** = 这张卡抽到的每张牌的**当前锚分(output+defense 归一)**之和。
  抽到主力攻击/格挡卡 → 高；抽到垃圾/0 锚分牌 → ≈0。凭空抽不出分。
- **加费 energy** = 这张卡给的能量 × 同一回合后续打出牌的锚分和（proxy，见下）。
  给能量但后面没打高分牌 → ≈0。
- **能力 power** = 这张卡上的 buff 层数 × 该 combat 后续打出的攻击牌数 × 小系数（估算维）。
  上 buff 但后面没攻击牌吃 → ≈0。这是唯一的「估算维」（下游真实伤害引擎不拆），
  权重最保守。

**[关键设计] 协同维只引用锚维分，不引用综合分。**
锚维是直接测量的硬值；协同维顺着锚维长。计算时序保证无序依赖：
`update_from_combat` 先把整场战斗的锚维(damage/block)全部累加完，**再**遍历同一场
细账算协同维——此时被引用的锚分已是「这局到目前为止的完整测量值」，不依赖卡的出牌
顺序，也永远不碰 card_score()/deck_strength() 这类综合量。→ 没有递归、没有循环。

关键性质（防作弊，对齐计划「防钻空子红线」）：
- **按局算、不跨局攒**：一张卡价值高度依赖当前 relic / 血量 / 牌组组合，
  每局从零长 → 模型学到的是「这张卡跟我这套配着好」，不是绝对好坏。
  → 通过 `reset()` 在 env.reset 时清空实现。
- **牌组实力 = 全部卡综合分之和（总分，不是平均分）**：拿到任何净正分的卡都让
  实力上升，**不存在「为保平均分而 skip」的动机**（第一层防 skip-all 保险）。
- 分只来自真实伤害 / 格挡（锚维）+ 引用锚分的协同（协同维）：不打仗不更新
  （`update_from_combat` 只在战斗结束被调），且协同维都 ≤ 其引用的锚分量级
  （归一系数 < 1），凭空刷不出来、也放大不到盖过真实战斗产出。

量纲：伤害 / 格挡各除以一个温和常数（`OUTPUT_NORM` / `DEFENSE_NORM`）做归一，
避免「一场打几百伤害」把综合分炸到三位数。归一后一张主力攻击卡单场综合分约
个位数量级，和 reward.py 里 W_STRENGTH_GROWTH 配合后增量奖励落在合理区间。
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional


# ============================================================
# 维度权重（综合分 = Σ 维度归一值 × 权重）
# ============================================================
#
# 所有权重初始值待训练时按两轴校验后调（注释统一标注）。

# 锚维（直接量、最实）——给合理正权重，初始两维同权。
W_OUTPUT: float = 1.0       # 输出维权重（初始值待训练时调）
W_DEFENSE: float = 1.0      # 防御维权重（初始值待训练时调）

# 协同三维（阶段 2 本轮实现）——**保守正值，明显低于锚维**。
# 理由：协同维含估算成分（尤其 power）、相对锚维更易被钻，权重压低让锚维主导综合分，
# 协同维只做「锦上添花」让奖励信号更丰富，不喧宾夺主。
# 量级关系：W_DRAW ≈ W_ENERGY > W_POWER；都远小于 W_OUTPUT/W_DEFENSE。
W_DRAW: float = 0.30        # 运转维权重（引用锚分、量纲已对齐；初始值待训练时调）
W_ENERGY: float = 0.30      # 加费维权重（引用锚分、量纲已对齐；初始值待训练时调）
W_POWER: float = 0.15       # 能力维权重（**估算维，最保守**；初始值待训练时调）

# 归一常数（温和，量纲稳定用）：单场一张主力攻击卡伤害几十 → 除 50 落到 ~1。
OUTPUT_NORM: float = 50.0   # 伤害归一除数（与 v8/model.py deck_strength 编码 /50 对齐）
DEFENSE_NORM: float = 50.0  # 格挡归一除数

# ---- 协同维内部系数（把协同原始量压到 ≤ 引用锚分量级，防放大）----
# 加费维 proxy 衰减：energy 维 = energy_amount × (同回合后续牌锚分和) × ENERGY_SYNERGY_COEF。
# 1 点能量名义上「换」回大致 1 张牌的产出，但能量是「促成」而非「直接产出」，
# 打半价：系数 0.5，确保加费维永远 ≤ 它促成的那些牌锚分和的一半，不盖过真实产出。
ENERGY_SYNERGY_COEF: float = 0.5
# 能力维（估算维）系数：层数 × 后续攻击牌数 × 这个小系数。
# 取很小（0.02）——1 层 buff 摊到 1 张后续攻击牌只贡献 0.02 归一分，
# 即「10 层 Strength + 后面 5 张攻击」≈ 10×5×0.02 = 1.0 归一分量级（与一张主力攻击卡相当），
# 既给出方向性信号又不让纯估算维炸开。
POWER_SYNERGY_COEF: float = 0.02


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
    env 调 update_from_combat(card_log) 累加锚维 + 协同维；任意时刻可
    deck_strength(deck) 取当前牌组总分。
    """

    def __init__(self) -> None:
        # card_key -> 本局累计原始量（未归一）
        self._output: Dict[str, float] = {}   # 累计真实伤害
        self._defense: Dict[str, float] = {}  # 累计格挡
        # 协同三维累计原始量（已含本模块归一前的「引用锚分之和」等量纲）。
        self._draw: Dict[str, float] = {}     # 抽到的牌的锚分之和（已是归一锚分量纲）
        self._energy: Dict[str, float] = {}   # 加费 proxy 累计（已是归一锚分量纲）
        self._power: Dict[str, float] = {}    # 能力估算累计（已是归一量纲）

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
    # 内部：锚分查询（协同维只引用它，绝不引用综合分）
    # ---------------------------------------------------------------------

    def _anchor_score(self, card_id: str) -> float:
        """单张卡的**锚维归一分**（output + defense），协同维唯一可引用的量。

        ★ 这是「无循环」的关键：协同维计算时只调这里，永不调 card_score()/card_dims()
        （那些含协同维 → 会递归）。锚分是直接测量值，不依赖任何协同量。
        """
        output_n = self._output.get(card_id, 0.0) / OUTPUT_NORM
        defense_n = self._defense.get(card_id, 0.0) / DEFENSE_NORM
        return output_n + defense_n

    # ---------------------------------------------------------------------
    # 战后累加（锚维 + 协同维）
    # ---------------------------------------------------------------------

    def update_from_combat(self, card_log: Iterable[Dict[str, Any]]) -> None:
        """从一场战斗的 per-card 细账累加锚维 + 协同维。

        card_log: env._last_combat_card_log 形态——List[dict]，每条
            {"turn", "event_type", "data"}，play_card 条目的 data 含
            {"card": card_id, "target", "effects": [
                {"type": "damage", "target", "amount": N},
                {"type": "block", "amount": N},
                {"type": "draw", "amount": N, "cards": [被抽到的 card_id, ...]},
                {"type": "energy", "amount": N},
                {"type": "power", "power": power_id, "amount": 层数},
                ...]}。

        只读 event_type=="play_card" 条目。其余 event_type（turn_start /
        enemy_move / combat_start ...）跳过。

        **两遍扫描，保证「协同维只引用已测全的锚分」、无序依赖、无循环：**
          Pass 1：把整场所有 play_card 的 damage/block 累加进锚维（_output / _defense）。
                  此后 _anchor_score(任意卡) = 这局到目前为止的完整锚分。
          Pass 2：再遍历同一场细账算协同维（draw / energy / power），引用 Pass 1
                  算好的锚分。因为锚分已经测全，协同维结果与卡出牌顺序无关。

        不打仗不调用 → 不更新；增量全来自细账真实 amount（锚维）或锚分引用（协同维）。
        """
        if not card_log:
            return

        # 先把 play_card 条目过滤成一个 list（两遍都用），顺带保留 turn 用于加费同回合关联。
        plays: List[Dict[str, Any]] = []
        for entry in card_log:
            if not isinstance(entry, dict):
                continue
            if entry.get("event_type") != "play_card":
                continue
            data = entry.get("data") or {}
            card_id = card_key_from_log(data.get("card", ""))
            if not card_id:
                continue
            plays.append(
                {
                    "turn": int(entry.get("turn", 0) or 0),
                    "card_id": card_id,
                    "effects": data.get("effects", []) or [],
                }
            )

        if not plays:
            return

        # ---- Pass 1：锚维（damage → output, block → defense）----
        for p in plays:
            card_id = p["card_id"]
            for eff in p["effects"]:
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

        # ---- Pass 2：协同维（draw / energy / power），引用 Pass 1 测全的锚分 ----
        # 预备：能力维需要「某 play 之后这场还打了几张攻击牌」。先标出每个 play 是否攻击。
        # （攻击牌 = 该 play 的 effects 里有 damage。）做个前缀/后缀计数。
        is_attack: List[bool] = [
            any(isinstance(e, dict) and e.get("type") == "damage" for e in p["effects"])
            for p in plays
        ]
        n_plays = len(plays)
        # suffix_attacks[i] = plays[i+1:] 里攻击牌张数（"这张牌之后还打了几张攻击"）
        suffix_attacks: List[int] = [0] * n_plays
        running = 0
        for i in range(n_plays - 1, -1, -1):
            suffix_attacks[i] = running
            if is_attack[i]:
                running += 1

        for idx, p in enumerate(plays):
            card_id = p["card_id"]
            turn = p["turn"]
            for eff in p["effects"]:
                if not isinstance(eff, dict):
                    continue
                t = eff.get("type")

                if t == "draw":
                    # 运转维：累加每张被抽到的牌的**当前锚分**。
                    # 抽到高 output/defense 牌得分高，抽到 0 锚分牌 ≈ 0（凭空抽不出分）。
                    drawn = eff.get("cards", []) or []
                    gain = 0.0
                    for drawn_id in drawn:
                        dk = card_key_from_log(drawn_id)
                        if dk:
                            gain += self._anchor_score(dk)
                    if gain:
                        self._draw[card_id] = self._draw.get(card_id, 0.0) + gain

                elif t == "energy":
                    # 加费维 proxy：energy_amount × 「**同一回合**这张牌之后打出的牌的锚分和」
                    #   × ENERGY_SYNERGY_COEF。
                    # proxy 选择理由（可解释、不易钻）：
                    #   - 「同回合 + 它之后」=最贴近「这点能量实际拿去打了哪些牌」的因果窗口。
                    #     给能量却没接牌（回合最后一张/下回合才用）→ 后续锚分和=0 → 不得分，
                    #     不能靠空打能量牌刷分。
                    #   - 引用的是后续牌的**锚分**（真实伤害/格挡量），不引用综合分 → 无循环。
                    #   - ×0.5 系数：能量是「促成」而非「直接产出」，打半价确保此维永远
                    #     ≤ 它促成牌锚分和的一半，盖不过真实战斗产出。
                    energy_amt = float(eff.get("amount", 0) or 0)
                    if energy_amt > 0:
                        downstream = 0.0
                        for j in range(idx + 1, n_plays):
                            if plays[j]["turn"] != turn:
                                break  # 出了本回合，停（同回合关联）
                            downstream += self._anchor_score(plays[j]["card_id"])
                        gain = energy_amt * downstream * ENERGY_SYNERGY_COEF
                        if gain:
                            self._energy[card_id] = self._energy.get(card_id, 0.0) + gain

                elif t == "power":
                    # 能力维（**估算维**）：buff 层数 × 该 combat **后续打出的攻击牌张数**
                    #   × POWER_SYNERGY_COEF（小系数）。
                    # 明确标注：下游真实伤害引擎不拆（一层 Strength 到底加了多少伤无法精确归因），
                    #   只能估。用「后续还有多少张攻击牌吃这个 buff」做方向性 proxy：
                    #   - 上 buff 后面攻击多 → power 维高；
                    #   - 上 buff 后面没攻击牌（如战斗末尾打 buff / 纯 buff 牌组）→ ≈0。
                    # 系数取极小（0.02），且这是权重最保守（W_POWER 最低）的维。
                    power_amt = float(eff.get("amount", 0) or 0)
                    if power_amt > 0 and suffix_attacks[idx] > 0:
                        gain = power_amt * float(suffix_attacks[idx]) * POWER_SYNERGY_COEF
                        if gain:
                            self._power[card_id] = self._power.get(card_id, 0.0) + gain

    # ---------------------------------------------------------------------
    # 综合分 / 牌组实力
    # ---------------------------------------------------------------------

    def card_dims(self, card_id: str) -> Dict[str, float]:
        """单张卡的各维归一值（debug / 两轴校验用）。

        协同三维的累计量（_draw/_energy/_power）已是「归一锚分量纲」（计算时引用的
        就是归一锚分），故这里**不再额外除归一常数**，直接取累计值即可。
        """
        return {
            "output": self._output.get(card_id, 0.0) / OUTPUT_NORM,
            "defense": self._defense.get(card_id, 0.0) / DEFENSE_NORM,
            "draw": self._draw.get(card_id, 0.0),
            "energy": self._energy.get(card_id, 0.0),
            "power": self._power.get(card_id, 0.0),
        }

    def card_score(self, card_id: str) -> float:
        """单张卡综合分 = 锚维 + 协同维 的加权和。

        = W_OUTPUT·output + W_DEFENSE·defense
          + W_DRAW·draw + W_ENERGY·energy + W_POWER·power。
        没出过力（不在任何细账里）的卡 → 各维 0 → 综合分 0（不正不负）。

        注：协同维只在 update_from_combat 里引用**锚维**算出来，这里把算好的值加权
        相加，不再触发任何引用——故 card_score 调 card_dims、card_dims 只读累计 dict，
        无递归、无循环。
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

    def deck_dims(self, deck: Optional[List[Dict[str, Any]]]) -> Dict[str, float]:
        """当前牌组的**5 维分各自的牌组聚合**（总和，与 deck_strength 同口径）。

        deck_strength() 给的是综合标量（5 维加权和的总分）；这里把每一维分开聚合，
        供 v8/model.py 当**输入特征**喂给模型——让模型在选卡/路线时「看到」自己牌组
        当前各维度多强（输出/防御/运转/加费/能力），而不只是一个标量。

        返回归一量纲值（与 card_dims 同口径：output/defense 已除归一常数，协同三维
        本就是归一量纲），各维在牌组上求和。同名重复卡各算一次（与 deck_strength 一致）。
        没出过力的卡各维 0 → 不拉聚合。空牌组 → 全 0。
        """
        agg = {"output": 0.0, "defense": 0.0, "draw": 0.0, "energy": 0.0, "power": 0.0}
        if not deck:
            return agg
        for entry in deck:
            if not isinstance(entry, dict):
                continue
            key = card_key_from_deck_entry(entry)
            d = self.card_dims(key)
            for k in agg:
                agg[k] += float(d.get(k, 0.0) or 0.0)
        return agg

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
    "ENERGY_SYNERGY_COEF",
    "POWER_SYNERGY_COEF",
]

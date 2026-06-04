"""LightspeedRelicProbe：用 sts_lightspeed（社区金标准 C++ 模拟器）验收遗物行为。

给定一条 RelicExpectation，用扩展后的 make_test_combat（带该遗物）构造受控战斗，
读「开战瞬间」快照；若 expectation 带 play_card 则再打一张牌读伤害。返回中性结果，
不在这里做断言（断言交给 test_lightspeed_relic_behavior.py）。

接入要点（2026-06-04 绑定扩展）：
  - make_test_combat 新增 relics 参数（list[RelicId]）：在 bc.init() 之前先把这些遗物
    obtainRelic 到本地 GameContext，这样 BattleContext::initRelics + executeActions 会
    真实触发 atBattleStart / atBattleStartPreDraw 效果（这正是要验收的核心）。
  - player_hp 在 init 前写到本地 gc：让 Blood Vial 这类开战回血叠在设定血量上，而非被
    init 后覆写清掉。
  - strength/block 用 INT_MIN 当「不覆写」哨兵（这里不传 → 不覆写 → Vajra/Anchor 等
    遗物施加的力量/护甲不会被清掉）。
  - get_combat_snapshot 新增导出字段：
    player: thorns / metallicize / regen / ritual / vigor
    enemy : metallicize / ritual

返回 RelicProbeResult：
  - player_snapshot : dict（开战瞬间玩家状态，含新字段）
  - enemies_snapshot: list[dict]（每只怪开战瞬间状态）
  - played          : bool（是否打了 play_card）
  - target_hp_delta : Optional[int]（play_card 后目标怪掉血）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from v8.backends.lightspeed_loader import load_lightspeed


@dataclass
class RelicProbeResult:
    player_snapshot: Dict[str, int]
    enemies_snapshot: List[Dict] = field(default_factory=list)
    played: bool = False
    target_hp_delta: Optional[int] = None
    play_success: Optional[bool] = None
    error: Optional[str] = None


class RelicNotMappedError(KeyError):
    """oracle relic 名在 lightspeed RelicId 枚举无对应。"""


class LightspeedRelicProbe:
    """实现遗物探针（鸭子类型）。"""

    def __init__(self):
        self._sts = load_lightspeed()

    def probe(self, exp) -> RelicProbeResult:
        """跑一条 RelicExpectation，返回中性结果。exp 是 tools.relic_oracle.RelicExpectation。"""
        sts = self._sts

        relic_enum = self._relic_enum(exp.relic)
        encounter = self._encounter_enum(exp.encounter)
        enemy_hps = list(exp.enemy_hps) if exp.enemy_hps else []

        gc = sts.GameContext(sts.CharacterClass.IRONCLAD, 1, 0)

        # 手牌：若要打牌验伤害，把 play_card 放在 hand[0]；否则不需要手牌。
        hand = []
        if exp.play_card:
            hand = [self._card_enum(exp.play_card)]

        bc = sts.make_test_combat(
            gc,
            encounter,
            player_hp=exp.player_hp,
            energy=exp.energy,
            hand=hand,
            enemy_hps=enemy_hps,
            relics=[relic_enum],
        )

        snap_before = sts.get_combat_snapshot(bc)
        player_snap = dict(snap_before["player"])
        enemies_snap = [dict(e) for e in snap_before["enemies"]]

        result = RelicProbeResult(
            player_snapshot=player_snap,
            enemies_snapshot=enemies_snap,
        )

        if exp.play_card:
            tgt = exp.play_target if exp.play_target >= 0 else 0
            hp_before = [e["hp"] for e in snap_before["enemies"]]
            ok = True
            try:
                sts.play_card(bc, 0, tgt)
            except Exception as e:  # noqa: BLE001  非法出牌 / binding 抛错
                ok = False
                result.error = repr(e)
            snap_after = sts.get_combat_snapshot(bc)
            enemies_after = snap_after["enemies"]
            result.played = True
            result.play_success = ok
            if tgt < len(enemies_after) and tgt < len(hp_before):
                result.target_hp_delta = hp_before[tgt] - enemies_after[tgt]["hp"]

        return result

    # ---- 枚举映射（靠 dir(sts.*) 实查，名一致直接 getattr）----
    def _relic_enum(self, name: str):
        enum = getattr(self._sts.RelicId, name, None)
        if enum is None:
            raise RelicNotMappedError(f"RelicId.{name} 不存在")
        return enum

    def _encounter_enum(self, name: str):
        enum = getattr(self._sts.MonsterEncounter, name, None)
        if enum is None:
            raise RelicNotMappedError(f"MonsterEncounter.{name} 不存在")
        return enum

    def _card_enum(self, name: str):
        enum = getattr(self._sts.CardId, name, None)
        if enum is None:
            raise RelicNotMappedError(f"CardId.{name} 不存在")
        return enum

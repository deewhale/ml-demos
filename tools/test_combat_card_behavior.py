"""第 1 层：卡牌行为对照测试。

读 card_oracle.ORACLE，用一个 CombatProbe 在受控战斗里打每张卡，断言结果对得上
独立 oracle。后端无关：换引擎只换 probe 工厂。

跑法：.venv/bin/python tools/test_combat_card_behavior.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = str(Path(__file__).resolve().parent.parent)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from typing import List

from tools.card_oracle import ORACLE, CardExpectation
from v8.backends.combat_probe import CardPlayResult, CombatProbe
from v8.backends.stsrl_combat_probe import StSRLCombatProbe


def check_expectation(probe: CombatProbe, exp: CardExpectation) -> List[str]:
    """跑一条期望，返回失败描述 list（空 = 通过）。"""
    hand_index = list(exp.setup.hand).index(exp.card_id)
    res: CardPlayResult = probe.play_single_card(
        exp.setup, hand_index=hand_index, target_index=exp.target_index
    )
    fails: List[str] = []
    if not res.success:
        fails.append(f"play_card 未成功（effects={res.effects}）")
    if exp.expect_enemy_hp_delta is not None and res.enemy_hp_delta != exp.expect_enemy_hp_delta:
        fails.append(f"敌人掉血 {res.enemy_hp_delta} != 期望 {exp.expect_enemy_hp_delta}")
    if exp.expect_player_block_delta is not None and res.player_block_delta != exp.expect_player_block_delta:
        fails.append(f"玩家格挡 +{res.player_block_delta} != 期望 +{exp.expect_player_block_delta}")
    if exp.expect_energy_delta is not None and res.energy_delta != exp.expect_energy_delta:
        fails.append(f"耗能 {res.energy_delta} != 期望 {exp.expect_energy_delta}")
    if exp.expect_enemy_status is not None:
        for name, layers in exp.expect_enemy_status.items():
            got = res.enemy_statuses.get(name, 0)
            if got != layers:
                fails.append(f"敌人状态 {name}={got} != 期望 {layers}")
    if exp.expect_player_status is not None:
        for name, layers in exp.expect_player_status.items():
            got = res.player_statuses.get(name, 0)
            if got != layers:
                fails.append(f"玩家状态 {name}={got} != 期望 {layers}")
    return fails


def run_all(probe: CombatProbe) -> int:
    """跑全表，返回失败卡数。"""
    total = len(ORACLE)
    failed = 0
    for exp in ORACLE:
        fails = check_expectation(probe, exp)
        if fails:
            failed += 1
            print(f"[FAIL] {exp.label or exp.card_id}")
            for f in fails:
                print(f"         - {f}")
            print(f"         (oracle 来源: {exp.source})")
        else:
            print(f"[ OK ] {exp.label or exp.card_id}")
    print(f"\n卡牌行为测试：{total - failed}/{total} 通过，{failed} 失败")
    return failed


def test_starting_cards_behavior():
    """起始三卡行为对照（StSRLSolver 后端）应全绿。"""
    probe = StSRLCombatProbe()
    failed = run_all(probe)
    assert failed == 0, f"{failed} 张卡行为与 oracle 不符"


if __name__ == "__main__":
    from v8.backends.stsrl_combat_probe import StSRLCombatProbe
    probe = StSRLCombatProbe()
    failed = run_all(probe)
    print(f"\n[审计模式] 完成；{failed} 张未对上 wiki（红灯=疑似引擎坏卡，需人工核）")

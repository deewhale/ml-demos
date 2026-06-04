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
    if exp.expect_unplayable:
        # 期望该卡当前不可打出：success 应为 False，且不掉血
        if res.success:
            fails.append("期望该卡不可打出，但引擎允许打出了")
    elif not res.success:
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
    if exp.expect_all_enemies_hp_delta is not None:
        for i, d in enumerate(res.all_enemy_hp_deltas):
            if d != exp.expect_all_enemies_hp_delta:
                fails.append(f"敌人[{i}]掉血 {d} != 期望全体 {exp.expect_all_enemies_hp_delta}")
    if exp.expect_all_enemies_status is not None:
        for i, st in enumerate(res.all_enemy_statuses):
            for name, layers in exp.expect_all_enemies_status.items():
                if st.get(name, 0) != layers:
                    fails.append(f"敌人[{i}]状态 {name}={st.get(name,0)} != 期望全体 {layers}")
    # ---- 牌堆维度 ----
    if exp.expect_draw_pile_size is not None and res.draw_pile_size != exp.expect_draw_pile_size:
        fails.append(f"抽牌堆张数 {res.draw_pile_size} != 期望 {exp.expect_draw_pile_size}")
    if exp.expect_discard_pile_size is not None and res.discard_pile_size != exp.expect_discard_pile_size:
        fails.append(f"弃牌堆张数 {res.discard_pile_size} != 期望 {exp.expect_discard_pile_size}")
    if exp.expect_exhaust_pile_size is not None and res.exhaust_pile_size != exp.expect_exhaust_pile_size:
        fails.append(f"消耗堆张数 {res.exhaust_pile_size} != 期望 {exp.expect_exhaust_pile_size}")
    if exp.expect_hand_size is not None and res.hand_size != exp.expect_hand_size:
        fails.append(f"手牌张数 {res.hand_size} != 期望 {exp.expect_hand_size}")

    def _check_contains(pile, spec, label):
        for name, n in spec.items():
            got = sum(1 for c in pile if c == name)
            if got < n:
                fails.append(f"{label} 含 {name}={got} < 期望 ≥{n}（pile={list(pile)}）")

    if exp.expect_draw_pile_contains is not None:
        _check_contains(res.draw_pile, exp.expect_draw_pile_contains, "抽牌堆")
    if exp.expect_discard_pile_contains is not None:
        _check_contains(res.discard_pile, exp.expect_discard_pile_contains, "弃牌堆")
    if exp.expect_exhaust_pile_contains is not None:
        _check_contains(res.exhaust_pile, exp.expect_exhaust_pile_contains, "消耗堆")
    if exp.expect_hand_contains is not None:
        _check_contains(res.hand, exp.expect_hand_contains, "手牌")
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

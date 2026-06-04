"""CombatProbe 接口与 StSRLCombatProbe 实现的测试。

跑法：.venv/bin/python tools/test_combat_probe.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = str(Path(__file__).resolve().parent.parent)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from v8.backends.combat_probe import CardPlayResult, CombatProbe, CombatSetup
from v8.backends.stsrl_combat_probe import StSRLCombatProbe


def test_combat_setup_and_result_are_neutral():
    """CombatSetup / CardPlayResult 可构造，且全是中性 Python 类型。"""
    setup = CombatSetup(hand=("Strike_R",))
    assert setup.hand == ("Strike_R",)
    assert setup.enemy_id == "JawWorm"
    assert setup.player_hp == 80
    assert setup.player_energy == 3

    res = CardPlayResult(
        success=True,
        enemy_hp_delta=6,
        player_block_delta=0,
        energy_delta=1,
        enemy_statuses={},
        player_statuses={},
        effects=[{"type": "damage", "target": "JawWorm", "amount": 6}],
        all_enemy_hp_deltas=[6],
        all_enemy_statuses=[{}],
    )
    assert res.success is True
    assert res.enemy_hp_delta == 6
    assert isinstance(res.effects, list)


def test_stsrl_probe_strike_deals_6():
    probe = StSRLCombatProbe()
    res = probe.play_single_card(
        CombatSetup(hand=("Strike_R",)), hand_index=0, target_index=0
    )
    assert res.success is True
    assert res.enemy_hp_delta == 6
    assert res.energy_delta == 1


def test_stsrl_probe_defend_blocks_5():
    probe = StSRLCombatProbe()
    res = probe.play_single_card(
        CombatSetup(hand=("Defend_R",)), hand_index=0, target_index=-1
    )
    assert res.success is True
    assert res.player_block_delta == 5
    assert res.energy_delta == 1


def test_stsrl_probe_bash_damage_and_vulnerable():
    probe = StSRLCombatProbe()
    res = probe.play_single_card(
        CombatSetup(hand=("Bash",)), hand_index=0, target_index=0
    )
    assert res.success is True
    assert res.enemy_hp_delta == 8
    assert res.energy_delta == 2
    assert res.enemy_statuses.get("Vulnerable", 0) == 2


def test_stsrl_probe_satisfies_protocol():
    assert isinstance(StSRLCombatProbe(), CombatProbe)


if __name__ == "__main__":
    test_combat_setup_and_result_are_neutral()
    test_stsrl_probe_strike_deals_6()
    test_stsrl_probe_defend_blocks_5()
    test_stsrl_probe_bash_damage_and_vulnerable()
    test_stsrl_probe_satisfies_protocol()
    print("ALL PASS")

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
    )
    assert res.success is True
    assert res.enemy_hp_delta == 6
    assert isinstance(res.effects, list)


if __name__ == "__main__":
    test_combat_setup_and_result_are_neutral()
    print("ALL PASS")

"""card_oracle 期望表的结构测试。

跑法：.venv/bin/python tools/test_card_oracle.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = str(Path(__file__).resolve().parent.parent)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from tools.card_oracle import ORACLE, CardExpectation


def test_oracle_entries_well_formed():
    assert len(ORACLE) >= 3
    card_ids = {e.card_id for e in ORACLE}
    assert {"Strike_R", "Defend_R", "Bash"} <= card_ids
    for e in ORACLE:
        assert isinstance(e, CardExpectation)
        assert e.card_id, "card_id 不能为空"
        assert len(e.setup.hand) >= 1, f"{e.card_id} 手牌不能为空"
        assert e.card_id in e.setup.hand, f"{e.card_id} 必须在手牌里"
        assert e.source, f"{e.card_id} 必须标 oracle 来源（独立标准答案出处）"


if __name__ == "__main__":
    test_oracle_entries_well_formed()
    print("ALL PASS")

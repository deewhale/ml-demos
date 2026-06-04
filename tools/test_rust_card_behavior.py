"""第 1 层 · Rust 引擎卡牌行为验收。

拿独立 wiki oracle 考 Rust 引擎（engine-rs）。仿 test_combat_card_behavior.py，
但用 RustCombatProbe。审计模式：不 hard-assert，红灯即疑似引擎坏卡。

预筛：本轮只跑 player_block==0 且 player_strength==0 的 entry（无预设子集），
且 card id 能映射到 Rust。其余 SKIP。

跑法：.venv/bin/python tools/test_rust_card_behavior.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = str(Path(__file__).resolve().parent.parent)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from typing import List

from tools.card_oracle import ORACLE, CardExpectation
from tools.test_combat_card_behavior import check_expectation
from v8.backends.rust_combat_probe import (
    RustCombatProbe,
    _PY_TO_RUST_ID,
)

# Python 引擎曾失败的 5 张，单独列出对照
PYTHON_BROKEN = {"Dark Shackles", "Panacea", "PanicButton", "J.A.X.", "Sword Boomerang"}


def _is_no_preset(exp: CardExpectation) -> bool:
    return exp.setup.player_block == 0 and exp.setup.player_strength == 0


def _can_map(exp: CardExpectation) -> bool:
    return exp.card_id in _PY_TO_RUST_ID


def run_audit(probe: RustCombatProbe) -> None:
    ran = passed = red = skipped = 0
    broken_results = {}  # card -> (status, detail)

    for exp in ORACLE:
        label = exp.label or exp.card_id
        if not _can_map(exp):
            skipped += 1
            print(f"[SKIP] {label} (Rust 无此 id)")
            if exp.card_id in PYTHON_BROKEN:
                broken_results[exp.card_id] = ("SKIP", "Rust 无此 id")
            continue
        if not _is_no_preset(exp):
            skipped += 1
            print(f"[SKIP] {label} (力量/格挡预设, 待后续)")
            continue

        ran += 1
        fails = check_expectation(probe, exp)
        if fails:
            red += 1
            print(f"[FAIL] {label}")
            for f in fails:
                print(f"         - {f}")
            print(f"         (oracle 来源: {exp.source})")
            if exp.card_id in PYTHON_BROKEN:
                broken_results[exp.card_id] = ("FAIL", "; ".join(fails))
        else:
            passed += 1
            print(f"[ OK ] {label}")
            if exp.card_id in PYTHON_BROKEN:
                broken_results[exp.card_id] = ("OK", "")

    print(
        f"\n[Rust 验收汇总] 跑了 {ran}，过 {passed}，"
        f"Rust 红灯 {red}，跳过 {skipped}"
    )

    print("\n[Python-broken 5 张 在 Rust 上的对照]")
    for card in sorted(PYTHON_BROKEN):
        status, detail = broken_results.get(card, ("NOT_IN_ORACLE", ""))
        line = f"  - {card}: {status}"
        if detail:
            line += f"  ({detail})"
        print(line)


if __name__ == "__main__":
    probe = RustCombatProbe()
    run_audit(probe)

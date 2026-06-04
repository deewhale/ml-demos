"""第 1 层 · sts_lightspeed（社区金标准 C++ 模拟器）卡牌行为验收。

拿独立 wiki oracle 考 sts_lightspeed。仿 test_rust_card_behavior.py /
test_combat_card_behavior.py，但用 LightspeedCombatProbe。审计模式：不 hard-assert，
红灯即“引擎或 binding 与 wiki 不符”，需人工判读（数值错 = 引擎真错；
状态缺失多半是 binding snapshot 不暴露该状态字段）。

本轮**不跳过力量/格挡预设**（lightspeed 支持 strength/block 注入）。
card id 映射不到的 [SKIP]。

跑法：.venv/bin/python tools/test_lightspeed_card_behavior.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = str(Path(__file__).resolve().parent.parent)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from tools.card_oracle import ORACLE, CardExpectation
from tools.test_combat_card_behavior import check_expectation
from v8.backends.lightspeed_combat_probe import (
    LightspeedCombatProbe,
    _PY_TO_CARDID,
)

# Python / Rust 引擎曾失败的卡，单独列出对照金标准做对没有
PRIOR_BROKEN = {
    "Dark Shackles",
    "Panacea",
    "PanicButton",
    "J.A.X.",
    "Sword Boomerang",
    "Disarm",
}


def _can_map(exp: CardExpectation) -> bool:
    return exp.card_id in _PY_TO_CARDID


def run_audit(probe: LightspeedCombatProbe) -> None:
    ran = passed = red = skipped = 0
    broken_results = {}  # card -> (status, detail)

    for exp in ORACLE:
        label = exp.label or exp.card_id
        if not _can_map(exp):
            skipped += 1
            print(f"[SKIP] {label} (lightspeed CardId 无此映射)")
            if exp.card_id in PRIOR_BROKEN:
                broken_results.setdefault(exp.card_id, ("SKIP", "无 CardId 映射"))
            continue

        ran += 1
        fails = check_expectation(probe, exp)
        if fails:
            red += 1
            print(f"[FAIL] {label}")
            for f in fails:
                print(f"         - {f}")
            print(f"         (oracle 来源: {exp.source})")
            if exp.card_id in PRIOR_BROKEN:
                broken_results[exp.card_id] = ("FAIL", "; ".join(fails))
        else:
            passed += 1
            print(f"[ OK ] {label}")
            if exp.card_id in PRIOR_BROKEN:
                broken_results.setdefault(exp.card_id, ("OK", ""))

    print(
        f"\n[lightspeed 验收汇总] 跑了 {ran}，过 {passed}，"
        f"红灯 {red}，跳过 {skipped}"
    )

    print("\n[Python/Rust 曾失败的卡 在 lightspeed 上的对照]")
    for card in sorted(PRIOR_BROKEN):
        status, detail = broken_results.get(card, ("NOT_IN_ORACLE", ""))
        line = f"  - {card}: {status}"
        if detail:
            line += f"  ({detail})"
        print(line)


if __name__ == "__main__":
    probe = LightspeedCombatProbe()
    run_audit(probe)

"""遗物行为验收：用 sts_lightspeed（金标准）跑 RELIC_ORACLE 全部，逐条断言。

oracle 是 wiki 独立标准答案；lightspeed 对不上如实记 [FAIL]（附引擎实际 vs wiki 期望），
绝不改 oracle 迁就引擎。重点验「开战触发类 atBattleStart 遗物」——遗物到底装上没、
开战效果有没有真触发。

跑法：.venv/bin/python tools/test_lightspeed_relic_behavior.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = str(Path(__file__).resolve().parent.parent)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from tools.relic_oracle import RELIC_ORACLE, RelicExpectation
from v8.backends.lightspeed_relic_probe import LightspeedRelicProbe


def _name(exp: RelicExpectation) -> str:
    return f"{exp.relic}" + (f" [{exp.label}]" if exp.label else "")


def _check_subset(actual: dict, expected: dict) -> list:
    """返回不匹配项列表 [(key, expected, actual), ...]；空 = 全过。"""
    mismatches = []
    for k, v in expected.items():
        a = actual.get(k, 0)
        if a != v:
            mismatches.append((k, v, a))
    return mismatches


def run() -> int:
    probe = LightspeedRelicProbe()
    total = len(RELIC_ORACLE)
    passed = 0
    fails = []

    print(f"=== 遗物行为验收 (lightspeed 金标准, {total} 条) ===\n")

    for exp in RELIC_ORACLE:
        name = _name(exp)
        try:
            res = probe.probe(exp)
        except Exception as e:  # noqa: BLE001  接入/枚举错误
            print(f"[FAIL] {name}: probe 抛错 {e!r}")
            fails.append((name, f"probe 抛错 {e!r}"))
            continue

        problems = []

        # 玩家开战状态断言
        if exp.expect_player:
            mm = _check_subset(res.player_snapshot, exp.expect_player)
            for k, v, a in mm:
                problems.append(f"player.{k}: wiki={v} lightspeed={a}")

        # 每只怪开战状态断言（全体性）
        if exp.expect_enemies:
            if not res.enemies_snapshot:
                problems.append("expect_enemies 但快照里没有敌人")
            for i, e in enumerate(res.enemies_snapshot):
                mm = _check_subset(e, exp.expect_enemies)
                for k, v, a in mm:
                    problems.append(f"enemy[{i}].{k}: wiki={v} lightspeed={a}")

        # play_card 伤害断言
        if exp.expect_enemy_hp_delta is not None:
            if not res.played:
                problems.append("expect_enemy_hp_delta 但没打牌")
            elif res.play_success is False:
                problems.append(f"play_card 失败 (error={res.error})")
            elif res.target_hp_delta != exp.expect_enemy_hp_delta:
                problems.append(
                    f"enemy_hp_delta: wiki={exp.expect_enemy_hp_delta} "
                    f"lightspeed={res.target_hp_delta}"
                )

        if problems:
            print(f"[FAIL] {name}")
            for p in problems:
                print(f"         {p}")
            print(f"         source: {exp.source}")
            fails.append((name, "; ".join(problems)))
        else:
            print(f"[OK]   {name}")
            passed += 1

    print(f"\n=== 汇总: {passed}/{total} 过, {len(fails)} 红 ===")
    if fails:
        print("\n红条目（lightspeed 实际 vs wiki 期望）：")
        for name, why in fails:
            print(f"  - {name}: {why}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(run())

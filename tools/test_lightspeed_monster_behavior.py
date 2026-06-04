"""怪物（敌人）行为验收 harness：拿 wiki 金标准 oracle 验 sts_lightspeed 引擎。

跑 MONSTER_ORACLE 全部条目，对每条用 LightspeedMonsterProbe 读开战瞬间快照，
逐项断言（HP 范围 / 首回合意图 / pre-battle 开战状态 / 敌人数量），打印 [OK]/[FAIL]
（FAIL 附实际 vs 期望）+ 汇总。

**审计模式**：oracle 是 wiki 独立标准答案，引擎对不上如实记，绝不改 oracle。

用法：
  .venv/bin/python tools/test_lightspeed_monster_behavior.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# 让本脚本能 import 仓库根下的 v8 包 + tools 同级模块。
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.monster_oracle import MONSTER_ORACLE, MonsterExpectation  # noqa: E402
from v8.backends.lightspeed_monster_probe import (  # noqa: E402
    LightspeedMonsterProbe,
    MonsterProbeResult,
)


def _check_one(exp: MonsterExpectation, res: MonsterProbeResult) -> list[str]:
    """对一条期望逐项断言，返回失败描述列表（空 = 全过）。"""
    fails: list[str] = []

    # ---- 群体断言：存活敌人数量 ----
    if exp.alive_count is not None and res.alive_count != exp.alive_count:
        fails.append(
            f"alive_count: 期望 {exp.alive_count}, 实际 {res.alive_count}"
        )

    # ---- 选目标敌人 ----
    target = None
    if exp.name_contains is not None:
        key = exp.name_contains.upper()
        for e in res.enemies:
            if key in str(e.get("name", "")).upper():
                target = e
                break
        if target is None:
            fails.append(
                f"找不到 name 含 '{exp.name_contains}' 的敌人; "
                f"实际敌人={[e.get('name') for e in res.enemies]}"
            )
            return fails
    else:
        if exp.enemy_index >= len(res.enemies):
            fails.append(
                f"enemy_index={exp.enemy_index} 越界; "
                f"存活敌人数={res.alive_count}"
            )
            return fails
        target = res.enemies[exp.enemy_index]

    name = target.get("name")
    hp = int(target.get("hp", -1))

    # ---- HP 范围 ----
    if exp.hp_min is not None and hp < exp.hp_min:
        fails.append(f"hp: {hp} < hp_min {exp.hp_min} (敌人={name})")
    if exp.hp_max is not None and hp > exp.hp_max:
        fails.append(f"hp: {hp} > hp_max {exp.hp_max} (敌人={name})")

    # ---- 首回合意图 ----
    if exp.is_attack is not None:
        actual_attack = bool(target.get("is_attack", False))
        if actual_attack != exp.is_attack:
            fails.append(
                f"is_attack: 期望 {exp.is_attack}, 实际 {actual_attack} "
                f"(intent={target.get('intent')}, dmg={target.get('intent_damage')})"
            )
    if exp.dmg_min is not None:
        d = int(target.get("intent_damage", -1))
        if d < exp.dmg_min:
            fails.append(f"intent_damage: {d} < dmg_min {exp.dmg_min} (intent={target.get('intent')})")
    if exp.dmg_max is not None:
        d = int(target.get("intent_damage", -1))
        if d > exp.dmg_max:
            fails.append(f"intent_damage: {d} > dmg_max {exp.dmg_max} (intent={target.get('intent')})")

    # ---- pre-battle 开战状态（精确值）----
    for field_name in ("metallicize", "artifact", "block", "strength", "strength_up"):
        expected = getattr(exp, field_name)
        if expected is not None:
            actual = int(target.get(field_name, 0))
            if actual != expected:
                extra = ""
                # strength 不匹配但同量 strength_up 存在：引擎把这只怪的开战力量表示成
                # GENERIC_STRENGTH_UP power（开回合 +X 力量），非开战瞬间的 Strength 叠层。
                # 这是「表示形式差异」，不是引擎算错——明确标注供审计区分。
                if field_name == "strength" and int(target.get("strength_up", 0)) == expected:
                    extra = (
                        f" [注: 引擎以 GENERIC_STRENGTH_UP power 表示, strength_up={expected}"
                        f"（开回合 +力量），开战瞬间无 Strength 叠层——表示形式差异，非数值错误]"
                    )
                fails.append(
                    f"{field_name}: 期望 {expected}, 实际 {actual} (敌人={name}){extra}"
                )

    return fails


def main() -> int:
    probe = LightspeedMonsterProbe(ascension=0, character="IRONCLAD")

    total = len(MONSTER_ORACLE)
    passed = 0
    failed_labels: list[str] = []

    print("=" * 78)
    print("怪物行为验收 harness — wiki 金标准 oracle vs sts_lightspeed (A0, IRONCLAD)")
    print("=" * 78)

    for exp in MONSTER_ORACLE:
        try:
            res = probe.probe(exp.encounter, exp.seed)
        except Exception as ex:  # 接入级错误（encounter 名错 / 绑定缺字段等）
            print(f"[FAIL] {exp.label:28s} <{exp.encounter}> 探针异常: {ex!r}")
            failed_labels.append(exp.label)
            continue

        fails = _check_one(exp, res)
        if not fails:
            # 摘要：目标敌人快照
            target = None
            if exp.name_contains:
                for e in res.enemies:
                    if exp.name_contains.upper() in str(e.get("name", "")).upper():
                        target = e
                        break
            elif exp.enemy_index < len(res.enemies):
                target = res.enemies[exp.enemy_index]
            summ = ""
            if target:
                summ = (
                    f"hp={target.get('hp')} intent={target.get('intent')} "
                    f"atk={target.get('is_attack')} dmg={target.get('intent_damage')}"
                )
            print(f"[OK]   {exp.label:28s} <{exp.encounter}> {summ}")
            passed += 1
        else:
            print(f"[FAIL] {exp.label:28s} <{exp.encounter}>")
            for f in fails:
                print(f"          - {f}")
            failed_labels.append(exp.label)

    print("=" * 78)
    print(f"汇总: {passed}/{total} 通过")
    if failed_labels:
        print(f"红: {', '.join(failed_labels)}")
    else:
        print("全部通过。")
    print("=" * 78)

    return 0 if not failed_labels else 1


if __name__ == "__main__":
    sys.exit(main())

"""LightspeedMonsterProbe：用 sts_lightspeed（社区金标准 C++ 模拟器）读怪物开战快照。

给定一条 MonsterExpectation，用绑定层 `make_encounter` 构造一场战斗（**不覆写敌人 hp /
不塞手牌 / 不动玩家 buff**，让引擎按 seed 生成真实怪物 hp、roll 首回合意图、施加 pre-battle
开战机制），读开战瞬间快照，返回中性结果（敌人列表的 hp / intent / 状态 / 数量）。

probe 只做「构造 + 读」，不做任何断言判断——断言逻辑全在
`tools/test_lightspeed_monster_behavior.py`（保持 probe 中性、引擎真值不被污染）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from v8.backends.lightspeed_loader import load_lightspeed


@dataclass
class MonsterProbeResult:
    """一次怪物探针的中性结果。"""

    encounter: str
    seed: int
    # 仅存活（有效）敌人列表，每个是中性 dict：
    #   idx / name / hp / max_hp / block / strength / strength_up / artifact /
    #   metallicize / vulnerable / weak / poison / ritual /
    #   intent / move_id / is_attack / intent_damage / intent_hits
    enemies: list[dict[str, Any]] = field(default_factory=list)
    alive_count: int = 0
    raw_enemy_count: int = 0  # 含空 slot 的原始 monsterCount


# 引擎用 "INVALID = 0" 这种空 slot 占位（如 AUTOMATON 遭遇有两个空 minion 槽）。
# 过滤掉：alive=False 或 name 以 INVALID 开头 或 max_hp<=0。
def _is_real_enemy(e: dict[str, Any]) -> bool:
    name = str(e.get("name", ""))
    if name.startswith("INVALID"):
        return False
    if int(e.get("max_hp", 0)) <= 0:
        return False
    return bool(e.get("alive", False))


class LightspeedMonsterProbe:
    """构造受控遭遇 + 读开战快照的探针。"""

    def __init__(self, ascension: int = 0, character: str = "IRONCLAD"):
        self._sts = load_lightspeed()
        self._ascension = ascension
        self._character = character

    def probe(self, encounter: str, seed: int) -> MonsterProbeResult:
        sts = self._sts
        char = getattr(sts.CharacterClass, self._character)
        enc = getattr(sts.MonsterEncounter, encounter)

        gc = sts.GameContext(char, seed, self._ascension)
        bc = sts.make_encounter(gc, enc)  # 不覆写敌人 hp / buff：读引擎真值
        snap = sts.get_combat_snapshot(bc)

        raw = list(snap.get("enemies", []))
        real = [e for e in raw if _is_real_enemy(e)]
        return MonsterProbeResult(
            encounter=encounter,
            seed=seed,
            enemies=real,
            alive_count=len(real),
            raw_enemy_count=len(raw),
        )

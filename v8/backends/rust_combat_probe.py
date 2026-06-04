"""RustCombatProbe：用 Rust 引擎（engine-rs / RustCombatEngine）实现 CombatProbe。

把 oracle 的 Python 卡 id 映射成 Rust 卡 id，造受控单卡战斗，读中性 CardPlayResult。

接入要点（2026-06-04 探针实测）：
  - RustCombatEngine(player_hp, player_max_hp, energy, deck, enemies, seed, relics=None)
    enemies = list[(enemy_id, hp, max_hp, move_damage, move_hits)]
  - start_combat() 随机抽牌，手牌上限 5。故 deck 里把目标卡塞多份提高命中。
  - get_legal_actions() 返回 Action：.action_type "PlayCard"/"EndTurn"、.index 手牌索引、
    .target 敌人索引（自身/无目标卡 .target == -1，不是 None）。
  - get_combat_snapshot() 返回 dict：enemies(list dict: hp/max_hp/block/statuses/...)、
    hand(list dict: card_id)、player_hp/player_block/energy/player_effects。
  - statuses / player_effects 是 list[{status_name, amount, status_id}]，需转成 {name: amount}。
  - Rust status_name 已是 STS 显示名（Vulnerable / Weakened / Strength），与 oracle 基本对齐；
    个别键（No Block）做归一化映射。
"""
from __future__ import annotations

from typing import Dict, List, Optional

from v8.backends.combat_probe import CardPlayResult, CombatSetup
from v8.backends.rust_engine_loader import load_sts_engine


# oracle（Python 引擎 id）→ Rust 引擎 id。
# 大部分 oracle id 与 Rust 注册表 id 完全一致（"Body Slam" / "Heavy Blade" / "Iron Wave" /
# "Sword Boomerang" / "Dark Shackles" / "J.A.X." / "PanicButton" 等），只有起始牌需要映射。
_PY_TO_RUST_ID: Dict[str, str] = {
    "Strike_R": "Strike",
    "Defend_R": "Defend",
    # 以下 id 在 Rust 注册表中同名（列出以示覆盖范围 / 文档化）
    "Bash": "Bash",
    "Anger": "Anger",
    "Clothesline": "Clothesline",
    "Twin Strike": "Twin Strike",
    "Thunderclap": "Thunderclap",
    "Iron Wave": "Iron Wave",
    "Pommel Strike": "Pommel Strike",
    "Shrug It Off": "Shrug It Off",
    "Inflame": "Inflame",
    "Body Slam": "Body Slam",
    "Heavy Blade": "Heavy Blade",
    "Flex": "Flex",
    "Carnage": "Carnage",
    "Dropkick": "Dropkick",
    "Hemokinesis": "Hemokinesis",
    "Pummel": "Pummel",
    "Searing Blow": "Searing Blow",
    "Uppercut": "Uppercut",
    "Battle Trance": "Battle Trance",
    "Bloodletting": "Bloodletting",
    "Disarm": "Disarm",
    "Entrench": "Entrench",
    "Ghostly Armor": "Ghostly Armor",
    "Seeing Red": "Seeing Red",
    "Bludgeon": "Bludgeon",
    "Feed": "Feed",
    "Impervious": "Impervious",
    "Limit Break": "Limit Break",
    "Barricade": "Barricade",
    "Berserk": "Berserk",
    "Juggernaut": "Juggernaut",
    "Blind": "Blind",
    "Dark Shackles": "Dark Shackles",
    "Finesse": "Finesse",
    "Flash of Steel": "Flash of Steel",
    "Good Instincts": "Good Instincts",
    "Panacea": "Panacea",
    "PanicButton": "PanicButton",
    "Swift Strike": "Swift Strike",
    "Trip": "Trip",
    "Bite": "Bite",
    "J.A.X.": "J.A.X.",
    "Cleave": "Cleave",
    "Intimidate": "Intimidate",
    "Shockwave": "Shockwave",
    "Immolate": "Immolate",
    "Reaper": "Reaper",
    "Dramatic Entrance": "Dramatic Entrance",
    "Sword Boomerang": "Sword Boomerang",
}

# 敌人 id 映射（JawWorm 在 Rust 同名）
_PY_TO_RUST_ENEMY: Dict[str, str] = {
    "JawWorm": "JawWorm",
}

# Rust status_name → oracle canonical 名（绝大多数已对齐，列出已知需归一项）。
# 归一化只统一键名，不动层数/数值——数值不准是引擎错。
_RUST_STATUS_TO_CANON: Dict[str, str] = {
    "Vulnerable": "Vulnerable",
    "Weakened": "Weakened",
    "Weak": "Weakened",          # 若 Rust 在别处用短名，统一到 oracle 的 Weakened
    "Strength": "Strength",
    "Shackled": "Shackled",
    "Artifact": "Artifact",
    "Barricade": "Barricade",
    "Berserk": "Berserk",
    "NoDraw": "NoDraw",
    "No Draw": "NoDraw",
    "NoBlock": "No Block",        # oracle 用 "No Block"（带空格）
    "No Block": "No Block",
}


class CardNotMappedError(KeyError):
    """oracle card_id 在 Rust 注册表无对应 id。"""


def map_card_id(py_id: str) -> str:
    if py_id in _PY_TO_RUST_ID:
        return _PY_TO_RUST_ID[py_id]
    raise CardNotMappedError(py_id)


def _normalize_statuses(status_list) -> Dict[str, int]:
    """Rust 的 list[{status_name, amount}] → {canonical_name: amount}。"""
    out: Dict[str, int] = {}
    for s in status_list or []:
        raw = s.get("status_name")
        if raw is None:
            continue
        canon = _RUST_STATUS_TO_CANON.get(raw, raw)
        out[canon] = s.get("amount", 0)
    return out


class RustCombatProbe:
    """实现 CombatProbe（鸭子类型）。"""

    def __init__(self):
        self._mod = load_sts_engine()

    def play_single_card(
        self,
        setup: CombatSetup,
        hand_index: int,
        target_index: int,
    ) -> CardPlayResult:
        if setup.player_block or setup.player_strength:
            raise NotImplementedError("rust probe 暂不支持力量/格挡预设")

        # 1. 目标卡的 Rust id（hand_index 指向 setup.hand 里的目标卡）
        py_card = setup.hand[hand_index]
        rust_card = map_card_id(py_card)

        # 2. deck：把目标卡塞 10 份，保证 start_combat 抽牌一定抽到
        deck = [rust_card] * 10

        # 3. enemies
        rust_enemy = _PY_TO_RUST_ENEMY.get(setup.enemy_id, setup.enemy_id)
        enemies = [
            (rust_enemy, setup.enemy_hp, setup.enemy_hp, 11, 1)
            for _ in range(setup.enemy_count)
        ]

        eng = self._mod.RustCombatEngine(
            setup.player_hp,
            setup.player_hp,
            setup.player_energy,
            deck,
            enemies,
            seed=42,
        )
        eng.start_combat()
        snap_before = eng.get_combat_snapshot()

        # 4. 找到要打的 PlayCard action
        action = self._find_play_action(eng, snap_before, rust_card, target_index)
        if action is None:
            return CardPlayResult(
                success=False,
                enemy_hp_delta=0,
                player_block_delta=0,
                energy_delta=0,
                enemy_statuses={},
                player_statuses={},
                effects=[],
                all_enemy_hp_deltas=[0] * setup.enemy_count,
                all_enemy_statuses=[{} for _ in range(setup.enemy_count)],
            )

        hp_before = [e["hp"] for e in snap_before["enemies"]]
        block_before = snap_before["player_block"]
        energy_before = snap_before["energy"]

        eng.take_action(action)
        snap_after = eng.get_combat_snapshot()

        enemies_after = snap_after["enemies"]
        all_hp_deltas = [
            hp_before[i] - enemies_after[i]["hp"] for i in range(len(enemies_after))
        ]
        all_statuses = [_normalize_statuses(e.get("statuses")) for e in enemies_after]
        read_idx = target_index if target_index >= 0 else 0

        return CardPlayResult(
            success=True,
            enemy_hp_delta=all_hp_deltas[read_idx],
            player_block_delta=snap_after["player_block"] - block_before,
            energy_delta=energy_before - snap_after["energy"],
            enemy_statuses=all_statuses[read_idx],
            player_statuses=_normalize_statuses(snap_after.get("player_effects")),
            effects=[],  # Rust 不给结构化 effects，留空（runner 不强依赖）
            all_enemy_hp_deltas=all_hp_deltas,
            all_enemy_statuses=all_statuses,
        )

    @staticmethod
    def _find_play_action(eng, snap, rust_card: str, target_index: int):
        """在 legal actions 里找 action_type=='PlayCard' 且手牌卡 id == rust_card、
        目标匹配的 action。

        Rust target 约定（实测）：
          - 单体卡：.target == 敌人索引（0/1/2...）
          - 自身/无目标/全体卡（power、AoE 如 Thunderclap/Cleave、随机分配如 Sword
            Boomerang）：.target == -1
        oracle 对单体伤害卡与 AoE 卡都用 target_index>=0；故先精确匹配敌人索引，
        匹配不到再回退到 .target==-1 的 action（覆盖 AoE / 全体卡）。"""
        hand = snap["hand"]
        candidates = []
        for a in eng.get_legal_actions():
            if a.action_type != "PlayCard":
                continue
            idx = a.index
            if idx is None or idx >= len(hand):
                continue
            if hand[idx]["card_id"] != rust_card:
                continue
            candidates.append(a)

        if target_index >= 0:
            # 优先精确命中敌人索引（单体卡）
            for a in candidates:
                if a.target == target_index:
                    return a
            # 回退：AoE / 全体 / 随机分配卡在 Rust 用 -1
            for a in candidates:
                if a.target == -1 or a.target is None:
                    return a
        else:
            # 自身 / 无目标卡
            for a in candidates:
                if a.target == -1 or a.target is None:
                    return a
        return None

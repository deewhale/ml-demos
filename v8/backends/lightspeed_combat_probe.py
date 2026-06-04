"""LightspeedCombatProbe：用 sts_lightspeed（社区金标准 C++ 模拟器）实现 CombatProbe。

把 oracle 的 Python 卡 id 映射成 lightspeed 的 `CardId` 枚举，造受控单卡战斗，
读中性 CardPlayResult。

接入要点（2026-06-04 探针实测，binding API）：
  - gc = sts.GameContext(CharacterClass.IRONCLAD, seed, ascension)
  - bc = sts.make_test_combat(gc, encounter, player_hp=, energy=, hand=[CardId...],
        enemy_hps=[...], strength=, block=)
        · encounter 是 MonsterEncounter 枚举；怪的数量/种类由 encounter 定，只能覆写
          每只 hp（enemy_hps 按 idx），不能任意加怪。
        · 单敌用 JAW_WORM（单怪）；多敌用 THREE_LOUSE（3 怪）验全体性。
  - sts.play_card(bc, hand_index, target_index)  target 默认 0
  - snap = sts.get_combat_snapshot(bc)
        player: {hp,max_hp,block,energy,strength,dexterity,artifact,vulnerable,weak,frail}
        enemies: [{idx,name,hp,max_hp,block,alive,strength,vulnerable,weak,poison},...]
        outcome,turn,cards_in_hand
  - 注意：snapshot 的玩家/敌人状态是**固定字段**（vulnerable/weak/strength/artifact/...），
    不是通用 dict。oracle 里要查的某些状态（Shackled / NoDraw / Barricade / Berserk /
    No Block / Dexterity 等）binding 当前不暴露 → 归一化字典里查不到 → 当作 0。
    这类“引擎可能做对了但 snapshot 不暴露”的项会在测试里体现为状态 FAIL，需人工判读，
    不是数值错。
"""
from __future__ import annotations

from typing import Dict, Optional

from v8.backends.combat_probe import CardPlayResult, CombatSetup
from v8.backends.lightspeed_loader import load_lightspeed


# oracle（Python 引擎 id）→ lightspeed CardId 枚举名（字符串，运行时 getattr 取枚举）。
# 以 sts.CardId 实际枚举名为准（dir(sts.CardId) 核对过）。绝大多数是 UPPER_SNAKE，
# 起始牌带 _RED 后缀，J.A.X. 在枚举里叫 JAX。
_PY_TO_CARDID: Dict[str, str] = {
    "Strike_R": "STRIKE_RED",
    "Defend_R": "DEFEND_RED",
    "Bash": "BASH",
    "Anger": "ANGER",
    "Clothesline": "CLOTHESLINE",
    "Twin Strike": "TWIN_STRIKE",
    "Thunderclap": "THUNDERCLAP",
    "Iron Wave": "IRON_WAVE",
    "Pommel Strike": "POMMEL_STRIKE",
    "Shrug It Off": "SHRUG_IT_OFF",
    "Inflame": "INFLAME",
    "Body Slam": "BODY_SLAM",
    "Heavy Blade": "HEAVY_BLADE",
    "Flex": "FLEX",
    "Carnage": "CARNAGE",
    "Dropkick": "DROPKICK",
    "Hemokinesis": "HEMOKINESIS",
    "Pummel": "PUMMEL",
    "Searing Blow": "SEARING_BLOW",
    "Uppercut": "UPPERCUT",
    "Battle Trance": "BATTLE_TRANCE",
    "Bloodletting": "BLOODLETTING",
    "Disarm": "DISARM",
    "Entrench": "ENTRENCH",
    "Ghostly Armor": "GHOSTLY_ARMOR",
    "Seeing Red": "SEEING_RED",
    "Bludgeon": "BLUDGEON",
    "Feed": "FEED",
    "Impervious": "IMPERVIOUS",
    "Limit Break": "LIMIT_BREAK",
    "Barricade": "BARRICADE",
    "Berserk": "BERSERK",
    "Juggernaut": "JUGGERNAUT",
    "Blind": "BLIND",
    "Dark Shackles": "DARK_SHACKLES",
    "Finesse": "FINESSE",
    "Flash of Steel": "FLASH_OF_STEEL",
    "Good Instincts": "GOOD_INSTINCTS",
    "Panacea": "PANACEA",
    "PanicButton": "PANIC_BUTTON",
    "Swift Strike": "SWIFT_STRIKE",
    "Trip": "TRIP",
    "Bite": "BITE",
    "J.A.X.": "JAX",
    "Cleave": "CLEAVE",
    "Intimidate": "INTIMIDATE",
    "Shockwave": "SHOCKWAVE",
    "Immolate": "IMMOLATE",
    "Reaper": "REAPER",
    "Dramatic Entrance": "DRAMATIC_ENTRANCE",
    "Sword Boomerang": "SWORD_BOOMERANG",
    # ---- 本轮新接：pile_inspection / deck_composition / x_cost / multi_play / other ----
    "Clash": "CLASH",
    "Headbutt": "HEADBUTT",
    "Perfected Strike": "PERFECTED_STRIKE",
    "Wild Strike": "WILD_STRIKE",
    "Reckless Charge": "RECKLESS_CHARGE",
    "Power Through": "POWER_THROUGH",
    "Sever Soul": "SEVER_SOUL",
    "Fiend Fire": "FIEND_FIRE",
    "Mind Blast": "MIND_BLAST",
    "Whirlwind": "WHIRLWIND",
    "Transmutation": "TRANSMUTATION",
    "Rampage": "RAMPAGE",
    "Burning Pact": "BURNING_PACT",
    "True Grit": "TRUE_GRIT",
    "Second Wind": "SECOND_WIND",
    "Warcry": "WARCRY",
    "Dual Wield": "DUAL_WIELD",
    "Havoc": "HAVOC",
    "Seeing Red": "SEEING_RED",
    "Offering": "OFFERING",
    "Bandage Up": "BANDAGE_UP",
    "Master of Strategy": "MASTER_OF_STRATEGY",
    "Infernal Blade": "INFERNAL_BLADE",
    "Jack Of All Trades": "JACK_OF_ALL_TRADES",
    "Hand of Greed": "HAND_OF_GREED",
    "HandOfGreed": "HAND_OF_GREED",
}


class CardNotMappedError(KeyError):
    """oracle card_id 在 lightspeed CardId 枚举无对应。"""


def _status_dict(entity: Dict) -> Dict[str, int]:
    """把 snapshot 实体的固定状态字段归一化成 oracle 用的 canonical 键。

    lightspeed snapshot 用小写直接字段：vulnerable / weak / strength / artifact / ...
    映射成 oracle 键：Vulnerable / Weakened / Strength / Artifact / ...
    （lightspeed 的 weak 即 STS 的虚弱 = oracle 的 Weakened）

    binding 不暴露的状态（Shackled / NoDraw / Barricade / Berserk / No Block /
    Dexterity 等）这里没法填 → check 时按 0 处理 → 体现为状态 FAIL，需人工判读。
    """
    out: Dict[str, int] = {}
    _maybe(out, "Vulnerable", entity.get("vulnerable"))
    _maybe(out, "Weakened", entity.get("weak"))
    _maybe(out, "Strength", entity.get("strength"))
    _maybe(out, "Artifact", entity.get("artifact"))
    _maybe(out, "Frail", entity.get("frail"))
    _maybe(out, "Dexterity", entity.get("dexterity"))
    _maybe(out, "Poison", entity.get("poison"))
    # ---- 卡牌验收新增暴露字段 -> oracle canonical 键 ----
    _maybe(out, "NoDraw", entity.get("no_draw"))         # Battle Trance
    _maybe(out, "No Block", entity.get("no_block"))      # Panic Button
    _maybe(out, "Barricade", entity.get("barricade"))    # Barricade
    _maybe(out, "Shackled", entity.get("shackled"))      # Dark Shackles（敌人）
    return out


def _maybe(d: Dict[str, int], key: str, val) -> None:
    if val:
        d[key] = int(val)


class LightspeedCombatProbe:
    """实现 CombatProbe（鸭子类型）。"""

    def __init__(self):
        self._sts = load_lightspeed()
        # 单敌 / 多敌 encounter（怪数由 encounter 定，多敌验 AoE 全体性即可）
        self._single_encounter = self._sts.MonsterEncounter.JAW_WORM
        self._multi_encounter = self._sts.MonsterEncounter.THREE_LOUSE

    def play_single_card(
        self,
        setup: CombatSetup,
        hand_index: int,
        target_index: int,
    ) -> CardPlayResult:
        sts = self._sts

        # 1. 映射目标卡 + 选 encounter
        py_card = setup.hand[hand_index]
        card_enum = self._card_enum(py_card)

        if setup.enemy_count > 1:
            encounter = self._multi_encounter
        else:
            encounter = self._single_encounter

        # 2. 手牌：按 setup.hand 原样映射每张牌（不再用 Defend 占位覆盖其它槽位）——
        #    Sever Soul / Fiend Fire / Burning Pact / Clash 等依赖手牌真实构成的卡需要
        #    确切的其它手牌，占位覆盖会污染判定。能映射的才进手牌；映射不到的占位用 Defend。
        hand = []
        for c in setup.hand:
            try:
                hand.append(self._card_enum(c))
            except CardNotMappedError:
                hand.append(sts.CardId.DEFEND_RED)
        if not hand:
            hand = [card_enum]

        # 出牌前能量：setup.energy 覆写优先，否则用 player_energy。
        energy = setup.energy if setup.energy >= 0 else setup.player_energy
        # 预置抽牌堆（Mind Blast / 牌堆机制 / deck_composition）
        draw_pile = [self._card_enum(c) for c in setup.draw_pile]

        gc = sts.GameContext(sts.CharacterClass.IRONCLAD, 1, 0)
        bc = sts.make_test_combat(
            gc,
            encounter,
            player_hp=setup.player_hp,
            energy=energy,
            hand=hand,
            enemy_hps=[setup.enemy_hp],  # 见下：会按实际怪数补齐
            strength=setup.player_strength,
            block=setup.player_block,
            draw_pile=draw_pile,
            clear_draw_pile=setup.clear_draw_pile,
        )

        snap_before = sts.get_combat_snapshot(bc)
        n_enemies = len(snap_before["enemies"])
        # enemy_hps 按实际怪数覆写 hp（前面只传了 1 个，重建以对齐怪数）
        if n_enemies > 1:
            bc = sts.make_test_combat(
                gc,
                encounter,
                player_hp=setup.player_hp,
                energy=energy,
                hand=hand,
                enemy_hps=[setup.enemy_hp] * n_enemies,
                strength=setup.player_strength,
                block=setup.player_block,
                draw_pile=draw_pile,
                clear_draw_pile=setup.clear_draw_pile,
            )
            snap_before = sts.get_combat_snapshot(bc)
            n_enemies = len(snap_before["enemies"])

        read_idx = target_index if target_index >= 0 else 0
        tgt = target_index if target_index >= 0 else 0

        hp_before = [e["hp"] for e in snap_before["enemies"]]
        block_before = snap_before["player"]["block"]
        energy_before = snap_before["player"]["energy"]

        success = True
        try:
            # binding 现在返回 bool：False = 引擎判定该卡当前不可打出（如 Clash 手里有
            # 非攻击牌），已安全 no-op，不会崩溃。
            played = sts.play_card(bc, hand_index, tgt)
            success = bool(played)
        except Exception:  # binding 抛错
            success = False

        snap_after = sts.get_combat_snapshot(bc)
        enemies_after = snap_after["enemies"]
        all_hp_deltas = [
            hp_before[i] - enemies_after[i]["hp"] for i in range(len(enemies_after))
        ]
        all_statuses = [_status_dict(e) for e in enemies_after]

        return CardPlayResult(
            success=success,
            enemy_hp_delta=all_hp_deltas[read_idx] if read_idx < len(all_hp_deltas) else 0,
            player_block_delta=snap_after["player"]["block"] - block_before,
            energy_delta=energy_before - snap_after["player"]["energy"],
            enemy_statuses=all_statuses[read_idx] if read_idx < len(all_statuses) else {},
            player_statuses=_status_dict(snap_after["player"]),
            effects=[],  # binding 不给结构化 effects
            all_enemy_hp_deltas=all_hp_deltas,
            all_enemy_statuses=all_statuses,
            hand=tuple(snap_after.get("hand", [])),
            draw_pile=tuple(snap_after.get("draw_pile", [])),
            discard_pile=tuple(snap_after.get("discard_pile", [])),
            exhaust_pile=tuple(snap_after.get("exhaust_pile", [])),
            hand_size=snap_after.get("hand_size", 0),
            draw_pile_size=snap_after.get("draw_pile_size", 0),
            discard_pile_size=snap_after.get("discard_pile_size", 0),
            exhaust_pile_size=snap_after.get("exhaust_pile_size", 0),
        )

    def _card_enum(self, py_id: str):
        name = _PY_TO_CARDID.get(py_id)
        if name is None:
            raise CardNotMappedError(py_id)
        enum = getattr(self._sts.CardId, name, None)
        if enum is None:
            raise CardNotMappedError(f"{py_id} -> CardId.{name} 不存在")
        return enum


def can_map(py_id: str) -> bool:
    name = _PY_TO_CARDID.get(py_id)
    if name is None:
        return False
    sts = load_lightspeed()
    return getattr(sts.CardId, name, None) is not None

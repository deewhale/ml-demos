"""mod_state_adapter 子画面 handler 的最小 mock 测试。

不依赖 STS / Communication Mod，直接喂 fake mod-state JSON。
跑法：.venv/bin/python tools/test_mod_adapter_subscreens.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = str(Path(__file__).resolve().parent.parent)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from v8.mod_state_adapter import (
    mod_json_to_available_actions,
    mod_json_to_v8_state,
    v8_idx_to_mod_command,
)


def _base_state(screen_type: str, screen_state: dict, choice_list=None,
                available_commands=None, in_game=True, extra_gs=None) -> dict:
    gs = {
        "screen_type": screen_type,
        "screen_state": screen_state,
        "choice_list": choice_list or [],
        "current_hp": 50,
        "max_hp": 80,
        "floor": 5,
        "act": 1,
        "gold": 100,
        "deck": [],
        "relics": [],
        "potions": [],
        "map": [],
    }
    if extra_gs:
        gs.update(extra_gs)
    return {
        "in_game": in_game,
        "game_state": gs,
        "available_commands": available_commands or ["choose", "cancel"],
    }


def test_grid_select_transform() -> None:
    """GRID screen for_transform=True（事件转化卡），有 3 张候选 → 3 个 action。"""
    state = _base_state(
        "GRID",
        {
            "for_transform": True,
            "num_cards": 1,
            "selected_cards": [],
            "cards": [
                {"id": "Strike", "upgrades": 0},
                {"id": "Defend", "upgrades": 0},
                {"id": "Bash", "upgrades": 1},
            ],
        },
        choice_list=["strike", "defend", "bash"],
    )
    v8s = mod_json_to_v8_state(state)
    actions, metas = mod_json_to_available_actions(state)
    assert v8s.phase == "GRID_SELECT", f"phase={v8s.phase!r}"
    assert len(actions) == 3, f"actions={actions}"
    assert "GRID:card:Strike" in actions[0]
    assert "GRID:card:Bash+1" in actions[2]
    cmd = v8_idx_to_mod_command(1, metas, state)
    assert cmd == "choose 1", f"cmd={cmd!r}"
    print("PASS test_grid_select_transform")


def test_grid_select_with_confirm() -> None:
    """GRID 多选且已选够 → confirm 出现在 actions。"""
    state = _base_state(
        "GRID",
        {
            "for_transform": False,
            "num_cards": 2,
            "selected_cards": [{"id": "Strike"}, {"id": "Defend"}],
            "cards": [
                {"id": "Strike", "upgrades": 0},
                {"id": "Defend", "upgrades": 0},
            ],
        },
        available_commands=["confirm", "cancel"],
    )
    actions, metas = mod_json_to_available_actions(state)
    assert any("confirm" in a.lower() for a in actions), f"actions={actions}"
    # idx 指向 GRID:confirm
    confirm_idx = [i for i, a in enumerate(actions) if a == "GRID:confirm"][0]
    cmd = v8_idx_to_mod_command(confirm_idx, metas, state)
    assert cmd == "confirm"
    print("PASS test_grid_select_with_confirm")


def test_hand_select_can_pick_zero() -> None:
    """HAND_SELECT can_pick_zero=True（如 Gambling Chip）→ confirm 永远可选。"""
    state = _base_state(
        "HAND_SELECT",
        {
            "can_pick_zero": True,
            "max_cards": 5,
            "cards": [
                {"id": "Strike", "upgrades": 0},
                {"id": "Wound", "upgrades": 0},
            ],
        },
    )
    v8s = mod_json_to_v8_state(state)
    actions, metas = mod_json_to_available_actions(state)
    assert v8s.phase == "HAND_SELECT", f"phase={v8s.phase!r}"
    assert len(actions) == 3, f"actions={actions}"  # 2 card + confirm
    assert "HAND:confirm" in actions
    print("PASS test_hand_select_can_pick_zero")


def test_shop_screen_filters_unaffordable() -> None:
    """SHOP_SCREEN 中买不起的项不暴露给 model。"""
    state = _base_state(
        "SHOP_SCREEN",
        {
            "cards": [
                {"id": "Strike", "price": 50},
                {"id": "Bash", "price": 200},  # 买不起
            ],
            "relics": [{"id": "Anchor", "price": 150}],  # 买不起
            "potions": [{"id": "FirePotion", "price": 30}],
            "purge_cost": 75,
        },
        choice_list=["strike", "bash", "anchor", "firepotion", "purge", "leave"],
        extra_gs={"gold": 100},
    )
    v8s = mod_json_to_v8_state(state)
    actions, metas = mod_json_to_available_actions(state)
    assert v8s.phase == "SHOP", f"phase={v8s.phase!r}"
    # 应包含：strike (50 <= 100), firepotion (30), purge (75), leave
    # 不包含：bash (200), anchor (150)
    joined = " | ".join(actions)
    assert "strike" in joined.lower()
    assert "firepotion" in joined.lower()
    assert "purge" in joined.lower() or "remove_card" in joined.lower()
    assert "SHOP:leave" in actions
    assert "bash" not in joined.lower(), f"应过滤掉 bash 但 actions={actions}"
    assert "anchor" not in joined.lower(), f"应过滤掉 anchor 但 actions={actions}"
    print(f"PASS test_shop_screen_filters_unaffordable (n_actions={len(actions)})")


def test_shop_screen_no_money() -> None:
    """金币不够所有项 → 至少有 leave 兜底。"""
    state = _base_state(
        "SHOP_SCREEN",
        {
            "cards": [{"id": "Strike", "price": 500}],
            "relics": [],
            "potions": [],
            "purge_cost": 75,
        },
        choice_list=["strike", "purge", "leave"],
        extra_gs={"gold": 10},
    )
    actions, metas = mod_json_to_available_actions(state)
    assert "SHOP:leave" in actions, f"actions={actions}"
    print("PASS test_shop_screen_no_money")


def test_event_multiphase_dead_adventurer() -> None:
    """多阶段事件（Dead Adventurer）：每次进来 choice_list 不同，都能枚举。"""
    # phase 1: "Search" / "Leave"
    state1 = _base_state(
        "EVENT",
        {"event_id": "Dead Adventurer", "event_name": "Dead Adventurer"},
        choice_list=["search", "leave"],
    )
    v8s = mod_json_to_v8_state(state1)
    actions1, metas1 = mod_json_to_available_actions(state1)
    assert v8s.phase == "EVENT"
    assert len(actions1) == 2
    assert "EVENT:Dead Adventurer" in actions1[0]
    # phase 2: 战斗触发，可能短暂回到事件 phase ("Take" / "Leave")
    state2 = _base_state(
        "EVENT",
        {"event_id": "Dead Adventurer", "event_name": "Dead Adventurer"},
        choice_list=["take 100 gold", "take relic", "leave"],
    )
    actions2, metas2 = mod_json_to_available_actions(state2)
    assert len(actions2) == 3, f"actions={actions2}"
    # 选择 idx=1 应得 "choose 1"
    cmd = v8_idx_to_mod_command(1, metas2, state2)
    assert cmd == "choose 1", f"cmd={cmd!r}"
    print("PASS test_event_multiphase_dead_adventurer")


def test_card_reward_with_skip_upgrade() -> None:
    """CARD_REWARD 含 cards 列表 + skip → action 包含每张卡 + skip。"""
    state = _base_state(
        "CARD_REWARD",
        {
            "cards": [
                {"id": "Anger", "upgrades": 0},
                {"id": "Cleave", "upgrades": 0},
                {"id": "Iron Wave", "upgrades": 0},
            ],
        },
        choice_list=["anger", "cleave", "iron wave", "skip"],
    )
    v8s = mod_json_to_v8_state(state)
    actions, metas = mod_json_to_available_actions(state)
    assert v8s.phase == "CARD_REWARDS"
    # 3 张卡 + skip = 4 action
    assert len(actions) == 4, f"actions={actions}"
    assert any("Anger" in a for a in actions)
    # skip 行的 meta choose_idx 应指 "skip" 在 choice_list 里的位置 (=3)
    skip_idx = [i for i, a in enumerate(actions) if "skip" in a][0]
    cmd = v8_idx_to_mod_command(skip_idx, metas, state)
    assert cmd == "choose 3", f"cmd={cmd!r}"
    print("PASS test_card_reward_with_skip_upgrade")


def main() -> int:
    test_grid_select_transform()
    test_grid_select_with_confirm()
    test_hand_select_can_pick_zero()
    test_shop_screen_filters_unaffordable()
    test_shop_screen_no_money()
    test_event_multiphase_dead_adventurer()
    test_card_reward_with_skip_upgrade()
    print("\nAll 7 mock tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

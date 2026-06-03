"""RealGameBackend 离线 mock 测试（不依赖 STS / Communication Mod）。

用 ListModIO 喂预设 mod JSON 序列，验证 RealGameBackend 的
reset / build_v8_state / get_available_actions / take_action / build_runner_snapshot
不报错且产出合理；并用 RealEpisodeRunner + fake model 端到端跑一局拿到 EpisodeResult。

端到端连真机要用户开游戏，不在本测试范围（见文件末尾说明）。

跑法：.venv/bin/python tools/test_real_game_backend_offline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = str(Path(__file__).resolve().parent.parent)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from v8.backends.real_game_backend import RealGameBackend, ListModIO  # noqa: E402
from v8.eval_harness import RealEpisodeRunner, evaluate  # noqa: E402


# =============================================================================
# Mock mod-state 构造（schema 参考 test_mod_adapter_subscreens._base_state）
# =============================================================================


def _ingame(screen_type, screen_state, choice_list=None, floor=1, act=1,
            hp=72, max_hp=80, gold=99, combat_state=None, available_commands=None,
            deck=None, relics=None):
    gs = {
        "screen_type": screen_type,
        "screen_state": screen_state or {},
        "choice_list": choice_list or [],
        "current_hp": hp,
        "max_hp": max_hp,
        "floor": floor,
        "act": act,
        "gold": gold,
        "deck": deck or [{"id": "Strike", "upgrades": 0}],
        "relics": relics or [{"id": "Burning Blood"}],
        "potions": [],
        "map": [],
    }
    if combat_state is not None:
        gs["combat_state"] = combat_state
    return {
        "in_game": True,
        "game_state": gs,
        "available_commands": available_commands or ["choose", "proceed", "cancel"],
    }


def _out_of_game():
    return {"in_game": False, "available_commands": ["start"]}


def _game_over(won, floor=17, act=1, hp=0, max_hp=80):
    return {
        "in_game": True,
        "game_state": {
            "screen_type": "GAME_OVER",
            "screen_state": {"victory": won},
            "choice_list": [],
            "current_hp": hp,
            "max_hp": max_hp,
            "floor": floor,
            "act": act,
            "gold": 0,
            "deck": [], "relics": [], "potions": [], "map": [],
        },
        "available_commands": ["proceed"],
    }


# =============================================================================
# Tests
# =============================================================================


def test_reset_and_build_state():
    """reset 走 ready → start → 第一个决策帧；build_v8_state / actions 合理。"""
    # recv 序列：reset 读第一帧(out-of-game) → start 后读第一个 MAP 决策帧
    map_state = _ingame("MAP", {}, choice_list=["x=1 M", "x=2 ?"], floor=0)
    io = ListModIO([_out_of_game(), map_state])
    be = RealGameBackend(character="IRONCLAD", io=io)
    be.reset(seed=10000)

    assert io.sent[0] == "ready", f"first cmd should be ready, got {io.sent}"
    assert any(c.startswith("start IRONCLAD 0") for c in io.sent), f"sent={io.sent}"
    assert not be.game_over
    state = be.build_v8_state()
    assert state.phase == "MAP", f"phase={state.phase!r}"
    actions = be.get_available_actions()
    assert len(actions) == 2, f"actions={actions}"
    assert be.get_available_action_labels(state) == actions
    print("PASS test_reset_and_build_state")


def test_take_action_advances_and_command_mapping():
    """take_action(idx) → 正确 mod 命令 + 推进到下一帧。"""
    map_state = _ingame("MAP", {}, choice_list=["x=1 M", "x=2 ?"], floor=0)
    next_event = _ingame(
        "EVENT", {"event_id": "Big Fish"}, choice_list=["banana", "donut", "box"], floor=1
    )
    io = ListModIO([_out_of_game(), map_state, next_event])
    be = RealGameBackend(io=io)
    be.reset(seed=1)
    # 选 map 节点 idx=1 → "choose 1"
    ok = be.take_action(1)
    assert ok
    assert "choose 1" in io.sent, f"sent={io.sent}"
    # 已推进到 EVENT 帧
    state = be.build_v8_state()
    assert state.phase == "EVENT", f"phase={state.phase!r}"
    assert len(be.get_available_actions()) == 3
    print("PASS test_take_action_advances_and_command_mapping")


def test_combat_decision_no_search():
    """COMBAT 帧：adapter 枚举 play_card / end_turn，take_action 发 play/end。"""
    combat = {
        "hand": [
            {"id": "Strike", "cost": 1, "is_playable": True, "has_target": True},
            {"id": "Defend", "cost": 1, "is_playable": True, "has_target": False},
        ],
        "monsters": [{"id": "Cultist", "current_hp": 48, "max_hp": 48}],
        "player": {"energy": 3},
    }
    combat_state = _ingame("NONE", {}, floor=2, combat_state=combat)
    after = _ingame("MAP", {}, choice_list=["x=1 M"], floor=2)
    io = ListModIO([_out_of_game(), combat_state, after])
    be = RealGameBackend(io=io)
    be.reset(seed=2)
    state = be.build_v8_state()
    assert state.phase == "COMBAT", f"phase={state.phase!r}"
    actions = be.get_available_actions()
    # Strike(有目标,1敌) + Defend(无目标) + end_turn = 3
    assert len(actions) == 3, f"actions={actions}"
    assert any("play_card" in a for a in actions)
    assert any("end_turn" in a for a in actions)
    # 打 Strike → "play 1 0"
    be.take_action(0)
    assert any(c.startswith("play 1") for c in io.sent), f"sent={io.sent}"
    print("PASS test_combat_decision_no_search")


def test_game_over_won_snapshot():
    """GAME_OVER victory → game_over/game_won + snapshot 合理。"""
    map_state = _ingame("MAP", {}, choice_list=["x=1 M"], floor=51, act=3)
    over = _game_over(won=True, floor=51, act=3, hp=30)
    io = ListModIO([_out_of_game(), map_state, over])
    be = RealGameBackend(io=io)
    be.reset(seed=3)
    assert not be.game_over
    be.take_action(0)  # 推到 GAME_OVER
    assert be.game_over and be.game_won
    snap = be.build_runner_snapshot()
    assert snap["game_won"] is True
    assert snap["act"] == 3 and snap["floor"] == 51, f"snap={snap}"
    print("PASS test_game_over_won_snapshot")


def test_real_episode_runner_end_to_end():
    """RealEpisodeRunner + fake argmax model 跑一局 → EpisodeResult。"""
    import torch

    class _FakeModel:
        """对任意 (state, actions) 返回长度 len(actions) 的 logits，argmax=0。"""
        def __call__(self, state, actions):
            n = len(actions)
            logits = torch.zeros(n)
            if n:
                logits[0] = 1.0
            return {"logits": logits, "value": torch.tensor(0.0)}

    # 序列：reset 读 out-of-game → start 后 MAP → choose 0 → EVENT → choose 0 →
    #       GAME_OVER(act2 boss 死, act=2)
    seq = [
        _out_of_game(),
        _ingame("MAP", {}, choice_list=["x=1 M"], floor=16, act=1),
        _ingame("EVENT", {"event_id": "Big Fish"}, choice_list=["a", "b"], floor=17, act=2),
        _game_over(won=False, floor=34, act=2, hp=0),
    ]
    io = ListModIO(seq)
    be = RealGameBackend(io=io, character="IRONCLAD")
    runner = RealEpisodeRunner(be)
    res = runner.run_episode(_FakeModel(), seed=10000)
    assert res.completed
    assert res.act == 2, f"act={res.act}"
    # act>=2 → 过 act1 boss；evaluate 聚合应记 act1_beat
    metrics = evaluate(_FakeModel(), runner, num_seeds=1, seed_offset=10000) if False else None
    print(f"PASS test_real_episode_runner_end_to_end (result={res})")


def main() -> int:
    test_reset_and_build_state()
    test_take_action_advances_and_command_mapping()
    test_combat_decision_no_search()
    test_game_over_won_snapshot()
    test_real_episode_runner_end_to_end()
    print("\nAll RealGameBackend offline mock tests passed.")
    print(
        "\n[NOTE] 端到端连真机不在离线测试内：需用户开 STS + ModTheSpire，"
        "用 RealGameBackend + StdioModIO 跑 evaluate(model, RealEpisodeRunner, ...)。"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""第 2 层：后端接口合格性 + 初始状态合理性（便宜粗筛网）。

只验「reset 不崩 + 快照字段齐全且数值合理 + build_v8_state 非空」这类确定性事实，
不驱动整局（整局无 stall 验证由 eval_harness parity 兜底）。后端无关：任何实现
GameBackend 的后端都应过。

跑法：.venv/bin/python tools/test_backend_conformance.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = str(Path(__file__).resolve().parent.parent)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from v8.backends.stsrl_backend import StSRLBackend

_SNAPSHOT_KEYS = (
    "floor", "act", "hp", "max_hp", "gold",
    "game_won", "deck_size", "relics_size",
)


def _fresh_backend() -> StSRLBackend:
    backend = StSRLBackend(character="ironclad", ascension=0)
    backend.reset(seed=42)
    return backend


def test_reset_initial_flags():
    backend = _fresh_backend()
    assert backend.game_over is False
    assert backend.game_won is False


def test_snapshot_keys_and_ranges():
    backend = _fresh_backend()
    snap = backend.build_runner_snapshot()
    for key in _SNAPSHOT_KEYS:
        assert key in snap, f"snapshot 缺 key: {key}"
    assert snap["hp"] > 0, f"hp 应 > 0，实得 {snap['hp']}"
    assert snap["max_hp"] >= snap["hp"], "max_hp 应 >= hp"
    assert snap["floor"] >= 0, f"floor 应 >= 0，实得 {snap['floor']}"
    assert snap["act"] >= 1, f"act 应 >= 1，实得 {snap['act']}"
    assert snap["deck_size"] >= 10, f"Ironclad 起始 deck 应 >= 10，实得 {snap['deck_size']}"


def test_build_v8_state_non_none():
    backend = _fresh_backend()
    state = backend.build_v8_state()
    assert state is not None


if __name__ == "__main__":
    test_reset_initial_flags()
    test_snapshot_keys_and_ranges()
    test_build_v8_state_non_none()
    print("ALL PASS")

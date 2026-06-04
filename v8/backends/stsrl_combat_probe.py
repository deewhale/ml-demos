"""StSRLCombatProbe：用 StSRLSolver 战斗引擎实现 CombatProbe。

封装 packages.engine.combat_engine 的 create_combat_from_enemies + play_card，
把结果转成中性 CardPlayResult。坑点（2026-06-04 探针实测）：
  - create_combat_from_enemies 收 Enemy 对象（create_enemy 造），非 EnemyCombatState
  - 必须先 start_combat()；它会随机抽牌，故之后直接覆写 state.hand 精确控制手牌
  - 能量在 state.energy，不在 player 上
  - 状态层数键名是 STS 显示名（"Vulnerable" / "Strength"；虚弱是 "Weakened"）
"""
from __future__ import annotations

from typing import Any, Dict

from sts_paths import ensure_on_sys_path

ensure_on_sys_path()

from packages.engine.combat_engine import create_combat_from_enemies  # noqa: E402
from packages.engine.content.enemies import create_enemy  # noqa: E402
from packages.engine.state.rng import Random as GameRandom  # noqa: E402

from v8.backends.combat_probe import CardPlayResult, CombatSetup  # noqa: E402

_ENEMY_SEED = 12345


class StSRLCombatProbe:
    """实现 CombatProbe（鸭子类型，无需显式继承 Protocol）。"""

    def play_single_card(
        self,
        setup: CombatSetup,
        hand_index: int,
        target_index: int,
    ) -> CardPlayResult:
        engine = self._build_engine(setup)
        read_idx = target_index if target_index >= 0 else 0
        enemy_hp_before = engine.state.enemies[read_idx].hp
        block_before = engine.state.player.block
        energy_before = engine.state.energy

        result: Dict[str, Any] = engine.play_card(hand_index, target_index)

        enemy_after = engine.state.enemies[read_idx]
        return CardPlayResult(
            success=bool(result.get("success", False)),
            enemy_hp_delta=enemy_hp_before - enemy_after.hp,
            player_block_delta=engine.state.player.block - block_before,
            energy_delta=energy_before - engine.state.energy,
            enemy_statuses=dict(enemy_after.statuses),
            player_statuses=dict(engine.state.player.statuses),
            effects=list(result.get("effects", []) or []),
        )

    @staticmethod
    def _build_engine(setup: CombatSetup):
        enemy = create_enemy(
            setup.enemy_id,
            GameRandom(_ENEMY_SEED),
            ascension=0,
            hp_rng=GameRandom(_ENEMY_SEED),
        )
        enemy.state.current_hp = setup.enemy_hp
        enemy.state.max_hp = setup.enemy_hp
        engine = create_combat_from_enemies(
            enemies=[enemy],
            player_hp=setup.player_hp,
            player_max_hp=setup.player_hp,
            deck=list(setup.hand),
            energy=setup.player_energy,
            relics=list(setup.relics),
        )
        engine.start_combat()
        engine.state.hand = list(setup.hand)
        return engine

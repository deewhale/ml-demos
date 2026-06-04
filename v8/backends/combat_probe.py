"""战斗探针能力接口（CombatProbe）。

正交于 GameBackend：GameBackend 是 env 驱动整局的契约，CombatProbe 是「构造受控
战斗、打单卡、读中性结果」的测试能力。只有模拟器后端（如 StSRLSolver）能实现它——
实机做不了受控单卡战斗。返回结构全为中性 Python 类型，不漏引擎对象。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Protocol, Tuple, runtime_checkable


@dataclass(frozen=True)
class CombatSetup:
    """受控战斗的初始条件。"""

    hand: Tuple[str, ...]            # 手牌卡 id（按顺序）
    enemy_id: str = "JawWorm"
    enemy_hp: int = 100
    player_hp: int = 80
    player_energy: int = 3
    relics: Tuple[str, ...] = ()


@dataclass(frozen=True)
class CardPlayResult:
    """打出单卡后的中性结果（不含任何引擎对象）。"""

    success: bool
    enemy_hp_delta: int              # 目标敌人掉血（前 - 后）
    player_block_delta: int          # 玩家格挡变化（后 - 前）
    energy_delta: int                # 消耗能量（前 - 后）
    enemy_statuses: Dict[str, int]   # 目标敌人出牌后的状态层数
    player_statuses: Dict[str, int]  # 玩家出牌后的状态层数
    effects: List[Dict[str, Any]]    # play_card 返回的结构化 effects（纯 dict）


@runtime_checkable
class CombatProbe(Protocol):
    """构造受控战斗 + 打单卡 + 读中性结果的能力接口。"""

    def play_single_card(
        self,
        setup: CombatSetup,
        hand_index: int,
        target_index: int,
    ) -> CardPlayResult:
        """在 setup 描述的受控战斗里打出 hand_index 处的卡（target_index<0 = 自我），
        返回中性 CardPlayResult。"""
        ...

"""
V8 极简模型（smoke 阶段，不是最终架构）。

目标：把 v8_data_collector.py 写出的 JSONL 一条记录编码成固定长度向量，
跑通 shared encoder + policy/value head 的训练循环。**不是 V8 设计文档里
的 Transformer + autoregressive policy**——那是后续。

固定输入维度：256（数值特征 + 卡名 hash bucket）。
最大动作数：MAX_ACTIONS=64（不够 padding，多出来截断）。
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


STATE_DIM = 256
HASH_BUCKETS = 256
MAX_HAND = 10
MAX_ENEMIES = 3
MAX_ACTIONS = 64


def _hash_bucket(s: str, n: int = HASH_BUCKETS) -> int:
    """卡名/动作字符串 → bucket 索引。稳定 hash（不依赖 PYTHONHASHSEED）。"""
    h = hashlib.md5(s.encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big") % n


def encode_state(record: Dict[str, Any]) -> torch.Tensor:
    """
    把一条 JSONL 记录编码成固定长度 STATE_DIM 向量。

    布局（256 维）:
      [0:8]    player 数值：hp_pct, block_pct, energy/10, turn/30, floor/57,
               act/3, draw_size/30, discard_size/30
      [8:14]   3 个 enemy 的 (hp_pct, block_pct)
      [14:19]  bottled_ai 教师评估 5 个标量特征：
                 battle_won, incoming_damage/30, player_hp_pct (=mirrors player.hp/max_hp),
                 dead_monsters/3, total_monster_health/200
               — 这些值由 bottled_ai ComparatorAssessment 直接给出，
               是教师对当前局面的「Ironclad 视角」总结
      [19:STATE_DIM]  剩余维度给 hand multi-hot（hash bucket 计数）
    """
    out: List[float] = []

    # ---- player 数值（8 维） ----
    p = record.get("player", {}) or {}
    max_hp = max(1, int(p.get("max_hp", 1)))
    out.append(float(p.get("hp", 0)) / max_hp)
    out.append(min(1.0, float(p.get("block", 0)) / max(1, max_hp)))
    out.append(float(p.get("energy", 0)) / 10.0)
    out.append(float(record.get("turn", 0)) / 30.0)
    out.append(float(record.get("floor", 0)) / 57.0)
    out.append(float(record.get("act", 0)) / 3.0)
    out.append(float(record.get("draw_pile_size", 0)) / 30.0)
    out.append(float(record.get("discard_pile_size", 0)) / 30.0)

    # ---- enemies（最多 3 个，每个 hp_pct + block）（6 维） ----
    enemies = record.get("enemies", []) or []
    for i in range(MAX_ENEMIES):
        if i < len(enemies):
            e = enemies[i]
            e_max = max(1, int(e.get("max_hp", 1)))
            out.append(float(e.get("hp", 0)) / e_max)
            out.append(min(1.0, float(e.get("block", 0)) / 50.0))
        else:
            out.append(0.0)
            out.append(0.0)

    # ---- bottled_ai 教师评估特征（5 维） ----
    # 字段值由 v8_data_collector → ComparatorAssessment 写出。
    # 缺失（采集阶段还没接通）时整段 0，模型仍可训练但失去教师信号。
    ba = record.get("bottled_ai_assessment") or {}
    ba_battle_won = 1.0 if ba.get("battle_won") else 0.0
    ba_incoming = float(ba.get("incoming_damage", 0) or 0) / 30.0
    # player_hp_pct 重复 player.hp/max_hp 但来自 bottled_ai 的口径，留作冗余信号
    ba_player_hp = float(p.get("hp", 0)) / max_hp
    ba_dead_mon = float(ba.get("dead_monsters", 0) or 0) / 3.0
    ba_tmh = float(ba.get("total_monster_health", 0) or 0) / 200.0
    out.extend([ba_battle_won, ba_incoming, ba_player_hp, ba_dead_mon, ba_tmh])

    # 当前累计 8 + 6 + 5 = 19 维。剩余给 hand multi-hot。
    hand_buckets = STATE_DIM - len(out)
    hand_vec = [0.0] * hand_buckets
    for card_id in (record.get("hand") or [])[:MAX_HAND]:
        idx = _hash_bucket(str(card_id), n=hand_buckets)
        hand_vec[idx] += 1.0
    out.extend(hand_vec)

    # 截断/补齐
    if len(out) < STATE_DIM:
        out.extend([0.0] * (STATE_DIM - len(out)))
    else:
        out = out[:STATE_DIM]

    return torch.tensor(out, dtype=torch.float32)


def encode_action(action_str: str, available_actions: List[str]) -> int:
    """把 chosen_action 字符串映射到 available_actions 里的 index。"""
    if not available_actions:
        return 0
    try:
        return available_actions.index(action_str)
    except ValueError:
        # 万一对不上（不太可能）→ 0
        return 0


class V8SmokeModel(nn.Module):
    """极简 shared encoder + policy/value head。总参数 < 100k。"""

    def __init__(self, state_dim: int = STATE_DIM, hidden: int = 128,
                 max_actions: int = MAX_ACTIONS):
        super().__init__()
        self.max_actions = max_actions
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden, max_actions)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        state: [B, STATE_DIM]
        returns: (policy_logits [B, MAX_ACTIONS], value [B])
        """
        h = self.encoder(state)
        logits = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)
        return logits, value


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


__all__ = [
    "STATE_DIM",
    "MAX_ACTIONS",
    "encode_state",
    "encode_action",
    "V8SmokeModel",
    "count_parameters",
]

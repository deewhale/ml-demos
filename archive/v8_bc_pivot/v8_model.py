"""
V8 元决策模型（pointer-network 风格 BC 模型）。

输入：state（deck/relics/scalars/phase/potions/map_position）
      + variable-length available_actions（list of str，如 'reward[card:0]'）
输出：每个 available_action 一个 logit；softmax 后选 idx。

设计要点：
- 卡牌/遗物/动作字符串走 hash → Embedding(VOCAB, EMB_DIM) lookup
  （不维护严格词表；冲突可接受，后续优化）
- deck/relics 用 mean pool（set encoder，简单稳定）
- 标量特征 (hp/max_hp/gold/floor/act/hp_pct/...) 单独 MLP
- phase one-hot embed
- state 各部分拼接 → state MLP → state_vec (D=128)
- 每个 action 字符串 → action_vec (D=128)（hash bucket → embedding，再 MLP 一层）
- score_i = state_vec · action_vec_i / sqrt(D)
- mask：padding 位置 logit = -1e9
- 训练 loss：CE(logits, teacher_label)

总参数估计：~500K（embedding 2048×64=131K 是大头，其余 MLP 几十 K）
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---- 词表 / 维度配置 ----
VOCAB_SIZE = 2048           # hash bucket 数（卡名/遗物名/动作字符串共用）
EMB_DIM = 64                # token embedding 维度
HIDDEN_DIM = 128            # state_vec / action_vec 维度
SCALAR_DIM = 8              # 标量特征数量（见 _encode_scalars）
PHASES = [
    "NEOW", "MAP_NAVIGATION", "COMBAT_REWARDS",
    "EVENT", "SHOP", "REST", "TREASURE", "BOSS_REWARDS",
]
PHASE_TO_IDX = {p: i for i, p in enumerate(PHASES)}
NUM_PHASES = len(PHASES)

MAX_DECK = 64               # deck token 截断（很少超过）
MAX_RELICS = 32
MAX_POTIONS = 5
MAX_ACTIONS = 64            # available_actions 截断 / padding 上限


def _hash_id(s: str, n: int = VOCAB_SIZE) -> int:
    """字符串 → [0, n) bucket 索引。稳定 hash（不依赖 PYTHONHASHSEED）。"""
    h = hashlib.md5(s.encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big") % n


# ----------------------------------------------------------------------------
# 编码：record → tensors（CPU 上做，loader 里调用）
# ----------------------------------------------------------------------------

def _encode_scalars(state: Dict[str, Any], floor: int, act: int) -> List[float]:
    """8 个标量特征。"""
    max_hp = max(1, int(state.get("max_hp", 1)))
    hp = float(state.get("hp", 0))
    return [
        hp / max_hp,                                # hp_pct
        max_hp / 100.0,                             # max_hp scaled
        float(state.get("gold", 0)) / 200.0,        # gold scaled
        float(floor) / 57.0,                        # floor pct
        float(act) / 3.0,                           # act pct
        float(len(state.get("deck") or [])) / 40.0, # deck size scaled
        float(len(state.get("relics") or [])) / 25.0,
        float(len(state.get("potions") or [])) / 5.0,
    ]


def encode_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    把一条 JSONL 记录编码成训练用的 dict（CPU tensor / list）。
    返回字段：
      deck_ids:      [Td]  long
      relic_ids:     [Tr]  long
      potion_ids:    [Tp]  long
      scalars:       [SCALAR_DIM] float
      phase_idx:     scalar long
      action_ids:    [Na]  long  (Na = 实际 action 数, ≤ MAX_ACTIONS)
      label:         scalar long  (teacher_label，调用方需先过滤)
      num_actions:   int
    """
    state = record.get("state") or {}
    floor = int(record.get("floor", 0))
    act = int(record.get("act", 0))

    deck = (state.get("deck") or [])[:MAX_DECK]
    relics = (state.get("relics") or [])[:MAX_RELICS]
    potions = (state.get("potions") or [])[:MAX_POTIONS]
    avail = (record.get("available_actions") or [])[:MAX_ACTIONS]

    deck_ids = [_hash_id(str(c)) for c in deck] or [0]
    relic_ids = [_hash_id(str(r)) for r in relics] or [0]
    potion_ids = [_hash_id(str(p)) for p in potions] or [0]
    action_ids = [_hash_id(str(a)) for a in avail] or [0]

    scalars = _encode_scalars(state, floor, act)
    phase = record.get("phase") or "NEOW"
    phase_idx = PHASE_TO_IDX.get(phase, 0)

    label = record.get("teacher_label")
    label = -1 if label is None else int(label)

    return {
        "deck_ids": deck_ids,
        "relic_ids": relic_ids,
        "potion_ids": potion_ids,
        "scalars": scalars,
        "phase_idx": phase_idx,
        "action_ids": action_ids,
        "label": label,
        "num_actions": len(avail),
    }


# ----------------------------------------------------------------------------
# 模型
# ----------------------------------------------------------------------------

class SetEncoder(nn.Module):
    """deck/relic/potion set encoder：embedding + mean pool + linear。"""

    def __init__(self, vocab: int, emb_dim: int, out_dim: int):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb_dim, padding_idx=0)
        self.proj = nn.Linear(emb_dim, out_dim)

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        ids:  [B, T]   long
        mask: [B, T]   float, 1=valid, 0=pad
        return: [B, out_dim]
        """
        e = self.emb(ids)                                  # [B, T, E]
        masked = e * mask.unsqueeze(-1)
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = masked.sum(dim=1) / denom                 # [B, E]
        return self.proj(pooled)                           # [B, out_dim]


class V8MetaModel(nn.Module):
    """
    Pointer-network 风格元决策模型。

    forward 输入是 batch 化好的 padded tensors；输出 logits 经过 mask（无效 action = -1e9）。
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        emb_dim: int = EMB_DIM,
        hidden: int = HIDDEN_DIM,
    ):
        super().__init__()
        self.hidden = hidden

        # 共享 token embedding（卡/遗物/药水/action 字符串都走 hash 到同一空间，
        # 避免 4 份 embedding；下游各分支用各自 proj head 区分语义）
        self.token_emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)

        # set encoders 用 set 内 mean pool，再走各自 proj
        self.deck_proj = nn.Linear(emb_dim, hidden)
        self.relic_proj = nn.Linear(emb_dim, hidden)
        self.potion_proj = nn.Linear(emb_dim, hidden)

        self.scalar_mlp = nn.Sequential(
            nn.Linear(SCALAR_DIM, 32),
            nn.ReLU(),
            nn.Linear(32, hidden),
        )

        self.phase_emb = nn.Embedding(NUM_PHASES, hidden)

        # 5 个 hidden 拼接 → state_vec
        self.state_mlp = nn.Sequential(
            nn.Linear(hidden * 5, hidden * 2),
            nn.ReLU(),
            nn.Linear(hidden * 2, hidden),
        )

        # action token → action_vec
        self.action_mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )

    def _pool_set(self, ids: torch.Tensor, mask: torch.Tensor,
                  proj: nn.Linear) -> torch.Tensor:
        """共享 token_emb 的 set encoder：mean pool → proj。"""
        e = self.token_emb(ids)                           # [B, T, E]
        masked = e * mask.unsqueeze(-1)
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = masked.sum(dim=1) / denom
        return proj(pooled)

    def encode_state(
        self,
        deck_ids: torch.Tensor, deck_mask: torch.Tensor,
        relic_ids: torch.Tensor, relic_mask: torch.Tensor,
        potion_ids: torch.Tensor, potion_mask: torch.Tensor,
        scalars: torch.Tensor,
        phase_idx: torch.Tensor,
    ) -> torch.Tensor:
        deck_h = self._pool_set(deck_ids, deck_mask, self.deck_proj)
        relic_h = self._pool_set(relic_ids, relic_mask, self.relic_proj)
        potion_h = self._pool_set(potion_ids, potion_mask, self.potion_proj)
        scalar_h = self.scalar_mlp(scalars)
        phase_h = self.phase_emb(phase_idx)
        cat = torch.cat([deck_h, relic_h, potion_h, scalar_h, phase_h], dim=-1)
        return self.state_mlp(cat)                        # [B, H]

    def encode_actions(self, action_ids: torch.Tensor) -> torch.Tensor:
        """action_ids: [B, Na]  →  [B, Na, H]"""
        e = self.token_emb(action_ids)
        return self.action_mlp(e)

    def forward(
        self,
        deck_ids: torch.Tensor, deck_mask: torch.Tensor,
        relic_ids: torch.Tensor, relic_mask: torch.Tensor,
        potion_ids: torch.Tensor, potion_mask: torch.Tensor,
        scalars: torch.Tensor,
        phase_idx: torch.Tensor,
        action_ids: torch.Tensor, action_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        返回 logits [B, Na]；padding 位置已设为 -1e9。
        """
        state_vec = self.encode_state(
            deck_ids, deck_mask, relic_ids, relic_mask,
            potion_ids, potion_mask, scalars, phase_idx,
        )                                                 # [B, H]
        action_vecs = self.encode_actions(action_ids)     # [B, Na, H]

        scale = self.hidden ** 0.5
        logits = torch.einsum("bh,bnh->bn", state_vec, action_vecs) / scale
        logits = logits.masked_fill(action_mask < 0.5, -1e9)
        return logits


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


__all__ = [
    "VOCAB_SIZE",
    "EMB_DIM",
    "HIDDEN_DIM",
    "MAX_ACTIONS",
    "PHASES",
    "PHASE_TO_IDX",
    "encode_record",
    "V8MetaModel",
    "count_parameters",
]

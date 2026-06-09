"""V8 双 head pointer network model。

按 docs/v8_implementation_design.md 组件 3 架构：
    Input: V8State
            ↓
    [Shared State Encoder]
       - 卡 / 遗物 / 药水 token embedding + set encoder（mean-pool）
       - 数字状态（hp/max_hp/floor/act/gold/...）MLP
       - Map encoder（flat 节点编码 + 一层简单邻接消息传递）
       - 牌组强度 5 维（card_scorer 聚合：输出/防御/运转/加费/能力）直接 concat
       - Phase one-hot concat
       - 全部 concat → MLP → state_vec [B, hidden_dim]
            ↓
       ┌────┴────────────────────┐
       ↓                          ↓
    [战斗 Head]                  [元决策 Head]
    state_vec + hand × monsters   state_vec + actions tokens
    + end_turn 选项                pointer over actions
       ↓                          ↓
    战斗 logits                   元决策 logits

    [Value Head]
    state_vec → MLP → scalar

设计原则对照（docs/v8_design_principles.md）：
- 共享 encoder（用户最早说"一个 model"）
- 战斗 + 元决策共一个 model
- 不引入 v8_strategy 概念
- 卡 / 遗物 / action 都是 token embedding，不硬编码 STS 卡库

不在本文件内：
- trainer / RL loop / PPO（下个 agent）
- reward function（下个 agent）
- env wrapper（下个 agent）
- 真实 action 枚举（v8/action_space.py 已 stub，下个 agent 接 GameRunner）
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from v8.action_space import KNOWN_PHASES
from v8.card_mech import N_MECH, card_mech_vector, mech_vector_for_action
from v8.state import V8State


# 牌组强度特征归一常数：card_scorer.deck_dims 是整副牌各维的「总和」，会随
# 牌组规模 + 战斗场次增长。除以这个温和常数把单维量级压回 ~O(1)，避免某维爆掉。
# 取 10.0：一副打到 act1 后段的牌单维聚合常落在个位到十几量级，/10 后落到 ~1。
DECK_STRENGTH_NORM: float = 10.0


# ===== Hash → token id =====


def hash_to_token_id(s: str, vocab_size: int = 2048) -> int:
    """字符串 → token id（md5 取模）。

    用 hash 而不是显式词表，是为了不硬编码 STS 卡库（设计原则要求）。
    Hash 冲突在 vocab_size=2048 下对 ~200 个 STS 实体概率较低，
    冲突时 model 仍能从其他特征区分；后续如发现冲突影响可换成
    维护 string→id 映射的 vocab。
    """
    if not s:
        return 0
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16) % vocab_size


# ===== Set encoder（mean-pool）=====


class TokenSetEncoder(nn.Module):
    """通用 token set encoder：token id list → set embedding。

    流程：token_ids → embedding lookup → 加 upgrade flag（可选）→ MLP → mean-pool。
    支持变长输入 + mask。
    """

    def __init__(self, vocab_size: int, embed_dim: int, out_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # +1 维给 upgrade flag（卡组时用，其他场景设 0）
        self.proj = nn.Sequential(
            nn.Linear(embed_dim + 1, out_dim),
            nn.ReLU(),
        )
        self.out_dim = out_dim

    def forward(
        self,
        token_ids: torch.Tensor,
        upgrade_flags: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            token_ids:    [B, N] long
            upgrade_flags:[B, N] float（0/1），None 时全 0
            mask:         [B, N] float（1=有效，0=padding），None 时全 1
        Returns:
            [B, out_dim]
        """
        emb = self.embed(token_ids)  # [B, N, embed_dim]
        if upgrade_flags is None:
            upgrade_flags = torch.zeros(
                token_ids.shape, dtype=torch.float32, device=emb.device
            )
        feat = torch.cat([emb, upgrade_flags.unsqueeze(-1)], dim=-1)  # [B, N, embed+1]
        feat = self.proj(feat)  # [B, N, out_dim]

        if mask is None:
            return feat.mean(dim=1)
        # mask mean
        mask_expanded = mask.unsqueeze(-1)  # [B, N, 1]
        feat = feat * mask_expanded
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)  # [B, 1]
        return feat.sum(dim=1) / denom


# ===== Map encoder =====


class MapEncoder(nn.Module):
    """Act 地图编码器。

    设计妥协：用 flat encoding + 一层简单邻接消息传递，而不是完整 GNN。
    理由：
    - act1 共 17 层，每层最多 7 个节点，规模小，flat 也能表达
    - 一层 message passing（邻居节点 embedding 平均）足够捕获"上下层连接"
    - 完整 GNN（PyG / DGL）依赖重，model 单 Mac MPS 跑得动是首要约束
    - 后续如需更强 graph 表达可升级到 multi-layer GAT，本期不做

    实现：
    - room_type one-hot（M / E / ? / $ / R / T / B / 空 = 8 维）+ x 坐标 + floor 坐标
    - per-node 通过 MLP 升到 embed_dim
    - 一层"邻居 mean-pool"：每个节点把自己 + 出边邻居的 embedding 平均
    - 全图 mean-pool 得 graph_vec
    - 当前位置节点 embedding 单独 concat（让 model 知道"我现在在哪"）
    """

    # M=monster, E=elite, ?=event, $=shop, R=rest, T=treasure, B=boss, _=empty
    ROOM_TYPES = ["M", "E", "?", "$", "R", "T", "B", "_"]
    NUM_ROOM_TYPES = len(ROOM_TYPES)

    def __init__(self, node_embed_dim: int = 32, out_dim: int = 64, max_floors: int = 18):
        super().__init__()
        # node 输入：room_type one-hot + x 坐标（归一化）+ floor 坐标（归一化）
        node_in_dim = self.NUM_ROOM_TYPES + 2
        self.node_mlp = nn.Sequential(
            nn.Linear(node_in_dim, node_embed_dim),
            nn.ReLU(),
        )
        # 邻居消息传递后再过一个 MLP
        self.message_mlp = nn.Sequential(
            nn.Linear(node_embed_dim * 2, node_embed_dim),
            nn.ReLU(),
        )
        # graph_vec + current_pos_emb 拼起来
        self.out_proj = nn.Sequential(
            nn.Linear(node_embed_dim * 2, out_dim),
            nn.ReLU(),
        )
        self.node_embed_dim = node_embed_dim
        self.out_dim = out_dim
        self.max_floors = max_floors

    @classmethod
    def _room_type_to_idx(cls, rt: str) -> int:
        if rt in cls.ROOM_TYPES:
            return cls.ROOM_TYPES.index(rt)
        return cls.ROOM_TYPES.index("_")

    def encode_single(
        self,
        map_nodes: List[List[Dict[str, Any]]],
        current_position: Optional[Dict[str, int]],
        device: torch.device,
    ) -> torch.Tensor:
        """单 sample encode（变长图，逐个处理后再 stack 到 batch）。

        Returns:
            [out_dim]
        """
        # 1. 先 flatten 成 node list + 记录 floor/x 索引 → flat idx
        node_features: List[torch.Tensor] = []
        node_lookup: Dict[tuple, int] = {}  # (floor, x) → flat idx
        adj_list: List[List[int]] = []  # adj_list[i] = neighbor flat idx list（出边）

        for floor_idx, floor_nodes in enumerate(map_nodes or []):
            for node in floor_nodes or []:
                rt = node.get("room_type", "_")
                x = int(node.get("x", 0))
                rt_idx = self._room_type_to_idx(rt)
                rt_one_hot = [0.0] * self.NUM_ROOM_TYPES
                rt_one_hot[rt_idx] = 1.0
                # 归一化坐标
                x_norm = x / 6.0  # STS 一层最多 7 个 x（0-6）
                f_norm = floor_idx / max(1, self.max_floors - 1)
                feat = torch.tensor(
                    rt_one_hot + [x_norm, f_norm], dtype=torch.float32, device=device
                )
                flat_idx = len(node_features)
                node_features.append(feat)
                node_lookup[(floor_idx, x)] = flat_idx
                adj_list.append(node.get("edges", []) or [])

        if not node_features:
            # 空图（NEOW 等），返回 zero
            return torch.zeros(self.out_dim, device=device)

        node_in = torch.stack(node_features, dim=0)  # [N, node_in_dim]
        node_emb = self.node_mlp(node_in)  # [N, node_embed_dim]

        # 2. 邻接消息传递：每个节点取自己 + 邻居（下一层 x 值匹配）embedding 的平均
        #    再过 message_mlp（concat self + neighbor_mean）
        N = node_emb.shape[0]
        neighbor_means = torch.zeros_like(node_emb)
        for i, neighbors_x_list in enumerate(adj_list):
            # 找到 i 对应的 floor_idx
            # 反查 node_lookup
            self_floor_x = None
            for (f, x), idx in node_lookup.items():
                if idx == i:
                    self_floor_x = (f, x)
                    break
            if self_floor_x is None:
                continue
            f, _ = self_floor_x
            # 邻居在下一层
            neighbor_indices = []
            for nx in neighbors_x_list:
                key = (f + 1, int(nx))
                if key in node_lookup:
                    neighbor_indices.append(node_lookup[key])
            if neighbor_indices:
                neighbor_means[i] = node_emb[neighbor_indices].mean(dim=0)
            else:
                neighbor_means[i] = node_emb[i]  # 没邻居就用自己

        msg_in = torch.cat([node_emb, neighbor_means], dim=-1)  # [N, 2*embed]
        node_emb2 = self.message_mlp(msg_in)  # [N, node_embed_dim]

        # 3. 全图 mean-pool
        graph_vec = node_emb2.mean(dim=0)  # [node_embed_dim]

        # 4. 当前位置 embedding
        if current_position is not None:
            cf = int(current_position.get("floor", -1))
            cx = int(current_position.get("x", -1))
            key = (cf, cx)
            if key in node_lookup:
                cur_emb = node_emb2[node_lookup[key]]
            else:
                cur_emb = torch.zeros(self.node_embed_dim, device=device)
        else:
            cur_emb = torch.zeros(self.node_embed_dim, device=device)

        combined = torch.cat([graph_vec, cur_emb], dim=-1)  # [2*embed]
        out = self.out_proj(combined)  # [out_dim]
        return out

    def forward(
        self,
        batch_map_nodes: List[List[List[Dict[str, Any]]]],
        batch_current_position: List[Optional[Dict[str, int]]],
        device: torch.device,
    ) -> torch.Tensor:
        """Batch forward。

        Args:
            batch_map_nodes:        list of map_nodes（每个是 list[floor]→list[node dict]）
            batch_current_position: list of current_position dict
        Returns:
            [B, out_dim]
        """
        outs = [
            self.encode_single(mn, cp, device)
            for mn, cp in zip(batch_map_nodes, batch_current_position)
        ]
        return torch.stack(outs, dim=0)


# ===== V8 Model 主体 =====


class V8Model(nn.Module):
    """V8 双 head pointer network model。

    单 model，共享 encoder + 战斗 head + 元决策 head + value head。
    """

    NUM_PHASES = len(KNOWN_PHASES)

    def __init__(
        self,
        vocab_size: int = 2048,
        embed_dim: int = 64,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # ----- Token embedding（共享给 deck/relics/potions/hand/monsters/actions）-----
        # 用 padding_idx=0 让 padding 位置 embedding 为 0
        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # ----- Set encoder：deck / relics / potions / hand 各一个（共享 token_embed）-----
        # 给 set encoder 自己的 proj head（卡组结构 vs 遗物结构略不同）
        set_out_dim = 64
        # deck：token embed + upgrade flag(1) + 客观机制向量(N_MECH)
        # 机制向量给模型「看见」每张牌的牌型/费用/伤害/稀有度等客观事实（非优劣评价）。
        self.deck_proj = nn.Sequential(
            nn.Linear(embed_dim + 1 + N_MECH, set_out_dim), nn.ReLU()
        )
        self.relic_proj = nn.Sequential(
            nn.Linear(embed_dim, set_out_dim), nn.ReLU()
        )
        self.potion_proj = nn.Sequential(
            nn.Linear(embed_dim, set_out_dim), nn.ReLU()
        )
        self.hand_proj = nn.Sequential(
            nn.Linear(embed_dim + 2, set_out_dim), nn.ReLU()  # +1 upgrade +1 cost
        )
        # 怪物 token：name token + hp + intent_dmg + intent_hits + block
        self.monster_proj = nn.Sequential(
            nn.Linear(embed_dim + 4, set_out_dim), nn.ReLU()
        )

        # ----- Map encoder -----
        map_out_dim = 64
        self.map_encoder = MapEncoder(node_embed_dim=32, out_dim=map_out_dim)

        # ----- 数字状态 MLP -----
        # 字段：hp/max_hp、hp_ratio、floor/17、act、gold/999、energy、num_potions
        num_state_dim = 8
        self.num_mlp = nn.Sequential(
            nn.Linear(num_state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        # ----- 牌组强度 5 维（card_scorer 牌组聚合：输出/防御/运转/加费/能力）-----
        # v5 起改用 card_scorer.deck_dims（按局真实战斗量出的 5 维），替换旧的
        # 模拟评分 4 维 dict（damage_dealt/damage_taken/turns_to_win/win_rate）。
        # 让模型在元决策（选卡/路线）时「看到」自己牌组当前各维多强。
        # 新特征 → 旧 ckpt 不兼容（输入维度变 + 语义变），v5 必须 fresh start。
        self.deck_strength_dim = 5

        # ----- Phase one-hot -----
        # 用 KNOWN_PHASES 的索引

        # ----- State 总融合 MLP -----
        # 拼接：deck(64) + relic(64) + potion(64) + map(64) + num(32) + deck_strength(5) + phase(NUM_PHASES)
        # 战斗外字段（hand/monsters/energy）不进 state_vec，只参与战斗 head
        state_in_dim = (
            set_out_dim * 3  # deck + relic + potion
            + map_out_dim
            + 32  # num_mlp 输出
            + self.deck_strength_dim
            + self.NUM_PHASES
        )
        self.state_fuse = nn.Sequential(
            nn.Linear(state_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # ----- 战斗 Head -----
        # state_vec + hand_card_emb + monster_emb → score
        # 用 bilinear-style：query = MLP(state) [hidden_dim]
        #                     key   = MLP(card_emb + monster_emb) [hidden_dim]
        # score = dot(query, key)
        self.combat_query = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # 战斗 head 的 key 输入：card_token_emb(64) + upgrade(1) + cost(1) + 机制(N_MECH)
        #   + monster_token_emb(64) + monster_stats(4)
        combat_key_in = embed_dim + 2 + N_MECH + embed_dim + 4
        self.combat_key = nn.Sequential(
            nn.Linear(combat_key_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # End turn 单独 key（特殊 action，不需要 card/target）
        self.end_turn_key = nn.Parameter(torch.randn(hidden_dim) * 0.01)

        # ----- 元决策 Head -----
        # state_vec + action_token_emb → score
        self.meta_query = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # 元决策 key 输入：action_token_emb(embed_dim) + 候选卡客观机制向量(N_MECH)
        #   机制向量让选卡头「看得见牌面」（牌型/费用/伤害/稀有度等客观事实），
        #   非选卡动作（地图/营火/拿金）机制向量为全 0。
        self.meta_key = nn.Sequential(
            nn.Linear(embed_dim + N_MECH, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # ----- Value Head -----
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # ----- Boss-aware encoding（boss token → hidden_dim 残差加到 state_vec）-----
        # 不改 state_fuse 输入维度 → 旧 ckpt strict=False load 全部已有 key 不 mismatch；
        # boss_proj 在旧 ckpt 里是 missing key，random init，需要续训学习。
        # 用 shared token_embed 把 boss name hash 成 embed_dim，再投影到 hidden_dim 后
        # add 到 state_vec（residual style，初始 ~0 不破坏既有行为）。
        self.boss_proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    # ===== 工具：state → batch tensor =====

    def _device(self) -> torch.device:
        return next(self.parameters()).device

    def _build_set_input(
        self,
        token_strings: List[str],
        upgrades: Optional[List[bool]] = None,
        device: torch.device = torch.device("cpu"),
    ):
        """构造 set encoder 输入（单 sample，未 batch）。"""
        ids = [hash_to_token_id(s, self.vocab_size) for s in token_strings] or [0]
        ups = upgrades if upgrades is not None else [False] * len(ids)
        if not upgrades and len(ups) < len(ids):
            ups = [False] * len(ids)
        ids_t = torch.tensor(ids, dtype=torch.long, device=device)
        ups_t = torch.tensor(
            [1.0 if u else 0.0 for u in ups[: len(ids)]],
            dtype=torch.float32,
            device=device,
        )
        return ids_t, ups_t

    def _set_pool(
        self,
        ids: torch.Tensor,
        proj: nn.Module,
        extra_feats: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """通用 set mean-pool encode（单 sample）。

        Args:
            ids:        [N] long
            proj:       MLP（embed_dim + extra_dim → out_dim）
            extra_feats:[N, extra_dim] 或 None
        Returns:
            [out_dim]
        """
        if ids.numel() == 0:
            ids = torch.zeros(1, dtype=torch.long, device=ids.device)
        emb = self.token_embed(ids)  # [N, embed_dim]
        if extra_feats is not None:
            feat = torch.cat([emb, extra_feats], dim=-1)
        else:
            feat = emb
        out = proj(feat)  # [N, out_dim]
        return out.mean(dim=0)  # [out_dim]

    def _phase_one_hot(self, phase: str, device: torch.device) -> torch.Tensor:
        v = torch.zeros(self.NUM_PHASES, dtype=torch.float32, device=device)
        if phase in KNOWN_PHASES:
            v[KNOWN_PHASES.index(phase)] = 1.0
        return v

    # ===== 单 state encode（内部 helper）=====

    def _encode_single_state(self, state: V8State, device: torch.device) -> torch.Tensor:
        """单 sample state → state_vec [hidden_dim]。"""
        # ---- deck ----
        deck_names = [c.get("name", "") for c in (state.deck or [])]
        deck_upgrades = [bool(c.get("upgraded", False)) for c in (state.deck or [])]
        if not deck_names:
            deck_names = [""]
            deck_upgrades = [False]
        deck_ids = torch.tensor(
            [hash_to_token_id(n, self.vocab_size) for n in deck_names],
            dtype=torch.long,
            device=device,
        )
        # extra = upgrade flag(1) + 客观机制向量(N_MECH)（按 per-card upgraded 取升级态机制）
        deck_extra = torch.tensor(
            [
                [1.0 if u else 0.0] + card_mech_vector(n, u)
                for n, u in zip(deck_names, deck_upgrades)
            ],
            dtype=torch.float32,
            device=device,
        )
        deck_vec = self._set_pool(deck_ids, self.deck_proj, deck_extra)

        # ---- relics ----
        relic_names = state.relics or [""]
        relic_ids = torch.tensor(
            [hash_to_token_id(n, self.vocab_size) for n in relic_names],
            dtype=torch.long,
            device=device,
        )
        relic_vec = self._set_pool(relic_ids, self.relic_proj)

        # ---- potions ----
        potion_names = [p for p in (state.potions or []) if p] or [""]
        potion_ids = torch.tensor(
            [hash_to_token_id(n, self.vocab_size) for n in potion_names],
            dtype=torch.long,
            device=device,
        )
        potion_vec = self._set_pool(potion_ids, self.potion_proj)

        # ---- map ----
        # 单 sample 直接用 map_encoder.encode_single
        map_vec = self.map_encoder.encode_single(
            state.map_nodes or [], state.current_position, device
        )

        # ---- 数字状态 ----
        hp = float(state.hp or 0)
        max_hp = float(state.max_hp or 1)
        hp_ratio = hp / max(1.0, max_hp)
        floor_norm = float(state.floor or 0) / 17.0
        act_norm = float(state.act or 1) / 3.0
        gold_norm = float(state.gold or 0) / 999.0
        energy = float(state.energy or 0) / 5.0
        num_potions = float(len([p for p in (state.potions or []) if p])) / 5.0
        max_hp_norm = max_hp / 100.0
        num_feats = torch.tensor(
            [hp_ratio, max_hp_norm, floor_norm, act_norm, gold_norm, energy, num_potions, hp / 100.0],
            dtype=torch.float32,
            device=device,
        )
        num_vec = self.num_mlp(num_feats)

        # ---- 牌组强度（card_scorer 5 维聚合：输出/防御/运转/加费/能力）----
        # state.deck_strength 由 env 填成 card_scorer.deck_dims 的 5 维 dict
        # （已是归一锚分量纲：output/defense 已 /OUTPUT_NORM，协同三维本就归一）。
        # 牌组聚合是「总和」会随牌组变大而增长，这里再除一个温和常数
        # DECK_STRENGTH_NORM 把整副牌的量级压回 ~O(1)，避免某维爆掉淹没其他特征。
        # 未评估过（None / 开局没打过仗）→ 全 0，安全。
        ds = state.deck_strength or {}
        strength_vec = torch.tensor(
            [
                float(ds.get("output", 0.0) or 0.0) / DECK_STRENGTH_NORM,
                float(ds.get("defense", 0.0) or 0.0) / DECK_STRENGTH_NORM,
                float(ds.get("draw", 0.0) or 0.0) / DECK_STRENGTH_NORM,
                float(ds.get("energy", 0.0) or 0.0) / DECK_STRENGTH_NORM,
                float(ds.get("power", 0.0) or 0.0) / DECK_STRENGTH_NORM,
            ],
            dtype=torch.float32,
            device=device,
        )

        # ---- phase one-hot ----
        phase_vec = self._phase_one_hot(state.phase or "", device)

        # ---- 全部拼起来 ----
        fused_in = torch.cat(
            [deck_vec, relic_vec, potion_vec, map_vec, num_vec, strength_vec, phase_vec],
            dim=-1,
        )
        state_vec = self.state_fuse(fused_in)

        # ---- Boss-aware residual：boss name → embed → MLP → add 到 state_vec ----
        # boss 为空（NEOW / 部分 reset 边界）时仍走 hash_to_token_id(""=0)，
        # token_embed[0] 因 padding_idx=0 永远为 0，boss_proj(0) 也 ~0，安全。
        boss_token_id = hash_to_token_id(state.boss or "", self.vocab_size)
        boss_id_t = torch.tensor([boss_token_id], dtype=torch.long, device=device)
        boss_emb = self.token_embed(boss_id_t).squeeze(0)  # [embed_dim]
        boss_residual = self.boss_proj(boss_emb)           # [hidden_dim]
        state_vec = state_vec + boss_residual

        return state_vec

    def encode_state(self, state) -> torch.Tensor:
        """V8State 或 list[V8State] → state_vec。

        - 单 state → [hidden_dim]
        - list of states → [B, hidden_dim]
        """
        device = self._device()
        if isinstance(state, V8State):
            return self._encode_single_state(state, device)
        if isinstance(state, list):
            outs = [self._encode_single_state(s, device) for s in state]
            return torch.stack(outs, dim=0)
        raise TypeError(f"encode_state expects V8State or list, got {type(state)}")

    # ===== 战斗 head =====

    def _combat_forward(
        self, state: V8State, state_vec: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """战斗 phase forward。

        Args:
            state:     单 V8State（必须 phase=="COMBAT"）
            state_vec: [hidden_dim]
        Returns:
            logits: [n_options]，最后一个 option 是 end_turn
        """
        hand = state.hand or []
        monsters = state.monsters or []

        # 没有手牌或没有怪物 → 只剩 end_turn
        if not hand or not monsters:
            query = self.combat_query(state_vec)  # [hidden_dim]
            end_score = (query * self.end_turn_key).sum()
            return end_score.unsqueeze(0)  # [1]

        # 构造 hand 卡 token + monster token cross product
        # hand: [H, embed_dim+2]（token + upgrade + cost）
        hand_ids = torch.tensor(
            [hash_to_token_id(c.get("name", ""), self.vocab_size) for c in hand],
            dtype=torch.long,
            device=device,
        )
        hand_emb = self.token_embed(hand_ids)  # [H, embed]
        # extra = upgrade(1) + cost(1) + 客观机制向量(N_MECH)
        hand_extra = torch.tensor(
            [
                [
                    1.0 if c.get("upgraded", False) else 0.0,
                    float(c.get("cost", 0)) / 5.0,
                ]
                + card_mech_vector(c.get("name", ""), bool(c.get("upgraded", False)))
                for c in hand
            ],
            dtype=torch.float32,
            device=device,
        )
        hand_full = torch.cat([hand_emb, hand_extra], dim=-1)  # [H, embed+2+N_MECH]

        # monsters: [M, embed_dim+4]（token + hp_ratio + intent_dmg + intent_hits + block）
        monster_ids = torch.tensor(
            [hash_to_token_id(m.get("name", ""), self.vocab_size) for m in monsters],
            dtype=torch.long,
            device=device,
        )
        monster_emb = self.token_embed(monster_ids)  # [M, embed]
        monster_extra = torch.tensor(
            [
                [
                    float(m.get("hp", 0)) / max(1.0, float(m.get("max_hp", 1))),
                    float(m.get("intent_dmg", 0)) / 30.0,
                    float(m.get("intent_hits", 0)) / 5.0,
                    float(m.get("block", 0)) / 30.0,
                ]
                for m in monsters
            ],
            dtype=torch.float32,
            device=device,
        )
        monster_full = torch.cat([monster_emb, monster_extra], dim=-1)  # [M, embed+4]

        # Cross product: [H, M, embed+2 + embed+4]
        H = hand_full.shape[0]
        M = monster_full.shape[0]
        h_exp = hand_full.unsqueeze(1).expand(H, M, -1)  # [H, M, embed+2]
        m_exp = monster_full.unsqueeze(0).expand(H, M, -1)  # [H, M, embed+4]
        pairs = torch.cat([h_exp, m_exp], dim=-1)  # [H, M, combat_key_in]
        pairs_flat = pairs.reshape(H * M, -1)
        keys = self.combat_key(pairs_flat)  # [H*M, hidden_dim]

        query = self.combat_query(state_vec)  # [hidden_dim]
        scores = (keys * query.unsqueeze(0)).sum(dim=-1)  # [H*M]

        # End turn score
        end_score = (query * self.end_turn_key).sum().unsqueeze(0)  # [1]

        logits = torch.cat([scores, end_score], dim=0)  # [H*M + 1]
        return logits

    # ===== 元决策 head =====

    def _meta_forward(
        self,
        state_vec: torch.Tensor,
        available_actions: List[str],
        device: torch.device,
    ) -> torch.Tensor:
        """元决策 phase forward。

        Args:
            state_vec:         [hidden_dim]
            available_actions: list of str
        Returns:
            logits: [n_options]
        """
        if not available_actions:
            # 没有可选 → 返回单位 logit（防 zero-len，由 env 兜底）
            return torch.zeros(1, device=device)

        action_ids = torch.tensor(
            [hash_to_token_id(a, self.vocab_size) for a in available_actions],
            dtype=torch.long,
            device=device,
        )
        action_emb = self.token_embed(action_ids)  # [A, embed]
        # 候选卡客观机制向量：选卡类标签抠卡名查表，非选卡动作为全 0。
        action_mech = torch.tensor(
            [mech_vector_for_action(a) for a in available_actions],
            dtype=torch.float32,
            device=device,
        )  # [A, N_MECH]
        action_full = torch.cat([action_emb, action_mech], dim=-1)  # [A, embed+N_MECH]
        keys = self.meta_key(action_full)  # [A, hidden_dim]

        query = self.meta_query(state_vec)  # [hidden_dim]
        scores = (keys * query.unsqueeze(0)).sum(dim=-1)  # [A]
        return scores

    # ===== 主 forward =====

    def forward(
        self,
        state: V8State,
        available_actions: List[str],
    ) -> Dict[str, torch.Tensor]:
        """单 state forward（训练 batch 由 trainer 自己循环或自定义 collate）。

        根据 state.phase 走战斗 head 或元决策 head。

        Returns:
            {
                "logits": [n_options]，
                "value":  [] 标量 tensor（state_vec → MLP → scalar）
            }
        """
        device = self._device()

        state_vec = self._encode_single_state(state, device)  # [hidden_dim]

        if state.phase == "COMBAT":
            logits = self._combat_forward(state, state_vec, device)
        else:
            logits = self._meta_forward(state_vec, available_actions, device)

        value = self.value_head(state_vec).squeeze(-1)  # scalar

        return {"logits": logits, "value": value}


__all__ = [
    "V8Model",
    "MapEncoder",
    "TokenSetEncoder",
    "hash_to_token_id",
]

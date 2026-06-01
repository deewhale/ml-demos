"""V8 State 数据结构。

按 docs/v8_implementation_design.md 组件 1（State Encoder）字段清单定义。

设计原则对照（docs/v8_design_principles.md）：
- 包含牌组强度 4 维（用户原话第 3 点核心）
- 完整 act 地图（路线评估需要 long-term planning，看完整可达图）
- 战斗内字段仅 COMBAT phase 有意义；元决策 phase 留空，避免假数据
- 信息对齐玩家：? 节点未访问只暴露 room type，不暴露具体事件名

不在本文件内：
- state → tensor 编码（那是 model 文件的事）
- v8_strategy 启发式（已 archive，禁止复活）
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class V8State:
    """V8 model 的 observation 数据结构。

    字段分 5 大类：数字状态 / 牌组 / 遗物 / 地图 / 战斗内。
    战斗内字段仅 phase == "COMBAT" 时填充，其他 phase 保持默认空值。

    序列化：用 to_dict / from_dict 可与 JSON / replay buffer 互转，
    供后续 RL trainer / 离线分析使用。
    """

    # ----- 数字状态 -----
    hp: int = 0
    max_hp: int = 0
    floor: int = 0
    act: int = 1
    gold: int = 0
    # 药水栏：list of 药水 ID 字符串（包含空槽用 "" 占位，与 StSRLSolver 一致）
    potions: List[str] = field(default_factory=list)

    # ----- 牌组（完整 deck，含升级状态）-----
    # 每张卡：{"name": str, "upgraded": bool}
    deck: List[Dict[str, Any]] = field(default_factory=list)

    # ----- 遗物 -----
    relics: List[str] = field(default_factory=list)

    # ----- 完整 act 地图 -----
    # map_nodes[floor_idx] = list of node dicts
    # 每个 node: {"room_type": str, "x": int, "edges": [int]}（edges 是下一层连接的 x list）
    # act1 共 17 层（含 boss），act 切换时整张图重置
    map_nodes: List[List[Dict[str, Any]]] = field(default_factory=list)
    # 当前位置 {"floor": int, "x": int}，未上图（NEOW）时为 None
    current_position: Optional[Dict[str, int]] = None

    # ----- 牌组强度（card_scorer 牌组聚合，v5 起）-----
    # 5 维数字：{"output": float, "defense": float, "draw": float,
    #          "energy": float, "power": float}
    # 由 env 在 reset/step 中从 CardScorer.deck_dims(deck) 填（按局真实战斗量出的
    # 各维牌组聚合）。让模型在元决策（选卡/路线）时看到自己牌组当前各维多强。
    # 未评估过（开局没打过仗）时为 None（model encoder 内有 None → 全 0 兜底）。
    # 旧语义（模拟评分 4 维 damage_dealt/damage_taken/turns_to_win/win_rate）已弃用。
    deck_strength: Optional[Dict[str, float]] = None

    # ----- 战斗内字段（仅 COMBAT phase 有意义）-----
    in_combat: bool = False
    # 每张卡：{"name": str, "upgraded": bool, "cost": int}
    hand: List[Dict[str, Any]] = field(default_factory=list)
    draw_pile: List[Dict[str, Any]] = field(default_factory=list)
    discard_pile: List[Dict[str, Any]] = field(default_factory=list)
    exhaust_pile: List[Dict[str, Any]] = field(default_factory=list)
    energy: int = 0
    # 每个敌人：
    #   {"hp": int, "max_hp": int,
    #    "intent_dmg": int, "intent_hits": int,
    #    "block": int, "powers": dict[str, int]}
    monsters: List[Dict[str, Any]] = field(default_factory=list)

    # ----- 当前 phase -----
    # 取值: NEOW / MAP / COMBAT / EVENT / SHOP / REST / TREASURE
    #       CARD_REWARDS / BOSS_REWARDS
    phase: str = ""

    # ----- 当前 act 的 boss 名（boss-aware encoding）-----
    # 取值: "Slime Boss" / "Hexaghost" / "The Guardian" / "Automaton" / ...
    # GameRunner reset 时即已确定（runner._boss_name），act 切换后会被覆盖成新 act 的 boss。
    # 进入 act1 boss 战之前 model 看得到这个字段，决策 boss-specific route / 选卡。
    # 旧 ckpt 兼容：token_embed 共享，新 token 走 hash → embed lookup，random init。
    boss: str = ""

    # ===== 序列化 =====

    def to_dict(self) -> Dict[str, Any]:
        """序列化为 plain dict（可 JSON dump）。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "V8State":
        """从 plain dict 反序列化。

        允许部分字段缺失（用 dataclass 默认值兜底），方便老数据兼容。
        """
        # 只取本类已声明的字段，多余字段静默丢弃
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**kwargs)


__all__ = ["V8State"]

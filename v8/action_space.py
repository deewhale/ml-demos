"""V8 Action Space schema。

按 docs/v8_implementation_design.md 组件 2（Action Space）设计：
不同 phase 不同 action 集，统一用 pointer network 架构（每 phase 输入
state + available_actions list，输出选择 idx）。

支持的 phase：
    NEOW           - 4 个 blessing 选 1
    MAP            - 下层可达节点选 1
    COMBAT         - 手牌 × target 或 end_turn（每回合多步）
    EVENT          - 事件选项中选 1
    SHOP           - 买卡 / 买药 / 删卡 / 买 relic / leave
    REST           - rest / smith
    TREASURE       - 开 / 不开
    CARD_REWARDS   - 3 张候选 + skip
    BOSS_REWARDS   - boss 后奖励选项

设计原则对照（docs/v8_design_principles.md）：
- 卡 / 节点 / 事件选项用 token embedding（不硬编码 STS 卡库枚举）
- 不复用 v8_strategy 启发式映射
- 不合法 action 由 environment 在 mask 阶段过滤，model 不会看到

不在本文件内：
- action 执行（GameRunner.take_action 已有，下个 agent 在 env 里串起来）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional


if TYPE_CHECKING:
    # 避免运行时循环 import（state 和 action_space 互引用就麻烦）
    from v8.state import V8State


# 已知 phase 名（V8 内部，与 docs/v8_implementation_design.md 一致）
KNOWN_PHASES = (
    "NEOW",
    "MAP",
    "COMBAT",
    "EVENT",
    "SHOP",
    "REST",
    "TREASURE",
    "CARD_REWARDS",
    "BOSS_REWARDS",
)


# V8 phase 名 → StSRLSolver GamePhase 名 映射
# StSRLSolver 用 MAP_NAVIGATION / COMBAT_REWARDS，V8 简化为 MAP / CARD_REWARDS。
_V8_TO_ENGINE_PHASE = {
    "MAP": "MAP_NAVIGATION",
    "CARD_REWARDS": "COMBAT_REWARDS",
}
_ENGINE_TO_V8_PHASE = {v: k for k, v in _V8_TO_ENGINE_PHASE.items()}


@dataclass
class V8Action:
    """每个 phase 的 action 表示。

    字段：
        phase:        发生在哪个 phase（KNOWN_PHASES 之一）
        chosen_idx:   在 available_actions 中选的 idx（pointer network 输出）
        action_str:   对应的具体 action 字符串描述（debug / log 用，不参与训练）
    """

    phase: str
    chosen_idx: int
    action_str: str


# =========================================================================
# action 描述构造（GameAction → str）
# =========================================================================

def _format_engine_action(action: Any) -> str:
    """把 StSRLSolver GameAction 对象转成可读字符串。

    各 action 类型来自 packages/engine/game.py：
        PathAction(node_index)                 → "MAP:node=<i>"
        NeowAction(choice_index)               → "NEOW:choice=<i>"
        CombatAction(action_type, ...)         → "COMBAT:<type>(card=<i>,target=<j>,potion=<k>)"
        RewardAction(reward_type, choice_index)→ "REWARD:<type>:choice=<i>"
        EventAction(choice_index)              → "EVENT:choice=<i>"
        ShopAction(action_type, item_index)    → "SHOP:<type>:item=<i>"
        RestAction(action_type, card_index)    → "REST:<type>(card=<i>)"
        TreasureAction(action_type)            → "TREASURE:<type>"
        BossRewardAction(relic_index)          → "BOSS:relic=<i>"

    用 duck typing（不 import GameAction 类型）以避免硬依赖 sys.path。
    """
    cls = type(action).__name__

    # 拿一组常见字段
    def g(name: str, default: Any = None) -> Any:
        return getattr(action, name, default)

    if cls == "PathAction":
        return f"MAP:node={g('node_index', -1)}"
    if cls == "NeowAction":
        return f"NEOW:choice={g('choice_index', -1)}"
    if cls == "CombatAction":
        atype = g("action_type", "?")
        return (
            f"COMBAT:{atype}(card={g('card_idx', -1)},"
            f"target={g('target_idx', -1)},potion={g('potion_idx', -1)})"
        )
    if cls == "RewardAction":
        return f"REWARD:{g('reward_type', '?')}:choice={g('choice_index', -1)}"
    if cls == "EventAction":
        return f"EVENT:choice={g('choice_index', -1)}"
    if cls == "ShopAction":
        return f"SHOP:{g('action_type', '?')}:item={g('item_index', -1)}"
    if cls == "RestAction":
        return f"REST:{g('action_type', '?')}(card={g('card_index', -1)})"
    if cls == "TreasureAction":
        return f"TREASURE:{g('action_type', '?')}"
    if cls == "BossRewardAction":
        return f"BOSS:relic={g('relic_index', -1)}"

    # fallback：直接 repr（保证不抛异常）
    try:
        return f"{cls}:{action!r}"
    except Exception:  # noqa: BLE001
        return cls


# =========================================================================
# state-only fallback（runner 不在时的占位枚举）
# =========================================================================

def _fallback_actions_from_state(state: "V8State") -> List[str]:
    """无 runner 时根据 V8State 字段构造合理 action list 占位。

    用于：单元测试、模型 forward 不依赖真实 runner、独立模式调试。
    """
    phase = state.phase or ""

    if phase == "NEOW":
        # Neow 总是 4 个 blessing 选项
        return [f"NEOW:choice={i}" for i in range(4)]

    if phase == "MAP":
        # state.map_nodes[next_floor] 中可达节点
        cur = state.current_position or {"floor": -1, "x": -1}
        cur_floor = int(cur.get("floor", -1))
        next_floor = cur_floor + 1
        if 0 <= next_floor < len(state.map_nodes):
            next_layer = state.map_nodes[next_floor]
            # cur_floor < 0 表示还没上图（NEOW 后第一步）：所有 next_floor 节点可走
            if cur_floor < 0:
                reachable = list(range(len(next_layer)))
            else:
                # 找当前层节点的 edges
                cur_x = int(cur.get("x", -1))
                cur_layer = state.map_nodes[cur_floor] if 0 <= cur_floor < len(state.map_nodes) else []
                reachable_x: List[int] = []
                for node in cur_layer:
                    if int(node.get("x", -1)) == cur_x:
                        reachable_x = list(node.get("edges", []) or [])
                        break
                # x → idx in next_layer
                reachable = [
                    i for i, n in enumerate(next_layer) if int(n.get("x", -1)) in reachable_x
                ]
            return [f"MAP:node={i}" for i in reachable] or ["MAP:node=0"]
        return ["MAP:node=0"]

    if phase == "CARD_REWARDS":
        # 默认 3 张候选 + skip
        return [
            "REWARD:card:choice=0",
            "REWARD:card:choice=1",
            "REWARD:card:choice=2",
            "REWARD:skip:choice=0",
        ]

    if phase == "BOSS_REWARDS":
        # 默认 3 个 boss relic 候选
        return [f"BOSS:relic={i}" for i in range(3)]

    if phase == "TREASURE":
        return ["TREASURE:take_relic", "TREASURE:skip"]

    if phase == "REST":
        # 至少 rest / smith；smith 需要可升级卡（state 里有 deck 即可）
        opts = ["REST:rest(card=-1)"]
        upgradable = sum(1 for c in state.deck if not c.get("upgraded", False))
        if upgradable > 0:
            # 每张可升级卡一个 smith 选项（idx 占位）
            for i, c in enumerate(state.deck):
                if not c.get("upgraded", False):
                    opts.append(f"REST:upgrade(card={i})")
        return opts

    if phase == "SHOP":
        # 简化占位：买 / 删 / 离开
        return [
            "SHOP:buy_card:item=0",
            "SHOP:buy_relic:item=0",
            "SHOP:buy_potion:item=0",
            "SHOP:remove_card:item=0",
            "SHOP:leave:item=-1",
        ]

    if phase == "EVENT":
        # 事件选项变长，state 没字段，给 placeholder 4 个
        return [f"EVENT:choice={i}" for i in range(4)]

    if phase == "COMBAT":
        # 手牌 × target + end_turn
        opts: List[str] = []
        n_targets = max(1, len(state.monsters))
        for ci, _card in enumerate(state.hand):
            for ti in range(n_targets):
                opts.append(f"COMBAT:play_card(card={ci},target={ti},potion=-1)")
        opts.append("COMBAT:end_turn(card=-1,target=-1,potion=-1)")
        return opts

    return [f"UNKNOWN:phase={phase}"]


# =========================================================================
# 主入口：get_available_actions
# =========================================================================

def get_available_actions(
    state: "V8State",
    runner: Optional[Any] = None,
) -> List[str]:
    """根据当前 state（+ optional runner）列出所有合法 actions（字符串描述形式）。

    参数：
        state:  V8State，model 决策的输入（必给）
        runner: StSRLSolver GameRunner 对象（可选）。给了就用它枚举真实 action，
                idx 与 runner.get_available_actions() 返回 list 一一对应；
                没给就用 state 字段构造 placeholder（合理 fallback，独立模式 / 单测用）。

    返回：
        list of str，每个字符串是一个 action 的可读描述，
        idx 即 pointer network 的输出 target。

    注意：
        - 给 runner 时，idx 与 runner 内部的 GameAction list 严格对齐，
          下游 RL env 拿 idx 后可用 runner.take_action(actions[idx]) 直接执行。
        - 不给 runner 时（独立模式）只能给 placeholder，下游 env 必须自己解析
          字符串或重新映射；本函数主要服务于训练前 smoke / 单测。
    """
    # 模式 1：runner 给了 → 真实枚举
    if runner is not None:
        try:
            engine_actions = runner.get_available_actions()
        except Exception as e:  # noqa: BLE001
            # 真出 bug 不要让 caller 整个挂；fallback 到 state-only 占位
            import logging
            logging.getLogger(__name__).warning(
                "runner.get_available_actions() failed: %s, falling back to state-only",
                e,
            )
            return _fallback_actions_from_state(state)
        return [_format_engine_action(a) for a in engine_actions]

    # 模式 2：runner 不在 → 用 state 字段做合理占位
    return _fallback_actions_from_state(state)


__all__ = [
    "V8Action",
    "KNOWN_PHASES",
    "get_available_actions",
]

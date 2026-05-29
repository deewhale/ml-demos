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
from typing import TYPE_CHECKING, Any, List, Optional, Tuple


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

def _card_instance_label(card: Any) -> str:
    """`CardInstance` / `Card` → 紧凑可读名（带 + 标记升级）。

    用 id（稳定字符串）而非 display name，避免本地化 / 大小写抖动。
    """
    if card is None:
        return "?"
    cid = getattr(card, "id", None) or getattr(card, "name", None) or "?"
    upgraded = bool(getattr(card, "upgraded", False))
    # SearingBlow 用 misc_value 表示升级次数（同 CardInstance.__repr__）
    misc = int(getattr(card, "misc_value", 0) or 0)
    if cid == "SearingBlow" and misc > 0:
        return f"{cid}+{misc}"
    return f"{cid}+" if upgraded else str(cid)


def _relic_label(relic: Any) -> str:
    if relic is None:
        return "?"
    return str(getattr(relic, "id", None) or getattr(relic, "name", None) or "?")


def _potion_label(potion: Any) -> str:
    if potion is None:
        return "?"
    return str(getattr(potion, "id", None) or getattr(potion, "name", None) or "?")


def _neow_choice_content(runner: Any, choice_index: int) -> str:
    """NeowAction(choice_index) → 该 blessing 的语义文字（祝福类型 + 缺点）。

    阶段 0 修复（docs/v8_rl_fix_plan_2026-05-29.md 改动 4）：原本 NEOW token 只编
    choice 序号，4 个选项 hash 不到不同 embedding → 模型塌到 idx=0。仿照
    _reward_card_content，从 runner.neow_blessings 取 blessing 文字注入 token。

    runner.neow_blessings 是 List[NeowBlessing]（见 engine/game.py:2642，
    NeowAction(i) 的 i 即此 list 下标）。NeowBlessing 字段：
        blessing_type: NeowBlessingType 枚举（.value 是稳定字符串，如 "ten_percent_hp_bonus"）
        drawback_type: NeowDrawbackType 枚举（NONE 表示无缺点）

    用 blessing_type.value（稳定、适合 hash），有缺点时再拼 drawback_type.value。
    """
    blessings = getattr(runner, "neow_blessings", None)
    if not blessings:
        return "?"
    idx = int(choice_index)
    if idx < 0 or idx >= len(blessings):
        return "?"
    b = blessings[idx]
    bt = getattr(b, "blessing_type", None)
    # 枚举优先取 .value（稳定字符串），退化到 str()
    bt_str = str(getattr(bt, "value", None) or bt or "?")
    dt = getattr(b, "drawback_type", None)
    dt_str = str(getattr(dt, "value", None) or dt or "")
    # NONE / 空 表示无缺点，不拼进 token
    if dt_str and dt_str.lower() not in ("none", ""):
        return f"{bt_str}|drawback={dt_str}"
    return bt_str


def _reward_card_content(runner: Any, choice_index: int) -> str:
    """RewardAction(reward_type='card', choice_index=encoded) → 候选卡 id。

    encoded = card_reward_idx * 100 + card_index（见 engine/game.py 第 2709 行）。
    runner.current_rewards.card_rewards[card_reward_idx].cards[card_index] 是 Card 对象。
    """
    rewards = getattr(runner, "current_rewards", None)
    if rewards is None:
        return "?"
    card_rewards = getattr(rewards, "card_rewards", None) or []
    cr_idx = int(choice_index) // 100
    c_idx = int(choice_index) % 100
    if cr_idx < 0 or cr_idx >= len(card_rewards):
        return "?"
    cards = getattr(card_rewards[cr_idx], "cards", None) or []
    if c_idx < 0 or c_idx >= len(cards):
        return "?"
    return _card_instance_label(cards[c_idx])


def _reward_relic_content(runner: Any, choice_index: int) -> str:
    """RewardAction(reward_type='relic', choice_index=reward_index) → relic id。

    choice_index 0 = current_rewards.relic, 1 = current_rewards.second_relic。
    """
    rewards = getattr(runner, "current_rewards", None)
    if rewards is None:
        return "?"
    if int(choice_index) == 0:
        rr = getattr(rewards, "relic", None)
    elif int(choice_index) == 1:
        rr = getattr(rewards, "second_relic", None)
    else:
        return "?"
    if rr is None:
        return "?"
    return _relic_label(getattr(rr, "relic", None))


def _event_choice_content(runner: Any, choice_index: int) -> Tuple[str, str]:
    """EventAction(choice_index) → (event_id, choice_name)。

    用 event_handler.get_available_choices 查 EventChoice.index == choice_index 的项。
    """
    ev_state = getattr(runner, "current_event_state", None)
    eh = getattr(runner, "event_handler", None)
    rs = getattr(runner, "run_state", None)
    if ev_state is None or eh is None or rs is None:
        return ("?", "?")
    event_id = str(getattr(ev_state, "event_id", "?") or "?")
    try:
        choices = eh.get_available_choices(ev_state, rs)
    except Exception:  # noqa: BLE001
        return (event_id, "?")
    for ch in choices or []:
        if int(getattr(ch, "index", -1)) == int(choice_index):
            # name 是内部稳定名（更适合 hash），text 是 UI 文案
            return (event_id, str(getattr(ch, "name", None) or getattr(ch, "text", None) or "?"))
    return (event_id, "?")


def _shop_item_content(runner: Any, action_type: str, item_index: int) -> str:
    """ShopAction(action_type, item_index) → 内容标识。

    - buy_colored_card / buy_colorless_card / buy_card / buy_relic / buy_potion:
      用 slot_index 在对应列表里找。
    - remove_card: item_index 是 deck 中 card_idx，去 run_state.deck 找。
    - leave: 无内容。
    """
    if action_type == "leave":
        return ""
    shop = getattr(runner, "current_shop", None)
    rs = getattr(runner, "run_state", None)

    def _find_by_slot(items: Any, slot: int, attr_name: str) -> Optional[Any]:
        if not items:
            return None
        for it in items:
            if int(getattr(it, "slot_index", -1)) == slot:
                return getattr(it, attr_name, None)
        return None

    idx = int(item_index)
    if shop is not None:
        if action_type in ("buy_colored_card", "buy_card"):
            card = _find_by_slot(getattr(shop, "colored_cards", None), idx, "card")
            if card is None:
                # 兜底：当 buy_card 时也搜 colorless
                card = _find_by_slot(getattr(shop, "colorless_cards", None), idx, "card")
            return _card_instance_label(card) if card is not None else "?"
        if action_type == "buy_colorless_card":
            card = _find_by_slot(getattr(shop, "colorless_cards", None), idx, "card")
            return _card_instance_label(card) if card is not None else "?"
        if action_type == "buy_relic":
            relic = _find_by_slot(getattr(shop, "relics", None), idx, "relic")
            return _relic_label(relic) if relic is not None else "?"
        if action_type == "buy_potion":
            potion = _find_by_slot(getattr(shop, "potions", None), idx, "potion")
            return _potion_label(potion) if potion is not None else "?"

    if action_type == "remove_card":
        # item_index 是 deck card_idx
        deck = getattr(rs, "deck", None) if rs is not None else None
        if deck is not None and 0 <= idx < len(deck):
            return _card_instance_label(deck[idx])
        return "?"

    return "?"


def _rest_card_content(runner: Any, action_type: str, card_index: int) -> str:
    """RestAction(action_type, card_index) → 卡 id（仅 upgrade / toke 等带 card 时）。"""
    if int(card_index) < 0:
        return ""
    rs = getattr(runner, "run_state", None)
    deck = getattr(rs, "deck", None) if rs is not None else None
    if deck is None or int(card_index) >= len(deck):
        return "?"
    return _card_instance_label(deck[int(card_index)])


def _boss_relic_content(runner: Any, relic_index: int) -> str:
    """BossRewardAction(relic_index) → relic id。"""
    if int(relic_index) < 0:
        return "skip"
    rewards = getattr(runner, "current_rewards", None)
    if rewards is None:
        return "?"
    br = getattr(rewards, "boss_relics", None)
    if br is None:
        return "?"
    relics = getattr(br, "relics", None) or []
    if int(relic_index) >= len(relics):
        return "?"
    return _relic_label(relics[int(relic_index)])


def _format_engine_action(action: Any, runner: Optional[Any] = None) -> str:
    """把 StSRLSolver GameAction 对象转成可读字符串。

    给 `runner` 时会把候选内容（卡名 / event choice name / shop item id 等）注入
    token 字符串里，让同一 choice 序号但不同内容能 hash 到不同 embedding（修复 mode collapse）。

    各 action 类型来自 packages/engine/game.py（带 runner 时新增内容字段）：
        PathAction(node_index)                 → "MAP:node=<i>"
        NeowAction(choice_index)               → "NEOW:<blessing_type>[|drawback=<d>]:choice=<i>"
        CombatAction(action_type, ...)         → "COMBAT:<type>(card=<i>,target=<j>,potion=<k>)"
        RewardAction(reward_type='card', i)    → "REWARD:card:<card_id>:choice=<i>"
        RewardAction(reward_type='skip_card',i)→ "REWARD:skip_card:choice=<i>"  # 内容无关
        RewardAction(reward_type='relic', i)   → "REWARD:relic:<relic_id>:choice=<i>"
        RewardAction(其他, i)                  → "REWARD:<type>:choice=<i>"
        EventAction(choice_index)              → "EVENT:<event_id>:<choice_name>:choice=<i>"
        ShopAction(buy_xxx, item_index)        → "SHOP:<type>:<item_id>:item=<i>"
        ShopAction(remove_card, item_index)    → "SHOP:remove_card:<card_id>:item=<i>"
        ShopAction(leave, ...)                 → "SHOP:leave"
        RestAction(rest, ...)                  → "REST:rest"
        RestAction(upgrade/toke, card_index)   → "REST:<type>:<card_id>(card=<i>)"
        TreasureAction(action_type)            → "TREASURE:<type>"
        BossRewardAction(relic_index)          → "BOSS:<relic_id>:relic=<i>"

    没给 runner 时退化为旧版（只编序号），保证 fallback / 单测能跑。
    用 duck typing（不 import GameAction 类型）以避免硬依赖 sys.path。
    """
    cls = type(action).__name__

    # 拿一组常见字段
    def g(name: str, default: Any = None) -> Any:
        return getattr(action, name, default)

    if cls == "PathAction":
        return f"MAP:node={g('node_index', -1)}"
    if cls == "NeowAction":
        cidx = g("choice_index", -1)
        if runner is not None:
            content = _neow_choice_content(runner, cidx)
            return f"NEOW:{content}:choice={cidx}"
        return f"NEOW:choice={cidx}"
    if cls == "CombatAction":
        atype = g("action_type", "?")
        return (
            f"COMBAT:{atype}(card={g('card_idx', -1)},"
            f"target={g('target_idx', -1)},potion={g('potion_idx', -1)})"
        )
    if cls == "RewardAction":
        rtype = g("reward_type", "?")
        cidx = g("choice_index", -1)
        if runner is not None:
            if rtype == "card":
                content = _reward_card_content(runner, cidx)
                return f"REWARD:card:{content}:choice={cidx}"
            if rtype == "relic":
                content = _reward_relic_content(runner, cidx)
                return f"REWARD:relic:{content}:choice={cidx}"
        return f"REWARD:{rtype}:choice={cidx}"
    if cls == "EventAction":
        cidx = g("choice_index", -1)
        if runner is not None:
            event_id, choice_name = _event_choice_content(runner, cidx)
            return f"EVENT:{event_id}:{choice_name}:choice={cidx}"
        return f"EVENT:choice={cidx}"
    if cls == "ShopAction":
        atype = g("action_type", "?")
        iidx = g("item_index", -1)
        if runner is not None:
            content = _shop_item_content(runner, atype, iidx)
            if atype == "leave":
                return "SHOP:leave"
            if content:
                return f"SHOP:{atype}:{content}:item={iidx}"
        return f"SHOP:{atype}:item={iidx}"
    if cls == "RestAction":
        atype = g("action_type", "?")
        cidx = g("card_index", -1)
        if runner is not None and int(cidx) >= 0:
            content = _rest_card_content(runner, atype, cidx)
            return f"REST:{atype}:{content}(card={cidx})"
        return f"REST:{atype}(card={cidx})"
    if cls == "TreasureAction":
        return f"TREASURE:{g('action_type', '?')}"
    if cls == "BossRewardAction":
        ridx = g("relic_index", -1)
        if runner is not None:
            content = _boss_relic_content(runner, ridx)
            return f"BOSS:{content}:relic={ridx}"
        return f"BOSS:relic={ridx}"

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
        return [_format_engine_action(a, runner=runner) for a in engine_actions]

    # 模式 2：runner 不在 → 用 state 字段做合理占位
    return _fallback_actions_from_state(state)


__all__ = [
    "V8Action",
    "KNOWN_PHASES",
    "get_available_actions",
]

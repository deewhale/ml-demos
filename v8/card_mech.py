"""卡牌客观机制特征查表（card mechanics lookup）。

目的：把每张卡「玩家能看见的客观机制事实」（牌型 / 费用 / 基础伤害 / 升级 / 固有 /
稀有度）编成定长向量，喂进 V8 模型的视野——让模型不再只看到卡名的 MD5 哈希 token、
对牌面机制「半瞎」。

红线（务必遵守）：
- 这里只编**客观机制事实**，绝不编「这张牌好不好 / 稀有=好」的优劣评价。
- rarity 作为**客观枚举 one-hot** 进观测（区分 basic/common/.../curse），不是优劣信号。
- 本模块不进奖励，只作模型输入特征。

数据源：sts_lightspeed 的 `Card` 绑定（`type / cost / base_damage / upgraded / innate /
rarity`）。机制是 (card_name, upgraded) 的纯函数（确定性），故可一次性建表、按名查。

键（name → 机制向量）同时覆盖两种命名口径，保证 deck / 手牌 / 候选卡都能命中：
- 引擎 enum 名（deck 用 `_enum_name(c.id)`，如 'BASH' / 'STRIKE_RED'）；
- 引擎 display 名（CARD_REWARD 标签用 C++ getName()，如 'Bash' / 'Pommel Strike'）。
统一 normalize（小写、去空格/下划线/加号）后做键，吸收两套口径 + 升级后缀差异。

候选卡标签形如 `REWARD_card=Thunderclap` / `SHOP_buy_card=Bash($50)` /
`CARD_SELECT=Strike`，`card_name_from_label` 把卡名抠出来再查表。
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

# 机制向量维度布局（务必与下方 _build_vector 一致）：
#   type one-hot [5]      : attack / skill / power / curse / status
#   cost_norm [1]         : cost/5（X-cost 记 0）
#   is_xcost [1]          : cost==-1 → 1
#   base_damage_norm [1]  : base_damage/30（无伤害 -1 记 0）
#   has_damage [1]        : base_damage>=0 → 1
#   upgraded [1]
#   innate [1]
#   rarity one-hot [6]    : basic / common / uncommon / rare / special / curse
N_MECH: int = 16

_TYPE_ORDER = ["ATTACK", "SKILL", "POWER", "CURSE", "STATUS"]
_RARITY_ORDER = ["BASIC", "COMMON", "UNCOMMON", "RARE", "SPECIAL", "CURSE"]

_ZERO_VEC: List[float] = [0.0] * N_MECH

# (name_normalized, upgraded) → 机制向量。lazy build。
_TABLE: Optional[Dict[tuple, List[float]]] = None


def _normalize(name: str) -> str:
    """归一卡名做查表键：小写、去空格/下划线/加号/数字后缀。

    吸收三套口径差异：
    - enum 名 'STRIKE_RED' / display 名 'Strike' → 'strikered' vs 'strike'（不同，
      故建表时两种键都登记，见 _build_table）；
    - 升级后缀 'Bash+' / 'Searing Blow+2' → 去掉 '+' 和尾随数字（机制向量自带
      upgraded 维，升级与否在那维区分，名字键统一到基名即可命中）。
    """
    if not name:
        return ""
    s = name.strip().lower()
    # 去掉升级后缀：尾随的 '+' 加可选数字（Searing Blow+3）
    s = re.sub(r"\+\d*$", "", s)
    # 去空格 / 下划线
    s = s.replace(" ", "").replace("_", "")
    return s


def _build_vector(c) -> List[float]:
    """单张 Card → 机制向量（纯客观）。"""
    v = [0.0] * N_MECH
    # type one-hot
    tname = getattr(getattr(c, "type", None), "name", "")
    if tname in _TYPE_ORDER:
        v[_TYPE_ORDER.index(tname)] = 1.0
    # cost
    cost = int(getattr(c, "cost", 0))
    if cost == -1:  # X-cost 哨兵
        v[6] = 1.0  # is_xcost
    else:
        v[5] = cost / 5.0  # cost_norm
    # base_damage
    bd = int(getattr(c, "base_damage", -1))
    if bd >= 0:
        v[7] = bd / 30.0  # base_damage_norm
        v[8] = 1.0        # has_damage
    # upgraded / innate
    v[9] = 1.0 if bool(getattr(c, "upgraded", False)) else 0.0
    v[10] = 1.0 if bool(getattr(c, "innate", False)) else 0.0
    # rarity one-hot（偏移 11）
    rname = getattr(getattr(c, "rarity", None), "name", "")
    if rname in _RARITY_ORDER:
        v[11 + _RARITY_ORDER.index(rname)] = 1.0
    return v


def _build_table() -> Dict[str, List[float]]:
    """从 lightspeed 绑定一次性建 name → 机制向量表（enum 名 + display 名双键）。

    引擎不可用（绑定缺失 / 非 lightspeed 环境）时返回空表，查表全 fallback 到 0 向量——
    安全降级，不致命（其他特征仍在；机制维退化成 0 等价于「这版引擎给不出机制」）。
    """
    table: Dict[str, List[float]] = {}
    try:
        from v8.backends.lightspeed_loader import load_lightspeed  # 延迟 import

        sts = load_lightspeed()
    except Exception:
        return table

    repr_re = re.compile(r"<slaythespire\.Card (.+?)\+?\d*>")
    for enum_name in dir(sts.CardId):
        if enum_name.startswith("_") or enum_name in ("name", "value", "INVALID"):
            continue
        try:
            cid = getattr(sts.CardId, enum_name)
            # 同名建升级 / 非升级两套机制向量（base_damage / cost / upgraded 维随升级变）。
            card0 = sts.Card(cid)
            card1 = sts.Card(cid)
            try:
                card1.upgrade()
            except Exception:
                card1 = card0  # 不可升级（curse/status/已是升级态）→ 用非升级向量兜底
            vec0 = _build_vector(card0)
            vec1 = _build_vector(card1)
        except Exception:
            continue
        m = repr_re.match(repr(card0))
        disp = _normalize(m.group(1)) if m else None
        for key in (_normalize(enum_name), disp):
            if key:
                # 键 = (normalized_name, upgraded)
                table[(key, False)] = vec0
                table[(key, True)] = vec1
    return table


def _get_table() -> Dict[tuple, List[float]]:
    global _TABLE
    if _TABLE is None:
        _TABLE = _build_table()
    return _TABLE


def card_mech_vector(name: str, upgraded: bool = False) -> List[float]:
    """(卡名, 是否升级) → 机制向量 [N_MECH]。未命中（未知卡 / 引擎不可用）返回全 0 向量。

    upgraded 影响 base_damage / cost / upgraded 维（如 Strike+ base_damage 6→9）。
    候选卡标签不带升级信息时默认 upgraded=False；deck / 手牌带 per-card upgraded bool。
    """
    if not name:
        return list(_ZERO_VEC)
    key = (_normalize(name), bool(upgraded))
    return _get_table().get(key, _ZERO_VEC)


# 候选卡标签里抠卡名：
#   REWARD_card=Thunderclap / SHOP_buy_card=Bash($50) / CARD_SELECT=Strike
_LABEL_CARD_RE = re.compile(
    r"(?:REWARD_card|SHOP_buy_card|CARD_SELECT)=(.+?)(?:\(\$\d+\))?\s*$"
)


def card_name_from_label(label: str) -> Optional[str]:
    """从元决策候选标签里抠出卡名；非「带卡名的选卡类标签」返回 None。"""
    if not label:
        return None
    m = _LABEL_CARD_RE.search(label.strip())
    if m:
        return m.group(1).strip()
    return None


def mech_vector_for_action(label: str) -> List[float]:
    """候选 action 标签 → 机制向量 [N_MECH]。

    选卡类标签（REWARD_card= / SHOP_buy_card= / CARD_SELECT=）抠卡名查表；
    非选卡动作（MAP-> / CAMPFIRE_ / REWARD_gold 等）无牌面机制，返回全 0 向量。
    """
    name = card_name_from_label(label)
    if name is None:
        return list(_ZERO_VEC)
    return card_mech_vector(name)


__all__ = [
    "N_MECH",
    "card_mech_vector",
    "card_name_from_label",
    "mech_vector_for_action",
]

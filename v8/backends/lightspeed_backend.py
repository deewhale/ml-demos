"""LightspeedBackend：把 sts_lightspeed（社区金标准 C++ 模拟器）包成 GameBackend。

第二个 GameBackend 实现（StSRLBackend 之后）。用 lightspeed 绑定的整局驱动 API
（get_state / step_choice / play_battle）驱动一整局 Ironclad，把战斗当黑盒交给
引擎 MCTS（play_battle），和现有「战斗走搜索、不进 RL trajectory」架构一致。

------------------------------------------------------------------------------
当前实现范围（stage2 中性核心）
------------------------------------------------------------------------------
实现了 GameBackend 的**中性核心**（lightspeed 给得出的）：
    reset / game_over / game_won / phase / force_terminate / close /
    build_v8_state / boss_name / seed_string / seed_int / build_runner_snapshot /
    get_available_actions / take_action / get_available_action_labels /
    run_combat_turn(no-op，战斗已在 take_action 内由 play_battle 一气呵成)

**透传债桩（lightspeed 给不出引擎对象，先返回 None / 最简桩，标 TODO）**：
    run_state / current_room_type / get_current_room_type / current_combat /
    current_event_state / event_handler / current_rewards / current_shop /
    neow_blessings / last_combat_card_log

这些是 StSRLSolver 特有的 duck-typed 引擎对象，env.py 的诊断日志 / node_reward
detect / event_stall guard 直接 introspection 其内部字段（见 protocol.py 顶部说明）。
本轮**不强求 env.py 完全跑通这些分支**——返回桩，让 env.py 的 try/except 兜底；
要让 V8Env+PPO 真正在 lightspeed 上训练，需把 env.py 对这些对象的直接访问中性化
（详见模块末尾 TODO 清单 / 任务报告）。

------------------------------------------------------------------------------
lightspeed 状态 → V8 phase / 动作映射
------------------------------------------------------------------------------
lightspeed get_state(gc) 返回 ScreenState 字符串（MAP_SCREEN / REWARDS / EVENT_SCREEN
/ SHOP_ROOM / REST_ROOM / TREASURE_ROOM / BOSS_RELIC_REWARDS / CARD_SELECT / BATTLE）。
映射到 V8 phase 名（NEOW / MAP / COMBAT / CARD_REWARDS / EVENT / SHOP / REST /
TREASURE / BOSS_REWARDS / RUN_COMPLETE）。

注：lightspeed 没有独立 NEOW screen——开局第一个决策点是 EVENT_SCREEN（Neow 事件），
choices 即 4 个祝福。这里统一按 EVENT 处理（label 已是可读串），不单独造 NEOW phase。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from v8.state import V8State
from v8.backends.lightspeed_loader import load_lightspeed
from v8.card_mech import card_mech_vector


logger = logging.getLogger(__name__)


# lightspeed ScreenState 名 → V8 phase 名 映射（build_v8_state / phase 用）。
# BATTLE 在 take_action 内由 play_battle 消化，元决策点永远不会停在 BATTLE。
_SCREEN_TO_V8_PHASE: Dict[str, str] = {
    "EVENT_SCREEN": "EVENT",
    "REWARDS": "CARD_REWARDS",
    "BOSS_RELIC_REWARDS": "BOSS_REWARDS",
    "CARD_SELECT": "CARD_REWARDS",  # 删卡 / 抓牌选择，归到选卡类
    "MAP_SCREEN": "MAP",
    "TREASURE_ROOM": "TREASURE",
    "REST_ROOM": "REST",
    "SHOP_ROOM": "SHOP",
    "BATTLE": "COMBAT",
    "INVALID": "RUN_COMPLETE",
}


# lightspeed Room 名 → V8 归一 room 名（_current_room_type_str 兼容用，给特征/日志）。
_ROOM_NAME_MAP: Dict[str, str] = {
    "MONSTER": "monster",
    "ELITE": "elite",
    "BOSS": "boss",
    "BOSS_TREASURE": "treasure",
    "REST": "rest",
    "SHOP": "shop",
    "EVENT": "event",
    "TREASURE": "treasure",
    "NONE": "unknown",
    "INVALID": "unknown",
}


def _enum_name(v: Any) -> str:
    """把 pybind11 enum（如 CardId.STRIKE_RED / MonsterEncounter.HEXAGHOST）取出
    其 .name（'STRIKE_RED'）；非 enum 退回 str()。"""
    if v is None:
        return ""
    name = getattr(v, "name", None)
    if name is not None:
        return str(name)
    return str(v)


# MonsterEncounter enum 名 → STS wiki 可读 boss 名（与 StSRLSolver _boss_name 口径靠拢）。
# 给不全也无妨：模型走 hash embedding，boss-aware 编码只要字符串稳定一致即可。
_BOSS_DISPLAY: Dict[str, str] = {
    "SLIME_BOSS": "Slime Boss",
    "HEXAGHOST": "Hexaghost",
    "THE_GUARDIAN": "The Guardian",
    "GUARDIAN": "The Guardian",
    "AUTOMATON": "Automaton",
    "BRONZE_AUTOMATON": "Automaton",
    "CHAMP": "The Champ",
    "THE_CHAMP": "The Champ",
    "COLLECTOR": "The Collector",
    "THE_COLLECTOR": "The Collector",
    "TIME_EATER": "Time Eater",
    "AWAKENED_ONE": "Awakened One",
    "DONU_AND_DECA": "Donu and Deca",
    "THE_HEART": "The Heart",
}


class LightspeedBackend:
    """sts_lightspeed C++ 模拟器的 GameBackend 实现（中性核心 + 透传债桩）。

    用法（不经 V8Env，直接驱动整局）：
        be = LightspeedBackend(ascension=0)
        be.reset(seed=42)
        while not be.game_over:
            state = be.build_v8_state()
            labels = be.get_available_action_labels(state)
            be.take_action(0)   # 选第 0 个动作（战斗会在内部 play_battle 打完）
    """

    def __init__(
        self,
        *,
        character: str = "ironclad",
        ascension: int = 0,
        verbose: bool = False,
        sim_count: int = 1000,
        boss_mult: float = 3.0,
        # 按房型分级的战斗搜索预算（play_battle sim_count）。默认 None = 关闭分级，
        # 所有战斗一视同仁用 `sim_count`（保持训练现状不变）。eval 时显式传分级
        # 配置（如 {"normal":1000,"elite":4000,"boss":10000}）开高 boss/精英搜索。
        # 房型判定：play_battle 前读 get_state()['room']（lightspeed 战斗中 room enum
        # 为 MONSTER/ELITE/BOSS），映射到 normal/elite/boss 三档；读不出退回 normal。
        combat_sim_counts: Optional[Dict[str, int]] = None,
        # 兼容 StSRLBackend 构造签名（env backend_factory 会传这些，lightspeed 用不上）
        solver_budgets: Optional[Dict[str, Tuple[float, int, int]]] = None,
        combat_net_wrapper: Optional[Any] = None,
    ):
        self.character = character
        self.ascension = ascension
        self.verbose = verbose
        self._sim_count = int(sim_count)
        self._boss_mult = float(boss_mult)
        # 分级配置（None = 不分级）。补全缺省档位（缺哪档用 flat sim_count 兜底）。
        self._combat_sim_counts: Optional[Dict[str, int]] = None
        if combat_sim_counts:
            self._combat_sim_counts = {
                "normal": int(combat_sim_counts.get("normal", self._sim_count)),
                "elite": int(combat_sim_counts.get("elite", self._sim_count)),
                "boss": int(combat_sim_counts.get("boss", self._sim_count)),
            }

        self._sts = load_lightspeed()
        self._gc: Optional[Any] = None
        self._seed: Optional[int] = None
        # 实时逐步战斗 (Stage 2): 当持有真 BattleContext 逐步打时非 None；黑盒路径
        # (play_battle) 全程为 None。两条路径互斥——实时路径绕过 _advance_past_battles。
        self._live_bc: Optional[Any] = None
        # 实时战斗模式开关（Stage3+4）：开启时 take_action 落进战斗后**不**自动 play_battle
        # 打完（_advance_past_battles 跳过），把战斗留给 env 走 start_combat 逐步驱动。
        # 默认关 = 黑盒路径（take_action 内 play_battle 一气呵成），训练现状不变。
        # env 在 _live_combat_available() 为真时调 enable_live_combat(True) 打开。
        self._live_combat_mode: bool = False
        # force_terminate 标志（lightspeed 无「强制 loss」API，用本地 flag 覆盖 game_over）
        self._forced_terminal: bool = False

        # perf 计数（env 读取累加）。lightspeed 战斗走引擎 MCTS（play_battle），
        # 不经我们的 adapter.pick_action；这里统计 play_battle 调用次数当近似。
        self.combat_search_calls: int = 0

    # =====================================================================
    # 内部：lightspeed 状态读取 helper
    # =====================================================================

    def _get_state(self) -> Dict[str, Any]:
        """读 lightspeed get_state(gc) 中性 dict。无 gc 时返回空。"""
        if self._gc is None:
            return {}
        return dict(self._sts.get_state(self._gc))

    def _char_class(self) -> Any:
        """character 字符串 → lightspeed CharacterClass（目前只支持 IRONCLAD）。"""
        cc = self._sts.CharacterClass
        name = (self.character or "ironclad").upper()
        return getattr(cc, name, cc.IRONCLAD)

    def _battle_sim_count(self, room: Any) -> int:
        """按当前战斗房型选 play_battle 的 sim_count。

        分级关闭（_combat_sim_counts is None）→ 一律 flat self._sim_count（训练现状）。
        分级开启 → 读 lightspeed room enum 名（MONSTER/ELITE/BOSS），映射到
        normal/elite/boss 三档；非战斗 room 或读不出 → 退回 normal 档。
        """
        if self._combat_sim_counts is None:
            return self._sim_count
        rn = str(getattr(room, "name", room) or "").upper()
        if "BOSS" in rn:
            tier = "boss"
        elif "ELITE" in rn:
            tier = "elite"
        else:
            tier = "normal"
        return self._combat_sim_counts[tier]

    def _advance_past_battles(self) -> None:
        """若当前停在 BATTLE screen，调 play_battle 打完，推进到下一个元决策点。

        play_battle 是黑盒：打完后 gc 要么进下一个 screen（胜），要么 outcome=LOSS（死）。
        单 take_action 后理论上最多触发一场战斗，但 event→combat 链式可能多场，故 while。
        每场战斗按当前 room 选 sim_count（分级开启时 boss/精英用更高搜索预算）。
        """
        if self._gc is None:
            return
        # 实时战斗模式：不自动 play_battle 打完，把战斗留给 env 逐步驱动（start_combat）。
        if self._live_combat_mode:
            return
        guard = 0
        while (
            self._sts.get_state(self._gc).get("in_battle", False)
            and self._gc.outcome == self._sts.GameOutcome.UNDECIDED
            and guard < 64
        ):
            guard += 1
            self.combat_search_calls += 1
            room = self._sts.get_state(self._gc).get("room", None)
            sim_count = self._battle_sim_count(room)
            self._sts.play_battle(self._gc, sim_count, self._boss_mult)

    # =====================================================================
    # 1. 生命周期
    # =====================================================================
    def reset(self, seed: int) -> None:
        """开新局：建 GameContext(IRONCLAD, seed, ascension)，推进过开局战斗（若有）。

        lightspeed GameContext 构造后直接落在第一个元决策点（一般是 Neow EVENT_SCREEN）；
        若构造即落 BATTLE（罕见），_advance_past_battles 兜底打完。
        """
        self._seed = int(seed)
        self._forced_terminal = False
        self.combat_search_calls = 0
        self._gc = self._sts.GameContext(
            self._char_class(), int(seed), int(self.ascension)
        )
        self._advance_past_battles()

    @property
    def game_over(self) -> bool:
        if self._forced_terminal:
            return True
        if self._gc is None:
            return True
        return self._gc.outcome != self._sts.GameOutcome.UNDECIDED

    @property
    def game_won(self) -> bool:
        if self._forced_terminal or self._gc is None:
            return False
        return self._gc.outcome == self._sts.GameOutcome.PLAYER_VICTORY

    @property
    def phase(self) -> str:
        """当前 phase（规范字符串，与 protocol.PHASE_* / StSRLBackend 同口径）。

        stage2 中性化：两个后端 phase 都返回规范字符串，env.py 用字符串比较
        （`phase in META_PHASES` / `phase == PHASE_COMBAT` 等）。lightspeed 由
        get_state['screen'] 经 _SCREEN_TO_V8_PHASE 映射；终态返回 RUN_COMPLETE。
        """
        if self.game_over:
            return "RUN_COMPLETE"
        st = self._get_state()
        screen = st.get("screen", "INVALID")
        return _SCREEN_TO_V8_PHASE.get(screen, "MAP")

    def force_terminate(self) -> None:
        """强制把 run 标 terminal（loss）。lightspeed 无该 API，用本地 flag 覆盖
        game_over / game_won（读取侧已尊重 _forced_terminal）。"""
        self._forced_terminal = True

    def close(self) -> None:
        self._gc = None

    # =====================================================================
    # 2. 状态
    # =====================================================================
    def build_v8_state(self) -> V8State:
        """从 lightspeed gc 构造 V8State（元决策 phase 用）。

        deck / relics 从 gc.deck / gc.relics 读（id 用 enum .name 字符串，模型走 hash
        embedding 无需严格词表）。map_nodes 暂留空（lightspeed 地图需走 SpireMap API
        单独构造，本轮先不接——是特征债，不阻塞整局驱动）。
        """
        st = self._get_state()
        gc = self._gc

        hp = int(st.get("hp", 0) or 0)
        max_hp = int(st.get("max_hp", 0) or 0)
        floor = int(st.get("floor", 0) or 0)
        act = int(st.get("act", 1) or 1)
        gold = int(st.get("gold", 0) or 0)

        deck: List[Dict[str, Any]] = []
        relics: List[str] = []
        if gc is not None:
            for c in gc.deck:
                deck.append({
                    "name": _enum_name(getattr(c, "id", None)),
                    "upgraded": bool(getattr(c, "upgraded", False)),
                })
            for r in gc.relics:
                relics.append(_enum_name(getattr(r, "id", None)))

        map_x = int(st.get("map_x", -1))
        map_y = int(st.get("map_y", -1))
        current_position: Optional[Dict[str, int]] = None
        if not (map_x == -1 and map_y == -1):
            current_position = {"floor": map_y, "x": map_x}

        phase_name = st.get("screen", "INVALID")
        phase_name = _SCREEN_TO_V8_PHASE.get(phase_name, "MAP")
        if self.game_over:
            phase_name = "RUN_COMPLETE"

        return V8State(
            hp=hp,
            max_hp=max_hp,
            floor=floor,
            act=act,
            gold=gold,
            potions=[],
            deck=deck,
            relics=relics,
            map_nodes=[],  # TODO(stage2): 从 SpireMap API 构造完整 act 地图（特征债）
            current_position=current_position,
            deck_strength=None,
            in_combat=(phase_name == "COMBAT"),
            phase=phase_name,
            boss=self.boss_name,
        )

    # ---- 透传债桩：lightspeed 给不出 StSRLSolver 引擎对象，先返回 None / 最简桩 ----
    # TODO(stage2): 这些是 env.py 诊断日志 / node_reward detect / event_stall guard
    # 直接 introspection 的引擎对象。要让 V8Env 完全跑通，需把 env.py 对它们的直接
    # 访问中性化（见模块末尾 TODO 清单）。返回桩让 env.py 的 try/except 兜底。

    @property
    def run_state(self) -> Any:
        """TODO(stage2): env.py 17 处直接读 .floor/.act/.current_hp/.max_hp/.gold/
        .deck/.relics/.map_position。这是最大透传债。返回最简鸭子对象，喂关键标量字段，
        让 env.py 的诊断日志 / node_reward / final_hp_ratio 不直接崩。"""
        return _RunStateShim(self)

    @property
    def current_room_type(self) -> Any:
        """TODO(stage2): StSRLSolver combat 内才设的 room 字符串。lightspeed 用
        get_state['room']（Room enum 名）近似——给 _current_room_type_str / boss 判定用。"""
        st = self._get_state()
        room = st.get("room", "NONE")
        return _ROOM_NAME_MAP.get(str(room), "unknown")

    def get_current_room_type(self) -> Any:
        """TODO(stage2): StSRLSolver 地图 RoomType enum getter。lightspeed 无对应
        引擎枚举对象，返回 None（env._current_room_type_str 会 fallback）。"""
        return None

    @property
    def current_combat(self) -> Any:
        """战斗对象。黑盒路径（play_battle 一气呵成）下永远 None；实时逐步战斗路径
        （start_combat 持有真 BattleContext）下返回该 BattleContext，供 env / 诊断
        introspection。Stage2 加实时路径后此处随 _live_bc 透出真对象。"""
        return self._live_bc

    @property
    def current_event_state(self) -> Any:
        """TODO(stage2): 事件状态对象（event_id / phase）。lightspeed 事件是 EVENT_SCREEN
        + choices，无独立 event_state 对象，返回 None（env [event] 日志会跳过）。"""
        return None

    @property
    def event_handler(self) -> Any:
        """TODO(stage2): 事件 handler（get_available_choices）。lightspeed 无对应对象，
        返回 None（env [event] 日志 introspection 会跳过）。"""
        return None

    @property
    def current_rewards(self) -> Any:
        """TODO(stage2): 奖励对象。lightspeed 奖励走 REWARDS screen choices，无独立对象，
        返回 None。"""
        return None

    @property
    def current_shop(self) -> Any:
        """TODO(stage2): 商店对象。lightspeed 商店走 SHOP_ROOM screen choices，返回 None。"""
        return None

    @property
    def neow_blessings(self) -> Any:
        """TODO(stage2): Neow 祝福 list。lightspeed Neow 是开局 EVENT_SCREEN choices，
        无独立 list 对象，返回 None。"""
        return None

    @property
    def last_combat_card_log(self) -> Any:
        """TODO(stage2): 上场战斗逐动作细账（CombatLogEntry list，供 card_scorer 评分）。
        lightspeed 战斗黑盒不导出 per-card 细账，返回 None（card_scorer 拿到空 → 不打分，
        deck_strength 特征退化为全 0，不阻塞整局驱动）。"""
        return None

    @property
    def boss_name(self) -> str:
        """当前 act boss 名（中性化 str）。从 gc.boss（MonsterEncounter enum）映射到
        STS wiki 可读名，映射不到时退回 enum .name（仍稳定一致，模型走 hash 无妨）。"""
        if self._gc is None:
            return ""
        raw = _enum_name(getattr(self._gc, "boss", None))
        return _BOSS_DISPLAY.get(raw, raw.replace("_", " ").title() if raw else "")

    @property
    def seed_string(self) -> str:
        """seed 的字符串表示（lightspeed get_seed_str 把整数 seed 转 UI 串）。"""
        if self._seed is None:
            return ""
        try:
            return str(self._sts.get_seed_str(int(self._seed)))
        except Exception:  # noqa: BLE001
            return str(self._seed)

    @property
    def seed_int(self) -> Optional[int]:
        return self._seed

    def build_runner_snapshot(self) -> Dict[str, Any]:
        """floor/act/hp/max_hp/gold/game_won/deck_size/relics_size 纯 dict
        （口径同 stsrl_backend，eval harness 跨进程回传用）。"""
        if self._gc is None:
            return {
                "floor": 0, "act": 1, "hp": 0, "max_hp": 0, "gold": 0,
                "game_won": False, "deck_size": 0, "relics_size": 0,
            }
        st = self._get_state()
        return {
            "floor": int(st.get("floor", 0) or 0),
            "act": int(st.get("act", 1) or 1),
            "hp": int(st.get("hp", 0) or 0),
            "max_hp": int(st.get("max_hp", 0) or 0),
            "gold": int(st.get("gold", 0) or 0),
            "game_won": bool(self.game_won),
            "deck_size": int(st.get("deck_size", 0) or 0),
            "relics_size": int(st.get("relics_size", 0) or 0),
        }

    # =====================================================================
    # 3. 动作
    # =====================================================================
    def get_available_actions(self) -> List[Any]:
        """当前合法动作 list（中性表示）。

        lightspeed get_state['choices'] = [{idx, label}]，idx 即 step_choice 的索引。
        返回 choice dict list（{idx, label}），take_action 接受 idx 或该 dict。
        战斗中（in_battle）不暴露动作（元决策点不会停在战斗内）→ 返回 []。
        """
        st = self._get_state()
        if st.get("in_battle", False):
            return []
        return list(st.get("choices", []))

    def take_action(self, action: Any) -> bool:
        """执行一个动作（接受 choice idx int 或 {idx,...} dict），战斗在内部打完。

        与 RealEpisodeRunner 的 take_action(idx) 约定一致：harness 传 int idx。
        执行后若进入战斗，_advance_past_battles 调 play_battle 推进到下一个元决策点。
        """
        if self._gc is None:
            return False
        if isinstance(action, dict):
            idx = int(action.get("idx", 0))
        else:
            idx = int(action)
        ok = bool(self._sts.step_choice(self._gc, idx))
        # 执行后可能落进战斗（MAP→MONSTER / EVENT→COMBAT 等）→ 打完
        self._advance_past_battles()
        return ok

    def get_available_action_labels(self, state: V8State) -> List[str]:  # noqa: ARG002
        """模型可见的 action 字符串描述（与 get_available_actions idx 对齐）。

        lightspeed choices 的 label 已是可读串（"MAP->node_x=3" / "REWARD_card=Thunderclap"
        / "CAMPFIRE_REST" / "SHOP_buy_card=..." 等），直接用——模型走 hash embedding，
        这些串足够区分动作语义。
        """
        return [str(c.get("label", "")) for c in self.get_available_actions()]

    # =====================================================================
    # 4. 战斗 —— (a) 实时逐步战斗能力（Stage 2，新增）
    # =====================================================================
    # 把战斗当 step-by-step 环境：start_combat 进战斗 → get_combat_state 读 V8State →
    # get_legal_combat_actions 拿合法动作 → step_combat_action 执行一个 → 循环到结束。
    # 与黑盒 play_battle 路径互斥、并存（不删黑盒）。Stage 3 再让 env 切过来用这条路径。

    def enable_live_combat(self, on: bool = True) -> None:
        """打开/关闭实时战斗模式（Stage3+4）：开启时 take_action 不再自动 play_battle
        打完，把战斗留给 env 走 start_combat / step_combat_action 逐步驱动。
        env 在 _live_combat_available() 为真时调用此方法打开。"""
        self._live_combat_mode = bool(on)

    def in_live_combat(self) -> bool:
        """当前是否处于实时逐步战斗中（持有未结束的真 BattleContext）。"""
        return self._live_bc is not None

    def start_combat(self) -> bool:
        """若当前停在 BATTLE screen，进战斗并持有真 BattleContext（不立即 playout 打完）。

        成功返回 True，self._live_bc 持有 init 好的 BattleContext，后续用
        get_combat_state / get_legal_combat_actions / step_combat_action 逐步驱动。
        非 BATTLE screen（或已在实时战斗中）返回 False。
        """
        if self._gc is None:
            return False
        if self._live_bc is not None:
            return True  # 已在实时战斗中，幂等
        st = self._get_state()
        if not st.get("in_battle", False):
            return False
        bc = self._sts.enter_battle(self._gc)
        if bc is None:
            return False
        self._live_bc = bc
        self.combat_search_calls += 1
        return True

    def get_combat_state(self) -> V8State:
        """实时战斗中：把 get_combat_snapshot 的 hand / energy / 每个 monster 的
        hp/block/intent 灌进 V8State（hand 带 card_mech 机制向量；in_combat=True、
        phase=COMBAT）。非实时战斗（无 _live_bc）退回 build_v8_state（元决策态）。

        hand 每张牌 dict：{name, upgraded, cost, mech}（mech 为 N_MECH 维客观机制向量，
        供模型战斗脑 _combat_forward 吃）。monsters 每个 dict：
        {hp, max_hp, block, intent_dmg, intent_hits, intent, powers}。
        """
        if self._live_bc is None:
            return self.build_v8_state()

        base = self.build_v8_state()
        snap = dict(self._sts.get_combat_snapshot(self._live_bc))

        player = dict(snap.get("player", {}))
        energy = int(player.get("energy", 0) or 0)

        # ---- hand：卡名 + 升级 + cost + 机制向量 ----
        # snapshot 的 hand 只给卡名（display 名）；升级标记体现在名字尾的 '+'，
        # card_mech 的 _normalize 已吸收 '+' 后缀，故用名字尾 '+' 判 upgraded。
        hand: List[Dict[str, Any]] = []
        for raw_name in list(snap.get("hand", [])):
            name = str(raw_name)
            upgraded = name.rstrip().endswith("+") or "+" in name.split()[-1] if name else False
            hand.append({
                "name": name,
                "upgraded": bool(upgraded),
                "mech": card_mech_vector(name, bool(upgraded)),
            })

        # ---- monsters：hp / block / intent ----
        monsters: List[Dict[str, Any]] = []
        for e in list(snap.get("enemies", [])):
            ed = dict(e)
            if not ed.get("alive", True):
                continue
            powers: Dict[str, int] = {}
            for pk in ("strength", "vulnerable", "weak", "poison",
                       "metallicize", "ritual", "artifact"):
                pv = int(ed.get(pk, 0) or 0)
                if pv:
                    powers[pk] = pv
            monsters.append({
                "hp": int(ed.get("hp", 0) or 0),
                "max_hp": int(ed.get("max_hp", 0) or 0),
                "block": int(ed.get("block", 0) or 0),
                "intent_dmg": int(ed.get("intent_damage", -1)),
                "intent_hits": int(ed.get("intent_hits", 0) or 0),
                "intent": str(ed.get("intent", "")),
                "powers": powers,
            })

        # 用战斗内真值覆盖（snapshot 的 player.hp 是战斗中实时血量）
        base.hp = int(player.get("hp", base.hp) or base.hp)
        base.max_hp = int(player.get("max_hp", base.max_hp) or base.max_hp)
        base.in_combat = True
        base.phase = "COMBAT"
        base.hand = hand
        base.energy = energy
        base.monsters = monsters
        return base

    def get_legal_combat_actions(self) -> List[Dict[str, Any]]:
        """实时战斗中：当前 InputState 下的合法战斗动作 list（dict，含足够信息回传执行）。

        包 get_battle_actions：每个 dict 含 type / bits / label（+ CARD 的 source_idx/
        target_idx/card_name、CARD_SELECT 的 select_idx/card_select_task）。bits 是引擎
        无损编码，step_combat_action 优先用它执行。非实时战斗或战斗已结束 → 返回 []。
        """
        if self._live_bc is None:
            return []
        if self._sts.battle_is_over(self._live_bc):
            return []
        actions = [dict(a) for a in self._sts.get_battle_actions(self._live_bc)]
        # 富化每个带目标动作的目标怪物特征（hp_ratio / intent），供模型战斗指针网
        # per-action 特征用——让模型「看见」这个动作打的是哪个怪、它多血/什么意图。
        # 从同一 BattleContext 的 combat snapshot 读敌人（idx 对齐 target_idx）。
        snap = dict(self._sts.get_combat_snapshot(self._live_bc))
        enemies = list(snap.get("enemies", []))
        by_idx: Dict[int, Dict[str, Any]] = {}
        for e in enemies:
            ed = dict(e)
            by_idx[int(ed.get("idx", -1))] = ed
        for a in actions:
            tgt = int(a.get("target_idx", -1))
            ed = by_idx.get(tgt)
            if ed is None:
                continue
            mhp = float(ed.get("max_hp", 0) or 0)
            a["target_hp_ratio"] = (
                float(ed.get("hp", 0) or 0) / mhp if mhp > 0 else 0.0
            )
            a["target_intent_dmg"] = int(ed.get("intent_damage", -1))
            a["target_intent_hits"] = int(ed.get("intent_hits", 0) or 0)
            a["target_block"] = int(ed.get("block", 0) or 0)
        return actions

    def get_combat_input_state(self) -> str:
        """实时战斗中当前 InputState 名（PLAYER_NORMAL / CARD_SELECT / EXECUTING_ACTIONS）。
        非实时战斗返回空串。"""
        if self._live_bc is None:
            return ""
        return str(self._sts.get_input_state(self._live_bc))

    def combat_turn(self) -> int:
        """实时战斗中当前回合数（从 combat snapshot 的 'turn' 字段读）。

        供 env 的战斗回合上限兜底（[combat_cap]）判断「打了太多回合」。
        非实时战斗（无 _live_bc）返回 0。
        """
        if self._live_bc is None:
            return 0
        snap = dict(self._sts.get_combat_snapshot(self._live_bc))
        return int(snap.get("turn", 0) or 0)

    def step_combat_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """实时战斗中执行一个模型选的动作（get_legal_combat_actions 返回的 dict）。

        包 execute_battle_action（优先用 'bits' 无损还原）。执行后若战斗分出胜负，
        调 exit_battle 把结果写回 gc（胜→推进下一 screen / 奖励；负→gc.outcome=LOSS）
        并清掉 _live_bc（回到元决策态）。

        返回 {ok, done, won, outcome}：
          ok      — 动作是否合法且执行成功
          done    — 这一步后战斗是否结束
          won     — 战斗结束时是否胜（未结束为 None）
          outcome — 战斗结果字符串
        """
        if self._live_bc is None:
            return {"ok": False, "done": False, "won": None, "outcome": "NO_COMBAT"}
        ok = bool(self._sts.execute_battle_action(self._live_bc, dict(action)))
        done = bool(self._sts.battle_is_over(self._live_bc))
        outcome = str(self._sts.battle_outcome(self._live_bc))
        won: Optional[bool] = None
        if done:
            won = outcome == "PLAYER_VICTORY"
            # 结算回写 gc（胜推进 / 负置 outcome），再清掉 live bc 回元决策态
            self._sts.exit_battle(self._live_bc, self._gc)
            self._live_bc = None
        return {"ok": ok, "done": done, "won": won, "outcome": outcome}

    # =====================================================================
    # 4. 战斗 —— (b) 黑盒路径（现役，保留不删）
    # =====================================================================
    def run_combat_turn(
        self,
        *,
        solver_budgets: Dict[str, Tuple[float, int, int]],  # noqa: ARG002
    ) -> Dict[str, Any]:
        """no-op：lightspeed 战斗已在 take_action 内由 play_battle 一气呵成。

        StSRLBackend 这里走 TurnSolver 逐回合搜索；lightspeed 把整场战斗当黑盒
        交给引擎 MCTS（play_battle），元决策点永远不会停在 COMBAT，故 env 的
        _advance_to_meta_decision 不会调到这里。保留签名兼容 GameBackend 协议。
        """
        return {"action_repr": "", "turn": 0, "ok": True, "no_actions": True,
                "fallback_used": False}


class _RunStateShim:
    """run_state 透传债的最简鸭子桩：把 lightspeed get_state 标量暴露成 env.py
    期望的 run_state.<attr> 字段（floor/act/current_hp/max_hp/gold/deck/relics/
    map_position），让 env.py 的诊断日志 / node_reward detect / final_hp_ratio
    不直接 AttributeError 崩。

    TODO(stage2): 这是过渡桩。理想做法是把 env.py 对 backend.run_state.<x> 的
    直接访问改成走 backend.build_v8_state() / build_runner_snapshot() 中性接口，
    届时可删除本 shim。deck/relics 用空 list（env 仅取 len 做诊断，不影响 reward 主轴）。
    """

    __slots__ = ("_be",)

    def __init__(self, backend: "LightspeedBackend"):
        self._be = backend

    def _st(self) -> Dict[str, Any]:
        return self._be._get_state()

    @property
    def floor(self) -> int:
        return int(self._st().get("floor", 0) or 0)

    @property
    def act(self) -> int:
        return int(self._st().get("act", 1) or 1)

    @property
    def current_hp(self) -> int:
        return int(self._st().get("hp", 0) or 0)

    # env.py [action]/[meta] 日志读 rs.hp（StSRLSolver run_state 同时有 hp/current_hp）
    @property
    def hp(self) -> int:
        return int(self._st().get("hp", 0) or 0)

    @property
    def max_hp(self) -> int:
        return int(self._st().get("max_hp", 0) or 0)

    @property
    def gold(self) -> int:
        return int(self._st().get("gold", 0) or 0)

    @property
    def deck(self) -> List[Any]:
        gc = self._be._gc
        return list(gc.deck) if gc is not None else []

    @property
    def relics(self) -> List[Any]:
        gc = self._be._gc
        return list(gc.relics) if gc is not None else []

    @property
    def map_position(self) -> Any:
        st = self._st()
        return _MapPosShim(int(st.get("map_x", -1)), int(st.get("map_y", -1)))


class _MapPosShim:
    """map_position 桩（env guard_cap 日志读 .x/.y）。"""

    __slots__ = ("x", "y")

    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y


__all__ = ["LightspeedBackend"]

"""V8 牌组强度评估接口。

按 docs/v8_implementation_design.md 组件 5 设计：
- 让 StSRLSolver 拿当前牌组打 act1 标准敌人（cultist / lagavulin / hexaghost），
  每个敌人 sim 3 次取平均，输出 4 维数字。
- 内部 cache by (sorted_deck, sorted_relics, hp_bucket, act)，避免重算。

设计原则对照（docs/v8_design_principles.md）：
- 用 search 实战 simulate（不是 static score），自动反映运转/加费/power 卡间接价值
- 输出 4 维：damage_dealt / damage_taken / turns_to_win / win_rate
- 不依赖 v8_strategy 启发式

实现状态：
- 接口签名 + cache 已就绪
- _simulate_combat 接 StSRLSolver TurnSolver 真跑战斗（Ironclad 默认）
"""

from __future__ import annotations

import atexit
import hashlib
import logging
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Dict, List, Optional, Tuple


logger = logging.getLogger(__name__)

# 单个 sim future 的 wall-clock 上限（秒）。SEARCH_BUDGET_S=5s + engine 启动 +
# pickle 传输 + 偶发慢化，给 60s 余量；超过即视为 worker 卡死，取消 + fallback 0。
_FUTURE_TIMEOUT_SEC: float = 60.0

# 监控：每 N 次 evaluate_deck 调用打一次 pool state（active/pending）
_EVAL_CALL_COUNT: int = 0
_POOL_LOG_INTERVAL: int = 20


# ----- 标准敌人配置（act1）-----
# act1 覆盖各档次：
#   Cultist     普通弱（开局首战）
#   GremlinNob  精英 1（中等难度，会 enrage）—— 加在这里提高 win_rate 离散粒度
#   Lagavulin   精英 2（中后期难，sleep 期需要扛 buff）
#   Hexaghost   boss（act1 最难，多段 multi-hit）
# 4 个敌人 × 3 sim = 12 sample/eval，win_rate 离散步长 1/12 ≈ 0.083，
# 比 mini-smoke v2 的 3 敌人 × 3 sim = 9 sample（步长 0.111）更细，
# 解决 mini-smoke v2 报告"win_rate 卡 0.667 = 2/3"区分度不足问题。
STANDARD_ENEMIES_ACT1: List[str] = ["Cultist", "GremlinNob", "Lagavulin", "Hexaghost"]
SIMS_PER_ENEMY: int = 3
# 单 turn 搜索时间预算（秒）。1.0s 已足够找到与 5.0s 完全相同的 win/dmg 结果；
# 在 Snecko Eye 拿到后（随机 cost）5.0s 会被全用满（每 turn 都 hit budget），
# 导致 evaluate_deck wall-time 50s+ → eval 30 seeds 一半 timeout。
#
# 2026-05-19 benchmark（post-boss deck + Snecko Eye 全 4 敌人 × 3 sim）：
#   budget=5.0s → 152s seq → ~17s parallel；ALL outcomes identical
#   budget=1.0s → 38s seq  → ~4s parallel；ALL outcomes identical
# 同 benchmark 也覆盖 starter deck + Snecko（最 pathological 情形），1.0s 与 5.0s
# 在 4 个敌人 × 3 sim 的 win_rate / damage_dealt / damage_taken 完全一致。
SEARCH_BUDGET_S: float = 1.0

# act2/act3 的标准敌人 user 后续训练时再扩展，先 act1 跑通
STANDARD_ENEMIES_BY_ACT: Dict[int, List[str]] = {
    1: STANDARD_ENEMIES_ACT1,
}


# ----- Cache -----
# key: deck_hash → value: 4 维 result dict
_DECK_EVAL_CACHE: Dict[str, Dict[str, float]] = {}

# ----- Cache 统计 -----
# 真 hit/miss 计数：smoke / 训练监控用，看 cache 是否真 work
_CACHE_HITS: int = 0
_CACHE_MISSES: int = 0


# ----- 并行执行（cache miss 路径内）-----
# 4 个 enemy × 3 sim = 12 个独立 sim，每个 sim 有自己的 engine / RNG / state，
# 互相之间无共享可变状态，是 embarrassingly parallel。
# macOS + MPS + fork() 已知有问题（torch / metal 在 fork 后状态损坏），所以强制 spawn。
# 池大小：min(12, cpu_count - 1)，给主进程留 1 核。
# 池只创建一次（lazy），atexit 注册关闭。
_PARALLEL_ENABLED: bool = os.environ.get("V8_DECK_EVALUATOR_PARALLEL", "1") != "0"
_POOL: Optional[ProcessPoolExecutor] = None


def _get_pool() -> Optional[ProcessPoolExecutor]:
    """懒创建 spawn-context process pool。disable 时返回 None。

    环境变量 cap（parallel_env.py 子进程模式用）：
        V8_DECK_EVALUATOR_MAX_WORKERS=N → 把默认 min(12, cpu-1) 进一步压到 N。
        典型场景：ParallelV8Env n_envs=4 时，每个子进程设 N=2，避免 4×12=48 个
        孙子进程爆 CPU。未设环境变量则保留旧上限。
    """
    global _POOL
    if not _PARALLEL_ENABLED:
        return None
    if _POOL is None:
        max_workers = min(12, max(1, (os.cpu_count() or 2) - 1))
        # env var cap（subprocess 模式下小化 worker 防爆 CPU）
        env_cap_raw = os.environ.get("V8_DECK_EVALUATOR_MAX_WORKERS", "").strip()
        if env_cap_raw:
            try:
                env_cap = int(env_cap_raw)
                if env_cap >= 1:
                    max_workers = min(max_workers, env_cap)
            except ValueError:
                logger.warning(
                    "deck_evaluator: invalid V8_DECK_EVALUATOR_MAX_WORKERS=%r, ignored",
                    env_cap_raw,
                )
        ctx = multiprocessing.get_context("spawn")
        _POOL = ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx)
        logger.info(
            "deck_evaluator: created spawn ProcessPoolExecutor max_workers=%d",
            max_workers,
        )
    return _POOL


def _shutdown_pool() -> None:
    """atexit 关闭 pool。"""
    global _POOL
    if _POOL is not None:
        try:
            _POOL.shutdown(wait=False, cancel_futures=True)
        except Exception:  # noqa: BLE001
            pass
        _POOL = None


def restart_pool() -> None:
    """重建 ProcessPoolExecutor（eval 前后调用，防 worker 状态累积 / 慢化）。

    用法（run_eval 入口 + 出口都调一次）：
        from v8.deck_evaluator import restart_pool
        restart_pool()  # 关旧的 + 让下次 _get_pool 重建

    线程安全说明：不在 eval 中途调；只在 eval boundary 处调（caller 保证）。
    """
    global _POOL
    if _POOL is not None:
        try:
            _POOL.shutdown(wait=False, cancel_futures=True)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "deck_evaluator.restart_pool: shutdown old pool failed: %s: %s",
                type(e).__name__, e,
            )
        _POOL = None
    logger.info("deck_evaluator: pool restarted (next call will lazy-create)")


atexit.register(_shutdown_pool)


def _simulate_combat_worker(
    enemy_id: str,
    deck: List[Dict],
    relics: List[str],
    hp: int,
    max_hp: int,
    act: int,
    seed: int,
) -> Optional[Dict[str, float]]:
    """子进程入口：单 (enemy, sim_idx) → outcome dict。

    顶层函数（pickleable）。失败返回 None，主进程过滤掉（等价于原来 try/except 后的 continue）。
    每个子进程首次调用时会触发 _ensure_solver_imports 懒加载。
    """
    try:
        return _simulate_combat(
            enemy_id=enemy_id,
            deck=deck,
            relics=relics,
            hp=hp,
            max_hp=max_hp,
            act=act,
            seed=seed,
        )
    except Exception as e:  # noqa: BLE001
        # 子进程里 logger 不一定回流到主进程，记进程内 log 即可；
        # 主进程见 None 时记 warning（与原 sequential 行为对齐）。
        logger.warning(
            "worker _simulate_combat failed (enemy=%s seed=%d): %s: %s",
            enemy_id, seed, type(e).__name__, e,
        )
        return None


def _deck_hash(
    deck: List[Dict],
    relics: List[str],
    hp: int,
    max_hp: int,
    act: int,
) -> str:
    """根据牌组 / 遗物 / HP / act 算 hash 用于 cache。

    HP 分 10 档（0-9），同档 HP 视为等价，减少 cache miss 同时保持评估准确。
    deck / relics 排序后入 hash，去除顺序敏感。
    """
    deck_repr = sorted([(c.get("name", ""), bool(c.get("upgraded", False))) for c in deck])
    relics_repr = sorted(relics)
    hp_bucket = (hp * 10) // max(max_hp, 1)
    key = repr((deck_repr, relics_repr, hp_bucket, act))
    return hashlib.md5(key.encode()).hexdigest()


def evaluate_deck(
    deck: List[Dict],
    relics: List[str],
    hp: int,
    max_hp: int,
    act: int = 1,
) -> Dict[str, float]:
    """评估牌组强度。

    输入：
        deck: list of {"name": str, "upgraded": bool}
        relics: list of relic ID 字符串
        hp / max_hp: 当前血量与上限（影响敌人面对的玩家强度）
        act: 当前所在 act（决定 standard enemies 集合）

    输出（4 维 dict，跨敌人取平均）：
        damage_dealt:  我方对敌人累计造成伤害 / sim
        damage_taken:  我方累计承受伤害 / sim
        turns_to_win:  胜局平均轮数（败局回退 max_turn）
        win_rate:      胜率 ∈ [0, 1]

    实现：
        1. cache 命中 → 直接返回
        2. 否则跑 STANDARD_ENEMIES_BY_ACT[act] × SIMS_PER_ENEMY 次 sim 取平均
        3. 写入 cache 后返回
    """
    global _CACHE_HITS, _CACHE_MISSES, _EVAL_CALL_COUNT

    _EVAL_CALL_COUNT += 1
    # 监控：每 N 次调用打一次 pool state（active workers / pending jobs），
    # 用来在 eval-hang 复现时定位是 pool 内累积 vs. inference 慢化。
    if _EVAL_CALL_COUNT % _POOL_LOG_INTERVAL == 0:
        try:
            _pool = _POOL  # 不要触发 lazy-create
            if _pool is not None:
                # ProcessPoolExecutor 的 _processes / _pending_work_items 是 _internal_ 字段，
                # 没有公开 API。读不到就跳过（不能为 log 让训练崩）。
                n_workers = len(getattr(_pool, "_processes", {}) or {})
                n_pending = len(getattr(_pool, "_pending_work_items", {}) or {})
                logger.info(
                    "[deck_eval] pool_active=%d queued=%d call_count=%d cache_size=%d",
                    n_workers, n_pending, _EVAL_CALL_COUNT, len(_DECK_EVAL_CACHE),
                )
        except Exception:  # noqa: BLE001
            pass

    cache_key = _deck_hash(deck, relics, hp, max_hp, act)
    cached = _DECK_EVAL_CACHE.get(cache_key)
    if cached is not None:
        _CACHE_HITS += 1
        return cached
    _CACHE_MISSES += 1

    enemy_pool = STANDARD_ENEMIES_BY_ACT.get(act, STANDARD_ENEMIES_ACT1)

    accum = {
        "damage_dealt": 0.0,
        "damage_taken": 0.0,
        "turns_to_win": 0.0,
        "win_rate": 0.0,
    }
    n_runs = 0

    # 12 个独立 sim 任务列表（4 enemy × 3 sim），每个 sim 互不干扰
    tasks: List[Tuple[str, int]] = [
        (enemy_id, sim_idx)
        for enemy_id in enemy_pool
        for sim_idx in range(SIMS_PER_ENEMY)
    ]

    pool = _get_pool()
    outcomes: List[Optional[Dict[str, float]]] = []

    if pool is not None:
        # 并行：spawn 子进程跑 12 个 sim
        # 注意：deck/relics 是 list[dict]/list[str]，pickleable；HP/act/seed 都是 int
        try:
            futures = [
                pool.submit(
                    _simulate_combat_worker,
                    enemy_id, deck, relics, hp, max_hp, act, sim_idx,
                )
                for enemy_id, sim_idx in tasks
            ]
            for fut in futures:
                try:
                    outcomes.append(fut.result(timeout=_FUTURE_TIMEOUT_SEC))
                except FutureTimeoutError:
                    # worker 卡死：cancel + fallback None，让训练继续。
                    # cancel 对已经 running 的 future 返回 False（进程仍在跑，但池仍可被复用）；
                    # 不能 join 这条 future，否则又会被它卡住。
                    try:
                        fut.cancel()
                    except Exception:  # noqa: BLE001
                        pass
                    logger.warning(
                        "[deck_eval] timeout (>%ss): future cancelled, fallback None",
                        _FUTURE_TIMEOUT_SEC,
                    )
                    outcomes.append(None)
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "evaluate_deck pool future failed: %s: %s",
                        type(e).__name__, e,
                    )
                    outcomes.append(None)
        except Exception as e:  # noqa: BLE001
            # 池整体崩了（不应发生，但 fallback 到 sequential 让训练不挂）
            logger.warning(
                "evaluate_deck parallel dispatch failed (%s: %s); falling back to sequential",
                type(e).__name__, e,
            )
            outcomes = []
            for enemy_id, sim_idx in tasks:
                try:
                    outcomes.append(_simulate_combat(
                        enemy_id=enemy_id, deck=deck, relics=relics,
                        hp=hp, max_hp=max_hp, act=act, seed=sim_idx,
                    ))
                except Exception as e2:  # noqa: BLE001
                    logger.warning(
                        "evaluate_deck sim failed (enemy=%s): %s: %s",
                        enemy_id, type(e2).__name__, e2,
                    )
                    outcomes.append(None)
    else:
        # 顺序：与历史行为完全一致（debug / V8_DECK_EVALUATOR_PARALLEL=0）
        for enemy_id, sim_idx in tasks:
            try:
                outcomes.append(_simulate_combat(
                    enemy_id=enemy_id, deck=deck, relics=relics,
                    hp=hp, max_hp=max_hp, act=act, seed=sim_idx,
                ))
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "evaluate_deck sim failed (enemy=%s): %s: %s",
                    enemy_id, type(e).__name__, e,
                )
                outcomes.append(None)

    for outcome in outcomes:
        if outcome is None:
            continue
        for k in accum:
            accum[k] += outcome[k]
        n_runs += 1

    if n_runs == 0:
        # 所有 sim 全失败：fallback 到全 0（让 caller 知道评估失效）
        result = {k: 0.0 for k in accum}
    else:
        result = {k: v / n_runs for k, v in accum.items()}

    _DECK_EVAL_CACHE[cache_key] = result
    return result


def clear_cache() -> None:
    """清空 cache。act 切换时调用（标准敌人换了，旧 cache 失效）。"""
    global _DECK_EVAL_CACHE
    _DECK_EVAL_CACHE = {}


def cache_size() -> int:
    """返回 cache 当前条目数（debug / 监控用）。"""
    return len(_DECK_EVAL_CACHE)


def get_cache_stats() -> Dict[str, float]:
    """返回 cache 真实命中率统计（smoke / 训练监控用）。

    输出：
        hits:       命中次数（evaluate_deck 调用且命中）
        misses:     miss 次数（evaluate_deck 调用且需要跑 sim）
        hit_rate:   hits / (hits + misses)，分母 0 时返回 0
        cache_size: 当前 cache 条目数
    """
    total = _CACHE_HITS + _CACHE_MISSES
    return {
        "hits": _CACHE_HITS,
        "misses": _CACHE_MISSES,
        "hit_rate": _CACHE_HITS / max(total, 1),
        "cache_size": len(_DECK_EVAL_CACHE),
    }


def reset_cache_stats() -> None:
    """重置 cache 命中/miss 计数。每 seed 开始前调用。

    注意：不清空 _DECK_EVAL_CACHE 本身（那个由 clear_cache 控制）；
    只重置 hit/miss 计数器。
    """
    global _CACHE_HITS, _CACHE_MISSES
    _CACHE_HITS = 0
    _CACHE_MISSES = 0


# =========================================================================
# 真实 simulate 实现
# =========================================================================

# 全局缓存（懒加载）：name → card_id 映射 / enemy_class 表
_CARD_NAME_TO_ID: Optional[Dict[str, str]] = None
_ENEMY_REGISTRY: Optional[Dict[str, type]] = None
# safety 上限：超过算败。act1 fights 中典型 turn 数 3-10；超过 15 turn 还没赢的 deck
# 强度本就极低（搜索每 turn hit budget 但仍 stall）。从 50 → 15 限制极端 worst-case
# 不影响 win/dmg 输出（2026-05-19 benchmark：starter+Snecko/post-boss+Snecko 等 deck
# max_turns=15 与 50 输出完全一致）。
_MAX_TURNS_HARD_CAP: int = 15


def _ensure_solver_imports() -> Tuple[type, type, type, callable, type]:
    """懒加载 StSRLSolver 模块。返回 (Random, TurnSolver, EndTurn, create_combat_from_enemies, ALL_CARDS_dict)。

    放在函数内是为了让本模块在 StSRLSolver 没装时也能 import（cache / hash 函数仍可用）。
    """
    global _CARD_NAME_TO_ID, _ENEMY_REGISTRY

    from sts_paths import ensure_on_sys_path
    ensure_on_sys_path()

    from packages.engine.combat_engine import create_combat_from_enemies  # type: ignore  # noqa: E402
    from packages.engine.state.combat import EndTurn  # type: ignore  # noqa: E402
    from packages.engine.state.rng import Random  # type: ignore  # noqa: E402
    from packages.training.turn_solver import TurnSolver  # type: ignore  # noqa: E402
    from packages.engine.content.cards import ALL_CARDS  # type: ignore  # noqa: E402

    # 构建 name → id 映射（一次性）：
    #   - V8Env 传过来的 deck["name"] 实际是 card.id（见 v8/env.py:145）
    #   - 历史版本只用 card.name 当 key，导致 caller 传 "Strike_R" 时 lookup miss → deck empty
    #   - 修复：所有卡都把 cid 自身作为 key 加入；Strike/Defend 同时保留 name→Ironclad_R fallback
    if _CARD_NAME_TO_ID is None:
        name_map: Dict[str, str] = {}
        for cid, card in ALL_CARDS.items():
            name = getattr(card, "name", cid)
            # 关键：cid 自身永远 self-map（V8Env 传的是 id）
            name_map[cid] = cid
            # Strike/Defend 等基础卡有多职业版本，name 重复（"Strike" 对应 _P/_R/_G/_B）
            # 用 name 作 key 时优先 Ironclad（_R）作为 fallback
            if name in ("Strike", "Defend"):
                if cid.endswith("_R"):
                    name_map[name] = cid
                elif name not in name_map:
                    # _R 还没出现，先放当前的占位（后续遇到 _R 会覆盖）
                    name_map[name] = cid
            else:
                # 其余卡 name 大多与 id 一致，直接覆盖
                name_map[name] = cid
        _CARD_NAME_TO_ID = name_map

    # 构建 enemy 类注册表（act1 标准 4 个：弱普通 + 2 精英 + boss）
    if _ENEMY_REGISTRY is None:
        from packages.engine.content.enemies import (  # type: ignore  # noqa: E402
            Cultist, GremlinNob, Lagavulin, Hexaghost,
        )
        _ENEMY_REGISTRY = {
            "Cultist": Cultist,
            "GremlinNob": GremlinNob,
            "Lagavulin": Lagavulin,
            "Hexaghost": Hexaghost,
        }

    return Random, TurnSolver, EndTurn, create_combat_from_enemies, ALL_CARDS


def _deck_to_card_ids(deck: List[Dict]) -> List[str]:
    """V8 deck dict → StSRLSolver card_id list（含 + 升级后缀）。

    未知 name 的卡：跳过（记 warning），避免传非法 ID 给引擎。
    """
    assert _CARD_NAME_TO_ID is not None  # 调 _ensure_solver_imports 后保证
    out: List[str] = []
    skipped: List[str] = []
    for c in deck:
        name = c.get("name", "")
        upgraded = bool(c.get("upgraded", False))
        cid = _CARD_NAME_TO_ID.get(name)
        if cid is None:
            skipped.append(name)
            continue
        if upgraded:
            cid = f"{cid}+1"
        out.append(cid)
    if skipped:
        # 升 warning：跑训练时若 deck 里有未知卡，应该被注意到
        logger.warning(
            "deck_to_card_ids: %d unknown card names skipped: %s",
            len(skipped), skipped[:5],  # 只打前 5 个避免 log 爆
        )
    return out


def _simulate_combat(
    enemy_id: str,
    deck: List[Dict],
    relics: List[str],
    hp: int,
    max_hp: int,
    act: int,
    seed: int = 0,
) -> Dict[str, float]:
    """跑一场 (deck vs enemy_id) 的战斗 sim，返回 4 维 outcome。

    流程：
        1. name → card_id 转换（升级版本 +1 后缀）
        2. 实例化 enemy（按 enemy_id 找类，构造 ai_rng / hp_rng）
        3. create_combat_from_enemies 起战 → engine.start_combat()
        4. while not is_combat_over: TurnSolver.solve_turn → 执行 plan → 必要时 EndTurn
        5. 记录 damage_dealt（敌人初始 max_hp 总和 − 剩余 hp 总和）/
           damage_taken（玩家初始 hp − 剩余 hp）/ turns / victory

    safety：
        - SEARCH_BUDGET_S 限制单次 turn solve 时间（5s）
        - _MAX_TURNS_HARD_CAP（50 turn）防死循环
        - 异常一律记 warning 并 raise（caller evaluate_deck 已 try/except）
    """
    Random, TurnSolver, EndTurn, create_combat_from_enemies, _ALL_CARDS = (
        _ensure_solver_imports()
    )

    enemy_cls = (_ENEMY_REGISTRY or {}).get(enemy_id)
    if enemy_cls is None:
        raise ValueError(f"unknown enemy_id={enemy_id!r}; supported: {list(_ENEMY_REGISTRY or {})}")

    # 转 deck（V8 dict → StSRLSolver string ID）
    card_ids = _deck_to_card_ids(deck)
    if not card_ids:
        # 历史 bug：seed 17 出过 27 次 "deck empty after name→id" fail——
        # 因为 _CARD_NAME_TO_ID 早期只 keyed by card.name，而 V8Env 传 "Strike_R"
        # （= card.id）匹配不到。当前修复已把 cid 自映射进 map（见 _ensure_solver_imports
        # 第 ~225 行），正常 V8 deck 不会触发本分支。
        # 兜底：若仍触发（如 deck 中卡牌 name 全部不在 ALL_CARDS），打详细 warning
        # 后 raise——外层 evaluate_deck 已 try/except + n_runs=0 fallback 全 0，
        # 不会让训练挂；但需要 warning 让用户注意到 deck 异常。
        sample_names = [c.get("name", "") for c in deck[:5]]
        logger.warning(
            "deck empty after name->id conversion: deck_size=%d sample_names=%s "
            "(check _CARD_NAME_TO_ID coverage)",
            len(deck), sample_names,
        )
        raise ValueError("deck empty after name->id conversion")

    # 实例化敌人（不同 seed 让 sim 间走不同 AI 分支）
    ai_rng = Random(1000 + seed)
    hp_rng = Random(2000 + seed)
    enemy = enemy_cls(ai_rng=ai_rng, ascension=0, hp_rng=hp_rng)

    # 构造引擎
    engine = create_combat_from_enemies(
        enemies=[enemy],
        player_hp=hp,
        player_max_hp=max_hp,
        deck=card_ids,
        energy=3,
        relics=list(relics or []),
        potions=[],
    )
    engine.start_combat()

    # 记录初始指标
    initial_player_hp = engine.state.player.hp
    initial_enemy_total_hp = sum(e.max_hp for e in engine.state.enemies)

    # solver：单 turn DFS / beam，时间预算转 ms
    solver = TurnSolver(
        time_budget_ms=SEARCH_BUDGET_S * 1000.0,
        node_budget=5000,
    )

    turns_played = 0
    while not engine.is_combat_over() and turns_played < _MAX_TURNS_HARD_CAP:
        plan = solver.solve_turn(engine, room_type="monster")
        if plan is None or len(plan) == 0:
            # 没招了：直接 EndTurn
            try:
                engine.execute_action(EndTurn())
            except Exception as e:  # noqa: BLE001
                logger.warning("execute_action(EndTurn) failed: %s", e)
                break
            turns_played += 1
            continue

        # 执行 plan
        ended = False
        for action in plan:
            if engine.is_combat_over():
                break
            try:
                engine.execute_action(action)
            except Exception as e:  # noqa: BLE001
                logger.warning("execute_action failed (action=%r): %s", action, e)
                break
            if isinstance(action, EndTurn):
                ended = True

        # plan 没含 EndTurn 的话补一个，保证 turn 推进
        if not ended and not engine.is_combat_over():
            try:
                engine.execute_action(EndTurn())
            except Exception as e:  # noqa: BLE001
                logger.warning("execute_action(EndTurn after plan) failed: %s", e)
                break
        turns_played += 1

    # 战斗结束统计
    final_player_hp = max(0, engine.state.player.hp)
    final_enemy_total_hp = sum(max(0, e.hp) for e in engine.state.enemies)
    damage_dealt = max(0, initial_enemy_total_hp - final_enemy_total_hp)
    damage_taken = max(0, initial_player_hp - final_player_hp)
    victory = bool(engine.is_victory()) if engine.is_combat_over() else False

    return {
        "damage_dealt": float(damage_dealt),
        "damage_taken": float(damage_taken),
        "turns_to_win": float(turns_played),
        "win_rate": 1.0 if victory else 0.0,
    }


__all__ = [
    "evaluate_deck",
    "clear_cache",
    "cache_size",
    "get_cache_stats",
    "reset_cache_stats",
    "STANDARD_ENEMIES_ACT1",
    "STANDARD_ENEMIES_BY_ACT",
    "SIMS_PER_ENEMY",
    "SEARCH_BUDGET_S",
]

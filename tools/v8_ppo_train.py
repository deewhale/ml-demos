"""V8 阶段 B：PPO 训练入口。

按 docs/v8_implementation_design.md §10 + docs/v8_design_principles.md：
- Phase A 已训好 combat head（监督 BC，从 turn_records 学搜索的出牌顺序/联动/target）
- Phase B 在战斗 head 之上 RL：学元决策（路线 / 选卡 / 事件 / 商店 / 休息 / 宝箱 / Neow / Boss）
- 战斗内仍由 TurnSolver 主导，model 通过 V8CombatNetWrapper 参与 leaf 评估
- 元决策 trajectory 才进 PPO（V8Env 自动跳过战斗 phase）

CLI 用法：
    # Smoke：管线快速验证（不要给 user 跑这个当真训练）
    python tools/v8_ppo_train.py --smoke

    # 真训练（user 启动，~83h）
    python tools/v8_ppo_train.py --num_episodes 1000 --batch_size 32

实现要点：
- 每 batch_size=N 局收 rollout，concat 后做一次 PPO update
- 每 eval_frequency 局跑一次 eval（deterministic，no_grad）
- 每 checkpoint_frequency 局存一次 ckpt（含 optimizer state）+ 元数据 JSON
- smoke 模式覆盖少量 episode，验证 wrapper hook、PPO update、ckpt 写盘
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time
from collections import Counter, deque
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import torch


# ============================================================
# Graceful shutdown：SIGINT 第一次软停（保存后退出），第二次硬退出
# ============================================================

# 模块级 flag：信号 handler 写、main loop 读
_STOP_REQUESTED: bool = False
_SIGINT_COUNT: int = 0


def _install_sigint_handler() -> None:
    """注册 SIGINT handler。
    - 第 1 次 Ctrl+C：设 _STOP_REQUESTED，main loop 跑完当前 batch + 保存 final ckpt + 写 summary 后退出
    - 第 2 次 Ctrl+C：立即 os._exit(130)（假定卡在某处无法软停）
    """
    def _handler(signum, frame):  # noqa: ARG001
        global _STOP_REQUESTED, _SIGINT_COUNT
        _SIGINT_COUNT += 1
        if _SIGINT_COUNT == 1:
            _STOP_REQUESTED = True
            # 直接 print（不走 logger，因为有些场合 logger 可能正卡在 flush）
            print(
                "\n[SIGINT] Ctrl+C received; will save final checkpoint + summary "
                "and exit after current batch finishes. Press Ctrl+C again to hard exit.",
                flush=True,
            )
        else:
            print(
                f"\n[SIGINT x{_SIGINT_COUNT}] hard exit (state may be lost).",
                flush=True,
            )
            os._exit(130)

    signal.signal(signal.SIGINT, _handler)

# 保证从仓库根可 import v8.*
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from v8.combat_net_wrapper import V8CombatNetWrapper
from v8.deck_evaluator import (
    get_cache_stats as get_deck_cache_stats,
    restart_pool as restart_deck_pool,
)
from v8.env import V8Env
from v8.model import V8Model
from v8.parallel_env import ParallelV8Env
from v8.trainer import RolloutStep, V8PPOTrainer

# Eval 内存监控用（可选；若环境无 psutil 用 None fallback）
try:
    import psutil  # type: ignore
    _PSUTIL_AVAILABLE = True
except ImportError:  # pragma: no cover
    psutil = None  # type: ignore
    _PSUTIL_AVAILABLE = False

# 强制 stdout 行缓冲（unbuffered），保证 heartbeat 实时可见。
# python -u / PYTHONUNBUFFERED=1 也行，这里再加一道保险。
try:
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
except Exception:  # noqa: BLE001
    pass

# 用 StreamHandler 显式绑 stdout，避免落进默认 stderr buffer 区
_handler = logging.StreamHandler(stream=sys.stdout)
_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logging.basicConfig(level=logging.INFO, handlers=[_handler], force=True)
logger = logging.getLogger("v8_ppo_train")


# ============================================================
# 工具
# ============================================================


def select_device(preferred: str) -> torch.device:
    """选 device：优先用户传的；mps 不可用回 cpu。"""
    if preferred == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        logger.warning("MPS 不可用，fallback 到 CPU")
        return torch.device("cpu")
    return torch.device(preferred)


def load_combat_head(model: V8Model, ckpt_path: str) -> Dict[str, Any]:
    """加载 Phase A 训好的 combat head 权重到 model。

    Phase A 的 ckpt 是 {model_state_dict, metadata}（与 v8_bc_train.py 一致）。
    Phase A 训的就是 V8Model 全套权重，所以直接 load_state_dict 即可。
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"combat head checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict")
    if state_dict is None:
        raise RuntimeError(f"checkpoint {ckpt_path} 缺 model_state_dict")

    # 容忍 shape 不匹配的参数：ckpt 里 shape 与当前 model 对应 param 不一致的
    # （如 v5 新增 deck_strength 特征令 state_fuse.0.weight 301→302），剔除后跳过，
    # 保持该层当前模型的随机初始化（让它 fresh 学），其余预热权重照常加载。
    model_state = model.state_dict()
    skipped_shape = []
    filtered_state_dict = {}
    for k, v in state_dict.items():
        if k in model_state and tuple(model_state[k].shape) != tuple(v.shape):
            skipped_shape.append((k, tuple(v.shape), tuple(model_state[k].shape)))
            continue
        filtered_state_dict[k] = v

    missing, unexpected = model.load_state_dict(filtered_state_dict, strict=False)
    loaded_n = len(filtered_state_dict) - len(unexpected)
    logger.info(
        "loaded combat head from %s (loaded=%d, skipped_shape=%d, missing=%d, unexpected=%d)",
        ckpt_path, loaded_n, len(skipped_shape), len(missing), len(unexpected),
    )
    if skipped_shape:
        for k, ck_shape, cur_shape in skipped_shape:
            logger.info(
                "  跳过 shape 不匹配 param (保持随机初始化): %s ckpt%s vs model%s",
                k, ck_shape, cur_shape,
            )
    if missing:
        logger.info("missing keys (前 5): %s", list(missing)[:5])
    if unexpected:
        logger.info("unexpected keys (前 5): %s", list(unexpected)[:5])
    return ckpt.get("metadata", {})


def save_metadata_json(
    json_path: str,
    *,
    num_episodes_so_far: int,
    eval_history: List[Dict[str, Any]],
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """checkpoint 旁边保存元数据 JSON。"""
    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "num_episodes_so_far": num_episodes_so_far,
        "eval_history": eval_history,
    }
    if extra:
        payload.update(extra)
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, default=str, ensure_ascii=False)


# ============================================================
# 评估
# ============================================================


def _log_eval_mem(tag: str, seed: int) -> None:
    """[eval-mem] log：每个 seed 开始/结束打 RSS + MPS allocated。

    监控 eval 慢化是否来自内存累积。psutil 不可用时只打 MPS。
    """
    parts: List[str] = []
    if _PSUTIL_AVAILABLE and psutil is not None:
        try:
            rss_mb = psutil.Process().memory_info().rss / 1e6
            parts.append(f"rss={rss_mb:.0f}MB")
        except Exception:  # noqa: BLE001
            parts.append("rss=?")
    if torch.backends.mps.is_available():
        try:
            mps_alloc_mb = torch.mps.current_allocated_memory() / 1e6
            parts.append(f"mps_alloc={mps_alloc_mb:.0f}MB")
        except Exception:  # noqa: BLE001
            parts.append("mps_alloc=?")
    if parts:
        logger.info("[eval-mem] %s seed=%d %s", tag, seed, " ".join(parts))


# ----- Eval stagnation attribution（2026-05-19 加）-----
# 用户要求：log 停滞 5 min 不立即 kill，先归因（同 event/combat/action 反复）。
# 命中 loop pattern → 真死循环 → kill；未命中 → 给 5 min grace；hard cap 3600s 兜底。
#
# 设计：在主线程 polling loop 中维护内存 deque（无需 hook env），每次 poll 读 env 暴露的
#   - _last_event_id（最近 event_id；非 EVENT phase 时为 None / 不变）
#   - _last_action_repr（最近一次 take_action 描述；env 内部已维护）
#   - runner.run_state.monsters_alive（战斗内 enemies）
# 与 polling 间隔（5s）对齐，poll 期间没新动作则同一个值会被重复 push，本身就是 mode collapse 信号。

# 阈值（参考 docstring 里 user 要求）
ATTR_EVENT_LOOP_THRESHOLD = 30          # 同 event_id 出现次数（event 死循环）
ATTR_COMBAT_HANG_THRESHOLD = 5          # 同 enemies 连续出现（combat 没推进）
ATTR_ACTION_COLLAPSE_THRESHOLD = 50     # 最近 50 个 action 全同一（mode collapse）


def _classify_hang_pattern(
    event_id_history: Deque[str],
    combat_enemies_history: Deque[str],
    action_history: Deque[str],
) -> Tuple[Optional[str], str]:
    """从 history deque 判定是否真死循环。

    Returns:
        (loop_type, detail): loop_type in {"event_loop", "combat_hang",
            "action_mode_collapse"} or None（未命中）。detail 是一行汇总。
    """
    # 1) Event loop：同 event_id 出现 >= 阈值
    if event_id_history:
        ev_counter: Counter = Counter(event_id_history)
        ev_top, ev_top_n = ev_counter.most_common(1)[0]
        if ev_top and ev_top != "?" and ev_top_n >= ATTR_EVENT_LOOP_THRESHOLD:
            return ("event_loop", f"event_id={ev_top} count={ev_top_n}/{len(event_id_history)}")

    # 2) Combat hang：同 enemies 连续 >= 阈值 出现（最后 N 个全相同）
    if len(combat_enemies_history) >= ATTR_COMBAT_HANG_THRESHOLD:
        recent = list(combat_enemies_history)[-ATTR_COMBAT_HANG_THRESHOLD:]
        # 过滤空（非战斗状态）
        non_empty = [x for x in recent if x and x != "?"]
        if len(non_empty) >= ATTR_COMBAT_HANG_THRESHOLD and len(set(non_empty)) == 1:
            return ("combat_hang", f"enemies={non_empty[0]} repeated={len(non_empty)}")

    # 3) Action mode collapse：最近 N 个 action 全同一
    if len(action_history) >= ATTR_ACTION_COLLAPSE_THRESHOLD:
        recent = list(action_history)[-ATTR_ACTION_COLLAPSE_THRESHOLD:]
        non_empty = [x for x in recent if x]
        if len(non_empty) >= ATTR_ACTION_COLLAPSE_THRESHOLD and len(set(non_empty)) == 1:
            return ("action_mode_collapse", f"action={non_empty[0]!r} repeated={len(non_empty)}")

    return (None, "")


def _attribution_dump(
    env: "V8Env",
    event_id_history: Deque[str],
    combat_enemies_history: Deque[str],
    action_history: Deque[str],
) -> str:
    """从 env runner + history 拼一个 multi-line attribution dump。"""
    lines: List[str] = []
    try:
        runner = env.runner
        rs = getattr(runner, "run_state", None) if runner is not None else None
        if rs is not None:
            floor = int(getattr(rs, "floor", 0) or 0)
            act = int(getattr(rs, "act", 1) or 1)
            screen = str(getattr(rs, "screen_type", "") or "")
            hp = int(getattr(rs, "current_hp", 0) or 0)
            max_hp = int(getattr(rs, "max_hp", 0) or 0)
            deck_size = len(getattr(rs, "deck", []) or [])
            phase = str(getattr(env, "_last_phase", "") or "")
            enemies = [
                (e.monster_id, int(e.current_hp))
                for e in (getattr(rs, "monsters_alive", []) or [])
            ]
            lines.append(
                f"  runner: floor={floor} act={act} screen={screen} "
                f"phase={phase} hp={hp}/{max_hp} deck_size={deck_size} enemies={enemies}"
            )
            lines.append(f"  env._last_event_id={getattr(env, '_last_event_id', None)!r}")
        else:
            lines.append("  runner: <unavailable>")
    except Exception as e:  # noqa: BLE001
        lines.append(f"  runner: read_err {type(e).__name__}: {e}")

    # History 统计
    if event_id_history:
        ev_counter: Counter = Counter(event_id_history)
        top3 = ev_counter.most_common(3)
        lines.append(f"  event_id_history len={len(event_id_history)} top3={top3}")
    if combat_enemies_history:
        en_counter: Counter = Counter(combat_enemies_history)
        top3 = en_counter.most_common(3)
        lines.append(f"  combat_enemies_history len={len(combat_enemies_history)} top3={top3}")
    if action_history:
        ac_counter: Counter = Counter(action_history)
        top3 = ac_counter.most_common(3)
        lines.append(f"  action_history len={len(action_history)} top3={top3}")
    return "\n".join(lines)


def run_eval(
    trainer: V8PPOTrainer,
    env: V8Env,
    num_seeds: int,
    seed_offset: int,
) -> Dict[str, Any]:
    """跑 num_seeds 局 deterministic eval（no_grad）。

    指标：
    - reached_boss_rate：摸到 act1 boss 房（floor>=16 或 final_act>=2）
    - act1_boss_beat_rate：穿过 act1 boss（final_act>=2）
    - act2_boss_beat_rate：穿过 act2 boss（final_act>=3）
    - won_game_rate：通关（runner.game_won）
    - floor_mean / floor_max / avg_steps
    - boss_reach_counts / boss_kill_counts：act1 boss 维度 per-boss 统计
      （act 切换会覆盖 runner._boss_name，所以 act1 击杀的 boss 名无法在 episode 末尾
      可靠归属——这里仅在 final_act==1 且 final_floor>=16 时按 _boss_name 计 reach，
      kill 维度对应 boss 计 0；总 act1 击杀数走 act1_boss_beat_rate）

    保留旧字段 beat_boss_rate = act1_boss_beat_rate（deprecated，下游迁移完毕后删）。

    慢化修复（4 道防线，2026-05-13 加）：
    - Fix A: 外层 torch.no_grad() 显式包裹（collect_rollout 已有 @torch.no_grad，
             此处冗余兜底任何非 rollout 的 forward）。
    - Fix B: 每个 seed 完成后 torch.mps.empty_cache() 释放 MPS 内存碎片。
    - Fix C: eval 前后 restart_deck_pool() 重建 worker 池，防 worker 状态累积。
    - Fix D: [eval-mem] log 监控 RSS + MPS allocated，便于复现时定位累积曲线。
    """
    floors: List[int] = []
    reached_boss = 0
    act1_beat = 0
    act2_beat = 0
    won_game = 0
    total_steps = 0
    # ---- 阶段 3：每次 eval 测当前策略熵（熵崩第一时间看见）----
    # 对每个 seed 的 eval rollout 算策略熵，最后按 seed 求均值。
    entropy_sum = 0.0
    entropy_n = 0
    # per-boss act1 维度：见 docstring，仅 final_act==1 且 reach 时填
    boss_reach_counts: Dict[str, int] = {}
    boss_kill_counts: Dict[str, int] = {}

    # ----- 单 seed 超时策略（2026-05-19 重做 + 2026-05-19 加归因）-----
    # 旧逻辑：300s/600s 硬 kill，把所有深局（玩到 act3+ 通关线）误杀。
    # 中代：progress 停滞 5min → 直接 kill，但 user 反馈停滞不一定是死循环（可能 PPO 内部慢）。
    # 新逻辑（带归因）：
    #   - EVAL_SEED_LONG_WARN_SEC=600s：单 seed 跑了 10min 还没结束 → log warning（不 kill）
    #   - EVAL_SEED_STAGNATION_SEC=300s：progress 信号停滞 5min → 跑 attribution：
    #       * 命中 loop pattern（event_loop / combat_hang / action_mode_collapse）
    #         → [eval] hang_confirmed → kill
    #       * 未命中 → [eval] stagnation_no_loop dump → 给 EVAL_SEED_GRACE_SEC 宽限
    #   - EVAL_SEED_GRACE_SEC=300s：no_loop 后累计再等这么多 → 仍无 progress → kill (timeout)
    #   - EVAL_SEED_HARD_CAP_SEC=3600s：极端兜底，防止 progress 信号误判导致 seed 永跑
    # Progress 信号定义：env.runner.run_state 里 (floor, act, screen_type) 的 tuple，
    # 变化即视为有进度；env 内部 phase-transition loop / combat turn 都会更新这几个字段。
    EVAL_SEED_LONG_WARN_SEC = 600.0
    EVAL_SEED_STAGNATION_SEC = 300.0
    EVAL_SEED_GRACE_SEC = 300.0
    EVAL_SEED_HARD_CAP_SEC = 3600.0
    POLL_INTERVAL_SEC = 5.0  # 主线程轮询间隔

    # ---- Fix C: eval 前重建 deck_evaluator pool（防 worker 累积慢化）----
    try:
        restart_deck_pool()
    except Exception as e:  # noqa: BLE001
        logger.warning("[eval] restart_deck_pool (pre) failed: %s: %s", type(e).__name__, e)

    # ---- Fix A: 外层禁 grad（用 set_grad_enabled bookend 避免整段缩进改动）----
    # collect_rollout 已 @torch.no_grad，此处仅作冗余兜底（任何非 rollout forward 也覆盖）。
    _eval_prev_grad_enabled = torch.is_grad_enabled()
    torch.set_grad_enabled(False)

    for i in range(num_seeds):
        seed = seed_offset + i
        seed_t0 = time.time()
        # ---- Fix D: 入口 memory log ----
        _log_eval_mem("start", seed)
        logger.info("[eval] seed=%d start", seed)
        rollout: List[Any] = []
        exc_holder: Dict[str, Any] = {}

        def _run_rollout(_seed: int = seed) -> None:
            try:
                rollout.extend(
                    trainer.collect_rollout(env, seed=_seed, deterministic=True)
                )
            except Exception as e:  # noqa: BLE001
                exc_holder["exc"] = e

        t = threading.Thread(target=_run_rollout, daemon=True)
        t.start()

        # ---- Progress 监控 loop ----
        # 每 POLL_INTERVAL_SEC 短 join 一次；读 env.runner 的 (floor, act, screen_type)
        # 当作 progress signal；信号变化 → 重置 last_progress_ts。
        # 触发 long_running warning（一次性）+ hang_detected kill + hard_cap kill。
        def _read_progress_signal() -> tuple:
            """从 env 读一个 progress tuple；任何读失败返回 ('?',)。"""
            try:
                runner = env.runner
                if runner is None:
                    return ("no_runner",)
                rs = getattr(runner, "run_state", None)
                if rs is None:
                    return ("no_run_state",)
                floor = int(getattr(rs, "floor", 0) or 0)
                act = int(getattr(rs, "act", 1) or 1)
                screen = str(getattr(rs, "screen_type", "") or "")
                # phase 用 env 自己暴露的（如果有）
                phase = str(getattr(env, "_last_phase", "") or "")
                # combat 内：用 enemies 数量 + 玩家 hp 当 sub-progress（防战斗超时被误判 hang）
                player_hp = int(getattr(rs, "current_hp", 0) or 0)
                enemies_t = tuple(
                    sorted(
                        (e.monster_id, int(e.current_hp))
                        for e in (getattr(rs, "monsters_alive", []) or [])
                    )
                ) if hasattr(rs, "monsters_alive") else ()
                return (floor, act, screen, phase, player_hp, enemies_t)
            except Exception:  # noqa: BLE001
                return ("read_err",)

        seed_start_ts = time.time()
        last_progress_ts = seed_start_ts
        last_progress_sig = _read_progress_signal()
        long_warning_fired = False
        kill_reason: Optional[str] = None

        # ---- Attribution history（每次 poll 维护，stagnation 触发时分析）----
        # 直接 push 当前 snapshot——同一个值反复 push 即代表 mode collapse / hang。
        event_id_history: Deque[str] = deque(maxlen=50)
        combat_enemies_history: Deque[str] = deque(maxlen=20)
        action_history: Deque[str] = deque(maxlen=50)

        # Grace 期相关：第一次 stagnation 触发后 dump、设置 grace_deadline；
        # 之后只要 progress 重新推进就 reset；grace 内仍卡 → kill。
        grace_deadline: Optional[float] = None
        attribution_logged_once = False

        while t.is_alive():
            t.join(timeout=POLL_INTERVAL_SEC)
            if not t.is_alive():
                break

            now = time.time()
            elapsed = now - seed_start_ts

            # 读 progress 信号
            cur_sig = _read_progress_signal()
            if cur_sig != last_progress_sig:
                last_progress_sig = cur_sig
                last_progress_ts = now
                # progress 重新推进 → 清 grace（之前归因为 no_loop，现在恢复了）
                if grace_deadline is not None:
                    logger.info(
                        "[eval] seed=%d grace_recovered progress resumed elapsed=%.0fs",
                        seed, elapsed,
                    )
                    grace_deadline = None
                    attribution_logged_once = False
            stagnation = now - last_progress_ts

            # ---- 维护 attribution history ----
            # event_id：env 暴露的 _last_event_id；非 EVENT phase 时为 None
            ev_id = getattr(env, "_last_event_id", None)
            event_id_history.append(str(ev_id) if ev_id else "")
            # combat enemies：从 cur_sig 取（tuple 在最后一位）
            enemies_repr = ""
            if isinstance(cur_sig, tuple) and len(cur_sig) >= 6:
                try:
                    enemies_t = cur_sig[5]
                    if enemies_t:
                        enemies_repr = repr(enemies_t)
                except Exception:  # noqa: BLE001
                    pass
            combat_enemies_history.append(enemies_repr)
            # action：env 暴露的 _last_action_repr（最后一次 take_action）
            action_repr = str(getattr(env, "_last_action_repr", "") or "")
            action_history.append(action_repr)

            # 600s elapsed warning（一次性）
            if not long_warning_fired and elapsed >= EVAL_SEED_LONG_WARN_SEC:
                long_warning_fired = True
                floor_now = cur_sig[0] if isinstance(cur_sig[0], int) else "?"
                act_now = cur_sig[1] if len(cur_sig) > 1 and isinstance(cur_sig[1], int) else "?"
                logger.warning(
                    "[eval] seed=%d long_running elapsed=%.0fs floor=%s act=%s "
                    "(not kill—deep run, still progressing)",
                    seed, elapsed, floor_now, act_now,
                )

            # ---- Stagnation handling with attribution ----
            if stagnation >= EVAL_SEED_STAGNATION_SEC:
                floor_now = cur_sig[0] if isinstance(cur_sig[0], int) else "?"
                act_now = cur_sig[1] if len(cur_sig) > 1 and isinstance(cur_sig[1], int) else "?"
                loop_type, loop_detail = _classify_hang_pattern(
                    event_id_history, combat_enemies_history, action_history,
                )
                if loop_type is not None:
                    # 真死循环 → kill
                    dump = _attribution_dump(
                        env, event_id_history, combat_enemies_history, action_history,
                    )
                    logger.warning(
                        "[eval] seed=%d hang_confirmed type=%s %s "
                        "(stagnation=%.0fs elapsed=%.0fs floor=%s act=%s)\n%s",
                        seed, loop_type, loop_detail, stagnation, elapsed,
                        floor_now, act_now, dump,
                    )
                    kill_reason = f"hang_{loop_type}"
                    break

                # 未命中 loop → 归因为 no_loop，给 grace
                if not attribution_logged_once:
                    dump = _attribution_dump(
                        env, event_id_history, combat_enemies_history, action_history,
                    )
                    logger.warning(
                        "[eval] seed=%d stagnation_no_loop stagnation=%.0fs elapsed=%.0fs "
                        "floor=%s act=%s — granting %.0fs grace\n%s",
                        seed, stagnation, elapsed, floor_now, act_now,
                        EVAL_SEED_GRACE_SEC, dump,
                    )
                    attribution_logged_once = True
                    grace_deadline = now + EVAL_SEED_GRACE_SEC

                # Grace 已到 → kill
                if grace_deadline is not None and now >= grace_deadline:
                    logger.warning(
                        "[eval] seed=%d grace_expired stagnation=%.0fs elapsed=%.0fs "
                        "floor=%s act=%s — terminating seed (bg thread leaked)",
                        seed, stagnation, elapsed, floor_now, act_now,
                    )
                    kill_reason = "grace_expired"
                    break

            # 极硬上限
            if elapsed >= EVAL_SEED_HARD_CAP_SEC:
                logger.warning(
                    "[eval] seed=%d hard_cap_hit elapsed=%.0fs (>%.0fs) — terminating seed "
                    "(bg thread leaked)",
                    seed, elapsed, EVAL_SEED_HARD_CAP_SEC,
                )
                kill_reason = "hard_cap"
                break

        if t.is_alive():
            # Hang / hard_cap：放弃 thread，跳到下一个 seed（与旧实现一致，thread 隔离 leak）
            if torch.backends.mps.is_available():
                try:
                    torch.mps.empty_cache()
                except Exception:  # noqa: BLE001
                    pass
            _log_eval_mem(f"abort_{kill_reason}", seed)
            continue
        if "exc" in exc_holder:
            e = exc_holder["exc"]
            logger.warning(
                "eval seed=%d crashed: %s: %s", seed, type(e).__name__, e,
            )
            if torch.backends.mps.is_available():
                try:
                    torch.mps.empty_cache()
                except Exception:  # noqa: BLE001
                    pass
            _log_eval_mem("crash", seed)
            continue

        seed_elapsed = time.time() - seed_t0
        logger.info("[eval] seed=%d done elapsed=%.1fs steps=%d", seed, seed_elapsed, len(rollout))

        if not rollout:
            # 仍然清 cache + 出口 log，保持 per-seed 不变量
            if torch.backends.mps.is_available():
                try:
                    torch.mps.empty_cache()
                except Exception:  # noqa: BLE001
                    pass
            _log_eval_mem("empty", seed)
            continue
        # ---- 阶段 3：测本 seed 的策略熵，累加（最后按 seed 求均值）----
        try:
            seed_entropy = trainer.policy_entropy(rollout)
            entropy_sum += seed_entropy
            entropy_n += 1
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "[eval] seed=%d policy_entropy failed: %s: %s",
                seed, type(e).__name__, e,
            )

        last = rollout[-1]
        # last.state 是这一步**之前**的 state；需要从 env 拿最新
        runner = env.runner
        if runner is not None:
            final_floor = int(getattr(runner.run_state, "floor", 0) or 0)
            final_act = int(getattr(runner.run_state, "act", 1) or 1)
            game_won = bool(runner.game_won)
            boss_name = str(getattr(runner, "_boss_name", "") or "")
        else:
            final_floor = int(getattr(last.state, "floor", 0) or 0)
            final_act = int(getattr(last.state, "act", 1) or 1)
            game_won = False
            boss_name = ""

        floors.append(final_floor)
        total_steps += len(rollout)
        # act1 boss 在 floor 16+；过 boss 进 act2 即 final_act>=2 或 game_won
        if final_floor >= 16 or final_act >= 2:
            reached_boss += 1
        if game_won or final_act >= 2:
            act1_beat += 1
        if game_won or final_act >= 3:
            act2_beat += 1
        if game_won:
            won_game += 1

        # per-boss：只在 final_act==1 且摸到 boss 房（floor>=16）时归属。
        # final_act>=2 时 _boss_name 已被切到 act2 boss，无法可靠回溯 act1 boss 名。
        if final_act == 1 and final_floor >= 16 and boss_name:
            boss_reach_counts[boss_name] = boss_reach_counts.get(boss_name, 0) + 1
            # 摸到没击杀：kill 计 0（确保 key 存在便于 log 输出 X/Y 形式）
            boss_kill_counts.setdefault(boss_name, 0)

        try:
            env.close()
        except Exception:  # noqa: BLE001
            pass

        # ---- Fix B + D: 每 seed 结束后清 MPS cache + 打出口 mem log ----
        if torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "[eval] mps.empty_cache failed seed=%d: %s: %s",
                    seed, type(e).__name__, e,
                )
        _log_eval_mem("end", seed)

    # ---- Fix A: 还原 grad enable 状态（与 set_grad_enabled bookend 配对）----
    torch.set_grad_enabled(_eval_prev_grad_enabled)

    # ---- Fix C: eval 完成后再重建一次 pool（防把累积带回训练阶段）----
    try:
        restart_deck_pool()
    except Exception as e:  # noqa: BLE001
        logger.warning("[eval] restart_deck_pool (post) failed: %s: %s", type(e).__name__, e)

    n = max(1, len(floors))
    act1_beat_rate = act1_beat / num_seeds
    return {
        "num_seeds": num_seeds,
        "completed": len(floors),
        "reached_boss_rate": reached_boss / num_seeds,
        "act1_boss_beat_rate": act1_beat_rate,
        "act2_boss_beat_rate": act2_beat / num_seeds,
        "won_game_rate": won_game / num_seeds,
        # 阶段 0 拆出的分层击杀指标（won_game_rate 是北极星，a1/a2 当过滤器）。
        # eval 侧 act1/act2 boss "beat" = 穿过该 act boss（进入下一 act 或通关），
        # 与 heartbeat 里 a1_boss_killed/a2_boss_killed 同口径。
        "a1_boss_killed_rate": act1_beat_rate,
        "a2_boss_killed_rate": act2_beat / num_seeds,
        # deprecated, use act1_boss_beat_rate
        "beat_boss_rate": act1_beat_rate,
        "floor_mean": sum(floors) / n if floors else 0.0,
        "floor_max": max(floors) if floors else 0,
        "avg_steps": total_steps / n if floors else 0.0,
        "boss_reach_counts": boss_reach_counts,
        "boss_kill_counts": boss_kill_counts,
        # ---- 阶段 3：当前策略熵（崩了第一时间看见）----
        "entropy": entropy_sum / entropy_n if entropy_n > 0 else 0.0,
    }


# ============================================================
# Main
# ============================================================


_PARSER_DEFAULTS: Dict[str, Any] = {
    "num_episodes": 1000,
    "batch_size": 32,
    "lr": 3e-4,
    "device": "mps",
    "combat_head_checkpoint": "sts_models/v8_combat_head_v1.pt",
    "output_dir": "sts_models/v8_ppo_rl",
    "eval_frequency": 100,
    "eval_seeds": 30,
    "checkpoint_frequency": 500,
    "max_steps_per_episode": 1500,
    "deck_eval_freq": 10,
    "n_envs": 1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V8 Phase B PPO 训练入口")
    parser.add_argument("--num_episodes", type=int, default=_PARSER_DEFAULTS["num_episodes"])
    parser.add_argument("--batch_size", type=int, default=_PARSER_DEFAULTS["batch_size"],
                        help="每多少 episodes 做一次 PPO update")
    parser.add_argument("--lr", type=float, default=_PARSER_DEFAULTS["lr"])
    parser.add_argument("--device", type=str, default=_PARSER_DEFAULTS["device"],
                        help="mps / cpu / cuda")
    parser.add_argument(
        "--combat_head_checkpoint",
        type=str,
        default=_PARSER_DEFAULTS["combat_head_checkpoint"],
    )
    parser.add_argument("--output_dir", type=str, default=_PARSER_DEFAULTS["output_dir"])
    parser.add_argument("--eval_frequency", type=int, default=_PARSER_DEFAULTS["eval_frequency"])
    parser.add_argument("--eval_seeds", type=int, default=_PARSER_DEFAULTS["eval_seeds"])
    parser.add_argument("--checkpoint_frequency", type=int, default=_PARSER_DEFAULTS["checkpoint_frequency"])
    parser.add_argument(
        "--max_steps_per_episode",
        type=int,
        default=_PARSER_DEFAULTS["max_steps_per_episode"],
        help="V8Env 单局最多 RL step 数（default=1500）；超时 force game_over 兜底"
             "（深局拖太久会浪费 wall time，1500 step 足够覆盖正常深 run）。smoke 会自动覆盖成 30。",
    )
    parser.add_argument(
        "--deck_eval_freq",
        type=int,
        default=_PARSER_DEFAULTS["deck_eval_freq"],
        help="每 N 次 post-step evaluate_deck 才真跑一次，中间复用上次 deck_strength"
             "（reward shaping signal 略微稀疏化，换加速）。default=10。",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="管线验证：覆盖小 episode/batch/freq + 写临时 output_dir + 限制 max_steps_per_episode",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="续训：从指定 ckpt 加载 model+optimizer，episode 计数从 ckpt metadata.episodes_done 继续。"
             "不指定则从随机/Phase-A combat head fresh init 开始。",
    )
    parser.add_argument(
        "--n_envs",
        type=int,
        default=_PARSER_DEFAULTS["n_envs"],
        help="并行 rollout collection 子进程数。1=旧串行行为（V8Env + combat_net_wrapper）；"
             ">1=ParallelV8Env，每步 broadcast 给 N 个子进程并行收 episode，主进程仍单 model"
             " forward。子进程内 combat_net_wrapper=None（regression：战斗 leaf 走纯搜索）。"
             " eval 路径保持串行（用 V8Env），不受影响。",
    )
    return parser.parse_args()


def apply_smoke_overrides(args: argparse.Namespace, *, defaults: Optional[Dict[str, Any]] = None) -> None:
    """smoke 模式：强制小 episode + 临时输出目录 + 卡每局 RL 步数。

    单局封顶 30 步（够覆盖几层 meta 决策 + 几场战斗），总耗时控制在 30min 内。

    若用户显式传了 --num_episodes / --batch_size / --eval_frequency /
    --checkpoint_frequency / --output_dir，则 smoke override 不再强行覆盖
    （方便 smoke-resume 之类的小型自定义场景）。defaults 用来识别 "是不是用户显式传的"。
    """
    d = defaults or {}
    def _is_default(name: str) -> bool:
        return getattr(args, name) == d.get(name)

    if _is_default("num_episodes"):
        args.num_episodes = 10
    if _is_default("batch_size"):
        args.batch_size = 2
    if _is_default("eval_frequency"):
        args.eval_frequency = 5
    if _is_default("eval_seeds"):
        args.eval_seeds = 2  # smoke eval 也只跑很少几个种子
    if _is_default("checkpoint_frequency"):
        args.checkpoint_frequency = 5
    # smoke：max_steps_per_episode 强制 30（兼容 None 或仍是 default 1500 两种情况）
    if args.max_steps_per_episode is None or _is_default("max_steps_per_episode"):
        args.max_steps_per_episode = 30
    if _is_default("output_dir"):
        smoke_root = _REPO_ROOT / "sts_models" / "v8_ppo_smoke"
        smoke_root.mkdir(parents=True, exist_ok=True)
        args.output_dir = str(smoke_root / f"run_{int(time.time())}")
    logger.info(
        "smoke 模式：output_dir=%s max_steps_per_episode=%d num_episodes=%d "
        "batch_size=%d eval_freq=%d ckpt_freq=%d",
        args.output_dir, args.max_steps_per_episode,
        args.num_episodes, args.batch_size,
        args.eval_frequency, args.checkpoint_frequency,
    )


def main() -> None:
    args = parse_args()
    if args.smoke:
        apply_smoke_overrides(args, defaults=_PARSER_DEFAULTS)

    # 安装 SIGINT handler（必须在 heavy import / 训练循环开始前）
    _install_sigint_handler()

    device = select_device(args.device)
    logger.info("device=%s", device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Self-check：output_dir 已有 ckpt 但没给 --resume_from，警告用户 ----
    # 让"以为是续训实际从零重训"的 concept error 在 log 第一行就能看到。
    # 不 abort，因为可能 user 故意 fresh restart 但写了已有目录（如 v2b→long_v2b_round2）；
    # 只是显式提示。
    try:
        existing_ckpts = [
            f for f in os.listdir(args.output_dir) if f.endswith(".pt")
        ] if os.path.exists(args.output_dir) else []
    except Exception:  # noqa: BLE001
        existing_ckpts = []
    if existing_ckpts and not args.resume_from:
        logger.warning(
            "[resume-check] output_dir=%s 已有 %d 个 .pt ckpt 但未传 --resume_from。"
            "本次训练将 fresh init（仅 Phase A combat head + 随机 RL 权重），"
            "不会继承既有 ckpt。如果是想续训请传 --resume_from=<ckpt_path>。",
            args.output_dir, len(existing_ckpts),
        )
        logger.warning(
            "[resume-check] existing ckpts (前 5): %s",
            existing_ckpts[:5],
        )

    # 1) Model + 加载 Phase A combat head
    model = V8Model()
    phase_a_meta = load_combat_head(model, args.combat_head_checkpoint)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "combat head loaded OK: ckpt=%s params=%d (%.2fM)",
        args.combat_head_checkpoint, n_params, n_params / 1e6,
    )

    # 2) Wrapper（让 model 在战斗 search leaf 评估时被调用）
    wrapper = V8CombatNetWrapper(model)

    # 3) Env
    env_kwargs: Dict[str, Any] = {"combat_net_wrapper": wrapper}
    if args.max_steps_per_episode is not None:
        env_kwargs["max_steps_per_episode"] = int(args.max_steps_per_episode)
    if args.deck_eval_freq is not None:
        env_kwargs["deck_eval_freq"] = int(args.deck_eval_freq)
    env = V8Env(**env_kwargs)

    # ---- Parallel env (rollout 加速)：n_envs > 1 时启动 ----
    # 注意：ParallelV8Env 子进程内 combat_net_wrapper 被强制 None（regression：
    # 战斗 leaf 评估退到纯搜索）。详见 docs/parallel_env_design.md。
    # eval 路径继续用主进程 env（V8Env + wrapper），保证 eval 与 ckpt 评估一致。
    parallel_env: Optional[ParallelV8Env] = None
    if int(args.n_envs) > 1:
        # ParallelV8Env 子进程不能 pickle wrapper（含 MPS V8Model）
        parallel_env_kwargs = {k: v for k, v in env_kwargs.items() if k != "combat_net_wrapper"}
        parallel_env_kwargs["combat_net_wrapper"] = None
        parallel_env = ParallelV8Env(
            n_envs=int(args.n_envs),
            env_kwargs=parallel_env_kwargs,
            # n_envs=4 + per_env=2 → 4 worker + 4×2 = 12 子进程，控总数
            deck_eval_workers_per_env=2,
        )
        logger.info(
            "[parallel] enabled n_envs=%d (rollout collection 走 ParallelV8Env, "
            "战斗 leaf eval 退到纯搜索；eval 路径仍走串行 V8Env+wrapper)",
            int(args.n_envs),
        )

    # 4) Trainer
    trainer = V8PPOTrainer(model=model, lr=args.lr, device=str(device))

    # ---- Resume：从指定 ckpt 加载 model+optimizer state，恢复 ep 计数 ----
    # ckpt metadata 历史上字段名不统一（episodes_done 或 episode 或 num_episodes_so_far），
    # 三个 key 都试一遍。
    start_episode = 0
    if args.resume_from:
        if not os.path.exists(args.resume_from):
            raise FileNotFoundError(f"--resume_from ckpt not found: {args.resume_from}")
        meta = trainer.load_checkpoint(args.resume_from)
        for k in ("episodes_done", "episode", "num_episodes_so_far"):
            if k in meta:
                start_episode = int(meta[k] or 0)
                logger.info(
                    "[resume] reading start_episode from metadata['%s']=%d",
                    k, start_episode,
                )
                break
        else:
            logger.warning(
                "[resume] ckpt %s metadata 中找不到 episodes_done/episode/"
                "num_episodes_so_far；start_episode 默认 0",
                args.resume_from,
            )
        logger.info(
            "[resume] loaded ckpt=%s, start_episode=%d, target=args.num_episodes=%d "
            "(将增量训 %d ep)",
            args.resume_from, start_episode, args.num_episodes,
            max(0, args.num_episodes - start_episode),
        )
        if start_episode >= args.num_episodes:
            logger.warning(
                "[resume] start_episode=%d >= --num_episodes=%d；不会再训。"
                "如要继续训，请把 --num_episodes 调大。",
                start_episode, args.num_episodes,
            )

    # ---- 启动期 banner，保证用户看得到所有关键 config ----
    logger.info(
        "[startup] output_dir=%s num_episodes=%d batch_size=%d eval_freq=%d ckpt_freq=%d "
        "max_steps_per_episode=%s smoke=%s resume_from=%s start_episode=%d",
        args.output_dir, args.num_episodes, args.batch_size,
        args.eval_frequency, args.checkpoint_frequency,
        args.max_steps_per_episode, args.smoke,
        args.resume_from, start_episode,
    )

    # 5) 训练 loop
    eval_history: List[Dict[str, Any]] = []
    train_log: List[Dict[str, Any]] = []
    num_episodes_done = start_episode
    t_start = time.time()

    # ----- milestone trackers（修复以前 % freq == 0 与 batch_size 不整除导致永不触发的 bug）-----
    # 旧逻辑：num_episodes_done % checkpoint_frequency == 0
    #   batch_size=32 / freq=500 → LCM=4000，1000 局训练永远 fire 不了
    # 新逻辑：num_episodes_done // freq 越过上一里程碑就 fire
    # resume 时：里程碑从 start_episode 对应的位置起算，避免立刻 fire 重复 ckpt
    last_ckpt_milestone = start_episode // args.checkpoint_frequency if args.checkpoint_frequency > 0 else 0
    last_eval_milestone = start_episode // args.eval_frequency if args.eval_frequency > 0 else 0

    # ----- 5h 墙钟 ckpt：episode-based ckpt 万一失效（bug / 进程僵死）也能保底 -----
    wall_ckpt_interval_sec = 5 * 3600  # 5 hours
    last_wall_ckpt_time = time.time()

    logger.info(
        "PPO 训练开始: num_episodes=%d batch_size=%d lr=%.2e eval_freq=%d ckpt_freq=%d "
        "wall_ckpt_interval=%ds (~%.1fh)",
        args.num_episodes, args.batch_size, args.lr,
        args.eval_frequency, args.checkpoint_frequency,
        wall_ckpt_interval_sec, wall_ckpt_interval_sec / 3600.0,
    )

    # ----- 退出原因（供最终 summary 标识，区分正常完成 vs SIGINT 软停）-----
    exit_reason = "completed"

    def _make_final_summary_payload() -> Dict[str, Any]:
        return {
            "phase": "B_ppo_rl",
            "args": vars(args),
            "episodes_done": num_episodes_done,
            "elapsed_sec": time.time() - t_start,
            "train_log_tail": train_log[-10:],
            "eval_history": eval_history,
            "wrapper_total_calls": wrapper.call_count,
            "wrapper_avg_value": wrapper.avg_value,
            "exit_reason": exit_reason,
        }

    def _save_final_checkpoint_and_summary() -> None:
        """统一保存 final ckpt + summary（正常完成 / SIGINT 软停都走这条路径）。"""
        final_ckpt_local = output_dir / "v8_ppo_final.pt"
        try:
            trainer.save_checkpoint(
                str(final_ckpt_local),
                metadata={
                    "phase": "B_ppo_rl_final",
                    "episodes_done": num_episodes_done,
                    "phase_a_meta": phase_a_meta,
                    "elapsed_sec": time.time() - t_start,
                    "exit_reason": exit_reason,
                },
            )
        except Exception as e:  # noqa: BLE001
            logger.error(
                "final ckpt save failed: %s: %s", type(e).__name__, e,
            )
        try:
            with open(output_dir / "v8_ppo_summary.json", "w") as f:
                json.dump(_make_final_summary_payload(), f, indent=2, default=str, ensure_ascii=False)
        except Exception as e:  # noqa: BLE001
            logger.error(
                "summary write failed: %s: %s", type(e).__name__, e,
            )
        logger.info(
            "训练结束 reason=%s episodes=%d 总耗时=%.1fs final_ckpt=%s wrapper_calls=%d",
            exit_reason, num_episodes_done, time.time() - t_start,
            final_ckpt_local, wrapper.call_count,
        )

    while num_episodes_done < args.num_episodes:
        # ---- 收 batch_size 个 rollout ----
        batch_rollouts: List[RolloutStep] = []
        batch_meta: List[Dict[str, Any]] = []
        batch_target = min(args.batch_size, args.num_episodes - num_episodes_done)
        prev_call_count = wrapper.call_count

        # ---- 性能累计（[perf] 行用）----
        batch_env_step_sec = 0.0
        batch_collect_fwd_sec = 0.0
        batch_eval_deck_sec = 0.0
        batch_eval_deck_calls = 0
        batch_search_calls = 0
        # deck_evaluator 全局 cache 计数 delta
        prev_cache = get_deck_cache_stats()
        prev_cache_hits = int(prev_cache.get("hits", 0) or 0)
        prev_cache_misses = int(prev_cache.get("misses", 0) or 0)

        if parallel_env is not None and batch_target >= parallel_env.n_envs:
            # ----------- 并行 rollout 路径（n_envs > 1）-----------
            n_envs_eff = parallel_env.n_envs
            # 一次收 n_envs 个 episode，多收几轮直到 batch_target
            n_rounds = batch_target // n_envs_eff
            remainder = batch_target - n_rounds * n_envs_eff
            if remainder != 0:
                # 简化：把 remainder 折成串行的 collect_rollout，不影响 batch
                logger.info(
                    "[parallel] batch_target=%d not multiple of n_envs=%d; "
                    "%d round(s) parallel + %d serial tail",
                    batch_target, n_envs_eff, n_rounds, remainder,
                )

            for round_idx in range(n_rounds):
                round_t0 = time.time()
                seeds = [
                    num_episodes_done + round_idx * n_envs_eff + k
                    for k in range(n_envs_eff)
                ]
                # set_episode 给每个子进程（guard_cap / [combat] 日志带 episode 标识）
                try:
                    parallel_env.set_episode(seeds)
                except Exception:  # noqa: BLE001
                    pass

                try:
                    sub_rollouts = trainer.collect_rollout_batched(
                        parallel_env, seeds=seeds, deterministic=False,
                    )
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "[parallel] rollout batched crashed seeds=%s: %s: %s (skip round)",
                        seeds, type(e).__name__, e,
                    )
                    continue

                round_secs = time.time() - round_t0
                rstats = getattr(trainer, "last_rollout_stats", {}) or {}
                batch_collect_fwd_sec += float(rstats.get("forward_time_sec", 0.0))
                batch_env_step_sec += float(rstats.get("env_step_time_sec", 0.0))

                # 跨进程一次性拿 runner snapshot 填 final_floor / beat_boss 等真值。
                # 单次广播 N envs 的额外开销 ~ms 级，相对 rollout 耗时可忽略。
                try:
                    snapshots = parallel_env.get_runner_snapshots()
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "[parallel] get_runner_snapshots failed: %s: %s (fallback 0/False)",
                        type(e).__name__, e,
                    )
                    snapshots = [
                        {"floor": 0, "act": 1, "game_won": False}
                        for _ in range(len(sub_rollouts))
                    ]

                # 每个 sub-rollout 也打 heartbeat（subprocess 内 [combat] 等日志已分别写 stderr）
                for i, rollout in enumerate(sub_rollouts):
                    ep_idx = seeds[i]
                    if not rollout:
                        logger.info(
                            "[heartbeat] ep=%d (parallel) empty rollout, skip",
                            ep_idx,
                        )
                        continue
                    ep_reward = sum(s.reward for s in rollout)
                    snap = snapshots[i] if i < len(snapshots) else {}
                    final_floor = int(snap.get("floor", 0) or 0)
                    final_act = int(snap.get("act", 1) or 1)
                    won_game = bool(snap.get("game_won", False))
                    # 阶段 0：拆诚实字段。a1_boss_killed=过第一幕 boss（进 act2,
                    # final_act>=2 或通关）；a2_boss_killed=过第二幕 boss（act>=3 或通关）；
                    # won_game=通关。beat_boss 保留为废弃别名 = won_game（web/etl 仍消费）。
                    a1_boss_killed = bool(won_game or final_act >= 2)
                    a2_boss_killed = bool(won_game or final_act >= 3)
                    beat_boss = won_game
                    batch_rollouts.extend(rollout)
                    batch_meta.append({
                        "ep": ep_idx,
                        "steps": len(rollout),
                        "reward_sum": ep_reward,
                        "final_floor": final_floor,
                        "final_act": final_act,
                        "a1_boss_killed": a1_boss_killed,
                        "a2_boss_killed": a2_boss_killed,
                        "won_game": won_game,
                        "beat_boss": beat_boss,  # deprecated 别名 = won_game
                        "secs": round_secs / n_envs_eff,  # 摊到每 ep
                    })
                    logger.info(
                        "[heartbeat] ep=%d (parallel) steps=%d round_secs=%.1f reward=%.3f "
                        "floor=%d a1_boss_killed=%s a2_boss_killed=%s won_game=%s beat_boss=%s",
                        ep_idx, len(rollout), round_secs, ep_reward,
                        final_floor, a1_boss_killed, a2_boss_killed, won_game, beat_boss,
                    )

            # ----------- 串行尾巴（remainder）-----------
            for k in range(remainder):
                ep_idx = num_episodes_done + n_rounds * n_envs_eff + k
                ep_t0 = time.time()
                try:
                    env.set_episode(ep_idx)
                except Exception:  # noqa: BLE001
                    pass
                try:
                    rollout = trainer.collect_rollout(env, seed=ep_idx, deterministic=False)
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "rollout ep=%d (tail) crashed: %s: %s (skip)",
                        ep_idx, type(e).__name__, e,
                    )
                    continue
                ep_reward = sum(s.reward for s in rollout)
                runner = env.runner
                final_floor = int(getattr(runner.run_state, "floor", 0) or 0) if runner else 0
                final_act = int(getattr(runner.run_state, "act", 1) or 1) if runner else 1
                won_game = bool(runner.game_won) if runner else False
                # 阶段 0：拆诚实字段（见 parallel 分支注释）
                a1_boss_killed = bool(won_game or final_act >= 2)
                a2_boss_killed = bool(won_game or final_act >= 3)
                beat_boss = won_game  # deprecated 别名
                ep_secs = time.time() - ep_t0
                rstats = getattr(trainer, "last_rollout_stats", {}) or {}
                batch_collect_fwd_sec += float(rstats.get("forward_time_sec", 0.0))
                batch_env_step_sec += float(rstats.get("env_step_time_sec", 0.0))
                batch_eval_deck_sec += float(rstats.get("eval_deck_time_sec", 0.0))
                batch_eval_deck_calls += int(rstats.get("eval_deck_calls", 0))
                batch_search_calls += int(rstats.get("combat_search_calls", 0))
                batch_rollouts.extend(rollout)
                batch_meta.append({
                    "ep": ep_idx,
                    "steps": len(rollout),
                    "reward_sum": ep_reward,
                    "final_floor": final_floor,
                    "final_act": final_act,
                    "a1_boss_killed": a1_boss_killed,
                    "a2_boss_killed": a2_boss_killed,
                    "won_game": won_game,
                    "beat_boss": beat_boss,  # deprecated 别名 = won_game
                    "secs": ep_secs,
                })
                logger.info(
                    "[heartbeat] ep=%d (tail-serial) steps=%d secs=%.1f reward=%.3f "
                    "floor=%d a1_boss_killed=%s a2_boss_killed=%s won_game=%s beat_boss=%s",
                    ep_idx, len(rollout), ep_secs, ep_reward, final_floor,
                    a1_boss_killed, a2_boss_killed, won_game, beat_boss,
                )
                try:
                    env.close()
                except Exception:  # noqa: BLE001
                    pass
        else:
            # ----------- 串行 rollout 路径（旧逻辑 / n_envs=1）-----------
            for k in range(batch_target):
                ep_idx = num_episodes_done + k
                ep_t0 = time.time()
                # 把 episode 编号告诉 env，guard_cap / [combat] 日志带上
                try:
                    env.set_episode(ep_idx)
                except Exception:  # noqa: BLE001
                    pass
                try:
                    rollout = trainer.collect_rollout(env, seed=ep_idx, deterministic=False)
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "rollout ep=%d crashed: %s: %s (skip)",
                        ep_idx, type(e).__name__, e,
                    )
                    # crash 也打 heartbeat，方便定位卡顿点
                    logger.info(
                        "[heartbeat] ep=%d CRASHED after %.1fs (%s)",
                        ep_idx, time.time() - ep_t0, type(e).__name__,
                    )
                    continue

                ep_reward = sum(s.reward for s in rollout)
                runner = env.runner
                final_floor = int(getattr(runner.run_state, "floor", 0) or 0) if runner else 0
                final_act = int(getattr(runner.run_state, "act", 1) or 1) if runner else 1
                won_game = bool(runner.game_won) if runner else False
                # 阶段 0：拆诚实字段（见 parallel 分支注释）
                a1_boss_killed = bool(won_game or final_act >= 2)
                a2_boss_killed = bool(won_game or final_act >= 3)
                beat_boss = won_game  # deprecated 别名
                ep_secs = time.time() - ep_t0

                # 拿 per-episode 性能 / 调用计数
                rstats = getattr(trainer, "last_rollout_stats", {}) or {}
                ep_fwd = float(rstats.get("forward_time_sec", 0.0))
                ep_env_step = float(rstats.get("env_step_time_sec", 0.0))
                ep_eval_deck_sec = float(rstats.get("eval_deck_time_sec", 0.0))
                ep_eval_deck_calls = int(rstats.get("eval_deck_calls", 0))
                ep_search_calls = int(rstats.get("combat_search_calls", 0))
                batch_collect_fwd_sec += ep_fwd
                batch_env_step_sec += ep_env_step
                batch_eval_deck_sec += ep_eval_deck_sec
                batch_eval_deck_calls += ep_eval_deck_calls
                batch_search_calls += ep_search_calls

                batch_rollouts.extend(rollout)
                batch_meta.append({
                    "ep": ep_idx,
                    "steps": len(rollout),
                    "reward_sum": ep_reward,
                    "final_floor": final_floor,
                    "final_act": final_act,
                    "a1_boss_killed": a1_boss_killed,
                    "a2_boss_killed": a2_boss_killed,
                    "won_game": won_game,
                    "beat_boss": beat_boss,  # deprecated 别名 = won_game
                    "secs": ep_secs,
                })
                # ---- 每局 heartbeat：silent 跑步是不可接受的 ----
                # 注意：beat_boss=%s 字段保留在原位置（web/etl/markers.py 正则依赖），
                # 阶段 0 新增的 a1/a2/won_game 追加在 search_calls 之前。
                logger.info(
                    "[heartbeat] ep=%d steps=%d secs=%.1f reward=%.3f floor=%d act=%d beat_boss=%s "
                    "a1_boss_killed=%s a2_boss_killed=%s won_game=%s "
                    "search_calls=%d eval_deck_calls=%d",
                    ep_idx, len(rollout), ep_secs, ep_reward,
                    final_floor, final_act, beat_boss,
                    a1_boss_killed, a2_boss_killed, won_game,
                    ep_search_calls, ep_eval_deck_calls,
                )
                try:
                    env.close()
                except Exception:  # noqa: BLE001
                    pass

        if not batch_rollouts:
            logger.warning("batch 全部 crash，跳过 update")
            num_episodes_done += batch_target
            continue

        # ---- PPO update ----
        upd_t0 = time.time()
        metrics = trainer.update(batch_rollouts, num_epochs=4)
        upd_secs = time.time() - upd_t0

        # ---- batch 日志 ----
        ep_steps = [m["steps"] for m in batch_meta]
        ep_rewards = [m["reward_sum"] for m in batch_meta]
        ep_floors = [m["final_floor"] for m in batch_meta]
        n_beat = sum(1 for m in batch_meta if m["beat_boss"])
        n_a1_killed = sum(1 for m in batch_meta if m.get("a1_boss_killed"))
        n_a2_killed = sum(1 for m in batch_meta if m.get("a2_boss_killed"))
        n_won = sum(1 for m in batch_meta if m.get("won_game"))
        wrapper_calls_in_batch = wrapper.call_count - prev_call_count

        num_episodes_done += batch_target
        log_entry = {
            "episodes_done": num_episodes_done,
            "batch_size": len(batch_meta),
            "mean_reward": sum(ep_rewards) / max(1, len(ep_rewards)),
            "mean_steps": sum(ep_steps) / max(1, len(ep_steps)),
            "mean_floor": sum(ep_floors) / max(1, len(ep_floors)),
            "beat_boss_count": n_beat,  # deprecated 别名 = won_game_count
            "a1_boss_killed_count": n_a1_killed,
            "a2_boss_killed_count": n_a2_killed,
            "won_game_count": n_won,
            "policy_loss": metrics.get("policy_loss", 0.0),
            "value_loss": metrics.get("value_loss", 0.0),
            "entropy": metrics.get("entropy", 0.0),
            "approx_kl": metrics.get("approx_kl", 0.0),
            "clip_frac": metrics.get("clip_frac", 0.0),
            # 阶段 3：自适应熵系数（本批用的 coef + 据本批熵调出的下批 coef）
            "entropy_coef": metrics.get("entropy_coef", 0.0),
            "entropy_coef_next": metrics.get("entropy_coef_next", 0.0),
            "update_secs": upd_secs,
            "wrapper_calls": wrapper_calls_in_batch,
        }
        train_log.append(log_entry)
        logger.info(
            "[ep=%d] reward_mean=%.3f steps_mean=%.1f floor_mean=%.1f beat=%d/%d "
            "policy=%.4f value=%.4f entropy=%.4f ent_coef=%.4f->%.4f kl=%.4f clip=%.3f "
            "wrapper_calls=%d upd=%.2fs",
            num_episodes_done, log_entry["mean_reward"], log_entry["mean_steps"],
            log_entry["mean_floor"], n_beat, len(batch_meta),
            log_entry["policy_loss"], log_entry["value_loss"], log_entry["entropy"],
            log_entry["entropy_coef"], log_entry["entropy_coef_next"],
            log_entry["approx_kl"], log_entry["clip_frac"],
            wrapper_calls_in_batch, upd_secs,
        )

        # ---- [perf] 性能分解（debug 用：定位是 CPU search / MPS forward / deck_eval 哪个 bottleneck）----
        upd_stats = getattr(trainer, "last_update_stats", {}) or {}
        upd_fwd_sec = float(upd_stats.get("update_forward_sec", 0.0))
        upd_total_sec = float(upd_stats.get("update_total_sec", upd_secs))
        # deck_evaluator cache hit/miss delta（本 batch 期间新增的 hits/misses）
        cur_cache = get_deck_cache_stats()
        cur_cache_hits = int(cur_cache.get("hits", 0) or 0)
        cur_cache_misses = int(cur_cache.get("misses", 0) or 0)
        d_cache_hits = cur_cache_hits - prev_cache_hits
        d_cache_misses = cur_cache_misses - prev_cache_misses
        batch_ep_total_sec = sum(m["secs"] for m in batch_meta)
        logger.info(
            "[perf] ep=%d batch_total=%.1fs collect_ep_sum=%.1fs env_step=%.1fs "
            "collect_fwd=%.1fs eval_deck=%.1fs(%d calls) search_calls=%d "
            "ppo_update=%.2fs(fwd=%.2fs) deck_cache_delta=hit%d/miss%d",
            num_episodes_done,
            batch_ep_total_sec + upd_total_sec,
            batch_ep_total_sec,
            batch_env_step_sec,
            batch_collect_fwd_sec,
            batch_eval_deck_sec, batch_eval_deck_calls,
            batch_search_calls,
            upd_total_sec, upd_fwd_sec,
            d_cache_hits, d_cache_misses,
        )

        # ---- Checkpoint 先于 Eval（修复：batch_v5_resume 教训）----
        # 之前顺序是 eval → ckpt：当 eval 在某 seed inference 慢化 600x 卡死时，
        # ckpt 永远落不了盘，5h+ 训练权重全部丢失。
        # 现在顺序：先 save ckpt（即便 eval 卡死，ckpt 已在磁盘上），再 run eval。
        if args.checkpoint_frequency > 0:
            cur_ckpt_milestone = num_episodes_done // args.checkpoint_frequency
            if cur_ckpt_milestone > last_ckpt_milestone:
                last_ckpt_milestone = cur_ckpt_milestone
                ckpt_path = output_dir / f"v8_ppo_ep{num_episodes_done}.pt"
                trainer.save_checkpoint(
                    str(ckpt_path),
                    metadata={
                        "phase": "B_ppo_rl",
                        "episodes_done": num_episodes_done,
                        "lr": args.lr,
                        "batch_size": args.batch_size,
                        "phase_a_meta": phase_a_meta,
                        "elapsed_sec": time.time() - t_start,
                    },
                )
                save_metadata_json(
                    str(output_dir / f"v8_ppo_ep{num_episodes_done}.json"),
                    num_episodes_so_far=num_episodes_done,
                    eval_history=eval_history,
                    extra={"args": vars(args)},
                )
                logger.info("[ckpt] saved before eval: %s", ckpt_path)

        # ---- Eval（修复：里程碑递进，不靠 % freq == 0；并放在 ckpt 之后）----
        if args.eval_frequency > 0:
            cur_eval_milestone = num_episodes_done // args.eval_frequency
            if cur_eval_milestone > last_eval_milestone:
                last_eval_milestone = cur_eval_milestone
                eval_t0 = time.time()
                # eval 用大 seed offset，跟训练 seed 不冲突
                eval_metrics = run_eval(
                    trainer, env,
                    num_seeds=args.eval_seeds,
                    seed_offset=10_000 + num_episodes_done,
                )
                eval_metrics["episodes_done"] = num_episodes_done
                eval_metrics["secs"] = time.time() - eval_t0
                eval_history.append(eval_metrics)
                logger.info(
                    "[eval@ep=%d] reached_a1_boss=%.2f a1_boss_beat=%.2f a2_boss_beat=%.2f "
                    "won_game=%.2f floor_mean=%.1f entropy=%.4f (%.1fs)",
                    num_episodes_done, eval_metrics["reached_boss_rate"],
                    eval_metrics["act1_boss_beat_rate"], eval_metrics["act2_boss_beat_rate"],
                    eval_metrics["won_game_rate"], eval_metrics["floor_mean"],
                    eval_metrics.get("entropy", 0.0),
                    eval_metrics["secs"],
                )
                # per-boss reach / kill 分布（仅 act1，act 切换后 _boss_name 被覆盖，
                # 故 kill 列无法按 boss 归属，恒为 0；reach 列反映"卡在哪个 boss"分布）
                _reach = eval_metrics.get("boss_reach_counts", {}) or {}
                _kill = eval_metrics.get("boss_kill_counts", {}) or {}
                if _reach or _kill:
                    _names = sorted(set(_reach.keys()) | set(_kill.keys()))
                    _parts = [
                        f"{name}={_kill.get(name, 0)}/{_reach.get(name, 0)}"
                        for name in _names
                    ]
                    logger.info(
                        "[eval@ep=%d] boss_kills: %s",
                        num_episodes_done, ", ".join(_parts),
                    )

        # ---- 5h 墙钟 ckpt（用户硬性要求：episode-ckpt 失效也得有保底）----
        now = time.time()
        if now - last_wall_ckpt_time >= wall_ckpt_interval_sec:
            wall_runtime_h = (now - t_start) / 3600.0
            ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
            wall_ckpt_path = output_dir / f"v8_ppo_wall_{ts_tag}.pt"
            try:
                trainer.save_checkpoint(
                    str(wall_ckpt_path),
                    metadata={
                        "phase": "B_ppo_rl_wall",
                        "episodes_done": num_episodes_done,
                        "lr": args.lr,
                        "batch_size": args.batch_size,
                        "phase_a_meta": phase_a_meta,
                        "elapsed_sec": now - t_start,
                        "wall_runtime_hours": wall_runtime_h,
                        "wall_ckpt_timestamp": ts_tag,
                    },
                )
                save_metadata_json(
                    str(output_dir / f"v8_ppo_wall_{ts_tag}.json"),
                    num_episodes_so_far=num_episodes_done,
                    eval_history=eval_history,
                    extra={
                        "args": vars(args),
                        "wall_runtime_hours": wall_runtime_h,
                        "kind": "wall_ckpt",
                    },
                )
                logger.info(
                    "[wall-ckpt] saved at runtime=%.2fh path=%s",
                    wall_runtime_h, wall_ckpt_path,
                )
            except Exception as e:  # noqa: BLE001
                logger.error(
                    "wall-clock ckpt save failed: %s: %s", type(e).__name__, e,
                )
            last_wall_ckpt_time = time.time()

        # ---- SIGINT 软停：当前 batch 已结束，存盘 + 退出 ----
        if _STOP_REQUESTED:
            exit_reason = "sigint"
            logger.info(
                "[SIGINT] stop_requested=True after batch ep=%d → saving and exiting",
                num_episodes_done,
            )
            break

    # ---- 最终 ckpt + summary（统一走 _save_final_checkpoint_and_summary）----
    _save_final_checkpoint_and_summary()

    # ---- 关并行子进程 ----
    if parallel_env is not None:
        try:
            parallel_env.close()
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "parallel_env close failed: %s: %s", type(e).__name__, e,
            )


if __name__ == "__main__":
    main()

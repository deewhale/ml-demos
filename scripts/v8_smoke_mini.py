"""V8 Smoke 5 seed mini 版（专项验证选卡机制）。

跟 v8_smoke.py 同样架构（不真 RL update，只 collect_rollout 验证 pipeline），
但只跑 5 seed（默认 0-4），并：

1. 每 seed 开始前 reset_cache_stats（per-seed 真实 cache 命中率）
2. 每 seed 跑完把 RolloutStep 序列化到 data/v8_smoke/per_seed_trajectory_<seed>.jsonl
3. 5 seed 跑完出"选卡专项报告"：CARD_REWARDS 决策数 / skip 比例 /
   平均 step reward / 正负 reward 数 / top 选卡分布 / cache 命中率

不动 v8/ 下任何核心组件，仅依赖：
- v8.env / v8.model / v8.trainer.collect_rollout（已有）
- v8.deck_evaluator.{reset_cache_stats, get_cache_stats}（本次新增）

CLI:
    python scripts/v8_smoke_mini.py [--seeds 0,1,2,3,4]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

# 保证从仓库根可 import v8.*
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from v8.env import V8Env
from v8.model import V8Model
from v8.trainer import V8PPOTrainer
from v8 import deck_evaluator as _deck_eval

OUTPUT_DIR = "data/v8_smoke"


def _safe_float(x: Any) -> float:
    try:
        f = float(x)
        if math.isnan(f) or math.isinf(f):
            return 0.0
        return f
    except Exception:
        return 0.0


def _serialize_step(step) -> Dict[str, Any]:
    """RolloutStep → JSONL record。处理 None / 非序列化字段。

    deck_strength_before = step.state.deck_strength（决策时观测到的 strength）。
    deck_strength_after 在外部 _attach_deck_strength_after 里基于 next step 推断 / 重评。
    """
    s = step.state
    deck = getattr(s, "deck", []) or []
    deck_strength = getattr(s, "deck_strength", None)
    available = step.available_actions or []
    chosen = (
        available[step.action_idx]
        if available and 0 <= step.action_idx < len(available)
        else None
    )
    record = {
        "phase": s.phase or "",
        "floor": int(getattr(s, "floor", 0) or 0),
        "act": int(getattr(s, "act", 1) or 1),
        "hp": int(getattr(s, "hp", 0) or 0),
        "max_hp": int(getattr(s, "max_hp", 0) or 0),
        "deck_size": len(deck),
        # 拷贝完整 deck（含 name/upgraded）：便于事后比较选卡前后的 deck diff
        "deck": [{"name": c.get("name", ""), "upgraded": bool(c.get("upgraded", False))} for c in deck],
        # deck_strength 可能是 None（CARD_REWARDS 决策时尚未评估）；保持 None
        "deck_strength": deck_strength if deck_strength is not None else None,
        # 对齐 reward 反推语义（reward.py 用 prev/next deck_strength 算 Δ）：
        #   deck_strength_before：决策前观测到的 strength（== state.deck_strength）
        #   deck_strength_after： 决策后 env.step 触发 evaluate_deck 的结果，
        #                        在 _attach_deck_strength_after 阶段填
        "deck_strength_before": deck_strength if deck_strength is not None else None,
        "deck_strength_after": None,
        # 遗物 / hp 用于 fallback 重评 deck_strength_after
        "relics": list(getattr(s, "relics", []) or []),
        "available_actions": list(available),
        "action_idx": int(step.action_idx),
        "chosen_action": chosen,
        "reward": _safe_float(step.reward),
        "value": _safe_float(step.value),
        "log_prob": _safe_float(step.log_prob),
        "done": bool(step.done),
    }
    return record


def _attach_deck_strength_after(records: List[Dict[str, Any]]) -> int:
    """填 records[i]['deck_strength_after']。

    策略（按优先级）：
        1. 用 records[i+1]['deck_strength_before']（即 next state.deck_strength，
           env 在 step 内已调过 evaluate_deck）—— 命中 cache 几乎 0 开销
        2. 如果 1 拿不到（最后一 step 或 next 也 None）但 phase == CARD_REWARDS
           且能拿到 next deck → 强制重评 evaluate_deck(next_deck, ...)
        3. 否则保持 None

    返回 reevaluated 次数（监控用）。
    """
    from v8.deck_evaluator import evaluate_deck  # 局部 import 避免顶部 import 循环

    n_records = len(records)
    n_reeval = 0

    for i, rec in enumerate(records):
        # 1. 用下一 step 的 deck_strength_before（== 该 step 的 state.deck_strength）
        next_rec = records[i + 1] if i + 1 < n_records else None
        if next_rec is not None and next_rec.get("deck_strength_before") is not None:
            rec["deck_strength_after"] = next_rec["deck_strength_before"]
            continue

        # 2. fallback：CARD_REWARDS 且 next deck 可拿 → 强制重评
        phase = rec.get("phase", "")
        if phase != "CARD_REWARDS":
            continue
        next_deck = (next_rec or {}).get("deck") if next_rec else None
        if not next_deck:
            continue
        try:
            ds = evaluate_deck(
                deck=next_deck,
                relics=rec.get("relics") or [],
                hp=int(rec.get("hp", 0) or 0),
                max_hp=int(rec.get("max_hp", 0) or 0),
                act=int(rec.get("act", 1) or 1),
            )
            rec["deck_strength_after"] = ds
            n_reeval += 1
        except Exception:  # noqa: BLE001
            # 评估失败不让 dump 挂；保持 None
            pass

    return n_reeval


def _should_keep_step(rec: Dict[str, Any]) -> bool:
    """trajectory dump 时过滤 no-op step。

    剔除：
      - chosen_action 含 "proceed"（CARD_REWARDS 后的 confirm step，model 没真选择）
      - 或 len(available_actions) == 1（只有一个选项 = 不是真决策）

    保留：所有有 ≥ 2 个 available 选项的真决策 step。
    """
    chosen = (rec.get("chosen_action") or "").lower()
    if "proceed" in chosen:
        return False
    if len(rec.get("available_actions") or []) <= 1:
        return False
    return True


def run_one_seed(
    seed: int,
    trainer: V8PPOTrainer,
    output_dir: str,
) -> Dict[str, Any]:
    """跑单 seed：reset_cache_stats → collect_rollout → dump trajectory → 取 cache_stats。

    返回 per_seed dict（含 status / floor / steps / cache stats / trajectory 路径）。
    """
    seed_start = time.time()
    seed_result: Dict[str, Any] = {"seed": seed, "status": "running"}

    # 真 cache 命中率：每 seed 开始时 reset
    _deck_eval.reset_cache_stats()

    env: Optional[V8Env] = None
    try:
        print(f"[seed {seed}] constructing V8Env ...", flush=True)
        env = V8Env()
        print(f"[seed {seed}] V8Env constructed", flush=True)
    except Exception as e:  # noqa: BLE001
        seed_result.update({
            "status": "crashed",
            "error": f"V8Env ctor failed: {type(e).__name__}: {e}",
            "traceback": traceback.format_exc()[:500],
            "time_seconds": round(time.time() - seed_start, 2),
        })
        print(f"[seed {seed}] CRASH in V8Env ctor: {e}", flush=True)
        print(seed_result["traceback"], flush=True)
        return seed_result

    # heartbeat: 每 30s 打一行 elapsed，证明进程没死
    heartbeat_stop = threading.Event()

    def _heartbeat() -> None:
        t0 = time.time()
        while not heartbeat_stop.wait(30):
            print(
                f"[seed {seed}] heartbeat elapsed={int(time.time() - t0)}s "
                f"(collect_rollout still running)",
                flush=True,
            )

    hb_thread = threading.Thread(target=_heartbeat, daemon=True)
    hb_thread.start()

    try:
        print(f"[seed {seed}] env.reset(seed={seed}) ...", flush=True)
        env.reset(seed=seed)
        print(f"[seed {seed}] reset done, calling collect_rollout ...", flush=True)
        rollout = trainer.collect_rollout(env, seed=seed)
        print(f"[seed {seed}] collect_rollout returned, steps={len(rollout)}", flush=True)

        # 序列化所有 RolloutStep → records（保留全部，便于推 deck_strength_after）
        all_records: List[Dict[str, Any]] = [_serialize_step(s) for s in rollout]

        # 修复 1：填 deck_strength_after（用 next step 的 before 或必要时 reeval）
        n_reeval = _attach_deck_strength_after(all_records)

        # 修复 2：dump 时过滤 proceed no-op + 单选项 step（trajectory 只留真决策）
        dropped_records = [r for r in all_records if not _should_keep_step(r)]
        kept_records = [r for r in all_records if _should_keep_step(r)]

        # dump trajectory（只 dump kept，dropped 数量记到 summary）
        traj_path = os.path.join(output_dir, f"per_seed_trajectory_{seed}.jsonl")
        with open(traj_path, "w") as f:
            for rec in kept_records:
                f.write(json.dumps(rec, default=str) + "\n")

        phases_seen = sorted({(s.state.phase or "") for s in rollout if s.state is not None})
        final_state = rollout[-1].state if rollout else None
        final_floor = int(getattr(final_state, "floor", 0) or 0) if final_state is not None else 0
        final_act = int(getattr(final_state, "act", 1) or 1) if final_state is not None else 1
        total_reward = sum(_safe_float(s.reward) for s in rollout)

        # 拿 cache stats（在 close() 前抓，避免 finally 顺序）
        cache_stats = _deck_eval.get_cache_stats()

        seed_result.update({
            "status": "ok",
            "num_steps": len(rollout),
            "num_steps_kept": len(kept_records),
            "num_steps_dropped": len(dropped_records),
            "deck_strength_reevaluated_count": n_reeval,
            "final_floor": final_floor,
            "final_act": final_act,
            "total_reward": round(total_reward, 4),
            "phases_seen": phases_seen,
            "cache_stats": cache_stats,
            "trajectory_path": traj_path,
            "time_seconds": round(time.time() - seed_start, 2),
        })
    except Exception as e:  # noqa: BLE001
        seed_result.update({
            "status": "crashed",
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc()[:800],
            "time_seconds": round(time.time() - seed_start, 2),
            "cache_stats": _deck_eval.get_cache_stats(),
        })
        print(f"[seed {seed}] CRASH: {type(e).__name__}: {e}", flush=True)
        print(seed_result["traceback"], flush=True)
    finally:
        heartbeat_stop.set()
        try:
            if env is not None:
                env.close()
        except Exception:  # noqa: BLE001
            pass

    return seed_result


def analyze_card_decisions(seeds: List[int], output_dir: str) -> Dict[str, Any]:
    """解析所有 trajectory，出选卡专项报告。

    依赖 deck 字段（在 _serialize_step 里完整 dump 了），通过相邻 step 的
    deck diff 推断"实际拿了哪张卡"——比单纯看 chosen_action="REWARD:card:choice=N"
    更有信息量。
    """
    all_decisions: List[Dict[str, Any]] = []  # CARD_REWARDS 阶段的 step
    next_state_decks: Dict[int, List[Dict[str, Any]]] = {}  # 决策后立即下一步的 deck

    # 加载所有 trajectory
    per_seed_trajectories: Dict[int, List[Dict[str, Any]]] = {}
    for seed in seeds:
        path = os.path.join(output_dir, f"per_seed_trajectory_{seed}.jsonl")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            steps = [json.loads(line) for line in f if line.strip()]
        per_seed_trajectories[seed] = steps

    # 收集 CARD_REWARDS 决策 + 下一步 deck（detect 实际选了哪张卡）
    for seed, steps in per_seed_trajectories.items():
        for i, step in enumerate(steps):
            if step.get("phase") != "CARD_REWARDS":
                continue
            decision = {**step, "seed": seed, "step_index": i}
            # 取下一步的 deck 看 diff
            if i + 1 < len(steps):
                decision["next_deck"] = steps[i + 1].get("deck", [])
            all_decisions.append(decision)

    total_decisions = len(all_decisions)
    if total_decisions == 0:
        return {
            "total_decisions": 0,
            "warning": "no CARD_REWARDS decisions found",
        }

    # skip 判定：chosen_action 含 "skip"，或者前后 deck size 没变
    skip_count = 0
    pick_count = 0
    picks_by_card: Counter = Counter()
    rewards: List[float] = []
    positive_reward = 0
    negative_reward = 0
    zero_reward = 0
    deck_strength_evaluated_count = 0
    deck_strength_after_present_count = 0

    for d in all_decisions:
        chosen = (d.get("chosen_action") or "").lower()
        is_skip = "skip" in chosen
        # 对照前后 deck 推断"实际加了哪张卡"
        prev_deck = d.get("deck", []) or []
        next_deck = d.get("next_deck", []) or []
        prev_names = Counter((c.get("name", ""), bool(c.get("upgraded", False))) for c in prev_deck)
        next_names = Counter((c.get("name", ""), bool(c.get("upgraded", False))) for c in next_deck)
        added = next_names - prev_names  # multiset diff: 加进 deck 的新卡
        if is_skip:
            skip_count += 1
        elif added:
            pick_count += 1
            for (name, upgraded), cnt in added.items():
                tag = f"{name}{'+' if upgraded else ''}"
                picks_by_card[tag] += cnt
        else:
            # chosen 不是 skip 但 deck 没变 → 可能 take_action 失败 / phase mismatch
            # 归类成 "no_change"
            picks_by_card["__no_change__"] += 1

        r = _safe_float(d.get("reward"))
        rewards.append(r)
        if r > 0:
            positive_reward += 1
        elif r < 0:
            negative_reward += 1
        else:
            zero_reward += 1

        if d.get("deck_strength") is not None:
            deck_strength_evaluated_count += 1
        if d.get("deck_strength_after") is not None:
            deck_strength_after_present_count += 1

    mean_reward = sum(rewards) / max(len(rewards), 1)

    return {
        "total_decisions": total_decisions,
        "skip_count": skip_count,
        "skip_rate": round(skip_count / max(total_decisions, 1), 4),
        "pick_count": pick_count,
        "no_deck_change_count": picks_by_card.get("__no_change__", 0),
        "mean_reward": round(mean_reward, 4),
        "positive_reward_count": positive_reward,
        "negative_reward_count": negative_reward,
        "zero_reward_count": zero_reward,
        "deck_strength_evaluated_count": deck_strength_evaluated_count,
        "deck_strength_after_present_count": deck_strength_after_present_count,
        "top_10_picks": picks_by_card.most_common(10),
        "all_picks": dict(picks_by_card),
    }


def run_smoke_mini(seeds: List[int]) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"[v8_smoke_mini] start: seeds={seeds} output_dir={OUTPUT_DIR}", flush=True)

    # 随机初始化 model（不预训练，纯 pipeline 验证）
    print("[v8_smoke_mini] building V8Model + V8PPOTrainer ...", flush=True)
    model = V8Model()
    model.eval()
    trainer = V8PPOTrainer(model, device="cpu")
    print("[v8_smoke_mini] model + trainer ready", flush=True)

    summary: Dict[str, Any] = {
        "seeds": seeds,
        "total_seeds": len(seeds),
        "completed": 0,
        "crashed": 0,
        "per_seed": [],
        "phase_coverage": {},
        "reached_boss_count": 0,
        "beat_boss_count": 0,
        "floor_mean": 0.0,
        "floor_min": 0,
        "floor_max": 0,
        "avg_episode_time": 0.0,
        "avg_steps_per_episode": 0.0,
    }

    start_time = time.time()
    floors: List[int] = []
    total_steps = 0

    for seed_idx, seed in enumerate(seeds):
        print(
            f"[v8_smoke_mini] >>> starting seed {seed} ({seed_idx + 1}/{len(seeds)}) ...",
            flush=True,
        )
        sr = run_one_seed(seed, trainer, OUTPUT_DIR)
        summary["per_seed"].append(sr)

        if sr["status"] == "ok":
            summary["completed"] += 1
            floors.append(sr.get("final_floor", 0))
            total_steps += sr.get("num_steps", 0)
            if sr.get("final_floor", 0) >= 16 or sr.get("final_act", 1) >= 2:
                summary["reached_boss_count"] += 1
            if sr.get("final_act", 1) >= 2:
                summary["beat_boss_count"] += 1
            for phase in sr.get("phases_seen", []) or []:
                if not phase:
                    continue
                summary["phase_coverage"][phase] = summary["phase_coverage"].get(phase, 0) + 1
        else:
            summary["crashed"] += 1

        # 增量保存
        try:
            with open(f"{OUTPUT_DIR}/smoke_mini_summary.json", "w") as f:
                json.dump(summary, f, indent=2, default=str)
        except Exception as e:  # noqa: BLE001
            print(f"[seed {seed}] WARN: incremental save failed: {e}", flush=True)

        cs = sr.get("cache_stats", {}) or {}
        print(
            f"[seed {seed}] {sr['status']} "
            f"floor={sr.get('final_floor', '?')} "
            f"act={sr.get('final_act', '?')} "
            f"steps={sr.get('num_steps', '?')} "
            f"kept={sr.get('num_steps_kept', '?')} "
            f"dropped={sr.get('num_steps_dropped', '?')} "
            f"reeval={sr.get('deck_strength_reevaluated_count', 0)} "
            f"reward={sr.get('total_reward', '?')} "
            f"time={sr.get('time_seconds', 0)}s "
            f"cache hits/misses={cs.get('hits', 0)}/{cs.get('misses', 0)} "
            f"hit_rate={cs.get('hit_rate', 0):.2%}",
            flush=True,
        )

    # 终结统计
    total_time = time.time() - start_time
    summary["total_time_seconds"] = round(total_time, 2)
    if floors:
        summary["floor_mean"] = round(sum(floors) / len(floors), 2)
        summary["floor_min"] = min(floors)
        summary["floor_max"] = max(floors)
    summary["avg_episode_time"] = round(total_time / max(summary["completed"], 1), 2)
    summary["avg_steps_per_episode"] = round(total_steps / max(summary["completed"], 1), 2)

    # 选卡专项分析
    card_report = analyze_card_decisions(seeds, OUTPUT_DIR)
    summary["card_decision_analysis"] = card_report

    # 写最终 summary
    try:
        with open(f"{OUTPUT_DIR}/smoke_mini_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
    except Exception as e:  # noqa: BLE001
        print(f"WARN: final save failed: {e}", flush=True)

    # 打印最终报告
    print("\n=== V8 Smoke MINI {} seed 完成 ===".format(len(seeds)), flush=True)
    print(f"总耗时: {total_time/60:.1f} min", flush=True)
    print(f"完成: {summary['completed']}/{summary['total_seeds']}", flush=True)
    print(f"crashed: {summary['crashed']}", flush=True)
    print(f"reached_boss: {summary['reached_boss_count']}/{summary['completed']}", flush=True)
    print(f"beat_boss: {summary['beat_boss_count']}/{summary['completed']}", flush=True)
    print(f"floor mean/min/max: {summary['floor_mean']}/{summary['floor_min']}/{summary['floor_max']}", flush=True)
    print(f"phase coverage: {summary['phase_coverage']}", flush=True)
    print(f"avg episode time: {summary['avg_episode_time']}s", flush=True)
    print(f"avg steps/episode: {summary['avg_steps_per_episode']}", flush=True)

    print("\n--- CARD_REWARDS 选卡专项 ---", flush=True)
    cr = card_report
    if cr.get("total_decisions", 0) == 0:
        print("WARN: no CARD_REWARDS decisions found", flush=True)
    else:
        print(f"总决策数: {cr['total_decisions']}", flush=True)
        print(f"skip: {cr['skip_count']} ({cr['skip_rate']:.1%})", flush=True)
        print(f"pick (deck 实际增加): {cr['pick_count']}", flush=True)
        print(f"chosen 非 skip 但 deck 没变: {cr['no_deck_change_count']}", flush=True)
        print(f"平均 step reward: {cr['mean_reward']}", flush=True)
        print(f"正/负/零 reward: {cr['positive_reward_count']}/{cr['negative_reward_count']}/{cr['zero_reward_count']}", flush=True)
        print(f"deck_strength_before 已评估的决策数: {cr['deck_strength_evaluated_count']}/{cr['total_decisions']}", flush=True)
        print(f"deck_strength_after  已填充的决策数: {cr.get('deck_strength_after_present_count', 0)}/{cr['total_decisions']}", flush=True)
        print("Top 10 选卡:", flush=True)
        for name, cnt in cr["top_10_picks"]:
            print(f"  {name}: {cnt}", flush=True)

    # 跨 seed 累计 cache 命中率（用 summary 里的 per_seed cache_stats 加总）
    total_h = 0
    total_m = 0
    total_kept = 0
    total_dropped = 0
    total_reeval = 0
    for sr in summary["per_seed"]:
        cs = sr.get("cache_stats", {}) or {}
        total_h += int(cs.get("hits", 0))
        total_m += int(cs.get("misses", 0))
        total_kept += int(sr.get("num_steps_kept", 0) or 0)
        total_dropped += int(sr.get("num_steps_dropped", 0) or 0)
        total_reeval += int(sr.get("deck_strength_reevaluated_count", 0) or 0)
    overall_hit_rate = total_h / max(total_h + total_m, 1)
    print(f"\n--- evaluate_deck 累计 cache 统计（5 seed 总和）---", flush=True)
    print(f"hits/misses: {total_h}/{total_m}", flush=True)
    print(f"overall hit_rate: {overall_hit_rate:.2%}", flush=True)
    print(f"\n--- trajectory dump 过滤统计（5 seed 总和）---", flush=True)
    print(f"kept (真决策): {total_kept}", flush=True)
    print(f"dropped (proceed no-op + 单选项): {total_dropped}", flush=True)
    print(f"deck_strength_after 重评次数: {total_reeval}", flush=True)
    summary["total_kept"] = total_kept
    summary["total_dropped"] = total_dropped
    summary["total_deck_strength_reevaluated"] = total_reeval
    summary["overall_cache_hit_rate"] = round(overall_hit_rate, 4)
    try:
        with open(f"{OUTPUT_DIR}/smoke_mini_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
    except Exception as e:  # noqa: BLE001
        print(f"WARN: final save (post-aggregate) failed: {e}", flush=True)


def _parse_seeds(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


if __name__ == "__main__":
    print(f"[v8_smoke_mini] __main__ entered, argv={sys.argv}", flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds",
        type=_parse_seeds,
        default=[0, 1, 2, 3, 4],
        help="逗号分隔的 seed 列表（默认 0,1,2,3,4）",
    )
    args = parser.parse_args()
    run_smoke_mini(args.seeds)

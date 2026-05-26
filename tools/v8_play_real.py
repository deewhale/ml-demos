"""V8 RL 真游戏 driver（MVP）：通过 CommunicationMod stdin/stdout 跑 Slay the Spire。

用法（mod 配置见 README 段）：
    把这个脚本配成 CommunicationMod 的 "command"，mod 会把它当子进程拉起来，
    stdin 收 V8 命令、stdout 推 game-state JSON（每行一条）。

启动示例（mod config.properties）：
    command=/path/to/.venv/bin/python /path/to/tools/v8_play_real.py \\
            --ckpt /path/to/v8_ppo_ep640.pt --character IRONCLAD

不在本脚本：
    - 战斗内多回合搜索（MVP 直接 argmax）
    - HAND_SELECT / GRID 等子 screen 处理（兜底 cancel）
"""

from __future__ import annotations

import argparse
import atexit
import datetime as _dt
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# --- repo path 注入 ---
# 通过脚本自身位置推断 repo 根（避免硬编码绝对路径）；
# 也允许 ML_DEMOS_REPO env var 覆盖（多 clone / 软链场景）
_REPO = os.environ.get("ML_DEMOS_REPO") or str(Path(__file__).resolve().parent.parent)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import torch  # noqa: E402

from v8.model import V8Model  # noqa: E402
from v8.mod_state_adapter import (  # noqa: E402
    mod_json_to_available_actions,
    mod_json_to_v8_state,
    v8_idx_to_mod_command,
)


_LOG_FILE_PATH = "/tmp/v8_play_real.stderr"
try:
    _LOG_FILE = open(_LOG_FILE_PATH, "a", buffering=1)
    atexit.register(_LOG_FILE.close)
except Exception:  # noqa: BLE001
    _LOG_FILE = None


def _log(msg: str) -> None:
    """所有日志走 stderr（stdout 是 mod 通信通道，不能污染），同时镜像到日志文件。"""
    line = f"[v8_play] {msg}"
    print(line, file=sys.stderr, flush=True)
    if _LOG_FILE is not None:
        try:
            print(line, file=_LOG_FILE, flush=True)
        except Exception:  # noqa: BLE001
            pass


def _send(cmd: str) -> None:
    """单条 mod 命令：stdout 一行。"""
    print(cmd, flush=True)


def _read_state() -> Optional[Dict[str, Any]]:
    """从 stdin 读一行 JSON，返回 dict（EOF / 空行返回 None）。"""
    line = sys.stdin.readline()
    if not line:
        return None
    line = line.strip()
    if not line:
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError as e:
        _log(f"JSON decode error: {e}; raw={line[:200]}")
        return None


def _select_device() -> torch.device:
    try:
        if torch.backends.mps.is_available():
            return torch.device("mps")
    except Exception:  # noqa: BLE001
        pass
    return torch.device("cpu")


def _load_model(ckpt_path: str, device: torch.device) -> V8Model:
    _log(f"loading model from {ckpt_path} on {device}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict: Dict[str, Any]
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and any(k.endswith(".weight") for k in ckpt.keys()):
        state_dict = ckpt
    else:
        # ckpt 本身可能就是 state_dict 或带 wrapper
        state_dict = ckpt
    model = V8Model()
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        _log(f"WARN missing keys: {missing[:5]}... ({len(missing)} total)")
    if unexpected:
        _log(f"WARN unexpected keys: {unexpected[:5]}... ({len(unexpected)} total)")
    model.to(device)
    model.eval()
    return model


# =============================================================================
# 多局指标收集 + summary
# =============================================================================

# STS 各 act boss 楼层（标准 ascension 0，IRONCLAD/SILENT/DEFECT/WATCHER 通用）：
#   Act 1 boss = floor 17, Act 2 boss = floor 34, Act 3 boss = floor 51
#   Act 4 final = floor 57（Heart）
# 我们用 floor / act 双信号判定（act 更可靠，floor 只做 sanity）
_BOSS_FLOORS = {1: 17, 2: 34, 3: 51, 4: 57}


def _new_game_metrics(game_idx: int, started_at: float) -> Dict[str, Any]:
    """初始化一局的 metrics dict。

    字段在 GAME_OVER 时由 _finalize_game_metrics 写入最终值。
    途中由 main loop 增量更新：
        - turns_played: 每次 end_turn 命令 +1
        - max_floor_reached / final_act: 每帧追踪
        - max_hp / hp_lost: 用 GAME_OVER 时的 max_hp + (max_hp - current_hp) 推
    """
    return {
        "game_idx": game_idx,
        "started_at": started_at,
        "ended_at": None,
        "elapsed_sec": None,
        "final_floor": 0,
        "final_act": 1,
        "max_floor_reached": 0,
        "max_hp": 0,
        "final_hp": 0,
        "hp_lost": 0,
        "turns_played": 0,
        "won_game": False,
        "act1_boss_beat": False,
        "act2_boss_beat": False,
        "act3_boss_beat": False,
        "cause_of_death": "unknown",  # "victory" / "act1_boss" / "act2_boss" /
                                       # "act3_boss" / "act4_heart" /
                                       # "act{N}_floor{F}" (non-boss death)
        "steps_executed": 0,
    }


def _update_game_progress(
    metrics: Dict[str, Any],
    floor: int,
    act: int,
    hp: int,
    max_hp: int,
) -> None:
    """每帧 in_game 时增量刷新 metrics 的当前进度字段（不算结束指标）。"""
    if floor > metrics["max_floor_reached"]:
        metrics["max_floor_reached"] = floor
    metrics["final_floor"] = floor
    metrics["final_act"] = act
    if max_hp > metrics["max_hp"]:
        metrics["max_hp"] = max_hp
    metrics["final_hp"] = hp


def _finalize_game_metrics(
    metrics: Dict[str, Any],
    won: bool,
    final_floor: int,
    final_act: int,
    final_hp: int,
    final_max_hp: int,
) -> None:
    """GAME_OVER 时收尾：判定 boss 通过 / cause_of_death / hp_lost。

    STS run-end 时 game_state.act/floor 仍指向死的那个楼层 / boss 楼层。
    判定规则（IRONCLAD 0 ascension）：
        - won=True → 通关 Act 3 boss（boss_beat=True for act1/2/3）；
          若 final_floor>=57（Act 4 Heart 通关）→ won_game=True 也包含 Heart kill
        - won=False → 死亡。如果 final_floor 是 boss 楼层（17/34/51/57）→
          act{N}_boss_beat 仍 False，cause_of_death=act{N}_boss
        - 否则 cause_of_death=act{N}_floor{F}（非 boss 死）
    "通过 actK boss" 推断：max_floor_reached > boss_floor 或 final_act > K，
    或 won 且 final_act >= K（won 意味着至少推进到 Act3 后）
    """
    metrics["won_game"] = won
    metrics["final_floor"] = final_floor
    metrics["final_act"] = final_act
    metrics["final_hp"] = final_hp
    if final_max_hp > metrics["max_hp"]:
        metrics["max_hp"] = final_max_hp
    metrics["hp_lost"] = max(0, metrics["max_hp"] - final_hp) if not won else max(0, metrics["max_hp"] - final_hp)

    # act{N}_boss_beat: 通过 max_floor / final_act 推断
    # 推过 act1 boss = max_floor > 17 或 final_act >= 2
    max_floor = metrics["max_floor_reached"]
    metrics["act1_boss_beat"] = bool(max_floor > _BOSS_FLOORS[1] or final_act >= 2)
    metrics["act2_boss_beat"] = bool(max_floor > _BOSS_FLOORS[2] or final_act >= 3)
    metrics["act3_boss_beat"] = bool(max_floor > _BOSS_FLOORS[3] or final_act >= 4 or won)

    # cause_of_death 归因
    if won:
        if max_floor >= _BOSS_FLOORS[4]:
            metrics["cause_of_death"] = "victory_heart"  # A4 通关
        else:
            metrics["cause_of_death"] = "victory_act3"   # A3 通关（普通 win）
    else:
        if final_floor == _BOSS_FLOORS[1]:
            metrics["cause_of_death"] = "act1_boss"
        elif final_floor == _BOSS_FLOORS[2]:
            metrics["cause_of_death"] = "act2_boss"
        elif final_floor == _BOSS_FLOORS[3]:
            metrics["cause_of_death"] = "act3_boss"
        elif final_floor == _BOSS_FLOORS[4]:
            metrics["cause_of_death"] = "act4_heart"
        else:
            metrics["cause_of_death"] = f"act{final_act}_floor{final_floor}"


def _write_summary(
    output_dir: str,
    ckpt_path: str,
    character: str,
    ascension: int,
    games: List[Dict[str, Any]],
) -> str:
    """把所有局的 metrics 汇总成 summary JSON 写盘。返回 summary 文件路径。"""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"v8_real_test_summary_{ts}.json"
    fpath = out_path / fname

    n = len(games)
    if n == 0:
        agg = {
            "n_games": 0,
            "note": "no games completed (script exited before any GAME_OVER)",
        }
    else:
        def _mean(key: str) -> float:
            return sum(float(g.get(key, 0) or 0) for g in games) / n

        def _rate(key: str) -> float:
            return sum(1 for g in games if g.get(key)) / n

        # cause_of_death 直方图
        cause_hist: Dict[str, int] = {}
        for g in games:
            c = str(g.get("cause_of_death", "unknown"))
            cause_hist[c] = cause_hist.get(c, 0) + 1

        agg = {
            "n_games": n,
            "won_game_rate": _rate("won_game"),
            "act1_boss_beat_rate": _rate("act1_boss_beat"),
            "act2_boss_beat_rate": _rate("act2_boss_beat"),
            "act3_boss_beat_rate": _rate("act3_boss_beat"),
            "reached_a1_boss_rate": sum(
                1 for g in games if g.get("max_floor_reached", 0) >= _BOSS_FLOORS[1]
            ) / n,
            "mean_final_floor": _mean("final_floor"),
            "mean_max_floor_reached": _mean("max_floor_reached"),
            "mean_turns_played": _mean("turns_played"),
            "mean_hp_lost": _mean("hp_lost"),
            "mean_elapsed_sec": _mean("elapsed_sec"),
            "cause_of_death_hist": cause_hist,
        }

    summary = {
        "ckpt": ckpt_path,
        "character": character,
        "ascension": ascension,
        "timestamp": ts,
        "aggregate": agg,
        "games": games,
    }
    fpath.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return str(fpath)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="V8 PPO checkpoint .pt")
    parser.add_argument("--character", default="IRONCLAD",
                        choices=["IRONCLAD", "SILENT", "DEFECT", "WATCHER"])
    parser.add_argument("--seed", type=str, default="",
                        help="seed string (optional)")
    parser.add_argument("--max_steps", type=int, default=2000,
                        help="安全上限：单局 step 超过即退出")
    parser.add_argument("--ascension", type=int, default=0)
    parser.add_argument("--num_games", type=int, default=1,
                        help="连打 N 局后退出（默认 1，单局兼容）")
    parser.add_argument("--output_dir", type=str,
                        default="./real_test_results/",
                        help="summary JSON 输出目录（默认 ./real_test_results/）")
    args = parser.parse_args()

    device = _select_device()
    model = _load_model(args.ckpt, device)
    _log(f"model ready; character={args.character} seed={args.seed!r}")
    _log(f"num_games={args.num_games} output_dir={args.output_dir}")

    # 与 mod 握手
    _send("ready")

    started = False
    step = 0
    last_floor = -1
    started_at = time.time()
    game_count = 0      # 已完成的局数（GAME_OVER 计数）
    num_games = max(1, int(args.num_games))
    games_metrics: List[Dict[str, Any]] = []
    cur_metrics: Optional[Dict[str, Any]] = None  # 当前局指标 dict（每局重建)

    # 防卡死兜底：同一 screen_type 连续 N step 没切换且发的 cmd 也没变 → abandon 这局
    # （场景：sub-screen 没 handler / model 反复选同一无效选项 / mod 不响应）
    SAME_SCREEN_STALL_LIMIT = 8     # 连续 N step 同 screen + 同 cmd 触发
    stall_screen: str = ""
    stall_last_cmd: str = ""
    stall_count: int = 0

    def _flush_summary(reason: str) -> str:
        path = _write_summary(
            args.output_dir, args.ckpt, args.character, args.ascension, games_metrics
        )
        _log(f"summary written ({reason}): {path}")
        return path

    try:
        while True:
            state_dict = _read_state()
            if state_dict is None:
                _log("stdin EOF, exiting")
                _flush_summary("stdin_eof")
                return 0

            step += 1
            if step > args.max_steps:
                _log(f"max_steps {args.max_steps} reached, exiting")
                # 兜底：若当前局未结束，先记录一笔 abandoned
                if cur_metrics is not None and cur_metrics.get("ended_at") is None:
                    cur_metrics["ended_at"] = time.time()
                    cur_metrics["elapsed_sec"] = cur_metrics["ended_at"] - cur_metrics["started_at"]
                    cur_metrics["cause_of_death"] = "abandoned_max_steps"
                    games_metrics.append(cur_metrics)
                    cur_metrics = None
                _flush_summary("max_steps")
                return 0

            # 1. 没开局 → start 命令
            if not state_dict.get("in_game"):
                if started:
                    # mod 在 dungeon load 期间会推多帧 in_game=False（过渡帧），不能直接退
                    # 真正的「局结束」要等 GAME_OVER screen 或 stdin EOF
                    _log("in_game=False after start (transition frame), waiting...")
                    continue
                # 一局开始（包括第一局和后续局）
                game_idx = game_count + 1
                _log(f"=== game {game_idx} / {num_games} start ===")
                seed_part = f" {args.seed}" if args.seed else ""
                # mod 协议：start <character> <ascension> <seed>
                # 多局模式下不重用 seed（让每局随机），仅在 num_games=1 时使用 --seed
                if num_games > 1 and game_idx > 1:
                    seed_part = ""
                start_cmd = f"start {args.character} {args.ascension}{seed_part}".strip()
                _log(f"sending: {start_cmd}")
                _send(start_cmd)
                started = True
                last_floor = -1
                started_at = time.time()
                step = 0  # 单局 max_steps 重新计数
                cur_metrics = _new_game_metrics(game_idx, started_at)
                continue

            gs = state_dict.get("game_state", {}) or {}
            screen_type = gs.get("screen_type", "")
            floor = int(gs.get("floor", 0) or 0)
            act = int(gs.get("act", 1) or 1)
            hp = int(gs.get("current_hp", 0) or 0)
            max_hp = int(gs.get("max_hp", 0) or 0)

            # 增量更新当前局进度
            if cur_metrics is not None:
                _update_game_progress(cur_metrics, floor, act, hp, max_hp)
                cur_metrics["steps_executed"] = step

            # 2. Game over
            if screen_type == "GAME_OVER":
                screen_state = gs.get("screen_state", {}) or {}
                won = bool(screen_state.get("victory", False))
                game_count += 1
                elapsed = time.time() - started_at

                # 落 metrics
                if cur_metrics is not None:
                    cur_metrics["ended_at"] = time.time()
                    cur_metrics["elapsed_sec"] = elapsed
                    _finalize_game_metrics(
                        cur_metrics, won, floor, act, hp, max_hp
                    )
                    games_metrics.append(cur_metrics)
                    cur_metrics = None

                _log(
                    f"GAME_OVER at floor={floor} act={act} hp={hp}/{max_hp} "
                    f"won={won} elapsed={elapsed:.1f}s"
                )
                _log(f"=== game {game_count} / {num_games} done: floor={floor} won={won} ===")

                # 增量 flush 一次，避免后续 hang 丢数据
                _flush_summary(f"after_game_{game_count}")

                # 发 proceed 把 game-over screen 关掉，回到主菜单
                _send("proceed")
                if game_count >= num_games:
                    _log(f"reached num_games={num_games}, exiting")
                    # 等一条最终状态再退（best-effort）
                    final = _read_state()
                    if final is not None and not final.get("in_game"):
                        _log("post-GAME_OVER got out-of-game state, exit")
                    _flush_summary("all_games_done")
                    return 0
                # 还要再打：重置 started，让下一次 in_game=False 触发新 start
                started = False
                continue

            if floor != last_floor:
                _log(f"floor {last_floor}→{floor} hp={hp}/{max_hp} screen={screen_type}")
                last_floor = floor

            # 3. 构 state + available（GRID / HAND_SELECT / SHOP / 多阶段 EVENT
            #    全部走 adapter，由 adapter 枚举 sub-screen 选项 → model 决策）
            try:
                v8_state = mod_json_to_v8_state(state_dict)
                actions, metas = mod_json_to_available_actions(state_dict)
            except Exception as e:  # noqa: BLE001
                _log(f"adapter ERROR {type(e).__name__}: {e}, sending proceed")
                _send("proceed")
                continue

            # 子画面诊断日志：GRID / HAND_SELECT / SHOP_SCREEN 时多打一行可见信息
            if screen_type in ("GRID", "HAND_SELECT", "SHOP_SCREEN"):
                ss_ext = (state_dict.get("game_state") or {}).get("screen_state", {}) or {}
                _log(
                    f"sub-screen={screen_type} "
                    f"num_cards={ss_ext.get('num_cards', '?')} "
                    f"can_pick_zero={ss_ext.get('can_pick_zero', '?')} "
                    f"for_transform={ss_ext.get('for_transform', '?')} "
                    f"for_upgrade={ss_ext.get('for_upgrade', '?')} "
                    f"for_purge={ss_ext.get('for_purge', '?')} "
                    f"n_actions={len(actions)}"
                )

            if not actions:
                # 没有可选 action：尝试按 available_commands 兜底
                available_cmds = state_dict.get("available_commands", []) or []
                fallback = "proceed" if "proceed" in available_cmds else (
                    "confirm" if "confirm" in available_cmds else (
                        "cancel" if "cancel" in available_cmds else "wait 30"
                    )
                )
                _log(f"no actions (phase={v8_state.phase} screen={screen_type}), sending {fallback}")
                _send(fallback)
                # 把 fallback 也算入 stall tracking，防 sub-screen 死循环
                if screen_type == stall_screen and fallback == stall_last_cmd:
                    stall_count += 1
                else:
                    stall_screen = screen_type
                    stall_last_cmd = fallback
                    stall_count = 1
                if stall_count >= SAME_SCREEN_STALL_LIMIT:
                    _log(
                        f"STALL: screen={screen_type} cmd={fallback} repeated "
                        f"{stall_count}x, abandoning game"
                    )
                    if cur_metrics is not None and cur_metrics.get("ended_at") is None:
                        cur_metrics["ended_at"] = time.time()
                        cur_metrics["elapsed_sec"] = cur_metrics["ended_at"] - cur_metrics["started_at"]
                        cur_metrics["cause_of_death"] = f"abandoned_stall_{screen_type}"
                        games_metrics.append(cur_metrics)
                        cur_metrics = None
                    _flush_summary("stall_abandoned")
                    # 尝试软退出：发 return / proceed 让 mod 回主菜单，然后 EOF 退
                    for retry_cmd in ("return", "proceed", "cancel"):
                        _send(retry_cmd)
                    return 0
                continue

            # 4. Model forward → argmax → command
            try:
                with torch.no_grad():
                    out = model(v8_state, actions)
                    logits = out["logits"]
                if logits.numel() == 0:
                    idx = 0
                else:
                    idx = int(torch.argmax(logits).item())
                if idx < 0 or idx >= len(actions):
                    idx = 0
                cmd = v8_idx_to_mod_command(idx, metas, state_dict)
                chosen_action = actions[idx] if idx < len(actions) else "?"
                # turns_played: 每次发 "end" (回合结束) 计 1
                if cur_metrics is not None and cmd.strip() == "end":
                    cur_metrics["turns_played"] += 1
                _log(
                    f"phase={v8_state.phase} screen={screen_type} "
                    f"n_actions={len(actions)} idx={idx} action={chosen_action} cmd={cmd}"
                )
                _send(cmd)

                # Stall 检测：同 screen + 同 cmd 连续 N 次 → 当作死循环 abandon
                if screen_type == stall_screen and cmd == stall_last_cmd:
                    stall_count += 1
                else:
                    stall_screen = screen_type
                    stall_last_cmd = cmd
                    stall_count = 1
                if stall_count >= SAME_SCREEN_STALL_LIMIT:
                    _log(
                        f"STALL: screen={screen_type} cmd={cmd} repeated "
                        f"{stall_count}x (model picked same action / mod didn't progress), "
                        f"abandoning game"
                    )
                    if cur_metrics is not None and cur_metrics.get("ended_at") is None:
                        cur_metrics["ended_at"] = time.time()
                        cur_metrics["elapsed_sec"] = cur_metrics["ended_at"] - cur_metrics["started_at"]
                        cur_metrics["cause_of_death"] = f"abandoned_stall_{screen_type}"
                        games_metrics.append(cur_metrics)
                        cur_metrics = None
                    _flush_summary("stall_abandoned")
                    for retry_cmd in ("return", "proceed", "cancel"):
                        _send(retry_cmd)
                    return 0
            except Exception as e:  # noqa: BLE001
                _log(f"model/forward ERROR {type(e).__name__}: {e}, sending proceed")
                _send("proceed")
                continue
    except KeyboardInterrupt:
        _log("KeyboardInterrupt, writing summary then exiting")
        _flush_summary("keyboard_interrupt")
        return 130
    except Exception as e:  # noqa: BLE001
        _log(f"FATAL {type(e).__name__}: {e}")
        _flush_summary("fatal_error")
        raise


if __name__ == "__main__":
    sys.exit(main())

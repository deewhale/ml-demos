"""ParallelV8Env — N 个 V8Env 子进程并行（phase 1 scaffolding）。

设计目标：
- 把 PPO rollout collection 从串行 N ep → 并行 N ep（~3.5x），主进程仍然单 model
  forward；子进程只跑 V8Env 模拟器 + 把 obs 回传。
- 子进程内 V8Env 独立持有 GameRunner / TurnSolverAdapter / deck_evaluator pool；
  主进程只负责 model forward + action 调度。

⚠️ phase 1 状态：
- 子进程 worker 已落（reset/step/get_available_actions/close 通讯协议）
- batched API（reset/step）已落
- model forward 仍由 caller（trainer）在主进程做
- **trainer 适配是 phase 2**——本文件不动 v8/trainer.py 也不动 tools/v8_ppo_train.py
- combat_net_wrapper 在 phase 1 默认 None（纯搜索）。phase 2 决定子进程是否各持
  wrapper 副本 vs. 把战斗 leaf eval 通过 pipe 回主进程。这是 phase 2 主要风险点。

deck_evaluator pool 策略（phase 1）：
- (b) 方案：每个子进程内独立 spawn pool。
- worker 数靠 V8_DECK_EVALUATOR_PARALLEL 环境变量；建议 phase 2 真训练前下调
  （n_envs=4 × 默认 12 worker = 48 进程，会撑爆 CPU）。
- 设计文档 docs/parallel_env_design.md 详述 (a)（中央 pool 通过 pipe 回主）vs.
  (b)（各自 pool）的权衡。

通讯协议：
    主 → 子（Pipe.send(cmd_tuple)）：
        ("reset", seed: int)               → 子: ("ok", V8State)
        ("step", action_idx: int)          → 子: ("ok", (V8State, float, bool, dict))
        ("get_actions", )                  → 子: ("ok", List[str])
        ("set_episode", ep_idx: int)       → 子: ("ok", None)
        ("close", )                        → 子: ("ok", None); 子进程退出
        ("ping", )                         → 子: ("ok", "pong")  smoke 用

    子 → 主（异常）：
        ("err", repr(exception))           主进程读到后 raise RuntimeError

为什么不用 SubprocVecEnv 风格（共享 array）：
- V8State 是含 dataclass / list / dict 的复杂结构，pickle 传输已经够用，
  shared memory 没有显著收益。
- pickle V8State + List[str] available_actions：实测 episode 内 step 数十次量级，
  每次 pickle ~KB，相对 search calls cost 可以忽略。

为什么不用 spawn 之外的 start_method：
- macOS + MPS 不能 fork（CLAUDE.md / deck_evaluator.py 同款约束）
- 主进程已用 spawn pool；子进程内的 deck_evaluator pool 自然也 spawn
"""

from __future__ import annotations

import atexit
import logging
import multiprocessing as mp
import os
import sys
import weakref
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 让 spawn 子进程能 import v8.*（与 tools/v8_ppo_train.py 同模式）
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from v8.env_base import V8EnvBase
from v8.state import V8State


logger = logging.getLogger(__name__)


# ============================================================
# 全局 reap registry（atexit 兜底）
# ============================================================
#
# worker 进程 daemon=False（必须，否则 deck_evaluator._get_pool spawn 子 pool 时
# Python 抛 "daemonic processes are not allowed to have children"）。
# 但 daemon=False 的代价：主进程退出前必须显式 join，否则子进程变孤儿。
# 用 atexit + weakref 兜底：每个 ParallelV8Env 实例 init 时注册自己，
# atexit 时若实例还活着且没 close，触发 close()（礼貌发 close cmd + join）。
#
# weakref 防止 atexit 持 strong ref 阻止 GC。

_OPEN_PARALLEL_ENVS: "weakref.WeakSet[ParallelV8Env]" = weakref.WeakSet()


def _atexit_close_all() -> None:
    """atexit hook：关掉所有还活着的 ParallelV8Env。"""
    for penv in list(_OPEN_PARALLEL_ENVS):
        try:
            penv.close()
        except Exception:  # noqa: BLE001
            pass


atexit.register(_atexit_close_all)


# ============================================================
# Worker：在子进程里 host 一个 V8Env，循环接 command 处理
# ============================================================


def _worker_main(
    worker_idx: int,
    conn: Connection,
    env_kwargs: Dict[str, Any],
    log_level: int = logging.INFO,
    deck_eval_workers: int = 2,
) -> None:
    """子进程入口：建 V8Env，loop 接 (cmd, *args)，处理后回 reply。

    子进程内：
    - logger 加 `[env-N]` 前缀（每个 env 独立标识）
    - V8Env 实例化用 env_kwargs（caller 传，可含 character / ascension /
      max_steps_per_episode / deck_eval_freq / verbose 等）
    - combat_net_wrapper 不能 pickle pytorch model 到子进程（spawn 也未必干净），
      所以子进程默认拿不到。phase 2 接受 combat search-only regression 作为 MVP。
    - deck_eval_workers 通过环境变量 V8_DECK_EVALUATOR_MAX_WORKERS 传给
      v8.deck_evaluator._get_pool（限制子进程内 pool 大小，防爆 CPU）
    """
    # 限制子进程内 deck_evaluator pool 大小（env var 在 deck_evaluator import 前设）
    os.environ["V8_DECK_EVALUATOR_MAX_WORKERS"] = str(int(deck_eval_workers))

    # 子进程独立日志（带 [env-N] 前缀），不继承主进程 handler
    h = logging.StreamHandler(stream=sys.stderr)
    h.setFormatter(logging.Formatter(
        f"%(asctime)s %(levelname)s [env-{worker_idx}] %(message)s"
    ))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(h)
    root.setLevel(log_level)

    # 子进程内 lazy import，避免主进程 fork-friendly 包被提前 spawn-loaded
    try:
        from v8.env import V8Env
    except Exception as e:  # noqa: BLE001
        try:
            conn.send(("err", f"V8Env import failed: {type(e).__name__}: {e}"))
        except Exception:  # noqa: BLE001
            pass
        return

    # 实例化 V8Env
    try:
        # combat_net_wrapper 不通过 pickle 传；如果 caller 传了非 None，强制忽略并 warn
        if env_kwargs.get("combat_net_wrapper") is not None:
            logger.warning(
                "ParallelV8Env worker: combat_net_wrapper passed to subprocess "
                "is ignored in phase 1; will use pure search."
            )
            env_kwargs = dict(env_kwargs)
            env_kwargs["combat_net_wrapper"] = None
        env = V8Env(**env_kwargs)
    except Exception as e:  # noqa: BLE001
        try:
            conn.send(("err", f"V8Env init failed: {type(e).__name__}: {e}"))
        except Exception:  # noqa: BLE001
            pass
        return

    # 主循环
    while True:
        try:
            msg = conn.recv()
        except (EOFError, KeyboardInterrupt):
            break
        if not isinstance(msg, tuple) or not msg:
            try:
                conn.send(("err", f"malformed cmd: {msg!r}"))
            except Exception:  # noqa: BLE001
                pass
            continue

        cmd = msg[0]
        try:
            if cmd == "reset":
                seed = int(msg[1])
                state = env.reset(seed=seed)
                conn.send(("ok", state))
            elif cmd == "step":
                action_idx = int(msg[1])
                result = env.step(action_idx)
                conn.send(("ok", result))
            elif cmd == "get_actions":
                actions = env.get_available_actions()
                conn.send(("ok", actions))
            elif cmd == "set_episode":
                ep_idx = int(msg[1])
                env.set_episode(ep_idx)
                conn.send(("ok", None))
            elif cmd == "reset_perf_counters":
                env.reset_perf_counters()
                conn.send(("ok", None))
            elif cmd == "ping":
                conn.send(("ok", "pong"))
            elif cmd == "close":
                try:
                    env.close()
                except Exception:  # noqa: BLE001
                    pass
                conn.send(("ok", None))
                break
            else:
                conn.send(("err", f"unknown cmd: {cmd!r}"))
        except Exception as e:  # noqa: BLE001
            # 任何子进程异常都包装回主，由主决定 raise / skip
            try:
                conn.send(("err", f"{type(e).__name__}: {e}"))
            except Exception:  # noqa: BLE001
                # pipe 已断 → 直接退出
                break

    try:
        conn.close()
    except Exception:  # noqa: BLE001
        pass


# ============================================================
# ParallelV8Env
# ============================================================


class ParallelV8Env(V8EnvBase):
    """N 个 V8Env 并行 wrapper（phase 1 scaffolding）。

    用法（phase 1，只测基础设施）：
        penv = ParallelV8Env(n_envs=2, env_kwargs={"max_steps_per_episode": 30})
        states = penv.reset(seeds=[1, 2])
        actions = penv.get_available_actions()  # List[List[str]]
        # 主进程 model forward 选 idx ...
        states, rewards, dones, infos = penv.step([0, 0])
        penv.close()

    phase 2 改动 plan 参见 docs/parallel_env_design.md。
    """

    def __init__(
        self,
        n_envs: int = 4,
        env_kwargs: Optional[Dict[str, Any]] = None,
        log_level: int = logging.INFO,
        deck_eval_workers_per_env: int = 2,
    ):
        """初始化 N 个子进程。

        Args:
            n_envs: 并行 env 数
            env_kwargs: 传给 V8Env 的 kwargs；combat_net_wrapper 若非 None 会被
                子进程强制忽略（参见 _worker_main）
            log_level: 子进程 root logger 等级
            deck_eval_workers_per_env: 子进程内 deck_evaluator pool max_workers 上限，
                通过环境变量 V8_DECK_EVALUATOR_MAX_WORKERS 传递（deck_evaluator
                读取该 var 作为 cap）。n_envs × per_env 应 ≤ 物理核数 - 2，
                典型值 n_envs=4 + per_env=2 = 8 子进程池，外加 4 个 worker 主进程，
                总 12 进程在 8-10 核 M-series 上仍可控。
        """
        if n_envs < 1:
            raise ValueError(f"n_envs must be >= 1, got {n_envs}")
        self.n_envs = int(n_envs)
        self._env_kwargs = dict(env_kwargs or {})
        self._log_level = log_level
        self._deck_eval_workers_per_env = max(1, int(deck_eval_workers_per_env))

        # macOS + MPS 强制 spawn
        ctx = mp.get_context("spawn")

        self._parent_conns: List[Connection] = []
        self._procs: List[mp.process.BaseProcess] = []
        self._closed: bool = False

        for i in range(self.n_envs):
            parent_conn, child_conn = ctx.Pipe(duplex=True)
            p = ctx.Process(
                target=_worker_main,
                args=(
                    i, child_conn, self._env_kwargs, self._log_level,
                    self._deck_eval_workers_per_env,
                ),
                # 重要：daemon=False。daemon=True 会让子进程不能再 spawn 子 pool
                # （deck_evaluator._get_pool 在子进程内会 fail with
                # "daemonic processes are not allowed to have children"）。
                # 代价：必须显式 close + join，atexit registry 兜底。
                daemon=False,
                name=f"v8-env-{i}",
            )
            p.start()
            # 子端在子进程内用，主进程关掉自己拿的副本
            child_conn.close()
            self._parent_conns.append(parent_conn)
            self._procs.append(p)

        # 注册到 atexit registry：进程退出前若没 close 自动关
        _OPEN_PARALLEL_ENVS.add(self)

        logger.info(
            "ParallelV8Env: spawned %d worker processes (pids=%s) "
            "deck_eval_workers_per_env=%d",
            self.n_envs,
            [p.pid for p in self._procs],
            self._deck_eval_workers_per_env,
        )

    # ---------------------------------------------------------------
    # 内部：广播 cmd / 收 reply
    # ---------------------------------------------------------------

    def _broadcast(self, cmd_per_env: List[Tuple[Any, ...]]) -> List[Any]:
        """给每个 env 发不同的 cmd，按下标顺序收 reply。

        子进程错误 → ("err", msg) → 主进程 raise RuntimeError(包含 env idx + msg)。
        """
        if self._closed:
            raise RuntimeError("ParallelV8Env: already closed")
        if len(cmd_per_env) != self.n_envs:
            raise ValueError(
                f"cmd_per_env len={len(cmd_per_env)} != n_envs={self.n_envs}"
            )

        # 1) 发
        for i, conn in enumerate(self._parent_conns):
            conn.send(cmd_per_env[i])

        # 2) 收
        results: List[Any] = [None] * self.n_envs
        for i, conn in enumerate(self._parent_conns):
            try:
                reply = conn.recv()
            except (EOFError, BrokenPipeError) as e:
                raise RuntimeError(
                    f"ParallelV8Env env-{i}: pipe broken ({type(e).__name__}: {e}); "
                    f"worker likely crashed. Check stderr."
                ) from e
            if not isinstance(reply, tuple) or len(reply) != 2:
                raise RuntimeError(
                    f"ParallelV8Env env-{i}: malformed reply {reply!r}"
                )
            tag, payload = reply
            if tag == "err":
                raise RuntimeError(f"ParallelV8Env env-{i}: worker error: {payload}")
            results[i] = payload
        return results

    # ---------------------------------------------------------------
    # V8EnvBase API
    # ---------------------------------------------------------------

    def reset(self, seeds: List[int]) -> List[V8State]:
        if len(seeds) != self.n_envs:
            raise ValueError(f"seeds len={len(seeds)} != n_envs={self.n_envs}")
        cmds = [("reset", int(s)) for s in seeds]
        return self._broadcast(cmds)  # type: ignore[return-value]

    def step(
        self, action_indices: List[int]
    ) -> Tuple[List[V8State], List[float], List[bool], List[Dict[str, Any]]]:
        if len(action_indices) != self.n_envs:
            raise ValueError(
                f"action_indices len={len(action_indices)} != n_envs={self.n_envs}"
            )
        cmds = [("step", int(a)) for a in action_indices]
        results = self._broadcast(cmds)
        # results[i] = (V8State, reward, done, info)
        states = [r[0] for r in results]
        rewards = [float(r[1]) for r in results]
        dones = [bool(r[2]) for r in results]
        infos = [dict(r[3]) for r in results]
        return states, rewards, dones, infos

    def get_available_actions(self) -> List[List[str]]:
        cmds: List[Tuple[Any, ...]] = [("get_actions",) for _ in range(self.n_envs)]
        return self._broadcast(cmds)  # type: ignore[return-value]

    def set_episode(self, ep_indices: List[int]) -> None:
        """各 env 设 episode idx（trainer 在 collect_rollout 前调用，guard_cap 日志用）。"""
        if len(ep_indices) != self.n_envs:
            raise ValueError(
                f"ep_indices len={len(ep_indices)} != n_envs={self.n_envs}"
            )
        cmds = [("set_episode", int(e)) for e in ep_indices]
        self._broadcast(cmds)

    def reset_perf_counters(self) -> None:
        cmds: List[Tuple[Any, ...]] = [
            ("reset_perf_counters",) for _ in range(self.n_envs)
        ]
        self._broadcast(cmds)

    def ping(self) -> List[str]:
        """smoke 用：每个 worker 回 'pong'。"""
        cmds: List[Tuple[Any, ...]] = [("ping",) for _ in range(self.n_envs)]
        return self._broadcast(cmds)  # type: ignore[return-value]

    def close(self) -> None:
        if self._closed:
            return
        # 礼貌关：发 close 等回 ack
        for conn in self._parent_conns:
            try:
                conn.send(("close",))
            except (BrokenPipeError, OSError):
                pass
        for conn in self._parent_conns:
            try:
                conn.recv()
            except Exception:  # noqa: BLE001
                pass
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass
        # 等子进程退出（给 5s）；超时 terminate
        for p in self._procs:
            p.join(timeout=5.0)
            if p.is_alive():
                logger.warning(
                    "ParallelV8Env: worker pid=%s didn't exit in 5s; terminating",
                    p.pid,
                )
                p.terminate()
                p.join(timeout=2.0)
        self._closed = True
        logger.info("ParallelV8Env: closed all workers")

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass


# ============================================================
# Smoke entry：python -m v8.parallel_env
# ============================================================


def _smoke() -> int:
    """最小 smoke：2 env spawn / ping / reset / get_actions / step 1 次 / close。

    退出码 0=ok，非 0=fail；任何 traceback 立即 raise（caller bash 看 stderr）。
    """
    # 主进程 logger 走 stdout（便于看顺序，跟 trainer 同款）
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
        force=True,
    )

    n = 2
    print(f"[smoke] spawning ParallelV8Env(n_envs={n}) ...", flush=True)
    penv = ParallelV8Env(
        n_envs=n,
        env_kwargs={
            "max_steps_per_episode": 30,
            "deck_eval_freq": 5,
            "verbose": False,
        },
    )

    try:
        print("[smoke] ping ...", flush=True)
        pongs = penv.ping()
        assert pongs == ["pong"] * n, f"ping reply mismatch: {pongs}"
        print(f"[smoke] ping ok: {pongs}", flush=True)

        print("[smoke] reset (seeds=[1, 2]) ...", flush=True)
        states = penv.reset(seeds=[1, 2])
        assert len(states) == n
        for i, s in enumerate(states):
            assert isinstance(s, V8State), f"env-{i} state type {type(s)}"
            print(
                f"[smoke] env-{i} reset: phase={s.phase} floor={s.floor} hp={s.hp}/{s.max_hp} "
                f"deck_size={len(s.deck)}",
                flush=True,
            )

        print("[smoke] get_available_actions ...", flush=True)
        acts = penv.get_available_actions()
        assert len(acts) == n
        for i, a in enumerate(acts):
            print(f"[smoke] env-{i} actions: {len(a)} options (first 3: {a[:3]})", flush=True)

        # 选 idx=0 step 一次（每个 env 独立选择）
        idxs = [0 for _ in range(n)]
        print(f"[smoke] step(action_indices={idxs}) ...", flush=True)
        next_states, rewards, dones, infos = penv.step(idxs)
        assert len(next_states) == n and len(rewards) == n and len(dones) == n
        for i in range(n):
            print(
                f"[smoke] env-{i} step: reward={rewards[i]:.3f} done={dones[i]} "
                f"next_phase={next_states[i].phase} info_phase_after={infos[i].get('phase_after')}",
                flush=True,
            )

        print("[smoke] close ...", flush=True)
    finally:
        penv.close()

    print("[smoke] OK", flush=True)
    return 0


__all__ = ["ParallelV8Env", "V8EnvBase"]


if __name__ == "__main__":
    # 让 multiprocessing spawn 在 __main__ 模式下 fork-safe
    mp.freeze_support()
    sys.exit(_smoke())

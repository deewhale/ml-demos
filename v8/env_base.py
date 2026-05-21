"""V8 RL env 抽象基类（phase 1 scaffolding）。

让 V8Env 和 ParallelV8Env 实现同一个接口，trainer 不感知具体是哪种。

设计取舍：
- 不强制 V8Env 立刻继承 V8EnvBase（避免 phase 1 touching env.py）；
  duck typing 仍工作。V8EnvBase 只作 protocol 文档 + 后续重构 anchor。
- ParallelV8Env 内部按这套 batched API 重 dispatch（每个 method 都返回 list）。

API 风格参考 gym.vector.AsyncVectorEnv：
- reset(seeds: List[int]) -> List[V8State]
- step(action_indices: List[int]) -> Tuple[List[V8State], List[float], List[bool], List[dict]]
- get_available_actions() -> List[List[str]]
- close() -> None

为什么不直接复用 gym.vector：
- V8State 是 dataclass 不是 numpy obs；gym.vector 强制 numpy
- get_available_actions 是 pointer-net 必需的动态 action 列表，gym.vector 不支持
- v8 的 set_episode / reset_perf_counters / runner 等暴露字段都是 V8Env-specific
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

from v8.state import V8State


class V8EnvBase(ABC):
    """V8 RL env 抽象基类（batched 接口）。

    单 env (V8Env) 也可以以 n=1 batch 适配；ParallelV8Env 是 n=N batch 的正经实现。
    """

    # 子类应填：env 实例数（V8Env=1, ParallelV8Env=n_envs）
    n_envs: int = 1

    @abstractmethod
    def reset(self, seeds: List[int]) -> List[V8State]:
        """重置所有 env，返回各自的 init state。

        seeds 长度必须 == n_envs。
        """

    @abstractmethod
    def step(
        self, action_indices: List[int]
    ) -> Tuple[List[V8State], List[float], List[bool], List[Dict[str, Any]]]:
        """批量执行 action，返回 (next_states, rewards, dones, infos)。

        action_indices 长度 == n_envs；每个 idx 对应该 env 当前 available_actions
        中的位置。done=True 的 env 不会被自动 reset（caller 显式管理）。
        """

    @abstractmethod
    def get_available_actions(self) -> List[List[str]]:
        """各 env 当前可选 action 字符串列表（长度 == n_envs）。"""

    @abstractmethod
    def close(self) -> None:
        """关闭所有 env / 子进程。"""


__all__ = ["V8EnvBase"]

"""V8 引擎后端层（GameBackend 中间层）。

把"引擎"从 env.py 解耦：env 只跟 GameBackend 接口说话，不直接触碰
StSRLSolver 的 GameRunner / TurnSolverAdapter。

后端实现：
- StSRLBackend（stsrl_backend.py）：包当前 StSRLSolver Python 引擎，第一个后端。
- 未来：真实游戏（CommunicationMod，参考 mod_state_adapter.py）/ Rust 引擎 / 其它。

stage1（2026-06）：纯重构，行为不变。详见 protocol.py 顶部 docstring 的接口契约
与 pass-through 债清单。
"""

from v8.backends.protocol import GameBackend

__all__ = ["GameBackend"]

"""V8CombatNetWrapper：让 V8Model 充当 StSRLSolver TurnSolverAdapter 的 combat_net。

按 docs/v8_design_principles.md 第 2 点（搜索 + 模型联合）要求：让 model 在战斗内
参与决策。StSRLSolver TurnSolverAdapter 已原生支持 `combat_net` 参数（见
external/StSRLSolver/.../turn_solver.py:1117-1152），其期待接口：

    combat_net.predict(combat_obs: np.ndarray) -> float

其中 combat_obs 由 StSRLSolver 内部的 CombatStateEncoder.encode(engine) 产生
（298 维 float32，见 turn_solver.py:1132-1139）。返回值是 leaf value（scalar），
search 在 leaf 节点用它替代 hand-rolled heuristic 评估。

设计要点：
- V8Model 的战斗 head 输出 [H*M+1] logits + value head 输出 scalar。但 search
  期待的是 leaf value（"这局面有多好"），所以这里用 V8Model 的 value_head 比 logits
  更合适。
- combat_obs 是 StSRLSolver 自己的 298 维向量，跟 V8State / V8Model 期待输入完全
  不同。本 wrapper 暂用 stub：返回 max(logits) 风格的"局面 score"（基于 V8Model
  对 obs 的简单 MLP 投影）；**真版本** 需要 model 内部支持把 CombatStateEncoder
  output 映射到 value（额外加一个 small head）。
- call_count：每次 predict 自增，smoke 验证 hook 是否真接通。

Round 2 范围：先 stub，验证 hook 能用。后续填实需要 model 加一个
combat_value_from_obs head（接 298-dim 输入 → scalar）。
"""

from __future__ import annotations

import logging
from typing import Optional, TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from v8.model import V8Model


logger = logging.getLogger(__name__)


class _CombatObsValueHead(nn.Module):
    """临时 head：把 StSRLSolver CombatStateEncoder 的 298 维 obs 映射到 scalar。

    Round 2 stub：随机权重，不参与训练。等收完 turn_records 数据后，model 加
    config 训这个 head（以 search 的 multi-turn evaluator 输出为 target，或者
    直接用 deck_evaluator outcome 作为 leaf reward 训）。

    保留它的目的：让 wrapper.predict 真返回个 model-derived float，证明 hook
    接通；同时为后续训练留 head 位置。
    """

    COMBAT_OBS_DIM: int = 298  # 与 StSRLSolver CombatStateEncoder.COMBAT_DIM 对齐

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(self.COMBAT_OBS_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class V8CombatNetWrapper:
    """让 V8Model 符合 TurnSolverAdapter combat_net 接口。

    用法：
        from v8.model import V8Model
        from v8.combat_net_wrapper import V8CombatNetWrapper

        model = V8Model()
        wrapper = V8CombatNetWrapper(model)
        env = V8Env(combat_net_wrapper=wrapper)
        # ... 跑 env.step ... search 内部会调 wrapper.predict(obs) → float

    Args:
        v8_model: 已实例化的 V8Model
        combat_obs_head: 可选 nn.Module，把 298 维 obs 映射成 scalar；不传则
            内部初始化 _CombatObsValueHead（随机权重 stub）

    属性：
        call_count: 累计 predict 调用次数（smoke 验证用，> 0 证明 hook 接通）
    """

    def __init__(
        self,
        v8_model: "V8Model",
        combat_obs_head: Optional[nn.Module] = None,
    ):
        self.model = v8_model
        # device 与 model 对齐
        try:
            self._device = next(v8_model.parameters()).device
        except StopIteration:
            self._device = torch.device("cpu")
        self._obs_head = combat_obs_head or _CombatObsValueHead().to(self._device)
        self._obs_head.eval()  # Round 2：暂不训练
        self.call_count: int = 0
        # 累计 sum 用于 smoke debug
        self._sum_value: float = 0.0

    def predict(self, combat_obs) -> float:
        """search leaf 评估：obs → scalar value。

        Args:
            combat_obs: numpy.ndarray shape [298,] float32（StSRLSolver
                CombatStateEncoder 输出）

        Returns:
            scalar float（被 search 当 leaf value 用）
        """
        self.call_count += 1

        try:
            arr = np.asarray(combat_obs, dtype=np.float32)
            if arr.ndim == 0:
                arr = arr.reshape(1)
            x = torch.from_numpy(arr).to(self._device)
            if x.dim() == 1:
                x = x.unsqueeze(0)
            with torch.no_grad():
                y = self._obs_head(x)
            val = float(y.item() if y.numel() == 1 else y.mean().item())
        except Exception as e:  # noqa: BLE001
            # search 不该被 wrapper 错误带挂；fail-safe 返回 0
            logger.warning(
                "V8CombatNetWrapper.predict failed: %s: %s (returning 0.0)",
                type(e).__name__, e,
            )
            val = 0.0

        self._sum_value += val
        return val

    def reset_stats(self) -> None:
        """smoke / debug 用，重置计数。"""
        self.call_count = 0
        self._sum_value = 0.0

    @property
    def avg_value(self) -> float:
        return self._sum_value / max(1, self.call_count)


__all__ = ["V8CombatNetWrapper", "_CombatObsValueHead"]

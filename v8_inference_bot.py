"""
V8 推理 Bot — V7Bot 子类，COMBAT phase 用 V8 模型决策，其他 phase 复用 V7。

设计
----
- COMBAT 决策点：构造 record（同 collector 格式）→ encode_state → model.forward
  → mask 非合法动作 → argmax → 返回 actions[idx]
- 非 COMBAT 决策：直接 super()._pick_action（V7 strategy 模块）
- 不写 JSONL，不调 solver（COMBAT 不会触达 V7 solver，节省时间）

这样 V8 model 接管的就是「战斗回合」决策这一个核心位，其他流程仍由 V7
strategy 与 reward/relic 选择保证可玩性。直接对比 final_floor / win 即可
评估 V8 战斗能力是否追上 V7。
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch

from v7_bot import V7Bot
from sts_paths import ensure_on_sys_path
ensure_on_sys_path()

from packages.engine.game import GamePhase  # noqa: E402

from v8_data_collector import build_combat_record
from v8_model import V8SmokeModel, encode_state, MAX_ACTIONS

logger = logging.getLogger(__name__)


class V8InferenceBot(V7Bot):
    """V7Bot 子类；COMBAT phase 用 V8 模型 argmax。"""

    def __init__(
        self,
        *,
        model_path: str,
        solver_budgets=None,
        verbose: bool = False,
    ):
        super().__init__(solver_budgets=solver_budgets, verbose=verbose)
        self._model = V8SmokeModel()
        state_dict = torch.load(model_path, map_location="cpu")
        self._model.load_state_dict(state_dict)
        self._model.eval()

    def _pick_action(self, runner, actions):
        # 非 COMBAT phase → V7 父类决策
        if runner.phase != GamePhase.COMBAT or runner.current_combat is None:
            return super()._pick_action(runner, actions)

        # COMBAT phase → V8 model
        try:
            rec = build_combat_record(runner, actions, chosen_action=None, seed=None)
            state_vec = encode_state(rec).unsqueeze(0)  # [1, STATE_DIM]
            with torch.no_grad():
                logits, _value = self._model(state_vec)  # logits: [1, MAX_ACTIONS]
            logits = logits[0]
            # mask 超出 actions 数量的位置
            n_avail = min(len(actions), MAX_ACTIONS)
            if n_avail == 0:
                return None
            masked = logits.clone()
            if MAX_ACTIONS > n_avail:
                masked[n_avail:] = float("-inf")
            idx = int(torch.argmax(masked).item())
            if idx >= len(actions):
                idx = 0
            return actions[idx]
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "V8 inference failed at floor=%d turn=%s: %s — fallback first action",
                runner.run_state.floor,
                getattr(runner.current_combat.state, "turn", "?"),
                e,
            )
            return actions[0]


__all__ = ["V8InferenceBot"]

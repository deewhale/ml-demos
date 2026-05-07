"""
V8 推理 Bot — V8Bot 子类，元决策 phase 用 V8MetaModel argmax，COMBAT 仍走 solver。

设计
----
- 元决策 phase（NEOW / MAP_NAVIGATION / COMBAT_REWARDS / EVENT / SHOP / REST /
  TREASURE / BOSS_REWARDS）：构造与训练一致的 record（复用
  `v8_data_collector._build_meta_state` + `_phase_options_summary` +
  `_action_to_str`），过 `encode_record` 拿 tensors → model.forward → argmax → idx →
  返回 actions[idx]。
- 如果 model idx 越界 / 找不到对应 action → fallback super()._pick_action（让
  V8Bot 走 v8_strategy.* 启发式）。
- COMBAT phase 直接 `return super()._pick_action(...)`（仍走 TurnSolver）。
- 不写 JSONL，不调 reward 决策的 model（COMBAT_REWARDS 在父类是 _pick_action 内部
  分支，super() 时是 _pick_reward_action；collector 训练数据是按 _pick_action 一
  次调用 = 一条 record 的，COMBAT_REWARDS 的 actions 列表本身就是当前可选的所有
  RewardAction，所以这里只 override _pick_action 就够了，不必再 override
  _pick_reward_action）。

state_dict 直接 load 到 V8MetaModel；device 优先 mps，fallback cpu。
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

import torch

from v8_bot import V8Bot
from sts_paths import ensure_on_sys_path
ensure_on_sys_path()

from packages.engine.game import GamePhase  # noqa: E402

# 与训练时同一套 schema 构造函数（_build_meta_state / _phase_options_summary /
# _action_to_str），保证推理 record 和训练 record 字段完全一致。
from v8_data_collector import (  # noqa: E402
    _META_PHASES,
    _action_to_str,
    _build_meta_state,
    _phase_options_summary,
    _teacher_label_for_phase,
)
from v8_model import (  # noqa: E402
    MAX_ACTIONS,
    V8MetaModel,
    encode_record,
)


# ----------------------------------------------------------------------
# 归因日志（attribution logger）
# ----------------------------------------------------------------------
# 模块级 list，每条 record 是一个 phase 的元决策细节（state + model idx +
# teacher idx + match）。flush 时写 JSONL 并清空。
_ATTRIBUTION_LOG: List[Dict[str, Any]] = []


def _record_attribution(rec: Dict[str, Any]) -> None:
    """把单条 attribution 加入模块级 buffer。"""
    _ATTRIBUTION_LOG.append(rec)


def flush_attribution_log(output_path: str) -> int:
    """flush buffer 到 JSONL 文件并清空，返回写出的条数。"""
    n = len(_ATTRIBUTION_LOG)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for r in _ATTRIBUTION_LOG:
            f.write(json.dumps(r, default=str) + "\n")
    _ATTRIBUTION_LOG.clear()
    return n

# CPU 限制（与 trainer 对齐；torch import 后只能用 set_num_threads）
torch.set_num_threads(2)
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

logger = logging.getLogger(__name__)


def _select_device() -> torch.device:
    """优先 mps，fallback cpu。"""
    try:
        if torch.backends.mps.is_available():
            return torch.device("mps")
    except Exception:  # noqa: BLE001
        pass
    return torch.device("cpu")


def _pad_seq(seq: List[int], pad_to: int) -> torch.Tensor:
    """单条序列 → [1, pad_to] long ids。"""
    pad_to = max(pad_to, 1)
    out = torch.zeros(1, pad_to, dtype=torch.long)
    L = min(len(seq), pad_to)
    if L > 0:
        out[0, :L] = torch.tensor(seq[:L], dtype=torch.long)
    return out


def _mask_seq(seq_len: int, pad_to: int) -> torch.Tensor:
    """单条序列对应的 mask → [1, pad_to] float。"""
    pad_to = max(pad_to, 1)
    out = torch.zeros(1, pad_to, dtype=torch.float32)
    L = min(seq_len, pad_to)
    if L > 0:
        out[0, :L] = 1.0
    return out


class V8InferenceBot(V8Bot):
    """V8Bot 子类；元决策 phase 用 V8MetaModel argmax 选 action。"""

    def __init__(
        self,
        *,
        model_path: str,
        solver_budgets=None,
        verbose: bool = False,
    ):
        super().__init__(solver_budgets=solver_budgets, verbose=verbose)
        self._device = _select_device()

        # 加载 state_dict（v8_trainer.py 存的就是 state_dict） — 见 v8_inference_bot.py:101
        self._model = V8MetaModel()
        state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
        self._model.load_state_dict(state_dict)
        self._model.to(self._device)
        self._model.eval()

        # 调试统计
        self._model_calls = 0
        self._model_oob = 0          # idx 越界 fallback
        self._model_exc = 0          # 异常 fallback
        self._first_exc: Optional[str] = None
        # forward 一次缓存的有效位 logits（cpu, [n_avail]）；attribution 用
        self._last_logits: Optional[torch.Tensor] = None
        logger.info(
            "V8InferenceBot loaded model_path=%s device=%s",
            model_path, self._device,
        )

    # ------------------------------------------------------------------
    # 单条 record → forward 一次拿到 logits，argmax → idx
    # ------------------------------------------------------------------

    def _model_pick_idx(
        self, runner, actions, n_avail: int
    ) -> Optional[int]:
        """构造 record + forward + argmax；越界返回 None。

        副作用：把这次 forward 的 logits（cpu tensor，仅有效位）缓存在
        `self._last_logits`，供 attribution logger 算 top-k probs。
        """
        self._last_logits = None
        if n_avail <= 0:
            return None

        # 与训练时 _build_meta_record 一致的 schema（精简版：推理只需 state +
        # available_actions + phase + floor + act；不需要 teacher_label / action）
        record = {
            "seed": None,
            "floor": runner.run_state.floor,
            "act": runner.run_state.act,
            "phase": runner.phase.name,
            "state": _build_meta_state(runner),
            "available_actions": [_action_to_str(a) for a in actions],
            "action": -1,
            "teacher_label": 0,            # encode_record 要 int；推理用不到
            "teacher_assessment": None,
            "bottled_ai_error": "not_in_combat",
        }

        encoded = encode_record(record)

        # padding 到 actions 实际长度（与训练 _pad_1d 行为一致：pad_to = max len）。
        # 注意：训练时 batch 内 pad_to = batch 内 max；推理 batch=1，所以 pad_to =
        # n_avail 即可。
        pad_to_actions = max(min(n_avail, MAX_ACTIONS), 1)
        action_ids = _pad_seq(encoded["action_ids"], pad_to_actions)
        action_mask = _mask_seq(min(n_avail, MAX_ACTIONS), pad_to_actions)

        deck_pad = max(len(encoded["deck_ids"]), 1)
        relic_pad = max(len(encoded["relic_ids"]), 1)
        potion_pad = max(len(encoded["potion_ids"]), 1)

        deck_ids = _pad_seq(encoded["deck_ids"], deck_pad)
        deck_mask = _mask_seq(len(encoded["deck_ids"]), deck_pad)
        relic_ids = _pad_seq(encoded["relic_ids"], relic_pad)
        relic_mask = _mask_seq(len(encoded["relic_ids"]), relic_pad)
        potion_ids = _pad_seq(encoded["potion_ids"], potion_pad)
        potion_mask = _mask_seq(len(encoded["potion_ids"]), potion_pad)

        scalars = torch.tensor([encoded["scalars"]], dtype=torch.float32)
        phase_idx = torch.tensor([encoded["phase_idx"]], dtype=torch.long)

        dev = self._device
        with torch.no_grad():
            logits = self._model(
                deck_ids.to(dev), deck_mask.to(dev),
                relic_ids.to(dev), relic_mask.to(dev),
                potion_ids.to(dev), potion_mask.to(dev),
                scalars.to(dev), phase_idx.to(dev),
                action_ids.to(dev), action_mask.to(dev),
            )                                              # [1, pad_to_actions]

        # mps 上 argmax 没问题；拿回 cpu int
        # 缓存有效位 logits 供 attribution 用（截到 n_avail）
        valid_n = min(n_avail, MAX_ACTIONS)
        try:
            self._last_logits = logits[0, :valid_n].detach().to("cpu").float()
        except Exception:  # noqa: BLE001
            self._last_logits = None

        idx = int(torch.argmax(logits, dim=-1)[0].item())
        if idx < 0 or idx >= n_avail:
            return None
        return idx

    # ------------------------------------------------------------------
    # override _pick_action：元决策走 model，COMBAT 走父类
    # ------------------------------------------------------------------

    def _pick_action(self, runner, actions):
        phase = runner.phase

        # COMBAT / 其他非元决策 phase → 父类
        if phase not in _META_PHASES:
            return super()._pick_action(runner, actions)

        if not actions:
            return super()._pick_action(runner, actions)

        # 元决策点：跑一次 model
        model_failed = False
        try:
            self._model_calls += 1
            idx = self._model_pick_idx(runner, actions, n_avail=len(actions))
        except Exception as e:  # noqa: BLE001
            self._model_exc += 1
            if self._first_exc is None:
                self._first_exc = f"{type(e).__name__}: {e}"
            logger.warning(
                "V8 model inference failed at floor=%d phase=%s: %s — fallback super()",
                runner.run_state.floor, phase.name, e,
            )
            idx = None
            model_failed = True

        # 同时算一次老师 idx（只用于归因记录，不影响决策；与 v8_data_collector
        # 训练 label 完全一致：调 _teacher_label_for_phase）
        teacher_idx: Optional[int] = None
        if not model_failed:
            try:
                t = _teacher_label_for_phase(runner, actions)
                if t == -1:
                    # COMBAT_REWARDS 的 sentinel：与父类返回的 action 同步，
                    # 这里取 model_idx 作为 fallback（与训练 label 兜底口径一致）
                    teacher_idx = idx if idx is not None else 0
                else:
                    teacher_idx = int(t)
            except Exception as e:  # noqa: BLE001
                logger.debug(
                    "teacher_idx calc failed (phase=%s): %s", phase.name, e
                )
                teacher_idx = None

        # 记录 attribution（model 成功 forward 时才记；fallback 走老师不记）
        if not model_failed and idx is not None:
            try:
                self._record_attribution_for_phase(
                    runner=runner,
                    actions=actions,
                    phase=phase,
                    model_idx=idx,
                    teacher_idx=teacher_idx,
                )
            except Exception as e:  # noqa: BLE001
                logger.debug("attribution record failed: %s", e)

        if idx is None:
            self._model_oob += 1
            return super()._pick_action(runner, actions)

        # 正常情况下直接返回 actions[idx]
        return actions[idx]

    # ------------------------------------------------------------------
    # 归因 record 构造（per-phase 决策细节）
    # ------------------------------------------------------------------

    def _record_attribution_for_phase(
        self,
        *,
        runner,
        actions,
        phase,
        model_idx: int,
        teacher_idx: Optional[int],
    ) -> None:
        """构造一条 attribution record 写入模块级 buffer。"""
        rs = runner.run_state

        # state summary（截断防爆）
        try:
            deck_ids = [getattr(c, "id", "?") for c in (rs.deck or [])]
        except Exception:  # noqa: BLE001
            deck_ids = []
        try:
            relic_ids = [getattr(r, "id", "?") for r in (rs.relics or [])]
        except Exception:  # noqa: BLE001
            relic_ids = []
        try:
            potion_ids = [
                (getattr(p, "id", "?") if p is not None else None)
                for p in (rs.potions or [])
            ]
        except Exception:  # noqa: BLE001
            potion_ids = []

        # 选项摘要 (人类可读)
        try:
            options_summary = _phase_options_summary(runner, actions)
        except Exception:  # noqa: BLE001
            options_summary = []
        # available_actions raw repr（截断到 MAX_ACTIONS）
        avail_strs = [_action_to_str(a) for a in actions[:MAX_ACTIONS]]

        # top-k probs from cached logits
        top3: List[Dict[str, Any]] = []
        if self._last_logits is not None and self._last_logits.numel() > 0:
            try:
                probs = torch.softmax(self._last_logits, dim=-1)
                k = int(min(3, probs.numel()))
                topv, topi = torch.topk(probs, k)
                for v, i in zip(topv.tolist(), topi.tolist()):
                    top3.append({"idx": int(i), "prob": float(v)})
            except Exception:  # noqa: BLE001
                top3 = []

        rec: Dict[str, Any] = {
            "phase": phase.name,
            "floor": int(getattr(rs, "floor", -1) or -1),
            "act": int(getattr(rs, "act", -1) or -1),
            "state_summary": {
                "hp": int(getattr(rs, "current_hp", 0) or 0),
                "max_hp": int(getattr(rs, "max_hp", 0) or 0),
                "gold": int(getattr(rs, "gold", 0) or 0),
                "deck_size": len(deck_ids),
                "deck": deck_ids[:60],          # 截断防爆
                "relics": relic_ids[:30],
                "potions": potion_ids[:5],
            },
            "n_options": len(actions),
            "available_actions": avail_strs,
            "options_summary": options_summary[:MAX_ACTIONS],
            "model_idx": int(model_idx),
            "model_top3": top3,
            "teacher_idx": (None if teacher_idx is None else int(teacher_idx)),
            "match": (
                None if teacher_idx is None else (int(model_idx) == int(teacher_idx))
            ),
        }
        _record_attribution(rec)


__all__ = [
    "V8InferenceBot",
    "flush_attribution_log",
    "_ATTRIBUTION_LOG",
]

"""
V7 Bot: 装配 TurnSolver + V7 evaluator + 策略层，跑一局完整 Watcher。

用法：
    from v7_bot import V7Bot
    bot = V7Bot()
    result = bot.run(seed=42, ascension=0, character="watcher")
"""

from __future__ import annotations

import json
import logging
import signal
import sys
import time
from typing import Any, Dict, List, Optional, TextIO

# 保证能 import StSRLSolver
sys.path.insert(0, "STSRLSOLVER_PATH")

from packages.engine.game import (
    GameRunner, GamePhase,
    PathAction, NeowAction, CombatAction, RewardAction,
    EventAction, ShopAction, RestAction, TreasureAction, BossRewardAction,
)
from packages.training.turn_solver import TurnSolverAdapter
from packages.engine.content.cards import ALL_CARDS

import v7_strategy as strat
from v7_evaluator import patch_turn_solver_eval


# 必败阈值：solver 最佳 score 低于此值视为"必败"，走 defensive fallback
_LOSS_THRESHOLD = -500_000.0

logger = logging.getLogger(__name__)


# Budget (降预算加速：elite 2s→500ms, boss 5s→1500ms)
SOLVER_BUDGETS = {
    "monster": (50.0,    5_000,   2_000),   # 50ms, cap 2s (保持)
    "elite":   (500.0,   20_000,  2_000),   # 500ms, cap 2s
    "boss":    (1_500.0, 40_000,  3_000),   # 1.5s, cap 3s (Run 6 baseline)
}


class V7Bot:
    """Self-contained V7 bot wrapping GameRunner + TurnSolver + strategy handlers."""

    def __init__(self, solver_budgets=None, verbose: bool = False):
        # Patch 评估函数（幂等）
        patch_turn_solver_eval()

        self._budgets = solver_budgets or SOLVER_BUDGETS
        self._verbose = verbose
        self._adapter = TurnSolverAdapter(
            time_budget_ms=50.0,
            node_budget=5_000,
            solver_budgets=self._budgets,
        )

    def run(
        self,
        seed,
        ascension: int = 0,
        character: str = "watcher",
        max_actions: int = 20_000,
        log_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """跑一局，返回 summary dict。"""
        t0 = time.monotonic()
        runner = GameRunner(
            seed=seed,
            ascension=ascension,
            character=character,
            skip_neow=False,    # V7 我们手动处理 Neow
            verbose=self._verbose,
        )
        self._adapter.reset()

        floor_hp: List[tuple] = []
        prev_floor = -1
        actions_taken = 0
        error_msg = None

        # ---------- per-floor logging ----------
        log_fh: Optional[TextIO] = open(log_path, "w") if log_path else None
        last_combat_sig = None  # (floor, room_type) of most recent combat-start we logged

        def _emit(rec):
            if log_fh is None:
                return
            try:
                log_fh.write(json.dumps(rec, default=str) + "\n")
                log_fh.flush()
            except Exception:  # noqa: BLE001
                pass

        def _room_type_str() -> str:
            try:
                rt = runner.get_current_room_type()
                return rt.name if rt is not None else "START"
            except Exception:  # noqa: BLE001
                return "UNKNOWN"

        def _snapshot_floor(tag: str):
            rs = runner.run_state
            rec = {
                "event": tag,
                "t": round(time.monotonic() - t0, 3),
                "seed": seed,
                "floor": rs.floor,
                "act": rs.act,
                "phase": runner.phase.name,
                "room_type": _room_type_str(),
                "hp": rs.current_hp,
                "max_hp": rs.max_hp,
                "gold": rs.gold,
                "deck_size": len(rs.deck),
                "relics": [r.id for r in rs.relics][:30],
                "actions_taken": actions_taken,
            }
            _emit(rec)

        def _snapshot_combat_start():
            nonlocal last_combat_sig
            cc = runner.current_combat
            rs = runner.run_state
            sig = (rs.floor, runner.current_room_type)
            if sig == last_combat_sig:
                return
            last_combat_sig = sig
            enemies = []
            if cc is not None:
                for e in cc.state.enemies:
                    enemies.append({
                        "id": getattr(e, "id", ""),
                        "name": getattr(e, "name", ""),
                        "hp": e.hp,
                        "max_hp": e.max_hp,
                        "type": getattr(e, "enemy_type", ""),
                    })
            _emit({
                "event": "combat_start",
                "t": round(time.monotonic() - t0, 3),
                "seed": seed,
                "floor": rs.floor,
                "act": rs.act,
                "room_type": runner.current_room_type,
                "hp": rs.current_hp,
                "max_hp": rs.max_hp,
                "enemies": enemies,
                "deck_size": len(rs.deck),
                "actions_taken": actions_taken,
            })

        prev_phase = None
        prev_hp = runner.run_state.current_hp
        # ---------------------------------------

        while not runner.game_over and actions_taken < max_actions:
            actions = runner.get_available_actions()
            if not actions:
                break

            cur_floor = runner.run_state.floor
            if cur_floor != prev_floor:
                floor_hp.append((cur_floor, runner.run_state.current_hp))
                prev_floor = cur_floor
                _snapshot_floor("floor_enter")

            # 侦测 combat 开始
            if (runner.phase.name == "COMBAT"
                    and prev_phase != "COMBAT"
                    and runner.current_combat is not None):
                _snapshot_combat_start()
                prev_hp = runner.run_state.current_hp
            # 侦测 combat 结束（离开 COMBAT phase）
            if prev_phase == "COMBAT" and runner.phase.name != "COMBAT":
                rs = runner.run_state
                _emit({
                    "event": "combat_end",
                    "t": round(time.monotonic() - t0, 3),
                    "seed": seed,
                    "floor": rs.floor,
                    "hp": rs.current_hp,
                    "hp_lost": prev_hp - rs.current_hp,
                    "actions_taken": actions_taken,
                })
                prev_hp = rs.current_hp
            prev_phase = runner.phase.name

            try:
                action = self._pick_action(runner, actions)
            except Exception as e:  # noqa: BLE001
                error_msg = f"_pick_action crash: {type(e).__name__}: {e}"
                logger.exception("V7 _pick_action crashed")
                # fallback 第一个
                action = actions[0]

            if action is None:
                action = actions[0]

            ok = runner.take_action(action)
            if not ok:
                # 跳过非法 action，防止死循环
                logger.warning("take_action returned False at floor=%d phase=%s action=%s",
                               runner.run_state.floor, runner.phase.name, action)
                # 随便选一个有效 action
                action = actions[0]
                runner.take_action(action)
            actions_taken += 1

        elapsed = time.monotonic() - t0
        stats = runner.get_run_statistics() if hasattr(runner, "get_run_statistics") else {}

        # final 记录
        if log_fh is not None:
            try:
                _emit({
                    "event": "game_end",
                    "t": round(elapsed, 3),
                    "seed": seed,
                    "floor": runner.run_state.floor,
                    "hp": runner.run_state.current_hp,
                    "won": bool(stats.get("game_won", False)),
                    "actions_taken": actions_taken,
                    "error": error_msg,
                })
                log_fh.close()
            except Exception:  # noqa: BLE001
                pass

        return {
            "seed": seed,
            "ascension": ascension,
            "character": character,
            "game_won": bool(stats.get("game_won", False)),
            "game_lost": bool(stats.get("game_lost", runner.game_over and not stats.get("game_won", False))),
            "final_floor": runner.run_state.floor,
            "final_act": stats.get("final_act", runner.run_state.act),
            "final_hp": runner.run_state.current_hp,
            "max_hp": runner.run_state.max_hp,
            "deck_size": len(runner.run_state.deck),
            "gold": runner.run_state.gold,
            "decisions": actions_taken,
            "elapsed_s": round(elapsed, 2),
            "floor_hp": floor_hp,
            "error": error_msg,
        }

    # ---------------------------------------------------------------
    # Defensive fallback (修法 A)
    # ---------------------------------------------------------------

    def _defensive_fallback_if_losing(self, runner, actions):
        """如果 solver 认为必败（最佳 score < _LOSS_THRESHOLD）且玩家 HP>0，
        返回一个 defensive action（优先打 block 最大的可负担 skill，其次 0 费 block 牌）。
        否则返回 None 让主路径使用 solver 结果。
        """
        scores = getattr(self._adapter, "last_solver_scores", None)
        if not scores:
            return None
        # scores 是 List[Tuple[desc, score]]，取最大 score
        try:
            best = max(s for _, s in scores)
        except (TypeError, ValueError):
            return None
        if best >= _LOSS_THRESHOLD:
            return None

        engine = getattr(runner, "current_combat", None)
        if engine is None:
            return None
        state = engine.state
        player = getattr(state, "player", None)
        if player is None or getattr(player, "hp", 0) <= 0:
            return None

        hand = list(getattr(state, "hand", []) or [])
        energy = getattr(state, "energy", 0)

        # 找可负担（cost<=energy 或 cost==0）的、base_block>0 的牌
        best_affordable = None  # (block, cost, card_idx)
        best_free = None
        for i, card_id in enumerate(hand):
            data = ALL_CARDS.get(card_id)
            if data is None:
                continue
            block = getattr(data, "base_block", -1)
            if block is None or block <= 0:
                continue
            cost = getattr(data, "cost", 0) or 0
            # X-cost 等异常成本，保守跳过
            if cost < 0:
                continue
            if cost == 0:
                if best_free is None or block > best_free[0]:
                    best_free = (block, cost, i)
            if cost <= energy:
                if best_affordable is None or block > best_affordable[0]:
                    best_affordable = (block, cost, i)

        # 构造 play_card 动作并在 actions 里找匹配
        def _find_play(card_idx):
            for a in actions:
                if isinstance(a, CombatAction) and a.action_type == "play_card" and a.card_idx == card_idx:
                    return a
            return None

        # 1) 能量允许：block 最高的 skill
        if best_affordable is not None:
            a = _find_play(best_affordable[2])
            if a is not None:
                return a
        # 2) 能量不够：block 最高的 0 费 skill
        if best_free is not None:
            a = _find_play(best_free[2])
            if a is not None:
                return a
        # 3) 都没有：end_turn
        for a in actions:
            if isinstance(a, CombatAction) and a.action_type == "end_turn":
                return a
        return None

    # ---------------------------------------------------------------
    # Dispatch
    # ---------------------------------------------------------------

    def _pick_action(self, runner, actions):
        phase = runner.phase
        rs = runner.run_state

        if phase == GamePhase.COMBAT:
            room_type = runner.current_room_type  # "monster" / "elite" / "boss"

            # === Solver timeout fix ===
            # Bug: TurnSolverAdapter._apply_room_type_budgets 只设置 inner solver,
            # 没设 MultiTurnSolver.time_budget_ms (默认 30s). 又因为 pick_action
            # 先调用 _get_turn_candidates(deadline=30s) 再 solve(30s), 单步 action
            # 最多 60s, elite+boss 组合爆炸可 >90s.
            # 修法: 把 multi_turn 外层预算强制拉到 cap_ms, 并用 SIGALRM 做硬超时保险.
            _rt_key = (room_type or "monster").lower()
            budgets = self._budgets.get(_rt_key)
            cap_ms = budgets[2] if budgets else 3000.0
            # 同步 multi_turn 的外层 deadline 与 cap_ms 一致
            try:
                self._adapter._multi_turn.time_budget_ms = float(cap_ms)
            except AttributeError:
                pass

            # 硬超时保险 (SIGALRM, 主线程), 预算 +2s buffer
            hard_timeout_s = max(1, int(cap_ms / 1000.0) + 2)
            a = None
            timed_out = False
            old_handler = None

            def _on_alarm(signum, frame):
                raise TimeoutError(f"solver hard timeout {hard_timeout_s}s (room={_rt_key})")

            try:
                old_handler = signal.signal(signal.SIGALRM, _on_alarm)
                signal.alarm(hard_timeout_s)
                a = self._adapter.pick_action(actions, runner, room_type=room_type)
            except TimeoutError as te:
                logger.warning("V7 solver hard-timeout: %s, fallback to defensive/first action", te)
                timed_out = True
                a = None
            finally:
                signal.alarm(0)
                if old_handler is not None:
                    signal.signal(signal.SIGALRM, old_handler)

            # 修法 A：solver 返回"必败"（最佳 score < _LOSS_THRESHOLD）
            # 且当前 HP > 0 时，走 defensive fallback 先苟活几回合
            fallback = self._defensive_fallback_if_losing(runner, actions)
            if fallback is not None:
                return fallback
            if a is None and timed_out:
                # 超时 fallback: 优先 EndTurn / 第一个合法动作
                for act in actions:
                    if getattr(act, "action_type", None) == "end_turn":
                        return act
                return actions[0]
            return a if a is not None else actions[0]

        # 进入其他 phase 前 reset adapter cache
        self._adapter.reset()

        if phase == GamePhase.MAP_NAVIGATION:
            idx = strat.pick_path(rs)
            # 保证是有效的 PathAction
            for a in actions:
                if isinstance(a, PathAction) and a.node_index == idx:
                    return a
            return actions[0]

        if phase == GamePhase.NEOW:
            if runner.neow_blessings is None:
                # 触发初始化
                runner.get_available_actions()
            blessings = runner.neow_blessings or []
            idx = strat.pick_neow(blessings) if blessings else 0
            # 找对应 NeowAction
            for a in actions:
                if isinstance(a, NeowAction) and a.choice_index == idx:
                    return a
            return actions[0]

        if phase == GamePhase.COMBAT_REWARDS:
            return self._pick_reward_action(runner, actions)

        if phase == GamePhase.EVENT:
            idx = strat.pick_event_action(rs, runner.current_event_state, actions)
            if 0 <= idx < len(actions):
                return actions[idx]
            return actions[0]

        if phase == GamePhase.SHOP:
            return strat.pick_shop_action(rs, runner.current_shop, actions)

        if phase == GamePhase.REST:
            return strat.pick_rest_action(rs, actions)

        if phase == GamePhase.TREASURE:
            # 默认拿 relic；sapphire key 跳过（A0 一般用不上）
            for a in actions:
                if isinstance(a, TreasureAction) and a.action_type == "take_relic":
                    return a
            return actions[0]

        if phase == GamePhase.BOSS_REWARDS:
            # 从 BossRelicChoices 里挑
            choices = None
            if runner.current_rewards and runner.current_rewards.boss_relics:
                choices = runner.current_rewards.boss_relics
            if choices is None or not choices.relics:
                return actions[0]
            idx = strat.pick_boss_relic(rs, choices)
            for a in actions:
                if isinstance(a, BossRewardAction) and a.relic_index == idx:
                    return a
            return actions[0]

        return actions[0]

    # ---------------------------------------------------------------
    # Reward phase
    # ---------------------------------------------------------------

    def _pick_reward_action(self, runner, actions):
        rs = runner.run_state
        rewards = runner.current_rewards

        # 分类
        gold_a = None
        potion_claim = None
        potion_skip = None
        emerald_claim = None
        emerald_skip = None
        relic_a = None
        proceed_a = None
        card_actions: List = []      # (card_reward_idx, action)
        skip_card_actions: List = [] # (card_reward_idx, action)
        singing_actions: List = []

        for a in actions:
            if not isinstance(a, RewardAction):
                continue
            t = a.reward_type
            if t == "gold":
                gold_a = a
            elif t == "potion":
                potion_claim = a
            elif t == "skip_potion":
                potion_skip = a
            elif t == "emerald_key":
                emerald_claim = a
            elif t == "skip_emerald_key":
                emerald_skip = a
            elif t == "relic":
                relic_a = a
            elif t == "proceed":
                proceed_a = a
            elif t == "card":
                reward_idx = a.choice_index // 100
                card_actions.append((reward_idx, a))
            elif t == "skip_card":
                skip_card_actions.append((a.choice_index, a))
            elif t == "singing_bowl":
                singing_actions.append(a)

        # 优先级：gold -> relic -> emerald -> card -> potion -> proceed
        if gold_a is not None:
            return gold_a
        if relic_a is not None:
            return relic_a
        if emerald_claim is not None:
            return emerald_claim

        # 处理卡牌奖励
        if card_actions and rewards is not None:
            # 按 reward_idx 分组，对每组选最好的
            # 找第一个未解决的 card reward
            for i, cr in enumerate(rewards.card_rewards):
                if cr.is_resolved:
                    continue
                # 在 card_actions 里找对应 reward_idx = i 的
                my_card_actions = [(ri, a) for ri, a in card_actions if ri == i]
                if not my_card_actions:
                    # 对应的 skip
                    for ri, a in skip_card_actions:
                        if ri == i:
                            return a
                    continue
                pick_idx = strat.pick_card_reward(rs, cr)
                if pick_idx is None:
                    # skip
                    for ri, a in skip_card_actions:
                        if ri == i:
                            return a
                    # 没 skip action 就拿第一张
                    return my_card_actions[0][1]
                # 找对应 card_index = pick_idx 的 action
                for ri, a in my_card_actions:
                    card_idx_in_action = a.choice_index % 100
                    if card_idx_in_action == pick_idx:
                        return a
                # fallback
                return my_card_actions[0][1]

        # Potion：MVP 直接不拿（留给 boss 的原则下，我们其实也没真正留）
        # 简化：有 claim 就拿，满了就 skip
        if potion_claim is not None:
            return potion_claim
        if potion_skip is not None:
            return potion_skip

        if emerald_skip is not None:
            return emerald_skip

        if proceed_a is not None:
            return proceed_a

        return actions[0]

"""
V6 Agent 诊断脚本 — 编码检查 + 1局游戏全量 instrumentation
==========================================================
用法: cd /path/to/StSRLSolver && uv run python3 /path/to/diagnose_v6.py
或:   python diagnose_v6.py  (需要 StSRLSolver 在 sys.path 中)

输出: stdout + v6_diagnostic_report.txt
"""

import sys
import os
import io
import time
import random
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict

# 路径设置
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))
from sts_paths import ensure_on_sys_path
ensure_on_sys_path()

# ============================================================
# 输出同时写 stdout 和文件
# ============================================================

class TeeWriter:
    """同时写 stdout 和文件"""
    def __init__(self, filepath):
        self.file = open(filepath, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, text):
        self.stdout.write(text)
        self.file.write(text)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()

REPORT_PATH = BASE_DIR / "v6_diagnostic_report.txt"
tee = TeeWriter(REPORT_PATH)


def out(msg=""):
    """打印一行到 stdout 和报告文件"""
    tee.write(msg + "\n")
    tee.flush()


# ============================================================
# Part 1: 编码健全性检查（不需要游戏模拟器）
# ============================================================

def check_encoding():
    out("=" * 60)
    out("=== PART 1: ENCODING SANITY CHECK ===")
    out("=" * 60)

    from data.sts_data import CARD_DATA, RELIC_DATA, POTION_DATA, UNIFIED_DIM, DIM_NAMES
    from sts_agent_v6 import encode_card, ENTITY_DIM

    issues = []
    warnings = []

    out(f"\nENTITY_DIM = {ENTITY_DIM}, UNIFIED_DIM = {UNIFIED_DIM}")
    if ENTITY_DIM != UNIFIED_DIM:
        issues.append(f"ENTITY_DIM ({ENTITY_DIM}) != UNIFIED_DIM ({UNIFIED_DIM})")
        out(f"  PROBLEM: ENTITY_DIM != UNIFIED_DIM")

    out(f"CARD_DATA entries: {len(CARD_DATA)}")
    out(f"RELIC_DATA entries: {len(RELIC_DATA)}")
    out(f"POTION_DATA entries: {len(POTION_DATA)}")

    # 测试已知卡牌
    test_cards = [
        {"id": "Strike_P", "name": "Strike", "type": "ATTACK", "cost": 1, "damage": 6, "block": 0},
        {"id": "Defend_P", "name": "Defend", "type": "SKILL", "cost": 1, "damage": 0, "block": 5},
        {"id": "Eruption", "name": "Eruption", "type": "ATTACK", "cost": 2, "damage": 9, "block": 0},
    ]

    for card in test_cards:
        vec = encode_card(card)
        n_nonzero = np.count_nonzero(vec)
        card_id = card["id"]
        in_data = card_id in CARD_DATA

        if vec.shape[0] != ENTITY_DIM:
            issues.append(f"{card_id}: shape=({vec.shape[0]},) != ENTITY_DIM ({ENTITY_DIM})")
            mark = "FAIL"
        elif n_nonzero == 0:
            issues.append(f"{card_id}: encoding has 0 non-zeros!")
            mark = "FAIL"
        else:
            mark = "OK"

        out(f"\n  [{mark}] {card_id}:")
        out(f"    shape=({vec.shape[0]},), non-zeros={n_nonzero}, in_CARD_DATA={in_data}")
        out(f"    first 10 dims: {vec[:10].tolist()}")

        # 检查关键维度
        from data.sts_data import _DIM_INDEX
        for dim_name in ["damage", "block", "cost", "is_attack", "is_skill", "is_power"]:
            if dim_name in _DIM_INDEX:
                idx = _DIM_INDEX[dim_name]
                out(f"    {dim_name} (dim {idx}): {vec[idx]:.3f}")

    # 测试未知卡（fallback 路径）
    out("\n  --- Fallback encoding test (unknown card) ---")
    unknown_card = {"id": "FAKE_CARD_XYZ_999", "name": "FakeCard", "type": "ATTACK",
                    "cost": 2, "damage": 10, "block": 0}
    vec_unknown = encode_card(unknown_card)
    n_nz = np.count_nonzero(vec_unknown)
    if vec_unknown.shape[0] != ENTITY_DIM:
        issues.append(f"Unknown card: shape mismatch ({vec_unknown.shape[0]} vs {ENTITY_DIM})")
        out(f"  [FAIL] Unknown card: shape=({vec_unknown.shape[0]},)")
    elif n_nz == 0:
        issues.append("Unknown card fallback produces all-zero vector")
        out(f"  [FAIL] Unknown card: all zeros (fallback broken)")
    else:
        out(f"  [OK] Unknown card: shape=({vec_unknown.shape[0]},), non-zeros={n_nz}")
        out(f"    first 10 dims: {vec_unknown[:10].tolist()}")

    # 空卡牌
    vec_empty = encode_card({})
    if np.count_nonzero(vec_empty) != 0:
        warnings.append("Empty card dict produces non-zero encoding")
        out(f"  [WARN] Empty card dict: {np.count_nonzero(vec_empty)} non-zeros (expected 0)")
    else:
        out(f"  [OK] Empty card dict: all zeros (correct)")

    return issues, warnings


# ============================================================
# Part 2: 运行游戏 + 收集诊断数据
# ============================================================

def run_instrumented_game():
    out("\n" + "=" * 60)
    out("=== PART 2: INSTRUMENTED GAME RUN ===")
    out("=" * 60)

    # 导入模拟器和训练脚本组件
    try:
        from packages.engine.game import GameRunner, GamePhase
        from packages.engine.content.cards import ALL_CARDS
    except ImportError as e:
        out(f"\n  [SKIP] 无法导入 StSRLSolver: {e}")
        out(f"  请在 StSRLSolver 目录下运行: cd /path/to/StSRLSolver && uv run python3 {__file__}")
        return None, [], []

    import sts_agent_v6 as v6
    import train_v6

    # 创建 agent
    model_path = BASE_DIR / "sts_models" / "agent_v6.pt"
    if model_path.exists():
        out(f"\n  加载模型权重: {model_path}")
    else:
        out(f"\n  模型文件不存在，使用随机初始化")

    agent = v6.AgentV6(device="cpu")
    agent.model.eval()
    agent.suppress_ppo = True  # 不触发 PPO 更新

    # ---- 诊断数据收集 ----
    diag = {
        "combat_decisions": [],    # (logits, probs, entropy, value, action, action_prob, reward_breakdown)
        "strategy_decisions": [],  # (type, logits, probs, action, value, reward)
        "all_transitions": [],     # 所有 Transition 对象的摘要
    }

    # Monkey-patch combat_act 收集详细诊断
    _orig_combat_act = agent.combat_act

    def _patched_combat_act(game_state):
        # 获取决策前的 trajectory 长度
        pre_len = len(agent.current_trajectory)

        # 手动计算 logits（和 combat_act 内部逻辑一致）
        gs = game_state.get("game_state", {})
        combat = gs.get("combat_state", {})
        hand = combat.get("hand", [])

        result = _orig_combat_act(game_state)

        # 从新增的 transition 提取信息
        if len(agent.current_trajectory) > pre_len:
            t = agent.current_trajectory[-1]
            card_action = t.action[0]

            # 重新计算 logits 用于诊断
            try:
                tokens = agent._build_tokens(game_state, include_combat=True)
                with torch.no_grad():
                    shared_repr = agent.model.ctx_transformer(tokens)
                    hand_encs = []
                    for c in hand[:v6.MAX_HAND]:
                        card_vec = v6.encode_card(c)
                        tv = torch.FloatTensor(card_vec).to(v6.DEVICE)
                        enc = agent.model.ctx_transformer.token_proj.project(
                            "card", v6.TOKEN_HAND_CARD, tv)
                        hand_encs.append(enc)
                    if hand_encs:
                        hand_encodings = torch.stack(hand_encs)
                    else:
                        hand_encodings = torch.zeros(0, v6.CTX_DIM, device=v6.DEVICE)

                    # card mask
                    card_mask = torch.zeros(v6.NUM_CARD_ACTIONS, device=v6.DEVICE)
                    available = game_state.get("available_commands", [])
                    for i, c in enumerate(hand[:v6.MAX_HAND]):
                        if c.get("is_playable", False):
                            card_mask[i] = 1.0
                    if "end" in available:
                        card_mask[v6.END_TURN_ACTION] = 1.0
                    if card_mask.sum() == 0:
                        card_mask[v6.END_TURN_ACTION] = 1.0

                    monster_encodings = torch.zeros(0, v6.CTX_DIM, device=v6.DEVICE)
                    monster_mask = torch.zeros(v6.MAX_MONSTERS, device=v6.DEVICE)

                    card_logits, _, value = agent.model.combat_head(
                        shared_repr, hand_encodings, monster_encodings, card_mask, monster_mask
                    )

                    probs = torch.softmax(card_logits, dim=-1)
                    entropy = -(probs * torch.log(probs + 1e-8)).sum().item()
                    action_prob = probs[card_action].item()

                    diag["combat_decisions"].append({
                        "logits": card_logits.cpu().numpy().tolist(),
                        "probs": probs.cpu().numpy().tolist(),
                        "entropy": entropy,
                        "value": value.item(),
                        "action": card_action,
                        "action_prob": action_prob,
                        "reward": t.reward,
                        "is_end_turn": card_action == v6.END_TURN_ACTION,
                        "hand_size": len(hand),
                        "n_playable": int(card_mask[:v6.MAX_HAND].sum().item()),
                    })
            except Exception as e:
                diag["combat_decisions"].append({
                    "error": str(e),
                    "action": card_action,
                    "reward": t.reward,
                })

        return result

    agent.combat_act = _patched_combat_act

    # Monkey-patch draft/path/choice 收集策略诊断
    for method_name, decision_type in [("pick_card", "draft"), ("path", "path"),
                                        ("choice", "choice"), ("_pick_from_candidates", "draft")]:
        orig = getattr(agent, method_name, None)
        if orig is None:
            continue

        def make_wrapper(orig_fn, dtype):
            def wrapper(*args, **kwargs):
                pre_len = len(agent.current_trajectory)
                result = orig_fn(*args, **kwargs)
                if len(agent.current_trajectory) > pre_len:
                    t = agent.current_trajectory[-1]
                    diag["strategy_decisions"].append({
                        "type": dtype,
                        "action": t.action,
                        "value": t.value,
                        "log_prob": t.log_prob,
                        "reward": t.reward,
                    })
                return result
            return wrapper

        setattr(agent, method_name, make_wrapper(orig, decision_type))

    # ---- 运行游戏 ----
    seed = random.randint(1, 99999)
    out(f"\n  Seed: {seed}")
    out(f"  Character: Watcher, Ascension: 0")
    out(f"  开始运行...")

    start_time = time.time()
    train_v6.reset_agent_for_game(agent)

    try:
        result = train_v6.run_one_game(agent, seed, ascension=0, training=True, verbose=True)
    except Exception as e:
        out(f"\n  [ERROR] 游戏运行失败: {e}")
        import traceback
        out(traceback.format_exc())
        result = None

    elapsed = time.time() - start_time
    out(f"  耗时: {elapsed:.1f}s")

    if result:
        out(f"  结果: {'WIN' if result['won'] else 'LOSE'} | floor={result['floor']} | "
            f"HP={result['hp']}/{result['max_hp']} | steps={result['steps']}")
        out(f"  combat decisions: {result['n_combat_decisions']}, "
            f"strategy decisions: {result['n_strategy_decisions']}")

    # 收集所有 trajectory 的 transition 摘要
    all_trajs = list(agent.trajectory_buffer) + ([agent.current_trajectory] if agent.current_trajectory else [])
    all_transitions = []
    for traj in all_trajs:
        for t in traj:
            all_transitions.append(t)

    return result, diag, all_transitions


# ============================================================
# Part 3: 分析和报告
# ============================================================

def analyze_and_report(game_result, diag, all_transitions, encoding_issues, encoding_warnings):
    out("\n" + "=" * 60)
    out("=== V6 DIAGNOSTIC REPORT ===")
    out("=" * 60)

    fatal_issues = []
    warnings = list(encoding_warnings)

    # ---- [ENCODING] ----
    out("\n[ENCODING]")
    if encoding_issues:
        for issue in encoding_issues:
            out(f"  PROBLEM: {issue}")
            fatal_issues.append(f"Encoding: {issue}")
    else:
        out(f"  All encoding checks passed")

    # ---- [GAME RESULT] ----
    out("\n[GAME RESULT]")
    if game_result:
        out(f"  Result: {'WIN' if game_result['won'] else 'LOSE'}")
        out(f"  Floor: {game_result['floor']}")
        out(f"  HP: {game_result['hp']}/{game_result['max_hp']}")
        out(f"  Steps: {game_result['steps']}")
    else:
        out(f"  Game did not complete (error or import failure)")
        fatal_issues.append("Game failed to run")
        _print_summary(fatal_issues, warnings)
        return

    # ---- [REWARD SIGNAL] ----
    out("\n[REWARD SIGNAL]")
    total = len(all_transitions)
    out(f"  Total transitions: {total}")

    if total == 0:
        fatal_issues.append("Zero transitions collected — no learning data")
        out(f"  FATAL: Zero transitions collected")
        _print_summary(fatal_issues, warnings)
        return

    by_type = defaultdict(list)
    for t in all_transitions:
        by_type[t.decision_type].append(t)

    for dtype in ["combat", "draft", "path", "choice"]:
        ts = by_type.get(dtype, [])
        if not ts:
            out(f"  {dtype:>8s}: 0 transitions")
            if dtype in ("draft", "path"):
                warnings.append(f"No {dtype} transitions (might be normal for short games)")
            continue

        rewards = [t.reward for t in ts]
        avg_r = np.mean(rewards)
        zero_pct = sum(1 for r in rewards if abs(r) < 1e-6) / len(rewards) * 100
        out(f"  {dtype:>8s}: {len(ts):4d} transitions | "
            f"avg_reward={avg_r:+.4f} | zero_reward={zero_pct:.0f}%")

        if dtype in ("draft", "path", "choice") and zero_pct > 99:
            fatal_issues.append(
                f"{dtype} decisions have {zero_pct:.0f}% zero reward — no learning signal for strategy layer"
            )

    # ---- [REWARD DISTRIBUTION] ----
    out("\n[REWARD DISTRIBUTION]")
    all_rewards = [t.reward for t in all_transitions]
    out(f"  Mean:   {np.mean(all_rewards):+.4f}")
    out(f"  Std:    {np.std(all_rewards):.4f}")
    out(f"  Min:    {np.min(all_rewards):+.4f}")
    out(f"  Max:    {np.max(all_rewards):+.4f}")
    out(f"  Median: {np.median(all_rewards):+.4f}")

    # 简易直方图
    positive = sum(1 for r in all_rewards if r > 0.01)
    negative = sum(1 for r in all_rewards if r < -0.01)
    zero = sum(1 for r in all_rewards if abs(r) <= 0.01)
    out(f"  Positive: {positive} ({positive/total*100:.1f}%)")
    out(f"  Near-zero: {zero} ({zero/total*100:.1f}%)")
    out(f"  Negative: {negative} ({negative/total*100:.1f}%)")

    # 终端奖励检查
    terminal_ts = [t for t in all_transitions if t.done]
    if terminal_ts:
        out(f"\n  Terminal transitions: {len(terminal_ts)}")
        for t in terminal_ts:
            out(f"    type={t.decision_type}, reward={t.reward:+.2f}, done={t.done}")
    else:
        warnings.append("No terminal transitions found (done=True missing)")
        out(f"  WARNING: No terminal transitions found")

    # 楼层里程碑奖励（已移除，进度通过终端奖励体现）
    out(f"\n  Floor milestone rewards: removed (progress via terminal reward only)")

    # ---- [POLICY BEHAVIOR] ----
    out("\n[POLICY BEHAVIOR]")
    combat_diag = diag.get("combat_decisions", [])
    if combat_diag:
        valid_entries = [d for d in combat_diag if "entropy" in d]
        if valid_entries:
            entropies = [d["entropy"] for d in valid_entries]
            values = [d["value"] for d in valid_entries]
            action_probs = [d["action_prob"] for d in valid_entries]
            end_turns = sum(1 for d in valid_entries if d.get("is_end_turn", False))

            out(f"  Combat decisions with diagnostics: {len(valid_entries)}")
            out(f"  Entropy: mean={np.mean(entropies):.3f}, "
                f"std={np.std(entropies):.3f}, "
                f"min={np.min(entropies):.3f}, max={np.max(entropies):.3f}")

            if np.mean(entropies) < 0.5:
                fatal_issues.append(f"Combat entropy very low ({np.mean(entropies):.3f}) — policy collapsed")
            elif np.mean(entropies) < 1.0:
                warnings.append(f"Combat entropy low ({np.mean(entropies):.3f}) — limited exploration")
            else:
                out(f"    -> Entropy looks healthy (good diversity)")

            out(f"  Value estimates: mean={np.mean(values):.3f}, "
                f"std={np.std(values):.3f}, "
                f"min={np.min(values):.3f}, max={np.max(values):.3f}")

            if np.std(values) < 0.01:
                warnings.append("Value estimates have near-zero variance — value head not learning")

            out(f"  Action probability: mean={np.mean(action_probs):.3f}")

            end_pct = end_turns / len(valid_entries) * 100
            out(f"  End-turn frequency: {end_pct:.1f}% ({end_turns}/{len(valid_entries)})")
            if end_pct > 60:
                fatal_issues.append(f"End-turn frequency {end_pct:.0f}% — agent not playing cards")
            elif end_pct > 45:
                warnings.append(f"End-turn frequency {end_pct:.0f}% — possibly too passive")
            elif end_pct < 5:
                warnings.append(f"End-turn frequency {end_pct:.0f}% — possibly never ending turn")

            # 手牌利用率
            playable_counts = [d.get("n_playable", 0) for d in valid_entries]
            hand_sizes = [d.get("hand_size", 0) for d in valid_entries]
            out(f"  Avg hand size: {np.mean(hand_sizes):.1f}, avg playable: {np.mean(playable_counts):.1f}")

        errors = [d for d in combat_diag if "error" in d]
        if errors:
            out(f"  Combat diagnostic errors: {len(errors)}")
            for e in errors[:3]:
                out(f"    {e['error']}")
    else:
        out(f"  No combat diagnostic data collected")
        warnings.append("No combat diagnostic data — combat_act may not have been called")

    strategy_diag = diag.get("strategy_decisions", [])
    if strategy_diag:
        out(f"\n  Strategy decisions: {len(strategy_diag)}")
        for sd in strategy_diag[:10]:
            out(f"    type={sd['type']}, action={sd['action']}, "
                f"value={sd['value']:.3f}, reward={sd['reward']:+.4f}")
    else:
        out(f"\n  No strategy diagnostic data collected")

    # ---- [HYPERPARAMETERS] ----
    out("\n[HYPERPARAMETERS]")
    import sts_agent_v6 as v6

    params = {
        "GAMMA": v6.GAMMA,
        "GAE_LAMBDA": v6.GAE_LAMBDA,
        "N_RUNS_PER_UPDATE": v6.N_RUNS_PER_UPDATE,
        "ENTROPY_COEFF_START": v6.ENTROPY_COEFF_START,
        "ENTROPY_COEFF_END": v6.ENTROPY_COEFF_END,
        "LR": v6.LR,
        "PPO_EPOCHS": v6.PPO_EPOCHS,
        "MINIBATCH_SIZE": v6.MINIBATCH_SIZE,
        "CLIP_RATIO": v6.CLIP_RATIO,
        "MAX_GRAD_NORM": v6.MAX_GRAD_NORM,
    }
    for name, val in params.items():
        flag = ""
        if name == "GAMMA" and val > 0.995:
            flag = " <- WARNING: very high for sparse rewards"
            warnings.append(f"GAMMA={val} very high — makes credit assignment harder with sparse strategy rewards")
        elif name == "N_RUNS_PER_UPDATE" and val < 8:
            flag = " <- WARNING: very small batch"
            warnings.append(f"N_RUNS_PER_UPDATE={val} — small batch may cause gradient instability")
        elif name == "ENTROPY_COEFF_START" and val < 0.03:
            flag = " <- WARNING: low initial exploration"
            warnings.append(f"ENTROPY_COEFF_START={val} — low initial exploration for complex action space")
        out(f"  {name}: {val}{flag}")

    # 模型参数量
    total_params = sum(p.numel() for p in v6.AgentV6(device="cpu").model.parameters())
    out(f"  Total model parameters: {total_params:,}")

    _print_summary(fatal_issues, warnings)


def _print_summary(fatal_issues, warnings):
    """打印最终汇总"""
    out("\n" + "-" * 60)

    if fatal_issues:
        out(f"\n[FATAL ISSUES] ({len(fatal_issues)})")
        for i, issue in enumerate(fatal_issues, 1):
            out(f"  {i}. {issue}")
    else:
        out(f"\n[FATAL ISSUES] None found")

    if warnings:
        out(f"\n[WARNINGS] ({len(warnings)})")
        for i, w in enumerate(warnings, 1):
            out(f"  {i}. {w}")
    else:
        out(f"\n[WARNINGS] None")

    out("\n" + "=" * 60)
    out(f"Report saved to: {REPORT_PATH}")


# ============================================================
# Main
# ============================================================

def main():
    out(f"V6 Diagnostic Script — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    out(f"Working dir: {BASE_DIR}")
    out(f"Device: {torch.device('mps' if torch.backends.mps.is_available() else 'cpu')}")
    out(f"Using CPU for diagnostics (deterministic)")

    # Part 1: 编码检查
    encoding_issues, encoding_warnings = check_encoding()

    # Part 2: 游戏运行
    game_result, diag, all_transitions = run_instrumented_game()

    # Part 3: 分析报告
    analyze_and_report(game_result, diag, all_transitions, encoding_issues, encoding_warnings)

    tee.close()


if __name__ == "__main__":
    main()

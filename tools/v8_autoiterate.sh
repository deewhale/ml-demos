#!/usr/bin/env bash
# =============================================================================
# v8_autoiterate.sh — 7×24 永不停的自驱训练迭代驱动
# =============================================================================
# 设计目标：无人值守稳跑数天，靠「每个实验有界 + 只留改善 + 删废 ckpt 防爆盘 +
#           错误不致命」保证安全，而不是靠总量封顶。
#
# 核心循环（永不退出，除非收到停止信号文件 /tmp/v8_iter_STOP）：
#   - 从队列文件 /tmp/v8_iter_queue.txt（人/Claude 注入）或内置 CONFIGS 轮转取一个 config
#   - 从 BEST_CKPT 续训 +TRIAL_EPISODES 局（小步快验），套上 config 旋钮
#   - 跑完 parse eval，算复合 score = won*1000 + a2*100 + floor_mean
#   - 改善则采纳（更新 BEST_CKPT/BEST_SCORE，保留该 ckpt），否则丢弃删 ckpt 省盘
#   - 更新 leaderboard 状态文件，检查停止信号，循环
#
# 安全要点：
#   - 每个被丢弃的 trial 立刻删 ckpt；只长期保留 BEST_CKPT + 最近 1 个采纳 ckpt
#   - 单 trial python 非 0 退出 / Traceback / 缺 exit_reason=completed → 标记失败、
#     删 ckpt、继续下一个（绝不让坏 trial 杀死整个 loop）
#   - 每 trial 有界（+TRIAL_EPISODES 局），绝不无限单跑
#   - hang 检测：trial 超 HANG_LIMIT_SEC 无 heartbeat 进展 → 杀该 trial、标记、继续
#   - 不硬编码 /Users；repo root 由脚本位置推导
# =============================================================================

set -u  # 未定义变量报错；注意：不要 set -e（单 trial 失败不能杀 loop）

# ---------------- 路径（不硬编码 /Users）----------------
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT" || { echo "FATAL: cannot cd to repo root $REPO_ROOT"; exit 1; }

PY="$REPO_ROOT/.venv/bin/python"
TRAIN="$REPO_ROOT/tools/v8_ppo_train.py"
ITER_DIR="$REPO_ROOT/sts_models/iter"          # 所有 trial 临时输出
BEST_DIR="$REPO_ROOT/sts_models/iter_best"     # 长期保留的 best ckpt
mkdir -p "$ITER_DIR" "$BEST_DIR"

# ---------------- 信号 / 状态 / 队列文件 ----------------
STOP_FILE="/tmp/v8_iter_STOP"
STATUS_FILE="/tmp/v8_autoiterate_status.md"
QUEUE_FILE="/tmp/v8_iter_queue.txt"
LOG_PREFIX="[autoiterate]"

# ---------------- 调参 ----------------
TRIAL_EPISODES="${TRIAL_EPISODES:-768}"   # 每个 trial 增量训多少局（小步快验）
BATCH_SIZE="${BATCH_SIZE:-32}"
EVAL_FREQ="${EVAL_FREQ:-384}"             # trial 内 eval 频率（+768 → 跑到末尾出新 eval）
CKPT_FREQ="${CKPT_FREQ:-768}"            # trial 内 ckpt 频率
# 采纳判定专用大种子复测（2026-06-08）：训练期中间 eval 仍 48 种子（省时），但**采纳判定**
# 改用对该 trial final ckpt 单独跑一次 ADOPT_EVAL_SEEDS 种子的 deterministic eval。
# 原因：通关是 ~6% 稀有事件，48 种子几乎永远测到 won=0，导致 score 里权重最大的 won×1000
# 项从不触发、搜索其实没在优化通关。96 种子是 48 的超集（同 seed_offset），可比且能测准 won。
ADOPT_EVAL_SEEDS="${ADOPT_EVAL_SEEDS:-96}"
DEVICE=mps
ENGINE_ARGS="--engine lightspeed --combat_tier"
HANG_LIMIT_SEC="${HANG_LIMIT_SEC:-10800}" # 单 trial 超 3h 无 heartbeat 进展 → 杀
HEARTBEAT_POLL_SEC=120                     # hang 检测轮询间隔

# ---------------- 起点：当前最优 ckpt + 基线分 ----------------
# 当前最优 = trial#29 c2_explore++ ckpt（ep8608）。
# 【2026-06-08 评分口径变更：48 种子 roll2 → ADOPT_EVAL_SEEDS(=96) 种子单次复测】
# 旧口径（48 种子 roll2）下 trial#29 score=91.15（a2=0.5417 floor=37.0 won=0）。但 48 种子
# 几乎永远测到 won=0，won×1000 项从不触发，搜索其实没在优化通关。新口径对 final ckpt
# 单独跑 96 种子（同 seed_offset=10000，48 的超集）测准 won。
# trial#29 ckpt 96 种子复测（同 c2 配置：--combat_tier --entropy_coef 0.08, boss_sim=6000）：
#   won=0.0000 a2=0.4271 a1=0.9896 floor=34.95 → score = 0 + 42.71 + 34.95 = 77.66
# （96 种子 a2 比 48 种子低，因多出的 48 个种子更难；且单次非 roll2。这是新口径的真实基线。）
# 注：trial#29 在 6000 boss 搜索预算下 96 局通关 0（能到 act3 floor49 但 act3 boss 团灭）；
#     更高搜索预算的 trial（boss_sim20k 等）才可能出通关种子，届时 won×1000 激活、能被采纳。
BEST_CKPT="${BEST_CKPT:-$REPO_ROOT/sts_models/iter_best/best_t0029_c2_explore++.pt}"
BEST_EP="${BEST_EP:-8608}"
BEST_SCORE="${BEST_SCORE:-77.66}"
BEST_LABEL="baseline_t0029_c2_explore++_96seed"
BEST_DESC="a1=0.99 a2=0.4271 won=0.00 floor=34.95 (96种子复测)"
# 采纳判定的 BEST ckpt 复测时要用的 config 旋钮（与 BEST 当初训练/eval 的战斗预算一致）。
# trial#29 = c2_explore++ = --entropy_coef 0.08（boss_sim 默认 6000）。
BEST_FLAGS="${BEST_FLAGS:---entropy_coef 0.08}"

# ---------------- 内置配置搜索空间（安全 CLI 旋钮组合）----------------
# 格式："label|<额外 CLI 旋钮>"。空旋钮 = baseline。
CONFIGS=(
  "c0_baseline|"
  "c1_explore+|--entropy_coef 0.05"
  "c2_explore++|--entropy_coef 0.08"
  "c3_combat++|--boss_sim_count 10000"
  "c4_lr-|--lr 1e-4"
  "c5_combo|--entropy_coef 0.05 --boss_sim_count 10000"
)
N_CONFIGS=${#CONFIGS[@]}
CONFIG_IDX=0

TRIAL_SEQ=0          # 全局递增序号（output dir 用，不用时间戳）
ROUND=0              # 跑完一整圈 CONFIGS 算一轮
ROUND_IMPROVED=0     # 本轮是否有改善
declare -a RECENT_RESULTS=()   # 最近 N 个 trial 结果摘要

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $LOG_PREFIX $*"; }

# ---------------- 算 score（从 summary.json 读 eval）----------------
# 输出三行：score / detail / exit_reason；解析失败 score=FAIL。
# 滚动均值口径（2026-06-06）：score 取**末 2 次 deterministic eval 的均值**，
# 不再用单次末值 eval_history[-1]——单峰 eval 噪声（±一两 pp / ±一两层）会让模型被
# 噪声峰当门槛卡死（v6 单峰 a2=0.4375=77.56 几乎无法被超过，但其实是抖出来的）。
# 一个 768 局 trial（eval_freq=384）正好出 2 次 eval，取末 2 次均值刚好平滑掉边界局翻转。
# 末 eval 不足 2 次（早停 / 配置异常）时退化为取全部可用 eval 的均值。
parse_summary() {
  local summary="$1"
  "$PY" - "$summary" <<'PYEOF'
import json, sys
p = sys.argv[1]
ROLL_N = 2  # 取末 N 次 eval 均值（与 trainer roll_mean 同口径思路）
try:
    d = json.load(open(p))
    er = d.get("exit_reason", "unknown")
    eh = d.get("eval_history") or []
    if not eh:
        print("FAIL"); print("no eval_history"); print(er); sys.exit(0)
    rec = eh[-ROLL_N:]              # 末 N 次（不足 N 则取全部可用）
    n = len(rec)
    def avg(key):
        return sum(float(e.get(key) or 0.0) for e in rec) / n
    won = avg("won_game_rate")
    a2 = avg("act2_boss_beat_rate")
    a1 = avg("act1_boss_beat_rate")
    fl = avg("floor_mean")
    ent = avg("entropy")
    score = won * 1000.0 + a2 * 100.0 + fl
    print(f"{score:.2f}")
    print(f"a1={a1:.2f} a2={a2:.4f} won={won:.2f} floor={fl:.1f} ent={ent:.3f} roll{n}")
    print(er)
except Exception as ex:
    print("FAIL"); print(f"{type(ex).__name__}: {ex}"); print("parse_error")
PYEOF
}

# ---------------- 采纳判定专用大种子复测 ----------------
# 对给定 ckpt 单独跑 ADOPT_EVAL_SEEDS 种子的 deterministic eval（只评估不训练），
# 把结果写到 <out_dir>/v8_ppo_summary.json（单条 eval_history）。
# 参数：$1=ckpt 路径  $2=输出目录  $3=config 旋钮（与该 ckpt 训练/eval 同战斗预算）
# 返回：0=成功（summary 存在且 exit_reason=eval_only_completed）；非 0=失败。
# 注：用 nice 降优先级，避免与正在跑的 trial 抢 MPS（本驱动串行，复测时无 trial 在跑，
#     但 nice 仍无害）。日志写 <out_dir>.adopt_eval.log。
run_adopt_eval() {
  local ckpt="$1" out_dir="$2" cfg_flags="$3"
  local elog="${out_dir}.adopt_eval.log"
  rm -rf "$out_dir"; mkdir -p "$out_dir"
  log "采纳复测：对 $ckpt 跑 $ADOPT_EVAL_SEEDS 种子 deterministic eval（flags='$cfg_flags'）→ $out_dir"
  # shellcheck disable=SC2086
  nice -n 5 "$PY" "$TRAIN" $ENGINE_ARGS \
      --eval_only --eval_seeds "$ADOPT_EVAL_SEEDS" \
      --resume_from "$ckpt" \
      --output_dir "$out_dir" \
      --device "$DEVICE" \
      $cfg_flags \
      > "$elog" 2>&1
  local rc=$?
  local summary="$out_dir/v8_ppo_summary.json"
  if [ "$rc" -ne 0 ] || [ ! -f "$summary" ]; then
    log "采纳复测失败：rc=$rc summary=$([ -f "$summary" ] && echo y || echo n)（见 $elog）"
    return 1
  fi
  if grep -qE 'Traceback|FATAL' "$elog" 2>/dev/null; then
    log "采纳复测日志含 Traceback/FATAL（见 $elog）→ 视为失败"
    return 1
  fi
  return 0
}

# ---------------- 更新状态文件（leaderboard）----------------
write_status() {
  local disk
  disk="$(du -sh "$ITER_DIR" 2>/dev/null | cut -f1)"
  local best_disk
  best_disk="$(du -sh "$BEST_DIR" 2>/dev/null | cut -f1)"
  {
    echo "# 永续自驱迭代状态（v8_autoiterate.sh）"
    echo
    echo "最后更新：$(date '+%Y-%m-%d %H:%M:%S')｜PID：$$｜repo：$REPO_ROOT"
    echo
    echo "## 当前最优"
    echo
    echo "- ckpt：\`$BEST_CKPT\`"
    echo "- 来源 label：$BEST_LABEL"
    echo "- score：**$BEST_SCORE**（= won×1000 + a2×100 + floor_mean，对 final ckpt 单独跑 ${ADOPT_EVAL_SEEDS} 种子复测口径）"
    echo "- ep：$BEST_EP"
    echo "- eval：$BEST_DESC"
    echo
    echo "## 进度"
    echo
    echo "- 已跑 trial：${TRIAL_SEQ}｜当前轮次：${ROUND}｜本轮已改善：${ROUND_IMPROVED} 次"
    echo "- trial 磁盘占用：${disk:-?}（${ITER_DIR}）｜best：${best_disk:-?}（${BEST_DIR}）"
    echo "- 每 trial 增量：$TRIAL_EPISODES 局｜hang 上限：$((HANG_LIMIT_SEC/3600))h"
    echo
    echo "## 最近 trial 结果（新→旧）"
    echo
    echo "| trial | label | score | 判定 | eval |"
    echo "|---|---|---|---|---|"
    local i
    for ((i=${#RECENT_RESULTS[@]}-1; i>=0; i--)); do
      echo "${RECENT_RESULTS[$i]}"
    done
    echo
    echo "## 控制"
    echo
    echo "- 停止：\`touch $STOP_FILE\`（当前 trial 跑完后优雅退出）"
    echo "- 注入实验：往 \`$QUEUE_FILE\` 追加一行 \`label|<CLI 旋钮>\`，驱动优先消费"
  } > "$STATUS_FILE"
}

# ---------------- 初始化队列文件（带格式说明）----------------
init_queue() {
  if [ ! -f "$QUEUE_FILE" ]; then
    {
      echo "# v8 自驱迭代实验队列（驱动优先消费未注释行）"
      echo "# 格式： label|<额外 CLI 旋钮>"
      echo "# 例：  my_exp|--entropy_coef 0.06 --boss_sim_count 12000"
      echo "# 例（baseline，无旋钮）： plain|"
      echo "# 注：以 # 开头或空行被忽略。每行消费后会被标记 [DONE]。"
    } > "$QUEUE_FILE"
  fi
}

# ---------------- 取下一个实验（队列优先，否则轮转 CONFIGS）----------------
# 设置全局 NEXT_LABEL / NEXT_FLAGS / NEXT_SRC。
pick_next() {
  NEXT_LABEL=""
  NEXT_FLAGS=""
  NEXT_SRC=""
  # 1) 队列：找第一个未 [DONE]、非注释、非空行
  if [ -f "$QUEUE_FILE" ]; then
    local lineno
    lineno="$("$PY" - "$QUEUE_FILE" <<'PYEOF'
import sys
p = sys.argv[1]
for i, raw in enumerate(open(p)):
    s = raw.strip()
    if not s or s.startswith("#") or s.startswith("[DONE]"):
        continue
    print(i)
    break
PYEOF
)"
    if [ -n "$lineno" ]; then
      local raw
      raw="$(sed -n "$((lineno+1))p" "$QUEUE_FILE")"
      NEXT_LABEL="${raw%%|*}"
      NEXT_FLAGS="${raw#*|}"
      NEXT_SRC="queue"
      # 标记该行已消费
      "$PY" - "$QUEUE_FILE" "$lineno" <<'PYEOF'
import sys
p, ln = sys.argv[1], int(sys.argv[2])
lines = open(p).read().splitlines()
if ln < len(lines):
    lines[ln] = "[DONE] " + lines[ln]
open(p, "w").write("\n".join(lines) + "\n")
PYEOF
      return
    fi
  fi
  # 2) 轮转内置 CONFIGS
  local cfg="${CONFIGS[$CONFIG_IDX]}"
  NEXT_LABEL="${cfg%%|*}"
  NEXT_FLAGS="${cfg#*|}"
  NEXT_SRC="builtin"
  CONFIG_IDX=$(( (CONFIG_IDX + 1) % N_CONFIGS ))
  # 轮转回到 0 = 跑完一圈
  if [ "$CONFIG_IDX" -eq 0 ]; then
    ROUND=$((ROUND + 1))
    if [ "$ROUND_IMPROVED" -eq 0 ]; then
      log "round $ROUND 无改善：CLI 旋钮搜索见顶，需代码级策略(奖励/架构)——已标记待 Claude 注入队列（但继续循环，eval 有噪声复跑可能有收获）"
    else
      log "round $ROUND 完成：本轮改善 $ROUND_IMPROVED 次"
    fi
    ROUND_IMPROVED=0
  fi
}

# ---------------- 记录一条 trial 结果到 RECENT_RESULTS（保留最近 12）----------------
push_result() {
  local row="$1"
  RECENT_RESULTS+=("$row")
  local n=${#RECENT_RESULTS[@]}
  if [ "$n" -gt 12 ]; then
    RECENT_RESULTS=("${RECENT_RESULTS[@]:$((n-12))}")
  fi
}

# ---------------- 删一个 trial 的所有 ckpt + 其 .log（省盘）----------------
purge_trial() {
  local dir="$1"
  if [ -n "$dir" ] && [ -d "$dir" ] && [[ "$dir" == "$ITER_DIR"/* ]]; then
    rm -rf "$dir"
    # 同名 .log（与 trial 目录同名 + .log）也删；只删 ITER_DIR 下、防误删
    if [[ "$dir.log" == "$ITER_DIR"/* ]]; then
      rm -f "$dir.log"
    fi
  fi
}

log "启动永续自驱迭代驱动。repo=$REPO_ROOT PID=$$"
log "起点 BEST_CKPT=$BEST_CKPT BEST_EP=$BEST_EP BEST_SCORE=$BEST_SCORE"
init_queue
write_status

# 上一个被采纳但非 baseline 的 ckpt（保留最近 1 个，再老的删）
PREV_ADOPTED_DIR=""

# =============================================================================
# 主循环：永不退出，除非 STOP_FILE 存在
# =============================================================================
while true; do
  if [ -f "$STOP_FILE" ]; then
    log "检测到停止信号 $STOP_FILE → 优雅退出"
    write_status
    break
  fi

  pick_next
  TRIAL_SEQ=$((TRIAL_SEQ + 1))
  local_label="$NEXT_LABEL"
  local_flags="$NEXT_FLAGS"
  TRIAL_DIR="$ITER_DIR/t$(printf '%04d' "$TRIAL_SEQ")_${local_label}"
  TARGET_EP=$((BEST_EP + TRIAL_EPISODES))
  TRIAL_LOG="$ITER_DIR/t$(printf '%04d' "$TRIAL_SEQ")_${local_label}.log"

  rm -rf "$TRIAL_DIR"
  mkdir -p "$TRIAL_DIR"

  # shellcheck disable=SC2086
  log "TRIAL #$TRIAL_SEQ [$NEXT_SRC] label=$local_label flags='$local_flags' resume=$BEST_CKPT ep:${BEST_EP}->${TARGET_EP} out=$TRIAL_DIR"

  # ---- 起训（后台），带 hang 检测 ----
  # shellcheck disable=SC2086
  "$PY" "$TRAIN" $ENGINE_ARGS \
      --num_episodes "$TARGET_EP" \
      --resume_from "$BEST_CKPT" \
      --batch_size "$BATCH_SIZE" \
      --eval_frequency "$EVAL_FREQ" \
      --checkpoint_frequency "$CKPT_FREQ" \
      --output_dir "$TRIAL_DIR" \
      --device "$DEVICE" \
      $local_flags \
      > "$TRIAL_LOG" 2>&1 &
  TRIAL_PID=$!

  # ---- hang 监测：心跳进度若 HANG_LIMIT_SEC 无变化 → 杀 ----
  last_hb=""
  last_hb_change=$(date +%s)
  trial_failed=0
  while kill -0 "$TRIAL_PID" 2>/dev/null; do
    sleep "$HEARTBEAT_POLL_SEC"
    if [ -f "$STOP_FILE" ]; then
      log "停止信号 → 杀当前 trial #$TRIAL_SEQ (PID $TRIAL_PID)"
      kill "$TRIAL_PID" 2>/dev/null; sleep 3; kill -9 "$TRIAL_PID" 2>/dev/null
      break
    fi
    cur_hb="$(grep -E '\[heartbeat\]|\[eval@ep=' "$TRIAL_LOG" 2>/dev/null | tail -1)"
    now=$(date +%s)
    if [ "$cur_hb" != "$last_hb" ]; then
      last_hb="$cur_hb"
      last_hb_change=$now
    elif [ $((now - last_hb_change)) -ge "$HANG_LIMIT_SEC" ]; then
      log "HANG 检测：trial #$TRIAL_SEQ 超 $((HANG_LIMIT_SEC/3600))h 无心跳进展 → 杀 (PID $TRIAL_PID)"
      kill "$TRIAL_PID" 2>/dev/null; sleep 3; kill -9 "$TRIAL_PID" 2>/dev/null
      trial_failed=1
      break
    fi
  done
  wait "$TRIAL_PID" 2>/dev/null
  TRIAL_RC=$?

  # 停止信号在 trial 中途触发：清理后退出
  if [ -f "$STOP_FILE" ]; then
    log "停止信号 → trial #$TRIAL_SEQ 中断，清理临时 ckpt 后退出"
    purge_trial "$TRIAL_DIR"
    write_status
    break
  fi

  SUMMARY="$TRIAL_DIR/v8_ppo_summary.json"

  # ---- 错误不致命：判定 trial 是否健康 ----
  # 用 grep -q 判存在（grep -c 在无匹配时退出 1 会污染计数；bash 3.2 无 mapfile，全程避坑）。
  if grep -qE 'Traceback|FATAL' "$TRIAL_LOG" 2>/dev/null; then trace_hits=1; else trace_hits=0; fi
  if [ "$trial_failed" -eq 1 ] || [ "$TRIAL_RC" -ne 0 ] || [ ! -f "$SUMMARY" ] || [ "$trace_hits" -gt 0 ]; then
    reason="rc=$TRIAL_RC failed=$trial_failed summary=$([ -f "$SUMMARY" ] && echo y || echo n) trace=$trace_hits"
    log "trial #$TRIAL_SEQ FAILED ($reason) → 删 ckpt 继续下一个"
    push_result "| #$TRIAL_SEQ | $local_label | - | ❌FAIL | $reason |"
    purge_trial "$TRIAL_DIR"
    write_status
    continue
  fi

  # ---- 训练健康门：训练 summary 的 exit_reason 必须 completed（确认训练正常跑完，
  #      不是中途崩/早停）。这里只做 gate，不用训练期 48 种子 eval 算分。----
  TRAIN_EXIT="$("$PY" - "$SUMMARY" <<'PYEOF'
import json, sys
try:
    print(json.load(open(sys.argv[1])).get("exit_reason", "unknown"))
except Exception:
    print("parse_error")
PYEOF
)"
  if [ "$TRAIN_EXIT" != "completed" ]; then
    log "trial #$TRIAL_SEQ 训练未正常完成（exit_reason=$TRAIN_EXIT）-> 删 ckpt 继续"
    push_result "| #$TRIAL_SEQ | $local_label | - | ❌FAIL | train_exit=$TRAIN_EXIT |"
    purge_trial "$TRIAL_DIR"
    write_status
    continue
  fi

  # ---- 采纳判定大种子复测：对 final ckpt 单独跑 ADOPT_EVAL_SEEDS 种子 deterministic eval ----
  # （训练期中间 eval 仍 48 种子省时；采纳判定改用 96 种子测准通关 won，见脚本顶部说明）。
  ADOPT_EVAL_DIR="$TRIAL_DIR/adopt_eval"
  if ! run_adopt_eval "$TRIAL_DIR/v8_ppo_final.pt" "$ADOPT_EVAL_DIR" "$local_flags"; then
    log "trial #$TRIAL_SEQ 采纳复测失败 → 删 ckpt 继续下一个"
    push_result "| #$TRIAL_SEQ | $local_label | - | ❌FAIL | adopt_eval_failed |"
    purge_trial "$TRIAL_DIR"
    write_status
    continue
  fi

  # ---- 解析 96 种子复测 eval + 算 score（bash 3.2 无 mapfile，用逐行 read）----
  ADOPT_SUMMARY="$ADOPT_EVAL_DIR/v8_ppo_summary.json"
  SCORE="FAIL"; DETAIL="?"; EXIT_REASON="?"
  _i=0
  while IFS= read -r _line; do
    case "$_i" in
      0) SCORE="$_line" ;;
      1) DETAIL="$_line" ;;
      2) EXIT_REASON="$_line" ;;
    esac
    _i=$((_i + 1))
  done < <(parse_summary "$ADOPT_SUMMARY")

  if [ "$SCORE" = "FAIL" ] || [ "$EXIT_REASON" != "eval_only_completed" ]; then
    log "trial #$TRIAL_SEQ 96种子复测解析失败（score=$SCORE exit_reason=$EXIT_REASON ${DETAIL}）-> 删 ckpt 继续"
    push_result "| #$TRIAL_SEQ | $local_label | $SCORE | ❌FAIL | adopt_exit=$EXIT_REASON $DETAIL |"
    purge_trial "$TRIAL_DIR"
    write_status
    continue
  fi

  # ---- 采纳判据（带噪声容忍）：score 明显超过 BEST 才采纳 ----
  # 复用 Python 做浮点比较，含 won/a2/floor 的"明显改善"判定。
  ADOPT="$("$PY" - "$SCORE" "$BEST_SCORE" <<'PYEOF'
import sys
s = float(sys.argv[1]); b = float(sys.argv[2])
# score = won*1000 + a2*100 + floor。改善判据（任一）：
#   - score 提升 >= 1.0（约等于 floor +1 或 a2 +1pp 的量级，过滤纯噪声）
print("YES" if (s - b) >= 1.0 else "NO")
PYEOF
)"

  if [ "$ADOPT" = "YES" ]; then
    NEW_FINAL="$TRIAL_DIR/v8_ppo_final.pt"
    # 移到 best 目录长期保留
    BEST_KEEP="$BEST_DIR/best_t$(printf '%04d' "$TRIAL_SEQ")_${local_label}.pt"
    cp -f "$NEW_FINAL" "$BEST_KEEP" 2>/dev/null
    # 删上上个采纳的 trial（只保留最近 1 个采纳 + baseline）
    if [ -n "$PREV_ADOPTED_DIR" ]; then
      purge_trial "$PREV_ADOPTED_DIR"
    fi
    PREV_ADOPTED_DIR="$TRIAL_DIR"
    OLD_SCORE="$BEST_SCORE"
    BEST_CKPT="$BEST_KEEP"
    BEST_SCORE="$SCORE"
    BEST_EP="$TARGET_EP"
    BEST_LABEL="$local_label (trial #$TRIAL_SEQ)"
    BEST_DESC="$DETAIL"
    ROUND_IMPROVED=$((ROUND_IMPROVED + 1))
    log "IMPROVED by $local_label: score $OLD_SCORE → $SCORE ($DETAIL) 采纳，新 BEST_CKPT=$BEST_CKPT BEST_EP=$BEST_EP"
    push_result "| #$TRIAL_SEQ | $local_label | $SCORE | ✅采纳 | $DETAIL |"
  else
    log "no gain $local_label: score=$SCORE (BEST=$BEST_SCORE) ($DETAIL) → 删 ckpt 省盘"
    push_result "| #$TRIAL_SEQ | $local_label | $SCORE | 丢弃 | $DETAIL |"
    purge_trial "$TRIAL_DIR"
  fi

  write_status
done

log "驱动退出。最终 BEST_CKPT=$BEST_CKPT BEST_SCORE=$BEST_SCORE"

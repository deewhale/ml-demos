#!/usr/bin/env bash
#
# v8_autoloop.sh — 无人值守自动训练循环驱动
#
# 作用：等当前批 v5 训练跑完后，自动续训 V8 PPO（sts_lightspeed + 分级搜索），
# 按简单决策树继续/停止，防空跑。全自动、有上限、状态可查。
#
# 决策树（每批跑完后）：
#   - 二幕率(act2)相对历史最好提升 ≥ 2pp，或 通关数(won_game) 增加 → 改善，继续下一批
#   - 连续 2 批「二幕率提升 < 2pp 且 通关没增」 → PLATEAU，停
#   - 出异常（log 有 Traceback/Error、python 非 0 退出、心跳 hang） → ERROR/HANG，停
#   - 训到 3 批上限（cum 9216） → CAP，停
#
# 用法：nohup bash tools/v8_autoloop.sh > /tmp/v8_autoloop.log 2>&1 &
#
set -u

# ---- repo root（相对推导，不硬编码绝对路径）----
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT" || { echo "无法 cd 到 repo root"; exit 1; }

PY=".venv/bin/python"
STATUS="/tmp/v8_autoloop_status.md"

# ---- 起始状态 ----
V5_PID=77644
PREV_FINAL="sts_models/v8_ppo_lightspeed_v5/v8_ppo_final.pt"
CUM=6144              # v5 累计目标
BATCH_IDX=6           # 下一批从 v6 起
CUM_CAP=9216          # 硬上限：v5(6144) + 3 批 × 1024
MAX_BATCHES=3
INCREMENT=1024

# 历史最好（用于改善判定）
BEST_ACT2=-1.0
BEST_WON=-1.0
NO_IMPROVE_STREAK=0   # 连续无改善批数（plateau 计数）

# ---- 工具函数 ----
ts() { date "+%Y-%m-%d %H:%M:%S"; }

status_header() {
  {
    echo "# 自动训练循环状态"
    echo ""
    echo "最后更新：$(ts)"
    echo ""
    echo "驱动 PID：$$ ｜ repo：$REPO_ROOT"
    echo ""
    echo "| 批次 | ep(累计目标) | 一幕(act1) | 二幕(act2) | 通关(won) | floor | 熵 | 决策 |"
    echo "|---|---|---|---|---|---|---|---|"
  } > "$STATUS"
}

status_row() {
  # 参数：batch ep act1 act2 won floor entropy decision
  echo "| $1 | $2 | $3 | $4 | $5 | $6 | $7 | $8 |" >> "$STATUS"
}

status_note() {
  echo "" >> "$STATUS"
  echo "$1" >> "$STATUS"
  # 刷新顶部时间戳
  sed -i '' "s/^最后更新：.*/最后更新：$(ts)/" "$STATUS" 2>/dev/null || true
}

# 解析一个 log 文件末段的 eval 行，回显 "act1 act2 won floor entropy ep"
# 取最后一条 [eval@ep=...] 实测行（非 roll3_mean）。
parse_eval() {
  local logf="$1"
  local line
  line="$(grep -E "\[eval@ep=[0-9]+\] reached_a1_boss=" "$logf" 2>/dev/null | tail -1)"
  if [ -z "$line" ]; then
    echo "NA NA NA NA NA NA"
    return
  fi
  local ep act1 act2 won floor ent
  ep="$(echo "$line"   | sed -nE 's/.*\[eval@ep=([0-9]+)\].*/\1/p')"
  act1="$(echo "$line" | sed -nE 's/.*a1_boss_beat=([0-9.]+).*/\1/p')"
  act2="$(echo "$line" | sed -nE 's/.*a2_boss_beat=([0-9.]+).*/\1/p')"
  won="$(echo "$line"  | sed -nE 's/.*won_game=([0-9.]+).*/\1/p')"
  floor="$(echo "$line"| sed -nE 's/.*floor_mean=([0-9.]+).*/\1/p')"
  ent="$(echo "$line"  | sed -nE 's/.*entropy=([0-9.]+).*/\1/p')"
  echo "${act1:-NA} ${act2:-NA} ${won:-NA} ${floor:-NA} ${ent:-NA} ${ep:-NA}"
}

# 浮点比较：返回 0(真) 若 $1 >= $2
fge() { awk -v a="$1" -v b="$2" 'BEGIN{exit !(a>=b)}'; }
# 返回 0(真) 若 ($1 - $2) >= $3   （改善幅度判定）
fdelta_ge() { awk -v a="$1" -v b="$2" -v d="$3" 'BEGIN{exit !((a-b)>=d)}'; }

# ---- 初始化状态文件 ----
status_header
status_note "等 v5（PID ${V5_PID}）跑完后开始自动续训。增量 ${INCREMENT}/批，最多 ${MAX_BATCHES} 批，累计上限 ${CUM_CAP}。"

echo "[$(ts)] autoloop 启动，驱动 PID=$$。等 v5（PID ${V5_PID}）跑完..."

# ---- 等 v5 跑完 ----
while kill -0 "$V5_PID" 2>/dev/null; do
  sleep 60
done
echo "[$(ts)] v5（PID ${V5_PID}）已结束，开始自动续训循环。"

# 记录 v5 最终 eval 作为基线
V5_LOG="/tmp/v8_ppo_lightspeed_v5.log"
read -r v5_act1 v5_act2 v5_won v5_floor v5_ent v5_ep <<< "$(parse_eval "$V5_LOG")"
status_row "v5(基线)" "${v5_ep}" "${v5_act1}" "${v5_act2}" "${v5_won}" "${v5_floor}" "${v5_ent}" "基线"
# 用 v5 初始化历史最好
[ "$v5_act2" != "NA" ] && BEST_ACT2="$v5_act2"
[ "$v5_won"  != "NA" ] && BEST_WON="$v5_won"
status_note "v5 基线：一幕=${v5_act1} 二幕=${v5_act2} 通关=${v5_won} floor=${v5_floor} 熵=${v5_ent}。开始续训。"

STOP_REASON=""
BATCHES_RUN=0

while [ "$BATCHES_RUN" -lt "$MAX_BATCHES" ]; do
  NEW_CUM=$(( CUM + INCREMENT ))
  if [ "$NEW_CUM" -gt "$CUM_CAP" ]; then
    STOP_REASON="CAP: 累计目标 ${NEW_CUM} 超过上限 ${CUM_CAP}，停。"
    break
  fi

  OUT="sts_models/v8_ppo_lightspeed_v${BATCH_IDX}"
  LOG="/tmp/v8_ppo_lightspeed_v${BATCH_IDX}.log"

  echo "[$(ts)] 起训 v${BATCH_IDX}：cum ${CUM}->${NEW_CUM}，resume_from=${PREV_FINAL}，out=${OUT}"
  status_note "[$(ts)] 起训 v${BATCH_IDX}（累计目标 ${NEW_CUM}），从 ${PREV_FINAL} 续训..."

  # 前序检查：resume_from 必须存在
  if [ ! -f "$PREV_FINAL" ]; then
    STOP_REASON="ERROR: v${BATCH_IDX} 的续训源 ${PREV_FINAL} 不存在，已停。"
    break
  fi

  # 起训（前台等这批退出——本脚本在 nohup 里）
  "$PY" tools/v8_ppo_train.py \
    --engine lightspeed \
    --combat_tier \
    --num_episodes "$NEW_CUM" \
    --resume_from "$PREV_FINAL" \
    --batch_size 32 \
    --eval_frequency 64 \
    --checkpoint_frequency 32 \
    --output_dir "$OUT" \
    --device mps \
    > "$LOG" 2>&1
  EXIT_CODE=$?

  BATCHES_RUN=$(( BATCHES_RUN + 1 ))

  # ---- 异常检测 ----
  if [ "$EXIT_CODE" -ne 0 ]; then
    read -r act1 act2 won floor ent ep <<< "$(parse_eval "$LOG")"
    status_row "v${BATCH_IDX}" "${ep}" "${act1}" "${act2}" "${won}" "${floor}" "${ent}" "退出码${EXIT_CODE}"
    STOP_REASON="ERROR/HANG: v${BATCH_IDX} python 退出码=${EXIT_CODE}（异常退出），已停。查 ${LOG}。"
    break
  fi
  if grep -qE "Traceback \(most recent call last\)" "$LOG" 2>/dev/null; then
    read -r act1 act2 won floor ent ep <<< "$(parse_eval "$LOG")"
    status_row "v${BATCH_IDX}" "${ep}" "${act1}" "${act2}" "${won}" "${floor}" "${ent}" "Traceback"
    STOP_REASON="ERROR/HANG: v${BATCH_IDX} log 出现 Traceback，已停。查 ${LOG}。"
    break
  fi
  # 完成标志校验：正常结束应有 reason=completed
  if ! grep -qE "训练结束 reason=completed" "$LOG" 2>/dev/null; then
    read -r act1 act2 won floor ent ep <<< "$(parse_eval "$LOG")"
    status_row "v${BATCH_IDX}" "${ep}" "${act1}" "${act2}" "${won}" "${floor}" "${ent}" "未正常完成"
    STOP_REASON="ERROR/HANG: v${BATCH_IDX} 未见正常完成标志（reason=completed），疑似 hang/中断，已停。查 ${LOG}。"
    break
  fi

  # ---- 解析这批 eval ----
  read -r act1 act2 won floor ent ep <<< "$(parse_eval "$LOG")"
  if [ "$act2" = "NA" ]; then
    status_row "v${BATCH_IDX}" "${ep}" "${act1}" "${act2}" "${won}" "${floor}" "${ent}" "无eval"
    STOP_REASON="ERROR: v${BATCH_IDX} 跑完但解析不到 eval 行，已停。查 ${LOG}。"
    break
  fi

  # ---- 决策树：改善 vs plateau ----
  # 改善：二幕相对历史最好提升 >= 2pp（0.02），或 通关数 > 历史最好
  improved=1   # 0=改善 1=未改善
  if fdelta_ge "$act2" "$BEST_ACT2" "0.02"; then improved=0; fi
  if awk -v a="$won" -v b="$BEST_WON" 'BEGIN{exit !(a>b)}'; then improved=0; fi

  if [ "$improved" -eq 0 ]; then
    NO_IMPROVE_STREAK=0
    DECISION="改善→续"
    # 更新历史最好
    fge "$act2" "$BEST_ACT2" && BEST_ACT2="$act2"
    fge "$won"  "$BEST_WON"  && BEST_WON="$won"
  else
    NO_IMPROVE_STREAK=$(( NO_IMPROVE_STREAK + 1 ))
    DECISION="无改善(${NO_IMPROVE_STREAK}/2)"
  fi

  status_row "v${BATCH_IDX}" "${ep}" "${act1}" "${act2}" "${won}" "${floor}" "${ent}" "$DECISION"
  status_note "[$(ts)] v${BATCH_IDX} 跑完：一幕=${act1} 二幕=${act2} 通关=${won} floor=${floor} 熵=${ent}（历史最好 二幕=${BEST_ACT2} 通关=${BEST_WON}）→ ${DECISION}"

  # plateau：连续 2 批无改善
  if [ "$NO_IMPROVE_STREAK" -ge 2 ]; then
    STOP_REASON="PLATEAU: 战斗已分级增强、量到顶，剩余墙是元决策(卡组/奖励/探索)，需策略改动复盘。连续 2 批二幕率提升<2pp 且通关未增。"
    break
  fi

  # ---- 推进到下一批 ----
  CUM="$NEW_CUM"
  PREV_FINAL="${OUT}/v8_ppo_final.pt"
  BATCH_IDX=$(( BATCH_IDX + 1 ))
done

# ---- 上限兜底 ----
if [ -z "$STOP_REASON" ]; then
  if [ "$BATCHES_RUN" -ge "$MAX_BATCHES" ]; then
    STOP_REASON="CAP: 已训到 ${MAX_BATCHES} 批（累计目标 ${CUM}），到上限，待人工复盘。"
  else
    STOP_REASON="STOP: 循环结束（原因未明确归类）。"
  fi
fi

echo "[$(ts)] 循环结束：$STOP_REASON"
status_note "---"
status_note "**停止原因**：$STOP_REASON"
status_note "**最终汇总**：共跑 ${BATCHES_RUN} 批，历史最好 二幕=${BEST_ACT2} 通关=${BEST_WON}。结束时间 $(ts)。"

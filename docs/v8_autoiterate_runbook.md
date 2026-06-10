# V8 训练 + 7×24 自驱迭代接管手册（会话 clear 后靠这份恢复）

> 路径全部用 `<repo>`（= 本仓库根）/ 仓库相对 / `/tmp`，**无家目录硬编码**。
> last-verified: 2026-06-08。

## 一句话

当前在 lightspeed 引擎上跑 V8 PPO + **7×24 永续自驱迭代**（`tools/v8_autoiterate.sh`）。
最优 agent `sts_models/iter_best/best_t0029_c2_explore++.pt`（ep8608）96 种子严格口径：
**一幕 ~99% / 二幕 ~43% / 通关 0 / floor_mean ~35**。卡在三幕 boss（到 floor49 团灭）。
CLI 旋钮搜索已 plateau，下一步转持续长训。

## 怎么查状态

```bash
cat /tmp/v8_autoiterate_status.md      # 排行榜 + 当前最优 + 进度 + 控制说明
tail -40 /tmp/v8_autoiterate.out       # 驱动滚动日志
ps -ef | grep v8_autoiterate | grep -v grep   # 驱动还活着吗（看 bash tools/v8_autoiterate.sh 的 PID）
ps -ef | grep v8_ppo_train | grep -v grep      # 当前 trial 的训练子进程
```

## 怎么停 / 重启

- **停**（优雅，当前 trial 跑完后退出）：`touch /tmp/v8_iter_STOP`
- **重启驱动**（挂了之后，在 `<repo>` 下）：
  ```bash
  cd <repo>
  nohup bash tools/v8_autoiterate.sh > /tmp/v8_autoiterate.out 2>&1 &
  ```
  重启前确认 `/tmp/v8_iter_STOP` 不存在（在就 `rm` 掉），否则会立刻退出。

## 驱动机制（tools/v8_autoiterate.sh）

永续循环，每轮：
1. **配置搜索**：从当前最优 ckpt（`BEST_CKPT`）续训 `TRIAL_EPISODES`（默认 768）局，
   套一组 CLI 旋钮（候选见脚本 `EXPERIMENTS` 数组：entropy_coef / boss_sim_count 等）。
   引擎固定 `--engine lightspeed --combat_tier`。
2. **队列优先**：若 `/tmp/v8_iter_queue.txt` 有未 `[DONE]` 行，**优先消费队列**。
   注入格式：往该文件追加一行 `label|<额外 CLI 旋钮>`，例：
   `my_exp|--entropy_coef 0.06 --boss_sim_count 12000`（`#` 开头 / 空行忽略；
   消费后标 `[DONE]`）。
3. **keep-best**：每个 trial 跑完按 score 排序，**只有改善才采纳**（更新 `BEST_CKPT`）。
4. **96 种子采纳复测**：采纳前对 final ckpt **单独跑 96 种子**复测，得诚实 score
   （`score = won×1000 + a2×100 + floor_mean`）。这是为什么 48 种子的「通关 6%」噪声
   不会污染采纳——96 种子口径 won=0。
5. **删废盘**：没采纳的 trial ckpt 删掉，防磁盘爆（`sts_models/iter/`）。
6. 写状态到 `/tmp/v8_autoiterate_status.md`，回到 1。
- hang 上限 3h（单 trial 超时杀），错误不致命（trial 失败跳过继续）。

## 当前最优 ckpt + 真实指标 + 起点对比

| 项 | 值 |
|---|---|
| ckpt | `sts_models/iter_best/best_t0029_c2_explore++.pt` |
| ep | 8608 |
| 配置 | `--combat_tier --entropy_coef 0.08`（boss_sim 默认 6000） |
| 96 种子口径 | a1=0.99 / a2=0.4271 / won=0.00 / floor_mean=34.95 |
| score | 77.66 |

> 注意诚实口径：**96 种子**采纳复测 won=0；旧 **48 种子** eval 报过「二幕 54% / 通关 6%」
> 是小样本噪声，**不要再引用 48 种子数**。归因复核里一组跑出真通关 6/96=6.25%，仍属稀有
> 事件，采纳口径记 won=0。

## ✅ 长训已启动（2026-06-08 17:21）——自驱迭代已停

**当前活跃：持续长训 v8_longtrain_v1**（不再跑自驱迭代驱动；驱动已优雅停掉退出）。

| 项 | 值 |
|---|---|
| 状态 | 运行中（PID 93465，nohup） |
| log | `/tmp/v8_longtrain_v1.log` |
| output_dir | `sts_models/v8_longtrain_v1/` |
| 起点 ckpt | `sts_models/iter_best/best_t0029_c2_explore++.pt`（ep8608） |
| 目标局数 | `--num_episodes 24000`（累计目标，增量训 15392 ep） |
| 配置 | `--engine lightspeed --combat_tier --entropy_coef 0.08`（= 最优 c2_explore++） |
| 其它旋钮 | `--batch_size 32 --eval_frequency 384 --checkpoint_frequency 512 --device mps` |

**长训巡检要点**：
- 活吗：`ps -ef | grep v8_ppo_train | grep -v grep`（PID 93465）
- 进度/心跳：`grep "\[heartbeat\]" /tmp/v8_longtrain_v1.log | tail -1`
- 周期 eval（每 384 局）：`grep -E "a1=|a2=|won=" /tmp/v8_longtrain_v1.log | tail -5`
- 异常：`grep -cE "Traceback|guard_cap|Error" /tmp/v8_longtrain_v1.log`
- ckpt 落盘：`ls sts_models/v8_longtrain_v1/`
- **早停红线**：eval 连续 ≥3-4 个周期不涨（a2 / won 无趋势）→ 早停，别白跑到 24000。
- **hang**：心跳间隔超 5min 怀疑 boss 深搜索；30min+ 无心跳立刻杀。
- 出成果（won>0 或 a2 显著涨）→ 把成果 ckpt 设为新 best，按需重启自驱迭代驱动。

**重启自驱迭代驱动**（若决定停长训回到搜索）：先 `rm -f /tmp/v8_iter_STOP`，
再 `cd <repo> && nohup bash tools/v8_autoiterate.sh > /tmp/v8_autoiterate.out 2>&1 &`。

### 转长训的背景（2026-06-08）

CLI 旋钮搜索（探索熵 0.05/0.08/0.10/0.12、boss 搜索 10k/20k、late_boss_bonus 牵引）
已 ≥2 轮全 no-gain，won 恒 0、a2 在 0.27~0.54 噪声带内无趋势上涨——**探索/搜索维度榨干**。
最后一档探索 hp_explore_max（entropy 0.12）17:07 跑完 score=64.18 < BEST 77.66 无采纳、won=0，
plateau 彻底坐实，遂转长训（自驱迭代驱动 17:19 优雅退出，BEST 保留）。

**下一步：对最优基座长训**，给 agent 时间学三幕 boss。命令模板（`<repo>` 下，把
`<best>` 换成当前 `iter_best/` 里的最优文件名）：

```bash
cd <repo>
.venv/bin/python tools/v8_ppo_train.py \
  --engine lightspeed --combat_tier --entropy_coef 0.08 \
  --num_episodes 24000 \
  --resume_from sts_models/iter_best/<best>.pt \
  --eval_frequency 384 --checkpoint_frequency 512 \
  --output_dir sts_models/v8_ppo_lightspeed_long1 --device mps
```

防空跑红线：
- 长训**只对最优基座**（别对随机/旧 ckpt 空烧）。
- 必须有 checkpoint（`--checkpoint_frequency`），随时可续 / 回滚。
- **eval 连续多次（≥3-4 个 eval 周期）不涨 → 早停**，别让它白跑到 24000。
- 长训期间自驱迭代驱动可先 `touch /tmp/v8_iter_STOP` 停掉，避免抢 MPS / 互相覆盖
  `BEST_CKPT` 记账；长训出成果后再把成果 ckpt 设为新 best 重启驱动。
- 注意：自驱迭代脚本用 `TRIAL_EPISODES` 环境变量控制单 trial 局数，若要在驱动里加长
  单 trial 而非另起长训，**重启驱动时设 `TRIAL_EPISODES=1536`**（别用 queue 的 flag 去
  覆盖 `--num_episodes`，会导致驱动 `BEST_EP` 记账错位）。

## 巡检 prompt 原文（clear 后照此 ScheduleWakeup 续巡检）

> ScheduleWakeup 时 `delaySeconds≈2700`，把下面整段原样作为 prompt；其中 `cd <repo>`
> 在真正 ScheduleWakeup 时换成真实仓库绝对路径。

```
[7×24 自驱迭代巡检] 读 /tmp/v8_autoiterate_status.md 和 /tmp/v8_autoiterate.out（驱动 PID 见 ps）。派 sub-agent：(1) 驱动(tools/v8_autoiterate.sh)活吗？挂了重启(nohup bash tools/v8_autoiterate.sh > /tmp/v8_autoiterate.out 2>&1 &)。STOP 在就停。(2) 看新实验96种子口径 won/a2/floor vs 最优，有真通关(won>0)或采纳没。(3) CLI搜索已plateau→若仍无改善则转持续长训(见 runbook 命令模板)给agent时间学三幕boss；reward级塑形按反作弊红线先report审别直接上。(4) 磁盘没爆/熵没塌/无hang/删废生效。(5) 简短报告(有意义变化才详细)。然后重新 ScheduleWakeup(delaySeconds≈2700,原样传这段prompt)，除非 STOP 或用户接管。
```

## 待审 / 待办

- **reward 级塑形**（血量管理 / 强牌组牵引）：有反作弊风险（可能教会苟活 / 逃战），
  **暂不仓促上**，要上需先设计 + 人工审，别直接进 reward.py。
- **t0064 孤儿 ckpt 删除**（~76M）：destructive，需用户授权后再删。
- **lightspeed 绑定 patch**：见下「重要隐患」。

## 重要隐患：lightspeed 绑定没进本仓库 git

sts_lightspeed 的 pybind11 绑定（combat probe / make_test_combat / get_state /
step_choice / play_battle / make_encounter / 遗物 / 意图字段等）都在 **gitignored** 的
`external/sts_lightspeed/bindings/slaythespire.cpp`，**没进本仓库 git，重 clone 会丢**。

已导出为补丁存档：`external/sts_lightspeed_patches/`
- `0001-dark-shackles-sign-fix.patch`（卡 bug 修）
- `0002-trip-cost-fix.patch`（卡 bug 修）
- `0003-pybind-bindings-combat-fullgame-relic-monster.patch`（**全套绑定** + 上述 2 个 bug 修也含在内的工作树改动）

**重建方法**（重 clone 后）：
```bash
cd <repo>/external
# 重 clone gamerpuppy/sts_lightspeed 到 sts_lightspeed/
cd sts_lightspeed
for p in ../sts_lightspeed_patches/*.patch; do git apply "$p" || git apply --3way "$p"; done
# 初始化 pybind11 子模块（绑定依赖它）
git submodule update --init --recursive
# 重编（用本仓库 venv 的 cmake）
<repo>/.venv/bin/cmake -S . -B build && <repo>/.venv/bin/cmake --build build --target slaythespire -j 10
# 验收（应 84/84 全过）
<repo>/.venv/bin/python tools/test_card_behavior.py
```
> 详细编译/验收口径见 `docs/v8_card_test_set_scheme.md` 和 CLAUDE.md「引擎验收测试台」段。

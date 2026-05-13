# CLAUDE.md

## 项目概述

ML 学习进阶项目：监督学习 → DQN → PPO+Transformer。最终目标：零先验知识的 Slay the Spire AI agent。

## 项目结构

### 活跃代码（V8 RL，训练中）
- `v8/env.py` — V8Env gym wrapper（封装 StSRLSolver runner）。负责元决策阶段
  （MAP / REWARD / CARD_REWARD / EVENT / ...）的状态/动作编码，并通过 search-net
  wrapper 驱动战斗。约 1100 行，含大量诊断日志（见下方「诊断日志」）。
- `v8/trainer.py` — V8PPOTrainer（PPO + clip + GAE + entropy bonus）。
  通过 env 收集 rollout，按 `eval_frequency` 周期跑确定性 eval。
- `v8/model.py` — V8Model（set-encoder + pointer-net actor + value head）。
- `v8/combat_net_wrapper.py` — 把 V8Model 桥接到 StSRLSolver 的 combat_net hook，
  供战斗内搜索调用；BC combat head 权重从 `sts_models/v8_combat_head_v1.pt` 加载（Phase A 产物）。
- `v8/deck_evaluator.py` — 战后牌组评分，9 worker ProcessPoolExecutor 并行模拟。
- `v8/reward.py` — step reward shaping（game_won / floor_progress / 战斗结果 / 牌组质量 delta）。
- `tools/v8_ppo_train.py` — 训练入口。常用参数：`--num_episodes` `--batch_size`
  `--checkpoint_frequency` `--eval_frequency` `--output_dir` `--smoke`。
- `data/sts_data.py` — STS 数据提取（V6 遗留，可能复用）。
- 归档但保留参考：`v8_bot.py`（战斗内 search infra，元决策 dispatch 已被 V8 RL 取代）、
  `v8_data_collector.py`（JSONL schema 参考，import 已 archive 模块所以无法直接运行）。

### 诊断日志（V8 RL 训练 / eval 输出，可 grep）

- `[startup]` / `[heartbeat]` — episode 级：ep#, steps, secs, reward, floor, beat_boss,
  search_calls, eval_deck_calls
- `[combat] enter|exit` — 每场战斗：ep, floor, act, room, enemies, hp before/after, turn_actions
- `[floor]` — 每次楼层变化：ep, floor, act, hp, room 类型（monster/elite/boss/rest/shop/event/treasure）
- `[deck]` — elite/boss 胜利 + reward 处理后：完整牌组（Counter，带升级标记如 Flex+1）+ 遗物
- `[perf]` — 每次 PPO update：batch_total, env_step, collect_fwd, eval_deck (calls/time),
  ppo_update (time/fwd), deck cache hit/miss
- `[guard_cap]` — env 内部 phase-transition loop 命中上限（20000 internal step）时触发，
  FORCE_TERMINATE 并附诊断 context（phase / floor / act / enemies 等）。
- `[event] enter` — 首次进入 EVENT phase 时打一次：ep, floor, act, event_id, phase, 可见 choices。
- `[event] choice` — env.step 处理 EventAction 时打一次：ep, floor, choice_idx, choice_text,
  事件前后 hp / gold / deck_size 等。
- `[event] phase_transition` — 同一事件内 phase 字符串变化（多 phase 事件）时打一次：
  ep, floor, event_id, old phase → new phase, 新 choices。
- `[event] exit` — 离开 EVENT phase 时打一次：ep, floor, event_id, reason（事件正常结束 /
  进入战斗 / 等），出门时的 hp / gold / deck_size。

### 文档
- `docs/v6_training_log.md` — V6 训练日志
- `docs/v6_architecture_review.md` — V3→V6 架构演进回顾
- `docs/archive/` — 旧版本设计文档和分析（V3-V5）

### 归档（不追踪）
- `archive/v3/` — V3 DQN agent
- `archive/v4/` — V4 PPO + Transformer
- `archive/v5/` — V5 TurnSolver + 策略层
- `demos/` — 教学 demo（MNIST、CartPole、简化 STS）
- `tools/` — 工具脚本（comm_bridge、test_solver 等）
- `sts_models/` — 模型权重

## 技术栈

- Python 3.12/3.13, PyTorch, Gymnasium
- Apple Silicon MPS 训练（multinomial sampling 回退 CPU）
- StSRLSolver 模拟器加速训练
- Communication Mod (ModTheSpire) 用于真实游戏集成
- bottled_ai 启发式规则作为行为克隆的数据源

## 编码约定

- 注释和文档使用中文
- 特征编码：字符串（卡/遗物/动作）→ MD5 hash → embedding 查表（vocab 2048, dim 64），无严格词表
- 标量特征：8 维（hp%, max_hp, gold, floor, act, deck_size, relics_size, potions_size）+ phase one-hot
- 模型结构：set encoder（mean-pool）+ pointer-network 动作打分，单 head 输出 logits
- 日志：决策日志写 `.log` 文件，统计写 `_stats.log`
- Device: 优先 MPS，fallback CPU

## 路径规范（防信息泄漏）
- 严禁在代码、注释、文档里硬编码 `/Users/<username>/...` 或 `~/Documents/code/github/...` 等本机/家目录绝对路径
- 外部仓库引用走 `sts_paths.py` 的 env var + 仓库相对路径方案
- 文档里写示例路径用 `<repo>` / `<your-clone-dir>` / `$STSRLSOLVER_PATH` 占位
- pre-commit hook (.git/hooks/pre-commit) 会拦截违规 staged 改动；新 clone 后需手动启用

## 开发规范

- 每个 demo 文件自包含，不超过 200 行
- agent 文件（`sts_agent*.py`）是单文件架构，包含所有 encoding/model/agent 逻辑
- 训练脚本（`train_v*.py`）负责 simulator 集成和数据收集
- 模型保存为 PyTorch state_dict 到 `sts_models/`

## 当前状态

<!-- last-verified: 2026-05-13 -->
- 2026-05-12: **V8 RL 已搭起，长跑暂停调查事件死循环 bug**。战斗内沿用 search +
  BC combat head (`sts_models/v8_combat_head_v1.pt`，Phase A 产物)，战斗外用纯 model
  RL（PPO）with dense reward shaping。`sts_models/v8_ppo_long_v1` 于 2026-05-12 10:08
  启动，~3.5h / 60 ep 后人为停掉以排查 deterministic eval 死循环（见下方「已知 bug」）。
- 2026-05-06: V8 BC（启发式 teacher）路线证明撞天花板（0/30 beat boss），
  已 archive 到 `archive/v8_bc_pivot/`。
- V8 BC 路线已 archive 的产物：`v8_strategy.py` / `v8_evaluator.py` /
  `v8_inference_bot.py` / 旧 `v8_model.py` / 旧 `v8_trainer.py` / `v8_teacher_*.py` /
  `v8_battle_state_adapter.py` / `scripts/v8_{collect_*,eval,sweep,train_value}.py` /
  `data/v8_*` / `sts_models/v8_{meta,value_head,smoke}_*.pt`。
  （注意：V8 RL 新模块在 `v8/` 包内，与已 archive 的旧 `v8_*.py` 文件不冲突。）
- 路线演进：V3-V5 (DQN/早期 PPO) → V6 (PPO+Transformer) → V7 (搜索+启发式，未完成)
  → V8 BC (监督学习, 已 archive) → **V8 RL (当前阶段)**。
- V6/V7 已归档，详见 docs/v6_architecture_review.md
- **维护规则**：每次切换大版本（如 V8→V9）或大阶段（如 V8 BC→V8 RL）必须同步更新
  「当前状态」和「活跃代码」段，刷新 last-verified 日期，与代码改动一起 commit。

### V8 RL 训练日志

- **trial100** (`sts_models/v8_ppo_trial100/`) — 100 ep 验证跑完成。
  23 boss kill / 100 ep（23% boss rate）；按 batch 拆 28% / 19% / 16% / 75%，
  最后一个 batch 75% 是噪声尖刺，不能当趋势看。证明 pipeline 闭环可跑。
- **`long_v1`**: 100 ep 训练，2026-05-12 启动后 ~60 ep 时人为停掉调查
  deterministic-eval bug，已弃用 (ckpt 在 `sts_models/v8_ppo_long_v1/v8_ppo_ep32.pt`)。
- **`long_v2`**: 2026-05-12 22:00 启动的 128 ep 训练，在 ep=42 因 deterministic-eval
  卡在 Mysterious Sphere event 循环（之前只 fix 了 menu，没 fix handler），
  ep=32 ckpt 保留，整个 run 弃用。
- **`long_v2b`**: 2026-05-12 23:17 启动的 128 ep 训练（applied 完整 fix）。
  截至 2026-05-13 10:19 ckpt ep=96 已落（含 ep=32/64/96 + 1 个 wall_ckpt），
  batch 4 进行中（ep=125/128）。trend 显示 RL **在真学**：act 1 boss 通过率从
  batch 1 的 28% 涨到 batch 4 的 64%，avg_reward 从 +0.1 涨到 +28.9。
  Ckpt 路径 `sts_models/v8_ppo_long_v2b/`。
- 2026-05-11 添加全套结构化诊断日志（`[startup]` / `[combat]` / `[floor]` / `[deck]` /
  `[perf]` / `[guard_cap]` / `[heartbeat]`），见上方「诊断日志」段。
- 2026-05-12 deterministic eval 死循环根因定位：StSRLSolver Python engine
  `_get_mysterious_sphere_choices` 没有按 `event_state.phase` 过滤可选 choices，
  导致 argmax 反复选同一个 event-action，事件不会推进、phase 不会切换。
  Sampling 模式因为有随机性所以不卡。注意 StSRLSolver 上游已经在 2026-04-21
  通过 PR #136/#137 彻底废弃整个 Python engine，迁到 Rust，我们 checkout 还停留在
  legacy Python 代码。
- 2026-05-12 同日为 env 加 event-level 诊断日志（`[event] enter` / `[event] choice` /
  `[event] phase_transition` / `[event] exit`），方便日后精确定位 event-state bug。
  早先尝试过的 `[floor_stall]` 检测器（连续 N step 无楼层变化兜底终止）当天已被 revert，
  不在当前代码里。
- 已知 quirk（不阻塞，留档）：`[guard_cap]` 触发于 `phase=COMBAT enemies=[]`
  （事件触发战斗后偶发），约 4-5% episode 命中，FORCE_TERMINATE 兜底干净。

### 已知 bug / 限制

- **StSRLSolver Python engine `_get_mysterious_sphere_choices` 和
  `_handle_mysterious_sphere` 缺 phase filter**（2026-05-12 发现，已在
  `external/StSRLSolver/` 分支 `fix/mysterious-sphere-phase-filter` 修复，
  commits `11b15c7a` + `f8006f30`）。上游已弃用整个 Python engine
  （PR #136/#137），fix 仅本地，不能 push（fork 被 GitHub abuse-prevention 禁用了）。
  **事实上我们 own 这个 fork**。

## 运行中的训练进程（2026-05-13 状态快照）

如新会话被 ScheduleWakeup 唤醒来 monitor，按以下信息接手：

- **进程**：v2b 训练在 nohup 下运行，父 PID 68304（python 子进程在父 PID 下）
- **日志文件**：`/tmp/v8_ppo_long_v2b.log` 全程 append；`/tmp/v8_ppo_long_v2b.exit` 在训练结束时出现
- **Output 目录**：`sts_models/v8_ppo_long_v2b/`，ckpt 文件 ep=32/64/96 + wall + 最终 ep=128 + final
- **参数**：num_episodes=128 batch_size=32 ckpt_freq=32 eval_freq=100
- **预期完成**：~12:30-13:30（剩 ~3 ep + PPO + 30 seeds 完整 game eval）
- **下一步**：训练完成后跑 metrics 分析（用户已授权，但说"等这一批完再做"）。下批训练参数 user 已定：每次 128 或 512 ep，ckpt_freq=32

接手 monitor 的检查清单（一行 bash 看全貌）：
- ckpt 落盘进度：`ls sts_models/v8_ppo_long_v2b/`
- 训练是否还活：`ps -ef | grep v8_ppo_train.py | grep -v grep`
- 异常监测：`grep -cE "\[guard_cap\]|MysteriousSphere event_phase=COMBAT_WON|Error|Traceback" /tmp/v8_ppo_long_v2b.log`
- 当前 ep：`grep "\[heartbeat\]" /tmp/v8_ppo_long_v2b.log | tail -1`
- 训练结束信号：`cat /tmp/v8_ppo_long_v2b.exit 2>&1`

## 子 Agent 协作规范

### 代码修改
- 修改函数/方法时，必须搜索所有调用方和引用，不能只改定义
- 修完后先跑 `python -c "import module"` 验证无语法/导入错误
- 删除代码前必须 grep 确认无其他文件引用

### 训练执行
- 启动训练前，先 `pkill -f` 清理同名旧进程
- 先跑 1 局测试，确认无运行时错误，再跑完整训练
- 失败的 agent 不应自动重试启动训练，应先报告错误等待指令
- 运行训练/测试时，使用单条 `&&` 串行命令在同一个 shell 中执行，禁止多次独立 bash 调用启动 Python 进程

### 主对话规则（强制）
- 主对话严禁直接使用 Read/Grep/Glob/Bash/Edit/Write 工具
- 所有文件读取、搜索、代码修改、命令执行必须通过 Agent 子 agent 完成
- 主对话只允许：讨论方向、确认方案、调度子 agent、总结子 agent 返回的结果
- 如果需要了解代码现状，派 Explore 子 agent 去看，不要自己读文件
- 违反此规则会浪费主对话上下文窗口，导致无法进行深度协作

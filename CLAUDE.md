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
- 标量特征：8 维 = `[hp_ratio, max_hp/100, floor/17, act/3, gold/999, energy/5, num_potions/5, hp/100]`
  （`deck_size` / `relics_size` 不在 scalar，通过 set-encoder mean-pool 隐式传递；
  note: `energy` 维度在元决策时永远为 0、`hp` 编了两次——已知冗余，待下次 schema 调整时清理）
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

## 训练迭代工作流（强制）

训练是**持续迭代过程**，不是"跑完一批问要不要继续"。每批完成后自动按决策树执行：

### 决策树

| 情况 | 自动动作 |
|---|---|
| 数据有改善（任何 metric 涨）| 起下批同参续训 |
| 训练过程出 bug（Traceback / hang / 数据异常）| 杀进程 → 翻 log → 修代码 → 重启续训 |
| 数据涨不动（连续 ≥3 批无改善）| 归因分析（翻 log 找 root cause + 提假设 + 实施 A/B 测试）→ 继续训练 |
| 系统级阻塞（OOM / disk / 模型架构必须改）| 报告 + 等用户指令 |
| destructive 操作（删大量 ckpt / 回滚已 push commit）| 报告 + 等用户指令 |

### 定时抽查（强制）

之前因 simulator bug 浪费 120+ 小时训练。任何"加速"都得配套"防错"。

**每 2-3 批训练完**：
1. 派 sub-agent 拉最近批的 5 局完整 trace
2. 对照 StS wiki 验证（敌人 hp / intent / 卡牌伤害 / relic 触发 / boss pre-battle effect）
3. 任何不一致 → 立刻报告 + 停训查 bug

**每 5 批 或 里程碑前后**（首次 act1 boss kill / 首次通关）：
1. 派 sub-agent 起 STS + ModTheSpire 实机
2. 用最新 ckpt 跑 10 局
3. 对比训练 metrics（胜率 / 平均楼层 / 各 boss 表现）vs 实机
4. drift > 30% → 报告 + 调查

### 报告原则

每批完成给简短报告，重点：
- 哪些 metric 改善 / 持平 / 倒退（vs 上批 + vs baseline）
- 模型行为变化（从 log 抽：选 AOE 卡偏好、避免某路径、deck 构筑特征等）
- 距目标多远（A0 通关率、act 进展、SlimeBoss 单点等）
- 已起的下批 PID（默认）

**不问用户**："继续吗" / "选 A 还是 B" / "做 X 还是 Y"。决策按上面的树自动办。

详见 memory: `feedback_continuous_iteration_no_interrupt.md` / `feedback_periodic_audit_required.md`

## 当前状态

<!-- last-verified: 2026-05-26 (batch_v28 启动 + v27 完成 a1_beat=16.7%; plateau 内持续小幅恢复, ESCALATION 仍待 user 决策) -->
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

完整训练历史（trial100 / long_v1-v4 / batch_v5-v22）已归档至 [docs/v8_training_log.md](docs/v8_training_log.md)。

CLAUDE.md 只保留最近活跃训练 + 还在用的知识。

**最近批次概览** (simulator-fix 后干净训练序列，a1_boss_beat eval 维度)：

| Batch | ep | a1_boss_beat | won_game | floor_mean | completed | 备注 |
|---|---|---|---|---|---|---|
| **v25** | **1408** | **3.3%** | **0.00** | **11.83** | **30/30** | **-26.7pp 历史最大单批 dip (2x 之前最深), 与 v21 历史最低持平; 14 reach / 1 kill** |
| **v26** | **1536** | **13.3%** | **0.00** | **11.93** | **30/30** | **+10pp 小幅回升但远低于 peak; N=3 plateau 严格触发 (v23-v26 未超 v22 peak)** |
| **v27** | **1664** | **16.7%** | **0.00** | **12.1** | **30/30** | **+3.3pp 连续 2 批小幅恢复; reached_boss=60% (+10pp); Hexaghost-heavy seed mix; plateau 持续** |

完整 v15-v27 细节 + audit findings 详见 [docs/v8_training_log.md](docs/v8_training_log.md)。

**[ESCALATION] N=3 plateau 触发, 归因方向待 user 决策**: v22 peak 30% 后, v23/v24/v25/v26/v27
连续 5 批未超 peak (16.7% / 30% / 3.3% / 13.3% / 16.7%, bimodal 抖动但 ceiling 卡 30%)。
autonomous loop 在 plateau 内继续小步迭代 (v28 已起), **等 user 决策归因方向**:
- 候选: reward shaping drift / entropy collapse / Adam moment 漂移 / 回滚 v22 或 v24 peak ckpt
- A/B 候选: 平行从 v22 / v24 重训对比 trajectory 稳定性; entropy bonus +20% 看是否减少 dip
- 实机测试基础设施已搭好 (`tools/run_real_test.sh` + multi-game `v8_play_real.py`),
  待修 subscreen handling (GRID/SHOP/multi-phase EVENT) 后可做 training vs 实机对照验证

- **`batch_v28` 启动 (2026-05-26, 续训, plateau 内自动起下批)**：
  从 v27 final ckpt 续训。**autonomous loop 在 plateau 内继续小步迭代, 等 user 决策归因方向**。
  - **续训源**：`sts_models/v8_ppo_batch_v27/v8_ppo_final.pt` (episodes_done=1664)
  - **参数**：`num_episodes=1792 batch_size=32 ckpt_freq=32 eval_freq=128`
    (n_envs=1 serial, 增量训 128 ep, ep 1665→1792)
  - **PID**：`60472`（nohup）；log `/tmp/v8_ppo_batch_v28.log`；output
    `sts_models/v8_ppo_batch_v28/`；exit signal file `/tmp/v8_ppo_batch_v28.exit`（如有）
  - **启动校验**：`[resume] start_episode=1664, target=1792 (将增量训 128 ep)`，0 Traceback
  - **预期**：~2-2.5h 训练 + final eval

接手 monitor 的检查清单（v28）：
- ckpt 落盘进度：`ls sts_models/v8_ppo_batch_v28/`
- 训练是否还活：`ps -ef | grep v8_ppo_train.py | grep -v grep`
- 异常监测：`grep -cE "\[guard_cap\]|MysteriousSphere event_phase=COMBAT_WON|Error|Traceback" /tmp/v8_ppo_batch_v28.log`
- 当前 ep：`grep "\[heartbeat\]" /tmp/v8_ppo_batch_v28.log | tail -1`
- 验证 resume 生效：`grep "\[resume\]" /tmp/v8_ppo_batch_v28.log`（应见 start_episode=1664）

### 已修复 bug
- **Action token mode-collapse bug** (2026-05-13): `v8/action_space.py` CARD_REWARD / EVENT / SHOP 三个 phase 的 token 字符串现在注入 card_name / event choice text / shop item name。**v2b ckpt 的 token-prior 已失效**，下批训练 fresh start。
- **StSRLSolver 持久 relic counter 不回写 + `get_pre_battle_effects()` 从未调用**
  (2026-05-22 发现 + 同日修复, fork commit `e567c65d`):
  NeowsLament 永远不递减（NEOW idx=0 → 全敌人 1 HP 全程持续），加上 8 个 boss/elite
  的 atBattleStart buff 全部失效。让 RL 学了一年的 exploit 而不是 STS。详见上方
  「Simulator bug 大爆发」段。**v3-v13 训练数据全部污染**，v14 从 trial100 重训。
- **Mushrooms event handler 缺 phase filter** (2026-05-14 发现, 2026-05-15 修,
  **2026-05-15 integration validated**):
  v4 ep=127 deterministic eval 卡 15h / 6352 次同 event choice 后定位根因。
  StSRLSolver fork 已修 (commit on `external/StSRLSolver/`)；env 侧加 event_stall
  guard 兜底 (`v8/env.py` commit `79b3d51`)：单 episode 同 event_id choice >= 30
  → FORCE_TERMINATE，防御未知同类 bug。**Integration 验证**：batch_v4 30-seed eval
  recovered run 内 Mushrooms 出现 1 次，phase INITIAL → COMBAT_WON → resolved 正常退出，
  event_stall guard 0 触发。

### Schema 变化
- Eval 输出新字段：`act1_boss_beat_rate` / `act2_boss_beat_rate` / `won_game_rate` / `boss_kill_counts` / `boss_reach_counts`。旧 `beat_boss_rate` 字段保留为 deprecated（= `act1_boss_beat_rate`）。
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

- **StSRLSolver fork 不能 push to GitHub**：fork 被 abuse-prevention 禁用了，
  所有 simulator 修复 commit 仅在本地存活。**事实上我们 own 这个 fork**，commits
  全在 `external/StSRLSolver/` 分支 `fix/mysterious-sphere-phase-filter`：
  - `11b15c7a` + `f8006f30`：Mysterious Sphere choices / handler 缺 phase filter
    (2026-05-12)
  - `b0626dc6`：Mushrooms choices / handler 缺 phase filter (2026-05-15)
  - `e567c65d`：persistent relic counter writeback + enemy
    `get_pre_battle_effects` wired (2026-05-22, **巨型 bug**, 让 NeowsLament
    永远不递减 + 8 个 boss/elite atBattleStart buff 全部失效)
  上游已弃用整个 Python engine（PR #136/#137 迁到 Rust），我们 checkout 还停留
  在 legacy Python 代码。

## Simulator fix milestone（2026-05-22）

2026-05-22 在 StSRLSolver 定位到 2 个致命 bug：
1. **Counter writeback 缺失**: `_end_combat` 没把 combat 侧 relic counter dict 回写到 `run_state.relics`，导致 NeowsLament 永远不递减（永久 1HP 全敌人）
2. **`get_pre_battle_effects()` 从未调用**: 8 个 boss/elite 的 atBattleStart buff 全部失效（Artifact/Unawakened/Regen/Curiosity/Time Warp/Invincible/BeatOfDeath/Surrounded）

修复 commit `e567c65d` on fork branch `fix/mysterious-sphere-phase-filter`（fork 不能 push to GitHub，本地存活）。

**影响**：v3-v13 训练数据全部污染（~140h compute），模型学了 NeowsLament 1HP exploit 而非真实 STS。v14/v14b 已 kill。v15 起为 simulator fix 后首个干净基线。

完整 bug 描述 + audit 细节 + v14b/v15 启动 timeline 详见 [docs/v8_training_log.md](docs/v8_training_log.md)。

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

### 续训规范（2026-05-15 立规，commit `bf2208f` 之后必须遵守）
- **默认续训**：所有新 batch_v<N> 必须 `--resume_from=<上批 best ckpt>`，除非有明确
  理由 fresh start（如：模型结构变了 / 加新 head / debug 隔离）。Fresh start 时必须
  在 commit message 里说明理由。
- **Ckpt 选择**：优先选上批 final ckpt；若 final 缺失（如 v4 卡死中止）用最新 wall ckpt
  或最大 ep 数的 ep_ckpt（如 `v8_ppo_long_v4/v8_ppo_wall_20260514_135748.pt`，
  metadata.episodes_done=96）。
- **`--num_episodes` 是累计目标值**，不是增量。比如续训从 ep=96 起想再训 160 ep，
  `--num_episodes` 写 256（96+160），不是 160。trainer 会自动按 start_episode 之差
  计算增量训练量；`[resume]` 日志会显示 `将增量训 N ep`。
- **Self-check 提示**：output_dir 已有 ckpt 但没传 `--resume_from` 时，日志会输出
  `[resume-check]` WARN 行；看到这行就停下检查是不是写错 output_dir / 漏传 resume_from。
- **背景**：2026-05-15 发现 batch_v3/v4/v5 一直是 fresh init 而非续训（CLI 没暴露
  `--resume_from`，trainer 内部支持），浪费 30h+ 训练时间。规范立此防再犯。

### 主对话规则（强制）
- 主对话严禁直接使用 Read/Grep/Glob/Bash/Edit/Write 工具
- 所有文件读取、搜索、代码修改、命令执行必须通过 Agent 子 agent 完成
- 主对话只允许：讨论方向、确认方案、调度子 agent、总结子 agent 返回的结果
- 如果需要了解代码现状，派 Explore 子 agent 去看，不要自己读文件
- 违反此规则会浪费主对话上下文窗口，导致无法进行深度协作

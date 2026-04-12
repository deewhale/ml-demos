# CLAUDE.md

## 项目概述

ML 学习进阶项目：监督学习 → DQN → PPO+Transformer。最终目标：零先验知识的 Slay the Spire AI agent。

## 项目结构

- `demo1_mnist.py`, `demo2_cartpole.py`, `demo3_slay_the_spire.py` — 三个教学 demo
- `sts_agent.py` — V3 agent (DQN, 已停止迭代)
- `sts_agent_v4.py` — V4 agent (PPO + Transformer shared backbone, 当前主力)
- `train_v4.py` — V4 训练管线 (StSRLSolver 集成)
- `train_v5.py` — V5 验证 (TurnSolver combat + v4 agent strategy)
- `pretrain_v5.py` — V5 behavior cloning 预训练
- `play_v5.py` — V5 实战运行
- `test_solver_game.py` — solver 可行性测试
- `docs/` — 架构文档和设计笔记
- `docs/v5_strategy_proposal.md` — V5 策略重写方案（已确认）
- `docs/v5_full_design.md` — V5 完整设计文档
- `docs/game_loop_bugs.md` — 游戏循环 bug 清单
- `docs/analysis_*.md` — V5 设计分析文档
- `tasks/` — PRD 和任务文档
- `sts_models/` — 模型权重存储

## 技术栈

- Python 3.12/3.13, PyTorch, Gymnasium
- Apple Silicon MPS 训练（multinomial sampling 回退 CPU）
- StSRLSolver 模拟器加速训练
- Communication Mod (ModTheSpire) 用于真实游戏集成

## 编码约定

- 注释和文档使用中文
- 特征编码：bloom filter hash（`hash_to_buckets`），不硬编码游戏知识
- 卡牌：逐卡 token（22-dim 数值特征 + 16-dim 可学习 ID 嵌入），power 用 hash bucket + 显式追踪
- 模型结构：shared encoder → task-specific heads → 每个 head 独立 value function
- 日志：决策日志写 `.log` 文件，统计写 `_stats.log`
- Device: 优先 MPS，fallback CPU

## 开发规范

- 每个 demo 文件自包含，不超过 200 行
- agent 文件（`sts_agent*.py`）是单文件架构，包含所有 encoding/model/agent 逻辑
- 训练脚本（`train_v*.py`）负责 simulator 集成和数据收集
- 模型保存为 PyTorch state_dict 到 `sts_models/`

## 当前状态

- V3: 34 局训练，floor 6-16 plateau，已归档
- V4: PPO + Transformer，正在测试验证
- V5: 策略层重写中 — Solver 接管 combat，RL 只训练 draft/path/choice

## V5 架构

- TurnSolver 处理所有 combat 决策（搜索树，按房间类型分配算力）
- Agent（Transformer + PPO）只处理策略层：draft（选牌）/ path（选路）/ choice（事件/休息/商店）
- Combat head 保留但冻结，作为 Solver 不适用时的退路
- 每局约 50 个策略决策，γ=0.99
- Reward: hp_value delta（层间）+ floor milestones + win bonus，无 STEP_COST
- 详细设计见 docs/v5_strategy_proposal.md

## 预训练 (Pretrain)

- `pretrain_v5.py` — 从人类 .run 数据做 behavior cloning 预训练
- 数据源：`SLAY_DATA_PATH/runs/` (800+ 局)
- 训练三个 head：draft（card_choices）、choice（campfire + boss_relics）、path（path_taken）
- 已知问题（待修复）：
  - 性能瓶颈：逐样本前向传播，未做 batch，全量数据 5 epochs 需 10+ 分钟
  - 保存格式已修复为 {"model": state_dict} 包装格式
  - 预训练效果待验证（修复前的模型未正确加载，结果不可靠）

### V5 待实现（见 docs/v5_full_design.md）
- 商店决策（买牌/买遗物/删牌/离开）
- 删牌接入 draft head
- Neow 祝福接入 choice head
- 事件编码优化（关键词检测 + 数字提取，CHOICE_OPTION_DIM 20→30）
- 遗物编码（新增 encode_relic，16-dim）
- 确认 _build_tokens 为逐卡 token 模式

## 已知问题

- 游戏循环有多个未修复 bug，详见 docs/game_loop_bugs.md
- 编码层（PRD 16 个 US）已完成，但 train_v5.py 的游戏流程未正确传递所有状态

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

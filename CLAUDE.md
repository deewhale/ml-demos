# CLAUDE.md

## 项目概述

ML 学习进阶项目：监督学习 → DQN → PPO+Transformer。最终目标：零先验知识的 Slay the Spire AI agent。

## 项目结构

### 活跃代码（V8 → 推 RL）
- `v8_bot.py` — 战斗内 search infra（GameRunner + TurnSolver + phase dispatch）。
  元决策原本调 `v8_strategy.*`，BC 路线 archive 后已替换为 `_StratStub`，
  调用元决策 dispatch 会抛 NotImplementedError，**待 RL 重构时换成 RL policy**。
- `v8_data_collector.py` — 元决策点采集（JSONL schema 设计可参考）。
  注意：本文件 import v8_strategy / v8_teacher_eval / v8_battle_state_adapter，
  这些模块已 archive，**当前 import 会失败**，RL 重构时再视情况删/改。
- `data/sts_data.py` — STS 数据提取（V6 遗留，可能复用）

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

<!-- last-verified: 2026-05-06 -->
- 2026-05-06: V8 BC（启发式 teacher）路线证明撞天花板（0/30 beat boss），
  已 archive 到 `archive/v8_bc_pivot/`。下一步推 RL（战斗内搜索 + 战斗外纯 model RL with dense reward shaping）。
- V8 BC 路线已 archive 的产物：`v8_strategy.py` / `v8_evaluator.py` /
  `v8_inference_bot.py` / `v8_model.py` / `v8_trainer.py` / `v8_teacher_*.py` /
  `v8_battle_state_adapter.py` / `scripts/v8_{collect_*,eval,sweep,train_value}.py` /
  `data/v8_*` / `sts_models/v8_{meta,value_head,smoke}_*.pt`。
- 保留：`v8_bot.py`（战斗内 search infra，元决策待 RL 重构）+
  `v8_data_collector.py`（schema 参考，当前因 import 已 archive 模块而无法直接 import）。
- 路线演进：V3-V5 (DQN/早期 PPO) → V6 (PPO+Transformer) → V7 (搜索+启发式，未完成) → V8 (监督学习, 已 archive) → V9 RL（待开始）
- V6/V7 已归档，详见 docs/v6_architecture_review.md
- **维护规则**：每次切换大版本（如 V8→V9）必须同步更新「当前状态」和「活跃代码」段，刷新 last-verified 日期，与代码改动一起 commit。

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

# ML Demos - 机器学习进阶项目

从监督学习到强化学习，最终目标：零先验知识的 Slay the Spire AI agent。

## 项目结构

### 活跃代码（V6）
- `sts_agent_v6.py` — V6 agent（统一效果编码 + Transformer + PPO）
- `train_v6.py` — V6 训练管线（多进程 + StSRLSolver 模拟器）
- `pretrain_v6.py` — V6 预训练（TurnSolver 教师 + behavior cloning）
- `data/sts_data.py` — STS 数据提取（卡牌/遗物/药水统一效果向量）

### 文档
- `docs/v6_training_log.md` — V6 训练日志
- `docs/v6_architecture_review.md` — V3→V6 架构演进回顾
- `docs/archive/` — 旧版本设计文档（V3-V5）

### 归档
- `archive/` — 旧版本代码（V3 DQN, V4 PPO+Transformer, V5 TurnSolver）
- `demos/` — 教学 demo（MNIST、CartPole、简化 STS）
- `tools/` — 工具脚本

## 环境依赖与 Setup

### 系统要求
- macOS / Linux（已在 macOS Darwin 25 验证）
- Python 3.12+（Python 3.9 装不了 torch 2.10+）
- git
- 可选：gh CLI（管理 GitHub repo）

### 一次性 Setup

1. clone ml-demos（你已经在这里了）

2. clone StSRLSolver 到 `external/StSRLSolver` 并 checkout 验证版本：
   ```bash
   git clone https://github.com/JackSwitzer/StSRLSolver.git external/StSRLSolver
   cd external/StSRLSolver && git checkout e82f8296 && cd ../..
   ```
   注意：StSRLSolver 当前 master 已把 Python engine 移到 Rust，**不要 git pull / merge**，否则 packages/engine 会消失。

3. 创建 venv 并安装依赖：
   ```bash
   python3.12 -m venv .venv
   .venv/bin/pip install "gymnasium>=0.29.0" "numpy>=1.24.0" "torch>=2.10.0" "psutil>=7.2.2"
   ```
   实测可行版本：torch 2.11.0、numpy 2.4.4、gymnasium 1.3.0、psutil 7.2.2

4. 启用 pre-commit hook（防绝对路径泄漏）：
   ```bash
   bash scripts/setup_hooks.sh
   ```

5. 验证：
   ```bash
   .venv/bin/python -c "from sts_paths import ensure_on_sys_path; ensure_on_sys_path(); from packages.engine.game import GameRunner; print('ok')"
   .venv/bin/python run_v7.py --n-games 1 --seeds 42
   ```

### StSRLSolver 路径配置

`sts_paths.py` 按以下顺序解析 StSRLSolver 路径：
1. 环境变量 `STSRLSOLVER_PATH`（如已设置）
2. 否则用 `<ml-demos>/external/StSRLSolver`

如果你 clone 到了别的位置：
```bash
export STSRLSOLVER_PATH=/path/to/StSRLSolver
```

### 不需要装的依赖

StSRLSolver 的 `pyproject.toml` 还列了 mlx / fastapi / uvicorn / websockets，V7/V8 当前用不到，**不要装**。

## 杀戮尖塔 AI Agent

通过 [CommunicationMod](https://github.com/ForgottenArbiter/CommunicationMod) 对接真实杀戮尖塔游戏。

### 设计理念

- 统一效果编码：从游戏数据提取卡牌/遗物/药水的真实效果向量
- Transformer + PPO：共享编码器，多任务决策头（战斗/选牌/选路/事件）
- Per-card reward：每张打出的卡牌获得独立奖励信号
- 架构演进：V3 (DQN) → V4 (PPO+Transformer) → V5 (Solver) → V6 (统一效果编码)

### 配置

**Mods 需要 (Steam 创意工坊):**
1. ModTheSpire — mod 加载器
2. BaseMod — 基础依赖
3. CommunicationMod — AI 通信接口
4. SuperFastMode — 去掉动画加速训练

## 核心概念速览

- **监督学习**: 给数据和答案 → 模型学规律 → 预测新数据
- **强化学习**: 智能体试错 → 做对了奖励/做错了惩罚 → 逐渐学会策略
- **DQN**: 用神经网络估算「这个状态下做某动作能拿多少分」

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

## 安装依赖

```bash
pip3 install numpy torch torchvision gymnasium matplotlib
```

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

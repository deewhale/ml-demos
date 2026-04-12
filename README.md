# ML Demos - 机器学习入门三连

从零开始理解机器学习，3 个递进式 demo。

## 安装依赖

```bash
pip3 install numpy torch torchvision gymnasium matplotlib
```

## Demo 列表

| Demo | 主题 | ML 类型 | 运行命令 |
|------|------|---------|---------|
| 1 | 手写数字识别 (MNIST) | 监督学习 | `python3 demo1_mnist.py` |
| 2 | AI 平衡小车 (CartPole) | 强化学习 (DQN) | `python3 demo2_cartpole.py` |
| 3 | 简化版杀戮尖塔 | 强化学习 (DQN + 卡牌) | `python3 demo3_slay_the_spire.py` |

## 杀戮尖塔实机 Agent (sts_agent.py)

通过 [CommunicationMod](https://github.com/ForgottenArbiter/CommunicationMod) 对接真实杀戮尖塔游戏，**纯学习，零硬编码**。

### 设计理念

不预设任何游戏知识，所有策略从实战中学习：
- 卡牌效果不需要人工标注 — 观察「打出前后状态变化」自动学习
- Buff/Debuff 用哈希桶编码 — 不需要知道力量、虚弱是什么，只需要知道它是个 power 且有数值
- 选卡从胜负信号中学习 — 赢了说明牌选得好，输了反向更新
- 卡牌用 ID 哈希标识 — 相同卡牌在网络中始终是同一个「指纹」，网络自己学这个指纹代表什么
- 组合效应通过 Self-Attention 学习 — 日晷看到0费攻击、Rushdown 看到姿态切换牌时 attention 权重高

### 配置

**Mods 需要 (Steam 创意工坊):**
1. ModTheSpire — mod 加载器
2. BaseMod — 基础依赖
3. CommunicationMod — AI 通信接口
4. SuperFastMode — 去掉动画加速训练

**CommunicationMod 配置** (`~/Library/Preferences/ModTheSpire/CommunicationMod/config.properties`):
```properties
command=python3 REPO_ROOT/sts_agent.py
runAtGameStart=true
verbose=true
```

**启动**: Steam → 杀戮尖塔 → ModTheSpire 全部勾选 → Play
**训练难度**: Ascension 20 (铁甲战士)
**训练模式**: 全自动循环，一局结束自动开下一局，无需人工干预

### 监控

```bash
# 每局结果汇总 (胜负/层数/HP/训练进度)
tail -f REPO_ROOT/sts_stats.log

# 实时决策日志
tail -f REPO_ROOT/sts_agent.log
```

### 架构 (v3)

| 模块 | 方法 | 学习信号 |
|------|------|---------|
| **DeckEncoder** | Self-Attention (卡牌+遗物→deck DNA) | 通过选卡/路线网络的梯度联合训练 |
| 战斗 | DQN (356维: 战场+deck DNA, 11动作) | 每步状态变化 + 战斗胜负 |
| 选卡 | 评估网络 (deck DNA+候选卡→分数) | 每局胜负 × 剩余HP |
| 路线 | 评估网络 (deck DNA+状态+节点→分数) | 每局胜负 |
| 篝火 | HP 阈值 (唯一硬编码) | — |

### DeckEncoder — 组合效应的核心

用 Self-Attention 编码卡牌+遗物的配合关系:
- 卡牌和遗物统一为「物品」(18维: 费用/伤害/格挡/类型/稀有度/哈希指纹)
- Multi-Head Attention (2头) 让物品之间互相关注 → 学习配合
- Mean/Max pooling + 数量统计(牌组厚度/遗物数) → 32维 deck DNA
- 训练后应能学到: 日晷+0费攻击=无限, Rushdown+姿态切换=抽牌引擎

### 状态编码

**战斗** (356维): 战场324维 + deck DNA 32维
- 玩家 (28维): HP%/格挡/能量/回合 + 24桶哈希 powers
- 怪物 (29×4=116维): HP%/格挡/意图/存活 + 24桶哈希 powers
- 手牌 (18×10=180维): 费用/伤害/格挡/类型 + 12维哈希指纹

**选卡** (54维): deck DNA 32维 + 上下文4维 + 候选卡18维

### 训练预估

| 阶段 | 局数 | 时间 (SuperFastMode) | 预期效果 |
|------|------|---------------------|---------|
| 基本出牌 | ~50 | ~2小时 | 学会看意图防御、有牌就出 |
| 战斗策略 | ~200 | ~8小时 | 学会优先级、能量管理 |
| 选卡偏好 | ~200+ | ~8小时 | 学会哪些牌好、何时跳过 |
| 组合发现 | ~500+ | ~20小时 | 开始识别卡牌/遗物配合 |

### 关键文件

- `sts_agent.py` — agent 主程序 (~210K 参数)
- `sts_agent.log` — 实时决策日志
- `sts_stats.log` — 每局结果统计
- `sts_models/agent_v3.pt` — 所有网络权重，跨局累积
- `demo3_slay_the_spire.py` — 简化版训练 demo (独立)

### 已知问题 & TODO

- [ ] 模型还没实际跑过完整一局，需要启动游戏验证
- [ ] 事件选择过于简单（总是第一项）
- [ ] 商店直接跳过（应学会买关键牌/移除诅咒）
- [ ] 药水使用未实现
- [ ] 前期 ε=0.35 探索率高，A20 早期会大量死亡
- [ ] DeckEncoder 只有 2 头 attention，复杂 combo 可能需要更深
- [ ] 选卡训练信号稀疏（只有局末胜负），可考虑加中间奖励
- [ ] 只支持铁甲战士，其他角色无需改代码（哈希编码通用）但需要单独训练

## 核心概念速览

- **监督学习**: 给数据和答案 → 模型学规律 → 预测新数据
- **强化学习**: 智能体试错 → 做对了奖励/做错了惩罚 → 逐渐学会策略
- **DQN**: 用神经网络估算「这个状态下做某动作能拿多少分」

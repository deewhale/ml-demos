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
  **redesign 后加 deck_strength 5 维输入特征**（卡组评分喂模型，不再当奖励，commit `6922a9e`）。
- `v8/combat_net_wrapper.py` — 把 V8Model 桥接到 StSRLSolver 的 combat_net hook。
  **当前已禁用**：env reset 里 `neural_eval=None`，战斗走 StSRLSolver 手写启发式搜索，
  combat_net（随机权重桩）不参与。BC combat head（`sts_models/v8_combat_head_v1.pt`，
  真权重）只作选牌 model 的预热初始化加载，**不冻结、训练里不被调用**（死参数，详见
  「当前状态」段）。
- `v8/card_scorer.py` — 按局滚动的 per-card 卡组评分（从战斗 per-card 细账
  `card_log` 累计）。**输出不进奖励**（2026-06-01 删 strength_reward 后）——只打日志
  + 作选牌 model 的输入特征。五维全部已实现（**非占位非 0**）：输出 / 防御两锚维 +
  运转 / 加费 / 能力三协同维，权重 `1.0 / 1.0 / 0.30 / 0.30 / 0.15`。
- `v8/deck_evaluator.py` — 旧战后牌组模拟评分（9 worker ProcessPoolExecutor）。
  **redesign 后 reward 不再调用**（已弃用模拟战评分），保留参考。
- `v8/reward.py` — step reward shaping。**2026-06-01 (commit `60ea0b9`) 删 `strength_reward`
  「卡组实力增长」(8×Δdeck) 后，改为纯「进度 + 过 boss + 通关」主轴**：
  step = `0.5×node + combat + floor_progress`，各项 floor=+3/层、boss_beat=+25、
  boss_hp=+15×到达 boss 时血量比、combat win=+1/lose=-5、通关=+100、
  hp_terminal=+5×final_hp（hp 罚 / 回合罚 / damage_ratio 此前已删；strength_reward 因刷分
  —— 死亡局净赚 +485 —— 删除）。战斗 per-card 细账由引擎 `CombatResult.card_log` /
  `game.last_combat_card_log` 导出，env 存 `_last_combat_card_log`
  （引擎 fork commit `75cb883d` + 主仓 `5cc87a9`），现仅供 `card_scorer` 打分作特征。
- `tools/v8_ppo_train.py` — 训练入口。常用参数：`--num_episodes` `--batch_size`
  `--checkpoint_frequency` `--eval_frequency` `--output_dir` `--smoke`。
- `data/sts_data.py` — STS 数据提取（V6 遗留，可能复用）。
- 归档但保留参考：`v8_bot.py`（战斗内 search infra，元决策 dispatch 已被 V8 RL 取代）、
  `v8_data_collector.py`（JSONL schema 参考，import 已 archive 模块所以无法直接运行）。

#### 引擎验收测试台（2026-06-04 新增，证伪 legacy 引擎 + 验证 lightspeed）
引擎无关的卡牌/遗物/怪物行为测试台：用独立 wiki oracle 当金标准，跨后端跑同一组用例验数值。
- `v8/backends/combat_probe.py` — 后端无关的战斗探针接口（各引擎实现它即可被测）。
- `v8/backends/stsrl_combat_probe.py` / `rust_combat_probe.py` / `rust_engine_loader.py` —
  legacy Python 与 Rust 版 StSRLSolver 的探针 + 加载器（证伪 legacy、评估 Rust 用）。
- `v8/backends/lightspeed_loader.py` / `lightspeed_combat_probe.py` /
  `lightspeed_relic_probe.py` / `lightspeed_monster_probe.py` — sts_lightspeed 的加载器 +
  战斗/遗物/怪物三类探针（现役准确引擎，验收通过）。
- `tools/card_oracle.py` / `relic_oracle.py` / `monster_oracle.py` — 独立 wiki 金标准期望值。
- `tools/test_card_behavior.py` / `test_relic_behavior.py` / `test_monster_behavior.py` —
  跑器：把 oracle 期望值喂给各后端探针逐条比对。

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
- `docs/v8_training_log.md` — V8 RL 完整训练历史（trial100 / long / batch_v* / redesign_*）
- `docs/v8_rl_diagnosis_2026-05-29.md` — V8 RL 全面诊断（推翻"随机数指挥"假设 + 标准 RL 对照）
- `docs/v8_rl_fix_plan_2026-05-29.md` — 六阶段修复计划 + 防钻空子红线
- `docs/v8_design_principles.md` — V8 设计原则（用户原话存档）
- `docs/parallel_env_design.md` — 并行 env 设计
- `docs/archive/` — 旧版本设计文档（由专门 agent 创建，放入 `v8_implementation_design.md` /
  `v8_alphazero_lite_design.md` / `v8_sweep_log.md` 等历史设计/分析）
- 注：`docs/v6_training_log.md` / `docs/v6_architecture_review.md` 已不存在（旧引用为死链，已移除）

### 归档（不追踪）
- `archive/v3/` — V3 DQN agent
- `archive/v4/` — V4 PPO + Transformer
- `archive/v5/` — V5 TurnSolver + 策略层
- `docs/archive/` — V8 历史设计文档（`v8_implementation_design.md` /
  `v8_alphazero_lite_design.md` / `v8_sweep_log.md`，已弃用但留参考）
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

<!-- last-verified: 2026-06-04 (本次变更: 建引擎无关测试台证伪 legacy Python StSRLSolver 系统性坏, 拿测试集验证 sts_lightspeed 为准确引擎, 现役方向是把训练引擎从 StSRLSolver 换到 lightspeed) -->
- 2026-06-04: **引擎换代——测试台证伪 legacy Python 引擎、锁定并验证 sts_lightspeed 为准确引擎，现役方向是换引擎到 lightspeed**。
  - **建测试台**：引擎无关的卡牌/遗物/怪物行为测试台（`CombatProbe` 接口 + 独立 wiki
    oracle + 后端无关跑器），证明能逮真 bug。新增代码见下方「活跃代码」段「引擎验收测试台」。
  - **证伪在用引擎**：用测试台证明在用的 **legacy Python StSRLSolver 系统性坏**——
    一整类 effect 串无 handler 被静默丢弃。确诊坏卡：Dark Shackles / Panacea /
    Panic Button / J.A.X. / Sword Boomerang（细账记 `docs/v8_engine_card_bug_ledger.md`）。
  - **Rust StSRLSolver 不够用**：战斗更准但整局只支持 Watcher，不适合直接训 Ironclad。
  - **锁定金标准 = sts_lightspeed**（gamerpuppy/sts_lightspeed，C++17 + pybind11，MIT，
    机制社区公认最准，全 Ironclad + 全四幕，clone 在 `external/sts_lightspeed`，gitignored）。
  - **测试集验收 lightspeed（mac 编译通 + 写绑定）**：卡牌可验数值 **51/51 对**、
    开战遗物 **11/11 对**、怪物 HP/意图/pre-battle **17/18 对**（1 个是表示差异非 bug）。
    历史「开战效果不触发」重灾区 lightspeed 全部正确；能从 Python 驱动打完整局 Ironclad
    （整局导航已绑）。
  - **现役方向**：把训练引擎从 StSRLSolver 换成 **sts_lightspeed**。剩余：补全卡/遗物/
    怪物穷尽覆盖 → 一轮实机 (CommunicationMod) 交叉验证 → 把 V8 训练接到 lightspeed
    后端（写 `LightspeedBackend` 实现 `GameBackend`、增量抹平 `v8/env.py` 透传债）。
  - 设计/计划细节见 `docs/superpowers/specs/2026-06-04-game-backend-test-harness-design.md`、
    `docs/superpowers/plans/2026-06-04-combat-probe-card-test-harness.md`、
    `docs/v8_card_test_set_scheme.md`。
- 2026-06-02: **奖励重设计转向坐实——「赢为主轴 + 评分当特征」证明有效，redesign_v6 现役**。
  - **证伪「评分当奖励」**：之前把「per-card 卡组评分增长」(`strength_reward`) 当奖励，
    归因证明严重跑偏——每打一场仗按伤害发奖占总奖励 73%、连死亡局都净赚 +485，
    模型被训成「多打架」而非「赢」。v1-v4 四条线（锚维 / 锚维续 / 协同 / 对齐）都卡在
    打过一幕 boss ~7-13%，低于旧线 32%。
  - **最终方向（用户拍板）**：奖励改成「真实进度 / 通关为主轴」（每层 +3 / 过 boss +25 /
    通关 +100 / 小胜负 / 存活），删 `strength_reward`（commit `60ea0b9`）；**卡组评分降级为
    模型输入特征**（deck_strength 5 维喂模型，commit `6922a9e`）；加「健康到达 boss」高效奖励
    （`W_BOSS_HP × 到 boss 血量比`，设计成奖好牌组、不逼模型 skip-all）。
  - **v5 验证成功**：这套（对齐奖励 + 评分特征 + 健康到达）**fresh 1024 局训到打过一幕 boss 30% /
    到达 60%**——清白对齐（无 exploit）、只 1024 局（旧线要 4900 才到 32%）、探索度还活着
    （熵 0.21）、牌组健康（~15）。**整个转向坐实。**
  - **前沿**：模型能稳定打一幕，二三幕几乎进不去（二幕碰过 3%、通关 0）。
  - **现役 `v8_ppo_redesign_v6`**（从 v5 final 续训到 2048 局，PID 3988，
    log `/tmp/v8_ppo_redesign_v6.log`），看拉长训练能否推过 30% / 往二幕渗透。
  - **关键诊断更正**：战斗随机桩是幽灵（隔离实验证战斗健康）、「24-50 回合」是计数 bug、
    真根因是奖励 reward hacking。详见
    [docs/v8_rl_diagnosis_2026-05-29.md](docs/v8_rl_diagnosis_2026-05-29.md)。
- 2026-06-01: **V8 RL 奖励再瘦身 + redesign_v5 现役**（已被 v5 完成 / v6 续训取代，留档）。redesign 时代分三段奖励演进：
  **v1-v3（2026-05-30~31）**用 5-29 redesign 设计的「卡组实力增长」(`strength_reward`)
  奖励训练（reward_mean ~200-360）；**v4（2026-06-01，commit `60ea0b9`）是删
  `strength_reward` 后第一批纯进度主轴**（reward_mean ~33-47，熵地板生效守住 0.186）；
  **v5（2026-06-01，commit `6922a9e`）再把 deck_strength 当模型输入特征 + 加「健康到达
  boss」防跳过奖励**。`strength_reward`（8×Δdeck 卡组实力增长）被证明是刷分元凶
  （死亡局也能净赚 +485），commit `60ea0b9` 彻底删除。
  现役 `reward.py`：step = 0.5×node + combat + floor_progress，
  各项 floor=+3/层、boss_beat=+25、boss_hp=+15×到达 boss 时血量比、combat win=+1/lose=-5、
  通关=+100、hp_terminal=+5×final_hp（hp 罚 / 回合罚 / damage_ratio 此前已删）。
  **战斗驱动 = StSRLSolver 手写启发式搜索**（小怪 50ms / elite 250ms / boss 500ms+10s cap），
  **不进 RL trajectory**；combat_net wrapper（随机权重桩）在 env reset 里已禁用
  （`neural_eval=None`）。BC combat head（`sts_models/v8_combat_head_v1.pt`，真权重）
  只作**选牌 model 的预热初始化加载**（loaded=45/skipped_shape=1/missing=5），
  **不冻结、训练里不被调用**（战斗走搜索）——是「加载了但既不冻结也不用」的死参数。
  redesign eval 概览：v1 a1 末 6.7% / 熵塌 0.060，v2 a1 末 13.3%（**ep768 act2 擦边
  3.3%，整个 redesign 时代唯一一次 act2 破零**）/ 熵 0.027，v3 崩 a1 末 0.00 / 熵彻底崩
  0.0096，v4 a1 末 10%（峰值 16.7%@ep640）/ 熵守住 0.186，**全程 won_game=0**。
  现役 `v8_ppo_redesign_v5`（**fresh start**，PID 85107，
  num_episodes=1024 batch_size=32 eval_freq=128 ckpt_freq=32 device=mps，当前约 ep64-81，
  零异常，log `/tmp/v8_ppo_redesign_v5.log`，output `sts_models/v8_ppo_redesign_v5`）。
  旧线 v40 已到 ep5120 结束存档（不再续）。
  详见 [docs/v8_rl_diagnosis_2026-05-29.md](docs/v8_rl_diagnosis_2026-05-29.md) 和
  [docs/v8_rl_fix_plan_2026-05-29.md](docs/v8_rl_fix_plan_2026-05-29.md)。
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
- V6/V7 已归档（`docs/v6_architecture_review.md` 已不存在，历史回顾参见 git log / `archive/`）
- **维护规则**：每次切换大版本（如 V8→V9）或大阶段（如 V8 BC→V8 RL）必须同步更新
  「当前状态」和「活跃代码」段，刷新 last-verified 日期，与代码改动一起 commit。

### V8 RL 训练日志

完整训练历史（trial100 / long_v1-v4 / batch_v5-v22）已归档至 [docs/v8_training_log.md](docs/v8_training_log.md)。

CLAUDE.md 只保留最近活跃训练 + 还在用的知识。

**最近批次概览** (simulator-fix 后干净训练序列，a1_boss_beat eval 维度)：

| Batch | ep | a1_boss_beat | won_game | floor_mean | completed | 备注 |
|---|---|---|---|---|---|---|
| **v37** | **2944** | **16.7%** | **0.00** | **12.53** | **30/30** | **⚠ 旧 reward (simulation-based deck_evaluator) 训练的最后一批; act2_boss_beat=3.33% (历史第二次破零, 上次为 batch_v32); reached_boss=66.7% (持续高位); floor_mean=12.53 (相对高); boss reach 均衡 Slime=5 / Hexa=4 / Guardian=6; boss_kill 仍全 0** |
| **v38** | **3872** | **~22% (均值)** | **0.00** | **~10.8 (均值)** | **N/A** | **⚠ 新 reward (real-combat) 第一批; 跑 928/1024 ep 后 ep=3882 hang 在 Guardian boss combat 36+ min 立刻杀; 7 中间评估 (ep 3072→3840): a1 = 13/13/40/27/27/17/20%, 峰值 40% (ep=3328); 比旧 reward 后期均值 (~17.6%) 略高; 无 final eval; ckpt 至 ep=3872** |
| **v39** | **4896** | **32.1% (均值)** | **0.00** | **~10.4 (均值)** | **N/A (8 中间评估)** | **✅ 新 reward 第二批 1024 ep 完整跑完; 8 中间评估 (ep 3968→4864) a1 = 27/33/30/37/40/40/30/20%, 均值 32.1% 创新奖励时代单批最高; 峰值 40% 在 ep=4480/4608 连续两次 (旧奖励 peak 30%); reach 均值 69.4%; act2_boss_beat / won_game 全程 0; 0 Traceback / 0 hang (v38 Guardian 现象未复现)** |
| **v40** | **5120 (结束)** | **—** | **0.00** | **—** | **存档** | **real-combat reward 时代最后一批; 到 ep5120 结束存档, 不再续。redesign 转向后 batch_v* 序列终止 (reward / 指标定义已变, 与下方 redesign 不可直接对比)** |

> 以下 redesign 行为奖励重设计时代概览（reward 信号已变，与上面 batch_v* 不可直接比）。
> **完整 redesign 细节见 [docs/v8_training_log.md](docs/v8_training_log.md)**（由专门 agent 维护）。

| Redesign | ep | a1_boss_beat (末) | won_game | 熵 (末) | 备注 |
|---|---|---|---|---|---|
| **redesign_v1** | 512 | **6.7%** | **0.00** | **0.060 (塌)** | strength_reward 时代首条正式线 (fresh start); v2 从 ep384 续, 丢弃 v1 ep385-526 |
| **redesign_v2** | 768 | **13.3%** | **0.00** | **0.027** | strength_reward 续训; **ep768 act2 擦边 3.3% = 整个 redesign 时代唯一一次 act2 破零** |
| **redesign_v3** | 640 | **0.00** | **0.00** | **0.0096 (彻底崩)** | strength_reward 续训; 崩了, 熵塌缩最严重 |
| **redesign_v4** | 640+ | **10% (峰 16.7%@ep640)** | **0.00** | **0.186 (守住)** | **删 strength_reward 第一批纯进度主轴 (commit `60ea0b9`); reward_mean ~33-47, 熵地板生效** |
| **redesign_v5** | **1024 (完成)** | **30% (到达 60%)** | **0.00** | **0.21 (活)** | **✅ 转向坐实: 对齐奖励 + deck_strength 当模型输入特征 + 健康到达 boss (commit `6922a9e`), fresh 1024 局即到一幕 30% (旧线要 4900 才到 32%); 清白对齐无 exploit, 探索度活 (熵 0.21), 牌组健康 ~15; 二幕碰过 3%、通关 0** |
| **redesign_v6** | **2048 (训练中)** | **训练中** | **训练中** | **训练中** | **⏳ 现役: 从 v5 final 续训到 2048 局, 看拉长训练能否推过 30% / 往二幕渗透; PID 3988, log /tmp/v8_ppo_redesign_v6.log** |

完整 v15-v40 + redesign_v1~v6 细节 + audit findings 详见 [docs/v8_training_log.md](docs/v8_training_log.md)。

**[重大转向] 奖励重设计 → batch_v* 序列终止, redesign 系列 (2026-05-30 ~ 06-01)**：
上表 v37-v40 是 **real-combat reward (hp/回合 shaping) 时代的历史记录**，保留备查。
2026-05-29~30 系统排查定位真根因 = 这套 reward 引发 reward hacking（skip 卡, 通关恒 0），
且偏离用户原设计。v40 已到 ep5120 结束存档（不再续）。redesign 系列三段奖励演进：
- **v1-v3 (2026-05-30~31)**：5-29 redesign 设计的「卡组实力增长 (strength_reward) + 进度 +
  过 boss」奖励。三批 a1 末值 6.7%/13.3%/0.00%，熵反复塌缩（0.060/0.027/0.0096），
  **v2 ep768 act2 擦边 3.3% = 整个 redesign 时代唯一一次 act2 破零**，**全程 won_game=0**。
- **v4 (2026-06-01)**：发现 `strength_reward` 是刷分元凶（死亡局净赚 +485），
  commit `60ea0b9` **彻底删除**，改纯「进度 + 过 boss + 通关」主轴（见上方「活跃代码」
  `reward.py` 公式）——**删 strength_reward 后第一批纯进度**。a1 末 10%（峰值 16.7%@ep640），
  熵地板生效守住 0.186。`v8_ppo_redesign_v4` 从头训。
- **v5 (2026-06-02，✅ 转向坐实)**：commit `6922a9e` 把 deck_strength 当**模型输入特征** +
  加「健康到达 boss」防跳过奖励。`v8_ppo_redesign_v5` **fresh 1024 局即到一幕 30% / 到达 60%**——
  清白对齐无 exploit、只 1024 局（旧线要 4900 才到 32%）、探索度活（熵 0.21）、牌组健康（~15）。
  **整个奖励重设计转向坐实**：评分当奖励证伪 → 赢为主轴 + 评分降级为模型输入特征 + 健康到达 boss。
- **v6 (2026-06-02，现役)**：从 v5 final **续训到 2048 局**，看拉长训练能否推过 30% /
  往二幕渗透。PID 3988，log `/tmp/v8_ppo_redesign_v6.log`。前沿：稳定打一幕，二三幕几乎进不去
  （二幕碰过 3%、通关 0）。
redesign 系列与 batch_v* 不可直接对比（reward / 指标定义已变）。诊断与修复细节见
[docs/v8_rl_diagnosis_2026-05-29.md](docs/v8_rl_diagnosis_2026-05-29.md) 和
[docs/v8_rl_fix_plan_2026-05-29.md](docs/v8_rl_fix_plan_2026-05-29.md)。

**[重大变更] reward 信号源切换 (commit `8b9485a`, 2026-05-27)**:
batch_v37 是 simulation-based reward (deck_evaluator) 训练的最后一批; batch_v38
起切换为 **real-combat-based reward**。改动：
- 去掉 `deck_evaluator.evaluate_deck()` 模拟战评分 (post-step deck reward)
- 加真实战斗结束 reward:
  `(won ? +30 : -30) - 1.0*hp_lost - 0.5*turns + 5.0*damage_ratio`
- 信号源从 "干净环境模拟战能赢的卡" → "真实游戏当前 relic/血量/状态下能赢的卡"
- 训练速度预期 ~5x 提升 (smoke 11s/ep vs 旧 50s/ep), 1024 ep 预计 ~3h vs 旧
  ~14h

**预期波动**: batch_v38 从 v37 ckpt resume 但 reward 信号源完全变了, 模型需要适应;
前几个 eval 周期 (ep=3072/3200/3328) 可能 dip 然后回升。v38 后续多批次比较时,
v37 作为 sim-reward 最后一批的 baseline。

- **`batch_v39` 完成 (2026-05-28, 新奖励第二批 1024 ep 完整跑完, 创单批均值新高)**:
  从 v38 ep=3872 续训完 1024 ep (ep 3873→4896) + 8 次中间评估全部完成, PID 88369 干净退出。
  - **训练侧**: 1024 ep, 0 Traceback, 0 hang (v38 Guardian 深搜索 hang 现象未复现)
  - **8 次中间评估 a1_boss_beat** (ep 3968→4864): 27/33/30/37/**40/40**/30/20%,
    均值 **32.1%** (新奖励时代单批最高), 峰值 **40% 连续两次** (ep=4480/4608)
  - **reach 均值 69.4%**, floor_mean ~10.4
  - **act 2 boss / won_game**: 8 次评估全部 0% (新奖励还没把模型推过 a1 boss)
  - **历史 peak 对比**: 旧奖励时代 peak 30% (v22/v24/v28/v30/v36), 新奖励 peak 40%
    (v38 单次, v39 连续两次)
  - **Ckpt 状态**: `sts_models/v8_ppo_batch_v39/` 含 ep=3904…4896 全部 + final + summary
  - 详见 [docs/v8_training_log.md](docs/v8_training_log.md)

- **`batch_v40` 启动 (2026-05-28, 续训, 1024 ep 长跑)**：
  从 v39 final ckpt 续训, 同参再训 1024 ep。
  - **续训源**：`sts_models/v8_ppo_batch_v39/v8_ppo_final.pt` (episodes_done=4896)
  - **参数**：`num_episodes=5920 batch_size=32 ckpt_freq=32 eval_freq=128`
    (n_envs=1 serial, **增量训 1024 ep**, ep 4897→5920)
  - **PID**：`12627`（nohup）；log `/tmp/v8_ppo_batch_v40.log`；output
    `sts_models/v8_ppo_batch_v40/`；exit signal file `/tmp/v8_ppo_batch_v40.exit`（如有）
  - **启动校验**：`[resume] start_episode=4896, target=5920 (将增量训 1024 ep)`，
    0 Traceback
  - **观察重点**:
    1. a1_beat 均值能否守住 32% 区间 / peak 40% 是否持续
    2. act 2 boss / won_game 破零 (新奖励时代仍是 0)
    3. v38 Guardian 深搜索 hang 是否再现, v39 整批未现但仍属潜在风险
  - **预期总时长**: ~3h 训练 + final eval

接手 monitor 的检查清单（redesign_v6）：
- ckpt 落盘进度：`ls sts_models/v8_ppo_redesign_v6/`
- 训练是否还活：`ps -ef | grep v8_ppo_train.py | grep -v grep`（PID 3988）
- 异常监测：`grep -cE "\[guard_cap\]|MysteriousSphere event_phase=COMBAT_WON|Error|Traceback" /tmp/v8_ppo_redesign_v6.log`
- 当前 ep：`grep "\[heartbeat\]" /tmp/v8_ppo_redesign_v6.log | tail -1`
- 验证 resume 生效：`grep "\[resume\]" /tmp/v8_ppo_redesign_v6.log`（v6 从 v5 final 续训到 2048 局，应见 start_episode≈1024）
- 观察重点：a1_beat 能否守住 / 推过 30%、二幕渗透 (二幕碰过 3%、通关 0)、熵地板
- 验证新 reward 生效：heartbeat 应见 `eval_deck_calls=0`（模拟战评分已弃用）
- hang 监测 (v38 教训): 心跳间隔超 5min → 怀疑 boss combat 深搜索循环, 30min+ 无心跳立刻杀

### 已修复 bug
- **回合计数标错名 + beat_boss 单指标过粗** (2026-05-29 修, commit `24df9cc`):
  日志 / reward 里的 `turns` 实为**动作数**（一回合多动作时被重复计数），导致
  "战斗 24-50 回合"的假象，误导排查方向。已正名为动作数。同批把单一 `beat_boss`
  指标拆成 `a1_boss_killed` / `a2_boss_killed` / `won_game` 三层，避免"过 act1 boss"
  和"通关"混在一个数里。
- **[假根因已推翻] 战斗随机桩不是头号根因** (2026-05-29 隔离实验):
  排查初期一度怀疑"战斗被随机数指挥"是通关恒 0 的头号原因。隔离实验证明战斗搜索
  健康（小怪 8-12 真实回合），随机桩噪声仅**边缘影响**（已拔除）。真根因是 reward
  hacking（见上方「当前状态」）。留档防再走弯路，详见
  [docs/v8_rl_diagnosis_2026-05-29.md](docs/v8_rl_diagnosis_2026-05-29.md)。
- **StSRLSolver 问题卡 / 破碎王冠选卡数量重复加减**
  (2026-05-25 发现 + 同日修复, fork commit `1413d69f`):
  `reward_handler._generate_card_reward` 在调 `generate_card_rewards` 前预先按
  Question Card +1 / Busted Crown -2 算了一次 `num_cards`，但
  `generate_card_rewards` 内部又按 `has_*` flag 算了一次，重复加减。结果：
  单问题卡 5 张（应 4）/ 单破碎王冠 1 张（min clamp 兜住碰巧对）/ 同时拿 1
  张（应 2）。修复：删 `reward_handler` 里的预先计算，统一让 `generate_card_rewards`
  处理。268 个 reward / relic_card test 全过 + 4 case 端到端验证全过。**影响 ~1%
  选卡决策**（拿到 Question Card 或 Busted Crown 的局），训练数据轻污染但远小于
  NeowsLament / Mushrooms 这类系统性 bug。
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
- 2026-05-29 (redesign) eval 输出加 `entropy` + a1/a2/won_game 分层指标（commit `740358b`）；
  单一 `beat_boss` 拆为 `a1_boss_killed` / `a2_boss_killed` / `won_game`（commit `24df9cc`，
  见上方「已修复 bug」）。引擎新增 `CombatResult.card_log` / `game.last_combat_card_log`，
  env 存 `_last_combat_card_log`（per-card 细账，供 `card_scorer` 评分）。
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
  - `1413d69f`：reward_handler 问题卡 / 破碎王冠选卡数量重复加减
    (2026-05-25, ~1% 选卡决策受影响)
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

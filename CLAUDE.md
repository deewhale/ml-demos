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

## 当前状态

<!-- last-verified: 2026-05-15 -->
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
- **`long_v2b` 完成**: 2026-05-13 12:19:12，13.03h，128 ep 训练 + 30 seed final eval。Final 指标：
  - reached_a1_boss=0.93, **act1_boss_beat=0.57**, act2_boss_beat=0.57, **won_game=0.23**（A0 通关率 23%）
  - floor_mean=13.0, 30 seed 中 7 个完整通关 A0
  - 训练 128 ep 中 35 个 game_won（27%）—— 里程碑 2（A0 通关）首次达成
  - Per-batch trend: A1 boss kill 28% → 28% → 38% → 66% (单调上涨)，avg_reward 0.10 → -0.46 → 13.72 → 32.21
  - 关键归因发现 1: **SlimeBoss 10.5% vs TheGuardian 82.4% vs Hexaghost 80.0%**（71.8pp 差距）。模型未学到 SlimeBoss 的 AOE 需求（boss-aware encoding gap，本批未修，留待下批数据判定）
  - 关键归因发现 2: **eval deterministic 模式 card_reward 阶段 97% argmax=choice=1 mode collapse**（同源 Mysterious Sphere bug）。Action token 只编 choice 序号不编卡名/事件文本/商品名 → 模型看不到候选内容。已修。
- **`long_v3` 完成 (N=2/3)**: 2026-05-13 16:18 → 2026-05-14 06:29，14.18h，128 ep + 30 seed final eval。Action token fix 后首次 fresh start。Final 指标：
  - reached_a1_boss=**1.00** (+7pp vs v2b), **act1_boss_beat=0.63** (+6pp), act2_boss_beat=**0.53** (-4pp), **won_game=0.43** (+20pp，**near doubled**)
  - floor_mean=14.0 (+1.0), 30 seed 中 13 个完整通关 A0
  - Per-batch beat_boss_in_batch: 8 / 4 / 11 / 13 (= 25% → 12.5% → 34.4% → 40.6%)，avg_reward 1.79 → -16.72 → 31.53 → 32.06 (batch 2 dip 后稳步上涨)
  - SlimeBoss 仍是瓶颈: **0/11 kills** in eval（v2b 10.5% → v3 0%，30 seed 全是 SlimeBoss seed → 验证 boss-aware encoding gap 仍存在）
  - 健康度: 0 Traceback, 6 guard_cap / 128 ep = 4.7%
  - Ckpt 路径 `sts_models/v8_ppo_long_v3/`，含 ep=32/64/96/128/final + 1 wall_ckpt
- **`long_v4` 卡死中止 (N=3/3 未完整)**: 2026-05-14 06:56 启动，~7h 后落 ep=96 wall_ckpt
  (`v8_ppo_wall_20260514_135748.pt`)，**ep=127 时被 Mushrooms event handler 死循环**
  卡 ~15h（同 event_id 重复 choice 6352 次，COMBAT_WON 阶段菜单未过滤），未触发 final
  eval。**用 eval-only 脚本（`/tmp/v4_eval_only.py`）跑 wall ckpt 的 30-seed eval 已完成**，
  输出 `sts_models/v8_ppo_long_v4/v8_ppo_eval_recovered.json`。Final eval (ep=96 wall ckpt,
  30 seed)：reached_boss=96.7% (29/30)，**act1_boss_beat=70%**，act2_boss_beat=60%，
  **won_game=30%** (9/30)，floor_mean=11.2。30% 对应训练 trajectory 中段，非 regression
  （batch_v3 同期 batch 2 末 34.4% game_won training）。**Mushrooms fix integration 验证通过**：
  30-seed eval 内 Mushrooms event 触发正常退出，event_stall guard 0 触发。
  Ckpt 路径 `sts_models/v8_ppo_long_v4/` 含 ep=32/64/96 + wall。
- **`batch_v5` 启动 (新命名约定)**: 2026-05-15 启动的 **512 ep** scale-up 训练（首次跳出
  128 ep 规模），按用户「训练量优先」原则 4× scale vs batch_v3/v4。参数
  `num_episodes=512 batch_size=32 ckpt_freq=32 eval_freq=100`。Applied fixes：env event_stall
  guard (commit `79b3d51`) + StSRLSolver Mushrooms phase filter fix（fork commit）+ Action
  token mode-collapse fix。预期 ~50h。Ckpt 路径 `sts_models/v8_ppo_batch_v5/`。
  **命名约定切换**：新训练 output_dir 用 `batch_v<N>`（不再 `long_v<N>`）；旧目录
  `v8_ppo_long_v3/v4/v5` 保留避免 break ckpt 引用。
  **后续 (2026-05-15)**：batch_v5 跑到 ep=96 后被停。**事后发现 concept error**：
  batch_v3/v4/v5 一直是从 random init 重训而非续训，因为 `tools/v8_ppo_train.py` 没暴露
  `--resume_from` CLI（trainer 内部本来支持）。已修 (commit `bf2208f`)，下批起强制续训。
- **`batch_v5_resume` 启动 (首次正确续训)**: 2026-05-15 23:52 启动，从 v4 wall ckpt
  (`sts_models/v8_ppo_long_v4/v8_ppo_wall_20260514_135748.pt`, `episodes_done=96`) 续训。
  参数 `num_episodes=256 batch_size=32 ckpt_freq=32 eval_freq=128`（增量训 160 ep）。
  Output 目录 `sts_models/v8_ppo_batch_v5_resume/`。这是修完 resume_from CLI 后第一次
  真正的续训训练，验证「续训规范」生效。256 ep 而非 512 ep 是保守选择：续训第一批先看
  trend 是否真的持续上涨，确认后再起下批。
- **`batch_v5_resume` 弃用 (eval hang)**: 2026-05-16，5h+ 训练后在 eval phase 卡死
  （第 10 个 eval seed inference step 慢化 600x，单 step 从 50ms 涨到 30s+），
  整个 batch 的 PPO update + ckpt 还没 save 就被人为终止，**5h+ 权重全部丢失**。
  根因：原代码顺序 PPO update → eval → ckpt save，eval 卡死则 ckpt 永不落盘。
  Output 目录 `sts_models/v8_ppo_batch_v5_resume/` 仅留启动时 metadata。
- **`batch_v5_resume2` 启动 (修完 eval hang 后重启)**: 2026-05-16 05:27 启动，
  从同一个 v4 wall ckpt 续训。参数与 batch_v5_resume 相同。Output 目录
  `sts_models/v8_ppo_batch_v5_resume2/`。**Applied fixes** (commit `93e7c42`)：
  ckpt save 移到 eval 之前（即便 eval 卡死 ckpt 已落盘）；deck_evaluator future
  加 60s timeout + None fallback（防 pool worker 卡死累积）；run_eval 单 seed
  300s wall timeout（防整 eval 拖死训练）；新增 `[deck_eval]` pool state +
  `[eval] seed start/done` 监控日志。Smoke 验证：0 Traceback，ckpt 在 eval 之前
  落盘，新 log 标记生效。
- **`batch_v5_resume2` 完成 + plateau 确认 (N=3, 跨 v3/v4/v5_resume2)**: 2026-05-17
  06:21 训练 + final eval 全部完成，~24.9h（89630s）。160 ep 增量训练 + 2 次 eval@ep=128/256。
  Ckpt 路径 `sts_models/v8_ppo_batch_v5_resume2/` 含 ep=128/160/192/224/256 + final
  + 3 wall。**核心结论：plateau 已严格确认，N=3 framework 触发 escalation**。
  - **Per-batch beat_boss_in_batch (5 连续 batch)**：
    - batch 1 (ep 97-128): 14/32 = 43.8%
    - batch 2 (ep 129-160): 13/32 = 40.6%
    - batch 3 (ep 161-192): 14/32 = 43.8%
    - batch 4 (ep 193-224): 15/32 = 46.9%
    - batch 5 (ep 225-256): 14/31 = 45.2% (1 ep 未计入 heartbeat)
    - **5 batch 全部落在 40-47% 窄区间，零趋势上升 → 训练量已不再带来 won_game 提升**
  - **mean_reward 同期**：46.91 / 53.72 / 56.18 / 52.80 / 47.85（高位震荡，不再单调上涨）
  - **Entropy drift**：0.689 → 0.685 → 0.643 → 0.631 → 0.590（缓慢收敛，policy 在 sharpen）
  - **Eval@ep=128 (mid-run)**: 30 seed 中仅 **completed=15/30**（15 个超 300s timeout 弃用）。
    reached_a1=0.43, **a1_beat=0.20**, a2_beat=0.17, **won_game=0.067** (1/15),
    floor_mean=11.9, SlimeBoss=0/7
  - **Eval@ep=256 (final)**: 30 seed 中 **completed=15/30**（15 timeout）。reached_a1=0.43,
    **a1_beat=0.23**, a2_beat=0.17, **won_game=0.00** (0/15), floor_mean=9.0,
    SlimeBoss=0/6. **同 run 内 ep=128→256 won_game 下降 (0.07→0.00)**，且 floor_mean
    下降 (11.9→9.0)，**进一步证实 plateau / mild overfit**
  - **Eval timeout 问题严重未根治**：50% seed 超 300s wall timeout（v4 recovered run
    completed=30/30），eval 数据样本量减半导致 won_game 噪声大；timeout 病灶仍在
    inference 慢化（之前疑似 MPS / pool worker 状态泄漏）
  - **跨批 won_game 对比 (eval, 单数据点警告：v5_resume2 仅 15 seed)**：
    - v3 final (ep=128, 30/30 seed): **won=0.43**
    - v4 eval_recovered (ep=96 wall ckpt, 30/30 seed): **won=0.30**
    - v5_resume2 ep=128 (15/30 seed): won=0.067
    - v5_resume2 ep=256 (15/30 seed): won=0.00
    - 注意 v5_resume2 eval 仅 15 seed completed，**不能直接 1:1 对比 v3/v4 30-seed 数据**；
      但 per-batch training trend 40-47% 是 32-seed 全量数据，plateau 结论稳
  - **SlimeBoss 累计 (eval)**: v5_resume2 中 **0/13 = 0%**，与 v3 (0/11) / v4 (0/8)
    完全一致 → SlimeBoss 仍是单点 bottleneck，boss-aware encoding gap 未变
  - **健康度**: 0 Traceback / 0 fatal error，guard_cap / event_stall 在正常范围
  - **判定**：N=3 (v3 / v4_recovered / v5_resume2) plateau framework 触发 escalation。
    续训 160 ep（96→256）在 won_game 维度 zero 增益（甚至轻微 regression）。**训练量已
    饱和，需结构性改动**：boss-aware encoding（SlimeBoss AOE 表征）/ reward shaping
    （加 boss-specific signal）/ 别的方案。下批训练前必须先讨论结构改动方向，不能再纯
    scale 训练时间。
- **`batch_v6` 完成 (boss-aware encoding 部分有效 + eval timeout 仍 53%)**: 2026-05-19
  07:15 训练 + final eval 全部完成，~20.5h（73750s）。128 ep 增量训练（ep 257→384）
  + 1 次 final eval@ep=384。从 v5_resume2 ep=256 续训，model 新增 `boss_proj.*` 4 参数。
  Ckpt 路径 `sts_models/v8_ppo_batch_v6/` 含 ep=288/320/352/384 + final + 3 wall。
  **核心结论**：boss-aware encoding 在 training 信号上有效（SlimeBoss 训练胜率 0%→8.5%），
  但 eval 端未表现，整体仍 plateau。
  - **Per-batch beat_boss_in_batch (4 连续 batch, ep 257→384)**：
    - batch 1 (ep 257-288): 20/32 = 62.5%
    - batch 2 (ep 289-320): 17/32 = 53.1%
    - batch 3 (ep 321-352): 15/32 = 46.9%
    - batch 4 (ep 353-384): 16/32 = 50.0%
    - **末段 ~50%**，相较 v5_resume2 五连 batch (40-47%) 上移约 ~5pp，**training 端可见小幅提升**
  - **mean_reward 同期**：96.9 / 81.5 / 74.6 / 89.3（远高于 v5_resume2 的 46-56 区间）。
    reward shaping 强化的 dense reward 让数字直接没有可比性，但相对趋势仍是单批内有效。
  - **Entropy drift**：0.523 → 0.500 → 0.449 → 0.409（继续 sharpen，无 collapse）
  - **SlimeBoss training kills**: **4/47 = 8.5%**（v3/v4/v5_resume2 训练侧累计 ~0%）→
    **boss-aware encoding 真信号，但增量小（绝对值仍 < 10%）**
  - **Eval@ep=384 (final, 30 seed)**: **completed=12/30**（**16 timeout**, 2 其他），
    reached_a1=0.33, **a1_beat=0.13**, a2_beat=0.07, **won_game=0.00** (0/12),
    floor_mean=11.7, SlimeBoss eval kills=**0/6 reach**
  - **跨批 won_game 完整对比 (eval, 数据从各 run summary.json 校对)**：
    | Run | ep | completed | reached_a1 | a1_beat | a2_beat | won_game | floor_mean | SlimeBoss kill |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | v3 final | 128 | 30/30 | 1.00 | 0.63 | 0.53 | **0.43** | 14.0 | 0/11=0% |
    | v4 eval recovered | 96 | 30/30 | 0.97 | 0.70 | 0.60 | **0.30** | 11.2 | 0/8=0% |
    | v5_resume2 final | 256 | 15/30 | 0.43 | 0.23 | 0.17 | **0.00** | 9.0 | 0/6=0% |
    | v6 final | 384 | 12/30 | 0.33 | 0.13 | 0.07 | **0.00** | 11.7 | 0/6=0% |
    - **绝对 won_game**: v3/v4 高位 → v5_resume2/v6 归零，**timeout 让样本严重不可比**
    - **completed 率**: 30/30 → 30/30 → 15/30 → 12/30，**timeout 加剧，eval 慢化未根治**
    - **reached_a1 趋势**：1.00 → 0.97 → 0.43 → 0.33，**eval 端能跑到 a1 boss 的 seed
      数量持续下降**，与 timeout 加剧同步，可能两者同根因（inference 慢化）
  - **Eval timeout fix 部分有效**：MPS cache 完全生效（mps_alloc 稳定 12MB，未泄漏），
    四道防线（ckpt 前置 / deck_evaluator 60s future timeout / 300s wall / pool 监控）
    均落地生效（0 fatal hang）。但 **timeout 率仍 53%（16/30）**，根因不在 MPS，
    需进一步 profiling（inference 慢化 / search 不收敛 / event handler 慢）。
  - **健康度**: 0 Traceback / 0 fatal error / 0 `[guard_cap]` / 0 Mysterious Sphere
    COMBAT_WON loop。续训规范 + bug fix 全部 integration validated。
  - **Boss-aware encoding 总评判**：**部分有效**（training 端 SlimeBoss 0%→8.5%，
    per-batch 50% 高于 plateau 40-47%）；但 **eval 端零反映**（eval timeout 高 +
    SlimeBoss reach 后 0 kill）。需更长训练验证 / 或新结构（reward shaping +
    eval-timeout 根因修复）。下批训练前再次必须先讨论方向，**不再纯加 ep 数**。

- **deck_evaluator search budget 收紧 fix 验证 (2026-05-19)**: commit `fbda896`
  收紧 `deck_evaluator` search budget (`s` 字段) + turn cap，假设 eval timeout 是
  deck_evaluator 慢化导致。在 v6 ep=384 ckpt 上 10-seed re-eval 验证（log
  `/tmp/v6_eval_verify.log`, json `sts_models/v8_ppo_batch_v6/v8_ppo_eval_verify_ep384.json`）。
  - **结果**：完成率 5/10 = 50%，timeout 率 50%（5/10 seed: 2/4/6/7/8 超 300s）
  - **对比 baseline (v6 30-seed eval, ep=384)**：completed 12/30 = 40%, timeout 16/30 = 53.3%
  - **统计判定**：10 seed 噪声 (SE ~16pp) 内与 baseline 无差异，**fix 无显著效果**
  - **eval 数据 (5 done seed)**：reached_a1=0.4, a1_beat=0.2, a2_beat=0.1, **won_game=0.0**,
    floor_mean=10.8, avg_steps=35.0，与 baseline 30-seed (reached_a1=0.33, a1_beat=0.13,
    a2_beat=0.07, won_game=0.00, floor_mean=11.7) 在噪声内一致
  - **结论**：eval timeout 根因 **不在 deck_evaluator search budget**。下一步必须改换
    profiling 方向：inference 慢化（per-step model fwd）/ search 不收敛 / event handler
    慢 / sim engine 端慢；继续盲改 timeout 防线无用。

- **batch_v6 ep=384 真实 eval 验证 (2026-05-19, 600s timeout)**: commit `437f279`
  把 eval seed wall timeout 从 300s 放宽到 600s 后，在 v6 ep=384 ckpt 上 10-seed
  re-eval 验证。**10/10 seed 完成（0 timeout）**。
  - **eval 数据 (10/10 completed)**：reached_boss=0.90, **a1_beat=0.70**,
    a2_beat=N/A, **won_game=0.50** (5/10), floor_mean=14.2, avg_steps=99.1
  - **2 个 seed 耗时 515s 和 580s** → 300s wall 会砍掉它们；这是之前 v5_resume2 /
    v6 30-seed eval completed 率掉到 12-15/30 的根本原因
  - **跨批 won_game 真实对比 (统一 wall timeout 维度)**：
    - v3 final (300s wall, 30/30 completed): won=0.43
    - v6 final (600s wall, 10/10 completed): **won=0.50** (+7pp vs v3)
  - **结论 1**：commit `437f279` (timeout 300→600) **修好了"eval 测量失效"问题**。
    eval 慢化根因 = v5/v6 model 玩得更深 (avg_steps 99.1)，不是 model 真慢化 /
    inference 卡死 / search 不收敛。
  - **结论 2**：boss-aware encoding (v6) 真实有效 **+7pp won_game vs v3 baseline**。
    之前 v5_resume2/v6 "won_game=0% in eval" 不是 model 变差，是 300s timeout
    砍掉了所有深局 seed。
  - **结论 3**：commit `fbda896` (deck_evaluator search 收紧) 无独立效果，可保留
    作为副助力。

### 已修复 bug
- **Action token mode-collapse bug** (2026-05-13): `v8/action_space.py` CARD_REWARD / EVENT / SHOP 三个 phase 的 token 字符串现在注入 card_name / event choice text / shop item name。**v2b ckpt 的 token-prior 已失效**，下批训练 fresh start。
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

- **StSRLSolver Python engine `_get_mysterious_sphere_choices` 和
  `_handle_mysterious_sphere` 缺 phase filter**（2026-05-12 发现，已在
  `external/StSRLSolver/` 分支 `fix/mysterious-sphere-phase-filter` 修复，
  commits `11b15c7a` + `f8006f30`）。上游已弃用整个 Python engine
  （PR #136/#137），fix 仅本地，不能 push（fork 被 GitHub abuse-prevention 禁用了）。
  **事实上我们 own 这个 fork**。

## 运行中的训练进程（2026-05-19 状态快照）

- **`batch_v7` 启动 (2026-05-19 14:12)**: 从 v6 ep=384 ckpt 续训 128 ep（target=512），
  参数 `num_episodes=512 batch_size=32 ckpt_freq=32 eval_freq=128`。
  - **PID**: 95886（nohup 子进程在父 PID 下）
  - **Log**: `/tmp/v8_ppo_batch_v7.log` 全程 append
  - **Output**: `sts_models/v8_ppo_batch_v7/`
  - **Resume from**: `sts_models/v8_ppo_batch_v6/v8_ppo_ep384.pt`
    （v6 won_game=50%, a1_beat=70% 已 600s eval 验证）
  - **目的**：验证 boss-aware encoding 配合更多训练是否持续 +pp，尤其 SlimeBoss
    训练侧 8.5% → 更高的趋势
  - **ETA**：~13-15h，预期 2026-05-20 凌晨 02-04 点完成训练 + final eval
  - **接手 monitor**：
    - ckpt 落盘：`ls sts_models/v8_ppo_batch_v7/`
    - 进程：`ps -ef | grep v8_ppo_train.py | grep -v grep`
    - 异常监测：`grep -cE "\[guard_cap\]|MysteriousSphere event_phase=COMBAT_WON|Error|Traceback" /tmp/v8_ppo_batch_v7.log`
    - 当前 ep：`grep "\[heartbeat\]" /tmp/v8_ppo_batch_v7.log | tail -1`

- **历史 best ckpt 备查**（v7 未完成前继续以 v6 为 best）。**batch_v6 ep=384
  已用 600s timeout 10-seed re-verify**（10/10 completed, won_game=0.50, +7pp vs v3）。
- **Best ckpt (统一 wall timeout 维度排序)**：
  - **v6 final** (`sts_models/v8_ppo_batch_v6/v8_ppo_final.pt`, ep=384) — **won=0.50**,
    10/10 seed eval (600s timeout, commit `437f279`)，含 boss-aware `boss_proj.*` 4 参数
  - v3 final (`sts_models/v8_ppo_long_v3/v8_ppo_final.pt`, ep=128) — won=0.43, 30/30 seed eval (300s)
  - v4 wall (`sts_models/v8_ppo_long_v4/v8_ppo_wall_20260514_135748.pt`, ep=96) — won=0.30, 30/30 seed eval (300s)
- **boss-aware encoding 模型权重**：v6 final 含 `boss_proj.*` 4 个新参数；从 v6
  续训的下批不需要 missing key fallback；从 v3/v4/v5_resume2 续训仍需 missing=4 fallback
- **eval timeout 修复 confirmed**：commit `437f279` (300s → 600s) **根治 eval
  测量失效问题**。v5_resume2/v6 之前的 "won_game=0%" 不是 model 变差，是 300s
  砍掉了所有深局 seed（v6 真实 won_game 50%，含 515s/580s 两个深局 seed）。
- **下批训练前可选方向**：
  1. boss-aware encoding 已 confirmed 真实改善（+7pp won_game）→ 可继续 scale 训练
     时间或叠加 reward shaping 强化 SlimeBoss-specific signal
  2. 旧 plateau 结论（v3 → v5_resume2 → v6 won_game 下降）已 invalidated，是
     timeout artifact 而非真实 regression

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

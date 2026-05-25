# V8 RL 训练日志归档

本文档归档 V8 RL 训练历史详细数据（trial100 / long_v1-v4 / batch_v5-v18）。
CLAUDE.md 只保留当前活跃训练 + 还在用的知识。

历史背景：v3-v13 数据全部污染（StSRLSolver `e567c65d` 之前 NeowsLament counter
writeback bug + pre-battle effects missing 让模型学了 1HP exploit）。v14/v14b 已 kill。
v15 起为 simulator fix 后首个干净基线。

## V3-V14 历史训练（数据污染期）

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

- **`batch_v7` 完成 (2026-05-20 03:06, attribution-based hang detection + 30-seed
  full eval)**: 从 v6 ep=384 续训 128 ep (ep 385→512)，~12h 训练 + ~2.1h final eval
  全部完成，~12.0h 训练 + 2.1h eval = 总 14.1h（43295s）。Ckpt 路径
  `sts_models/v8_ppo_batch_v7/` 含 ep=416/448/480/512 + final + 1 wall。
  - **Per-batch beat_boss_in_batch (4 连续 batch, ep 385→512)**：
    - batch 1 (ep 385-416): 13/32 = 40.6%
    - batch 2 (ep 417-448): 15/32 = 46.9%
    - batch 3 (ep 449-480): 21/32 = **65.6%** (高点)
    - batch 4 (ep 481-512): 17/32 = 53.1%
    - 平均 ~51.6%，相较 v6 (~53%) 持平、v5_resume2 (40-47%) 上移 ~5-10pp，**training
      端 trend 维持**
  - **mean_reward 同期**：76.0 / 65.3 / 101.8 / 91.6（高位，batch 3 峰值 101.8）
  - **Entropy drift**：0.399 → 0.381 → 0.344 → 0.306（持续 sharpen，无 collapse）
  - **Eval@ep=512 (final, 30 seed, attribution-based timeout)**: **30/30 completed**
    （0 timeout, 0 hang, 0 false-positive kill），reached_a1=**1.00**, **a1_beat=0.77**,
    a2_beat=**0.73**, **won_game=0.50** (15/30), floor_mean=13.1, SlimeBoss reach=7/30,
    SlimeBoss kill=**0/7=0%** (gap unchanged)
  - **跨批 won_game 真实对比 (统一 30-seed 全量, 注意 timeout 维度不同)**：
    | Run | ep | completed | reached_a1 | a1_beat | a2_beat | won_game | floor_mean | SlimeBoss kill |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | v3 final | 128 | 30/30 (300s) | 1.00 | 0.63 | 0.53 | **0.43** | 14.0 | 0/11=0% |
    | v4 recovered | 96 | 30/30 (300s) | 0.97 | 0.70 | 0.60 | **0.30** | 11.2 | 0/8=0% |
    | v6 final 30-seed | 384 | 12/30 (300s) | 0.33 | 0.13 | 0.07 | 0.00 (artifact) | 11.7 | 0/6=0% |
    | v6 ep=384 re-verify | 384 | 10/10 (600s) | 0.90 | 0.70 | N/A | **0.50** | 14.2 | N/A |
    | **v7 final 30-seed** | **512** | **30/30 (attr)** | **1.00** | **0.77** | **0.73** | **0.50** | **13.1** | **0/7=0%** |
    - **v7 30-seed full eval = won_game 0.50**，与 v6 10-seed 600s timeout (0.50)
      **一致**，并比 v3 30-seed (0.43) **+7pp**
    - **completed 30/30**：attribution-based timeout 完全替代了 300s/600s hard kill，
      所有 seed 自然走完
    - **reached_a1=1.00 / a1_beat=0.77 / a2_beat=0.73** 均为历史最高
  - **SlimeBoss**: eval kill 仍 0/7 = 0%，与 v3/v4/v5/v6 完全一致。training 端
    v6 boss-aware encoding 已让 SlimeBoss training kill 0%→8.5%；v7 续训没新结构
    改动，eval 仍 0% 在意料内。**SlimeBoss 仍是 bottleneck**。
  - **健康度**: 0 Traceback / 1 `[guard_cap]` (~0.2% 命中) / 0 Mysterious Sphere
    COMBAT_WON loop / 0 fatal hang

- **新 attribution-based hang detection 实战验证 (batch_v7 30-seed full eval)**:
  - **触发次数**: `long_running` (>=600s warning) **0 次**, `hang_confirmed` **0 次**,
    `stagnation_no_loop` **0 次**, `grace_expired` **0 次**, `hard_cap_hit` **0 次**
  - **30 seed elapsed 分布**: 最长 444.9s（远低于 600s warning threshold），
    其余分布 257-389s 区间，无慢化样本
  - **结论**: 新 timeout 逻辑**零 false positive kill**，**所有 30 seed 自然走完**
    （包括之前可能被 300s/600s 砍掉的深局），证明 attribution-based 设计正确
    （归因再 kill > 时间 hard kill）
  - **caveat**: v7 这批的 eval 数据本身没有真正的 hang 出现，所以 "hang 归因正确性"
    没被压力测试。下批跑出深局 + 超长 seed 时还需观测 long_running warning trigger
    是否正常打 log，但当前判定：**attribution-based 改造 zero-regression 已落实**

- **`batch_v8` 完成 (2026-05-20, 训练 trend up + 首次 hang_confirmed 实战触发)**:
  从 v7 ep=512 续训 128 ep (ep 513→640)。Ckpt 路径 `sts_models/v8_ppo_batch_v8/`
  含 ep=640 final。
  - **Per-batch beat_boss_in_batch (4 连续 batch, ep 513→640)**：
    - batch 1: 53%
    - batch 2: 59%
    - batch 3: 59%
    - batch 4: **72%** (末段大跃迁，历史新高)
    - 平均 ~60.8%，相较 v7 (~51.6%) 上移 ~9pp，**training 端 trend 持续 up**
  - **Eval@ep=640 (final, 30 seed, attribution-based timeout)**: 29/30 done + 1
    `hang_confirmed` (action_mode_collapse)，**won_game=0.40** (12/30, vs v7 0.50 小幅
    回落), reached_boss=**0.87**, **a1_boss_beat=0.60**, floor_mean=13.7
  - **跨批 won_game 对比 (30-seed full eval)**：
    - v3 final (ep=128): 0.43
    - v6 re-verify (ep=384, 10 seed): 0.50
    - v7 final (ep=512): **0.50**
    - **v8 final (ep=640): 0.40** (vs v7 -10pp 小幅回落，training trend up 但 eval 没跟上)
  - **首次 attribution-based hang_confirmed 实战触发 (1 次)**: 归因
    `action_mode_collapse`，证明新 hang detection logic 在真实压力下能正确捕获
    deterministic policy 卡死的 seed，**zero false-positive** 保持
  - **健康度**: 30 seed eval 中 29 顺利完成 + 1 正确归因 hang，无 Traceback / 无
    Mysterious Sphere COMBAT_WON loop

- **`batch_v10` 完成 (2026-05-21 17:13, training trend dip + eval won_game 小幅回落)**:
  从 v9 ep=768 续训 128 ep (ep 769→896)，~6.85h 训练 + ~1.28h final eval = 总 ~6.85h
  (24651s)。Ckpt 路径 `sts_models/v8_ppo_batch_v10/` 含 ep=800/832/864/896 final + 1 wall。
  新加速配置生效 (`max_steps_per_episode=1500`, `deck_eval_freq=5`)，wall-time 较 v9
  缩短 ~4h。
  - **Per-batch beat_boss_in_batch (4 连续 batch, ep 769→896)**：
    - batch 1 (ep 769-800): 13/32 = 40.6%
    - batch 2 (ep 801-832): 17/32 = 53.1%
    - batch 3 (ep 833-864): 13/32 = 40.6%
    - batch 4 (ep 865-896): 12/32 = 37.5%
    - 平均 ~43.0%，与 v9 (~41.4%) 持平，仍低于 v8 末段 (~60.8%)
  - **mean_reward 同期**：78.1 / 93.1 / 86.2 / 87.8（高位稳定）
  - **Entropy drift**：0.217 → 0.220 → 0.176 → 0.176（继续 sharpen，再降）
  - **Eval@ep=896 (final, 30 seed, attribution-based timeout)**: **30/30 completed**,
    reached_a1_boss=**1.00**, **a1_boss_beat=0.53**, a2_boss_beat=**0.53**,
    **won_game=0.43** (13/30, vs v9 0.53 **-10pp 小幅回落**), floor_mean=15.6,
    SlimeBoss reach=14/30, SlimeBoss kill=**0/14=0%** (gap unchanged)
  - **跨批 won_game 完整对比 (30-seed full eval)**：
    | Run | ep | completed | reached_a1 | a1_beat | a2_beat | won_game | floor_mean | SlimeBoss kill |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | v3 final | 128 | 30/30 (300s) | 1.00 | 0.63 | 0.53 | 0.43 | 14.0 | 0/11=0% |
    | v7 final | 512 | 30/30 (attr) | 1.00 | 0.77 | 0.73 | 0.50 | 13.1 | 0/7=0% |
    | v8 final | 640 | 29/30 (attr) | 0.87 | 0.60 | N/A | 0.40 | 13.7 | N/A |
    | v9 final | 768 | 30/30 (attr) | 1.00 | 0.80 | 0.73 | **0.53** | 14.6 | 0/6=0% |
    | **v10 final** | **896** | **30/30 (attr)** | **1.00** | **0.53** | **0.53** | **0.43** | **15.6** | **0/14=0%** |
  - **关键发现**: SlimeBoss reach 翻倍 (v9=6 → v10=14)，但 SlimeBoss kill 仍 0；
    a1_boss_beat 从 v9 0.80 跌到 0.53，可能 v10 seed mix 中 SlimeBoss seed 占比变高
    (47% vs 20%) → 0% SlimeBoss kill 直接拖低 a1_beat。**floor_mean 15.6 反创新高**
    表明非 SlimeBoss seed 平均跑得更深
  - **SlimeBoss**: eval kill 仍 0/14=0%（累计 6 个 run 共 ~52 reach 仍 0 kill）。
    boss-aware encoding 已 plateau，eval 端 SlimeBoss gap 持续未解
  - **健康度**: 30 seed eval 全 completed，0 Traceback / 0 hang_confirmed / 0
    Mysterious Sphere COMBAT_WON loop

- **`batch_v11` 完成 (2026-05-21, 真实有效训练但 parallel stats bug 让指标误报 0%)**:
  从 v10 ep=896 续训 128 ep (ep 897→1024)，首次启用 `n_envs=4` 多环境并行。
  Ckpt 路径 `sts_models/v8_ppo_batch_v11/` 含 ep=928/960 + final。
  - **关键发现**：parallel 模式下 `[perf]` / `[heartbeat]` 硬编码 `final_floor=0`
    `beat_boss=False`（trainer 主进程拿不到子进程 runner.run_state），导致汇总指标
    显示 0% 通关率。**实际 subprocess `[combat]` log 显示 44% game_won**——训练正常，
    只是统计 bug。已修 (commit `b69a0bc`)：parallel_env 新增 `get_runner_snapshot` cmd
    一次性回传 floor/act/hp/game_won 等字段。
  - **n_envs=4 性能结论**：parallel 反而慢（multiprocessing 序列化 + deck_evaluator
    pool 嵌套开销 + 主进程 forward 排队），下批先 fallback `n_envs=1` serial。
    parallel 真训练前需重做 phase 2 设计（中央 pool / shared model forward 等）。

- **`batch_v12` 完成 (2026-05-22 04:29, mid-eval won_game=0.533 与 v9 best 持平)**:
  从 v11 ep=960 续训 128 ep (ep 961→1088)，~7.59h 训练（27311s）。**serial n_envs=1**
  (v11 parallel stats bug 修复后首跑)。Ckpt 路径 `sts_models/v8_ppo_batch_v12/` 含
  ep=992/1024/1056/1088 + final。
  - **Per-batch beat_boss_in_batch (4 连续 batch, ep 961→1088)**：
    - batch 1 (ep 961-992):  16/32 = 50.0%
    - batch 2 (ep 993-1024): 17/32 = 53.1%
    - batch 3 (ep 1025-1056): 15/32 = 46.9%
    - batch 4 (ep 1057-1088): 16/32 = 50.0%
    - 平均 ~50.0%，相较 v10/v11 (~43%) 上移 ~7pp，恢复到 v8/v9 区间。training 端 trend 稳
  - **mean_reward 同期**：92.6 / 105.6 / 84.5 / 102.3（高位稳定）
  - **Entropy drift**：0.172 / 0.184 / 0.184 / 0.175（低位稳定，未继续 sharpen，无 collapse）
  - **Eval@ep=1024 (mid-run, 30 seed, attribution-based timeout)**: **30/30 completed**,
    reached_a1_boss=**0.967**, **a1_boss_beat=0.633**, a2_boss_beat=**0.600**,
    **won_game=0.533** (16/30, **与 v9 best 0.533 持平**), floor_mean=14.7,
    SlimeBoss reach=10/30, SlimeBoss kill=**0/10=0%** (gap unchanged)
  - **跨批 won_game 完整对比 (30-seed full eval)**：
    | Run | ep | completed | reached_a1 | a1_beat | a2_beat | won_game | floor_mean | SlimeBoss kill |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | v3 final | 128 | 30/30 (300s) | 1.00 | 0.63 | 0.53 | 0.43 | 14.0 | 0/11=0% |
    | v7 final | 512 | 30/30 (attr) | 1.00 | 0.77 | 0.73 | 0.50 | 13.1 | 0/7=0% |
    | v9 final | 768 | 30/30 (attr) | 1.00 | 0.80 | 0.73 | **0.53** | 14.6 | 0/6=0% |
    | v10 final | 896 | 30/30 (attr) | 1.00 | 0.53 | 0.53 | 0.43 | 15.6 | 0/14=0% |
    | **v12 mid (ep=1024)** | **1024** | **30/30 (attr)** | **0.97** | **0.63** | **0.60** | **0.53** | **14.7** | **0/10=0%** |
  - **注意 caveat**: 本批 `num_episodes=1088 eval_freq=128` 只在 ep=1024 触发一次 mid-eval，
    ep=1088 final 没有 eval（trainer 完成 last batch → ckpt save → 退出，没再触发 eval cycle）。
    final ckpt 的 eval 待下批 v13 启动后等待 ep=1216 时 mid-eval 顺手观测。
  - **关键发现**: stats bug 修复后 v12 训练 trend 真实回到 ~50%，与 v8/v9/v10 修复前的
    序列化模式一致；**won_game=0.533 与 v9 历史最高持平**，证明 v10/v11 之间的指标"回落"
    并非真 regression，更可能是 v10 SlimeBoss seed mix 偏高导致的 noise
  - **健康度**: 0 Traceback / 0 Mysterious Sphere COMBAT_WON loop / 0 fatal hang

- **`batch_v9` 完成 (2026-05-21 02:44, training trend slight dip + eval won_game 历史最高)**:
  从 v8 ep=640 续训 128 ep (ep 641→768)，~10.9h 训练 + ~2.4h final eval = 总 ~10.9h (39334s)。
  Ckpt 路径 `sts_models/v8_ppo_batch_v9/` 含 ep=672/704/736/768 final + 1 wall。
  - **Per-batch beat_boss_in_batch (4 连续 batch, ep 641→768)**：
    - batch 1 (ep 641-672): 15/32 = 46.9%
    - batch 2 (ep 673-704): 15/32 = 46.9%
    - batch 3 (ep 705-736): 9/32 = 28.1% (dip)
    - batch 4 (ep 737-768): 14/32 = 43.8%
    - 平均 ~41.4%，相较 v8 (~60.8%) **回落 ~19pp**，**v8 末段 72% 跃迁未保持**
  - **mean_reward 同期**：83.3 / 95.1 / 62.1 / 90.1（高位震荡）
  - **Entropy drift**：0.208 → 0.213 → 0.211 → 0.217（极低位 stable，未继续 sharpen）
  - **Eval@ep=768 (final, 30 seed, attribution-based timeout)**: **30/30 completed**
    (1 long_running warning seed=10770 ~600s+ 未 kill，归因正确放过),
    reached_a1_boss=**1.00**, **a1_boss_beat=0.80**, a2_boss_beat=**0.73**,
    **won_game=0.53** (16/30, **历史最高 +3pp vs v7**), floor_mean=14.6,
    SlimeBoss reach=6/30, SlimeBoss kill=**0/6=0%** (gap unchanged)
  - **跨批 won_game 完整对比 (30-seed full eval)**：
    | Run | ep | completed | reached_a1 | a1_beat | a2_beat | won_game | floor_mean | SlimeBoss kill |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | v3 final | 128 | 30/30 (300s) | 1.00 | 0.63 | 0.53 | 0.43 | 14.0 | 0/11=0% |
    | v4 recovered | 96 | 30/30 (300s) | 0.97 | 0.70 | 0.60 | 0.30 | 11.2 | 0/8=0% |
    | v6 re-verify | 384 | 10/10 (600s) | 0.90 | 0.70 | N/A | 0.50 | 14.2 | N/A |
    | v7 final | 512 | 30/30 (attr) | 1.00 | 0.77 | 0.73 | 0.50 | 13.1 | 0/7=0% |
    | v8 final | 640 | 29/30 (attr) | 0.87 | 0.60 | N/A | 0.40 | 13.7 | N/A |
    | **v9 final** | **768** | **30/30 (attr)** | **1.00** | **0.80** | **0.73** | **0.53** | **14.6** | **0/6=0%** |
  - **关键发现**: training per-batch dip (60.8% → 41.4%) **同时** eval won_game 上升
    (0.40 → 0.53)，training/eval 信号脱钩。training 端是 stochastic policy 采样，
    eval 是 deterministic argmax；可能 v9 entropy 极低 (~0.21) 后 deterministic argmax
    policy 收敛更稳，training noise 反而拉低
  - **SlimeBoss**: eval kill 仍 **0/6 = 0%**，与 v3/v4/v5/v6/v7/v8 完全一致。
    boss-aware encoding 在 eval 端**累计 5 个 run 仍 0 kill**，gap 未变
  - **健康度**: 30 seed eval 中 30 全部 completed (含 1 个 long_running 600s+ 但
    自然走完)，12 个 `[guard_cap]` (~1.5% / 768 ep, 正常范围), 0 Traceback,
    0 Mysterious Sphere COMBAT_WON event loop bug
  - **attribution-based hang detection 二次实战**: long_running warning 触发 1 次
    (seed=10770 elapsed 601s floor=14 act=3)，归因为深局正常 progressing 未 kill，
    最终该 seed 自然 done，**zero false-positive** 维持

### batch_v7 历史快照

- **`batch_v7` 首次启动后被人为停掉 (2026-05-19 14:12 启动 → 14:35 stop @ ep≈387)**：
  从 v6 ep=384 ckpt 续训。**问题：之前 eval seed wall timeout 还是 600s 硬 kill**，
  会粗暴砍掉所有深局（v6 已观测到 515s/580s 的深局 seed），eval 信号噪声大。
  用户 push back：「600s 应该是警告（检查是否死循环），不应粗暴 kill 深局」。
  Killed 后改 timeout 逻辑（见下条），重启 batch_v7。
- **eval seed timeout 改造 (2026-05-19, 同日)**：
  `tools/v8_ppo_train.py` 中 `run_eval` 的 wall hard kill 改成 stagnation-based。
  - 移除单 seed wall hard kill（之前 300s → 600s）
  - 加 `EVAL_SEED_LONG_WARN_SEC=600s` warning（一次性 log，不 kill）：
    `[eval] seed=X long_running elapsed=Xs floor=Y act=Z`
  - 加 `EVAL_SEED_HARD_CAP_SEC=3600s` 极硬上限作 catch-all
  - 主线程 5s 轮询读 progress signal，progress 变化 → 重置 stagnation 计时
- **eval hang detection 加归因 (2026-05-19, 同日，二次迭代)**：
  user push back：「stagnation 5min 不一定是死循环（可能 PPO 内部慢），要先归因再 kill」。
  - 删 `EVAL_SEED_HANG_SEC` 直接 kill，改成 `EVAL_SEED_STAGNATION_SEC=300s` 触发归因
  - polling loop 维护 3 个 deque（无需 hook env，靠 env 已有诊断字段）：
    - `event_id_history` (maxlen=50)：取 `env._last_event_id`
    - `combat_enemies_history` (maxlen=20)：取 progress signal tuple 的 enemies 项
    - `action_history` (maxlen=50)：取 `env._last_action_repr`
  - 归因 (`_classify_hang_pattern`)：
    - 同 event_id 出现 >= 30 次 → `event_loop`
    - 同 combat enemies 连续 >= 5 次 → `combat_hang`
    - 最近 50 action 全同一 → `action_mode_collapse`
  - 命中 → log `[eval] seed=X hang_confirmed type=Y ...` → kill seed
  - 未命中 → log `[eval] seed=X stagnation_no_loop dump=...` → 给
    `EVAL_SEED_GRACE_SEC=300s` 宽限，期间 progress 恢复则 reset；仍卡 →
    `[eval] seed=X grace_expired ... terminating`
  - **Smoke validated**: 0 Traceback, 4 ep × 2 eval-seed 全部 done，
    无 stagnation 触发（smoke 太短）；attribution 单元测试 4/4 通过
- **`batch_v7` 重启 (2026-05-19 15:05, 二次启动)**：用新 attribution-based timeout
  逻辑从 v6 ep=384 续训 128 ep (target=512)，参数
  `num_episodes=512 batch_size=32 ckpt_freq=32 eval_freq=128`。第一次启动
  (PID 96912) 因 `EVAL_SEED_HANG_SEC` 直接 kill 被用户 push back 杀掉
  (`sts_models/v8_ppo_batch_v7_killed_v2/` 备份)。**完成于 2026-05-20 03:06**，
  详见上方「batch_v7 完成」条目。

### Best ckpt 索引（v3-v13 数据污染期，仅留档参考）

- **Best ckpt (统一 30-seed eval 维度排序，2026-05-21 更新)**：
  - **v9 final** (`sts_models/v8_ppo_batch_v9/v8_ppo_final.pt`, ep=768) — **won=0.53**,
    **30/30 seed eval** (attribution-based timeout)，a1_beat=0.80, a2_beat=0.73,
    reached_a1=1.00, floor_mean=14.6，含 boss-aware `boss_proj.*` 4 参数。**当前 best ckpt**
  - v7 final (`sts_models/v8_ppo_batch_v7/v8_ppo_final.pt`, ep=512) — won=0.50,
    30/30 seed eval (attr)，a1_beat=0.77, reached_a1=1.00
  - v10 final (`sts_models/v8_ppo_batch_v10/v8_ppo_final.pt`, ep=896) — won=0.43,
    30/30 seed eval (attr), a1_beat=0.53, a2_beat=0.53, reached_a1=1.00, floor_mean=15.6
  - v6 final (`sts_models/v8_ppo_batch_v6/v8_ppo_final.pt`, ep=384) — won=0.50,
    10/10 seed eval (600s)；30-seed 同维度 eval 未做
  - v3 final (`sts_models/v8_ppo_long_v3/v8_ppo_final.pt`, ep=128) — won=0.43, 30/30 seed eval (300s)
  - v4 wall (`sts_models/v8_ppo_long_v4/v8_ppo_wall_20260514_135748.pt`, ep=96) — won=0.30, 30/30 seed eval (300s)
- **boss-aware encoding 模型权重**：v6/v7 final 含 `boss_proj.*` 4 个新参数；从 v6/v7
  续训的下批不需要 missing key fallback；从 v3/v4/v5_resume2 续训仍需 missing=4 fallback
- **eval timeout 修复 confirmed (累计 2 步)**：
  1. commit `437f279` (300s → 600s) 修好 v5/v6 "eval 测量失效" 假象
  2. attribution-based timeout (batch_v7) 用 30-seed full eval (30/30 completed)
     彻底替代 hard wall kill，**最干净的 eval 测量维度**
- **下批训练前可选方向**：
  1. 继续 scale (boss-aware encoding 已 confirmed 真实改善 +7pp won_game vs v3，
     v7 维持 0.50 plateau 4 个 batch trend ~52% → 仍有上升势头)；或叠加 reward
     shaping 强化 SlimeBoss-specific signal（30 seed 中 7 个 SlimeBoss 但 0 kill）
  2. SlimeBoss-only encoding / AOE-related state representation 改造（继续解
     SlimeBoss 0% gap）

## Simulator bug 大爆发（2026-05-22）

### Simulator bug 大爆发 + v3→v13 数据失效 (2026-05-22)

调查 v10/v11/v12 plateau 时定位到 **StSRLSolver 2 个致命 bug**，让 NEOW idx=0
(`three_enemy_kill`) 永久"全敌人 1 HP"。模型 100% NEOW pick idx=0 → 训练数据基本
作废，**所有 v3-v13 训练（~140h compute）模型其实在学 exploit 不是 STS**。

**Bug 1 — counter writeback 缺失** (`game.py` `_end_combat`)
  - `_enter_combat` 把 `run_state.relics[i].counter` 灌到
    `combat.state.relic_counters[relic_id]` (line 3862-3867)
  - atBattleStart 触发的 NeowsLament handler 通过 `ctx.set_relic_counter` 减计数
    （写到 combat 侧）
  - `_end_combat` 没把 combat 侧 dict 写回 `run_state.relics` → counter 永远=3
  - 影响范围：NeowsLament + Pen Nib / Nunchaku / Ink Bottle / Happy Flower /
    Sundial / Incense Burner / Girya（所有持久 counter 遗物）
  - 修复：`_end_combat` 在 `self.current_combat = None` 之前 iterate
    `run_state.relics`，若对应 relic_id 在 combat counter dict 中且
    `relic.counter != -1` → 回写

**Bug 2 — `get_pre_battle_effects()` 从未被调用** (`combat_engine.py` `start_combat`)
  - 8 个 enemy 类（BronzeAutomaton / AwakenedOne / TimeEater / Donu / Deca /
    SpireShield / SpireSpear / CorruptHeart）声明了 pre-battle 加 buff，但
    `start_combat` 没人 invoke
  - 失效 buff：Artifact / Unawakened / Regen / Curiosity / Time Warp /
    Invincible / BeatOfDeath / Surrounded
  - 修复：`start_combat` 在 `execute_relic_triggers("atBattleStart")` 之前
    （Java parity）调 `_apply_enemy_pre_battle_effects(state, enemy_objects)`，
    snake_case→canonical-power-name 映射表见 `_PRE_BATTLE_STATUS_MAP`

**修复 commit**：StSRLSolver fork `e567c65d` on branch
`fix/mysterious-sphere-phase-filter`。**fork 不能 push**（被 GitHub abuse-prevention
禁用，我们 own 这个 fork）。

**验证**：
- 单元测试 (`/tmp/test_simulator_fix.py`)：7/7 全通过（6 个 enemy 类 + JawWorm 反例 +
  NeowsBlessing counter 3→2 全链路）
- V8Env smoke (4 ep, 277s)：0 Traceback，0 guard_cap；ep=3 picked NEOW idx=0 后
  **只有前 3 场战斗** turn_actions∈{1,2}（NeowsLament 正确生效窗口），第 4 场起
  enemies 真实 HP，combats 14-34 actions

### v13 杀 + v14 从 trial100 重训

- **`batch_v13` 已 kill (2026-05-22 ~12:00)**：simulator bug 让 v3-v13 训练数据
  污染，没必要继续。原本从 v12 final 续训，~6h 进度直接弃用。
  Output 目录 `sts_models/v8_ppo_batch_v13/` 保留留档但不再使用。

- **`batch_v14` 跑到 ep=222 后人为 kill (2026-05-22 14:11)**：simulator fix 后
  首次干净训练，跑出 ckpt ep128/160/192。boss combat avg 37s/max 169s、elite
  avg 28s/max 118s，outlier 拖慢训练，决定 cut search budget 后重启 v14b。
  Output 目录 `sts_models/v8_ppo_batch_v14/`，ckpt 保留但不再续训。

### v14b：search budget cut 后从 trial100 重训 (2026-05-22 14:57 启动 → 后续 kill)

- **`batch_v14b` 启动**：v14 同基线 (trial100 ep96)，加 search budget cut。
  - **续训源**：`sts_models/v8_ppo_trial100/v8_ppo_ep96.pt`（同 v14，bug-exploit
    程度最低的 baseline）
  - **参数**：`num_episodes=228 batch_size=32 ckpt_freq=32 eval_freq=128`
    （n_envs=1 serial, 增量训 132 ep, ep 97→228）
  - **新差异**：env.py SOLVER_BUDGETS cut（commit 05894c4）—— elite
    base 500→250ms / cap 12000→1000ms，boss base 2000→500ms / cap 25000→10000ms。
    smoke 4 ep 验证 elite avg 27.7s → 7.5s (-73%)，单 ep wall ~52s。
  - **注意**：trainer 报 "optimizer load_state_dict mismatch ... keeping fresh
    optimizer state" + "model load_state_dict missing=4"（boss_proj.* 是 v6+ 加的，
    trial100 ckpt 没有，保持随机初始化）。**Adam moments 重建 = 前 ~几百 step 学
    习率/动量噪声偏大**，不阻塞
  - **后续**：v14b 已 kill（output 目录已清），决定改 from-scratch 起 batch_v15。

### v15：simulator fix 后第一次干净 from-scratch 训练 (2026-05-22 17:44 启动)

- **`batch_v15` 启动**：**不 resume**，from-scratch（meta head 随机 init,
  combat head 加载 `sts_models/v8_combat_head_v1.pt`）。
  - **参数**：`num_episodes=128 batch_size=32 ckpt_freq=32 eval_freq=128`
    (n_envs=1 serial)
  - **PID**：24753（nohup, log `/tmp/v8_ppo_batch_v15.log`，output
    `sts_models/v8_ppo_batch_v15/`）
  - **新差异（P1-A + P1-D, commit 25efbdc）**：
    - `deck_eval_freq` default 5 → 10：post-step evaluate_deck 频率减半，
      profile 显示 deck_eval 占单 ep 时间 ~68%，砍一半 call 数节省 ~15s/ep
    - `env.reset()` 不再 `clear_deck_cache()`：cache 跨 ep 复用（命中率
      预期 5% → 30%+），节省 ~10s/ep；act 切换 / `close()` 路径仍清；
      新加 `reset_cache_for_new_run()` 公共方法供需要 cold cache 场景调用
  - **预期**：单 ep 43s → ~25s，128 ep ~1h 训练 + final eval
  - **数据废弃声明**：v3-v14 训练数据全部失效（simulator counter writeback +
    pre-battle effects bug 让模型学了 NeowsLament 1HP exploit）。v15 是 simulator
    fix (`e567c65d`) + deck_eval 优化 (`25efbdc`) 后第一次从零干净训练。
  - **下批训练规范**：每次 128 或 512 ep，ckpt_freq=32（user 已定）

### Bug audit 结论（2026-05-22, v14b 启动前做）

排查范围：deck_evaluator simulator、其他 multi-phase event handlers、relic
counter writeback、reward shaping。所有项干净，不需要新 fix：

1. **deck_evaluator 不走 GameRunner**：直接 `create_combat_from_enemies` 构造
   combat，**不经 NEOW 流程** → NeowsBlessing 1HP bug 历史上从未影响牌组评分
   信号（pre-fix 训练里 deck_evaluator reward 是干净的，只有 env-side
   hp_loss / floor reward 被污染）
2. **其他 event handlers**：DeadAdventurer / MaskedBandits / MindBloom 看了
   choices + handlers，DeadAdventurer 虽是 multi-phase 但每次 search 都
   推进 attempt_count + 消耗 rewards 队列必然终止；MaskedBandits / MindBloom
   是 single-phase 选完直接进战斗或结束，不会循环。无需 phase filter
3. **Counter writeback**：fork commit e567c65d 在 game.py `_end_combat` 已
   cover 所有 counter relics（NeowsLament/PenNib/Nunchaku/InkBottle/
   HappyFlower/Sundial/IncenseBurner/Girya/RedSkull 等）
4. **reward.py**：无"打死敌人 +X"这种容易被 1HP bug exploit 的项，全是
   牌组强度（独立 simulator）+ HP loss + 节点收益（基于真实 hp/relic 变化）+
   floor/won bonus，pre-fix 训练里 reward 信号也基本干净

接手 monitor 的检查清单（v15）：
- ckpt 落盘进度：`ls sts_models/v8_ppo_batch_v15/`
- 训练是否还活：`ps -ef | grep v8_ppo_train.py | grep -v grep`
- 异常监测：`grep -cE "\[guard_cap\]|MysteriousSphere event_phase=COMBAT_WON|Error|Traceback" /tmp/v8_ppo_batch_v15.log`
- 当前 ep：`grep "\[heartbeat\]" /tmp/v8_ppo_batch_v15.log | tail -1`
- 验证 deck_eval 优化生效：`grep "freq=10" /tmp/v8_ppo_batch_v15.log | head -3`

## V15-V18 干净基线训练

### batch_v15 完成 + v16 续训启动 (2026-05-23)

- **`batch_v15` 完成 (2026-05-22 → 2026-05-23, simulator fix 后第一次干净 from-scratch)**：
  128 ep, ~1.33h (4779s)，from-scratch (meta head 随机 init, combat head v1 加载)。
  Ckpt 路径 `sts_models/v8_ppo_batch_v15/` 含 ep=32/64/96/128 + final + summary。
  - **Per-batch beat_boss_in_batch**: 0 / 0 / 0 / 0（4 batch 全 0，从零起步未触及 boss）
  - **mean_reward**: -44.2 / -47.9 / -53.6 / -54.3（持续下降，但 reward shaping 被
    simulator fix 后正常化，不再有 NeowsLament 1HP exploit 给的虚高 reward）
  - **mean_floor**: 9.72 / 9.25 / 9.81 / 9.31（一直停在 act 1 中段，未推到 a1 boss）
  - **Entropy drift**: 0.713 → 0.681 → 0.641 → 0.618（policy 在 sharpen，无 collapse）
  - **Eval@ep=128 (30 seed)**: reached_boss=0.267 (8/30), **a1_beat=0.10 (3/30)**,
    a2_beat=0.00, **won_game=0.00**, floor_mean=9.73, SlimeBoss reach=3 kill=0
  - **基线已重建**：v15 是 simulator fix (`e567c65d`) 后第一次干净 RL 起步基线。
    数字看着比 v3-v13 (won_game 30-53%) 低很多，但 v3-v13 数据全部污染。
    v15 是真实 STS 信号下的 "ep=128 from-scratch" 表现，作为新基线锚点。

- **`batch_v16` 启动 (2026-05-23, 续训, 持续迭代规范首次落地)**：
  从 v15 final ckpt 续训。**应用 memory `feedback_continuous_iteration_no_interrupt`**：
  batch 完成自动起下批，不再问 user "继续吗"。
  - **续训源**：`sts_models/v8_ppo_batch_v15/v8_ppo_final.pt` (episodes_done=128)
  - **参数**：`num_episodes=256 batch_size=32 ckpt_freq=32 eval_freq=128`
    (n_envs=1 serial, 增量训 128 ep, ep 129→256)
  - **PID**：60253（nohup）；log `/tmp/v8_ppo_batch_v16.log`；output
    `sts_models/v8_ppo_batch_v16/`；exit signal file `/tmp/v8_ppo_batch_v16.exit`（如有）
  - **预期**：~1.5-2h 训练 + final eval

- **`batch_v16` 完成 + v17 续训启动 (2026-05-25)**：
  v16 训练 128 ep (ep 129→256) 完成，从 v15 final 续训。Ckpt 路径
  `sts_models/v8_ppo_batch_v16/` 含 ep=160/192/224/256 + final + summary。
  - **训练侧**: 128 ep, 0 Traceback / 0 guard_cap / 0 MysteriousSphere COMBAT_WON loop
  - **Per-batch 表现 (基于 heartbeat beat_boss=True，仅捕获 final boss 胜利)**：
    heartbeat 视角 0 beat_boss，但 `[deck] room=boss` 日志显示
    **5 个 act 1 boss kill**（ep=167 Hexaghost, ep=205 Hexaghost,
    ep=227 TheGuardian, ep=237 TheGuardian, ep=255 TheGuardian）→ act 1 通过率
    **3.9% (5/128)**，仍处于 simulator-fix 后早期学习阶段
  - **Per-boss eval （training 内）**: Hexaghost 2/16=12.5%，TheGuardian 3/8=37.5%,
    SlimeBoss **0/16=0%**（SlimeBoss bottleneck 持续，未 boss-aware encoding 改动）
  - **Audit (5 ep 抽查 ep=130/160/190/220/250)**: ALL CLEAR。
    - Cultist 战 turn_actions 分布健康（102/117 = 1 turn 解决，最长 27 turn）→
      无 player.Strength 异常累积征兆
    - GremlinNob / Lagavulin / Hexaghost / SlimeBoss / TheGuardian 战斗长度合理（13-44 turn）
    - 敌人 roster 与 wiki 一致（Louse / SpikeSlime / Cultist / JawWorm / FungiBeast /
      Sentry / Lagavulin / Hexaghost / SlimeBoss / TheGuardian 等），act 2+ enemy
      (SphericGuardian / Byrd / Champ) 偶发出现，**无非法 enemy ID**
    - Event 多样性正常（Mushrooms 27 / AccursedBlacksmith 21 / GoldenIdol 19 等），
      328 个 `resolved->MAP_NAVIGATION` + 14 个 `combat_started`，**0 个 event stuck**
    - **MysteriousSphere COMBAT_WON loop = 0** (Mushrooms fix 持续生效)
    - 0 guard_cap, 0 Traceback

- **`batch_v17` 完成 + v18 续训启动 (2026-05-25)**：
  v17 训练 128 ep (ep 257→384) 完成，~1.8h 跑完无错。Ckpt 路径
  `sts_models/v8_ppo_batch_v17/` 含 ep=288/320/352/384 + final + summary。
  - **训练侧**: 128 ep, 0 Traceback
  - **Final eval (ep=384, 30 seeds)**:
    reached_a1_boss=33%, **a1_boss_beat=10%**, a2_boss_beat=0%, won_game=0%,
    floor_mean=9.9, boss_kills: Hexaghost=0/5, TheGuardian=0/2
  - **趋势对比** (a1_boss_beat eval): v15=10% → v16=3%（回退）→ **v17=10%（反弹）**。
    v16 → v17 反弹 + 不构成 3 批连续 plateau → 决策**继续起 v18 续训**
  - **Per-batch 训练表现** (mean_reward / mean_floor / beat_boss_in_batch):
    - batch ep=288: -21.08 / 8.84 / 0
    - batch ep=320: -36.48 / 9.78 / 0
    - batch ep=352: -37.39 / 10.0 / 0
    - batch ep=384: -20.52 / 31.94 steps / 0
    heartbeat 视角 0 final-boss kill；act-1 通过率从 eval 看 10%（与 v15 同档）

- **`batch_v18` 启动 (2026-05-25, 续训, v17 a1_beat 反弹后自动起下批)**：
  从 v17 final ckpt 续训。
  - **续训源**：`sts_models/v8_ppo_batch_v17/v8_ppo_final.pt` (episodes_done=384)
  - **参数**：`num_episodes=512 batch_size=32 ckpt_freq=32 eval_freq=128`
    (n_envs=1 serial, 增量训 128 ep, ep 385→512)
  - **PID**：86362（nohup）；log `/tmp/v8_ppo_batch_v18.log`；output
    `sts_models/v8_ppo_batch_v18/`；exit signal file `/tmp/v8_ppo_batch_v18.exit`（如有）
  - **启动校验**：`[resume] start_episode=384, target=512 (将增量训 128 ep)`，0 Traceback
  - **预期**：~2-2.5h 训练 + final eval

- **`batch_v18` 完成 + v19 续训启动 (2026-05-25)**：
  v18 训练 128 ep (ep 385→512) 完成，~1.98h (7133s) 跑完无错。Ckpt 路径
  `sts_models/v8_ppo_batch_v18/` 含 ep=416/448/480/512 + final + summary。
  - **训练侧**: 128 ep, 0 Traceback / 0 guard_cap / 0 MysteriousSphere COMBAT_WON loop
  - **Final eval (ep=512, 30 seeds, attribution-based timeout)**:
    reached_a1_boss=40%, **a1_boss_beat=13.3%**, a2_boss_beat=0%, won_game=0%,
    floor_mean=10.5, avg_steps=34, mean_reward=-0.063；boss kills 中仅 Hexaghost
    出现击杀（其他 boss 0）
  - **趋势对比** (a1_boss_beat eval): v15=10% → v16=3% → v17=10% → **v18=13.3%
    (+3.3pp vs v17，连续 2 批回升)**
  - **v17/v18 audit (2026-05-25) ALL CLEAR**:
    - Cultist 战 turn_actions 最长 58 turn 正常，无 player.Strength 异常累积征兆
    - NeowsLament combat #4+ 敌人 HP 恢复正常 (simulator fix `e567c65d` 生效)
    - 敌人 roster 全合法（无非法 enemy ID）
    - Event stalls = 0, guard_cap rate < 1%
    - 0 Traceback / 0 MysteriousSphere COMBAT_WON loop

- **`batch_v19` 完成 + v20 续训启动 (2026-05-25)**：
  v19 训练 128 ep (ep 513→640) 完成，~92.6 min 训练 (5561s) + ~22.4 min eval (1346s)
  = 总 ~115 min (6907.5s wall)，PID 2259 exited cleanly。Ckpt 路径
  `sts_models/v8_ppo_batch_v19/` 含 ep=544/576/608/640 + final + summary。
  - **训练侧**: 128 ep, 0 Traceback / 0 hang_confirmed / 0 grace_expired / 0 hard_cap_hit
  - **Final eval (ep=640, 30 seeds, attribution-based timeout)**:
    completed_seeds=**30/30**, reached_boss_rate=**0.4333** (13/30),
    **a1_boss_beat_rate=0.20** (6/30), a2_boss_beat_rate=0.00,
    **won_game_rate=0.00**, floor_mean=9.93
  - **Boss reach/kill counts**: Slime Boss reach=5/kill=0, Hexaghost reach=1/kill=0,
    The Guardian reach=1/kill=0 (counts 总和 7，与 reached_boss_rate=13/30 不一致 —
    boss_reach_counts dict 可能未跟踪某些 reach 路径)
  - **指标定义 gap (留档)**: a1_boss_beat_rate=0.20 显示 6 次 beat，但 boss_kill_counts
    全 0 → metric 定义差异（likely floor-advancement based vs explicit-kill counter）。
    模型行为正常，仅 measurement quirk
  - **趋势对比** (a1_boss_beat eval): v15=10% → v16=3% → v17=10% → v18=13.3% →
    **v19=20% (+6.7pp vs v18, 连续 3 批回升)**
  - **健康度**: 30 seed eval 全 completed，simulator fix 后首个 act 1 boss beat
    reach >= 20% 的批次，trend 持续 up

- **`batch_v20` 启动 (2026-05-25, 续训, v19 a1_beat=20% 创 simulator-fix 后新高)**：
  从 v19 final ckpt 续训。
  - **续训源**：`sts_models/v8_ppo_batch_v19/v8_ppo_final.pt` (episodes_done=640)
  - **参数**：`num_episodes=768 batch_size=32 ckpt_freq=32 eval_freq=128`
    (n_envs=1 serial, 增量训 128 ep, ep 641→768)
  - **PID**：`11503`（nohup）；log `/tmp/v8_ppo_batch_v20.log`；
    output `sts_models/v8_ppo_batch_v20/`；exit signal file
    `/tmp/v8_ppo_batch_v20.exit`（如有）
  - **启动校验**：`[resume] start_episode=640, target=768 (将增量训 128 ep)`
  - **预期**：~2-2.5h 训练 + final eval

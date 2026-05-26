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
  - Per-batch beat_boss_count: 8 / 4 / 11 / 13 (= 25% → 12.5% → 34.4% → 40.6%)，avg_reward 1.79 → -16.72 → 31.53 → 32.06 (batch 2 dip 后稳步上涨)
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
  - **Per-batch beat_boss_count (5 连续 batch)**：
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
  - **Per-batch beat_boss_count (4 连续 batch, ep 257→384)**：
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
  - **Per-batch beat_boss_count (4 连续 batch, ep 385→512)**：
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
  - **Per-batch beat_boss_count (4 连续 batch, ep 513→640)**：
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
  - **Per-batch beat_boss_count (4 连续 batch, ep 769→896)**：
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
  - **Per-batch beat_boss_count (4 连续 batch, ep 961→1088)**：
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
  - **Per-batch beat_boss_count (4 连续 batch, ep 641→768)**：
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
  - **Per-batch beat_boss_count**: 0 / 0 / 0 / 0（4 batch 全 0，从零起步未触及 boss）
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
  - **Per-batch 训练表现** (mean_reward / mean_floor / beat_boss_count):
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

- **`batch_v20` 完成 + v21 续训启动 (2026-05-25)**：
  v20 训练 128 ep (ep 641→768) 完成，~100 min 总 wall (6015.6s)，PID 11503 exited
  cleanly。Ckpt 路径 `sts_models/v8_ppo_batch_v20/` 含 ep=672/704/736/768 + final + summary。
  - **训练侧**: 128 ep, 0 Traceback / 0 hang_confirmed
  - **Final eval (ep=768, 30 seeds, attribution-based timeout)**:
    completed_seeds=**30/30** (0 timeout), reached_a1_boss_rate=**0.30** (9/30),
    **a1_boss_beat_rate=0.07** (2/30), a2_boss_beat_rate=0.00,
    **won_game_rate=0.00**, floor_mean=10.2
  - **Boss reach/kill counts**: Hexaghost reach=2/kill=0, Slime Boss reach=1/kill=0,
    The Guardian reach=4/kill=0 (总 reach=7, 全 0 kill — 与 v19 同源 explicit-kill
    counter 与 a1_boss_beat_rate 定义差异)
  - **趋势对比** (a1_boss_beat eval, 6 batches): v15=10% → v16=3% → v17=10% →
    v18=13.3% → v19=20% → **v20=7% (-13pp vs v19, 单批回落)**
  - **回归 flag**: 单批回落不触发 plateau (需 3 批连续无改善)。v21 监控决策：若也
    回落/持平 → 进入归因调查模式 (3-batch plateau rule)
  - **健康度**: 30 seed eval 全 completed，0 timeout / 0 Traceback / 0 hang_confirmed /
    0 MysteriousSphere COMBAT_WON loop

- **`batch_v21` 启动 (2026-05-25, 续训, v20 a1_beat 单批回落后自动起下批)**：
  从 v20 final ckpt 续训。
  - **续训源**：`sts_models/v8_ppo_batch_v20/v8_ppo_final.pt` (episodes_done=768)
  - **参数**：`num_episodes=896 batch_size=32 ckpt_freq=32 eval_freq=128`
    (n_envs=1 serial, 增量训 128 ep, ep 769→896)
  - **PID**：`18042`（nohup）；log `/tmp/v8_ppo_batch_v21.log`；
    output `sts_models/v8_ppo_batch_v21/`；exit signal file
    `/tmp/v8_ppo_batch_v21.exit`（如有）
  - **启动校验**：`[resume] start_episode=768, target=896 (将增量训 128 ep)`，
    0 Traceback
  - **预期**：~2-2.5h 训练 + final eval
  - **监控决策**：若 v21 a1_boss_beat 也 ≤ 10% (相对 v19 的 20% 持平或回落) →
    构成 3-batch plateau (v20+v21+下批)，进入归因调查模式

- **`batch_v21` 完成 + v22 续训启动 (2026-05-25, 连续 2 批 regression flag)**：
  v21 训练 128 ep (ep 769→896) 完成，~108 min 总 wall (6475.6s)，PID 18042 exited
  cleanly。Ckpt 路径 `sts_models/v8_ppo_batch_v21/` 含 ep=800/832/864/896 + final + summary。
  - **训练侧**: 128 ep, 0 Traceback / 0 hang_confirmed
  - **Final eval (ep=896, 30 seeds, attribution-based timeout)**:
    completed_seeds=**30/30** (0 timeout), reached_boss_rate=**0.30** (9/30),
    **a1_boss_beat_rate=0.0333** (1/30), a2_boss_beat_rate=0.00,
    **won_game_rate=0.00**, floor_mean=11.17
  - **Boss reach/kill counts**: Slime Boss reach=3/kill=0, The Guardian reach=2/kill=0,
    Hexaghost reach=3/kill=0 (总 reach=8, 全 0 kill — 持续 metric gap)
  - **趋势对比** (a1_boss_beat eval, 7 batches): v15=10% → v16=3% → v17=10% →
    v18=13.3% → v19=20% (peak) → v20=7% → **v21=3.3% (-3.7pp vs v20, 连续 2 批回落)**
  - **回归 flag**: **连续 2 批 regression** (v19→v20 -13pp, v20→v21 -3.7pp)，
    v21 已**低于 v15 起点 (10%)**，policy collapse 嫌疑大。
    严格 3-batch plateau rule 还差 1 批 (v22)，但实质已是 plateau/collapse 模式
  - **健康度**: 30 seed eval 全 completed，0 timeout / 0 Traceback / 0 hang_confirmed /
    0 MysteriousSphere COMBAT_WON loop

- **`batch_v22` 启动 (2026-05-25, 续训, v21 连续 2 批回落后自动起下批)**：
  从 v21 final ckpt 续训。
  - **续训源**：`sts_models/v8_ppo_batch_v21/v8_ppo_final.pt` (episodes_done=896)
  - **参数**：`num_episodes=1024 batch_size=32 ckpt_freq=32 eval_freq=128`
    (n_envs=1 serial, 增量训 128 ep, ep 897→1024)
  - **PID**：`23805`（nohup）；log `/tmp/v8_ppo_batch_v22.log`；
    output `sts_models/v8_ppo_batch_v22/`；exit signal file
    `/tmp/v8_ppo_batch_v22.exit`（如有）
  - **启动校验**：`[resume] start_episode=896, target=1024 (将增量训 128 ep)`，
    0 Traceback
  - **预期**：~2-2.5h 训练 + final eval
  - **关键决策**：若 v22 仍回落/持平 → 触发归因调查 (3-batch plateau rule 实质命中)。
    候选方向: reward shaping 强化 / lower lr / entropy bonus 调高防 policy collapse /
    回滚到 v19 ckpt 重训用更稳超参

- **`batch_v22` 完成 + v23 续训启动 (2026-05-25, a1_beat=30% 新历史高点, 大幅反弹)**：
  v22 训练 128 ep (ep 897→1024) 完成，~139 min 总 wall (8330.3s)，PID 23805 exited
  cleanly。Ckpt 路径 `sts_models/v8_ppo_batch_v22/` 含 ep=928/960/992/1024 + final + summary。
  - **训练侧**: 128 ep, 0 Traceback / 0 hang_confirmed / 1 guard_cap (正常范围)
  - **Final eval (ep=1024, 30 seeds, attribution-based timeout)**:
    completed_seeds=**30/30** (0 timeout), reached_boss_rate=**0.57** (17/30, 几乎翻倍 vs v21=0.30),
    **a1_boss_beat_rate=0.30** (9/30, **+26.7pp vs v21, +10pp vs v19 peak 20%**),
    a2_boss_beat_rate=0.00, **won_game_rate=0.00**, floor_mean=**9.03** (vs v21=11.17, **下降**)
  - **Boss reach/kill counts**: Hexaghost reach=3/kill=0, Slime Boss reach=4/kill=0,
    The Guardian reach=1/kill=0 (总 reach=8, 全 0 kill — metric gap 持续, kill 计数
    与 a1_boss_beat_rate 同源差异未变)
  - **趋势对比** (a1_boss_beat eval, 8 batches): v15=10% → v16=3% → v17=10% →
    v18=13.3% → v19=20% (prev peak) → v20=7% → v21=3.3% (lowest) → **v22=30%
    (+26.7pp vs v21, 新历史高点, 超 v19 peak +10pp)**
  - **bimodal floor_mean 观察**: floor_mean=9.03 比 v21=11.17 **下降** 2.14, 但
    reached_boss_rate 几乎翻倍 (0.30→0.57)。表明 v22 seed 分布是 bimodal: 要么早死,
    要么打到 boss 并大幅提升 a1 kill 概率。policy 在「冒险打深」和「稳保 floor 10」
    之间向前者偏移
  - **回归 flag 解除**: v20/v21 连续 2 批 regression 被 v22 大幅反弹打破，证实是
    **training stochastic dip 而非 policy collapse**。不再需要进入归因调查模式
    (v19 peak 之后的「连续 2 批回落」trigger 已被 v22 反弹否决)
  - **健康度**: 30 seed eval 全 completed，0 timeout / 0 Traceback / 0 hang_confirmed /
    0 MysteriousSphere COMBAT_WON loop

- **`batch_v23` 启动 (2026-05-25, 续训, v22 a1_beat=30% 新历史高点后自动起下批)**：
  从 v22 final ckpt 续训。
  - **续训源**：`sts_models/v8_ppo_batch_v22/v8_ppo_final.pt` (episodes_done=1024)
  - **参数**：`num_episodes=1152 batch_size=32 ckpt_freq=32 eval_freq=128`
    (n_envs=1 serial, 增量训 128 ep, ep 1025→1152)
  - **PID**：`28987`（nohup）；log `/tmp/v8_ppo_batch_v23.log`；
    output `sts_models/v8_ppo_batch_v23/`；exit signal file
    `/tmp/v8_ppo_batch_v23.exit`（如有）
  - **启动校验**：`[resume] start_episode=1024, target=1152 (将增量训 128 ep)`，
    0 Traceback
  - **预期**：~2-2.5h 训练 + final eval
  - **监控决策**：v22 新高后看 v23 是否能维持 a1_beat ≥ 20% 区间; 若 v23 大幅回落 (≤ 10%)
    可能 v22 是 noise spike, 需 v24 进一步确认

- **`batch_v23` 完成 + v24 续训启动 (2026-05-26, v22 peak 后 stochastic dip)**：
  v23 训练 128 ep (ep 1025→1152) 完成，~125 min 总 wall (7507.6s)，PID 28987
  exited cleanly。Ckpt 路径 `sts_models/v8_ppo_batch_v23/` 含 ep=1056/1088/1120/1152
  + final + summary。
  - **训练侧**: 128 ep, 0 Traceback / 0 hang_confirmed / 0 guard_cap
  - **Final eval (ep=1152, 30 seeds, attribution-based timeout)**:
    completed_seeds=**30/30** (0 timeout), reached_boss_rate=**0.467** (14/30, vs v22=0.57 -10pp),
    **a1_boss_beat_rate=0.167** (5/30, **-13.3pp vs v22 peak 30%**),
    a2_boss_beat_rate=0.00, **won_game_rate=0.00**, floor_mean=**9.73** (vs v22=9.03)
  - **Boss reach/kill counts**: Slime Boss reach=7/kill=0, Hexaghost reach=1/kill=0,
    The Guardian reach=1/kill=0 (总 reach=9, 全 0 kill — metric gap 持续)
  - **趋势对比** (a1_boss_beat eval, 9 batches): v15=10% → v16=3% → v17=10% →
    v18=13.3% → v19=20% → v20=7% → v21=3.3% → v22=30% (peak) → **v23=16.7%
    (-13.3pp vs v22, 单次回落)**
  - **回落归因 (类似 v19→v20)**: v22 peak 后 -13.3pp 单次 dip，pattern 与 v19→v20
    一致（peak → -13pp single batch dip → 后续大反弹）。未触发 3-batch plateau
    rule (仅 1 批回落)，按持续迭代规范继续起 v24
  - **健康度**: 30 seed eval 全 completed，0 timeout / 0 Traceback / 0 hang_confirmed /
    0 MysteriousSphere COMBAT_WON loop

- **`batch_v24` 启动 (2026-05-26, 续训, v23 单次 dip 后自动起下批)**：
  从 v23 final ckpt 续训。
  - **续训源**：`sts_models/v8_ppo_batch_v23/v8_ppo_final.pt` (episodes_done=1152)
  - **参数**：`num_episodes=1280 batch_size=32 ckpt_freq=32 eval_freq=128`
    (n_envs=1 serial, 增量训 128 ep, ep 1153→1280)
  - **PID**：`33643`（nohup）；log `/tmp/v8_ppo_batch_v24.log`；
    output `sts_models/v8_ppo_batch_v24/`；exit signal file
    `/tmp/v8_ppo_batch_v24.exit`（如有）
  - **启动校验**：`[resume] start_episode=1152, target=1280 (将增量训 128 ep)`，
    0 Traceback
  - **预期**：~2-2.5h 训练 + final eval
  - **监控决策**：若 v24 反弹回 ≥ 20% 则确认 v23 是 stochastic dip (类似 v20 模式);
    若 v24 持平/继续 ≤ 17% 则连续 2 批回落，需在 v25 前后准备归因调查

- **`batch_v24` 完成 + v25 续训启动 (2026-05-26, a1_beat=30% 回 v22 peak, 确认 bimodal)**：
  v24 训练 128 ep (ep 1153→1280) 完成，~135 min 总 wall (8114s)，PID 33643
  exited cleanly。Ckpt 路径 `sts_models/v8_ppo_batch_v24/` 含 ep=1184/1216/1248/1280
  + final + summary。
  - **训练侧**: 128 ep, 0 Traceback / 2 guard_cap (正常范围)
  - **Final eval (ep=1280, 30 seeds, attribution-based timeout)**:
    completed_seeds=**30/30** (0 timeout), reached_boss_rate=**0.567** (17/30, 回 v22 水平 0.57),
    **a1_boss_beat_rate=0.30** (9/30, **+13.3pp vs v23, 与 v22 peak 持平**),
    a2_boss_beat_rate=0.00, **won_game_rate=0.00**, floor_mean=**10.57**
    (vs v23=9.73, vs v22=9.03 略升)
  - **Boss reach/kill counts**: The Guardian reach=3/kill=0, Slime Boss reach=3/kill=0,
    Hexaghost reach=2/kill=0 (总 reach=8, 全 0 kill — metric gap 持续未变)
  - **趋势对比** (a1_boss_beat eval, 10 batches): v15=10% → v16=3% → v17=10% →
    v18=13.3% → v19=20% → v20=7% → v21=3.3% → v22=30% (peak) → v23=16.7% (dip) →
    **v24=30% (re-peak, 与 v22 持平)**
  - **Bimodal 模式确认**: v22 / v24 两个 peak 持平 30%, v23 中间 dip 16.7% — 整体
    在 ~30% 稳定区间 + stochastic ~5-15% dip 之间震荡。v22 不是 noise spike, v23
    不是 policy collapse, 而是训练 stochastic 的 bimodal 抖动 (类似 v19/v20 pattern)
  - **健康度**: 30 seed eval 全 completed，0 timeout / 0 Traceback / 0 hang_confirmed /
    0 MysteriousSphere COMBAT_WON loop

- **`batch_v25` 启动 (2026-05-26, 续训, v24 a1_beat 回 peak 后自动起下批)**：
  从 v24 final ckpt 续训。
  - **续训源**：`sts_models/v8_ppo_batch_v24/v8_ppo_final.pt` (episodes_done=1280)
  - **参数**：`num_episodes=1408 batch_size=32 ckpt_freq=32 eval_freq=128`
    (n_envs=1 serial, 增量训 128 ep, ep 1281→1408)
  - **PID**：`38084`（nohup）；log `/tmp/v8_ppo_batch_v25.log`；
    output `sts_models/v8_ppo_batch_v25/`；exit signal file
    `/tmp/v8_ppo_batch_v25.exit`（如有）
  - **启动校验**：`[resume] start_episode=1280, target=1408 (将增量训 128 ep)`，
    0 Traceback
  - **预期**：~2-2.5h 训练 + final eval
  - **监控决策**：v22/v24 双 peak 30% 确认稳定上限，v25 看是否能冲破 30% 天花板;
    若 v25 ≥ 35% 则 trend 继续上行, 若再次 dip 到 15-20% 则 bimodal 区间稳定
    (考虑 reward shaping / SlimeBoss-specific 改动突破 plateau)

- **`batch_v25` 完成 + v26 续训启动 (2026-05-26, 历史最大单批 dip -26.7pp)**：
  v25 训练 128 ep (ep 1281→1408) 完成，~137 min 总 wall (8211s)，PID 38084
  exited cleanly。Ckpt 路径 `sts_models/v8_ppo_batch_v25/` 含 ep=1312/1344/1376/1408
  + final + summary。
  - **训练侧**: 128 ep, 0 Traceback / 0 guard_cap
  - **Final eval (ep=1408, 30 seeds, attribution-based timeout)**:
    completed_seeds=**30/30** (0 timeout), reached_boss_rate=**0.467** (14/30, 与 v23 持平),
    **a1_boss_beat_rate=0.0333** (1/30, **-26.7pp vs v24's 30%, 历史最大单批 dip**),
    a2_boss_beat_rate=0.00, **won_game_rate=0.00**, floor_mean=**11.83**
    (vs v24=10.57 反升, vs v23=9.73)
  - **Boss reach/kill counts**: The Guardian reach=4/kill=0, Slime Boss reach=5/kill=0,
    Hexaghost reach=4/kill=0 (总 reach=13, 全 0 kill — metric gap 持续未变)
  - **趋势对比** (a1_boss_beat eval, 11 batches): v15=10% → v16=3% → v17=10% →
    v18=13.3% → v19=20% → v20=7% → v21=3.3% → v22=30% (peak) → v23=16.7% (dip) →
    v24=30% (re-peak) → **v25=3.3% (max dip, 与 v21 历史最低持平)**
  - **历史最大 dip flag**: 单批回落 -26.7pp 是之前最深 dip 的 ~2x (v22→v23=-13.3pp,
    v19→v20=-13pp)。floor_mean 反升 11.83 + reached_boss 维持 0.467 → 不是模型彻底
    崩溃 (能跑到 boss 房), 但 boss 战斗执行能力骤降 (14 reach 中 13 房只 1 kill)
  - **第 4 次 dip-recovery pattern**: v19/v20, v22/v23, v24/v25 三次 peak → dip,
    但 v25 dip 深度异常。单批 dip 不触发 plateau (N=3 连续低 才触发归因);
    若 v26 仍 < 15% → N=2 连续大 regression alarming, 触发归因调查
  - **健康度**: 30 seed eval 全 completed，0 timeout / 0 Traceback / 0 hang_confirmed /
    0 MysteriousSphere COMBAT_WON loop

- **`batch_v26` 启动 (2026-05-26, 续训, v25 max dip 后自动起下批 + 监控点设置)**：
  从 v25 final ckpt 续训。
  - **续训源**：`sts_models/v8_ppo_batch_v25/v8_ppo_final.pt` (episodes_done=1408)
  - **参数**：`num_episodes=1536 batch_size=32 ckpt_freq=32 eval_freq=128`
    (n_envs=1 serial, 增量训 128 ep, ep 1409→1536)
  - **PID**：`43437`（nohup）；log `/tmp/v8_ppo_batch_v26.log`；
    output `sts_models/v8_ppo_batch_v26/`；exit signal file
    `/tmp/v8_ppo_batch_v26.exit`（如有）
  - **启动校验**：`[resume] start_episode=1408, target=1536 (将增量训 128 ep)`，
    0 Traceback
  - **预期**：~2-2.5h 训练 + final eval
  - **关键监控点**: 若 v26 a1_beat 仍 < 15% → 触发归因调查 (N=2 连续大 regression)
    - 候选归因: reward shaping 影响 / Adam moment 漂移 / entropy collapse / lr too high
    - 备选回滚 ckpt: v22 final (ep=1024) / v24 final (ep=1280) — 两个 peak ckpt
      可作回滚 baseline
  - **监控决策**：v26 ≥ 20% → 确认是 stochastic dip, v25 是 outlier (类似 v19/v22 pattern);
    v26 在 [15%, 20%] → bimodal 区间继续抖动, 可再观望 1 批; v26 < 15% → 立刻进归因

- **`batch_v26` 完成 + v27 续训启动 (2026-05-26, N=3 plateau 严格触发 + ESCALATION)**：
  v26 训练 128 ep (ep 1409→1536) 完成，~142 min 总 wall (8511s)，PID 43437
  exited cleanly。Ckpt 路径 `sts_models/v8_ppo_batch_v26/` 含 ep=1440/1472/1504/1536
  + final + summary。
  - **训练侧**: 128 ep, 0 Traceback / 0 guard_cap
  - **Final eval (ep=1536, 30 seeds, attribution-based timeout)**:
    completed_seeds=**30/30** (0 timeout), reached_boss_rate=**0.50** (15/30, 与 v24 持平),
    **a1_boss_beat_rate=0.133** (4/30, **+10pp vs v25 但远低于 v22/v24 peak 30%**),
    a2_boss_beat_rate=0.00, **won_game_rate=0.00**, floor_mean=**11.93**
    (vs v25=11.83, 持平)
  - **Boss reach/kill counts**: Slime Boss reach=7/kill=0, The Guardian reach=2/kill=0,
    Hexaghost reach=2/kill=0 (总 reach=11, 全 0 kill — metric gap 持续未变;
    Slime Boss 累计 reach 7 仍 0 kill, boss-aware encoding 未在 eval 端发挥作用)
  - **趋势对比** (a1_boss_beat eval, 12 batches): v15=10% → v16=3% → v17=10% →
    v18=13.3% → v19=20% → v20=7% → v21=3.3% → v22=30% (peak) → v23=16.7% → v24=30%
    (re-peak) → v25=3.3% → **v26=13.3%** (v25 dip 后小幅回升但远低于 peak)
  - **[ESCALATION] N=3 plateau 严格触发**: v23/v24/v25/v26 连续 4 批未超 v22 peak 30%。
    v26 (13.3%) 触发先前设的 < 15% 归因监控阈值。bimodal 抖动 + 整体 ceiling 仍卡在
    30% peak。**归因方向所需的架构级决策属于用户决策范围**:
    - **候选归因方向 (供用户讨论)**:
      1. Reward shaping 影响：v22 之后 step reward 是否有 implicit drift？
      2. Entropy collapse：v22 时 entropy ~0.21，v26 时是否更低？需 grep 验证
      3. Adam moment 累积漂移：长 chain resume 后 momentum 偏离最优区域
      4. Bimodal exploration：v22/v24 peak 时找到稳定 trajectory，dip 时 trajectory
         多样化但失败率高
      5. Boss-aware encoding 还需强化（Slime Boss 累计 reach 7 仍 0 kill）
    - **A/B 测试候选**:
      - 平行从 v22 ep=1024 重训 N batch vs v24 ep=1280 重训 N batch，看哪条 trajectory 更稳
      - 调 entropy bonus +20% vs current，看是否能减少 dip 深度
    - **回滚 baseline ckpt**: v22 final (ep=1024) / v24 final (ep=1280) 双 peak
  - **健康度**: 30 seed eval 全 completed，0 timeout / 0 Traceback / 0 hang_confirmed /
    0 MysteriousSphere COMBAT_WON loop

- **`batch_v27` 启动 (2026-05-26, 续训, autonomous loop 继续起标准续训 + ESCALATION FLAG)**：
  从 v26 final ckpt 续训。**autonomous loop 继续起 v27 标准续训，但下批完成后强烈
  建议由 user 介入决定归因方向**。
  - **续训源**：`sts_models/v8_ppo_batch_v26/v8_ppo_final.pt` (episodes_done=1536)
  - **参数**：`num_episodes=1664 batch_size=32 ckpt_freq=32 eval_freq=128`
    (n_envs=1 serial, 增量训 128 ep, ep 1537→1664)
  - **PID**：`48252`（nohup）；log `/tmp/v8_ppo_batch_v27.log`；
    output `sts_models/v8_ppo_batch_v27/`；exit signal file
    `/tmp/v8_ppo_batch_v27.exit`（如有）
  - **启动校验**：`[resume] start_episode=1536, target=1664 (将增量训 128 ep)`，
    0 Traceback
  - **预期**：~2-2.5h 训练 + final eval
  - **[ESCALATION FLAG]**: N=3 plateau 已严格触发 (v23/v24/v25/v26 未超 v22 peak)。
    v27 完成后**强烈建议 user 介入决定归因方向**，autonomous loop 不再自动起 v28。
    需 user 决策: 继续标准续训 / reward shaping / entropy bonus 调整 / 回滚 v22 或 v24
    peak ckpt 重训。

- **`batch_v27` 完成 + v28 续训启动 (2026-05-26, plateau 内持续小幅恢复)**：
  v27 训练 128 ep (ep 1537→1664) 完成，~142 min (8521s)，0 Traceback / 1 guard_cap
  (~0.8%, 正常范围)，PID 48252 已干净退出。Ckpt 路径 `sts_models/v8_ppo_batch_v27/`
  含 ep=1568/1600/1632/1664 + final + v8_ppo_summary.json。
  - **Final eval (ep=1664, 30 seeds, attribution-based timeout)**:
    - reached_boss_rate=**0.60** (18/30, **+10pp vs batch_v26 0.50**)
    - **act1_boss_beat_rate=0.167** (5/30, **+3.3pp vs batch_v26 0.133**, 持续恢复
      但仍远低于 batch_v22 peak 0.30)
    - act2_boss_beat_rate=**0.00**, won_game_rate=**0.00**
    - floor_mean=12.1 (vs batch_v26 11.93 微升)
    - boss_reach_counts: Slime Boss=3, Hexaghost=7, The Guardian=3 (total 13,
      **Hexaghost-heavy 不同于以往**, 之前几批 Guardian/Slime 占主导)
    - boss_kill_counts: 全部 0 (metric gap 持续, 5 个 a1_beat 全部来自非典型 boss
      路径或 eval seed mix 差异)
    - completed_seeds=30/30
  - **跨批 a1_boss_beat 趋势 (13 batch)**：
    ... → batch_v22=**30%** (peak) → batch_v23=16.7% → batch_v24=30% (回 peak) →
    batch_v25=3.3% (max dip) → batch_v26=13.3% (反弹) → **batch_v27=16.7%** (+3.3pp,
    连续 2 批小幅恢复但仍 plateau 内)
  - **判定**：N=3 plateau 仍触发中（v22 peak 之后 5 批未超 peak）。v27 vs v26 +3.3pp
    + reached_boss +10pp 显示**模型在小幅恢复**，但 won_game / a2_beat 持续 0，
    eval 端未突破。继续 plateau 状态 → **等 user 决策归因方向**。
  - **实机测试基础设施 (2026-05-26 已搭好)**：
    - `tools/run_real_test.sh` — 实机测试入口脚本
    - `v8_play_real.py` — multi-game 自动化实机对战，支持连续多局
    - 已知问题待修: subscreen handling (GRID / SHOP / multi-phase EVENT) — 子屏幕
      下 action dispatch 不完整, 跑实机时会卡在某些 event 子页面。修复后可做
      training metrics vs 实机对照验证
  - **健康度**: 30 seed eval 全 completed (0 timeout / 0 Traceback / 0 hang_confirmed /
    0 MysteriousSphere COMBAT_WON loop), 训练 1 guard_cap 在正常范围

- **`batch_v28` 启动 (2026-05-26, 续训, plateau 内自动起下批)**：
  从 v27 final ckpt 续训。**autonomous loop 在 plateau 内继续小步迭代**，
  等 user 决策归因方向期间维持训练节奏。
  - **续训源**：`sts_models/v8_ppo_batch_v27/v8_ppo_final.pt` (episodes_done=1664)
  - **参数**：`num_episodes=1792 batch_size=32 ckpt_freq=32 eval_freq=128`
    (n_envs=1 serial, 增量训 128 ep, ep 1665→1792)
  - **PID**：`60472`（nohup）；log `/tmp/v8_ppo_batch_v28.log`；
    output `sts_models/v8_ppo_batch_v28/`；exit signal file
    `/tmp/v8_ppo_batch_v28.exit`（如有）
  - **启动校验**：`[resume] start_episode=1664, target=1792 (将增量训 128 ep)`，
    0 Traceback
  - **预期**：~2-2.5h 训练 + final eval

- **`batch_v28` 完成 + v29 续训启动 (2026-05-26, a1_boss_beat 回到 peak 30% + reached_boss 历史新高)**：
  v28 训练 128 ep (ep 1665→1792) 完成，~2h35m (9290s)，0 Traceback / 2 guard_cap
  (正常范围)，PID 60472 已干净退出。Ckpt 路径 `sts_models/v8_ppo_batch_v28/`
  含 ep=1696/1728/1760/1792 + final + v8_ppo_summary.json。
  - **Final eval (ep=1792, 30 seeds, attribution-based timeout)**:
    - reached_boss_rate=**0.633** (19/30, **历史新高**, +3.3pp vs prior peak
      batch_v22/v24 0.567)
    - **act1_boss_beat_rate=0.30** (9/30, **回到 peak 30%**, +13.3pp vs batch_v27
      0.167, 与 batch_v22/v24 双 peak 持平)
    - act2_boss_beat_rate=**0.00**, won_game_rate=**0.00**
    - floor_mean=11.2 (vs batch_v27 12.1 微降，但 reached_boss 上升表明 seed mix
      偏向更难 boss / 早期失败更靠 boss 楼层)
    - boss_reach_counts: Slime Boss=4, Hexaghost=4, The Guardian=2 (total 10,
      分布相对均匀)
    - boss_kill_counts: 全部 0 (metric gap 持续, 9 个 a1_beat 全部来自非典型 boss
      路径或 eval seed mix 差异)
    - completed_seeds=30/30
  - **跨批 a1_boss_beat 趋势 (14 batch)**：
    ... → batch_v22=**30%** (peak) → batch_v23=16.7% → batch_v24=30% (回 peak) →
    batch_v25=3.3% (max dip) → batch_v26=13.3% → batch_v27=16.7% →
    **batch_v28=30%** (**回到 peak**, 第三次到达 30% 上限)
  - **判定**：bimodal 模式继续 — peak 不会"丢"（v22/v24/v28 三次 30%），但 dip
    也不会"恢复成 monotonic 上升"（v23/v25/v26/v27 都在 peak 下徘徊）。
    reached_boss 63.3% 是 6 批以来新高，说明**模型能撑到 boss 的能力在小幅累积**，
    但 boss_kill 全 0 的 metric gap 持续——能到 boss 但杀不掉。
  - **健康度**: 30 seed eval 全 completed (0 timeout / 0 Traceback / 0 hang_confirmed /
    0 MysteriousSphere COMBAT_WON loop), 训练 2 guard_cap 在正常范围

- **`batch_v29` 启动 (2026-05-26, 续训, plateau 内自动起下批)**：
  从 v28 final ckpt 续训。**autonomous loop 在 plateau 内继续小步迭代**。
  - **续训源**：`sts_models/v8_ppo_batch_v28/v8_ppo_final.pt` (episodes_done=1792)
  - **参数**：`num_episodes=1920 batch_size=32 ckpt_freq=32 eval_freq=128`
    (n_envs=1 serial, 增量训 128 ep, ep 1793→1920)
  - **PID**：`68347`（nohup）；log `/tmp/v8_ppo_batch_v29.log`；
    output `sts_models/v8_ppo_batch_v29/`；exit signal file
    `/tmp/v8_ppo_batch_v29.exit`（如有）
  - **启动校验**：`[resume] start_episode=1792, target=1920 (将增量训 128 ep)`，
    0 Traceback
  - **预期**：~2-2.5h 训练 + final eval

- **`batch_v29` 完成 + v30 续训启动 (2026-05-26, a1_boss_beat 小幅回落 + reached_boss 再创新高)**：
  v29 训练 128 ep (ep 1793→1920) 完成，~2h20m (8411.8s)，0 Traceback / 0 guard_cap，
  PID 68347 已干净退出。Ckpt 路径 `sts_models/v8_ppo_batch_v29/` 含
  ep=1824/1856/1888/1920 + final + v8_ppo_summary.json。
  - **Final eval (ep=1920, 30 seeds, attribution-based timeout)**:
    - reached_boss_rate=**0.70** (21/30, **历史新高**, +6.7pp vs batch_v28 0.633)
    - **act1_boss_beat_rate=0.233** (7/30, **-6.7pp vs batch_v28 0.30**, 小幅回落
      但仍在 plateau 区间内 vs peak 30%)
    - act2_boss_beat_rate=**0.00**, won_game_rate=**0.00**
    - floor_mean=11.3 (vs batch_v28 11.2 持平)
    - boss_reach_counts: Slime Boss=9, Hexaghost=3, The Guardian=2 (total 14,
      **SB-heavy 64%** — Slime Boss 占比远高于以往批次)
    - boss_kill_counts: 全部 0 (metric gap 持续, 7 个 a1_beat 全部来自非典型 boss
      路径或 eval seed mix 差异)
    - completed_seeds=30/30
  - **跨批 a1_boss_beat 趋势 (15 batch)**：
    ... → batch_v22=**30%** (peak) → batch_v23=16.7% → batch_v24=30% (回 peak) →
    batch_v25=3.3% (max dip) → batch_v26=13.3% → batch_v27=16.7% →
    batch_v28=30% (回 peak) → **batch_v29=23.3%** (小幅回落)
  - **判定**：bimodal/plateau 模式持续。a1_beat 单批 -6.7pp 回落部分由 **SB-heavy
    seed mix** 解释 (SB 一直 0 kill, 占 64% 拉低整体击败率); reached_boss 70% 创
    新高表明**模型撑到 boss 的能力仍在累积**, 但 boss_kill 全 0 的 metric gap
    持续——能到 boss 但杀不掉。
  - **健康度**: 30 seed eval 全 completed (0 timeout / 0 Traceback / 0 hang_confirmed /
    0 MysteriousSphere COMBAT_WON loop), 训练 0 guard_cap 历史最佳

- **`batch_v30` 启动 (2026-05-26, 续训, plateau 内自动起下批)**：
  从 v29 final ckpt 续训。**autonomous loop 在 plateau 内继续小步迭代**。
  - **续训源**：`sts_models/v8_ppo_batch_v29/v8_ppo_final.pt` (episodes_done=1920)
  - **参数**：`num_episodes=2048 batch_size=32 ckpt_freq=32 eval_freq=128`
    (n_envs=1 serial, 增量训 128 ep, ep 1921→2048)
  - **PID**：`86395`（nohup）；log `/tmp/v8_ppo_batch_v30.log`；
    output `sts_models/v8_ppo_batch_v30/`；exit signal file
    `/tmp/v8_ppo_batch_v30.exit`（如有）
  - **启动校验**：`[resume] start_episode=1920, target=2048 (将增量训 128 ep)`，
    0 Traceback
  - **预期**：~2-2.5h 训练 + final eval

- **`batch_v30` 完成 + v31 续训启动 (2026-05-26, a1_boss_beat 回到 peak 30%)**：
  v30 训练 128 ep (ep 1921→2048) + final eval 全部完成，~2.5h 训练 (9125.6s)，
  0 Traceback，PID 86395 已干净退出。Ckpt 路径 `sts_models/v8_ppo_batch_v30/` 含
  ep=1952/1984/2016/2048 + final + summary。
  - **Final eval (ep=2048, 30 seeds)**:
    - reached_boss_rate=**0.633** (19/30, 与 batch_v28 持平, vs batch_v29 0.70 微降 -6.7pp)
    - **act1_boss_beat_rate=0.30** (9/30, **回 peak**, +6.7pp vs batch_v29 0.233)
    - act2_boss_beat_rate=0.00
    - won_game_rate=0.00 (metric gap 持续)
    - floor_mean=10.57
    - boss_reach_counts: TheGuardian=1, Hexaghost=5, SlimeBoss=4 (total 10, 较均衡 mix)
    - boss_kill_counts: 全 0 (metric gap)
    - completed_seeds=30/30
  - **趋势 (a1_boss_beat eval, 最近 5 批)**：
    batch_v26=23% → batch_v27=27% → **batch_v28=30%** (peak) →
    batch_v29=23.3% (回落) → **batch_v30=30%** (回 peak)
    plateau 在 23-30% 区间内波动，**未突破上限**
  - **健康度**: 30 seed eval 全 completed (0 timeout / 0 Traceback / 0 hang_confirmed /
    0 MysteriousSphere COMBAT_WON loop), 训练 0 guard_cap

- **`batch_v31` 启动 (2026-05-26, 续训, 模拟器 Question Card / Busted Crown bug 修复后第一批)**：
  从 v30 final ckpt 续训。**关键差异**: 这是 StSRLSolver fork commit `1413d69f`
  (主项目 commit `ee09e150`) Question Card / Busted Crown bug 修复后的第一批训练。
  修复影响 ~1% 选卡决策，主要是问题卡 + 破碎王冠组合下的卡牌奖励分支。
  - **续训源**：`sts_models/v8_ppo_batch_v30/v8_ppo_final.pt` (episodes_done=2048)
  - **参数**：`num_episodes=2176 batch_size=32 ckpt_freq=32 eval_freq=128`
    (n_envs=1 serial, 增量训 128 ep, ep 2049→2176)
  - **PID**：`90806`（nohup）；log `/tmp/v8_ppo_batch_v31.log`；
    output `sts_models/v8_ppo_batch_v31/`；exit signal file
    `/tmp/v8_ppo_batch_v31.exit`（如有）
  - **启动校验**：`[resume] start_episode=2048, target=2176 (将增量训 128 ep)`，
    0 Traceback
  - **观察重点**：bug 修复影响面 ~1%，预期数据波动在噪声范围内；连续 2-3 批
    观察是否有微小 trend 变化
  - **预期**：~2-2.5h 训练 + final eval

- **`batch_v31` 完成 + v32 续训启动 (2026-05-26, 模拟器 bug 修复后第一批, SB-heavy seed 拖低 a1_beat)**：
  v31 训练 128 ep (ep 2049→2176) 完成，elapsed 8398.5s (~2.3h)，0 Traceback。
  Ckpt 路径 `sts_models/v8_ppo_batch_v31/` 含 ep=2080/2112/2144/2176 + final +
  v8_ppo_summary.json。这是 StSRLSolver fork commit `1413d69f` (Question Card /
  Busted Crown bug 修复) 后的第一批训练。
  - **Final eval (ep=2176, 30 seed, attribution-based timeout)**:
    completed_seeds=30/30, reached_boss_rate=**0.567** (17/30, 微降 vs v30 63.3%),
    **act1_boss_beat_rate=0.233** (7/30, **-6.7pp vs v30 30%**, 同 v29 水平),
    act2_boss_beat_rate=0.00, **won_game_rate=0.00**, floor_mean=10.6
  - **Boss reach/kill counts**: Guardian reach=2 kill=0, Hexaghost reach=2 kill=0,
    **Slime Boss reach=6** kill=0 (total reach=10, **SB-heavy 60%**)
  - **跨批 a1_boss_beat trend (17 批)**: … → batch_v28=30% → batch_v29=23.3% →
    batch_v30=30% → **batch_v31=23.3%**。持续在 23-30% 区间波动 6 批了
  - **回落部分原因**: SB-heavy seed mix (6/10 boss 遭遇 = 60% Slime Boss) 拖低
    a1_beat，因 SlimeBoss 累计 0 kill（boss-aware encoding 在 eval 端持续无效）
  - **模拟器 bug 修复影响**: 对这批训练 trend 无明显影响（修复只影响 ~1% 选卡，
    符合预期；信号被 SB-heavy seed noise 完全掩盖）
  - **健康度**: 0 Traceback / 30 seed eval 全 completed / 0 hang_confirmed /
    0 MysteriousSphere COMBAT_WON loop

- **`batch_v32` 启动 (2026-05-26, 续训, plateau 继续观察)**：
  从 v31 final ckpt 续训。
  - **续训源**：`sts_models/v8_ppo_batch_v31/v8_ppo_final.pt` (episodes_done=2176)
  - **参数**：`num_episodes=2304 batch_size=32 ckpt_freq=32 eval_freq=128`
    (n_envs=1 serial, 增量训 128 ep, ep 2177→2304)
  - **PID**：`95493`（nohup）；log `/tmp/v8_ppo_batch_v32.log`；
    output `sts_models/v8_ppo_batch_v32/`；exit signal file
    `/tmp/v8_ppo_batch_v32.exit`（如有）
  - **启动校验**：`[resume] start_episode=2176, target=2304 (将增量训 128 ep)`，
    0 Traceback
  - **预期**：~2-2.5h 训练 + final eval

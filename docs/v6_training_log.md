# V6 Training Log

## 2026-04-12 02:55 - Crash Fix & Training Restart

### Bug: trajectory_buffer accumulation in worker processes

**Symptom**: Training with workers > 1 "dies silently" after 1 PPO update. The process appears to hang during the PPO update step.

**Root Cause**: `collect_one_game()` in `train_v6.py` did not clear `_worker_agent.trajectory_buffer` between games. Since `multiprocessing.Pool` reuses worker processes, each subsequent game call accumulated ALL previous games' trajectories in the buffer.

With 2 workers and 16 games per batch: each worker ran 8 games, accumulating 1+2+...+8 = 36 trajectory lists per worker. Total: ~80 trajectories instead of the expected 16. This caused the PPO update to process ~5x more transitions, making it take 10+ minutes instead of ~2 minutes, appearing as a silent hang.

With 4 workers the duplication was less extreme (~40 vs 16) but still caused massive slowdowns, especially as the PPO loop does per-transition forward passes (not batched).

**Fix** (train_v6.py, `collect_one_game()`):
- Added `_worker_agent.trajectory_buffer = []` at the start of each game
- Changed return to use `list()` copies to prevent cross-game interference

**Verification**:
- 1 worker: 16 trajectories, 172s (was already working)
- 2 workers: 16 trajectories, 106s (was hanging at 80 trajectories)
- 4 workers: 16 trajectories, 98s (was crashing/hanging)

**Training restarted**: `--from-scratch --n-games 500 --workers 4`
Output: `v6_training_v3.log`

## Round 1 训练结果 (500局完成)

**时间**: 2026-04-12
**配置**: 500局, 4 workers, from-scratch, per-card immediate rewards

**结果**:
- 496/500 局完成，0 崩溃
- 平均层数: 6.1（从头到尾持平，无学习信号）
- 最高层数: 16
- 胜率: 0%
- 熵: 1.04 → 0.96（策略在收敛但没有提升表现）

**诊断**: 训练流水线稳定运行，但模型没有学到有效策略。平均 floor 6.1 从第一批到最后一批完全没变化。需要分析原因：
- reward 设计是否合理？
- 模型是否能从输入中获取有效信息？
- PPO 超参数是否合适？
- 需要 curriculum learning？

<!-- git smoke test 2026-04-29 -->

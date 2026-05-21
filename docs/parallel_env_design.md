# Parallel V8 Env 设计文档

## 目标

把 V8 RL PPO 训练 rollout collection 从串行 N 局 → 并行 N 局，目标加速 **~3.5×**
（n_envs=4 时；理论上限 4×，扣掉 ipc + 主进程 forward 串行化）。

phase 1（本期）：研究 + 设计 + scaffolding。不改 trainer / 不改 main loop。
phase 2（下一期）：trainer 适配 + 真训练验证。

## 当前架构（串行）

调用栈：
- `tools/v8_ppo_train.py main()`
  - `for k in range(batch_target):` 串行收 `batch_size` 个 episode
    - `trainer.collect_rollout(env, seed=ep_idx)` 跑完一整局 STS
      - `while True:` 步循环
        - `model.forward(state, actions)` → 主进程 MPS forward
        - `env.step(action_idx)` → 子线程不存在，主进程一气呵成跑：
          - meta phase transition / event handling
          - **战斗内**：`env._advance_to_meta_decision()` 内部循环跑 `TurnSolver.pick_action`
            （CPU 重，单局可能几十秒到几分钟）
          - **post-battle**：`evaluate_deck()` 调 `deck_evaluator` 的 spawn pool
            （12 worker × 4 enemies × 3 sim）
  - PPO update：在 batch_size 个 ep concat 后跑一次

关键资源：
- **deck_evaluator pool**：`_POOL: ProcessPoolExecutor`（spawn, max_workers=12）
  module-global，懒创建，`atexit` shutdown，`restart_pool()` 显式重建。
- **logger**：单进程，所有 [combat] / [floor] / [event] / [perf] / [heartbeat]
  都写 stdout/stderr。trainer 配置 StreamHandler 绑 stdout 强制 line_buffering。
- **V8CombatNetWrapper**：含 `V8Model`，作为 `combat_net` 注入 `TurnSolverAdapter`，
  在战斗 search leaf eval 时 forward 一次（MPS / CPU）。**与主进程 model 共享权重**
  （wrapper 持 model ref，不 copy）；call_count 累加在 wrapper instance。

V8Env gym-like：
- `reset(seed) -> V8State`
- `step(action_idx) -> (V8State, reward, done, info)`
- `get_available_actions() -> List[str]`（pointer-net 必需，动态长度）
- `close()`
- 额外字段：`set_episode(ep_idx)`、`reset_perf_counters()`、暴露 `runner` 属性
  让 trainer eval 拿 `final_floor / final_act / game_won`。

## ParallelV8Env 设计

### 进程模型

```
主进程
├── model (V8Model on MPS)
├── trainer (V8PPOTrainer)
├── ParallelV8Env
│   ├── worker-0 (子进程, V8Env 实例) ←Pipe→
│   ├── worker-1 (子进程, V8Env 实例) ←Pipe→
│   ├── worker-2 (子进程, V8Env 实例) ←Pipe→
│   └── worker-3 (子进程, V8Env 实例) ←Pipe→
└── PPO update (batch_size × n_envs ep concat)
```

每个子进程：
- 独立 GameRunner / TurnSolverAdapter
- 独立 deck_evaluator pool（见下方权衡）
- 独立 logger，stderr 前缀 `[env-N]`
- 独立 V8Env per-episode 计数器

### 通讯协议

主 → 子（`Pipe.send((cmd, *args))`）：

| cmd | args | reply |
|---|---|---|
| `reset` | seed: int | (V8State,) |
| `step` | action_idx: int | (V8State, reward, done, info) |
| `get_actions` | – | List[str] |
| `set_episode` | ep_idx: int | None |
| `reset_perf_counters` | – | None |
| `ping` | – | "pong" |
| `close` | – | None（子进程退出）|

错误：子进程任何异常 → `("err", repr(e))`，主进程 `recv()` 后 `raise RuntimeError`。

### deck_evaluator pool 处理

**phase 1 选择：(b) 每个子进程内独立 spawn pool。**

| 方案 | 优点 | 缺点 |
|---|---|---|
| (a) 中央 pool，子→主 pipe 转 deck eval 请求 | worker 总数可控（12 不会变成 48）；cache 全局共享 | pipe 把 deck eval 任务回主进程会让主进程成新瓶颈（每 ep 几十次 evaluate_deck）；实现复杂；cache 主进程持有时 deck hash 仍要传 |
| (b) 每个子进程独立 pool | 实现最简单（V8Env 内 import deck_evaluator 即可）；deck cache 在子进程内有效（同一 episode 内大量复用） | n_envs=4 × 12 worker = 48 子进程，撑爆 CPU（M-series 一般 8-10 物理核）；cache 不跨子进程共享 |

**phase 1 用 (b) 但**给 worker 设小一些：
- 通过 `V8_DECK_EVALUATOR_PARALLEL=0` 可关闭子进程内并行，回退串行 sim。
- phase 2 建议：env_kwargs 加 `deck_eval_workers=3`，让 `deck_evaluator._get_pool`
  接受 override，n_envs=4 → 4×3=12 worker，跟现状串行 12 一致。
- 进一步：phase 2 视情况评估 (a)，若发现 cache miss 太多（cache 不跨子进程共享）。

### combat_net_wrapper 处理（phase 1 vs phase 2）

**phase 1**：子进程内 `combat_net_wrapper=None`，纯搜索（与 trial100 之前等价）。
- 原因：spawn 不能可靠 pickle `V8Model + MPS tensors`；fork 在 macOS 已知不安全。
- 影响：训练效果会回到"无 model 战斗 leaf eval"。不能直接用 phase 1 的并行训练成果，
  必须等 phase 2 把 wrapper 接通才能上真训练。

**phase 2 方案**：
- 方案 X：每个子进程内独立 V8Model（CPU），权重定期同步（每 batch 用
  `state_dict` pickle 推送）。CPU forward 比 MPS 慢但子进程无 GPU 冲突；leaf eval
  call 次数不算多（每 turn ~1 次）。
- 方案 Y：子进程在 leaf eval 时通过 pipe 把 (state, candidates) 发回主进程做
  forward，主进程批量 batched forward 后回传。理论 GPU 利用率高，实现复杂，
  且主进程的 pipe latency 可能反而拖慢。
- 方案 Z：phase 2 直接接受"无 model 战斗 leaf"作为 acceptable regression，
  让 PPO 学得更慢但管线干净。

phase 2 决策点：先 benchmark phase 1 在 `combat_net=None` 下的速度提升幅度
（如果 search 占主导，wrapper 缺失影响小，方案 Z 可行）。

### 不暴露的字段 / 限制

- `env.runner` / `env._last_event_id` / `env._last_action_repr` 这些属性目前 trainer
  和 eval-attribution 直接读。**ParallelV8Env 不复制**，phase 2 trainer 改造要让 eval
  loop 不再依赖这些字段（或加 `get_runner_snapshot()` 命令把所需字段一次 pipe 回来）。
- `wrapper.call_count` 当前是主进程 instance attr；phase 1 子进程没 wrapper → 这个
  统计在并行模式下为 0。phase 2 视方案 X/Y/Z 重新设计统计回传。

## API（V8EnvBase）

```python
class V8EnvBase(ABC):
    n_envs: int
    def reset(self, seeds: List[int]) -> List[V8State]: ...
    def step(self, action_indices: List[int]) -> Tuple[List[V8State], List[float], List[bool], List[dict]]: ...
    def get_available_actions(self) -> List[List[str]]: ...
    def close(self) -> None: ...
```

`V8Env`（单 env）目前不强制继承 `V8EnvBase`，由 trainer 用 duck typing 兼容。
phase 2 决定是否给 V8Env 加 `as_batched()` adapter 或让 trainer 直接走两条路径。

## phase 2 改动 plan（trainer 适配）

`tools/v8_ppo_train.py`：
1. 启动期改成 `ParallelV8Env(n_envs=args.n_envs, env_kwargs=...)`。
2. `for k in range(batch_target)` 串行循环 → `for batch_round in range(batch_target // n_envs)`，
   每轮一次性收 `n_envs` 局。
3. eval：phase 1 用 N=1 ParallelV8Env（或保留 V8Env），保证 eval per-seed 顺序 +
   `env.runner` 直读不受影响。

`v8/trainer.py`：
1. `collect_rollout(env, seed)` → 新增 `collect_rollout_batched(parallel_env, seeds: List[int])`。
   返回 `List[List[RolloutStep]]`（n_envs 条 trajectory）。
2. 内部主循环：
   ```
   states = penv.reset(seeds)
   active = [True] * n_envs
   while any(active):
       acts_lists = penv.get_available_actions()
       # 对每个 active env 跑 model forward（pointer-net 不同长，必须 per-env forward；
       # phase 2 可考虑 padding + batched forward 进一步加速）
       action_idxs = []
       for i in range(n_envs):
           if not active[i]:
               action_idxs.append(0)
               continue
           out = self.model(states[i], acts_lists[i])
           ... sample/argmax ...
           action_idxs.append(idx)
       next_states, rewards, dones, infos = penv.step(action_idxs)
       # 记 trajectory；done=True 的 env 标 inactive
       ...
   ```
3. 注意 `done=True` 后 env 仍在子进程里 "等下次 reset"，不能再 step，得 mask 掉。

`v8/combat_net_wrapper.py` / `v8/model.py`：phase 2 决定方案 X/Y/Z 后再改。

工程量预估：
- trainer 改 ~3-4 小时（含 done masking 边界条件 + 测试）
- main loop 改 ~2 小时
- combat_net_wrapper 方案落地 ~1 天（含 IPC 设计 + 同步策略 + benchmark）
- 全套 smoke + perf 对比：~半天

## 风险点

1. **deck_evaluator 子进程内 pool 撑爆 CPU**。n_envs=4 × 12 worker 默认 = 48 进程。
   phase 2 必须给 `deck_evaluator._get_pool()` 加 max_workers override 参数，
   外部通过 env var / env_kwargs 控制。建议 n_envs × workers ≤ 物理核数 - 2。

1b. **【phase 1 smoke 发现的真问题】**`daemonic processes are not allowed to have
    children`：worker 进程当前 `daemon=True`，触发 deck_evaluator `_get_pool()`
    spawn 子 pool 时 Python 抛 `AssertionError`，deck_evaluator 自动 fallback 到
    串行 sim（log: `evaluate_deck parallel dispatch failed ... falling back to
    sequential`）。phase 1 smoke 验证 worker 通讯正常，但 evaluate_deck 实际是
    串行的，**phase 2 真训练必须修**。选项：
    - 去掉 `_worker_main` 的 `daemon=True`（自己负责 shutdown 时 reap 子孙进程）
    - 走方案 (a) 中央 pool（子进程不再起 pool）
    - 强制子进程内 `V8_DECK_EVALUATOR_PARALLEL=0` + 接受单 sim 的串行成本

2. **combat_net_wrapper 缺失导致并行训练效果 ≠ 串行训练效果**。phase 1 跑出来的
   "快" 数字不是 apples-to-apples，必须 phase 2 接通 wrapper 后重测。

3. **V8State pickle cost**。phase 1 每次 step 来回 pickle 一次 V8State（含完整 deck /
   relics / map_nodes ~几 KB）；预计 episode 内几十 step，总 ipc ~MB，对比 search +
   evaluate_deck 几秒级别可忽略。**phase 2 需要 benchmark 实测**。

4. **logger flooding**。4 个子进程 [floor] / [event] / [combat] / [action] 日志全打 stderr，
   一行行交错，调试时不易看。phase 2 加 `[env-N]` 前缀已经够用；
   下一步可考虑按 env 分文件落盘。

5. **子进程 V8Env init 失败传不回主进程**。worker 入口已加 try/except + `conn.send(("err", ...))`，
   但如果连 conn.send 都失败（malformed pickle），主进程的 `recv()` 会被 `EOFError`
   触发，已 raise RuntimeError 兜底。需要在 phase 2 加 health check（启动后 ping 一遍）。

6. **eval 路径**：当前 `run_eval()` 严重依赖 `env.runner` 直接读字段 + 单进程
   threading.Thread 看 progress signal。phase 2 不要并行 eval，直接保留串行 V8Env
   做 eval；并行只用于训练 rollout 收集。

## 未解决问题

- **deck cache 跨 env 共享**：目前 cache 是 module-global dict，子进程间不共享。
  同一 episode 内（cache hit 率高）影响不大，但跨 episode 重启的 starter deck
  会被多次重算。phase 2 真训练前测一下 cache miss 率，若过高考虑方案 (a)
  中央 pool 顺便共享 cache。

- **SIGINT 软停**：当前 `_install_sigint_handler` 在主进程。phase 2 子进程也要响应
  SIGINT 礼貌关闭，否则主进程关后子进程会成孤儿。

- **per-env perf counter 聚合**：`env.eval_deck_calls` / `env.combat_search_calls`
  是子进程内 V8Env 字段；要回 phase 2 trainer，得新加 `get_perf_counters()` 命令。

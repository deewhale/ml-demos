# STS V8 RL 训练数据 Web 可视化

V8 RL 训练数据 web 可视化界面，含中文 mapping。FastAPI + sqlite3 + 纯 HTML/JS + Chart.js，零构建步骤。

## 快速启动

依赖：项目根 `.venv`（Python 3.12/3.13），已装 `fastapi` + `uvicorn`。其余只用标准库。

```bash
# 1. 首次需要装 fastapi/uvicorn（已装可跳过）
.venv/bin/pip install fastapi uvicorn

# 2. ETL 全流程入库（首次跑或 schema 更新后跑）
.venv/bin/python -m web.etl.scan_batches \
  && .venv/bin/python -m web.etl.parse_log \
  && .venv/bin/python -m web.etl.extract_localization \
  && .venv/bin/python -m web.etl.load_mappings

# 3. 启动 web
.venv/bin/uvicorn web.app:app --port 8008
# 浏览器访问 http://localhost:8008
```

注：`extract_localization` 从本地 Steam `desktop-1.0.jar` 抽中文 mapping JSON。Jar 不存在时
脚本会跳过中文，UI 显示英文兜底。`scan_batches` + `parse_log` 是数据入库主流程，下次只需
增量重跑（在 web 内点「重新扫描」按钮等价）。

## 目录结构

```
web/
├── app.py                  # FastAPI 入口，挂载 5 个 router + 静态资源
├── db.py                   # sqlite3 connection helper + init_db()
├── schema.sql              # 全部表结构（batches / metrics / eval / episodes / ... / i18n）
├── etl/                    # 入库脚本
│   ├── scan_batches.py     # 扫 sts_models/v8_ppo_batch_* 的 summary.json 入库
│   ├── parse_log.py        # 流式解析 /tmp/v8_ppo_batch_*.log 入库 6 表
│   ├── markers.py          # log marker 正则集合
│   ├── extract_localization.py  # 从 jar 抽中文 mapping JSON
│   └── load_mappings.py    # 把 content_py + jar JSON 灌进 i18n_entries
├── routers/                # API 实现
│   ├── training.py         # /api/training/*
│   ├── eval.py             # /api/eval/*
│   ├── episode.py          # /api/episode/*
│   ├── deck.py             # /api/deck/*
│   ├── lookup.py           # /api/lookup/*  英中查询
│   └── admin.py            # /api/admin/rescan  P6 添加
├── static/                 # 前端
│   ├── index.html          # 单页骨架 + 左侧导航 + 重新扫描按钮
│   ├── app.css             # 全部样式
│   ├── app.js              # 路由 + i18n 缓存 + 通用 UI helper
│   ├── page_training.js    # 训练曲线
│   ├── page_eval.js        # Eval 详情
│   ├── page_episode.js     # 单局回放
│   └── page_deck.js        # 牌组聚合
└── data/                   # 数据产物（不入 git）
    ├── sts_viz.sqlite      # 主 db
    └── localization_zho.json  # jar 抽出来的中文 mapping
```

## 更新数据

训练完成新 batch 后，有两种方式重新入库：

**A. 命令行**

```bash
.venv/bin/python -m web.etl.scan_batches    # 增量按 mtime 跳过未变化的
.venv/bin/python -m web.etl.parse_log       # 解析所有 batch 的 log
```

**B. 浏览器**

打开 web 后左侧导航底部点「重新扫描」按钮，等约 1s 看到「完成: X batch / Y ep / Z.Zs」即可。
按钮内部调用 `POST /api/admin/rescan` 同步触发 ETL，完成后自动刷新当前页。

中文 mapping 不会随训练数据更新；只在更新 jar 或 content_py 后才需要重跑 `load_mappings`。

## 中文 Mapping 来源

`extract_localization.py` 从 Steam 安装目录的 `desktop-1.0.jar` 里抽 `localization/zho/*.json`，
合并写到 `web/data/localization_zho.json`。Jar 路径默认扫描：

- `~/Downloads/*.jar`（手动下载）
- Mac: `~/Library/Application Support/Steam/steamapps/common/SlayTheSpire/SlayTheSpire.app/Contents/Resources/desktop-1.0.jar`
- Linux/SteamOS: `~/.steam/steam/steamapps/common/SlayTheSpire/desktop-1.0.jar`
- Windows: `C:/Program Files (x86)/Steam/steamapps/common/SlayTheSpire/desktop-1.0.jar`

可用 `--jar PATH` 指定。Jar 不存在则跳过中文抽取，`load_mappings` 仍能跑（只有英文行）。

`load_mappings.py` 把英文（来自 `external/StSRLSolver/packages/engine/content/*.py`）和中文（jar）
合并写入 `i18n_entries` 表。P6 添加了「no-space alias」逻辑：日志里 `event_id` 等是
CamelCase 无空格（`BonfireElementals`），i18n 表里 en_id 带空格（`Bonfire Elementals`），
ETL 时会自动多插入一条 `en_id.replace(' ','')` 的 alias 行（`source='alias_no_space'`），
让 LEFT JOIN 两种 key 都能命中中文。

覆盖率（当前）：card 100% / relic 100% / monster 94% / event 98% / potion 100%。

## 数据库位置

`web/data/sts_viz.sqlite`。可以直接用 sqlite3 cli 查：

```bash
sqlite3 web/data/sts_viz.sqlite
sqlite> .tables
sqlite> SELECT batch_id, episodes_done, exit_reason FROM batches ORDER BY scanned_at;
sqlite> SELECT kind, COUNT(*) FROM i18n_entries GROUP BY kind;
```

## 页面说明

| 页面 | 路径 | 干啥 |
| --- | --- | --- |
| 训练曲线 | `#/training` | 跨 batch 同指标 line chart（mean_reward / beat_boss_count / entropy 等） |
| Eval 详情 | `#/eval` | 单 batch 单 eval 时点的 metric 卡片 + 跨 batch won_game 趋势 + boss 击杀分布 |
| 单局回放 | `#/episode` | 选 batch + 胜局过滤 → 左侧 episode 列表，右侧 floor/combat/event/deck 时间轴 |
| 牌组 | `#/deck` | 胜局牌组卡牌频次柱状图，可过滤 floor 下限 + top N |

## 端口冲突时

`uvicorn web.app:app --port 8888`（或任意未占用端口）。前端 fetch 全部用相对路径，端口换了不影响。

## 已知 quirk

- `extract_localization` 在没装 jar 的机器上会 warning 跳过，`load_mappings` 仍会写入英文行，
  UI 用 `lookupI18N` 兜底显示英文 `en_name`，不会崩
- 训练目录命名约定：`sts_models/v8_ppo_batch_v<N>` 才会被 `scan_batches` 扫到（前缀 `batch_`
  必须有）；旧的 `v8_ppo_long_v*` 目录不在 P6 范围内
- 4 个孤儿目录（无 `v8_ppo_summary.json`）在 rescan 时会被计入 `batches_errors`，正常
- `BonfireElementals` 这类 alias 行 source 字段是 `alias_no_space`，原始行 source 仍是
  `jar_zho`；下游查询不区分 source，只看 zh_name

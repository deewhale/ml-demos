// 训练曲线页：episode-centric，无 batch 概念
// - 顶部 metric 多选 chip + 刷新按钮
// - 主区 CSS Grid 多张小 chart card：每张一个 metric，单条连续曲线
// - x 轴 = 已完成局数（episodes_done / ep）
// - 奖励 / 步数：用 training/episodes（每局 1 个原始值，无任何平滑/聚合），
//   连成连续折线（pointRadius:0），默认视窗缩放到最近 ~300 局，可缩放/平移看全程。
// - PPO 指标（entropy/loss/kl 等）：用 training/metrics_all（PPO update 级，
//   每 32 局一个聚合点），连成线。x 轴 = episodes_done。
// - Chart.js zoom plugin 支持拖拽缩放 + 平移（pan）

(function () {
    // 默认初始视窗：只显示最近 N 局，避免 3840 点太密。可缩放/平移看全程。
    const DEFAULT_WINDOW = 300;

    const METRICS = [
        { key: "mean_reward", label: "奖励（每局）", color: "#2563eb" },
        { key: "mean_steps", label: "步数（每局）", color: "#0891b2" },
        { key: "entropy", label: "策略熵", color: "#7c3aed" },
        { key: "policy_loss", label: "策略损失", color: "#d97706" },
        { key: "value_loss", label: "价值损失", color: "#db2777" },
        { key: "approx_kl", label: "近似 KL 散度", color: "#65a30d" },
        { key: "clip_frac", label: "裁剪比例", color: "#9333ea" },
        { key: "update_secs", label: "更新耗时(秒)", color: "#0d9488" },
        { key: "wrapper_calls", label: "Wrapper 调用", color: "#6b7280" },
    ];

    // 以下 metric 使用 per-episode 数据（episodes 表，每局一个原始点，无平滑/聚合）
    // 其余 metric 用 metrics_all（每 32 局一个聚合点，PPO update 级）
    // 楼层/Boss 进度改由「死亡楼层分布」+「Act 进度分布」两图展示，不再做 per-ep 散点
    // 奖励/步数 = 每局 1 个原始值（reward / steps），连成连续折线（pointRadius:0），
    //   默认缩放到最近 ~300 局，拖拽缩放 / 平移可看全部 3840 局。
    const PER_EP_FIELD = {
        mean_reward: "reward",
        mean_steps: "steps",
    };

    const DEFAULT_METRICS = [
        "mean_reward",
        "mean_steps",
        "entropy",
        "policy_loss",
        "value_loss",
    ];

    // ===== 行为指标定义 =====
    const BEHAVIOR_METRICS = [
        { key: "card_skip_rate", label: "跳过选牌率", color: "#dc2626", format: "pct" },
        { key: "rest_rate", label: "休息 vs 升级（休息占比）", color: "#16a34a", format: "pct" },
        { key: "cards_picked_per_ep", label: "每局拿卡数", color: "#2563eb", format: "num" },
        { key: "shop_buy_rate", label: "商店购买率", color: "#d97706", format: "pct" },
    ];

    const state = {
        selectedMetrics: DEFAULT_METRICS.slice(),
        metricsRows: [],     // /api/training/metrics_all 返回的 flat array
        episodesRows: [],    // /api/training/episodes 返回的 flat array
        deathData: [],       // /api/training/death_distribution 返回的楼层分布
        behaviorRows: [],    // /api/training/behavior 返回的窗口聚合
        actProgressionRows: [], // /api/training/act_progression 返回的窗口聚合
        charts: {},
        deathChart: null,    // 死亡楼层分布 Chart.js 实例
        behaviorCharts: {},  // 行为指标 Chart.js 实例
        actProgressionChart: null, // Act 进度分布 Chart.js 实例
    };

    async function fetchMetricsAll() {
        const res = await fetch("/api/training/metrics_all");
        if (!res.ok) throw new Error("加载训练指标失败: " + res.status);
        const data = await res.json();
        return Array.isArray(data) ? data : (Array.isArray(data.rows) ? data.rows : []);
    }

    async function fetchEpisodes() {
        const res = await fetch("/api/training/episodes");
        if (!res.ok) throw new Error("加载训练局数据失败: " + res.status);
        const data = await res.json();
        return Array.isArray(data) ? data : (Array.isArray(data.rows) ? data.rows : []);
    }

    async function fetchDeathDistribution(epMin, epMax) {
        const q = new URLSearchParams();
        if (epMin != null) q.append("ep_min", String(epMin));
        if (epMax != null) q.append("ep_max", String(epMax));
        const url = "/api/training/death_distribution" + (q.toString() ? "?" + q.toString() : "");
        const res = await fetch(url);
        if (!res.ok) throw new Error("加载死亡楼层分布失败: " + res.status);
        return res.json();
    }

    async function fetchBehavior() {
        const res = await fetch("/api/training/behavior");
        if (!res.ok) throw new Error("加载行为指标失败: " + res.status);
        const data = await res.json();
        return Array.isArray(data) ? data : [];
    }

    async function fetchActProgression() {
        const res = await fetch("/api/training/act_progression");
        if (!res.ok) throw new Error("加载 Act 进度失败: " + res.status);
        const data = await res.json();
        return Array.isArray(data) ? data : [];
    }

    // (rolling 平滑已移除 — 全部用原始数据)

    function renderControls(root) {
        const metricChips = METRICS
            .map((m) => {
                const active = state.selectedMetrics.includes(m.key) ? "active" : "";
                const style = active
                    ? `style="background:${m.color};border-color:${m.color};color:#fff;"`
                    : "";
                return `<span class="chip metric-chip ${active}" data-metric="${m.key}" ${style}>${m.label}</span>`;
            })
            .join("");

        root.innerHTML = `
            <h2>训练曲线</h2>
            <div class="controls">
                <label>指标:</label>
                <div id="metric-chip-row">${metricChips}</div>
                <button id="metric-all-btn" type="button">全选</button>
                <button id="metric-none-btn" type="button">清空</button>
                <button id="refresh-btn" type="button">刷新</button>
                <button id="reset-zoom-btn" type="button">重置缩放</button>
            </div>
            <div class="controls">
                <span class="muted">x 轴 = 已完成局数 · 奖励/步数每局 1 个原始值连成折线（无平滑/聚合）· 默认显示最近 300 局 · 滚轮/拖拽缩放、按住平移 · 「重置缩放」看全程 · 楼层 / Boss 进度见下方「死亡楼层分布」「Act 进度分布」</span>
            </div>
            <div id="metric-grid" class="metric-grid"></div>
        `;

        root.querySelectorAll(".chip[data-metric]").forEach((el) => {
            el.addEventListener("click", () => {
                const mk = el.dataset.metric;
                if (state.selectedMetrics.includes(mk)) {
                    state.selectedMetrics = state.selectedMetrics.filter((x) => x !== mk);
                } else {
                    state.selectedMetrics.push(mk);
                }
                renderControls(root);
                redrawAll();
            });
        });

        root.querySelector("#metric-all-btn").addEventListener("click", () => {
            state.selectedMetrics = METRICS.map((m) => m.key);
            renderControls(root);
            redrawAll();
        });
        root.querySelector("#metric-none-btn").addEventListener("click", () => {
            state.selectedMetrics = [];
            renderControls(root);
            redrawAll();
        });

        root.querySelector("#refresh-btn").addEventListener("click", async () => {
            const btn = root.querySelector("#refresh-btn");
            btn.disabled = true;
            try {
                const [metrics, episodes] = await Promise.all([fetchMetricsAll(), fetchEpisodes()]);
                state.metricsRows = metrics;
                state.episodesRows = episodes;
                renderControls(root);
                redrawAll();
                // 同步刷新死亡分布图（如果面板存在）
                if (document.getElementById("death-dist-chart-wrap")) {
                    refreshDeathChart();
                }
            } catch (e) {
                console.warn("refresh failed", e);
            } finally {
                if (btn) btn.disabled = false;
            }
        });

        // 「重置缩放」= 显示全部局数（不是回到默认的最近窗口）
        root.querySelector("#reset-zoom-btn").addEventListener("click", () => {
            for (const chart of Object.values(state.charts)) {
                if (!chart) continue;
                if (chart.$fullRange && chart.scales && chart.scales.x) {
                    chart.options.scales.x.min = chart.$fullRange.min;
                    chart.options.scales.x.max = chart.$fullRange.max;
                    chart.update("none");
                } else if (chart.resetZoom) {
                    chart.resetZoom();
                }
            }
        });
    }

    function destroyAllCharts() {
        for (const k of Object.keys(state.charts)) {
            try { state.charts[k].destroy(); } catch (_) {}
        }
        state.charts = {};
    }

    // 返回 { full:{min,max}, view:{min,max} }
    //   full = 全部数据的 x 范围（重置缩放 / pan limits 用）
    //   view = 初始视窗（缩放到最近 DEFAULT_WINDOW 局）
    function computeXRange() {
        let xMin = Infinity, xMax = -Infinity;
        for (const r of state.metricsRows) {
            const x = r.episodes_done || 0;
            if (x < xMin) xMin = x;
            if (x > xMax) xMax = x;
        }
        for (const r of state.episodesRows) {
            const x = r.ep || 0;
            if (x < xMin) xMin = x;
            if (x > xMax) xMax = x;
        }
        if (!isFinite(xMin) || !isFinite(xMax) || xMin === xMax) return null;
        const full = { min: xMin, max: xMax };
        const viewMin = Math.max(xMin, xMax - DEFAULT_WINDOW);
        const view = { min: viewMin, max: xMax };
        return { full, view };
    }

    function renderOneChart(metricKey, container, xRange, isLastRow) {
        const meta = METRICS.find((m) => m.key === metricKey);
        const metricLabel = (meta && meta.label) || window.metricZh(metricKey) || metricKey;
        const color = (meta && meta.color) || "#2563eb";

        const card = document.createElement("div");
        card.className = "metric-card metric-card-chart";
        card.innerHTML = `
            <div class="metric-card-title">${metricLabel}</div>
            <div class="metric-card-canvas-wrap"><canvas></canvas></div>
        `;
        container.appendChild(card);
        const canvas = card.querySelector("canvas");

        const usePerEp = PER_EP_FIELD.hasOwnProperty(metricKey);
        let datasets;

        if (usePerEp) {
            const field = PER_EP_FIELD[metricKey];
            const raw = state.episodesRows
                .filter((r) => r[field] !== null && r[field] !== undefined)
                .map((r) => ({ x: r.ep || 0, y: Number(r[field]) }));

            // 每局 1 个原始值，连成连续折线（无平滑、无散点）。
            // pointRadius:0 → 不画点，3840 个点渲染成一条锯齿折线而非挤成一团；
            // tension:0 → 锯齿直连，忠实呈现原始信号；borderWidth:1 → 细线减少糊成块。
            datasets = [
                {
                    label: metricLabel,
                    data: raw,
                    borderColor: color,
                    backgroundColor: color,
                    showLine: true,
                    pointRadius: 0,
                    borderWidth: 1,
                    tension: 0,
                    spanGaps: true,
                },
            ];
        } else {
            const points = state.metricsRows
                .filter((r) => r[metricKey] !== null && r[metricKey] !== undefined)
                .map((r) => ({ x: r.episodes_done || 0, y: r[metricKey] }));

            datasets = [
                {
                    label: metricLabel,
                    data: points,
                    borderColor: color,
                    backgroundColor: color,
                    tension: 0.2,
                    pointRadius: 2,
                    borderWidth: 2,
                    spanGaps: true,
                },
            ];
        }

        const xZh = "已完成局数";
        const xScale = {
            type: "linear",
            display: true,
            title: { display: isLastRow, text: isLastRow ? xZh : "" },
            ticks: { display: true, font: { size: 10 }, maxTicksLimit: 6 },
        };
        // 初始视窗缩放到最近 DEFAULT_WINDOW 局（view）；缩放/平移可看全程。
        const fullRange = xRange ? xRange.full : null;
        if (xRange && xRange.view) {
            xScale.min = xRange.view.min;
            xScale.max = xRange.view.max;
        }

        const zoomOpts = {
            zoom: {
                wheel: { enabled: true },
                drag: { enabled: true, backgroundColor: "rgba(37,99,235,0.1)", borderColor: "#2563eb", borderWidth: 1 },
                mode: "x",
            },
            pan: { enabled: true, mode: "x" },
        };
        // 限制缩放/平移不超出实际数据范围
        if (fullRange) {
            zoomOpts.limits = { x: { min: fullRange.min, max: fullRange.max } };
        }

        const chart = new Chart(canvas, {
            type: "line",
            data: { datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                parsing: false,
                scales: {
                    x: xScale,
                    y: {
                        title: { display: false },
                        ticks: { font: { size: 10 } },
                    },
                },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        mode: "nearest",
                        intersect: false,
                        callbacks: {
                            title: (items) => {
                                if (!items.length) return "";
                                return `${xZh} = ${items[0].parsed.x}`;
                            },
                            label: (ctx) => {
                                const v = ctx.parsed.y;
                                const vStr = (v == null)
                                    ? "—"
                                    : (Math.abs(v) >= 1 ? v.toFixed(2) : v.toFixed(4));
                                return `${ctx.dataset.label || metricLabel} = ${vStr}`;
                            },
                        },
                    },
                    zoom: zoomOpts,
                },
            },
        });
        // 记下全程范围，供「重置缩放」按钮把视窗拉到全部局数（而非回到默认窗口）
        chart.$fullRange = fullRange;
        state.charts[metricKey] = chart;
    }

    function redrawAll() {
        destroyAllCharts();
        const grid = document.getElementById("metric-grid");
        if (!grid) return;
        grid.innerHTML = "";
        if (state.selectedMetrics.length === 0) {
            grid.innerHTML = `<div class="empty">请至少选 1 个指标</div>`;
            return;
        }
        if (!state.metricsRows.length && !state.episodesRows.length) {
            grid.innerHTML = `<div class="empty">暂无训练指标数据</div>`;
            return;
        }
        const xRange = computeXRange();
        const total = state.selectedMetrics.length;
        state.selectedMetrics.forEach((mk, idx) => {
            const isLastRow = idx >= total - 2;
            renderOneChart(mk, grid, xRange, isLastRow);
        });
    }

    // ===== 死亡楼层分布图 =====
    function destroyDeathChart() {
        if (state.deathChart) {
            try { state.deathChart.destroy(); } catch (_) {}
            state.deathChart = null;
        }
    }

    function renderDeathChart() {
        destroyDeathChart();
        const container = document.getElementById("death-dist-chart-wrap");
        if (!container) return;
        if (!state.deathData || !state.deathData.length) {
            container.innerHTML = `<div class="empty">暂无楼层分布数据</div>`;
            return;
        }

        container.innerHTML = `<canvas id="death-dist-canvas"></canvas>`;
        const canvas = document.getElementById("death-dist-canvas");
        if (!canvas) return;

        const data = state.deathData;
        const labels = data.map(d => d.floor);
        const deaths = data.map(d => d.count - d.wins);
        const wins = data.map(d => d.wins);

        // Boss 楼层注解线
        const bossAnnotations = {};
        const bossFloors = [
            { floor: 16, label: "Act 1 Boss" },
            { floor: 33, label: "Act 2 Boss" },
            { floor: 50, label: "Act 3 Boss" },
        ];
        bossFloors.forEach((b, i) => {
            if (labels.includes(b.floor) || b.floor <= Math.max(...labels)) {
                bossAnnotations["boss" + i] = {
                    type: "line",
                    xMin: b.floor,
                    xMax: b.floor,
                    borderColor: "rgba(31, 41, 55, 0.5)",
                    borderWidth: 1.5,
                    borderDash: [5, 3],
                    label: {
                        display: true,
                        content: b.label,
                        position: "start",
                        backgroundColor: "rgba(31, 41, 55, 0.75)",
                        color: "#fff",
                        font: { size: 10 },
                        padding: 3,
                    },
                };
            }
        });

        state.deathChart = new Chart(canvas, {
            type: "bar",
            data: {
                labels: labels,
                datasets: [
                    {
                        label: "失败",
                        data: deaths,
                        backgroundColor: "rgba(239, 68, 68, 0.7)",
                        borderColor: "#dc2626",
                        borderWidth: 1,
                    },
                    {
                        label: "通关",
                        data: wins,
                        backgroundColor: "rgba(34, 197, 94, 0.7)",
                        borderColor: "#16a34a",
                        borderWidth: 1,
                    },
                ],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                scales: {
                    x: {
                        stacked: true,
                        title: { display: true, text: "结束楼层" },
                        ticks: { font: { size: 10 } },
                    },
                    y: {
                        stacked: true,
                        title: { display: true, text: "局数" },
                        beginAtZero: true,
                        ticks: { stepSize: 1, font: { size: 10 } },
                    },
                },
                plugins: {
                    legend: { display: true, position: "top", labels: { font: { size: 11 } } },
                    tooltip: {
                        callbacks: {
                            title(items) {
                                if (!items.length) return "";
                                return `第 ${items[0].label} 楼`;
                            },
                        },
                    },
                    annotation: {
                        annotations: bossAnnotations,
                    },
                },
            },
        });
    }

    async function refreshDeathChart() {
        const epMinEl = document.getElementById("death-ep-min");
        const epMaxEl = document.getElementById("death-ep-max");
        const epMin = epMinEl && epMinEl.value ? parseInt(epMinEl.value, 10) : null;
        const epMax = epMaxEl && epMaxEl.value ? parseInt(epMaxEl.value, 10) : null;
        try {
            state.deathData = await fetchDeathDistribution(
                isNaN(epMin) ? null : epMin,
                isNaN(epMax) ? null : epMax,
            );
        } catch (e) {
            console.warn("death distribution fetch failed", e);
            state.deathData = [];
        }
        renderDeathChart();
    }

    // ===== 行为指标图表 =====
    function destroyBehaviorCharts() {
        for (const k of Object.keys(state.behaviorCharts)) {
            try { state.behaviorCharts[k].destroy(); } catch (_) {}
        }
        state.behaviorCharts = {};
    }

    function renderBehaviorCharts() {
        destroyBehaviorCharts();
        const grid = document.getElementById("behavior-grid");
        if (!grid) return;
        grid.innerHTML = "";

        if (!state.behaviorRows || !state.behaviorRows.length) {
            grid.innerHTML = `<div class="empty">暂无行为指标数据</div>`;
            return;
        }

        BEHAVIOR_METRICS.forEach((bm, idx) => {
            const card = document.createElement("div");
            card.className = "metric-card metric-card-chart";
            card.innerHTML = `
                <div class="metric-card-title">${bm.label}</div>
                <div class="metric-card-canvas-wrap"><canvas></canvas></div>
            `;
            grid.appendChild(card);
            const canvas = card.querySelector("canvas");

            const points = state.behaviorRows
                .filter((r) => r[bm.key] !== null && r[bm.key] !== undefined)
                .map((r) => ({
                    x: Math.round((r.ep_start + r.ep_end) / 2),
                    y: bm.format === "pct" ? r[bm.key] * 100 : r[bm.key],
                }));

            const isLastRow = idx >= BEHAVIOR_METRICS.length - 2;
            const xZh = "已完成局数";

            const chart = new Chart(canvas, {
                type: "line",
                data: {
                    datasets: [
                        {
                            label: bm.label,
                            data: points,
                            borderColor: bm.color,
                            backgroundColor: bm.color + "20",
                            fill: true,
                            tension: 0.3,
                            pointRadius: 3,
                            borderWidth: 2,
                            spanGaps: true,
                        },
                    ],
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: false,
                    parsing: false,
                    scales: {
                        x: {
                            type: "linear",
                            display: true,
                            title: { display: isLastRow, text: isLastRow ? xZh : "" },
                            ticks: { font: { size: 10 }, maxTicksLimit: 6 },
                        },
                        y: {
                            title: { display: false },
                            ticks: {
                                font: { size: 10 },
                                callback: function (v) {
                                    return bm.format === "pct" ? v.toFixed(0) + "%" : v.toFixed(1);
                                },
                            },
                            beginAtZero: true,
                            max: bm.format === "pct" ? 100 : undefined,
                        },
                    },
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            mode: "nearest",
                            intersect: false,
                            callbacks: {
                                title: (items) => {
                                    if (!items.length) return "";
                                    return `${xZh} = ${items[0].parsed.x}`;
                                },
                                label: (ctx) => {
                                    const v = ctx.parsed.y;
                                    if (v == null) return bm.label + " = —";
                                    return bm.format === "pct"
                                        ? `${bm.label} = ${v.toFixed(1)}%`
                                        : `${bm.label} = ${v.toFixed(2)}`;
                                },
                            },
                        },
                        zoom: {
                            zoom: {
                                drag: { enabled: true, backgroundColor: "rgba(37,99,235,0.1)", borderColor: "#2563eb", borderWidth: 1 },
                                mode: "x",
                            },
                        },
                    },
                },
            });
            state.behaviorCharts[bm.key] = chart;
        });
    }

    // ===== Act 进度分布图（stacked area） =====
    function destroyActProgressionChart() {
        if (state.actProgressionChart) {
            try { state.actProgressionChart.destroy(); } catch (_) {}
            state.actProgressionChart = null;
        }
    }

    function renderActProgressionChart() {
        destroyActProgressionChart();
        const container = document.getElementById("act-progression-chart-wrap");
        if (!container) return;

        if (!state.actProgressionRows || !state.actProgressionRows.length) {
            container.innerHTML = `<div class="empty">暂无 Act 进度数据</div>`;
            return;
        }

        container.innerHTML = `<canvas id="act-progression-canvas"></canvas>`;
        const canvas = document.getElementById("act-progression-canvas");
        if (!canvas) return;

        const data = state.actProgressionRows;
        const xLabels = data.map(d => Math.round((d.ep_start + d.ep_end) / 2));

        // Stacked areas from bottom (widest → narrowest 闯关里程碑):
        // 打过 Act1 boss → 打过 Act2 boss → 到达 Act3 boss → 通关
        // Each layer's data is cumulative percentage (already 0-1 from API, convert to 0-100)
        const layers = [
            { key: "beat_a1_boss", label: "打过 Act 1 Boss (F17+)", color: "rgba(134, 239, 172, 0.7)", border: "#16a34a" },
            { key: "beat_a2_boss", label: "打过 Act 2 Boss (F34+)", color: "rgba(74, 222, 128, 0.7)", border: "#15803d" },
            { key: "reached_a3_boss", label: "到达 Act 3 Boss (F50+)", color: "rgba(34, 197, 94, 0.7)", border: "#166534" },
            { key: "won", label: "通关 (F55+)", color: "rgba(22, 163, 74, 0.7)", border: "#14532d" },
        ];

        const datasets = layers.map((layer, idx) => ({
            label: layer.label,
            data: data.map((d, i) => ({ x: xLabels[i], y: (d[layer.key] || 0) * 100 })),
            borderColor: layer.border,
            backgroundColor: layer.color,
            fill: idx === 0 ? "origin" : "-1",
            tension: 0.3,
            pointRadius: 2,
            borderWidth: 1.5,
            spanGaps: true,
            order: layers.length - idx,
        }));

        const xZh = "已完成局数";

        state.actProgressionChart = new Chart(canvas, {
            type: "line",
            data: { datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                parsing: false,
                interaction: { mode: "index", intersect: false },
                scales: {
                    x: {
                        type: "linear",
                        display: true,
                        title: { display: true, text: xZh },
                        ticks: { font: { size: 10 }, maxTicksLimit: 8 },
                    },
                    y: {
                        stacked: true,
                        title: { display: true, text: "比例 (%)" },
                        min: 0,
                        max: 100,
                        ticks: {
                            font: { size: 10 },
                            callback: function (v) { return v + "%"; },
                        },
                    },
                },
                plugins: {
                    legend: { display: true, position: "top", labels: { font: { size: 11 } } },
                    tooltip: {
                        mode: "index",
                        intersect: false,
                        callbacks: {
                            title: (items) => {
                                if (!items.length) return "";
                                return `${xZh} = ${items[0].parsed.x}`;
                            },
                            label: (ctx) => {
                                const v = ctx.parsed.y;
                                return `${ctx.dataset.label} = ${v != null ? v.toFixed(1) + "%" : "—"}`;
                            },
                        },
                    },
                    zoom: {
                        zoom: {
                            drag: { enabled: true, backgroundColor: "rgba(37,99,235,0.1)", borderColor: "#2563eb", borderWidth: 1 },
                            mode: "x",
                        },
                    },
                },
            },
        });
    }

    async function render(root, params) {
        root.innerHTML = window.UI.loading("载入训练数据...");
        try {
            const [metrics, episodes, deathDist, behavior, actProg] = await Promise.all([
                fetchMetricsAll(),
                fetchEpisodes(),
                fetchDeathDistribution(null, null),
                fetchBehavior(),
                fetchActProgression(),
            ]);
            state.metricsRows = metrics;
            state.episodesRows = episodes;
            state.deathData = deathDist;
            state.behaviorRows = behavior;
            state.actProgressionRows = actProg;
        } catch (e) {
            root.innerHTML = window.UI.error(e.message);
            return;
        }

        if (state.metricsRows.length === 0 && state.episodesRows.length === 0) {
            root.innerHTML = window.UI.empty(
                "暂无训练数据，请先点击\"重新扫描\"或运行训练"
            );
            return;
        }

        if (params.metrics) {
            state.selectedMetrics = params.metrics.split(",").filter((k) =>
                METRICS.some((m) => m.key === k)
            );
        } else if (params.metric) {
            state.selectedMetrics = [params.metric];
        } else {
            state.selectedMetrics = DEFAULT_METRICS.slice();
        }
        if (state.selectedMetrics.length === 0) {
            state.selectedMetrics = DEFAULT_METRICS.slice();
        }

        renderControls(root);
        redrawAll();

        // 死亡楼层分布区域（追加到 root 末尾）
        const deathSection = document.createElement("div");
        deathSection.innerHTML = `
            <div class="death-dist-section">
                <h3 class="death-dist-title">死亡楼层分布</h3>
                <div class="controls death-dist-controls">
                    <label>局范围:</label>
                    <input type="number" id="death-ep-min" placeholder="起始局" style="width:80px;">
                    <span class="muted">~</span>
                    <input type="number" id="death-ep-max" placeholder="结束局" style="width:80px;">
                    <button type="button" id="death-filter-btn">筛选</button>
                    <button type="button" id="death-reset-btn">重置</button>
                </div>
                <div id="death-dist-chart-wrap" class="death-dist-chart-wrap"></div>
            </div>
        `;
        root.appendChild(deathSection);

        // 渲染死亡分布图
        renderDeathChart();

        // 筛选按钮事件
        document.getElementById("death-filter-btn").addEventListener("click", refreshDeathChart);
        document.getElementById("death-reset-btn").addEventListener("click", () => {
            const epMinEl = document.getElementById("death-ep-min");
            const epMaxEl = document.getElementById("death-ep-max");
            if (epMinEl) epMinEl.value = "";
            if (epMaxEl) epMaxEl.value = "";
            refreshDeathChart();
        });

        // ===== Act 进度分布区域 =====
        const actProgressionSection = document.createElement("div");
        actProgressionSection.innerHTML = `
            <div class="act-progression-section">
                <h3 style="margin: 24px 0 4px 0; font-size: 17px; color: #1f2937;">Act 进度分布</h3>
                <div class="controls" style="padding: 8px 16px;">
                    <span class="muted" style="font-size: 12px;">各 Act 到达率随训练变化 · 窗口 = 64 局 · 堆叠面积图</span>
                </div>
                <div id="act-progression-chart-wrap" class="act-progression-chart-wrap"></div>
            </div>
        `;
        root.appendChild(actProgressionSection);
        renderActProgressionChart();

        // ===== 行为指标区域 =====
        const behaviorSection = document.createElement("div");
        behaviorSection.innerHTML = `
            <div class="behavior-section">
                <h3 style="margin: 24px 0 4px 0; font-size: 17px; color: #1f2937;">Agent 行为指标</h3>
                <div class="controls" style="padding: 8px 16px;">
                    <span class="muted" style="font-size: 12px;">从 meta_decisions 聚合 · 窗口 = 64 局 · x 轴 = 窗口中点局数</span>
                </div>
                <div id="behavior-grid" class="metric-grid"></div>
            </div>
        `;
        root.appendChild(behaviorSection);
        renderBehaviorCharts();
    }

    window.PAGES.training = { render };
})();

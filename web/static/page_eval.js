// 评估结果页：episode-centric，无 batch 概念
// - 左侧：所有评估时点按局数升序的列表
// - 右上：当前选中时点的指标卡（通关率 / 抵达 Boss 率 / etc）
// - 右下：通关率随局数的趋势曲线
// - 底部：当前时点的 Boss 抵达 / 击败柱状图
// 中文化：所有 label / tooltip 用 metricZh / roomZh

(function () {
    let evalResults = [];
    let selectedEp = null;
    let bossRows = [];
    let trendChart = null;
    let floorTrendChart = null;
    let bossChart = null;

    async function fetchEvalResults() {
        const res = await fetch("/api/eval/results");
        if (!res.ok) throw new Error("加载评估结果失败: " + res.status);
        const data = await res.json();
        return Array.isArray(data) ? data : [];
    }

    async function fetchBossBreakdown(ep) {
        const res = await fetch(`/api/eval/boss_breakdown?ep=${ep}`);
        if (!res.ok) throw new Error("加载 Boss 分布失败: " + res.status);
        return res.json();
    }

    function pct(v) {
        if (v == null) return "—";
        return (v * 100).toFixed(1) + "%";
    }

    function fmt(v, digits = 2) {
        if (v == null) return "—";
        return Number(v).toFixed(digits);
    }

    function rateTone(rate) {
        if (rate == null) return "rate-none";
        if (rate >= 0.5) return "rate-high";
        if (rate >= 0.25) return "rate-mid";
        return "rate-low";
    }

    function renderMetricCards(container, evalRow) {
        if (!evalRow) {
            container.innerHTML = window.UI.empty("暂无评估数据");
            return;
        }
        const cards = [
            { label: "通关率", value: pct(evalRow.won_game_rate), tone: rateTone(evalRow.won_game_rate), big: true },
            { label: "抵达 Boss 率", value: pct(evalRow.reached_boss_rate), tone: rateTone(evalRow.reached_boss_rate), big: true },
            { label: "一层 Boss 击败率", value: pct(evalRow.act1_boss_beat_rate), tone: rateTone(evalRow.act1_boss_beat_rate), big: true },
            { label: "二层 Boss 击败率", value: pct(evalRow.act2_boss_beat_rate), tone: rateTone(evalRow.act2_boss_beat_rate), big: true },
            { label: "平均楼层", value: fmt(evalRow.floor_mean, 1) },
            { label: "完成 / 种子数", value: `${evalRow.completed} / ${evalRow.num_seeds}` },
            { label: "平均步数", value: fmt(evalRow.avg_steps, 1) },
            { label: "总耗时", value: fmt(evalRow.secs, 0) + " 秒" },
        ];
        container.innerHTML = cards
            .map(
                (c) => `
                <div class="metric-card ${c.tone || ""} ${c.big ? "metric-big" : ""}">
                    <div class="label">${c.label}</div>
                    <div class="value">${c.value}</div>
                </div>`
            )
            .join("");
    }

    function destroyCharts() {
        if (trendChart) { trendChart.destroy(); trendChart = null; }
        if (floorTrendChart) { floorTrendChart.destroy(); floorTrendChart = null; }
        if (bossChart) { bossChart.destroy(); bossChart = null; }
    }

    function drawTrendChart() {
        const rateCanvas = document.getElementById("trend-canvas");
        const floorCanvas = document.getElementById("floor-trend-canvas");
        if (!rateCanvas) return;
        if (trendChart) { trendChart.destroy(); trendChart = null; }
        if (floorTrendChart) { floorTrendChart.destroy(); floorTrendChart = null; }

        // -- Rate metrics chart (0-100%) --
        const rateMetrics = [
            { key: "reached_boss_rate",    label: "到达Boss率",       color: "#2563eb" },
            { key: "act1_boss_beat_rate",  label: "Act1 Boss击杀率",  color: "#16a34a" },
            { key: "act2_boss_beat_rate",  label: "Act2 Boss击杀率",  color: "#d97706" },
            { key: "won_game_rate",        label: "通关率",           color: "#dc2626" },
        ];

        const rateDatasets = rateMetrics.map((m) => ({
            label: m.label,
            data: evalResults
                .filter((r) => r[m.key] != null)
                .map((r) => ({ x: r.episodes_done, y: r[m.key] })),
            borderColor: m.color,
            backgroundColor: m.color,
            pointRadius: 3,
            borderWidth: 2,
            tension: 0,
            showLine: true,
        }));

        if (rateDatasets.some((ds) => ds.data.length > 0)) {
            trendChart = new Chart(rateCanvas, {
                type: "line",
                data: { datasets: rateDatasets },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    parsing: false,
                    scales: {
                        x: { type: "linear", title: { display: true, text: "已完成局数" } },
                        y: { title: { display: true, text: "比率" }, min: 0, max: 1,
                             ticks: { callback: (v) => (v * 100).toFixed(0) + "%" } },
                    },
                    plugins: {
                        legend: { display: true, position: "top" },
                        tooltip: {
                            callbacks: {
                                title: (items) => items.length ? `已完成局数 = ${items[0].parsed.x}` : "",
                                label: (ctx) => {
                                    const v = ctx.parsed.y;
                                    const vStr = (v == null) ? "—" : (v * 100).toFixed(1) + "%";
                                    return `${ctx.dataset.label} = ${vStr}`;
                                },
                            },
                        },
                    },
                    onClick: (evt, elements) => {
                        if (!elements.length) return;
                        const idx = elements[0].index;
                        const dsIdx = elements[0].datasetIndex;
                        const pt = rateDatasets[dsIdx].data[idx];
                        if (pt) {
                            selectedEp = pt.x;
                            renderEvalList(document.getElementById("eval-list-pane"));
                            refreshEpDetail();
                        }
                    },
                },
            });
        }

        // -- Floor metrics chart (absolute values) --
        if (!floorCanvas) return;
        const floorMetrics = [
            { key: "floor_mean", label: "平均楼层", color: "#7c3aed" },
            { key: "floor_max",  label: "最高楼层", color: "#0891b2" },
        ];

        const floorDatasets = floorMetrics.map((m) => ({
            label: m.label,
            data: evalResults
                .filter((r) => r[m.key] != null)
                .map((r) => ({ x: r.episodes_done, y: r[m.key] })),
            borderColor: m.color,
            backgroundColor: m.color,
            pointRadius: 3,
            borderWidth: 2,
            tension: 0,
            showLine: true,
        }));

        if (floorDatasets.some((ds) => ds.data.length > 0)) {
            floorTrendChart = new Chart(floorCanvas, {
                type: "line",
                data: { datasets: floorDatasets },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    parsing: false,
                    scales: {
                        x: { type: "linear", title: { display: true, text: "已完成局数" } },
                        y: { title: { display: true, text: "楼层" }, beginAtZero: true },
                    },
                    plugins: {
                        legend: { display: true, position: "top" },
                        tooltip: {
                            callbacks: {
                                title: (items) => items.length ? `已完成局数 = ${items[0].parsed.x}` : "",
                                label: (ctx) => {
                                    const v = ctx.parsed.y;
                                    const vStr = (v == null) ? "—" : Number(v).toFixed(1);
                                    return `${ctx.dataset.label} = ${vStr}`;
                                },
                            },
                        },
                    },
                    onClick: (evt, elements) => {
                        if (!elements.length) return;
                        const idx = elements[0].index;
                        const dsIdx = elements[0].datasetIndex;
                        const pt = floorDatasets[dsIdx].data[idx];
                        if (pt) {
                            selectedEp = pt.x;
                            renderEvalList(document.getElementById("eval-list-pane"));
                            refreshEpDetail();
                        }
                    },
                },
            });
        }
    }

    function drawBossChart() {
        const canvas = document.getElementById("boss-canvas");
        if (!canvas) return;
        if (bossChart) { bossChart.destroy(); bossChart = null; }

        const labels = bossRows.map((r) =>
            r.zh_name || r.en_name || r.boss_en_id
        );
        const reachData = bossRows.map((r) => r.reach || 0);
        const killData = bossRows.map((r) => r.kill || 0);

        bossChart = new Chart(canvas, {
            type: "bar",
            data: {
                labels,
                datasets: [
                    {
                        label: "抵达",
                        data: reachData,
                        backgroundColor: "#93c5fd",
                        borderColor: "#2563eb",
                    },
                    {
                        label: "击败",
                        data: killData,
                        backgroundColor: "#86efac",
                        borderColor: "#16a34a",
                    },
                ],
            },
            options: {
                indexAxis: "y",
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { beginAtZero: true, ticks: { stepSize: 1 }, title: { display: true, text: "种子数" } },
                },
                plugins: {
                    legend: { position: "top" },
                    tooltip: {
                        callbacks: {
                            label: (ctx) => {
                                return `${ctx.dataset.label}: ${ctx.parsed.x}`;
                            },
                        },
                    },
                },
            },
        });
    }

    function renderEvalList(container) {
        if (!evalResults.length) {
            container.innerHTML = window.UI.empty("无评估记录");
            return;
        }
        const rows = evalResults
            .map((r) => {
                const cls = r.episodes_done === selectedEp ? "selected" : "";
                return `
                    <tr class="${cls}" data-ep="${r.episodes_done}">
                        <td class="num">${r.episodes_done}</td>
                        <td class="num">${pct(r.won_game_rate)}</td>
                        <td class="num">${pct(r.act1_boss_beat_rate)}</td>
                        <td class="num">${fmt(r.floor_mean, 1)}</td>
                    </tr>
                `;
            })
            .join("");
        container.innerHTML = `
            <table class="simple ep-table">
                <thead>
                    <tr>
                        <th>局数</th><th>通关率</th><th>一层 Boss 击败率</th><th>平均楼层</th>
                    </tr>
                </thead>
                <tbody>${rows}</tbody>
            </table>
        `;
        container.querySelectorAll("tr[data-ep]").forEach((tr) => {
            tr.addEventListener("click", async () => {
                selectedEp = parseInt(tr.dataset.ep, 10);
                renderEvalList(container);
                await refreshEpDetail();
            });
        });
    }

    async function refreshEpDetail() {
        const row = evalResults.find((r) => r.episodes_done === selectedEp);
        renderMetricCards(document.getElementById("metric-grid"), row);
        if (row) {
            try {
                bossRows = await fetchBossBreakdown(row.episodes_done);
            } catch (e) {
                bossRows = [];
                console.error(e);
            }
        } else {
            bossRows = [];
        }
        drawBossChart();
    }

    async function render(root, params) {
        destroyCharts();
        root.innerHTML = window.UI.loading("载入评估数据...");
        try {
            evalResults = await fetchEvalResults();
        } catch (e) {
            root.innerHTML = window.UI.error(e.message);
            return;
        }

        if (!evalResults.length) {
            root.innerHTML = window.UI.empty("暂无评估数据");
            return;
        }

        // 按 episodes_done 升序
        evalResults.sort((a, b) => (a.episodes_done || 0) - (b.episodes_done || 0));

        // 默认选最后一个 eval 时点（最新 ep）
        if (params.ep) {
            const want = parseInt(params.ep, 10);
            const hit = evalResults.find((r) => r.episodes_done === want);
            selectedEp = hit ? want : evalResults[evalResults.length - 1].episodes_done;
        } else {
            selectedEp = evalResults[evalResults.length - 1].episodes_done;
        }

        root.innerHTML = `
            <h2>评估结果</h2>
            <div class="controls">
                <span class="muted">点击左侧某一行或图表上的点查看详情</span>
            </div>
            <div class="ep-layout">
                <div class="ep-list-pane" id="eval-list-pane"></div>
                <div class="tl-pane">
                    <div class="metric-grid" id="metric-grid"></div>
                    <div class="chart-card">
                        <h3>评估指标趋势（比率）</h3>
                        <div class="chart-wrap"><canvas id="trend-canvas"></canvas></div>
                    </div>
                    <div class="chart-card">
                        <h3>楼层趋势</h3>
                        <div class="chart-wrap"><canvas id="floor-trend-canvas"></canvas></div>
                    </div>
                    <div class="chart-card">
                        <h3>Boss 击败 / 抵达分布（当前时点）</h3>
                        <div class="chart-wrap"><canvas id="boss-canvas"></canvas></div>
                    </div>
                </div>
            </div>
        `;

        renderEvalList(document.getElementById("eval-list-pane"));
        await refreshEpDetail();
        drawTrendChart();
    }

    window.PAGES.eval = { render };
})();

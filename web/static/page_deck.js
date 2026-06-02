// 牌组聚合页：episode-centric，无 batch 概念
// - 控件：胜负过滤 + floor 下限 + top N + 稀有度过滤 + 显示基础卡 + ep 范围
// - 数据：单次调 /api/deck/card_frequency，后端返回 flat array
// - 中文化：所有 label / tooltip 用中文；自定义 plugin 在 bar 末端显示频次
// - 颜色按 rarity；过滤基础卡；稀有度多选 chip

(function () {
    let onlyBeatBoss = true;
    let floorMin = 0;
    let topN = 40;
    let showBasic = false;
    let epMin = "";
    let epMax = "";
    const ALL_RARITIES = ["RARE", "UNCOMMON", "COMMON", "SPECIAL", "CURSE", "UNKNOWN"];
    let activeRarities = new Set(ALL_RARITIES);
    let chartInstance = null;
    let relicChartInstance = null;
    let pickRateChartInstance = null;
    let rawData = [];
    let relicData = [];
    let pickRateData = [];
    let pickRateSortKey = "offered"; // offered | picked | pick_rate | card
    let pickRateSortAsc = false;

    const RARITY_COLORS = {
        BASIC: "#9e9e9e",
        COMMON: "#90a4ae",
        UNCOMMON: "#42a5f5",
        RARE: "#ffc107",
        SPECIAL: "#ab47bc",
        CURSE: "#424242",
        UNKNOWN: "#cbd5e1",
    };

    const RARITY_LABELS = {
        BASIC: "基础",
        COMMON: "普通",
        UNCOMMON: "罕见",
        RARE: "稀有",
        SPECIAL: "特殊",
        CURSE: "诅咒",
        UNKNOWN: "未分类",
    };

    async function fetchCardFreq() {
        const q = new URLSearchParams({
            floor_min: String(floorMin),
        });
        if (onlyBeatBoss) q.append("beat_boss", "1");
        if (epMin !== "") q.append("ep_min", String(epMin));
        if (epMax !== "") q.append("ep_max", String(epMax));
        const res = await fetch(`/api/deck/card_frequency?${q.toString()}`);
        if (!res.ok) throw new Error("加载卡牌频次失败: " + res.status);
        const data = await res.json();
        return Array.isArray(data) ? data : [];
    }

    async function fetchPickRates() {
        const q = new URLSearchParams();
        if (epMin !== "") q.append("ep_min", String(epMin));
        if (epMax !== "") q.append("ep_max", String(epMax));
        const res = await fetch(`/api/deck/pick_rates?${q.toString()}`);
        if (!res.ok) throw new Error("加载卡牌选择率失败: " + res.status);
        const data = await res.json();
        return Array.isArray(data) ? data : [];
    }

    async function fetchRelicFreq() {
        const q = new URLSearchParams();
        if (onlyBeatBoss) q.append("beat_boss", "1");
        if (epMin !== "") q.append("ep_min", String(epMin));
        if (epMax !== "") q.append("ep_max", String(epMax));
        const res = await fetch(`/api/deck/relic_frequency?${q.toString()}`);
        if (!res.ok) throw new Error("加载遗物频次失败: " + res.status);
        const data = await res.json();
        return Array.isArray(data) ? data : [];
    }

    function escapeHtml(s) {
        if (s == null) return "";
        return String(s)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;");
    }

    function destroyChart() {
        if (chartInstance) {
            chartInstance.destroy();
            chartInstance = null;
        }
    }

    function destroyRelicChart() {
        if (relicChartInstance) {
            relicChartInstance.destroy();
            relicChartInstance = null;
        }
    }

    const valueLabelPlugin = {
        id: "valueLabel",
        afterDatasetsDraw(chart) {
            const { ctx } = chart;
            ctx.save();
            ctx.font = "11px -apple-system, sans-serif";
            ctx.fillStyle = "#374151";
            ctx.textAlign = "left";
            ctx.textBaseline = "middle";
            chart.data.datasets.forEach((dataset, di) => {
                const meta = chart.getDatasetMeta(di);
                meta.data.forEach((bar, i) => {
                    const v = dataset.data[i];
                    if (v == null) return;
                    const { x, y } = bar.tooltipPosition();
                    ctx.fillText(String(v), x + 4, y);
                });
            });
            ctx.restore();
        },
    };

    // ===== 遗物频次图 =====
    const RELIC_TIER_COLORS = {
        Common: "#90a4ae",
        Uncommon: "#42a5f5",
        Rare: "#ffc107",
        Boss: "#ef4444",
        Shop: "#22c55e",
        Event: "#a855f7",
        Starter: "#9e9e9e",
    };
    const RELIC_DEFAULT_COLOR = "#78909c";

    function drawRelicChart() {
        const canvas = document.getElementById("relic-canvas");
        if (!canvas) return;
        destroyRelicChart();
        if (!relicData.length) {
            const wrap = canvas.parentElement;
            if (wrap) wrap.innerHTML = `<div class="empty">暂无遗物数据</div>`;
            return;
        }

        const top = relicData.slice(0, topN);
        const labels = top.map((r) => r.display);
        const data = top.map((r) => r.freq);
        const colors = top.map(() => RELIC_DEFAULT_COLOR);

        relicChartInstance = new Chart(canvas, {
            type: "bar",
            data: {
                labels,
                datasets: [
                    {
                        label: "出现次数",
                        data,
                        backgroundColor: colors,
                        borderColor: "#1f2937",
                        borderWidth: 0,
                    },
                ],
            },
            options: {
                indexAxis: "y",
                responsive: true,
                maintainAspectRatio: false,
                layout: { padding: { right: 36 } },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: (ctx) => {
                                const r = top[ctx.dataIndex];
                                const totalEps = r.total_eps || 1;
                                const pct = ((r.freq / totalEps) * 100).toFixed(1);
                                return `${r.display}：${r.freq} 次（${pct}% 局持有）`;
                            },
                            afterLabel: (ctx) => {
                                const r = top[ctx.dataIndex];
                                return `引擎 ID: ${r.relic_en_id}`;
                            },
                        },
                    },
                },
                scales: {
                    x: { beginAtZero: true, title: { display: true, text: "出现次数" } },
                    y: { ticks: { autoSkip: false, font: { size: 11 } } },
                },
            },
            plugins: [valueLabelPlugin],
        });
    }

    function updateRelicTitle() {
        const titleEl = document.getElementById("relic-chart-title");
        if (!titleEl) return;
        const winFilter = onlyBeatBoss ? "通关局" : "全部局";
        const rangeNote = (epMin !== "" || epMax !== "")
            ? ` · 局 ${epMin || "0"}~${epMax || "最新"}`
            : "";
        titleEl.textContent = `遗物出现频次（${winFilter}${rangeNote}）`;
    }

    function destroyPickRateChart() {
        if (pickRateChartInstance) {
            pickRateChartInstance.destroy();
            pickRateChartInstance = null;
        }
    }

    function getPickRateFiltered() {
        // 只展示 offered >= 5 的卡（减少噪声）
        return pickRateData.filter((r) => r.offered >= 5);
    }

    function pickRateColor(rate) {
        // 渐变：红 (0%) -> 黄 (50%) -> 绿 (100%)
        if (rate <= 0.5) {
            const t = rate / 0.5;
            const r = Math.round(239 * (1 - t) + 234 * t);
            const g = Math.round(68 * (1 - t) + 179 * t);
            const b = Math.round(68 * (1 - t) + 8 * t);
            return `rgb(${r},${g},${b})`;
        } else {
            const t = (rate - 0.5) / 0.5;
            const r = Math.round(234 * (1 - t) + 34 * t);
            const g = Math.round(179 * (1 - t) + 197 * t);
            const b = Math.round(8 * (1 - t) + 94 * t);
            return `rgb(${r},${g},${b})`;
        }
    }

    function drawPickRateChart() {
        const canvas = document.getElementById("pick-rate-canvas");
        if (!canvas) return;
        destroyPickRateChart();
        const filtered = getPickRateFiltered();
        if (!filtered.length) {
            const wrap = canvas.parentElement;
            if (wrap) wrap.innerHTML = `<div class="empty">暂无选择率数据（需至少 5 次提供）</div>`;
            return;
        }

        // 按 pick_rate 降序排列展示
        const sorted = [...filtered].sort((a, b) => b.pick_rate - a.pick_rate);
        const top = sorted.slice(0, topN);
        const labels = top.map((r) => r.zh || r.card);
        const data = top.map((r) => Math.round(r.pick_rate * 1000) / 10); // percent with 1 decimal
        const colors = top.map((r) => pickRateColor(r.pick_rate));

        const height = Math.max(400, top.length * 22);
        canvas.parentElement.style.height = height + "px";

        pickRateChartInstance = new Chart(canvas, {
            type: "bar",
            data: {
                labels,
                datasets: [
                    {
                        label: "选择率 %",
                        data,
                        backgroundColor: colors,
                        borderColor: "#1f2937",
                        borderWidth: 0,
                    },
                ],
            },
            options: {
                indexAxis: "y",
                responsive: true,
                maintainAspectRatio: false,
                layout: { padding: { right: 50 } },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: (ctx) => {
                                const r = top[ctx.dataIndex];
                                return `${r.zh || r.card}：提供 ${r.offered} 次，选择 ${r.picked} 次（${(r.pick_rate * 100).toFixed(1)}%）`;
                            },
                            afterLabel: (ctx) => {
                                const r = top[ctx.dataIndex];
                                return `引擎 ID: ${r.card}`;
                            },
                        },
                    },
                },
                scales: {
                    x: {
                        beginAtZero: true,
                        max: 100,
                        title: { display: true, text: "选择率 %" },
                        ticks: { callback: (v) => v + "%" },
                    },
                    y: { ticks: { autoSkip: false, font: { size: 11 } } },
                },
            },
            plugins: [
                {
                    id: "pickRateValueLabel",
                    afterDatasetsDraw(chart) {
                        const { ctx } = chart;
                        ctx.save();
                        ctx.font = "11px -apple-system, sans-serif";
                        ctx.fillStyle = "#374151";
                        ctx.textAlign = "left";
                        ctx.textBaseline = "middle";
                        chart.data.datasets.forEach((dataset, di) => {
                            const meta = chart.getDatasetMeta(di);
                            meta.data.forEach((bar, i) => {
                                const v = dataset.data[i];
                                if (v == null) return;
                                const { x, y } = bar.tooltipPosition();
                                ctx.fillText(v.toFixed(1) + "%", x + 4, y);
                            });
                        });
                        ctx.restore();
                    },
                },
            ],
        });
    }

    function updatePickRateTitle() {
        const titleEl = document.getElementById("pick-rate-chart-title");
        if (!titleEl) return;
        const rangeNote = (epMin !== "" || epMax !== "")
            ? ` · 局 ${epMin || "0"}~${epMax || "最新"}`
            : "";
        const count = getPickRateFiltered().length;
        titleEl.textContent = `卡牌选择率（被提供时选择的比例${rangeNote}） · ${count} 种卡牌`;
    }

    function renderPickRateTable() {
        const tbody = document.querySelector("#pick-rate-table tbody");
        if (!tbody) return;
        const filtered = getPickRateFiltered();

        // 排序
        const sorted = [...filtered].sort((a, b) => {
            let cmp = 0;
            if (pickRateSortKey === "card") {
                const aLabel = a.zh || a.card;
                const bLabel = b.zh || b.card;
                cmp = aLabel.localeCompare(bLabel, "zh-CN");
            } else {
                cmp = (a[pickRateSortKey] || 0) - (b[pickRateSortKey] || 0);
            }
            return pickRateSortAsc ? cmp : -cmp;
        });

        tbody.innerHTML = sorted
            .map(
                (r, idx) => {
                    const pct = (r.pick_rate * 100).toFixed(1);
                    const barWidth = Math.round(r.pick_rate * 100);
                    const barColor = pickRateColor(r.pick_rate);
                    return `
                    <tr>
                        <td>${idx + 1}</td>
                        <td>${escapeHtml(r.zh || r.card)}</td>
                        <td class="muted">${escapeHtml(r.card)}</td>
                        <td class="num">${r.offered}</td>
                        <td class="num">${r.picked}</td>
                        <td class="num">
                            <div style="display:flex;align-items:center;gap:6px;justify-content:flex-end;">
                                <div style="width:60px;height:10px;background:#e5e7eb;border-radius:5px;overflow:hidden;">
                                    <div style="width:${barWidth}%;height:100%;background:${barColor};border-radius:5px;"></div>
                                </div>
                                <span>${pct}%</span>
                            </div>
                        </td>
                    </tr>
                `;
                }
            )
            .join("");
    }

    function setupPickRateTableSort() {
        const ths = document.querySelectorAll("#pick-rate-table th[data-sort]");
        ths.forEach((th) => {
            th.style.cursor = "pointer";
            th.addEventListener("click", () => {
                const key = th.dataset.sort;
                if (pickRateSortKey === key) {
                    pickRateSortAsc = !pickRateSortAsc;
                } else {
                    pickRateSortKey = key;
                    pickRateSortAsc = key === "card"; // card 默认升序，数值默认降序
                }
                // 更新排序指示
                ths.forEach((t) => t.classList.remove("sort-asc", "sort-desc"));
                th.classList.add(pickRateSortAsc ? "sort-asc" : "sort-desc");
                renderPickRateTable();
            });
        });
    }

    function getFilteredData() {
        return rawData.filter((r) => {
            const rar = r.rarity || "UNKNOWN";
            if (rar === "BASIC" && !showBasic) return false;
            if (rar === "BASIC") return true;
            return activeRarities.has(rar);
        });
    }

    function drawChart() {
        const canvas = document.getElementById("deck-canvas");
        if (!canvas) return;
        destroyChart();
        const filtered = getFilteredData();
        const top = filtered.slice(0, topN);
        if (!top.length) return;

        const labels = top.map((r) => r.display);
        const data = top.map((r) => r.freq);
        const colors = top.map((r) => RARITY_COLORS[r.rarity || "UNKNOWN"]);

        chartInstance = new Chart(canvas, {
            type: "bar",
            data: {
                labels,
                datasets: [
                    {
                        label: "出现次数",
                        data,
                        backgroundColor: colors,
                        borderColor: "#1f2937",
                        borderWidth: 0,
                    },
                ],
            },
            options: {
                indexAxis: "y",
                responsive: true,
                maintainAspectRatio: false,
                layout: { padding: { right: 36 } },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: (ctx) => {
                                const r = top[ctx.dataIndex];
                                const up = r.upgraded ? "（已升级）" : "";
                                const rar = RARITY_LABELS[r.rarity || "UNKNOWN"];
                                return `${r.display}${up}：${r.freq} 次 · ${rar}`;
                            },
                            afterLabel: (ctx) => {
                                const r = top[ctx.dataIndex];
                                return `引擎 ID: ${r.card_en_id}${r.upgraded ? "+1" : ""}`;
                            },
                        },
                    },
                },
                scales: {
                    x: { beginAtZero: true, title: { display: true, text: "出现次数" } },
                    y: { ticks: { autoSkip: false, font: { size: 11 } } },
                },
            },
            plugins: [valueLabelPlugin],
        });
    }

    function renderLegend() {
        const legendEl = document.getElementById("deck-legend");
        if (!legendEl) return;
        const items = ALL_RARITIES.concat(["BASIC"])
            .map((k) => {
                const visible = (k === "BASIC") ? showBasic : activeRarities.has(k);
                const opacity = visible ? 1.0 : 0.35;
                return `<span class="rarity-legend" style="opacity:${opacity};">
                    <span class="legend-swatch" style="background:${RARITY_COLORS[k]}"></span>
                    ${RARITY_LABELS[k]}
                </span>`;
            })
            .join("");
        legendEl.innerHTML = items;
    }

    function renderTable() {
        const tbody = document.querySelector("#deck-table tbody");
        if (!tbody) return;
        const filtered = getFilteredData();
        const total = filtered.reduce((s, r) => s + r.freq, 0) || 1;
        const top = filtered.slice(0, topN);
        tbody.innerHTML = top
            .map(
                (r, idx) => {
                    const rar = r.rarity || "UNKNOWN";
                    const rarLabel = RARITY_LABELS[rar];
                    return `
                    <tr>
                        <td>${idx + 1}</td>
                        <td><span class="card-chip rarity-${rar}${r.upgraded ? " upgraded" : ""}">${escapeHtml(r.display)}</span></td>
                        <td><span class="rarity-badge rarity-${rar}">${rarLabel}</span></td>
                        <td class="muted">${escapeHtml(r.card_en_id)}${r.upgraded ? "+1" : ""}</td>
                        <td class="num">${r.freq}</td>
                        <td class="num muted">${((r.freq / total) * 100).toFixed(1)}%</td>
                    </tr>
                `;
                }
            )
            .join("");
    }

    function renderRarityChips() {
        const wrap = document.getElementById("deck-rarity-chips");
        if (!wrap) return;
        wrap.innerHTML = ALL_RARITIES
            .map((k) => {
                const active = activeRarities.has(k);
                return `<span class="chip rarity-chip rarity-${k} ${active ? "active" : ""}" data-rarity="${k}">${RARITY_LABELS[k]}</span>`;
            })
            .join(" ");
        wrap.querySelectorAll(".chip[data-rarity]").forEach((ch) => {
            ch.addEventListener("click", () => {
                const k = ch.dataset.rarity;
                if (activeRarities.has(k)) activeRarities.delete(k);
                else activeRarities.add(k);
                renderRarityChips();
                renderLegend();
                drawChart();
                renderTable();
                updateTitle();
            });
        });
    }

    function updateTitle() {
        const titleEl = document.getElementById("deck-chart-title");
        if (!titleEl) return;
        const winFilter = onlyBeatBoss ? "通关局" : "全部局";
        const basicNote = showBasic ? "含基础卡" : "已过滤基础卡";
        const rangeNote = (epMin !== "" || epMax !== "")
            ? ` · 局 ${epMin || "0"}~${epMax || "最新"}`
            : "";
        titleEl.textContent = `卡牌出现频次（${winFilter} · 楼层 ≥ ${floorMin} · ${basicNote}${rangeNote}）`;
    }

    async function reload() {
        const chartCard = document.getElementById("deck-chart-card");
        if (chartCard) chartCard.querySelector(".deck-loading").style.display = "block";
        try {
            const [cards, relics, picks] = await Promise.all([
                fetchCardFreq(), fetchRelicFreq(), fetchPickRates(),
            ]);
            rawData = cards;
            relicData = relics;
            pickRateData = picks;
        } catch (e) {
            rawData = [];
            relicData = [];
            pickRateData = [];
            console.error(e);
        }
        updateTitle();
        updateRelicTitle();
        updatePickRateTitle();
        renderLegend();
        drawChart();
        drawRelicChart();
        drawPickRateChart();
        renderTable();
        renderPickRateTable();
        if (chartCard) chartCard.querySelector(".deck-loading").style.display = "none";
    }

    async function render(root, params) {
        destroyChart();
        destroyRelicChart();
        destroyPickRateChart();
        root.innerHTML = window.UI.loading("载入牌组数据...");

        root.innerHTML = `
            <h2>牌组组成</h2>
            <div class="controls">
                <label><input type="checkbox" id="deck-only-win" ${onlyBeatBoss ? "checked" : ""}/> 只看通关局</label>
                <label title="累计绝对楼层下限。act1: 1-17，act2: 18-34，act3: 35-51">绝对楼层下限（act1 1-17 / act2 18-34 / act3 35-51）</label>
                <input type="number" id="deck-floor-min" value="${floorMin}" min="0" max="51" style="width: 60px;"/>
                <label>显示前</label>
                <input type="number" id="deck-top-n" value="${topN}" min="5" max="200" style="width: 60px;"/>
                <span style="color:#6b7280; font-size: 12px;">张</span>
                <label><input type="checkbox" id="deck-show-basic" ${showBasic ? "checked" : ""}/> 显示基础卡</label>
                <label>局范围</label>
                <input type="number" id="deck-ep-min" placeholder="起始局" value="${escapeHtml(String(epMin))}" style="width: 80px;"/>
                <span style="color:#6b7280; font-size: 12px;">~</span>
                <input type="number" id="deck-ep-max" placeholder="结束局" value="${escapeHtml(String(epMax))}" style="width: 80px;"/>
            </div>
            <div class="controls" style="padding-top: 6px; padding-bottom: 6px;">
                <span style="font-size:12px;color:#4b5563;">稀有度过滤：</span>
                <div id="deck-rarity-chips" style="display:flex;flex-wrap:wrap;gap:4px;"></div>
            </div>
            <div class="chart-card" id="deck-chart-card">
                <h3 id="deck-chart-title">卡牌出现频次</h3>
                <div id="deck-legend" class="rarity-legend-wrap"></div>
                <div class="deck-loading loading">加载中...</div>
                <div class="chart-wrap" style="height: 700px;"><canvas id="deck-canvas"></canvas></div>
            </div>
            <div class="chart-card">
                <h3>详细列表</h3>
                <table class="simple" id="deck-table">
                    <thead>
                        <tr>
                            <th>排名</th><th>卡牌</th><th>稀有度</th><th>引擎 ID</th><th>频次</th><th>占比</th>
                        </tr>
                    </thead>
                    <tbody></tbody>
                </table>
            </div>
            <div class="chart-card" id="relic-chart-card">
                <h3 id="relic-chart-title">遗物出现频次</h3>
                <div class="chart-wrap" style="height: 500px;"><canvas id="relic-canvas"></canvas></div>
            </div>
            <div class="chart-card" id="pick-rate-chart-card">
                <h3 id="pick-rate-chart-title">卡牌选择率（被提供时选择的比例）</h3>
                <div class="chart-wrap" style="height: 700px;"><canvas id="pick-rate-canvas"></canvas></div>
            </div>
            <div class="chart-card">
                <h3>选择率详细列表</h3>
                <p style="font-size:12px;color:#6b7280;margin:0 0 8px 0;">只显示被提供 ≥5 次的卡牌 · 点击列头排序</p>
                <table class="simple" id="pick-rate-table">
                    <thead>
                        <tr>
                            <th>排名</th>
                            <th data-sort="card">卡牌</th>
                            <th>引擎 ID</th>
                            <th data-sort="offered" class="num">提供次数</th>
                            <th data-sort="picked" class="num">选择次数</th>
                            <th data-sort="pick_rate" class="num sort-desc">选择率</th>
                        </tr>
                    </thead>
                    <tbody></tbody>
                </table>
            </div>
        `;

        renderRarityChips();
        setupPickRateTableSort();

        root.querySelector("#deck-only-win").addEventListener("change", async (e) => {
            onlyBeatBoss = e.target.checked;
            await reload();
        });
        root.querySelector("#deck-floor-min").addEventListener("change", async (e) => {
            floorMin = parseInt(e.target.value, 10) || 0;
            await reload();
        });
        root.querySelector("#deck-top-n").addEventListener("change", async (e) => {
            topN = Math.max(5, parseInt(e.target.value, 10) || 40);
            updateTitle();
            updateRelicTitle();
            drawChart();
            drawRelicChart();
            renderTable();
        });
        root.querySelector("#deck-show-basic").addEventListener("change", (e) => {
            showBasic = e.target.checked;
            updateTitle();
            renderLegend();
            drawChart();
            renderTable();
        });
        root.querySelector("#deck-ep-min").addEventListener("change", async (e) => {
            const v = e.target.value.trim();
            epMin = v === "" ? "" : parseInt(v, 10) || "";
            await reload();
        });
        root.querySelector("#deck-ep-max").addEventListener("change", async (e) => {
            const v = e.target.value.trim();
            epMax = v === "" ? "" : parseInt(v, 10) || "";
            await reload();
        });

        await reload();
    }

    window.PAGES.deck = { render };
})();

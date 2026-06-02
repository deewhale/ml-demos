// 路由 + 公共逻辑：hash 路由 -> window.PAGES[name].render(root, params)
// i18n 缓存初始化（P4 才填充，现在空表也能正常工作）
// P6 新增：通用 loading/empty/error 渲染 helper + 重新扫描按钮

(function () {
    // 全局 i18n 缓存：{card: {en_id: zh_name}, relic: {...}, monster: {...}, event: {...}, potion: {...}}
    window.I18N = { card: {}, relic: {}, monster: {}, event: {}, potion: {} };

    // 房间类型中文化（floor_events.room / combats.room 等）
    window.ROOM_ZH = {
        monster: "普通战",
        elite: "精英",
        boss: "Boss 战",
        rest: "休息",
        shop: "商店",
        event: "事件",
        treasure: "宝箱",
        unknown: "未知",
    };
    window.roomZh = function (room) {
        if (!room) return "";
        return window.ROOM_ZH[room] || room;
    };

    // 指标名英→中映射，用于 chart title / axis label / tooltip
    window.METRIC_ZH = {
        mean_reward: "奖励（每局）",
        mean_steps: "步数（每局）",
        mean_floor: "平均楼层",
        beat_boss_count: "击败 Boss",
        policy_loss: "策略损失",
        value_loss: "价值损失",
        entropy: "策略熵",
        approx_kl: "近似 KL 散度",
        clip_frac: "裁剪比例",
        update_secs: "更新耗时(秒)",
        wrapper_calls: "Wrapper 调用",
        won_game_rate: "通关率",
        reached_boss_rate: "抵达 Boss 率",
        act1_boss_beat_rate: "一层 Boss 击败率",
        act2_boss_beat_rate: "二层 Boss 击败率",
        beat_boss_rate: "击败 Boss 率（已废弃）",
        floor_mean: "平均楼层",
        floor_max: "最高楼层",
        avg_steps: "平均步数",
        num_seeds: "种子数",
        completed: "完成数",
        secs: "耗时(秒)",
        episodes_done: "已完成局数",

        reward: "奖励",
        steps: "步数",
        floor: "楼层",
        floor_reached: "抵达楼层",
        beat_boss: "击败 Boss",
        ep: "局序号",
    };
    window.metricZh = function (key) {
        if (!key) return "";
        return window.METRIC_ZH[key] || key;
    };

    // 通用 UI helper：loading / empty / error。各页面可用。
    // 用法：window.UI.loading('载入中...') 返回 HTML 字符串
    window.UI = {
        loading: function (text) {
            return `<div class="loading">${text || "加载中..."}</div>`;
        },
        empty: function (text) {
            return `<div class="empty">${text || "暂无数据，请先点击\"重新扫描\""}</div>`;
        },
        error: function (msg) {
            const text = (msg == null || msg === "") ? "未知错误" : String(msg);
            return `<div class="error">加载失败: ${text}</div>`;
        },
    };

    // 通用兜底查询：zh_name 缺失则用 en_name 兜底，再缺用 en_id
    window.lookupI18N = function (kind, en_id, upgraded) {
        if (!en_id) return "";
        const table = window.I18N[kind] || {};
        const entry = table[en_id];
        // entry 可能是字符串（旧版）或 {en_name, zh_name}
        let label;
        if (entry && typeof entry === "object") {
            label = entry.zh_name || entry.en_name || en_id;
        } else if (typeof entry === "string") {
            label = entry || en_id;
        } else {
            label = en_id;
        }
        if (upgraded) label = label + "+1";
        return label;
    };

    // 启动时一次性拉 5 类 lookup（P4 后端实现：/api/lookup/all 不带 kind 返回完整字典）
    async function preloadI18N() {
        try {
            const res = await fetch("/api/lookup/all");
            if (!res.ok) {
                console.warn("[i18n] preload failed:", res.status);
                return;
            }
            const all = await res.json();
            // all = {card: {en_id: {en_id,en_name,zh_name,zh_desc,source}}, relic: {...}, ...}
            const kinds = ["card", "relic", "monster", "event", "potion"];
            for (const k of kinds) {
                if (all[k] && typeof all[k] === "object") {
                    window.I18N[k] = all[k];
                }
            }
        } catch (e) {
            console.warn("[i18n] preload error:", e);
        }
    }

    // 解析 hash: #/training?foo=bar -> {name: 'training', params: {foo:'bar'}}
    function parseHash() {
        const raw = window.location.hash || "#/training";
        const stripped = raw.startsWith("#/") ? raw.slice(2) : raw.slice(1);
        const [name, query] = stripped.split("?");
        const params = {};
        if (query) {
            for (const seg of query.split("&")) {
                const [k, v] = seg.split("=");
                if (k) params[decodeURIComponent(k)] = decodeURIComponent(v || "");
            }
        }
        return { name: name || "training", params };
    }

    function setActiveNav(name) {
        document.querySelectorAll("#sidebar a").forEach((a) => {
            const href = a.getAttribute("href") || "";
            const target = href.startsWith("#/") ? href.slice(2).split("?")[0] : "";
            if (target === name) a.classList.add("active");
            else a.classList.remove("active");
        });
    }

    async function renderRoute() {
        const root = document.getElementById("app-root");
        if (!root) return;
        const { name, params } = parseHash();
        setActiveNav(name);
        const page = window.PAGES[name];
        if (!page || typeof page.render !== "function") {
            root.innerHTML = `<div class="error">未知页面: ${name}</div>`;
            return;
        }
        try {
            await page.render(root, params);
        } catch (e) {
            console.error(e);
            root.innerHTML = `<div class="error">页面渲染错误: ${e.message}</div>`;
        }
    }

    // P6：重新扫描按钮 → POST /api/admin/rescan → 重新拉 i18n + 重渲当前页
    function setupRescanButton() {
        const btn = document.getElementById("rescan-btn");
        const status = document.getElementById("rescan-status");
        if (!btn) return;
        btn.addEventListener("click", async () => {
            btn.disabled = true;
            status.className = "";
            status.textContent = "扫描中...";
            try {
                const res = await fetch("/api/admin/rescan", { method: "POST" });
                if (!res.ok) {
                    const txt = await res.text();
                    throw new Error(`HTTP ${res.status}: ${txt}`);
                }
                const j = await res.json();
                status.className = "ok";
                status.textContent = `完成: ${j.episodes_parsed} 局 / ${j.elapsed_sec}s`;
                // 重新拉 i18n（mapping 可能有新增）+ 重渲当前页
                await preloadI18N();
                await renderRoute();
            } catch (e) {
                status.className = "err";
                status.textContent = `失败: ${e.message || e}`;
            } finally {
                btn.disabled = false;
            }
        });
    }

    window.addEventListener("hashchange", renderRoute);
    window.addEventListener("DOMContentLoaded", async () => {
        setupRescanButton();
        await preloadI18N();
        if (!window.location.hash) {
            // replaceState 不触发 hashchange，避免双重 render
            history.replaceState(null, "", "#/training");
        }
        renderRoute();
    });
})();

// 单局回放页：episode-centric，无 batch 概念
// - 左侧：按局数排序的 episode 列表
// - 右侧：选中局的终局总结 + Act 分组 timeline
// - 控件：三态过滤（全部 / 通关 / 失败）+ 排序选择
// - 中文化：所有 label / 表头 / room 用 roomZh
// - Act 颜色分组 / 终局总结面板 / 战斗 turn 级 lazy fetch 保留不变

(function () {
    let episodes = [];
    let resultFilter = "all";  // "all" / "won" / "lost"
    let sortKey = "ep";
    let sortOrder = "desc";
    let selectedEp = null;
    let currentTimeline = null;
    let hpChart = null;  // HP 曲线 Chart.js 实例

    const SORT_KEYS_ZH = {
        reward: "奖励",
        floor_reached: "抵达楼层",
        steps: "步数",
        ep: "局序号",
        secs: "耗时",
    };

    const ACT_LABELS = {
        1: "第一层 · 凋零城外",
        2: "第二层 · 万象森林",
        3: "第三层 · 灰心高塔",
        4: "第四层 · 尖塔之心",
    };

    const RARITY_GROUPS = [
        { key: "RARE", label: "稀有" },
        { key: "UNCOMMON", label: "罕见" },
        { key: "COMMON", label: "普通" },
        { key: "BASIC", label: "基础" },
        { key: "SPECIAL", label: "特殊" },
        { key: "CURSE", label: "诅咒" },
        { key: "UNKNOWN", label: "未分类" },
    ];

    async function fetchEpisodeList() {
        const q = new URLSearchParams({
            sort: sortKey,
            order: sortOrder,
            limit: "200",
        });
        if (resultFilter === "won") q.append("beat_boss", "1");
        else if (resultFilter === "lost") q.append("beat_boss", "0");
        const res = await fetch(`/api/episode/list?${q.toString()}`);
        if (!res.ok) throw new Error("加载局列表失败: " + res.status);
        const data = await res.json();
        return Array.isArray(data) ? data : [];
    }

    async function fetchTimeline(ep) {
        const res = await fetch(`/api/episode/${ep}/timeline`);
        if (!res.ok) throw new Error("加载时间轴失败: " + res.status);
        return res.json();
    }

    async function fetchCombatDetail(ep, act, floor) {
        const url = `/api/episode/${ep}/combat/${act}/${floor}/detail`;
        const res = await fetch(url);
        if (!res.ok) throw new Error("加载战斗详情失败: " + res.status);
        return res.json();
    }

    const combatDetailCache = new Map();

    function fmt(v, digits = 2) {
        if (v == null) return "—";
        return Number(v).toFixed(digits);
    }

    function escapeHtml(s) {
        if (s == null) return "";
        return String(s)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;");
    }

    function roomBadge(room) {
        const colorMap = {
            monster: ["#dbeafe", "#1d4ed8"],
            elite: ["#fee2e2", "#b91c1c"],
            boss: ["#1f2937", "#fbbf24"],
            rest: ["#fef3c7", "#92400e"],
            shop: ["#dcfce7", "#166534"],
            event: ["#ede9fe", "#6d28d9"],
            treasure: ["#fef9c3", "#a16207"],
            unknown: ["#e5e7eb", "#374151"],
        };
        const [bg, fg] = colorMap[room] || ["#e5e7eb", "#374151"];
        const label = window.roomZh(room) || room || "?";
        return `<span class="room-badge" style="background:${bg};color:${fg};">${label}</span>`;
    }

    function exitReasonZh(r) {
        const map = {
            victory: "胜利",
            defeat: "失败",
            timeout: "超时",
            orphan_no_exit: "未结束",
        };
        return r ? (map[r] || r) : "";
    }

    function actLabel(act) {
        return ACT_LABELS[act] || `第${act}层`;
    }

    function renderEpisodeList(container) {
        if (!episodes.length) {
            container.innerHTML = window.UI.empty("无符合条件的局");
            return;
        }
        const rows = episodes
            .map((e) => {
                const cls = e.ep === selectedEp ? "selected" : "";
                const beat = e.beat_boss ? "通关" : "败";
                const beatCls = e.beat_boss ? "won" : "lost";
                return `
                    <tr class="${cls}" data-ep="${e.ep}">
                        <td>${e.ep}</td>
                        <td class="num">${fmt(e.reward, 1)}</td>
                        <td class="num">${e.floor_reached}</td>
                        <td class="num">${e.steps}</td>
                        <td class="cell-center beat-cell ${beatCls}">${beat}</td>
                    </tr>
                `;
            })
            .join("");
        container.innerHTML = `
            <table class="simple ep-table">
                <thead>
                    <tr>
                        <th>局</th><th>奖励</th><th>楼层</th><th>步数</th><th>结果</th>
                    </tr>
                </thead>
                <tbody>${rows}</tbody>
            </table>
        `;
        container.querySelectorAll("tr[data-ep]").forEach((tr) => {
            tr.addEventListener("click", async () => {
                selectedEp = parseInt(tr.dataset.ep, 10);
                renderEpisodeList(container);
                await loadTimeline();
            });
        });
    }

    function renderEndgame(eg) {
        if (!eg) return "";
        const wonCls = eg.won ? "won" : "lost";
        const title = eg.won ? "通关" : (eg.death_at ? "失败" : "未完成");
        let titleSub = "";
        if (!eg.won && eg.death_at) {
            const a = eg.death_at.act;
            const f = eg.death_at.floor;
            const enemies = (eg.death_at.enemies_zh || []).join("、") || "未知敌人";
            titleSub = `<span class="endgame-sub">死于 ${actLabel(a)} · 第 ${f} 楼 · ${escapeHtml(enemies)}</span>`;
        } else if (eg.won) {
            titleSub = `<span class="endgame-sub">通关 ${actLabel(eg.final_act || 3)} · 第 ${eg.final_floor} 楼</span>`;
        } else {
            titleSub = `<span class="endgame-sub">到达 ${actLabel(eg.final_act || 1)} · 第 ${eg.final_floor} 楼</span>`;
        }

        const hpCur = eg.final_hp ?? 0;
        const hpMax = eg.final_hp_max ?? 1;
        const hpRatio = Math.max(0, Math.min(1, hpCur / hpMax));
        const hpPct = (hpRatio * 100).toFixed(0);
        let hpBarColor = "#16a34a";
        if (hpRatio < 0.3) hpBarColor = "#dc2626";
        else if (hpRatio < 0.6) hpBarColor = "#f59e0b";

        const finalDeck = eg.final_deck || {};
        const deckGroupsHtml = RARITY_GROUPS
            .filter((g) => (finalDeck[g.key] || []).length > 0)
            .map((g) => {
                const chips = (finalDeck[g.key] || [])
                    .map((c) => {
                        const cls = `card-chip rarity-${c.rarity || "UNKNOWN"}${c.upgraded ? " upgraded" : ""}`;
                        return `<span class="${cls}" title="${escapeHtml(c.card_en_id)}">${escapeHtml(c.display)}×${c.count}</span>`;
                    })
                    .join(" ");
                return `
                    <div class="endgame-deck-row">
                        <span class="endgame-deck-label rarity-label-${g.key}">${g.label}</span>
                        <div class="endgame-deck-chips">${chips}</div>
                    </div>
                `;
            })
            .join("");

        const relicsHtml = (eg.final_relics || [])
            .map((r) => `<span class="relic-chip" title="${escapeHtml(r.en_id)}">${escapeHtml(r.display)}</span>`)
            .join(" ");

        return `
            <div class="endgame-panel ${wonCls}">
                <div class="endgame-title">
                    <span class="endgame-status">${title}</span>
                    ${titleSub}
                </div>
                <div class="endgame-stats">
                    <div class="endgame-stat">
                        <span class="endgame-stat-label">最终 HP</span>
                        <div class="hp-bar"><div class="hp-fill" style="width:${hpPct}%;background:${hpBarColor};"></div></div>
                        <span class="endgame-stat-value">${hpCur} / ${hpMax}</span>
                    </div>
                    <div class="endgame-stat">
                        <span class="endgame-stat-label">奖励</span>
                        <span class="endgame-stat-value">${fmt(eg.total_reward, 2)}</span>
                    </div>
                    <div class="endgame-stat">
                        <span class="endgame-stat-label">总步数</span>
                        <span class="endgame-stat-value">${eg.total_steps ?? "—"}</span>
                    </div>
                    <div class="endgame-stat">
                        <span class="endgame-stat-label">牌组数</span>
                        <span class="endgame-stat-value">${eg.deck_size ?? "—"} 张</span>
                    </div>
                </div>
                <details class="endgame-section" open>
                    <summary>最终牌组</summary>
                    <div class="endgame-deck">${deckGroupsHtml || "<i class='muted'>无牌组数据</i>"}</div>
                </details>
                <details class="endgame-section" open>
                    <summary>最终遗物 (${(eg.final_relics || []).length})</summary>
                    <div class="endgame-relics">${relicsHtml || "<i class='muted'>无遗物</i>"}</div>
                </details>
            </div>
        `;
    }

    function renderDecision(dec) {
        if (!dec || !dec.phase) return "";

        if (dec.phase === "CARD_REWARDS") {
            // 遗物奖励子步骤（如战胜 elite 后的 relic reward）
            if (dec.type === "relic_reward" && dec.chosen_relic) {
                const relicZh = escapeHtml(dec.chosen_relic_zh || dec.chosen_relic);
                return `
                    <div class="decision relic-reward">
                        <span class="decision-label">遗物奖励：</span>
                        <span class="relic-option chosen">${relicZh}</span>
                    </div>
                `;
            }
            const options = (dec.options || []);
            const chosenCard = dec.chosen_card || null;
            const isSkip = dec.type === "skip" || !chosenCard;
            const chips = options.map((o) => {
                // 选项可能是卡牌或遗物
                if (o.relic) {
                    const label = escapeHtml(o.zh || o.relic || "?");
                    return `<span class="relic-option">${label}</span>`;
                }
                const label = escapeHtml(o.zh || o.card || "?");
                const isChosen = !isSkip && (o.card === chosenCard);
                if (isChosen) {
                    return `<span class="card-option chosen">${label}</span>`;
                }
                return `<span class="card-option">${label}</span>`;
            }).join("");
            const skipTag = isSkip ? `<span class="skip-label">跳过</span>` : "";
            return `
                <div class="decision card-reward">
                    <span class="decision-label">卡牌奖励：</span>
                    ${chips}${skipTag}
                </div>
            `;
        }

        if (dec.phase === "REST") {
            if (dec.type === "upgrade" && dec.chosen_card) {
                const cardZh = escapeHtml(dec.chosen_card_zh || dec.chosen_card);
                // 构造可升级卡列表
                const options = (dec.options || []);
                const optChips = options.map((o) => {
                    const label = escapeHtml(o.zh || o.card || "?");
                    const isChosen = o.card === dec.chosen_card;
                    if (isChosen) {
                        return `<span class="card-option chosen">${label}</span>`;
                    }
                    return `<span class="card-option">${label}</span>`;
                }).join("");
                return `
                    <div class="decision rest-decision">
                        <span class="decision-label">休息站：升级</span>
                        ${optChips || `<span class="card-option chosen">${cardZh}</span>`}
                    </div>
                `;
            }
            return `<div class="decision rest-decision"><span class="decision-label">休息站：</span>休息（回复 HP）</div>`;
        }

        if (dec.phase === "SHOP") {
            let actionHtml = "";
            if (dec.type === "buy") {
                const itemZh = escapeHtml(
                    dec.chosen_card_zh || dec.chosen_card ||
                    dec.chosen_relic_zh || dec.chosen_relic ||
                    dec.chosen_potion_zh || dec.chosen_potion || ""
                );
                const typeLabel = dec.item_type === "potion" ? "药水" :
                    dec.item_type === "relic" ? "遗物" :
                    (dec.item_type && dec.item_type.includes("card")) ? "卡牌" : "";
                actionHtml = `购买${typeLabel ? "（" + typeLabel + "）" : ""}：<span class="card-option chosen">${itemZh}</span>`;
            } else if (dec.type === "remove") {
                const cardZh = escapeHtml(dec.chosen_card_zh || dec.chosen_card || "");
                actionHtml = `移除卡牌：<span class="card-option chosen">${cardZh}</span>`;
            } else if (dec.type === "leave") {
                actionHtml = "离开";
            } else {
                actionHtml = "浏览";
            }
            return `<div class="decision shop-decision"><span class="decision-label">商店：</span>${actionHtml}</div>`;
        }

        if (dec.phase === "BOSS_REWARDS") {
            const relicZh = escapeHtml(dec.chosen_relic_zh || dec.chosen_relic || "");
            const options = (dec.options || []);
            const chips = options.map((o) => {
                const label = escapeHtml(o.zh || o.relic || "?");
                const isChosen = o.relic === dec.chosen_relic;
                if (isChosen) {
                    return `<span class="relic-option chosen">${label}</span>`;
                }
                return `<span class="relic-option">${label}</span>`;
            }).join("");
            return `
                <div class="decision boss-reward">
                    <span class="decision-label">Boss 遗物：</span>
                    ${chips || `<span class="relic-option chosen">${relicZh}</span>`}
                </div>
            `;
        }

        if (dec.phase === "NEOW") {
            // Neow 祝福选择（只有 choice index，没有文本描述）
            const choiceIdx = dec.chosen ? dec.chosen.replace(/.*choice=/, "") : "?";
            return `<div class="decision neow-decision"><span class="decision-label">Neow 祝福：</span>选择 #${escapeHtml(choiceIdx)}</div>`;
        }

        if (dec.phase === "TREASURE") {
            return `<div class="decision treasure-decision"><span class="decision-label">宝箱：</span>获取遗物</div>`;
        }

        if (dec.phase === "MAP") {
            return `<div class="decision map-decision"><span class="decision-label">地图选择：</span>${escapeHtml(dec.chosen || "")}</div>`;
        }

        // Generic fallback for unknown decision phases
        return `<div class="decision"><span class="decision-label">${escapeHtml(dec.phase)}：</span>${escapeHtml(dec.type || dec.chosen || "")}</div>`;
    }

    function renderDecisions(decisions) {
        if (!decisions || !decisions.length) return "";
        // Filter out MAP decisions by default (they add noise), unless it is the only decision
        const nonMap = decisions.filter((d) => d.phase !== "MAP");
        const toRender = nonMap.length > 0 ? nonMap : decisions;
        return toRender.map(renderDecision).join("");
    }

    function renderTimelineItem(item) {
        const parts = [];

        // Floor header: floor number + room type in Chinese + enemies if combat
        const roomLabel = window.roomZh(item.room) || item.room || "?";
        let headerExtra = "";
        if (item.combat) {
            const enemyNames = (item.combat.enemies_zh || item.combat.enemies_en || []).map(escapeHtml).join("、");
            if (enemyNames) headerExtra = ` - ${enemyNames}`;
        }
        const hpBefore = item.combat ? (item.combat.floor_hp_before ?? item.combat.hp_before ?? item.hp_cur) : item.hp_cur;
        const hpAfter = item.combat ? (item.combat.floor_hp_after ?? item.combat.hp_after ?? item.hp_cur) : item.hp_cur;

        parts.push(`
            <div class="tl-header">
                <span class="tl-floor">F${item.floor}</span>
                ${roomBadge(item.room)}
                <span class="tl-header-label">${escapeHtml(roomLabel)}${headerExtra}</span>
                <span class="tl-hp">HP ${item.hp_cur}/${item.hp_max}</span>
            </div>
        `);

        if (item.combat) {
            const c = item.combat;
            const enemies = (c.enemies_zh || []).map(escapeHtml).join("、");
            const reasonCls = c.exit_reason === "victory" ? "win" : "lose";
            const cbHp = c.floor_hp_before ?? c.hp_before;
            const caHp = c.floor_hp_after ?? c.hp_after;
            const diff = (cbHp != null && caHp != null) ? (caHp - cbHp) : null;
            let diffTxt = "";
            if (diff != null) {
                if (diff < 0) diffTxt = `<span class="hp-diff lose">(${diff})</span>`;
                else if (diff > 0) diffTxt = `<span class="hp-diff heal">(+${diff})</span>`;
                else diffTxt = `<span class="hp-diff neutral">(无损)</span>`;
            }
            const dataAct = item.act;
            const dataFloor = item.floor;
            parts.push(`
                <details class="tl-block combat combat-expandable" data-act="${dataAct}" data-floor="${dataFloor}">
                    <summary>
                        <span class="tl-tag">战斗</span>
                        <span>${enemies || "—"}</span>
                        <span class="tl-arrow">→</span>
                        <span class="tl-reason ${reasonCls}">${escapeHtml(exitReasonZh(c.exit_reason))}</span>
                        <span class="tl-detail">HP ${cbHp ?? "?"}→${caHp ?? "?"} ${diffTxt}</span>
                        <span class="tl-detail">回合数=${c.turn_actions}</span>
                        <span class="combat-toggle-hint">点击查看回合详情</span>
                    </summary>
                    <div class="combat-detail-body"><div class="loading">载入战斗详情...</div></div>
                </details>
            `);
        }

        if (item.event) {
            const ev = item.event;
            const evName = escapeHtml(ev.zh_name || ev.event_id || "");
            const choices = (ev.choices || [])
                .map((c) => {
                    const sel = c.idx === ev.chosen_idx;
                    const txt = escapeHtml(c.text || "");
                    return sel
                        ? `<span class="choice chosen">${c.idx}: ${txt}</span>`
                        : `<span class="choice">${c.idx}: ${txt}</span>`;
                })
                .join(" ");
            parts.push(`
                <div class="tl-block event">
                    <span class="tl-tag">事件</span>
                    <span class="event-name">${evName}</span>
                    <span class="tl-detail">阶段=${escapeHtml(ev.event_phase || "")}</span>
                    <div class="event-choices">${choices || "<i>无可选</i>"}</div>
                    <span class="tl-reason">${escapeHtml(ev.exit_reason || "")}</span>
                </div>
            `);
        }

        // Render decisions (card rewards, rest, shop, etc.)
        const decisionsHtml = renderDecisions(item.decisions);
        if (decisionsHtml) {
            parts.push(`<div class="tl-decisions">${decisionsHtml}</div>`);
        }

        if (item.deck) {
            const d = item.deck;
            const cards = (d.cards || [])
                .map(
                    (c) => {
                        const zhName = c.display || window.lookupI18N("card", c.card_en_id, c.upgraded) || c.card_en_id;
                        const cls = `card-chip rarity-${c.rarity || "UNKNOWN"}${c.upgraded ? " upgraded" : ""}`;
                        return `<span class="${cls}" title="${escapeHtml(c.card_en_id)}">${escapeHtml(zhName)}×${c.count}</span>`;
                    }
                )
                .join(" ");
            const relics = (d.relics_zh || d.relics || []).map((r) => {
                if (typeof r === "string") {
                    // Try i18n lookup for relic en_id strings
                    return escapeHtml(window.lookupI18N("relic", r) || r);
                }
                // Already an object with display/en_id
                return escapeHtml(r.display || window.lookupI18N("relic", r.en_id) || r.en_id || r);
            }).join("、");
            parts.push(`
                <details class="tl-block deck">
                    <summary>
                        <span class="tl-tag">牌组</span>
                        <span>${d.deck_size} 张</span>
                        <span class="tl-detail">${relics ? "遗物: " + relics : ""}</span>
                    </summary>
                    <div class="card-list">${cards}</div>
                </details>
            `);
        }

        return `<div class="timeline-item act-${item.act}">${parts.join("")}</div>`;
    }

    function renderCombatPlay(p) {
        if (p.action_type === "end_turn") {
            const pd = p.player_delta_raw ? `<span class="play-delta player">${escapeHtml(p.player_delta_raw)}</span>` : "";
            const ed = p.enemy_delta_raw ? `<span class="play-delta enemy">${escapeHtml(p.enemy_delta_raw)}</span>` : "";
            return `
                <div class="combat-play end-turn">
                    <span class="play-idx">#${p.action_idx}</span>
                    <span class="play-action">回合结束</span>
                    ${pd}${ed}
                </div>
            `;
        }
        const cardName = p.card_zh && p.card_zh !== p.card_en_id
            ? `${escapeHtml(p.card_zh)}<span class="play-card-en">(${escapeHtml(p.card_en_id || "")})</span>`
            : escapeHtml(p.card_en_id || "?");
        const slot = (p.hand_slot != null) ? `<span class="play-slot">手牌#${p.hand_slot}</span>` : "";
        const target = p.target_en_id
            ? `<span class="play-arrow">→</span><span class="play-target">${escapeHtml(p.target_zh || p.target_en_id)}</span>`
            : "";
        const pd = p.player_delta_raw ? `<span class="play-delta player">${escapeHtml(p.player_delta_raw)}</span>` : "";
        const ed = p.enemy_delta_raw ? `<span class="play-delta enemy">${escapeHtml(p.enemy_delta_raw)}</span>` : "";
        return `
            <div class="combat-play">
                <span class="play-idx">#${p.action_idx}</span>
                <span class="play-card">${cardName}</span>
                ${slot}
                ${target}
                ${pd}${ed}
            </div>
        `;
    }

    function renderCombatTurn(t) {
        const handChips = (t.hand || [])
            .map((h) => {
                const zh = h.card_zh && h.card_zh !== h.card_en_id ? `${escapeHtml(h.card_zh)}` : escapeHtml(h.card_en_id || "?");
                const en = h.card_zh && h.card_zh !== h.card_en_id ? `<span class="hand-en">${escapeHtml(h.card_en_id || "")}</span>` : "";
                const cost = (h.cost != null && h.cost !== "?") ? `<span class="hand-cost">${escapeHtml(String(h.cost))}</span>` : "";
                return `<span class="hand-chip" title="slot ${h.slot}">${cost}${zh}${en}</span>`;
            })
            .join("");

        const enemiesHtml = (t.enemies_state || [])
            .map((e) => {
                const zh = e.zh_name && e.zh_name !== e.en_id ? `${escapeHtml(e.zh_name)} <span class="enemy-en">(${escapeHtml(e.en_id || "")})</span>` : escapeHtml(e.en_id || "?");
                const hpPct = e.hp_max ? Math.max(0, Math.min(100, (e.hp / e.hp_max) * 100)) : 0;
                const block = e.block ? `<span class="enemy-block">护甲 ${e.block}</span>` : "";
                const buffs = e.buffs ? `<span class="enemy-buffs">${escapeHtml(e.buffs)}</span>` : "";
                const intent = e.intent_raw ? `<span class="enemy-intent" title="intent">意图: ${escapeHtml(e.intent_raw)}</span>` : "";
                return `
                    <div class="enemy-row">
                        <div class="enemy-name">${zh}</div>
                        <div class="enemy-hpbar"><div class="enemy-hpfill" style="width:${hpPct.toFixed(1)}%"></div><span class="enemy-hptxt">${e.hp}/${e.hp_max}</span></div>
                        ${block}${buffs}${intent}
                    </div>
                `;
            })
            .join("");

        const playerBuffs = t.player_buffs_raw ? `<span class="player-buffs">${escapeHtml(t.player_buffs_raw)}</span>` : "";
        const playerBlock = t.block ? `<span class="player-block">护甲 ${t.block}</span>` : "";
        const playsHtml = (t.plays || []).map(renderCombatPlay).join("");

        return `
            <div class="combat-turn">
                <div class="combat-turn-header">
                    <span class="turn-num">回合 ${t.turn_num}</span>
                    <span class="player-stat">能量 ${t.energy}/${t.energy_max}</span>
                    <span class="player-stat hp">HP ${t.hp_cur}/${t.hp_max}</span>
                    ${playerBlock}${playerBuffs}
                </div>
                <div class="combat-turn-body">
                    <div class="combat-side">
                        <div class="side-label">敌人</div>
                        <div class="enemies-list">${enemiesHtml || "<i class='muted'>无</i>"}</div>
                    </div>
                    <div class="combat-side">
                        <div class="side-label">手牌</div>
                        <div class="hand-list">${handChips || "<i class='muted'>无</i>"}</div>
                    </div>
                </div>
                <div class="combat-plays">
                    <div class="side-label">出牌序列</div>
                    ${playsHtml || "<i class='muted'>无</i>"}
                </div>
            </div>
        `;
    }

    function renderCombatDetail(detail) {
        if (!detail) return `<div class="muted">无数据</div>`;
        const turns = detail.turns || [];
        const headerLine = `
            <div class="combat-detail-header">
                <span>第 ${detail.abs_floor ?? detail.floor} 楼（Act ${detail.act}）</span>
                <span>敌人: ${(detail.enemies_zh || detail.enemies_en || []).map(escapeHtml).join("、") || "—"}</span>
                <span>HP ${detail.hp_before ?? "?"} → ${detail.hp_after ?? "?"}</span>
                <span>结果: ${escapeHtml(exitReasonZh(detail.exit_reason) || detail.exit_reason || "?")}</span>
                <span>总 action 数: ${detail.turn_actions}</span>
                <span>记录回合数: ${turns.length}</span>
            </div>
        `;
        if (!turns.length) {
            return headerLine + `<div class="muted">无 turn 数据</div>`;
        }
        return headerLine + `<div class="combat-turn-list">${turns.map(renderCombatTurn).join("")}</div>`;
    }

    async function handleCombatToggle(details) {
        const body = details.querySelector(".combat-detail-body");
        if (!body) return;
        if (details.dataset.loaded === "1") return;
        const act = parseInt(details.dataset.act, 10);
        const floor = parseInt(details.dataset.floor, 10);
        const cacheKey = `${selectedEp}_${act}_${floor}`;
        try {
            let detail;
            if (combatDetailCache.has(cacheKey)) {
                detail = combatDetailCache.get(cacheKey);
            } else {
                detail = await fetchCombatDetail(selectedEp, act, floor);
                combatDetailCache.set(cacheKey, detail);
            }
            body.innerHTML = renderCombatDetail(detail);
            details.dataset.loaded = "1";
        } catch (e) {
            body.innerHTML = `<div class="error">${escapeHtml(e.message)}</div>`;
        }
    }

    // ===== HP 曲线图：每楼层 HP 变化 =====
    const ROOM_COLORS = {
        monster: "rgba(239, 68, 68, 0.12)",    // 红
        elite:   "rgba(220, 38, 38, 0.20)",    // 深红
        boss:    "rgba(31, 41, 55, 0.18)",      // 暗色
        rest:    "rgba(34, 197, 94, 0.12)",     // 绿
        shop:    "rgba(234, 179, 8, 0.12)",     // 金
        event:   "rgba(139, 92, 246, 0.12)",    // 紫
        treasure:"rgba(253, 224, 71, 0.12)",    // 浅黄
    };

    function destroyHpChart() {
        if (hpChart) {
            try { hpChart.destroy(); } catch (_) {}
            hpChart = null;
        }
    }

    function renderHpChart(container, timeline) {
        destroyHpChart();
        if (!timeline || !timeline.length) {
            container.innerHTML = "";
            return;
        }

        // 构造数据点：x = 绝对楼层，y = hp
        // abs_floor 直接来自后端（兜底 fallback 老数据：(act-1)*17 + floor）
        const points = timeline.map(item => {
            const absFloor = item.abs_floor ?? ((item.act - 1) * 17 + item.floor);
            return { absFloor, hp_cur: item.hp_cur, hp_max: item.hp_max, room: item.room, act: item.act, floor: item.floor };
        });

        const hpData = points.map(p => ({ x: p.absFloor, y: p.hp_cur }));
        const maxHpData = points.map(p => ({ x: p.absFloor, y: p.hp_max }));

        // 计算 y 轴上限
        const yMax = Math.max(...points.map(p => p.hp_max || 0), ...points.map(p => p.hp_cur || 0)) + 5;
        const xMin = Math.min(...points.map(p => p.absFloor));
        const xMax = Math.max(...points.map(p => p.absFloor));

        // 构造房间类型背景色条（Chart.js 插件方式）
        const roomBands = points.map((p, i) => {
            const left = (i === 0) ? p.absFloor - 0.5 : (points[i - 1].absFloor + p.absFloor) / 2;
            const right = (i === points.length - 1) ? p.absFloor + 0.5 : (p.absFloor + points[i + 1].absFloor) / 2;
            return { left, right, color: ROOM_COLORS[p.room] || "rgba(0,0,0,0.03)" };
        });

        // Boss 楼层竖线
        const bossFloors = points.filter(p => p.room === "boss").map(p => p.absFloor);

        container.innerHTML = `
            <div class="hp-chart-card">
                <div class="hp-chart-title">HP 变化曲线</div>
                <div class="hp-chart-canvas-wrap"><canvas id="hp-progression-canvas"></canvas></div>
                <div class="hp-chart-legend">
                    <span class="hp-legend-item"><span class="hp-swatch" style="background:#ef4444;"></span>当前 HP</span>
                    <span class="hp-legend-item"><span class="hp-swatch" style="background:#9ca3af;border-style:dashed;"></span>最大 HP</span>
                    <span class="hp-legend-item"><span class="hp-swatch" style="background:rgba(239,68,68,0.12);border:1px solid #fca5a5;"></span>战斗</span>
                    <span class="hp-legend-item"><span class="hp-swatch" style="background:rgba(34,197,94,0.12);border:1px solid #86efac;"></span>休息</span>
                    <span class="hp-legend-item"><span class="hp-swatch" style="background:rgba(139,92,246,0.12);border:1px solid #c4b5fd;"></span>事件</span>
                    <span class="hp-legend-item"><span class="hp-swatch" style="background:rgba(234,179,8,0.12);border:1px solid #fde68a;"></span>商店</span>
                </div>
            </div>
        `;

        const canvas = document.getElementById("hp-progression-canvas");
        if (!canvas) return;

        // 房间背景色 + Boss 竖线自定义插件
        const roomBgPlugin = {
            id: "roomBackground",
            beforeDraw(chart) {
                const { ctx, chartArea: area, scales: { x: xScale, y: yScale } } = chart;
                if (!area) return;
                // 房间色条
                for (const band of roomBands) {
                    const left = Math.max(xScale.getPixelForValue(band.left), area.left);
                    const right = Math.min(xScale.getPixelForValue(band.right), area.right);
                    if (right <= left) continue;
                    ctx.fillStyle = band.color;
                    ctx.fillRect(left, area.top, right - left, area.bottom - area.top);
                }
                // Boss 竖线
                ctx.save();
                ctx.strokeStyle = "rgba(31, 41, 55, 0.35)";
                ctx.lineWidth = 1.5;
                ctx.setLineDash([4, 3]);
                for (const bf of bossFloors) {
                    const px = xScale.getPixelForValue(bf);
                    if (px < area.left || px > area.right) continue;
                    ctx.beginPath();
                    ctx.moveTo(px, area.top);
                    ctx.lineTo(px, area.bottom);
                    ctx.stroke();
                }
                ctx.restore();
            },
        };

        hpChart = new Chart(canvas, {
            type: "line",
            data: {
                datasets: [
                    {
                        label: "当前 HP",
                        data: hpData,
                        borderColor: "#ef4444",
                        backgroundColor: "rgba(239, 68, 68, 0.15)",
                        fill: true,
                        tension: 0.25,
                        pointRadius: 2.5,
                        pointBackgroundColor: "#ef4444",
                        borderWidth: 2,
                        order: 1,
                    },
                    {
                        label: "最大 HP",
                        data: maxHpData,
                        borderColor: "#9ca3af",
                        borderDash: [6, 3],
                        backgroundColor: "transparent",
                        fill: false,
                        tension: 0.1,
                        pointRadius: 0,
                        borderWidth: 1.5,
                        order: 2,
                    },
                ],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                parsing: false,
                interaction: { mode: "index", intersect: false },
                scales: {
                    x: {
                        type: "linear",
                        title: { display: true, text: "绝对楼层" },
                        min: xMin - 0.5,
                        max: xMax + 0.5,
                        ticks: { stepSize: 1, font: { size: 10 } },
                    },
                    y: {
                        title: { display: true, text: "HP" },
                        min: 0,
                        max: yMax,
                        ticks: { font: { size: 10 } },
                    },
                },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            title(items) {
                                if (!items.length) return "";
                                const xVal = items[0].parsed.x;
                                const pt = points.find(p => p.absFloor === xVal);
                                if (pt) {
                                    const roomLabel = window.roomZh(pt.room) || pt.room || "?";
                                    return `第 ${pt.act} 层 F${pt.floor}（${roomLabel}）`;
                                }
                                return `楼层 ${xVal}`;
                            },
                            label(ctx) {
                                return `${ctx.dataset.label}: ${ctx.parsed.y}`;
                            },
                        },
                    },
                },
            },
            plugins: [roomBgPlugin],
        });
    }

    function renderTimeline(container) {
        if (!currentTimeline) {
            container.innerHTML = `<div class="loading">点击左侧某局查看回放</div>`;
            return;
        }
        const s = currentTimeline.summary;
        const eg = currentTimeline.endgame;

        const timeline = currentTimeline.timeline || [];
        const groups = [];
        let curAct = null;
        for (const it of timeline) {
            if (it.act !== curAct) {
                groups.push({ act: it.act, items: [] });
                curAct = it.act;
            }
            groups[groups.length - 1].items.push(it);
        }
        const groupsHtml = groups
            .map((g) => {
                const items = g.items.map(renderTimelineItem).join("");
                return `
                    <div class="act-section act-${g.act}">
                        <div class="act-divider act-${g.act}">${actLabel(g.act)}</div>
                        ${items}
                    </div>
                `;
            })
            .join("");

        const endgameHtml = renderEndgame(eg);

        container.innerHTML = `
            <div class="tl-summary">
                <h3>第 ${currentTimeline.ep} 局</h3>
                <div class="summary-row">
                    <span>奖励: <b>${fmt(s.reward, 2)}</b></span>
                    <span>楼层: <b>${s.floor_reached}</b></span>
                    <span>步数: <b>${s.steps}</b></span>
                    <span>耗时: <b>${fmt(s.secs, 1)} 秒</b></span>
                    <span>搜索次数: <b>${s.search_calls}</b></span>
                </div>
            </div>
            ${endgameHtml}
            <div id="hp-chart-container"></div>
            <div class="timeline-list">${groupsHtml || '<div class="loading">无时间轴数据</div>'}</div>
        `;

        // 渲染 HP 曲线图
        const hpContainer = container.querySelector("#hp-chart-container");
        if (hpContainer && timeline.length) {
            renderHpChart(hpContainer, timeline);
        }

        container.querySelectorAll("details.combat-expandable").forEach((d) => {
            d.addEventListener("toggle", () => {
                if (d.open) handleCombatToggle(d);
            });
        });
    }

    async function loadTimeline() {
        const right = document.getElementById("tl-pane");
        if (!right) return;
        combatDetailCache.clear();
        if (selectedEp == null) {
            currentTimeline = null;
            renderTimeline(right);
            return;
        }
        right.innerHTML = `<div class="loading">载入时间轴...</div>`;
        try {
            currentTimeline = await fetchTimeline(selectedEp);
        } catch (e) {
            right.innerHTML = `<div class="error">加载时间轴失败: ${escapeHtml(e.message)}</div>`;
            return;
        }
        renderTimeline(right);
    }

    async function reloadEpisodeList() {
        const list = document.getElementById("ep-list-pane");
        if (!list) return;
        list.innerHTML = `<div class="loading">载入局列表...</div>`;
        try {
            episodes = await fetchEpisodeList();
        } catch (e) {
            list.innerHTML = `<div class="error">${escapeHtml(e.message)}</div>`;
            return;
        }
        // 选择默认 ep
        if (episodes.length && (selectedEp == null || !episodes.find((x) => x.ep === selectedEp))) {
            selectedEp = episodes[0].ep;
        }
        renderEpisodeList(list);
        await loadTimeline();
    }

    async function render(root, params) {
        root.innerHTML = window.UI.loading("载入局数据...");

        // URL 参数支持 ?ep=<global_ep> 直接定位
        if (params.ep) {
            selectedEp = parseInt(params.ep, 10);
        } else {
            selectedEp = null;
        }

        const sortOpts = ["ep", "reward", "floor_reached", "steps", "secs"]
            .map((s) => `<option value="${s}" ${s === sortKey ? "selected" : ""}>${SORT_KEYS_ZH[s] || s}</option>`)
            .join("");

        const filterOpts = [
            { v: "all", t: "全部" },
            { v: "won", t: "只看通关" },
            { v: "lost", t: "只看失败" },
        ]
            .map((o) => `<option value="${o.v}" ${o.v === resultFilter ? "selected" : ""}>${o.t}</option>`)
            .join("");

        root.innerHTML = `
            <h2>单局回放</h2>
            <div class="controls">
                <label>结果:</label>
                <select id="ep-filter-sel">${filterOpts}</select>
                <label>排序:</label>
                <select id="ep-sort-sel">${sortOpts}</select>
                <select id="ep-order-sel">
                    <option value="desc" ${sortOrder === "desc" ? "selected" : ""}>从大到小</option>
                    <option value="asc" ${sortOrder === "asc" ? "selected" : ""}>从小到大</option>
                </select>
                <span class="muted">点击行查看回放</span>
            </div>
            <div class="ep-layout">
                <div class="ep-list-pane" id="ep-list-pane"></div>
                <div class="tl-pane" id="tl-pane"></div>
            </div>
        `;

        root.querySelector("#ep-filter-sel").addEventListener("change", async (e) => {
            resultFilter = e.target.value;
            selectedEp = null;
            await reloadEpisodeList();
        });
        root.querySelector("#ep-sort-sel").addEventListener("change", async (e) => {
            sortKey = e.target.value;
            await reloadEpisodeList();
        });
        root.querySelector("#ep-order-sel").addEventListener("change", async (e) => {
            sortOrder = e.target.value;
            await reloadEpisodeList();
        });

        await reloadEpisodeList();
    }

    window.PAGES.episode = { render };
})();

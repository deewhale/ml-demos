# 训练曲线 API：summary + 全指标时间序列 + episode 列表 + 死亡楼层分布
from fastapi import APIRouter, Query

from ..db import get_conn

router = APIRouter()


# 允许暴露的训练指标白名单（防 SQL 注入 + 防奇怪 metric 名）
TRAIN_METRIC_COLS = {
    "mean_reward",
    "mean_steps",
    "mean_floor",
    "beat_boss_count",
    "policy_loss",
    "value_loss",
    "entropy",
    "approx_kl",
    "clip_frac",
    "update_secs",
    "wrapper_calls",
}


@router.get("/summary")
def training_summary():
    """训练总览：total_episodes, latest_ep, metrics_available"""
    conn = get_conn()
    try:
        row_tm = conn.execute(
            "SELECT COUNT(*) AS cnt, MAX(episodes_done) AS latest FROM training_metrics"
        ).fetchone()
        row_ep = conn.execute(
            "SELECT COUNT(*) AS cnt FROM episodes"
        ).fetchone()
        return {
            "total_episodes": row_ep["cnt"] if row_ep else 0,
            "latest_ep": row_tm["latest"] if row_tm else None,
            "metrics_available": row_tm["cnt"] > 0 if row_tm else False,
        }
    finally:
        conn.close()


@router.get("/metrics_all")
def metrics_all(
    ep_min: int | None = Query(None, description="最小 episodes_done（含）"),
    ep_max: int | None = Query(None, description="最大 episodes_done（含）"),
):
    """全指标时间序列，可选 ep 范围过滤"""
    conn = get_conn()
    try:
        cols = ["episodes_done", "batch_size"] + sorted(TRAIN_METRIC_COLS)
        where: list[str] = []
        params: list = []
        if ep_min is not None:
            where.append("episodes_done >= ?")
            params.append(ep_min)
        if ep_max is not None:
            where.append("episodes_done <= ?")
            params.append(ep_max)
        where_sql = (" WHERE " + " AND ".join(where)) if where else ""
        sql = f"SELECT {', '.join(cols)} FROM training_metrics{where_sql} ORDER BY episodes_done ASC"
        rows = conn.execute(sql, params).fetchall()
        return {"rows": [dict(r) for r in rows]}
    finally:
        conn.close()


@router.get("/episodes")
def training_episodes(
    ep_min: int | None = Query(None, description="最小 ep（含）"),
    ep_max: int | None = Query(None, description="最大 ep（含）"),
    beat_boss: int | None = Query(None, description="0 / 1 过滤；省略=不过滤"),
):
    """episode 级数据列表"""
    conn = get_conn()
    try:
        where: list[str] = []
        params: list = []
        if ep_min is not None:
            where.append("ep >= ?")
            params.append(ep_min)
        if ep_max is not None:
            where.append("ep <= ?")
            params.append(ep_max)
        if beat_boss is not None:
            where.append("beat_boss = ?")
            params.append(beat_boss)
        where_sql = (" WHERE " + " AND ".join(where)) if where else ""
        sql = (
            f"SELECT ep, reward, floor_reached, steps, beat_boss, secs "
            f"FROM episodes{where_sql} ORDER BY ep ASC"
        )
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


@router.get("/death_distribution")
def death_distribution(
    ep_min: int | None = Query(None, description="最小 ep（含）"),
    ep_max: int | None = Query(None, description="最大 ep（含）"),
):
    """死亡楼层分布：按 floor_reached 聚合，区分通关 / 失败。"""
    conn = get_conn()
    try:
        where: list[str] = []
        params: list = []
        if ep_min is not None:
            where.append("ep >= ?")
            params.append(ep_min)
        if ep_max is not None:
            where.append("ep <= ?")
            params.append(ep_max)
        where_sql = (" WHERE " + " AND ".join(where)) if where else ""

        rows = conn.execute(
            f"""
            SELECT floor_reached, COUNT(*) AS count,
                   SUM(CASE WHEN beat_boss = 1 THEN 1 ELSE 0 END) AS wins
            FROM episodes{where_sql}
            GROUP BY floor_reached
            ORDER BY floor_reached ASC
            """,
            params,
        ).fetchall()
        return [
            {"floor": r["floor_reached"], "count": r["count"], "wins": r["wins"]}
            for r in rows
        ]
    finally:
        conn.close()


@router.get("/act_progression")
def act_progression(
    window: int = Query(64, ge=1, description="滑动窗口大小（按 episode 数）"),
):
    """Act 进度分布：按滑动窗口统计闯关里程碑的到达比率。

    绝对楼层 = (act-1)*17 + floor，模拟器 floor 每 Act 0-index（floor 16 = boss）。
    boss 里程碑（连续楼层）：Act1 boss=16、Act2 boss=33、Act3 boss=50、心脏=55。

    返回每个窗口的（聚焦「打到多深」）:
    - beat_a1_boss:  % floor >= 17 (打过 Act 1 boss，进入 Act 2)
    - beat_a2_boss:  % floor >= 34 (打过 Act 2 boss，进入 Act 3)
    - reached_a3_boss: % floor >= 50 (到达 Act 3 boss 层)
    - won: % floor >= 55 (通关 / 到达心脏层后)
    """
    conn = get_conn()
    try:
        rows = conn.execute(
            "SELECT ep, floor_reached FROM episodes ORDER BY ep ASC"
        ).fetchall()
        if not rows:
            return []

        eps = [(r["ep"], r["floor_reached"]) for r in rows]
        result = []
        for win_start in range(0, len(eps), window):
            win = eps[win_start : win_start + window]
            if not win:
                break
            n = len(win)
            ep_start = win[0][0]
            ep_end = win[-1][0]

            beat_a1_boss = sum(1 for _, f in win if f >= 17) / n
            beat_a2_boss = sum(1 for _, f in win if f >= 34) / n
            reached_a3_boss = sum(1 for _, f in win if f >= 50) / n
            won = sum(1 for _, f in win if f >= 55) / n

            result.append({
                "ep_start": ep_start,
                "ep_end": ep_end,
                "beat_a1_boss": round(beat_a1_boss, 4),
                "beat_a2_boss": round(beat_a2_boss, 4),
                "reached_a3_boss": round(reached_a3_boss, 4),
                "won": round(won, 4),
            })

        return result
    finally:
        conn.close()


@router.get("/behavior")
def behavior_stats(
    ep_min: int | None = Query(None, description="最小 ep（含）"),
    ep_max: int | None = Query(None, description="最大 ep（含）"),
    window: int = Query(64, ge=1, description="滑动窗口大小（按 episode 数）"),
):
    """Agent 行为指标：从 meta_decisions 聚合，按滑动窗口返回时间序列。

    返回每个窗口的:
    - card_skip_rate: CARD_REWARDS 跳过率
    - rest_rate: REST 决策中选择休息（vs 升级）的比例
    - cards_picked_per_ep: 窗口内平均每局拿卡数
    - shop_buy_rate / shop_remove_rate: SHOP 购买 / 移除比例
    """
    conn = get_conn()
    try:
        # --- 1. 拉 CARD_REWARDS per-ep 聚合 ---
        card_where = ["phase = 'CARD_REWARDS'"]
        card_params: list = []
        if ep_min is not None:
            card_where.append("ep >= ?")
            card_params.append(ep_min)
        if ep_max is not None:
            card_where.append("ep <= ?")
            card_params.append(ep_max)
        card_rows = conn.execute(
            f"""
            SELECT ep,
                   COUNT(*) AS total,
                   SUM(CASE WHEN decision_type = 'skip' THEN 1 ELSE 0 END) AS skips,
                   SUM(CASE WHEN decision_type = 'pick' THEN 1 ELSE 0 END) AS picks
            FROM meta_decisions
            WHERE {' AND '.join(card_where)}
            GROUP BY ep ORDER BY ep
            """,
            card_params,
        ).fetchall()
        card_by_ep = {r["ep"]: dict(r) for r in card_rows}

        # --- 2. 拉 REST per-ep 聚合 ---
        rest_where = ["phase = 'REST'"]
        rest_params: list = []
        if ep_min is not None:
            rest_where.append("ep >= ?")
            rest_params.append(ep_min)
        if ep_max is not None:
            rest_where.append("ep <= ?")
            rest_params.append(ep_max)
        rest_rows = conn.execute(
            f"""
            SELECT ep,
                   COUNT(*) AS total,
                   SUM(CASE WHEN decision_type = 'rest' THEN 1 ELSE 0 END) AS rests,
                   SUM(CASE WHEN decision_type = 'upgrade' THEN 1 ELSE 0 END) AS upgrades
            FROM meta_decisions
            WHERE {' AND '.join(rest_where)}
            GROUP BY ep ORDER BY ep
            """,
            rest_params,
        ).fetchall()
        rest_by_ep = {r["ep"]: dict(r) for r in rest_rows}

        # --- 3. 拉 SHOP per-ep 聚合 ---
        shop_where = ["phase = 'SHOP'"]
        shop_params: list = []
        if ep_min is not None:
            shop_where.append("ep >= ?")
            shop_params.append(ep_min)
        if ep_max is not None:
            shop_where.append("ep <= ?")
            shop_params.append(ep_max)
        shop_rows = conn.execute(
            f"""
            SELECT ep,
                   COUNT(*) AS total,
                   SUM(CASE WHEN decision_type = 'buy' THEN 1 ELSE 0 END) AS buys,
                   SUM(CASE WHEN decision_type = 'remove' THEN 1 ELSE 0 END) AS removes,
                   SUM(CASE WHEN decision_type = 'leave' THEN 1 ELSE 0 END) AS leaves
            FROM meta_decisions
            WHERE {' AND '.join(shop_where)}
            GROUP BY ep ORDER BY ep
            """,
            shop_params,
        ).fetchall()
        shop_by_ep = {r["ep"]: dict(r) for r in shop_rows}

        # --- 4. 取全部 ep 列表（排序） ---
        all_eps = sorted(
            set(card_by_ep.keys()) | set(rest_by_ep.keys()) | set(shop_by_ep.keys())
        )
        if not all_eps:
            return []

        # --- 5. 按窗口聚合 ---
        result = []
        for win_start in range(0, len(all_eps), window):
            win_eps = all_eps[win_start : win_start + window]
            if not win_eps:
                break
            ep_start = win_eps[0]
            ep_end = win_eps[-1]
            n_eps = len(win_eps)

            # card rewards
            card_total = sum(card_by_ep.get(e, {}).get("total", 0) for e in win_eps)
            card_skips = sum(card_by_ep.get(e, {}).get("skips", 0) for e in win_eps)
            card_picks = sum(card_by_ep.get(e, {}).get("picks", 0) for e in win_eps)
            card_skip_rate = (card_skips / card_total) if card_total > 0 else None

            # cards picked per ep
            cards_picked_per_ep = card_picks / n_eps if n_eps > 0 else None

            # rest vs upgrade
            rest_total = sum(rest_by_ep.get(e, {}).get("total", 0) for e in win_eps)
            rest_rests = sum(rest_by_ep.get(e, {}).get("rests", 0) for e in win_eps)
            rest_rate = (rest_rests / rest_total) if rest_total > 0 else None

            # shop
            shop_total = sum(shop_by_ep.get(e, {}).get("total", 0) for e in win_eps)
            shop_buys = sum(shop_by_ep.get(e, {}).get("buys", 0) for e in win_eps)
            shop_removes = sum(shop_by_ep.get(e, {}).get("removes", 0) for e in win_eps)
            shop_buy_rate = (shop_buys / shop_total) if shop_total > 0 else None
            shop_remove_rate = (shop_removes / shop_total) if shop_total > 0 else None

            def _round(v):
                return round(v, 4) if v is not None else None

            result.append({
                "ep_start": ep_start,
                "ep_end": ep_end,
                "card_skip_rate": _round(card_skip_rate),
                "rest_rate": _round(rest_rate),
                "cards_picked_per_ep": _round(cards_picked_per_ep),
                "shop_buy_rate": _round(shop_buy_rate),
                "shop_remove_rate": _round(shop_remove_rate),
            })

        return result
    finally:
        conn.close()

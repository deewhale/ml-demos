# Eval 详情 API：eval 时间序列 + boss 分布（中文 join）
from fastapi import APIRouter, HTTPException, Query

from ..db import get_conn

router = APIRouter()


@router.get("/results")
def eval_results(
    ep_min: int | None = Query(None, description="最小 episodes_done（含）"),
    ep_max: int | None = Query(None, description="最大 episodes_done（含）"),
):
    """所有 eval 时点，按 episodes_done 升序"""
    conn = get_conn()
    try:
        where: list[str] = []
        params: list = []
        if ep_min is not None:
            where.append("episodes_done >= ?")
            params.append(ep_min)
        if ep_max is not None:
            where.append("episodes_done <= ?")
            params.append(ep_max)
        where_sql = (" WHERE " + " AND ".join(where)) if where else ""
        rows = conn.execute(
            f"""
            SELECT episodes_done, num_seeds, completed,
                   reached_boss_rate, act1_boss_beat_rate, act2_boss_beat_rate,
                   won_game_rate, beat_boss_rate, floor_mean, floor_max,
                   avg_steps, secs
            FROM eval_results{where_sql}
            ORDER BY episodes_done ASC
            """,
            params,
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


@router.get("/boss_breakdown")
def boss_breakdown(ep: int = Query(..., description="episodes_done 值")):
    """某次 eval 的 boss 击杀 / 触达分布，附 i18n 中文名"""
    conn = get_conn()
    try:
        rows = conn.execute(
            """
            SELECT
                ebc.boss_en_id,
                i.zh_name,
                i.en_name,
                ebc.reach_count,
                ebc.kill_count
            FROM eval_boss_counts ebc
            LEFT JOIN i18n_entries i
              ON i.kind = 'monster' AND i.en_id = ebc.boss_en_id
            WHERE ebc.episodes_done = ?
            ORDER BY ebc.reach_count DESC, ebc.boss_en_id ASC
            """,
            (ep,),
        ).fetchall()
        return [
            {
                "boss_en_id": r["boss_en_id"],
                "zh_name": r["zh_name"],
                "en_name": r["en_name"],
                "reach": r["reach_count"],
                "kill": r["kill_count"],
            }
            for r in rows
        ]
    finally:
        conn.close()

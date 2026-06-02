# 扫 sts_models/v8_ppo_batch_v* 目录，把每个 v8_ppo_summary.json 入库
# 用法:
#   python -m web.etl.scan_batches                 # 增量（按 mtime 跳过未变化）
#   python -m web.etl.scan_batches --force         # 强制全量重扫
#
# 注意：v14 之前的 batch 因 simulator bug 污染，自动跳过。

import argparse
import json
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

from ..db import PROJECT_ROOT, get_conn, init_db


# 训练目录命名前缀，去掉这个前缀即 source_key
BATCH_DIR_PREFIX = "v8_ppo_batch_"
SUMMARY_FILENAME = "v8_ppo_summary.json"

# v14 之前的数据因 simulator bug 污染，跳过
MIN_CLEAN_BATCH_NUM = 14

# source_key 里提取数字部分的正则（如 "v15" → 15, "v8_smoke" → 8）
_RE_SOURCE_NUM = re.compile(r"^v(\d+)")


def _source_key_num(source_key: str) -> int | None:
    """从 source_key 提取起始数字。'v15' → 15, 'v8_smoke' → 8, 非 v 开头 → None"""
    m = _RE_SOURCE_NUM.match(source_key)
    return int(m.group(1)) if m else None


def discover_batch_dirs() -> list[Path]:
    """扫描 sts_models 下所有 v8_ppo_batch_v* 目录"""
    root = PROJECT_ROOT / "sts_models"
    if not root.exists():
        return []
    return sorted(
        [p for p in root.glob(f"{BATCH_DIR_PREFIX}*") if p.is_dir()]
    )


def _existing_summary_mtime(conn: sqlite3.Connection, source_key: str) -> float | None:
    """查 etl_metadata 拿已记录的 summary_mtime"""
    row = conn.execute(
        "SELECT summary_mtime FROM etl_metadata WHERE source_key = ?", (source_key,)
    ).fetchone()
    return row["summary_mtime"] if row else None


def find_log_path(source_key: str) -> str | None:
    """约定日志在 /tmp/v8_ppo_batch_<source_key>.log。存在则返回绝对路径。"""
    candidate = Path(f"/tmp/v8_ppo_batch_{source_key}.log")
    return str(candidate) if candidate.exists() else None


def scan_one_batch(
    conn: sqlite3.Connection, batch_dir: Path, force: bool
) -> tuple[bool, str]:
    """处理单个 batch 目录。返回 (是否实际写入, 状态字符串)"""
    source_key = batch_dir.name[len(BATCH_DIR_PREFIX):]

    # v14 之前的 batch 因 simulator bug 污染，跳过
    num = _source_key_num(source_key)
    if num is not None and num < MIN_CLEAN_BATCH_NUM:
        return False, "skipped_polluted"

    summary_path = batch_dir / SUMMARY_FILENAME

    if not summary_path.exists():
        return False, f"no_summary"

    mtime = summary_path.stat().st_mtime

    # 增量判定：未 force 且 mtime 未变 -> 跳过
    if not force:
        prev = _existing_summary_mtime(conn, source_key)
        if prev is not None and abs(prev - mtime) < 1e-6:
            return False, "skipped_unchanged"

    # 读 summary
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[scan_batches] WARN: {summary_path} parse error: {exc}", file=sys.stderr)
        return False, f"parse_error"

    # 必要字段缺失则 warning 跳过
    if "args" not in summary:
        print(f"[scan_batches] WARN: {source_key} missing 'args', skip", file=sys.stderr)
        return False, "missing_args"

    log_path = find_log_path(source_key)
    scanned_at = datetime.now(timezone.utc).isoformat()

    # 收集本 batch 所有 episodes_done 值（用于幂等 DELETE）
    all_episodes_done: set[int] = set()
    for m in summary.get("train_log_tail", []) or []:
        ep = m.get("episodes_done")
        if ep is not None:
            all_episodes_done.add(ep)
    for e in summary.get("eval_history", []) or []:
        ep = e.get("episodes_done")
        if ep is not None:
            all_episodes_done.add(ep)

    cur = conn.cursor()

    # 幂等：DELETE 旧数据（按 episodes_done 集合）
    if all_episodes_done:
        placeholders = ",".join("?" * len(all_episodes_done))
        ep_list = list(all_episodes_done)
        cur.execute(f"DELETE FROM eval_boss_counts WHERE episodes_done IN ({placeholders})", ep_list)
        cur.execute(f"DELETE FROM eval_results WHERE episodes_done IN ({placeholders})", ep_list)
        cur.execute(f"DELETE FROM training_metrics WHERE episodes_done IN ({placeholders})", ep_list)

    # etl_metadata: DELETE + INSERT
    cur.execute("DELETE FROM etl_metadata WHERE source_key = ?", (source_key,))
    cur.execute(
        """
        INSERT INTO etl_metadata
            (source_key, batch_dir, log_path, summary_mtime, scanned_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            source_key,
            str(batch_dir),
            log_path,
            mtime,
            scanned_at,
        ),
    )

    # training_metrics
    metrics_rows = []
    for m in summary.get("train_log_tail", []) or []:
        ep = m.get("episodes_done")
        if ep is None:
            continue
        metrics_rows.append(
            (
                ep,
                m.get("batch_size"),
                m.get("mean_reward"),
                m.get("mean_steps"),
                m.get("mean_floor"),
                m.get("beat_boss_count"),
                m.get("policy_loss"),
                m.get("value_loss"),
                m.get("entropy"),
                m.get("approx_kl"),
                m.get("clip_frac"),
                m.get("update_secs"),
                m.get("wrapper_calls"),
            )
        )
    if metrics_rows:
        cur.executemany(
            """
            INSERT INTO training_metrics
                (episodes_done, batch_size, mean_reward, mean_steps,
                 mean_floor, beat_boss_count, policy_loss, value_loss, entropy,
                 approx_kl, clip_frac, update_secs, wrapper_calls)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            metrics_rows,
        )

    # eval_results + eval_boss_counts
    for e in summary.get("eval_history", []) or []:
        ep = e.get("episodes_done")
        if ep is None:
            continue
        cur.execute(
            """
            INSERT INTO eval_results
                (episodes_done, num_seeds, completed, reached_boss_rate,
                 act1_boss_beat_rate, act2_boss_beat_rate, won_game_rate, beat_boss_rate,
                 floor_mean, floor_max, avg_steps, secs)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ep,
                e.get("num_seeds"),
                e.get("completed"),
                e.get("reached_boss_rate"),
                e.get("act1_boss_beat_rate"),
                e.get("act2_boss_beat_rate"),
                e.get("won_game_rate"),
                e.get("beat_boss_rate"),
                e.get("floor_mean"),
                e.get("floor_max"),
                e.get("avg_steps"),
                e.get("secs"),
            ),
        )

        reach = e.get("boss_reach_counts") or {}
        kill = e.get("boss_kill_counts") or {}
        # 取两 dict 的并集 key
        boss_ids = set(reach.keys()) | set(kill.keys())
        boss_rows = [
            (ep, boss_id, int(reach.get(boss_id, 0) or 0), int(kill.get(boss_id, 0) or 0))
            for boss_id in boss_ids
        ]
        if boss_rows:
            cur.executemany(
                """
                INSERT INTO eval_boss_counts
                    (episodes_done, boss_en_id, reach_count, kill_count)
                VALUES (?, ?, ?, ?)
                """,
                boss_rows,
            )

    conn.commit()
    return True, "inserted"


def scan_all(force: bool = False) -> dict:
    """无参编程入口：扫描所有 batch 目录入库，返回统计 dict。

    给 web.routers.admin 用，避免重复实现命令行解析。
    """
    init_db()
    conn = get_conn()
    try:
        batch_dirs = discover_batch_dirs()
        if not batch_dirs:
            print("[scan_batches] no batch directories found under sts_models/")
            return {"inserted": 0, "skipped": 0, "errors": 0, "total_dirs": 0}

        n_inserted = 0
        n_skipped = 0
        n_error = 0
        for bd in batch_dirs:
            wrote, status = scan_one_batch(conn, bd, force=force)
            tag = "WROTE" if wrote else "SKIP"
            print(f"[scan_batches] {tag} {bd.name}: {status}")
            if wrote:
                n_inserted += 1
            elif status.startswith("skipped"):
                n_skipped += 1
            else:
                n_error += 1
        print(
            f"[scan_batches] done: inserted={n_inserted} skipped={n_skipped} "
            f"error_or_empty={n_error} total_dirs={len(batch_dirs)}"
        )
        return {
            "inserted": n_inserted,
            "skipped": n_skipped,
            "errors": n_error,
            "total_dirs": len(batch_dirs),
        }
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="扫训练 summary.json 入库")
    parser.add_argument("--force", action="store_true", help="忽略 mtime 强制重扫")
    args = parser.parse_args()
    scan_all(force=args.force)


if __name__ == "__main__":
    main()

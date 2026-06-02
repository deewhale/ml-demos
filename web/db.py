# SQLite 连接和路径常量
# 所有 router / ETL 通过 get_conn() 拿连接，init_db() 一次性建表

import sqlite3
from pathlib import Path

# 项目根目录（web/ 上一层）
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 数据库路径：web/data/sts_viz.sqlite
DB_DIR = Path(__file__).resolve().parent / "data"
DB_PATH = DB_DIR / "sts_viz.sqlite"

# schema.sql 路径
SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


def get_conn() -> sqlite3.Connection:
    """拿一个 sqlite3 连接。自动启 foreign_keys + WAL + busy_timeout，row_factory 用 Row。

    每次调用都开新连接，调用方自行 close（或 with as）。
    WAL 模式允许读写并发（rescan 写入期间 GET 请求不会 "database is locked"）。
    """
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


def migrate_v3() -> bool:
    """schema v3 迁移：episode-centric，去掉所有 batch 概念。

    检测方式：
    1. DB 文件存在 + batches 表还在 → 旧 schema，删除 DB 文件让 init_db 重建。
    2. DB 文件存在 + 缺少必要的新表（如 meta_decisions）→ schema 过时，同样重建。
    返回 True 表示删了旧 DB；False 表示无需迁移。
    """
    if not DB_PATH.exists():
        return False

    conn = sqlite3.connect(str(DB_PATH))
    try:
        has_batches = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='batches'"
        ).fetchone()
        has_meta_decisions = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='meta_decisions'"
        ).fetchone()
    finally:
        conn.close()

    need_rebuild = False
    if has_batches is not None:
        print("[migrate_v3] old schema detected (batches table present), deleting DB for rebuild ...")
        need_rebuild = True
    elif has_meta_decisions is None:
        print("[migrate_v3] schema outdated (meta_decisions table missing), deleting DB for rebuild ...")
        need_rebuild = True

    if not need_rebuild:
        return False

    DB_PATH.unlink()
    return True


def init_db() -> None:
    """读 schema.sql 建表。幂等（IF NOT EXISTS）。自动跑 v3 迁移。"""
    DB_DIR.mkdir(parents=True, exist_ok=True)
    migrated = migrate_v3()
    conn = get_conn()
    try:
        schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
        conn.executescript(schema_sql)
        conn.commit()
        if migrated:
            print("[migrate_v3] schema rebuilt (episode-centric, no batch_id)")
    finally:
        conn.close()


if __name__ == "__main__":
    # python -m web.db 可手动初始化
    init_db()
    print(f"[db] initialized {DB_PATH}")

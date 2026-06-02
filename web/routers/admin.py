"""Admin endpoints（P6）。

主要给前端「重新扫描」按钮用：触发 ETL 流水线（scan_batches -> parse_log）重新入库。

Endpoints:
  POST /api/admin/rescan?force=false      跑 scan_batches.scan_all + parse_log.parse_all_batches，
                                          返回 {status, sources_scanned, episodes_parsed, elapsed_sec}
"""
from __future__ import annotations

import logging
import time
import traceback

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from ..etl.load_mappings import load_mappings as load_i18n
from ..etl.parse_log import parse_all_batches
from ..etl.scan_batches import scan_all

logger = logging.getLogger("web.admin")
router = APIRouter()


@router.post("/rescan")
def rescan(force: bool = Query(False, description="忽略 mtime 强制重扫所有源")):
    """同步触发 ETL 流水线：scan_batches -> parse_log。

    v8 解析约 0.13s，加起来 < 1s，前端可直接同步等。
    ETL 内部异常会被捕获并返回 500 + detail，而非裸 traceback。
    """
    t0 = time.time()
    try:
        scan_stats = scan_all(force=force)
        parse_stats = parse_all_batches(single_source=None)
        # i18n 翻译表自动刷新（确保 rescan 后中文数据可用）
        try:
            load_i18n()
        except Exception as i18n_exc:
            logger.warning("i18n load failed (non-fatal): %s", i18n_exc)
    except Exception as exc:
        elapsed = time.time() - t0
        tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
        logger.error("rescan failed after %.2fs:\n%s", elapsed, "".join(tb))
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "detail": f"{type(exc).__name__}: {exc}",
                "elapsed_sec": round(elapsed, 2),
                "force": force,
            },
        )
    elapsed = time.time() - t0
    return {
        "status": "ok",
        "sources_scanned": scan_stats.get("inserted", 0),
        "sources_skipped": scan_stats.get("skipped", 0),
        "sources_errors": scan_stats.get("errors", 0),
        "total_source_dirs": scan_stats.get("total_dirs", 0),
        "logs_processed": parse_stats.get("sources_processed", 0),
        "episodes_parsed": parse_stats.get("total_episodes", 0),
        "elapsed_sec": round(elapsed, 2),
        "force": force,
    }

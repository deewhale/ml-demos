# V8 RL 训练数据可视化 FastAPI 入口
# 启动: .venv/bin/uvicorn web.app:app --port 8765

import logging
import sqlite3
import traceback
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from .db import init_db
from .routers import admin as admin_router
from .routers import deck as deck_router
from .routers import episode as episode_router
from .routers import eval as eval_router
from .routers import lookup as lookup_router
from .routers import training as training_router

logger = logging.getLogger("web")

app = FastAPI(title="STS V8 RL Visualization", version="0.1.0")

# CORS：允许本地开发任意来源（静态资源同源时不影响，跨源开发时防 500）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# 开发环境：静态资源禁缓存，防止浏览器缓存旧 JS/CSS
@app.middleware("http")
async def no_cache_static(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response


@app.exception_handler(sqlite3.OperationalError)
async def sqlite_error_handler(request: Request, exc: sqlite3.OperationalError):
    """SQLite 错误（database is locked / no such table 等）→ 503 而非裸 500"""
    logger.error("SQLite OperationalError on %s: %s", request.url, exc)
    return JSONResponse(
        status_code=503,
        content={"detail": f"database temporarily unavailable: {exc}"},
    )


@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception):
    """兜底异常 handler：打完整 traceback 到 stderr，返回 500 + 可调试 detail"""
    tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
    logger.error("Unhandled exception on %s:\n%s", request.url, "".join(tb))
    return JSONResponse(
        status_code=500,
        content={"detail": f"{type(exc).__name__}: {exc}"},
    )


@app.on_event("startup")
def _startup_init_db():
    # 首次启动时自动建表（幂等），保证 router 拿到的连接能查到 schema
    init_db()


# 5 个子 router 挂到对应前缀
app.include_router(training_router.router, prefix="/api/training", tags=["training"])
app.include_router(eval_router.router, prefix="/api/eval", tags=["eval"])
app.include_router(episode_router.router, prefix="/api/episode", tags=["episode"])
app.include_router(deck_router.router, prefix="/api/deck", tags=["deck"])
app.include_router(lookup_router.router, prefix="/api/lookup", tags=["lookup"])
app.include_router(admin_router.router, prefix="/api/admin", tags=["admin"])

# 静态资源
_static_dir = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


@app.get("/", include_in_schema=False)
def root_redirect():
    return RedirectResponse(url="/static/index.html")


@app.get("/healthz", include_in_schema=False)
def healthz():
    return {"status": "ok"}

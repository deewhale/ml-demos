"""Rust 引擎（StSRLSolver origin/main 的 engine-rs，pyo3 绑定）加载器。

定位并加载 libsts_engine.dylib，缓存返回 `sts_engine` 扩展模块。

.dylib 定位优先级：
  1. 环境变量 STS_RUST_DYLIB（显式覆盖）
  2. 默认：<repo>/external/stsrl_rust_main/packages/engine-rs/target/debug/libsts_engine.dylib
     （repo root 由本文件相对位置推导，不硬编码家目录）

找不到时抛 FileNotFoundError，提示先 build。
"""
from __future__ import annotations

import importlib.machinery
import importlib.util
import os
from pathlib import Path
from typing import Optional

# 本文件在 <repo>/v8/backends/rust_engine_loader.py → parents[2] = <repo>
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DYLIB = (
    _REPO_ROOT
    / "external"
    / "stsrl_rust_main"
    / "packages"
    / "engine-rs"
    / "target"
    / "debug"
    / "libsts_engine.dylib"
)

_BUILD_HINT = (
    "Rust 引擎 .dylib 未找到。请先在持久 worktree 构建：\n"
    '  . "$HOME/.cargo/env"\n'
    "  PYO3_PYTHON=<repo>/.venv/bin/python \\\n"
    '  RUSTFLAGS="-C link-arg=-undefined -C link-arg=dynamic_lookup" \\\n'
    "  cargo build --manifest-path "
    "<repo>/external/stsrl_rust_main/packages/engine-rs/Cargo.toml "
    "--features extension-module\n"
    "或设置环境变量 STS_RUST_DYLIB 指向已构建的 libsts_engine.dylib。"
)

_CACHED_MODULE = None


def _resolve_dylib() -> Path:
    env = os.environ.get("STS_RUST_DYLIB")
    if env:
        p = Path(env).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(
                f"STS_RUST_DYLIB={env} 指向的文件不存在。\n{_BUILD_HINT}"
            )
        return p
    if not _DEFAULT_DYLIB.exists():
        raise FileNotFoundError(
            f"默认路径无 .dylib：{_DEFAULT_DYLIB}\n{_BUILD_HINT}"
        )
    return _DEFAULT_DYLIB


def load_sts_engine(dylib_path: Optional[str] = None):
    """加载并缓存 sts_engine 扩展模块。

    dylib_path 显式给则用它，否则走 env / 默认路径解析。
    """
    global _CACHED_MODULE
    if _CACHED_MODULE is not None and dylib_path is None:
        return _CACHED_MODULE

    p = Path(dylib_path).expanduser().resolve() if dylib_path else _resolve_dylib()
    if not p.exists():
        raise FileNotFoundError(f"无 .dylib：{p}\n{_BUILD_HINT}")

    loader = importlib.machinery.ExtensionFileLoader("sts_engine", str(p))
    spec = importlib.util.spec_from_loader("sts_engine", loader)
    if spec is None:
        raise ImportError(f"无法为 {p} 创建 spec")
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)

    if dylib_path is None:
        _CACHED_MODULE = module
    return module

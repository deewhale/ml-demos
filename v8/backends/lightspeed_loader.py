"""sts_lightspeed（社区金标准 C++ 模拟器，pybind11 绑定）加载器。

定位并 import `slaythespire` 扩展模块，缓存返回。

.so 定位优先级：
  1. 环境变量 STS_LIGHTSPEED_BUILD（显式指向 build 目录）
  2. 默认：<repo>/external/sts_lightspeed/build
     （repo root 由本文件相对位置推导：parents[2] = <repo>，不硬编码家目录）

把 build 目录加进 sys.path 后 `import slaythespire`。找不到 .so 抛清晰错误，
提示先 build。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# 本文件在 <repo>/v8/backends/lightspeed_loader.py → parents[2] = <repo>
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_BUILD = _REPO_ROOT / "external" / "sts_lightspeed" / "build"

_BUILD_HINT = (
    "sts_lightspeed 扩展模块（slaythespire*.so）未找到。请先构建：\n"
    "  在 <repo>/external/sts_lightspeed 下 cmake build 出 build/ 目录，\n"
    "  产物形如 build/slaythespire.cpython-3xx-darwin.so。\n"
    "或设置环境变量 STS_LIGHTSPEED_BUILD 指向含该 .so 的 build 目录。"
)

_CACHED_MODULE = None


def _resolve_build_dir() -> Path:
    env = os.environ.get("STS_LIGHTSPEED_BUILD")
    if env:
        p = Path(env).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(
                f"STS_LIGHTSPEED_BUILD={env} 指向的目录不存在。\n{_BUILD_HINT}"
            )
        return p
    if not _DEFAULT_BUILD.exists():
        raise FileNotFoundError(
            f"默认 build 目录不存在：{_DEFAULT_BUILD}\n{_BUILD_HINT}"
        )
    return _DEFAULT_BUILD


def _has_so(build_dir: Path) -> bool:
    return any(build_dir.glob("slaythespire*.so"))


def load_lightspeed():
    """定位并 import sts_lightspeed 的 `slaythespire` 模块，缓存返回。"""
    global _CACHED_MODULE
    if _CACHED_MODULE is not None:
        return _CACHED_MODULE

    build_dir = _resolve_build_dir()
    if not _has_so(build_dir):
        raise FileNotFoundError(
            f"{build_dir} 下没有 slaythespire*.so。\n{_BUILD_HINT}"
        )

    build_str = str(build_dir)
    if build_str not in sys.path:
        sys.path.insert(0, build_str)

    import slaythespire  # noqa: E402  延迟到路径就绪后 import

    _CACHED_MODULE = slaythespire
    return _CACHED_MODULE

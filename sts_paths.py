"""StSRLSolver 路径解析。优先 env var，否则用 repo 相对路径。"""
import os
import sys


def stsrl_solver_path() -> str:
    env = os.environ.get("STSRLSOLVER_PATH")
    if env:
        return env
    # repo 相对：本文件所在目录的 external/StSRLSolver
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "external", "StSRLSolver")


def ensure_on_sys_path() -> None:
    p = stsrl_solver_path()
    if not os.path.isdir(p):
        raise RuntimeError(
            f"StSRLSolver not found at {p}. "
            f"Set STSRLSOLVER_PATH env var or clone to external/StSRLSolver."
        )
    if p not in sys.path:
        sys.path.insert(0, p)

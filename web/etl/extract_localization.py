"""从 Steam Slay the Spire jar 提取简中本地化 → JSON。

CLI:
  python -m web.etl.extract_localization              # 默认扫描 + 兜底路径
  python -m web.etl.extract_localization --jar PATH   # 显式指定 jar
  python -m web.etl.extract_localization --zho_dir D  # 直接读 D/{cards,relics,...}.json
  python -m web.etl.extract_localization --out PATH   # 自定义输出（默认 web/data/localization_zho.json）

输出 schema:
{
  "source": "jar:/abs/path.jar"  或  "zho_dir:/abs/path",
  "entries": {
     "card":    {"<en_id>": {"zh_name", "zh_desc", "raw"}},
     "relic":   {...},
     "monster": {...},
     "event":   {...},
     "potion":  {...}
  }
}

注意：
- 必须用 zipfile 标准库读 jar，禁 subprocess unzip/jar
- jar 内简中路径是 localization/zhs/ (zhs=简体, zht=繁体)
- 文件名是 cards.json / relics.json / monsters.json / events.json / potions.json
  （lowercase plural，与 STS 旧 mod 命名 CardStrings.json 不同）
- JSON schema（实测）:
    cards.json:    {"<id>": {"NAME": str, "DESCRIPTION": str}}
    relics.json:   {"<id>": {"NAME": str, "FLAVOR": str, "DESCRIPTIONS": [str]}}
    monsters.json: {"<id>": {"NAME": str, "MOVES": [str], "DIALOG": [str]}}
    events.json:   {"<id>": {"NAME": str, "DESCRIPTIONS": [str]}}
    potions.json:  {"<id>": {"NAME": str, "DESCRIPTIONS": [str]}}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional

# 5 类对应：(kind, jar 内文件名)
_KIND_TO_JAR_FILE = [
    ("card", "cards.json"),
    ("relic", "relics.json"),
    ("monster", "monsters.json"),
    ("event", "events.json"),
    ("potion", "potions.json"),
]

# Steam 默认安装路径候选（macOS / Windows / Linux）
_STEAM_JAR_CANDIDATES = [
    # macOS
    Path.home() / "Library/Application Support/Steam/steamapps/common/SlayTheSpire/SlayTheSpire.app/Contents/Resources/desktop-1.0.jar",
    # Windows（如果跑在 WSL 上才会有意义）
    Path("C:/Program Files (x86)/Steam/steamapps/common/SlayTheSpire/desktop-1.0.jar"),
    # Linux
    Path.home() / ".steam/steam/steamapps/common/SlayTheSpire/desktop-1.0.jar",
    Path.home() / ".local/share/Steam/steamapps/common/SlayTheSpire/desktop-1.0.jar",
]


def _find_jar_path(explicit: Optional[str]) -> Optional[Path]:
    """三级 fallback 找 jar 路径。

    优先级：
      1. --jar 显式参数
      2. 环境变量 STS_JAR_PATH
      3. Steam 标准路径（mac/win/linux）
      4. ~/Downloads/*.jar 兜底扫描（按用户提示）
    """
    if explicit:
        p = Path(explicit).expanduser()
        if p.exists():
            return p
        print(f"[extract] WARN --jar 路径不存在: {p}", file=sys.stderr)

    env = os.environ.get("STS_JAR_PATH")
    if env:
        p = Path(env).expanduser()
        if p.exists():
            return p
        print(f"[extract] WARN STS_JAR_PATH 路径不存在: {p}", file=sys.stderr)

    for cand in _STEAM_JAR_CANDIDATES:
        if cand.exists():
            return cand

    downloads = Path.home() / "Downloads"
    if downloads.is_dir():
        # 优先匹配 desktop-1.0.jar / desktop*.jar
        for pattern in ("desktop-1.0.jar", "desktop*.jar", "*.jar"):
            for cand in sorted(downloads.glob(pattern)):
                # 只要是 STS jar（大小 > 100MB 用作启发式过滤）
                try:
                    if cand.stat().st_size > 100 * 1024 * 1024:
                        return cand
                except OSError:
                    continue

    return None


def _normalize_card_entry(raw: Dict[str, Any]) -> Dict[str, Any]:
    """cards.json entry → {zh_name, zh_desc, raw}."""
    name = raw.get("NAME") or ""
    desc = raw.get("DESCRIPTION") or ""
    if isinstance(desc, list):
        desc = " ".join(desc)
    return {"zh_name": name, "zh_desc": desc, "raw": raw}


def _normalize_descriptions_entry(raw: Dict[str, Any]) -> Dict[str, Any]:
    """relics/events/potions 用 DESCRIPTIONS list；relics 还多个 FLAVOR。"""
    name = raw.get("NAME") or ""
    desc_list = raw.get("DESCRIPTIONS") or []
    if isinstance(desc_list, list):
        desc = " ".join(s for s in desc_list if isinstance(s, str))
    else:
        desc = str(desc_list)
    return {"zh_name": name, "zh_desc": desc, "raw": raw}


def _normalize_monster_entry(raw: Dict[str, Any]) -> Dict[str, Any]:
    """monsters.json: name 是核心，moves / dialog 都塞 raw。"""
    name = raw.get("NAME") or ""
    return {"zh_name": name, "zh_desc": "", "raw": raw}


_NORMALIZERS = {
    "card": _normalize_card_entry,
    "relic": _normalize_descriptions_entry,
    "monster": _normalize_monster_entry,
    "event": _normalize_descriptions_entry,
    "potion": _normalize_descriptions_entry,
}


def extract_from_jar(jar_path: Path) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """从 jar 提取 5 类简中条目。"""
    out: Dict[str, Dict[str, Dict[str, Any]]] = {k: {} for k, _ in _KIND_TO_JAR_FILE}
    with zipfile.ZipFile(str(jar_path)) as zf:
        names = set(zf.namelist())
        for kind, fname in _KIND_TO_JAR_FILE:
            path_in_jar = f"localization/zhs/{fname}"
            if path_in_jar not in names:
                print(f"[extract] WARN 缺失: {path_in_jar}", file=sys.stderr)
                continue
            raw_bytes = zf.read(path_in_jar)
            data = json.loads(raw_bytes.decode("utf-8"))
            normalizer = _NORMALIZERS[kind]
            for en_id, entry_raw in data.items():
                if not isinstance(entry_raw, dict):
                    # 极少数 entry 可能是裸字符串，包一层
                    entry_raw = {"NAME": str(entry_raw)}
                out[kind][en_id] = normalizer(entry_raw)
    return out


def extract_from_dir(zho_dir: Path) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """从已解压的目录读 5 个 JSON 文件。"""
    out: Dict[str, Dict[str, Dict[str, Any]]] = {k: {} for k, _ in _KIND_TO_JAR_FILE}
    for kind, fname in _KIND_TO_JAR_FILE:
        f = zho_dir / fname
        if not f.exists():
            print(f"[extract] WARN 缺失: {f}", file=sys.stderr)
            continue
        data = json.loads(f.read_text(encoding="utf-8"))
        normalizer = _NORMALIZERS[kind]
        for en_id, entry_raw in data.items():
            if not isinstance(entry_raw, dict):
                entry_raw = {"NAME": str(entry_raw)}
            out[kind][en_id] = normalizer(entry_raw)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="从 Steam STS jar 提取简中本地化")
    parser.add_argument("--jar", type=str, default=None, help="jar 文件绝对路径")
    parser.add_argument("--zho_dir", type=str, default=None, help="已解压 zho 目录")
    parser.add_argument(
        "--out",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "data" / "localization_zho.json"),
        help="输出 JSON 路径",
    )
    args = parser.parse_args()

    if args.zho_dir:
        zho_dir = Path(args.zho_dir).expanduser()
        if not zho_dir.is_dir():
            print(f"[extract] ERROR --zho_dir 不是目录: {zho_dir}", file=sys.stderr)
            return 2
        entries = extract_from_dir(zho_dir)
        source = f"zho_dir:{zho_dir}"
    else:
        jar_path = _find_jar_path(args.jar)
        if jar_path is None:
            print(
                "[extract] ERROR 找不到 jar。试试 --jar PATH 或设 STS_JAR_PATH 环境变量。",
                file=sys.stderr,
            )
            return 2
        print(f"[extract] 使用 jar: {jar_path} ({jar_path.stat().st_size // (1024*1024)} MB)")
        entries = extract_from_jar(jar_path)
        source = f"jar:{jar_path}"

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"source": source, "entries": entries}
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    # 打印每类条目数
    for kind in ("card", "relic", "monster", "event", "potion"):
        print(f"[extract] {kind}: {len(entries.get(kind, {}))} 条")
    size_kb = out_path.stat().st_size / 1024
    print(f"[extract] 写入 {out_path} ({size_kb:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

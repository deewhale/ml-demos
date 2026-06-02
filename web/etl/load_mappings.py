"""Mapping JSON + content_py → SQLite i18n_entries 入库。

CLI:
  python -m web.etl.load_mappings                          # 默认读 web/data/localization_zho.json
  python -m web.etl.load_mappings --localization PATH      # 自定义

流程:
  1. 加载 content_py 5 类（cards / relics / enemies / events / potions）拿英文 source-of-truth
  2. DELETE FROM i18n_entries
  3. 批量 INSERT 英文行 (source='content_py')
  4. 如果 localization JSON 存在 → 按 jar key 查找对应 en_id（cards 用 CARD_ID_ALIASES 别名表 normalize），UPSERT zh_name + zh_desc + raw_json + source='jar_zho'
  5. jar 里有但 py 找不到的 entry → 作为 orphan 行 INSERT (source='jar_zho_only')
  6. 末尾打印每类的 (total, with_zh) 统计
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 项目根（web/etl/ 上两层）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_ENGINE_PACKAGES = _PROJECT_ROOT / "external" / "StSRLSolver" / "packages"

# 默认本地化 JSON 路径
_DEFAULT_LOCALIZATION = Path(__file__).resolve().parent.parent / "data" / "localization_zho.json"


def _import_content_py() -> Dict[str, Any]:
    """import content_py 5 类。失败时抛异常（不做 ast fallback，保持简单）。

    returns: {
      'cards':   dict[str, Card],
      'card_aliases': dict[str, str],     # alias_id -> canonical_id
      'relics':  dict[str, Relic],
      'enemies': dict[str, EnemyClass],   # ENEMY_CLASSES, value 是 class
      'events':  dict[str, Event],
      'potions': dict[str, Potion],
    }
    """
    if str(_ENGINE_PACKAGES) not in sys.path:
        sys.path.insert(0, str(_ENGINE_PACKAGES))
    from engine.content.cards import ALL_CARDS, CARD_ID_ALIASES  # type: ignore
    from engine.content.relics import ALL_RELICS  # type: ignore
    from engine.content.enemies import ENEMY_CLASSES  # type: ignore
    from engine.content.events import ALL_EVENTS  # type: ignore
    from engine.content.potions import ALL_POTIONS  # type: ignore

    return {
        "cards": dict(ALL_CARDS),
        "card_aliases": dict(CARD_ID_ALIASES),
        "relics": dict(ALL_RELICS),
        "enemies": dict(ENEMY_CLASSES),
        "events": dict(ALL_EVENTS),
        "potions": dict(ALL_POTIONS),
    }


def _en_name_for(kind: str, en_id: str, obj: Any) -> str:
    """从 content_py 对象拿英文 display name。enemies 是 class，没 .name，用 en_id。"""
    if kind == "monster":
        # ENEMY_CLASSES value 是 class，无统一 name；用 en_id 本身
        return en_id
    name = getattr(obj, "name", None)
    if isinstance(name, str) and name:
        return name
    return en_id


def _enum_value(v: Any) -> Any:
    """把 enum（如 CardRarity.RARE） → 'RARE'；非 enum 原样返回。"""
    if v is None:
        return None
    # IntEnum/Enum 都有 .value 属性；优先用 .value
    val = getattr(v, "value", None)
    if val is not None and isinstance(val, (str, int)):
        return val
    return v


def _meta_for(kind: str, en_id: str, obj: Any) -> Optional[str]:
    """对 cards 提取 rarity / card_type / cost / color 写到 meta_json；其它 kind 暂不存。

    返回 JSON 字符串或 None。
    """
    if kind != "card":
        return None
    meta: Dict[str, Any] = {}
    rarity = _enum_value(getattr(obj, "rarity", None))
    card_type = _enum_value(getattr(obj, "card_type", None))
    color = _enum_value(getattr(obj, "color", None))
    cost = getattr(obj, "cost", None)
    if rarity is not None:
        meta["rarity"] = rarity
    if card_type is not None:
        meta["card_type"] = card_type
    if color is not None:
        meta["color"] = color
    if cost is not None and isinstance(cost, (int, float)):
        meta["cost"] = cost
    if not meta:
        return None
    return json.dumps(meta, ensure_ascii=False)


def _build_english_rows(content: Dict[str, Any]) -> List[Tuple[str, str, str, Optional[str]]]:
    """生成英文 source-of-truth 行 [(kind, en_id, en_name, meta_json), ...]."""
    rows: List[Tuple[str, str, str, Optional[str]]] = []
    for en_id, obj in content["cards"].items():
        rows.append((
            "card",
            en_id,
            _en_name_for("card", en_id, obj),
            _meta_for("card", en_id, obj),
        ))
    for en_id, obj in content["relics"].items():
        rows.append((
            "relic",
            en_id,
            _en_name_for("relic", en_id, obj),
            _meta_for("relic", en_id, obj),
        ))
    for en_id, obj in content["enemies"].items():
        rows.append((
            "monster",
            en_id,
            _en_name_for("monster", en_id, obj),
            _meta_for("monster", en_id, obj),
        ))
    for en_id, obj in content["events"].items():
        rows.append((
            "event",
            en_id,
            _en_name_for("event", en_id, obj),
            _meta_for("event", en_id, obj),
        ))
    for en_id, obj in content["potions"].items():
        rows.append((
            "potion",
            en_id,
            _en_name_for("potion", en_id, obj),
            _meta_for("potion", en_id, obj),
        ))
    return rows


def _normalize_jar_key(kind: str, jar_key: str, content: Dict[str, Any]) -> Optional[str]:
    """jar key → content_py en_id。

    cards 走 CARD_ID_ALIASES 别名表。其他 kind 直接相等匹配。
    返回 None 表示 py 没有此 en_id。
    """
    if kind == "card":
        canonical = content["card_aliases"].get(jar_key, jar_key)
        if canonical in content["cards"]:
            return canonical
        return None
    elif kind == "relic":
        return jar_key if jar_key in content["relics"] else None
    elif kind == "monster":
        return jar_key if jar_key in content["enemies"] else None
    elif kind == "event":
        return jar_key if jar_key in content["events"] else None
    elif kind == "potion":
        return jar_key if jar_key in content["potions"] else None
    return None


def load_mappings(localization_path: Optional[Path] = None) -> Dict[str, Any]:
    """主入口。返回统计字典 {'cards': (total, with_zh), ...}."""
    # 1. 加载 content_py
    content = _import_content_py()
    print(
        f"[load] content_py: cards={len(content['cards'])} relics={len(content['relics'])} "
        f"enemies={len(content['enemies'])} events={len(content['events'])} potions={len(content['potions'])}"
    )

    # 2. 读 localization JSON（可选）
    localization: Optional[Dict[str, Any]] = None
    loc_path = localization_path or _DEFAULT_LOCALIZATION
    if loc_path.exists():
        try:
            localization = json.loads(loc_path.read_text(encoding="utf-8"))
            print(f"[load] localization: {loc_path} ({loc_path.stat().st_size // 1024} KB)")
        except Exception as e:
            print(f"[load] WARN 读 localization 失败: {e}", file=sys.stderr)
            localization = None
    else:
        print(f"[load] WARN 没有 localization JSON ({loc_path})，只入英文行")

    # 3. 入库
    # 延迟 import 避免 content_py 失败时卡 db 初始化
    from ..db import get_conn, init_db

    init_db()  # 幂等建表
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM i18n_entries")

        # 英文行先 INSERT（带 meta_json：cards 含 rarity / card_type / cost）
        en_rows = _build_english_rows(content)
        cur.executemany(
            "INSERT INTO i18n_entries "
            "(kind, en_id, en_name, zh_name, zh_desc, source, raw_json, meta_json) "
            "VALUES (?, ?, ?, NULL, NULL, 'content_py', NULL, ?)",
            en_rows,
        )
        # 各 kind 总数（来自 content_py）
        py_counts: Dict[str, int] = {}
        for kind, _, _, _ in en_rows:
            py_counts[kind] = py_counts.get(kind, 0) + 1

        # 中文 UPSERT + 孤儿 INSERT
        upserts: Dict[str, int] = {k: 0 for k in ("card", "relic", "monster", "event", "potion")}
        orphans: Dict[str, List[str]] = {k: [] for k in upserts}
        if localization is not None:
            entries_by_kind = localization.get("entries") or {}
            for kind, kind_dict in entries_by_kind.items():
                if kind not in upserts:
                    continue
                for jar_key, payload in kind_dict.items():
                    en_id = _normalize_jar_key(kind, jar_key, content)
                    zh_name = payload.get("zh_name") or None
                    zh_desc = payload.get("zh_desc") or None
                    raw = payload.get("raw")
                    raw_json = json.dumps(raw, ensure_ascii=False) if raw is not None else None
                    if en_id is not None:
                        cur.execute(
                            "UPDATE i18n_entries SET zh_name=?, zh_desc=?, source='jar_zho', raw_json=? "
                            "WHERE kind=? AND en_id=?",
                            (zh_name, zh_desc, raw_json, kind, en_id),
                        )
                        if cur.rowcount > 0:
                            upserts[kind] += 1
                    else:
                        # 孤儿：jar 有但 py 没有
                        cur.execute(
                            "INSERT OR REPLACE INTO i18n_entries "
                            "(kind, en_id, en_name, zh_name, zh_desc, source, raw_json) "
                            "VALUES (?, ?, NULL, ?, ?, 'jar_zho_only', ?)",
                            (kind, jar_key, zh_name, zh_desc, raw_json),
                        )
                        orphans[kind].append(jar_key)

        # monster alias 注入：eval 数据里 boss_en_id 用带空格的 STS 内部 display name
        # （如 'Slime Boss'），但 content_py 的 ENEMY_CLASSES key 是无空格 'SlimeBoss'。
        # 直接复制一份带空格的 alias 行（含已 UPSERT 的中文），让 lookup 双向都能查到。
        # 列表参考 STS Java 端 *.id 与 *Strings.json key 的实际差异（手工对齐）。
        _MONSTER_ALIASES = {
            "Slime Boss": "SlimeBoss",
            "Jaw Worm": "JawWorm",
            "Fungi Beast": "FungiBeast",
            "Gremlin Nob": "GremlinNob",
            "Snake Plant": "SnakePlant",
            "Spheric Guardian": "SphericGuardian",
            "Writhing Mass": "WrithingMass",
            "Giant Head": "GiantHead",
            "Awakened One": "AwakenedOne",
            "Time Eater": "TimeEater",
            "Bronze Automaton": "BronzeAutomaton",
            "The Guardian": "TheGuardian",
            "The Collector": "TheCollector",
            "Spire Shield": "SpireShield",
            "Spire Spear": "SpireSpear",
            "Corrupt Heart": "CorruptHeart",
        }
        alias_inserted = 0
        for alias_id, canon_id in _MONSTER_ALIASES.items():
            row = cur.execute(
                "SELECT en_name, zh_name, zh_desc, raw_json, meta_json FROM i18n_entries "
                "WHERE kind='monster' AND en_id=?",
                (canon_id,),
            ).fetchone()
            if row is None:
                continue
            cur.execute(
                "INSERT OR REPLACE INTO i18n_entries "
                "(kind, en_id, en_name, zh_name, zh_desc, source, raw_json, meta_json) "
                "VALUES ('monster', ?, ?, ?, ?, 'alias', ?, ?)",
                (
                    alias_id,
                    row["en_name"] or canon_id,
                    row["zh_name"],
                    row["zh_desc"],
                    row["raw_json"],
                    row["meta_json"],
                ),
            )
            alias_inserted += 1
        if alias_inserted:
            print(f"[load] monster alias 注入 {alias_inserted} 条（带空格 display name）")

        # 通用 no-space alias 注入（P6 修复 event_id alias 缺口）：
        # 训练日志里 event_id（以及部分 card/relic/potion）为 CamelCase 无空格
        # （如 'BonfireElementals' / 'AccursedBlacksmith'），但 i18n_entries 表里
        # en_id 是带空格的 display name（'Bonfire Elementals'）。LEFT JOIN 无法匹配，
        # 中文显示为英文 fallback。
        # 解决：每条带空格 en_id 额外 INSERT 一条 en_id.replace(' ','') 的 alias 行
        # （source='alias_no_space'），复用同一行的中文/raw。
        # monster 跳过（上面 _MONSTER_ALIASES 手工双向映射已经覆盖）。
        no_space_inserted: Dict[str, int] = {k: 0 for k in ("card", "relic", "event", "potion")}
        space_rows = cur.execute(
            "SELECT kind, en_id, en_name, zh_name, zh_desc, raw_json, meta_json "
            "FROM i18n_entries "
            "WHERE kind IN ('card','relic','event','potion') AND en_id LIKE '% %'"
        ).fetchall()
        for row in space_rows:
            kind = row["kind"]
            no_space_id = row["en_id"].replace(" ", "")
            if no_space_id == row["en_id"]:
                continue
            # 用 INSERT OR REPLACE 兜底（如果 no_space_id 已存在则覆盖，
            # 但实际上 content_py 自身 en_id 不带空格，这里 alias 是新行）
            cur.execute(
                "INSERT OR REPLACE INTO i18n_entries "
                "(kind, en_id, en_name, zh_name, zh_desc, source, raw_json, meta_json) "
                "VALUES (?, ?, ?, ?, ?, 'alias_no_space', ?, ?)",
                (
                    kind,
                    no_space_id,
                    row["en_name"] or no_space_id,
                    row["zh_name"],
                    row["zh_desc"],
                    row["raw_json"],
                    row["meta_json"],
                ),
            )
            no_space_inserted[kind] += 1
        for kind, n in no_space_inserted.items():
            if n:
                print(f"[load] {kind} no-space alias 注入 {n} 条")

        conn.commit()

        # 4. 验证查询
        verify_rows = cur.execute(
            "SELECT kind, COUNT(*) AS total, "
            "SUM(CASE WHEN zh_name IS NOT NULL AND zh_name != '' THEN 1 ELSE 0 END) AS with_zh "
            "FROM i18n_entries GROUP BY kind ORDER BY kind"
        ).fetchall()
        stats: Dict[str, Tuple[int, int]] = {}
        print("[load] === 入库统计 ===")
        for row in verify_rows:
            kind = row["kind"]
            total = row["total"]
            with_zh = row["with_zh"] or 0
            ratio = (with_zh / total * 100) if total else 0
            print(f"[load] {kind}: total={total} with_zh={with_zh} ({ratio:.1f}%)")
            stats[kind] = (total, with_zh)

        # 5. 孤儿条目（jar 有但 py 没有）
        print("[load] === 孤儿条目（jar_zho_only，没有 content_py 对应）===")
        for kind in ("card", "relic", "monster", "event", "potion"):
            if orphans[kind]:
                print(f"[load] {kind} 孤儿 {len(orphans[kind])} 个: {orphans[kind][:20]}")

        return {"stats": stats, "orphans": orphans, "upserts": upserts}
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="把 mapping JSON + content_py 入库")
    parser.add_argument(
        "--localization",
        type=str,
        default=str(_DEFAULT_LOCALIZATION),
        help="localization JSON 路径（默认 web/data/localization_zho.json）",
    )
    args = parser.parse_args()
    load_mappings(Path(args.localization).expanduser())
    return 0


if __name__ == "__main__":
    sys.exit(main())

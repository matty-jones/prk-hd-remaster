"""Import the ao-vk texture catalogue and reconcile it with the RDB."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from . import rdb
from .registry import Registry

KNOWN_COLUMNS = {
    "type", "id", "kind", "name", "fmt", "w", "h", "nbytes", "alpha", "seam",
    "flat", "cutout", "category", "atlas", "text", "family", "family_n",
    "in_family", "cluster", "duplicate_of", "risk", "mean_r", "mean_g",
    "mean_b", "dhash",
}

SURFACE = {"terrain", "stone", "architecture", "metal", "interior"}


def _text(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _int(value, default: int = 0) -> int:
    text = _text(value)
    if text == "":
        return default
    return int(float(text))


def _float(value, default: float = 0.0) -> float:
    text = _text(value)
    if text == "":
        return default
    return float(text)


def _flag(value) -> int:
    text = _text(value).lower()
    if text in ("", "0", "false", "no"):
        return 0
    if text in ("1", "true", "yes"):
        return 1
    try:
        return 1 if float(text) != 0 else 0
    except ValueError:
        return 0


def disposition_for(row: dict) -> str:
    """Map one catalogue row to an exclusive processing disposition.

    Duplicate rows are marked ``duplicate`` and resolved onto their canonical
    row's disposition after the whole file is loaded.
    """
    fmt = _text(row.get("fmt")).lower()
    category = _text(row.get("category")).lower()
    name = _text(row.get("name")).lower()
    risk = _text(row.get("risk")).lower()
    table = _int(row.get("type"))
    if fmt == "unreadable":
        return "decode-error"
    if table == 1010008 or category == "ui":
        return "skip-ui"
    if category == "shadow" or name.startswith("shadow"):
        return "skip-mask"
    if risk == "auto":
        if _flag(row.get("atlas")) or _flag(row.get("text")) or _flag(row.get("cutout")):
            return "specialist"
        if _float(row.get("alpha")) > 0.02:
            return "specialist"
        if category not in SURFACE:
            return "needs-classification"
        return "remaster"
    if risk == "specialist":
        return "specialist"
    if risk == "review":
        return "needs-classification"
    if risk == "skip":
        if _text(row.get("duplicate_of")):
            return "duplicate"
        return "skip-placeholder"
    return "needs-classification"


def _tags(row: dict) -> list[str]:
    tags = []
    if _float(row.get("alpha")) <= 0.02:
        tags.append("opaque")
    else:
        tags.append("alpha")
    if _float(row.get("seam"), 99) <= 2.2:
        tags.append("possibly_tileable")
    category = _text(row.get("category"))
    if category:
        tags.append(category)
    for flag, label in (("flat", "flat"), ("cutout", "cutout"), ("atlas", "atlas"), ("text", "text")):
        if _flag(row.get(flag)):
            tags.append(label)
    kind = _text(row.get("kind"))
    if kind:
        tags.append(kind)
    return tags


def row_to_asset(row: dict, prompt_version: str) -> dict:
    unknown = {key: value for key, value in row.items() if key not in KNOWN_COLUMNS}
    metadata = {key: row.get(key, "") for key in sorted(KNOWN_COLUMNS)}
    metadata.update(unknown)
    table = _int(row.get("type"))
    resource_id = _int(row.get("id"))
    return {
        "key": f"{table}:{resource_id}",
        "rdb_type": table,
        "rdb_id": resource_id,
        "source_sha256": None,
        "pixel_sha256": None,
        "source_width": _int(row.get("w")),
        "source_height": _int(row.get("h")),
        "source_format": _text(row.get("fmt")),
        "name": _text(row.get("name")),
        "kind": _text(row.get("kind")),
        "family": _text(row.get("family")),
        "material_class": _text(row.get("category")),
        "tags": _tags(row),
        "disposition": disposition_for(row),
        "family_prompt_version": prompt_version,
        "source_metadata": metadata,
        "risk": _text(row.get("risk")),
        "seam": _float(row.get("seam"), 99.0),
        "alpha": _float(row.get("alpha")),
        "duplicate_of": _text(row.get("duplicate_of")),
    }


def resolve_duplicates(assets: list[dict]) -> None:
    by_key = {asset["key"]: asset for asset in assets}
    for asset in assets:
        if asset["disposition"] != "duplicate":
            continue
        seen = set()
        cursor = asset
        inherited = "needs-classification"
        canonical = ""
        while cursor is not None and cursor["key"] not in seen:
            seen.add(cursor["key"])
            parent_key = cursor.get("duplicate_of") or ""
            if not parent_key:
                inherited = cursor["disposition"] if cursor["disposition"] != "duplicate" else "needs-classification"
                canonical = cursor["key"]
                break
            parent = by_key.get(parent_key)
            if parent is None:
                inherited = "needs-classification"
                canonical = parent_key
                break
            if parent["disposition"] != "duplicate":
                inherited = parent["disposition"]
                canonical = parent["key"]
                break
            cursor = parent
        asset["disposition"] = inherited
        asset["canonical_key"] = canonical
        metadata = asset["source_metadata"]
        metadata["inherited_disposition"] = inherited
        metadata["canonical_key"] = canonical


def read_csv(csv_path: Path) -> list[dict]:
    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path} has no header")
        return list(reader)


def import_csv(csv_path: Path, registry: Registry, prompt_version: str) -> dict:
    rows = read_csv(csv_path)
    assets = [row_to_asset(row, prompt_version) for row in rows]
    resolve_duplicates(assets)
    for asset in assets:
        registry.upsert_asset(asset, commit=False)
    registry.conn.commit()
    counts = Counter(asset["disposition"] for asset in assets)
    risks = Counter(asset["risk"] for asset in assets)
    cohort = [asset for asset in assets if asset["risk"] == "auto"]
    categories = Counter(asset["material_class"] for asset in cohort)
    return {
        "rows": len(assets),
        "dispositions": dict(counts),
        "risks": dict(risks),
        "cohort_auto": len(cohort),
        "cohort_categories": dict(categories),
        "unknown_columns": sorted({
            key for row in rows for key in row if key not in KNOWN_COLUMNS
        }),
    }


def reconcile(registry: Registry, db_path: Path | None = None, lod_tables: dict | None = None) -> dict:
    """Coverage report. Opens the RDB read-only when a path is given."""
    assets = registry.all_assets()
    by_risk = Counter(asset["risk"] for asset in assets)
    by_disp = Counter(asset["disposition"] for asset in assets)
    auto = [asset for asset in assets if asset["risk"] == "auto"]
    by_cat = Counter(asset["material_class"] for asset in auto)
    skip_reasons = Counter()
    for asset in assets:
        if asset["risk"] != "skip" and asset["disposition"] != "decode-error":
            continue
        metadata = asset["source_metadata"] if isinstance(asset["source_metadata"], dict) else {}
        if asset["disposition"] == "decode-error" or metadata.get("fmt") == "unreadable":
            skip_reasons["unreadable"] += 1
        elif asset.get("duplicate_of"):
            skip_reasons["duplicate"] += 1
        else:
            skip_reasons["placeholder_or_flat"] += 1
    report = {
        "imported": len(assets),
        "risks": dict(by_risk),
        "dispositions": dict(by_disp),
        "cohort_auto": len(auto),
        "cohort_categories": dict(by_cat),
        "exclusion_reasons": dict(skip_reasons),
        "decodable_estimate": len(assets) - skip_reasons["unreadable"] - skip_reasons["placeholder_or_flat"] - 0,
    }
    # The catalogue keeps duplicate and unreadable rows. Decodable images are
    # every imported row that is not an unreadable or placeholder skip.
    report["decodable_estimate"] = sum(
        1 for asset in assets if asset["disposition"] not in {"decode-error", "skip-placeholder"}
    )
    if db_path is None:
        report["rdb"] = None
        return report
    connection = rdb.connect(db_path)
    try:
        counts = {table: rdb.table_count(connection, table) for table in CANON_TABLES}
        missing = {}
        for table in CANON_TABLES:
            present = rdb.table_ids(connection, table)
            wanted = {asset["rdb_id"] for asset in assets if asset["rdb_type"] == table}
            missing[str(table)] = sorted(wanted - present)
        report["rdb"] = {
            "counts": {str(key): value for key, value in counts.items()},
            "missing_ids": missing,
            "lod": rdb.lod_census(connection, lod_tables or {}),
        }
    finally:
        connection.close()
    return report


CANON_TABLES = (1010004, 1010006, 1010008, 1010009, 1010011)


def format_reconcile(report: dict) -> str:
    lines = [
        f"imported {report['imported']} catalogue rows",
        f"risk=auto cohort {report['cohort_auto']}",
        "categories: " + json.dumps(report["cohort_categories"], sort_keys=True),
        "dispositions: " + json.dumps(report["dispositions"], sort_keys=True),
        "skip/decode reasons: " + json.dumps(report["exclusion_reasons"], sort_keys=True),
        f"rows with a usable disposition (not decode-error or placeholder): {report['decodable_estimate']}",
    ]
    rdb_report = report.get("rdb")
    if rdb_report:
        lines.append("rdb counts: " + json.dumps(rdb_report["counts"], sort_keys=True))
        missing = {key: value for key, value in rdb_report["missing_ids"].items() if value}
        lines.append(f"catalogue ids missing from rdb: {sum(len(v) for v in missing.values())}")
        for entry in rdb_report.get("lod") or []:
            lines.append(
                f"lod {entry['table']} -> {entry['parent']}: {entry['rows']} rows, "
                f"{entry['shared_ids']} shared, disposition {entry['disposition']}"
            )
    return "\n".join(lines) + "\n"

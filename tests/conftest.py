"""Synthetic fixtures. No Project Rubi-Ka art."""

from __future__ import annotations

import io
import sqlite3
from pathlib import Path

import pytest
from PIL import Image

from prk_remaster.config import load_config


def tiny_png(color=(20, 140, 60), size=32) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (size, size), color).save(buffer, format="PNG")
    return buffer.getvalue()


def ground_blob(payload: bytes, name: str = "sand") -> bytes:
    header = name.encode("latin1") + b"\x00"
    header = header + b"\xcd" * (24 - len(header))
    return header + payload


def tiny_jpeg(color=(90, 70, 40), size=32) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (size, size), color).save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("PRK_WORK_ROOT", str(tmp_path / "work"))
    loaded = load_config()
    loaded["prk_root"] = str(tmp_path / "client")
    loaded["material_rules"]["resolution"] = {
        "terrain": 64, "stone": 64, "architecture": 64, "interior": 64, "metal": 64, "default": 64,
    }
    return loaded


def write_rdb(path: Path, rows: dict[int, dict[int, bytes]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    try:
        for table, items in rows.items():
            connection.execute(f"CREATE TABLE rdb_{table} (id INTEGER PRIMARY KEY, version INTEGER, data BLOB)")
            for resource_id, blob in items.items():
                connection.execute(
                    f"INSERT INTO rdb_{table} (id, version, data) VALUES (?, ?, ?)",
                    (resource_id, 1000000, blob),
                )
        connection.commit()
    finally:
        connection.close()


def csv_text(rows: list[dict]) -> str:
    header = [
        "type", "id", "kind", "name", "fmt", "w", "h", "nbytes", "alpha", "seam",
        "flat", "cutout", "category", "atlas", "text", "family", "family_n",
        "in_family", "cluster", "duplicate_of", "risk", "mean_r", "mean_g",
        "mean_b", "dhash", "extra_note",
    ]
    lines = [",".join(header)]
    for row in rows:
        lines.append(",".join(str(row.get(column, "")) for column in header))
    return "\n".join(lines) + "\n"


def auto_row(**overrides) -> dict:
    row = {
        "type": 1010006, "id": 1, "kind": "ground", "name": "sand", "fmt": "jpeg",
        "w": 32, "h": 32, "nbytes": 10, "alpha": 0, "seam": 0.4, "flat": 0,
        "cutout": 0, "category": "terrain", "atlas": 0, "text": 0, "family": "ground:sand",
        "family_n": 1, "in_family": 0, "cluster": -1, "duplicate_of": "", "risk": "auto",
        "mean_r": 90, "mean_g": 70, "mean_b": 40, "dhash": 1, "extra_note": "kept",
    }
    row.update(overrides)
    return row

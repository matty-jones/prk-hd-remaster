"""Read-only access to a Project Rubi-Ka resource database."""

from __future__ import annotations

import io
import sqlite3
import struct
from pathlib import Path

from PIL import Image

CANON = {
    1010004: "world",
    1010006: "ground",
    1010008: "icon",
    1010009: "misc",
    1010011: "skin",
}


class RdbError(RuntimeError):
    pass


def connect(path: Path | str) -> sqlite3.Connection:
    """Open the database in SQLite read-only mode. Writes raise."""
    database = Path(path).resolve()
    if not database.is_file():
        raise RdbError(f"RDB not found: {database}")
    uri = f"file:{database}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def image_bytes(table: int, data: bytes) -> tuple[str, bytes]:
    """Return (embedded name, compressed image bytes) for one row.

    Ground rows store a 24-byte Latin-1 name in front of the JPEG or PNG.
    A few rows overwrite the PNG signature's first four bytes with 02 00 00 00.
    """
    if not data:
        return "", b""
    if int(table) == 1010006:
        name = data[:24].split(b"\x00", 1)[0].decode("latin1", "replace")
        payload = data[24:]
    else:
        name, payload = "", data
    if payload[:4] == b"\x02\x00\x00\x00" and payload[4:8] == b"\r\n\x1a\n":
        payload = b"\x89PNG" + payload[4:]
    return name, payload


def load_names(connection: sqlite3.Connection) -> dict[int, dict[int, str]]:
    """Parse the client resource directory in rdb_1000010."""
    row = connection.execute("SELECT data FROM rdb_1000010").fetchone()
    if row is None or not row[0]:
        return {}
    blob = row[0]
    count = struct.unpack_from("<I", blob, 0)[0]
    offset = 4
    names: dict[int, dict[int, str]] = {}
    for _ in range(count):
        table, count_ids = struct.unpack_from("<II", blob, offset)
        offset += 8
        table_names: dict[int, str] = {}
        for _ in range(count_ids):
            resource_id, length = struct.unpack_from("<II", blob, offset)
            table_names[resource_id] = blob[offset + 8 : offset + 8 + length].decode("latin1")
            offset += 8 + length
        names[table] = table_names
    return names


def open_image(payload: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(payload))
    image.load()
    return image


def extract(connection: sqlite3.Connection, table: int, resource_id: int) -> dict:
    query = f"SELECT id, data FROM rdb_{int(table)} WHERE id = ?"
    row = connection.execute(query, (int(resource_id),)).fetchone()
    if row is None:
        raise RdbError(f"missing RDB texture {table}:{resource_id}")
    embedded, payload = image_bytes(int(table), row["data"])
    image = open_image(payload)
    return {
        "rdb_type": int(table),
        "rdb_id": int(resource_id),
        "embedded_name": embedded,
        "payload": payload,
        "image": image,
        "source_format": image.format or "unknown",
        "width": image.size[0],
        "height": image.size[1],
    }


def table_ids(connection: sqlite3.Connection, table: int) -> set[int]:
    rows = connection.execute(f"SELECT id FROM rdb_{int(table)}").fetchall()
    return {int(row[0]) for row in rows}


def table_count(connection: sqlite3.Connection, table: int) -> int:
    row = connection.execute(f"SELECT COUNT(*) FROM rdb_{int(table)}").fetchone()
    return int(row[0]) if row else 0


def lod_census(connection: sqlite3.Connection, lod_tables: dict) -> list[dict]:
    """Count lower-quality copies. Does not decode them."""
    census = []
    for lod, parent in lod_tables.items():
        lod_i = int(lod)
        parent_i = int(parent)
        rows = table_count(connection, lod_i)
        shared = connection.execute(
            f"SELECT COUNT(*) FROM rdb_{parent_i} a JOIN rdb_{lod_i} b ON a.id = b.id"
        ).fetchone()[0]
        census.append({
            "table": lod_i,
            "parent": parent_i,
            "rows": rows,
            "shared_ids": int(shared),
            "disposition": "skip-lod",
        })
    return census

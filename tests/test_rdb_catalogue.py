import sqlite3
from pathlib import Path

import pytest

from prk_remaster import rdb
from prk_remaster.catalogue import import_csv, reconcile
from prk_remaster.registry import Registry
from tests.conftest import auto_row, csv_text, tiny_jpeg, tiny_png, write_rdb


def test_image_bytes_repairs_png_and_reads_ground_name():
    png = tiny_png()
    wrapped = b"\x02\x00\x00\x00" + png[4:]
    name, payload = rdb.image_bytes(1010004, wrapped)
    assert name == ""
    assert payload.startswith(b"\x89PNG")
    assert rdb.open_image(payload).size == (32, 32)
    header = b"sand\x00" + b"\xcd" * 19
    jpeg = tiny_jpeg()
    gname, gpayload = rdb.image_bytes(1010006, header + jpeg)
    assert gname == "sand"
    assert gpayload.startswith(b"\xff\xd8")


def test_rdb_connection_is_read_only(tmp_path):
    database = tmp_path / "rdb.db"
    write_rdb(database, {1010004: {7: tiny_png()}})
    before = database.read_bytes()
    connection = rdb.connect(database)
    with pytest.raises(sqlite3.OperationalError):
        connection.execute("INSERT INTO rdb_1010004 (id, version, data) VALUES (8, 1, x'00')")
    connection.close()
    assert database.read_bytes() == before
    record = rdb.extract(rdb.connect(database), 1010004, 7)
    assert record["width"] == 32
    record["image"].close()


def test_catalogue_dispositions_and_reconcile(tmp_path):
    rows = [
        auto_row(id=1, name="sand"),
        auto_row(id=2, name="sand", risk="skip", duplicate_of="1010006:1"),
        auto_row(id=3, name="shadow1", category="shadow", risk="specialist"),
        auto_row(type=1010008, id=4, kind="icon", name="icon", category="ui", risk="specialist"),
        auto_row(id=5, name="", fmt="unreadable", risk="skip", category="other"),
        auto_row(id=6, name="brick", category="stone", family="brick"),
        auto_row(id=7, name="floor", category="interior", risk="review"),
    ]
    csv_path = tmp_path / "assets.csv"
    csv_path.write_text(csv_text(rows), encoding="utf-8")
    registry = Registry(tmp_path / "registry.sqlite")
    summary = import_csv(csv_path, registry, "v1")
    assert summary["rows"] == 7
    assert summary["unknown_columns"] == ["extra_note"]
    assert registry.asset("1010006:1")["disposition"] == "remaster"
    assert "kept" in str(registry.asset("1010006:1")["source_metadata"])
    duplicate = registry.asset("1010006:2")
    assert duplicate["disposition"] == "remaster"
    assert duplicate["canonical_key"] == "1010006:1"
    assert registry.asset("1010006:3")["disposition"] == "skip-mask"
    assert registry.asset("1010008:4")["disposition"] == "skip-ui"
    assert registry.asset("1010006:5")["disposition"] == "decode-error"
    assert registry.asset("1010006:7")["disposition"] == "needs-classification"
    report = reconcile(registry)
    assert report["cohort_auto"] == 2
    assert report["cohort_categories"]["terrain"] == 1
    assert report["cohort_categories"]["stone"] == 1
    assert report["exclusion_reasons"]["unreadable"] == 1
    assert report["exclusion_reasons"]["duplicate"] == 1


def test_real_catalogue_cohort_when_present():
    csv_path = Path.home() / "Games/ao-vk/TEMP/texture-catalogue/assets.csv"
    if not csv_path.is_file():
        pytest.skip("catalogue is not on this machine")
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        registry = Registry(Path(tmp) / "registry.sqlite")
        summary = import_csv(csv_path, registry, "v1")
    assert summary["rows"] == 20859
    assert summary["cohort_auto"] == 1076
    assert summary["cohort_categories"] == {
        "terrain": 312, "architecture": 288, "metal": 186, "interior": 176, "stone": 114,
    }
    assert summary["risks"]["skip"] == 449

import json
from pathlib import Path

from PIL import Image

from prk_remaster.deploy import disable, install, snapshot_rdb, uninstall
from prk_remaster.imaging import atomic_save_png
from prk_remaster.pack import map_filename


def _pack(root: Path) -> Path:
    root.mkdir(parents=True)
    for suffix, mode, color in (
        ("d", "RGB", (10, 20, 30)),
        ("n", "RGB", (128, 128, 255)),
        ("h", "L", 140),
        ("orm", "RGB", (255, 200, 0)),
    ):
        atomic_save_png(Image.new(mode, (8, 8), color), root / map_filename(1010006, 15, suffix))
    manifest = {
        "name": "demo",
        "assets": [{
            "key": "1010006:15",
            "rdb_type": 1010006,
            "rdb_id": 15,
            "source_sha256": "abc",
            "shared_from": None,
            "maps": {},
        }],
    }
    (root / "pack-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_install_disable_uninstall_does_not_touch_rdb(tmp_path):
    client = tmp_path / "client"
    rdb = client / "cd_image" / "rdb.db"
    rdb.parent.mkdir(parents=True)
    rdb.write_bytes(b"sqlite-fixture-not-a-real-database")
    before = rdb.read_bytes()
    stamp = snapshot_rdb(client)
    pack = _pack(tmp_path / "demo")
    plan = install(client, pack, receipt_dir=tmp_path / "receipts", dry_run=True)
    assert plan["order_after"] == ["demo"]
    assert not (client / "randy-vk" / "packs" / "demo").exists()
    install(client, pack, receipt_dir=tmp_path / "receipts", dry_run=False)
    assert (client / "randy-vk" / "packs" / "demo" / map_filename(1010006, 15, "d")).is_file()
    order = (client / "randy-vk" / "packs" / "order.txt").read_text(encoding="utf-8")
    assert order.splitlines()[0] == "demo"
    assert rdb.read_bytes() == before
    assert snapshot_rdb(client) == stamp
    held = disable(client, "demo", receipt_dir=tmp_path / "receipts", yes=False)
    assert held["applied"] is False
    assert "demo" in (client / "randy-vk" / "packs" / "order.txt").read_text(encoding="utf-8")
    disable(client, "demo", receipt_dir=tmp_path / "receipts", yes=True)
    assert "demo" not in (client / "randy-vk" / "packs" / "order.txt").read_text(encoding="utf-8")
    assert (client / "randy-vk" / "packs" / "demo").is_dir()
    preview = uninstall(client, "demo", receipt_dir=tmp_path / "receipts", dry_run=True, yes=True)
    assert preview["applied"] is False
    uninstall(client, "demo", receipt_dir=tmp_path / "receipts", yes=True)
    assert not (client / "randy-vk" / "packs" / "demo").exists()
    assert rdb.read_bytes() == before

from prk_remaster.catalogue import import_csv
from prk_remaster.imaging import pixel_sha256
from prk_remaster.pbr import cpu_bump_depth
from prk_remaster.pack import build_pack, map_filename, validate_pack
from prk_remaster.registry import Registry
from prk_remaster.runner import run_cohort
from tests.conftest import auto_row, csv_text, ground_blob, tiny_png, write_rdb
from PIL import Image


def _edit(image, prompt, seed, settings):
    return image


def test_runner_packs_and_resumes(cfg, tmp_path):
    client = tmp_path / "client"
    png = tiny_png((40, 120, 70))
    other = tiny_png((180, 40, 40))
    write_rdb(client / "cd_image" / "rdb.db", {
        1010006: {1: ground_blob(png), 2: ground_blob(png), 3: ground_blob(other, "rock")},
    })
    cfg["prk_root"] = str(client)
    rows = [
        auto_row(id=1),
        auto_row(id=2, name="sand_b", family="ground:sand"),
        auto_row(id=3, name="rock", category="stone", family="ground:rock"),
    ]
    csv_path = tmp_path / "assets.csv"
    csv_path.write_text(csv_text(rows), encoding="utf-8")
    registry = Registry(tmp_path / "work" / "registry.sqlite")
    import_csv(csv_path, registry, "v1")
    calls = {"n": 0}

    def edit(image, prompt, seed, settings):
        calls["n"] += 1
        return image

    run_dir = tmp_path / "work" / "runs" / "cohort"
    first = run_cohort(
        cfg, registry, run_dir, cohort="first-1076", preset="aggressive", run_id="cohort",
        edit_fn=edit, pbr_fn=cpu_bump_depth, measure_vram=False,
    )
    assert calls["n"] == 2
    summary = first["summary"]["counts"]
    assert summary.get("qa_passed") == 3
    assert registry.job("cohort", "1010006:2")["canonical_key"] == "1010006:1"
    left = pixel_sha256(Image.open(run_dir / "assets" / "1010006_1" / "source.png"))
    right = pixel_sha256(Image.open(run_dir / "assets" / "1010006_2" / "source.png"))
    assert left == right
    run_cohort(
        cfg, registry, run_dir, cohort="first-1076", preset="aggressive", run_id="cohort",
        resume=True, edit_fn=edit, pbr_fn=cpu_bump_depth, measure_vram=False,
    )
    assert calls["n"] == 2
    manifest = build_pack(registry, "cohort", tmp_path / "pack", "prk-hd-v01")
    assert validate_pack(tmp_path / "pack", manifest) == []
    assert (tmp_path / "pack" / map_filename(1010006, 2, "d")).is_file()
    assert (tmp_path / "pack" / map_filename(1010006, 3, "orm")).is_file()
    assert registry.job("cohort", "1010006:1")["status"] == "packed"


def test_retry_after_failure(cfg, tmp_path):
    client = tmp_path / "client"
    write_rdb(client / "cd_image" / "rdb.db", {1010006: {1: ground_blob(tiny_png())}})
    cfg["prk_root"] = str(client)
    csv_path = tmp_path / "assets.csv"
    csv_path.write_text(csv_text([auto_row(id=1)]), encoding="utf-8")
    registry = Registry(tmp_path / "work" / "registry.sqlite")
    import_csv(csv_path, registry, "v1")
    state = {"n": 0}

    def edit(image, prompt, seed, settings):
        state["n"] += 1
        if state["n"] == 1:
            raise RuntimeError("worker crashed")
        return image

    run_dir = tmp_path / "run"
    run_cohort(
        cfg, registry, run_dir, cohort="first-1076", preset="aggressive", run_id="r",
        edit_fn=edit, pbr_fn=cpu_bump_depth, measure_vram=False,
    )
    assert registry.job("r", "1010006:1")["status"] == "failed"
    run_cohort(
        cfg, registry, run_dir, cohort="first-1076", preset="aggressive", run_id="r",
        resume=True, retry_failed=True, edit_fn=edit, pbr_fn=cpu_bump_depth, measure_vram=False,
    )
    assert registry.job("r", "1010006:1")["status"] == "qa_passed"
    assert state["n"] == 2

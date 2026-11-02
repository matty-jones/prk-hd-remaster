import os

import numpy as np
from PIL import Image

from prk_remaster.config import load_config
from prk_remaster.imaging import sha256_file
from prk_remaster.registry import Registry
from prk_remaster.salvage import run_salvage


def test_salvage_anchors_without_writing_the_source_run(tmp_path):
    cfg = load_config()
    cfg["work_root"] = str(tmp_path)
    source = np.full((48, 48, 3), (40, 28, 24), np.uint8)
    source[16:24, :] = (150, 40, 160)
    qwen = np.full((96, 96, 3), (180, 190, 200), np.uint8)
    qwen[::8, :] = (70, 130, 220)
    v1 = tmp_path / "runs" / "v1run" / "assets" / "1010004_9"
    v1.mkdir(parents=True)
    Image.fromarray(source).save(v1 / "source.png")
    Image.fromarray(qwen).save(v1 / "raw_qwen.png")
    Image.fromarray(qwen).save(v1 / "albedo_final.png")
    before = sha256_file(v1 / "raw_qwen.png")
    registry = Registry(tmp_path / "registry.sqlite")
    registry.upsert_asset({
        "key": "1010004:9",
        "rdb_type": 1010004,
        "rdb_id": 9,
        "name": "ai_building_neutral_wall_tech_stripe_blue.png",
        "family": "ai_building",
        "material_class": "architecture",
        "disposition": "remaster",
        "risk": "auto",
        "source_width": 48,
        "source_height": 48,
        "seam": 0.4,
        "alpha": 0,
    })
    registry.create_run("v1run", "first-1076", "aggressive", cfg)
    registry.upsert_job({
        "run_id": "v1run",
        "asset_key": "1010004:9",
        "status": "packed",
        "stage": "pack",
        "seed": 7,
        "prompt": "old",
        "cache_key": "old",
        "target_w": 96,
        "target_h": 96,
        "attempt": 0,
        "rejection_reason": None,
        "canonical_key": None,
        "detail": {"generation_label": "qwen-native"},
    })
    result = run_salvage(
        cfg, registry, from_run="v1run", run_id="salvage-test", preset="aggressive",
        stop_after="anchor",
    )
    assert result["run_id"] == "salvage-test"
    assert sha256_file(v1 / "raw_qwen.png") == before
    assert sha256_file(v1 / "albedo_final.png") == before
    anchored = tmp_path / "runs" / "salvage-test" / "assets" / "1010004_9"
    assert (anchored / "albedo_final.png").is_file()
    assert (anchored / "source.png").stat().st_ino == (v1 / "source.png").stat().st_ino
    assert os.path.samefile(anchored / "raw_qwen.png", v1 / "raw_qwen.png")
    job = registry.job("salvage-test", "1010004:9")
    assert job["status"] == "generated"
    assert job["detail"]["anchored"] is True
    assert "cable" not in (job.get("prompt") or "")
    assert (tmp_path / "runs" / "salvage-test" / "gallery" / "dramatic-triptych.png").is_file()

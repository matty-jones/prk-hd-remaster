import numpy as np
from PIL import Image

from prk_remaster.config import load_config
from prk_remaster.imaging import pixel_sha256
from prk_remaster.pbr import cpu_bump_depth
from prk_remaster.pack import map_filename
from prk_remaster.refine import run_refine
from prk_remaster.registry import Registry
from prk_remaster.scale import (
    band_rms,
    meso_agreement,
    prefer_regen,
    restrain_albedo,
    scale_metrics,
    scale_mode,
    split_bands,
    texel_px,
)
from prk_remaster.scale import _lab


def _policy():
    return load_config()["scale_policy"]


def _slab(size: int = 64) -> Image.Image:
    image = np.zeros((size, size, 3), np.uint8)
    image[:, : size // 2] = (70, 60, 40)
    image[:, size // 2 :] = (170, 145, 90)
    speck = image.copy()
    speck[::2, ::2] = np.clip(speck[::2, ::2].astype(np.int16) + 90, 0, 255)
    return Image.fromarray(speck)


def test_scale_mode_restrains_floors_and_keeps_cable_and_terrain():
    policy = _policy()
    assert scale_mode("", "manufactured", "interior", policy) == "restrained"
    assert scale_mode("wall", "manufactured", "architecture", policy) == "restrained"
    assert scale_mode("stone", "organic", "stone", policy) == "restrained"
    assert scale_mode("cable", "manufactured", "metal", policy) == "expressive"
    assert scale_mode("dirt", "organic", "terrain", policy) == "expressive"
    assert scale_mode("grass", "organic", "terrain", policy) == "expressive"
    assert scale_mode("", "organic", "terrain", policy) == "expressive"


def test_restrain_drops_speckle_and_keeps_the_slab():
    policy = _policy()
    source = Image.new("RGB", (16, 16), (120, 100, 60))
    albedo = _slab(64)
    restrained, residual = restrain_albedo(albedo, source.size, "restrained", policy)
    before = scale_metrics(source, albedo, policy)
    after = scale_metrics(source, restrained, policy)
    assert after["micro_rms"] < before["micro_rms"] * 0.55
    assert meso_agreement(restrained, albedo, source.size, policy) > 0.9
    assert residual.std() > 1.0
    left = np.asarray(restrained)[:, :16].astype(np.float32).mean()
    right = np.asarray(restrained)[:, -16:].astype(np.float32).mean()
    assert right > left + 40
    same, zero = restrain_albedo(albedo, source.size, "expressive", policy)
    assert np.array_equal(np.asarray(same), np.asarray(albedo))
    assert float(zero.max()) == 0.0
    # The micro band is the part finer than one source texel, not the slab.
    parts = split_bands(_lab(albedo)[:, :, 0], texel_px(source.size, albedo.size), policy)
    assert band_rms(parts["micro"]) > band_rms(parts["macro"]) * 0.2


def test_prefer_regen_requires_a_quieter_image_that_keeps_the_tile():
    policy = _policy()
    source = Image.new("RGB", (16, 16), (120, 100, 60))
    noisy = _slab(64)
    split, _residual = restrain_albedo(noisy, source.size, "restrained", policy)
    quieter, _again = restrain_albedo(split, source.size, "restrained", policy)
    use, info = prefer_regen(source, quieter, noisy, split, quieter, policy)
    assert info["meso_agreement"] > 0.85
    assert use is True
    reject, reason = prefer_regen(source, quieter, noisy, split, noisy, policy)
    assert reject is False
    assert reason["reason"] == "regen is not cleaner than the split"


def test_refine_restrains_a_floor_and_links_dirt(cfg, tmp_path):
    registry = Registry(tmp_path / "work" / "registry.sqlite")
    floor = {
        "key": "1010009:30643", "rdb_type": 1010009, "rdb_id": 30643,
        "name": "", "family": "", "material_class": "interior", "disposition": "keep",
        "seam": 99.0, "risk": "auto",
    }
    dirt = {
        "key": "1010004:10946", "rdb_type": 1010004, "rdb_id": 10946,
        "name": "std-dirt32x32.png", "family": "std_dirt", "material_class": "terrain",
        "disposition": "keep", "seam": 99.0, "risk": "auto",
    }
    registry.upsert_asset(floor)
    registry.upsert_asset(dirt)
    registry.create_run("v3", "first-1076", "semantic", {"config_revision": "v0.3.0"})
    registry.update_run("v3", status="packed", detail={"kind": "reconstruct", "from_run": "v1"})
    runs = tmp_path / "work" / "runs"
    floor_dir = runs / "v3" / "assets" / "1010009_30643"
    dirt_dir = runs / "v3" / "assets" / "1010004_10946"
    floor_dir.mkdir(parents=True)
    dirt_dir.mkdir(parents=True)
    source = Image.new("RGB", (16, 16), (120, 100, 60))
    albedo = _slab(64)
    source.save(floor_dir / "source.png")
    albedo.save(floor_dir / "albedo_final.png")
    source.save(dirt_dir / "source.png")
    albedo.save(dirt_dir / "albedo_final.png")
    Image.new("RGB", (64, 64), (128, 128, 255)).save(dirt_dir / "normal.png")
    Image.new("L", (64, 64), 128).save(dirt_dir / "height.png")
    Image.new("RGB", (64, 64), (210, 180, 0)).save(dirt_dir / "orm.png")
    for key, detail in (
        ("1010009:30643", {"subject_class": "", "kind": "manufactured", "effective_material": "interior"}),
        ("1010004:10946", {"subject_class": "dirt", "kind": "organic", "effective_material": "terrain"}),
    ):
        registry.upsert_job({
            "run_id": "v3", "asset_key": key, "status": "packed", "stage": "pack",
            "seed": 1, "prompt": "", "cache_key": key, "target_w": 64, "target_h": 64,
            "attempt": 0, "rejection_reason": None, "canonical_key": None, "detail": detail,
        })
    calls = {"n": 0}

    def pbr(image, settings):
        calls["n"] += 1
        return cpu_bump_depth(image, settings)

    first = run_refine(
        cfg, registry, from_run="v3", run_id="v4", stop_after="pack",
        skip_regen=True, pbr_fn=pbr, measure_vram=False,
    )
    assert first["stage"] == "pack"
    assert calls["n"] == 1
    refined = Image.open(runs / "v4" / "assets" / "1010009_30643" / "albedo_final.png")
    original = Image.open(floor_dir / "albedo_final.png")
    assert scale_metrics(source, refined, _policy())["micro_rms"] < scale_metrics(source, original, _policy())["micro_rms"] * 0.55
    assert (runs / "v4" / "assets" / "1010009_30643" / "normal_base.png").is_file()
    linked = runs / "v4" / "assets" / "1010004_10946" / "albedo_final.png"
    assert pixel_sha256(Image.open(linked)) == pixel_sha256(Image.open(dirt_dir / "albedo_final.png"))
    assert linked.stat().st_ino == (dirt_dir / "albedo_final.png").stat().st_ino
    pack = tmp_path / "work" / "dist" / "prk-hd-v04"
    assert (pack / map_filename(1010009, 30643, "d")).is_file()
    assert (pack / map_filename(1010004, 10946, "n")).is_file()
    assert (runs / "v4" / "gallery" / "scale-report.json").is_file()
    run_refine(
        cfg, registry, from_run="v3", run_id="v4", stop_after="pack",
        resume=True, skip_regen=True, pbr_fn=pbr, measure_vram=False,
    )
    assert calls["n"] == 1

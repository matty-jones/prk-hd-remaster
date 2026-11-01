import numpy as np
from PIL import Image

from prk_remaster.config import load_config
from prk_remaster.pbr import build_orm, height_from_depth, policy_for, renormalize_normal
from prk_remaster.planner import derive_seed, plan_jobs, rejection_reason, target_size
from prk_remaster.quality import check_albedo, check_normal, colour_shift
from prk_remaster.registry import Registry
from prk_remaster.tiles import repair_seam, seam_is_worse, seam_ratio
from tests.conftest import auto_row, csv_text
from prk_remaster.catalogue import import_csv


def test_seam_repair_and_gate():
    array = np.full((64, 64, 3), 40, np.uint8)
    array[:, 0] = 255
    broken = Image.fromarray(array)
    before = seam_ratio(broken)
    repaired = repair_seam(broken, band=8)
    assert seam_ratio(repaired) < before
    assert seam_is_worse(8.0, 0.4)
    assert not seam_is_worse(1.2, 0.4)
    reasons = check_albedo(broken, target_w=64, target_h=64, source_seam=0.4, tileable=True, cfg={})
    assert any(reason.startswith("periodic seam") for reason in reasons)
    shifted = Image.new("RGB", (64, 64), (10, 10, 10))
    assert colour_shift(Image.new("RGB", (64, 64), (200, 180, 20)), shifted) > 0.2
    assert check_albedo(shifted, target_w=64, target_h=64, source_seam=0.2, tileable=True, cfg={}) == []


def test_black_and_inverted_normal_fail():
    black = Image.new("RGB", (32, 32), (0, 0, 0))
    assert any("black" in reason for reason in check_albedo(
        black, target_w=32, target_h=32, source_seam=0.2, tileable=True, cfg={},
    ))
    flat = renormalize_normal(Image.new("RGB", (16, 16), (255, 0, 0)))
    array = np.asarray(flat).astype(np.float32)
    vectors = array / 255.0 * 2.0 - 1.0
    assert np.allclose(np.linalg.norm(vectors, axis=-1), 1.0, atol=0.03)
    outward = Image.new("RGB", (16, 16), (128, 128, 255))
    assert check_normal(outward, outward) == []
    inward = np.asarray(outward).copy()
    inward[:, :, 2] = 0
    reasons = check_normal(Image.fromarray(inward), outward)
    assert any("inward" in reason for reason in reasons)


def test_orm_channel_order_is_not_brightness():
    rules = load_config()["material_rules"]
    bright = Image.new("RGB", (32, 32), (250, 250, 250))
    height = height_from_depth(Image.new("L", (32, 32), 200), relief=0.8, invert=False)
    terrain = build_orm(bright, height, policy_for("sand", "terrain", rules), rules)
    channels = np.asarray(terrain)
    assert channels[:, :, 2].max() <= 1
    assert channels[:, :, 1].mean() > 180
    assert channels[:, :, 0].mean() > 200
    painted = build_orm(
        Image.new("RGB", (32, 32), (180, 180, 180)),
        height,
        policy_for("wallpanel", "metal", rules),
        rules,
    )
    assert np.asarray(painted)[:, :, 2].mean() > 100
    red = Image.new("RGB", (32, 32), (200, 30, 30))
    painted_red = build_orm(red, height, policy_for("wallpanel", "metal", rules), rules)
    assert np.asarray(painted_red)[:, :, 2].mean() < 20
    bare = build_orm(red, height, policy_for("chrome_hull", "metal", rules), rules)
    assert np.asarray(bare)[:, :, 2].mean() > 200
    gradient = np.tile(np.linspace(0, 255, 32, dtype=np.uint8), (32, 1))
    field = Image.fromarray(gradient)
    plain = np.asarray(height_from_depth(field, 1, False))
    flipped = np.asarray(height_from_depth(field, 1, True))
    assert plain[:, -1].mean() > plain[:, 0].mean()
    assert flipped[:, 0].mean() > flipped[:, -1].mean()


def test_planner_rejects_shadow_and_is_stable(tmp_path):
    rows = [
        auto_row(id=1, name="sand"),
        auto_row(id=9, name="shadow_sand", category="terrain", risk="auto"),
    ]
    csv_path = tmp_path / "assets.csv"
    csv_path.write_text(csv_text(rows), encoding="utf-8")
    registry = Registry(tmp_path / "registry.sqlite")
    import_csv(csv_path, registry, "v1")
    registry.update_asset_hashes("1010006:1", "abc", "pix", 32, 32, "JPEG")
    cfg = load_config()
    cfg["material_rules"]["resolution"]["terrain"] = 64
    registry.create_run("run-a", "first-1076", "aggressive", cfg)
    jobs = plan_jobs(registry, cfg, "run-a")
    by_key = {job["asset_key"]: job for job in jobs}
    assert by_key["1010006:9"]["status"] == "rejected"
    assert "shadow" in by_key["1010006:9"]["rejection_reason"]
    assert by_key["1010006:1"]["status"] == "pending"
    assert by_key["1010006:1"]["target_w"] == 64
    seed = derive_seed(
        cfg["model_lock"]["qwen"]["files"]["diffusion"]["filename"],
        cfg["config_revision"], 1010006, 1, "abc",
    )
    assert by_key["1010006:1"]["seed"] == seed
    again = plan_jobs(registry, cfg, "run-a")
    assert {job["asset_key"]: job["seed"] for job in again} == {job["asset_key"]: job["seed"] for job in jobs}
    registry.update_job("run-a", "1010006:1", status="generated", detail={"generation_label": "qwen-native"})
    kept = plan_jobs(registry, cfg, "run-a")
    kept_job = next(job for job in kept if job["asset_key"] == "1010006:1")
    assert kept_job["status"] == "generated"
    assert kept_job["detail"]["generation_label"] == "qwen-native"
    asset = registry.asset("1010006:9")
    assert rejection_reason(asset, cfg["exclusions"])
    assert target_size(64, 32, "terrain", cfg["material_rules"]) == (64, 32)

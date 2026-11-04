import numpy as np
from PIL import Image

from prk_remaster.detail import lock_coarse_colour, structure_reference
from prk_remaster.pbr import blend_detail_normal, stamp_detail_height
from prk_remaster.quality import check_normal, highpass_energy, native_fidelity
from prk_remaster.tiles import wrap_blur


def _corr(left, right) -> float:
    a = np.asarray(left, dtype=np.float32).ravel()
    b = np.asarray(right, dtype=np.float32).ravel()
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-6:
        return 0.0
    return float(np.dot(a, b) / denom)


def test_colour_lock_moves_the_mean_and_keeps_the_grain():
    height = width = 96
    source = np.full((height, width, 3), (150, 90, 40), np.uint8)
    generated = np.full((height, width, 3), (40, 80, 170), np.float32)
    yy, xx = np.mgrid[0:height, 0:width]
    checker = ((xx + yy) % 2).astype(np.float32)
    generated += (checker * 2.0 - 1.0)[..., None] * 36.0
    generated_image = Image.fromarray(np.clip(generated, 0, 255).astype(np.uint8))
    source_image = Image.fromarray(source)
    locked = lock_coarse_colour(generated_image, source_image, scale=4.0)
    locked_mean = np.asarray(locked, dtype=np.float32).mean(axis=(0, 1))
    generated_mean = np.asarray(generated_image, dtype=np.float32).mean(axis=(0, 1))
    source_mean = source.astype(np.float32).mean(axis=(0, 1))
    assert np.linalg.norm(locked_mean - source_mean) < np.linalg.norm(generated_mean - source_mean) * 0.45
    luma = np.asarray(locked.convert("L"), dtype=np.float32)
    high = luma - wrap_blur(luma, 1.2)
    assert _corr(high, checker * 2.0 - 1.0) > 0.5


def test_structure_reference_keeps_a_lane_and_drops_pixel_blocks():
    small = np.full((8, 8, 3), 36, np.uint8)
    small[:, 3:5] = (210, 170, 40)
    rng = np.random.default_rng(0)
    small = np.clip(small.astype(np.int16) + rng.integers(-35, 35, size=small.shape), 0, 255).astype(np.uint8)
    source = Image.fromarray(small)
    reference = np.asarray(structure_reference(source, (64, 64), kind="manufactured", subject_class="cable"))
    nearest = np.asarray(source.resize((64, 64), Image.Resampling.NEAREST), dtype=np.float32)
    lane = reference[:, 24:40].astype(np.float32).mean()
    left = reference[:, 4:16].astype(np.float32).mean()
    assert lane > left + 40

    def energy(rgb: np.ndarray) -> float:
        luma = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        high = luma - wrap_blur(luma, 1.1)
        return float(np.mean(high ** 2))

    assert energy(reference.astype(np.float32)) < energy(nearest) * 0.55


def test_native_detail_sees_specks_that_a_256_downsample_erases():
    source = Image.new("RGB", (32, 32), (80, 70, 50))
    big = np.full((512, 512, 3), (80, 70, 50), np.uint8)
    big[::2, ::2] = (190, 160, 110)
    image = Image.fromarray(big)
    report = native_fidelity(source, image)
    assert report["detail_gain"] > 2.0
    assert report["fine_energy"] > highpass_energy(image) * 4.0
    assert report["structure"] > 0.9


def test_restrained_reference_softens_block_edges_more_than_a_wall():
    small = np.full((8, 8, 3), 40, np.uint8)
    small[::2, ::2] = 200
    source = Image.fromarray(small)
    tight = np.asarray(structure_reference(source, (64, 64), kind="manufactured", subject_class="wall"))
    wide = np.asarray(structure_reference(
        source, (64, 64), kind="manufactured", subject_class="wall", scale_mode="restrained",
    ))

    def energy(rgb: np.ndarray) -> float:
        luma = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        high = luma - wrap_blur(luma, 1.1)
        return float(np.mean(high ** 2))

    assert energy(wide) < energy(tight) * 0.75


def test_detail_normal_adds_grain_and_stays_a_unit_normal():
    albedo_array = np.full((48, 48, 3), 90, np.uint8)
    albedo_array[::2, ::2] = 200
    albedo = Image.fromarray(albedo_array)
    flat = Image.new("RGB", (48, 48), (128, 128, 255))
    untouched = blend_detail_normal(flat, albedo, 0.0)
    assert np.max(np.abs(np.asarray(untouched).astype(np.int16) - np.asarray(flat))) <= 1
    detailed = blend_detail_normal(flat, albedo, 0.85)
    assert check_normal(detailed, albedo) == []
    assert np.asarray(detailed)[:, :, 0].std() > 2.0
    height = stamp_detail_height(Image.new("L", (48, 48), 128), albedo, 0.85)
    assert height.size == albedo.size
    assert np.asarray(height).std() > 2.0

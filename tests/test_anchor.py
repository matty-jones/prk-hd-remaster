import cv2
import numpy as np
from PIL import Image

from prk_remaster.anchor import anchor_image, compress_highlights, source_trust
from prk_remaster.tiles import wrap_blur
from prk_remaster.quality import choose_donor, decide_route, fidelity_report, highpass_correlation
from prk_remaster.tiles import seam_ratio


def _runs(row: np.ndarray, thresh: float) -> int:
    hot = row > thresh
    count = 0
    previous = False
    for value in hot:
        if value and not previous:
            count += 1
        previous = bool(value)
    return count


def _horizontal_bands(image: Image.Image) -> int:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    chroma = (rgb.max(axis=2) - rgb.min(axis=2)).mean(axis=1)
    thresh = float(np.median(chroma) + 0.45 * (chroma.max() - np.median(chroma)))
    return _runs(chroma, thresh)


def test_anchor_restores_stripe_hue_and_keeps_grain():
    height, width = 160, 160
    source = np.full((height, width, 3), (28, 26, 32), np.uint8)
    source[48:56, :] = (140, 40, 170)
    source[104:112, :] = (40, 170, 190)
    rng = np.random.default_rng(1)
    qwen = np.full((height, width, 3), (176, 182, 188), np.float32)
    for y in range(8, height, 16):
        qwen[y:y + 4, :] = (70, 130, 220)
    checker = ((np.indices((height, width)).sum(axis=0) % 2) * 2 - 1).astype(np.float32)
    qwen += rng.normal(0, 14, size=qwen.shape) * checker[..., None]
    qwen_image = Image.fromarray(np.clip(qwen, 0, 255).astype(np.uint8))
    source_image = Image.fromarray(source)
    result, _diag = anchor_image(source_image, qwen_image, kind="graphic", preset="aggressive")
    anchored = fidelity_report(source_image, result, qwen_image)
    original = fidelity_report(source_image, qwen_image, qwen_image)
    assert anchored["delta_e"] < original["delta_e"] * 0.55
    purple = np.asarray(result)[48:56].astype(np.float32).mean(axis=(0, 1))
    qwen_purple = np.asarray(qwen_image)[48:56].astype(np.float32).mean(axis=(0, 1))
    source_purple = source[48:56].astype(np.float32).mean(axis=(0, 1))
    assert np.linalg.norm(purple - source_purple) < np.linalg.norm(qwen_purple - source_purple)
    assert _horizontal_bands(result) <= 3
    assert _horizontal_bands(result) < _horizontal_bands(qwen_image)
    assert anchored["detail_energy"] > anchored["source_hf"] * 1.1


def test_sharp_source_keeps_its_detail_and_blur_accepts_grain():
    rng = np.random.default_rng(2)
    height = width = 128
    yy, xx = np.mgrid[0:height, 0:width]
    base = np.full((height, width, 3), 90, np.float32)
    sharp = np.clip(base + ((xx + 2 * yy) % 5 == 0)[..., None] * 50, 0, 255).astype(np.uint8)
    qwen = np.clip(base + (yy % 4 == 0)[..., None] * 55, 0, 255).astype(np.uint8)
    sharp_image = Image.fromarray(sharp)
    qwen_image = Image.fromarray(qwen)
    result, _diag = anchor_image(sharp_image, qwen_image, kind="organic")
    assert highpass_correlation(result, sharp_image) > highpass_correlation(result, qwen_image)

    blurred = Image.fromarray(np.clip(cv2.GaussianBlur(sharp, (0, 0), 8.0), 0, 255).astype(np.uint8))
    checker = ((xx + yy) % 2).astype(np.float32)
    grain = np.clip(
        np.asarray(blurred, dtype=np.float32) + rng.normal(0, 16, size=(height, width, 3)) * checker[..., None],
        0, 255,
    ).astype(np.uint8)
    grain_image = Image.fromarray(grain)
    blurred_result, _diag = anchor_image(blurred, grain_image, kind="organic")
    assert highpass_correlation(blurred_result, grain_image) > highpass_correlation(blurred_result, blurred)


def test_wrap_blur_does_not_open_a_periodic_seam():
    height, width = 128, 128
    yy = np.arange(height)[:, None]
    xx = np.arange(width)[None, :]
    luma = 128 + 40 * np.sin(2 * np.pi * xx / width * 4) + 30 * np.sin(2 * np.pi * yy / height * 3)
    rgb = np.clip(np.dstack([luma, luma * 0.9, luma * 0.75]), 0, 255).astype(np.uint8)
    image = Image.fromarray(rgb)
    blurred = Image.fromarray(np.clip(wrap_blur(rgb.astype(np.float32), 2.0), 0, 255).astype(np.uint8))
    assert seam_ratio(blurred) <= seam_ratio(image) + 0.15
    qwen = Image.fromarray(np.clip(rgb.astype(np.float32) * np.array([1.05, 0.85, 0.7]), 0, 255).astype(np.uint8))
    anchored, _diag = anchor_image(image, qwen, kind="organic")
    assert seam_ratio(anchored) < 2.2


def test_highlight_spikes_are_pulled_down():
    field = np.full((64, 64, 3), 70, np.uint8)
    field[30:34, 30:34] = 255
    out = np.asarray(compress_highlights(Image.fromarray(field)))
    assert out[32, 32].mean() < 200
    assert out[0, 0].mean() > 50


def test_blocky_source_is_trusted_less_than_clean_detail():
    blocks = np.zeros((32, 32), np.uint8)
    for y in range(0, 32, 8):
        for x in range(0, 32, 8):
            blocks[y:y + 8, x:x + 8] = 40 + x + y
    block_rgb = np.dstack([blocks, blocks, blocks])
    lines = np.full((128, 128, 3), 80, np.uint8)
    lines[:, ::3] = 210
    assert source_trust(Image.fromarray(block_rgb)) < source_trust(Image.fromarray(lines))


def test_choose_donor_keeps_the_better_anchored_v1():
    assert choose_donor(
        {"structure": 0.82, "delta_e": 8.0, "detail_gain": 1.2},
        {"structure": 0.40, "delta_e": 22.0, "detail_gain": 3.0},
    ) == "v1"


def test_decide_route_regenerates_a_material_conflict():
    calm = {
        "structure": 0.9, "chroma_de": 4, "source_structure": 0.3,
        "hf_agreement": 0.8, "qwen_hf": 1, "source_hf": 1,
    }
    assert decide_route(calm, {"material_conflict": False, "subject_class": "panel"}) == "anchor"
    clash = dict(calm)
    clash["chroma_de"] = 20
    assert decide_route(clash, {"material_conflict": True, "subject_class": "grass"}) == "regen"

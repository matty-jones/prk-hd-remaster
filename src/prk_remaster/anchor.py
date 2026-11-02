"""Rebuild an albedo from the source layout and the Qwen detail donor.

Coarse colour and large features stay with the source. Qwen contributes grain,
and edges only where the source already has an edge in the same place. Filters
are circular so a tiling texture does not grow a seam.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from .quality import (
    RETRY_DELTA_E_ABOVE,
    RETRY_STRUCTURE_BELOW,
    REVIEW_STRUCTURE_BELOW,
    fidelity_report,
    highpass_correlation,
)
from .tiles import wrap_blur


def _luma(rgb: np.ndarray) -> np.ndarray:
    return rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


def _grad_mag(plane: np.ndarray) -> np.ndarray:
    gx = np.roll(plane, -1, axis=1) - np.roll(plane, 1, axis=1)
    gy = np.roll(plane, -1, axis=0) - np.roll(plane, 1, axis=0)
    return np.hypot(gx, gy)


def source_trust(image: Image.Image) -> float:
    """How much of the source's own high frequency is real detail rather than mush or JPEG blocks."""
    luma = np.asarray(image.convert("L"), dtype=np.float32)
    sharp = _sharpness(luma)
    block = _blockiness(luma)
    if min(image.size) <= 48:
        block = max(block, 0.7)
    return float(np.clip(sharp * (1.0 - 0.85 * block), 0.0, 1.0))


def _sharpness(luma: np.ndarray) -> float:
    lap = cv2.Laplacian(luma, cv2.CV_32F, borderType=cv2.BORDER_REFLECT)
    blurred = cv2.GaussianBlur(luma, (0, 0), sigmaX=1.2, sigmaY=1.2)
    lap_b = cv2.Laplacian(blurred, cv2.CV_32F, borderType=cv2.BORDER_REFLECT)
    ratio = float(lap.var() / (lap_b.var() + 1e-3))
    return float(np.clip((ratio - 1.15) / 3.0, 0.0, 1.0))


def _blockiness(luma: np.ndarray) -> float:
    height, width = luma.shape
    if min(height, width) < 16:
        return 1.0
    boundary = []
    interior = []
    for x in range(1, width):
        step = float(np.mean(np.abs(luma[:, x] - luma[:, x - 1])))
        if x % 8 == 0:
            boundary.append(step)
        else:
            interior.append(step)
    if not boundary or not interior:
        return 0.0
    score = float(np.mean(boundary) / (np.mean(interior) + 1e-3))
    return float(np.clip((score - 1.08) / 0.7, 0.0, 1.0))


def compress_highlights(image: Image.Image) -> Image.Image:
    """Pull baked specular spikes toward the local colour so they are not copied into albedo."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lightness = lab[:, :, 0]
    sigma = max(2.0, min(image.size) * 0.04)
    local = wrap_blur(lightness, sigma)
    excess = lightness - local
    scale = float(np.percentile(np.abs(excess), 92)) + 6.0
    limit = max(18.0, scale * 2.0)
    hot = excess > limit
    lab[:, :, 0] = np.where(hot, local + limit * 0.35, lightness)
    out = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    return Image.fromarray(out, "RGB")


def _grain_gain(preset: str) -> float:
    if preset == "faithful":
        return 0.55
    return 1.0


def _structure_value(source: np.ndarray, qwen: np.ndarray) -> float:
    sigma = max(1.0, min(source.shape[:2]) / 12.0)
    left = wrap_blur(_luma(source), sigma)
    right = wrap_blur(_luma(qwen), sigma)
    gx = np.roll(left, -1, axis=1) - np.roll(left, 1, axis=1)
    gy = np.roll(left, -1, axis=0) - np.roll(left, 1, axis=0)
    hx = np.roll(right, -1, axis=1) - np.roll(right, 1, axis=1)
    hy = np.roll(right, -1, axis=0) - np.roll(right, 1, axis=0)
    a = np.stack([gx, gy], axis=-1).astype(np.float32).ravel()
    b = np.stack([hx, hy], axis=-1).astype(np.float32).ravel()
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-6:
        return 1.0
    return float(np.dot(a, b) / denom)


def _local_agreement(source_high: np.ndarray, qwen_high: np.ndarray, sigma: float) -> np.ndarray:
    sl = _luma(source_high)
    ql = _luma(qwen_high)
    sx = np.roll(sl, -1, axis=1) - np.roll(sl, 1, axis=1)
    sy = np.roll(sl, -1, axis=0) - np.roll(sl, 1, axis=0)
    qx = np.roll(ql, -1, axis=1) - np.roll(ql, 1, axis=1)
    qy = np.roll(ql, -1, axis=0) - np.roll(ql, 1, axis=0)
    dot = sx * qx + sy * qy
    denom = np.hypot(sx, sy) * np.hypot(qx, qy) + 1e-3
    agree = np.clip(dot / denom, 0.0, 1.0).astype(np.float32)
    return wrap_blur(agree, max(1.0, sigma * 0.5))


def _coherence(plane: np.ndarray) -> np.ndarray:
    """High where edges keep a direction. Grain and noise score low."""
    gx = np.roll(plane, -1, axis=1) - np.roll(plane, 1, axis=1)
    gy = np.roll(plane, -1, axis=0) - np.roll(plane, 1, axis=0)
    nx = np.roll(gx, 1, axis=1)
    ny = np.roll(gy, 1, axis=1)
    dot = gx * nx + gy * ny
    denom = np.hypot(gx, gy) * np.hypot(nx, ny) + 1e-3
    return wrap_blur(np.clip(dot / denom, 0.0, 1.0).astype(np.float32), 1.2)


def _edge_keep_mask(source: np.ndarray, qwen: np.ndarray, sigma: float, gain: float) -> np.ndarray:
    """Zero Qwen where it introduces a strong edge the source does not have."""
    sl = _luma(wrap_blur(source, sigma))
    ql = _luma(wrap_blur(qwen, sigma))
    sm = _grad_mag(sl)
    qm = _grad_mag(ql)
    s_cut = float(np.percentile(sm, 72)) + 1e-3
    q_cut = float(np.percentile(qm, 72)) + 1e-3
    new_edge = (qm > q_cut) & (sm < s_cut * 0.55)
    mask = np.full(sm.shape, np.float32(gain), dtype=np.float32)
    mask[new_edge] = 0.0
    return wrap_blur(mask, max(0.8, sigma * 0.4))


def _nudge_chroma(donor: np.ndarray, source: np.ndarray, amount: float) -> np.ndarray:
    """Shift the donor's average chroma toward the source without touching its contrast."""
    def to_lab(rgb: np.ndarray) -> np.ndarray:
        u8 = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
        return cv2.cvtColor(u8, cv2.COLOR_RGB2LAB).astype(np.float32)

    lab_d = to_lab(donor)
    lab_s = to_lab(source)
    for channel in (1, 2):
        shift = float(lab_s[:, :, channel].mean() - lab_d[:, :, channel].mean())
        lab_d[:, :, channel] = np.clip(lab_d[:, :, channel] + amount * shift, 0, 255)
    return cv2.cvtColor(lab_d.astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32)


def _lab_lock(mixed: np.ndarray, source: np.ndarray, sigma: float) -> np.ndarray:
    """Low-frequency Lab from the source. Fine chroma and luminance variation may remain."""
    def to_lab(rgb: np.ndarray) -> np.ndarray:
        u8 = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
        return cv2.cvtColor(u8, cv2.COLOR_RGB2LAB).astype(np.float32)

    lab_s = to_lab(source)
    lab_m = to_lab(mixed)
    a_s = wrap_blur(lab_s[:, :, 1], sigma)
    b_s = wrap_blur(lab_s[:, :, 2], sigma)
    a_hp = lab_m[:, :, 1] - wrap_blur(lab_m[:, :, 1], sigma)
    b_hp = lab_m[:, :, 2] - wrap_blur(lab_m[:, :, 2], sigma)
    light_s = lab_s[:, :, 0]
    light_m = lab_m[:, :, 0]
    low = wrap_blur(light_s, sigma)
    high = light_m - wrap_blur(light_m, sigma)
    source_high = light_s - low
    src_energy = wrap_blur(source_high ** 2, sigma)
    mix_energy = wrap_blur(high ** 2, sigma)
    scale = np.ones_like(high, dtype=np.float32)
    richer = src_energy > mix_energy * 1.4
    scale = np.where(richer, np.sqrt((src_energy + 1e-3) / (mix_energy + 1e-3)), scale)
    scale = np.clip(scale, 0.75, 1.8).astype(np.float32)
    locked = np.empty_like(lab_s)
    locked[:, :, 0] = np.clip(low + high * scale, 0, 255)
    locked[:, :, 1] = np.clip(a_s + 0.3 * a_hp, 0, 255)
    locked[:, :, 2] = np.clip(b_s + 0.3 * b_hp, 0, 255)
    return cv2.cvtColor(locked.astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32)


def anchor_image(
    source: Image.Image,
    qwen: Image.Image,
    *,
    kind: str = "organic",
    preset: str = "aggressive",
    tightness: float = 0.0,
) -> tuple[Image.Image, dict]:
    """Return an albedo whose layout and palette follow ``source`` and whose grain follows ``qwen``."""
    qwen = qwen.convert("RGB")
    source = source.convert("RGB")
    prepared = compress_highlights(source.resize(qwen.size, Image.Resampling.LANCZOS))
    base = np.asarray(prepared, dtype=np.float32)
    donor = np.asarray(qwen, dtype=np.float32)
    trust = source_trust(source)
    gain = _grain_gain(preset) * (1.0 - 0.65 * float(np.clip(tightness, 0.0, 1.0)))
    minor = float(min(donor.shape[:2]))
    scales = {"graphic": 0.007, "manufactured": 0.014, "organic": 0.022}
    feature = max(1.25, scales.get(kind, 0.02) * minor)
    agreement = _structure_value(base, donor)
    if agreement < 0.55:
        feature = max(1.25, feature * 0.5)
    feature *= 1.0 - 0.45 * float(np.clip(tightness, 0.0, 1.0))
    feature = max(1.1, feature)

    source_u8 = Image.fromarray(np.clip(np.rint(base), 0, 255).astype(np.uint8), "RGB")
    hf_corr = highpass_correlation(source_u8, qwen)
    # Same surface: the donor is a sharper version of this picture, not a new design.
    # Grass blades are coherent and not pixel-aligned, so the edge gate would drop them.
    same_surface = kind == "organic" and agreement >= 0.72 and hf_corr >= 0.65
    if same_surface:
        # The edit already depicts this surface. Keep its blades and contrast, and
        # only nudge the average chroma toward the source. Replacing the low-pass
        # field clips and posterises grass.
        locked = _nudge_chroma(donor, base, 0.28)
    else:
        source_low = wrap_blur(base, feature)
        source_high = base - source_low
        qwen_high = donor - wrap_blur(donor, feature)
        mask = _edge_keep_mask(base, donor, feature, gain)
        agree_px = _local_agreement(source_high, qwen_high, feature)
        # Coherent edges pass only when they agree with the source, so a new stripe or a
        # moved emblem does not survive. Incoherent grain passes, more freely on a soft source.
        agreed = np.clip(agree_px, 0.0, 1.0)
        coherent = _coherence(_luma(qwen_high)) > 0.45
        q_weight = np.where(
            coherent,
            mask * agreed * 0.9,
            mask * (0.25 + 0.75 * (1.0 - trust)),
        ).astype(np.float32)
        q_weight = np.clip(q_weight, 0.0, 1.0)
        high = source_high * (1.0 - q_weight[..., None]) + qwen_high * q_weight[..., None]
        mixed = source_low + high
        coarse = max(feature * 3.0, 0.08 * minor)
        mixed = wrap_blur(base, coarse) + (mixed - wrap_blur(mixed, coarse))
        locked = _lab_lock(mixed, base, max(2.0, coarse * 0.7))
    image = Image.fromarray(np.clip(np.rint(locked), 0, 255).astype(np.uint8), "RGB")
    return image, {
        "trust": round(trust, 4),
        "agreement": round(agreement, 4),
        "hf_corr": round(float(hf_corr), 4),
        "same_surface": bool(same_surface),
        "feature_sigma": round(float(feature), 3),
        "grain_gain": round(float(gain), 4),
        "kind": kind,
    }


def anchor_with_retry(
    source: Image.Image,
    qwen: Image.Image,
    *,
    kind: str = "organic",
    preset: str = "aggressive",
    fidelity: dict | None = None,
) -> tuple[Image.Image, dict]:
    """Anchor once, then tighten if structure or colour is still outside the band."""
    fidelity = fidelity or {}
    image, diag = anchor_image(source, qwen, kind=kind, preset=preset, tightness=0.0)
    report = fidelity_report(source, image, qwen)
    review = False
    tightened = False
    structure_below = float(fidelity.get("structure_retry_below", RETRY_STRUCTURE_BELOW))
    delta_above = float(fidelity.get("delta_e_retry_above", RETRY_DELTA_E_ABOVE))
    review_below = float(fidelity.get("review_structure_below", REVIEW_STRUCTURE_BELOW))
    if report["structure"] < structure_below or report["delta_e"] > delta_above:
        tight, diag_tight = anchor_image(source, qwen, kind=kind, preset=preset, tightness=1.0)
        report_tight = fidelity_report(source, tight, qwen)
        soft = diag["trust"] < 0.35
        detail_lost = report_tight["detail_gain"] < 1.05 and report["detail_gain"] > report_tight["detail_gain"] + 0.05
        if soft and detail_lost:
            review = True
        else:
            image, report, diag = tight, report_tight, diag_tight
            tightened = True
    if report["structure"] < review_below or report["delta_e"] > delta_above:
        review = True
    return image, {**diag, **report, "tightened": tightened, "review": review}

"""Hard quality gates. Large colour change is a diagnostic, not a failure."""

from __future__ import annotations

from PIL import Image

from .tiles import seam_is_worse, seam_ratio, shape_correlation, wrap_blur


def _luma_stats(image: Image.Image) -> tuple[float, float]:
    import numpy as np
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    luma = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return float(luma.mean()), float(luma.std())


def check_albedo(image: Image.Image, *, target_w: int, target_h: int, source_seam: float, tileable: bool, cfg: dict) -> list[str]:
    reasons = []
    if image is None or image.size[0] < 1 or image.size[1] < 1:
        return ["albedo is missing or empty"]
    if image.size != (target_w, target_h):
        reasons.append(f"albedo size {image.size[0]}x{image.size[1]} != target {target_w}x{target_h}")
    mean, std = _luma_stats(image)
    if mean < 2.0 and std < 1.0:
        reasons.append("albedo is empty or black")
    if tileable:
        ratio = seam_ratio(image)
        if seam_is_worse(
            ratio,
            source_seam,
            above=float(cfg.get("seam_fail_above", 2.2)),
            worsen_ratio=float(cfg.get("seam_worsen_ratio", 1.75)),
            worsen_delta=float(cfg.get("seam_worsen_delta", 0.8)),
        ):
            reasons.append(f"periodic seam {ratio:.2f} is worse than source {source_seam:.2f}")
    return reasons


def check_normal(image: Image.Image, albedo: Image.Image) -> list[str]:
    import numpy as np
    if image.size != albedo.size:
        return [f"normal size {image.size} does not match albedo {albedo.size}"]
    array = np.asarray(image.convert("RGB"), dtype=np.float32)
    if array.shape[-1] != 3:
        return ["normal map is not RGB"]
    vectors = array / 255.0 * 2.0 - 1.0
    length = np.linalg.norm(vectors, axis=-1)
    if float(np.mean((length < 0.85) | (length > 1.15))) > 0.05:
        return ["normal vectors are not unit length"]
    if float(vectors[:, :, 2].mean()) <= 0:
        return ["normal map faces inward (mean Z is not positive)"]
    return []


def check_orm(image: Image.Image, albedo: Image.Image) -> list[str]:
    import numpy as np
    if image.size != albedo.size:
        return [f"orm size {image.size} does not match albedo {albedo.size}"]
    array = np.asarray(image.convert("RGB"))
    if array.ndim != 3 or array.shape[-1] != 3:
        return ["orm is not an RGB image (R occlusion, G roughness, B metallic)"]
    if array.min() < 0 or array.max() > 255:
        return ["orm channel outside 0..255"]
    return []


def check_height(image: Image.Image, albedo: Image.Image) -> list[str]:
    if image.size != albedo.size:
        return [f"height size {image.size} does not match albedo {albedo.size}"]
    if image.mode not in {"L", "RGB"}:
        return [f"height mode {image.mode} is not greyscale-compatible"]
    return []


def check_alignment(source: Image.Image, albedo: Image.Image, *, tileable: bool) -> list[str]:
    """Non-tileable surfaces fail when the large shapes move. Tileable ones do not."""
    if tileable:
        return []
    if shape_correlation(source, albedo) < 0.35:
        return ["large shapes moved relative to the source; UV registration looks lost"]
    return []


def colour_shift(source: Image.Image, albedo: Image.Image) -> float:
    """Mean absolute luma difference, recorded as a diagnostic only."""
    import numpy as np
    left = np.asarray(source.convert("L").resize((64, 64), Image.Resampling.BOX), dtype=np.float32)
    right = np.asarray(albedo.convert("L").resize((64, 64), Image.Resampling.BOX), dtype=np.float32)
    return float(np.mean(np.abs(left - right)) / 255.0)


# Routing thresholds. Large colour change still does not hard-fail a pack.
ROUTE_STRUCTURE_BELOW = 0.55
ROUTE_CHROMA_ABOVE = 12.0
ROUTE_HF_AGREEMENT_BELOW = 0.30
ROUTE_SOURCE_STRUCTURE = 0.12
RETRY_STRUCTURE_BELOW = 0.72
RETRY_DELTA_E_ABOVE = 18.0
REVIEW_STRUCTURE_BELOW = 0.60
FINE_SUBJECTS = frozenset({"cable", "grass", "vegetation", "chain", "pipe", "wood", "rope", "wire"})


def _rgb_array(image: Image.Image) -> "np.ndarray":
    import numpy as np
    return np.asarray(image.convert("RGB"), dtype=np.float32)


def cie_lab(image: Image.Image):
    """CIE L*a*b* from an sRGB image. L is 0..100."""
    import numpy as np
    rgb = _rgb_array(image) / 255.0
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    matrix = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float32,
    )
    xyz = linear @ matrix.T
    xyz = xyz / np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
    delta = 6.0 / 29.0
    f = np.where(xyz > delta ** 3, np.cbrt(xyz), xyz / (3 * delta ** 2) + 4.0 / 29.0)
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1).astype(np.float32)


def _fit_pair(source: Image.Image, other: Image.Image, longest: int = 96) -> tuple[Image.Image, Image.Image]:
    source = source.convert("RGB")
    other = other.convert("RGB").resize(source.size, Image.Resampling.BOX)
    scale = longest / max(source.size)
    if scale < 1:
        size = (max(8, int(round(source.size[0] * scale))), max(8, int(round(source.size[1] * scale))))
        source = source.resize(size, Image.Resampling.BOX)
        other = other.resize(size, Image.Resampling.BOX)
    return source, other


def _luma_np(image: Image.Image):
    import numpy as np
    rgb = _rgb_array(image)
    return rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


def _grad_components(plane):
    import numpy as np
    gx = np.roll(plane, -1, axis=1) - np.roll(plane, 1, axis=1)
    gy = np.roll(plane, -1, axis=0) - np.roll(plane, 1, axis=0)
    return gx, gy


def _corr(left, right) -> float:
    import numpy as np
    a = np.asarray(left, dtype=np.float32).ravel()
    b = np.asarray(right, dtype=np.float32).ravel()
    a = a - a.mean()
    b = b - b.mean()
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a < 1e-6 and norm_b < 1e-6:
        return 1.0
    if norm_a < 1e-6 or norm_b < 1e-6:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _lowpass_luma(image: Image.Image):
    luma = _luma_np(image)
    sigma = max(1.0, min(luma.shape) / 12.0)
    return wrap_blur(luma, sigma)


def structure_agreement(source: Image.Image, albedo: Image.Image) -> float:
    """Wrap-aware correlation of low-pass gradients. 1 means the large features match."""
    import numpy as np
    left, right = _fit_pair(source, albedo)
    gx, gy = _grad_components(_lowpass_luma(left))
    hx, hy = _grad_components(_lowpass_luma(right))
    return _corr(np.stack([gx, gy], axis=-1), np.stack([hx, hy], axis=-1))


def _structure_energy(image: Image.Image) -> float:
    import numpy as np
    mag = np.hypot(*_grad_components(_lowpass_luma(image)))
    return float(np.clip(mag.mean() / 25.0, 0.0, 1.0))


def _hf_plane(image: Image.Image):
    image = image.convert("RGB")
    longest = max(image.size)
    if longest > 256:
        scale = 256 / longest
        size = (max(8, int(image.size[0] * scale)), max(8, int(image.size[1] * scale)))
        image = image.resize(size, Image.Resampling.BOX)
    luma = _luma_np(image)
    return luma - wrap_blur(luma, 1.25)


def highpass_energy(image: Image.Image) -> float:
    import numpy as np
    plane = _hf_plane(image)
    return float(np.mean(plane ** 2))


def highpass_correlation(left: Image.Image, right: Image.Image) -> float:
    left = left.convert("RGB")
    right = right.convert("RGB")
    if right.size != left.size:
        right = right.resize(left.size, Image.Resampling.BOX)
    return _corr(_hf_plane(left), _hf_plane(right))


def _band_agreement(source: Image.Image, albedo: Image.Image) -> float:
    import numpy as np
    left = cie_lab(source)
    right = cie_lab(albedo)
    sat_l = np.hypot(left[..., 1], left[..., 2])
    sat_r = np.hypot(right[..., 1], right[..., 2])
    return float(max(_corr(sat_l.mean(axis=1), sat_r.mean(axis=1)), _corr(sat_l.mean(axis=0), sat_r.mean(axis=0))))


def fidelity_report(source: Image.Image, albedo: Image.Image, donor: Image.Image | None = None) -> dict:
    """Colour, structure, and detail measures. These route salvage; they do not reject a pack."""
    import numpy as np
    left, right = _fit_pair(source, albedo, 96)
    lab_s = cie_lab(left)
    lab_a = cie_lab(right)
    delta = lab_s - lab_a
    delta_e = float(np.mean(np.linalg.norm(delta, axis=-1)))
    chroma_de = float(np.mean(np.linalg.norm(delta[..., 1:], axis=-1)))
    std_s = float(lab_s[..., 0].std())
    std_a = float(lab_a[..., 0].std())
    lanczos = source.convert("RGB").resize(albedo.size, Image.Resampling.LANCZOS)
    source_hf = highpass_energy(lanczos)
    detail_energy = highpass_energy(albedo)
    qwen_hf = highpass_energy(donor) if donor is not None else source_hf
    hf_agreement = highpass_correlation(albedo, donor) if donor is not None else 1.0
    return {
        "delta_e": round(delta_e, 3),
        "chroma_de": round(chroma_de, 3),
        "luma_std_ratio": round(std_a / max(std_s, 1e-3), 3),
        "structure": round(structure_agreement(source, albedo), 4),
        "source_structure": round(_structure_energy(left), 4),
        "band_agreement": round(_band_agreement(left, right), 4),
        "detail_energy": round(detail_energy, 3),
        "source_hf": round(source_hf, 3),
        "qwen_hf": round(qwen_hf, 3),
        "detail_gain": round(detail_energy / max(source_hf, 1e-3), 3),
        "hf_agreement": round(hf_agreement, 4),
    }


def decide_route(metrics: dict, brief: dict, fidelity: dict | None = None) -> str:
    """Anchor when the donor still depicts the source. Regenerate when the material or the subject is wrong."""
    fidelity = fidelity or {}
    structure_below = float(fidelity.get("structure_regen_below", ROUTE_STRUCTURE_BELOW))
    chroma_above = float(fidelity.get("chroma_conflict_above", ROUTE_CHROMA_ABOVE))
    hf_below = float(fidelity.get("hf_agreement_below", ROUTE_HF_AGREEMENT_BELOW))
    conflict = bool(brief.get("material_conflict")) and float(metrics.get("chroma_de") or 0) > chroma_above
    structure_lost = (
        float(metrics.get("structure") or 1) < structure_below
        and float(metrics.get("source_structure") or 0) > ROUTE_SOURCE_STRUCTURE
    )
    subject = str(brief.get("subject_class") or "")
    hf_bad = (
        subject in FINE_SUBJECTS
        and float(metrics.get("hf_agreement") or 1) < hf_below
        and float(metrics.get("qwen_hf") or 0) > float(metrics.get("source_hf") or 0) * 1.15
    )
    if conflict or structure_lost or hf_bad:
        return "regen"
    return "anchor"


def choose_donor(anchored_v1: dict, anchored_v2: dict) -> str:
    """Keep a regenerated edit only when structure and colour are not worse. Detail breaks a tie."""
    if float(anchored_v2["structure"]) < float(anchored_v1["structure"]) - 0.02:
        return "v1"
    if float(anchored_v2["delta_e"]) > float(anchored_v1["delta_e"]) + 1.0:
        return "v1"
    v1_key = (
        float(anchored_v1["structure"]),
        -float(anchored_v1["delta_e"]),
        float(anchored_v1.get("detail_gain") or 0),
    )
    v2_key = (
        float(anchored_v2["structure"]),
        -float(anchored_v2["delta_e"]),
        float(anchored_v2.get("detail_gain") or 0),
    )
    if v2_key > v1_key:
        return "v2"
    return "v1"

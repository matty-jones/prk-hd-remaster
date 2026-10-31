"""Hard quality gates. Large colour change is a diagnostic, not a failure."""

from __future__ import annotations

from PIL import Image

from .tiles import seam_is_worse, seam_ratio, shape_correlation


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

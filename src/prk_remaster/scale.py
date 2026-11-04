"""Material-scale split for an albedo.

Macrostructure (wider than about four source texels) and mesostructure
(about one to four texels) stay. Microstructure finer than one source texel
is turned down on flat manufactured surfaces. Cable, soil, and grass keep it.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .quality import structure_agreement
from .tiles import wrap_blur

EXPRESSIVE_CLASSES = frozenset({
    "cable", "wire", "chain", "rope", "dirt", "sand", "grass", "vegetation",
})
DRESSED_CLASSES = frozenset({
    "stone", "wall", "tile", "plaster", "floor", "brick", "wood", "panel", "concrete",
})
RESTRAINED_MATERIALS = frozenset({"interior", "architecture", "metal"})

DIAGNOSIS_KEYS = (
    "1010009:30643",
    "1010004:20908",
    "1010004:10946",
    "1010006:2",
    "1010009:8544",
    "1010004:224626",
)

_FALLBACK_GAINS = {
    "restrained": {
        "micro_albedo_gain": 0.25,
        "meso_albedo_gain": 1.0,
        "macro_albedo_gain": 1.0,
        "micro_normal_weight": 0.35,
        "micro_roughness": 0.10,
    },
    "expressive": {
        "micro_albedo_gain": 1.0,
        "meso_albedo_gain": 1.0,
        "macro_albedo_gain": 1.0,
        "micro_normal_weight": 0.0,
        "micro_roughness": 0.0,
    },
}


def scale_mode(subject_class: str, kind: str, effective_material: str, policy: dict | None = None) -> str:
    """Restrained flats lose albedo micro. Expressive materials keep it."""
    policy = policy or {}
    subject = (subject_class or "").strip().lower()
    material = (effective_material or "").strip().lower()
    expressive = set(policy.get("expressive_classes") or EXPRESSIVE_CLASSES)
    dressed = set(policy.get("dressed_classes") or DRESSED_CLASSES)
    restrained = set(policy.get("restrained_materials") or RESTRAINED_MATERIALS)
    if subject in expressive:
        return "expressive"
    if material == "terrain":
        return "expressive"
    if material in restrained:
        return "restrained"
    if material == "stone" or subject in dressed:
        return "restrained"
    if (kind or "").strip().lower() == "manufactured":
        return "restrained"
    return "expressive"


def mode_gains(policy: dict | None, mode: str) -> dict:
    base = dict(_FALLBACK_GAINS["expressive" if mode != "restrained" else "restrained"])
    found = ((policy or {}).get("modes") or {}).get(mode) or {}
    for key, value in found.items():
        base[key] = value
    return base


def texel_px(source_size: tuple[int, int], output_size: tuple[int, int]) -> float:
    return max(output_size) / max(max(source_size), 1)


def _sigmas(texel: float, policy: dict | None) -> tuple[float, float]:
    policy = policy or {}
    micro = max(0.8, float(texel) * float(policy.get("micro_texel", 0.75)))
    macro = max(micro + 0.5, float(texel) * float(policy.get("macro_texel", 4.0)))
    return micro, macro


def split_bands(luma: np.ndarray, texel: float, policy: dict | None = None) -> dict:
    """Lightness bands. Micro is finer than about one source texel."""
    micro_sigma, macro_sigma = _sigmas(texel, policy)
    low = wrap_blur(luma, micro_sigma)
    macro = wrap_blur(luma, macro_sigma)
    return {
        "macro": macro,
        "meso": low - macro,
        "micro": luma - low,
        "micro_sigma": micro_sigma,
        "macro_sigma": macro_sigma,
    }


def band_rms(plane: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(plane, dtype=np.float32) ** 2)))


def _corr(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float32).ravel()
    b = np.asarray(right, dtype=np.float32).ravel()
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-6:
        return 0.0
    return float(np.dot(a, b) / denom)


def _lab(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)


def _from_lab(lab: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(np.clip(np.rint(lab), 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    return Image.fromarray(rgb, "RGB")


def _project(micro: np.ndarray, lattice: np.ndarray) -> tuple[float, float, float]:
    """Correlation, RMS of the lattice-aligned part, RMS of the remainder."""
    correlation = _corr(micro, lattice)
    a = np.asarray(micro, dtype=np.float32).ravel()
    b = np.asarray(lattice, dtype=np.float32).ravel()
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.dot(b, b))
    if denom < 1e-6:
        return correlation, 0.0, band_rms(a)
    coef = float(np.dot(a, b) / denom)
    locked = (coef * b).reshape(micro.shape)
    invented = (a - coef * b).reshape(micro.shape)
    return correlation, band_rms(locked), band_rms(invented)


def scale_metrics(source: Image.Image, albedo: Image.Image, policy: dict | None = None) -> dict:
    """Band energy and how much of the micro band lines up with source blocks."""
    texel = texel_px(source.size, albedo.size)
    parts = split_bands(_lab(albedo)[:, :, 0], texel, policy)
    lattice = split_bands(
        _lab(source.convert("RGB").resize(albedo.size, Image.Resampling.NEAREST))[:, :, 0],
        texel,
        policy,
    )["micro"]
    correlation, locked, invented = _project(parts["micro"], lattice)
    return {
        "texel_px": round(texel, 3),
        "micro_rms": round(band_rms(parts["micro"]), 3),
        "meso_rms": round(band_rms(parts["meso"]), 3),
        "macro_rms": round(band_rms(parts["macro"]), 3),
        "lattice_corr": round(correlation, 4),
        "lattice_locked_rms": round(locked, 3),
        "invented_rms": round(invented, 3),
        "micro_sigma": round(float(parts["micro_sigma"]), 3),
        "macro_sigma": round(float(parts["macro_sigma"]), 3),
    }


def meso_agreement(left: Image.Image, right: Image.Image, source_size: tuple[int, int], policy: dict | None = None) -> float:
    """Correlation of the one-to-four texel band. 1 means the slab pattern matches."""
    if right.size != left.size:
        right = right.convert("RGB").resize(left.size, Image.Resampling.LANCZOS)
    texel = texel_px(source_size, left.size)
    a = split_bands(_lab(left)[:, :, 0], texel, policy)["meso"]
    b = split_bands(_lab(right)[:, :, 0], texel, policy)["meso"]
    return _corr(a, b)


def restrain_albedo(
    image: Image.Image,
    source_size: tuple[int, int],
    mode: str,
    policy: dict | None = None,
) -> tuple[Image.Image, np.ndarray]:
    """Return the albedo and the lightness that was removed from it.

    Expressive mode returns the same pixels and a zero residual.
    """
    gains = mode_gains(policy, mode)
    micro_gain = float(gains["micro_albedo_gain"])
    meso_gain = float(gains["meso_albedo_gain"])
    macro_gain = float(gains["macro_albedo_gain"])
    rgb = image.convert("RGB")
    if micro_gain >= 0.999 and meso_gain >= 0.999 and macro_gain >= 0.999:
        return rgb, np.zeros((rgb.size[1], rgb.size[0]), dtype=np.float32)
    lab = _lab(rgb)
    texel = texel_px(source_size, rgb.size)
    parts = split_bands(lab[:, :, 0], texel, policy)
    rebuilt = parts["macro"] * macro_gain + parts["meso"] * meso_gain + parts["micro"] * micro_gain
    removed = parts["micro"] * (1.0 - micro_gain)
    lab[:, :, 0] = np.clip(rebuilt, 0.0, 255.0)
    return _from_lab(lab), removed.astype(np.float32)


def prefer_regen(
    source: Image.Image,
    v1: Image.Image | None,
    v3: Image.Image,
    split: Image.Image,
    regen: Image.Image,
    policy: dict | None = None,
) -> tuple[bool, dict]:
    """Use a regenerated albedo only when it is quieter than the split and still the same tile."""
    if regen.size != split.size:
        regen = regen.convert("RGB").resize(split.size, Image.Resampling.LANCZOS)
    if v3.size != split.size:
        v3 = v3.convert("RGB").resize(split.size, Image.Resampling.LANCZOS)
    split_m = scale_metrics(source, split, policy)
    regen_m = scale_metrics(source, regen, policy)
    meso = meso_agreement(regen, v3, source.size, policy)
    structure_split = structure_agreement(source, split)
    structure_regen = structure_agreement(source, regen)
    info = {
        "meso_agreement": round(meso, 4),
        "structure_split": round(structure_split, 4),
        "structure_regen": round(structure_regen, 4),
        "split_micro_rms": split_m["micro_rms"],
        "regen_micro_rms": regen_m["micro_rms"],
    }
    if v1 is None:
        info["reason"] = "no v1 reference"
        return False, info
    if v1.size != split.size:
        v1 = v1.convert("RGB").resize(split.size, Image.Resampling.LANCZOS)
    v1_micro = scale_metrics(source, v1, policy)["micro_rms"]
    info["v1_micro_rms"] = v1_micro
    if meso < 0.85:
        info["reason"] = "meso drifted from v3"
        return False, info
    if structure_regen + 0.02 < structure_split:
        info["reason"] = "structure worse than the split"
        return False, info
    info["v1_gap_regen"] = round(abs(regen_m["micro_rms"] - v1_micro), 3)
    info["v1_gap_split"] = round(abs(split_m["micro_rms"] - v1_micro), 3)
    # Closer to v1's band energy is not cleaner. v1 still carries edge energy.
    # A regen replaces the split only when its own micro band is actually quieter.
    if regen_m["micro_rms"] >= split_m["micro_rms"]:
        info["reason"] = "regen is not cleaner than the split"
        return False, info
    info["reason"] = "regen is quieter and keeps the tile"
    return True, info


def _center_box(size: tuple[int, int], span: int) -> tuple[int, int, int, int]:
    width, height = size
    side = min(span, width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return left, top, left + side, top + side


def _fit_width(image: Image.Image, width: int) -> Image.Image:
    scale = width / max(image.size[0], 1)
    height = max(1, int(round(image.size[1] * scale)))
    return image.resize((width, height), Image.Resampling.BOX)


def _lod(crop: Image.Image, divisor: int, span: int) -> Image.Image:
    small_w = max(1, crop.size[0] // divisor)
    small_h = max(1, crop.size[1] // divisor)
    small = crop.resize((small_w, small_h), Image.Resampling.BOX)
    return small.resize((span, span), Image.Resampling.NEAREST)


def _sheet_font(size: int):
    from PIL import ImageFont
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if path.is_file():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def write_distance_sheet(columns: list[tuple[str, Image.Image]], dest: Path, caption: str) -> Path:
    """Native crop, two box mips, and an oblique squash. Columns are one material."""
    from .imaging import atomic_save_png

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not columns:
        image = Image.new("RGB", (320, 48), (16, 16, 16))
        atomic_save_png(image, dest)
        return dest
    span = 320
    prepared = []
    for label, image in columns:
        rgb = image.convert("RGB")
        crop = rgb.crop(_center_box(rgb.size, span))
        if crop.size != (span, span):
            crop = crop.resize((span, span), Image.Resampling.NEAREST)
        oblique = crop.resize((span, max(1, span // 4)), Image.Resampling.BOX)
        prepared.append((label, _fit_width(rgb, span), crop, _lod(crop, 8, span), _lod(crop, 16, span), oblique))
    gap = 8
    header = 36
    row_gap = 6
    width = gap + len(prepared) * (span + gap)
    row_heights = [
        max(item[1].size[1] for item in prepared),
        span,
        span,
        span,
        max(item[5].size[1] for item in prepared),
    ]
    height = header + sum(row_heights) + row_gap * len(row_heights) + gap
    sheet = Image.new("RGB", (width, height), (12, 12, 12))
    draw = ImageDraw.Draw(sheet)
    font = _sheet_font(14)
    small = _sheet_font(12)
    draw.text((gap, 8), caption[:240], fill=(235, 235, 235), font=font)
    left = gap
    for label, fit, native, lod8, lod16, oblique in prepared:
        draw.text((left, header - 16), label, fill=(190, 190, 190), font=small)
        top = header
        for panel in (fit, native, lod8, lod16, oblique):
            sheet.paste(panel, (left, top))
            top += panel.size[1] + row_gap
        left += span + gap
    atomic_save_png(sheet, dest)
    return dest

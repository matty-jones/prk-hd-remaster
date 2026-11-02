"""Turn a filename and the source image into a structured material brief.

The brief is what the prompt compiler and the anchor are allowed to believe.
A generic or contradictory name does not invent a subject.
"""

from __future__ import annotations

import math
import re

import numpy as np
from PIL import Image

from .quality import cie_lab

_TOKEN = re.compile(r"[a-z0-9]+")
_SIZE = re.compile(r"^\d+x\d+$")
_TRAILING_SIZE = re.compile(r"\d+x\d+$")
BANNED_CAPTION = (
    "photo", "photograph", "scene", "landscape", "camera", "person", "character",
    "portrait", "room interior", "factory floor", "building exterior",
)
_KIND_BY_CATEGORY = {
    "terrain": "organic",
    "stone": "organic",
    "metal": "manufactured",
    "architecture": "manufactured",
    "interior": "manufactured",
}


def tokenize(name: str) -> list[str]:
    text = (name or "").lower().replace(".png", " ").replace(".jpg", " ").replace(".jpeg", " ")
    tokens = []
    for token in _TOKEN.findall(text):
        if _SIZE.match(token) or token.isdigit():
            continue
        peeled = _TRAILING_SIZE.sub("", token)
        if not peeled or peeled.isdigit():
            continue
        tokens.append(peeled)
    return tokens


def _hue_name(a: float, b: float, saturation: float) -> str:
    if saturation < 8:
        return "neutral grey"
    angle = math.degrees(math.atan2(b, a))
    # Lab: +a red, +b yellow, -a green, -b blue.
    names = (
        (-180, "green"),
        (-135, "cyan"),
        (-60, "blue"),
        (-15, "purple"),
        (20, "red"),
        (70, "orange"),
        (110, "yellow"),
        (160, "green"),
        (180, "green"),
    )
    for limit, name in names:
        if angle <= limit:
            return name
    return "green"


def _tone(lightness: float) -> str:
    if lightness < 28:
        return "very dark"
    if lightness < 45:
        return "dark"
    if lightness < 68:
        return "mid-tone"
    return "light"


def _runs(values: np.ndarray, thresh: float) -> list[tuple[int, int]]:
    runs = []
    start = None
    for index, hot in enumerate(values > thresh):
        if hot and start is None:
            start = index
        elif not hot and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(values)))
    return runs


def _axis_bands(saturation: np.ndarray) -> tuple[int, str, list[tuple[int, int]]]:
    row = saturation.mean(axis=1)
    col = saturation.mean(axis=0)

    def count(axis: np.ndarray) -> tuple[int, list[tuple[int, int]]]:
        peak = float(axis.max())
        base = float(np.median(axis))
        if peak - base < 6:
            return 0, []
        thresh = base + 0.5 * (peak - base)
        found = [item for item in _runs(axis, thresh) if item[1] - item[0] >= 1]
        return len(found), found

    rows, row_runs = count(row)
    cols, col_runs = count(col)
    if rows >= cols and rows:
        return rows, "horizontal", row_runs
    if cols:
        return cols, "vertical", col_runs
    return 0, "", []


def visual_reading(image: Image.Image) -> dict:
    """Palette and structure measured from the source, without a learned model."""
    rgb_image = image.convert("RGB")
    sample = rgb_image if max(rgb_image.size) <= 96 else rgb_image.resize(
        (max(8, rgb_image.size[0] * 64 // max(rgb_image.size)), max(8, rgb_image.size[1] * 64 // max(rgb_image.size))),
        Image.Resampling.BOX,
    )
    rgb = np.asarray(sample, dtype=np.float32)
    lab = cie_lab(sample)
    mean_l = float(lab[..., 0].mean())
    mean_a = float(lab[..., 1].mean())
    mean_b = float(lab[..., 2].mean())
    saturation = np.hypot(lab[..., 1], lab[..., 2])
    mean_sat = float(saturation.mean())
    mean_r = float(rgb[..., 0].mean())
    mean_g = float(rgb[..., 1].mean())
    mean_b_rgb = float(rgb[..., 2].mean())
    vegetation = mean_g > mean_r * 1.12 and mean_g > mean_b_rgb * 1.08 and mean_sat > 8
    bands, axis, runs = _axis_bands(saturation)
    band_hues = []
    for start, end in runs[:4]:
        if axis == "horizontal":
            patch = lab[start:end]
        else:
            patch = lab[:, start:end]
        if patch.size == 0:
            continue
        band_hues.append(_hue_name(float(patch[..., 1].mean()), float(patch[..., 2].mean()), float(np.hypot(patch[..., 1], patch[..., 2]).mean())))
    luma = lab[..., 0]
    blur = luma.mean()
    center = luma[luma.shape[0] // 4: luma.shape[0] * 3 // 4, luma.shape[1] // 4: luma.shape[1] * 3 // 4]
    border = np.concatenate([
        luma[: luma.shape[0] // 8, :].ravel(),
        luma[-max(1, luma.shape[0] // 8):, :].ravel(),
    ])
    center_motif = float(center.std()) > float(border.std()) * 1.35 and float(center.std()) > 8 and abs(float(center.mean()) - blur) > 4
    return {
        "mean_l": round(mean_l, 2),
        "mean_a": round(mean_a, 2),
        "mean_b": round(mean_b, 2),
        "mean_sat": round(mean_sat, 2),
        "tone": _tone(mean_l),
        "hue_name": _hue_name(mean_a, mean_b, mean_sat),
        "vegetation": bool(vegetation),
        "band_count": int(bands),
        "band_axis": axis,
        "band_hues": band_hues,
        "center_motif": bool(center_motif),
    }


def describe_appearance(reading: dict) -> str:
    parts = [f"{reading['tone']} {reading['hue_name']} field"]
    if reading.get("band_count"):
        hues = ", ".join(reading.get("band_hues") or []) or "accent"
        noun = "band" if reading["band_count"] == 1 else "bands"
        parts.append(f"{reading['band_count']} {reading['band_axis']} {noun} ({hues})")
    if reading.get("center_motif"):
        parts.append("a high-contrast motif centred in the tile")
    if reading.get("vegetation"):
        parts.append("green vegetation")
    return ", ".join(parts)


def _color_disagrees(token: str, reading: dict) -> bool:
    if token == "dark":
        return reading["mean_l"] > 62
    if token == "light":
        return reading["mean_l"] < 32
    if token in {"neutral", "desat", "desaturated", "grey", "gray"}:
        return reading["mean_sat"] > 25
    if reading["mean_sat"] < 10:
        return token not in set(reading.get("band_hues") or []) and token != reading["hue_name"]
    return token != reading["hue_name"] and token not in set(reading.get("band_hues") or [])


def build_brief(asset: dict, image: Image.Image, lexicon: dict) -> dict:
    """Structured description used by the anchor and the prompt compiler."""
    reading = visual_reading(image)
    generic = set(lexicon.get("generic") or [])
    noise = set(lexicon.get("noise") or [])
    colors = lexicon.get("colors") or {}
    settings = lexicon.get("settings") or {}
    subjects = lexicon.get("subjects") or {}
    tokens = [token for token in tokenize(asset.get("name") or "") if token not in noise]
    known = [subjects[token] for token in tokens if token in subjects]
    known.sort(key=lambda item: int(item.get("rank") or 0), reverse=True)
    setting = next((settings[token] for token in tokens if token in settings), "")
    color_tokens = [colors[token] for token in tokens if token in colors]
    disagreed = [token for token in color_tokens if _color_disagrees(token, reading)]
    category = asset.get("material_class") or ""
    primary = known[0] if known else {}
    specific = bool(primary) and int(primary.get("rank") or 0) >= 55 and primary.get("class") not in {"metal", "wall", "plate", "stone"}
    only_generic = not tokens or all(token in generic or token in colors or token in settings for token in tokens)
    name_confidence = 0.0
    if specific:
        name_confidence = 0.86
    elif primary and int(primary.get("rank") or 0) >= 30 and not only_generic:
        name_confidence = 0.5
    elif primary:
        name_confidence = 0.28
    subject = ""
    enhance = "fine surface grain of the material already visible"
    hold = "the original palette, contrast, and feature placement"
    kind = _KIND_BY_CATEGORY.get(category, "manufactured")
    subject_class = ""
    material = category
    if name_confidence >= 0.45 and primary:
        subject = str(primary.get("subject") or "")
        if setting and "factory" not in setting:
            subject = f"{subject} on {setting}"
        enhance = str(primary.get("enhance") or enhance)
        hold = str(primary.get("hold") or hold)
        kind = str(primary.get("kind") or kind)
        subject_class = str(primary.get("class") or "")
        material = str(primary.get("material") or category)
    if reading["center_motif"] and name_confidence < 0.45:
        kind = "graphic"
        hold = "the centred motif's shape, contrast, and placement, " + hold
    elif reading["band_count"] and name_confidence < 0.45:
        kind = "graphic"
        hold = "the original accent bands and their positions, " + hold
    vegetation_conflict = bool(reading["vegetation"]) and category in {"stone", "metal", "architecture"}
    if vegetation_conflict and name_confidence < 0.7:
        subject = "short grass"
        subject_class = "grass"
        material = "terrain"
        kind = "organic"
        enhance = "distinct blades, dry tips, and small clumps of the same grass"
        hold = "the original green and the boundary between grass and any bare soil"
        name_confidence = max(name_confidence, 0.55)
    effective = material or category or "unknown"
    if effective not in {"terrain", "stone", "architecture", "interior", "metal"}:
        effective = category or "unknown"
    return {
        "subject": subject,
        "subject_class": subject_class,
        "kind": kind,
        "enhance": enhance,
        "hold": hold,
        "appearance": describe_appearance(reading),
        "name_confidence": round(name_confidence, 3),
        "material_conflict": bool(vegetation_conflict and category != effective),
        "effective_material": effective,
        "color_disagreement": disagreed,
        "setting": setting,
        "reading": reading,
        "caption_used": False,
    }


def needs_caption(brief: dict, route: str = "anchor") -> bool:
    if route == "regen":
        return True
    if brief.get("material_conflict"):
        return True
    if float(brief.get("name_confidence") or 0) < 0.45:
        return True
    if not brief.get("subject"):
        return True
    return False


def sanitize_caption(text: str, limit: int = 180) -> str:
    cleaned = " ".join(str(text or "").replace("\n", " ").split())
    lowered = cleaned.lower()
    if not cleaned:
        return ""
    if lowered.startswith(("do not", "don't", "dont ", "no ")):
        return ""
    if any(word in lowered for word in BANNED_CAPTION):
        return ""
    return cleaned[:limit].rstrip(" ,.;")


_GRASS_WORDS = ("grass", "vegetation", "moss", "lawn")
_NOT_GRASS = ("stone", "wall", "metal", "brick", "concrete", "pipe", "wood", "tile")


def _pixel_grass(brief: dict) -> bool:
    return brief.get("effective_material") == "terrain" and any(
        word in str(brief.get("subject") or "").lower() for word in _GRASS_WORDS
    )


def _drops_grass(text: str) -> bool:
    lowered = text.lower()
    return any(word in lowered for word in _NOT_GRASS)


def apply_caption(brief: dict, caption: dict | None) -> dict:
    """Let a caption fill an unreliable brief. A confident filename keeps its subject."""
    updated = dict(brief)
    if not caption:
        return updated
    unreliable = (
        float(updated.get("name_confidence") or 0) < 0.45
        or bool(updated.get("material_conflict"))
        or not updated.get("subject")
        or caption.get("name_agreement") is False
    )
    if not unreliable:
        updated["caption_used"] = False
        return updated
    subject = sanitize_caption(caption.get("subject") or "")
    pattern = sanitize_caption(caption.get("pattern") or "")
    enhance = sanitize_caption(caption.get("safe_enhancements") or "")
    material = sanitize_caption(caption.get("material") or "").lower()
    # A green field already read as grass is not renamed into stone or a wall.
    if _pixel_grass(updated):
        def _grass_detail(text: str) -> str:
            if not text or _drops_grass(text):
                return ""
            if not any(word in text.lower() for word in _GRASS_WORDS):
                return ""
            return text

        subject = _grass_detail(subject)
        pattern = _grass_detail(pattern)
        enhance = _grass_detail(enhance)
        if material and material not in {"terrain", "grass", "vegetation"}:
            material = ""
    if subject:
        updated["subject"] = subject
        updated["caption_used"] = True
    if enhance:
        updated["enhance"] = enhance
    if pattern:
        updated["hold"] = f"the visible pattern ({pattern}), {updated.get('hold') or 'the original placement'}"
    if "grass" in subject.lower() or "vegetation" in subject.lower() or material in {"grass", "vegetation", "terrain"}:
        updated["effective_material"] = "terrain"
        updated["kind"] = "organic"
        updated["subject_class"] = updated.get("subject_class") or "grass"
        updated["material_conflict"] = True
    elif material in {"metal", "stone", "terrain", "interior", "architecture"} and float(updated.get("name_confidence") or 0) < 0.45:
        updated["effective_material"] = material
    return updated


def apply_v3_detail(brief: dict, prompts: dict) -> dict:
    """Use the v3 microstructure line for a known subject. The filename subject stays."""
    table = prompts.get("v3_detail") or {}
    text = str(table.get(brief.get("subject_class") or "") or "").strip()
    updated = dict(brief)
    setting = str(brief.get("setting") or "")
    subject = str(brief.get("subject") or "")
    if setting and subject.endswith(setting):
        subject = subject[: -(len(setting))].removesuffix(" on ").strip()
        updated["subject"] = subject
    if not text:
        return updated if updated.get("subject") != brief.get("subject") else brief
    updated["enhance"] = text
    return updated


def compile_prompt(asset: dict, brief: dict, prompts: dict, target_w: int, target_h: int, version: str = "v2") -> str:
    """Positive-only prompt. CFG 1 drops a negative prompt, so restrictions are phrased as what to keep."""
    block = prompts.get(version) or prompts.get("v2") or {}
    shared = (block.get("shared") or "").strip()
    uncertain = float(brief.get("name_confidence") or 0) < 0.35 or not brief.get("subject")
    body = (block.get("uncertain") if uncertain else block.get("certain")) or ""
    fields = {
        "name": asset.get("name") or "unnamed",
        "family": asset.get("family") or "none",
        "subject": brief.get("subject") or "the surface visible in the supplied image",
        "appearance": brief.get("appearance") or "the original palette",
        "enhance": brief.get("enhance") or "fine surface grain of the material already visible",
        "hold": brief.get("hold") or "the original palette, contrast, and feature placement",
        "width": target_w,
        "height": target_h,
    }
    text = (shared + "\n" + body.strip()).format(**fields)
    text += (
        f"\nSource size {asset.get('source_width')}x{asset.get('source_height')}, "
        f"target {target_w}x{target_h}, aspect preserved."
    )
    return text

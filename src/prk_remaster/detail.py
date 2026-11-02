"""Layout reference and coarse colour lock for a Qwen-Image-2.1 edit.

The edit graph drops the vision tokens and splices a VAE latent of the
reference into the sequence at the canvas size. A Lanczos upscale of the
source fills that latent with blur the original texels never contained, and
Turbo at CFG 1 redraws it. The reference built here keeps lanes, panels, and
broad regions, and leaves the finer band empty so the model has to invent it.
Colour is restored afterwards from a wide blur, without replacing that grain.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from .tiles import wrap_blur


def _to_lab(rgb: np.ndarray) -> np.ndarray:
    u8 = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
    return cv2.cvtColor(u8, cv2.COLOR_RGB2LAB).astype(np.float32)


def _from_lab(lab: np.ndarray) -> np.ndarray:
    u8 = np.clip(np.rint(lab), 0, 255).astype(np.uint8)
    return cv2.cvtColor(u8, cv2.COLOR_LAB2RGB).astype(np.float32)


def reference_sigma(scale: float, kind: str, subject_class: str) -> float:
    """Output-pixel radius of one source texel, tightened or loosened by material."""
    if subject_class in {"dirt", "sand", "soil"}:
        factor = 0.85
    elif subject_class in {"cable", "wire", "chain", "rope"}:
        factor = 0.70
    elif subject_class in {"grass", "vegetation"}:
        factor = 0.10
    elif subject_class == "wall":
        factor = 0.20
    elif kind == "graphic":
        factor = 0.28
    elif kind == "organic":
        factor = 0.32
    else:
        factor = 0.55
    return float(np.clip(scale * factor, 0.8, 18.0))


def structure_reference(
    source: Image.Image,
    size: tuple[int, int],
    *,
    kind: str = "organic",
    subject_class: str = "",
) -> Image.Image:
    """Canvas-sized layout. Strong edges stay. Sub-texel mush does not."""
    up = np.asarray(source.convert("RGB").resize(size, Image.Resampling.LANCZOS))
    scale = max(size) / max(max(source.size), 1)
    space = reference_sigma(scale, kind, subject_class)
    colour = 46.0 if subject_class in {"dirt", "sand", "soil"} else 32.0
    if kind == "organic" and subject_class not in {"dirt", "sand", "soil"}:
        colour = 22.0
    if kind == "graphic":
        colour = 18.0
    pad = max(2, int(round(space * 2.0)))
    pad = min(pad, max(1, min(size) // 5))
    padded = np.pad(up, ((pad, pad), (pad, pad), (0, 0)), mode="wrap")
    filtered = cv2.bilateralFilter(padded, 0, colour, space)
    cropped = filtered[pad:-pad, pad:-pad]
    return Image.fromarray(np.ascontiguousarray(cropped), "RGB")


def lock_coarse_colour(generated: Image.Image, source: Image.Image, *, scale: float) -> Image.Image:
    """Move broad colour onto the source. Leave generated grain in the luminance.

    Chroma is locked much more tightly than lightness. A wide chroma blur kept
    the edit's own hue on every strand and grain (blue cable, green soil).
    ``scale`` is accepted so callers can record the upscale; the radii are in
    output pixels, which is where the invented detail actually sits.
    """
    del scale
    generated_rgb = np.asarray(generated.convert("RGB"), dtype=np.float32)
    source_rgb = np.asarray(
        source.convert("RGB").resize(generated.size, Image.Resampling.LANCZOS),
        dtype=np.float32,
    )
    lab_g = _to_lab(generated_rgb)
    lab_s = _to_lab(source_rgb)
    locked = np.empty_like(lab_g)
    for channel, sigma, gain in ((0, 12.0, 1.0), (1, 5.0, 0.25), (2, 5.0, 0.25)):
        low_s = wrap_blur(lab_s[:, :, channel], sigma)
        high_g = lab_g[:, :, channel] - wrap_blur(lab_g[:, :, channel], sigma)
        locked[:, :, channel] = low_s + high_g * gain
    rgb = np.clip(np.rint(_from_lab(locked)), 0, 255).astype(np.uint8)
    return Image.fromarray(rgb, "RGB")

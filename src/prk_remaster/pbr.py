"""Height and ORM derived from PBRFusion depth plus material-class rules.

PBRFusion supplies a normal map and a relative depth field. Roughness, metallic,
and occlusion are explicit policies. They are not claimed as measured PBR.
"""

from __future__ import annotations

import re

import cv2
import numpy as np
from PIL import Image

from .imaging import atomic_save_png


def policy_for(name: str | None, category: str | None, rules: dict) -> dict:
    tokens = set(re.findall(r"[a-z0-9]+", (name or "").lower()))
    classes = rules["classes"]
    if category == "metal":
        if tokens & set(rules.get("painted_tokens") or []):
            return dict(classes["metal_painted"])
        if tokens & set(rules.get("bare_metal_tokens") or []):
            return dict(classes["metal_bare"])
        return dict(classes["metal_painted"])
    policy = classes.get(category or "") or classes["unknown"]
    return dict(policy)


def renormalize_normal(image: Image.Image, flip_y: bool = False) -> Image.Image:
    """Re-unit tangent-space normals. OpenGL +Y is left untouched unless flip_y."""
    array = np.asarray(image.convert("RGB"), dtype=np.float32)
    vectors = array / 255.0 * 2.0 - 1.0
    if flip_y:
        vectors[:, :, 1] *= -1.0
    length = np.linalg.norm(vectors, axis=-1, keepdims=True)
    vectors = vectors / np.clip(length, 1e-6, None)
    encoded = np.clip(np.rint((vectors * 0.5 + 0.5) * 255.0), 0, 255).astype(np.uint8)
    return Image.fromarray(encoded)


def resize_normal(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    array = np.asarray(image.convert("RGB"))
    resized = cv2.resize(array, size, interpolation=cv2.INTER_LINEAR)
    return renormalize_normal(Image.fromarray(resized))


def height_from_depth(depth: Image.Image, relief: float, invert: bool) -> Image.Image:
    """Apparent relief. White is high after the invert flag is applied."""
    field = np.asarray(depth.convert("L"), dtype=np.float32) / 255.0
    if invert:
        field = 1.0 - field
    low, high = np.percentile(field, [2, 98])
    if high - low < 1e-4:
        scaled = np.full_like(field, 0.5)
    else:
        scaled = np.clip((field - low) / (high - low), 0.0, 1.0)
    scaled = np.clip(0.5 + (scaled - 0.5) * float(relief), 0.0, 1.0)
    return Image.fromarray(np.clip(np.rint(scaled * 255.0), 0, 255).astype(np.uint8))


def _occlusion(height: np.ndarray, strength: float, floor: float) -> np.ndarray:
    laplacian = cv2.Laplacian(height.astype(np.float32), cv2.CV_32F)
    valleys = np.clip(laplacian, 0, None)
    scale = float(np.percentile(valleys, 99)) + 1e-6
    valleys = np.clip(valleys / scale, 0.0, 1.0)
    return np.clip(1.0 - strength * valleys, floor, 1.0)


def build_orm(albedo: Image.Image, height: Image.Image, policy: dict, rules: dict) -> Image.Image:
    """R occlusion, G roughness, B metallic. Metalness is not copied from luma."""
    rgb = np.asarray(albedo.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    blur = cv2.GaussianBlur(gray, (0, 0), 2.0)
    detail = np.abs(gray - blur)
    detail = np.clip(detail / (float(np.percentile(detail, 95)) + 1e-6), 0.0, 1.0)
    rough = np.clip(
        float(policy["roughness"]) + (detail - 0.35) * float(policy.get("roughness_detail", 0.0)),
        0.05,
        1.0,
    )
    mode = policy.get("metallic_mode", "nonmetal")
    metal = np.full(gray.shape, float(policy["metallic"]), dtype=np.float32)
    if mode == "painted" and policy.get("allow_exposed"):
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        saturation = hsv[:, :, 1] / 255.0
        value = hsv[:, :, 2] / 255.0
        exposed = (saturation < 0.08) & (value > 0.45) & (value < 0.92)
        metal = np.where(exposed, float(policy.get("exposed_metallic", 0.7)), metal).astype(np.float32)
    height_field = np.asarray(height.convert("L"), dtype=np.float32) / 255.0
    occlusion = _occlusion(
        height_field,
        float(rules.get("ao_strength", 0.18)),
        float(rules.get("ao_floor", 0.82)),
    )
    packed = np.stack([
        np.clip(np.rint(occlusion * 255.0), 0, 255),
        np.clip(np.rint(rough * 255.0), 0, 255),
        np.clip(np.rint(metal * 255.0), 0, 255),
    ], axis=-1).astype(np.uint8)
    return Image.fromarray(packed)


def derive_maps(
    albedo: Image.Image,
    normal: Image.Image,
    depth: Image.Image,
    *,
    name: str,
    category: str,
    rules: dict,
    invert_height: bool,
) -> dict[str, Image.Image]:
    policy = policy_for(name, category, rules)
    normal_map = resize_normal(normal, albedo.size)
    if depth.size != albedo.size:
        depth = depth.convert("L").resize(albedo.size, Image.Resampling.BILINEAR)
    height = height_from_depth(depth, float(policy.get("relief", 0.5)), invert_height)
    orm = build_orm(albedo, height, policy, rules)
    return {
        "albedo": albedo.convert("RGB"),
        "normal": normal_map,
        "height": height,
        "orm": orm,
        "policy": policy,
    }


def save_maps(maps: dict, directory, stem: str) -> dict[str, str]:
    """Write ao-vk suffixes. ``stem`` is ``{type}_{id}``."""
    roles = {"albedo": "d", "normal": "n", "height": "h", "orm": "orm"}
    written = {}
    for role, suffix in roles.items():
        path = directory / f"{stem}_{suffix}.png"
        atomic_save_png(maps[role], path)
        written[role] = str(path)
    return written


def synthetic_bump(size: int = 64) -> Image.Image:
    """A radial bump used to decide whether predicted depth is white-high."""
    yy, xx = np.mgrid[0:size, 0:size]
    center = (size - 1) / 2
    radius = np.sqrt((xx - center) ** 2 + (yy - center) ** 2) / (size * 0.45)
    field = np.clip(1.0 - radius, 0.0, 1.0) ** 2
    gray = np.clip(np.rint(field * 255.0), 0, 255).astype(np.uint8)
    rgb = np.stack([gray, gray, gray], axis=-1)
    return Image.fromarray(rgb)


def depth_is_white_high(depth: Image.Image) -> bool:
    """True when the centre of a radial bump is brighter than the corners."""
    array = np.asarray(depth.convert("L"), dtype=np.float32)
    height, width = array.shape
    center = array[height // 2 - 2 : height // 2 + 2, width // 2 - 2 : width // 2 + 2].mean()
    corners = np.mean([
        array[:4, :4].mean(),
        array[:4, -4:].mean(),
        array[-4:, :4].mean(),
        array[-4:, -4:].mean(),
    ])
    return bool(center > corners)


def cpu_bump_depth(albedo: Image.Image, settings: dict | None = None) -> tuple[Image.Image, Image.Image]:
    """Deterministic stand-in used when a test supplies no diffusion backend.

    Bright texels become high depth. The normal is the OpenGL +Y Sobel of that
    height so the channels describe the same features.
    """
    gray = np.asarray(albedo.convert("L"), dtype=np.float32) / 255.0
    depth = Image.fromarray(np.clip(np.rint(gray * 255.0), 0, 255).astype(np.uint8))
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    normal = np.zeros(gray.shape + (3,), dtype=np.float32)
    normal[:, :, 0] = -dx
    normal[:, :, 1] = -dy
    normal[:, :, 2] = 1.0
    length = np.linalg.norm(normal, axis=-1, keepdims=True)
    normal = normal / np.clip(length, 1e-6, None)
    encoded = np.clip(np.rint((normal * 0.5 + 0.5) * 255.0), 0, 255).astype(np.uint8)
    return Image.fromarray(encoded), depth

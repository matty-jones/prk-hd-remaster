"""Periodic seam measurement, shift conditioning, and diagnostic previews."""

from __future__ import annotations

import numpy as np
from PIL import Image


def _luma(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    return rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


def seam_ratio(image: Image.Image) -> float:
    """Boundary step divided by the median interior step, matching the catalogue.

    A value at or under 2.2 is the catalogue's tileable heuristic. The two axes
    are compared separately and the worse one is returned.
    """
    luma = _luma(image)
    height, width = luma.shape
    if min(height, width) < 16:
        return 99.0
    step_x = max(width // 16, 1)
    columns = [
        float(np.mean(np.abs(luma[:, x] - luma[:, x - 1])))
        for x in range(step_x, width - step_x, step_x)
    ] or [1.0]
    step_y = max(height // 16, 1)
    rows = [
        float(np.mean(np.abs(luma[y, :] - luma[y - 1, :])))
        for y in range(step_y, height - step_y, step_y)
    ] or [1.0]
    edge_x = float(np.mean(np.abs(luma[:, 0] - luma[:, -1])))
    edge_y = float(np.mean(np.abs(luma[0, :] - luma[-1, :])))
    ratio_x = edge_x / max(float(np.median(columns)), 2.0)
    ratio_y = edge_y / max(float(np.median(rows)), 2.0)
    return float(max(ratio_x, ratio_y))


def seam_is_worse(new_ratio: float, source_ratio: float, *, above: float = 2.2, worsen_ratio: float = 1.75, worsen_delta: float = 0.8) -> bool:
    """Fail only when a tileable source grew a clearly worse periodic seam."""
    if new_ratio <= above:
        return False
    return new_ratio > max(source_ratio * worsen_ratio, source_ratio + worsen_delta)


def wrap_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur with periodic boundaries."""
    array = np.asarray(image, dtype=np.float32)
    if sigma < 0.35:
        return array.copy()
    single = array.ndim == 2
    if single:
        array = array[:, :, None]
    height, width, channels = array.shape
    fy = np.fft.fftfreq(height).astype(np.float32)[:, None]
    fx = np.fft.rfftfreq(width).astype(np.float32)[None, :]
    kernel = np.exp(-2.0 * (np.pi ** 2) * (sigma ** 2) * (fy ** 2 + fx ** 2)).astype(np.float32)
    out = np.empty_like(array)
    for channel in range(channels):
        spectrum = np.fft.rfft2(array[:, :, channel])
        out[:, :, channel] = np.fft.irfft2(spectrum * kernel, s=(height, width)).astype(np.float32)
    if single:
        return out[:, :, 0]
    return out


def circular_shift(image: Image.Image, dx: int, dy: int) -> Image.Image:
    array = np.asarray(image)
    shifted = np.roll(np.roll(array, int(dy), axis=0), int(dx), axis=1)
    return Image.fromarray(shifted)


def repair_seam(image: Image.Image, band: int = 8) -> Image.Image:
    """Pull opposite edges together with a short ramp.

    The correction is confined to ``band`` pixels. It does not blur the tile.
    """
    array = np.asarray(image.convert("RGB"), dtype=np.float32)
    height, width, _channels = array.shape
    band = max(1, min(band, width // 4, height // 4))
    delta_x = array[:, 0, :] - array[:, -1, :]
    for index in range(band):
        weight = (band - index) / band
        array[:, index, :] -= delta_x * 0.5 * weight
        array[:, width - 1 - index, :] += delta_x * 0.5 * weight
    delta_y = array[0, :, :] - array[-1, :, :]
    for index in range(band):
        weight = (band - index) / band
        array[index, :, :] -= delta_y * 0.5 * weight
        array[height - 1 - index, :, :] += delta_y * 0.5 * weight
    return Image.fromarray(np.clip(np.rint(array), 0, 255).astype(np.uint8))


def preview_3x3(image: Image.Image, cell: int = 256) -> Image.Image:
    fitted = image.convert("RGB")
    scale = min(1.0, cell / max(fitted.size))
    if scale < 1:
        size = (max(1, int(fitted.size[0] * scale)), max(1, int(fitted.size[1] * scale)))
        fitted = fitted.resize(size, Image.Resampling.BOX)
    array = np.asarray(fitted)
    tiled = np.tile(array, (3, 3, 1))
    return Image.fromarray(tiled)


def relight(albedo: Image.Image, normal: Image.Image, light=(0.35, 0.55, 0.76)) -> Image.Image:
    """Shade a flat preview so inverted normals and dead relief are visible."""
    colour = np.asarray(albedo.convert("RGB"), dtype=np.float32) / 255.0
    normals = np.asarray(normal.convert("RGB").resize(albedo.size, Image.Resampling.BILINEAR), dtype=np.float32)
    vectors = normals / 255.0 * 2.0 - 1.0
    length = np.linalg.norm(vectors, axis=-1, keepdims=True)
    vectors = vectors / np.clip(length, 1e-6, None)
    direction = np.array(light, dtype=np.float32)
    direction = direction / np.linalg.norm(direction)
    shade = np.clip(vectors @ direction, 0.0, 1.0)
    lit = colour * (0.22 + 0.78 * shade[..., None])
    return Image.fromarray(np.clip(np.rint(lit * 255.0), 0, 255).astype(np.uint8))


def shape_correlation(left: Image.Image, right: Image.Image) -> float:
    """Downsampled luma correlation, used to notice a large UV shift."""
    size = (32, 32)
    a = np.asarray(left.convert("L").resize(size, Image.Resampling.BOX), dtype=np.float32).ravel()
    b = np.asarray(right.convert("L").resize(size, Image.Resampling.BOX), dtype=np.float32).ravel()
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-6:
        return 1.0
    return float(np.dot(a, b) / denom)

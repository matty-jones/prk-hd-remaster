"""Image bytes, hashes, and atomic PNG writes."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def pixel_sha256(image: Image.Image) -> str:
    """Hash decoded RGBA pixels and their shape, including alpha."""
    rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    digest = hashlib.sha256()
    digest.update(f"{rgba.shape[1]}x{rgba.shape[0]}x4".encode("ascii"))
    digest.update(rgba.tobytes())
    return digest.hexdigest()


def atomic_save_png(image: Image.Image, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    image.save(temporary, format="PNG")
    temporary.replace(path)
    return path


def load_png(path: Path) -> Image.Image:
    with Image.open(path) as image:
        image.load()
        return image.copy()

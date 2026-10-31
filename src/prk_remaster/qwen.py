"""Qwen-Image-2.1-Turbo image-conditioned edit via the official ComfyUI graph."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from .comfy_client import ComfyClient
from .config import REPO_ROOT

WORKFLOW_PATH = REPO_ROOT / "workflows" / "qwen-21-turbo-edit.json"
REQUIRED_NODES = [
    "UNETLoader",
    "CLIPLoader",
    "VAELoader",
    "LoadImage",
    "TextEncodeQwenImage21",
    "QwenImage21Cache",
    "ManualSigmas",
    "KSamplerSelect",
    "RandomNoise",
    "CFGGuider",
    "SamplerCustomAdvanced",
    "VAEDecode",
    "SaveImage",
]


def load_workflow() -> dict:
    return json.loads(WORKFLOW_PATH.read_text(encoding="utf-8"))


def build_workflow(
    *,
    image_name: str,
    prompt: str,
    seed: int,
    diffusion: str,
    text_encoder: str,
    vae: str,
    sigmas: str,
    cache_device: str = "cpu",
    cache_dtype: str = "int8",
    resolution: int = 0,
) -> dict:
    """Fill the versioned API graph. CFG stays 1 and the Turbo sigma list is fixed."""
    graph = load_workflow()
    graph["1"]["inputs"]["unet_name"] = diffusion
    graph["2"]["inputs"]["clip_name"] = text_encoder
    graph["3"]["inputs"]["vae_name"] = vae
    graph["4"]["inputs"]["image"] = image_name
    graph["5"]["inputs"]["prompt"] = prompt
    graph["5"]["inputs"]["negative_prompt"] = ""
    graph["5"]["inputs"]["resolution"] = int(resolution)
    graph["6"]["inputs"]["device"] = cache_device
    graph["6"]["inputs"]["dtype"] = cache_dtype
    graph["7"]["inputs"]["sigmas"] = sigmas
    graph["9"]["inputs"]["noise_seed"] = int(seed)
    graph["10"]["inputs"]["cfg"] = 1.0
    graph["8"]["inputs"]["sampler_name"] = "euler"
    return graph


def edit_image(
    client: ComfyClient,
    image: Image.Image,
    prompt: str,
    seed: int,
    settings: dict,
    lock: dict,
) -> Image.Image:
    uploaded = client.upload_image(image, f"prk-src-{seed}.png")
    files = lock["qwen"]["files"]
    workflow = build_workflow(
        image_name=uploaded,
        prompt=prompt,
        seed=seed,
        diffusion=files["diffusion"]["filename"],
        text_encoder=files["text_encoder"]["filename"],
        vae=files["vae"]["filename"],
        sigmas=lock["qwen"]["sigmas"],
        cache_device=settings.get("cache_device", "cpu"),
        cache_dtype=settings.get("cache_dtype", "int8"),
        resolution=int(settings.get("qwen_resolution", 0)),
    )
    prompt_id = client.submit(workflow)
    entry = client.wait(prompt_id)
    images = client.result_images(entry)
    # SaveImage is the last consumer. Prefer the node id we assigned it.
    chosen = None
    outputs = entry.get("outputs") or {}
    if "13" in outputs and outputs["13"].get("images"):
        chosen = outputs["13"]["images"][0]
    else:
        chosen = images[-1]
    return client.download(chosen)

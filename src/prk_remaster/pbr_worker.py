"""PBRFusion 4 normal and depth via ComfyUI. ORM is built afterwards on the CPU."""

from __future__ import annotations

import json

from PIL import Image

from .comfy_client import ComfyClient
from .config import REPO_ROOT

WORKFLOW_PATH = REPO_ROOT / "workflows" / "pbrfusion4-normal-depth.json"
NODE_NAME = "AIXPOLY_PBRFusion4SimpleDepthAndNormalGenerator"


def load_workflow() -> dict:
    return json.loads(WORKFLOW_PATH.read_text(encoding="utf-8"))


def build_workflow(image_name: str, settings: dict) -> dict:
    graph = load_workflow()
    graph["1"]["inputs"]["image"] = image_name
    node = graph["2"]["inputs"]
    node["intensity_blend"] = float(settings.get("pbr_intensity_blend", 0.2))
    node["opengl_normal"] = True
    node["optimize"] = bool(settings.get("pbr_optimize", False))
    node["bilateral_d"] = int(settings.get("pbr_bilateral_d", 10))
    node["bilateral_sigma_color"] = float(settings.get("pbr_bilateral_sigma_color", 50))
    node["bilateral_sigma_space"] = float(settings.get("pbr_bilateral_sigma_space", 50))
    return graph


def estimate(client: ComfyClient, albedo: Image.Image, settings: dict) -> tuple[Image.Image, Image.Image]:
    """Return (OpenGL normal, filtered depth)."""
    uploaded = client.upload_image(albedo, "prk-albedo.png")
    workflow = build_workflow(uploaded, settings)
    prompt_id = client.submit(workflow)
    entry = client.wait(prompt_id)
    outputs = entry.get("outputs") or {}
    depth_info = (outputs.get("3") or {}).get("images", [None])[0]
    normal_info = (outputs.get("4") or {}).get("images", [None])[0]
    if depth_info is None or normal_info is None:
        raise RuntimeError("PBRFusion did not return depth and normal images")
    depth = client.download(depth_info).convert("L")
    normal = client.download(normal_info).convert("RGB")
    return normal, depth

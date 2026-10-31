"""Load factory configuration and resolve local paths."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

PATH_KEYS = ("prk_root", "catalogue_csv", "ao_vk_root", "work_root")


def load_yaml(path: Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not contain a mapping")
    return data


def load_config(path: Path | None = None) -> dict:
    if path is None:
        env = os.environ.get("PRK_REMASTER_CONFIG")
        path = Path(env) if env else REPO_ROOT / "config" / "default.yaml"
    cfg_path = Path(path).resolve()
    cfg = load_yaml(cfg_path)
    cfg["_config_path"] = str(cfg_path)
    cfg["_repo_root"] = str(REPO_ROOT)
    for key in PATH_KEYS:
        if cfg.get(key):
            cfg[key] = str(Path(os.path.expanduser(str(cfg[key]))).resolve())
    if os.environ.get("PRK_WORK_ROOT"):
        cfg["work_root"] = str(Path(os.environ["PRK_WORK_ROOT"]).resolve())
    cfg_dir = cfg_path.parent
    cfg["family_prompts"] = load_yaml(cfg_dir / "family-prompts.yaml")
    cfg["material_rules"] = load_yaml(cfg_dir / "material-rules.yaml")
    cfg["exclusions"] = load_yaml(cfg_dir / "exclusions.yaml")
    cfg["model_lock"] = load_yaml(cfg_dir / "model-lock.yaml")
    return cfg


def work_paths(cfg: dict) -> dict[str, Path]:
    root = Path(cfg["work_root"])
    return {
        "root": root,
        "registry": root / "registry.sqlite",
        "runs": root / "runs",
        "dist": root / "dist",
        "cache": root / "cache",
        "comfy": root / "comfyui",
        "license": root / "license-acceptance.json",
        "receipts": root / "install-receipts",
        "logs": root / "logs",
    }


def ensure_work_dirs(cfg: dict) -> dict[str, Path]:
    paths = work_paths(cfg)
    for key, path in paths.items():
        if key == "registry":
            path.parent.mkdir(parents=True, exist_ok=True)
        elif key != "license":
            path.mkdir(parents=True, exist_ok=True)
    return paths

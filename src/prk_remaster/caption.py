"""Cache source captions for unreliable briefs. One Qwen2.5-VL-3B, not a second generator."""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)

from .config import REPO_ROOT
from .imaging import sha256_file

WORKER = REPO_ROOT / "scripts" / "caption_worker.py"


def caption_python(cfg: dict) -> Path:
    configured = ((cfg.get("fidelity") or {}).get("caption_python")) or ""
    if configured:
        return Path(configured)
    return Path(cfg["work_root"]) / "comfyui" / ".venv" / "bin" / "python"


def caption_model_dir(cfg: dict) -> Path:
    return Path(cfg["work_root"]) / "models" / "captioner" / "Qwen2.5-VL-3B-Instruct"


def ensure_caption_model(cfg: dict) -> Path:
    dest = caption_model_dir(cfg)
    if (dest / "config.json").is_file():
        return dest
    lock = (cfg.get("model_lock") or {}).get("captioner") or {}
    repo = lock.get("repo") or "Qwen/Qwen2.5-VL-3B-Instruct"
    dest.parent.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=repo, local_dir=str(dest))
    if not (dest / "config.json").is_file():
        raise RuntimeError(f"captioner download did not produce {dest}")
    return dest


def caption_path(cache_dir: Path, source: Path) -> Path:
    return cache_dir / f"{sha256_file(source)}.json"


def cached_caption(cache_dir: Path, source: Path) -> dict | None:
    path = caption_path(cache_dir, source)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_caption(cache_dir: Path, source: Path, payload: dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    caption_path(cache_dir, source).write_text(json.dumps(payload), encoding="utf-8")


def caption_many(cfg: dict, jobs: list[dict], cache_dir: Path) -> dict[str, dict]:
    """Caption each source once. ``jobs`` use keys image, name, category, family. Failures yield {}."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    found: dict[str, dict] = {}
    pending = []
    for job in jobs:
        source = Path(job["image"])
        existing = cached_caption(cache_dir, source)
        if existing is not None:
            found[str(source)] = existing
            continue
        pending.append(job)
    if not pending:
        return found
    python = caption_python(cfg)
    if not python.is_file() or not WORKER.is_file():
        return found
    try:
        model = ensure_caption_model(cfg)
    except Exception:
        return found
    manifest = cache_dir / "_batch.jsonl"
    lines = []
    for job in pending:
        source = Path(job["image"])
        lines.append(json.dumps({
            "image": str(source),
            "out": str(caption_path(cache_dir, source)),
            "name": job.get("name") or "",
            "category": job.get("category") or "",
            "family": job.get("family") or "",
        }))
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        subprocess.run(
            [str(python), str(WORKER), "--model", str(model), "--manifest", str(manifest)],
            check=True,
            timeout=max(600, 90 * len(pending)),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("caption worker failed: %s", exc)
    for job in pending:
        source = Path(job["image"])
        existing = cached_caption(cache_dir, source)
        if existing is not None:
            found[str(source)] = existing
    return found

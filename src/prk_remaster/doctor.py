"""Local environment checks. Nothing here writes to the game client."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .comfy_client import ComfyClient
from .models import license_accepted, missing_weights
from .qwen import REQUIRED_NODES
from . import pbr_worker


def _gpu() -> tuple[bool, str]:
    if shutil.which("nvidia-smi") is None:
        return False, "nvidia-smi is not on PATH"
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            text=True,
            timeout=10,
        ).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return False, str(exc)
    return True, output


def doctor(cfg: dict, *, offline: bool = False) -> list[tuple[str, bool, str]]:
    checks = []
    prk = Path(cfg["prk_root"])
    rdb = prk / "cd_image" / "rdb.db"
    checks.append(("prk root", prk.is_dir(), str(prk)))
    checks.append(("rdb readable", rdb.is_file(), str(rdb)))
    csv_path = Path(cfg["catalogue_csv"])
    checks.append(("catalogue csv", csv_path.is_file(), str(csv_path)))
    work = Path(cfg["work_root"])
    checks.append(("work root writable", work.exists() and os_access_write(work), str(work)))
    ok, detail = _gpu()
    checks.append(("gpu", ok, detail))
    from .config import work_paths
    accepted = license_accepted(work_paths(cfg)["license"])
    checks.append(("qwen licence accepted", accepted, "private local experimentation" if accepted else "not accepted"))
    missing = missing_weights(cfg)
    checks.append(("model weights", not missing, "missing " + ", ".join(missing) if missing else "present"))
    caption_dir = Path(cfg["work_root"]) / "models" / "captioner" / "Qwen2.5-VL-3B-Instruct"
    present = (caption_dir / "config.json").is_file()
    required = bool((cfg.get("fidelity") or {}).get("require_captioner"))
    checks.append(("captioner", present or not required, "present" if present else "not downloaded"))
    if offline:
        checks.append(("comfyui", True, "skipped (--offline)"))
        return checks
    try:
        client = ComfyClient(cfg["comfy_url"])
    except Exception as exc:  # noqa: BLE001
        checks.append(("comfyui", False, str(exc)))
        return checks
    alive = client.healthy()
    checks.append(("comfyui", alive, cfg["comfy_url"] if alive else "not answering"))
    if alive:
        try:
            needed = REQUIRED_NODES + [pbr_worker.NODE_NAME]
            absent = client.missing_nodes(needed)
            checks.append(("comfy nodes", not absent, "missing " + ", ".join(absent) if absent else "present"))
        except Exception as exc:  # noqa: BLE001
            checks.append(("comfy nodes", False, str(exc)))
    return checks


def os_access_write(path: Path) -> bool:
    return os_access(path)


def os_access(path: Path) -> bool:
    import os
    return os.access(path, os.W_OK)


def format_doctor(checks: list[tuple[str, bool, str]]) -> str:
    lines = []
    for name, ok, detail in checks:
        flag = "ok" if ok else "FAIL"
        lines.append(f"[{flag}] {name}: {detail}")
    return "\n".join(lines) + "\n"

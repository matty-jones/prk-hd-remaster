"""Model licence gate, ComfyUI bootstrap, and weight checksums."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import yaml

from .comfy_client import ComfyClient
from .config import REPO_ROOT, work_paths
from .imaging import sha256_file
from .registry import utc_now

LICENSE_NAME = "Qwen Research License"


class ModelError(RuntimeError):
    pass


def accept_license(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "license": LICENSE_NAME,
        "commercial_use": False,
        "purpose": "private local experimentation",
        "accepted_at": utc_now(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def license_accepted(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return payload.get("license") == LICENSE_NAME and payload.get("commercial_use") is False


def require_license(path: Path) -> None:
    if not license_accepted(path):
        raise ModelError(
            "Qwen Research License has not been accepted for private local use. "
            "Re-run with --accept-qwen-research-license. Weights are not downloaded without that flag."
        )


def _run(command: list[str], **kwargs) -> None:
    subprocess.run(command, check=True, **kwargs)


def bootstrap_comfy(cfg: dict) -> dict[str, str]:
    """Clone ComfyUI and the PBRFusion nodes. Idempotent."""
    paths = work_paths(cfg)
    script = REPO_ROOT / "scripts" / "bootstrap-models.sh"
    env = os.environ.copy()
    env["PRK_WORK_ROOT"] = str(paths["root"])
    _run(["bash", str(script)], env=env)
    comfy = paths["comfy"]
    node = comfy / "custom_nodes" / "COMFYUI-PBRFusion4"
    return {
        "comfyui": subprocess.check_output(["git", "-C", str(comfy), "rev-parse", "HEAD"], text=True).strip(),
        "pbr_nodes": subprocess.check_output(["git", "-C", str(node), "rev-parse", "HEAD"], text=True).strip(),
    }


def _repo_filename(repo: str, basename: str) -> str:
    from huggingface_hub import HfApi
    files = HfApi().list_repo_files(repo)
    matches = [item for item in files if item.endswith("/" + basename) or item == basename]
    if not matches:
        raise ModelError(f"{basename} is not in {repo}")
    matches.sort(key=len)
    return matches[0]


def _link(source: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_symlink() or dest.is_file():
        if dest.is_file() and not dest.is_symlink() and sha256_file(dest) == sha256_file(source):
            return
        dest.unlink()
    dest.symlink_to(source)


def download_weights(cfg: dict) -> dict[str, str]:
    """Download the locked files and return basename -> sha256."""
    from huggingface_hub import hf_hub_download
    paths = work_paths(cfg)
    comfy_models = paths["comfy"] / "models"
    lock = cfg["model_lock"]
    cache = paths["root"] / "models" / "hf"
    cache.mkdir(parents=True, exist_ok=True)
    digests = {}
    qwen_repo = lock["qwen"]["repo"]
    for _role, spec in lock["qwen"]["files"].items():
        filename = spec["filename"]
        remote = _repo_filename(qwen_repo, filename)
        local = hf_hub_download(repo_id=qwen_repo, filename=remote, local_dir=cache / "qwen")
        digest = sha256_file(local)
        expected = spec.get("sha256")
        if expected and expected != digest:
            raise ModelError(f"{filename} sha256 {digest} != lock {expected}")
        _link(Path(local), comfy_models / spec["dest"] / filename)
        digests[filename] = digest
    pbr = lock["pbrfusion4"]
    remote = _repo_filename(pbr["repo"], pbr["filename"])
    local = hf_hub_download(repo_id=pbr["repo"], filename=remote, local_dir=cache / "pbrfusion4")
    digest = sha256_file(local)
    if pbr.get("sha256") and pbr["sha256"] != digest:
        raise ModelError(f"PBRFusion sha256 {digest} != lock {pbr['sha256']}")
    _link(Path(local), comfy_models / pbr["dest"] / pbr["filename"])
    digests[pbr["filename"]] = digest
    return digests


def write_lock_digests(cfg: dict, digests: dict[str, str], revisions: dict[str, str] | None = None) -> None:
    path = Path(cfg["_config_path"]).parent / "model-lock.yaml"
    lock = cfg["model_lock"]
    for spec in lock["qwen"]["files"].values():
        if spec["filename"] in digests:
            spec["sha256"] = digests[spec["filename"]]
    if lock["pbrfusion4"]["filename"] in digests:
        lock["pbrfusion4"]["sha256"] = digests[lock["pbrfusion4"]["filename"]]
    if revisions:
        lock["comfyui"]["revision"] = revisions.get("comfyui")
        lock["pbr_nodes"]["revision"] = revisions.get("pbr_nodes")
    path.write_text(yaml.safe_dump(lock, sort_keys=False), encoding="utf-8")


def launch_comfy(cfg: dict, wait_seconds: float = 180.0) -> int:
    import time
    paths = work_paths(cfg)
    client = ComfyClient(cfg["comfy_url"])
    if client.healthy():
        return 0
    python = paths["comfy"] / ".venv" / "bin" / "python"
    if not python.is_file():
        raise ModelError(f"ComfyUI venv is missing at {python}")
    log_path = paths["logs"] / "comfy.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("a", encoding="utf-8")
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    process = subprocess.Popen(
        [
            str(python), "main.py",
            "--listen", str(cfg.get("comfy_host", "127.0.0.1")),
            "--port", str(cfg.get("comfy_port", 8188)),
            "--disable-auto-launch",
        ],
        cwd=paths["comfy"],
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )
    (paths["root"] / "comfy.pid").write_text(str(process.pid), encoding="utf-8")
    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        if client.healthy():
            return process.pid
        if process.poll() is not None:
            raise ModelError(f"ComfyUI exited {process.returncode}; see {log_path}")
        time.sleep(2)
    raise ModelError(f"ComfyUI did not answer on {cfg['comfy_url']} within {wait_seconds:.0f}s")


def setup(cfg: dict, *, accept: bool, download: bool = True, launch: bool = False) -> dict:
    paths = work_paths(cfg)
    if accept:
        accept_license(paths["license"])
    require_license(paths["license"])
    revisions = bootstrap_comfy(cfg)
    digests = download_weights(cfg) if download else {}
    if digests or revisions:
        write_lock_digests(cfg, digests, revisions)
    pid = launch_comfy(cfg) if launch else None
    return {"revisions": revisions, "digests": digests, "pid": pid, "license": str(paths["license"])}


def installed_weight_paths(cfg: dict) -> list[Path]:
    models = work_paths(cfg)["comfy"] / "models"
    lock = cfg["model_lock"]
    paths = []
    for spec in lock["qwen"]["files"].values():
        paths.append(models / spec["dest"] / spec["filename"])
    pbr = lock["pbrfusion4"]
    paths.append(models / pbr["dest"] / pbr["filename"])
    return paths


def missing_weights(cfg: dict) -> list[str]:
    return [path.name for path in installed_weight_paths(cfg) if not path.is_file()]

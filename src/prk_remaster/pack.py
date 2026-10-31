"""Build an ao-vk texture pack. Filenames use the RDB type and id."""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

from .imaging import atomic_save_png, load_png, sha256_file
from .registry import Registry, utc_now

MAP_ROLES = {"albedo_final": "d", "normal": "n", "height": "h", "orm": "orm"}
REQUIRED = ("d", "n", "h", "orm")
NAME_RE_TABLES = {1010004, 1010006, 1010009, 1010011}


class PackError(RuntimeError):
    pass


def map_filename(rdb_type: int, rdb_id: int, suffix: str) -> str:
    return f"{int(rdb_type)}_{int(rdb_id)}_{suffix}.png"


def _copy_map(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_name(dest.name + ".tmp")
    shutil.copyfile(src, temporary)
    temporary.replace(dest)


def build_pack(registry: Registry, run_id: str, dest: Path, name: str) -> dict:
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    jobs = registry.jobs(run_id)
    by_key = {job["asset_key"]: job for job in jobs}
    included = []
    missing = []
    for job in jobs:
        if job["status"] not in {"qa_passed", "packed"}:
            continue
        if job.get("canonical_key"):
            continue
        asset = registry.asset(job["asset_key"])
        if asset is None:
            missing.append(job["asset_key"])
            continue
        outputs = {item["role"]: item for item in registry.outputs(run_id, job["asset_key"])}
        entry = _write_asset(registry, run_id, asset, outputs, dest, shared_from=None)
        if entry is None:
            missing.append(job["asset_key"])
            continue
        included.append(entry)
        registry.update_job(run_id, job["asset_key"], status="packed", stage="pack")
        for alias in registry.aliases_of(asset["key"]):
            if alias["disposition"] != asset["disposition"]:
                continue
            alias_entry = _write_asset(registry, run_id, alias, outputs, dest, shared_from=asset["key"])
            if alias_entry:
                included.append(alias_entry)
        for other in jobs:
            canon = other.get("canonical_key")
            if canon != asset["key"]:
                continue
            other_asset = registry.asset(other["asset_key"])
            if other_asset is None:
                continue
            alias_entry = _write_asset(registry, run_id, other_asset, outputs, dest, shared_from=asset["key"])
            if alias_entry:
                included.append(alias_entry)
                registry.update_job(run_id, other["asset_key"], status="packed", stage="pack", canonical_key=asset["key"])
    manifest = {
        "name": name,
        "run_id": run_id,
        "created_at": utc_now(),
        "assets": included,
        "missing_passed_jobs": missing,
    }
    manifest_path = dest / "pack-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    problems = validate_pack(dest, manifest)
    if problems:
        raise PackError("; ".join(problems))
    zip_path = dest.parent / f"{name}.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(dest.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=str(Path(name) / path.relative_to(dest)))
    manifest["zip"] = str(zip_path)
    manifest["pack_dir"] = str(dest)
    return manifest


def _write_asset(registry, run_id, asset, outputs, dest, shared_from) -> dict | None:
    maps = {}
    for role, suffix in MAP_ROLES.items():
        item = outputs.get(role)
        if item is None or not item.get("path") or not Path(item["path"]).is_file():
            return None
        filename = map_filename(asset["rdb_type"], asset["rdb_id"], suffix)
        target = dest / filename
        _copy_map(Path(item["path"]), target)
        maps[suffix] = {
            "file": filename,
            "sha256": sha256_file(target),
            "bytes": target.stat().st_size,
        }
    return {
        "key": asset["key"],
        "rdb_type": asset["rdb_type"],
        "rdb_id": asset["rdb_id"],
        "source_sha256": asset.get("source_sha256"),
        "shared_from": shared_from,
        "maps": maps,
    }


def validate_pack(dest: Path, manifest: dict | None = None) -> list[str]:
    dest = Path(dest)
    problems = []
    if manifest is None:
        path = dest / "pack-manifest.json"
        if not path.is_file():
            return ["pack-manifest.json is missing"]
        manifest = json.loads(path.read_text(encoding="utf-8"))
    seen = set()
    for asset in manifest.get("assets") or []:
        key = asset["key"]
        if key in seen:
            problems.append(f"duplicate manifest entry {key}")
        seen.add(key)
        table = int(asset["rdb_type"])
        resource_id = int(asset["rdb_id"])
        if table not in NAME_RE_TABLES:
            problems.append(f"{key} is not a full-quality texture table")
        if f"{table}:{resource_id}" != key:
            problems.append(f"{key} does not match type/id fields")
        for suffix in REQUIRED:
            filename = map_filename(table, resource_id, suffix)
            path = dest / filename
            if not path.is_file():
                problems.append(f"missing {filename}")
                continue
            try:
                with load_png(path) as image:
                    image.load()
            except Exception as exc:  # noqa: BLE001 - report the bad file
                problems.append(f"{filename} is not a readable PNG ({exc})")
    return problems


def expected_files(registry: Registry, run_id: str) -> list[str]:
    names = []
    for job in registry.jobs(run_id):
        if job["status"] not in {"qa_passed", "packed", "pending", "extracted", "generated", "pbr"}:
            continue
        asset = registry.asset(job["asset_key"])
        if asset is None:
            continue
        for suffix in REQUIRED:
            names.append(map_filename(asset["rdb_type"], asset["rdb_id"], suffix))
    return names

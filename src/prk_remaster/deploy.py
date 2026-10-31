"""Install and remove an ao-vk pack without touching the game database."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from .pack import validate_pack
from .registry import utc_now

GAME_MARKERS = ("anarchy.exe", "anarchyonline.exe")


class DeployError(RuntimeError):
    pass


def rdb_path(prk_root: Path) -> Path:
    return Path(prk_root) / "cd_image" / "rdb.db"


def snapshot_rdb(prk_root: Path) -> dict:
    path = rdb_path(prk_root)
    if not path.is_file():
        raise DeployError(f"RDB not found at {path}")
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def assert_rdb_unchanged(before: dict, prk_root: Path) -> None:
    after = snapshot_rdb(prk_root)
    if after["size"] != before["size"] or after["mtime_ns"] != before["mtime_ns"]:
        raise DeployError("cd_image/rdb.db changed; install aborted")


def game_running(prk_root: Path) -> bool:
    """True when an Anarchy process is using this client directory."""
    root = str(Path(prk_root).resolve())
    proc = Path("/proc")
    if not proc.is_dir():
        return False
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\x00", b" ").decode("utf-8", "replace")
            cwd = os.readlink(entry / "cwd")
        except OSError:
            continue
        lowered = command.lower()
        if not any(marker in lowered for marker in GAME_MARKERS):
            continue
        if root in command or root in cwd:
            return True
    return False


def _order_path(prk_root: Path) -> Path:
    return Path(prk_root) / "randy-vk" / "packs" / "order.txt"


def read_order(prk_root: Path) -> list[str]:
    path = _order_path(prk_root)
    if not path.is_file():
        return []
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        lines.append(stripped)
    return lines


def write_order(prk_root: Path, names: list[str]) -> None:
    path = _order_path(prk_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(names) + ("\n" if names else "")
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(body, encoding="utf-8")
    temporary.replace(path)


def _backup_order(prk_root: Path, receipt_dir: Path) -> str | None:
    path = _order_path(prk_root)
    receipt_dir.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        return None
    stamp = utc_now().replace(":", "").replace("+00:00", "Z")
    backup = path.with_name(f"order.txt.bak.{stamp}")
    shutil.copy2(path, backup)
    shutil.copy2(path, receipt_dir / backup.name)
    return str(backup)


def install_plan(prk_root: Path, pack_dir: Path, name: str | None = None) -> dict:
    pack_dir = Path(pack_dir)
    problems = validate_pack(pack_dir)
    if problems:
        raise DeployError("pack failed validation: " + "; ".join(problems))
    name = name or pack_dir.name
    destination = Path(prk_root) / "randy-vk" / "packs" / name
    files = sorted(path.name for path in pack_dir.glob("*.png"))
    order = read_order(prk_root)
    return {
        "name": name,
        "source": str(pack_dir),
        "destination": str(destination),
        "files": files,
        "order_before": order,
        "order_after": [name] + [item for item in order if item != name],
        "would_replace_folder": destination.exists(),
        "rdb": str(rdb_path(prk_root)),
    }


def install(prk_root: Path, pack_dir: Path, *, name: str | None = None, receipt_dir: Path, dry_run: bool = False) -> dict:
    prk_root = Path(prk_root)
    plan = install_plan(prk_root, pack_dir, name)
    plan["dry_run"] = dry_run
    if dry_run:
        return plan
    if game_running(prk_root):
        raise DeployError("PRK looks running; close the game before installing a pack")
    before = snapshot_rdb(prk_root)
    receipt_dir = Path(receipt_dir)
    receipt_dir.mkdir(parents=True, exist_ok=True)
    backup = _backup_order(prk_root, receipt_dir)
    destination = Path(plan["destination"])
    if destination.exists():
        raise DeployError(f"{destination} already exists; refusing to overwrite another pack")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(pack_dir, destination)
    write_order(prk_root, plan["order_after"])
    assert_rdb_unchanged(before, prk_root)
    receipt = {
        "action": "install",
        "at": utc_now(),
        "name": plan["name"],
        "destination": str(destination),
        "order_backup": backup,
        "order_after": plan["order_after"],
        "rdb_before": before,
        "files": len(plan["files"]),
    }
    receipt_path = receipt_dir / f"install-{plan['name']}.json"
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    # Touch nothing else in the client. Confirm the database file is still the same inode size.
    assert_rdb_unchanged(before, prk_root)
    receipt["receipt"] = str(receipt_path)
    return receipt


def disable(prk_root: Path, name: str, *, receipt_dir: Path, dry_run: bool = False, yes: bool = False) -> dict:
    order = read_order(prk_root)
    plan = {"action": "disable", "name": name, "order_before": order, "order_after": [item for item in order if item != name]}
    if dry_run or not yes:
        plan["applied"] = False
        return plan
    before = snapshot_rdb(prk_root)
    _backup_order(prk_root, Path(receipt_dir))
    write_order(prk_root, plan["order_after"])
    assert_rdb_unchanged(before, prk_root)
    plan["applied"] = True
    return plan


def uninstall(prk_root: Path, name: str, *, receipt_dir: Path, dry_run: bool = False, yes: bool = False) -> dict:
    destination = Path(prk_root) / "randy-vk" / "packs" / name
    order = read_order(prk_root)
    plan = {
        "action": "uninstall",
        "name": name,
        "remove_dir": str(destination),
        "dir_exists": destination.is_dir(),
        "order_before": order,
        "order_after": [item for item in order if item != name],
    }
    if dry_run or not yes:
        plan["applied"] = False
        return plan
    if not destination.is_dir():
        raise DeployError(f"pack folder not found: {destination}")
    before = snapshot_rdb(prk_root)
    _backup_order(prk_root, Path(receipt_dir))
    shutil.rmtree(destination)
    write_order(prk_root, plan["order_after"])
    assert_rdb_unchanged(before, prk_root)
    # The database file must not have been opened for write. Size and mtime are the check.
    os.sync()
    assert_rdb_unchanged(before, prk_root)
    plan["applied"] = True
    return plan

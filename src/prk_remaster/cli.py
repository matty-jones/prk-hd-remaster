"""prk-remaster command line."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from . import __version__
from .catalogue import format_reconcile, import_csv, reconcile
from .config import ensure_work_dirs, load_config, work_paths
from .deploy import disable, install, uninstall
from .doctor import doctor, format_doctor
from .gallery import write_gallery
from .models import ModelError, setup
from .pack import build_pack, expected_files
from .planner import dumps_plan, load_ids, plan_jobs
from .registry import Registry
from .report import write_report
from .runner import run_cohort
from .salvage import run_salvage


def _config(args):
    cfg = load_config(getattr(args, "config", None))
    if getattr(args, "prk_root", None):
        cfg["prk_root"] = str(Path(args.prk_root).expanduser().resolve())
    if getattr(args, "comfy_url", None):
        cfg["comfy_url"] = args.comfy_url
    return cfg


def _registry(cfg) -> Registry:
    paths = ensure_work_dirs(cfg)
    return Registry(paths["registry"])


def _resolve_run(registry: Registry, run: str | None, cohort: str | None = None) -> str:
    if run in (None, "", "latest"):
        found = registry.latest_run_id(cohort if run == "latest" else None)
        if found is None:
            found = registry.latest_run_id()
        if found is None:
            raise SystemExit("no runs recorded")
        return found
    return run


def _types(value: str | None) -> list[int] | None:
    if not value:
        return None
    return [int(item) for item in value.split(",") if item.strip()]


def cmd_doctor(args) -> int:
    cfg = _config(args)
    text = format_doctor(doctor(cfg, offline=args.offline))
    sys.stdout.write(text)
    failed = [line for line in text.splitlines() if line.startswith("[FAIL]")]
    return 1 if failed else 0


def cmd_catalogue_import(args) -> int:
    cfg = _config(args)
    registry = _registry(cfg)
    csv_path = Path(args.csv or cfg["catalogue_csv"])
    summary = import_csv(csv_path, registry, cfg.get("prompt_version", "v1"))
    sys.stdout.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return 0


def cmd_catalogue_reconcile(args) -> int:
    cfg = _config(args)
    registry = _registry(cfg)
    db = Path(cfg["prk_root"]) / "cd_image" / "rdb.db"
    report = reconcile(registry, db if db.is_file() else None, cfg["exclusions"].get("lod_tables"))
    sys.stdout.write(format_reconcile(report))
    return 0


def cmd_models_setup(args) -> int:
    cfg = _config(args)
    ensure_work_dirs(cfg)
    try:
        result = setup(cfg, accept=args.accept_qwen_research_license, launch=not args.no_launch)
    except ModelError as exc:
        sys.stderr.write(str(exc) + "\n")
        return 2
    sys.stdout.write(json.dumps(result, indent=2, default=str) + "\n")
    return 0


def cmd_run(args) -> int:
    cfg = _config(args)
    paths = ensure_work_dirs(cfg)
    registry = Registry(paths["registry"])
    run_id = None if args.run in (None, "", "latest") else args.run
    if args.resume and run_id is None:
        run_id = registry.latest_run_id(args.cohort)
    run_dir = Path(args.output_dir) if args.output_dir else paths["runs"] / (run_id or "pending")
    result = run_cohort(
        cfg, registry, run_dir,
        cohort=args.cohort, preset=args.preset, run_id=run_id, resume=args.resume,
        limit=args.limit, family=args.family, types=_types(args.types),
        ids=load_ids(args.ids_file), force=args.force, retry_failed=args.retry_failed,
        dry_run=args.dry_run, measure_vram=not args.dry_run,
    )
    if args.dry_run:
        sys.stdout.write(dumps_plan(result["jobs"]))
        return 0
    # The run directory uses the id assigned inside run_cohort.
    actual = result["run_id"]
    if not args.output_dir:
        final_dir = paths["runs"] / actual
        if run_dir != final_dir and run_dir.exists() and not final_dir.exists():
            run_dir.rename(final_dir)
    sys.stdout.write(json.dumps({"run_id": actual, "summary": result.get("summary"), "detail": result.get("detail")}, indent=2) + "\n")
    return 0


def cmd_salvage(args) -> int:
    cfg = _config(args)
    paths = ensure_work_dirs(cfg)
    registry = Registry(paths["registry"])
    keys = None
    if args.keys:
        keys = {item.strip() for item in args.keys.split(",") if item.strip()}
    if args.ids_file:
        keys = load_ids(args.ids_file)
    result = run_salvage(
        cfg, registry,
        from_run=args.from_run,
        run_id=None if args.run in (None, "", "latest") else args.run,
        preset=args.preset,
        keys=keys,
        stop_after=args.stop_after,
        force=args.force,
        resume=args.resume,
        regen_cap=args.regen_cap,
        pack_name=args.pack_name,
    )
    sys.stdout.write(json.dumps(result, indent=2) + "\n")
    return 0


def cmd_status(args) -> int:
    cfg = _config(args)
    registry = _registry(cfg)
    run_id = _resolve_run(registry, args.run)
    summary = registry.summary(run_id)
    sys.stdout.write(json.dumps(summary, indent=2) + "\n")
    return 0


def cmd_gallery(args) -> int:
    cfg = _config(args)
    paths = ensure_work_dirs(cfg)
    registry = Registry(paths["registry"])
    run_id = _resolve_run(registry, args.run)
    run_dir = Path(args.output_dir) if args.output_dir else paths["runs"] / run_id
    index = write_gallery(registry, run_id, run_dir)
    sys.stdout.write(str(index) + "\n")
    return 0


def cmd_report(args) -> int:
    cfg = _config(args)
    paths = ensure_work_dirs(cfg)
    registry = Registry(paths["registry"])
    run_id = _resolve_run(registry, args.run)
    run = registry.run(run_id) or {}
    dest = Path(args.output_dir) if args.output_dir else paths["runs"] / run_id / "REPORT.md"
    path = write_report(registry, run_id, dest, extra=run.get("detail"))
    sys.stdout.write(str(path) + "\n")
    return 0


def cmd_pack_build(args) -> int:
    cfg = _config(args)
    paths = ensure_work_dirs(cfg)
    registry = Registry(paths["registry"])
    run_id = _resolve_run(registry, args.run)
    if args.dry_run:
        for name in expected_files(registry, run_id):
            sys.stdout.write(name + "\n")
        return 0
    dest = paths["dist"] / args.name
    manifest = build_pack(registry, run_id, dest, args.name)
    sys.stdout.write(json.dumps({"pack": manifest["pack_dir"], "assets": len(manifest["assets"]), "zip": manifest["zip"]}, indent=2) + "\n")
    return 0


def cmd_pack_install(args) -> int:
    cfg = _config(args)
    paths = ensure_work_dirs(cfg)
    pack = Path(args.pack)
    if not pack.is_absolute():
        pack = Path.cwd() / pack
    receipt = install(
        Path(cfg["prk_root"]), pack, name=args.name, receipt_dir=paths["receipts"], dry_run=args.dry_run,
    )
    sys.stdout.write(json.dumps(receipt, indent=2) + "\n")
    return 0


def cmd_pack_disable(args) -> int:
    cfg = _config(args)
    paths = ensure_work_dirs(cfg)
    result = disable(
        Path(cfg["prk_root"]), args.name, receipt_dir=paths["receipts"], dry_run=args.dry_run, yes=args.yes,
    )
    sys.stdout.write(json.dumps(result, indent=2) + "\n")
    return 0


def cmd_pack_uninstall(args) -> int:
    cfg = _config(args)
    paths = ensure_work_dirs(cfg)
    result = uninstall(
        Path(cfg["prk_root"]), args.name, receipt_dir=paths["receipts"], dry_run=args.dry_run, yes=args.yes,
    )
    sys.stdout.write(json.dumps(result, indent=2) + "\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="prk-remaster")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--config", type=Path)
    sub = parser.add_subparsers(dest="command", required=True)

    doctor_p = sub.add_parser("doctor")
    doctor_p.add_argument("--prk-root")
    doctor_p.add_argument("--comfy-url")
    doctor_p.add_argument("--offline", action="store_true")
    doctor_p.set_defaults(func=cmd_doctor)

    catalogue = sub.add_parser("catalogue")
    cat_sub = catalogue.add_subparsers(dest="catalogue_cmd", required=True)
    imp = cat_sub.add_parser("import")
    imp.add_argument("--prk-root")
    imp.add_argument("--csv")
    imp.set_defaults(func=cmd_catalogue_import)
    rec = cat_sub.add_parser("reconcile")
    rec.add_argument("--prk-root")
    rec.set_defaults(func=cmd_catalogue_reconcile)

    models = sub.add_parser("models")
    models_sub = models.add_subparsers(dest="models_cmd", required=True)
    setup_p = models_sub.add_parser("setup")
    setup_p.add_argument("--profile", default="rtx5080-16gb")
    setup_p.add_argument("--accept-qwen-research-license", action="store_true")
    setup_p.add_argument("--no-launch", action="store_true")
    setup_p.set_defaults(func=cmd_models_setup)

    run_p = sub.add_parser("run")
    run_p.add_argument("--cohort", default="first-1076")
    run_p.add_argument("--preset", default="aggressive")
    run_p.add_argument("--run", default="latest")
    run_p.add_argument("--resume", action="store_true")
    run_p.add_argument("--limit", type=int)
    run_p.add_argument("--family")
    run_p.add_argument("--types")
    run_p.add_argument("--ids-file")
    run_p.add_argument("--force", action="store_true")
    run_p.add_argument("--retry-failed", action="store_true")
    run_p.add_argument("--dry-run", action="store_true")
    run_p.add_argument("--output-dir")
    run_p.add_argument("--comfy-url")
    run_p.add_argument("--prk-root")
    run_p.set_defaults(func=cmd_run)

    salvage_p = sub.add_parser("salvage")
    salvage_p.add_argument("--from-run", required=True)
    salvage_p.add_argument("--run", default="latest")
    salvage_p.add_argument("--preset", default="aggressive")
    salvage_p.add_argument("--keys", help="Comma-separated asset keys")
    salvage_p.add_argument("--ids-file")
    salvage_p.add_argument("--stop-after", default="pack", choices=("anchor", "regen", "pbr", "pack"))
    salvage_p.add_argument("--force", action="store_true")
    salvage_p.add_argument("--resume", action="store_true")
    salvage_p.add_argument("--regen-cap", type=int)
    salvage_p.add_argument("--pack-name", default="prk-hd-v02")
    salvage_p.add_argument("--comfy-url")
    salvage_p.add_argument("--prk-root")
    salvage_p.set_defaults(func=cmd_salvage)

    for name, func in (("status", cmd_status), ("gallery", cmd_gallery), ("report", cmd_report)):
        item = sub.add_parser(name)
        item.add_argument("--run", default="latest")
        item.add_argument("--output-dir")
        item.set_defaults(func=func)

    pack = sub.add_parser("pack")
    pack_sub = pack.add_subparsers(dest="pack_cmd", required=True)
    build = pack_sub.add_parser("build")
    build.add_argument("--run", default="latest")
    build.add_argument("--name", default="prk-hd-v01")
    build.add_argument("--dry-run", action="store_true")
    build.set_defaults(func=cmd_pack_build)
    inst = pack_sub.add_parser("install")
    inst.add_argument("--prk-root")
    inst.add_argument("--pack", required=True)
    inst.add_argument("--name")
    inst.add_argument("--dry-run", action="store_true")
    inst.set_defaults(func=cmd_pack_install)
    dis = pack_sub.add_parser("disable")
    dis.add_argument("--prk-root")
    dis.add_argument("--name", required=True)
    dis.add_argument("--dry-run", action="store_true")
    dis.add_argument("--yes", action="store_true")
    dis.set_defaults(func=cmd_pack_disable)
    un = pack_sub.add_parser("uninstall")
    un.add_argument("--prk-root")
    un.add_argument("--name", required=True)
    un.add_argument("--dry-run", action="store_true")
    un.add_argument("--yes", action="store_true")
    un.set_defaults(func=cmd_pack_uninstall)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

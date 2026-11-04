"""Restrain microstructure on flat v3 surfaces. Does not rewrite the v3 run.

Expressive materials (cable, soil, grass, other terrain) are hardlinked.
Interior, architecture, metal, and dressed stone get a quieter albedo, a new
structural normal, and the removed grain on the normal and the roughness.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import signal
from pathlib import Path

from PIL import Image

from .config import REPO_ROOT, ensure_work_dirs
from .detail import lock_coarse_colour, structure_reference
from .gallery import PRIORITY_KEYS
from .imaging import atomic_save_png, load_png
from .pack import build_pack
from .pbr import blend_detail_field, build_orm, policy_for, stamp_detail_field
from .planner import cache_key, reduced_long_side
from .reconstruct import _edit_once, _repair
from .registry import Registry
from .runner import (
    _asset_dir,
    _default_edit,
    _default_pbr,
    _mirror_aliases,
    _pbr_stage,
    _qa,
    _record,
    _settings,
    is_oom,
    new_run_id,
)
from .scale import (
    DIAGNOSIS_KEYS,
    _lab,
    meso_agreement,
    mode_gains,
    prefer_regen,
    restrain_albedo,
    scale_metrics,
    scale_mode,
    split_bands,
    texel_px,
    write_distance_sheet,
)
from .semantics import apply_v3_detail, build_brief, compile_prompt

log = logging.getLogger("prk.refine")
SCALE_VERSION = 1
DONE = {"qa_passed", "packed"}
STAGES = ("albedo", "pbr", "pack")
_PAUSE = False


def find_refine_run(registry: Registry, from_run: str) -> str | None:
    rows = registry.conn.execute(
        "SELECT run_id, detail FROM runs ORDER BY created_at DESC"
    ).fetchall()
    for row in rows:
        detail = json.loads(row["detail"] or "{}")
        if detail.get("kind") == "refine" and detail.get("from_run") == from_run:
            return row["run_id"]
    return None


def _request_pause(signum, _frame) -> None:
    global _PAUSE
    _PAUSE = True
    log.info("pause requested (signal %s); finishing the current texture", signum)


def _should_pause(run_dir: Path) -> bool:
    return _PAUSE or (run_dir / "STOP").exists()


def _link(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        try:
            if dest.stat().st_ino == src.stat().st_ino and dest.stat().st_dev == src.stat().st_dev:
                return
        except OSError:
            pass
        dest.unlink()
    try:
        os.link(src, dest)
    except OSError:
        shutil.copy2(src, dest)


def _ordered(keys: list[str]) -> list[str]:
    rank = {key: index for index, key in enumerate(PRIORITY_KEYS)}
    diagnosis = {key: index for index, key in enumerate(DIAGNOSIS_KEYS)}
    return sorted(keys, key=lambda key: (diagnosis.get(key, 100), rank.get(key, 10_000), key))


def _resume_text(from_run: str, run_id: str) -> str:
    return (
        f"cd {REPO_ROOT}\n"
        "source .venv/bin/activate\n"
        f"prk-remaster refine --from-run {from_run} --run {run_id} --resume\n"
    )


def _write_resume(run_dir: Path, from_run: str, run_id: str) -> None:
    (run_dir / "RESUME.txt").write_text(_resume_text(from_run, run_id), encoding="utf-8")


def _load_stats(run_dir: Path) -> dict:
    path = run_dir / "stats.json"
    if not path.is_file():
        return {"gpu_seconds": 0.0, "peak_vram_mib": 0.0}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"gpu_seconds": 0.0, "peak_vram_mib": 0.0}


def _save_stats(run_dir: Path, stats: dict) -> None:
    (run_dir / "stats.json").write_text(json.dumps(stats), encoding="utf-8")


def _sweep_comfy(cfg: dict) -> None:
    """Drop ComfyUI's extra copies. The run already stores the maps it keeps."""
    root = Path(cfg["work_root"]) / "comfyui"
    for folder, pattern in (("output", "prk-*"), ("input", "prk-src-*")):
        directory = root / folder
        if not directory.is_dir():
            continue
        for path in directory.glob(pattern):
            if path.is_file():
                path.unlink()


def _run_detail(from_run: str, record: dict, stage: str) -> dict:
    return {
        "kind": "refine", "from_run": from_run, "stage": stage,
        "prompt_version": record["prompt_version"], "config_revision": record["config_revision"],
        "scale_version": SCALE_VERSION,
    }


def _asset_mode(asset: dict, detail: dict, policy: dict) -> str:
    return scale_mode(
        str(detail.get("subject_class") or ""),
        str(detail.get("kind") or ""),
        str(detail.get("effective_material") or asset.get("material_class") or ""),
        policy,
    )


def _gain_value(policy: dict, mode: str) -> float:
    return round(float(mode_gains(policy, mode)["micro_albedo_gain"]), 4)


def _albedo_current(detail: dict, gain: float, directory: Path) -> bool:
    return (
        detail.get("scale_version") == SCALE_VERSION
        and detail.get("micro_albedo_gain") == gain
        and (directory / "albedo_final.png").is_file()
    )


def run_refine(
    cfg: dict,
    registry: Registry,
    *,
    from_run: str,
    run_id: str | None = None,
    keys: set[str] | None = None,
    stop_after: str = "pack",
    force: bool = False,
    resume: bool = False,
    limit: int | None = None,
    pack_name: str | None = None,
    skip_regen: bool = False,
    pbr_fn=None,
    edit_fn=None,
    measure_vram: bool = True,
) -> dict:
    if stop_after not in STAGES:
        raise ValueError(f"unsupported stop_after {stop_after}")
    recipe = cfg.get("refine") or {}
    policy = cfg.get("scale_policy") or {}
    pack_name = pack_name or recipe.get("pack_name") or "prk-hd-v04"
    paths = ensure_work_dirs(cfg)
    from_dir = paths["runs"] / from_run
    if not from_dir.is_dir():
        raise SystemExit(f"source run not found: {from_dir}")
    source_run = registry.run(from_run) or {}
    if resume and run_id is None:
        run_id = find_refine_run(registry, from_run)
        if run_id is None:
            raise SystemExit(f"no refine run to resume from {from_run}")
    if run_id is None:
        run_id = new_run_id()
    if run_id == from_run:
        raise SystemExit("refine refuses to write into the source run")
    run_dir = paths["runs"] / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    if from_dir.resolve() == run_dir.resolve() or from_dir.resolve() in run_dir.resolve().parents:
        raise SystemExit("refine refuses to write inside the source run")
    stop = run_dir / "STOP"
    if resume and stop.exists():
        stop.unlink()
    _write_resume(run_dir, from_run, run_id)
    record = {
        "config_revision": recipe.get("config_revision", "v0.3.1"),
        "prompt_version": recipe.get("prompt_version", "v3"),
    }
    if registry.run(run_id) is None:
        registry.create_run(run_id, "first-1076", "refine", record)
    registry.update_run(run_id, status="running", detail=_run_detail(from_run, record, "albedo"))
    registry.events_path = run_dir / "events.jsonl"
    selected = _select(registry, from_run, keys)
    _ensure_jobs(registry, run_id, from_run, from_dir, run_dir, selected, record)
    stats = _load_stats(run_dir)
    compare = [str(item) for item in (recipe.get("compare_keys") or [])]
    global _PAUSE
    _PAUSE = False
    previous_term = signal.getsignal(signal.SIGTERM)
    previous_int = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGTERM, _request_pause)
    signal.signal(signal.SIGINT, _request_pause)
    try:
        paused = _albedos(
            registry, cfg, run_id, run_dir, from_dir, from_run, source_run, policy, record,
            stats, force, limit, keys, compare, skip_regen, edit_fn, measure_vram,
        )
        _save_stats(run_dir, stats)
        _write_sheet(registry, run_id, run_dir, from_dir, paths["runs"], source_run, policy)
        if paused or _PAUSE or stop.exists():
            _mark_paused(registry, run_id, from_run, record, "albedo")
            return {"run_id": run_id, "stage": "paused", "resume": _resume_text(from_run, run_id).strip()}
        if stop_after == "albedo":
            registry.update_run(run_id, status="generated", detail=_run_detail(from_run, record, "albedo"))
            return {"run_id": run_id, "stage": "albedo", "jobs": len(registry.jobs(run_id))}
        paused = _maps(registry, cfg, run_id, run_dir, policy, stats, force, pbr_fn, measure_vram)
        _save_stats(run_dir, stats)
        if paused or _PAUSE or stop.exists():
            _mark_paused(registry, run_id, from_run, record, "pbr")
            return {"run_id": run_id, "stage": "paused", "resume": _resume_text(from_run, run_id).strip()}
        if stop_after == "pbr":
            registry.update_run(run_id, status="pbr", detail=_run_detail(from_run, record, "pbr"))
            return {"run_id": run_id, "stage": "pbr", "jobs": len(registry.jobs(run_id))}
        _mirror_aliases(registry, run_id)
        dest = paths["dist"] / pack_name
        manifest = build_pack(registry, run_id, dest, pack_name, archive=False)
        registry.update_run(run_id, status="packed", detail={
            **_run_detail(from_run, record, "pack"), "pack": str(dest),
            "gpu_seconds": round(float(stats.get("gpu_seconds") or 0), 1),
        })
        return {
            "run_id": run_id, "stage": "pack", "pack": manifest.get("pack_dir"),
            "assets": len(manifest.get("assets") or []),
        }
    finally:
        signal.signal(signal.SIGTERM, previous_term)
        signal.signal(signal.SIGINT, previous_int)


def _mark_paused(registry, run_id, from_run, record, stage) -> None:
    registry.update_run(run_id, status="paused", detail=_run_detail(from_run, record, stage))
    log.info("paused during %s", stage)


def _select(registry: Registry, from_run: str, keys: set[str] | None) -> dict[str, dict]:
    chosen = {}
    for job in registry.jobs(from_run):
        if job["status"] not in DONE:
            continue
        if keys is not None and job["asset_key"] not in keys:
            continue
        chosen[job["asset_key"]] = job
    return chosen


def _ensure_jobs(registry, run_id, from_run, from_dir, run_dir, selected, record) -> None:
    for key, source_job in selected.items():
        asset = registry.asset(key)
        if asset is None:
            continue
        canonical = source_job.get("canonical_key")
        if canonical not in selected:
            canonical = None
        existing = registry.job(run_id, key)
        if canonical:
            if existing is None:
                registry.upsert_job({
                    "run_id": run_id, "asset_key": key, "status": "extracted", "stage": "alias",
                    "seed": source_job.get("seed"), "prompt": "", "cache_key": source_job.get("cache_key"),
                    "target_w": source_job.get("target_w"), "target_h": source_job.get("target_h"),
                    "attempt": 0, "rejection_reason": None, "canonical_key": canonical,
                    "detail": {"from_run": from_run, "alias": True, "scale_version": SCALE_VERSION},
                })
            continue
        source = from_dir / "assets" / key.replace(":", "_") / "source.png"
        if not source.is_file():
            registry.upsert_job({
                "run_id": run_id, "asset_key": key, "status": "blocked", "stage": "link",
                "seed": source_job.get("seed"), "prompt": "", "cache_key": "",
                "target_w": 0, "target_h": 0, "attempt": 0,
                "rejection_reason": "source albedo inputs are missing", "canonical_key": None,
                "detail": {"from_run": from_run},
            })
            continue
        directory = _asset_dir(run_dir, key)
        _link(source, directory / "source.png")
        if existing and existing["status"] not in {"blocked", "failed"}:
            continue
        detail = dict(source_job.get("detail") or {})
        detail.update({
            "from_run": from_run,
            "scale_version": None,
            "generated": False,
        })
        registry.upsert_job({
            "run_id": run_id, "asset_key": key, "status": "extracted", "stage": "link",
            "seed": source_job.get("seed") or 1, "prompt": source_job.get("prompt") or "",
            "cache_key": cache_key(
                asset.get("source_sha256") or key, "refine",
                record["prompt_version"], record["config_revision"], "albedo",
            ),
            "target_w": int(source_job.get("target_w") or 0),
            "target_h": int(source_job.get("target_h") or 0),
            "attempt": 0, "rejection_reason": None, "canonical_key": None,
            "detail": detail,
        })
        registry.log_event(run_id, key, "refine_linked")


def _albedos(
    registry, cfg, run_id, run_dir, from_dir, from_run, source_run, policy, record,
    stats, force, limit, keys, compare, skip_regen, edit_fn, measure_vram,
) -> bool:
    pending = []
    for job in registry.jobs(run_id):
        if job.get("canonical_key") or job["status"] in {"blocked", "rejected"}:
            continue
        if keys is not None and job["asset_key"] not in keys:
            continue
        pending.append(job["asset_key"])
    pending = _ordered(pending)
    if limit is not None:
        pending = pending[:limit]
    log.info("albedo queue %s", len(pending))
    for index, key in enumerate(pending, start=1):
        if _should_pause(run_dir):
            log.info("pausing before %s", key)
            return True
        job = registry.job(run_id, key)
        if job is None:
            continue
        try:
            _one_albedo(registry, cfg, run_id, run_dir, from_dir, job, policy, force)
        except Exception as exc:  # noqa: BLE001
            registry.update_job(run_id, key, status="failed", rejection_reason=str(exc)[:500], stage="albedo")
            registry.log_event(run_id, key, "albedo_error", error=str(exc)[:500])
            log.info("albedo failed %s: %s", key, str(exc)[:200])
        if index % 25 == 0:
            log.info("albedo progress %s/%s", index, len(pending))
    if not skip_regen:
        _regen_compares(
            registry, cfg, run_id, run_dir, from_dir, source_run, policy, record,
            stats, force, compare, edit_fn, measure_vram,
        )
    return False


def _one_albedo(registry, cfg, run_id, run_dir, from_dir, job, policy, force) -> None:
    key = job["asset_key"]
    directory = _asset_dir(run_dir, key)
    asset = registry.asset(key) or {}
    detail = dict(job.get("detail") or {})
    mode = _asset_mode(asset, detail, policy)
    gain = _gain_value(policy, mode)
    src_dir = from_dir / "assets" / key.replace(":", "_")
    if _albedo_current(detail, gain, directory) and not force:
        return
    if mode == "expressive":
        for name in ("albedo_final.png", "normal.png", "height.png", "orm.png"):
            src = src_dir / name
            if not src.is_file():
                raise FileNotFoundError(f"missing {src}")
            _link(src, directory / name)
        for role, name in (
            ("albedo_final", "albedo_final.png"),
            ("normal", "normal.png"),
            ("height", "height.png"),
            ("orm", "orm.png"),
            ("source", "source.png"),
        ):
            _record(registry, run_id, key, role, directory / name)
        detail.update({
            "scale_mode": mode,
            "scale_version": SCALE_VERSION,
            "micro_albedo_gain": gain,
            "albedo_choice": "linked",
            "detail_stamped": True,
            "linked_from": str(src_dir),
        })
        registry.update_job(run_id, key, status="qa_passed", stage="albedo", detail=detail, rejection_reason=None)
        registry.log_event(run_id, key, "refine_linked_expressive", gain=gain)
        return
    donor_src = src_dir / "albedo_final.png"
    if not donor_src.is_file():
        raise FileNotFoundError(f"missing {donor_src}")
    _link(donor_src, directory / "albedo_donor.png")
    source = load_png(directory / "source.png").convert("RGB")
    donor = load_png(directory / "albedo_donor.png").convert("RGB")
    restrained, _residual = restrain_albedo(donor, source.size, mode, policy)
    restrained = _repair(restrained, asset, cfg)
    atomic_save_png(restrained, directory / "albedo_final.png")
    _record(registry, run_id, key, "source", directory / "source.png")
    _record(registry, run_id, key, "albedo_final", directory / "albedo_final.png")
    _record(registry, run_id, key, "albedo_donor", directory / "albedo_donor.png")
    detail.update({
        "scale_mode": mode,
        "scale_version": SCALE_VERSION,
        "micro_albedo_gain": gain,
        "albedo_choice": "split",
        "detail_stamped": False,
        "fidelity": scale_metrics(source, restrained, policy),
    })
    target = restrained.size
    registry.update_job(
        run_id, key, status="generated", stage="albedo", detail=detail,
        target_w=target[0], target_h=target[1], rejection_reason=None,
    )
    registry.log_event(run_id, key, "refine_restrained", gain=gain, micro_rms=detail["fidelity"]["micro_rms"])


def _regen_compares(
    registry, cfg, run_id, run_dir, from_dir, source_run, policy, record,
    stats, force, compare, edit_fn, measure_vram,
) -> None:
    wanted = [key for key in compare if registry.job(run_id, key)]
    if not wanted:
        return
    own_edit = edit_fn is None
    free_fn = None
    if own_edit:
        try:
            from .models import launch_comfy
            launch_comfy(cfg)
        except Exception as exc:  # noqa: BLE001
            log.info("floor regen skipped; comfy did not start: %s", str(exc)[:200])
            return
        edit_fn, free_fn = _default_edit(cfg)
    settings = _settings(cfg)
    runs = from_dir.parent
    v1_id = (source_run.get("detail") or {}).get("from_run") or ""
    for key in wanted:
        if _should_pause(run_dir):
            break
        job = registry.job(run_id, key)
        if job is None or job.get("canonical_key"):
            continue
        detail = dict(job.get("detail") or {})
        if detail.get("scale_mode") != "restrained":
            continue
        directory = _asset_dir(run_dir, key)
        if detail.get("regen_compared") and (directory / "albedo_regen.png").is_file() and not force:
            continue
        try:
            _regen_one(
                registry, cfg, run_id, run_dir, directory, job, policy, record, stats,
                edit_fn, settings, runs, v1_id, from_dir, measure_vram,
            )
        except Exception as exc:  # noqa: BLE001
            detail["regen_compared"] = True
            detail["regen_error"] = str(exc)[:300]
            registry.update_job(run_id, key, detail=detail)
            log.info("floor regen kept the split for %s: %s", key, str(exc)[:200])
    if free_fn is not None:
        try:
            free_fn()
        except Exception:
            log.info("comfy free after regen did not complete")


def _regen_one(
    registry, cfg, run_id, run_dir, directory, job, policy, record, stats,
    edit_fn, settings, runs, v1_id, from_dir, measure_vram,
) -> None:
    del measure_vram, run_dir
    key = job["asset_key"]
    asset = registry.asset(key) or {}
    source = load_png(directory / "source.png").convert("RGB")
    brief = apply_v3_detail(
        build_brief(asset, source, cfg.get("name_lexicon") or {}),
        cfg.get("family_prompts") or {},
        policy,
    )
    brief["scale_mode"] = "restrained"
    target = (int(job.get("target_w") or 0), int(job.get("target_h") or 0))
    if min(target) < 32:
        target = source.size
    prompt = compile_prompt(
        asset, brief, cfg.get("family_prompts") or {}, target[0], target[1], version=record["prompt_version"],
    )
    (directory / "prompt_regen.txt").write_text(prompt + "\n", encoding="utf-8")
    reference = structure_reference(
        source, target, kind=brief.get("kind") or "manufactured",
        subject_class=brief.get("subject_class") or "", scale_mode="restrained",
    )
    try:
        edited, elapsed, peak = _edit_once(edit_fn, reference, prompt, int(job.get("seed") or 1), settings, stats)
    except Exception as exc:
        if not is_oom(exc) or max(target) <= 1024:
            raise
        smaller = reduced_long_side(target[0], target[1], 1024, int(cfg["material_rules"].get("snap", 32)))
        reference = structure_reference(
            source, smaller, kind=brief.get("kind") or "manufactured",
            subject_class=brief.get("subject_class") or "", scale_mode="restrained",
        )
        edited, elapsed, peak = _edit_once(edit_fn, reference, prompt, int(job.get("seed") or 1), settings, stats)
        target = smaller
    scale = max(edited.size) / max(max(source.size), 1)
    locked = _repair(lock_coarse_colour(edited.convert("RGB"), source, scale=scale), asset, cfg)
    atomic_save_png(locked, directory / "albedo_regen.png")
    _record(registry, run_id, key, "albedo_regen", directory / "albedo_regen.png")
    split = load_png(directory / "albedo_final.png").convert("RGB")
    v3 = load_png(from_dir / "assets" / key.replace(":", "_") / "albedo_final.png").convert("RGB")
    v1_path = runs / v1_id / "assets" / key.replace(":", "_") / "albedo_final.png"
    v1 = load_png(v1_path).convert("RGB") if v1_id and v1_path.is_file() else None
    use, info = prefer_regen(source, v1, v3, split, locked, policy)
    detail = dict((registry.job(run_id, key) or job).get("detail") or {})
    detail["regen_compared"] = True
    detail["regen_choice"] = info
    detail["regen_seconds"] = round(elapsed, 3)
    if peak:
        detail["regen_vram_mib"] = peak
    if use:
        atomic_save_png(locked, directory / "albedo_final.png")
        _record(registry, run_id, key, "albedo_final", directory / "albedo_final.png")
        detail["albedo_choice"] = "regen"
        detail["detail_stamped"] = False
        log.info("floor regen replaces the split for %s (%s)", key, info.get("reason"))
    else:
        log.info("floor regen kept the split for %s (%s)", key, info.get("reason"))
    registry.update_job(run_id, key, status="generated", stage="albedo", detail=detail, prompt=prompt)
    registry.log_event(run_id, key, "refine_regen", **{k: info[k] for k in info if k != "reason"}, reason=info.get("reason"))


def _maps(registry, cfg, run_id, run_dir, policy, stats, force, pbr_fn, measure_vram) -> bool:
    jobs = _ordered([
        job["asset_key"] for job in registry.jobs(run_id)
        if not job.get("canonical_key") and job["status"] not in {"blocked", "rejected", "failed"}
    ])
    needs_gpu = False
    for key in jobs:
        job = registry.job(run_id, key)
        if job is None:
            continue
        detail = job.get("detail") or {}
        if detail.get("scale_mode") == "restrained" and (force or not detail.get("detail_stamped")):
            needs_gpu = True
            break
    own_pbr = pbr_fn is None
    free_fn = None
    if needs_gpu and own_pbr:
        from .models import launch_comfy
        _sweep_comfy(cfg)
        launch_comfy(cfg)
        pbr_fn, free_fn = _default_pbr(cfg)
    elif pbr_fn is None:
        pbr_fn = _cpu_pbr
    try:
        if free_fn is not None:
            try:
                free_fn()
            except Exception:
                log.info("comfy free before pbr did not complete")
        for index, key in enumerate(jobs, start=1):
            if _should_pause(run_dir):
                log.info("pausing before pbr %s", key)
                return True
            job = registry.job(run_id, key)
            if job is None:
                continue
            detail = dict(job.get("detail") or {})
            if detail.get("scale_mode") != "restrained":
                continue
            if detail.get("detail_stamped") and not force and job["status"] in DONE:
                continue
            try:
                if force:
                    for name in ("normal_base.png", "height_base.png"):
                        extra = _asset_dir(run_dir, key) / name
                        if extra.exists():
                            extra.unlink()
                if _needs_pbr(job, force):
                    _pbr_stage(registry, cfg, run_id, run_dir, job, pbr_fn, stats, measure_vram and own_pbr, free_fn)
                    current = registry.job(run_id, key)
                    if current and current["status"] == "pbr":
                        marked = dict(current.get("detail") or {})
                        marked["refine_pbr"] = True
                        registry.update_job(run_id, key, detail=marked)
                current = registry.job(run_id, key)
                if current and current["status"] not in {"failed", "blocked"}:
                    _stamp(registry, cfg, run_id, run_dir, current, policy, force)
                    current = registry.job(run_id, key)
                if current and current["status"] == "pbr":
                    _qa(registry, cfg, run_id, run_dir, current)
                    for name in ("depth_raw.png", "preview_relit.png"):
                        extra = _asset_dir(run_dir, key) / name
                        if extra.is_file():
                            extra.unlink()
            except Exception as exc:  # noqa: BLE001
                registry.update_job(run_id, key, status="failed", rejection_reason=str(exc)[:500], stage="pbr")
                registry.log_event(run_id, key, "refine_pbr_error", error=str(exc)[:500])
                log.info("pbr failed %s: %s", key, str(exc)[:200])
            if index % 20 == 0:
                _save_stats(run_dir, stats)
                if own_pbr:
                    _sweep_comfy(cfg)
                log.info("pbr progress %s/%s", index, len(jobs))
    finally:
        if free_fn is not None:
            try:
                free_fn()
            except Exception:
                log.info("comfy free after pbr did not complete")
    return False


def _cpu_pbr(albedo, _settings):
    from .pbr import cpu_bump_depth
    return cpu_bump_depth(albedo)


def _needs_pbr(job: dict, force: bool) -> bool:
    """A copied v3 timing must not count. Only a normal built in this pass does."""
    if force:
        return True
    detail = job.get("detail") or {}
    if detail.get("detail_stamped"):
        return False
    return not detail.get("refine_pbr")


def _stamp(registry, cfg, run_id, run_dir, job, policy, force: bool) -> None:
    key = job["asset_key"]
    detail = dict(job.get("detail") or {})
    if detail.get("detail_stamped") and not force:
        return
    directory = _asset_dir(run_dir, key)
    normal_path = directory / "normal.png"
    height_path = directory / "height.png"
    base_normal = directory / "normal_base.png"
    base_height = directory / "height_base.png"
    if not normal_path.is_file() or not height_path.is_file():
        raise FileNotFoundError(f"missing structural maps for {key}")
    _keep_base(normal_path, base_normal)
    _keep_base(height_path, base_height)
    albedo = load_png(directory / "albedo_final.png").convert("RGB")
    source = load_png(directory / "source.png").convert("RGB")
    structural = load_png(base_normal)
    height = load_png(base_height)
    gains = mode_gains(policy, "restrained")
    weight = float(gains["micro_normal_weight"])
    field = _residual_field(directory, albedo, source.size, detail, policy)
    detailed = blend_detail_field(structural, field, weight)
    stamped = stamp_detail_field(height, field, weight)
    asset = registry.asset(key) or {}
    category = detail.get("effective_material") or asset.get("material_class") or ""
    rules = cfg["material_rules"]
    orm = build_orm(
        albedo, stamped, policy_for(asset.get("name") or "", category, rules), rules,
        residual=field, residual_roughness=float(gains["micro_roughness"]),
    )
    atomic_save_png(detailed, normal_path)
    atomic_save_png(stamped, height_path)
    atomic_save_png(orm, directory / "orm.png")
    _record(registry, run_id, key, "normal", normal_path)
    _record(registry, run_id, key, "normal_base", base_normal)
    _record(registry, run_id, key, "height", height_path)
    _record(registry, run_id, key, "orm", directory / "orm.png")
    detail["detail_stamped"] = True
    detail["detail_weight"] = weight
    detail["pbr_label"] = "pbr-1024-plus-scale"
    registry.update_job(run_id, key, status="pbr", stage="pbr", detail=detail)
    registry.log_event(run_id, key, "scale_stamped", weight=weight)


def _keep_base(src: Path, dest: Path) -> None:
    if dest.exists():
        return
    try:
        os.link(src, dest)
    except OSError:
        shutil.copy2(src, dest)


def _residual_field(directory: Path, albedo: Image.Image, source_size: tuple[int, int], detail: dict, policy: dict):
    """Lightness removed from the albedo. A regen keeps only its own micro band."""
    if detail.get("albedo_choice") == "regen":
        return split_bands(_lab(albedo)[:, :, 0], texel_px(source_size, albedo.size), policy)["micro"]
    donor_path = directory / "albedo_donor.png"
    if not donor_path.is_file():
        raise FileNotFoundError(donor_path)
    donor = load_png(donor_path).convert("RGB")
    if donor.size != albedo.size:
        donor = donor.resize(albedo.size, Image.Resampling.LANCZOS)
    return _lab(donor)[:, :, 0] - _lab(albedo)[:, :, 0]


def _write_sheet(registry, run_id, run_dir, from_dir, runs: Path, source_run: dict, policy: dict) -> None:
    v1_id = (source_run.get("detail") or {}).get("from_run") or ""
    report = {}
    gallery = run_dir / "gallery"
    for key in DIAGNOSIS_KEYS:
        if registry.job(run_id, key) is None:
            continue
        directory = _asset_dir(run_dir, key)
        albedo_path = directory / "albedo_final.png"
        source_path = directory / "source.png"
        if not albedo_path.is_file() or not source_path.is_file():
            continue
        source = load_png(source_path).convert("RGB")
        refined = load_png(albedo_path).convert("RGB")
        v3_path = from_dir / "assets" / key.replace(":", "_") / "albedo_final.png"
        v3 = load_png(v3_path).convert("RGB") if v3_path.is_file() else None
        v1_path = runs / v1_id / "assets" / key.replace(":", "_") / "albedo_final.png" if v1_id else None
        v1 = load_png(v1_path).convert("RGB") if v1_path is not None and v1_path.is_file() else None
        columns = [("source", source.resize(refined.size, Image.Resampling.NEAREST))]
        entry = {"refined": scale_metrics(source, refined, policy)}
        if v1 is not None:
            columns.append(("v1", v1.resize(refined.size, Image.Resampling.LANCZOS) if v1.size != refined.size else v1))
            entry["v1"] = scale_metrics(source, columns[-1][1], policy)
        if v3 is not None:
            shown = v3 if v3.size == refined.size else v3.resize(refined.size, Image.Resampling.LANCZOS)
            columns.append(("v3", shown))
            entry["v3"] = scale_metrics(source, shown, policy)
            entry["meso_vs_v3"] = round(meso_agreement(refined, shown, source.size, policy), 4)
        choice = (registry.job(run_id, key).get("detail") or {}).get("albedo_choice")
        if choice == "linked":
            columns.append(("linked", refined))
        elif choice == "regen":
            columns.append(("regen", refined))
        else:
            columns.append(("restrained", refined))
        regen_path = directory / "albedo_regen.png"
        if choice != "regen" and regen_path.is_file():
            regen = load_png(regen_path).convert("RGB")
            columns.append(("regen", regen if regen.size == refined.size else regen.resize(refined.size, Image.Resampling.LANCZOS)))
            entry["regen"] = scale_metrics(source, columns[-1][1], policy)
        entry["albedo_choice"] = choice
        report[key] = entry
        micro = entry["refined"]["micro_rms"]
        write_distance_sheet(
            columns, gallery / f"{key.replace(':', '_')}-scale.png",
            f"{key}  micro {micro:.2f}  choice {choice}",
        )
    if report:
        (gallery / "scale-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        log.info("scale sheet %s keys", len(report))

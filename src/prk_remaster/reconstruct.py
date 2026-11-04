"""Semantic reconstruction of the whole cohort. Does not read or write v1 or v2 albedos."""

from __future__ import annotations

import json
import logging
import os
import signal
import shutil
from pathlib import Path

from PIL import Image

from .config import REPO_ROOT, ensure_work_dirs
from .detail import lock_coarse_colour, structure_reference
from .gallery import PRIORITY_KEYS, write_crop_strip
from .imaging import atomic_save_png, load_png
from .pack import build_pack
from .pbr import build_orm, blend_detail_normal, detail_weight, policy_for, stamp_detail_height
from .planner import cache_key, reduced_long_side
from .quality import check_albedo, native_fidelity
from .registry import Registry
from .runner import (
    _asset_dir,
    _call_with_vram,
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
from .semantics import apply_v3_detail, build_brief, compile_prompt
from .tiles import circular_shift, repair_seam

log = logging.getLogger("prk.reconstruct")
LOCK_VERSION = 2
V1_DONE = {"qa_passed", "packed"}
STAGES = ("generate", "pbr", "pack")
_PAUSE = False


def find_reconstruct_run(registry: Registry, from_run: str) -> str | None:
    rows = registry.conn.execute(
        "SELECT run_id, detail FROM runs ORDER BY created_at DESC"
    ).fetchall()
    for row in rows:
        detail = json.loads(row["detail"] or "{}")
        if detail.get("kind") == "reconstruct" and detail.get("from_run") == from_run:
            return row["run_id"]
    return None


def _request_pause(signum, _frame) -> None:
    global _PAUSE
    _PAUSE = True
    log.info("pause requested (signal %s); finishing the current texture", signum)


def _link(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        return
    try:
        os.link(src, dest)
    except OSError:
        shutil.copy2(src, dest)


def _ordered(keys: list[str]) -> list[str]:
    rank = {key: index for index, key in enumerate(PRIORITY_KEYS)}
    return sorted(keys, key=lambda key: (rank.get(key, 10_000), key))


def _resume_text(from_run: str, run_id: str) -> str:
    return (
        f"cd {REPO_ROOT}\n"
        "source .venv/bin/activate\n"
        f"prk-remaster reconstruct --from-run {from_run} --run {run_id} --resume\n"
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


def _repair(image: Image.Image, asset: dict, cfg: dict) -> Image.Image:
    reasons = check_albedo(
        image, target_w=image.size[0], target_h=image.size[1],
        source_seam=float(asset.get("seam") or 99), tileable=True, cfg=cfg,
    )
    if any(reason.startswith("periodic seam") for reason in reasons):
        repaired = repair_seam(image, band=int(cfg.get("seam_retry_band", 8)))
        again = check_albedo(
            repaired, target_w=image.size[0], target_h=image.size[1],
            source_seam=float(asset.get("seam") or 99), tileable=True, cfg=cfg,
        )
        if not any(reason.startswith("periodic seam") for reason in again):
            return repaired
    return image


def _edit_once(edit_fn, reference: Image.Image, prompt: str, seed: int, settings: dict, stats: dict):
    shifted = circular_shift(reference, reference.size[0] // 2, reference.size[1] // 2)
    edited, elapsed, peak = _call_with_vram(lambda: edit_fn(shifted, prompt, seed, settings), True)
    stats["gpu_seconds"] = float(stats.get("gpu_seconds") or 0) + elapsed
    if peak:
        stats["peak_vram_mib"] = max(float(stats.get("peak_vram_mib") or 0), peak)
    edited = edited.convert("RGB")
    return circular_shift(edited, -(edited.size[0] // 2), -(edited.size[1] // 2)), elapsed, peak


def run_reconstruct(
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
) -> dict:
    if stop_after not in STAGES:
        raise ValueError(f"unsupported stop_after {stop_after}")
    recipe = cfg.get("reconstruct") or {}
    pack_name = pack_name or recipe.get("pack_name") or "prk-hd-v03"
    paths = ensure_work_dirs(cfg)
    from_dir = paths["runs"] / from_run
    if not from_dir.is_dir():
        raise SystemExit(f"source run not found: {from_dir}")
    if resume and run_id is None:
        run_id = find_reconstruct_run(registry, from_run)
        if run_id is None:
            raise SystemExit(f"no reconstruct run to resume from {from_run}")
    if run_id is None:
        run_id = new_run_id()
    if run_id == from_run:
        raise SystemExit("reconstruct refuses to write into the source run")
    run_dir = paths["runs"] / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    if from_dir.resolve() == run_dir.resolve() or from_dir.resolve() in run_dir.resolve().parents:
        raise SystemExit("reconstruct refuses to write inside the source run")
    stop = run_dir / "STOP"
    if resume and stop.exists():
        stop.unlink()
    _write_resume(run_dir, from_run, run_id)
    record = {
        "config_revision": recipe.get("config_revision", "v0.3.0"),
        "prompt_version": recipe.get("prompt_version", "v3"),
    }
    if registry.run(run_id) is None:
        registry.create_run(run_id, "first-1076", "semantic", record)
    registry.update_run(run_id, status="running", detail={
        "kind": "reconstruct", "from_run": from_run, "prompt_version": record["prompt_version"],
        "config_revision": record["config_revision"],
    })
    registry.events_path = run_dir / "events.jsonl"
    selected = _select(registry, from_run, keys)
    _ensure_jobs(registry, run_id, from_run, from_dir, run_dir, selected, record)
    stats = _load_stats(run_dir)
    global _PAUSE
    _PAUSE = False
    previous_term = signal.getsignal(signal.SIGTERM)
    previous_int = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGTERM, _request_pause)
    signal.signal(signal.SIGINT, _request_pause)
    try:
        paused = _generate(registry, cfg, run_id, run_dir, from_dir, record, stats, force, limit, keys)
        _save_stats(run_dir, stats)
        _write_crops(registry, run_id, run_dir, from_dir)
        if paused or _PAUSE or stop.exists():
            _mark_paused(registry, run_id, from_run, "generate")
            return {"run_id": run_id, "stage": "paused", "resume": _resume_text(from_run, run_id).strip()}
        if stop_after == "generate":
            registry.update_run(run_id, status="generated", detail=_detail(from_run, record, "generate"))
            return {"run_id": run_id, "stage": "generate", "jobs": len(registry.jobs(run_id))}
        paused = _pbr(registry, cfg, run_id, run_dir, stats, force)
        _save_stats(run_dir, stats)
        _write_crops(registry, run_id, run_dir, from_dir)
        if paused or _PAUSE or stop.exists():
            _mark_paused(registry, run_id, from_run, "pbr")
            return {"run_id": run_id, "stage": "paused", "resume": _resume_text(from_run, run_id).strip()}
        if stop_after == "pbr":
            registry.update_run(run_id, status="pbr", detail=_detail(from_run, record, "pbr"))
            return {"run_id": run_id, "stage": "pbr", "jobs": len(registry.jobs(run_id))}
        dest = paths["dist"] / pack_name
        manifest = build_pack(registry, run_id, dest, pack_name, archive=False)
        registry.update_run(run_id, status="packed", detail={
            **_detail(from_run, record, "pack"), "pack": str(dest),
            "gpu_seconds": round(float(stats.get("gpu_seconds") or 0), 1),
        })
        return {
            "run_id": run_id, "stage": "pack", "pack": manifest.get("pack_dir"),
            "assets": len(manifest.get("assets") or []),
        }
    finally:
        signal.signal(signal.SIGTERM, previous_term)
        signal.signal(signal.SIGINT, previous_int)


def _detail(from_run: str, record: dict, stage: str) -> dict:
    return {
        "kind": "reconstruct", "from_run": from_run, "stage": stage,
        "prompt_version": record["prompt_version"], "config_revision": record["config_revision"],
    }


def _mark_paused(registry, run_id, from_run, stage) -> None:
    registry.update_run(run_id, status="paused", detail={"kind": "reconstruct", "from_run": from_run, "stage": stage})
    log.info("paused during %s", stage)


def _select(registry: Registry, from_run: str, keys: set[str] | None) -> dict[str, dict]:
    chosen = {}
    for job in registry.jobs(from_run):
        if job["status"] not in V1_DONE:
            continue
        if keys is not None and job["asset_key"] not in keys:
            continue
        chosen[job["asset_key"]] = job
    return chosen


def _ensure_jobs(registry, run_id, from_run, from_dir, run_dir, selected, record) -> None:
    for key, v1_job in selected.items():
        asset = registry.asset(key)
        if asset is None:
            continue
        canonical = v1_job.get("canonical_key")
        if canonical not in selected:
            canonical = None
        existing = registry.job(run_id, key)
        if canonical:
            if existing is None:
                registry.upsert_job({
                    "run_id": run_id, "asset_key": key, "status": "extracted", "stage": "alias",
                    "seed": v1_job.get("seed"), "prompt": "", "cache_key": v1_job.get("cache_key"),
                    "target_w": v1_job.get("target_w"), "target_h": v1_job.get("target_h"),
                    "attempt": 0, "rejection_reason": None, "canonical_key": canonical,
                    "detail": {"from_run": from_run, "alias": True, "prompt_version": record["prompt_version"]},
                })
            continue
        source = from_dir / "assets" / key.replace(":", "_") / "source.png"
        if not source.is_file():
            registry.upsert_job({
                "run_id": run_id, "asset_key": key, "status": "blocked", "stage": "link",
                "seed": v1_job.get("seed"), "prompt": "", "cache_key": "",
                "target_w": 0, "target_h": 0, "attempt": 0,
                "rejection_reason": "v1 source is missing", "canonical_key": None,
                "detail": {"from_run": from_run},
            })
            continue
        directory = _asset_dir(run_dir, key)
        _link(source, directory / "source.png")
        if existing and existing["status"] not in {"blocked", "failed"}:
            continue
        registry.upsert_job({
            "run_id": run_id, "asset_key": key, "status": "extracted", "stage": "link",
            "seed": v1_job.get("seed") or 1, "prompt": "",
            "cache_key": cache_key(
                asset.get("source_sha256") or key, "reconstruct",
                record["prompt_version"], record["config_revision"], "generate",
            ),
            "target_w": int(v1_job.get("target_w") or 0),
            "target_h": int(v1_job.get("target_h") or 0),
            "attempt": 0, "rejection_reason": None, "canonical_key": None,
            "detail": {"from_run": from_run, "prompt_version": record["prompt_version"], "generated": False},
        })
        registry.log_event(run_id, key, "reconstruct_linked")


def _should_pause(run_dir: Path) -> bool:
    return _PAUSE or (run_dir / "STOP").exists()


def _generate(registry, cfg, run_id, run_dir, from_dir, record, stats, force: bool, limit: int | None, keys: set[str] | None) -> bool:
    pending = []
    for job in registry.jobs(run_id):
        if job.get("canonical_key") or job["status"] in {"blocked", "rejected"}:
            continue
        if keys is not None and job["asset_key"] not in keys:
            continue
        directory = _asset_dir(run_dir, job["asset_key"])
        detail = job.get("detail") or {}
        done = (
            detail.get("generated")
            and detail.get("lock_version") == LOCK_VERSION
            and (directory / "albedo_final.png").is_file()
        )
        if done and not force:
            for name in ("raw_qwen.png", "structure.png"):
                extra = directory / name
                if extra.is_file():
                    extra.unlink()
        if done and not force:
            continue
        pending.append(job)
    pending = _ordered([job["asset_key"] for job in pending])
    by_key = {job["asset_key"]: job for job in registry.jobs(run_id)}
    if limit is not None:
        pending = pending[:limit]
    if not pending:
        log.info("generate queue is empty")
        return False
    log.info("generate queue %s", len(pending))
    edit_fn, free_fn = _default_edit(cfg)
    try:
        free_fn()
    except Exception:
        log.info("comfy free before generate did not complete")
    settings = _settings(cfg)
    for index, key in enumerate(pending, start=1):
        if _should_pause(run_dir):
            log.info("pausing before %s", key)
            return True
        job = registry.job(run_id, key) or by_key[key]
        try:
            _generate_one(registry, cfg, run_id, run_dir, job, edit_fn, settings, stats, force, record)
        except Exception as exc:  # noqa: BLE001
            registry.update_job(run_id, key, status="failed", rejection_reason=str(exc)[:500], stage="generate")
            registry.log_event(run_id, key, "generate_error", error=str(exc)[:500])
            log.info("generate failed %s: %s", key, str(exc)[:200])
        if index % 10 == 0:
            _save_stats(run_dir, stats)
        log.info("generate progress %s/%s", index, len(pending))
    try:
        free_fn()
    except Exception:
        log.info("comfy free after generate did not complete")
    return False


def _generate_one(registry, cfg, run_id, run_dir, job, edit_fn, settings, stats, force, record) -> None:
    key = job["asset_key"]
    directory = _asset_dir(run_dir, key)
    asset = registry.asset(key) or {}
    source = load_png(directory / "source.png").convert("RGB")
    brief = apply_v3_detail(
        build_brief(asset, source, cfg.get("name_lexicon") or {}),
        cfg.get("family_prompts") or {},
        cfg.get("scale_policy") or {},
    )
    target = (int(job.get("target_w") or 0), int(job.get("target_h") or 0))
    if min(target) < 32:
        target = source.size
    prompt = compile_prompt(
        asset, brief, cfg.get("family_prompts") or {}, target[0], target[1], version=record["prompt_version"],
    )
    (directory / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")
    raw_path = directory / "raw_qwen.png"
    elapsed = 0.0
    peak = None
    label = (job.get("detail") or {}).get("generation_label") or ""
    if force or not raw_path.is_file():
        reference = structure_reference(
            source, target, kind=brief.get("kind") or "organic", subject_class=brief.get("subject_class") or "",
            scale_mode=brief.get("scale_mode") or "",
        )
        try:
            edited, elapsed, peak = _edit_once(edit_fn, reference, prompt, int(job.get("seed") or 1), settings, stats)
        except Exception as exc:
            if not is_oom(exc) or max(target) <= 1024:
                raise
            smaller = reduced_long_side(target[0], target[1], 1024, int(cfg["material_rules"].get("snap", 32)))
            reference = structure_reference(
                source, smaller, kind=brief.get("kind") or "organic", subject_class=brief.get("subject_class") or "",
                scale_mode=brief.get("scale_mode") or "",
            )
            edited, elapsed, peak = _edit_once(edit_fn, reference, prompt, int(job.get("seed") or 1), settings, stats)
            label = "qwen-1024"
            target = smaller
        edited = edited.convert("RGB")
        if not label:
            label = "qwen-native" if max(edited.size) >= 2048 else f"qwen-{max(edited.size)}"
        atomic_save_png(edited, raw_path)
        _record(registry, run_id, key, "raw_qwen", raw_path)
        _save_stats(run_dir, stats)
    edited = load_png(raw_path).convert("RGB")
    scale = max(edited.size) / max(max(source.size), 1)
    locked = lock_coarse_colour(edited, source, scale=scale)
    locked = _repair(locked, asset, cfg)
    albedo = directory / "albedo_final.png"
    atomic_save_png(locked, albedo)
    _record(registry, run_id, key, "source", directory / "source.png")
    _record(registry, run_id, key, "albedo_final", albedo)
    metrics = native_fidelity(source, locked)
    detail = dict((registry.job(run_id, key) or job).get("detail") or {})
    detail.update({
        "generated": True,
        "generation_label": label or detail.get("generation_label") or "qwen-native",
        "prompt_version": record["prompt_version"],
        "kind": brief.get("kind"),
        "subject": brief.get("subject"),
        "subject_class": brief.get("subject_class"),
        "effective_material": brief.get("effective_material"),
        "material_conflict": brief.get("material_conflict"),
        "generate_seconds": round(elapsed, 3) if elapsed else detail.get("generate_seconds"),
        "fidelity": metrics,
        "review": bool(metrics["structure"] < 0.45),
        "detail_stamped": False,
        "lock_version": LOCK_VERSION,
    })
    if peak:
        detail["vram_mib"] = peak
    registry.update_job(
        run_id, key, status="generated", stage="generate", prompt=prompt,
        target_w=locked.size[0], target_h=locked.size[1], detail=detail, rejection_reason=None,
    )
    registry.log_event(run_id, key, "reconstructed", **metrics)
    # The data volume cannot hold a second raw 2K copy of the cohort.
    for name in ("raw_qwen.png", "structure.png"):
        extra = directory / name
        if extra.is_file():
            extra.unlink()
    log.info(
        "reconstructed %s gain %.2f structure %.3f coarse_dE %.2f",
        key, metrics["detail_gain"], metrics["structure"], metrics["coarse_delta_e"],
    )


def _pbr(registry, cfg, run_id, run_dir, stats, force: bool) -> bool:
    pbr_fn, free_fn = _default_pbr(cfg)
    try:
        free_fn()
    except Exception:
        log.info("comfy free before pbr did not complete")
    jobs = _ordered([
        job["asset_key"] for job in registry.jobs(run_id)
        if not job.get("canonical_key") and job["status"] in {"generated", "pbr"}
    ])
    for index, key in enumerate(jobs, start=1):
        if _should_pause(run_dir):
            log.info("pausing before pbr %s", key)
            return True
        job = registry.job(run_id, key)
        if job is None:
            continue
        if job["status"] == "generated":
            _pbr_stage(registry, cfg, run_id, run_dir, job, pbr_fn, stats, True, free_fn)
        current = registry.job(run_id, key)
        if current and current["status"] == "pbr":
            _stamp(registry, cfg, run_id, run_dir, current, force)
            current = registry.job(run_id, key)
        if current and current["status"] == "pbr":
            _qa(registry, cfg, run_id, run_dir, current)
            for name in ("depth_raw.png", "preview_relit.png"):
                extra = _asset_dir(run_dir, key) / name
                if extra.is_file():
                    extra.unlink()
        if index % 20 == 0:
            _save_stats(run_dir, stats)
            log.info("pbr progress %s/%s", index, len(jobs))
    _mirror_aliases(registry, run_id)
    try:
        free_fn()
    except Exception:
        log.info("comfy free after pbr did not complete")
    return False


def _stamp(registry, cfg, run_id, run_dir, job, force: bool) -> None:
    key = job["asset_key"]
    detail = dict(job.get("detail") or {})
    if detail.get("detail_stamped") and not force:
        return
    directory = _asset_dir(run_dir, key)
    albedo = load_png(directory / "albedo_final.png").convert("RGB")
    normal = load_png(directory / "normal.png")
    height = load_png(directory / "height.png")
    weight = detail_weight(detail.get("subject_class") or "", detail.get("kind") or "")
    detailed = blend_detail_normal(normal, albedo, weight)
    stamped = stamp_detail_height(height, albedo, weight)
    asset = registry.asset(key) or {}
    category = detail.get("effective_material") or asset.get("material_class") or ""
    policy = policy_for(asset.get("name") or "", category, cfg["material_rules"])
    orm = build_orm(albedo, stamped, policy, cfg["material_rules"])
    atomic_save_png(detailed, directory / "normal.png")
    atomic_save_png(stamped, directory / "height.png")
    atomic_save_png(orm, directory / "orm.png")
    _record(registry, run_id, key, "normal", directory / "normal.png")
    _record(registry, run_id, key, "height", directory / "height.png")
    _record(registry, run_id, key, "orm", directory / "orm.png")
    detail["detail_stamped"] = True
    detail["detail_weight"] = weight
    detail["pbr_label"] = "pbr-1024-plus-detail"
    registry.update_job(run_id, key, detail=detail)
    registry.log_event(run_id, key, "detail_stamped", weight=weight)


def _v2_albedo(from_dir: Path, registry: Registry, from_run: str, key: str) -> Path | None:
    # Comparison only. The v2 salvage run is never written.
    rows = registry.conn.execute("SELECT run_id, detail FROM runs ORDER BY created_at DESC").fetchall()
    for row in rows:
        payload = json.loads(row["detail"] or "{}")
        if payload.get("kind") == "salvage" and payload.get("from_run") == from_run:
            path = from_dir.parent / row["run_id"] / "assets" / key.replace(":", "_") / "albedo_final.png"
            if path.is_file():
                return path
    return None


def _crop_box(size: tuple[int, int], span: int = 512) -> tuple[int, int, int, int]:
    width, height = size
    side = min(span, width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return left, top, left + side, top + side


def _write_crops(registry, run_id, run_dir, from_dir) -> None:
    dest = run_dir / "gallery" / "crops"
    run = registry.run(run_id) or {}
    from_run = (run.get("detail") or {}).get("from_run") or ""
    for key in PRIORITY_KEYS:
        job = registry.job(run_id, key)
        if job is None or job.get("canonical_key"):
            continue
        directory = _asset_dir(run_dir, key)
        albedo_path = directory / "albedo_final.png"
        source_path = directory / "source.png"
        if not albedo_path.is_file() or not source_path.is_file():
            continue
        albedo = load_png(albedo_path).convert("RGB")
        source = load_png(source_path).convert("RGB").resize(albedo.size, Image.Resampling.NEAREST)
        box = _crop_box(albedo.size)
        asset = registry.asset(key) or {}
        metrics = (job.get("detail") or {}).get("fidelity") or {}
        caption = (
            f"{key}  {asset.get('name') or ''}  "
            f"gain {metrics.get('detail_gain', '')}  struct {metrics.get('structure', '')}  "
            f"coarse dE {metrics.get('coarse_delta_e', '')}"
        )
        panels = [("source", source.crop(box)), ("v3", albedo.crop(box))]
        v2 = _v2_albedo(from_dir, registry, from_run, key) if from_run else None
        if v2 is not None:
            panels.insert(1, ("v2", load_png(v2).convert("RGB").resize(albedo.size, Image.Resampling.LANCZOS).crop(box)))
        normal_path = directory / "normal.png"
        if normal_path.is_file():
            panels.append(("normal", load_png(normal_path).convert("RGB").crop(box)))
        write_crop_strip(panels, dest / f"{key.replace(':', '_')}.png", caption)

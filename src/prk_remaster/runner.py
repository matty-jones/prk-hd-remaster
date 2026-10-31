"""Resumable cohort execution. One GPU stage at a time."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from pathlib import Path

from PIL import Image

from . import pbr, qwen, rdb
from .comfy_client import ComfyClient
from .imaging import atomic_save_png, load_png, pixel_sha256, sha256_bytes, sha256_file
from .planner import derive_seed, link_identical_pixels, plan_jobs, reduced_long_side
from .quality import check_albedo, check_height, check_normal, check_orm, colour_shift
from .registry import Registry
from .tiles import circular_shift, preview_3x3, relight, repair_seam

log = logging.getLogger("prk_remaster")

NEXT = {
    "pending": "extract",
    "extracted": "generate",
    "generated": "pbr",
    "pbr": "qa",
}


def new_run_id() -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{uuid.uuid4().hex[:6]}"


def is_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or ("cuda" in text and "memory" in text)


def sample_vram() -> float | None:
    import subprocess
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
        )
        return float(output.strip().splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def _call_with_vram(fn, measure: bool):
    if not measure:
        started = time.perf_counter()
        result = fn()
        return result, time.perf_counter() - started, None
    peaks: list[float] = []
    stop = threading.Event()

    def poll() -> None:
        while not stop.is_set():
            value = sample_vram()
            if value is not None:
                peaks.append(value)
            stop.wait(1.0)

    thread = threading.Thread(target=poll, daemon=True)
    thread.start()
    started = time.perf_counter()
    try:
        result = fn()
    finally:
        stop.set()
        thread.join(timeout=2)
    peak = max(peaks) if peaks else sample_vram()
    return result, time.perf_counter() - started, peak


def _asset_dir(run_dir: Path, key: str) -> Path:
    path = run_dir / "assets" / key.replace(":", "_")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _record(registry: Registry, run_id: str, key: str, role: str, path: Path) -> None:
    registry.add_output(run_id, key, role, path, sha256_file(path), path.stat().st_size)


def _settings(cfg: dict) -> dict:
    return {
        "cache_device": cfg.get("cache_device", "cpu"),
        "cache_dtype": cfg.get("cache_dtype", "int8"),
        "qwen_resolution": cfg.get("qwen_resolution", 0),
        "pbr_optimize": cfg.get("pbr_optimize", False),
        "pbr_intensity_blend": cfg.get("pbr_intensity_blend", 0.2),
        "pbr_bilateral_d": cfg.get("pbr_bilateral_d", 10),
        "pbr_bilateral_sigma_color": cfg.get("pbr_bilateral_sigma_color", 50),
        "pbr_bilateral_sigma_space": cfg.get("pbr_bilateral_sigma_space", 50),
    }


def _default_edit(cfg):
    client = ComfyClient(cfg["comfy_url"], timeout=float(cfg.get("comfy_timeout_seconds", 1800)))

    def edit(image, prompt, seed, settings):
        return qwen.edit_image(client, image, prompt, seed, settings, cfg["model_lock"])

    return edit, client.free


def _default_pbr(cfg):
    client = ComfyClient(cfg["comfy_url"], timeout=float(cfg.get("comfy_timeout_seconds", 1800)))

    def estimate(albedo, settings):
        from . import pbr_worker
        return pbr_worker.estimate(client, albedo, settings)

    return estimate, client.free


def run_cohort(
    cfg: dict,
    registry: Registry,
    run_dir: Path,
    *,
    cohort: str,
    preset: str,
    run_id: str | None = None,
    resume: bool = False,
    limit: int | None = None,
    family: str | None = None,
    types: list[int] | None = None,
    ids: set[str] | None = None,
    force: bool = False,
    retry_failed: bool = False,
    dry_run: bool = False,
    edit_fn=None,
    pbr_fn=None,
    free_fn=None,
    measure_vram: bool = False,
) -> dict:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if resume and run_id is None:
        run_id = registry.latest_run_id(cohort)
    if run_id is None:
        run_id = new_run_id()
    if registry.run(run_id) is None:
        registry.create_run(run_id, cohort, preset, cfg)
    registry.events_path = run_dir / "events.jsonl"
    jobs = plan_jobs(
        registry, cfg, run_id,
        cohort=cohort, family=family, types=types, ids=ids,
        limit=limit, force=force, retry_failed=retry_failed,
    )
    if dry_run:
        return {"run_id": run_id, "jobs": jobs, "dry_run": True}
    registry.update_run(run_id, status="running")
    if edit_fn is None:
        edit_fn, default_free = _default_edit(cfg)
        free_fn = free_fn or default_free
    if pbr_fn is None:
        pbr_fn, pbr_free = _default_pbr(cfg)
        # Free after each stage. The PBR client is a second handle on the same server.
        previous_free = free_fn
        def free_fn():
            if previous_free:
                previous_free()
            pbr_free()
    db_path = Path(cfg["prk_root"]) / "cd_image" / "rdb.db"
    connection = rdb.connect(db_path)
    stats = {"gpu_seconds": 0.0, "peak_vram_mib": 0.0, "wall_start": time.perf_counter()}
    try:
        for job in registry.jobs(run_id):
            if job["status"] == "pending":
                _extract(connection, registry, cfg, run_id, run_dir, job)
        link_identical_pixels(registry, run_id)
        for job in registry.jobs(run_id):
            if job["status"] == "extracted" and not job.get("canonical_key"):
                _generate(registry, cfg, run_id, run_dir, job, edit_fn, stats, measure_vram, force)
        if free_fn:
            free_fn()
        for job in registry.jobs(run_id):
            if job["status"] == "generated" and not job.get("canonical_key"):
                _pbr_stage(registry, cfg, run_id, run_dir, job, pbr_fn, stats, measure_vram, free_fn)
        if free_fn:
            free_fn()
        for job in registry.jobs(run_id):
            if job["status"] == "pbr" and not job.get("canonical_key"):
                _qa(registry, cfg, run_id, run_dir, job)
        _mirror_aliases(registry, run_id)
    finally:
        connection.close()
    wall = time.perf_counter() - stats["wall_start"]
    detail = {
        "wall_seconds": round(wall, 3),
        "gpu_seconds": round(stats["gpu_seconds"], 3),
        "peak_vram_mib": stats["peak_vram_mib"] or None,
        "preset": preset,
        "cohort": cohort,
    }
    summary = registry.summary(run_id)
    terminal = {"packed", "rejected", "blocked", "failed", "qa_passed"}
    done = summary["total"] and all(status in terminal for status in summary["counts"])
    registry.update_run(run_id, status="complete" if done else "partial", detail=detail)
    return {"run_id": run_id, "summary": summary, "detail": detail}


def _extract(connection, registry, cfg, run_id, run_dir, job) -> None:
    key = job["asset_key"]
    registry.update_job(run_id, key, stage="extract")
    registry.log_event(run_id, key, "extract_start")
    log.info("extract %s", key)
    asset = registry.asset(key)
    try:
        record = rdb.extract(connection, asset["rdb_type"], asset["rdb_id"])
    except Exception as exc:  # noqa: BLE001
        registry.update_job(run_id, key, status="blocked", rejection_reason=f"decode failed: {exc}", stage="extract")
        registry.log_event(run_id, key, "extract_blocked", error=str(exc))
        return
    image = record["image"]
    directory = _asset_dir(run_dir, key)
    source = directory / "source.png"
    atomic_save_png(image.convert("RGB"), source)
    source_sha = sha256_bytes(record["payload"])
    pixels = pixel_sha256(image)
    registry.update_asset_hashes(
        key, source_sha, pixels, record["width"], record["height"], record["source_format"],
    )
    _record(registry, run_id, key, "source", source)
    model = cfg["model_lock"]["qwen"]["files"]["diffusion"]["filename"]
    seed = derive_seed(model, cfg.get("config_revision"), asset["rdb_type"], asset["rdb_id"], source_sha)
    from .planner import cache_key
    key_hash = cache_key(
        source_sha, model, cfg.get("config_revision"), cfg.get("prompt_version"),
        seed, "generate", cfg.get("stage_version"),
    )
    prompt_path = directory / "prompt.txt"
    prompt_path.write_text(job.get("prompt") or "", encoding="utf-8")
    registry.update_job(
        run_id, key, status="extracted", stage="extract", seed=seed, cache_key=key_hash,
    )
    registry.log_event(run_id, key, "extracted", sha256=source_sha)


def _generate(registry, cfg, run_id, run_dir, job, edit_fn, stats, measure_vram, force) -> None:
    key = job["asset_key"]
    directory = _asset_dir(run_dir, key)
    cache_dir = Path(cfg["work_root"]) / "cache" / (job.get("cache_key") or "none")
    cached = cache_dir / "albedo_final.png"
    if cached.is_file() and not force:
        raw = directory / "raw_qwen.png"
        albedo = directory / "albedo_final.png"
        atomic_save_png(load_png(cached), raw)
        atomic_save_png(load_png(cached), albedo)
        _record(registry, run_id, key, "raw_qwen", raw)
        _record(registry, run_id, key, "albedo_final", albedo)
        detail = dict(job.get("detail") or {})
        detail["cache_hit"] = True
        label_file = cache_dir / "generation_label.txt"
        stored = label_file.read_text(encoding="utf-8").strip() if label_file.is_file() else ""
        detail["generation_label"] = stored or detail.get("generation_label") or "qwen-cache"
        registry.update_job(run_id, key, status="generated", stage="generate", detail=detail)
        registry.log_event(run_id, key, "generate_cache_hit")
        return
    source = load_png(directory / "source.png").convert("RGB")
    for round_index in range(2):
        job = registry.job(run_id, key)
        target = (int(job["target_w"]), int(job["target_h"]))
        if min(target) < 1:
            registry.update_job(run_id, key, status="blocked", rejection_reason="target size is empty")
            return
        conditioning = source.resize(target, Image.Resampling.LANCZOS)
        shifted = circular_shift(conditioning, target[0] // 2, target[1] // 2)
        registry.update_job(run_id, key, stage="generate", attempt=int(job.get("attempt") or 0) + (1 if round_index else 0))
        try:
            edited, elapsed, peak = _call_with_vram(
                lambda: edit_fn(shifted, job.get("prompt") or "", int(job["seed"]), _settings(cfg)),
                measure_vram,
            )
        except Exception as exc:  # noqa: BLE001
            if is_oom(exc) and round_index == 0 and max(target) > 1024:
                smaller = reduced_long_side(target[0], target[1], 1024, int(cfg["material_rules"].get("snap", 32)))
                detail = dict(job.get("detail") or {})
                detail["generation_label"] = "qwen-1024"
                detail["oom"] = str(exc)[:300]
                registry.update_job(
                    run_id, key, target_w=smaller[0], target_h=smaller[1], detail=detail,
                    attempt=int(job.get("attempt") or 0) + 1, stage="generate",
                )
                registry.log_event(run_id, key, "generate_oom_retry", error=str(exc)[:300])
                continue
            status = "blocked" if is_oom(exc) else "failed"
            registry.update_job(
                run_id, key, status=status, rejection_reason=str(exc)[:500],
                attempt=int(job.get("attempt") or 0) + 1,
            )
            registry.log_event(run_id, key, "generate_error", error=str(exc)[:500])
            return
        stats["gpu_seconds"] += elapsed
        if peak:
            stats["peak_vram_mib"] = max(stats["peak_vram_mib"], peak)
        edited = edited.convert("RGB")
        label = (job.get("detail") or {}).get("generation_label")
        if not label:
            label = "qwen-native" if max(edited.size) >= 2048 else f"qwen-{max(edited.size)}"
        final = circular_shift(edited, -(edited.size[0] // 2), -(edited.size[1] // 2))
        if final.size != target:
            # Keep the model's actual frame. Do not bilinear-enlarge it and call that native.
            registry.update_job(run_id, key, target_w=final.size[0], target_h=final.size[1])
            target = final.size
        raw_path = directory / "raw_qwen.png"
        atomic_save_png(final, raw_path)
        _record(registry, run_id, key, "raw_qwen", raw_path)
        asset = registry.asset(key)
        reasons = check_albedo(
            final, target_w=target[0], target_h=target[1],
            source_seam=float(asset.get("seam") or 99), tileable=True, cfg=cfg,
        )
        seam_reasons = [reason for reason in reasons if reason.startswith("periodic seam")]
        if seam_reasons:
            repaired = repair_seam(final, band=int(cfg.get("seam_retry_band", 8)))
            repaired_reasons = check_albedo(
                repaired, target_w=target[0], target_h=target[1],
                source_seam=float(asset.get("seam") or 99), tileable=True, cfg=cfg,
            )
            registry.log_event(run_id, key, "seam_retry", before=seam_reasons, after=repaired_reasons)
            if not any(item.startswith("periodic seam") for item in repaired_reasons):
                final = repaired
                reasons = [item for item in repaired_reasons if not item.startswith("periodic seam")]
            else:
                reasons = repaired_reasons
        if any(item.startswith("albedo is empty") or item.startswith("albedo size") for item in reasons):
            registry.update_job(run_id, key, status="rejected", rejection_reason="; ".join(reasons))
            return
        if any(item.startswith("periodic seam") for item in reasons):
            registry.update_job(run_id, key, status="rejected", rejection_reason="; ".join(reasons))
            registry.log_event(run_id, key, "rejected", reason="; ".join(reasons))
            return
        albedo_path = directory / "albedo_final.png"
        atomic_save_png(final, albedo_path)
        _record(registry, run_id, key, "albedo_final", albedo_path)
        cache_dir.mkdir(parents=True, exist_ok=True)
        atomic_save_png(final, cached)
        (cache_dir / "generation_label.txt").write_text(label + "\n", encoding="utf-8")
        detail = dict(registry.job(run_id, key).get("detail") or {})
        detail["generation_label"] = label
        detail["generate_seconds"] = round(elapsed, 3)
        detail["colour_shift"] = colour_shift(source, final)
        if peak:
            detail["vram_mib"] = peak
        registry.update_job(run_id, key, status="generated", stage="generate", detail=detail)
        registry.log_event(run_id, key, "generated", label=label, seconds=round(elapsed, 3))
        log.info("generated %s %s in %.1fs", key, label, elapsed)
        return


def _pbr_stage(registry, cfg, run_id, run_dir, job, pbr_fn, stats, measure_vram, free_fn) -> None:
    key = job["asset_key"]
    directory = _asset_dir(run_dir, key)
    albedo = load_png(directory / "albedo_final.png").convert("RGB")
    shifted = circular_shift(albedo, albedo.size[0] // 2, albedo.size[1] // 2)
    registry.update_job(run_id, key, stage="pbr")
    settings = _settings(cfg)
    if stats.get("pbr_force_optimize"):
        settings["pbr_optimize"] = True
    normal = depth = None
    elapsed = 0.0
    peak = None
    for attempt in range(2):
        try:
            (normal, depth), elapsed, peak = _call_with_vram(
                lambda: pbr_fn(shifted, settings),
                measure_vram,
            )
            break
        except Exception as exc:  # noqa: BLE001
            if is_oom(exc) and attempt == 0 and not settings.get("pbr_optimize"):
                settings = dict(settings)
                settings["pbr_optimize"] = True
                stats["pbr_force_optimize"] = True
                registry.log_event(run_id, key, "pbr_oom_retry", error=str(exc)[:300])
                log.info("pbr oom %s; retrying at 1024", key)
                if free_fn:
                    free_fn()
                continue
            status = "blocked" if is_oom(exc) else "failed"
            registry.update_job(run_id, key, status=status, rejection_reason=f"pbr: {exc}"[:500])
            registry.log_event(run_id, key, "pbr_error", error=str(exc)[:500])
            return
    assert normal is not None and depth is not None
    stats["gpu_seconds"] += elapsed
    if peak:
        stats["peak_vram_mib"] = max(stats["peak_vram_mib"], peak)
    normal = circular_shift(normal.convert("RGB"), -(normal.size[0] // 2), -(normal.size[1] // 2))
    depth = circular_shift(depth.convert("L"), -(depth.size[0] // 2), -(depth.size[1] // 2))
    asset = registry.asset(key)
    maps = pbr.derive_maps(
        albedo, normal, depth,
        name=asset.get("name") or "",
        category=asset.get("material_class") or "",
        rules=cfg["material_rules"],
        invert_height=bool(cfg.get("height_invert", False)),
    )
    normal_path = directory / "normal.png"
    depth_path = directory / "depth_raw.png"
    height_path = directory / "height.png"
    orm_path = directory / "orm.png"
    atomic_save_png(maps["normal"], normal_path)
    atomic_save_png(maps["height"], height_path)
    atomic_save_png(maps["orm"], orm_path)
    atomic_save_png(depth.convert("L"), depth_path)
    for role, path in (
        ("normal", normal_path), ("depth_raw", depth_path),
        ("height", height_path), ("orm", orm_path),
    ):
        _record(registry, run_id, key, role, path)
    detail = dict(job.get("detail") or {})
    detail["pbr_seconds"] = round(elapsed, 3)
    detail["pbr_label"] = "pbr-1024-upscaled" if settings.get("pbr_optimize") else "pbr-native"
    detail["orm_policy"] = maps["policy"].get("metallic_mode")
    detail["height_invert"] = bool(cfg.get("height_invert", False))
    registry.update_job(run_id, key, status="pbr", stage="pbr", detail=detail)
    registry.log_event(run_id, key, "pbr_done", seconds=round(elapsed, 3))


def _qa(registry, cfg, run_id, run_dir, job) -> None:
    key = job["asset_key"]
    directory = _asset_dir(run_dir, key)
    asset = registry.asset(key)
    albedo = load_png(directory / "albedo_final.png")
    normal = load_png(directory / "normal.png")
    height = load_png(directory / "height.png")
    orm = load_png(directory / "orm.png")
    source = load_png(directory / "source.png")
    reasons = []
    reasons += check_albedo(
        albedo, target_w=int(job["target_w"]), target_h=int(job["target_h"]),
        source_seam=float(asset.get("seam") or 99), tileable=True, cfg=cfg,
    )
    reasons += check_normal(normal, albedo)
    reasons += check_height(height, albedo)
    reasons += check_orm(orm, albedo)
    tile = preview_3x3(albedo)
    lit = relight(albedo, normal)
    tile_path = directory / "preview_3x3.png"
    lit_path = directory / "preview_relit.png"
    atomic_save_png(tile, tile_path)
    atomic_save_png(lit, lit_path)
    _record(registry, run_id, key, "preview_3x3", tile_path)
    _record(registry, run_id, key, "preview_relit", lit_path)
    if reasons:
        registry.update_job(run_id, key, status="rejected", rejection_reason="; ".join(reasons), stage="qa")
        registry.log_event(run_id, key, "qa_rejected", reasons=reasons)
        return
    registry.update_job(run_id, key, status="qa_passed", stage="qa", rejection_reason=None)
    registry.log_event(run_id, key, "qa_passed")


def _mirror_aliases(registry: Registry, run_id: str) -> None:
    for job in registry.jobs(run_id):
        canonical = job.get("canonical_key")
        if not canonical:
            continue
        source = registry.job(run_id, canonical)
        if source is None:
            continue
        if source["status"] in {"qa_passed", "packed"}:
            registry.update_job(
                run_id, job["asset_key"], status="qa_passed", stage="alias",
                rejection_reason=None, canonical_key=canonical,
            )
        elif source["status"] in {"rejected", "blocked", "failed"}:
            registry.update_job(
                run_id, job["asset_key"], status=source["status"], stage="alias",
                rejection_reason=source.get("rejection_reason"), canonical_key=canonical,
            )

"""Rebuild a finished Qwen run in place of a new run. The source run is only read."""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path

from PIL import Image

from .anchor import anchor_with_retry
from .caption import caption_many, write_caption
from .config import ensure_work_dirs
from .gallery import CONTROL_KEYS, DRAMATIC_KEYS, write_comparison_sheet
from .imaging import atomic_save_png, load_png
from .pack import build_pack
from .planner import cache_key, reduced_long_side
from .quality import check_albedo, choose_donor, decide_route, fidelity_report, highpass_correlation
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
from .semantics import apply_caption, build_brief, compile_prompt, needs_caption
from .tiles import circular_shift, repair_seam

log = logging.getLogger("prk.salvage")
V1_DONE = {"qa_passed", "packed"}
STAGES = ("anchor", "regen", "pbr", "pack")


def find_salvage_run(registry: Registry, from_run: str) -> str | None:
    rows = registry.conn.execute(
        "SELECT run_id, detail FROM runs ORDER BY created_at DESC"
    ).fetchall()
    for row in rows:
        detail = json.loads(row["detail"] or "{}")
        if detail.get("kind") == "salvage" and detail.get("from_run") == from_run:
            return row["run_id"]
    return None


def _link(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    try:
        os.link(src, dest)
    except OSError:
        shutil.copy2(src, dest)


def _donor(directory: Path) -> Path | None:
    for name in ("raw_qwen.png", "albedo_final.png"):
        path = directory / name
        if path.is_file():
            return path
    return None


def _repair(image: Image.Image, asset: dict, cfg: dict) -> Image.Image:
    reasons = check_albedo(
        image,
        target_w=image.size[0],
        target_h=image.size[1],
        source_seam=float(asset.get("seam") or 99),
        tileable=True,
        cfg=cfg,
    )
    if any(reason.startswith("periodic seam") for reason in reasons):
        repaired = repair_seam(image, band=int(cfg.get("seam_retry_band", 8)))
        again = check_albedo(
            repaired,
            target_w=image.size[0],
            target_h=image.size[1],
            source_seam=float(asset.get("seam") or 99),
            tileable=True,
            cfg=cfg,
        )
        if not any(reason.startswith("periodic seam") for reason in again):
            return repaired
    return image


def _route_metrics(source: Image.Image, qwen: Image.Image, anchored: Image.Image) -> tuple[dict, dict, dict]:
    donor = fidelity_report(source, qwen, qwen)
    lanczos = source.convert("RGB").resize(qwen.size, Image.Resampling.LANCZOS)
    donor["hf_agreement"] = round(highpass_correlation(lanczos, qwen), 4)
    anchored_metrics = fidelity_report(source, anchored, qwen)
    merged = dict(anchored_metrics)
    # Conflict looks at how far v1 drifted. Structure looks at the anchored albedo.
    merged["chroma_de"] = donor["chroma_de"]
    merged["hf_agreement"] = donor["hf_agreement"]
    merged["qwen_hf"] = donor["detail_energy"]
    merged["source_hf"] = donor["source_hf"]
    merged["source_structure"] = donor["source_structure"]
    return donor, anchored_metrics, merged


def _score(detail: dict) -> float:
    donor = detail.get("donor_fidelity") or {}
    fidelity = detail.get("fidelity") or {}
    conflict = 2.0 if detail.get("material_conflict") else 0.0
    return conflict + (1.0 - float(fidelity.get("structure") or 0.0)) + float(donor.get("chroma_de") or 0.0) / 20.0


def _rows(registry: Registry, run_id: str, run_dir: Path, keys: list[str]) -> list[dict]:
    rows = []
    for key in keys:
        job = registry.job(run_id, key)
        if job is None:
            continue
        directory = run_dir / "assets" / key.replace(":", "_")
        detail = job.get("detail") or {}
        fidelity = detail.get("fidelity") or {}
        asset = registry.asset(key) or {}
        rows.append({
            "key": key,
            "name": asset.get("name") or "",
            "source": directory / "source.png",
            "qwen": directory / "raw_qwen.png",
            "anchored": directory / "albedo_final.png",
            "metrics": (
                f"dE {fidelity.get('delta_e')}  struct {fidelity.get('structure')}  "
                f"detail {fidelity.get('detail_gain')}  {detail.get('route')}  {detail.get('donor', 'v1')}"
            ),
        })
    return rows


def _write_sheets(registry: Registry, run_id: str, run_dir: Path) -> None:
    gallery = run_dir / "gallery"
    dramatic = [key for key in (*DRAMATIC_KEYS, *CONTROL_KEYS) if registry.job(run_id, key)]
    write_comparison_sheet(
        _rows(registry, run_id, run_dir, dramatic),
        gallery / "dramatic-triptych.png",
        "source, v1 Qwen, anchored",
    )
    ranked = sorted(
        registry.jobs(run_id),
        key=lambda job: float((job.get("detail") or {}).get("fidelity", {}).get("delta_e") or 0),
        reverse=True,
    )
    worst = [job["asset_key"] for job in ranked if not job.get("canonical_key")][:12]
    write_comparison_sheet(
        _rows(registry, run_id, run_dir, worst),
        gallery / "worst-triptych.png",
        "largest remaining colour difference",
    )


def _write_triage(run_dir: Path, jobs: list[dict], cap: int) -> None:
    regen = [job for job in jobs if (job.get("detail") or {}).get("route") == "regen"]
    lines = [
        "# Salvage triage",
        "",
        f"Jobs: {len(jobs)}",
        f"Regen queue: {len(regen)}",
        f"Regen cap: {cap}",
        "",
    ]
    for job in sorted(regen, key=lambda item: _score(item.get("detail") or {}), reverse=True):
        detail = job.get("detail") or {}
        lines.append(
            f"- {job['asset_key']} {detail.get('subject') or ''} "
            f"conflict={detail.get('material_conflict')} score={_score(detail):.3f}"
        )
    (run_dir / "TRIAGE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _edit_once(edit_fn, source: Image.Image, prompt: str, seed: int, settings: dict, target: tuple[int, int], stats: dict):
    conditioning = source.convert("RGB").resize(target, Image.Resampling.LANCZOS)
    shifted = circular_shift(conditioning, target[0] // 2, target[1] // 2)
    edited, elapsed, peak = _call_with_vram(lambda: edit_fn(shifted, prompt, seed, settings), True)
    stats["gpu_seconds"] += elapsed
    if peak:
        stats["peak_vram_mib"] = max(stats["peak_vram_mib"], peak)
    edited = edited.convert("RGB")
    edited = circular_shift(edited, -(edited.size[0] // 2), -(edited.size[1] // 2))
    return edited


def run_salvage(
    cfg: dict,
    registry: Registry,
    *,
    from_run: str,
    run_id: str | None = None,
    preset: str = "aggressive",
    keys: set[str] | None = None,
    stop_after: str = "pack",
    force: bool = False,
    resume: bool = False,
    regen_cap: int | None = None,
    pack_name: str = "prk-hd-v02",
) -> dict:
    if stop_after not in STAGES:
        raise ValueError(f"unsupported stop_after {stop_after}")
    fidelity = cfg.get("fidelity") or {}
    cap = int(regen_cap if regen_cap is not None else fidelity.get("regen_cap", 200))
    paths = ensure_work_dirs(cfg)
    from_dir = paths["runs"] / from_run
    if not from_dir.is_dir():
        raise SystemExit(f"source run not found: {from_dir}")
    if resume and run_id is None:
        run_id = find_salvage_run(registry, from_run)
    if run_id is None:
        run_id = new_run_id()
    if run_id == from_run:
        raise SystemExit("salvage refuses to write into the source run")
    run_dir = paths["runs"] / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    salvage_cfg = dict(cfg)
    salvage_cfg["config_revision"] = fidelity.get("config_revision", "v0.2.0")
    salvage_cfg["prompt_version"] = fidelity.get("prompt_version", "v2")
    if registry.run(run_id) is None:
        registry.create_run(run_id, "first-1076", preset, salvage_cfg)
        registry.update_run(run_id, detail={"kind": "salvage", "from_run": from_run, "prompt_version": "v2"})
    registry.events_path = run_dir / "events.jsonl"
    selected = _select(registry, from_run, keys)
    _ensure_jobs(registry, cfg, run_id, from_run, from_dir, run_dir, selected, salvage_cfg)
    _anchor_all(registry, cfg, run_id, run_dir, preset, fidelity, force)
    jobs = registry.jobs(run_id)
    _write_triage(run_dir, jobs, cap)
    _write_sheets(registry, run_id, run_dir)
    if stop_after == "anchor":
        registry.update_run(run_id, status="anchored", detail={"kind": "salvage", "from_run": from_run, "stage": "anchor"})
        return {"run_id": run_id, "stage": "anchor", "jobs": len(jobs)}
    _regen(registry, cfg, run_id, run_dir, preset, fidelity, cap, force)
    _write_sheets(registry, run_id, run_dir)
    if stop_after == "regen":
        registry.update_run(run_id, status="regenerated", detail={"kind": "salvage", "from_run": from_run, "stage": "regen"})
        return {"run_id": run_id, "stage": "regen", "jobs": len(registry.jobs(run_id))}
    _pbr(registry, cfg, run_id, run_dir)
    _write_sheets(registry, run_id, run_dir)
    if stop_after == "pbr":
        registry.update_run(run_id, status="pbr", detail={"kind": "salvage", "from_run": from_run, "stage": "pbr"})
        return {"run_id": run_id, "stage": "pbr", "jobs": len(registry.jobs(run_id))}
    dest = paths["dist"] / pack_name
    manifest = build_pack(registry, run_id, dest, pack_name, archive=False)
    registry.update_run(run_id, status="packed", detail={"kind": "salvage", "from_run": from_run, "stage": "pack", "pack": str(dest)})
    return {"run_id": run_id, "stage": "pack", "pack": manifest.get("pack_dir"), "assets": len(manifest.get("assets") or [])}


def _select(registry: Registry, from_run: str, keys: set[str] | None) -> dict[str, dict]:
    chosen = {}
    for job in registry.jobs(from_run):
        if job["status"] not in V1_DONE:
            continue
        if keys is not None and job["asset_key"] not in keys:
            continue
        chosen[job["asset_key"]] = job
    return chosen


def _ensure_jobs(registry, cfg, run_id, from_run, from_dir, run_dir, selected, salvage_cfg) -> None:
    for key, v1_job in selected.items():
        asset = registry.asset(key)
        if asset is None:
            continue
        canonical = v1_job.get("canonical_key")
        if canonical not in selected:
            canonical = None
        existing = registry.job(run_id, key)
        v1_dir = from_dir / "assets" / key.replace(":", "_")
        if canonical:
            if existing is None:
                registry.upsert_job({
                    "run_id": run_id,
                    "asset_key": key,
                    "status": "extracted",
                    "stage": "alias",
                    "seed": v1_job.get("seed"),
                    "prompt": "",
                    "cache_key": v1_job.get("cache_key"),
                    "target_w": v1_job.get("target_w"),
                    "target_h": v1_job.get("target_h"),
                    "attempt": 0,
                    "rejection_reason": None,
                    "canonical_key": canonical,
                    "detail": {"from_run": from_run, "alias": True},
                })
            continue
        donor = _donor(v1_dir)
        source = v1_dir / "source.png"
        if donor is None or not source.is_file():
            registry.upsert_job({
                "run_id": run_id,
                "asset_key": key,
                "status": "blocked",
                "stage": "link",
                "seed": v1_job.get("seed"),
                "prompt": "",
                "cache_key": "",
                "target_w": 0,
                "target_h": 0,
                "attempt": 0,
                "rejection_reason": "v1 source or qwen output is missing",
                "canonical_key": None,
                "detail": {"from_run": from_run},
            })
            continue
        directory = _asset_dir(run_dir, key)
        if from_dir.resolve() in directory.resolve().parents:
            raise RuntimeError("refusing to write the salvage tree inside the source run")
        _link(source, directory / "source.png")
        _link(donor, directory / "raw_qwen.png")
        if existing and existing["status"] not in {"blocked", "failed"}:
            continue
        image = load_png(directory / "raw_qwen.png")
        registry.upsert_job({
            "run_id": run_id,
            "asset_key": key,
            "status": "extracted",
            "stage": "link",
            "seed": v1_job.get("seed") or 1,
            "prompt": "",
            "cache_key": cache_key(
                asset.get("source_sha256") or key,
                "salvage",
                salvage_cfg.get("prompt_version"),
                salvage_cfg.get("config_revision"),
                "anchor",
            ),
            "target_w": image.size[0],
            "target_h": image.size[1],
            "attempt": 0,
            "rejection_reason": None,
            "canonical_key": None,
            "detail": {"from_run": from_run, "generation_label": (v1_job.get("detail") or {}).get("generation_label")},
        })
        registry.log_event(run_id, key, "salvage_linked")


def _anchor_all(registry, cfg, run_id, run_dir, preset, fidelity, force: bool) -> None:
    jobs = [job for job in registry.jobs(run_id) if not job.get("canonical_key") and job["status"] not in {"blocked", "rejected"}]
    for index, job in enumerate(jobs, start=1):
        key = job["asset_key"]
        detail = job.get("detail") or {}
        directory = _asset_dir(run_dir, key)
        if detail.get("anchored") and (directory / "albedo_final.png").is_file() and not force:
            continue
        asset = registry.asset(key)
        if asset is None:
            continue
        source = load_png(directory / "source.png")
        qwen = load_png(directory / "raw_qwen.png")
        brief = build_brief(asset, source, cfg.get("name_lexicon") or {})
        image, diag = anchor_with_retry(
            source, qwen, kind=brief.get("kind") or "organic", preset=preset, fidelity=fidelity,
        )
        image = _repair(image, asset, cfg)
        albedo = directory / "albedo_final.png"
        atomic_save_png(image, albedo)
        _record(registry, run_id, key, "source", directory / "source.png")
        _record(registry, run_id, key, "raw_qwen", directory / "raw_qwen.png")
        _record(registry, run_id, key, "albedo_final", albedo)
        donor_metrics, anchored_metrics, merged = _route_metrics(source, qwen, image)
        route = decide_route(merged, brief, fidelity)
        prompt = compile_prompt(asset, brief, cfg.get("family_prompts") or {}, image.size[0], image.size[1])
        (directory / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")
        stored = dict(detail)
        stored.update({
            "anchored": True,
            "route": route,
            "regen_pending": route == "regen",
            "regen_done": False,
            "donor": "v1",
            "kind": brief.get("kind"),
            "effective_material": brief.get("effective_material"),
            "material_conflict": brief.get("material_conflict"),
            "subject": brief.get("subject"),
            "subject_class": brief.get("subject_class"),
            "name_confidence": brief.get("name_confidence"),
            "appearance": brief.get("appearance"),
            "enhance": brief.get("enhance"),
            "hold": brief.get("hold"),
            "donor_fidelity": {
                "delta_e": donor_metrics.get("delta_e"),
                "chroma_de": donor_metrics.get("chroma_de"),
                "structure": donor_metrics.get("structure"),
                "hf_agreement": donor_metrics.get("hf_agreement"),
                "detail_gain": donor_metrics.get("detail_gain"),
            },
            "review": bool(diag.get("review") or route == "regen"),
            "tightened": bool(diag.get("tightened")),
            "trust": diag.get("trust"),
            "prompt_version": "v2",
        })
        stored["fidelity"] = {
            "delta_e": anchored_metrics["delta_e"],
            "chroma_de": anchored_metrics["chroma_de"],
            "luma_std_ratio": anchored_metrics["luma_std_ratio"],
            "structure": anchored_metrics["structure"],
            "detail_gain": anchored_metrics["detail_gain"],
            "band_agreement": anchored_metrics["band_agreement"],
        }
        registry.update_job(
            run_id, key, status="generated", stage="anchor", prompt=prompt,
            target_w=image.size[0], target_h=image.size[1], detail=stored, rejection_reason=None,
        )
        registry.log_event(run_id, key, "anchored", route=route)
        if index % 25 == 0 or index == len(jobs):
            log.info("anchored %s/%s", index, len(jobs))


def _regen(registry, cfg, run_id, run_dir, preset, fidelity, cap: int, force: bool) -> None:
    pending = []
    for job in registry.jobs(run_id):
        detail = job.get("detail") or {}
        if job.get("canonical_key"):
            continue
        if detail.get("regen_pending") and not detail.get("regen_done"):
            pending.append(job)
    if len(pending) > cap:
        pending.sort(key=lambda job: _score(job.get("detail") or {}), reverse=True)
        for job in pending[cap:]:
            detail = dict(job.get("detail") or {})
            detail["route"] = "anchor"
            detail["regen_pending"] = False
            detail["regen_capped"] = True
            detail["review"] = True
            registry.update_job(run_id, job["asset_key"], detail=detail)
            registry.log_event(run_id, job["asset_key"], "regen_capped")
        pending = pending[:cap]
    if not pending:
        log.info("regen queue is empty")
        return
    log.info("regen queue %s", len(pending))
    _caption_queue(registry, cfg, run_id, run_dir, pending)
    edit_fn, free_fn = _default_edit(cfg)
    stats = {"gpu_seconds": 0.0, "peak_vram_mib": 0.0}
    try:
        free_fn()
    except Exception:
        log.info("comfy free before regen did not complete")
    settings = _settings(cfg)
    for job in pending:
        key = job["asset_key"]
        current = registry.job(run_id, key) or job
        detail = dict(current.get("detail") or {})
        directory = _asset_dir(run_dir, key)
        v2_path = directory / "raw_qwen_v2.png"
        source = load_png(directory / "source.png")
        try:
            if force or not v2_path.is_file():
                edited = _edit_with_oom(
                    edit_fn, source, current.get("prompt") or "", int(current.get("seed") or 1),
                    settings, (int(current["target_w"]), int(current["target_h"])), stats, cfg,
                )
                atomic_save_png(edited, v2_path)
            edited = load_png(v2_path)
        except Exception as exc:  # noqa: BLE001
            detail["regen_done"] = True
            detail["regen_pending"] = False
            detail["regen_error"] = str(exc)[:400]
            detail["review"] = True
            registry.update_job(run_id, key, detail=detail)
            registry.log_event(run_id, key, "regen_error", error=str(exc)[:400])
            log.info("regen failed %s", key)
            continue
        asset = registry.asset(key) or {}
        kind = detail.get("kind") or "organic"
        image, _diag = anchor_with_retry(source, edited, kind=kind, preset=preset, fidelity=fidelity)
        image = _repair(image, asset, cfg)
        v1_fidelity = detail.get("fidelity") or {}
        v2_report = fidelity_report(source, image, edited)
        winner = choose_donor(v1_fidelity, v2_report)
        if winner == "v2":
            atomic_save_png(image, directory / "albedo_final.png")
            _record(registry, run_id, key, "albedo_final", directory / "albedo_final.png")
            detail["fidelity"] = {
                "delta_e": v2_report["delta_e"],
                "chroma_de": v2_report["chroma_de"],
                "luma_std_ratio": v2_report["luma_std_ratio"],
                "structure": v2_report["structure"],
                "detail_gain": v2_report["detail_gain"],
                "band_agreement": v2_report["band_agreement"],
            }
            detail["donor"] = "v2"
        detail["regen_done"] = True
        detail["regen_pending"] = False
        detail["review"] = winner == "v1"
        fields = {"status": "generated", "stage": "regen", "detail": detail}
        if winner == "v2":
            fields["target_w"] = image.size[0]
            fields["target_h"] = image.size[1]
        registry.update_job(run_id, key, **fields)
        registry.log_event(run_id, key, "regen_done", donor=detail.get("donor"))
        log.info("regen %s kept %s", key, detail.get("donor"))
    try:
        free_fn()
    except Exception:
        log.info("comfy free after regen did not complete")


def _edit_with_oom(edit_fn, source, prompt, seed, settings, target, stats, cfg):
    try:
        return _edit_once(edit_fn, source, prompt, seed, settings, target, stats)
    except Exception as exc:
        if not is_oom(exc) or max(target) <= 1024:
            raise
        smaller = reduced_long_side(target[0], target[1], 1024, int(cfg["material_rules"].get("snap", 32)))
        return _edit_once(edit_fn, source, prompt, seed, settings, smaller, stats)


def _caption_queue(registry, cfg, run_id, run_dir, pending) -> None:
    from .comfy_client import ComfyClient
    try:
        ComfyClient(cfg["comfy_url"], timeout=30).free()
    except Exception:
        log.info("caption stage did not free comfy")
    groups: dict[str, list] = {}
    cache_dir = Path(cfg["work_root"]) / "cache" / "captions"
    for job in pending:
        asset = registry.asset(job["asset_key"]) or {}
        directory = _asset_dir(run_dir, job["asset_key"])
        source = directory / "source.png"
        brief = build_brief(asset, load_png(source), cfg.get("name_lexicon") or {})
        if not needs_caption(brief, "regen"):
            continue
        family = asset.get("family") or ""
        reading = brief.get("reading") or {}
        if family and (brief.get("material_conflict") or float(brief.get("name_confidence") or 0) < 0.45):
            group = f"{family}|{reading.get('hue_name')}|{reading.get('tone')}"
        else:
            group = job["asset_key"]
        groups.setdefault(group, []).append((job, asset, source, brief))
    if not groups:
        return
    representatives = []
    for members in groups.values():
        _job, asset, source, _brief = members[0]
        representatives.append({
            "image": str(source),
            "name": asset.get("name") or "",
            "category": asset.get("material_class") or "",
            "family": asset.get("family") or "",
        })
    found = caption_many(cfg, representatives, cache_dir)
    for members in groups.values():
        _job, _asset, source, _brief = members[0]
        payload = found.get(str(source)) or {}
        for job, asset, member_source, brief in members:
            if payload and member_source != source:
                write_caption(cache_dir, member_source, payload)
            updated = apply_caption(brief, payload)
            prompt = compile_prompt(
                asset, updated, cfg.get("family_prompts") or {},
                int(job.get("target_w") or 0), int(job.get("target_h") or 0),
            )
            detail = dict((registry.job(run_id, job["asset_key"]) or job).get("detail") or {})
            detail["subject"] = updated.get("subject")
            detail["kind"] = updated.get("kind")
            detail["effective_material"] = updated.get("effective_material")
            detail["material_conflict"] = updated.get("material_conflict")
            detail["subject_class"] = updated.get("subject_class")
            detail["caption_used"] = bool(updated.get("caption_used"))
            directory = _asset_dir(run_dir, job["asset_key"])
            (directory / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")
            registry.update_job(run_id, job["asset_key"], prompt=prompt, detail=detail)


def _pbr(registry, cfg, run_id, run_dir) -> None:
    pbr_fn, free_fn = _default_pbr(cfg)
    stats = {"gpu_seconds": 0.0, "peak_vram_mib": 0.0, "pbr_force_optimize": bool(cfg.get("pbr_optimize"))}
    try:
        free_fn()
    except Exception:
        log.info("comfy free before pbr did not complete")
    for job in registry.jobs(run_id):
        if job.get("canonical_key") or job["status"] != "generated":
            continue
        _pbr_stage(registry, cfg, run_id, run_dir, job, pbr_fn, stats, True, free_fn)
        current = registry.job(run_id, job["asset_key"])
        if current and current["status"] == "pbr":
            _qa(registry, cfg, run_id, run_dir, current)
    _mirror_aliases(registry, run_id)
    try:
        free_fn()
    except Exception:
        log.info("comfy free after pbr did not complete")

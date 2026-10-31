"""Cohort selection, prompts, seeds, and target resolution."""

from __future__ import annotations

import hashlib
from pathlib import Path

from .registry import Registry, utc_now

STAGE_ORDER = ("pending", "extracted", "generated", "pbr", "qa_passed", "packed")


def derive_seed(*parts) -> int:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % (2**31 - 1)
    return value or 1


def cache_key(*parts) -> str:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def snap(value: float, step: int) -> int:
    return max(step, int(round(value / step)) * step)


def target_size(width: int, height: int, category: str, rules: dict) -> tuple[int, int]:
    """Long-side target, aspect preserved, dimensions snapped to the VAE multiple."""
    table = rules.get("resolution") or {}
    long_side = int(table.get(category) or table.get("default") or 1024)
    if category == "metal" and min(width, height) < int(rules.get("metal_small_source_min_side", 128)):
        long_side = min(long_side, int(rules.get("metal_small_source_long_side", 1024)))
    long_side = max(int(rules.get("snap", 32)), long_side)
    scale = long_side / max(width, height, 1)
    step = int(rules.get("snap", 32))
    return snap(width * scale, step), snap(height * scale, step)


def reduced_long_side(width: int, height: int, long_side: int = 1024, step: int = 32) -> tuple[int, int]:
    scale = long_side / max(width, height, 1)
    return snap(width * scale, step), snap(height * scale, step)


def _flag(metadata: dict, name: str) -> bool:
    value = metadata.get(name, 0)
    if isinstance(value, str):
        return value.strip().lower() not in ("", "0", "false", "no")
    return bool(value)


def rejection_reason(asset: dict, exclusions: dict) -> str | None:
    name = (asset.get("name") or "").lower()
    prefix = (exclusions.get("shadow_name_prefix") or "shadow").lower()
    if name.startswith(prefix):
        return f"name starts with {prefix}; shadow masks are not re-authored as albedo"
    category = asset.get("material_class") or ""
    if category in set(exclusions.get("blocked_categories") or []):
        return f"category {category} is excluded from the generic albedo cohort"
    metadata = asset.get("source_metadata") or {}
    for flag in exclusions.get("blocked_flags") or []:
        if _flag(metadata, flag):
            return f"catalogue flag {flag} is set; routed out of generic generation"
    limit = float(exclusions.get("alpha_above", 0.02))
    if float(asset.get("alpha") or 0) > limit:
        return f"alpha fraction {asset.get('alpha')} exceeds {limit}"
    return None


def build_prompt(asset: dict, prompts: dict, target_w: int, target_h: int) -> str:
    category = asset.get("material_class") or ""
    body = (prompts.get("categories") or {}).get(category) or prompts.get("default_category") or ""
    shared = prompts.get("shared") or ""
    fields = {
        "name": asset.get("name") or "unnamed",
        "family": asset.get("family") or "none",
        "width": target_w,
        "height": target_h,
    }
    text = (shared.strip() + "\n" + body.strip()).format(**fields)
    text += (
        f"\nSource size {asset.get('source_width')}x{asset.get('source_height')}, "
        f"target {target_w}x{target_h}, aspect preserved."
    )
    tags = asset.get("tags") or []
    if tags:
        text += "\nSemantic tags: " + ", ".join(tags) + "."
    if "possibly_tileable" in tags:
        text += "\nTreat the frame as one period of a repeating surface."
    return text


def parse_id_file(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    keys = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        keys.add(line)
    return keys


def plan_jobs(
    registry: Registry,
    cfg: dict,
    run_id: str,
    *,
    cohort: str = "first-1076",
    family: str | None = None,
    types: list[int] | None = None,
    ids: set[str] | None = None,
    limit: int | None = None,
    force: bool = False,
    retry_failed: bool = False,
) -> list[dict]:
    """Create or refresh jobs for the selected cohort. Returns the job rows."""
    if cohort != "first-1076":
        raise ValueError(f"unsupported cohort {cohort}")
    lock = cfg["model_lock"]
    model_revision = lock["qwen"]["files"]["diffusion"]["filename"]
    assets = registry.cohort_assets(risk="auto", family=family, types=types, ids=ids)
    if limit is not None:
        assets = assets[: int(limit)]
    prompts = cfg["family_prompts"]
    rules = cfg["material_rules"]
    exclusions = cfg["exclusions"]
    jobs = []
    for asset in assets:
        existing = registry.job(run_id, asset["key"])
        if existing and existing["status"] in {"qa_passed", "packed"} and not force:
            jobs.append(existing)
            continue
        if existing and existing["status"] in {"failed", "blocked"} and not retry_failed and not force:
            jobs.append(existing)
            continue
        if existing and existing["status"] == "rejected" and not force:
            jobs.append(existing)
            continue
        reason = rejection_reason(asset, exclusions)
        width = int(asset["source_width"] or 0)
        height = int(asset["source_height"] or 0)
        if width < 1 or height < 1:
            reason = reason or "source dimensions are missing"
            target = (0, 0)
        else:
            target = target_size(width, height, asset["material_class"], rules)
        seed = derive_seed(
            model_revision,
            cfg.get("config_revision"),
            asset["rdb_type"],
            asset["rdb_id"],
            asset.get("source_sha256") or asset["key"],
        )
        prompt = build_prompt(asset, prompts, target[0], target[1]) if not reason else ""
        key = cache_key(
            asset.get("source_sha256") or asset["key"],
            model_revision,
            cfg.get("config_revision"),
            cfg.get("prompt_version"),
            seed,
            "generate",
            cfg.get("stage_version"),
        )
        if existing and existing.get("cache_key") and existing["cache_key"] != key and existing["status"] in {"qa_passed", "packed"} and not force:
            job = dict(existing)
            job["status"] = "blocked"
            job["rejection_reason"] = "recipe changed; pass --force to overwrite"
            registry.upsert_job(job)
            jobs.append(registry.job(run_id, asset["key"]))
            continue
        attempt = 0
        if existing and retry_failed:
            attempt = int(existing.get("attempt") or 0)
        job = {
            "run_id": run_id,
            "asset_key": asset["key"],
            "status": "rejected" if reason else "pending",
            "stage": "plan",
            "seed": seed,
            "prompt": prompt,
            "cache_key": key,
            "target_w": target[0],
            "target_h": target[1],
            "attempt": attempt,
            "rejection_reason": reason,
            "canonical_key": None,
            "detail": {
                "material_class": asset["material_class"],
                "family": asset["family"],
                "model": model_revision,
                "prompt_version": cfg.get("prompt_version"),
                "planned_at": utc_now(),
            },
        }
        registry.upsert_job(job)
        registry.log_event(run_id, asset["key"], "planned", status=job["status"], reason=reason)
        jobs.append(registry.job(run_id, asset["key"]))
    return jobs


def link_identical_pixels(registry: Registry, run_id: str) -> int:
    """Share one generation across exact pixel duplicates in this run.

    Perceptual hashes are ignored. Assets whose pixels differ are never linked,
    even when the catalogue dHash collides.
    """
    groups: dict[tuple, list] = {}
    for job in registry.jobs(run_id):
        if job["status"] in {"rejected", "blocked", "failed"}:
            continue
        asset = registry.asset(job["asset_key"])
        if asset is None or not asset.get("pixel_sha256"):
            continue
        signature = (asset["pixel_sha256"], asset["disposition"], asset["material_class"])
        groups.setdefault(signature, []).append((asset, job))
    linked = 0
    for members in groups.values():
        if len(members) < 2:
            continue
        members.sort(key=lambda item: (item[0]["rdb_type"], item[0]["rdb_id"]))
        canonical = members[0][0]["key"]
        for asset, job in members[1:]:
            if job.get("canonical_key") == canonical:
                continue
            detail = dict(job.get("detail") or {})
            detail["shared_from"] = canonical
            registry.update_job(run_id, asset["key"], canonical_key=canonical, detail=detail)
            linked += 1
    return linked


def describe_job(job: dict) -> str:
    reason = job.get("rejection_reason") or ""
    return (
        f"{job['asset_key']} {job['status']} {job.get('target_w')}x{job.get('target_h')} "
        f"seed={job.get('seed')} {reason}"
    ).rstrip()


def load_ids(path: str | None) -> set[str] | None:
    if not path:
        return None
    return parse_id_file(Path(path))


def dumps_plan(jobs: list[dict]) -> str:
    return "\n".join(describe_job(job) for job in jobs) + ("\n" if jobs else "")

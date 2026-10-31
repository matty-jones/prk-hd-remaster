"""Human-readable run report. Counts come from the registry, not from estimates."""

from __future__ import annotations

import json
from pathlib import Path

from . import __version__
from .registry import Registry


def write_report(registry: Registry, run_id: str, dest: Path, *, extra: dict | None = None) -> Path:
    run = registry.run(run_id) or {}
    summary = registry.summary(run_id)
    jobs = registry.jobs(run_id)
    detail = run.get("detail") or {}
    if extra:
        detail = {**detail, **extra}
    lines = [
        f"# PRK HD remaster run {run_id}",
        "",
        f"Software version `{__version__}`. This report separates the factory from texture outputs.",
        "",
        "## Cohort",
        "",
        f"- Jobs recorded: {summary['total']}",
        f"- Status counts: `{json.dumps(summary['counts'], sort_keys=True)}`",
        "",
        "A completed cohort has every `risk=auto` id in exactly one of `packed`, `rejected`, or `blocked`.",
        "",
        "## Outputs",
        "",
    ]
    packed = [job for job in jobs if job["status"] == "packed"]
    passed = [job for job in jobs if job["status"] in {"qa_passed", "packed"}]
    lines.append(f"- QA passed or packed: {len(passed)}")
    lines.append(f"- Packed jobs: {len(packed)}")
    lines.append("- Maps per packed surface: albedo `_d`, OpenGL normal `_n`, height `_h`, ORM `_orm`.")
    lines.append("- ORM roughness and metalness are material-class rules plus spatial variation, not a learned PBR estimate.")
    lines.append("- Height is relative relief from PBRFusion filtered depth, not metric displacement.")
    lines.append("")
    lines.append("## Failures")
    lines.append("")
    failures = [job for job in jobs if job["status"] in {"rejected", "blocked", "failed"}]
    if not failures:
        lines.append("None recorded.")
    else:
        for job in failures:
            lines.append(f"- `{job['asset_key']}` {job['status']}: {job.get('rejection_reason') or 'no reason stored'}")
    lines.append("")
    lines.append("## Timing and hardware")
    lines.append("")
    for key in ("wall_seconds", "gpu_seconds", "peak_vram_mib", "gpu_name", "driver", "disk_bytes"):
        if key in detail:
            lines.append(f"- {key}: {detail[key]}")
    if "wall_seconds" not in detail:
        lines.append("- Timing is recorded when a production run executes on this machine.")
    lines.append("")
    lines.append("## Models")
    lines.append("")
    lines.append(f"- Detail: `{json.dumps(detail.get('models', {}), sort_keys=True)}`")
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    lines.append("- Turbo is sampled at CFG 1 with the published 8-step sigma list. A negative prompt is not used.")
    lines.append("- A bilinear upscale is never described as native 2K. If generation falls back to 1024, the job detail says so.")
    lines.append("- In-game before/after screenshots are a user session. See the gallery CAPTURE.md. This report does not claim that comparison was captured.")
    lines.append("")
    path = Path(dest)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path

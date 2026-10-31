"""Self-contained review gallery. No network requests."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from PIL import Image

from .imaging import atomic_save_png, load_png
from .registry import Registry
from .tiles import preview_3x3, relight

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>PRK HD review __RUN__</title>
<style>
body { margin: 0; font: 14px/1.4 sans-serif; background: #161616; color: #eee; }
header { padding: 16px 20px; background: #222; position: sticky; top: 0; }
input, select { background: #111; color: #eee; border: 1px solid #444; padding: 4px 6px; }
.counts span { margin-right: 12px; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 12px; padding: 12px; }
.card { background: #242424; padding: 8px; }
.card img { width: 100%; height: auto; background: #111; cursor: zoom-in; }
.card h3 { font-size: 13px; margin: 6px 0; }
.meta { color: #aaa; font-size: 12px; }
#viewer { display: none; position: fixed; inset: 0; background: rgba(0,0,0,.88); overflow: auto; padding: 24px; }
#viewer img { max-width: 100%; }
.family { padding: 8px 20px 20px; }
</style>
</head>
<body>
<header>
  <h1>PRK HD review <span id="run"></span></h1>
  <div class="counts" id="counts"></div>
  <p>
    <input id="q" placeholder="family, id, category" size="32">
    <select id="status"><option value="">any status</option></select>
    <select id="category"><option value="">any category</option></select>
  </p>
  <p class="meta">In-game screenshots belong in in-game-evidence/. Capture steps are in CAPTURE.md next to this file.</p>
</header>
<div id="families" class="family"></div>
<div id="grid" class="grid"></div>
<div id="viewer"></div>
<script>
const DATA = __DATA__;
const counts = document.getElementById("counts");
const tally = {};
DATA.assets.forEach(a => { tally[a.status] = (tally[a.status] || 0) + 1; });
counts.innerHTML = Object.keys(tally).sort().map(k => `<span>${k}: ${tally[k]}</span>`).join("")
  + `<span>shown cohort: ${DATA.assets.length}</span><span>target: ${DATA.cohort_target}</span>`;
const statusEl = document.getElementById("status");
const categoryEl = document.getElementById("category");
const gridEl = document.getElementById("grid");
const familiesEl = document.getElementById("families");
const viewerEl = document.getElementById("viewer");
const statuses = [...new Set(DATA.assets.map(a => a.status))].sort();
const categories = [...new Set(DATA.assets.map(a => a.category))].sort();
for (const value of statuses) statusEl.append(new Option(value, value));
for (const value of categories) categoryEl.append(new Option(value, value));
document.getElementById("run").textContent = DATA.run_id;
function thumb(asset, role) {
  const file = asset.files[role];
  if (!file) return "";
  return `<img data-full="${file}" src="${file}" alt="${role}">`;
}
function render() {
  const q = document.getElementById("q").value.toLowerCase();
  const st = statusEl.value;
  const cat = categoryEl.value;
  const rows = DATA.assets.filter(a => {
    if (st && a.status !== st) return false;
    if (cat && a.category !== cat) return false;
    if (!q) return true;
    return (a.key + " " + a.family + " " + a.name + " " + a.category).toLowerCase().includes(q);
  });
  gridEl.innerHTML = rows.map(a => `
    <article class="card">
      <h3>${a.key} · ${a.status}</h3>
      <div class="meta">${a.category} / ${a.family || "no family"} / ${a.name || "unnamed"} / ${a.target}</div>
      ${thumb(a, "source")}${thumb(a, "albedo")}${thumb(a, "normal")}${thumb(a, "height")}${thumb(a, "roughness")}${thumb(a, "tile")}${thumb(a, "relit")}
      <p class="meta">${a.reason || ""}</p>
    </article>`).join("");
  const families = {};
  rows.forEach(a => {
    const key = a.family || "(single)";
    (families[key] = families[key] || []).push(a);
  });
  familiesEl.innerHTML = "<h2>Families</h2>" + Object.keys(families).sort().map(name => {
    const group = families[name];
    const hero = group.find(a => a.status === "qa_passed" || a.status === "packed") || group[0];
    return `<p><strong>${name}</strong> ${group.length} · hero ${hero.key} ${hero.status}</p>`;
  }).join("");
}
document.getElementById("q").addEventListener("input", render);
statusEl.addEventListener("change", render);
categoryEl.addEventListener("change", render);
gridEl.addEventListener("click", event => {
  const full = event.target.getAttribute && event.target.getAttribute("data-full");
  if (!full) return;
  viewerEl.style.display = "block";
  viewerEl.innerHTML = `<img src="${full}" alt="">`;
});
viewerEl.addEventListener("click", () => { viewerEl.style.display = "none"; viewerEl.innerHTML = ""; });
render();
</script>
</body>
</html>
"""

CAPTURE = """# In-game capture

The factory does not claim these screenshots were taken.

1. Close Project Rubi-Ka. Installing a pack while the client is open does not reload it.
2. Install the pack with `prk-remaster pack install` and start the game.
3. In the client log, confirm a line naming pack `prk-hd-v01` and a file count.
4. Stand on a ground tile that is in the pack (a `ground:sand` or `ground:grass` id from the gallery).
5. Note the time of day, graphics settings, and camera position.
6. Take a screenshot. Press Ctrl+Shift+I on the surface and record the `type:id` from the log or clipboard.
7. Disable the pack with `prk-remaster pack disable --yes` and restart the client.
8. Return to the same spot and time of day and take the matching screenshot.
9. Put both images in `in-game-evidence/` with the texture id in the filename.

A visible material change is pending that user session. Do not treat this file as evidence that it happened.
"""


def _thumb(src: Path, dest: Path, size: int = 192) -> None:
    if not src.is_file():
        return
    image = load_png(src)
    image.thumbnail((size, size))
    atomic_save_png(image.convert("RGB") if image.mode != "RGB" else image, dest)


def write_gallery(registry: Registry, run_id: str, run_dir: Path, dest: Path | None = None) -> Path:
    dest = Path(dest or (run_dir / "gallery"))
    if dest.exists():
        shutil.rmtree(dest)
    thumbs = dest / "thumbs"
    thumbs.mkdir(parents=True)
    (dest / "in-game-evidence").mkdir()
    (dest / "CAPTURE.md").write_text(CAPTURE, encoding="utf-8")
    assets = []
    for job in registry.jobs(run_id):
        outputs = {item["role"]: Path(item["path"]) for item in registry.outputs(run_id, job["asset_key"]) if item.get("path")}
        asset = registry.asset(job["asset_key"]) or {}
        files = {}
        stem = job["asset_key"].replace(":", "_")
        for role, label in (
            ("source", "source"),
            ("albedo_final", "albedo"),
            ("normal", "normal"),
            ("height", "height"),
        ):
            path = outputs.get(role)
            if path and path.is_file():
                target = thumbs / f"{stem}_{label}.png"
                _thumb(path, target)
                files[label] = f"thumbs/{target.name}"
        albedo = outputs.get("albedo_final")
        normal = outputs.get("normal")
        if albedo and albedo.is_file() and normal and normal.is_file():
            tile = preview_3x3(load_png(albedo), cell=160)
            rel = relight(load_png(albedo), load_png(normal))
            tile_path = thumbs / f"{stem}_tile.png"
            rel_path = thumbs / f"{stem}_relit.png"
            atomic_save_png(tile, tile_path)
            rel.thumbnail((192, 192))
            atomic_save_png(rel, rel_path)
            files["tile"] = f"thumbs/{tile_path.name}"
            files["relit"] = f"thumbs/{rel_path.name}"
            orm = outputs.get("orm")
            if orm and orm.is_file():
                roughness = load_png(orm).getchannel("G")
                rough_path = thumbs / f"{stem}_roughness.png"
                atomic_save_png(roughness.convert("RGB"), rough_path)
                files["roughness"] = f"thumbs/{rough_path.name}"
        assets.append({
            "key": job["asset_key"],
            "status": job["status"],
            "category": (asset.get("material_class") or ""),
            "family": asset.get("family") or "",
            "name": asset.get("name") or "",
            "target": f"{job.get('target_w')}x{job.get('target_h')}",
            "reason": job.get("rejection_reason") or "",
            "files": files,
        })
    payload = {"run_id": run_id, "cohort_target": 1076, "assets": assets}
    html = HTML.replace("__RUN__", run_id).replace("__DATA__", json.dumps(payload))
    index = dest / "index.html"
    index.write_text(html, encoding="utf-8")
    return index

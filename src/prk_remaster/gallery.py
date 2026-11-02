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
.card button, #pager button { background: #333; color: #eee; border: 1px solid #555; margin: 2px 4px 2px 0; padding: 2px 6px; cursor: pointer; }
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
    <span id="pager"><button type="button" id="prev">Previous</button><span id="pageinfo"></span><button type="button" id="next">Next</button></span>
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
const PAGE = 12;
let page = 0;
let rows = DATA.assets;
function thumb(asset, role) {
  const file = asset.files[role];
  if (!file) return "";
  return `<img data-full="${file}" src="${file}" alt="${role}">`;
}
function extra(asset, role, label) {
  const file = asset.files[role];
  if (!file) return "";
  return `<button type="button" data-full="${file}">${label}</button>`;
}
function applyFilter() {
  const q = document.getElementById("q").value.toLowerCase();
  const st = statusEl.value;
  const cat = categoryEl.value;
  rows = DATA.assets.filter(a => {
    if (st && a.status !== st) return false;
    if (cat && a.category !== cat) return false;
    if (!q) return true;
    return (a.key + " " + a.family + " " + a.name + " " + a.category).toLowerCase().includes(q);
  });
  page = 0;
  render();
}
function render() {
  const pages = Math.max(1, Math.ceil(rows.length / PAGE));
  if (page < 0) page = 0;
  if (page >= pages) page = pages - 1;
  const slice = rows.slice(page * PAGE, page * PAGE + PAGE);
  document.getElementById("pageinfo").textContent = ` page ${page + 1} of ${pages} · ${rows.length} matches `;
  gridEl.innerHTML = slice.map(a => `
    <article class="card">
      <h3>${a.key} · ${a.status}</h3>
      <div class="meta">${a.category} / ${a.family || "no family"} / ${a.name || "unnamed"} / ${a.target}</div>
      ${thumb(a, "source")}${thumb(a, "albedo")}
      <p>${extra(a, "normal", "normal")}${extra(a, "height", "height")}${extra(a, "roughness", "roughness")}${extra(a, "tile", "3x3")}${extra(a, "relit", "relit")}</p>
      <p class="meta">${a.reason || ""}</p>
    </article>`).join("");
  const families = {};
  rows.forEach(a => {
    const key = a.family || "(single)";
    (families[key] = families[key] || []).push(a);
  });
  const names = Object.keys(families).sort();
  familiesEl.innerHTML = `<h2>Families (${names.length})</h2>` + names.slice(0, 40).map(name => {
    const group = families[name];
    const hero = group.find(a => a.status === "qa_passed" || a.status === "packed") || group[0];
    return `<p><strong>${name}</strong> ${group.length} · hero ${hero.key} ${hero.status}</p>`;
  }).join("") + (names.length > 40 ? "<p class='meta'>Filter to see the other families.</p>" : "");
}
document.getElementById("q").addEventListener("input", applyFilter);
statusEl.addEventListener("change", applyFilter);
categoryEl.addEventListener("change", applyFilter);
document.getElementById("prev").addEventListener("click", () => { page -= 1; render(); });
document.getElementById("next").addEventListener("click", () => { page += 1; render(); });
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


DRAMATIC_KEYS = (
    "1010004:10946",
    "1010004:44528",
    "1010004:36111",
    "1010006:215",
    "1010004:255754",
    "1010004:23133",
    "1010004:99474",
    "1010009:95482",
    "1010009:9691",
    "1010009:8544",
    "1010004:20388",
    "1010004:20908",
)
CONTROL_KEYS = ("1010006:2",)


def _sheet_font(size: int):
    from PIL import ImageFont
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if path.is_file():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _fit(image: Image.Image, size: int) -> Image.Image:
    fitted = image.convert("RGB")
    fitted.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), (18, 18, 18))
    canvas.paste(fitted, ((size - fitted.size[0]) // 2, (size - fitted.size[1]) // 2))
    return canvas


def write_comparison_sheet(rows: list[dict], dest: Path, title: str) -> Path:
    """Three columns: source, v1 Qwen, anchored albedo. ``rows`` carry paths and a metric line."""
    from PIL import ImageDraw
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        image = Image.new("RGB", (640, 80), (16, 16, 16))
        ImageDraw.Draw(image).text((12, 24), title + " (empty)", fill=(230, 230, 230), font=_sheet_font(18))
        atomic_save_png(image, dest)
        return dest
    cell = 220
    label_h = 52
    header = 36
    gap = 8
    width = gap + 3 * (cell + gap)
    height = header + gap + len(rows) * (cell + label_h + gap)
    sheet = Image.new("RGB", (width, height), (12, 12, 12))
    draw = ImageDraw.Draw(sheet)
    font = _sheet_font(14)
    small = _sheet_font(12)
    draw.text((gap, 8), title, fill=(240, 240, 240), font=_sheet_font(18))
    for index, row in enumerate(rows):
        top = header + index * (cell + label_h + gap)
        panels = (
            ("source", row.get("source")),
            ("qwen v1", row.get("qwen")),
            ("anchored", row.get("anchored")),
        )
        for column, (label, path) in enumerate(panels):
            left = gap + column * (cell + gap)
            if path and Path(path).is_file():
                sheet.paste(_fit(load_png(path), cell), (left, top))
            else:
                draw.rectangle((left, top, left + cell, top + cell), outline=(80, 80, 80))
            draw.text((left, top + cell + 2), label, fill=(200, 200, 200), font=small)
        caption = f"{row.get('key', '')}  {row.get('name') or ''}"
        metrics = row.get("metrics") or ""
        draw.text((gap, top + cell + 18), caption[:90], fill=(230, 230, 230), font=font)
        draw.text((gap, top + cell + 34), str(metrics)[:140], fill=(170, 170, 170), font=small)
    atomic_save_png(sheet, dest)
    return dest

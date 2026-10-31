# PRK HD Remaster Factory

Local factory that re-authors Project Rubi-Ka surface textures with Qwen-Image-2.1-Turbo, builds normal and height maps with PBRFusion 4, packs ORM from material rules, and installs an ao-vk texture pack. Original game files are not modified. Generated art and model weights stay outside Git.

The first production cohort is the catalogue's 1,076 `risk=auto` materials (312 terrain, 288 architecture, 186 metal, 176 interior, 114 stone).

## Setup

```bash
uv sync
uv run prk-remaster doctor --prk-root "$HOME/Games/Project-Rubi-Ka/client" --offline
uv run prk-remaster catalogue import \
  --prk-root "$HOME/Games/Project-Rubi-Ka/client" \
  --csv "$HOME/Games/ao-vk/TEMP/texture-catalogue/assets.csv"
uv run prk-remaster catalogue reconcile --prk-root "$HOME/Games/Project-Rubi-Ka/client"
uv run prk-remaster models setup --profile rtx5080-16gb --accept-qwen-research-license
```

`--accept-qwen-research-license` records that the Qwen Research License was accepted for private local experimentation. Weights are not downloaded without it. PBRFusion 4 is Apache-2.0. See [docs/LICENSING.md](docs/LICENSING.md).

ComfyUI listens on `127.0.0.1` only. Models, the job database, and outputs live in `/mnt/m/prk-hd-remaster` (see [config/default.yaml](config/default.yaml)).

## Production

```bash
uv run prk-remaster run --cohort first-1076 --preset aggressive --resume
uv run prk-remaster status --run latest
uv run prk-remaster gallery --run latest
uv run prk-remaster report --run latest
uv run prk-remaster pack build --run latest --name prk-hd-v01
uv run prk-remaster pack install --dry-run --prk-root "$HOME/Games/Project-Rubi-Ka/client" \
  --pack /mnt/m/prk-hd-remaster/dist/prk-hd-v01
uv run prk-remaster pack install --prk-root "$HOME/Games/Project-Rubi-Ka/client" \
  --pack /mnt/m/prk-hd-remaster/dist/prk-hd-v01
```

`--limit` is for a smoke run, not the production target. Close the game before installing. Disable the pack with `pack disable --name prk-hd-v01 --yes` and remove it with `pack uninstall --name prk-hd-v01 --yes`.

In-game before/after screenshots are a user session. Steps are in [docs/OPERATIONS.md](docs/OPERATIONS.md).

## Tests

```bash
uv run pytest
```

Tests use synthetic images and a tiny SQLite fixture. They do not read game art except an optional check that the local catalogue still contains 1,076 `risk=auto` rows.

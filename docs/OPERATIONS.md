# Operations

Paths below match this machine. Override them with `--prk-root`, `--csv`, `--comfy-url`, or `PRK_WORK_ROOT`.

## Daily commands

```bash
uv run prk-remaster doctor --prk-root "$HOME/Games/Project-Rubi-Ka/client"
uv run prk-remaster catalogue import \
  --csv "$HOME/Games/ao-vk/TEMP/texture-catalogue/assets.csv"
uv run prk-remaster catalogue reconcile --prk-root "$HOME/Games/Project-Rubi-Ka/client"
uv run prk-remaster run --cohort first-1076 --preset aggressive --resume
uv run prk-remaster status --run latest
uv run prk-remaster gallery --run latest
uv run prk-remaster report --run latest
uv run prk-remaster pack build --run latest --name prk-hd-v01
```

`run` is resumable. A crashed ComfyUI process or an interrupted Python process leaves each texture at its last finished stage (`pending`, `extracted`, `generated`, `pbr`, `qa_passed`). Run the same command again. `--retry-failed` requeues `failed` and `blocked` jobs. `--force` is required to replace a finished result with a different recipe.

`reconstruct` is the v3 semantic pass. It writes a new run and does not modify the v1 run or the v2 salvage run. A `STOP` file in the run directory, or SIGTERM, finishes the current texture and returns. Resume with the command in that run's `RESUME.txt`:

```bash
uv run prk-remaster reconstruct --from-run <v1-run-id> --run <v3-run-id> --resume
```

`--limit 1` checks that a real image comes back. It is not the production cohort.

`reconstruct --pack-name prk-hd-v05` draws the cohort again with the current prompts. Flat surfaces use the worn-stone wording from the shop-floor regen. Expressive surfaces use the detail wording. Leave `--pack-name` set so the run does not replace `dist/prk-hd-v03`.

`refine` is the v4 material-scale pass. It reads a finished v3 run, hardlinks expressive assets, rewrites restrained albedos, rebuilds their lighting, and packs `prk-hd-v04`. It does not modify the source run and it does not install the pack. A `STOP` file or SIGTERM finishes the current texture. Resume with:

```bash
uv run prk-remaster refine --from-run <v3-run-id> --run <v4-run-id> --resume
```

Useful filters: `--family ground:sand`, `--types 1010006`, `--ids-file ids.txt` (one `type:id` per line).

## Install and roll back

Close Project Rubi-Ka first. The installer refuses to run when an Anarchy process for this client is visible, and it refuses to replace an existing pack folder.

```bash
uv run prk-remaster pack install --dry-run \
  --prk-root "$HOME/Games/Project-Rubi-Ka/client" \
  --pack /mnt/m/prk-hd-remaster/dist/prk-hd-v01
uv run prk-remaster pack install \
  --prk-root "$HOME/Games/Project-Rubi-Ka/client" \
  --pack /mnt/m/prk-hd-remaster/dist/prk-hd-v01
```

That copies the folder to `<client>/randy-vk/packs/prk-hd-v01/` and prepends `prk-hd-v01` to `randy-vk/packs/order.txt`. ao-vk uses the first listed folder that contains any map for a texture, and every map for that texture comes from that one folder. A previous `order.txt` is copied beside it and into the work directory's `install-receipts/`.

`cd_image/rdb.db` is never opened for write. `randy31.dll` and the rest of the client are not modified.

```bash
uv run prk-remaster pack disable --name prk-hd-v01 --yes \
  --prk-root "$HOME/Games/Project-Rubi-Ka/client"
uv run prk-remaster pack uninstall --name prk-hd-v01 --yes \
  --prk-root "$HOME/Games/Project-Rubi-Ka/client"
```

Disable only removes the `order.txt` line. Uninstall also deletes that pack folder. Neither command deletes other packs. Without `--yes`, disable and uninstall print the plan and stop. `--dry-run` does the same.

Restart the client after install or disable. Packs are read at startup. The log line looks like `materials: pack 'prk-hd-v01', N files`.

## In-game comparison

This has to be done in a game session. The factory does not claim it has been done.

1. Install `prk-hd-v01` with the game closed.
2. Start Project Rubi-Ka and confirm the log names pack `prk-hd-v01`.
3. Stand on a remastered ground tile. The gallery family `ground:sand` or `ground:grass` lists ids.
4. Note graphics settings, time of day, and where you are standing.
5. Screenshot. Press Ctrl+Shift+I on the surface. The log and clipboard report `type:id`.
6. Quit, run `pack disable --name prk-hd-v01 --yes`, start the client again.
7. Return to the same place and time of day and take the matching screenshot.
8. Save both under the gallery's `in-game-evidence/` directory, with the texture id in the filename.

## Cohort rules

`first-1076` is every catalogue row with `risk=auto`. A row in that set whose name starts with `shadow`, or which carries an atlas, text, or cut-out flag, is `rejected` and is not sent to Qwen. Other catalogue risks are stored as `specialist`, `needs-classification`, `skip-ui`, `skip-mask`, `skip-placeholder`, or `decode-error`. LOD tables are counted as `skip-lod` by `catalogue reconcile` and are not remastered.

Every cohort id should finish as `packed`, `rejected`, or `blocked`. The run report lists the ones that did not succeed.

## Disk and memory

The root filesystem is nearly full. Keep weights and outputs on `/mnt/m`. Qwen and PBRFusion are not loaded together. GPU work is one image at a time.

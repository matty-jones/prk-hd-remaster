# Progress

Save point for the PRK HD remaster factory. Updated as the system is built and run.

## Environment

- Repo: `/home/blasky/Games/prk-hd-remaster`
- PRK client: `/home/blasky/Games/Project-Rubi-Ka/client` with `cd_image/rdb.db` (read-only).
- Catalogue: `/home/blasky/Games/ao-vk/TEMP/texture-catalogue/assets.csv` (20,859 rows). Cohort `risk=auto` is 1,076 textures: terrain 312, architecture 288, metal 186, interior 176, stone 114. `risk=skip` is 449 rows and stays in the import.
- ao-vk checkout: `/home/blasky/Games/ao-vk` at `ab587fd30ec3384b36a1d1dd7f99615e2ad46578`. Pack layout is `<client>/randy-vk/packs/<name>/<type>_<id>_{d,n,h,orm}.png`. `order.txt` is first-match-wins. The client has `randy-vk/fx` and `randy-vk/portals` and no `packs/` directory yet.
- GPU: NVIDIA GeForce RTX 5080, 16303 MiB, driver 615.78.08, capability 12.0. PyTorch 2.9.0+cu128 is installed on the system interpreter.
- Disk: `/` has about 48 GB free. Work root is `/mnt/m/prk-hd-remaster` (about 211 GB free). No ComfyUI install and no Qwen-Image or PBRFusion weights were present at the start.

## Implementation

The factory is a `uv` project, command `prk-remaster`. Stages are catalogue import, read-only RDB extract, Qwen edit through the official 8-step Turbo ComfyUI graph, PBRFusion normal/depth, rule-based ORM and height, QA, pack, and install. Tests use synthetic images.

## Not done yet

- ComfyUI and the locked weights are not installed until `models setup` runs.
- The 1,076-texture cohort has not been executed. Do not invent counts for it.
- In-game screenshots are pending a user session. See `docs/OPERATIONS.md`.

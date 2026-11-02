# Progress

Save point for the PRK HD remaster factory. Updated as the system is built and run.

## Environment

- Repo: `/home/blasky/Games/prk-hd-remaster`
- PRK client: `/home/blasky/Games/Project-Rubi-Ka/client` with `cd_image/rdb.db` (read-only).
- Catalogue: `/home/blasky/Games/ao-vk/TEMP/texture-catalogue/assets.csv` (20,859 rows). Cohort `risk=auto` is 1,076 textures: terrain 312, architecture 288, metal 186, interior 176, stone 114. `risk=skip` is 449 rows and stays in the import.
- ao-vk checkout: `/home/blasky/Games/ao-vk` at `ab587fd30ec3384b36a1d1dd7f99615e2ad46578`. Pack layout is `<client>/randy-vk/packs/<name>/<type>_<id>_{d,n,h,orm}.png`. `order.txt` is first-match-wins. The client has `randy-vk/fx` and `randy-vk/portals` and no `packs/` directory yet.
- GPU: NVIDIA GeForce RTX 5080, 16303 MiB, driver 615.78.08, capability 12.0. PyTorch 2.9.0+cu128 is installed on the system interpreter.
- Disk: `/` has about 48 GB free. Work root is `/mnt/m/prk-hd-remaster`. ComfyUI, weights, the registry, and runs live there.

## Implementation

The factory is a `uv` project, command `prk-remaster`. Stages are catalogue import, read-only RDB extract, Qwen edit through the official 8-step Turbo ComfyUI graph, PBRFusion normal/depth, rule-based ORM and height, QA, pack, and install. Tests use synthetic images. `uv run pytest`: 26 passed.

## Models

ComfyUI `0dbfd493` and PBRFusion nodes `70274b60` are installed under the work root. Doctor reports the GPU, the local Qwen research-license receipt, all four weight files, and the required nodes. Checksums are in `config/model-lock.yaml`. PBRFusion matches the published SHA-256.

Smoke on `1010004:5942` (run `20261031-132341-a2fa23`):

- Qwen edit 128 to 2048, label `qwen-native`, 94.9 s including load, peak 15786 MiB. Albedo SHA-256 `2d20a0e6cfb3145d543acf90fb493c4ba8343132382b91e40aa18d21dec2663a`. Seam 1.14 versus source seam 1.18.
- Radial bump probe: depth is white-high, so `height_invert` is false.
- PBRFusion at 2048 failed an allocation. The 1024 path succeeded in 3.8 s, peak 12557 MiB, labelled `pbr-1024-upscaled`. Job status `qa_passed`.

## Cohort

Run `20261031-132341-a2fa23` completed at 00:09 MST on 1 November 2026. All 1,076 jobs are `qa_passed`. None were rejected, blocked, or failed. Labels: 1,028 `qwen-native`, 48 `qwen-1024`, and 1,076 `pbr-1024-upscaled`. The resume session took 25,755 s wall, 21,037 s of GPU time, and peaked at 15,809 MiB.

Pack `dist/prk-hd-v01` is the v1 output and was not modified by the salvage pass. In-game screenshots stay pending a user session.

## Source-anchored salvage

Run `20261101-215758-d79815` (`prompt_version` v2, `config_revision` v0.2.0) reads v1 and does not write into it. `source.png` and `raw_qwen.png` are hardlinks. The anchor keeps source colour and large layout, and keeps Qwen grain. On an organic texture whose high frequencies already agree with the source (grass `1010006:2`), the anchor keeps the Qwen picture and only nudges average chroma, so a sharp edit is not pulled back to the JPEG.

Triage queued 6 regenerations, under the cap of 200. Two of those kept the new edit (`1010006:128`, `1010006:346`). The other four kept the anchored v1 image, including brock `1010006:215`, which is terrain rather than stone. Qwen2.5-VL-3B-Instruct is pinned in `config/model-lock.yaml` and cached by source SHA-256. A caption that contradicts a pixel grass reading is dropped. One caption survived, on `1010006:128`.

PBRFusion and ORM were rerun from the corrected albedos. ORM uses `effective_material` when the filename and the pixels disagree. All 1,076 jobs are `packed`. Pack `dist/prk-hd-v02` has 1,168 files including aliases, is not zipped, and is not installed. Review sheets are `runs/20261101-215758-d79815/gallery/dramatic-triptych.png` and `worst-triptych.png`.

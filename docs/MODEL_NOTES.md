# Model notes

One stack is used. There is no upscaler bake-off.

## Qwen-Image-2.1-Turbo

ComfyUI graph: [workflows/qwen-21-turbo-edit.json](../workflows/qwen-21-turbo-edit.json).

- Diffusion: `qwen_image_2.1_turbo_int8_convrot.safetensors` (Comfy-Org/Qwen-Image-2.1).
- Text encoder: `qwen3vl_8b_w4a8.safetensors`, CLIP type `qwen_image`. The smaller encoder is used because this machine has 16 GB of VRAM and 31 GB of system RAM. The 9.35 GB INT8 encoder is the larger published alternative.
- VAE: `qwen_image_2.1_vae_bf16.safetensors`.
- Sampler: euler, CFG 1, ManualSigmas `1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0` (the eight-step schedule from the official Turbo image-edit template, plus the terminal 0).
- The negative prompt is empty. CFG 1 does not use it.
- The prompt enhancer is off.
- KV cache device `cpu`, dtype `int8`, so the cache does not sit in VRAM next to the generator.
- The encoder latent is sampled directly (`resolution` 0 keeps the conditioning image size). A different latent size shifts the edit, so the pipeline does not swap in an empty latent of another size.
- Conditioning images are resized to the target, circular-shifted by half, edited, then shifted back so the tile seam sits in the middle of the generation.

If CUDA runs out of memory, the job is retried once at a 1024 long side and labelled `qwen-1024`. A bilinear enlarge is not reported as native 2K. A second out-of-memory is `blocked`.

Diffusers `QwenImage21Pipeline` on `Qwen/Qwen-Image-2.1-Turbo` is the fallback only if this ComfyUI graph cannot execute on the RTX 5080. It is not a second model comparison. `config/model-lock.yaml` records `diffusers_fallback.enabled`.

Measured on the RTX 5080 (driver 615.78.08), ComfyUI `0dbfd493`, 31 October 2026. Catalogue texture `1010004:5942` (architecture, 128x128 source) edited to 2048x2048 in 94.9 seconds including model load. Label `qwen-native`. Peak `nvidia-smi` memory during that edit was 15786 MiB. Albedo SHA-256 `2d20a0e6cfb3145d543acf90fb493c4ba8343132382b91e40aa18d21dec2663a`. The graph executed, so the Diffusers fallback stays disabled.

With DynamicVRAM left as ComfyUI ships it, the worker called `soft_empty_cache` after every prompt, and the next 2048 edit spent about 31 seconds staging weights again before the 2.6-second steps. `scripts/bootstrap-models.sh` changes that local ComfyUI checkout so the CUDA cache is dropped only when `/free` is requested. The Qwen pass can then keep the Turbo weights resident. The change is not in the ComfyUI git revision; re-running the bootstrap script reapplies it.

## PBRFusion 4

Graph: [workflows/pbrfusion4-normal-depth.json](../workflows/pbrfusion4-normal-depth.json).

- Weight: `PBRFusion4.safetensors`, published SHA-256 `83c53a5566bc7c114f17fb15baac14cc6a1f7b40cd16f72a41da5aa43c065609`.
- Node: `AIXPOLY_PBRFusion4SimpleDepthAndNormalGenerator` with `opengl_normal` true.
- It supplies a normal and a relative depth. It does not supply a learned albedo or a learned ORM.
- Qwen is freed before PBRFusion is loaded. `/free` only flags the queue, so the client waits until the Comfy process itself drops under 4096 MiB before the next prompt.
- Height is percentile-normalised filtered depth with a class relief scale. White is high. A 512 px radial bump (centre brighter than the corners in the albedo) produced depth whose centre was also brighter, extrema 35 to 234, so `height_invert` stays false. Depth SHA-256 `7c252f0fc26d1c7158901f19bb929201090bd0e66286c22d2284ad77f0640a5f`.
- Full-resolution inference on the 2048 albedo failed with a CUDA allocation error while the Comfy process was already down to about 2940 MiB. The same albedo then passed at the node's 1024 long side in 3.8 seconds, peak 12557 MiB, and the node resized the maps back to 2048. That result is labelled `pbr-1024-upscaled`. `pbr_optimize` is therefore true for this machine. A later job in the same process skips the full-resolution attempt after the first allocation failure.
- ORM is R occlusion, G roughness, B metallic. Roughness and metalness come from [config/material-rules.yaml](../config/material-rules.yaml) plus local contrast and, for painted metal, low-saturation exposed chips. Metalness is not taken from brightness. Occlusion is a weak valley term so it does not double-darken the game's ambient term.

## Source-anchored salvage

Fidelity is applied after the Turbo edit. The graph stays CFG 1 with the fixed eight-step sigmas. v1 outputs stay valid because salvage uses `prompt_version` v2 only in its own run.

`prk-remaster salvage` hardlinks the v1 source and Qwen albedo into a new run, then rebuilds `albedo_final.png`. Coarse colour and layout come from the source. Qwen contributes grain, and edges only where the source already has an edge in the same place. Filters are circular. Baked highlights are pulled down before the colour lock, and 8-pixel JPEG energy on a tiny source is not treated as detail. An organic texture that already matches the source at coarse scale and in the high band keeps the Qwen image, with a small average-chroma nudge.

Briefs come from `config/name-lexicon.yaml` plus measured colour and structure. The compiled prompt is positive-only. When the filename and the pixels disagree, `effective_material` is what ORM uses. The catalogue class on the asset is left as imported.

The captioner is `Qwen/Qwen2.5-VL-3B-Instruct`, recorded under `captioner` in `config/model-lock.yaml`. Weights live at `{work_root}/models/captioner/Qwen2.5-VL-3B-Instruct`. It runs only for an unreliable brief or a texture already queued for regeneration, from `scripts/caption_worker.py` against the ComfyUI PyTorch install. The worker uses the GPU when at least 9 GiB is free and otherwise stays on CPU. A caption that renames a pixel-read grass surface to stone or a wall is discarded. If the captioner cannot run, the lexicon and the visual measures still choose the route.

Run `20261101-215758-d79815` anchored all 1,076 v1 textures, regenerated 6, and kept 2 of those new edits. Pack `dist/prk-hd-v02` was built without a zip and without installing over `prk-hd-v01`.

## What gets measured

`doctor` and the run report record GPU name, driver, peak `nvidia-smi` memory, and wall time split from model inference time. Those numbers are written only after a real run. They are not estimated here.

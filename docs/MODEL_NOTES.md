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

## PBRFusion 4

Graph: [workflows/pbrfusion4-normal-depth.json](../workflows/pbrfusion4-normal-depth.json).

- Weight: `PBRFusion4.safetensors`, published SHA-256 `83c53a5566bc7c114f17fb15baac14cc6a1f7b40cd16f72a41da5aa43c065609`.
- Node: `AIXPOLY_PBRFusion4SimpleDepthAndNormalGenerator` with `opengl_normal` true.
- It supplies a normal and a relative depth. It does not supply a learned albedo or a learned ORM.
- Qwen is freed before PBRFusion is loaded.
- Height is percentile-normalised filtered depth with a class relief scale. White is high. `height_invert` is set from a radial bump probe when the model is available.
- ORM is R occlusion, G roughness, B metallic. Roughness and metalness come from [config/material-rules.yaml](../config/material-rules.yaml) plus local contrast and, for painted metal, low-saturation exposed chips. Metalness is not taken from brightness. Occlusion is a weak valley term so it does not double-darken the game's ambient term.

## What gets measured

`doctor` and the run report record GPU name, driver, peak `nvidia-smi` memory, and wall time split from model inference time. Those numbers are written only after a real run. They are not estimated here.

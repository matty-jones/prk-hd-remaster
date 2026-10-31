#!/usr/bin/env bash
# Idempotent ComfyUI + PBRFusion node install. Weights are downloaded by
# `prk-remaster models setup` after the Qwen Research License is accepted.
set -euo pipefail

ROOT="${PRK_WORK_ROOT:-/mnt/m/prk-hd-remaster}"
COMFY="${ROOT}/comfyui"
NODE="${COMFY}/custom_nodes/COMFYUI-PBRFusion4"

mkdir -p "${ROOT}/logs" "${COMFY}"

if [[ ! -d "${COMFY}/.git" ]]; then
  git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git "${COMFY}"
fi

if [[ ! -x "${COMFY}/.venv/bin/python" ]]; then
  python3 -m venv "${COMFY}/.venv"
fi

# shellcheck disable=SC1091
source "${COMFY}/.venv/bin/activate"
python -m pip install -U pip
python -m pip install -r "${COMFY}/requirements.txt"
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

if [[ ! -d "${NODE}/.git" ]]; then
  git clone --depth 1 https://github.com/Night1099/COMFYUI-PBRFusion4.git "${NODE}"
fi
if [[ -f "${NODE}/requirements.txt" ]]; then
  python -m pip install -r "${NODE}/requirements.txt"
fi

# Keep Turbo weights resident between prompts. /free still unloads them.
python - "${COMFY}/main.py" <<'PY'
import pathlib, sys
path = pathlib.Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
marker = "prk-remaster: drop the CUDA cache only when /free asked for it."
if marker not in text:
    old = "                    gc.collect()\n                    comfy.model_management.soft_empty_cache()\n"
    new = (
        "                    gc.collect()\n"
        "                    # prk-remaster: drop the CUDA cache only when /free asked for it.\n"
        "                    if free_memory:\n"
        "                        comfy.model_management.soft_empty_cache()\n"
    )
    if old not in text:
        raise SystemExit(f"{path} does not contain the expected cache-clear block")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
PY

echo "comfyui $(git -C "${COMFY}" rev-parse HEAD)"
echo "pbrfusion-nodes $(git -C "${NODE}" rev-parse HEAD)"

"""Describe source textures with one loaded Qwen2.5-VL-3B. The manifest is JSON lines."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

PROMPT = """You describe one tiling game texture, not a place and not a photograph.
Filename hint: {name}. It may be wrong, generic, or empty.
Catalogue class: {category}. Family: {family}.
Reply with JSON only, using these keys:
subject, material, pattern, safe_enhancements, name_agreement.
subject is the visible surface in a few words.
material is one of terrain, stone, metal, architecture, interior.
pattern is the visible arrangement.
safe_enhancements are fine surface details of that same subject.
name_agreement is true only when the filename matches the image.
Describe only what is visible.
"""


def parse_caption(text: str) -> dict:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {}
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    agreement = data.get("name_agreement")
    if isinstance(agreement, str):
        agreement = agreement.strip().lower() in {"true", "yes", "1"}
    return {
        "subject": str(data.get("subject") or ""),
        "material": str(data.get("material") or ""),
        "pattern": str(data.get("pattern") or ""),
        "safe_enhancements": str(data.get("safe_enhancements") or ""),
        "name_agreement": bool(agreement) if agreement is not None else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    rows = [json.loads(line) for line in Path(args.manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
    # The 16 GB card is shared with ComfyUI and the desktop. Use the GPU only when
    # it can hold the 3B model; otherwise stay on CPU so a full card does not abort the batch.
    free = torch.cuda.mem_get_info()[0] if torch.cuda.is_available() else 0
    if free >= 9 * 1024 ** 3:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model, torch_dtype=torch.float16, device_map="auto", max_memory={0: "8GiB", "cpu": "24GiB"},
        )
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model, torch_dtype=torch.float32)
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model)
    for row in rows:
        image = Image.open(row["image"]).convert("RGB")
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT.format(
                    name=row.get("name") or "unnamed",
                    category=row.get("category") or "unknown",
                    family=row.get("family") or "none",
                )},
            ],
        }]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt")
        inputs = {key: value.to(model.device) if hasattr(value, "to") else value for key, value in inputs.items()}
        output = model.generate(**inputs, max_new_tokens=220)
        trimmed = output[:, inputs["input_ids"].shape[1]:]
        decoded = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        payload = parse_caption(decoded)
        payload["raw"] = decoded[:800]
        Path(row["out"]).write_text(json.dumps(payload), encoding="utf-8")
        print(row.get("name") or row["image"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from PIL import Image

from prk_remaster.config import load_config
from prk_remaster.pbr import policy_for
from prk_remaster.semantics import apply_caption, build_brief, compile_prompt, sanitize_caption


def _cfg():
    return load_config()


def test_cable_prompt_describes_cables_and_not_a_factory_scene():
    cfg = _cfg()
    image = Image.new("RGB", (32, 64), (120, 90, 40))
    asset = {
        "name": "dock_factory_metal_cable.png",
        "family": "dock_factory",
        "material_class": "metal",
        "source_width": 32,
        "source_height": 64,
    }
    brief = build_brief(asset, image, cfg["name_lexicon"])
    text = compile_prompt(asset, brief, cfg["family_prompts"], 512, 1024)
    lowered = text.lower()
    assert brief["subject_class"] == "cable"
    assert "cable" in lowered
    assert "strand" in lowered
    assert "a factory" not in lowered
    assert "scene" not in lowered
    assert "photograph" not in lowered
    rules = cfg["material_rules"]
    assert policy_for(asset["name"], brief["effective_material"], rules)["metallic_mode"] == "bare"


def test_brock_green_image_is_not_described_as_stone():
    cfg = _cfg()
    image = Image.new("RGB", (64, 64), (70, 150, 45))
    asset = {"name": "brock", "family": "ground:brock", "material_class": "stone", "source_width": 64, "source_height": 64}
    brief = build_brief(asset, image, cfg["name_lexicon"])
    text = compile_prompt(asset, brief, cfg["family_prompts"], 256, 256).lower()
    assert brief["material_conflict"]
    assert brief["effective_material"] == "terrain"
    assert "stone" not in text
    assert "grass" in text
    updated = apply_caption(brief, {
        "subject": "Stone Pile",
        "material": "stone",
        "pattern": "grass patches on a stone surface",
        "safe_enhancements": "smoothed edges of the stones",
        "name_agreement": False,
    })
    revised = compile_prompt(asset, updated, cfg["family_prompts"], 256, 256).lower()
    assert updated["effective_material"] == "terrain"
    assert "stone" not in revised
    assert "grass" in revised


def test_empty_filename_does_not_invent_a_subject():
    cfg = _cfg()
    image = Image.new("RGB", (32, 32), (90, 90, 90))
    asset = {"name": "", "family": "", "material_class": "interior", "source_width": 32, "source_height": 32}
    brief = build_brief(asset, image, cfg["name_lexicon"])
    text = compile_prompt(asset, brief, cfg["family_prompts"], 128, 128).lower()
    assert brief["name_confidence"] < 0.35
    assert brief["subject"] == ""
    assert "cable" not in text
    assert "grass" not in text
    assert "limestone" not in text


def test_caption_cannot_smuggle_a_scene_into_a_confident_name():
    cfg = _cfg()
    image = Image.new("RGB", (32, 64), (120, 90, 40))
    asset = {"name": "dock_factory_metal_cable.png", "family": "dock_factory", "material_class": "metal"}
    brief = build_brief(asset, image, cfg["name_lexicon"])
    updated = apply_caption(brief, {
        "subject": "a photograph of a factory floor",
        "safe_enhancements": "add a person",
        "name_agreement": True,
    })
    assert updated["subject_class"] == "cable"
    assert "photograph" not in updated["subject"].lower()
    assert sanitize_caption("a photograph of a dock") == ""
    assert sanitize_caption("braided cable sheath") == "braided cable sheath"

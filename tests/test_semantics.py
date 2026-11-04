from PIL import Image

from prk_remaster.config import load_config
from prk_remaster.pbr import policy_for
from prk_remaster.semantics import apply_caption, apply_v3_detail, build_brief, compile_prompt, sanitize_caption


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


def test_v3_prompt_asks_for_braid_and_treats_the_reference_as_layout():
    cfg = _cfg()
    image = Image.new("RGB", (32, 64), (120, 90, 40))
    asset = {
        "name": "dock_factory_metal_cable.png",
        "family": "dock_factory",
        "material_class": "metal",
        "source_width": 32,
        "source_height": 64,
    }
    brief = apply_v3_detail(build_brief(asset, image, cfg["name_lexicon"]), cfg["family_prompts"])
    text = compile_prompt(asset, brief, cfg["family_prompts"], 512, 1024, version="v3").lower()
    assert "layout reference" in text
    assert "braid" in text
    assert "strand" in text
    assert "does not copy the reference blur" in text
    assert "dock fitting" not in text


def test_v3_dirt_prompt_asks_for_hard_grains():
    cfg = _cfg()
    image = Image.new("RGB", (32, 32), (90, 70, 40))
    asset = {"name": "std-dirt32x32.png", "family": "std_dirt", "material_class": "terrain", "source_width": 32, "source_height": 32}
    brief = apply_v3_detail(build_brief(asset, image, cfg["name_lexicon"]), cfg["family_prompts"])
    text = compile_prompt(asset, brief, cfg["family_prompts"], 2048, 2048, version="v3").lower()
    assert brief["subject_class"] == "dirt"
    assert "grain" in text
    assert "not a soft blur" in text


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


def test_v3_floor_prompt_keeps_meso_and_quiets_micro():
    cfg = _cfg()
    image = Image.new("RGB", (32, 32), (140, 120, 70))
    asset = {"name": "", "family": "", "material_class": "interior", "source_width": 128, "source_height": 128}
    brief = apply_v3_detail(build_brief(asset, image, cfg["name_lexicon"]), cfg["family_prompts"], cfg["scale_policy"])
    text = compile_prompt(asset, brief, cfg["family_prompts"], 2048, 2048, version="v3").lower()
    assert brief["scale_mode"] == "restrained"
    assert "quiet" in text
    assert "compression" in text
    assert "pixel blocks" in text
    assert "sharp fine grain" not in text


def test_v3_cable_stays_expressive_when_the_class_is_known():
    cfg = _cfg()
    image = Image.new("RGB", (32, 64), (120, 90, 40))
    asset = {
        "name": "dock_factory_metal_cable.png",
        "family": "dock_factory",
        "material_class": "metal",
        "source_width": 32,
        "source_height": 64,
    }
    brief = apply_v3_detail(build_brief(asset, image, cfg["name_lexicon"]), cfg["family_prompts"], cfg["scale_policy"])
    text = compile_prompt(asset, brief, cfg["family_prompts"], 512, 1024, version="v3").lower()
    assert brief["scale_mode"] == "expressive"
    assert "braid" in text
    assert "does not copy the reference blur" in text


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

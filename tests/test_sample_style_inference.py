from __future__ import annotations

import sys
from pathlib import Path

import yaml

from klein4b.sample_style_inference import (
    DEFAULT_BEST_LORA_PATH,
    build_ai_toolkit_command,
    negative_prompt_additions_for_eye_state,
    render_sample_style_config,
)


def test_render_sample_style_config_uses_best_klein_settings(tmp_path: Path) -> None:
    reference = tmp_path / "selfie.png"
    lora = tmp_path / "weights.safetensors"
    output_dir = tmp_path / "out"

    config_text = render_sample_style_config(
        name="structured_marble_test",
        training_folder=output_dir,
        reference_path=reference,
        prompt="Change image 1 into a <mrblbust> from the reference portrait, test prompt",
        lora_path=lora,
    )
    payload = yaml.safe_load(config_text)
    process = payload["config"]["process"][0]
    sample = process["sample"]

    assert process["network"]["pretrained_lora_path"] == str(lora)
    assert process["network"]["linear"] == 64
    assert process["network"]["conv"] == 32
    assert process["train"]["steps"] == 0
    assert process["train"]["disable_sampling"] is False
    assert process["model"]["arch"] == "flux2_klein_4b"
    assert sample["width"] == 768
    assert sample["height"] == 1024
    assert sample["seed"] == 7
    assert sample["guidance_scale"] == 2.5
    assert sample["sample_steps"] == 24
    assert sample["samples"][0]["ctrl_img_1"] == str(reference)
    assert sample["samples"][0]["prompt"].startswith("Change image 1 into a <mrblbust>")
    assert "head-only crop" in sample["neg"]
    assert "individual hair strands" in sample["neg"]
    assert "pupils" in sample["neg"]
    assert "irises" in sample["neg"]
    assert "painted eyes" in sample["neg"]
    assert "modern design" in sample["neg"]
    assert "contemporary decorative details" in sample["neg"]
    assert "hair bow" in sample["neg"]
    assert "ribbon" in sample["neg"]
    assert "hair clip" in sample["neg"]
    assert "headband" in sample["neg"]
    assert "circlet" in sample["neg"]
    assert "tiara" in sample["neg"]
    assert "headscarf" in sample["neg"]
    assert "turban" in sample["neg"]
    assert "baseball cap" in sample["neg"]
    assert "bright white marble" in sample["neg"]
    assert "clean white stone" in sample["neg"]
    assert "overbright highlights" in sample["neg"]
    assert "high-key lighting" in sample["neg"]
    assert "smooth marble finish" in sample["neg"]
    assert "teeth" in sample["neg"]
    assert "open mouth" in sample["neg"]
    assert "parted lips" in sample["neg"]


def test_render_sample_style_config_can_add_conditional_negative_terms(tmp_path: Path) -> None:
    reference = tmp_path / "selfie.png"
    lora = tmp_path / "weights.safetensors"
    output_dir = tmp_path / "out"

    config_text = render_sample_style_config(
        name="structured_marble_test",
        training_folder=output_dir,
        reference_path=reference,
        prompt="Change image 1 into a <mrblbust> from the reference portrait",
        lora_path=lora,
        negative_prompt_additions=("closed eyes", "lowered eyelids"),
    )
    payload = yaml.safe_load(config_text)
    negative_prompt = payload["config"]["process"][0]["sample"]["neg"]

    assert "closed eyes" in negative_prompt
    assert "lowered eyelids" in negative_prompt


def test_open_eye_negative_prompt_additions_are_conditional() -> None:
    open_terms = negative_prompt_additions_for_eye_state("open")

    assert "closed eyes" in open_terms
    assert "lowered eyelids" in open_terms
    assert negative_prompt_additions_for_eye_state("unknown") == open_terms
    assert negative_prompt_additions_for_eye_state("closed") == ()


def test_default_best_lora_path_points_to_step_6500() -> None:
    assert DEFAULT_BEST_LORA_PATH.name.endswith("000006500.safetensors")


def test_build_ai_toolkit_command_points_to_run_py(tmp_path: Path) -> None:
    command = build_ai_toolkit_command(tmp_path / "vendor" / "ai-toolkit", tmp_path / "config.yaml")

    assert command == [
        sys.executable,
        str(tmp_path / "vendor" / "ai-toolkit" / "run.py"),
        str(tmp_path / "config.yaml"),
    ]

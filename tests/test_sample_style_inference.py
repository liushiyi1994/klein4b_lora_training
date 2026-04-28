from __future__ import annotations

import sys
from pathlib import Path

import yaml

from klein4b.sample_style_inference import (
    DEFAULT_BEST_LORA_PATH,
    build_ai_toolkit_command,
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


def test_default_best_lora_path_points_to_step_6500() -> None:
    assert DEFAULT_BEST_LORA_PATH.name.endswith("000006500.safetensors")


def test_build_ai_toolkit_command_points_to_run_py(tmp_path: Path) -> None:
    command = build_ai_toolkit_command(tmp_path / "vendor" / "ai-toolkit", tmp_path / "config.yaml")

    assert command == [
        sys.executable,
        str(tmp_path / "vendor" / "ai-toolkit" / "run.py"),
        str(tmp_path / "config.yaml"),
    ]

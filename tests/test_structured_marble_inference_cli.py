from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import yaml
from PIL import Image

VALID_PLAN = {
    "reference_identity": {
        "age_band": "child",
        "gender_presentation": "feminine",
        "eye_state": "open",
        "head_pose": "slight left-facing three-quarter head pose",
        "face_structure": ["rounded cheeks", "small compact jaw", "short nose"],
        "hair_or_headwear": ["short bob-like hair silhouette with soft bangs"],
        "broad_expression": "calm closed-mouth expression",
    },
    "target_style": {
        "bust_framing": "centered chest-up Greek marble statue bust",
        "statue_angle": "matching the reference head angle",
        "drapery_and_torso": ["simple classical drapery", "visible shoulders"],
        "headpiece_or_ornament": [],
    },
    "safety_overrides": {
        "identity_source_policy": "use only the reference portrait for identity",
        "eye_policy": "blank sculpted stone eyes or closed carved eyelids",
        "material_policy": "all hair and ornaments are carved marble",
        "banned_details": ["pupils", "irises", "catchlights"],
    },
}


def load_cli_module():
    module_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "run_structured_marble_inference.py"
    )
    spec = importlib.util.spec_from_file_location("run_structured_marble_inference", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_replays_cached_prompt_plan_without_openai_or_ai_toolkit(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = load_cli_module()
    reference = tmp_path / "selfie.png"
    pseudo_target = tmp_path / "pseudo.png"
    previous_best = tmp_path / "previous.jpg"
    prompt_plan_json = tmp_path / "plan.json"
    output_dir = tmp_path / "out"
    Image.new("RGB", (12, 16), "red").save(reference)
    Image.new("RGB", (12, 16), "blue").save(pseudo_target)
    Image.new("RGB", (12, 16), "green").save(previous_best)
    prompt_plan_json.write_text(json.dumps(VALID_PLAN), encoding="utf-8")

    def fail_openai(*args: object, **kwargs: object) -> object:
        raise AssertionError("OpenAI planner must not run in cached replay")

    monkeypatch.setattr(module, "plan_prompt_with_openai", fail_openai)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_structured_marble_inference.py",
            "--reference",
            str(reference),
            "--pseudo-target",
            str(pseudo_target),
            "--previous-best",
            str(previous_best),
            "--prompt-plan-json",
            str(prompt_plan_json),
            "--skip-openai",
            "--no-run-ai-toolkit",
            "--output-dir",
            str(output_dir),
        ],
    )

    module.main()

    run_config = json.loads((output_dir / "run_config.json").read_text(encoding="utf-8"))
    assert run_config["reference"] == str(reference)
    assert run_config["pseudo_target"] == str(pseudo_target)
    assert run_config["pseudo_target_role"] == "comparison-only"
    assert run_config["used_openai"] is False
    assert (output_dir / "prompt_plan.json").exists()
    assert (
        (output_dir / "prompt.txt")
        .read_text(encoding="utf-8")
        .startswith("Change image 1 into a <mrblbust>")
    )
    assert (output_dir / "sample_style_inference.yaml").exists()
    sample_config = yaml.safe_load(
        (output_dir / "sample_style_inference.yaml").read_text(encoding="utf-8")
    )
    negative_prompt = sample_config["config"]["process"][0]["sample"]["neg"]
    assert "closed eyes" in negative_prompt
    assert "lowered eyelids" in negative_prompt
    assert (output_dir / "contact_sheet.jpg").exists()


def test_cli_runs_ai_toolkit_through_python_and_records_generated_output(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = load_cli_module()
    reference = tmp_path / "selfie.png"
    prompt_plan_json = tmp_path / "plan.json"
    output_dir = tmp_path / "out"
    ai_toolkit_dir = tmp_path / "vendor" / "ai-toolkit"
    Image.new("RGB", (12, 16), "red").save(reference)
    prompt_plan_json.write_text(json.dumps(VALID_PLAN), encoding="utf-8")
    ai_toolkit_dir.mkdir(parents=True)
    calls: list[dict[str, object]] = []

    def fake_run(command: list[str], check: bool) -> None:
        calls.append({"command": command, "check": check})
        samples_dir = output_dir / "structured_marble_inference" / "samples"
        samples_dir.mkdir(parents=True)
        Image.new("RGB", (12, 16), "purple").save(samples_dir / "sample.jpg")

    def fail_openai(*args: object, **kwargs: object) -> object:
        raise AssertionError("OpenAI planner must not run in cached replay")

    monkeypatch.setattr(module, "plan_prompt_with_openai", fail_openai)
    monkeypatch.setattr(module.subprocess, "run", fake_run)

    module.main(
        [
            "--reference",
            str(reference),
            "--prompt-plan-json",
            str(prompt_plan_json),
            "--skip-openai",
            "--output-dir",
            str(output_dir),
            "--ai-toolkit-dir",
            str(ai_toolkit_dir),
        ]
    )

    generated_path = output_dir / "generated.jpg"
    run_config = json.loads((output_dir / "run_config.json").read_text(encoding="utf-8"))

    assert calls == [
        {
            "command": [
                sys.executable,
                str(ai_toolkit_dir / "run.py"),
                str(output_dir / "sample_style_inference.yaml"),
            ],
            "check": True,
        }
    ]
    assert generated_path.exists()
    assert run_config["generated"] == str(generated_path)
    assert run_config["ai_toolkit_dir"] == str(ai_toolkit_dir)
    assert run_config["ai_toolkit_command"] == calls[0]["command"]
    assert (output_dir / "contact_sheet.jpg").exists()


def test_cli_can_plan_with_bedrock_nova_provider(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = load_cli_module()
    reference = tmp_path / "selfie.png"
    output_dir = tmp_path / "out"
    Image.new("RGB", (12, 16), "red").save(reference)
    calls: list[dict[str, object]] = []

    def fail_openai(*args: object, **kwargs: object) -> object:
        raise AssertionError("OpenAI planner must not run for Bedrock Nova provider")

    def fake_bedrock_plan(**kwargs: object) -> object:
        calls.append(dict(kwargs))
        return module.parse_prompt_plan(VALID_PLAN)

    monkeypatch.setattr(module, "plan_prompt_with_openai", fail_openai)
    monkeypatch.setattr(module, "plan_prompt_with_bedrock_nova", fake_bedrock_plan)

    module.main(
        [
            "--reference",
            str(reference),
            "--output-dir",
            str(output_dir),
            "--planner-provider",
            "bedrock-nova",
            "--model",
            "us.amazon.nova-2-lite-v1:0",
            "--aws-region",
            "us-east-1",
            "--no-run-ai-toolkit",
        ]
    )

    run_config = json.loads((output_dir / "run_config.json").read_text(encoding="utf-8"))

    assert calls == [
        {
            "reference_path": reference,
            "model": "us.amazon.nova-2-lite-v1:0",
            "region_name": "us-east-1",
        }
    ]
    assert run_config["planner_provider"] == "bedrock-nova"
    assert run_config["model"] == "us.amazon.nova-2-lite-v1:0"
    assert run_config["aws_region"] == "us-east-1"
    assert run_config["used_openai"] is False


def test_cli_can_preprocess_reference_before_planning_and_sampling(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = load_cli_module()
    reference = tmp_path / "selfie.png"
    prompt_plan_json = tmp_path / "plan.json"
    output_dir = tmp_path / "out"
    Image.new("RGB", (12, 16), "red").save(reference)
    prompt_plan_json.write_text(json.dumps(VALID_PLAN), encoding="utf-8")
    preprocess_calls: list[dict[str, object]] = []

    def fake_preprocess_reference(**kwargs: object) -> object:
        preprocess_calls.append(dict(kwargs))
        effective_reference = output_dir / "reference_preprocessed.jpg"
        Image.new("RGB", (9, 12), "orange").save(effective_reference)

        class Result:
            effective_reference_path = effective_reference
            metadata = {"enabled": True}

        return Result()

    monkeypatch.setattr(module, "preprocess_reference_image", fake_preprocess_reference)

    module.main(
        [
            "--reference",
            str(reference),
            "--prompt-plan-json",
            str(prompt_plan_json),
            "--skip-openai",
            "--no-run-ai-toolkit",
            "--output-dir",
            str(output_dir),
            "--preprocess-reference",
            "--scrfd-model",
            str(tmp_path / "det.onnx"),
        ]
    )

    run_config = json.loads((output_dir / "run_config.json").read_text(encoding="utf-8"))
    sample_config = yaml.safe_load(
        (output_dir / "sample_style_inference.yaml").read_text(encoding="utf-8")
    )

    assert preprocess_calls == [
        {
            "reference_path": reference,
            "output_path": output_dir / "reference_preprocessed.jpg",
            "metadata_path": output_dir / "preprocess_metadata.json",
            "detector": preprocess_calls[0]["detector"],
        }
    ]
    assert run_config["reference_original"] == str(reference)
    assert run_config["reference"] == str(output_dir / "reference_preprocessed.jpg")
    assert run_config["preprocess_reference"] is True
    assert sample_config["config"]["process"][0]["sample"]["samples"][0]["ctrl_img_1"] == str(
        output_dir / "reference_preprocessed.jpg"
    )


def test_cli_rejects_skip_openai_without_prompt_plan_json(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = load_cli_module()
    reference = tmp_path / "selfie.png"
    output_dir = tmp_path / "out"
    Image.new("RGB", (12, 16), "red").save(reference)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_structured_marble_inference.py",
            "--reference",
            str(reference),
            "--skip-openai",
            "--output-dir",
            str(output_dir),
        ],
    )

    try:
        module.main()
    except SystemExit as error:
        assert error.code == 2
    else:
        raise AssertionError("--skip-openai without --prompt-plan-json must fail")

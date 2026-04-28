from __future__ import annotations

import base64
import importlib.util
import json
from io import BytesIO
from pathlib import Path

from PIL import Image

from klein4b.marble_prompt_planning import parse_prompt_plan
from klein4b.sagemaker_inference import (
    GenerationResult,
    SageMakerMarbleModel,
    input_fn,
    model_fn,
    output_fn,
    predict_fn,
)

VALID_PLAN = {
    "reference_identity": {
        "age_band": "child",
        "gender_presentation": "feminine",
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


def png_bytes(color: str = "red") -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (12, 16), color).save(buffer, format="PNG")
    return buffer.getvalue()


def test_input_fn_decodes_json_base64_and_rejects_pseudo_targets() -> None:
    image_bytes = png_bytes()
    payload = {
        "request_id": "case-001",
        "image_base64": base64.b64encode(image_bytes).decode("ascii"),
        "metadata": {"source": "unit-test"},
    }

    request = input_fn(json.dumps(payload).encode("utf-8"), "application/json")

    assert request.request_id == "case-001"
    assert request.image_bytes == image_bytes
    assert request.image_format == "png"
    assert request.metadata == {"source": "unit-test"}

    payload["pseudo_target_base64"] = payload["image_base64"]
    try:
        input_fn(json.dumps(payload).encode("utf-8"), "application/json")
    except ValueError as error:
        assert "pseudo target" in str(error)
    else:
        raise AssertionError("production input_fn must reject pseudo-target inputs")


def test_input_fn_accepts_raw_image_bytes() -> None:
    image_bytes = png_bytes("blue")

    request = input_fn(image_bytes, "image/png")

    assert request.image_bytes == image_bytes
    assert request.image_format == "png"
    assert request.request_id.startswith("request-")


def test_model_fn_loads_defaults_from_model_dir_and_environment(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    lora = model_dir / "marble.safetensors"
    lora.write_bytes(b"fake-weights")
    output_root = tmp_path / "outputs"
    ai_toolkit_dir = tmp_path / "ai-toolkit"
    ai_toolkit_dir.mkdir()
    monkeypatch.setenv("KLEIN4B_OUTPUT_ROOT", str(output_root))
    monkeypatch.setenv("KLEIN4B_AI_TOOLKIT_DIR", str(ai_toolkit_dir))
    monkeypatch.setenv("AWS_REGION", "us-east-2")

    model = model_fn(str(model_dir))

    assert model.output_root == output_root
    assert model.lora_path == lora
    assert model.ai_toolkit_dir == ai_toolkit_dir
    assert model.planner_provider == "bedrock-nova"
    assert model.planner_model == "us.amazon.nova-2-lite-v1:0"
    assert model.aws_region == "us-east-2"


def test_predict_fn_writes_artifacts_and_returns_json_ready_prediction(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model = SageMakerMarbleModel(
        output_root=tmp_path / "outputs",
        lora_path=tmp_path / "weights.safetensors",
        ai_toolkit_dir=tmp_path / "ai-toolkit",
        planner_provider="bedrock-nova",
        planner_model="us.amazon.nova-2-lite-v1:0",
        aws_region="us-east-2",
    )
    model.lora_path.write_bytes(b"fake-weights")
    model.ai_toolkit_dir.mkdir()
    calls: list[dict[str, object]] = []

    def fake_plan(**kwargs: object) -> object:
        calls.append(dict(kwargs))
        return parse_prompt_plan(VALID_PLAN)

    def fake_generation(**kwargs: object) -> GenerationResult:
        output_dir = kwargs["output_dir"]
        assert isinstance(output_dir, Path)
        samples_dir = output_dir / "structured_marble_inference" / "samples"
        samples_dir.mkdir(parents=True)
        generated = samples_dir / "sample.jpg"
        Image.new("RGB", (12, 16), "purple").save(generated)
        return GenerationResult(
            generated_path=generated,
            wall_seconds=4.5,
            sampler_seconds=1.25,
            log_path=output_dir / "ai_toolkit.log",
        )

    monkeypatch.setattr("klein4b.sagemaker_inference.plan_prompt_with_bedrock_nova", fake_plan)
    monkeypatch.setattr(
        "klein4b.sagemaker_inference._run_ai_toolkit_generation",
        fake_generation,
    )

    request = input_fn(
        json.dumps(
            {
                "request_id": "prod-case",
                "image_base64": base64.b64encode(png_bytes()).decode("ascii"),
            }
        ).encode("utf-8"),
        "application/json",
    )

    prediction = predict_fn(request, model)
    response = json.loads(output_fn(prediction, "application/json"))

    output_dir = model.output_root / "prod-case"
    assert calls == [
        {
            "reference_path": output_dir / "reference.png",
            "model": "us.amazon.nova-2-lite-v1:0",
            "region_name": "us-east-2",
        }
    ]
    assert (output_dir / "reference.png").exists()
    assert (output_dir / "prompt_plan.json").exists()
    assert (output_dir / "prompt.txt").exists()
    assert (output_dir / "sample_style_inference.yaml").exists()
    assert (output_dir / "generated.jpg").exists()
    assert (output_dir / "run_config.json").exists()
    assert (output_dir / "timing.json").exists()
    assert "pseudo_target" not in (output_dir / "run_config.json").read_text(encoding="utf-8")
    assert response["request_id"] == "prod-case"
    assert response["content_type"] == "image/jpeg"
    assert response["planner_provider"] == "bedrock-nova"
    assert response["model"] == "us.amazon.nova-2-lite-v1:0"
    assert response["timings"]["generate_wall_seconds"] == 4.5
    assert response["timings"]["sampler_seconds"] == 1.25
    assert base64.b64decode(response["image_base64"]) == prediction.image_bytes


def test_output_fn_can_return_raw_jpeg(tmp_path: Path) -> None:
    model = SageMakerMarbleModel(
        output_root=tmp_path,
        lora_path=tmp_path / "weights.safetensors",
        ai_toolkit_dir=tmp_path / "ai-toolkit",
        planner_provider="bedrock-nova",
        planner_model="us.amazon.nova-2-lite-v1:0",
        aws_region="us-east-2",
    )
    request = input_fn(png_bytes(), "image/png")
    generated = tmp_path / "generated.jpg"
    Image.new("RGB", (12, 16), "purple").save(generated)
    prediction = model.build_prediction(
        request=request,
        output_dir=tmp_path,
        generated_path=generated,
        prompt_plan=VALID_PLAN,
        prompt="prompt",
        timings={"total_seconds": 1.0},
    )

    assert output_fn(prediction, "image/jpeg") == prediction.image_bytes


def test_sagemaker_entry_point_re_exports_handler_functions() -> None:
    entry_point = Path(__file__).resolve().parents[1] / "sagemaker" / "inference.py"
    spec = importlib.util.spec_from_file_location("sagemaker_inference_entry_point", entry_point)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.model_fn is model_fn
    assert module.input_fn is input_fn
    assert module.predict_fn is predict_fn
    assert module.output_fn is output_fn

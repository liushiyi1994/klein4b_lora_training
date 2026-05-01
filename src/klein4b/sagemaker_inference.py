from __future__ import annotations

import base64
import json
import os
import re
import shutil
import subprocess
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from PIL import Image

from klein4b.marble_prompt_planning import (
    parse_prompt_plan,
    plan_prompt_with_bedrock_nova,
    plan_prompt_with_openai,
    render_marble_prompt,
)
from klein4b.reference_preprocessing import ScrfdFaceDetector, preprocess_reference_image
from klein4b.sample_style_inference import (
    build_ai_toolkit_command,
    find_latest_sample,
    negative_prompt_additions_for_eye_state,
    render_sample_style_config,
)

CONFIG_NAME = "structured_marble_inference"
DEFAULT_OPENAI_MODEL = "gpt-5.4-mini"
DEFAULT_BEDROCK_NOVA_MODEL = "us.amazon.nova-2-lite-v1:0"
DEFAULT_PLANNER_PROVIDER = "bedrock-nova"
DEFAULT_AWS_REGION = "us-east-2"
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class SageMakerInferenceRequest:
    image_bytes: bytes
    image_format: str
    request_id: str
    metadata: dict[str, object]


@dataclass(frozen=True)
class GenerationResult:
    generated_path: Path
    wall_seconds: float
    sampler_seconds: float | None
    log_path: Path


@dataclass(frozen=True)
class SageMakerInferencePrediction:
    request_id: str
    image_bytes: bytes
    content_type: str
    prompt_plan: dict[str, object]
    prompt: str
    timings: dict[str, float]
    output_dir: Path
    planner_provider: str
    model: str


@dataclass(frozen=True)
class SageMakerMarbleModel:
    output_root: Path
    lora_path: Path
    ai_toolkit_dir: Path
    planner_provider: str
    planner_model: str
    aws_region: str
    preprocess_reference: bool = False
    scrfd_model_path: Path | None = None

    def build_prediction(
        self,
        *,
        request: SageMakerInferenceRequest,
        output_dir: Path,
        generated_path: Path,
        prompt_plan: Mapping[str, object],
        prompt: str,
        timings: Mapping[str, float],
    ) -> SageMakerInferencePrediction:
        return SageMakerInferencePrediction(
            request_id=request.request_id,
            image_bytes=generated_path.read_bytes(),
            content_type="image/jpeg",
            prompt_plan=dict(prompt_plan),
            prompt=prompt,
            timings=dict(timings),
            output_dir=output_dir,
            planner_provider=self.planner_provider,
            model=self.planner_model,
        )


def model_fn(model_dir: str) -> SageMakerMarbleModel:
    return load_model(model_dir=Path(model_dir))


def input_fn(
    request_body: bytes | str,
    request_content_type: str,
) -> SageMakerInferenceRequest:
    content_type = _normalize_media_type(request_content_type)
    body = _body_to_bytes(request_body)
    if content_type == "application/json":
        return _input_from_json(body)
    if content_type in {"application/x-image", "image/jpeg", "image/png", "image/webp"}:
        image_format = _detect_image_format(body)
        return SageMakerInferenceRequest(
            image_bytes=body,
            image_format=image_format,
            request_id=f"request-{uuid.uuid4().hex}",
            metadata={},
        )
    raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(
    input_data: SageMakerInferenceRequest,
    model: SageMakerMarbleModel,
) -> SageMakerInferencePrediction:
    total_start = time.perf_counter()
    output_dir = model.output_root / input_data.request_id
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_original_path, reference_path, preprocess_metadata_path = _prepare_reference_image(
        input_data=input_data,
        model=model,
        output_dir=output_dir,
    )

    plan_start = time.perf_counter()
    prompt_plan = _plan_prompt(reference_path=reference_path, model=model)
    plan_seconds = time.perf_counter() - plan_start
    prompt_plan_payload = _prompt_plan_to_payload(prompt_plan)

    render_start = time.perf_counter()
    prompt_plan_path = output_dir / "prompt_plan.json"
    prompt_plan_path.write_text(
        json.dumps(prompt_plan_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    prompt = render_marble_prompt(prompt_plan)
    prompt_path = output_dir / "prompt.txt"
    prompt_path.write_text(prompt + "\n", encoding="utf-8")

    ai_toolkit_config_path = output_dir / "sample_style_inference.yaml"
    ai_toolkit_config_path.write_text(
        render_sample_style_config(
            name=CONFIG_NAME,
            training_folder=output_dir,
            reference_path=reference_path,
            prompt=prompt,
            lora_path=model.lora_path,
            negative_prompt_additions=negative_prompt_additions_for_eye_state(
                prompt_plan.reference_identity.eye_state
            ),
        ),
        encoding="utf-8",
    )
    command = build_ai_toolkit_command(model.ai_toolkit_dir, ai_toolkit_config_path)
    render_config_seconds = time.perf_counter() - render_start

    generation = _run_ai_toolkit_generation(
        command=command,
        output_dir=output_dir,
        samples_dir=output_dir / CONFIG_NAME / "samples",
        log_path=output_dir / "ai_toolkit.log",
    )
    generated_path = _copy_generated_jpeg(generation.generated_path, output_dir / "generated.jpg")

    timings = {
        "plan_seconds": plan_seconds,
        "render_config_seconds": render_config_seconds,
        "generate_wall_seconds": generation.wall_seconds,
        "total_seconds": time.perf_counter() - total_start,
    }
    if generation.sampler_seconds is not None:
        timings["sampler_seconds"] = generation.sampler_seconds
        timings["generation_overhead_seconds"] = (
            generation.wall_seconds - generation.sampler_seconds
        )

    timing_path = output_dir / "timing.json"
    timing_path.write_text(json.dumps(timings, indent=2) + "\n", encoding="utf-8")

    run_config = {
        "reference_original": str(reference_original_path),
        "reference": str(reference_path),
        "preprocess_reference": model.preprocess_reference,
        "preprocess_metadata": _optional_path(preprocess_metadata_path),
        "lora": str(model.lora_path),
        "planner_provider": model.planner_provider,
        "model": model.planner_model,
        "aws_region": model.aws_region,
        "used_openai": model.planner_provider == "openai",
        "ai_toolkit_dir": str(model.ai_toolkit_dir),
        "ai_toolkit_command": command,
        "ai_toolkit_config": str(ai_toolkit_config_path),
        "prompt": str(prompt_path),
        "prompt_plan": str(prompt_plan_path),
        "generated": str(generated_path),
        "timing": str(timing_path),
        "request_metadata": input_data.metadata,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2) + "\n",
        encoding="utf-8",
    )

    return model.build_prediction(
        request=input_data,
        output_dir=output_dir,
        generated_path=generated_path,
        prompt_plan=prompt_plan_payload,
        prompt=prompt,
        timings=timings,
    )


def output_fn(
    prediction: SageMakerInferencePrediction,
    response_content_type: str,
) -> str | bytes:
    accept = _normalize_media_type(response_content_type)
    if accept in {"application/json", "*/*"}:
        payload = {
            "request_id": prediction.request_id,
            "content_type": prediction.content_type,
            "image_base64": base64.b64encode(prediction.image_bytes).decode("ascii"),
            "prompt_plan": prediction.prompt_plan,
            "prompt": prediction.prompt,
            "timings": prediction.timings,
            "output_dir": str(prediction.output_dir),
            "planner_provider": prediction.planner_provider,
            "model": prediction.model,
        }
        return json.dumps(payload)
    if accept == "image/jpeg":
        return prediction.image_bytes
    raise ValueError(f"Unsupported accept type: {response_content_type}")


def load_model(
    *,
    model_dir: Path,
    output_root: Path | None = None,
    lora_path: Path | None = None,
    ai_toolkit_dir: Path | None = None,
    planner_provider: str | None = None,
    planner_model: str | None = None,
    aws_region: str | None = None,
) -> SageMakerMarbleModel:
    provider = planner_provider or os.environ.get(
        "KLEIN4B_PLANNER_PROVIDER",
        DEFAULT_PLANNER_PROVIDER,
    )
    if provider not in {"bedrock-nova", "openai"}:
        raise ValueError(f"Unsupported planner provider: {provider}")
    resolved_model = planner_model or os.environ.get("KLEIN4B_PLANNER_MODEL")
    if resolved_model is None:
        resolved_model = (
            DEFAULT_BEDROCK_NOVA_MODEL if provider == "bedrock-nova" else DEFAULT_OPENAI_MODEL
        )

    resolved_output_root = output_root or Path(
        os.environ.get("KLEIN4B_OUTPUT_ROOT", "/tmp/klein4b_sagemaker_outputs")
    )
    resolved_lora_path = _resolve_lora_path(model_dir, lora_path)
    resolved_ai_toolkit_dir = ai_toolkit_dir or Path(
        os.environ.get("KLEIN4B_AI_TOOLKIT_DIR", str(REPO_ROOT / "vendor" / "ai-toolkit"))
    )
    region = (
        aws_region
        or os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
        or DEFAULT_AWS_REGION
    )

    if not resolved_lora_path.exists():
        raise FileNotFoundError(f"LoRA weights not found: {resolved_lora_path}")
    if not resolved_ai_toolkit_dir.exists():
        raise FileNotFoundError(f"AI Toolkit directory not found: {resolved_ai_toolkit_dir}")

    return SageMakerMarbleModel(
        output_root=resolved_output_root,
        lora_path=resolved_lora_path,
        ai_toolkit_dir=resolved_ai_toolkit_dir,
        planner_provider=provider,
        planner_model=resolved_model,
        aws_region=region,
        preprocess_reference=_env_bool("KLEIN4B_PREPROCESS_REFERENCE", default=False),
        scrfd_model_path=_optional_env_path("KLEIN4B_SCRFD_MODEL_PATH"),
    )


def _input_from_json(body: bytes) -> SageMakerInferenceRequest:
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Invalid JSON request body") from error
    if not isinstance(payload, Mapping):
        raise ValueError("JSON request body must be an object")
    for key in payload:
        if "pseudo_target" in str(key).lower() or "pseudo-target" in str(key).lower():
            raise ValueError("Production inference does not accept pseudo target inputs")
    image_base64 = payload.get("image_base64", payload.get("reference_image_base64"))
    if not isinstance(image_base64, str) or not image_base64.strip():
        raise ValueError("JSON request body must include image_base64")
    try:
        image_bytes = base64.b64decode(image_base64, validate=True)
    except ValueError as error:
        raise ValueError("image_base64 is not valid base64") from error
    request_id = _sanitize_request_id(payload.get("request_id"))
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata must be a JSON object when provided")
    return SageMakerInferenceRequest(
        image_bytes=image_bytes,
        image_format=_detect_image_format(image_bytes),
        request_id=request_id,
        metadata=dict(metadata),
    )


def _plan_prompt(reference_path: Path, model: SageMakerMarbleModel) -> object:
    if model.planner_provider == "bedrock-nova":
        return plan_prompt_with_bedrock_nova(
            reference_path=reference_path,
            model=model.planner_model,
            region_name=model.aws_region,
        )
    return plan_prompt_with_openai(reference_path=reference_path, model=model.planner_model)


def _prepare_reference_image(
    *,
    input_data: SageMakerInferenceRequest,
    model: SageMakerMarbleModel,
    output_dir: Path,
) -> tuple[Path, Path, Path | None]:
    if not model.preprocess_reference:
        reference_path = output_dir / f"reference.{input_data.image_format}"
        reference_path.write_bytes(input_data.image_bytes)
        return reference_path, reference_path, None

    original_path = output_dir / f"reference_original.{input_data.image_format}"
    original_path.write_bytes(input_data.image_bytes)
    detector = _build_reference_detector(model)
    metadata_path = output_dir / "preprocess_metadata.json"
    result = preprocess_reference_image(
        reference_path=original_path,
        output_path=output_dir / "reference_preprocessed.jpg",
        metadata_path=metadata_path,
        detector=detector,
    )
    return original_path, result.effective_reference_path, metadata_path


def _build_reference_detector(model: SageMakerMarbleModel) -> ScrfdFaceDetector:
    if model.scrfd_model_path is None:
        raise ValueError("KLEIN4B_SCRFD_MODEL_PATH is required when KLEIN4B_PREPROCESS_REFERENCE=1")
    return ScrfdFaceDetector(model.scrfd_model_path)


def _optional_path(path: Path | None) -> str | None:
    if path is None:
        return None
    return str(path)


def _optional_env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    return Path(value)


def _env_bool(name: str, *, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _run_ai_toolkit_generation(
    *,
    command: list[str],
    output_dir: Path,
    samples_dir: Path,
    log_path: Path,
) -> GenerationResult:
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_file:
        subprocess.run(command, check=True, stdout=log_file, stderr=subprocess.STDOUT)
    wall_seconds = time.perf_counter() - start
    generated_path = find_latest_sample(samples_dir)
    return GenerationResult(
        generated_path=generated_path,
        wall_seconds=wall_seconds,
        sampler_seconds=_parse_sampler_seconds(log_path),
        log_path=log_path,
    )


def _copy_generated_jpeg(source_path: Path, destination_path: Path) -> Path:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if source_path.suffix.lower() in {".jpg", ".jpeg"}:
        shutil.copy2(source_path, destination_path)
        return destination_path
    with Image.open(source_path) as image:
        image.convert("RGB").save(destination_path, format="JPEG", quality=95)
    return destination_path


def _parse_sampler_seconds(log_path: Path) -> float | None:
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    matches = re.findall(
        r"Generating (?:Samples|Images):\s+100%.*?\[\s*(\d+):(\d+)<",
        text,
        flags=re.S,
    )
    if not matches:
        return None
    minutes, seconds = matches[-1]
    return float(int(minutes) * 60 + int(seconds))


def _resolve_lora_path(model_dir: Path, explicit_lora_path: Path | None) -> Path:
    env_lora_path = os.environ.get("KLEIN4B_LORA_PATH")
    if explicit_lora_path is not None:
        return explicit_lora_path
    if env_lora_path:
        return Path(env_lora_path)
    candidates = sorted(model_dir.rglob("*.safetensors"))
    if candidates:
        return candidates[0]
    return model_dir / "model.safetensors"


def _sanitize_request_id(value: object) -> str:
    if value is None:
        return f"request-{uuid.uuid4().hex}"
    if not isinstance(value, str) or not value.strip():
        raise ValueError("request_id must be a non-empty string when provided")
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip(".-")
    if not sanitized:
        raise ValueError("request_id must contain at least one safe path character")
    return sanitized[:120]


def _detect_image_format(image_bytes: bytes) -> str:
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image_format = (image.format or "").lower()
            image.verify()
    except Exception as error:
        raise ValueError("Request image bytes are not a valid image") from error
    if image_format == "jpeg":
        return "jpg"
    if image_format in {"png", "webp"}:
        return image_format
    raise ValueError(f"Unsupported image format: {image_format}")


def _body_to_bytes(request_body: bytes | str) -> bytes:
    if isinstance(request_body, bytes):
        return request_body
    return request_body.encode("utf-8")


def _normalize_media_type(content_type: str) -> str:
    return content_type.split(";", 1)[0].strip().lower()


def _prompt_plan_to_payload(prompt_plan: object) -> dict[str, object]:
    plan = parse_prompt_plan(_prompt_plan_to_candidate_payload(prompt_plan))
    reference = plan.reference_identity
    target = plan.target_style
    safety = plan.safety_overrides
    return {
        "reference_identity": {
            "age_band": reference.age_band,
            "gender_presentation": reference.gender_presentation,
            "eye_state": reference.eye_state,
            "head_pose": reference.head_pose,
            "face_structure": list(reference.face_structure),
            "hair_or_headwear": list(reference.hair_or_headwear),
            "broad_expression": reference.broad_expression,
        },
        "target_style": {
            "bust_framing": target.bust_framing,
            "statue_angle": target.statue_angle,
            "drapery_and_torso": list(target.drapery_and_torso),
            "headpiece_or_ornament": list(target.headpiece_or_ornament),
        },
        "safety_overrides": {
            "identity_source_policy": safety.identity_source_policy,
            "eye_policy": safety.eye_policy,
            "material_policy": safety.material_policy,
            "banned_details": list(safety.banned_details),
        },
    }


def _prompt_plan_to_candidate_payload(prompt_plan: object) -> Mapping[str, object]:
    if isinstance(prompt_plan, Mapping):
        return prompt_plan
    reference = prompt_plan.reference_identity
    target = prompt_plan.target_style
    safety = prompt_plan.safety_overrides
    return {
        "reference_identity": {
            "age_band": reference.age_band,
            "gender_presentation": reference.gender_presentation,
            "eye_state": reference.eye_state,
            "head_pose": reference.head_pose,
            "face_structure": list(reference.face_structure),
            "hair_or_headwear": list(reference.hair_or_headwear),
            "broad_expression": reference.broad_expression,
        },
        "target_style": {
            "bust_framing": target.bust_framing,
            "statue_angle": target.statue_angle,
            "drapery_and_torso": list(target.drapery_and_torso),
            "headpiece_or_ornament": list(target.headpiece_or_ornament),
        },
        "safety_overrides": {
            "identity_source_policy": safety.identity_source_policy,
            "eye_policy": safety.eye_policy,
            "material_policy": safety.material_policy,
            "banned_details": list(safety.banned_details),
        },
    }

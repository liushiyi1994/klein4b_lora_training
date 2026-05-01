from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from klein4b.marble_prompt_planning import (  # noqa: E402
    parse_prompt_plan,
    plan_prompt_with_bedrock_nova,
    plan_prompt_with_openai,
    render_marble_prompt,
)
from klein4b.reference_preprocessing import (  # noqa: E402
    ScrfdFaceDetector,
    preprocess_reference_image,
)
from klein4b.sample_style_inference import (  # noqa: E402
    DEFAULT_BEST_LORA_PATH,
    build_ai_toolkit_command,
    find_latest_sample,
    make_comparison_contact_sheet,
    negative_prompt_additions_for_eye_state,
    render_sample_style_config,
)

CONFIG_NAME = "structured_marble_inference"
DEFAULT_OPENAI_MODEL = "gpt-5.4-mini"
DEFAULT_BEDROCK_NOVA_MODEL = "us.amazon.nova-2-lite-v1:0"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan and run structured selfie-to-marble sample-style inference."
    )
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pseudo-target", type=Path, default=None)
    parser.add_argument("--previous-best", type=Path, default=None)
    parser.add_argument("--lora", type=Path, default=DEFAULT_BEST_LORA_PATH)
    parser.add_argument("--planner-provider", choices=("openai", "bedrock-nova"), default="openai")
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--aws-region",
        default=os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1",
    )
    parser.add_argument("--prompt-plan-json", type=Path, default=None)
    parser.add_argument("--skip-openai", action="store_true")
    parser.add_argument("--no-run-ai-toolkit", action="store_true")
    parser.add_argument("--ai-toolkit-dir", type=Path, default=REPO_ROOT / "vendor" / "ai-toolkit")
    parser.add_argument(
        "--preprocess-reference",
        action="store_true",
        help="Run SCRFD loose face crop before prompt planning and AI Toolkit sampling.",
    )
    parser.add_argument(
        "--scrfd-model",
        type=Path,
        default=os.environ.get("KLEIN4B_SCRFD_MODEL_PATH"),
        help="Path to caller-supplied SCRFD ONNX detector weights.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.skip_openai and args.prompt_plan_json is None:
        parser.error("--skip-openai requires --prompt-plan-json")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    planner_model = _resolve_planner_model(args.planner_provider, args.model)
    reference_path = _resolve_effective_reference_path(args)

    if args.skip_openai:
        prompt_plan_payload = json.loads(args.prompt_plan_json.read_text(encoding="utf-8"))
        prompt_plan = parse_prompt_plan(prompt_plan_payload)
        used_openai = False
    elif args.planner_provider == "bedrock-nova":
        prompt_plan = plan_prompt_with_bedrock_nova(
            reference_path=reference_path,
            model=planner_model,
            region_name=args.aws_region,
        )
        prompt_plan_payload = _prompt_plan_to_payload(prompt_plan)
        used_openai = False
    else:
        prompt_plan = plan_prompt_with_openai(reference_path=reference_path, model=planner_model)
        prompt_plan_payload = _prompt_plan_to_payload(prompt_plan)
        used_openai = True

    prompt_plan_path = args.output_dir / "prompt_plan.json"
    prompt_plan_path.write_text(json.dumps(prompt_plan_payload, indent=2) + "\n", encoding="utf-8")

    prompt = render_marble_prompt(prompt_plan)
    prompt_path = args.output_dir / "prompt.txt"
    prompt_path.write_text(prompt + "\n", encoding="utf-8")

    ai_toolkit_config_path = args.output_dir / "sample_style_inference.yaml"
    ai_toolkit_config_path.write_text(
        render_sample_style_config(
            name=CONFIG_NAME,
            training_folder=args.output_dir,
            reference_path=reference_path,
            prompt=prompt,
            lora_path=args.lora,
            negative_prompt_additions=negative_prompt_additions_for_eye_state(
                prompt_plan.reference_identity.eye_state
            ),
        ),
        encoding="utf-8",
    )

    generated_path = None
    command = build_ai_toolkit_command(args.ai_toolkit_dir, ai_toolkit_config_path)
    if not args.no_run_ai_toolkit:
        subprocess.run(command, check=True)
        latest_sample = find_latest_sample(args.output_dir / CONFIG_NAME / "samples")
        generated_path = args.output_dir / f"generated{latest_sample.suffix.lower()}"
        shutil.copy2(latest_sample, generated_path)

    run_config_path = args.output_dir / "run_config.json"
    run_config = {
        "reference_original": str(args.reference),
        "reference": str(reference_path),
        "preprocess_reference": args.preprocess_reference,
        "preprocess_metadata": _optional_path(
            args.output_dir / "preprocess_metadata.json" if args.preprocess_reference else None
        ),
        "pseudo_target": _optional_path(args.pseudo_target),
        "pseudo_target_role": "comparison-only",
        "previous_best": _optional_path(args.previous_best),
        "lora": str(args.lora),
        "planner_provider": args.planner_provider,
        "model": planner_model,
        "aws_region": args.aws_region,
        "used_openai": used_openai,
        "no_run_ai_toolkit": args.no_run_ai_toolkit,
        "ai_toolkit_dir": str(args.ai_toolkit_dir),
        "ai_toolkit_command": command,
        "ai_toolkit_config": str(ai_toolkit_config_path),
        "prompt": str(prompt_path),
        "prompt_plan": str(prompt_plan_path),
        "generated": _optional_path(generated_path),
    }
    run_config_path.write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")

    make_comparison_contact_sheet(
        output_path=args.output_dir / "contact_sheet.jpg",
        reference_path=reference_path,
        pseudo_target_path=args.pseudo_target,
        previous_best_path=args.previous_best,
        generated_path=generated_path,
    )


def _optional_path(path: Path | None) -> str | None:
    if path is None:
        return None
    return str(path)


def _resolve_effective_reference_path(args: argparse.Namespace) -> Path:
    if not args.preprocess_reference:
        return args.reference
    if args.scrfd_model is None:
        raise ValueError("--scrfd-model is required with --preprocess-reference")
    detector = ScrfdFaceDetector(model_path=Path(args.scrfd_model))
    result = preprocess_reference_image(
        reference_path=args.reference,
        output_path=args.output_dir / "reference_preprocessed.jpg",
        metadata_path=args.output_dir / "preprocess_metadata.json",
        detector=detector,
    )
    return result.effective_reference_path


def _resolve_planner_model(planner_provider: str, model: str | None) -> str:
    if model is not None:
        return model
    if planner_provider == "bedrock-nova":
        return DEFAULT_BEDROCK_NOVA_MODEL
    return DEFAULT_OPENAI_MODEL


def _prompt_plan_to_payload(prompt_plan) -> dict[str, object]:
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


if __name__ == "__main__":
    main()

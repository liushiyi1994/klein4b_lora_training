from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from klein4b.sagemaker_inference import (  # noqa: E402
    input_fn,
    load_model,
    output_fn,
    predict_fn,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the SageMaker marble inference handler locally."
    )
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--request-id", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--accept", default="application/json")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--lora", type=Path, default=None)
    parser.add_argument("--ai-toolkit-dir", type=Path, default=None)
    parser.add_argument("--planner-provider", choices=("bedrock-nova", "openai"), default=None)
    parser.add_argument("--model", dest="planner_model", default=None)
    parser.add_argument("--aws-region", default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    model = load_model(
        model_dir=args.model_dir,
        output_root=args.output_root,
        lora_path=args.lora,
        ai_toolkit_dir=args.ai_toolkit_dir,
        planner_provider=args.planner_provider,
        planner_model=args.planner_model,
        aws_region=args.aws_region,
    )
    request = _build_request(args.reference, args.request_id)
    prediction = predict_fn(request, model)
    response = output_fn(prediction, args.accept)
    _write_response(response, args.output)


def _content_type_for_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".webp":
        return "image/webp"
    return "application/x-image"


def _build_request(reference_path: Path, request_id: str | None) -> object:
    image_bytes = reference_path.read_bytes()
    if request_id is None:
        return input_fn(image_bytes, _content_type_for_path(reference_path))
    payload = {
        "request_id": request_id,
        "image_base64": base64.b64encode(image_bytes).decode("ascii"),
    }
    return input_fn(json.dumps(payload).encode("utf-8"), "application/json")


def _write_response(response: str | bytes, output_path: Path | None) -> None:
    if output_path is None:
        if isinstance(response, bytes):
            sys.stdout.buffer.write(response)
        else:
            print(response)
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(response, bytes):
        output_path.write_bytes(response)
    else:
        output_path.write_text(response + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from klein4b.inference import build_marble_pair_prompt  # noqa: E402

MODEL_NAME = "black-forest-labs/FLUX.2-klein-base-4B"
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")


@dataclass(frozen=True)
class CheckpointSpec:
    label: str
    path: Path


@dataclass(frozen=True)
class VariantSpec:
    label: str
    reference_preprocess: str
    image_condition_scale: float


def parse_checkpoint(value: str) -> CheckpointSpec:
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError("checkpoint must be LABEL=PATH")
    return CheckpointSpec(label=label, path=Path(path))


def parse_variant(value: str) -> VariantSpec:
    parts = value.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("variant must be LABEL:REFERENCE_PREPROCESS:SCALE")
    label, reference_preprocess, scale = parts
    if reference_preprocess != "original":
        raise argparse.ArgumentTypeError("only original reference preprocessing is supported")
    try:
        image_condition_scale = float(scale)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("variant scale must be a float") from exc
    return VariantSpec(
        label=label,
        reference_preprocess=reference_preprocess,
        image_condition_scale=image_condition_scale,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate fixed-prompt contact sheets for marble LoRA checkpoint sweeps."
    )
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, default=None)
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--prompt-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=parse_checkpoint, action="append", required=True)
    parser.add_argument(
        "--variant",
        type=parse_variant,
        action="append",
        default=None,
        help="Reference variant as LABEL:REFERENCE_PREPROCESS:SCALE.",
    )
    parser.add_argument("--id", dest="ids", action="append", default=None)
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    parser.add_argument("--num-inference-steps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--lora-scale", type=float, default=1.0)
    return parser


def discover_ids(reference_dir: Path, requested_ids: list[str] | None) -> list[str]:
    if requested_ids:
        return requested_ids
    return sorted(
        path.stem
        for path in reference_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def find_image(directory: Path, identity: str) -> Path:
    for extension in IMAGE_EXTENSIONS:
        path = directory / f"{identity}{extension}"
        if path.exists():
            return path
    raise FileNotFoundError(f"No image found for {identity} in {directory}")


def maybe_find_image(directory: Path | None, identity: str) -> Path | None:
    if directory is None:
        return None
    try:
        return find_image(directory, identity)
    except FileNotFoundError:
        return None


def load_prompt(identity: str, prompt_dir: Path | None, dataset_dir: Path | None) -> str:
    if prompt_dir is not None:
        return (prompt_dir / f"{identity}.txt").read_text(encoding="utf-8").strip()
    if dataset_dir is None:
        raise ValueError("--dataset-dir is required when --prompt-dir is not provided")
    caption_path = dataset_dir / "targets" / f"{identity}.txt"
    return build_marble_pair_prompt(caption_path.read_text(encoding="utf-8"))


def load_reference(reference_dir: Path, identity: str, variant: VariantSpec) -> Image.Image:
    if variant.reference_preprocess != "original":
        raise ValueError(f"Unsupported reference preprocessing: {variant.reference_preprocess}")
    return Image.open(find_image(reference_dir, identity)).convert("RGB")


def callable_accepts_kwarg(callable_obj: object, kwarg: str) -> bool:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return False
    return any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD or name == kwarg
        for name, parameter in signature.parameters.items()
    )


def resize_for_contact(image: Image.Image, width: int = 220) -> Image.Image:
    result = image.convert("RGB").copy()
    result.thumbnail((width, int(width * 2)), Image.Resampling.LANCZOS)
    return result


def make_contact_sheet(
    images: list[Image.Image],
    labels: list[str],
    columns: int = 4,
) -> Image.Image:
    if len(images) != len(labels):
        raise ValueError("images and labels must have the same length")
    thumbs = [resize_for_contact(image) for image in images]
    cell_width = max(image.width for image in thumbs)
    cell_height = max(image.height for image in thumbs) + 26
    rows = (len(thumbs) + columns - 1) // columns
    canvas = Image.new("RGB", (cell_width * columns, cell_height * rows), "black")
    draw = ImageDraw.Draw(canvas)
    for index, (image, label) in enumerate(zip(thumbs, labels, strict=True)):
        x = (index % columns) * cell_width
        y = (index // columns) * cell_height
        draw.text((x + 4, y + 6), label, fill="white")
        canvas.paste(image, (x, y + 26))
    return canvas


def generate_sweep(args: argparse.Namespace) -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for local inference")

    from diffusers import Flux2KleinPipeline

    start = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir = args.output_dir / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    contacts_dir = args.output_dir / "contacts"
    contacts_dir.mkdir(parents=True, exist_ok=True)

    ids = discover_ids(args.reference_dir, args.ids)
    variants = args.variant or [VariantSpec("orig", "original", 1.0)]

    print(f"fixed-prompt checkpoint sweep start: {datetime.now().astimezone().isoformat()}")
    pipe = Flux2KleinPipeline.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    supports_image_condition_scale = callable_accepts_kwarg(
        pipe.__call__,
        "image_condition_scale",
    )

    generated: dict[tuple[str, str, str], Path] = {}
    prompts: dict[str, str] = {}
    previous_lora_loaded = False

    for checkpoint in args.checkpoint:
        if previous_lora_loaded and hasattr(pipe, "unload_lora_weights"):
            pipe.unload_lora_weights()
        print(f"Loading LoRA {checkpoint.label}: {checkpoint.path}")
        pipe.load_lora_weights(str(checkpoint.path))
        previous_lora_loaded = True

        for variant in variants:
            checkpoint_output_dir = args.output_dir / f"{variant.label}_{checkpoint.label}"
            checkpoint_output_dir.mkdir(parents=True, exist_ok=True)
            reference_output_dir = args.output_dir / "reference_inputs" / variant.label
            reference_output_dir.mkdir(parents=True, exist_ok=True)

            for identity in ids:
                prompt = prompts.get(identity)
                if prompt is None:
                    prompt = load_prompt(identity, args.prompt_dir, args.dataset_dir)
                    prompts[identity] = prompt
                    (prompts_dir / f"{identity}.txt").write_text(prompt, encoding="utf-8")

                reference = load_reference(args.reference_dir, identity, variant)
                reference.save(reference_output_dir / f"{identity}.png")
                print(f"Generating {identity} {checkpoint.label} {variant.label}")
                call_kwargs = {
                    "image": reference,
                    "prompt": prompt,
                    "generator": torch.Generator(device="cuda").manual_seed(args.seed),
                    "guidance_scale": args.guidance_scale,
                    "num_inference_steps": args.num_inference_steps,
                    "attention_kwargs": {"scale": args.lora_scale},
                }
                if supports_image_condition_scale:
                    call_kwargs["image_condition_scale"] = variant.image_condition_scale
                result = pipe(**call_kwargs).images[0]
                output_path = checkpoint_output_dir / f"{identity}.png"
                result.save(output_path)
                generated[(checkpoint.label, variant.label, identity)] = output_path

    contact_paths: list[Path] = []
    for identity in ids:
        for variant in variants:
            images = [load_reference(args.reference_dir, identity, variant)]
            labels = [f"ref:{variant.label}"]
            target_path = maybe_find_image(args.target_dir, identity)
            if target_path is not None:
                images.append(Image.open(target_path).convert("RGB"))
                labels.append("target")
            for checkpoint in args.checkpoint:
                images.append(Image.open(generated[(checkpoint.label, variant.label, identity)]))
                labels.append(checkpoint.label)
            contact = make_contact_sheet(images, labels)
            contact_path = contacts_dir / f"{identity}_{variant.label}_contact.jpg"
            contact.save(contact_path)
            if variant == variants[0]:
                legacy_contact_path = contacts_dir / f"{identity}_contact.jpg"
                contact.save(legacy_contact_path)
                contact_paths.append(legacy_contact_path)

    summary_images = [Image.open(path).convert("RGB") for path in contact_paths]
    if summary_images:
        summary = make_contact_sheet(
            summary_images,
            [path.stem for path in contact_paths],
            columns=1,
        )
        summary.save(args.output_dir / "summary_contact.jpg")

    elapsed = time.time() - start
    run_config = {
        "reference_dir": str(args.reference_dir),
        "target_dir": str(args.target_dir) if args.target_dir is not None else None,
        "dataset_dir": str(args.dataset_dir) if args.dataset_dir is not None else None,
        "prompt_dir": str(args.prompt_dir) if args.prompt_dir is not None else None,
        "checkpoints": [
            {"label": checkpoint.label, "path": str(checkpoint.path)}
            for checkpoint in args.checkpoint
        ],
        "variants": [
            {
                "label": variant.label,
                "reference_preprocess": variant.reference_preprocess,
                "image_condition_scale": variant.image_condition_scale,
            }
            for variant in variants
        ],
        "ids": ids,
        "guidance_scale": args.guidance_scale,
        "num_inference_steps": args.num_inference_steps,
        "seed": args.seed,
        "lora_scale": args.lora_scale,
        "elapsed_seconds": elapsed,
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2),
        encoding="utf-8",
    )
    print(f"Done in {elapsed:.1f}s: {args.output_dir}")


def main() -> None:
    args = build_parser().parse_args()
    generate_sweep(args)


if __name__ == "__main__":
    main()

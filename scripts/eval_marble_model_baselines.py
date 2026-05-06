from __future__ import annotations

import argparse
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

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")
DEFAULT_MODELS = ("flux-dev", "z-image", "qwen-edit")
DEFAULT_KLEIN_DIR = Path("outputs/eval/sample_style_unquantized_6500_all_test_img/generated_named")
DEFAULT_NEGATIVE_PROMPT = (
    "head-only crop, close-up face, cropped shoulders, cropped torso, loose rocks, "
    "separate rock pile, rock pedestal, debris below bust, flat plinth, individual "
    "hair strands, natural hair color, glossy hair, polished marble, wet shine, "
    "specular hotspots, frontal studio light, head-on key light, beauty lighting, "
    "perfect portrait lighting, head-on beauty lighting, spotlight on face, smooth "
    "porcelain face, overexposed white face, clean bright cheeks, shiny forehead, "
    "shiny nose, shiny lips, red or orange light on face, red or orange light on hair, "
    "glowing skin, glowing hair, soft hair strands, realistic hair texture, modern "
    "eyewear, modern accessories, duplicate figures, side-by-side views, collage"
)
DEFAULT_PROMPT = (
    "Change image 1 into a centered chest-up Greek marble statue bust with blank "
    "white sculpted stone eyes, carved hair made from the same weathered grey marble "
    "as the face, visible shoulders and drapery, a broken stone base with subtle "
    "localized lava glow, rough pitted low-albedo stone texture, asymmetrical "
    "off-axis lighting, dark ember background, and no modern accessories."
)


@dataclass(frozen=True)
class ModelSpec:
    label: str
    output_label: str
    model_id: str
    pipeline_name: str
    mode: str


def build_model_specs(args: argparse.Namespace) -> dict[str, ModelSpec]:
    return {
        "flux-dev": ModelSpec(
            label="flux-dev",
            output_label="flux_dev",
            model_id=args.flux_dev_model,
            pipeline_name="FluxImg2ImgPipeline",
            mode="img2img",
        ),
        "z-image": ModelSpec(
            label="z-image",
            output_label="z_image",
            model_id=args.z_image_model,
            pipeline_name="ZImageImg2ImgPipeline",
            mode="img2img",
        ),
        "qwen-edit": ModelSpec(
            label="qwen-edit",
            output_label="qwen_edit",
            model_id=args.qwen_edit_model,
            pipeline_name="QwenImageEditPipeline",
            mode="edit",
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare marble reference-image inference baselines against Klein outputs."
    )
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, default=None)
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--prompt-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--klein-dir",
        type=Path,
        default=DEFAULT_KLEIN_DIR,
        help="Directory of existing Klein best-result images named by test id.",
    )
    parser.add_argument(
        "--no-klein",
        action="store_true",
        help="Do not include the existing Klein best-result column.",
    )
    parser.add_argument(
        "--model",
        choices=DEFAULT_MODELS,
        action="append",
        default=None,
        help="Model baseline to run. Defaults to all baselines.",
    )
    parser.add_argument("--id", dest="ids", action="append", default=None)
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    parser.add_argument("--num-inference-steps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument(
        "--img2img-strength",
        type=float,
        default=0.85,
        help="Strength for FLUX-dev and Z-Image img2img baselines.",
    )
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--negative-prompt-file", type=Path, default=None)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--cpu-offload",
        action="store_true",
        help="Use pipeline CPU offload when supported instead of moving the whole model to CUDA.",
    )
    parser.add_argument("--flux-dev-model", default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--z-image-model", default="Tongyi-MAI/Z-Image")
    parser.add_argument("--qwen-edit-model", default="Qwen/Qwen-Image-Edit")
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
    if directory is None or not directory.exists():
        return None
    try:
        return find_image(directory, identity)
    except FileNotFoundError:
        return None


def load_prompt(identity: str, prompt_dir: Path | None, dataset_dir: Path | None) -> str:
    if prompt_dir is not None:
        return (prompt_dir / f"{identity}.txt").read_text(encoding="utf-8").strip()
    if dataset_dir is not None:
        caption_path = dataset_dir / "targets" / f"{identity}.txt"
        return build_marble_pair_prompt(caption_path.read_text(encoding="utf-8"))
    return DEFAULT_PROMPT


def load_prompts(
    ids: list[str],
    prompt_dir: Path | None,
    dataset_dir: Path | None,
    output_dir: Path,
) -> dict[str, str]:
    prompts_dir = output_dir / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    prompts = {identity: load_prompt(identity, prompt_dir, dataset_dir) for identity in ids}
    for identity, prompt in prompts.items():
        (prompts_dir / f"{identity}.txt").write_text(prompt, encoding="utf-8")
    return prompts


def load_negative_prompt(args: argparse.Namespace) -> str:
    if args.negative_prompt_file is not None:
        return args.negative_prompt_file.read_text(encoding="utf-8").strip()
    return args.negative_prompt


def torch_dtype(torch_module: object, dtype_name: str) -> object:
    if dtype_name == "bf16":
        return torch_module.bfloat16
    if dtype_name == "fp16":
        return torch_module.float16
    return torch_module.float32


def load_pipeline(
    spec: ModelSpec,
    args: argparse.Namespace,
    torch_module: object,
) -> object:
    import diffusers

    pipeline_cls = getattr(diffusers, spec.pipeline_name)
    pipe = pipeline_cls.from_pretrained(
        spec.model_id,
        torch_dtype=torch_dtype(torch_module, args.dtype),
    )
    if args.cpu_offload and hasattr(pipe, "enable_model_cpu_offload"):
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(args.device)
    return pipe


def build_call_kwargs(
    spec: ModelSpec,
    reference: Image.Image,
    prompt: str,
    negative_prompt: str,
    args: argparse.Namespace,
    generator: object,
) -> dict[str, object]:
    common = {
        "image": reference,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "height": args.height,
        "width": args.width,
        "num_inference_steps": args.num_inference_steps,
        "generator": generator,
    }
    if spec.mode == "edit":
        return {
            **common,
            "true_cfg_scale": args.guidance_scale,
        }
    return {
        **common,
        "guidance_scale": args.guidance_scale,
        "strength": args.img2img_strength,
    }


def generate_for_model(
    spec: ModelSpec,
    ids: list[str],
    prompts: dict[str, str],
    negative_prompt: str,
    args: argparse.Namespace,
    torch_module: object,
) -> dict[str, Path]:
    pipe = load_pipeline(spec, args, torch_module)
    model_output_dir = args.output_dir / "generated" / spec.output_label
    model_output_dir.mkdir(parents=True, exist_ok=True)

    generated: dict[str, Path] = {}
    for identity in ids:
        reference = Image.open(find_image(args.reference_dir, identity)).convert("RGB")
        generator = torch_module.Generator(device=args.device).manual_seed(args.seed)
        call_kwargs = build_call_kwargs(
            spec,
            reference,
            prompts[identity],
            negative_prompt,
            args,
            generator,
        )
        print(f"Generating {identity} with {spec.label}")
        result = pipe(**call_kwargs).images[0]
        output_path = model_output_dir / f"{identity}.png"
        result.save(output_path)
        generated[identity] = output_path

    del pipe
    if hasattr(torch_module, "cuda") and hasattr(torch_module.cuda, "empty_cache"):
        torch_module.cuda.empty_cache()
    return generated


def resize_for_contact(image: Image.Image, width: int = 220) -> Image.Image:
    result = image.convert("RGB").copy()
    result.thumbnail((width, int(width * 2)), Image.Resampling.LANCZOS)
    return result


def make_contact_sheet(
    images: list[Image.Image],
    labels: list[str],
    columns: int | None = None,
) -> Image.Image:
    if len(images) != len(labels):
        raise ValueError("images and labels must have the same length")
    if not images:
        raise ValueError("at least one image is required")
    thumbs = [resize_for_contact(image) for image in images]
    column_count = columns or len(thumbs)
    cell_width = max(image.width for image in thumbs)
    cell_height = max(image.height for image in thumbs) + 26
    rows = (len(thumbs) + column_count - 1) // column_count
    canvas = Image.new("RGB", (cell_width * column_count, cell_height * rows), "black")
    draw = ImageDraw.Draw(canvas)
    for index, (image, label) in enumerate(zip(thumbs, labels, strict=True)):
        x = (index % column_count) * cell_width
        y = (index // column_count) * cell_height
        draw.text((x + 4, y + 6), label, fill="white")
        canvas.paste(image, (x, y + 26))
    return canvas


def write_contacts(
    ids: list[str],
    selected_specs: list[ModelSpec],
    generated: dict[str, dict[str, Path]],
    args: argparse.Namespace,
) -> list[Path]:
    contacts_dir = args.output_dir / "contacts"
    contacts_dir.mkdir(parents=True, exist_ok=True)
    include_klein = not args.no_klein and args.klein_dir is not None and args.klein_dir.exists()
    contact_paths: list[Path] = []

    for identity in ids:
        images = [Image.open(find_image(args.reference_dir, identity)).convert("RGB")]
        labels = ["reference"]

        target_path = maybe_find_image(args.target_dir, identity)
        if target_path is not None:
            images.append(Image.open(target_path).convert("RGB"))
            labels.append("target")

        klein_path = maybe_find_image(args.klein_dir, identity) if include_klein else None
        if klein_path is not None:
            images.append(Image.open(klein_path).convert("RGB"))
            labels.append("klein_6500")

        for spec in selected_specs:
            images.append(Image.open(generated[spec.label][identity]).convert("RGB"))
            labels.append(spec.output_label)

        contact = make_contact_sheet(images, labels)
        contact_path = contacts_dir / f"{identity}_contact.jpg"
        contact.save(contact_path)
        contact_paths.append(contact_path)

    return contact_paths


def write_summary(output_dir: Path, contact_paths: list[Path]) -> None:
    if not contact_paths:
        return
    summary_images = [Image.open(path).convert("RGB") for path in contact_paths]
    summary = make_contact_sheet(
        summary_images,
        [path.stem for path in contact_paths],
        columns=1,
    )
    summary.save(output_dir / "summary_contact.jpg")


def generate_baselines(args: argparse.Namespace) -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for local inference")

    start = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ids = discover_ids(args.reference_dir, args.ids)
    selected_labels = args.model or list(DEFAULT_MODELS)
    specs_by_label = build_model_specs(args)
    selected_specs = [specs_by_label[label] for label in selected_labels]
    prompts = load_prompts(ids, args.prompt_dir, args.dataset_dir, args.output_dir)
    negative_prompt = load_negative_prompt(args)

    print(f"model baseline comparison start: {datetime.now().astimezone().isoformat()}")
    generated: dict[str, dict[str, Path]] = {}
    for spec in selected_specs:
        generated[spec.label] = generate_for_model(
            spec,
            ids,
            prompts,
            negative_prompt,
            args,
            torch,
        )

    contact_paths = write_contacts(ids, selected_specs, generated, args)
    write_summary(args.output_dir, contact_paths)

    elapsed = time.time() - start
    run_config = {
        "reference_dir": str(args.reference_dir),
        "target_dir": str(args.target_dir) if args.target_dir is not None else None,
        "dataset_dir": str(args.dataset_dir) if args.dataset_dir is not None else None,
        "prompt_dir": str(args.prompt_dir) if args.prompt_dir is not None else None,
        "klein_dir": str(args.klein_dir) if args.klein_dir is not None else None,
        "included_klein_baseline": (
            not args.no_klein and args.klein_dir is not None and args.klein_dir.exists()
        ),
        "ids": ids,
        "models": selected_labels,
        "model_ids": {spec.label: spec.model_id for spec in selected_specs},
        "guidance_scale": args.guidance_scale,
        "num_inference_steps": args.num_inference_steps,
        "seed": args.seed,
        "width": args.width,
        "height": args.height,
        "img2img_strength": args.img2img_strength,
        "dtype": args.dtype,
        "device": args.device,
        "cpu_offload": args.cpu_offload,
        "negative_prompt": negative_prompt,
        "generated": {
            model: {identity: str(path) for identity, path in paths.items()}
            for model, paths in generated.items()
        },
        "elapsed_seconds": elapsed,
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2),
        encoding="utf-8",
    )
    print(f"Done in {elapsed:.1f}s: {args.output_dir}")


def main() -> None:
    args = build_parser().parse_args()
    generate_baselines(args)


if __name__ == "__main__":
    main()

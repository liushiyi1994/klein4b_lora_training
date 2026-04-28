from __future__ import annotations

import sys
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageFont

DEFAULT_MODEL_NAME = "black-forest-labs/FLUX.2-klein-base-4B"
DEFAULT_BEST_LORA_PATH = (
    Path("outputs")
    / "runs"
    / "marble_v4_pairs_rich_result_caption_rank64_unquantized"
    / "20260424-211309"
    / "marble_bust_v4_pairs_rich_result_caption_rank64_unquantized_style"
    / "marble_bust_v4_pairs_rich_result_caption_rank64_unquantized_style_000006500.safetensors"
)
DEFAULT_NEGATIVE_PROMPT = (
    "head-only crop, close-up face, cropped shoulders, cropped torso, loose rocks, "
    "separate rock pile, rock pedestal, debris below bust, flat plinth, individual hair "
    "strands, natural hair color, glossy hair, polished marble, wet shine, specular "
    "hotspots, frontal studio light, head-on key light, beauty lighting, perfect portrait "
    "lighting, head-on beauty lighting, spotlight on face, smooth porcelain face, "
    "overexposed white face, clean bright cheeks, shiny forehead, shiny nose, shiny lips, "
    "pupils, irises, colored eyes, catchlights, painted eyes, realistic human eyes, "
    "red or orange light on face, red or orange light on hair, glowing skin, glowing hair, "
    "soft hair strands, realistic hair texture, modern design, modern decorative details, "
    "contemporary decorative details, contemporary ornamentation, modern eyewear, "
    "modern accessories, "
    "fabric bow, hair bow, ribbon, hair ribbon, hair clip, hair band, headband, "
    "decorative headband, circlet, tiara, forehead band, colorful hair "
    "accessory, fabric cap, baseball cap, headscarf, head wrap, headwrap, turban, "
    "bonnet, veil, "
    "duplicate figures, side-by-side views, collage"
)


def render_sample_style_config(
    name: str,
    training_folder: Path,
    reference_path: Path,
    prompt: str,
    lora_path: Path = DEFAULT_BEST_LORA_PATH,
) -> str:
    payload = {
        "job": "extension",
        "config": {
            "name": name,
            "process": [
                {
                    "type": "diffusion_trainer",
                    "training_folder": str(training_folder),
                    "device": "cuda",
                    "trigger_word": "<mrblbust>",
                    "network": {
                        "type": "lora",
                        "linear": 64,
                        "linear_alpha": 64,
                        "conv": 32,
                        "conv_alpha": 32,
                        "pretrained_lora_path": str(lora_path),
                    },
                    "train": {
                        "steps": 0,
                        "skip_first_sample": True,
                        "disable_sampling": False,
                        "batch_size": 1,
                        "train_unet": True,
                        "train_text_encoder": False,
                        "gradient_checkpointing": True,
                        "noise_scheduler": "flowmatch",
                        "optimizer": "adamw8bit",
                        "dtype": "bf16",
                    },
                    "model": {
                        "arch": "flux2_klein_4b",
                        "name_or_path": DEFAULT_MODEL_NAME,
                        "quantize": False,
                        "quantize_te": False,
                        "low_vram": True,
                        "model_kwargs": {"match_target_res": False},
                    },
                    "sample": {
                        "sampler": "flowmatch",
                        "width": 768,
                        "height": 1024,
                        "samples": [{"prompt": prompt, "ctrl_img_1": str(reference_path)}],
                        "neg": DEFAULT_NEGATIVE_PROMPT,
                        "seed": 7,
                        "guidance_scale": 2.5,
                        "sample_steps": 24,
                    },
                }
            ],
        },
    }

    return yaml.safe_dump(payload, sort_keys=False)


def build_ai_toolkit_command(ai_toolkit_dir: Path, config_path: Path) -> list[str]:
    return [sys.executable, str(ai_toolkit_dir / "run.py"), str(config_path)]


def find_latest_sample(samples_dir: Path) -> Path:
    candidates = sorted(samples_dir.glob("*"))
    image_candidates = [
        path for path in candidates if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    ]
    if not image_candidates:
        raise FileNotFoundError(f"No sample images found in {samples_dir}")
    return image_candidates[-1]


def make_comparison_contact_sheet(
    output_path: Path,
    reference_path: Path,
    pseudo_target_path: Path | None,
    previous_best_path: Path | None,
    generated_path: Path | None,
) -> None:
    items = [("reference", reference_path)]
    if pseudo_target_path is not None:
        items.append(("pseudo target", pseudo_target_path))
    if previous_best_path is not None:
        items.append(("previous best", previous_best_path))
    if generated_path is not None:
        items.append(("generated", generated_path))

    thumb_width = 220
    thumb_height = 280
    label_height = 24
    padding = 16
    width = padding + len(items) * thumb_width + (len(items) - 1) * padding + padding
    height = padding + label_height + thumb_height + padding
    sheet = Image.new("RGB", (width, height), "black")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for index, (label, image_path) in enumerate(items):
        x = padding + index * (thumb_width + padding)
        draw.text((x, padding), label, fill="white", font=font)
        with Image.open(image_path) as image:
            thumbnail = image.convert("RGB")
            thumbnail.thumbnail((thumb_width, thumb_height), Image.Resampling.LANCZOS)
            image_x = x + (thumb_width - thumbnail.width) // 2
            image_y = padding + label_height + (thumb_height - thumbnail.height) // 2
            sheet.paste(thumbnail, (image_x, image_y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)

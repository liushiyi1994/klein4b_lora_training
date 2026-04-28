from __future__ import annotations

from pathlib import Path

import yaml

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
    "red or orange light on face, red or orange light on hair, glowing skin, glowing hair, "
    "soft hair strands, realistic hair texture, modern eyewear, modern accessories, "
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
    return [str(ai_toolkit_dir / "run.py"), str(config_path)]

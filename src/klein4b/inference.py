from __future__ import annotations

from pathlib import Path


def build_demo_prompt(trigger_word: str) -> str:
    return (
        f"{trigger_word}. Change image 1 into a centered close-up beauty portrait with "
        "refined cosmetic styling, preserving the person's identity, pose, and framing. "
        "Keep the background simple and the lighting clean and natural."
    )


def inference_defaults() -> dict[str, float | int]:
    return {"guidance_scale": 4.0, "num_inference_steps": 24}


def run_local_inference(
    reference_path: Path,
    prompt: str,
    output_path: Path,
    lora_path: Path | None,
) -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for local inference")

    from diffusers import Flux2KleinPipeline
    from PIL import Image

    pipe = Flux2KleinPipeline.from_pretrained(
        "black-forest-labs/FLUX.2-klein-base-4B",
        torch_dtype=torch.bfloat16,
    )
    if lora_path is not None:
        pipe.load_lora_weights(str(lora_path))
    pipe.to("cuda")

    kwargs = inference_defaults()
    image = Image.open(reference_path).convert("RGB")
    result = pipe(
        image=image,
        prompt=prompt,
        generator=torch.Generator(device="cuda").manual_seed(7),
        **kwargs,
    ).images[0]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path)

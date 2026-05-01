from __future__ import annotations

from pathlib import Path


def build_demo_prompt(trigger_word: str) -> str:
    return (
        f"{trigger_word}. Change image 1 into a centered close-up beauty portrait with "
        "refined cosmetic styling, preserving the person's identity, pose, and framing. "
        "Keep the background simple and the lighting clean and natural."
    )


def build_marble_pair_prompt(pair_caption: str) -> str:
    prompt = pair_caption.strip()
    if prompt.startswith("transform into"):
        prompt = prompt.replace("transform into", "Change image 1 into", 1)
    return (
        f"{prompt}. Do not preserve skin material, selfie lighting, skin shine, "
        "makeup shine, wet lips, or catchlights from the reference. Use a rough "
        "pitted low-albedo face surface with dry chalky unpolished stone, uneven "
        "grey-brown mineral patina, subdued off-axis ambient lighting, and "
        "asymmetrical stone lighting with one side of the face slightly darker "
        "than the other, shadowed eye sockets, and shallow carved shadows under "
        "the brow ridge, nose, lower lip, and chin; no full-face even "
        "illumination, no frontal studio light, no head-on key light, no "
        "perfect portrait lighting, no smooth beauty-render face, no polished "
        "cheeks, no shiny forehead, no shiny nose, no shiny lips, no glossy "
        "polished marble, no wet shine, no specular hotspots. The eyes must be "
        "blank white sculpted stone eyes or closed carved eyelids; no pupils, "
        "no irises, no colored eyes, no catchlights. Hair must be carved from "
        "the same marble as the face. Use hairstyle only as sculptural shape "
        "with broad chiseled grooves, solid stone masses, and marble veins "
        "continuing through the hair; ignore natural hair color from the "
        "reference and preserve only the hairstyle silhouette, braid shapes, "
        "and carved lock structure; no individual hair strands, no natural "
        "hair color, no glossy hair, no soft hair strands."
    )


def inference_defaults() -> dict[str, float | int]:
    return {"guidance_scale": 4.0, "num_inference_steps": 24}


def run_local_inference(
    reference_path: Path,
    prompt: str,
    output_path: Path,
    lora_path: Path | None,
    lora_scale: float = 1.0,
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
    call_kwargs = {
        "image": image,
        "prompt": prompt,
        "generator": torch.Generator(device="cuda").manual_seed(7),
        **kwargs,
    }
    if lora_path is not None:
        call_kwargs["attention_kwargs"] = {"scale": lora_scale}

    result = pipe(**call_kwargs).images[0]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path)

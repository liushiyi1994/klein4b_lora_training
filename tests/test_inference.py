from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest

from klein4b.inference import (
    build_demo_prompt,
    build_marble_pair_prompt,
    inference_defaults,
    run_local_inference,
)


def test_build_demo_prompt_is_transformation_focused() -> None:
    prompt = build_demo_prompt("K4BMAKEUP03")
    assert prompt.startswith("K4BMAKEUP03.")
    assert "preserving the person's identity" in prompt
    assert "image 2" not in prompt


def test_build_marble_pair_prompt_emphasizes_identity_and_blank_eyes() -> None:
    prompt = build_marble_pair_prompt(
        "transform into a <mrblbust> marble statue bust of a young adult man, "
        "preserve facial structure and identity"
    )

    assert prompt.startswith("Change image 1 into a <mrblbust>")
    assert "preserve facial structure and identity" in prompt
    assert "blank white sculpted stone eyes" in prompt
    assert "closed carved eyelids" in prompt
    assert "no pupils" in prompt
    assert "no irises" in prompt
    assert "no catchlights" in prompt
    assert "Hair must be carved from the same marble as the face" in prompt
    assert "Use hairstyle only as sculptural shape" in prompt
    assert "broad chiseled grooves" in prompt
    assert "marble veins continuing through the hair" in prompt
    assert "ignore natural hair color from the reference" in prompt
    assert "preserve only the hairstyle silhouette" in prompt
    assert "no individual hair strands" in prompt
    assert "no natural hair color" in prompt
    assert "no soft hair strands" in prompt


def test_build_marble_pair_prompt_suppresses_reference_lighting_and_gloss() -> None:
    prompt = build_marble_pair_prompt(
        "transform into a <mrblbust> marble statue bust, preserve facial structure"
    )
    prompt_lower = prompt.lower()

    assert "do not preserve skin material" in prompt_lower
    assert "selfie lighting" in prompt_lower
    assert "no frontal studio light" in prompt_lower
    assert "no head-on key light" in prompt_lower
    assert "no perfect portrait lighting" in prompt_lower
    assert "rough pitted low-albedo face surface" in prompt_lower
    assert "asymmetrical stone lighting" in prompt_lower
    assert "one side of the face slightly darker" in prompt_lower
    assert "shadowed eye sockets" in prompt_lower
    assert "no full-face even illumination" in prompt_lower
    assert "no smooth beauty-render face" in prompt_lower
    assert "no polished cheeks" in prompt_lower
    assert "no shiny forehead" in prompt_lower


def test_inference_defaults_match_photorealistic_demo_use_case() -> None:
    defaults = inference_defaults()
    assert defaults["guidance_scale"] == 4.0
    assert defaults["num_inference_steps"] == 24


def test_run_local_inference_uses_lora_seed_and_saves_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: dict[str, object] = {}

    class FakeGenerator:
        def __init__(self, device: str) -> None:
            calls["generator_device"] = device

        def manual_seed(self, seed: int) -> "FakeGenerator":
            calls["seed"] = seed
            return self

    fake_torch = types.ModuleType("torch")
    fake_torch.bfloat16 = "bfloat16"
    fake_torch.Generator = FakeGenerator
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: True)

    class FakeResultImage:
        def save(self, path: Path) -> None:
            calls["saved_to"] = path

    class FakePipeline:
        def load_lora_weights(self, path: str) -> None:
            calls["lora_path"] = path

        def to(self, device: str) -> None:
            calls["pipeline_device"] = device

        def __call__(self, **kwargs: object) -> types.SimpleNamespace:
            calls["pipe_kwargs"] = kwargs
            return types.SimpleNamespace(images=[FakeResultImage()])

    class FakeFlux2KleinPipeline:
        @staticmethod
        def from_pretrained(model_name: str, torch_dtype: object) -> FakePipeline:
            calls["model_name"] = model_name
            calls["torch_dtype"] = torch_dtype
            return FakePipeline()

    fake_diffusers = types.ModuleType("diffusers")
    fake_diffusers.Flux2KleinPipeline = FakeFlux2KleinPipeline

    class FakeOpenedImage:
        def convert(self, mode: str) -> str:
            calls["image_mode"] = mode
            return "converted-image"

    def fake_open(path: Path) -> FakeOpenedImage:
        calls["opened_path"] = path
        return FakeOpenedImage()

    fake_image_module = types.SimpleNamespace(open=fake_open)
    fake_pil = types.ModuleType("PIL")
    fake_pil.Image = fake_image_module

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)
    monkeypatch.setitem(sys.modules, "PIL", fake_pil)

    reference_path = tmp_path / "reference.png"
    output_path = tmp_path / "nested" / "result.png"
    lora_path = tmp_path / "weights.safetensors"

    run_local_inference(
        reference_path=reference_path,
        prompt="demo prompt",
        output_path=output_path,
        lora_path=lora_path,
        lora_scale=1.25,
    )

    assert calls["model_name"] == "black-forest-labs/FLUX.2-klein-base-4B"
    assert calls["torch_dtype"] == "bfloat16"
    assert calls["lora_path"] == str(lora_path)
    assert calls["pipeline_device"] == "cuda"
    assert calls["opened_path"] == reference_path
    assert calls["image_mode"] == "RGB"
    assert calls["generator_device"] == "cuda"
    assert calls["seed"] == 7
    assert output_path.parent.is_dir()
    assert calls["saved_to"] == output_path
    assert calls["pipe_kwargs"] == {
        "image": "converted-image",
        "prompt": "demo prompt",
        "generator": calls["pipe_kwargs"]["generator"],
        "guidance_scale": 4.0,
        "num_inference_steps": 24,
        "attention_kwargs": {"scale": 1.25},
    }


def test_run_local_inference_requires_cuda(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)

    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    with pytest.raises(RuntimeError, match="CUDA is required for local inference"):
        run_local_inference(
            reference_path=tmp_path / "reference.png",
            prompt="demo prompt",
            output_path=tmp_path / "result.png",
            lora_path=None,
        )


def test_run_inference_cli_wires_prompt_and_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_calls: dict[str, object] = {}
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "run_inference.py"
    spec = importlib.util.spec_from_file_location("task5_run_inference", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def fake_run_local_inference(
        *,
        reference_path: Path,
        prompt: str,
        output_path: Path,
        lora_path: Path | None,
        lora_scale: float = 1.0,
    ) -> None:
        run_calls["reference_path"] = reference_path
        run_calls["prompt"] = prompt
        run_calls["output_path"] = output_path
        run_calls["lora_path"] = lora_path
        run_calls["lora_scale"] = lora_scale

    monkeypatch.setattr(module, "run_local_inference", fake_run_local_inference)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_inference.py",
            "--reference",
            str(tmp_path / "reference.png"),
            "--output",
            str(tmp_path / "result.png"),
            "--lora",
            str(tmp_path / "weights.safetensors"),
            "--lora-scale",
            "1.2",
        ],
    )

    module.main()

    assert run_calls["reference_path"] == tmp_path / "reference.png"
    assert run_calls["output_path"] == tmp_path / "result.png"
    assert run_calls["lora_path"] == tmp_path / "weights.safetensors"
    assert run_calls["lora_scale"] == 1.2
    assert run_calls["prompt"] == build_demo_prompt("K4BMAKEUP03")

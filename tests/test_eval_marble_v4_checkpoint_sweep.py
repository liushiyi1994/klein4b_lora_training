from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

from PIL import Image


def load_sweep_module() -> types.ModuleType:
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "eval_marble_v4_checkpoint_sweep.py"
    )
    spec = importlib.util.spec_from_file_location("eval_marble_v4_checkpoint_sweep", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_checkpoint_sweep_writes_outputs_and_run_config(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls: dict[str, object] = {"loaded_loras": [], "prompts": [], "scales": []}

    reference_dir = tmp_path / "refs"
    target_dir = tmp_path / "targets"
    prompt_dir = tmp_path / "prompts_in"
    output_dir = tmp_path / "eval"
    reference_dir.mkdir()
    target_dir.mkdir()
    prompt_dir.mkdir()
    Image.new("RGB", (8, 8), "red").save(reference_dir / "face_a.png")
    Image.new("RGB", (8, 8), "blue").save(target_dir / "face_a.png")
    (prompt_dir / "face_a.txt").write_text("fixed prompt for face_a", encoding="utf-8")

    checkpoint_a = tmp_path / "a.safetensors"
    checkpoint_b = tmp_path / "b.safetensors"
    checkpoint_a.write_text("a", encoding="utf-8")
    checkpoint_b.write_text("b", encoding="utf-8")

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

    class FakePipeline:
        def load_lora_weights(self, path: str) -> None:
            calls["loaded_loras"].append(path)

        def to(self, device: str) -> "FakePipeline":
            calls["device"] = device
            return self

        def __call__(self, **kwargs: object) -> types.SimpleNamespace:
            calls["prompts"].append(kwargs["prompt"])
            calls["scales"].append(kwargs["attention_kwargs"]["scale"])
            calls["image_condition_scale"] = kwargs["image_condition_scale"]
            return types.SimpleNamespace(images=[Image.new("RGB", (8, 8), "green")])

    class FakeFlux2KleinPipeline:
        @staticmethod
        def from_pretrained(model_name: str, torch_dtype: object) -> FakePipeline:
            calls["model_name"] = model_name
            calls["torch_dtype"] = torch_dtype
            return FakePipeline()

    fake_diffusers = types.ModuleType("diffusers")
    fake_diffusers.Flux2KleinPipeline = FakeFlux2KleinPipeline
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)

    module = load_sweep_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_marble_v4_checkpoint_sweep.py",
            "--reference-dir",
            str(reference_dir),
            "--target-dir",
            str(target_dir),
            "--prompt-dir",
            str(prompt_dir),
            "--output-dir",
            str(output_dir),
            "--checkpoint",
            f"quantized_8500={checkpoint_a}",
            "--checkpoint",
            f"unquantized_4000={checkpoint_b}",
            "--variant",
            "orig:original:0.75",
            "--guidance-scale",
            "2.5",
            "--num-inference-steps",
            "24",
            "--seed",
            "7",
            "--lora-scale",
            "1.0",
        ],
    )

    module.main()

    assert calls["model_name"] == "black-forest-labs/FLUX.2-klein-base-4B"
    assert calls["torch_dtype"] == "bfloat16"
    assert calls["device"] == "cuda"
    assert calls["loaded_loras"] == [str(checkpoint_a), str(checkpoint_b)]
    assert calls["prompts"] == ["fixed prompt for face_a", "fixed prompt for face_a"]
    assert calls["scales"] == [1.0, 1.0]
    assert calls["image_condition_scale"] == 0.75
    assert calls["generator_device"] == "cuda"
    assert calls["seed"] == 7

    assert (output_dir / "orig_quantized_8500" / "face_a.png").exists()
    assert (output_dir / "orig_unquantized_4000" / "face_a.png").exists()
    assert (output_dir / "contacts" / "face_a_contact.jpg").exists()
    assert (output_dir / "summary_contact.jpg").exists()
    assert (output_dir / "prompts" / "face_a.txt").read_text(encoding="utf-8") == (
        "fixed prompt for face_a"
    )

    run_config = json.loads((output_dir / "run_config.json").read_text(encoding="utf-8"))
    assert run_config["reference_dir"] == str(reference_dir)
    assert run_config["target_dir"] == str(target_dir)
    assert run_config["prompt_dir"] == str(prompt_dir)
    assert run_config["ids"] == ["face_a"]
    assert run_config["guidance_scale"] == 2.5
    assert run_config["num_inference_steps"] == 24
    assert run_config["seed"] == 7
    assert run_config["lora_scale"] == 1.0
    assert run_config["checkpoints"][0]["label"] == "quantized_8500"
    assert run_config["checkpoints"][1]["label"] == "unquantized_4000"


def test_checkpoint_sweep_omits_image_condition_scale_when_pipeline_does_not_support_it(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls: dict[str, object] = {"loaded_loras": [], "prompts": []}

    reference_dir = tmp_path / "refs"
    prompt_dir = tmp_path / "prompts_in"
    output_dir = tmp_path / "eval"
    reference_dir.mkdir()
    prompt_dir.mkdir()
    Image.new("RGB", (8, 8), "red").save(reference_dir / "face_a.png")
    (prompt_dir / "face_a.txt").write_text("fixed prompt for face_a", encoding="utf-8")

    checkpoint = tmp_path / "checkpoint.safetensors"
    checkpoint.write_text("checkpoint", encoding="utf-8")

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

    class FakePipeline:
        def load_lora_weights(self, path: str) -> None:
            calls["loaded_loras"].append(path)

        def to(self, device: str) -> "FakePipeline":
            calls["device"] = device
            return self

        def __call__(
            self,
            *,
            image: Image.Image,
            prompt: str,
            generator: FakeGenerator,
            guidance_scale: float,
            num_inference_steps: int,
            attention_kwargs: dict[str, float],
        ) -> types.SimpleNamespace:
            calls["prompts"].append(prompt)
            calls["guidance_scale"] = guidance_scale
            calls["num_inference_steps"] = num_inference_steps
            calls["attention_kwargs"] = attention_kwargs
            return types.SimpleNamespace(images=[Image.new("RGB", (8, 8), "green")])

    class FakeFlux2KleinPipeline:
        @staticmethod
        def from_pretrained(model_name: str, torch_dtype: object) -> FakePipeline:
            calls["model_name"] = model_name
            calls["torch_dtype"] = torch_dtype
            return FakePipeline()

    fake_diffusers = types.ModuleType("diffusers")
    fake_diffusers.Flux2KleinPipeline = FakeFlux2KleinPipeline
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)

    module = load_sweep_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_marble_v4_checkpoint_sweep.py",
            "--reference-dir",
            str(reference_dir),
            "--prompt-dir",
            str(prompt_dir),
            "--output-dir",
            str(output_dir),
            "--checkpoint",
            f"unquantized_4000={checkpoint}",
            "--variant",
            "orig:original:0.75",
        ],
    )

    module.main()

    assert calls["loaded_loras"] == [str(checkpoint)]
    assert calls["prompts"] == ["fixed prompt for face_a"]
    assert calls["guidance_scale"] == 2.5
    assert calls["num_inference_steps"] == 24
    assert calls["attention_kwargs"] == {"scale": 1.0}
    assert (output_dir / "orig_unquantized_4000" / "face_a.png").exists()

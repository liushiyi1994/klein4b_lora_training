from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest
from PIL import Image


def load_baselines_module() -> types.ModuleType:
    script_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "eval_marble_model_baselines.py"
    )
    spec = importlib.util.spec_from_file_location("eval_marble_model_baselines", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_model_baseline_runner_writes_outputs_contacts_and_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    reference_dir = tmp_path / "refs"
    target_dir = tmp_path / "targets"
    prompt_dir = tmp_path / "prompts"
    klein_dir = tmp_path / "klein"
    output_dir = tmp_path / "eval"
    for directory in (reference_dir, target_dir, prompt_dir, klein_dir):
        directory.mkdir()

    Image.new("RGB", (12, 16), "red").save(reference_dir / "face_a.png")
    Image.new("RGB", (12, 16), "blue").save(target_dir / "face_a.jpg")
    Image.new("RGB", (12, 16), "purple").save(klein_dir / "face_a.jpg")
    (prompt_dir / "face_a.txt").write_text("fixed marble prompt", encoding="utf-8")

    class FakeGenerator:
        def __init__(self, device: str) -> None:
            self.device = device

        def manual_seed(self, seed: int) -> "FakeGenerator":
            self.seed = seed
            return self

    fake_torch = types.ModuleType("torch")
    fake_torch.bfloat16 = "bfloat16"
    fake_torch.float16 = "float16"
    fake_torch.float32 = "float32"
    fake_torch.Generator = FakeGenerator
    fake_torch.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        empty_cache=lambda: None,
    )

    class FakePipeline:
        label = "unset"
        color = "white"

        @classmethod
        def from_pretrained(cls, model_id: str, torch_dtype: object) -> "FakePipeline":
            pipe = cls()
            pipe.model_id = model_id
            pipe.torch_dtype = torch_dtype
            return pipe

        def to(self, device: str) -> "FakePipeline":
            self.device = device
            return self

        def __call__(self, **kwargs: object) -> types.SimpleNamespace:
            calls.append(
                (
                    self.label,
                    {
                        "model_id": self.model_id,
                        "torch_dtype": self.torch_dtype,
                        "device": self.device,
                        **kwargs,
                    },
                )
            )
            return types.SimpleNamespace(images=[Image.new("RGB", (10, 14), self.color)])

    class FakeFlux(FakePipeline):
        label = "flux-dev"
        color = "green"

    class FakeZImage(FakePipeline):
        label = "z-image"
        color = "yellow"

    class FakeQwen(FakePipeline):
        label = "qwen-edit"
        color = "orange"

    fake_diffusers = types.ModuleType("diffusers")
    fake_diffusers.FluxImg2ImgPipeline = FakeFlux
    fake_diffusers.ZImageImg2ImgPipeline = FakeZImage
    fake_diffusers.QwenImageEditPipeline = FakeQwen
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)

    module = load_baselines_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_marble_model_baselines.py",
            "--reference-dir",
            str(reference_dir),
            "--target-dir",
            str(target_dir),
            "--prompt-dir",
            str(prompt_dir),
            "--klein-dir",
            str(klein_dir),
            "--output-dir",
            str(output_dir),
            "--model",
            "flux-dev",
            "--model",
            "z-image",
            "--model",
            "qwen-edit",
            "--guidance-scale",
            "2.5",
            "--num-inference-steps",
            "24",
            "--seed",
            "7",
            "--width",
            "768",
            "--height",
            "1024",
            "--img2img-strength",
            "0.82",
        ],
    )

    module.main()

    assert [label for label, _ in calls] == ["flux-dev", "z-image", "qwen-edit"]

    flux_call = calls[0][1]
    assert flux_call["model_id"] == "black-forest-labs/FLUX.1-dev"
    assert flux_call["torch_dtype"] == "bfloat16"
    assert flux_call["device"] == "cuda"
    assert flux_call["prompt"] == "fixed marble prompt"
    assert flux_call["guidance_scale"] == 2.5
    assert flux_call["num_inference_steps"] == 24
    assert flux_call["strength"] == 0.82
    assert flux_call["width"] == 768
    assert flux_call["height"] == 1024
    assert flux_call["generator"].seed == 7

    z_image_call = calls[1][1]
    assert z_image_call["strength"] == 0.82
    assert z_image_call["guidance_scale"] == 2.5

    qwen_call = calls[2][1]
    assert "strength" not in qwen_call
    assert qwen_call["true_cfg_scale"] == 2.5
    assert qwen_call["num_inference_steps"] == 24

    assert (output_dir / "generated" / "flux_dev" / "face_a.png").exists()
    assert (output_dir / "generated" / "z_image" / "face_a.png").exists()
    assert (output_dir / "generated" / "qwen_edit" / "face_a.png").exists()
    assert (output_dir / "contacts" / "face_a_contact.jpg").exists()
    assert (output_dir / "summary_contact.jpg").exists()
    assert (output_dir / "prompts" / "face_a.txt").read_text(encoding="utf-8") == (
        "fixed marble prompt"
    )

    run_config = json.loads((output_dir / "run_config.json").read_text(encoding="utf-8"))
    assert run_config["reference_dir"] == str(reference_dir)
    assert run_config["target_dir"] == str(target_dir)
    assert run_config["prompt_dir"] == str(prompt_dir)
    assert run_config["klein_dir"] == str(klein_dir)
    assert run_config["ids"] == ["face_a"]
    assert run_config["models"] == ["flux-dev", "z-image", "qwen-edit"]
    assert run_config["included_klein_baseline"] is True
    assert run_config["guidance_scale"] == 2.5
    assert run_config["num_inference_steps"] == 24
    assert run_config["img2img_strength"] == 0.82
    assert run_config["model_ids"]["qwen-edit"] == "Qwen/Qwen-Image-Edit"


def test_model_baseline_runner_requires_cuda(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    reference_dir = tmp_path / "refs"
    prompt_dir = tmp_path / "prompts"
    reference_dir.mkdir()
    prompt_dir.mkdir()
    Image.new("RGB", (8, 8), "red").save(reference_dir / "face_a.png")
    (prompt_dir / "face_a.txt").write_text("fixed marble prompt", encoding="utf-8")

    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    module = load_baselines_module()
    args = module.build_parser().parse_args(
        [
            "--reference-dir",
            str(reference_dir),
            "--prompt-dir",
            str(prompt_dir),
            "--output-dir",
            str(tmp_path / "eval"),
            "--model",
            "flux-dev",
        ]
    )

    with pytest.raises(RuntimeError, match="CUDA is required"):
        module.generate_baselines(args)

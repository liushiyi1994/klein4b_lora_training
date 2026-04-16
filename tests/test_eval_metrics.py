import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from klein4b.eval_metrics import ssim_score
from klein4b.image_grid import make_four_up_grid


def test_ssim_score_is_one_for_identical_images() -> None:
    pixels = np.zeros((32, 32, 3), dtype=np.uint8)
    image = Image.fromarray(pixels)
    assert ssim_score(image, image) == 1.0


def test_make_four_up_grid_has_expected_width() -> None:
    image = Image.new("RGB", (16, 16), color="white")
    grid = make_four_up_grid([image, image, image, image], ["ref", "base", "lora", "target"])
    assert grid.size[0] == 64


def test_make_four_up_grid_rejects_invalid_input() -> None:
    image = Image.new("RGB", (16, 16), color="white")
    mismatched = Image.new("RGB", (12, 16), color="white")

    with pytest.raises(ValueError, match="exactly 4 images"):
        make_four_up_grid([image, image, image], ["ref", "base", "lora", "target"])

    with pytest.raises(ValueError, match="exactly 4 labels"):
        make_four_up_grid([image, image, image, image], ["ref", "base", "lora"])

    with pytest.raises(ValueError, match="same size"):
        make_four_up_grid(
            [image, image, mismatched, image],
            ["ref", "base", "lora", "target"],
        )


def test_compare_before_after_writes_grid_and_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "compare_before_after.py"
    spec = importlib.util.spec_from_file_location("task6_compare_before_after", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    identity = "000123"
    data_root = tmp_path / "data" / "demo_ffhq_makeup"
    outputs_root = tmp_path / "outputs" / "eval"
    data_root.mkdir(parents=True)
    (outputs_root / "base").mkdir(parents=True)
    (outputs_root / "lora").mkdir(parents=True)

    reference_path = tmp_path / "reference.png"
    target_path = tmp_path / "target.png"
    Image.new("RGB", (8, 8), color=(10, 10, 10)).save(reference_path)
    Image.new("RGB", (8, 8), color=(20, 20, 20)).save(target_path)
    Image.new("RGB", (8, 8), color=(30, 30, 30)).save(outputs_root / "base" / f"{identity}.png")
    Image.new("RGB", (8, 8), color=(20, 20, 20)).save(outputs_root / "lora" / f"{identity}.png")

    manifest = [
        {
            "identity_id": identity,
            "split": "test",
            "reference_path": str(reference_path),
            "target_path": str(target_path),
        }
    ]
    (data_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)

    module.main()

    grid_path = outputs_root / "grids" / f"{identity}.png"
    metrics_path = outputs_root / "metrics" / f"{identity}.json"
    assert grid_path.exists()
    assert metrics_path.exists()
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["identity_id"] == identity
    assert metrics["base_ssim_to_target"] < metrics["lora_ssim_to_target"]
    assert metrics["lora_ssim_to_target"] == 1.0

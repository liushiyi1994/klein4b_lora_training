# Klein4B Bootstrap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible local WSL bootstrap for FLUX.2 Klein 4B LoRA training, paired-demo dataset prep, local reference-image inference, and before/after evaluation.

**Architecture:** Keep the repository thin and deterministic. This project owns dataset prep, config rendering, training launch, inference, and evaluation, while a pinned checkout of AI Toolkit handles LoRA training. Use the `FFHQ-Makeup` demo dataset as a smoke test: `bare.jpg` supplies identity at inference, `makeup_03.jpg` supplies the supervised style target during training and evaluation.

**Tech Stack:** Python 3.12, PyTorch 2.9.1/cu128, AI Toolkit CLI, Diffusers `Flux2KleinPipeline`, Hugging Face Hub, Pillow, PyYAML, LPIPS, scikit-image, pytest, Ruff

---

## Research Notes To Preserve In The Implementation

- AI Toolkit README currently lists `python >=3.10 (3.12 recommended)` and installs `torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128`.
- AI Toolkit explicitly supports `black-forest-labs/FLUX.2-klein-base-4B`.
- BFL’s FLUX.2 Klein training guide recommends **base** variants for fine-tuning, `8e-5` to `1e-4` LoRA learning rates, `1500-2500` steps for style LoRAs, and `512px` as the fast-iteration starting resolution.
- BFL’s FLUX.2 Klein style-training example recommends **20-40 training images**, trigger words, and captions that describe visible content but omit the target style.
- The Hugging Face `Flux2KleinPipeline` supports local image-conditioned generation with an `image=` argument, so the inference harness can stay local and use the trained LoRA directly.

## Planned File Structure

**Create**
- `.python-version`
- `.gitignore`
- `pyproject.toml`
- `requirements.txt`
- `README.md`
- `configs/ai_toolkit.lock.json`
- `configs/train_flux2_klein_makeup_demo.template.yaml`
- `scripts/setup_local_env.sh`
- `scripts/bootstrap_ai_toolkit.sh`
- `scripts/check_cuda.py`
- `scripts/bootstrap_demo_dataset.py`
- `scripts/train_lora.py`
- `scripts/run_inference.py`
- `scripts/compare_before_after.py`
- `src/klein4b/__init__.py`
- `src/klein4b/paths.py`
- `src/klein4b/ai_toolkit.py`
- `src/klein4b/demo_dataset.py`
- `src/klein4b/training.py`
- `src/klein4b/inference.py`
- `src/klein4b/eval_metrics.py`
- `src/klein4b/image_grid.py`
- `tests/test_paths.py`
- `tests/test_ai_toolkit.py`
- `tests/test_demo_dataset.py`
- `tests/test_training.py`
- `tests/test_inference.py`
- `tests/test_eval_metrics.py`

---

### Task 1: Scaffold The Python Project And Stable Paths

**Files:**
- Create: `.python-version`
- Create: `.gitignore`
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `src/klein4b/__init__.py`
- Create: `src/klein4b/paths.py`
- Test: `tests/test_paths.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_paths.py
from pathlib import Path

from klein4b.paths import data_dir, outputs_dir, repo_root


def test_repo_root_contains_agents_md() -> None:
    root = repo_root()
    assert root.name == "klein4b"
    assert (root / "AGENTS.md").exists()


def test_standard_runtime_directories_are_root_relative() -> None:
    root = repo_root()
    assert data_dir() == root / "data"
    assert outputs_dir() == root / "outputs"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_paths.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'klein4b'`

- [ ] **Step 3: Write minimal implementation**

```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=69", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "klein4b"
version = "0.1.0"
requires-python = ">=3.12,<3.13"
description = "Bootstrap training pipeline for FLUX.2 Klein 4B LoRA experiments."

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I"]
```

```text
# .python-version
3.12
```

```text
# requirements.txt
huggingface_hub>=0.34.4
pyyaml>=6.0.2
pillow>=11.1.0
numpy>=2.1.1
diffusers>=0.35.0
transformers>=4.55.0
accelerate>=1.10.0
safetensors>=0.5.3
lpips>=0.1.4
scikit-image>=0.25.2
pytest>=8.4.1
ruff>=0.12.0
```

```text
# .gitignore
.venv/
__pycache__/
.pytest_cache/
.ruff_cache/
data/
outputs/
vendor/
*.pyc
```

```python
# src/klein4b/__init__.py
__all__ = ["__version__"]

__version__ = "0.1.0"
```

```python
# src/klein4b/paths.py
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    return repo_root() / "data"


def outputs_dir() -> Path:
    return repo_root() / "outputs"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_paths.py -q`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add .python-version .gitignore pyproject.toml requirements.txt src/klein4b/__init__.py src/klein4b/paths.py tests/test_paths.py
git commit -m "chore: scaffold python package and repo paths"
```

### Task 2: Pin AI Toolkit And Bootstrap The Local WSL Environment

**Files:**
- Create: `configs/ai_toolkit.lock.json`
- Create: `scripts/setup_local_env.sh`
- Create: `scripts/bootstrap_ai_toolkit.sh`
- Create: `scripts/check_cuda.py`
- Create: `src/klein4b/ai_toolkit.py`
- Test: `tests/test_ai_toolkit.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ai_toolkit.py
from pathlib import Path

from klein4b.ai_toolkit import (
    AIToolkitLock,
    load_ai_toolkit_lock,
    torch_install_command,
)


def test_lock_file_has_expected_repo_and_commit() -> None:
    lock = load_ai_toolkit_lock(Path("configs/ai_toolkit.lock.json"))
    assert lock.repo_url == "https://github.com/ostris/ai-toolkit.git"
    assert lock.commit == "8f67f5022e5ea39e8325cddb32fd6a0ab46a0813"


def test_torch_install_command_targets_cu128() -> None:
    command = torch_install_command()
    assert "torch==2.9.1" in command
    assert "cu128" in command
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ai_toolkit.py -q`
Expected: FAIL with `ModuleNotFoundError` or import errors for `klein4b.ai_toolkit`

- [ ] **Step 3: Write minimal implementation**

```json
// configs/ai_toolkit.lock.json
{
  "repo_url": "https://github.com/ostris/ai-toolkit.git",
  "commit": "8f67f5022e5ea39e8325cddb32fd6a0ab46a0813",
  "torch": {
    "version": "2.9.1",
    "torchvision": "0.24.1",
    "torchaudio": "2.9.1",
    "index_url": "https://download.pytorch.org/whl/cu128"
  }
}
```

```python
# src/klein4b/ai_toolkit.py
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class AIToolkitLock:
    repo_url: str
    commit: str
    torch: dict[str, str]


def load_ai_toolkit_lock(path: Path) -> AIToolkitLock:
    payload = json.loads(path.read_text())
    return AIToolkitLock(
        repo_url=payload["repo_url"],
        commit=payload["commit"],
        torch=payload["torch"],
    )


def torch_install_command() -> str:
    lock = load_ai_toolkit_lock(Path("configs/ai_toolkit.lock.json"))
    torch_cfg = lock.torch
    return (
        "pip install --no-cache-dir "
        f"torch=={torch_cfg['version']} "
        f"torchvision=={torch_cfg['torchvision']} "
        f"torchaudio=={torch_cfg['torchaudio']} "
        f"--index-url {torch_cfg['index_url']}"
    )
```

```bash
# scripts/setup_local_env.sh
#!/usr/bin/env bash
set -euo pipefail

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install --no-cache-dir torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -e .
```

```bash
# scripts/bootstrap_ai_toolkit.sh
#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/ostris/ai-toolkit.git"
COMMIT="8f67f5022e5ea39e8325cddb32fd6a0ab46a0813"
TARGET_DIR="vendor/ai-toolkit"

mkdir -p vendor
if [ ! -d "${TARGET_DIR}/.git" ]; then
  git clone "${REPO_URL}" "${TARGET_DIR}"
fi

git -C "${TARGET_DIR}" fetch --all --tags
git -C "${TARGET_DIR}" checkout "${COMMIT}"

source .venv/bin/activate
pip install -r "${TARGET_DIR}/requirements.txt"
```

```python
# scripts/check_cuda.py
import torch

print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device_name={torch.cuda.get_device_name(0)}")
    print(f"bf16_supported={torch.cuda.is_bf16_supported()}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ai_toolkit.py -q`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add configs/ai_toolkit.lock.json scripts/setup_local_env.sh scripts/bootstrap_ai_toolkit.sh scripts/check_cuda.py src/klein4b/ai_toolkit.py tests/test_ai_toolkit.py
git commit -m "chore: pin ai toolkit and bootstrap local environment"
```

### Task 3: Build The Demo Dataset Subset And AI Toolkit-Ready Captions

**Files:**
- Create: `src/klein4b/demo_dataset.py`
- Create: `scripts/bootstrap_demo_dataset.py`
- Test: `tests/test_demo_dataset.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_demo_dataset.py
from pathlib import Path

from klein4b.demo_dataset import (
    build_split_map,
    demo_caption,
    parse_identity_from_name,
)


def test_parse_identity_from_name() -> None:
    assert parse_identity_from_name("images/002386_makeup_03.jpg") == "002386"
    assert parse_identity_from_name("images/002386_bare.jpg") == "002386"


def test_split_map_is_deterministic() -> None:
    identities = [f"{i:06d}" for i in range(60)]
    split_map = build_split_map(identities, train_count=40, val_count=10, test_count=10, seed=7)
    assert len(split_map["train"]) == 40
    assert len(split_map["val"]) == 10
    assert len(split_map["test"]) == 10
    assert split_map["train"][0] == "000000"


def test_demo_caption_uses_trigger_without_style_words() -> None:
    caption = demo_caption("K4BMAKEUP03")
    assert caption.startswith("K4BMAKEUP03.")
    body = caption.partition(".")[2].lower()
    assert "makeup" not in body
    assert "lipstick" not in body
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_demo_dataset.py -q`
Expected: FAIL with import errors for `klein4b.demo_dataset`

- [ ] **Step 3: Write minimal implementation**

```python
# src/klein4b/demo_dataset.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import zipfile

from huggingface_hub import hf_hub_download


DATASET_REPO = "cyberagent/FFHQ-Makeup"


def parse_identity_from_name(path: str) -> str:
    name = Path(path).name
    return name.split("_")[0]


def build_split_map(
    identities: list[str],
    train_count: int,
    val_count: int,
    test_count: int,
    seed: int,
) -> dict[str, list[str]]:
    ordered = sorted(identities)
    random.Random(seed).shuffle(ordered)
    return {
        "train": sorted(ordered[:train_count]),
        "val": sorted(ordered[train_count:train_count + val_count]),
        "test": sorted(ordered[train_count + val_count:train_count + val_count + test_count]),
    }


def demo_caption(trigger_word: str) -> str:
    return (
        f"{trigger_word}. A centered close-up portrait of a person facing the camera, "
        "with shoulders visible, neutral framing, even studio lighting, and a simple background."
    )


def download_demo_zip(cache_dir: Path) -> Path:
    return Path(
        hf_hub_download(
            repo_id=DATASET_REPO,
            repo_type="dataset",
            filename="FFHQ-Makeup.zip",
            local_dir=cache_dir,
            local_dir_use_symlinks=False,
        )
    )


def available_identities(zip_path: Path, style_slot: str = "makeup_03") -> list[str]:
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
    identities = []
    for name in sorted(names):
        if not name.endswith(f"_{style_slot}.jpg"):
            continue
        identity = parse_identity_from_name(name)
        bare = f"images/{identity}_bare.jpg"
        target = f"images/{identity}_{style_slot}.jpg"
        if bare in names and target in names:
            identities.append(identity)
    return identities
```

```python
# scripts/bootstrap_demo_dataset.py
from __future__ import annotations

import json
from pathlib import Path
import zipfile

from klein4b.demo_dataset import available_identities, build_split_map, demo_caption, download_demo_zip
from klein4b.paths import data_dir


def main() -> None:
    root = data_dir() / "demo_ffhq_makeup"
    train_dir = root / "train"
    cache_dir = root / "cache"
    train_dir.mkdir(parents=True, exist_ok=True)

    zip_path = download_demo_zip(cache_dir)
    identities = available_identities(zip_path)
    split_map = build_split_map(identities, train_count=40, val_count=10, test_count=10, seed=7)

    manifest_rows = []
    trigger = "K4BMAKEUP03"

    with zipfile.ZipFile(zip_path) as zf:
        for split, ids in split_map.items():
            split_dir = root / split
            refs_dir = root / "references" / split
            split_dir.mkdir(parents=True, exist_ok=True)
            refs_dir.mkdir(parents=True, exist_ok=True)
            for identity in ids:
                bare_name = f"images/{identity}_bare.jpg"
                target_name = f"images/{identity}_makeup_03.jpg"
                ref_path = refs_dir / f"{identity}.jpg"
                tgt_path = split_dir / f"{identity}.jpg"
                txt_path = split_dir / f"{identity}.txt"

                ref_path.write_bytes(zf.read(bare_name))
                if split == "train":
                    tgt_path.write_bytes(zf.read(target_name))
                    txt_path.write_text(demo_caption(trigger))
                    caption_path = str(txt_path)
                else:
                    tgt_path.write_bytes(zf.read(target_name))
                    caption_path = None

                manifest_rows.append(
                    {
                        "split": split,
                        "identity_id": identity,
                        "reference_path": str(ref_path),
                        "target_path": str(tgt_path),
                        "caption_path": caption_path,
                    }
                )

    (root / "manifest.json").write_text(json.dumps(manifest_rows, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_demo_dataset.py -q`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add src/klein4b/demo_dataset.py scripts/bootstrap_demo_dataset.py tests/test_demo_dataset.py
git commit -m "feat: add demo dataset bootstrap pipeline"
```

### Task 4: Render The AI Toolkit Training Config And Launch Training Runs

**Files:**
- Create: `configs/train_flux2_klein_makeup_demo.template.yaml`
- Create: `src/klein4b/training.py`
- Create: `scripts/train_lora.py`
- Test: `tests/test_training.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_training.py
from pathlib import Path
import sys

from klein4b.training import build_training_command, render_training_config


def test_render_training_config_uses_flux2_klein_base_4b(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "demo_ffhq_makeup" / "train"
    dataset_dir.mkdir(parents=True)
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()

    config_text = render_training_config(dataset_dir=dataset_dir, output_dir=output_dir)

    assert 'black-forest-labs/FLUX.2-klein-base-4B' in config_text
    assert 'trigger_word: "K4BMAKEUP03"' in config_text
    assert 'steps: 1800' in config_text
    assert 'resolution:' in config_text


def test_build_training_command_points_to_ai_toolkit_run_py() -> None:
    command = build_training_command(Path("vendor/ai-toolkit"), Path("outputs/runs/demo/train_config.yaml"))
    assert command == [
        sys.executable,
        "vendor/ai-toolkit/run.py",
        "outputs/runs/demo/train_config.yaml",
    ]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_training.py -q`
Expected: FAIL with import errors for `klein4b.training`

- [ ] **Step 3: Write minimal implementation**

```yaml
# configs/train_flux2_klein_makeup_demo.template.yaml
job: extension
config:
  name: demo_ffhq_makeup_makeup03
  process:
    - type: diffusion_trainer
      training_folder: "{{ training_folder }}"
      device: "cuda"
      trigger_word: "K4BMAKEUP03"
      network:
        type: "lora"
        linear: 64
        linear_alpha: 32
        conv: 32
        conv_alpha: 16
      datasets:
        - folder_path: "{{ dataset_dir }}"
          caption_ext: "txt"
          resolution:
            - 512
      train:
        batch_size: 1
        steps: 1800
        lr: 0.00009
        optimizer: "adamw8bit"
        timestep_type: "weighted"
        content_or_style: "balanced"
      model:
        name_or_path: "black-forest-labs/FLUX.2-klein-base-4B"
        quantize: true
        low_vram: false
meta:
  name: "demo_ffhq_makeup_makeup03"
  version: "1.0"
```

```python
# src/klein4b/training.py
from pathlib import Path
import sys


def render_training_config(dataset_dir: Path, output_dir: Path) -> str:
    template = Path("configs/train_flux2_klein_makeup_demo.template.yaml").read_text()
    return (
        template
        .replace("{{ training_folder }}", str(output_dir))
        .replace("{{ dataset_dir }}", str(dataset_dir))
    )


def build_training_command(ai_toolkit_dir: Path, config_path: Path) -> list[str]:
    return [sys.executable, str(ai_toolkit_dir / "run.py"), str(config_path)]
```

```python
# scripts/train_lora.py
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import subprocess

from klein4b.training import build_training_command, render_training_config


def main() -> None:
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_dir = Path("outputs/runs") / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    config_path = run_dir / "train_config.yaml"
    config_path.write_text(
        render_training_config(
            dataset_dir=Path("data/demo_ffhq_makeup/train"),
            output_dir=run_dir,
        )
    )

    command = build_training_command(Path("vendor/ai-toolkit"), config_path)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_training.py -q`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add configs/train_flux2_klein_makeup_demo.template.yaml src/klein4b/training.py scripts/train_lora.py tests/test_training.py
git commit -m "feat: add ai toolkit config rendering and training launcher"
```

### Task 5: Add Local Reference-Image Inference For Base And LoRA Runs

**Files:**
- Create: `src/klein4b/inference.py`
- Create: `scripts/run_inference.py`
- Test: `tests/test_inference.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_inference.py
from klein4b.inference import build_demo_prompt, inference_defaults


def test_build_demo_prompt_is_transformation_focused() -> None:
    prompt = build_demo_prompt("K4BMAKEUP03")
    assert prompt.startswith("K4BMAKEUP03.")
    assert "preserving the person's identity" in prompt
    assert "image 2" not in prompt


def test_inference_defaults_match_photorealistic_demo_use_case() -> None:
    defaults = inference_defaults()
    assert defaults["guidance_scale"] == 4.0
    assert defaults["num_inference_steps"] == 24
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_inference.py -q`
Expected: FAIL with import errors for `klein4b.inference`

- [ ] **Step 3: Write minimal implementation**

```python
# src/klein4b/inference.py
from __future__ import annotations

from pathlib import Path

import torch
from diffusers import Flux2KleinPipeline
from PIL import Image


def build_demo_prompt(trigger_word: str) -> str:
    return (
        f"{trigger_word}. Change image 1 into a centered close-up beauty portrait with "
        "refined cosmetic styling, preserving the person's identity, pose, and framing. "
        "Keep the background simple and the lighting clean and natural."
    )


def inference_defaults() -> dict[str, float | int]:
    return {"guidance_scale": 4.0, "num_inference_steps": 24}


def run_local_inference(reference_path: Path, prompt: str, output_path: Path, lora_path: Path | None) -> None:
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
```

```python
# scripts/run_inference.py
from __future__ import annotations

import argparse
from pathlib import Path

from klein4b.inference import build_demo_prompt, run_local_inference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lora", type=Path, default=None)
    args = parser.parse_args()

    run_local_inference(
        reference_path=args.reference,
        prompt=build_demo_prompt("K4BMAKEUP03"),
        output_path=args.output,
        lora_path=args.lora,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_inference.py -q`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add src/klein4b/inference.py scripts/run_inference.py tests/test_inference.py
git commit -m "feat: add local base and lora inference harness"
```

### Task 6: Add Evaluation Metrics, Comparison Grids, And The Operator README

**Files:**
- Create: `src/klein4b/eval_metrics.py`
- Create: `src/klein4b/image_grid.py`
- Create: `scripts/compare_before_after.py`
- Create: `README.md`
- Test: `tests/test_eval_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eval_metrics.py
import numpy as np
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_eval_metrics.py -q`
Expected: FAIL with import errors for `klein4b.eval_metrics` or `klein4b.image_grid`

- [ ] **Step 3: Write minimal implementation**

```python
# src/klein4b/eval_metrics.py
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity


def ssim_score(image_a: Image.Image, image_b: Image.Image) -> float:
    arr_a = np.asarray(image_a.convert("RGB"))
    arr_b = np.asarray(image_b.convert("RGB"))
    score = structural_similarity(arr_a, arr_b, channel_axis=2)
    return round(float(score), 6)
```

```python
# src/klein4b/image_grid.py
from PIL import Image, ImageDraw


def make_four_up_grid(images: list[Image.Image], labels: list[str]) -> Image.Image:
    width, height = images[0].size
    canvas = Image.new("RGB", (width * 4, height + 28), color="black")
    draw = ImageDraw.Draw(canvas)
    for index, (image, label) in enumerate(zip(images, labels, strict=True)):
        canvas.paste(image, (index * width, 28))
        draw.text((index * width + 4, 6), label, fill="white")
    return canvas
```

```python
# scripts/compare_before_after.py
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from klein4b.eval_metrics import ssim_score
from klein4b.image_grid import make_four_up_grid


def main() -> None:
    manifest = json.loads(Path("data/demo_ffhq_makeup/manifest.json").read_text())
    test_rows = [row for row in manifest if row["split"] == "test"]

    for row in test_rows:
        identity = row["identity_id"]
        ref = Image.open(row["reference_path"]).convert("RGB")
        base = Image.open(f"outputs/eval/base/{identity}.png").convert("RGB")
        lora = Image.open(f"outputs/eval/lora/{identity}.png").convert("RGB")
        target = Image.open(row["target_path"]).convert("RGB")

        grid = make_four_up_grid([ref, base, lora, target], ["reference", "base", "lora", "target"])
        out_dir = Path("outputs/eval/grids")
        out_dir.mkdir(parents=True, exist_ok=True)
        grid.save(out_dir / f"{identity}.png")

        metrics = {
            "identity_id": identity,
            "base_ssim_to_target": ssim_score(base, target),
            "lora_ssim_to_target": ssim_score(lora, target),
        }
        metrics_dir = Path("outputs/eval/metrics")
        metrics_dir.mkdir(parents=True, exist_ok=True)
        (metrics_dir / f"{identity}.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
```

````markdown
# README.md

## Local Bootstrap

```bash
bash scripts/setup_local_env.sh
bash scripts/bootstrap_ai_toolkit.sh
source .venv/bin/activate
python scripts/check_cuda.py
```

## Demo Dataset

```bash
python scripts/bootstrap_demo_dataset.py
```

## Train

```bash
python scripts/train_lora.py
```

## Inference

```bash
python scripts/run_inference.py \
  --reference data/demo_ffhq_makeup/references/test/002386.jpg \
  --output outputs/eval/base/002386.png
LORA_PATH=$(find outputs/runs -name '*.safetensors' | sort | tail -n 1)
python scripts/run_inference.py \
  --reference data/demo_ffhq_makeup/references/test/002386.jpg \
  --lora "${LORA_PATH}" \
  --output outputs/eval/lora/002386.png
```

## Compare

```bash
python scripts/compare_before_after.py
```
````

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_eval_metrics.py -q`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add src/klein4b/eval_metrics.py src/klein4b/image_grid.py scripts/compare_before_after.py README.md tests/test_eval_metrics.py
git commit -m "feat: add evaluation grids metrics and operator docs"
```

## Verification Checklist

- [ ] Run `bash scripts/setup_local_env.sh`
- [ ] Run `bash scripts/bootstrap_ai_toolkit.sh`
- [ ] Run `source .venv/bin/activate && python scripts/check_cuda.py`
- [ ] Run `pytest -q`
- [ ] Run `python scripts/bootstrap_demo_dataset.py`
- [ ] Run `python scripts/train_lora.py`
- [ ] Run held-out base inference into `outputs/eval/base/`
- [ ] Run held-out LoRA inference into `outputs/eval/lora/`
- [ ] Run `python scripts/compare_before_after.py`
- [ ] Confirm every test identity has a grid in `outputs/eval/grids/`

## Self-Review

- Spec coverage:
  - WSL local environment: Task 2
  - demo dataset prep: Task 3
  - AI Toolkit LoRA config and launch: Task 4
  - local reference-image inference: Task 5
  - before/after comparison flow: Task 6
- Placeholder scan:
  - No `TODO`, `TBD`, or deferred implementation markers remain in task steps.
- Type consistency:
  - `K4BMAKEUP03` is the single trigger word across dataset prep, training, and inference.
  - The training model id and inference model id both use `black-forest-labs/FLUX.2-klein-base-4B`.

# Repository Guidelines

## Project Scope & Layout
This repo bootstraps a **selfie -> Greek marble statue bust** pipeline on **FLUX.2 Klein 4B** with a **style-only LoRA**. Keep identity out of the LoRA; the selfie stays a reference image at inference time. Use `configs/` for versioned AI Toolkit templates and lockfiles, `src/klein4b/` for reusable Python helpers, `scripts/` for CLI entry points, `data/` for local datasets, `tests/` for regression coverage, and `outputs/` for checkpoints, samples, and eval artifacts. Treat `data/raw/` and user selfies as private and untracked.

## Build, Test, and Development Commands
Prefer checked-in scripts over notebooks or shell history:

- `bash scripts/setup_local_env.sh`: create `.venv`, install the pinned CUDA 12.8 PyTorch stack, and install the package.
- `bash scripts/bootstrap_ai_toolkit.sh`: clone the pinned AI Toolkit checkout into `vendor/ai-toolkit/`.
- `python scripts/bootstrap_demo_dataset.py`: build the small FFHQ-Makeup demo split under `data/demo_ffhq_makeup/`.
- `python scripts/train_lora.py --config configs/train_flux2_klein_makeup_demo.template.yaml`: render a run config and launch LoRA training.
- `python scripts/run_inference.py --reference path/to/selfie.jpg --output outputs/eval/base/test.png [--lora path/to/weights.safetensors]`: run base or LoRA inference.
- `python scripts/compare_before_after.py`, `pytest -q`, `ruff check .`, `ruff format .`: compare outputs, test, lint, and format.

## Coding Style & Naming Conventions
Target Python 3.12 with small, testable modules. Use 4-space indentation in Python and 2 spaces in YAML/JSON/Markdown. Use `snake_case` for files, functions, and config names; use `PascalCase` for classes. Name configs by intent, such as `marble_lora.yaml` or `dataset_filtering.yaml`. Keep prompts, negative prompts, and augmentation rules explicit in config instead of burying them in code.

## Training & Testing Guidelines
Optimize for **style consistency**, not identity memorization. The current demo dataset is FFHQ-Makeup and exists only to validate the local fine-tune loop; the real marble dataset should enforce stone texture, blank white eyes, sculpted hair, bust-only framing, and no modern accessories. Add tests for dataset validation, config rendering, prompt assembly, and eval metrics. Save fixed-reference comparisons under `outputs/eval/` before merging model-affecting changes.

## Commit & Pull Request Guidelines
Use focused, imperative commits such as `data: add marble caption filter` or `train: tune lora rank for flux klein`. Pull requests should include the training goal, dataset/config changes, validation commands, sample outputs, and any known failure cases. Include before/after grids or metric summaries for model-affecting changes.

## Security & Data Handling
Never commit selfies, private datasets, secrets, or generated credentials. Keep local paths, API keys, and experiment caches in ignored files such as `.env.local` or tool-specific cache directories.

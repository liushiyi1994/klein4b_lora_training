# Z-Image Marble Base Design

## Objective
Add a second local training and inference path for the **selfie/reference image -> Greek marble statue bust** project using **Z-Image** in the same local environment and AI Toolkit flow already used for FLUX.2 Klein 4B.

The first milestone is not a full migration away from FLUX. It is a parallel, reproducible **Z-Image base** pipeline that can:

1. download the model from Hugging Face,
2. train a style-only LoRA with AI Toolkit,
3. run local image-conditioned inference from a reference selfie,
4. save evaluation outputs using the same comparison workflow already used for FLUX.

## Constraints
- Runtime: local WSL environment with RTX 5090
- Training launcher: local AI Toolkit checkout under `vendor/ai-toolkit/`
- Inference: local Diffusers pipeline, same machine, same `.venv`
- Learning objective: **style only**
- Identity must remain in the **reference image conditioning**, not inside the LoRA
- The existing FLUX pipeline must keep working while Z-Image is added

## Model Decision
### Recommended Phase 1 Target
Use **`Tongyi-MAI/Z-Image`** as the training target.

This is the cleanest match for the current goal:
- the official Z-Image family positions **Z-Image base** as the fine-tunable model,
- the local AI Toolkit checkout has a native `zimage` architecture path,
- the local Diffusers install already exposes both `ZImagePipeline` and `ZImageImg2ImgPipeline`.

### Optional Variant
Treat **`slzd/Z-Image-Base-0.36.0.dev0-fp32`** as an **optional full-precision experiment**, not the default bootstrap path.

It is useful if we later decide the quantized base path is leaving quality on the table, but it should not be the first milestone because:
- it increases memory risk on a 32 GB card,
- it adds one more axis of change while the main unknown is model behavior, not precision,
- AI Toolkit already has a straightforward base-model path using `Tongyi-MAI/Z-Image`.

### Out of Scope for This Design
Do **not** train on `Tongyi-MAI/Z-Image-Turbo` in this phase. AI Toolkit supports Turbo through a training adapter, but the clean long-term path is the base model, and the user explicitly prefers base-first.

## Approach Options
### Option A: Base-first parallel path
Create a dedicated Z-Image training and inference path next to the existing FLUX path.

Pros:
- lowest risk to the current working FLUX flow,
- easiest handoff to another agent,
- simplest debugging because FLUX and Z-Image stay isolated.

Cons:
- some duplicate code around training/inference helpers.

### Option B: Refactor current FLUX utilities into a generic model abstraction first
Generalize the current training and inference helpers so both FLUX and Z-Image share one path.

Pros:
- cleaner long-term architecture if many more models are added.

Cons:
- too much surface area for the first Z-Image milestone,
- higher risk of breaking the current FLUX work while still learning Z-Image-specific requirements.

### Option C: Full-precision-first Z-Image bootstrap
Start directly from `slzd/Z-Image-Base-0.36.0.dev0-fp32`.

Pros:
- removes quantization as a possible quality limiter.

Cons:
- VRAM risk,
- slower iteration,
- more moving pieces before the baseline is established.

## Recommended Approach
Use **Option A**.

Add a **parallel Z-Image path** that mirrors the current FLUX workflow:
- checked-in AI Toolkit config template,
- small wrapper script to render and launch training,
- local img2img inference script,
- focused tests for config rendering and inference call wiring,
- saved eval outputs under `outputs/eval/`.

Keep the current FLUX code untouched except where shared helpers are already trivially reusable.

## Repo Design
### New Configs
- `configs/train_zimage_marble_bust_v1.template.yaml`

This template should be the Z-Image equivalent of the current marble FLUX templates:
- same marble-bust dataset folder shape,
- same trigger token strategy if needed,
- same sample prompt intent,
- same run output convention.

Default model settings should target the official base repo:
- `arch: "zimage"`
- `name_or_path: "Tongyi-MAI/Z-Image"`
- `quantize: true`
- `quantize_te: true`
- `qtype: "qfloat8"`
- `qtype_te: "qfloat8"`
- `low_vram: true`

The template should be easy to clone into a second variant later for fp32 experiments.

### New Python Modules
- `src/klein4b/zimage_training.py`
- `src/klein4b/zimage_inference.py`

Keep these separate from the current FLUX modules. They should own:
- Z-Image default template lookup,
- Z-Image config rendering helpers,
- Z-Image-specific inference defaults,
- Z-Image pipeline construction.

### New Scripts
- `scripts/train_zimage_lora.py`
- `scripts/run_zimage_inference.py`

These should mirror the ergonomics of the current FLUX scripts:
- `--config`
- `--dataset-dir`
- `--output-root`
- `--reference`
- `--output`
- `--lora`
- `--lora-scale`

### New Tests
- `tests/test_zimage_training.py`
- `tests/test_zimage_inference.py`

Keep the test style consistent with the current repo:
- verify config rendering contains the expected model settings,
- verify the CLI writes the rendered config into a timestamped run dir,
- verify inference constructs the expected Diffusers call without requiring real GPU inference in unit tests.

## Dataset Design
For phase 1, reuse the current marble dataset shape:
- image files in a folder,
- one `.txt` caption per image,
- existing manual-caption v4 data is the most useful starting point.

The dataset contract should remain:
- bust-only framing,
- marble style supervision only,
- identity excluded from training data objectives,
- reference image used only at inference time.

No new dataset format should be invented for Z-Image unless the model proves it requires one.

## Training Flow
### Phase 1 Default
Train a **style-only LoRA** on `Tongyi-MAI/Z-Image` using AI Toolkit.

The training path should look like the current FLUX workflow:
1. choose a checked-in YAML template,
2. render placeholders into a timestamped run directory,
3. launch `vendor/ai-toolkit/run.py <rendered-config>`,
4. archive and evaluate the resulting checkpoints.

### AI Toolkit Expectations
The local AI Toolkit checkout already supports:
- `arch: "zimage"`
- model loading from Hugging Face repos,
- Z-Image LoRA targeting,
- flowmatch scheduler defaults.

That means the other agent should **not** add a custom trainer or custom AI Toolkit extension first. The first pass should stay inside existing toolkit capabilities.

### FP32 Follow-on
If phase 1 succeeds and quality still looks bottlenecked by the quantized base path, create a follow-up config variant that swaps:
- `name_or_path: "slzd/Z-Image-Base-0.36.0.dev0-fp32"`
- `quantize: false`
- `quantize_te: false`

This should be treated as a controlled experiment after the quantized base path is already working.

## Inference Flow
Use **`ZImageImg2ImgPipeline`** for local reference-image inference.

The interface should stay close to the current FLUX inference workflow:
- input: reference selfie path,
- input: marble prompt string,
- optional input: LoRA weights path,
- output: rendered image path.

The prompt-building strategy should stay aligned with the current marble work:
- visible structural descriptors only,
- no race labels,
- no expression descriptors,
- explicit material and lighting instructions,
- style block aligned with the training captions.

The first inference milestone is simple:
- base Z-Image without LoRA,
- Z-Image + trained LoRA,
- same seed and prompt for repeatable comparison.

## Evaluation Flow
Save outputs under a new Z-Image-specific eval directory, for example:
- `outputs/eval/test_selfie_zimage_v1/`
- `outputs/eval/test_img_zimage_v1/`

Minimum comparison outputs:
- reference image,
- base Z-Image output,
- Z-Image LoRA output,
- target image where one exists.

The contact-sheet format should stay visually comparable to the current FLUX evaluation output so model-to-model review is easy.

Once the Z-Image path works, a useful comparison run is:
- current best FLUX checkpoint,
- Z-Image base,
- Z-Image LoRA,
- original target where available.

## Success Criteria
This design is successful when another agent can:
- download `Tongyi-MAI/Z-Image` locally from Hugging Face,
- train a LoRA with AI Toolkit from a checked-in repo config,
- run local image-conditioned inference with `ZImageImg2ImgPipeline`,
- save comparison sheets for `test_selfie.jpg` and the existing `test_img/` set,
- evaluate whether Z-Image reduces the persistent FLUX portrait-lighting bias.

## Risks and Guardrails
### Main Risk
Z-Image may still preserve portrait-photo lighting from the reference image, even if the base model differs from FLUX.

The first Z-Image milestone should therefore answer one concrete question:
**does Z-Image produce more convincing matte, asymmetrical marble lighting under image conditioning than the current FLUX LoRA path?**

### Guardrails
- Do not refactor the existing FLUX pipeline first.
- Do not introduce a new dataset format first.
- Do not make fp32 the bootstrap default.
- Do not claim Z-Image is better based on text-only samples; use image-conditioned evals.

## Implementation Boundary
This design should stop at:
- a working Z-Image base training path,
- a working Z-Image img2img inference path,
- saved local eval artifacts,
- a clear answer on whether Z-Image is worth pushing further.

It should not yet include:
- a full model-agnostic abstraction over every image model in the repo,
- production UX changes,
- automatic checkpoint sweeps,
- large-scale retraining grid search,
- full-precision default training.

## Sources
- Z-Image base model card: https://huggingface.co/Tongyi-MAI/Z-Image
- Z-Image fp32 mirror: https://huggingface.co/slzd/Z-Image-Base-0.36.0.dev0-fp32
- Diffusers Z-Image pipelines: https://huggingface.co/docs/diffusers/en/api/pipelines/z_image
- Local AI Toolkit support:
  - `vendor/ai-toolkit/README.md`
  - `vendor/ai-toolkit/ui/src/app/jobs/new/options.ts`
  - `vendor/ai-toolkit/extensions_built_in/diffusion_models/z_image/z_image.py`

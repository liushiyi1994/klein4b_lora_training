# Marble Model Inference Baseline Comparison

## Goal

Compare other base/edit models against the current best Klein result without training new LoRAs.

Current Klein baseline:

```text
outputs/eval/sample_style_unquantized_6500_all_test_img/generated_named/
```

This directory contains the selected unquantized Klein rank64 step 6500 sample-style outputs.

## Models

The comparison runner supports:

- `flux-dev`: `black-forest-labs/FLUX.1-dev` through `FluxImg2ImgPipeline`
- `z-image`: `Tongyi-MAI/Z-Image` through `ZImageImg2ImgPipeline`
- `qwen-edit`: `Qwen/Qwen-Image-Edit` through `QwenImageEditPipeline`

Klein is included as an existing output column. The script does not regenerate Klein, because the
best Klein result uses the AI Toolkit sample-style path with the trained LoRA.

## Shared Settings

Use the same visible comparison settings as the Klein sample-style selection:

- fixed reference images from `test_img/`
- optional target images from `test_img_target/`
- fixed marble prompts from the previous prompt directory when available
- `seed: 7`
- `width: 768`
- `height: 1024`
- `guidance/cfg: 2.5`
- `steps: 24`
- same long marble negative prompt where the model pipeline supports it

FLUX-dev and Z-Image are img2img pipelines, so they also need `img2img_strength`. Start with
`0.85`. Lower values usually preserve the selfie too much; higher values may ignore reference
structure.

## Command

```bash
.venv/bin/python scripts/eval_marble_model_baselines.py \
  --reference-dir test_img \
  --target-dir test_img_target \
  --prompt-dir outputs/eval/test_img_rich_result_caption_rank64_continue_4000_10000_fixed_prompt_g25_clean/prompts \
  --klein-dir outputs/eval/sample_style_unquantized_6500_all_test_img/generated_named \
  --output-dir outputs/eval/model_baseline_comparison \
  --model flux-dev \
  --model z-image \
  --model qwen-edit \
  --guidance-scale 2.5 \
  --num-inference-steps 24 \
  --seed 7 \
  --width 768 \
  --height 1024 \
  --img2img-strength 0.85
```

If the old fixed prompt directory is not present, omit `--prompt-dir`. The script will use a
generic marble bust prompt, which is still useful for a first model sanity check but is less directly
comparable to the Klein checkpoint sweep.

## Output

```text
outputs/eval/model_baseline_comparison/
  generated/
    flux_dev/<id>.png
    z_image/<id>.png
    qwen_edit/<id>.png
  contacts/
    <id>_contact.jpg
  prompts/
    <id>.txt
  run_config.json
  summary_contact.jpg
```

Each contact sheet uses this order:

```text
reference | target if present | klein_6500 | flux_dev | z_image | qwen_edit
```

Use `summary_contact.jpg` for quick selection, then inspect the per-ID contact sheets for details.

## What To Judge

Compare each model against Klein on:

- bust completeness: shoulders, torso, drapery, broken base
- marble material: matte low-albedo stone, pitting, weathered grey-brown patina
- face handling: blank sculpted eyes, no skin shine, no beauty-light catchlights
- hair handling: carved stone masses, no natural hair color or individual strands
- reference alignment: broad head angle, hair silhouette, and identity-adjacent structure
- prompt obedience: no modern accessories, no collage, no separate rock pile

## Important Caveat

This is not a perfectly identical pipeline comparison.

Klein and Qwen-Edit are edit/control-image style paths. FLUX-dev and Z-Image are img2img paths, so
the reference image behaves more like an initial canvas and needs `--img2img-strength`. The script
records this setting in `run_config.json` so the contact sheets can be interpreted correctly.

The purpose is to answer whether another base/edit model is promising enough to justify a full LoRA
training run. It is not a final trained-model comparison.

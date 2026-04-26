# Marble V4 Best Result: Unquantized Rank64 Checkpoint 6500

## Recommendation

Use the unquantized bf16 rank64/conv32 LoRA at step 6500 with AI Toolkit sample-style inference.

Recommended checkpoint:

```text
outputs/runs/marble_v4_pairs_rich_result_caption_rank64_unquantized/20260424-211309/marble_bust_v4_pairs_rich_result_caption_rank64_unquantized_style/marble_bust_v4_pairs_rich_result_caption_rank64_unquantized_style_000006500.safetensors
```

Recommended inference mode:

- AI Toolkit `diffusion_trainer` sample hook with `steps: 0`
- `network.pretrained_lora_path` pointing at the 6500 checkpoint
- `sample.sampler: flowmatch`
- `sample.width: 768`
- `sample.height: 1024`
- `sample.seed: 7`
- `sample.guidance_scale: 2.5`
- `sample.sample_steps: 24`
- Reference image passed as `ctrl_img_1`
- Keep the long marble negative prompt from the training config

The local all-test contact sheet for this selection is:

```text
outputs/eval/sample_style_unquantized_6500_all_test_img/summary_contact.jpg
```

These generated outputs are intentionally ignored and are not committed.

## Why 6500

Step 6500 gives the best observed balance between style strength and reference usability:

- Strong Greek marble bust composition with visible shoulders, torso, and drapery.
- Better full-bust structure than the earlier 4500 test.
- Less overcooked than later checkpoints, which can push harder into repeated dataset composition or artifact emphasis.
- Good preservation of broad face angle, hair mass, and identity-adjacent facial structure for the fixed test references.
- Cleaner use of blank sculpted eyes, weathered grey stone, carved hair, dark ember background, and localized lava in the broken base.

This is a visual selection, not an automated metric selection. The decision was based on side-by-side contact sheets against fixed references and available targets.

## How We Found It

1. Started from the current best quantized rich-caption rank64 run as the behavioral baseline.

2. Created a clean unquantized bf16 training config:

```text
configs/train_flux2_klein_marble_bust_v4_pairs_rich_result_caption_rank64_unquantized.template.yaml
```

Important config differences:

- `model.quantize: false`
- `model.quantize_te: false`
- no `qtype` / `qtype_te`
- kept `dtype: bf16`
- kept LoRA `linear: 64`, `conv: 32`
- kept `batch_size: 1`
- kept `gradient_checkpointing: true`
- kept `optimizer: adamw8bit`
- trained to `10000`
- saved every `500`

3. Ran the full unquantized training job:

```bash
.venv/bin/python scripts/train_lora.py \
  --config configs/train_flux2_klein_marble_bust_v4_pairs_rich_result_caption_rank64_unquantized.template.yaml \
  --output-root outputs/runs/marble_v4_pairs_rich_result_caption_rank64_unquantized
```

The run completed without OOM on the 5090. Checkpoint timestamps showed about 1.36 seconds per step, roughly 44 steps per minute, and about 3.85 hours wall time to final.

4. Ran the requested fixed-prompt diffusers sweep against the current best quantized 8500 baseline.

Compared:

- quantized 8500
- unquantized 4000
- unquantized 6000
- unquantized 7000
- unquantized 8000
- unquantized 8500
- unquantized 9000
- unquantized 9500
- unquantized 10000

Settings:

- fixed prompt files from the previous clean eval
- `guidance_scale: 2.5`
- `num_inference_steps: 24`
- `seed: 7`
- base Klein diffusers inference

Output:

```text
outputs/eval/test_img_rich_result_caption_rank64_unquantized_vs_quant8500_fixed_prompt_g25/
```

This sweep was useful for an apples-to-apples checkpoint comparison, but it did not match the in-training sample look.

5. Investigated the mismatch between diffusers inference and training samples.

The training samples looked more aligned with the desired marble bust style, especially around the 4500-6500 range. The key difference was that the samples were produced through AI Toolkit's internal sample path, not through the public diffusers `Flux2KleinPipeline` call.

6. Ran sample-style tests for unquantized 4500 and 6500 on `test_selfie`.

Output:

```text
outputs/eval/sample_style_unquantized_4500_6500_test_selfie/compare/test_selfie_sample_style_4500_6500_contact.jpg
```

Result:

- 4500 was closer/frontal and darker.
- 6500 had a cleaner full bust, better drapery, and stronger production-style composition.

7. Ran sample-style 6500 across all `test_img` references, including targets where present.

Output:

```text
outputs/eval/sample_style_unquantized_6500_all_test_img/summary_contact.jpg
```

Verification:

- 6 generated images
- 6 contact sheets
- targets included for 5 IDs
- no target for `test_selfie`

This confirmed 6500 as the best current result for actual inference.

## Inference Pipeline To Use

Use AI Toolkit's sample-style path for production inference. The practical pattern is:

```yaml
job: extension
config:
  name: sample_style_inference
  process:
    - type: diffusion_trainer
      training_folder: outputs/eval/sample_style_inference
      device: cuda
      trigger_word: "<mrblbust>"
      network:
        type: lora
        linear: 64
        linear_alpha: 64
        conv: 32
        conv_alpha: 32
        pretrained_lora_path: outputs/runs/marble_v4_pairs_rich_result_caption_rank64_unquantized/20260424-211309/marble_bust_v4_pairs_rich_result_caption_rank64_unquantized_style/marble_bust_v4_pairs_rich_result_caption_rank64_unquantized_style_000006500.safetensors
      train:
        steps: 0
        skip_first_sample: true
        disable_sampling: false
        batch_size: 1
        train_unet: true
        train_text_encoder: false
        gradient_checkpointing: true
        noise_scheduler: flowmatch
        optimizer: adamw8bit
        dtype: bf16
      model:
        arch: flux2_klein_4b
        name_or_path: black-forest-labs/FLUX.2-klein-base-4B
        quantize: false
        quantize_te: false
        low_vram: true
        model_kwargs:
          match_target_res: false
      sample:
        sampler: flowmatch
        width: 768
        height: 1024
        samples:
          - prompt: "Change image 1 into a <mrblbust> centered chest-up Greek marble statue bust ..."
            ctrl_img_1: path/to/selfie.jpg
        neg: "head-only crop, close-up face, cropped shoulders, cropped torso, loose rocks, separate rock pile, rock pedestal, debris below bust, flat plinth, individual hair strands, natural hair color, glossy hair, polished marble, wet shine, specular hotspots, frontal studio light, head-on key light, beauty lighting, perfect portrait lighting, head-on beauty lighting, spotlight on face, smooth porcelain face, overexposed white face, clean bright cheeks, shiny forehead, shiny nose, shiny lips, red or orange light on face, red or orange light on hair, glowing skin, glowing hair, soft hair strands, realistic hair texture, modern eyewear, modern accessories, duplicate figures, side-by-side views, collage"
        seed: 7
        guidance_scale: 2.5
        sample_steps: 24
```

Run it with:

```bash
.venv/bin/python vendor/ai-toolkit/run.py path/to/sample_style_inference.yaml
```

The generated image will be under:

```text
<training_folder>/<config.name>/samples/*.jpg
```

The trainer path also writes a checkpoint and optimizer file even when `steps: 0`; ignore those for inference.

## Diffusers Inference vs Sample-Style Inference

The earlier diffusers path used:

```python
pipe = Flux2KleinPipeline.from_pretrained(...)
pipe.load_lora_weights(...)
pipe(image=reference, prompt=prompt, ...)
```

The selected sample-style path uses:

```text
AI Toolkit diffusion_trainer
  -> loads flux2_klein_4b
  -> creates the rank64/conv32 LoRA network
  -> loads checkpoint through network.pretrained_lora_path
  -> encodes positive and negative prompts
  -> passes the reference as ctrl_img_1
  -> samples with the AI Toolkit flowmatch scheduler
```

Main practical differences:

- Diffusers passes a generic `image`; sample-style passes `ctrl_img_1`, which routes through FLUX.2 control-image conditioning.
- Diffusers did not use the long config negative prompt in our sweep; sample-style does.
- Diffusers loads LoRA through its adapter system; sample-style recreates the same LoRA network target structure used during training.
- Diffusers is useful for generic checkpoint sweeps; sample-style matches the in-training sample behavior and currently produces the better bust results.

## Current Limitation

The current best result is based on visual inspection of fixed contact sheets. There is no automated score yet for identity preservation, bust completeness, matte-stone quality, or modern-clothing suppression. If this becomes a repeated workflow, add a small script that generates the sample-style YAML, runs AI Toolkit, copies the latest sample to a stable output path, and writes a contact sheet.

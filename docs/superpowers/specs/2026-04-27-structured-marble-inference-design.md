# Structured Marble Inference Design

## Objective

Build and test a production-shaped inference path for the current best FLUX.2 Klein 4B marble bust setup:

1. use `gpt-5.4-mini` as a vision prompt planner,
2. generate a strict JSON description from the reference portrait selfie only,
3. render a deterministic final-image prompt from that JSON,
4. run AI Toolkit sample-style Klein inference with the known best checkpoint/settings,
5. compare the new output against the previous best sample-style output.

This branch is for testing the pipeline behavior first. It should not retrain the LoRA or change the dataset.

## Current Best Inference Contract

Use the existing best checkpoint:

```text
outputs/runs/marble_v4_pairs_rich_result_caption_rank64_unquantized/20260424-211309/marble_bust_v4_pairs_rich_result_caption_rank64_unquantized_style/marble_bust_v4_pairs_rich_result_caption_rank64_unquantized_style_000006500.safetensors
```

Use AI Toolkit sample-style inference:

- model: `black-forest-labs/FLUX.2-klein-base-4B`
- architecture: `flux2_klein_4b`
- LoRA rank: `linear: 64`, `conv: 32`
- `ctrl_img_1` for the reference portrait
- width: `768`
- height: `1024`
- seed: `7`
- guidance scale: `2.5`
- sample steps: `24`
- train steps: `0`
- long marble negative prompt from the v4 unquantized training config

## Evidence Behind The Change

The additional-set comparison showed that specific final-image prompts beat one generic prompt. The generic prompt often collapses toward similar adult philosopher busts and loses age, hair, headwear, child/female cues, and composition. The reference image contributes identity, pose, and hair silhouette through `ctrl_img_1`; the prompt must still describe the desired final statue.

The prompt-planning rule is:

- input image: identity, pose, broad expression, face structure, hair/headwear silhouette
- fixed project style: Greek marble bust style, stone material, blank eyes, bust framing, broken base, localized lava, weathering, and dark ember background
- prompt: final statue description built from reference-derived JSON plus fixed style constraints

## OpenAI API Design

Use the OpenAI Responses API with `gpt-5.4-mini`.

The request should include:

- planner instructions as text,
- the reference portrait selfie as the only image input,
- a strict JSON schema output format.

The VLM must not receive the pseudo target. When a pseudo target path is provided to the CLI, it is only a human comparison artifact for the contact sheet and run record; it does not affect JSON planning or prompt rendering.

The branch should keep OpenAI use isolated in one module so unit tests can fake the client without importing the real SDK. Runtime code should read the API key from the standard OpenAI client environment behavior and should not store keys in config or output artifacts.

## Prompt Plan JSON

The planner returns one object with exactly these top-level sections:

```json
{
  "reference_identity": {
    "age_band": "child | young adult | adult | middle-aged adult | elderly adult | unknown",
    "gender_presentation": "masculine | feminine | androgynous | unknown",
    "head_pose": "short visual phrase",
    "face_structure": ["visible structural cue"],
    "hair_or_headwear": ["visible silhouette cue"],
    "broad_expression": "neutral visual phrase"
  },
  "target_style": {
    "bust_framing": "short final-image phrase",
    "statue_angle": "short final-image phrase",
    "drapery_and_torso": ["style/composition cue"],
    "headpiece_or_ornament": ["only if useful"],
    "stone_surface": ["matte weathered marble cue"],
    "weathering": ["patina/crack/chip cue"],
    "base_and_lava": ["broken lower base cue"],
    "background": "dark ember background phrase"
  },
  "safety_overrides": {
    "identity_source_policy": "short phrase",
    "eye_policy": "short phrase",
    "material_policy": "short phrase",
    "banned_details": ["detail to suppress"]
  }
}
```

The schema must avoid race and ethnicity fields entirely. The prompt renderer should also defensively reject descriptor strings that contain obvious race or ethnicity labels, because this project explicitly excludes those labels.

`target_style` is produced from the project instructions and the reference image composition only. It is not extracted from a pseudo target image.

## Prompt Rendering

The deterministic renderer converts `PromptPlan` JSON into one final-image paragraph.

It must start with:

```text
Change image 1 into a <mrblbust> from the reference portrait,
```

The paragraph should then preserve reference-derived structural cues and add target-style cues. It must include these fixed constraints regardless of model output:

- use the reference portrait as the only identity, pose, face-structure, and hair/headwear source,
- require blank sculpted stone eyes or closed carved eyelids,
- ban pupils, irises, colored eyes, catchlights, painted eyes, and realistic human eyes,
- require hair, eyebrows, facial hair, head coverings, and ornaments to be carved from the same marble as the face,
- require matte weathered grey marble, rough pitted low-albedo stone, grey-brown mineral patina, grime in recesses, chipped edges,
- require localized lava only in the broken lower base,
- avoid glossy marble, wet shine, specular hotspots, selfie lighting, beauty lighting, modern accessories, duplicate figures, side-by-side views, and collage.

The renderer should be pure and unit-tested. It should not call OpenAI or AI Toolkit.

## AI Toolkit YAML Rendering

Add a helper that renders a temporary AI Toolkit inference YAML for a single image or a small batch. The helper should:

- include `network.pretrained_lora_path`,
- recreate the rank64/conv32 LoRA network shape,
- set `train.steps: 0`,
- set `train.disable_sampling: false`,
- set `sample.samples[].prompt`,
- set `sample.samples[].ctrl_img_1`,
- include the long negative prompt,
- use the best 768x1024, seed 7, guidance 2.5, 24-step settings.

The generated YAML is an artifact under the output directory, not a checked-in config template.

## CLI Design

Add `scripts/run_structured_marble_inference.py`.

Inputs:

- `--reference PATH` required,
- `--output-dir PATH` required,
- `--pseudo-target PATH` optional comparison-only image; never sent to OpenAI and never used for prompt rendering,
- `--previous-best PATH` optional,
- `--lora PATH` defaulting to the best 6500 checkpoint,
- `--model gpt-5.4-mini` default,
- `--prompt-plan-json PATH` optional cache/replay mode,
- `--skip-openai` only valid with `--prompt-plan-json`,
- `--no-run-ai-toolkit` to stop after JSON/prompt/YAML rendering.

Outputs:

- `prompt_plan.json`,
- `prompt.txt`,
- `sample_style_inference.yaml`,
- `run_config.json`,
- stable generated image path copied from AI Toolkit samples,
- contact sheet with reference, pseudo target when present, previous best when present, and new output.

## Test Strategy

Use TDD for implementation.

Required unit coverage:

- prompt plan schema/validation rejects race and ethnicity descriptors,
- renderer includes required fixed constraints,
- OpenAI planner builds a Responses request with exactly one image input and strict JSON schema,
- CLI treats `--pseudo-target` as comparison-only and never passes it to the planner,
- CLI supports cached JSON replay without an OpenAI call,
- AI Toolkit YAML renderer emits `ctrl_img_1`, best checkpoint/settings, negative prompt, and rank64/conv32 network settings,
- contact sheet includes the expected columns when previous best and pseudo target are provided.

Manual verification:

1. run the pipeline on one known additional-set reference,
2. compare against the previous best output from `outputs/eval/additional_test_set_klein6500_sample_style/generated_named/`,
3. inspect whether the structured prompt preserves age, hair/headwear, child/female cues, and target-like composition better than the generic prompt ablation.

## Baseline Branch Fixes

The global worktree exposes two pre-existing baseline issues:

- `scripts/eval_marble_v4_checkpoint_sweep.py` imports `build_marble_pair_prompt`, but clean `main` does not define it,
- `tests/test_paths.py` assumes the repository checkout directory is named `klein4b`, which fails in a global worktree.

This branch may fix those narrowly so the inference worktree can have a meaningful test baseline. Those fixes should be kept separate from the new structured inference behavior.

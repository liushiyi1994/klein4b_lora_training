# Klein 9B Next Training Round

## Current Branch

Use this branch for the next training setup:

```bash
git switch feature/klein9b-base-training
```

The branch adds a Klein 9B base training template:

```text
configs/train_flux2_klein9b_marble_bust_v7_30_rank64_unquantized.template.yaml
```

The template targets `black-forest-labs/FLUX.2-klein-base-9B` with unquantized
training, rank 64 LoRA settings, and a paired dataset layout under:

```text
data/marble-bust-data/v7_30/dataset
data/marble-bust-data/v7_30/control
```

Keep `data/`, selfies, approved generated targets, and model outputs untracked.

## Goal

The Charlie's Tale image pipeline takes a user selfie and generates a Greek
marble statue bust portrait. The output should preserve likeness while applying
the project statue style:

- white blank eyes with no pupils or irises
- simplified carved hair, eyebrows, mouth, and teeth
- neutral or restrained expression
- Greek bust framing with clothing, drapery, or armor
- consistent dark background and amber particle atmosphere
- matte sculptural marble or weathered stone material

The LoRA should learn style only. Identity should come from the inference-time
reference image, not from memorized training subjects.

## Current Problems

Recent Klein 4B LoRA runs show these recurring failures:

- Outfit and helmet hallucination: crests misalign, helmet forms become
  nonsensical, and armor blends into face, hair, or ears.
- Melted visual structure: scales, armor panels, hair masses, and facial
  boundaries become soft, liquid, or unstable.
- Weak training data signal: the current synthetic set has about 72 pairs, but
  some targets already contain AI artifacts, bad helmets, weak lighting, or
  incorrect sculptural details.
- Poor material and lighting: marble can look flat, glossy, too smooth,
  spotlighted, overbright, or inconsistent between face, bust, and background.
- Overbroad generalization: too many helmet, armor, outfit, and pose variants
  are being compressed into too little data.
- Identity/style instability: outputs can drift identity, keep the face too
  human, or damage the statue conversion.

## Working Hypothesis

The biggest risk is not rank or base-model size. The biggest risk is target
quality.

A LoRA trained on flawed synthetic targets will learn those flaws. If target
images contain melted armor, hallucinated helmets, bad crest geometry, glossy
skin-like marble, or inconsistent lighting, the LoRA can amplify those artifacts
instead of correcting them.

For the next round, 30 clean approved images are likely more useful than 72
mixed-quality images. Add more images only if they meet the same quality bar.

## Recommended Data Strategy

Start with a narrow v7 set:

- Use only approved clean targets.
- Target 30 to 40 images for the first Klein 9B run.
- Keep one coherent statue look: same material, lighting, background, bust
  framing, and particle treatment.
- Limit outfit complexity. Prefer no helmet or one helmet family for the first
  run.
- Avoid mixing many helmet types, armor types, clothing silhouettes, and
  lighting styles in the same small LoRA.
- Reject targets with melted boundaries, distorted crests, armor-face blending,
  glossy highlights, human-looking eyes, modern accessories, or unstable hands,
  shoulders, or bust bases.
- Keep captions style-focused. Do not include identity-specific names,
  biography, or private subject details.

If separate visual looks are required, prefer separate subsets and possibly
separate LoRAs:

- clean draped bust, no helmet
- one helmet family with simple armor
- ornate armor/helmet variant after the simple looks work

## Recommended Training Setup

### fal.ai Offline Package Preparation

For the next round, prepare data for fal.ai first instead of launching local AI
Toolkit training. The fal edit trainer expects a zip of paired files:

```text
<root>_start.<ext>  # reference/control selfie
<root>_end.<ext>    # approved target bust
<root>.txt          # optional per-pair edit instruction
```

Create the offline package with:

```bash
python scripts/prepare_fal_training_package.py \
  --dataset-root data/marble-bust-data/v7_30 \
  --output-root outputs/fal_training_packages \
  --package-name marble_v7_30_klein9b_fal \
  --steps 3600 \
  --learning-rate 0.00004
```

This writes:

```text
outputs/fal_training_packages/marble_v7_30_klein9b_fal.zip
outputs/fal_training_packages/marble_v7_30_klein9b_fal_manifest.json
outputs/fal_training_packages/marble_v7_30_klein9b_fal_request.json
```

The command is offline only. It does not upload files, call fal.ai, require
`FAL_KEY`, or spend money. After the zip is uploaded later, place the uploaded
zip URL into `image_data_url` in the request JSON and submit it to:

```text
fal-ai/flux-2-klein-9b-base-trainer/edit
```

Prefer per-pair captions under `dataset/<stem>.txt`. If captions are not ready
and you intentionally want one fallback edit instruction, pass both
`--default-caption` and `--allow-missing-captions`; otherwise missing captions
fail validation.

### Local AI Toolkit Fallback

Use the new Klein 9B template first:

```bash
python scripts/train_lora.py \
  --config configs/train_flux2_klein9b_marble_bust_v7_30_rank64_unquantized.template.yaml \
  --output-root outputs/runs/marble_v7_30_klein9b_rank64_unquantized
```

Initial settings:

- base: `black-forest-labs/FLUX.2-klein-base-9B`
- arch: `flux2_klein_9b`
- quantization: off
- LoRA rank: `linear: 64`, `conv: 32`
- training length: `3600` steps, roughly 120 passes over 30 images at batch size 1
- learning rate: `0.00004`
- text encoder training: off
- `content_or_style: "style"`

Keep this config as the local fallback and as the source of current training
intent. For fal.ai, the repo no longer controls AI Toolkit-specific settings
such as `arch`, `quantize`, `content_or_style`, sample prompts, checkpoint save
frequency, or optimizer internals. fal controls the trainer runtime; this repo
controls the paired zip contents, captions, step count, learning rate, and output
LoRA format.

Keep rank 64 for the first clean 9B run. Move to rank 96 only if a clean set
still underfits, such as weak marble texture, weak blank eyes, poor bust
framing, or inconsistent carved-hair style. Do not increase rank to compensate
for bad targets.

## Evaluation Checklist

After each checkpoint sweep, review fixed references for:

- likeness preservation from the reference image
- blank white eyes with no pupils, irises, dark centers, or eye markings
- clean boundaries between face, hair, helmet, ears, armor, and drapery
- Greek bust framing with visible shoulders and upper torso
- stable helmet or no helmet, depending on the subset goal
- matte marble material with useful shadowing and sculptural depth
- no glossy skin, wet shine, beauty lighting, or overbright face
- no modern accessories copied from the reference
- no melted scales, armor panels, crest details, or robe folds

If failures cluster around one look, split that look into its own subset instead
of broadening the caption or increasing rank.

## Next Actions

1. Place the approved v7 targets under `data/marble-bust-data/v7_30/dataset`.
2. Place matching selfie/reference controls under
   `data/marble-bust-data/v7_30/control`.
3. Ensure image stems match between target, control, and caption files.
4. Audit each target against the rejection criteria above.
5. Train the unquantized Klein 9B rank64 config.
6. Run checkpoint samples on fixed references and compare against prior 4B runs.
7. Decide whether to add more clean data, split looks into separate LoRAs, or
   try rank 96.

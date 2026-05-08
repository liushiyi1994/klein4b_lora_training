# Klein 9B Next Training Round

## Current Branch

Use this branch for the next training setup:

```bash
git switch feature/klein9b-base-training
```

The branch adds Klein 9B base training templates:

```text
configs/train_flux2_klein9b_marble_bust_v7_30_rank64_unquantized.template.yaml
configs/train_flux2_klein9b_marble_bust_v7_60_no_statue_ref_rank64_unquantized.template.yaml
configs/train_flux2_klein9b_marble_bust_v7_60_with_statue_ref_rank64_unquantized.template.yaml
```

All templates target `black-forest-labs/FLUX.2-klein-base-9B` with unquantized
training and rank 64 LoRA settings.

The current 60-image training data is organized under:

```text
data/marble-bust-data/v7_30/training/wearable_v1_15_no_statue_ref
data/marble-bust-data/v7_30/training/wearable_v1_15_with_statue_ref
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

For the next round, the 60 approved images are enough for an A/B training test.
Add more images only if they meet the same quality bar.

## Recommended Data Strategy

Start with the approved v7 set:

- Use only approved clean targets.
- Keep material, lighting, background, bust framing, and particle treatment
  consistent.
- Keep the four wearable families explicit in captions: Dionysian reveler,
  armored general, draped scholar, and helmeted hero-warrior.
- Avoid adding more helmet, armor, clothing, or lighting variants until this
  60-image set has been evaluated.
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

There are two 60-image setups:

1. `no_statue_ref`: target bust, caption, and selfie/reference portrait only.
   This is closest to the current production inference shape.
2. `with_statue_ref`: target bust, caption, selfie/reference portrait, and a
   second reference statue image for wearable detail. This tests whether an
   explicit statue detail reference reduces helmet and armor hallucination.

Run both with the same model, rank, learning rate, and step count so the
comparison isolates the data setup.

The current local data root is:

```text
/home/liush/projects/charlie_tale_ft/data_synthesis/klein4b_lora_training/data/marble-bust-data/v7_30/training
```

No-statue-reference run:

```bash
python scripts/train_lora.py \
  --config configs/train_flux2_klein9b_marble_bust_v7_60_no_statue_ref_rank64_unquantized.template.yaml \
  --dataset-dir /home/liush/projects/charlie_tale_ft/data_synthesis/klein4b_lora_training/data/marble-bust-data/v7_30/training \
  --output-root outputs/runs/marble_v7_60_no_statue_ref_klein9b_rank64_unquantized
```

With-statue-reference run:

```bash
python scripts/train_lora.py \
  --config configs/train_flux2_klein9b_marble_bust_v7_60_with_statue_ref_rank64_unquantized.template.yaml \
  --dataset-dir /home/liush/projects/charlie_tale_ft/data_synthesis/klein4b_lora_training/data/marble-bust-data/v7_30/training \
  --output-root outputs/runs/marble_v7_60_with_statue_ref_klein9b_rank64_unquantized
```

Initial settings:

- base: `black-forest-labs/FLUX.2-klein-base-9B`
- arch: `flux2_klein_9b`
- quantization: off
- LoRA rank: `linear: 64`, `conv: 32`
- training length: `6000` steps, roughly 100 passes over 60 images at batch size 1
- learning rate: `0.00004`
- text encoder training: off
- `content_or_style: "style"`

Keep rank 64 for the first clean 9B runs. Move to rank 96 only if the clean set
still underfits, such as weak marble texture, weak blank eyes, poor bust
framing, or inconsistent carved-hair style. Do not increase rank to compensate
for bad targets.

If the `with_statue_ref` run wins, production inference must also provide a
second statue reference image or a deterministic statue-reference selection
step. If production will only have the user selfie, prefer the `no_statue_ref`
run unless the quality gap is large enough to justify changing inference.

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

1. Audit the 60 targets against the rejection criteria above.
2. Train the no-statue-reference config.
3. Train the with-statue-reference config.
4. Run checkpoint samples on the same fixed references for both runs.
5. Compare identity, blank eyes, wearable structure, helmet stability, armor
   boundaries, material, lighting, and background consistency.
6. Decide whether to use the production-like no-statue-ref setup, change
   inference to include statue references, split looks into separate LoRAs, or
   try rank 96.

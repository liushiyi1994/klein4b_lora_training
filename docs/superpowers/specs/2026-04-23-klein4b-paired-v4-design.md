# Klein4B Paired V4 Design

## Objective
Add a second **paired-data** training path to this repo for the existing **FLUX.2 Klein 4B** marble-bust project.

This path should stay in the current repo and current local environment. It is not a model-family migration. The goal is to test whether **paired reference->target supervision** helps the model stop carrying over selfie lighting, skin smoothness, and portrait-photo rendering into the marble result.

The first milestone is:

1. build a new **v4 paired dataset** from the existing selfie/bust pairs,
2. manually review and rewrite every pair caption to match the v4 weathered-face direction,
3. add a separate **paired Klein4B training config** that uses control images during training,
4. run paired inference/eval separately from the current target-only LoRA path.

## Constraints
- Keep using **`black-forest-labs/FLUX.2-klein-base-4B`**
- Keep using the local **AI Toolkit** checkout in `vendor/ai-toolkit/`
- Keep the current target-only v4 training path intact
- Do not overwrite old dataset versions in place
- Do not use race labels
- Do not use expression descriptors
- Identity should still come from structure in the reference/control image, not from LoRA memorization

## Problem Statement
The current best FLUX path is a style-only LoRA trained on target bust images plus captions. It improved material and hair treatment, but image-conditioned inference still carries over:

- head-on portrait lighting,
- smooth skin-like face planes,
- beauty-photo highlight placement,
- overly clean facial rendering.

That failure mode is consistent with the model preserving too much of the source image’s lighting and surface behavior. A paired training path is the right experiment when the question becomes:

**Can Klein4B learn “keep identity structure, but replace the source photo rendering with matte, weathered marble bust rendering”?**

## Acceptance Criteria
The paired Klein4B experiment should be judged against these three criteria:

1. **Target bust style reads correctly**
   - The output should read immediately as a weathered Greek marble bust.
   - It should preserve the intended material direction: matte stone, carved hair/headwear, blank sculpted eyes or carved eyelids, broken base, and localized lava only in the base cracks.
   - It does **not** need to match the paired target image one-to-one in every detail. The target is style supervision, not an exact reconstruction requirement.

2. **Identity is preserved well enough to remain recognizable**
   - The transformed bust should still read as the same person from the source image.
   - This means preserving the major structural cues: face shape, proportions, nose/lip/jaw structure, age band, gender presentation where visually clear, head angle, and hair/headwear silhouette.
   - It does **not** require exact photographic likeness or perfect one-to-one similarity.

3. **Studio-photo lighting is removed**
   - The result should not look like a selfie or portrait photo rendered in stone.
   - It should avoid frontal beauty lighting, even face illumination, shiny forehead/nose/lips, smooth skin-like planes, and clean studio highlight placement.
   - The face should read as stone under subdued, non-beauty lighting rather than a polished portrait.

## Existing Evidence
The local AI Toolkit checkout already supports control-image paths for `flux2_klein_4b`. The UI config metadata exposes:

- `datasets.multi_control_paths`
- `sample.multi_ctrl_imgs`

The underlying `flux2` model code also already handles control images in generation and training-related plumbing. That means this experiment does **not** require a custom trainer before starting.

## Approaches
### Option A: New paired v4 dataset + separate paired Klein4B config
Create a new dataset version dedicated to paired training and a new training config dedicated to control-image supervision.

Pros:
- clean provenance,
- easy comparison against current bust-only v4,
- lowest risk to current work,
- easiest handoff to another agent.

Cons:
- duplicates some files from older dataset versions,
- requires manual caption review across all pairs.

### Option B: Rewrite `v3/pairs` in place and reuse it as the paired dataset
Reuse the current pair folder and only change captions/configs.

Pros:
- faster.

Cons:
- destroys dataset-version meaning,
- makes future comparison confusing,
- mixes old and new intent into one version label.

### Option C: Auto-generate paired captions from existing v4 bust captions
Programmatically derive pair captions instead of manually reviewing them.

Pros:
- fastest.

Cons:
- directly conflicts with the user’s instruction to review captions manually,
- likely to miss image-specific structural cues,
- weaker supervision quality.

## Recommended Approach
Use **Option A**.

Create a new paired dataset version under a new v4-specific folder, manually rewrite all pair captions, and add a dedicated paired Klein4B training path. This keeps the current target-only v4 LoRA path available as the baseline while making the paired experiment clean to evaluate.

## Dataset Design
### New Dataset Root
- `data/marble-bust-data/v4_pairs_weathered_face/`

### Dataset Structure
- `targets/`
  - bust target images, one per pair item
- `control/`
  - selfie/source images, matched by basename
- `captions/` is **not** needed as a separate folder if AI Toolkit expects caption files beside the target images

Recommended actual on-disk layout:

- `data/marble-bust-data/v4_pairs_weathered_face/targets/<id>.jpg`
- `data/marble-bust-data/v4_pairs_weathered_face/targets/<id>.txt`
- `data/marble-bust-data/v4_pairs_weathered_face/control/<id>.jpg`

The basenames must match so AI Toolkit can pair `folder_path` target images with `control_path` source images.

### Source Material
Start from the existing paired assets currently in:

- `data/marble-bust-data/v3/pairs/*_input.jpg`
- `data/marble-bust-data/v3/pairs/*_target.jpg`
- `data/marble-bust-data/v3/pairs/*.txt`

The new v4 paired dataset should be a transformed copy, not an in-place edit.

### Caption Rules
Every caption should be manually reviewed and rewritten to match the v4 weathered-face direction.

Required rules:
- no race labels,
- no expression descriptors,
- no adjectives tied to mood or emotion,
- preserve visible structural identity cues only:
  - age band,
  - gender presentation where visually clear,
  - face shape,
  - jaw, cheekbone, nose, lip structure,
  - hair/headwear silhouette,
  - head angle.

Required style block:
- matte weathered grey-white or dark ash-grey marble,
- dry chalky unpolished stone,
- rough pitted low-albedo face planes,
- grime/mineral patina where visible in the target,
- blank sculpted eyes or carved eyelids,
- hair/headwear carved from the same marble,
- localized lava only inside broken base cracks,
- no glossy polished marble,
- no wet shine,
- no specular hotspots,
- no frontal studio light,
- no head-on beauty lighting,
- no orange spill onto face/hair/torso/background.

### Caption Shape
The pair captions should remain **transformation instructions**, not plain target descriptions.

Recommended pattern:
- start with `transform into` or equivalent edit framing,
- say to preserve facial structure and identity,
- specify only structural target attributes,
- then enforce the marble material and anti-lighting constraints.

## Training Design
### New Config
- `configs/train_flux2_klein_marble_bust_v4_pairs_weathered_face.template.yaml`

### Model
- `arch: "flux2_klein_4b"`
- `name_or_path: "black-forest-labs/FLUX.2-klein-base-4B"`

### Dataset Config
Use:
- `folder_path` -> target bust images
- `control_path_1` -> source selfie images

For this first milestone, use **one** control image per sample. Do not start with multi-control training. The target image filename in `folder_path` and the source image filename in `control_path_1` must share the same basename so AI Toolkit can match them.

### Separation from Existing Training
This must be a distinct config from:
- `configs/train_flux2_klein_marble_bust_v4_manual_weathered_face.template.yaml`

Do not retrofit paired controls into the existing target-only config. The two training paths need to remain directly comparable.

### Run Naming
Recommended run root:
- `outputs/runs/marble_v4_pairs_weathered_face/`

Candidate checkpoints should be archived the same way as the existing v3/v4 runs, because AI Toolkit retention is limited.

## Inference Design
### New Inference Path
The existing inference code should stay usable, but the paired experiment needs a separate evaluation convention so results do not get mixed with the target-only run outputs.

Add either:
- a new paired eval script, or
- a small extension to the current eval flow that clearly names the paired outputs.

Recommended output roots:
- `outputs/eval/test_selfie_klein_v4_pairs_weathered_face/`
- `outputs/eval/test_img_klein_v4_pairs_weathered_face/`

For sampled validation prompts in the AI Toolkit config, use `ctrl_img_1` to point at the source selfie/control image.

### Prompt Strategy
Keep the same prompt principles already established for marble inference:
- visible structural descriptors only,
- no race labels,
- no expression descriptors,
- explicit anti-gloss and anti-studio-light constraints,
- blank sculpted eyes and carved marble hair.

The paired training path should not remove prompt discipline. It should reduce the model’s tendency to ignore that prompt discipline during image conditioning.

## Evaluation Design
Minimum comparison set:
- source selfie/control image,
- original target bust image where available,
- current best target-only FLUX result,
- new paired Klein4B result.

The most important evaluation rows are:
- `test_selfie.jpg`
- `1.png`
- `7.png`
- `eastasian_child_neutral_1.png`
- `european_middleaged_neutral_3.png`
- `southasian_middleaged_male_3.png`

The main evaluation question is:

**Does paired Klein4B training reduce frontal portrait-light carryover and smooth skin-like facial rendering more than the current target-only v4 LoRA path?**

## Risks and Guardrails
### Risk 1: Overfitting identity mappings
Because the dataset is paired, the model may memorize specific face mappings instead of learning the general transformation rule.

Guardrail:
- keep the dataset small but clean,
- keep prompts structural rather than identity-specific,
- compare on held-out and off-dataset references.

### Risk 2: Control image dominates style learning
The model may still copy too much source-photo lighting if the captions are weak or too generic.

Guardrail:
- manual caption review is mandatory,
- captions must positively describe the weathered matte stone material and explicitly reject photo lighting.

### Risk 3: Confusing experimental provenance
If the paired dataset overwrites old versions, comparisons become hard to trust.

Guardrail:
- create a new v4 paired dataset version,
- keep v3 pairs and current v4 bust-only data unchanged.

## Implementation Boundary
This design should stop at:
- a reviewed v4 paired dataset,
- a separate paired Klein4B training config,
- a reproducible paired training launch path,
- separate eval outputs for paired vs target-only comparison.

It should not yet include:
- a generalized multi-model training abstraction,
- new UI/product work,
- automatic large-scale sweeps,
- architectural refactors to the current FLUX path unless directly needed for the paired pipeline.

## Recommended File Set
- `data/marble-bust-data/v4_pairs_weathered_face/`
- `configs/train_flux2_klein_marble_bust_v4_pairs_weathered_face.template.yaml`
- `scripts/prepare_marble_v4_pairs_dataset.py`
- `tests/test_marble_dataset.py` updates or a dedicated paired-dataset test file
- `tests/test_training.py` updates for the new paired config
- optional new eval helper if the current ones become too awkward

## Success Criteria
Another agent should be able to:
- regenerate the paired v4 dataset from existing pair assets,
- inspect the rewritten pair captions,
- launch paired Klein4B training from a checked-in config,
- run paired eval outputs side by side with current target-only FLUX outputs,
- make a grounded call on whether paired training is better for the marble transformation task according to the three acceptance criteria above.

## Sources
- Local paired-dataset assets under `data/marble-bust-data/v3/pairs/`
- Current target-only v4 captions under `data/marble-bust-data/v4_manual_weathered_face/busts/`
- AI Toolkit `flux2_klein_4b` options in `vendor/ai-toolkit/ui/src/app/jobs/new/options.ts`
- AI Toolkit FLUX2 model control-image plumbing in `vendor/ai-toolkit/extensions_built_in/diffusion_models/flux2/flux2_model.py`

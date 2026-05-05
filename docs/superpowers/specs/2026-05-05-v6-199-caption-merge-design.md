# V6 199 Caption Merge Design

## Goal

Build a reproducible local preparation path for the 199-row selfie-to-bust
dataset from:

```text
https://github.com/watsondg/charliestale-ml/tree/data-prep/data-synthesis/editing-dataset-control
```

The prepared dataset should keep the paired-control layout expected by AI
Toolkit:

```text
data/marble-bust-data/v6_199/
  control/   # selfie/reference images
  dataset/   # bust images and rewritten captions
  archetypes.json
  manifest.json
  CAPTION_REPORT.md
```

The rewritten captions should merge the existing v5 marble-caption style with
the new dataset captions. They should preserve the new captions' demographic
wording, including race or ethnicity labels when present, while using consistent
helmet and garment detail patterns from `archetypes.json`.

## Source Inputs

- Remote control images: `editing-dataset-control/control/{NNN}.png`
- Remote bust images and captions:
  `editing-dataset-control/dataset/{NNN}.png` and `{NNN}.txt`
- Remote row metadata: `editing-dataset/manifest.json`
  - Maps `NNN` to `source_id` and `clothing_slug`.
- Local archetype mapping:
  `data/marble-bust-data/v6_199/archetypes.json`
  - Maps `clothing_slug` to `[helmet_archetype, garment_archetype]`.

The preparation command should not infer archetypes from caption text when the
manifest and `archetypes.json` provide an exact mapping.

## Caption Contract

Each rewritten caption should be a single prose caption, not trigger-token
training text. It should follow this shape:

```text
transform into a <mrblbust> from the reference portrait, preserving <source
demographic and selfie identity details>. The target is a <pose/framing from
the source caption> with blank sculpted eyes, <fixed helmet/garment detail
phrase>, matte weathered grey or grey-brown stone, grime in the relevant carved
grooves, chipped shoulders, broken lower bust, localized lava only in the lower
cracks and fractures, dark ember background, no glossy polished marble, no wet
shine, no specular hotspots, no frontal studio light, no head-on beauty
lighting, no orange spill on face hair torso or background
```

Rules:

- Preserve the source demographic phrase, including race or ethnicity labels if
  present.
- Preserve source selfie details such as age/gender wording, face shape, hair,
  facial features, pose, gaze, and expression.
- Use `<mrblbust>` as the style trigger.
- Use prose-only archetype descriptions; do not emit raw keys such as
  `ctCorinthian`.
- Replace vague source garment/helmet sentences with deterministic archetype
  detail phrases.
- Keep v5-style material and negative-detail language consistent across every
  row.
- Do not add pseudo-target language or production-inference prompt-planner
  language. These are training captions for paired targets.

## Archetype Phrase Tables

Implementation should define a small local table for helmet prose and garment
prose. These phrases are the source of consistency across rows.

Helmet and headwear phrases:

- `ctCorinthian`: carved Corinthian helmet with cheek guards, raised brow band,
  and vertical crest channels.
- `ctAttic`: carved Attic helmet with open face, brow ridge, cheek guards, and
  shallow crest band.
- `ctPhrygian`: carved Phrygian helmet with forward-curving cap, folded crown,
  and side cheek guards.
- `ctChalcidian`: carved Chalcidian helmet with open face, cheek pieces, and
  clean arched brow.
- `ctPilos`: conical pilos helmet with simple raised rim and smooth carved cap.
- `ctWinged`: winged helmet with a tall crest, raised side wings, and carved
  brow guard.
- `ctFloralWreath`: floral wreath circling the head with carved petals and leaf
  clusters.
- `ctIvyWreath`: ivy-and-grape wreath circling the head with carved leaves and
  grape clusters.
- `ctVineWreath`: vine wreath circling the head with twisting carved stems and
  leaves.
- `ctTopknotBaldric`: topknot bound with a carved fillet and clean hair bands.
- `ctStephane`: stephane diadem above the brow with a smooth raised crown band.
- `ctBare`: no helmet phrase; preserve the source caption's carved hair or bare
  head shape.

Garment phrases:

- `ctWarriorCuirass`: warrior cuirass with shoulder plates, diagonal cloak, and
  carved chest relief.
- `ctOrnateCuirass`: ornate cuirass with raised chest medallion, shoulder
  plates, cloak edge, and carved relief trim.
- `ctAthenaAegisChiton`: aegis over a chiton with scaled chest panel and layered
  cloth folds.
- `ctAthenaAegisPeplos`: aegis over a peplos with scaled chest panel, broad
  shoulder drape, and deep robe folds.
- `ctSageHimationDiagonal`: diagonal himation pinned at one shoulder with broad
  carved robe folds.
- `ctCivilianHimationChiton`: chiton with himation pulled across one shoulder,
  layered neckline, and soft vertical folds.
- `ctTogate`: toga gathered into a heavy front fold with broad wrapped bands
  across the torso.
- `ctPeplosChiton`: peplos over chiton with stacked neckline layers and vertical
  drapery channels.
- `ctBeltedPeplos`: high-belted peplos with cinched waist, sleeve edge, and
  wet-drapery folds.

Rows with the same `[helmet_archetype, garment_archetype]` pair should receive
the same garment/helmet detail pattern, aside from pose and identity text
preserved from their source caption.

## CLI Shape

Add a checked-in script, expected name:

```bash
python scripts/prepare_marble_v6_199_dataset.py \
  --source /path/to/charliestale-ml/data-synthesis \
  --target data/marble-bust-data/v6_199
```

Defaults should support the sparse clone path used during local work when
practical, but the script must accept explicit paths so the user can point at a
fresh checkout.

The script should:

1. Validate that all 199 ids have control image, bust image, caption, manifest
   row, and archetype mapping.
2. Copy control images into `target/control/{NNN}.png`.
3. Copy bust images into `target/dataset/{NNN}.png`.
4. Rewrite captions into `target/dataset/{NNN}.txt`.
5. Write `target/manifest.json` with row id, source id, clothing slug,
   archetypes, source paths, target paths, and rewritten caption.
6. Write `target/CAPTION_REPORT.md` with counts and archetype coverage.

The script may replace the generated `control/` and `dataset/` subdirectories
inside the target, but it must preserve the existing target-level
`archetypes.json`.

## Validation And Tests

Use test-first implementation. Focused tests should cover:

- Parsing representative remote captions into preserved selfie details and
  target pose/framing.
- Preserving race/ethnicity labels when present in source captions.
- Mapping `NNN -> clothing_slug -> archetype pair` through manifest and
  `archetypes.json`.
- Rendering identical archetype detail prose for rows with the same archetype
  pair.
- Writing a complete 199-row control/dataset mirror with matching `.png` and
  `.txt` files.
- Failing clearly when a manifest row, source image, source caption, or
  archetype mapping is missing.

Existing tests for v5 config rendering and current inference behavior should
remain unchanged.

## Out Of Scope

- Training a new LoRA.
- Changing the production SageMaker inference prompt planner.
- Changing existing v4/v5 configs.
- Removing race or ethnicity labels from the source captions.
- Emitting trigger-token captions.

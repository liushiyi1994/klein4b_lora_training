# Wearable V1 Nano Batch Handoff

Date: 2026-05-07
Branch: `feature/data-synthesis`

## Current Data Paths

- Nano batch root: `data/marble-bust-data/v7_30/synthetic_generation/wearable_v1_15/`
- Main review sheet: `data/marble-bust-data/v7_30/synthetic_generation/wearable_v1_15/review_contact_sheet.jpg`
- Per-archetype review sheets:
  - `review_CSDIONYSIAN_REVELER.jpg`
  - `review_CSARMORED_GENERAL.jpg`
  - `review_CSDRAPED_SCHOLAR.jpg`
  - `review_CSHELMETED_HERO_WARRIOR.jpg`
- Pair QC manifest: `data/marble-bust-data/v7_30/synthetic_generation/wearable_v1_15/pair_qc.jsonl`
- Approved-only job manifest: `data/marble-bust-data/v7_30/synthetic_generation/wearable_v1_15/approved_jobs.jsonl`
- Statue reference pool: `data/marble-bust-data/v7_30/statue_reference_pool/wearable_v1/`
- Reusable output rubric: `docs/training/wearable_output_rubric.md`

## Pair QC Result

Manual review checked the generated bust against its copied statue reference for all 20 Nano rows. The goal was wearable alignment: wreath, drape, armor, or helmet must transfer from the reference without bare torso, anatomical chest cues, or wrong headgear.

Summary:

- 16 approved
- 1 approved with note
- 2 need regeneration
- 1 reference rejected
- 17 rows are listed in `approved_jobs.jsonl`

Approved counts after QC:

- `CSDIONYSIAN_REVELER`: 5
- `CSARMORED_GENERAL`: 3
- `CSDRAPED_SCHOLAR`: 5
- `CSHELMETED_HERO_WARRIOR`: 4

Rejected or needs regeneration:

| Row | Status | Reason |
| --- | --- | --- |
| `013_CSARMORED_GENERAL_middleeastern_elderly_female_2` | `reference_rejected` | The bell cuirass reference is a standalone bronze close-up with breast/anatomical forms and no bust context. It is now inactive in the reference manifest. |
| `015_CSHELMETED_HERO_WARRIOR_african_youngadult_male_1` | `needs_regeneration` | Helmet is present, but the torso reads too bare/athletic and not covered enough by cloak, armor, aegis, or drape. |
| `017_CSARMORED_GENERAL_southasian_elderly_neutral_3` | `needs_regeneration` | The reference is valid, but the generated bust loses the lion pauldron/Medusa cuirass and turns it into generic breastplate scrolls. |

Borderline but currently approved:

- `009_CSARMORED_GENERAL_hispaniclatino_middleaged_male_1`: usable armored/draped silhouette, but cuirass transfer is softer than the torso reference. Review again before final training.

## Reference Pool State

Active references after this review:

- `CSDIONYSIAN_REVELER`: 6 active
- `CSARMORED_GENERAL`: 4 active
- `CSDRAPED_SCHOLAR`: 6 active
- `CSHELMETED_HERO_WARRIOR`: 5 active

The armored class is temporarily under the previous 5-reference target because rank 2 and rank 5 were rejected after generation review. The generator now permits 4 active references per archetype so bad references do not stay active just to satisfy the old count. Before generating a full 15-per-class set, find at least one new Greek-style armored bust reference with a clear cuirass, high covered neckline, and no breast/nipple/anatomical torso cues.

## Prompt / Script Notes

Current Nano script:

```bash
python3 scripts/generate_wearable_synthetic_examples.py --limit 20 --generate --force
```

The prompt profile is `wearable_v2_smaller_9x16` in `src/klein4b/wearable_synthetic_jobs.py`.

Important constraints already in the prompt:

- 9:16 output
- bust occupies 45-55 percent of frame height
- no straight-on frontal busts
- match selfie eye state; open selfie eyes should become open blank marble eyes, with no pupils
- chest and shoulders fully covered
- no nipples, breast forms, bare torso, pectoral separation, or anatomical cuirass
- third image controls wearable cue only, not subject identity or hairstyle

## GPT Image 2 / FAL Note

The GPT Image 2 comparison path was wired in `scripts/generate_gpt_image2_fal_examples.py`, but fal returned:

```text
Enterprise ready endpoint 'openai/gpt-image-2/edit' is not automatically allowed
```

So the GPT Image 2 folder contains copied input assets/metadata only, not generated outputs:

`data/marble-bust-data/v7_30/synthetic_generation/wearable_v1_15_gpt_image_2_fal/`

## Next Steps On Another Machine

1. Pull this branch.
2. Review `review_contact_sheet.jpg` and `pair_qc.jsonl`.
3. Do not train on rows 013, 015, or 017.
4. Replace the rejected armored rank 5 reference before scaling to 15 per archetype. Until that replacement exists, the generator will cycle the 4 active armored refs and row 013 will re-render from rank 6 instead of the rejected rank 5 source.
5. Regenerate at least:
   - row 015 with stronger covered-torso helmeted prompt behavior
   - row 017 with stronger lion pauldron/Medusa cuirass transfer
   - row 013, either from the current active-rank cycle or after adding a new valid armored reference
6. Use `approved_jobs.jsonl` as the current clean subset if packaging a small interim training run.

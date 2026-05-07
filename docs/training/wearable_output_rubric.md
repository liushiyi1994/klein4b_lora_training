# Wearable Marble Bust Output Rubric

Use this rubric for generated marble bust examples made from:

1. style references
2. a statue/clothing reference
3. a subject selfie/control image

The goal is not to judge whether the generated statue is generally attractive. The goal is to decide whether the pair is clean enough for style LoRA training without teaching the model bad clothing, bad anatomy, wrong framing, or wrong statue behavior.

## Review Outcome

Assign exactly one status per generated row.

| Status | Use When | Training Use |
| --- | --- | --- |
| `approved` | No hard-fail issues and the wearable cue is clearly correct. | Safe for the current clean set. |
| `approved_with_note` | No hard-fail issues, but one non-critical quality issue should be tracked. | Usable for small/interim runs; re-review before final full training. |
| `needs_regeneration` | The reference is valid, but generation failed important prompt or alignment requirements. | Do not train. Regenerate with the same or improved prompt/reference. |
| `reference_rejected` | The statue/clothing reference itself is bad for this archetype or causes unsafe failure modes. | Do not train. Mark reference inactive and replace it. |
| `rejected` | The image has severe visual or policy-quality failure unrelated to a reusable reference issue. | Do not train. |

## Hard-Fail Gates

If any gate fails, the row cannot be `approved`.

### 1. Reference Validity

Fail the reference if:

- The wearable cue is not visible enough to guide generation.
- It is a detached product/object shot when the class requires a worn cue.
- It has strong anatomical chest, nipple, navel, or bare-torso signals.
- It belongs to the wrong archetype.
- It is not plausibly ancient Greek or Greek-style for the intended class.
- It is too cropped, headless, dark, blurry, or visually confusing to transfer.

Bad reference outcome: `reference_rejected`.

### 2. Wearable Alignment

Fail the generated output if the target archetype is not clearly visible:

- Dionysian: no clear carved wreath or no draped bust cue.
- Armored general: no clear cuirass/body armor, or armor becomes a shirt, bodice, bare torso, or generic drape.
- Draped scholar: no clear himation/chiton folds, or it becomes armor/helmet/wreath class.
- Helmeted hero-warrior: no worn helmet, detached/floating helmet, or helmet silhouette contradicts the reference.

Valid reference but bad transfer outcome: `needs_regeneration`.

### 3. Torso Safety

Fail if any of these appear:

- Bare chest, exposed sternum, exposed pectorals, exposed abdomen.
- Nipples, areolas, nipple-like armor bumps, pectoral discs, or breast cups.
- Visible navel, six-pack, muscle separation, or anatomical torso landmarks.
- Child subject with adult chest anatomy.
- Drapery slipping open so the torso reads as skin instead of clothing.

Outcome: usually `needs_regeneration`; mark `reference_rejected` if the reference encourages this failure.

### 4. Statue Material

Fail if the bust is not a single marble object:

- Colored eyes, pupils, irises, catchlights, painted face, skin tone, makeup, fabric color, bronze armor, green wreath leaves, or metal helmet.
- Photographic skin texture, pores, freckles, hair strands, or realistic human eyes.
- Zombie-like cracks, veins, scratches, or weathering across the face/body.

Small marble texture variation is acceptable. Color/material mismatch is not.

### 5. Identity And Eye State

Fail if:

- The output ignores the subject selfie so strongly that age/gender/facial structure is unusable.
- Open-eye selfies become fully closed eyes without reason.
- Open generated eyes contain pupils, irises, color, or glossy human catchlights.
- The output copies the statue reference face/hair instead of the subject.

The generated face does not need perfect identity. It should preserve the subject's broad age, gender presentation, facial structure, and hair category while becoming a Hellenic marble ideal.

### 6. Composition

Fail if:

- Not 9:16 for current v7 generation.
- Bust is cropped by frame edges.
- Bust is too large to have clear negative space.
- Straight-on frontal bust when the prompt requires three-quarter.
- Multiple figures, duplicate views, side-by-side views, or extra busts.
- Loose rubble/debris appears around or below the bust.

Minor scale variation is acceptable if the bust remains fully visible and centered.

## Scored Quality Dimensions

After hard gates, score each dimension from 0 to 3.

| Score | Meaning |
| --- | --- |
| 3 | Clean, clear, training-ready. |
| 2 | Usable, with minor weakness. |
| 1 | Weak; likely needs regeneration unless all other dimensions are excellent. |
| 0 | Fails the dimension. |

| Dimension | 3-Point Standard |
| --- | --- |
| Wearable transfer | Reference cue is clearly present and archetype-specific. |
| Torso coverage | Chest and shoulders are continuously clothed/armored with no anatomy leakage. |
| Marble/statue style | Single colorless carved marble material; no painted or living-human details. |
| Subject preservation | Broad identity, age, hair category, and face structure survive the transformation. |
| Eye handling | Eye state follows the selfie; open eyes are blank carved marble with no pupils. |
| Pose/framing | Three-quarter, centered, fully visible, 9:16, not too large. |
| Visual cleanliness | No bad cracks, blur, edge artifacts, extra figures, modern accessories, or debris. |
| Archetype consistency | Does not drift into another class or mix incompatible class symbols. |

Recommended status from score:

- `approved`: no hard-fail gates, all dimensions >= 2, and total score >= 20.
- `approved_with_note`: no hard-fail gates, one dimension is 1 or total score is 17-19.
- `needs_regeneration`: any generation hard-fail gate, or total score < 17.
- `reference_rejected`: reference validity gate fails.

## Archetype-Specific Checklist

### `CSDIONYSIAN_REVELER`

Required:

- Carved grape, ivy, floral, or leaf wreath on the head.
- Draped shoulder/chest garment, usually himation/chiton.
- No helmet, cuirass, shield, warrior armor, or bare torso.

Common reject reasons:

- Plain hair with no wreath.
- Wreath becomes colored leaves or flowers.
- Drapery opens into bare chest.

### `CSARMORED_GENERAL`

Required:

- Bare head unless the prompt/reference intentionally says otherwise.
- Clear cuirass, breastplate, pauldron, shoulder straps, gorgoneion, or dominant armor cue.
- Armor reads as plate/garment, not anatomical skin.

Common reject reasons:

- Smooth shirt/bodice instead of armor.
- Breast-shaped armor cups.
- Nipples or pectoral anatomy.
- Overly object-like torso copied from a detached cuirass reference.

### `CSDRAPED_SCHOLAR`

Required:

- Bare head.
- Clear himation/chiton drapery across chest and shoulders.
- Scholar/philosopher/orator silhouette, not warrior or Dionysian.

Common reject reasons:

- Head-only bust with no readable drape.
- Bare shoulder/chest.
- Accidental helmet, wreath, armor, or crown.

### `CSHELMETED_HERO_WARRIOR`

Required:

- Greek-style helmet worn on the head.
- Helmet silhouette follows the reference: low dome, cheek guards, pilos, Corinthian, crest only if present in the reference.
- Chest/shoulders covered by cloak, cuirass, aegis, or drape.

Common reject reasons:

- No helmet, floating helmet, or helmet held as an object.
- Invented plume/crest/horns/wings not in the reference.
- Bare athletic torso under the helmet.

## Review Procedure

Review in this order:

1. Inspect the statue/clothing reference alone. If it is bad, stop and mark `reference_rejected`.
2. Compare generated output to the reference for wearable cue transfer.
3. Check torso safety before judging style quality.
4. Check marble material and eye behavior.
5. Check identity preservation against the selfie.
6. Check framing and pose.
7. Assign status, total score, and one short note.

## Suggested QC JSONL Fields

```json
{
  "id": "013_CSARMORED_GENERAL_middleeastern_elderly_female_2",
  "archetype": "CSARMORED_GENERAL",
  "reference_filename": "05_commons_greek_bell_cuirass.jpg",
  "status": "reference_rejected",
  "score_total": 9,
  "scores": {
    "wearable_transfer": 1,
    "torso_coverage": 0,
    "marble_style": 2,
    "subject_preservation": 2,
    "eye_handling": 2,
    "pose_framing": 1,
    "visual_cleanliness": 1,
    "archetype_consistency": 0
  },
  "notes": "Reference is a standalone bronze bell cuirass close-up with anatomical chest forms and no bust context."
}
```

Keep notes short and concrete. A future reviewer should understand whether the next action is to keep, regenerate, or replace the reference.

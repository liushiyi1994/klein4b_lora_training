# Marble Bust Inference Output Rubric

Use this rubric for end-user inference outputs where the input is a selfie and the output is a marble bust. Optional inputs may include style reference busts, a wearable/statue reference, an archetype prompt, or a previous approved example.

Unlike training-data QC, inference QC treats identity as a hard requirement. A visually strong marble bust is still a failure if it does not look like the person in the selfie.

## Inputs To Review

Review the output with all available inputs:

1. subject selfie
2. generated bust
3. prompt or archetype text, if available
4. style reference busts, if used
5. wearable/statue reference, if used

If a reference bust is used, judge it as style/material/framing guidance unless the prompt explicitly says it should control clothing or headgear. The selfie controls identity.

## Review Outcome

Assign exactly one status per inference result.

| Status | Use When | User Action |
| --- | --- | --- |
| `pass` | Identity is preserved and no hard-fail issues are present. | Accept output. |
| `pass_with_note` | Identity passes and there are only minor style/framing issues. | Accept or optionally regenerate. |
| `regenerate` | The prompt/reference setup is valid, but the output fails a hard gate. | Regenerate with same or tightened prompt. |
| `prompt_needs_revision` | Failure appears caused by missing, conflicting, or weak prompt instructions. | Revise prompt before regenerating. |
| `reference_problem` | Style/wearable/reference bust is misleading, low quality, or conflicts with the task. | Replace reference before regenerating. |
| `fail` | Severe output failure not clearly fixed by a simple prompt/reference change. | Do not use output. |

## Hard-Fail Gates

If any gate fails, the output cannot be `pass`.

### 1. Identity Preservation

Identity must pass for all accepted inference results.

Pass identity when the output preserves most of the subject's recognizable structure:

- apparent age band
- gender presentation or neutral presentation
- face shape and proportions
- hair length/category and broad hairstyle
- major facial hair, if present
- nose, mouth, jaw, cheek, and brow structure at a broad level
- head orientation or expression when the prompt asks to preserve it

Fail identity if:

- The bust looks like a different person.
- Age changes substantially, such as adult to child or young adult to elderly.
- Gender presentation changes unintentionally.
- Hair length/category changes substantially, such as short hair becoming long flowing hair, bald becoming full hair, or curly becoming straight when not covered by helmet.
- Facial hair is invented or removed when it is identity-relevant.
- The face copies the reference bust/statue instead of the selfie.
- The output becomes a generic Greek face with little subject resemblance.
- Ethnicity or broad facial structure is erased into a single generic classical template.

Outcome: usually `regenerate`; use `prompt_needs_revision` if the prompt did not explicitly prioritize identity.

### 2. Statue Material

Fail if the bust does not read as a single carved marble statue:

- skin color, painted face, makeup, colored lips, colored hair, colored eyes
- visible pupils, irises, catchlights, glossy human eyes, or photographic eye detail
- bronze, fabric, green leaves, metal helmet, or other non-marble material
- realistic pores, freckles, skin texture, individual hair strands, or living-human surface
- zombie-like cracks, veins, scratches, or weathering across face, neck, hair, clothing, or armor

Small natural marble tonal variation is acceptable. The whole figure should still read as one unpainted stone object.

### 3. Eye Handling

Fail if:

- Open selfie eyes become closed without a clear prompt reason.
- Open generated eyes have pupils, irises, eye color, or catchlights.
- Eyes look alive, wet, painted, or human-photographic.
- Eyelids or sockets are malformed enough to break identity.

Pass when open selfie eyes become open blank carved marble eyes, and closed selfie eyes remain peacefully closed if that is the intended pose.

### 4. Bust And Composition

Fail if:

- Output is not a bust or head-and-shoulders sculpture.
- Bust is cropped by the frame edge.
- Bust is too large with no negative space, unless the requested format explicitly asks for close-up.
- Output contains multiple figures, duplicate views, side-by-side angles, extra heads, or extra busts.
- Output is straight-on frontal when the prompt requires three-quarter.
- Loose rocks, rubble, debris, text, watermark, UI, or modern background objects appear.
- The generated image is not the requested aspect ratio or delivery format.

For current v7 style, preferred output is 9:16, centered, fully visible, with enough dark negative space around the bust.

### 5. Torso And Clothing Safety

Fail if:

- Bare chest, exposed sternum, pectorals, abdomen, or bare torso appears.
- Nipples, areolas, breast cups, nipple-like armor bumps, pectoral discs, navel, or six-pack anatomy appear.
- Drapery opens to show torso skin.
- Child subject has adult chest anatomy.
- Clothing or armor is missing when the prompt/archetype requires it.

The chest and shoulders should be fully covered by carved drapery, armor, cloak, aegis, or bust termination.

### 6. Prompt And Reference Adherence

Fail if:

- The requested archetype is missing or wrong.
- Reference bust/style material, lighting, or framing is ignored.
- Wearable/statue reference clothing or headgear is ignored when it was provided for that purpose.
- Headgear is invented when prompt says bare head.
- Helmet, wreath, drape, or armor from a reference is copied into the wrong archetype.
- The output follows the reference bust identity more than the selfie identity.

If the prompt/reference instructions conflict, mark `prompt_needs_revision` or `reference_problem` instead of judging the model output alone.

## Scored Quality Dimensions

After hard gates, score each dimension from 0 to 3.

| Score | Meaning |
| --- | --- |
| 3 | Strong, user-facing quality. |
| 2 | Acceptable, with minor weakness. |
| 1 | Weak; use only if the user accepts tradeoffs. |
| 0 | Fails the dimension. |

| Dimension | 3-Point Standard |
| --- | --- |
| Identity preservation | Clearly resembles the selfie after marble stylization. |
| Age/gender/hair fidelity | Age band, gender presentation, and hair category are preserved unless intentionally changed. |
| Marble/statue material | Single unpainted carved marble object with statue-like surfaces. |
| Eye handling | Eye state follows selfie/prompt; open eyes are blank carved marble with no pupils. |
| Bust composition | Head-and-shoulders bust, centered, fully visible, correct aspect ratio, enough negative space. |
| Torso/clothing safety | Fully covered chest and shoulders, no anatomical torso cues. |
| Reference/style adherence | Follows style refs and optional wearable/reference bust without copying wrong identity. |
| Prompt/archetype adherence | Output matches requested archetype, pose, headgear, framing, and constraints. |
| Visual polish | Sharp enough, coherent sculpture, no artifacts, no extra figures, no debris or watermarks. |

Recommended status from score:

- `pass`: no hard-fail gates, identity score is 3, all other dimensions >= 2, and total score >= 23.
- `pass_with_note`: no hard-fail gates, identity score >= 2, no other dimension is 0, and total score is 19-22.
- `regenerate`: identity score < 2, any hard-fail gate, or total score < 19.
- `prompt_needs_revision`: output fails because instructions were missing, ambiguous, or contradictory.
- `reference_problem`: output fails because a reference input is poor or misleading.

## Identity Pass/Fail Examples

### Identity Pass

The bust can pass identity even if:

- facial details are idealized into Greek marble planes
- skin color is removed
- hair is simplified into carved groups
- expression becomes calmer
- small asymmetries are harmonized

It must still read as the same person at a glance when compared with the selfie.

### Identity Fail

The bust fails identity if:

- the subject's short hair becomes long statue hair because of a reference bust
- a young face becomes elderly or an elderly face becomes young
- a clean-shaven person gains a beard, or a bearded person loses a beard
- a child becomes an adult heroic bust
- the output copies the statue reference's face instead of the selfie
- the face is only a generic classical mask with no selfie-specific structure

## Reference Bust Handling

When a reference bust is included, review these separately:

- Style transfer: material, lighting, background, scale, and bust termination match the reference.
- Identity boundary: face, age, hair, and facial hair still come from the selfie.
- Wearable boundary: headgear/clothing comes from the prompt or wearable reference, not accidentally from the style bust.
- Framing boundary: output may match the reference composition but must not crop or enlarge beyond requested limits.

If the reference bust is beautiful but causes identity loss, the result fails.

## Review Procedure

Review in this order:

1. Compare generated bust to selfie for identity. If identity fails, mark `regenerate` or `prompt_needs_revision`.
2. Check eye handling.
3. Check marble material and living-human artifacts.
4. Check torso and clothing safety.
5. Check composition, aspect ratio, single-bust requirement, and negative space.
6. Compare against prompt, archetype, and any reference bust/wearable input.
7. Assign status, scores, and one short actionable note.

## Suggested QC JSON Fields

```json
{
  "input_selfie": "path/to/selfie.png",
  "generated_bust": "path/to/output.png",
  "reference_bust": "path/to/reference_bust.png",
  "wearable_reference": "path/to/wearable_reference.jpg",
  "prompt_id": "v7_inference_prompt_001",
  "status": "pass_with_note",
  "hard_fail": false,
  "score_total": 22,
  "scores": {
    "identity_preservation": 3,
    "age_gender_hair_fidelity": 2,
    "marble_style": 3,
    "eye_handling": 3,
    "bust_composition": 2,
    "torso_clothing_safety": 3,
    "reference_style_adherence": 2,
    "prompt_archetype_adherence": 2,
    "visual_polish": 2
  },
  "failure_reasons": [],
  "notes": "Identity is strong and marble eyes are correct; bust is slightly larger than the reference framing."
}
```

Keep notes actionable. The next reviewer should know whether to accept, regenerate, change the prompt, or replace a reference.

# Klein4B Paired V4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a paired-data FLUX.2 Klein 4B training and evaluation path that uses a new manually reviewed `v4_pairs_weathered_face` dataset to keep recognizable identity while removing studio-light carryover and preserving the weathered marble bust style.

**Architecture:** Add a second dataset-prep path that materializes AI Toolkit-compatible `targets/` and `control/` folders, forces manual caption rewrite with explicit stubs and caption auditing, and leaves the existing target-only v4 path untouched. Then add a dedicated paired AI Toolkit config and a separate paired-eval module that runs baseline-vs-paired image-conditioned inference and writes four-up comparison sheets.

**Tech Stack:** Python 3.12, local AI Toolkit (`flux2_klein_4b`), Diffusers `Flux2KleinPipeline`, Pillow, pytest, Ruff, local ignored `data/` and `outputs/` directories

---

## Planned File Structure

**Create**
- `configs/train_flux2_klein_marble_bust_v4_pairs_weathered_face.template.yaml`
- `scripts/prepare_marble_v4_pairs_dataset.py`
- `scripts/eval_marble_v4_pairs.py`
- `src/klein4b/paired_eval.py`
- `tests/test_paired_eval.py`

**Modify**
- `src/klein4b/marble_dataset.py`
- `tests/test_marble_dataset.py`
- `tests/test_training.py`

**Leave Unchanged On Purpose**
- `configs/train_flux2_klein_marble_bust_v4_manual_weathered_face.template.yaml`
- `scripts/train_lora.py`
- `src/klein4b/inference.py`
- `scripts/run_inference.py`

Those files already work for the target-only path. The paired experiment should stay parallel to them instead of refactoring them first.

---

### Task 1: Add Manual Pair-Caption Audit Helpers

**Files:**
- Modify: `src/klein4b/marble_dataset.py`
- Modify: `tests/test_marble_dataset.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_marble_dataset.py
from pathlib import Path

from klein4b.marble_dataset import (
    audit_manual_pair_caption_dir,
    validate_manual_pair_caption,
)


def test_validate_manual_pair_caption_rejects_stub_race_and_expression_terms() -> None:
    errors = validate_manual_pair_caption(
        "REWRITE REQUIRED. transform into a <mrblbust> smiling eastasian child, "
        "glossy polished marble face under studio light"
    )

    assert "caption still contains the rewrite placeholder" in errors
    assert "caption contains banned term: eastasian" in errors
    assert "caption contains banned term: smiling" in errors
    assert "caption is missing required phrase: preserve recognizable identity" in errors
    assert "caption is missing required phrase: dry chalky unpolished stone" in errors
    assert "caption is missing required phrase: no frontal studio light" in errors


def test_validate_manual_pair_caption_accepts_reviewed_v4_pair_caption() -> None:
    caption = (
        "transform into a <mrblbust> centered chest-up Greek marble statue bust of "
        "an adult woman, preserve recognizable identity and facial structure, keep "
        "the oval face, straight narrow nose, full lower lip, thick rope-like hair "
        "mass behind the shoulder, and slight three-quarter head turn, dark ash-grey "
        "weathered marble, dry chalky unpolished stone, rough pitted low-albedo face "
        "surface, brown-grey mineral patina across the face neck and torso where "
        "visible, blank sculpted eyes, hair headwear eyebrows and eyelashes carved "
        "from the same marble, jagged broken lower bust with localized lava only in "
        "the bottom cracks, no glossy polished marble, no wet shine, no specular "
        "hotspots, no frontal studio light, no head-on beauty lighting, no orange "
        "spill on face hair torso or background"
    )

    assert validate_manual_pair_caption(caption) == []


def test_audit_manual_pair_caption_dir_returns_per_file_errors(tmp_path: Path) -> None:
    targets_dir = tmp_path / "targets"
    targets_dir.mkdir()
    (targets_dir / "1.txt").write_text(
        "transform into a <mrblbust> centered chest-up Greek marble statue bust, "
        "preserve recognizable identity and facial structure, dark ash-grey "
        "weathered marble, dry chalky unpolished stone, rough pitted low-albedo "
        "face surface, blank sculpted eyes, hair carved from the same marble, "
        "localized lava only in the bottom cracks, no glossy polished marble, "
        "no frontal studio light, no head-on beauty lighting, no orange spill "
        "on face hair torso or background",
        encoding="utf-8",
    )
    (targets_dir / "2.txt").write_text(
        "transform into a <mrblbust> smiling european adult man",
        encoding="utf-8",
    )

    errors = audit_manual_pair_caption_dir(targets_dir)

    assert "1.txt" not in errors
    assert "2.txt" in errors
    assert "caption contains banned term: european" in errors["2.txt"]
    assert "caption contains banned term: smiling" in errors["2.txt"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `./.venv/bin/pytest tests/test_marble_dataset.py -q`

Expected: FAIL with `ImportError` because `validate_manual_pair_caption` and `audit_manual_pair_caption_dir` do not exist yet.

- [ ] **Step 3: Write the minimal implementation**

```python
# src/klein4b/marble_dataset.py
PAIR_CAPTION_BANNED_TERMS = (
    "eastasian",
    "southasian",
    "middleeastern",
    "european",
    "asian",
    "smiling",
    "smile",
    "grinning",
    "laughing",
    "frowning",
    "neutral expression",
    "happy",
    "sad",
    "angry",
    "surprised",
)

PAIR_CAPTION_REQUIRED_PHRASES = (
    "transform into a <mrblbust>",
    "preserve recognizable identity",
    "dry chalky unpolished stone",
    "rough pitted low-albedo face surface",
    "localized lava only in the bottom cracks",
    "no glossy polished marble",
    "no frontal studio light",
    "no head-on beauty lighting",
    "no orange spill on face hair torso or background",
)


def validate_manual_pair_caption(caption: str) -> list[str]:
    normalized = " ".join(caption.strip().lower().split())
    errors: list[str] = []

    if "rewrite required" in normalized:
        errors.append("caption still contains the rewrite placeholder")

    for term in PAIR_CAPTION_BANNED_TERMS:
        if term in normalized:
            errors.append(f"caption contains banned term: {term}")

    for phrase in PAIR_CAPTION_REQUIRED_PHRASES:
        if phrase not in normalized:
            errors.append(f"caption is missing required phrase: {phrase}")

    if "blank sculpted eyes" not in normalized and "closed carved eyelids" not in normalized:
        errors.append(
            "caption is missing required phrase: blank sculpted eyes or closed carved eyelids"
        )

    if "carved from the same marble" not in normalized:
        errors.append("caption is missing required phrase: carved from the same marble")

    return errors


def audit_manual_pair_caption_dir(targets_dir: Path) -> dict[str, list[str]]:
    errors_by_file: dict[str, list[str]] = {}
    for caption_path in sorted(targets_dir.glob("*.txt")):
        errors = validate_manual_pair_caption(caption_path.read_text(encoding="utf-8"))
        if errors:
            errors_by_file[caption_path.name] = errors
    return errors_by_file
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `./.venv/bin/pytest tests/test_marble_dataset.py -q`

Expected: PASS for the new caption-audit tests and the existing marble-dataset tests.

- [ ] **Step 5: Commit**

```bash
git add src/klein4b/marble_dataset.py tests/test_marble_dataset.py
git commit -m "data: add paired caption audit helpers"
```

---

### Task 2: Build The Paired Dataset Scaffold And CLI

**Files:**
- Modify: `src/klein4b/marble_dataset.py`
- Create: `scripts/prepare_marble_v4_pairs_dataset.py`
- Modify: `tests/test_marble_dataset.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_marble_dataset.py
import importlib.util
import sys
from pathlib import Path

import pytest

from klein4b.marble_dataset import build_marble_v4_pairs_dataset


def test_build_marble_v4_pairs_dataset_copies_and_renames_assets(tmp_path: Path) -> None:
    source_pairs_dir = tmp_path / "v3" / "pairs"
    target_dir = tmp_path / "v4_pairs_weathered_face"
    source_pairs_dir.mkdir(parents=True)
    (source_pairs_dir / "1_input.jpg").write_bytes(b"input")
    (source_pairs_dir / "1_target.jpg").write_bytes(b"target")
    (source_pairs_dir / "1.txt").write_text("legacy pair caption", encoding="utf-8")

    report = build_marble_v4_pairs_dataset(source_pairs_dir, target_dir)

    assert report.pair_count == 1
    assert report.control_count == 1
    assert report.target_count == 1
    assert report.caption_count == 1
    assert (target_dir / "control" / "1.jpg").read_bytes() == b"input"
    assert (target_dir / "targets" / "1.jpg").read_bytes() == b"target"
    stub = (target_dir / "targets" / "1.txt").read_text(encoding="utf-8")
    assert "REWRITE REQUIRED" in stub
    assert "preserve recognizable identity and facial structure" in stub
    assert "data/marble-bust-data/v4_manual_weathered_face/busts/1.txt" in (
        target_dir / "CAPTION_REVIEW.md"
    ).read_text(encoding="utf-8")
    manifest = (target_dir / "manifest.json").read_text(encoding="utf-8")
    assert '"version": "v4_pairs_weathered_face"' in manifest
    assert '"caption_policy": "manual_review_required"' in manifest


def test_prepare_marble_v4_pairs_dataset_cli_prints_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "prepare_marble_v4_pairs_dataset.py"
    )
    spec = importlib.util.spec_from_file_location("prepare_pairs_cli", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    source_pairs_dir = tmp_path / "v3" / "pairs"
    target_dir = tmp_path / "v4_pairs_weathered_face"
    source_pairs_dir.mkdir(parents=True)
    (source_pairs_dir / "1_input.jpg").write_bytes(b"input")
    (source_pairs_dir / "1_target.jpg").write_bytes(b"target")
    (source_pairs_dir / "1.txt").write_text("legacy pair caption", encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_marble_v4_pairs_dataset.py",
            "--source",
            str(source_pairs_dir),
            "--target",
            str(target_dir),
        ],
    )

    module.main()

    assert (target_dir / "targets" / "1.txt").exists()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `./.venv/bin/pytest tests/test_marble_dataset.py -q`

Expected: FAIL because `build_marble_v4_pairs_dataset` and the new CLI script do not exist yet.

- [ ] **Step 3: Write the minimal implementation**

```python
# src/klein4b/marble_dataset.py
PAIR_CAPTION_REWRITE_STUB = """REWRITE REQUIRED.
Replace this file after visually reviewing the matching control and target images.

Required caption shape:
- start with `transform into a <mrblbust>`
- preserve recognizable identity and facial structure
- use only structural descriptors that are visible in the control and target
- do not use race labels
- do not use expression descriptors
- include the full v4 weathered-marble anti-gloss and anti-studio-light block
"""


@dataclass(frozen=True)
class MarblePairedDatasetReport:
    source_pairs_dir: Path
    target_dir: Path
    pair_ids: tuple[str, ...]
    control_count: int
    target_count: int
    caption_count: int

    @property
    def pair_count(self) -> int:
        return len(self.pair_ids)


def _find_pair_image(source_pairs_dir: Path, sample_id: str, suffix: str) -> Path:
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        candidate = source_pairs_dir / f"{sample_id}_{suffix}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Missing {suffix} image for sample {sample_id} in {source_pairs_dir}"
    )


def build_manual_pair_caption_stub(sample_id: str) -> str:
    return (
        PAIR_CAPTION_REWRITE_STUB
        + f"\nReference v4 bust caption: data/marble-bust-data/v4_manual_weathered_face/busts/{sample_id}.txt\n"
        + f"Control image: control/{sample_id}.jpg\n"
        + f"Target image: targets/{sample_id}.jpg\n"
    )


def write_pair_caption_review(report: MarblePairedDatasetReport) -> None:
    lines = [
        "# Paired Caption Review",
        "",
        "Every `targets/<id>.txt` file must be rewritten by hand before training.",
        "",
        "## Files To Review",
        "",
    ]
    for sample_id in report.pair_ids:
        lines.extend(
            [
                f"- `{sample_id}`",
                f"  - control: `control/{sample_id}.jpg`",
                f"  - target: `targets/{sample_id}.jpg`",
                (
                    "  - v4 bust caption reference: "
                    f"`data/marble-bust-data/v4_manual_weathered_face/busts/{sample_id}.txt`"
                ),
            ]
        )
    (report.target_dir / "CAPTION_REVIEW.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def build_marble_v4_pairs_dataset(
    source_pairs_dir: Path,
    target_dir: Path,
) -> MarblePairedDatasetReport:
    if not source_pairs_dir.exists():
        raise FileNotFoundError(f"Source pair dataset does not exist: {source_pairs_dir}")
    if target_dir.exists():
        shutil.rmtree(target_dir)

    control_dir = target_dir / "control"
    targets_dir = target_dir / "targets"
    control_dir.mkdir(parents=True)
    targets_dir.mkdir(parents=True)

    pair_ids: list[str] = []
    control_count = 0
    target_count = 0
    caption_count = 0

    for legacy_caption_path in sorted(source_pairs_dir.glob("*.txt")):
        sample_id = legacy_caption_path.stem
        control_source = _find_pair_image(source_pairs_dir, sample_id, "input")
        target_source = _find_pair_image(source_pairs_dir, sample_id, "target")

        control_dest = control_dir / f"{sample_id}{control_source.suffix.lower()}"
        target_dest = targets_dir / f"{sample_id}{target_source.suffix.lower()}"
        caption_dest = targets_dir / f"{sample_id}.txt"

        shutil.copy2(control_source, control_dest)
        shutil.copy2(target_source, target_dest)
        caption_dest.write_text(build_manual_pair_caption_stub(sample_id), encoding="utf-8")

        pair_ids.append(sample_id)
        control_count += 1
        target_count += 1
        caption_count += 1

    manifest = {
        "version": "v4_pairs_weathered_face",
        "caption_policy": "manual_review_required",
        "pair_count": len(pair_ids),
    }
    (target_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    report = MarblePairedDatasetReport(
        source_pairs_dir=source_pairs_dir,
        target_dir=target_dir,
        pair_ids=tuple(pair_ids),
        control_count=control_count,
        target_count=target_count,
        caption_count=caption_count,
    )
    write_pair_caption_review(report)
    return report
```

```python
# scripts/prepare_marble_v4_pairs_dataset.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from klein4b.marble_dataset import build_marble_v4_pairs_dataset  # noqa: E402
from klein4b.paths import data_dir  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create the v4 paired marble-bust dataset scaffold for manual caption review."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=data_dir() / "marble-bust-data" / "v3" / "pairs",
        help="Source v3 pairs directory containing *_input, *_target, and *.txt files.",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=data_dir() / "marble-bust-data" / "v4_pairs_weathered_face",
        help="Target dataset directory to rebuild.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = build_marble_v4_pairs_dataset(args.source, args.target)
    print(f"Wrote {report.target_dir}")
    print(f"Pairs copied: {report.pair_count}")
    print(f"Manual caption rewrite required: {report.caption_count}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `./.venv/bin/pytest tests/test_marble_dataset.py -q`

Expected: PASS for the new scaffold tests and the existing marble-dataset tests.

- [ ] **Step 5: Commit**

```bash
git add src/klein4b/marble_dataset.py scripts/prepare_marble_v4_pairs_dataset.py tests/test_marble_dataset.py
git commit -m "data: scaffold paired v4 marble dataset"
```

---

### Task 3: Rewrite The Numeric-ID Pair Captions By Hand

**Files:**
- Modify: `data/marble-bust-data/v4_pairs_weathered_face/targets/1.txt`
- Modify: `data/marble-bust-data/v4_pairs_weathered_face/targets/7.txt`
- Modify: `data/marble-bust-data/v4_pairs_weathered_face/targets/8.txt`
- Modify: `data/marble-bust-data/v4_pairs_weathered_face/targets/9.txt`
- Modify: `data/marble-bust-data/v4_pairs_weathered_face/targets/11.txt`
- Modify: `data/marble-bust-data/v4_pairs_weathered_face/targets/12.txt`
- Modify: `data/marble-bust-data/v4_pairs_weathered_face/targets/14.txt`
- Modify: `data/marble-bust-data/v4_pairs_weathered_face/targets/15.txt`
- Modify: `data/marble-bust-data/v4_pairs_weathered_face/targets/17.txt`
- Modify: `data/marble-bust-data/v4_pairs_weathered_face/targets/23.txt`
- Modify: `data/marble-bust-data/v4_pairs_weathered_face/targets/27.txt`
- Modify: `data/marble-bust-data/v4_pairs_weathered_face/targets/28.txt`
- Modify: `data/marble-bust-data/v4_pairs_weathered_face/targets/29.txt`
- Modify: `data/marble-bust-data/v4_pairs_weathered_face/targets/32.txt`
- Modify: `data/marble-bust-data/v4_pairs_weathered_face/targets/35.txt`
- Modify: `data/marble-bust-data/v4_pairs_weathered_face/targets/36.txt`
- Modify: `data/marble-bust-data/v4_pairs_weathered_face/targets/37.txt`
- Modify: `data/marble-bust-data/v4_pairs_weathered_face/targets/39.txt`

- [ ] **Step 1: Generate the local scaffold**

Run: `./.venv/bin/python scripts/prepare_marble_v4_pairs_dataset.py`

Expected:

```text
Wrote /home/liush/projects/charlie_tale_ft/klein4b/data/marble-bust-data/v4_pairs_weathered_face
Pairs copied: 29
Manual caption rewrite required: 29
```

- [ ] **Step 2: Rewrite the numeric-id caption files after visual review**

For each file listed in this task:

1. Open the control image at `data/marble-bust-data/v4_pairs_weathered_face/control/<id>.jpg`.
2. Open the paired target image at `data/marble-bust-data/v4_pairs_weathered_face/targets/<id>.jpg`.
3. Open the matching target-only v4 caption at `data/marble-bust-data/v4_manual_weathered_face/busts/<id>.txt`.
4. Replace the stub text in `data/marble-bust-data/v4_pairs_weathered_face/targets/<id>.txt` with one line that keeps only structural identity cues from the pair and keeps the full v4 material tail.

Build every rewritten caption in this exact sequence:

```text
1. Start with `transform into a <mrblbust> centered chest-up Greek marble statue bust of`.
2. Immediately add only the age band and gender presentation that are visually clear in the pair.
3. Add `, preserve recognizable identity and facial structure, keep`.
4. Add only the visible structural clauses from the images: face shape, jaw, nose, lips, head angle, and hair or headwear silhouette.
5. Append this fixed material tail verbatim:
   `dark ash-grey weathered marble, dry chalky unpolished stone, rough pitted low-albedo face surface, brown-grey mineral patina across the face neck and torso where visible, blank sculpted eyes or closed carved eyelids, hair headwear eyebrows and eyelashes carved from the same marble, jagged broken lower bust with localized lava only in the bottom cracks, no glossy polished marble, no wet shine, no specular hotspots, no frontal studio light, no head-on beauty lighting, no orange spill on face hair torso or background`
```

Keep these constraints exact while editing:

```text
- no race labels
- no expression descriptors
- no mood or emotion adjectives
- preserve recognizable identity, not one-to-one likeness
- keep only descriptors visible in the source and target images
```

Two fully written examples to copy the tone and level of detail:

```text
transform into a <mrblbust> centered chest-up Greek marble statue bust of an adult woman, preserve recognizable identity and facial structure, keep the oval face, straight narrow nose, full lower lip, slight three-quarter head turn, and thick rope-like hair mass falling behind the shoulder, dark ash-grey weathered marble, dry chalky unpolished stone, rough pitted low-albedo face surface, brown-grey mineral patina across the face neck and torso where visible, blank sculpted eyes, hair headwear eyebrows and eyelashes carved from the same marble, jagged broken lower bust with localized lava only in the bottom cracks, no glossy polished marble, no wet shine, no specular hotspots, no frontal studio light, no head-on beauty lighting, no orange spill on face hair torso or background
```

```text
transform into a <mrblbust> centered chest-up Greek marble statue bust of an adult man, preserve recognizable identity and facial structure, keep the broad jaw, straight nose, firm mouth, forward-facing head angle, and short carved wavy hair combed toward the brow, dark ash-grey weathered marble, dry chalky unpolished stone, rough pitted low-albedo face surface, brown-grey mineral patina across the face neck and torso where visible, blank sculpted eyes, hair headwear eyebrows and eyelashes carved from the same marble, jagged broken lower bust with localized lava only in the bottom cracks, no glossy polished marble, no wet shine, no specular hotspots, no frontal studio light, no head-on beauty lighting, no orange spill on face hair torso or background
```

- [ ] **Step 3: Audit the numeric-id captions**

Run:

```bash
./.venv/bin/python - <<'PY'
from pathlib import Path

from klein4b.marble_dataset import validate_manual_pair_caption

ids = ["1", "7", "8", "9", "11", "12", "14", "15", "17", "23", "27", "28", "29", "32", "35", "36", "37", "39"]
targets_dir = Path("data/marble-bust-data/v4_pairs_weathered_face/targets")
errors = {}
for sample_id in ids:
    caption_path = targets_dir / f"{sample_id}.txt"
    caption_errors = validate_manual_pair_caption(caption_path.read_text(encoding="utf-8"))
    if caption_errors:
        errors[sample_id] = caption_errors

if errors:
    for sample_id, caption_errors in sorted(errors.items()):
        print(sample_id)
        for error in caption_errors:
            print(f"  - {error}")
    raise SystemExit(1)

print(f"OK: {len(ids)} numeric-id captions passed manual audit")
PY
```

Expected: `OK: 18 numeric-id captions passed manual audit`

- [ ] **Step 4: Do not commit the dataset files**

`data/` is ignored in this repo. Leave these edits as local working data and move on.

---

### Task 4: Rewrite The Named Pair Captions By Hand

**Files:**
- Modify: `data/marble-bust-data/v4_pairs_weathered_face/targets/eastasian_child_neutral_1.txt`
- Modify: `data/marble-bust-data/v4_pairs_weathered_face/targets/eastasian_child_neutral_3.txt`
- Modify: `data/marble-bust-data/v4_pairs_weathered_face/targets/eastasian_elderly_female_3.txt`
- Modify: `data/marble-bust-data/v4_pairs_weathered_face/targets/eastasian_youngadult_female_1.txt`
- Modify: `data/marble-bust-data/v4_pairs_weathered_face/targets/european_middleaged_neutral_3.txt`
- Modify: `data/marble-bust-data/v4_pairs_weathered_face/targets/european_youngadult_male_1.txt`
- Modify: `data/marble-bust-data/v4_pairs_weathered_face/targets/middleeastern_child_neutral_1.txt`
- Modify: `data/marble-bust-data/v4_pairs_weathered_face/targets/southasian_child_female_2.txt`
- Modify: `data/marble-bust-data/v4_pairs_weathered_face/targets/southasian_elderly_male_2.txt`
- Modify: `data/marble-bust-data/v4_pairs_weathered_face/targets/southasian_elderly_neutral_2.txt`
- Modify: `data/marble-bust-data/v4_pairs_weathered_face/targets/southasian_middleaged_male_3.txt`

- [ ] **Step 1: Rewrite the named caption files after visual review**

Use the same review flow and the same fixed material tail from Task 3. The file names still contain older demographic labels; the new caption text must not repeat those labels. Keep only what is visible:

```text
- child / adult / elderly when visually clear
- woman / man when visually clear
- face shape
- jaw, cheekbone, nose, and lip structure
- hair or headwear silhouette
- head angle
```

Start each file from the matching v4 bust caption at `data/marble-bust-data/v4_manual_weathered_face/busts/<id>.txt`, but convert it into edit framing:

```text
replace the leading `a <mrblbust>` with `transform into a <mrblbust>`
insert `preserve recognizable identity and facial structure` immediately after the subject phrase
remove any phrase that describes race or expression
keep the full weathered-marble anti-gloss and anti-studio-light block
```

Three fully written examples for the named files:

```text
transform into a <mrblbust> centered chest-up Greek marble statue bust of a child, preserve recognizable identity and facial structure, keep the short forward-swept fringe, rounded cheeks, small nose, soft jawline, and slight leftward head turn, dark ash-grey weathered marble, dry chalky unpolished stone, rough pitted low-albedo face surface, brown-grey mineral patina across the face neck and torso where visible, blank sculpted eyes, hair headwear eyebrows and eyelashes carved from the same marble, jagged broken lower bust with localized lava only in the bottom cracks, no glossy polished marble, no wet shine, no specular hotspots, no frontal studio light, no head-on beauty lighting, no orange spill on face hair torso or background
```

```text
transform into a <mrblbust> centered chest-up Greek marble statue bust of an elderly woman, preserve recognizable identity and facial structure, keep the narrow face, longer nose, thin lips, deeper eye sockets, and short carved hair tucked close to the head, dark ash-grey weathered marble, dry chalky unpolished stone, rough pitted low-albedo face surface, brown-grey mineral patina across the face neck and torso where visible, blank sculpted eyes or closed carved eyelids, hair headwear eyebrows and eyelashes carved from the same marble, jagged broken lower bust with localized lava only in the bottom cracks, no glossy polished marble, no wet shine, no specular hotspots, no frontal studio light, no head-on beauty lighting, no orange spill on face hair torso or background
```

```text
transform into a <mrblbust> centered chest-up Greek marble statue bust of an adult man, preserve recognizable identity and facial structure, keep the broader jaw, straight nose, medium-full lips, forward-facing head angle, and short textured hair silhouette, dark ash-grey weathered marble, dry chalky unpolished stone, rough pitted low-albedo face surface, brown-grey mineral patina across the face neck and torso where visible, blank sculpted eyes, hair headwear eyebrows and eyelashes carved from the same marble, jagged broken lower bust with localized lava only in the bottom cracks, no glossy polished marble, no wet shine, no specular hotspots, no frontal studio light, no head-on beauty lighting, no orange spill on face hair torso or background
```

- [ ] **Step 2: Audit the named captions**

Run:

```bash
./.venv/bin/python - <<'PY'
from pathlib import Path

from klein4b.marble_dataset import validate_manual_pair_caption

ids = [
    "eastasian_child_neutral_1",
    "eastasian_child_neutral_3",
    "eastasian_elderly_female_3",
    "eastasian_youngadult_female_1",
    "european_middleaged_neutral_3",
    "european_youngadult_male_1",
    "middleeastern_child_neutral_1",
    "southasian_child_female_2",
    "southasian_elderly_male_2",
    "southasian_elderly_neutral_2",
    "southasian_middleaged_male_3",
]
targets_dir = Path("data/marble-bust-data/v4_pairs_weathered_face/targets")
errors = {}
for sample_id in ids:
    caption_path = targets_dir / f"{sample_id}.txt"
    caption_errors = validate_manual_pair_caption(caption_path.read_text(encoding="utf-8"))
    if caption_errors:
        errors[sample_id] = caption_errors

if errors:
    for sample_id, caption_errors in sorted(errors.items()):
        print(sample_id)
        for error in caption_errors:
            print(f"  - {error}")
    raise SystemExit(1)

print(f"OK: {len(ids)} named captions passed manual audit")
PY
```

Expected: `OK: 11 named captions passed manual audit`

- [ ] **Step 3: Audit the whole paired dataset**

Run:

```bash
./.venv/bin/python - <<'PY'
from pathlib import Path

from klein4b.marble_dataset import audit_manual_pair_caption_dir

targets_dir = Path("data/marble-bust-data/v4_pairs_weathered_face/targets")
errors = audit_manual_pair_caption_dir(targets_dir)
if errors:
    for filename, file_errors in sorted(errors.items()):
        print(filename)
        for error in file_errors:
            print(f"  - {error}")
    raise SystemExit(1)

print(f"OK: {len(list(targets_dir.glob('*.txt')))} paired captions passed full audit")
PY
```

Expected: `OK: 29 paired captions passed full audit`

- [ ] **Step 4: Do not commit the dataset files**

`data/` is ignored in this repo. Leave these edits as local working data and move on.

---

### Task 5: Add The Paired Klein4B Training Template

**Files:**
- Create: `configs/train_flux2_klein_marble_bust_v4_pairs_weathered_face.template.yaml`
- Modify: `tests/test_training.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_training.py
from pathlib import Path

from klein4b.training import render_training_config


def test_render_training_config_sets_marble_v4_pairs_defaults(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "marble-bust-data" / "v4_pairs_weathered_face"
    (dataset_dir / "targets").mkdir(parents=True)
    (dataset_dir / "control").mkdir()
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    template_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "train_flux2_klein_marble_bust_v4_pairs_weathered_face.template.yaml"
    )

    config_text = render_training_config(
        template_path=template_path,
        dataset_dir=dataset_dir,
        output_dir=output_dir,
    )

    assert 'arch: "flux2_klein_4b"' in config_text
    assert 'name: marble_bust_v4_pairs_weathered_face_style' in config_text
    assert str(dataset_dir / "targets") in config_text
    assert str(dataset_dir / "control") in config_text
    assert "control_path_1" in config_text
    assert "samples:" in config_text
    assert "ctrl_img_1" in config_text
    assert "preserve recognizable identity and facial structure" in config_text
    assert "no frontal studio light" in config_text
    assert "no head-on beauty lighting" in config_text
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `./.venv/bin/pytest tests/test_training.py -q`

Expected: FAIL because the paired training template does not exist yet.

- [ ] **Step 3: Write the minimal implementation**

```yaml
# configs/train_flux2_klein_marble_bust_v4_pairs_weathered_face.template.yaml
job: extension
config:
  name: marble_bust_v4_pairs_weathered_face_style
  process:
    - type: diffusion_trainer
      training_folder: "{{ training_folder }}"
      device: "cuda"
      trigger_word: "<mrblbust>"
      network:
        type: "lora"
        linear: 32
        linear_alpha: 32
        conv: 16
        conv_alpha: 16
      save:
        dtype: "bf16"
        save_every: 200
        max_step_saves_to_keep: 3
      datasets:
        - folder_path: "{{ dataset_dir }}/targets"
          control_path_1: "{{ dataset_dir }}/control"
          caption_ext: "txt"
          cache_latents_to_disk: true
          resolution:
            - 768
      train:
        batch_size: 1
        steps: 2000
        gradient_accumulation_steps: 1
        train_unet: true
        train_text_encoder: false
        gradient_checkpointing: true
        lr: 0.00006
        noise_scheduler: "flowmatch"
        optimizer: "adamw8bit"
        timestep_type: "weighted"
        content_or_style: "balanced"
        dtype: "bf16"
        unload_text_encoder: false
      model:
        arch: "flux2_klein_4b"
        name_or_path: "black-forest-labs/FLUX.2-klein-base-4B"
        quantize: true
        quantize_te: true
        qtype: "qfloat8"
        qtype_te: "qfloat8"
        low_vram: true
        model_kwargs:
          match_target_res: false
      sample:
        sampler: "flowmatch"
        sample_every: 200
        width: 768
        height: 1024
        samples:
          - prompt: "Change image 1 into a <mrblbust> centered chest-up Greek marble statue bust of an adult woman, preserve recognizable identity and facial structure, keep the oval face, straight narrow nose, full lower lip, slight three-quarter head turn, and thick rope-like hair mass behind the shoulder, dark ash-grey weathered marble, dry chalky unpolished stone, rough pitted low-albedo face surface, brown-grey mineral patina across the face neck and torso where visible, blank sculpted eyes, hair headwear eyebrows and eyelashes carved from the same marble, jagged broken lower bust with localized lava only in the bottom cracks, no glossy polished marble, no wet shine, no specular hotspots, no frontal studio light, no head-on beauty lighting, no orange spill on face hair torso or background"
            ctrl_img_1: "{{ dataset_dir }}/control/1.jpg"
          - prompt: "Change image 1 into a <mrblbust> centered chest-up Greek marble statue bust of an adult woman, preserve recognizable identity and facial structure, keep the narrower chin, small straight nose, soft lips, slight downward head angle, and carved hair volume swept away from the forehead, dark ash-grey weathered marble, dry chalky unpolished stone, rough pitted low-albedo face surface, brown-grey mineral patina across the face neck and torso where visible, blank sculpted eyes, hair headwear eyebrows and eyelashes carved from the same marble, jagged broken lower bust with localized lava only in the bottom cracks, no glossy polished marble, no wet shine, no specular hotspots, no frontal studio light, no head-on beauty lighting, no orange spill on face hair torso or background"
            ctrl_img_1: "{{ dataset_dir }}/control/7.jpg"
          - prompt: "Change image 1 into a <mrblbust> centered chest-up Greek marble statue bust of a child, preserve recognizable identity and facial structure, keep the rounded cheeks, smaller nose, short forward fringe, and slight leftward head turn, dark ash-grey weathered marble, dry chalky unpolished stone, rough pitted low-albedo face surface, brown-grey mineral patina across the face neck and torso where visible, blank sculpted eyes, hair headwear eyebrows and eyelashes carved from the same marble, jagged broken lower bust with localized lava only in the bottom cracks, no glossy polished marble, no wet shine, no specular hotspots, no frontal studio light, no head-on beauty lighting, no orange spill on face hair torso or background"
            ctrl_img_1: "{{ dataset_dir }}/control/eastasian_child_neutral_1.jpg"
          - prompt: "Change image 1 into a <mrblbust> centered chest-up Greek marble statue bust of an adult person, preserve recognizable identity and facial structure, keep the longer face, straight nose, firmer mouth, and short carved hair or headwear silhouette, dark ash-grey weathered marble, dry chalky unpolished stone, rough pitted low-albedo face surface, brown-grey mineral patina across the face neck and torso where visible, blank sculpted eyes, hair headwear eyebrows and eyelashes carved from the same marble, jagged broken lower bust with localized lava only in the bottom cracks, no glossy polished marble, no wet shine, no specular hotspots, no frontal studio light, no head-on beauty lighting, no orange spill on face hair torso or background"
            ctrl_img_1: "{{ dataset_dir }}/control/european_middleaged_neutral_3.jpg"
          - prompt: "Change image 1 into a <mrblbust> centered chest-up Greek marble statue bust of an adult man, preserve recognizable identity and facial structure, keep the broad jaw, straight nose, medium-full lips, forward head angle, and short textured hair silhouette, dark ash-grey weathered marble, dry chalky unpolished stone, rough pitted low-albedo face surface, brown-grey mineral patina across the face neck and torso where visible, blank sculpted eyes, hair headwear eyebrows and eyelashes carved from the same marble, jagged broken lower bust with localized lava only in the bottom cracks, no glossy polished marble, no wet shine, no specular hotspots, no frontal studio light, no head-on beauty lighting, no orange spill on face hair torso or background"
            ctrl_img_1: "{{ dataset_dir }}/control/southasian_middleaged_male_3.jpg"
        neg: "head-only crop, close-up face, cropped shoulders, cropped torso, loose rocks, separate rock pile, rock pedestal, debris below bust, flat plinth, individual hair strands, natural hair color, glossy hair, polished marble, wet shine, specular hotspots, frontal studio light, head-on key light, beauty lighting, perfect portrait lighting, spotlight on face, smooth porcelain face, overexposed white face, clean bright cheeks, shiny forehead, shiny nose, shiny lips, red or orange light on face, red or orange light on hair, glowing skin, glowing hair, soft hair strands, realistic hair texture, modern eyewear, modern accessories, duplicate figures, side-by-side views, collage"
        seed: 7
        guidance_scale: 4
        sample_steps: 20
meta:
  name: "marble_bust_v4_pairs_weathered_face_style"
  version: "4.0"
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `./.venv/bin/pytest tests/test_training.py -q`

Expected: PASS for the new paired-config test and the existing training tests.

- [ ] **Step 5: Commit**

```bash
git add configs/train_flux2_klein_marble_bust_v4_pairs_weathered_face.template.yaml tests/test_training.py
git commit -m "train: add paired klein4b v4 config"
```

---

### Task 6: Add A Dedicated Paired Evaluation Module And CLI

**Files:**
- Create: `src/klein4b/paired_eval.py`
- Create: `scripts/eval_marble_v4_pairs.py`
- Create: `tests/test_paired_eval.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_paired_eval.py
from pathlib import Path

from klein4b.paired_eval import DEFAULT_PAIR_EVAL_IDS, build_pair_eval_paths


def test_build_pair_eval_paths_returns_expected_layout(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "v4_pairs_weathered_face"
    output_dir = tmp_path / "eval"

    paths = build_pair_eval_paths(dataset_dir, output_dir, "1")

    assert paths.sample_id == "1"
    assert paths.control_path == dataset_dir / "control" / "1.jpg"
    assert paths.target_path == dataset_dir / "targets" / "1.jpg"
    assert paths.caption_path == dataset_dir / "targets" / "1.txt"
    assert paths.baseline_output_path == output_dir / "baseline" / "1.png"
    assert paths.paired_output_path == output_dir / "paired" / "1.png"
    assert paths.contact_output_path == output_dir / "contacts" / "1_contact.jpg"


def test_default_pair_eval_ids_match_the_spec_focus_set() -> None:
    assert DEFAULT_PAIR_EVAL_IDS == (
        "1",
        "7",
        "eastasian_child_neutral_1",
        "european_middleaged_neutral_3",
        "southasian_middleaged_male_3",
    )
```

```python
# tests/test_paired_eval.py
import importlib.util
import sys
from pathlib import Path

import pytest


def test_eval_marble_v4_pairs_cli_wires_arguments(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_calls: dict[str, object] = {}
    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "eval_marble_v4_pairs.py"
    )
    spec = importlib.util.spec_from_file_location("eval_pairs_cli", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def fake_run_pair_eval_set(**kwargs: object) -> None:
        run_calls.update(kwargs)

    monkeypatch.setattr(module, "run_pair_eval_set", fake_run_pair_eval_set)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_marble_v4_pairs.py",
            "--dataset-dir",
            str(tmp_path / "dataset"),
            "--baseline-lora",
            str(tmp_path / "baseline.safetensors"),
            "--paired-lora",
            str(tmp_path / "paired.safetensors"),
            "--output-dir",
            str(tmp_path / "eval"),
            "--ids",
            "1",
            "7",
        ],
    )

    module.main()

    assert run_calls["dataset_dir"] == tmp_path / "dataset"
    assert run_calls["baseline_lora_path"] == tmp_path / "baseline.safetensors"
    assert run_calls["paired_lora_path"] == tmp_path / "paired.safetensors"
    assert run_calls["output_dir"] == tmp_path / "eval"
    assert run_calls["sample_ids"] == ("1", "7")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `./.venv/bin/pytest tests/test_paired_eval.py -q`

Expected: FAIL because `klein4b.paired_eval` and the new CLI script do not exist yet.

- [ ] **Step 3: Write the minimal implementation**

```python
# src/klein4b/paired_eval.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from klein4b.image_grid import make_four_up_grid
from klein4b.inference import build_marble_pair_prompt, run_local_inference

DEFAULT_PAIR_EVAL_IDS = (
    "1",
    "7",
    "eastasian_child_neutral_1",
    "european_middleaged_neutral_3",
    "southasian_middleaged_male_3",
)


@dataclass(frozen=True)
class PairEvalPaths:
    sample_id: str
    control_path: Path
    target_path: Path
    caption_path: Path
    baseline_output_path: Path
    paired_output_path: Path
    contact_output_path: Path


def build_pair_eval_paths(dataset_dir: Path, output_dir: Path, sample_id: str) -> PairEvalPaths:
    return PairEvalPaths(
        sample_id=sample_id,
        control_path=dataset_dir / "control" / f"{sample_id}.jpg",
        target_path=dataset_dir / "targets" / f"{sample_id}.jpg",
        caption_path=dataset_dir / "targets" / f"{sample_id}.txt",
        baseline_output_path=output_dir / "baseline" / f"{sample_id}.png",
        paired_output_path=output_dir / "paired" / f"{sample_id}.png",
        contact_output_path=output_dir / "contacts" / f"{sample_id}_contact.jpg",
    )


def save_pair_contact_sheet(paths: PairEvalPaths) -> None:
    control = Image.open(paths.control_path).convert("RGB")
    target = Image.open(paths.target_path).convert("RGB")
    baseline = Image.open(paths.baseline_output_path).convert("RGB")
    paired = Image.open(paths.paired_output_path).convert("RGB")

    target_size = target.size
    images = [
        control.resize(target_size),
        target,
        baseline.resize(target_size),
        paired.resize(target_size),
    ]
    contact = make_four_up_grid(images, ["control", "target", "target-only", "paired"])
    paths.contact_output_path.parent.mkdir(parents=True, exist_ok=True)
    contact.save(paths.contact_output_path)


def write_summary_sheet(contact_paths: list[Path], output_path: Path) -> None:
    rows = [Image.open(path).convert("RGB") for path in contact_paths]
    if not rows:
        return

    width = max(row.width for row in rows)
    height = sum(row.height for row in rows)
    canvas = Image.new("RGB", (width, height), color="black")

    y_offset = 0
    for row in rows:
        canvas.paste(row, (0, y_offset))
        y_offset += row.height

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def run_pair_eval_set(
    *,
    dataset_dir: Path,
    baseline_lora_path: Path,
    paired_lora_path: Path,
    output_dir: Path,
    sample_ids: tuple[str, ...] = DEFAULT_PAIR_EVAL_IDS,
    baseline_lora_scale: float = 1.0,
    paired_lora_scale: float = 1.0,
) -> None:
    contact_paths: list[Path] = []

    for sample_id in sample_ids:
        paths = build_pair_eval_paths(dataset_dir, output_dir, sample_id)
        pair_caption = paths.caption_path.read_text(encoding="utf-8")
        prompt = build_marble_pair_prompt(pair_caption)

        run_local_inference(
            reference_path=paths.control_path,
            prompt=prompt,
            output_path=paths.baseline_output_path,
            lora_path=baseline_lora_path,
            lora_scale=baseline_lora_scale,
        )
        run_local_inference(
            reference_path=paths.control_path,
            prompt=prompt,
            output_path=paths.paired_output_path,
            lora_path=paired_lora_path,
            lora_scale=paired_lora_scale,
        )
        save_pair_contact_sheet(paths)
        contact_paths.append(paths.contact_output_path)

    write_summary_sheet(
        contact_paths,
        output_dir / "test_img_klein_v4_pairs_weathered_face_contact.jpg",
    )
```

```python
# scripts/eval_marble_v4_pairs.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from klein4b.paired_eval import DEFAULT_PAIR_EVAL_IDS, run_pair_eval_set  # noqa: E402
from klein4b.paths import data_dir, outputs_dir  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run baseline-vs-paired evaluation for the paired Klein4B marble dataset."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=data_dir() / "marble-bust-data" / "v4_pairs_weathered_face",
        help="Paired dataset root with control/ and targets/ subdirectories.",
    )
    parser.add_argument("--baseline-lora", type=Path, required=True)
    parser.add_argument("--paired-lora", type=Path, required=True)
    parser.add_argument("--baseline-scale", type=float, default=1.0)
    parser.add_argument("--paired-scale", type=float, default=1.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=outputs_dir() / "eval" / "test_img_klein_v4_pairs_weathered_face",
        help="Evaluation output directory.",
    )
    parser.add_argument(
        "--ids",
        nargs="*",
        default=list(DEFAULT_PAIR_EVAL_IDS),
        help="Specific sample ids to evaluate.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_pair_eval_set(
        dataset_dir=args.dataset_dir,
        baseline_lora_path=args.baseline_lora,
        paired_lora_path=args.paired_lora,
        output_dir=args.output_dir,
        sample_ids=tuple(args.ids),
        baseline_lora_scale=args.baseline_scale,
        paired_lora_scale=args.paired_scale,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `./.venv/bin/pytest tests/test_paired_eval.py -q`

Expected: PASS for the new paired-eval tests.

- [ ] **Step 5: Commit**

```bash
git add src/klein4b/paired_eval.py scripts/eval_marble_v4_pairs.py tests/test_paired_eval.py
git commit -m "eval: add paired klein4b comparison flow"
```

---

### Task 7: Verify The Full Paired Pipeline Wiring

**Files:**
- Modify: none
- Verify: `src/klein4b/marble_dataset.py`
- Verify: `configs/train_flux2_klein_marble_bust_v4_pairs_weathered_face.template.yaml`
- Verify: `scripts/prepare_marble_v4_pairs_dataset.py`
- Verify: `scripts/eval_marble_v4_pairs.py`
- Verify: `tests/test_marble_dataset.py`
- Verify: `tests/test_training.py`
- Verify: `tests/test_paired_eval.py`

- [ ] **Step 1: Run the focused verification suite**

Run:

```bash
./.venv/bin/pytest tests/test_marble_dataset.py tests/test_training.py tests/test_paired_eval.py -q
```

Expected: all focused paired-dataset, paired-config, and paired-eval tests pass.

- [ ] **Step 2: Run the full repo verification**

Run:

```bash
./.venv/bin/pytest -q
./.venv/bin/ruff check .
```

Expected:

```text
all tests pass
ruff check reports no violations
```

- [ ] **Step 3: Verify both new CLIs expose the expected interface**

Run:

```bash
./.venv/bin/python scripts/prepare_marble_v4_pairs_dataset.py --help
./.venv/bin/python scripts/eval_marble_v4_pairs.py --help
```

Expected:

```text
both commands exit with code 0
the dataset script shows --source and --target
the eval script shows --dataset-dir, --baseline-lora, --paired-lora, --baseline-scale, --paired-scale, --output-dir, and --ids
```

- [ ] **Step 4: Record the first real run commands in the working notes**

Use these exact commands after code review is complete and the paired captions have passed the manual audit:

```bash
./.venv/bin/python scripts/prepare_marble_v4_pairs_dataset.py
./.venv/bin/python scripts/train_lora.py --config configs/train_flux2_klein_marble_bust_v4_pairs_weathered_face.template.yaml --dataset-dir data/marble-bust-data/v4_pairs_weathered_face --output-root outputs/runs/marble_v4_pairs_weathered_face
./.venv/bin/python scripts/eval_marble_v4_pairs.py --dataset-dir data/marble-bust-data/v4_pairs_weathered_face --baseline-lora outputs/runs/marble_v3_matte_lava/candidates/marble_bust_v3_matte_lava_style_1600_candidate.safetensors --paired-lora "$(find outputs/runs/marble_v4_pairs_weathered_face -type f -name 'marble_bust_v4_pairs_weathered_face_style*_candidate.safetensors' | sort | tail -n 1)" --output-dir outputs/eval/test_img_klein_v4_pairs_weathered_face --baseline-scale 1.0 --paired-scale 1.0
```

- [ ] **Step 5: No commit in this task**

This task is verification and runbook capture only.

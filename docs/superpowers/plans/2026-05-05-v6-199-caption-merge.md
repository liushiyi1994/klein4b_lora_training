# V6 199 Caption Merge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible v6 dataset preparation path that copies the 199 remote selfie/control pairs locally and rewrites captions in v5 marble style with deterministic prose-only archetype detail.

**Architecture:** Add a focused `klein4b.marble_v6_dataset` module for parsing source captions, rendering archetype-enriched captions, validating source rows, copying files, and writing reports. Add a thin CLI script that calls the module. Keep all training config and inference code unchanged.

**Tech Stack:** Python 3.12, dataclasses, `json`, `shutil`, `pathlib`, `pytest`, existing `src/` package layout.

---

## File Structure

| Path | Responsibility |
|---|---|
| `src/klein4b/marble_v6_dataset.py` | New reusable module for v6 caption parsing, archetype phrase rendering, dataset validation/copying, manifest writing, and report writing. |
| `scripts/prepare_marble_v6_199_dataset.py` | New CLI entry point for preparing `data/marble-bust-data/v6_199` from a checkout of `charliestale-ml/data-synthesis`. |
| `tests/test_marble_v6_dataset.py` | New unit tests for parser, caption renderer, builder validation, and output layout. |
| `tests/test_prepare_marble_v6_199_dataset_cli.py` | New CLI smoke tests that monkeypatch the builder and verify argument/default behavior. |

The implementation must not modify:

- `src/klein4b/marble_dataset.py`
- `src/klein4b/sagemaker_inference.py`
- existing v4/v5 configs
- existing v5 tests or the currently untracked v5 config

---

## Task 1: Caption Parser And Archetype Renderer

**Files:**
- Create: `src/klein4b/marble_v6_dataset.py`
- Test: `tests/test_marble_v6_dataset.py`

- [ ] **Step 1: Write failing parser and renderer tests**

Create `tests/test_marble_v6_dataset.py` with:

```python
from __future__ import annotations

from pathlib import Path

import pytest

from klein4b.marble_v6_dataset import (
    ArchetypePair,
    CaptionParseError,
    parse_v6_source_caption,
    render_archetype_details,
    render_v6_caption,
)


SOURCE_CAPTION_001 = (
    "Convert this portrait into a CSTALE marble bust of an African child female — "
    "short coiled hair gathered into two puffs, round face, full cheeks, almond eyes, "
    "broad nose with rounded tip, full lips. The figure wears a winged helmet with a "
    "tall crest. An ornate cuirass shapes the torso, with a cloak draped diagonally "
    "across one shoulder. The body is in strict left profile; the head is turned back "
    "over the left shoulder, chin lifted, with a heroic upward gaze. A serene "
    "archaic-smile expression."
)


SOURCE_CAPTION_BARE = (
    "Convert this portrait into a CSTALE marble bust of a European young-adult "
    "gender-neutral person — shoulder-length wavy hair loose and flowing, oval face, "
    "flat cheeks, almond-shaped eyes, straight nose with refined tip, medium full lips. "
    "A cuirass shapes the chest, with a cloak draped diagonally across one shoulder. "
    "The body is in three-quarter view turned to the right; the head is turned back "
    "over the right shoulder, with a direct gaze toward the viewer. A serene "
    "archaic-smile expression."
)


def test_parse_v6_source_caption_preserves_demographic_race_label_and_identity() -> None:
    parts = parse_v6_source_caption(SOURCE_CAPTION_001)

    assert parts.identity == (
        "an African child female with short coiled hair gathered into two puffs, "
        "round face, full cheeks, almond eyes, broad nose with rounded tip, full lips"
    )
    assert parts.pose == (
        "in strict left profile; the head is turned back over the left shoulder, "
        "chin lifted, with a heroic upward gaze"
    )
    assert parts.expression == "a serene archaic-smile expression"


def test_parse_v6_source_caption_handles_bare_rows_without_helmet_sentence() -> None:
    parts = parse_v6_source_caption(SOURCE_CAPTION_BARE)

    assert "European young-adult gender-neutral person" in parts.identity
    assert "shoulder-length wavy hair loose and flowing" in parts.identity
    assert parts.pose.startswith("in three-quarter view turned to the right")
    assert parts.expression == "a serene archaic-smile expression"


def test_render_archetype_details_uses_prose_only_and_no_raw_tokens() -> None:
    details = render_archetype_details(
        ArchetypePair(helmet="ctWinged", garment="ctOrnateCuirass")
    )

    assert "ctWinged" not in details
    assert "ctOrnateCuirass" not in details
    assert "winged helmet with a tall crest" in details
    assert "raised chest medallion" in details
    assert "carved relief trim" in details


def test_render_archetype_details_for_bare_head_omits_helmet_phrase() -> None:
    details = render_archetype_details(
        ArchetypePair(helmet="ctBare", garment="ctWarriorCuirass")
    )

    assert "helmet" not in details
    assert "bare" not in details.lower()
    assert details.startswith("warrior cuirass")


def test_render_v6_caption_merges_v5_style_with_source_identity_and_archetypes() -> None:
    caption = render_v6_caption(
        source_caption=SOURCE_CAPTION_001,
        archetypes=ArchetypePair(helmet="ctWinged", garment="ctOrnateCuirass"),
    )

    assert caption.startswith("transform into a <mrblbust> from the reference portrait")
    assert "preserving an African child female" in caption
    assert "short coiled hair gathered into two puffs" in caption
    assert "winged helmet with a tall crest" in caption
    assert "ornate cuirass" in caption
    assert "blank sculpted eyes" in caption
    assert "matte weathered grey-brown stone" in caption
    assert "localized lava only in the lower cracks and fractures" in caption
    assert "no glossy polished marble" in caption
    assert "no orange spill on face hair torso or background" in caption
    assert "CSTALE" not in caption
    assert "ctWinged" not in caption


def test_render_v6_caption_raises_clear_error_for_unknown_archetype() -> None:
    with pytest.raises(KeyError, match="unknown helmet archetype"):
        render_archetype_details(ArchetypePair(helmet="ctMissing", garment="ctWarriorCuirass"))


def test_parse_v6_source_caption_raises_clear_error_for_unexpected_shape() -> None:
    with pytest.raises(CaptionParseError, match="expected caption prefix"):
        parse_v6_source_caption("not a valid source caption")
```

- [ ] **Step 2: Run parser test to verify it fails**

Run:

```bash
pytest tests/test_marble_v6_dataset.py::test_parse_v6_source_caption_preserves_demographic_race_label_and_identity -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'klein4b.marble_v6_dataset'`.

- [ ] **Step 3: Implement minimal caption parser and renderer**

Create `src/klein4b/marble_v6_dataset.py`:

```python
from __future__ import annotations

import re
from dataclasses import dataclass


class CaptionParseError(ValueError):
    """Raised when a v6 source caption does not match the expected structure."""


@dataclass(frozen=True)
class ArchetypePair:
    helmet: str
    garment: str


@dataclass(frozen=True)
class SourceCaptionParts:
    identity: str
    pose: str
    expression: str


HELMET_DETAIL_PHRASES: dict[str, str | None] = {
    "ctCorinthian": "carved Corinthian helmet with cheek guards, raised brow band, and vertical crest channels",
    "ctAttic": "carved Attic helmet with open face, brow ridge, cheek guards, and shallow crest band",
    "ctPhrygian": "carved Phrygian helmet with forward-curving cap, folded crown, and side cheek guards",
    "ctChalcidian": "carved Chalcidian helmet with open face, cheek pieces, and clean arched brow",
    "ctPilos": "conical pilos helmet with simple raised rim and smooth carved cap",
    "ctWinged": "winged helmet with a tall crest, raised side wings, and carved brow guard",
    "ctFloralWreath": "floral wreath circling the head with carved petals and leaf clusters",
    "ctIvyWreath": "ivy-and-grape wreath circling the head with carved leaves and grape clusters",
    "ctVineWreath": "vine wreath circling the head with twisting carved stems and leaves",
    "ctTopknotBaldric": "topknot bound with a carved fillet and clean hair bands",
    "ctStephane": "stephane diadem above the brow with a smooth raised crown band",
    "ctBare": None,
}

GARMENT_DETAIL_PHRASES: dict[str, str] = {
    "ctWarriorCuirass": "warrior cuirass with shoulder plates, diagonal cloak, and carved chest relief",
    "ctOrnateCuirass": "ornate cuirass with raised chest medallion, shoulder plates, cloak edge, and carved relief trim",
    "ctAthenaAegisChiton": "aegis over a chiton with scaled chest panel and layered cloth folds",
    "ctAthenaAegisPeplos": "aegis over a peplos with scaled chest panel, broad shoulder drape, and deep robe folds",
    "ctSageHimationDiagonal": "diagonal himation pinned at one shoulder with broad carved robe folds",
    "ctCivilianHimationChiton": "chiton with himation pulled across one shoulder, layered neckline, and soft vertical folds",
    "ctTogate": "toga gathered into a heavy front fold with broad wrapped bands across the torso",
    "ctPeplosChiton": "peplos over chiton with stacked neckline layers and vertical drapery channels",
    "ctBeltedPeplos": "high-belted peplos with cinched waist, sleeve edge, and wet-drapery folds",
}

SOURCE_PREFIX = "Convert this portrait into a CSTALE marble bust of "
MATERIAL_SUFFIX = (
    "matte weathered grey-brown stone, dry chalky unpolished surface, grime in "
    "hair grooves, helmet channels, armor relief, robe folds, garment seams, and "
    "drapery channels, chipped shoulders, broken lower bust, localized lava only "
    "in the lower cracks and fractures, dark ember background, no glossy polished "
    "marble, no wet shine, no specular hotspots, no frontal studio light, no "
    "head-on beauty lighting, no orange spill on face hair torso or background"
)


def parse_v6_source_caption(caption: str) -> SourceCaptionParts:
    normalized = _normalize_spaces(caption)
    if not normalized.startswith(SOURCE_PREFIX):
        raise CaptionParseError("expected caption prefix")

    sentences = _split_sentences(normalized)
    if len(sentences) < 3:
        raise CaptionParseError("expected identity, pose, and expression sentences")

    identity_sentence = sentences[0]
    identity_raw = identity_sentence.removeprefix(SOURCE_PREFIX).rstrip(".")
    identity = _render_identity(identity_raw)

    pose_sentence = next((sentence for sentence in sentences if sentence.startswith("The body is ")), None)
    if pose_sentence is None:
        raise CaptionParseError("expected body/head pose sentence")
    pose = pose_sentence.removeprefix("The body is ").rstrip(".")

    expression_sentence = sentences[-1].rstrip(".")
    if not expression_sentence.startswith("A "):
        raise CaptionParseError("expected final expression sentence")
    expression = expression_sentence[0].lower() + expression_sentence[1:]

    return SourceCaptionParts(identity=identity, pose=pose, expression=expression)


def render_archetype_details(archetypes: ArchetypePair) -> str:
    try:
        helmet = HELMET_DETAIL_PHRASES[archetypes.helmet]
    except KeyError as error:
        raise KeyError(f"unknown helmet archetype: {archetypes.helmet}") from error
    try:
        garment = GARMENT_DETAIL_PHRASES[archetypes.garment]
    except KeyError as error:
        raise KeyError(f"unknown garment archetype: {archetypes.garment}") from error

    if helmet is None:
        return garment
    return f"{helmet}, {garment}"


def render_v6_caption(source_caption: str, archetypes: ArchetypePair) -> str:
    parts = parse_v6_source_caption(source_caption)
    details = render_archetype_details(archetypes)
    return (
        "transform into a <mrblbust> from the reference portrait, "
        f"preserving {parts.identity}. "
        "The target is a Greek marble statue bust with "
        f"the body {parts.pose}, {parts.expression}, blank sculpted eyes, "
        f"{details}, {MATERIAL_SUFFIX}"
    )


def _render_identity(identity_raw: str) -> str:
    if " — " not in identity_raw:
        return identity_raw.strip()
    demographic, traits = identity_raw.split(" — ", 1)
    return f"{demographic.strip()} with {traits.strip()}"


def _normalize_spaces(text: str) -> str:
    return " ".join(text.strip().split())


def _split_sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=\.)\s+", text) if part.strip()]
```

- [ ] **Step 4: Run parser and renderer tests to verify they pass**

Run:

```bash
pytest tests/test_marble_v6_dataset.py -v
```

Expected: PASS for all tests in this new file.

- [ ] **Step 5: Commit Task 1**

Run:

```bash
git add src/klein4b/marble_v6_dataset.py tests/test_marble_v6_dataset.py
git commit -m "data: add v6 marble caption renderer"
```

---

## Task 2: Dataset Builder

**Files:**
- Modify: `src/klein4b/marble_v6_dataset.py`
- Test: `tests/test_marble_v6_dataset.py`

- [ ] **Step 1: Add failing builder tests**

Append to `tests/test_marble_v6_dataset.py`:

```python
import json

from klein4b.marble_v6_dataset import build_v6_199_dataset


def write_fake_v6_source(root: Path) -> None:
    control_dir = root / "editing-dataset-control" / "control"
    dataset_dir = root / "editing-dataset-control" / "dataset"
    control_dir.mkdir(parents=True)
    dataset_dir.mkdir(parents=True)
    for index, caption in {"001": SOURCE_CAPTION_001, "002": SOURCE_CAPTION_BARE}.items():
        (control_dir / f"{index}.png").write_bytes(f"control-{index}".encode("ascii"))
        (dataset_dir / f"{index}.png").write_bytes(f"bust-{index}".encode("ascii"))
        (dataset_dir / f"{index}.txt").write_text(caption, encoding="utf-8")

    editing_manifest_dir = root / "editing-dataset"
    editing_manifest_dir.mkdir()
    (editing_manifest_dir / "manifest.json").write_text(
        json.dumps(
            [
                {
                    "index": "001",
                    "source_id": "african_child_female_1",
                    "clothing_slug": "marble_mars_winged_helmet_gorgoneion_cuirass",
                    "prompt": SOURCE_CAPTION_001,
                },
                {
                    "index": "002",
                    "source_id": "european_youngadult_neutral_1",
                    "clothing_slug": "bronze_muscled_cuirass_anatomical",
                    "prompt": SOURCE_CAPTION_BARE,
                },
            ],
            indent=2,
        ),
        encoding="utf-8",
    )


def write_fake_archetypes(target: Path) -> None:
    target.mkdir(parents=True)
    (target / "archetypes.json").write_text(
        json.dumps(
            {
                "bronze_muscled_cuirass_anatomical": ["ctBare", "ctWarriorCuirass"],
                "marble_mars_winged_helmet_gorgoneion_cuirass": [
                    "ctWinged",
                    "ctOrnateCuirass",
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def test_build_v6_199_dataset_copies_pairs_and_rewrites_captions(tmp_path: Path) -> None:
    source = tmp_path / "remote" / "data-synthesis"
    target = tmp_path / "v6_199"
    write_fake_v6_source(source)
    write_fake_archetypes(target)

    report = build_v6_199_dataset(source_dir=source, target_dir=target, expected_count=2)

    assert report.row_count == 2
    assert (target / "control" / "001.png").read_bytes() == b"control-001"
    assert (target / "dataset" / "001.png").read_bytes() == b"bust-001"
    caption_001 = (target / "dataset" / "001.txt").read_text(encoding="utf-8")
    caption_002 = (target / "dataset" / "002.txt").read_text(encoding="utf-8")
    assert "preserving an African child female" in caption_001
    assert "winged helmet with a tall crest" in caption_001
    assert "ctWinged" not in caption_001
    assert "European young-adult gender-neutral person" in caption_002
    assert "helmet" not in caption_002.split("warrior cuirass", 1)[0]
    assert (target / "archetypes.json").exists()
    assert (target / "manifest.json").exists()
    assert (target / "CAPTION_REPORT.md").exists()


def test_build_v6_199_dataset_writes_manifest_with_archetype_metadata(tmp_path: Path) -> None:
    source = tmp_path / "remote" / "data-synthesis"
    target = tmp_path / "v6_199"
    write_fake_v6_source(source)
    write_fake_archetypes(target)

    build_v6_199_dataset(source_dir=source, target_dir=target, expected_count=2)

    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    assert manifest[0]["index"] == "001"
    assert manifest[0]["source_id"] == "african_child_female_1"
    assert manifest[0]["clothing_slug"] == "marble_mars_winged_helmet_gorgoneion_cuirass"
    assert manifest[0]["helmet_archetype"] == "ctWinged"
    assert manifest[0]["garment_archetype"] == "ctOrnateCuirass"
    assert manifest[0]["target_caption_path"].endswith("dataset/001.txt")
    assert "rewritten_caption" in manifest[0]


def test_build_v6_199_dataset_preserves_existing_archetypes_file(tmp_path: Path) -> None:
    source = tmp_path / "remote" / "data-synthesis"
    target = tmp_path / "v6_199"
    write_fake_v6_source(source)
    write_fake_archetypes(target)
    original_archetypes = (target / "archetypes.json").read_text(encoding="utf-8")

    build_v6_199_dataset(source_dir=source, target_dir=target, expected_count=2)

    assert (target / "archetypes.json").read_text(encoding="utf-8") == original_archetypes


def test_build_v6_199_dataset_fails_when_archetype_mapping_is_missing(tmp_path: Path) -> None:
    source = tmp_path / "remote" / "data-synthesis"
    target = tmp_path / "v6_199"
    write_fake_v6_source(source)
    target.mkdir(parents=True)
    (target / "archetypes.json").write_text("{}", encoding="utf-8")

    with pytest.raises(KeyError, match="missing archetype mapping"):
        build_v6_199_dataset(source_dir=source, target_dir=target, expected_count=2)


def test_build_v6_199_dataset_fails_when_expected_count_differs(tmp_path: Path) -> None:
    source = tmp_path / "remote" / "data-synthesis"
    target = tmp_path / "v6_199"
    write_fake_v6_source(source)
    write_fake_archetypes(target)

    with pytest.raises(ValueError, match="expected 199 rows, found 2"):
        build_v6_199_dataset(source_dir=source, target_dir=target, expected_count=199)
```

- [ ] **Step 2: Run builder test to verify it fails**

Run:

```bash
pytest tests/test_marble_v6_dataset.py::test_build_v6_199_dataset_copies_pairs_and_rewrites_captions -v
```

Expected: FAIL with `ImportError` or `AttributeError` because `build_v6_199_dataset` does not exist.

- [ ] **Step 3: Implement dataset builder**

Append these imports and dataclasses to `src/klein4b/marble_v6_dataset.py`:

```python
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class V6BuildReport:
    source_dir: Path
    target_dir: Path
    row_count: int
    archetype_counts: dict[str, int]
```

Add these functions below `render_v6_caption`:

```python
def build_v6_199_dataset(
    *,
    source_dir: Path,
    target_dir: Path,
    expected_count: int = 199,
) -> V6BuildReport:
    manifest_rows = _load_source_manifest(source_dir / "editing-dataset" / "manifest.json")
    if len(manifest_rows) != expected_count:
        raise ValueError(f"expected {expected_count} rows, found {len(manifest_rows)}")

    archetypes = _load_archetypes(target_dir / "archetypes.json")
    plan = _build_copy_plan(source_dir=source_dir, target_dir=target_dir, manifest_rows=manifest_rows, archetypes=archetypes)

    _replace_generated_subdir(target_dir / "control")
    _replace_generated_subdir(target_dir / "dataset")
    (target_dir / "control").mkdir(parents=True, exist_ok=True)
    (target_dir / "dataset").mkdir(parents=True, exist_ok=True)

    output_manifest: list[dict[str, object]] = []
    archetype_counter: Counter[str] = Counter()
    for row in plan:
        shutil.copy2(row["source_control_path"], row["target_control_path"])
        shutil.copy2(row["source_bust_path"], row["target_bust_path"])
        Path(row["target_caption_path"]).write_text(str(row["rewritten_caption"]) + "\n", encoding="utf-8")
        archetype_key = f"{row['helmet_archetype']} + {row['garment_archetype']}"
        archetype_counter[archetype_key] += 1
        output_manifest.append(_manifest_output_row(row))

    (target_dir / "manifest.json").write_text(
        json.dumps(output_manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_caption_report(
        target_dir=target_dir,
        source_dir=source_dir,
        row_count=len(output_manifest),
        archetype_counts=dict(sorted(archetype_counter.items())),
    )
    return V6BuildReport(
        source_dir=source_dir,
        target_dir=target_dir,
        row_count=len(output_manifest),
        archetype_counts=dict(sorted(archetype_counter.items())),
    )


def _load_source_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"source manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"source manifest must be a list: {path}")
    rows = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("source manifest rows must be objects")
        for key in ("index", "source_id", "clothing_slug"):
            if not isinstance(item.get(key), str) or not item[key]:
                raise ValueError(f"source manifest row missing string field: {key}")
        rows.append(item)
    return sorted(rows, key=lambda row: row["index"])


def _load_archetypes(path: Path) -> dict[str, ArchetypePair]:
    if not path.exists():
        raise FileNotFoundError(f"archetypes file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"archetypes file must be an object: {path}")
    archetypes: dict[str, ArchetypePair] = {}
    for slug, pair in payload.items():
        if not isinstance(slug, str):
            raise ValueError("archetype keys must be strings")
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError(f"{slug}: archetype mapping must be a two-item list")
        helmet, garment = pair
        if not isinstance(helmet, str) or not isinstance(garment, str):
            raise ValueError(f"{slug}: archetype names must be strings")
        archetypes[slug] = ArchetypePair(helmet=helmet, garment=garment)
    return archetypes


def _build_copy_plan(
    *,
    source_dir: Path,
    target_dir: Path,
    manifest_rows: list[dict[str, Any]],
    archetypes: dict[str, ArchetypePair],
) -> list[dict[str, object]]:
    control_dir = source_dir / "editing-dataset-control" / "control"
    dataset_dir = source_dir / "editing-dataset-control" / "dataset"
    plan: list[dict[str, object]] = []
    for row in manifest_rows:
        index = row["index"]
        clothing_slug = row["clothing_slug"]
        if clothing_slug not in archetypes:
            raise KeyError(f"missing archetype mapping for clothing_slug: {clothing_slug}")
        pair = archetypes[clothing_slug]
        source_control_path = control_dir / f"{index}.png"
        source_bust_path = dataset_dir / f"{index}.png"
        source_caption_path = dataset_dir / f"{index}.txt"
        for path in (source_control_path, source_bust_path, source_caption_path):
            if not path.exists():
                raise FileNotFoundError(f"required source file not found: {path}")
        source_caption = source_caption_path.read_text(encoding="utf-8")
        rewritten_caption = render_v6_caption(source_caption=source_caption, archetypes=pair)
        plan.append(
            {
                "index": index,
                "source_id": row["source_id"],
                "clothing_slug": clothing_slug,
                "helmet_archetype": pair.helmet,
                "garment_archetype": pair.garment,
                "source_control_path": source_control_path,
                "source_bust_path": source_bust_path,
                "source_caption_path": source_caption_path,
                "target_control_path": target_dir / "control" / f"{index}.png",
                "target_bust_path": target_dir / "dataset" / f"{index}.png",
                "target_caption_path": target_dir / "dataset" / f"{index}.txt",
                "rewritten_caption": rewritten_caption,
            }
        )
    return plan


def _replace_generated_subdir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _manifest_output_row(row: dict[str, object]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key in (
        "index",
        "source_id",
        "clothing_slug",
        "helmet_archetype",
        "garment_archetype",
        "source_control_path",
        "source_bust_path",
        "source_caption_path",
        "target_control_path",
        "target_bust_path",
        "target_caption_path",
        "rewritten_caption",
    ):
        value = row[key]
        output[key] = str(value) if isinstance(value, Path) else value
    return output


def _write_caption_report(
    *,
    target_dir: Path,
    source_dir: Path,
    row_count: int,
    archetype_counts: dict[str, int],
) -> None:
    lines = [
        "# V6 199 Caption Report",
        "",
        f"- Source: `{source_dir}`",
        f"- Target: `{target_dir}`",
        f"- Rows written: {row_count}",
        "- Caption policy: v5 marble material style with deterministic prose archetype details.",
        "",
        "## Archetype Coverage",
        "",
    ]
    for archetype_key, count in archetype_counts.items():
        lines.append(f"- `{archetype_key}`: {count}")
    lines.append("")
    (target_dir / "CAPTION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
```

- [ ] **Step 4: Run builder tests to verify they pass**

Run:

```bash
pytest tests/test_marble_v6_dataset.py -v
```

Expected: PASS for parser, renderer, and builder tests.

- [ ] **Step 5: Run formatting for the touched Python files**

Run:

```bash
ruff format src/klein4b/marble_v6_dataset.py tests/test_marble_v6_dataset.py
ruff check src/klein4b/marble_v6_dataset.py tests/test_marble_v6_dataset.py
```

Expected: both commands exit 0.

- [ ] **Step 6: Commit Task 2**

Run:

```bash
git add src/klein4b/marble_v6_dataset.py tests/test_marble_v6_dataset.py
git commit -m "data: build v6 marble dataset mirror"
```

---

## Task 3: CLI Entry Point

**Files:**
- Create: `scripts/prepare_marble_v6_199_dataset.py`
- Test: `tests/test_prepare_marble_v6_199_dataset_cli.py`

- [ ] **Step 1: Write failing CLI tests**

Create `tests/test_prepare_marble_v6_199_dataset_cli.py`:

```python
from __future__ import annotations

import importlib.util
from pathlib import Path


def load_cli_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "prepare_marble_v6_199_dataset.py"
    spec = importlib.util.spec_from_file_location("prepare_marble_v6_199_dataset", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_help_exits_without_building_dataset() -> None:
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "scripts/prepare_marble_v6_199_dataset.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--source" in result.stdout
    assert "--target" in result.stdout


def test_cli_calls_builder_with_explicit_paths(monkeypatch, tmp_path: Path) -> None:
    module = load_cli_module()
    source = tmp_path / "charliestale-ml" / "data-synthesis"
    target = tmp_path / "v6_199"
    calls: list[dict[str, object]] = []

    class Report:
        row_count = 199
        archetype_counts = {"ctWinged + ctOrnateCuirass": 5}

    def fake_build_v6_199_dataset(**kwargs: object) -> object:
        calls.append(dict(kwargs))
        return Report()

    monkeypatch.setattr(module, "build_v6_199_dataset", fake_build_v6_199_dataset)

    exit_code = module.main(["--source", str(source), "--target", str(target)])

    assert exit_code == 0
    assert calls == [{"source_dir": source, "target_dir": target, "expected_count": 199}]


def test_cli_supports_expected_count_override_for_smoke_tests(monkeypatch, tmp_path: Path) -> None:
    module = load_cli_module()
    calls: list[dict[str, object]] = []

    class Report:
        row_count = 2
        archetype_counts = {}

    def fake_build_v6_199_dataset(**kwargs: object) -> object:
        calls.append(dict(kwargs))
        return Report()

    monkeypatch.setattr(module, "build_v6_199_dataset", fake_build_v6_199_dataset)

    exit_code = module.main(
        [
            "--source",
            str(tmp_path / "source"),
            "--target",
            str(tmp_path / "target"),
            "--expected-count",
            "2",
        ]
    )

    assert exit_code == 0
    assert calls[0]["expected_count"] == 2
```

- [ ] **Step 2: Run CLI test to verify it fails**

Run:

```bash
pytest tests/test_prepare_marble_v6_199_dataset_cli.py::test_cli_help_exits_without_building_dataset -v
```

Expected: FAIL because `scripts/prepare_marble_v6_199_dataset.py` does not exist.

- [ ] **Step 3: Implement CLI script**

Create `scripts/prepare_marble_v6_199_dataset.py`:

```python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from klein4b.marble_v6_dataset import build_v6_199_dataset  # noqa: E402
from klein4b.paths import data_dir  # noqa: E402


def default_source_dir() -> Path:
    env_value = os.environ.get("KLEIN4B_V6_SOURCE_DIR")
    if env_value:
        return Path(env_value)
    return REPO_ROOT.parent / "charliestale-ml" / "data-synthesis"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare the v6 199-row marble bust control dataset with rewritten captions."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=default_source_dir(),
        help="Path to charliestale-ml/data-synthesis containing editing-dataset-control and editing-dataset/manifest.json.",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=data_dir() / "marble-bust-data" / "v6_199",
        help="Target v6_199 dataset directory containing archetypes.json.",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=199,
        help="Expected manifest row count. Keep 199 for the production v6 dataset.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_v6_199_dataset(
        source_dir=args.source,
        target_dir=args.target,
        expected_count=args.expected_count,
    )
    print(f"Wrote {report.row_count} rows to {args.target}")
    print(f"Archetype groups: {len(report.archetype_counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run CLI tests to verify they pass**

Run:

```bash
pytest tests/test_prepare_marble_v6_199_dataset_cli.py -v
```

Expected: PASS.

- [ ] **Step 5: Run formatting for CLI files**

Run:

```bash
ruff format scripts/prepare_marble_v6_199_dataset.py tests/test_prepare_marble_v6_199_dataset_cli.py
ruff check scripts/prepare_marble_v6_199_dataset.py tests/test_prepare_marble_v6_199_dataset_cli.py
```

Expected: both commands exit 0.

- [ ] **Step 6: Commit Task 3**

Run:

```bash
git add scripts/prepare_marble_v6_199_dataset.py tests/test_prepare_marble_v6_199_dataset_cli.py
git commit -m "data: add v6 dataset preparation cli"
```

---

## Task 4: Real Dataset Smoke Run

**Files:**
- Generated/ignored: `data/marble-bust-data/v6_199/control/`
- Generated/ignored: `data/marble-bust-data/v6_199/dataset/`
- Generated/ignored: `data/marble-bust-data/v6_199/manifest.json`
- Generated/ignored: `data/marble-bust-data/v6_199/CAPTION_REPORT.md`

No source file changes are expected in this task.

- [ ] **Step 1: Confirm the remote source checkout exists**

Use either a fresh checkout:

```bash
git clone --depth 1 --branch data-prep https://github.com/watsondg/charliestale-ml.git /tmp/charliestale-ml-data-prep
```

or the sparse checkout already created during analysis:

```text
/tmp/charliestale-ml-data-prep.Qd1oI7/repo/data-synthesis
```

Verify:

```bash
test -d /tmp/charliestale-ml-data-prep.Qd1oI7/repo/data-synthesis/editing-dataset-control
test -f /tmp/charliestale-ml-data-prep.Qd1oI7/repo/data-synthesis/editing-dataset/manifest.json
```

Expected: both commands exit 0.

- [ ] **Step 2: Run the preparation command**

Run:

```bash
python scripts/prepare_marble_v6_199_dataset.py \
  --source /tmp/charliestale-ml-data-prep.Qd1oI7/repo/data-synthesis \
  --target data/marble-bust-data/v6_199
```

Expected output includes:

```text
Wrote 199 rows to data/marble-bust-data/v6_199
Archetype groups:
```

- [ ] **Step 3: Verify output counts**

Run:

```bash
find data/marble-bust-data/v6_199/control -maxdepth 1 -type f -name '*.png' | wc -l
find data/marble-bust-data/v6_199/dataset -maxdepth 1 -type f -name '*.png' | wc -l
find data/marble-bust-data/v6_199/dataset -maxdepth 1 -type f -name '*.txt' | wc -l
```

Expected output:

```text
199
199
199
```

- [ ] **Step 4: Inspect representative rewritten captions**

Run:

```bash
sed -n '1p' data/marble-bust-data/v6_199/dataset/001.txt
sed -n '1p' data/marble-bust-data/v6_199/dataset/100.txt
sed -n '1p' data/marble-bust-data/v6_199/dataset/199.txt
```

Expected:

- Each starts with `transform into a <mrblbust> from the reference portrait`.
- `001.txt` preserves `African child female`.
- No line contains raw `ct` archetype tokens.
- Each includes v5-style matte stone, grime, broken lower bust, localized lava, and negative lighting terms.

- [ ] **Step 5: Run full fast verification**

Run:

```bash
pytest -q
ruff check .
```

Expected:

```text
100+ passed
All checks passed!
```

The exact test count will increase after the new tests are added.

- [ ] **Step 6: Do not commit ignored data outputs**

Run:

```bash
git status --short
```

Expected: generated `data/` outputs are ignored and do not appear. Only planned source/test/script files should appear if any task has not yet been committed. Existing unrelated dirty files may still appear:

```text
 M tests/test_training.py
?? configs/train_flux2_klein_marble_bust_v5_clothing_armor_rank64_unquantized.template.yaml
```

---

## Final Verification

After all implementation commits:

```bash
pytest -q
ruff check .
python scripts/prepare_marble_v6_199_dataset.py \
  --source /tmp/charliestale-ml-data-prep.Qd1oI7/repo/data-synthesis \
  --target data/marble-bust-data/v6_199
```

Expected:

- All tests pass.
- Ruff passes.
- The script writes exactly 199 control images, 199 bust images, and 199 rewritten captions.
- `data/marble-bust-data/v6_199/archetypes.json` remains unchanged.
- Rewritten captions are prose-only and preserve race labels where the source captions contain them.

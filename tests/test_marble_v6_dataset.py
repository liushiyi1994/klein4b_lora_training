from __future__ import annotations

import json
from pathlib import Path

import pytest

from klein4b.marble_v6_dataset import (
    ArchetypePair,
    CaptionParseError,
    build_v6_199_dataset,
    parse_v6_source_caption,
    render_archetype_details,
    render_v6_caption,
)

SOURCE_CAPTION_001 = (
    "Convert this portrait into a CSTALE marble bust of an African child female \u2014 "
    "short coiled hair gathered into two puffs, round face, full cheeks, almond eyes, "
    "broad nose with rounded tip, full lips. The figure wears a winged helmet with a "
    "tall crest. An ornate cuirass shapes the torso, with a cloak draped diagonally "
    "across one shoulder. The body is in strict left profile; the head is turned back "
    "over the left shoulder, chin lifted, with a heroic upward gaze. A serene "
    "archaic-smile expression."
)


SOURCE_CAPTION_BARE = (
    "Convert this portrait into a CSTALE marble bust of a European young-adult "
    "gender-neutral person \u2014 shoulder-length wavy hair loose and flowing, oval face, "
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
    details = render_archetype_details(ArchetypePair(helmet="ctWinged", garment="ctOrnateCuirass"))

    assert "ctWinged" not in details
    assert "ctOrnateCuirass" not in details
    assert "winged helmet with a tall crest" in details
    assert "raised chest medallion" in details
    assert "carved relief trim" in details


def test_render_archetype_details_for_bare_head_omits_helmet_phrase() -> None:
    details = render_archetype_details(ArchetypePair(helmet="ctBare", garment="ctWarriorCuirass"))

    assert "helmet" not in details
    assert "bare" not in details.lower()
    assert details.startswith("warrior cuirass")


def test_render_archetype_details_for_attic_helmet_includes_tall_crest() -> None:
    details = render_archetype_details(
        ArchetypePair(helmet="ctAttic", garment="ctAthenaAegisChiton")
    )

    assert "tall crested Attic Athena helmet" in details
    assert "raised horsehair crest" in details
    assert "laurel relief" in details


def test_render_archetype_details_uses_corinthian_pushed_up_slug_prompt() -> None:
    details = render_archetype_details(
        ArchetypePair(helmet="ctCorinthian", garment="ctAthenaAegisPeplos"),
        clothing_slug="bronze_athena_corinthian_pushed_up",
    )

    assert "pushed-up Corinthian Athena helmet" in details
    assert "broad rear fan-shaped horsehair crest" in details
    assert "no separate animal statue or freestanding finial" in details


def test_render_archetype_details_uses_chalcidian_figural_cheek_slug_prompt() -> None:
    details = render_archetype_details(
        ArchetypePair(helmet="ctChalcidian", garment="ctWarriorCuirass"),
        clothing_slug="bronze_chalcidian_helmet_figural_cheek",
    )

    assert "smooth close-fitting Chalcidian helmet cap" in details
    assert "separate hinged cheek guard plates" in details
    assert "no crest, no tall plume, no folded Phrygian cap" in details


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
    assert "featureless blank marble eye surfaces" in caption
    assert "matte weathered grey-brown stone" in caption
    assert "localized lava only in the lower cracks and fractures" in caption
    assert "no glossy polished marble" in caption
    assert "no orange spill on face hair torso or background" in caption
    assert "CSTALE" not in caption
    assert "ctWinged" not in caption


def test_render_v6_caption_for_helmeted_rows_removes_reference_headwear() -> None:
    caption = render_v6_caption(
        source_caption=SOURCE_CAPTION_001,
        archetypes=ArchetypePair(helmet="ctWinged", garment="ctOrnateCuirass"),
    )

    assert "ignore and remove any hat or modern headwear from the reference image" in caption
    assert "replace it with" in caption


def test_render_v6_caption_replaces_gaze_with_head_angle_and_blank_eye_surface() -> None:
    caption = render_v6_caption(
        source_caption=SOURCE_CAPTION_001,
        archetypes=ArchetypePair(helmet="ctWinged", garment="ctOrnateCuirass"),
    )

    assert "gaze" not in caption
    assert "heroic upward head angle" in caption
    assert "almond eyes" not in caption
    assert "almond eye openings" in caption
    assert "featureless blank marble eye surfaces" in caption
    assert "pupil-less" in caption
    assert "no circular markings or dark centers" in caption


def test_render_v6_caption_applies_clothing_slug_specific_helmet_details() -> None:
    caption = render_v6_caption(
        source_caption=SOURCE_CAPTION_001,
        archetypes=ArchetypePair(helmet="ctChalcidian", garment="ctWarriorCuirass"),
        clothing_slug="bronze_chalcidian_helmet_figural_cheek",
    )

    assert "smooth close-fitting Chalcidian helmet cap" in caption
    assert "separate hinged cheek guard plates" in caption
    assert "no crest, no tall plume, no folded Phrygian cap" in caption


def test_render_v6_caption_raises_clear_error_for_unknown_archetype() -> None:
    with pytest.raises(KeyError, match="unknown helmet archetype"):
        render_archetype_details(ArchetypePair(helmet="ctMissing", garment="ctWarriorCuirass"))


def test_parse_v6_source_caption_raises_clear_error_for_unexpected_shape() -> None:
    with pytest.raises(CaptionParseError, match="expected caption prefix"):
        parse_v6_source_caption("not a valid source caption")


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


def test_build_v6_199_dataset_writes_manifest_with_archetype_metadata(
    tmp_path: Path,
) -> None:
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


def test_build_v6_199_dataset_fails_when_archetype_mapping_is_missing(
    tmp_path: Path,
) -> None:
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

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

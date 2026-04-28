from pathlib import Path

from klein4b.marble_dataset import (
    HAIR_MATERIAL_CLAUSE,
    V3_MATTE_LAVA_CLAUSE,
    build_marble_v2_dataset,
    build_marble_v3_dataset,
    normalize_marble_caption,
    normalize_marble_v3_caption,
)


def test_normalize_marble_caption_adds_stone_hair_material_once() -> None:
    caption = (
        "a <mrblbust> marble statue bust of a young adult woman, "
        "long wavy hair parted in the center"
    )

    normalized = normalize_marble_caption(caption)
    renormalized = normalize_marble_caption(normalized)

    assert HAIR_MATERIAL_CLAUSE in normalized
    assert normalized.count(HAIR_MATERIAL_CLAUSE) == 1
    assert renormalized == normalized
    assert "broad chiseled grooves" in normalized
    assert "marble veins continuing through the hair" in normalized
    assert "no individual hair strands" in normalized
    assert "no natural hair color" in normalized
    assert "no glossy hair" in normalized
    assert "no soft hair strands" in normalized


def test_normalize_marble_v3_caption_localizes_lava_and_matte_material() -> None:
    caption = (
        "a <mrblbust> marble statue bust, the bottom of the bust terminates in "
        "a jagged irregular broken edge with missing three-dimensional chunks, "
        "not cut flat, glowing orange-red embers and lava seep from that broken "
        "edge, no loose rocks or debris below"
    )

    normalized = normalize_marble_v3_caption(caption)
    renormalized = normalize_marble_v3_caption(normalized)

    assert "glowing orange-red embers and lava seep" not in normalized
    assert "small localized orange-red lava fissures" in normalized
    assert "matte weathered grey-white marble" in normalized
    assert "dry chalky unpolished stone" in normalized
    assert "no red or orange light spilling onto the face" in normalized
    assert "no glossy polished marble" in normalized
    assert "no specular hotspots" in normalized
    assert "no overexposed white face" in normalized
    assert V3_MATTE_LAVA_CLAUSE in normalized
    assert normalized.count(V3_MATTE_LAVA_CLAUSE) == 1
    assert renormalized == normalized


def test_build_marble_v2_dataset_copies_images_and_rewrites_captions(
    tmp_path: Path,
) -> None:
    source = tmp_path / "v1"
    target = tmp_path / "v2"
    (source / "busts").mkdir(parents=True)
    (source / "pairs").mkdir()
    (source / "busts" / "1.png").write_bytes(b"image")
    (source / "busts" / "1.txt").write_text(
        "a <mrblbust> marble statue bust with braided hair",
        encoding="utf-8",
    )
    (source / "pairs" / "1_input.jpg").write_bytes(b"input")
    (source / "pairs" / "1_target.jpg").write_bytes(b"target")
    (source / "pairs" / "1.txt").write_text(
        "transform into a <mrblbust> marble statue bust with wavy hair",
        encoding="utf-8",
    )
    (source / "manifest.json").write_text('{"version": "v1"}', encoding="utf-8")

    report = build_marble_v2_dataset(source, target)

    assert report.caption_count == 2
    assert report.image_count == 3
    assert (target / "busts" / "1.png").read_bytes() == b"image"
    assert (target / "pairs" / "1_input.jpg").read_bytes() == b"input"
    assert HAIR_MATERIAL_CLAUSE in (target / "busts" / "1.txt").read_text(encoding="utf-8")
    assert HAIR_MATERIAL_CLAUSE in (target / "pairs" / "1.txt").read_text(encoding="utf-8")
    assert "v2" in (target / "CAPTION_REPORT.md").read_text(encoding="utf-8")


def test_build_marble_v3_dataset_copies_images_and_rewrites_captions(
    tmp_path: Path,
) -> None:
    source = tmp_path / "v1"
    target = tmp_path / "v3"
    (source / "busts").mkdir(parents=True)
    (source / "busts" / "1.png").write_bytes(b"image")
    (source / "busts" / "1.txt").write_text(
        "a <mrblbust> marble statue bust with braided hair, "
        "glowing orange-red embers and lava seep from that broken edge",
        encoding="utf-8",
    )
    (source / "manifest.json").write_text('{"version": "v1"}', encoding="utf-8")

    report = build_marble_v3_dataset(source, target)
    caption = (target / "busts" / "1.txt").read_text(encoding="utf-8")

    assert report.caption_count == 1
    assert report.image_count == 1
    assert (target / "busts" / "1.png").read_bytes() == b"image"
    assert "small localized orange-red lava fissures" in caption
    assert V3_MATTE_LAVA_CLAUSE in caption
    assert "matte_marble_localized_lava_v1" in (target / "manifest.json").read_text(
        encoding="utf-8"
    )
    assert "v3" in (target / "CAPTION_REPORT.md").read_text(encoding="utf-8")

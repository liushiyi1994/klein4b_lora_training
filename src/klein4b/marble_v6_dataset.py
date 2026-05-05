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
    "ctCorinthian": (
        "carved Corinthian helmet with cheek guards, raised brow band, "
        "and vertical crest channels"
    ),
    "ctAttic": (
        "carved Attic helmet with open face, brow ridge, cheek guards, "
        "and shallow crest band"
    ),
    "ctPhrygian": (
        "carved Phrygian helmet with forward-curving cap, folded crown, "
        "and side cheek guards"
    ),
    "ctChalcidian": (
        "carved Chalcidian helmet with open face, cheek pieces, and clean arched brow"
    ),
    "ctPilos": "conical pilos helmet with simple raised rim and smooth carved cap",
    "ctWinged": "winged helmet with a tall crest, raised side wings, and carved brow guard",
    "ctFloralWreath": "floral wreath circling the head with carved petals and leaf clusters",
    "ctIvyWreath": (
        "ivy-and-grape wreath circling the head with carved leaves and grape clusters"
    ),
    "ctVineWreath": "vine wreath circling the head with twisting carved stems and leaves",
    "ctTopknotBaldric": "topknot bound with a carved fillet and clean hair bands",
    "ctStephane": "stephane diadem above the brow with a smooth raised crown band",
    "ctBare": None,
}

GARMENT_DETAIL_PHRASES: dict[str, str] = {
    "ctWarriorCuirass": (
        "warrior cuirass with shoulder plates, diagonal cloak, and carved chest relief"
    ),
    "ctOrnateCuirass": (
        "ornate cuirass with raised chest medallion, shoulder plates, cloak edge, "
        "and carved relief trim"
    ),
    "ctAthenaAegisChiton": (
        "aegis over a chiton with scaled chest panel and layered cloth folds"
    ),
    "ctAthenaAegisPeplos": (
        "aegis over a peplos with scaled chest panel, broad shoulder drape, "
        "and deep robe folds"
    ),
    "ctSageHimationDiagonal": (
        "diagonal himation pinned at one shoulder with broad carved robe folds"
    ),
    "ctCivilianHimationChiton": (
        "chiton with himation pulled across one shoulder, layered neckline, "
        "and soft vertical folds"
    ),
    "ctTogate": (
        "toga gathered into a heavy front fold with broad wrapped bands across the torso"
    ),
    "ctPeplosChiton": (
        "peplos over chiton with stacked neckline layers and vertical drapery channels"
    ),
    "ctBeltedPeplos": (
        "high-belted peplos with cinched waist, sleeve edge, and wet-drapery folds"
    ),
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

    identity_raw = sentences[0].removeprefix(SOURCE_PREFIX).rstrip(".")
    identity = _render_identity(identity_raw)

    pose_sentence = next(
        (sentence for sentence in sentences if sentence.startswith("The body is ")),
        None,
    )
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
    if " \u2014 " in identity_raw:
        demographic, traits = identity_raw.split(" \u2014 ", 1)
        return f"{demographic.strip()} with {traits.strip()}"
    if " - " in identity_raw:
        demographic, traits = identity_raw.split(" - ", 1)
        return f"{demographic.strip()} with {traits.strip()}"
    return identity_raw.strip()


def _normalize_spaces(text: str) -> str:
    return " ".join(text.strip().split())


def _split_sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=\.)\s+", text) if part.strip()]

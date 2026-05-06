from __future__ import annotations

import json
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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


@dataclass(frozen=True)
class V6BuildReport:
    source_dir: Path
    target_dir: Path
    row_count: int
    archetype_counts: dict[str, int]


HELMET_DETAIL_PHRASES: dict[str, str | None] = {
    "ctCorinthian": (
        "carved Corinthian helmet with cheek guards, raised brow band, and vertical crest channels"
    ),
    "ctAttic": (
        "tall crested Attic Athena helmet with raised horsehair crest, open face, "
        "brow ridge, cheek guards, shallow crest band, and laurel relief"
    ),
    "ctPhrygian": (
        "carved Phrygian helmet with forward-curving cap, folded crown, and side cheek guards"
    ),
    "ctChalcidian": (
        "carved Chalcidian helmet with open face, cheek pieces, and clean arched brow"
    ),
    "ctPilos": "conical pilos helmet with simple raised rim and smooth carved cap",
    "ctWinged": "winged helmet with a tall crest, raised side wings, and carved brow guard",
    "ctFloralWreath": "floral wreath circling the head with carved petals and leaf clusters",
    "ctIvyWreath": ("ivy-and-grape wreath circling the head with carved leaves and grape clusters"),
    "ctVineWreath": "vine wreath circling the head with twisting carved stems and leaves",
    "ctTopknotBaldric": "topknot bound with a carved fillet and clean hair bands",
    "ctStephane": "stephane diadem above the brow with a smooth raised crown band",
    "ctBare": None,
}
SLUG_HELMET_DETAIL_PHRASES: dict[str, str] = {
    "bronze_athena_corinthian_pushed_up": (
        "pushed-up Corinthian Athena helmet with a smooth helmet bowl raised above the brow, "
        "open face, visor-like raised brow band, cheek guards lifted beside the temples, "
        "and a broad rear fan-shaped horsehair crest attached behind the helmet with vertical "
        "plume ridges, no separate animal statue or freestanding finial on top"
    ),
    "bronze_chalcidian_helmet_figural_cheek": (
        "smooth close-fitting Chalcidian helmet cap with rounded skull bowl, open face, "
        "clean arched brow band, clear ear opening, separate hinged cheek guard plates "
        "with figural cheek outline, neck guard, and incised edge lines, no crest, no tall "
        "plume, no folded Phrygian cap"
    ),
}

GARMENT_DETAIL_PHRASES: dict[str, str] = {
    "ctWarriorCuirass": (
        "warrior cuirass with shoulder plates, diagonal cloak, and carved chest relief"
    ),
    "ctOrnateCuirass": (
        "ornate cuirass with raised chest medallion, shoulder plates, cloak edge, "
        "and carved relief trim"
    ),
    "ctAthenaAegisChiton": ("aegis over a chiton with scaled chest panel and layered cloth folds"),
    "ctAthenaAegisPeplos": (
        "aegis over a peplos with scaled chest panel, broad shoulder drape, and deep robe folds"
    ),
    "ctSageHimationDiagonal": (
        "diagonal himation pinned at one shoulder with broad carved robe folds"
    ),
    "ctCivilianHimationChiton": (
        "chiton with himation pulled across one shoulder, layered neckline, and soft vertical folds"
    ),
    "ctTogate": ("toga gathered into a heavy front fold with broad wrapped bands across the torso"),
    "ctPeplosChiton": (
        "peplos over chiton with stacked neckline layers and vertical drapery channels"
    ),
    "ctBeltedPeplos": ("high-belted peplos with cinched waist, sleeve edge, and wet-drapery folds"),
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
REFERENCE_HEADWEAR_REPLACEMENT = (
    "ignore and remove any hat or modern headwear from the reference image, replace it with"
)
BLANK_EYE_PHRASE = (
    "featureless blank marble eye surfaces, pupil-less, iris-less, "
    "with no circular markings or dark centers"
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


def render_archetype_details(
    archetypes: ArchetypePair,
    *,
    clothing_slug: str | None = None,
) -> str:
    try:
        helmet = HELMET_DETAIL_PHRASES[archetypes.helmet]
    except KeyError as error:
        raise KeyError(f"unknown helmet archetype: {archetypes.helmet}") from error
    try:
        garment = GARMENT_DETAIL_PHRASES[archetypes.garment]
    except KeyError as error:
        raise KeyError(f"unknown garment archetype: {archetypes.garment}") from error

    if clothing_slug in SLUG_HELMET_DETAIL_PHRASES:
        helmet = SLUG_HELMET_DETAIL_PHRASES[clothing_slug]

    if helmet is None:
        return garment
    return f"{REFERENCE_HEADWEAR_REPLACEMENT} {helmet}, {garment}"


def render_v6_caption(
    source_caption: str,
    archetypes: ArchetypePair,
    *,
    clothing_slug: str | None = None,
) -> str:
    parts = parse_v6_source_caption(source_caption)
    identity = _render_blank_eye_identity(parts.identity)
    pose = _render_blank_eye_pose(parts.pose)
    details = render_archetype_details(archetypes, clothing_slug=clothing_slug)
    return (
        "transform into a <mrblbust> from the reference portrait, "
        f"preserving {identity}. "
        "The target is a Greek marble statue bust with "
        f"the body {pose}, {parts.expression}, {BLANK_EYE_PHRASE}, "
        f"{details}, {MATERIAL_SUFFIX}"
    )


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
    plan = _build_copy_plan(
        source_dir=source_dir,
        target_dir=target_dir,
        manifest_rows=manifest_rows,
        archetypes=archetypes,
    )

    _replace_generated_subdir(target_dir / "control")
    _replace_generated_subdir(target_dir / "dataset")
    (target_dir / "control").mkdir(parents=True, exist_ok=True)
    (target_dir / "dataset").mkdir(parents=True, exist_ok=True)

    output_manifest: list[dict[str, object]] = []
    archetype_counter: Counter[str] = Counter()
    for row in plan:
        shutil.copy2(row["source_control_path"], row["target_control_path"])
        shutil.copy2(row["source_bust_path"], row["target_bust_path"])
        Path(row["target_caption_path"]).write_text(
            str(row["rewritten_caption"]) + "\n",
            encoding="utf-8",
        )
        archetype_key = f"{row['helmet_archetype']} + {row['garment_archetype']}"
        archetype_counter[archetype_key] += 1
        output_manifest.append(_manifest_output_row(row))

    (target_dir / "manifest.json").write_text(
        json.dumps(output_manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    archetype_counts = dict(sorted(archetype_counter.items()))
    _write_caption_report(
        target_dir=target_dir,
        source_dir=source_dir,
        row_count=len(output_manifest),
        archetype_counts=archetype_counts,
    )
    return V6BuildReport(
        source_dir=source_dir,
        target_dir=target_dir,
        row_count=len(output_manifest),
        archetype_counts=archetype_counts,
    )


def _render_identity(identity_raw: str) -> str:
    if " \u2014 " in identity_raw:
        demographic, traits = identity_raw.split(" \u2014 ", 1)
        return f"{demographic.strip()} with {traits.strip()}"
    if " - " in identity_raw:
        demographic, traits = identity_raw.split(" - ", 1)
        return f"{demographic.strip()} with {traits.strip()}"
    return identity_raw.strip()


def _render_blank_eye_pose(pose: str) -> str:
    replacements = {
        "with a heroic upward gaze": "with a heroic upward head angle",
        "with a downcast contemplative gaze": "with a downward contemplative head angle",
        "with a direct gaze toward the viewer": "with the face oriented toward the viewer",
        "with an off-side gaze away from the viewer": "with the face angled away from the viewer",
    }
    for old, new in replacements.items():
        pose = pose.replace(old, new)
    return pose.replace(" gaze", " head angle")


def _render_blank_eye_identity(identity: str) -> str:
    return re.sub(r"\beyes\b", "eye openings", identity)


def _normalize_spaces(text: str) -> str:
    return " ".join(text.strip().split())


def _split_sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=\.)\s+", text) if part.strip()]


def _load_source_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"source manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"source manifest must be a list: {path}")

    rows: list[dict[str, Any]] = []
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
        rewritten_caption = render_v6_caption(
            source_caption=source_caption,
            archetypes=pair,
            clothing_slug=clothing_slug,
        )
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

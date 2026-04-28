from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

HAIR_MATERIAL_CLAUSE = (
    "hair, beard, eyebrows, and eyelashes are carved from the same white marble "
    "as the face, using broad chiseled grooves, solid stone masses, and marble "
    "veins continuing through the hair; no individual hair strands, no natural "
    "hair color, no glossy hair, no soft hair strands"
)

V3_MATTE_LAVA_CLAUSE = (
    "matte weathered grey-white marble with a dry chalky unpolished stone "
    "surface, controlled dramatic dark background with sparse ember particles "
    "behind the bust, localized orange-red lava glow only inside cracks of the "
    "broken bottom base, no red or orange light spilling onto the face, hair, "
    "torso, shoulders, or background, no glossy polished marble, no wet shine, "
    "no specular hotspots, no overexposed white face"
)

V1_LAVA_PHRASE = "glowing orange-red embers and lava seep from that broken edge"
V3_LOCALIZED_LAVA_PHRASE = (
    "small localized orange-red lava fissures glow only inside cracks of that broken bottom edge"
)


@dataclass(frozen=True)
class MarbleCaptionReport:
    source_dir: Path
    target_dir: Path
    caption_count: int
    image_count: int


def normalize_marble_caption(caption: str) -> str:
    normalized = " ".join(caption.strip().split())
    if HAIR_MATERIAL_CLAUSE in normalized:
        return normalized
    return f"{normalized}, {HAIR_MATERIAL_CLAUSE}"


def normalize_marble_v3_caption(caption: str) -> str:
    normalized = normalize_marble_caption(caption)
    normalized = normalized.replace(V1_LAVA_PHRASE, V3_LOCALIZED_LAVA_PHRASE)
    if V3_MATTE_LAVA_CLAUSE in normalized:
        return normalized
    return f"{normalized}, {V3_MATTE_LAVA_CLAUSE}"


def build_marble_dataset(
    source_dir: Path,
    target_dir: Path,
    *,
    version: str,
    caption_policy: str,
) -> MarbleCaptionReport:
    if not source_dir.exists():
        raise FileNotFoundError(f"Source dataset does not exist: {source_dir}")
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True)

    normalizer = {
        "v2": normalize_marble_caption,
        "v3": normalize_marble_v3_caption,
    }[version]

    caption_count = 0
    image_count = 0
    for source_path in sorted(source_dir.rglob("*")):
        if source_path.is_dir():
            continue
        relative_path = source_path.relative_to(source_dir)
        target_path = target_dir / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if source_path.suffix.lower() == ".txt":
            target_path.write_text(
                normalizer(source_path.read_text(encoding="utf-8")),
                encoding="utf-8",
            )
            caption_count += 1
        else:
            shutil.copy2(source_path, target_path)
            if source_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                image_count += 1

    manifest_path = target_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(manifest, dict):
            manifest["version"] = version
            manifest["caption_policy"] = caption_policy
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    report = MarbleCaptionReport(
        source_dir=source_dir,
        target_dir=target_dir,
        caption_count=caption_count,
        image_count=image_count,
    )
    write_caption_report(report, version=version, caption_policy=caption_policy)
    return report


def build_marble_v2_dataset(source_dir: Path, target_dir: Path) -> MarbleCaptionReport:
    return build_marble_dataset(
        source_dir,
        target_dir,
        version="v2",
        caption_policy="marble_hair_material_v1",
    )


def build_marble_v3_dataset(source_dir: Path, target_dir: Path) -> MarbleCaptionReport:
    return build_marble_dataset(
        source_dir,
        target_dir,
        version="v3",
        caption_policy="matte_marble_localized_lava_v1",
    )


def write_caption_report(
    report: MarbleCaptionReport,
    *,
    version: str,
    caption_policy: str,
) -> None:
    body = "\n".join(
        [
            f"# Marble Bust Dataset {version} Caption Report",
            "",
            f"- Source: `{report.source_dir}`",
            f"- Target: `{report.target_dir}`",
            f"- Captions rewritten: {report.caption_count}",
            f"- Images copied: {report.image_count}",
            f"- Caption policy: {caption_policy}.",
            "",
        ]
    )
    (report.target_dir / "CAPTION_REPORT.md").write_text(body, encoding="utf-8")

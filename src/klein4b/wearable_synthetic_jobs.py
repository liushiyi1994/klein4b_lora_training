from __future__ import annotations

import json
import os
import re
import shutil
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

DEFAULT_SUBJECTS_DIR = Path(
    "/tmp/charliestale-ml-structured-captions/data-synthesis/assets/subjects"
)
DEFAULT_CAPTIONS_DIR = Path("/tmp/charliestale-ml-structured-captions/data-synthesis/captions")
DEFAULT_WEARABLE_MANIFEST = Path(
    "data/marble-bust-data/v7_30/statue_reference_pool/wearable_v1/manifest.jsonl"
)
DEFAULT_OUTPUT_DIR = Path(
    "data/marble-bust-data/v7_30/synthetic_generation/wearable_v1_15"
)
DEFAULT_GOLDEN_EXAMPLES = (
    Path("/home/liushiyi1994/projects/charlie_tale/klein_training/golden_example/1.png"),
    Path("/home/liushiyi1994/projects/charlie_tale/klein_training/golden_example/2.png"),
)
DEFAULT_IMAGE_MODEL = "gemini-3-pro-image-preview"
DEFAULT_CAPTION_MODEL = "gemini-2.5-flash"
DEFAULT_RETRY_DELAY_SECONDS = 5
DEFAULT_MAX_RETRIES = 5
MIN_ACTIVE_REFERENCES_PER_ARCHETYPE = 4
PROMPT_PROFILE_VERSION = "wearable_v2_smaller_9x16"

ARCHETYPE_ORDER = (
    "CSDIONYSIAN_REVELER",
    "CSARMORED_GENERAL",
    "CSDRAPED_SCHOLAR",
    "CSHELMETED_HERO_WARRIOR",
)
ETHNICITIES = (
    "african",
    "eastasian",
    "european",
    "hispaniclatino",
    "middleeastern",
    "southasian",
)
AGES = ("youngadult", "middleaged", "elderly", "child")
GENDERS = ("male", "female", "neutral")


@dataclass(frozen=True)
class Subject:
    id: str
    path: Path
    ethnicity: str
    age: str
    gender: str
    replicate: int

    @property
    def demographic_phrase(self) -> str:
        ethnicity = {
            "african": "African",
            "eastasian": "East Asian",
            "european": "European",
            "hispaniclatino": "Hispanic or Latino",
            "middleeastern": "Middle Eastern",
            "southasian": "South Asian",
        }.get(self.ethnicity, self.ethnicity)
        age = {
            "youngadult": "young-adult",
            "middleaged": "middle-aged adult",
            "elderly": "elderly adult",
            "child": "child",
        }.get(self.age, self.age)
        if self.gender == "neutral":
            return f"a {ethnicity} {age} gender-neutral person"
        article = "an" if ethnicity[0].lower() in {"a", "e", "i", "o", "u"} else "a"
        return f"{article} {ethnicity} {age} {self.gender}"


@dataclass(frozen=True)
class WearableReference:
    archetype: str
    rank: int
    path: Path
    filename: str
    wearable_cue: str
    source_url: str | None


@dataclass(frozen=True)
class PromptProfile:
    contract_label: str
    positive_contract: str
    reference_contract: str
    negative_contract: str

    def render(self, wearable_cue: str) -> str:
        return (
            f"{self.contract_label}: {self.positive_contract} "
            f"Clothing reference cue: {wearable_cue}. "
            f"{self.reference_contract} {self.negative_contract}"
        )


@dataclass(frozen=True)
class SyntheticJob:
    index: int
    archetype: str
    archetype_slot: int
    subject: Subject
    reference: WearableReference
    golden_examples: tuple[Path, Path]
    captions_dir: Path | None = None

    @property
    def id(self) -> str:
        return f"{self.index:03d}_{self.archetype}_{self.subject.id}"


class TransientApiError(RuntimeError):
    pass


@dataclass(frozen=True)
class JobPlanConfig:
    subjects_dir: Path = DEFAULT_SUBJECTS_DIR
    wearable_manifest: Path = DEFAULT_WEARABLE_MANIFEST
    output_dir: Path = DEFAULT_OUTPUT_DIR
    per_archetype: int = 15
    captions_dir: Path | None = DEFAULT_CAPTIONS_DIR
    golden_examples: tuple[Path, Path] = field(default_factory=lambda: DEFAULT_GOLDEN_EXAMPLES)


CAPTION_INSTRUCTION = (
    "Output exactly two lines. LINE 1 is a machine-readable header in this exact "
    "format: [gender: <male|female|neutral>; age: <child|adult>]. LINE 2 begins "
    "with a short demographic phrase of the form 'a <ethnicity> <gender-word> "
    "<age-word>, ' where ethnicity is one of African, European, East Asian, South "
    "Asian, Hispanic or Latino, Middle Eastern. Then add 6 to 10 comma-separated "
    "sculptor-useful facial descriptors: hair length, hair texture, hairstyle, "
    "facial hair only if visible, face shape, cheeks, eyes, nose, lips, and optional "
    "brow or jaw. Do not mention color, personality, expression, clothing, "
    "background, lighting, teeth, open mouth, or complete sentences."
)

PREAMBLE = (
    "use the first two images as style references, keep lighting condition the same, "
    "keep statue surface color same, and use their dark ember background and centered "
    "marble bust composition. Keep the final image as a 9:16 portrait composition. "
    "Match the golden references for material, lighting, background, and centered "
    "placement, but make the bust smaller inside the tall frame. Statue occupies 45 "
    "to 55 percent of the full frame height, measured from the highest carved point "
    "including any wreath or helmet to the lowest broken edge. Leave at least 20 "
    "percent empty dark space above the highest point and at least 15 percent empty "
    "dark space below the lowest broken edge. Keep generous side margins; the bust "
    "should not exceed 72 percent of the frame width. Fully visible. Centered. Clear "
    "negative space on all sides. No edge contact. Camera pulled back, not close-up. "
    "Never render a straight-on frontal bust: the shoulders, torso, helmet, and face "
    "must all be in a three-quarter angle, not frontal, with one side plane visibly "
    "receding. Do not make the marble bust dark gray, black, blue-gray, or "
    "underexposed; keep the face, helmet, armor, and drapery as readable medium-light "
    "marble against the dark background, with clear separation along the jaw, neck, "
    "shoulders, and folds. "
    "The first two images control only material, lighting, background, particle "
    "atmosphere, bust scale, and framing. Do not copy helmets, clothing, headgear, "
    "pose, face, or identity from the first two style references. Use the third image "
    "as a clothing and armor "
    "reference - match the garment silhouette and the major design elements visible "
    "there, but render everything as the same carved stone material as the first two "
    "style references. KEEP ORNAMENTATION MINIMAL - render only the carved features "
    "explicitly described and what is clearly visible in the third reference image. "
    "Do NOT invent additional decorative bands, beaded edgings, relief motifs, or "
    "ornamental borders that are not described or visible in the reference. If a "
    "surface is not described as decorated, render it as plain marble. Every part of "
    "the bust - face, hair, helmet, armor, drapery, wreath, accessories - is carved "
    "from the same single block of marble in the same surface color; nothing is "
    "rendered as bronze, fabric, leaves, or any other material color. FOLLOW THE "
    "HEADGEAR DESCRIPTION EXACTLY: if the description says the head is bare with no "
    "headgear, render the head bare even if the third reference image shows a helmet, "
    "crown, or wreath - the third image is only a guide for garment, armor, and "
    "ornamental detail, not for what sits on the head. Never stack or layer head "
    "coverings. The chest and shoulders are ALWAYS fully covered by the specified "
    "armor or drapery - render an opaque continuous surface from the neckline down to "
    "the broken bottom edge of the bust. The torso must read as a clothed or armored "
    "surface, NEVER as bare skin. If the subject is a child, the chest is completely "
    "flat and unsexed under all armor or drapery. ABSOLUTELY NO BARE CHEST. No "
    "exposed skin on the chest, ever. No partially open drapery revealing the chest. "
    "No one-shoulder chitons that leave the opposite chest bare. No drapery slipping "
    "off to expose skin. ABSOLUTELY NO NIPPLES anywhere, ever. NO visible navels, "
    "abdominal six-pack definition, pectoral muscle separation, or other anatomical "
    "body landmarks anywhere on the bust. No breasts, breast cups, cleavage, mammary "
    "forms, or chest mounds; female, neutral, and child subjects still have a fully "
    "covered non-anatomical marble garment or armor surface from neckline to broken "
    "edge. If the outfit is described as a cuirass or breastplate, render it as a "
    "single smooth carved armor surface - a continuous plate, not muscled body "
    "anatomy and not a bare torso. The bust is composed "
    "CHEST-UP - visible from the upper chest to the top of the head, not from the "
    "waist or torso. HAIR BELONGS TO THE SUBJECT, NOT THE CLOTHING REFERENCE: render "
    "the subject's hair length, texture, and style exactly as the subject caption "
    "describes it. If the third reference image shows long flowing hair, a topknot, "
    "a bun, a ponytail, or any other hairstyle, ignore it; the third image is only a "
    "guide for headgear and garment shape, never for what hair the subject has."
)

AMBER_PROSE = (
    "Tiny glowing amber embers drift through the air around the bust in an irregular, "
    "scattered pattern, a few sparks close to the marble and others floating further "
    "out into the dark background, with bright molten lava seeping from the broken "
    "edge of the bust itself. The background is empty dark space - NO loose rocks, "
    "rubble, broken chunks, or debris scattered around or below the bust."
)

EYE_CONTRACT = (
    "Match the fourth image eye state. If the fourth image shows open eyes, render "
    "open blank marble eyes with carved eyelids and smooth unpainted eye surfaces. "
    "Keep the eyes statue-like and colorless: absolutely no iris, pupil, catchlight, "
    "colored eye, or painted eye detail. Do not close the eyes unless the fourth "
    "image clearly shows the subject's eyes closed."
)

POSE_VARIANTS = (
    "The body is angled in three-quarter view turned to the subject's left. The head "
    "is turned to the subject's right and tilted gently toward one shoulder; the pose "
    "is not frontal and the gaze angles slightly past the camera. The bust has no "
    "arms visible, with a clean bust termination at mid-chest.",
    "The body is angled in three-quarter view turned to the subject's left. The head "
    "is also turned slightly to the subject's left, held level and upright; the pose "
    "is not frontal and the far shoulder recedes visibly. The bust has no arms "
    "visible, with a clean bust termination at mid-chest.",
    "The body is angled in three-quarter view turned to the subject's right. The head "
    "is turned to the subject's left, chin lifted slightly upward in a dignified pose; "
    "the pose is not frontal and one cheek plane is more prominent than the other. "
    "The bust has one arm only faintly suggested at the side of the chest, with the "
    "hand not prominently detailed.",
)

PROMPT_PROFILES = {
    "CSDIONYSIAN_REVELER": PromptProfile(
        contract_label="Dionysian profile contract",
        positive_contract=(
            "No helmet, cap, brow band, cheek guards, or metal warrior headgear. "
            "The wreath is mandatory and must be visibly carved around the head as "
            "grape leaves, ivy leaves, flowers, small fruit clusters, or a clearly "
            "Greek Dionysian vegetal crown. A himation or chiton-himation combination "
            "must cover the shoulders and upper chest in soft diagonal folds."
        ),
        reference_contract=(
            "Use the third image to decide the wreath silhouette and drapery direction; "
            "copy only the large readable garment and wreath forms, not the reference "
            "face, pose, skin exposure, or hairstyle."
        ),
        negative_contract=(
            "Reject warrior cues in this class: no helmet, no cheek guards, no cuirass, "
            "no breastplate, no shield, and no robe-only torso that loses the wreath."
        ),
    ),
    "CSARMORED_GENERAL": PromptProfile(
        contract_label="Armored general profile contract",
        positive_contract=(
            "The head is entirely bare with no helmet, wreath, crown, or head covering. "
            "The final bust must read as armored, not as a draped scholar; a cuirass "
            "plate must be clearly visible across the upper chest. Render the torso "
            "as a Greek-style breastplate or cuirass plate, not as robe-only drapery, "
            "with a high covered neckline and optional cloak or shoulder protection "
            "only when supported by the reference. The armor reference is binding: "
            "preserve the third image's cuirass silhouette, shoulder straps, pauldrons, "
            "neckline, and any single dominant central emblem when visible."
        ),
        reference_contract=(
            "Use the third image to preserve the main armor silhouette, shoulder line, "
            "chest plate shape, and any single large central emblem if clearly visible. "
            "If the third image is torso-only armor, place that armor on the bust while "
            "keeping the subject face from the fourth image."
        ),
        negative_contract=(
            "Do not turn the armor into cloth folds, a bare anatomical torso, a Roman "
            "muscle cuirass with nipples, not a breast-shaped plate, and not an ornate fantasy "
            "breastplate. Do not replace the cuirass with a plain shirt, smooth bodice, "
            "bare chest, generic tunic, or unarmored torso."
        ),
    ),
    "CSDRAPED_SCHOLAR": PromptProfile(
        contract_label="Draped scholar profile contract",
        positive_contract=(
            "The head is entirely bare with no helmet, wreath, crown, or head covering. "
            "A chiton and heavy himation wrap the torso, crossing the chest and both "
            "shoulders in deep classical folds. The bust must read as a Greek "
            "philosopher or scholar through layered cloth, calm plain surfaces, and "
            "fully covered upper torso."
        ),
        reference_contract=(
            "Use the third image to choose the direction and depth of the drapery folds "
            "and the high covered neckline; keep the output bust chest-up."
        ),
        negative_contract=(
            "No armor, cuirass, helmet, crown, or wreath. No headless torso crop, no "
            "bare shoulder, no open drapery exposing the chest, and no decorative trim "
            "unless the third reference visibly requires it."
        ),
    ),
    "CSHELMETED_HERO_WARRIOR": PromptProfile(
        contract_label="Helmeted hero-warrior profile contract",
        positive_contract=(
            "A Greek-style helmet is mandatory and must sit flush on the subject's "
            "head. Match the major helmet silhouette visible in the third reference: "
            "dome, brow line, cheek guards, neck guard, crest, or plume only when "
            "clearly present. The helmet, face, hair, and garment are all one carved "
            "marble material. The chest and shoulders are covered by a cloak, cuirass, "
            "aegis, or draped shoulder garment; the helmeted class must never become "
            "a bare chest or bare athletic torso."
        ),
        reference_contract=(
            "Use the third image to transfer the helmet shape onto the subject while "
            "preserving the subject's face and only the visible hair not covered by "
            "the helmet."
        ),
        negative_contract=(
            "The helmet must be worn on the head, not floating, held, displayed beside "
            "the head, or copied as a separate object. Use a smooth low helmet when "
            "the third reference shows a smooth low helmet; no crest or plume unless "
            "the third reference clearly shows one. Do not add extra crests, horns, "
            "wings, fantasy ornaments, or a bare uncovered warrior head."
        ),
    ),
}


def parse_subject_id(subject_id: str, path: Path | None = None) -> Subject:
    try:
        ethnicity, age, gender, replicate_text = subject_id.rsplit("_", 3)
        replicate = int(replicate_text)
    except ValueError as exc:
        raise ValueError(f"Invalid subject id: {subject_id!r}") from exc
    return Subject(
        id=subject_id,
        path=path or Path(f"{subject_id}.png"),
        ethnicity=ethnicity,
        age=age,
        gender=gender,
        replicate=replicate,
    )


def build_job_plan(config: JobPlanConfig) -> list[SyntheticJob]:
    subjects = _load_subjects(config.subjects_dir)
    references_by_archetype = _load_wearable_references(config.wearable_manifest)
    jobs_by_archetype: dict[str, list[SyntheticJob]] = {}
    next_index = 0

    for archetype_index, archetype in enumerate(ARCHETYPE_ORDER):
        references = references_by_archetype[archetype]
        selected_subjects = _select_subjects(subjects, config.per_archetype, archetype_index)
        jobs_by_archetype[archetype] = []
        for slot, subject in enumerate(selected_subjects):
            reference = references[slot % len(references)]
            jobs_by_archetype[archetype].append(
                SyntheticJob(
                    index=next_index,
                    archetype=archetype,
                    archetype_slot=slot,
                    subject=subject,
                    reference=reference,
                    golden_examples=config.golden_examples,
                    captions_dir=config.captions_dir,
                )
            )
            next_index += 1

    ordered: list[SyntheticJob] = []
    for slot in range(config.per_archetype):
        for archetype in ARCHETYPE_ORDER:
            ordered.append(jobs_by_archetype[archetype][slot])
    return [replace(job, index=i) for i, job in enumerate(ordered)]


def render_jobs(jobs: list[SyntheticJob], *, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "jobs.jsonl"
    rendered_dirs: list[Path] = []
    rows: list[dict[str, Any]] = []
    for job in jobs:
        job_dir = output_dir / job.id
        job_dir.mkdir(parents=True, exist_ok=True)
        style_paths = _copy_golden_examples(job, job_dir)
        statue_reference = _copy_as(job.reference.path, job_dir / f"statue_reference{job.reference.path.suffix}")
        control = _copy_as(job.subject.path, job_dir / f"control{job.subject.path.suffix}")
        prompt = render_prompt(job)
        (job_dir / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")
        metadata = _metadata_for(job, job_dir, style_paths, statue_reference, control)
        (job_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        rows.append(metadata)
        rendered_dirs.append(job_dir)

    manifest_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    return rendered_dirs


def render_prompt(job: SyntheticJob) -> str:
    subject_caption = _subject_caption(job)
    outfit = PROMPT_PROFILES[job.archetype].render(job.reference.wearable_cue)
    pose = POSE_VARIANTS[job.index % len(POSE_VARIANTS)]
    return (
        f"{PREAMBLE} Create a marble bust sculpture of {subject_caption} that resembles "
        f"the person in the fourth image (the subject portrait), rendered as a classical "
        f"Greek statue carved from exactly the same marble material and surface color as "
        f"the references (do not brighten or whiten it), in the Archaic/Classical Greek "
        f"sculptural tradition. The marble bust is colorless; disregard any color or mood "
        f"implied by the subject portrait - render skin, hair, and eyes entirely as "
        f"uncolored carved marble, not as a painted portrait. The face, hair, and torso "
        f"surfaces are smooth and intact. NO surface cracks, hairline fractures, "
        f"scratches, or vein-like lines across the face, neck, hair, armor, or drapery; "
        f"any damage exists only as missing three-dimensional chunks chipped "
        f"away at the broken bottom edge. The skin is rendered as calm planes of marble "
        f"without wrinkles, freckles, pores, or photographic skin texture. The hair is "
        f"carved as flowing ribbon-like locks and soft grouped strands rather than "
        f"individually etched hairs. The face wears a serene calm expression, the lips "
        f"softly closed and meeting in an even line with a faint archaic smile, the brow "
        f"relaxed and untroubled. {outfit} {pose} {AMBER_PROSE} Render a single bust, one "
        f"figure only, centered in the frame - no duplicates, no side-by-side views, no "
        f"multiple angles. The bottom of the bust terminates in a jagged, irregular "
        f"broken edge with missing three-dimensional chunks - NOT cut flat - and "
        f"glowing orange-red embers and lava seep from that broken edge. No loose rocks "
        f"or debris below. {EYE_CONTRACT}"
    )


def generate_jobs(
    jobs: list[SyntheticJob],
    *,
    output_dir: Path,
    env_path: Path,
    model: str = DEFAULT_IMAGE_MODEL,
    aspect_ratio: str = "9:16",
    force: bool = False,
) -> list[Path]:
    from google import genai
    from google.genai import types

    api_key = load_api_key_from_env(env_path)
    if not api_key:
        raise RuntimeError(
            "No Gemini API key found. Set GEMINI_API_KEY, GOOGLE_API_KEY, "
            "GOOGLE_AI_STUDIO_KEY, or google_ai_stuidio_key in .env."
        )
    client = genai.Client(api_key=api_key)
    generated: list[Path] = []
    for job in jobs:
        job_dir = output_dir / job.id
        output_path = job_dir / "generated.png"
        if output_path.exists() and not force:
            generated.append(output_path)
            continue
        prompt = (job_dir / "prompt.txt").read_text(encoding="utf-8")
        contents = [
            Image.open(job_dir / "style_1.png"),
            Image.open(job_dir / "style_2.png"),
            Image.open(next(job_dir.glob("statue_reference.*"))),
            Image.open(next(job_dir.glob("control.*"))),
            prompt,
        ]
        response = call_with_retries(
            lambda: client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                    image_config=types.ImageConfig(aspect_ratio=aspect_ratio),
                ),
            ),
            label=f"generate {job.id}",
        )
        image = _first_response_image(response)
        if image is None:
            raise RuntimeError(f"No image returned for {job.id}")
        image.save(output_path)
        generated.append(output_path)
    return generated


def make_review_contact_sheet(
    job_dirs: list[Path],
    output_path: Path,
    *,
    thumb_size: tuple[int, int] = (140, 140),
) -> Path:
    columns = (
        ("style_1", "style_1.*"),
        ("style_2", "style_2.*"),
        ("statue_ref", "statue_reference.*"),
        ("control", "control.*"),
        ("generated", "generated.png"),
    )
    label_height = 36
    row_header_height = 24
    gap = 10
    cell_width = thumb_size[0]
    row_height = row_header_height + thumb_size[1] + label_height + gap
    width = gap + len(columns) * (cell_width + gap)
    height = gap + max(1, len(job_dirs)) * row_height
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)

    y = gap
    for job_dir in job_dirs:
        draw.text((gap, y), job_dir.name, fill=(0, 0, 0))
        image_y = y + row_header_height
        for column_index, (label, pattern) in enumerate(columns):
            x = gap + column_index * (cell_width + gap)
            image_path = _first_matching_file(job_dir, pattern)
            if image_path is not None:
                thumb = _thumbnail_for_contact_sheet(image_path, thumb_size)
                sheet.paste(thumb, (x, image_y))
            else:
                draw.rectangle(
                    (x, image_y, x + thumb_size[0], image_y + thumb_size[1]),
                    outline=(180, 180, 180),
                    fill=(245, 245, 245),
                )
                draw.text((x + 8, image_y + 58), "missing", fill=(120, 120, 120))
            draw.text((x, image_y + thumb_size[1] + 4), label, fill=(0, 0, 0))
        y += row_height

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)
    return output_path


def ensure_captions(
    jobs: list[SyntheticJob],
    *,
    captions_dir: Path,
    env_path: Path,
    model: str = DEFAULT_CAPTION_MODEL,
    strict: bool = False,
) -> None:
    from google import genai

    api_key = load_api_key_from_env(env_path)
    if not api_key:
        raise RuntimeError(
            "No Gemini API key found for captioning. Set GEMINI_API_KEY, GOOGLE_API_KEY, "
            "GOOGLE_AI_STUDIO_KEY, or google_ai_stuidio_key in .env."
        )
    captions_dir.mkdir(parents=True, exist_ok=True)
    client = genai.Client(api_key=api_key)
    seen: set[str] = set()
    for job in jobs:
        if job.subject.id in seen:
            continue
        seen.add(job.subject.id)
        caption_path = captions_dir / f"{job.subject.id}.txt"
        if caption_path.exists():
            continue
        try:
            response = call_with_retries(
                lambda: client.models.generate_content(
                    model=model,
                    contents=[Image.open(job.subject.path), CAPTION_INSTRUCTION],
                ),
                label=f"caption {job.subject.id}",
            )
            caption = (response.text or "").strip()
            if not caption:
                raise RuntimeError(f"Caption model returned empty text for {job.subject.id}")
        except Exception as exc:
            if strict:
                raise
            caption = _fallback_caption_for_subject(job.subject)
            print(
                f"caption {job.subject.id}: using fallback caption after error: {exc}",
                flush=True,
            )
        caption_path.write_text(caption + "\n", encoding="utf-8")


def load_api_key_from_env(env_path: Path | None = None) -> str | None:
    keys = (
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "GOOGLE_AI_STUDIO_KEY",
        "google_ai_studio_key",
        "google_ai_stuidio_key",
    )
    for key in keys:
        value = os.environ.get(key)
        if value:
            return value
    values: dict[str, str] = {}
    if env_path is not None and env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    for key in keys:
        if values.get(key):
            return values[key]
    return None


def call_with_retries(
    func,
    *,
    label: str,
    max_retries: int = DEFAULT_MAX_RETRIES,
    delay_seconds: int = DEFAULT_RETRY_DELAY_SECONDS,
):
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return func()
        except Exception as exc:
            last_error = exc
            if not _is_transient_error(exc) or attempt == max_retries:
                raise
            print(
                f"{label}: transient API error, retrying in {delay_seconds}s "
                f"(attempt {attempt}/{max_retries - 1})",
                flush=True,
            )
            time.sleep(delay_seconds)
    raise last_error  # type: ignore[misc]


def _is_transient_error(exc: Exception) -> bool:
    if isinstance(exc, TransientApiError):
        return True
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "503",
            "502",
            "bad gateway",
            "server error",
            "unavailable",
            "high demand",
            "overload",
            "timeout",
            "deadline",
            "temporarily",
        )
    )


def _load_subjects(subjects_dir: Path) -> dict[str, Subject]:
    if not subjects_dir.exists():
        raise FileNotFoundError(f"subjects directory not found: {subjects_dir}")
    subjects: dict[str, Subject] = {}
    for path in sorted(subjects_dir.glob("*.png")):
        subject = parse_subject_id(path.stem, path)
        subjects[subject.id] = subject
    if not subjects:
        raise ValueError(f"no subject PNG files found in {subjects_dir}")
    return subjects


def _load_wearable_references(manifest_path: Path) -> dict[str, list[WearableReference]]:
    rows = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    grouped: dict[str, list[WearableReference]] = {archetype: [] for archetype in ARCHETYPE_ORDER}
    for row in rows:
        if row.get("assessment_pass") is not True:
            continue
        archetype = row["archetype"]
        if archetype not in grouped:
            continue
        path = Path(row["local_path"])
        if not path.exists():
            raise FileNotFoundError(path)
        grouped[archetype].append(
            WearableReference(
                archetype=archetype,
                rank=int(row["rank"]),
                path=path,
                filename=row["filename"],
                wearable_cue=row.get("wearable_cue") or row["filename"],
                source_url=row.get("source_url"),
            )
        )
    for archetype, refs in grouped.items():
        refs.sort(key=lambda ref: ref.rank)
        if len(refs) < MIN_ACTIVE_REFERENCES_PER_ARCHETYPE:
            raise ValueError(
                f"expected at least {MIN_ACTIVE_REFERENCES_PER_ARCHETYPE} references "
                f"for {archetype}, found {len(refs)}"
            )
    return grouped


def _select_subjects(
    subjects: dict[str, Subject],
    count: int,
    archetype_offset: int,
) -> list[Subject]:
    selected: list[Subject] = []
    seen: set[str] = set()
    i = 0
    while len(selected) < count:
        ethnicity = ETHNICITIES[(i + archetype_offset) % len(ETHNICITIES)]
        age = AGES[((i // len(GENDERS)) + archetype_offset) % len(AGES)]
        gender = GENDERS[(i + archetype_offset) % len(GENDERS)]
        replicate = ((i + archetype_offset) % 3) + 1
        subject_id = f"{ethnicity}_{age}_{gender}_{replicate}"
        subject = subjects.get(subject_id)
        if subject is not None and subject.id not in seen:
            selected.append(subject)
            seen.add(subject.id)
        i += 1
        if i > len(subjects) * 2 and len(selected) < count:
            for subject in sorted(subjects.values(), key=lambda item: item.id):
                if subject.id not in seen:
                    selected.append(subject)
                    seen.add(subject.id)
                    if len(selected) == count:
                        break
            break
    if len(selected) < count:
        raise ValueError(f"not enough unique subjects for {count} jobs")
    return selected


def _subject_caption(job: SyntheticJob) -> str:
    if job.captions_dir is not None:
        caption_path = job.captions_dir / f"{job.subject.id}.txt"
        if caption_path.exists():
            return _sanitize_subject_caption(
                _strip_traits_header(caption_path.read_text(encoding="utf-8"))
            )
    return _strip_traits_header(_fallback_caption_for_subject(job.subject))


def _fallback_caption_for_subject(subject: Subject) -> str:
    age = "child" if subject.age == "child" else "adult"
    return (
        f"[gender: {subject.gender}; age: {age}]\n"
        f"{subject.demographic_phrase}, face structure and hairstyle matching the fourth image"
    )


def _strip_traits_header(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines and lines[0].startswith("[gender:"):
        return " ".join(lines[1:]).strip()
    return " ".join(lines).strip()


def _sanitize_subject_caption(caption: str) -> str:
    sanitized = caption
    accessory_patterns = (
        r"\s+with a headband",
        r"\s+with headband",
        r"\s+with a hair band",
        r"\s+with hair band",
        r"\s+with glasses",
        r"\s+wearing glasses",
        r"\s+with eyeglasses",
        r"\s+wearing eyeglasses",
        r"\s+with headphones",
        r"\s+wearing headphones",
        r"\s+with earbuds",
        r"\s+wearing earbuds",
    )
    for pattern in accessory_patterns:
        sanitized = re.sub(pattern, "", sanitized, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", sanitized).strip(" ,")


def _copy_golden_examples(job: SyntheticJob, job_dir: Path) -> tuple[Path, Path]:
    copied: list[Path] = []
    for index, path in enumerate(job.golden_examples, 1):
        copied.append(_copy_as(path, job_dir / f"style_{index}{path.suffix}"))
    return copied[0], copied[1]


def _copy_as(source: Path, destination: Path) -> Path:
    if not source.exists():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def _first_matching_file(directory: Path, pattern: str) -> Path | None:
    return next(iter(sorted(directory.glob(pattern))), None)


def _thumbnail_for_contact_sheet(path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as image:
        thumb = image.convert("RGB")
        thumb.thumbnail(size)
        canvas = Image.new("RGB", size, "white")
        x = (size[0] - thumb.width) // 2
        y = (size[1] - thumb.height) // 2
        canvas.paste(thumb, (x, y))
        return canvas


def _metadata_for(
    job: SyntheticJob,
    job_dir: Path,
    style_paths: tuple[Path, Path],
    statue_reference: Path,
    control: Path,
) -> dict[str, Any]:
    return {
        "id": job.id,
        "job_dir": str(job_dir),
        "archetype": job.archetype,
        "archetype_slot": job.archetype_slot,
        "subject_id": job.subject.id,
        "subject": {
            "ethnicity": job.subject.ethnicity,
            "age": job.subject.age,
            "gender": job.subject.gender,
            "replicate": job.subject.replicate,
            "source_path": str(job.subject.path),
        },
        "style_references": [str(path) for path in style_paths],
        "statue_reference": str(statue_reference),
        "control": str(control),
        "prompt": str(job_dir / "prompt.txt"),
        "reference_rank": job.reference.rank,
        "reference_filename": job.reference.filename,
        "reference_source_url": job.reference.source_url,
        "wearable_cue": job.reference.wearable_cue,
        "model": DEFAULT_IMAGE_MODEL,
        "rendered_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def _first_response_image(response: Any) -> Any | None:
    parts = getattr(response, "parts", None)
    if not parts:
        return None
    for part in parts:
        if getattr(part, "inline_data", None):
            return part.as_image()
    return None

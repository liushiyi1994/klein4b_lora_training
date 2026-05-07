from __future__ import annotations

import json
import sys
import types
from pathlib import Path

from PIL import Image

import klein4b.wearable_synthetic_jobs as wearable_synthetic_jobs
from klein4b.wearable_synthetic_jobs import (
    DEFAULT_GOLDEN_EXAMPLES,
    POSE_VARIANTS,
    JobPlanConfig,
    TransientApiError,
    build_job_plan,
    call_with_retries,
    ensure_captions,
    load_api_key_from_env,
    make_review_contact_sheet,
    parse_subject_id,
    render_jobs,
)


def write_png(path: Path, color: str = "red") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 24), color=color).save(path)


def write_subjects(subjects_dir: Path) -> None:
    ethnicities = [
        "african",
        "eastasian",
        "european",
        "hispaniclatino",
        "middleeastern",
        "southasian",
    ]
    ages = ["youngadult", "middleaged", "elderly", "child"]
    genders = ["male", "female", "neutral"]
    for ethnicity in ethnicities:
        for age in ages:
            for gender in genders:
                for replicate in range(1, 4):
                    write_png(subjects_dir / f"{ethnicity}_{age}_{gender}_{replicate}.png")


def write_wearable_manifest(pool_dir: Path) -> Path:
    rows = []
    for archetype in [
        "CSDIONYSIAN_REVELER",
        "CSARMORED_GENERAL",
        "CSDRAPED_SCHOLAR",
        "CSHELMETED_HERO_WARRIOR",
    ]:
        for rank in range(1, 7):
            ref = pool_dir / archetype / f"{rank:02d}_{archetype.lower()}.png"
            write_png(ref, color="blue")
            rows.append(
                {
                    "archetype": archetype,
                    "rank": rank,
                    "filename": ref.name,
                    "local_path": str(ref),
                    "wearable_cue": f"{archetype} wearable cue {rank}",
                    "source_url": f"https://example.test/{archetype}/{rank}",
                    "assessment_pass": True,
                }
            )
    manifest = pool_dir / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return manifest


def test_parse_subject_id_accepts_multiword_ethnicity() -> None:
    subject = parse_subject_id("hispaniclatino_youngadult_female_2")

    assert subject.ethnicity == "hispaniclatino"
    assert subject.age == "youngadult"
    assert subject.gender == "female"
    assert subject.replicate == 2


def test_build_job_plan_creates_15_unique_people_per_archetype(tmp_path: Path) -> None:
    subjects_dir = tmp_path / "subjects"
    write_subjects(subjects_dir)
    manifest = write_wearable_manifest(tmp_path / "wearable")

    jobs = build_job_plan(
        JobPlanConfig(
            subjects_dir=subjects_dir,
            wearable_manifest=manifest,
            output_dir=tmp_path / "out",
            per_archetype=15,
        )
    )

    assert len(jobs) == 60
    assert [job.archetype for job in jobs[:4]] == [
        "CSDIONYSIAN_REVELER",
        "CSARMORED_GENERAL",
        "CSDRAPED_SCHOLAR",
        "CSHELMETED_HERO_WARRIOR",
    ]

    by_archetype: dict[str, list[str]] = {}
    for job in jobs:
        by_archetype.setdefault(job.archetype, []).append(job.subject.id)

    assert set(by_archetype) == {
        "CSDIONYSIAN_REVELER",
        "CSARMORED_GENERAL",
        "CSDRAPED_SCHOLAR",
        "CSHELMETED_HERO_WARRIOR",
    }
    for subject_ids in by_archetype.values():
        assert len(subject_ids) == 15
        assert len(set(subject_ids)) == 15

    dionysian_refs = [
        job.reference.rank for job in jobs if job.archetype == "CSDIONYSIAN_REVELER"
    ]
    assert dionysian_refs == [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3]


def test_build_job_plan_skips_rejected_references_and_cycles_remaining(tmp_path: Path) -> None:
    subjects_dir = tmp_path / "subjects"
    write_subjects(subjects_dir)
    manifest = write_wearable_manifest(tmp_path / "wearable")
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        if row["archetype"] == "CSHELMETED_HERO_WARRIOR" and row["rank"] == 5:
            row["assessment_pass"] = False
            row["rejection_reason"] = "review rejected weak helmet reference"
        if row["archetype"] == "CSARMORED_GENERAL" and row["rank"] == 2:
            row["assessment_pass"] = False
            row["rejection_reason"] = "review rejected anatomical cuirass reference"
        if row["archetype"] == "CSARMORED_GENERAL" and row["rank"] == 5:
            row["assessment_pass"] = False
            row["rejection_reason"] = "review rejected object-only bell cuirass reference"
    manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    jobs = build_job_plan(
        JobPlanConfig(
            subjects_dir=subjects_dir,
            wearable_manifest=manifest,
            output_dir=tmp_path / "out",
            per_archetype=15,
        )
    )

    helmeted_refs = [
        job.reference.rank for job in jobs if job.archetype == "CSHELMETED_HERO_WARRIOR"
    ]
    assert helmeted_refs == [1, 2, 3, 4, 6, 1, 2, 3, 4, 6, 1, 2, 3, 4, 6]
    armored_refs = [job.reference.rank for job in jobs if job.archetype == "CSARMORED_GENERAL"]
    assert armored_refs == [1, 3, 4, 6, 1, 3, 4, 6, 1, 3, 4, 6, 1, 3, 4]


def test_render_jobs_writes_review_assets_and_manifest(tmp_path: Path) -> None:
    subjects_dir = tmp_path / "subjects"
    write_subjects(subjects_dir)
    manifest = write_wearable_manifest(tmp_path / "wearable")
    golden = [tmp_path / "golden1.png", tmp_path / "golden2.png"]
    for path in golden:
        write_png(path, color="green")
    output_dir = tmp_path / "out"

    jobs = build_job_plan(
        JobPlanConfig(
            subjects_dir=subjects_dir,
            wearable_manifest=manifest,
            output_dir=output_dir,
            per_archetype=15,
            golden_examples=golden,
        )
    )

    rendered = render_jobs(jobs[:3], output_dir=output_dir)

    assert len(rendered) == 3
    assert (output_dir / "jobs.jsonl").exists()
    rows = [
        json.loads(line)
        for line in (output_dir / "jobs.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 3
    first_job_dir = output_dir / "000_CSDIONYSIAN_REVELER_african_youngadult_male_1"
    assert (first_job_dir / "style_1.png").exists()
    assert (first_job_dir / "style_2.png").exists()
    assert (first_job_dir / "statue_reference.png").exists()
    assert (first_job_dir / "control.png").exists()
    assert (first_job_dir / "prompt.txt").exists()
    assert (first_job_dir / "metadata.json").exists()
    prompt = (first_job_dir / "prompt.txt").read_text(encoding="utf-8")
    assert "use the first two images as style references" in prompt
    assert "Keep the final image as a 9:16 portrait composition" in prompt
    assert "Statue occupies 45 to 55 percent of the full frame height" in prompt
    assert "at least 20 percent empty dark space above the highest point" in prompt
    assert "Never render a straight-on frontal bust" in prompt
    assert "Do not make the marble bust dark gray, black, blue-gray, or underexposed" in prompt
    assert "No breasts, breast cups, cleavage, mammary forms, or chest mounds" in prompt
    assert "NO surface cracks, hairline fractures, scratches, or vein-like lines" in prompt
    assert "Match the fourth image eye state" in prompt
    assert "If the fourth image shows open eyes, render open blank marble eyes" in prompt
    assert "absolutely no iris, pupil, catchlight, colored eye, or painted eye detail" in prompt
    assert "fully closed" not in prompt.lower()
    assert "Do not copy helmets, clothing, headgear, pose, face, or identity from the first two" in prompt
    assert "that resembles the person in the fourth image" in prompt
    assert "CSDIONYSIAN_REVELER wearable cue 1" in prompt
    assert "No helmet, cap, brow band, cheek guards, or metal warrior headgear" in prompt

    armored_prompt = (
        output_dir / "001_CSARMORED_GENERAL_eastasian_middleaged_female_2" / "prompt.txt"
    ).read_text(encoding="utf-8")
    assert "The final bust must read as armored, not as a draped scholar" in armored_prompt
    assert "a cuirass plate must be clearly visible" in armored_prompt
    assert "The armor reference is binding" in armored_prompt
    assert "Do not replace the cuirass with a plain shirt, smooth bodice, bare chest" in armored_prompt
    assert "not a breast-shaped plate" in armored_prompt


def test_default_golden_examples_follow_current_9x16_files() -> None:
    assert [path.name for path in DEFAULT_GOLDEN_EXAMPLES] == ["1.png", "2.png"]


def test_make_review_contact_sheet_writes_generated_review_grid(tmp_path: Path) -> None:
    subjects_dir = tmp_path / "subjects"
    write_subjects(subjects_dir)
    manifest = write_wearable_manifest(tmp_path / "wearable")
    golden = [tmp_path / "golden1.png", tmp_path / "golden2.png"]
    for path in golden:
        write_png(path, color="green")
    output_dir = tmp_path / "out"
    jobs = build_job_plan(
        JobPlanConfig(
            subjects_dir=subjects_dir,
            wearable_manifest=manifest,
            output_dir=output_dir,
            per_archetype=15,
            golden_examples=(golden[0], golden[1]),
        )
    )
    job_dirs = render_jobs(jobs[:4], output_dir=output_dir)
    for index, job_dir in enumerate(job_dirs):
        write_png(job_dir / "generated.png", color="purple" if index % 2 else "yellow")

    sheet_path = make_review_contact_sheet(job_dirs, output_dir / "review_contact_sheet.jpg")

    assert sheet_path == output_dir / "review_contact_sheet.jpg"
    assert sheet_path.exists()
    with Image.open(sheet_path) as image:
        assert image.width >= 700
        assert image.height >= 800


def test_archetype_prompt_profiles_are_specific(tmp_path: Path) -> None:
    subjects_dir = tmp_path / "subjects"
    write_subjects(subjects_dir)
    manifest = write_wearable_manifest(tmp_path / "wearable")
    golden = [tmp_path / "golden1.png", tmp_path / "golden2.png"]
    for path in golden:
        write_png(path, color="green")
    output_dir = tmp_path / "out"

    jobs = build_job_plan(
        JobPlanConfig(
            subjects_dir=subjects_dir,
            wearable_manifest=manifest,
            output_dir=output_dir,
            per_archetype=15,
            golden_examples=(golden[0], golden[1]),
        )
    )

    render_jobs(jobs[:4], output_dir=output_dir)

    prompts = {
        job.archetype: (output_dir / job.id / "prompt.txt").read_text(encoding="utf-8")
        for job in jobs[:4]
    }
    assert "Dionysian profile contract" in prompts["CSDIONYSIAN_REVELER"]
    assert "The wreath is mandatory and must be visibly carved around the head" in prompts[
        "CSDIONYSIAN_REVELER"
    ]
    assert "Armored general profile contract" in prompts["CSARMORED_GENERAL"]
    assert "not as robe-only drapery" in prompts["CSARMORED_GENERAL"]
    assert "Draped scholar profile contract" in prompts["CSDRAPED_SCHOLAR"]
    assert "No armor, cuirass, helmet, crown, or wreath" in prompts["CSDRAPED_SCHOLAR"]
    assert "Helmeted hero-warrior profile contract" in prompts["CSHELMETED_HERO_WARRIOR"]
    assert "The helmet must be worn on the head, not floating, held, displayed beside the head" in prompts[
        "CSHELMETED_HERO_WARRIOR"
    ]
    assert "no crest or plume unless the third reference clearly shows one" in prompts[
        "CSHELMETED_HERO_WARRIOR"
    ]
    assert "covered by a cloak, cuirass, aegis, or draped shoulder garment" in prompts[
        "CSHELMETED_HERO_WARRIOR"
    ]


def test_pose_variants_are_three_quarter_not_frontal() -> None:
    forbidden_phrases = (
        "squared forward",
        "straight toward the viewer",
        "straight-on frontal",
        "front-facing",
    )

    for pose in POSE_VARIANTS:
        lowered = pose.lower()
        assert "three-quarter" in lowered
        assert "not frontal" in lowered
        for phrase in forbidden_phrases:
            assert phrase not in lowered


def test_render_jobs_suppresses_modern_headwear_from_caption(tmp_path: Path) -> None:
    subjects_dir = tmp_path / "subjects"
    write_subjects(subjects_dir)
    manifest = write_wearable_manifest(tmp_path / "wearable")
    golden = [tmp_path / "golden1.png", tmp_path / "golden2.png"]
    for path in golden:
        write_png(path, color="green")
    captions_dir = tmp_path / "captions"
    captions_dir.mkdir()
    (captions_dir / "african_youngadult_male_1.txt").write_text(
        "[gender: male; age: adult]\n"
        "a African male adult, short wavy hair pulled back with a headband, "
        "oval face, smooth cheeks, almond eyes, broad nose, full lips\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"

    jobs = build_job_plan(
        JobPlanConfig(
            subjects_dir=subjects_dir,
            wearable_manifest=manifest,
            output_dir=output_dir,
            per_archetype=15,
            captions_dir=captions_dir,
            golden_examples=(golden[0], golden[1]),
        )
    )

    render_jobs(jobs[:1], output_dir=output_dir)

    prompt = (
        output_dir / "000_CSDIONYSIAN_REVELER_african_youngadult_male_1" / "prompt.txt"
    ).read_text(encoding="utf-8")
    assert "headband" not in prompt.lower()
    assert "short wavy hair pulled back" in prompt


def test_load_api_key_accepts_existing_dotenv_key_name(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    env_path = tmp_path / ".env"
    env_path.write_text("google_ai_stuidio_key=secret-value\n", encoding="utf-8")

    assert load_api_key_from_env(env_path) == "secret-value"


def test_ensure_captions_writes_fallback_when_caption_model_is_busy(
    tmp_path: Path,
    monkeypatch,
) -> None:
    subjects_dir = tmp_path / "subjects"
    write_subjects(subjects_dir)
    manifest = write_wearable_manifest(tmp_path / "wearable")
    jobs = build_job_plan(
        JobPlanConfig(
            subjects_dir=subjects_dir,
            wearable_manifest=manifest,
            output_dir=tmp_path / "out",
            per_archetype=15,
            captions_dir=tmp_path / "captions",
        )
    )
    fake_genai = types.SimpleNamespace(Client=lambda api_key: object())
    monkeypatch.setitem(sys.modules, "google", types.SimpleNamespace(genai=fake_genai))
    monkeypatch.setattr(wearable_synthetic_jobs, "load_api_key_from_env", lambda env_path: "secret")

    def always_busy(*args, **kwargs) -> str:
        raise TransientApiError("503 UNAVAILABLE high demand")

    monkeypatch.setattr(wearable_synthetic_jobs, "call_with_retries", always_busy)

    ensure_captions(
        jobs[:1],
        captions_dir=tmp_path / "captions",
        env_path=tmp_path / ".env",
    )

    caption = (tmp_path / "captions" / f"{jobs[0].subject.id}.txt").read_text(encoding="utf-8")
    assert "[gender: male; age: adult]" in caption
    assert "face structure and hairstyle matching the fourth image" in caption


def test_call_with_retries_retries_transient_503(monkeypatch) -> None:
    monkeypatch.setattr("klein4b.wearable_synthetic_jobs.time.sleep", lambda _: None)
    attempts = {"count": 0}

    def flaky_call() -> str:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise TransientApiError("503 UNAVAILABLE high demand")
        return "ok"

    assert call_with_retries(flaky_call, label="caption", max_retries=2, delay_seconds=0) == "ok"
    assert attempts["count"] == 2


def test_call_with_retries_retries_transient_502_bad_gateway(monkeypatch) -> None:
    monkeypatch.setattr("klein4b.wearable_synthetic_jobs.time.sleep", lambda _: None)
    attempts = {"count": 0}

    def flaky_call() -> str:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("502 Bad Gateway temporary server error")
        return "ok"

    assert call_with_retries(flaky_call, label="generate", max_retries=2, delay_seconds=0) == "ok"
    assert attempts["count"] == 2

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from klein4b.wearable_synthetic_jobs import (  # noqa: E402
    DEFAULT_CAPTION_MODEL,
    DEFAULT_CAPTIONS_DIR,
    DEFAULT_GOLDEN_EXAMPLES,
    DEFAULT_IMAGE_MODEL,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SUBJECTS_DIR,
    DEFAULT_WEARABLE_MANIFEST,
    JobPlanConfig,
    build_job_plan,
    ensure_captions,
    generate_jobs,
    make_review_contact_sheet,
    render_jobs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render and optionally generate wearable-first marble bust synthetic jobs."
    )
    parser.add_argument("--subjects-dir", type=Path, default=DEFAULT_SUBJECTS_DIR)
    parser.add_argument("--captions-dir", type=Path, default=DEFAULT_CAPTIONS_DIR)
    parser.add_argument("--wearable-manifest", type=Path, default=DEFAULT_WEARABLE_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--golden-example",
        type=Path,
        action="append",
        default=None,
        help="Pass exactly two style reference images. Defaults to the golden_example folder.",
    )
    parser.add_argument("--per-archetype", type=int, default=15)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Render/generate only the first N round-robin jobs for review.",
    )
    parser.add_argument("--generate", action="store_true", help="Call Nano Banana Pro after render.")
    parser.add_argument(
        "--caption-missing",
        action="store_true",
        help="Caption selected subjects before rendering prompts.",
    )
    parser.add_argument("--env", type=Path, default=REPO_ROOT / ".env")
    parser.add_argument("--model", default=DEFAULT_IMAGE_MODEL)
    parser.add_argument("--caption-model", default=DEFAULT_CAPTION_MODEL)
    parser.add_argument("--aspect-ratio", default="9:16")
    parser.add_argument(
        "--contact-sheet",
        type=Path,
        default=None,
        help="Review grid path. Defaults to review_contact_sheet.jpg in the output directory.",
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    golden_examples = tuple(args.golden_example or DEFAULT_GOLDEN_EXAMPLES)
    if len(golden_examples) != 2:
        parser.error("--golden-example must be provided exactly twice when overriding defaults")

    config = JobPlanConfig(
        subjects_dir=args.subjects_dir,
        captions_dir=args.captions_dir,
        wearable_manifest=args.wearable_manifest,
        output_dir=args.output_dir,
        per_archetype=args.per_archetype,
        golden_examples=(golden_examples[0], golden_examples[1]),
    )
    jobs = build_job_plan(config)
    selected_jobs = jobs[: args.limit] if args.limit is not None else jobs

    if args.caption_missing or args.generate:
        ensure_captions(
            selected_jobs,
            captions_dir=args.captions_dir,
            env_path=args.env,
            model=args.caption_model,
        )

    rendered = render_jobs(selected_jobs, output_dir=args.output_dir)
    print(f"Rendered {len(rendered)} job(s) under {args.output_dir}")
    contact_sheet = args.contact_sheet or args.output_dir / "review_contact_sheet.jpg"

    if not args.generate:
        sheet_path = make_review_contact_sheet(rendered, contact_sheet)
        print(f"Review contact sheet written to {sheet_path}")
        print("Generation skipped. Pass --generate to call Nano Banana Pro.")
        return 0

    generated = generate_jobs(
        selected_jobs,
        output_dir=args.output_dir,
        env_path=args.env,
        model=args.model,
        aspect_ratio=args.aspect_ratio,
        force=args.force,
    )
    for path in generated:
        print(f"Generated {path}")
    sheet_path = make_review_contact_sheet(rendered, contact_sheet)
    print(f"Review contact sheet written to {sheet_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

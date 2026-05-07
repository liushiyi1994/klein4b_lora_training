from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from klein4b.fal_training import (  # noqa: E402
    OUTPUT_LORA_FORMATS,
    FalPackageOptions,
    FalTrainingPackageError,
    build_fal_training_package,
)
from klein4b.paths import data_dir, outputs_dir  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare an offline fal.ai FLUX.2 Klein 9B edit trainer zip, manifest, "
            "and request JSON. This does not upload files or submit training."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=data_dir() / "marble-bust-data" / "v7_30",
        help="Dataset root containing dataset/ target images and control/ reference images.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=outputs_dir() / "fal_training_packages",
        help="Directory where the package zip, manifest, and request JSON are written.",
    )
    parser.add_argument(
        "--package-name",
        default="marble_v7_30_klein9b_fal",
        help="Output file stem for the zip, manifest, and request JSON.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=3600,
        help="fal training steps. Must be 100 to 10000 in increments of 100.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.00004,
        help="fal learning_rate request value.",
    )
    parser.add_argument(
        "--image-data-url",
        default="",
        help="Future uploaded zip URL to place in request JSON. Leave blank for offline prep.",
    )
    parser.add_argument(
        "--output-lora-format",
        choices=OUTPUT_LORA_FORMATS,
        default="fal",
        help="fal output_lora_format request value.",
    )
    parser.add_argument(
        "--default-caption",
        default=None,
        help="Fallback edit instruction for samples missing per-pair ROOT.txt captions.",
    )
    parser.add_argument(
        "--allow-missing-captions",
        action="store_true",
        help="Allow missing per-pair captions only when --default-caption is also set.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = build_fal_training_package(
            FalPackageOptions(
                dataset_root=args.dataset_root,
                output_dir=args.output_root,
                package_name=args.package_name,
                steps=args.steps,
                learning_rate=args.learning_rate,
                output_lora_format=args.output_lora_format,
                image_data_url=args.image_data_url,
                default_caption=args.default_caption,
                allow_missing_captions=args.allow_missing_captions,
            )
        )
    except FalTrainingPackageError as error:
        parser.exit(status=2, message=f"error: {error}\n")

    print(f"zip: {result.zip_path}")
    print(f"manifest: {result.manifest_path}")
    print(f"request json: {result.request_json_path}")
    print(f"sample count: {result.sample_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

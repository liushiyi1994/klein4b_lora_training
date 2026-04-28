from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from klein4b.marble_dataset import build_marble_v3_dataset  # noqa: E402
from klein4b.paths import data_dir  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create marble-bust-data v3 with matte localized-lava captions."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=data_dir() / "marble-bust-data" / "v1",
        help="Source marble dataset version directory.",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=data_dir() / "marble-bust-data" / "v3",
        help="Target marble dataset version directory.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = build_marble_v3_dataset(args.source, args.target)
    print(f"Wrote {report.target_dir}")
    print(f"Captions rewritten: {report.caption_count}")
    print(f"Images copied: {report.image_count}")


if __name__ == "__main__":
    main()

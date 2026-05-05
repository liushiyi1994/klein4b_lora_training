from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from klein4b.marble_v6_dataset import build_v6_199_dataset  # noqa: E402
from klein4b.paths import data_dir  # noqa: E402


def default_source_dir() -> Path:
    env_value = os.environ.get("KLEIN4B_V6_SOURCE_DIR")
    if env_value:
        return Path(env_value)
    return REPO_ROOT.parent / "charliestale-ml" / "data-synthesis"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare the v6 199-row marble bust control dataset with rewritten captions."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=default_source_dir(),
        help=(
            "Path to charliestale-ml/data-synthesis containing editing-dataset-control "
            "and editing-dataset/manifest.json."
        ),
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=data_dir() / "marble-bust-data" / "v6_199",
        help="Target v6_199 dataset directory containing archetypes.json.",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=199,
        help="Expected manifest row count. Keep 199 for the production v6 dataset.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_v6_199_dataset(
        source_dir=args.source,
        target_dir=args.target,
        expected_count=args.expected_count,
    )
    print(f"Wrote {report.row_count} rows to {args.target}")
    print(f"Archetype groups: {len(report.archetype_counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

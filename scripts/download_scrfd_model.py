from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from klein4b.scrfd_model import (  # noqa: E402
    DEFAULT_SCRFD_MODEL_URL,
    download_scrfd_model,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download the default SCRFD ONNX detector for reference preprocessing."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination ONNX path. Defaults to data/models/scrfd/scrfd_person_2.5g.onnx.",
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_SCRFD_MODEL_URL,
        help="SCRFD ONNX download URL.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload even if the destination file already exists.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = download_scrfd_model(
        output_path=args.output,
        url=args.url,
        force=args.force,
    )
    print(f"SCRFD model: {result.path}")
    print(f"Manifest: {result.manifest_path}")
    print(f"Bytes: {result.bytes}")
    print(f"SHA256: {result.sha256}")
    print(f"export KLEIN4B_SCRFD_MODEL_PATH={shlex.quote(str(result.path))}")


if __name__ == "__main__":
    main()

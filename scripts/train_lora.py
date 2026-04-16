from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from klein4b.paths import data_dir, outputs_dir  # noqa: E402
from klein4b.training import (  # noqa: E402
    build_training_command,
    default_training_template_path,
    render_training_config,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render a versioned AI Toolkit config and launch local LoRA training."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=default_training_template_path(),
        help="Path to the checked-in AI Toolkit YAML template.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=data_dir() / "demo_ffhq_makeup" / "train",
        help="Folder containing captioned training images.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=outputs_dir() / "runs",
        help="Root folder where timestamped run directories are created.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_dir = args.output_root / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    config_path = run_dir / "train_config.yaml"
    config_path.write_text(
        render_training_config(
            template_path=args.config,
            dataset_dir=args.dataset_dir,
            output_dir=run_dir,
        ),
        encoding="utf-8",
    )

    command = build_training_command(REPO_ROOT / "vendor" / "ai-toolkit", config_path)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

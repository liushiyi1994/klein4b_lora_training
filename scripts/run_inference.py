from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from klein4b.inference import build_demo_prompt, run_local_inference  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lora", type=Path, default=None)
    parser.add_argument("--lora-scale", type=float, default=1.0)
    args = parser.parse_args()

    run_local_inference(
        reference_path=args.reference,
        prompt=build_demo_prompt("K4BMAKEUP03"),
        output_path=args.output,
        lora_path=args.lora,
        lora_scale=args.lora_scale,
    )


if __name__ == "__main__":
    main()

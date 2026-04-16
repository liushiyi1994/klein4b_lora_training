from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from klein4b.eval_metrics import ssim_score  # noqa: E402
from klein4b.image_grid import make_four_up_grid  # noqa: E402


def main() -> None:
    manifest_path = REPO_ROOT / "data" / "demo_ffhq_makeup" / "manifest.json"
    eval_root = REPO_ROOT / "outputs" / "eval"
    base_dir = eval_root / "base"
    lora_dir = eval_root / "lora"
    grids_dir = eval_root / "grids"
    metrics_dir = eval_root / "metrics"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    test_rows = [row for row in manifest if row["split"] == "test"]

    for row in test_rows:
        identity = row["identity_id"]
        ref = Image.open(row["reference_path"]).convert("RGB")
        base = Image.open(base_dir / f"{identity}.png").convert("RGB")
        lora = Image.open(lora_dir / f"{identity}.png").convert("RGB")
        target = Image.open(row["target_path"]).convert("RGB")

        grid = make_four_up_grid(
            [ref, base, lora, target],
            ["reference", "base", "lora", "target"],
        )
        grids_dir.mkdir(parents=True, exist_ok=True)
        grid.save(grids_dir / f"{identity}.png")

        metrics = {
            "identity_id": identity,
            "base_ssim_to_target": ssim_score(base, target),
            "lora_ssim_to_target": ssim_score(lora, target),
        }
        metrics_dir.mkdir(parents=True, exist_ok=True)
        (metrics_dir / f"{identity}.json").write_text(
            json.dumps(metrics, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()

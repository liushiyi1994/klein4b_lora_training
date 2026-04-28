from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from klein4b.sagemaker_inference import input_fn, model_fn, output_fn, predict_fn  # noqa: E402

__all__ = ["input_fn", "model_fn", "output_fn", "predict_fn"]

from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "prepare_fal_training_package.py"


def write_pair(dataset_root: Path, stem: str) -> None:
    control_dir = dataset_root / "control"
    dataset_dir = dataset_root / "dataset"
    control_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (control_dir / f"{stem}.png").write_bytes(b"control")
    (dataset_dir / f"{stem}.png").write_bytes(b"target")
    (dataset_dir / f"{stem}.txt").write_text(
        "Change image 1 into a <mrblbust> Greek marble bust.\n",
        encoding="utf-8",
    )


def test_prepare_fal_training_package_cli_help_exits_without_packaging() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "usage:" in result.stdout
    assert "--dataset-root" in result.stdout
    assert "--image-data-url" in result.stdout


def test_prepare_fal_training_package_cli_writes_offline_package(tmp_path: Path) -> None:
    dataset_root = tmp_path / "v7_30"
    output_root = tmp_path / "fal_packages"
    write_pair(dataset_root, "case_a")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--dataset-root",
            str(dataset_root),
            "--output-root",
            str(output_root),
            "--package-name",
            "fal_cli_test",
            "--steps",
            "1000",
            "--learning-rate",
            "0.00005",
            "--image-data-url",
            "https://example.invalid/fal_cli_test.zip",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "fal_cli_test.zip" in result.stdout
    assert "sample count: 1" in result.stdout

    zip_path = output_root / "fal_cli_test.zip"
    request_path = output_root / "fal_cli_test_request.json"
    assert zip_path.exists()
    assert request_path.exists()

    with zipfile.ZipFile(zip_path) as package:
        assert set(package.namelist()) == {
            "case_a_start.png",
            "case_a_end.png",
            "case_a.txt",
        }

    request_payload = json.loads(request_path.read_text(encoding="utf-8"))
    assert request_payload == {
        "image_data_url": "https://example.invalid/fal_cli_test.zip",
        "steps": 1000,
        "learning_rate": 0.00005,
        "output_lora_format": "fal",
    }

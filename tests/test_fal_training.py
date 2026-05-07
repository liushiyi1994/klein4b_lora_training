from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from klein4b.fal_training import (
    FAL_KLEIN_9B_EDIT_TRAINER_ENDPOINT,
    FalPackageOptions,
    FalTrainingPackageError,
    build_fal_training_package,
)


def write_pair(
    dataset_root: Path,
    stem: str,
    *,
    control_suffix: str = ".png",
    target_suffix: str = ".jpg",
    caption: str | None = "Change image 1 into a <mrblbust> weathered Greek marble bust.",
) -> None:
    control_dir = dataset_root / "control"
    dataset_dir = dataset_root / "dataset"
    control_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (control_dir / f"{stem}{control_suffix}").write_bytes(f"control-{stem}".encode("ascii"))
    (dataset_dir / f"{stem}{target_suffix}").write_bytes(f"target-{stem}".encode("ascii"))
    if caption is not None:
        (dataset_dir / f"{stem}.txt").write_text(caption + "\n", encoding="utf-8")


def test_build_fal_training_package_writes_zip_manifest_and_request_json(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "v7_30"
    write_pair(dataset_root, "case_a")
    output_dir = tmp_path / "packages"

    result = build_fal_training_package(
        FalPackageOptions(
            dataset_root=dataset_root,
            output_dir=output_dir,
            package_name="fal_marble_test",
            steps=3600,
            learning_rate=0.00004,
            output_lora_format="fal",
        )
    )

    assert result.sample_count == 1
    assert result.zip_path == output_dir / "fal_marble_test.zip"
    assert result.manifest_path == output_dir / "fal_marble_test_manifest.json"
    assert result.request_json_path == output_dir / "fal_marble_test_request.json"

    with zipfile.ZipFile(result.zip_path) as package:
        assert set(package.namelist()) == {
            "case_a_start.png",
            "case_a_end.jpg",
            "case_a.txt",
        }
        assert package.read("case_a_start.png") == b"control-case_a"
        assert package.read("case_a_end.jpg") == b"target-case_a"
        assert b"<mrblbust>" in package.read("case_a.txt")

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["endpoint_id"] == FAL_KLEIN_9B_EDIT_TRAINER_ENDPOINT
    assert manifest["sample_count"] == 1
    assert manifest["samples"] == [
        {
            "stem": "case_a",
            "control_source": str(dataset_root / "control" / "case_a.png"),
            "target_source": str(dataset_root / "dataset" / "case_a.jpg"),
            "caption_source": str(dataset_root / "dataset" / "case_a.txt"),
            "start_member": "case_a_start.png",
            "end_member": "case_a_end.jpg",
            "caption_member": "case_a.txt",
        }
    ]

    request_payload = json.loads(result.request_json_path.read_text(encoding="utf-8"))
    assert request_payload == {
        "image_data_url": "",
        "steps": 3600,
        "learning_rate": 0.00004,
        "output_lora_format": "fal",
    }


def test_build_fal_training_package_rejects_missing_caption_without_default(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "v7_30"
    write_pair(dataset_root, "case_a", caption=None)

    with pytest.raises(FalTrainingPackageError, match="missing caption"):
        build_fal_training_package(
            FalPackageOptions(
                dataset_root=dataset_root,
                output_dir=tmp_path / "packages",
                package_name="fal_marble_test",
            )
        )


def test_build_fal_training_package_allows_missing_caption_with_explicit_default(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "v7_30"
    write_pair(dataset_root, "case_a", caption=None)

    result = build_fal_training_package(
        FalPackageOptions(
            dataset_root=dataset_root,
            output_dir=tmp_path / "packages",
            package_name="fal_marble_test",
            default_caption="Change image 1 into a <mrblbust> Greek marble bust.",
            allow_missing_captions=True,
        )
    )

    with zipfile.ZipFile(result.zip_path) as package:
        assert set(package.namelist()) == {
            "case_a_start.png",
            "case_a_end.jpg",
        }

    request_payload = json.loads(result.request_json_path.read_text(encoding="utf-8"))
    assert request_payload["default_caption"] == (
        "Change image 1 into a <mrblbust> Greek marble bust."
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["samples"][0]["caption_source"] == "default_caption"
    assert manifest["samples"][0]["caption_member"] is None


def test_build_fal_training_package_rejects_unmatched_control_and_target_stems(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "v7_30"
    write_pair(dataset_root, "case_a")
    (dataset_root / "control" / "case_a.png").unlink()

    with pytest.raises(FalTrainingPackageError, match="missing control image"):
        build_fal_training_package(
            FalPackageOptions(
                dataset_root=dataset_root,
                output_dir=tmp_path / "packages",
                package_name="fal_marble_test",
            )
        )


def test_build_fal_training_package_validates_fal_training_options(tmp_path: Path) -> None:
    dataset_root = tmp_path / "v7_30"
    write_pair(dataset_root, "case_a")

    with pytest.raises(FalTrainingPackageError, match="steps must be"):
        build_fal_training_package(
            FalPackageOptions(
                dataset_root=dataset_root,
                output_dir=tmp_path / "packages",
                package_name="fal_marble_test",
                steps=50,
            )
        )

    with pytest.raises(FalTrainingPackageError, match="output_lora_format"):
        build_fal_training_package(
            FalPackageOptions(
                dataset_root=dataset_root,
                output_dir=tmp_path / "packages",
                package_name="fal_marble_test",
                output_lora_format="diffusers",
            )
        )

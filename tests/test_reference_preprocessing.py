from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from klein4b.reference_preprocessing import (
    FaceDetection,
    ReferencePreprocessError,
    make_loose_portrait_crop_box,
    preprocess_reference_image,
    select_reference_face,
)


class FakeDetector:
    def __init__(self, detections: list[FaceDetection]) -> None:
        self.detections = detections
        self.called_with: Path | None = None

    def detect_faces(self, image_path: Path) -> list[FaceDetection]:
        self.called_with = image_path
        return self.detections


def test_select_reference_face_prefers_large_centered_face() -> None:
    detections = [
        FaceDetection(bbox=(5.0, 5.0, 45.0, 45.0), score=0.99),
        FaceDetection(bbox=(80.0, 50.0, 160.0, 150.0), score=0.91),
        FaceDetection(bbox=(170.0, 10.0, 210.0, 50.0), score=0.98),
    ]

    selected = select_reference_face(detections, image_size=(240, 200))

    assert selected == detections[1]


def test_make_loose_portrait_crop_box_expands_face_to_three_by_four_crop() -> None:
    detection = FaceDetection(bbox=(80.0, 60.0, 140.0, 140.0), score=0.95)

    crop_box = make_loose_portrait_crop_box(
        detection,
        image_size=(240, 320),
    )

    left, top, right, bottom = crop_box
    assert left < 80
    assert top < 60
    assert right > 140
    assert bottom > 140
    assert pytest.approx((right - left) / (bottom - top), rel=0.02) == 0.75
    assert crop_box == (20, 18, 200, 258)


def test_preprocess_reference_image_saves_crop_and_metadata(tmp_path: Path) -> None:
    reference_path = tmp_path / "reference.png"
    output_path = tmp_path / "reference_preprocessed.jpg"
    metadata_path = tmp_path / "preprocess_metadata.json"
    Image.new("RGB", (240, 320), color="red").save(reference_path)
    detector = FakeDetector([FaceDetection(bbox=(80.0, 60.0, 140.0, 140.0), score=0.95)])

    result = preprocess_reference_image(
        reference_path=reference_path,
        output_path=output_path,
        metadata_path=metadata_path,
        detector=detector,
    )

    assert detector.called_with == reference_path
    assert result.effective_reference_path == output_path
    assert output_path.exists()
    assert metadata_path.exists()
    with Image.open(output_path) as image:
        assert image.size == (180, 240)
        assert image.mode == "RGB"
    assert result.metadata["enabled"] is True
    assert result.metadata["selected_face"]["bbox"] == [80.0, 60.0, 140.0, 140.0]
    assert result.metadata["crop_box"] == [20, 18, 200, 258]
    assert result.metadata["output_size"] == [180, 240]


def test_preprocess_reference_image_raises_when_no_face_is_detected(tmp_path: Path) -> None:
    reference_path = tmp_path / "reference.png"
    Image.new("RGB", (240, 320), color="red").save(reference_path)

    with pytest.raises(ReferencePreprocessError, match="No face detected"):
        preprocess_reference_image(
            reference_path=reference_path,
            output_path=tmp_path / "reference_preprocessed.jpg",
            metadata_path=tmp_path / "preprocess_metadata.json",
            detector=FakeDetector([]),
        )

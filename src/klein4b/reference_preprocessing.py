from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from PIL import Image


class ReferencePreprocessError(ValueError):
    """Raised when a reference image cannot be preprocessed safely."""


@dataclass(frozen=True)
class FaceDetection:
    bbox: tuple[float, float, float, float]
    score: float


class FaceDetector(Protocol):
    def detect_faces(self, image_path: Path) -> list[FaceDetection]:
        """Return detected faces in source-image coordinates."""


@dataclass(frozen=True)
class ReferencePreprocessResult:
    original_reference_path: Path
    effective_reference_path: Path
    metadata: dict[str, object]


class ScrfdFaceDetector:
    """InsightFace/SCRFD detector wrapper.

    This wrapper expects a caller-supplied ONNX path. It intentionally does not
    auto-download InsightFace model packs because the official pretrained
    weights are research/non-commercial unless separately licensed.
    """

    def __init__(
        self,
        model_path: Path,
        *,
        providers: tuple[str, ...] = ("CPUExecutionProvider",),
        det_size: tuple[int, int] = (640, 640),
    ) -> None:
        self.model_path = model_path
        self.providers = providers
        self.det_size = det_size
        self._detector: object | None = None

    def detect_faces(self, image_path: Path) -> list[FaceDetection]:
        import cv2
        import insightface
        import numpy as np

        detector = self._load_detector(insightface)
        with Image.open(image_path) as image:
            rgb = np.asarray(image.convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        bboxes, _keypoints = detector.detect(bgr, max_num=0)
        detections: list[FaceDetection] = []
        for bbox in bboxes:
            x1, y1, x2, y2, score = [float(value) for value in bbox[:5]]
            detections.append(FaceDetection(bbox=(x1, y1, x2, y2), score=score))
        return detections

    def _load_detector(self, insightface_module: object) -> object:
        if self._detector is None:
            detector = insightface_module.model_zoo.get_model(
                str(self.model_path),
                providers=list(self.providers),
            )
            if detector is None:
                raise ReferencePreprocessError(
                    f"SCRFD model could not be loaded from {self.model_path}"
                )
            detector.prepare(ctx_id=-1, input_size=self.det_size)
            self._detector = detector
        return self._detector


def preprocess_reference_image(
    *,
    reference_path: Path,
    output_path: Path,
    metadata_path: Path,
    detector: FaceDetector,
) -> ReferencePreprocessResult:
    detections = detector.detect_faces(reference_path)
    with Image.open(reference_path) as image:
        source = image.convert("RGB")
        image_size = source.size
        selected = select_reference_face(detections, image_size=image_size)
        crop_box = make_loose_portrait_crop_box(selected, image_size=image_size)
        crop = source.crop(crop_box)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        crop.save(output_path, format="JPEG", quality=95)

    metadata = {
        "enabled": True,
        "original_reference": str(reference_path),
        "effective_reference": str(output_path),
        "detected_faces": [
            {
                "bbox": _bbox_to_list(detection.bbox),
                "score": detection.score,
            }
            for detection in detections
        ],
        "selected_face": {
            "bbox": _bbox_to_list(selected.bbox),
            "score": selected.score,
        },
        "crop_box": list(crop_box),
        "source_size": list(image_size),
        "output_size": [crop_box[2] - crop_box[0], crop_box[3] - crop_box[1]],
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return ReferencePreprocessResult(
        original_reference_path=reference_path,
        effective_reference_path=output_path,
        metadata=metadata,
    )


def select_reference_face(
    detections: list[FaceDetection],
    *,
    image_size: tuple[int, int],
) -> FaceDetection:
    if not detections:
        raise ReferencePreprocessError("No face detected in reference image")

    image_width, image_height = image_size
    image_center_x = image_width / 2
    image_center_y = image_height / 2
    image_diag_sq = float(image_width * image_width + image_height * image_height)

    def rank(detection: FaceDetection) -> float:
        x1, y1, x2, y2 = detection.bbox
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        distance_sq = (center_x - image_center_x) ** 2 + (center_y - image_center_y) ** 2
        center_penalty = distance_sq / max(image_diag_sq, 1.0)
        return area * detection.score * (1.0 - min(center_penalty, 0.95))

    return max(detections, key=rank)


def make_loose_portrait_crop_box(
    detection: FaceDetection,
    *,
    image_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    image_width, image_height = image_size
    x1, y1, x2, y2 = detection.bbox
    face_width = max(1.0, x2 - x1)
    face_height = max(1.0, y2 - y1)
    face_center_x = (x1 + x2) / 2
    face_center_y = (y1 + y2) / 2

    crop_width = max(face_width * 3.0, face_height * 2.25)
    crop_height = crop_width / 0.75
    if crop_width > image_width:
        crop_width = float(image_width)
        crop_height = min(float(image_height), crop_width / 0.75)
    if crop_height > image_height:
        crop_height = float(image_height)
        crop_width = min(float(image_width), crop_height * 0.75)

    left = face_center_x - crop_width / 2
    top = face_center_y - crop_height * 0.34
    left, top = _clamp_crop_origin(
        left=left,
        top=top,
        crop_width=crop_width,
        crop_height=crop_height,
        image_width=image_width,
        image_height=image_height,
    )
    right = left + crop_width
    bottom = top + crop_height
    return (
        int(round(left)),
        int(round(top)),
        int(round(right)),
        int(round(bottom)),
    )


def _clamp_crop_origin(
    *,
    left: float,
    top: float,
    crop_width: float,
    crop_height: float,
    image_width: int,
    image_height: int,
) -> tuple[float, float]:
    max_left = max(0.0, image_width - crop_width)
    max_top = max(0.0, image_height - crop_height)
    return (
        min(max(left, 0.0), max_left),
        min(max(top, 0.0), max_top),
    )


def _bbox_to_list(bbox: tuple[float, float, float, float]) -> list[float]:
    return [float(value) for value in bbox]

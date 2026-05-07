from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path


FAL_KLEIN_9B_EDIT_TRAINER_ENDPOINT = "fal-ai/flux-2-klein-9b-base-trainer/edit"
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")
OUTPUT_LORA_FORMATS = ("fal", "comfy")


class FalTrainingPackageError(ValueError):
    """Raised when a dataset cannot be packaged for fal training."""


@dataclass(frozen=True)
class FalPackageOptions:
    dataset_root: Path
    output_dir: Path
    package_name: str = "marble_v7_30_klein9b_fal"
    steps: int = 3600
    learning_rate: float = 0.00004
    output_lora_format: str = "fal"
    image_data_url: str = ""
    default_caption: str | None = None
    allow_missing_captions: bool = False


@dataclass(frozen=True)
class FalPackageResult:
    zip_path: Path
    manifest_path: Path
    request_json_path: Path
    sample_count: int


@dataclass(frozen=True)
class _FalSample:
    stem: str
    control_source: Path
    target_source: Path
    caption_source: Path | None
    start_member: str
    end_member: str
    caption_member: str | None


def build_fal_training_package(options: FalPackageOptions) -> FalPackageResult:
    """Write a fal-compatible training zip, manifest, and request JSON."""

    _validate_options(options)
    dataset_dir = options.dataset_root / "dataset"
    control_dir = options.dataset_root / "control"
    samples = _discover_samples(
        dataset_dir=dataset_dir,
        control_dir=control_dir,
        default_caption=options.default_caption,
        allow_missing_captions=options.allow_missing_captions,
    )

    options.output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = options.output_dir / f"{options.package_name}.zip"
    manifest_path = options.output_dir / f"{options.package_name}_manifest.json"
    request_json_path = options.output_dir / f"{options.package_name}_request.json"

    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as package:
        for sample in samples:
            package.write(sample.control_source, sample.start_member)
            package.write(sample.target_source, sample.end_member)
            if sample.caption_source is not None and sample.caption_member is not None:
                package.write(sample.caption_source, sample.caption_member)

    manifest = _build_manifest(
        options=options,
        samples=samples,
        zip_path=zip_path,
        request_json_path=request_json_path,
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    request_payload = _build_request_payload(options)
    request_json_path.write_text(
        json.dumps(request_payload, indent=2) + "\n",
        encoding="utf-8",
    )

    return FalPackageResult(
        zip_path=zip_path,
        manifest_path=manifest_path,
        request_json_path=request_json_path,
        sample_count=len(samples),
    )


def _validate_options(options: FalPackageOptions) -> None:
    if not options.package_name or Path(options.package_name).name != options.package_name:
        raise FalTrainingPackageError("package_name must be a plain file stem")
    if options.steps < 100 or options.steps > 10000 or options.steps % 100 != 0:
        raise FalTrainingPackageError("steps must be between 100 and 10000 in increments of 100")
    if options.learning_rate <= 0:
        raise FalTrainingPackageError("learning_rate must be greater than 0")
    if options.output_lora_format not in OUTPUT_LORA_FORMATS:
        raise FalTrainingPackageError(
            f"output_lora_format must be one of: {', '.join(OUTPUT_LORA_FORMATS)}"
        )
    if options.default_caption is not None and not options.default_caption.strip():
        raise FalTrainingPackageError("default_caption must not be blank")
    if options.allow_missing_captions and options.default_caption is None:
        raise FalTrainingPackageError(
            "allow_missing_captions requires an explicit default_caption"
        )


def _discover_samples(
    *,
    dataset_dir: Path,
    control_dir: Path,
    default_caption: str | None,
    allow_missing_captions: bool,
) -> list[_FalSample]:
    target_images = _collect_images(dataset_dir, "target")
    control_images = _collect_images(control_dir, "control")
    if not target_images:
        raise FalTrainingPackageError(f"no target images found in {dataset_dir}")

    missing_controls = sorted(set(target_images) - set(control_images))
    if missing_controls:
        raise FalTrainingPackageError(
            "missing control image for target stem(s): " + ", ".join(missing_controls)
        )

    missing_targets = sorted(set(control_images) - set(target_images))
    if missing_targets:
        raise FalTrainingPackageError(
            "missing target image for control stem(s): " + ", ".join(missing_targets)
        )

    samples: list[_FalSample] = []
    for stem in sorted(target_images):
        target_source = target_images[stem]
        control_source = control_images[stem]
        caption_source = dataset_dir / f"{stem}.txt"
        caption_member: str | None = f"{stem}.txt"

        if not caption_source.exists():
            if default_caption is None:
                raise FalTrainingPackageError(
                    f"missing caption for {stem}: add {caption_source.name} "
                    "or provide a default_caption"
                )
            if not allow_missing_captions:
                raise FalTrainingPackageError(
                    f"missing caption for {stem}: pass allow_missing_captions "
                    "to use default_caption"
                )
            caption_source = None
            caption_member = None

        samples.append(
            _FalSample(
                stem=stem,
                control_source=control_source,
                target_source=target_source,
                caption_source=caption_source,
                start_member=f"{stem}_start{control_source.suffix.lower()}",
                end_member=f"{stem}_end{target_source.suffix.lower()}",
                caption_member=caption_member,
            )
        )

    return samples


def _collect_images(directory: Path, role: str) -> dict[str, Path]:
    if not directory.exists():
        raise FalTrainingPackageError(f"{role} directory not found: {directory}")
    if not directory.is_dir():
        raise FalTrainingPackageError(f"{role} path is not a directory: {directory}")

    images: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if path.stem in images:
            raise FalTrainingPackageError(
                f"multiple {role} images found for stem {path.stem}: "
                f"{images[path.stem].name}, {path.name}"
            )
        images[path.stem] = path

    return images


def _build_manifest(
    *,
    options: FalPackageOptions,
    samples: list[_FalSample],
    zip_path: Path,
    request_json_path: Path,
) -> dict[str, object]:
    return {
        "endpoint_id": FAL_KLEIN_9B_EDIT_TRAINER_ENDPOINT,
        "dataset_root": str(options.dataset_root),
        "zip_path": str(zip_path),
        "request_json_path": str(request_json_path),
        "sample_count": len(samples),
        "steps": options.steps,
        "learning_rate": options.learning_rate,
        "output_lora_format": options.output_lora_format,
        "default_caption": options.default_caption,
        "samples": [_manifest_sample(sample) for sample in samples],
    }


def _manifest_sample(sample: _FalSample) -> dict[str, object]:
    return {
        "stem": sample.stem,
        "control_source": str(sample.control_source),
        "target_source": str(sample.target_source),
        "caption_source": str(sample.caption_source)
        if sample.caption_source is not None
        else "default_caption",
        "start_member": sample.start_member,
        "end_member": sample.end_member,
        "caption_member": sample.caption_member,
    }


def _build_request_payload(options: FalPackageOptions) -> dict[str, object]:
    payload: dict[str, object] = {
        "image_data_url": options.image_data_url,
        "steps": options.steps,
        "learning_rate": options.learning_rate,
        "output_lora_format": options.output_lora_format,
    }
    if options.default_caption is not None:
        payload["default_caption"] = options.default_caption
    return payload

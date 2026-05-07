from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_DEPS = REPO_ROOT / ".python-deps" / "fal-client"
if LOCAL_DEPS.exists() and str(LOCAL_DEPS) not in sys.path:
    sys.path.insert(0, str(LOCAL_DEPS))

DEFAULT_INPUT_ROOT = Path(
    "data/marble-bust-data/v7_30/synthetic_generation/wearable_v1_15"
)
DEFAULT_OUTPUT_ROOT = Path(
    "data/marble-bust-data/v7_30/synthetic_generation/wearable_v1_15_gpt_image_2_fal"
)
DEFAULT_ROWS = ("001", "003", "005", "007", "009", "010", "011")
DEFAULT_MODEL = "openai/gpt-image-2/edit"
FAL_ENV_KEYS = (
    "FAL_KEY",
    "FAL_API_KEY",
    "FAL_AI_KEY",
    "fal_key",
    "fal_api_key",
    "fai_api_key",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate GPT Image 2 FAL comparison examples from rendered wearable jobs."
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--env", type=Path, default=REPO_ROOT / ".env")
    parser.add_argument("--row", action="append", default=None, help="Three-digit row id.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--quality", default="high")
    parser.add_argument("--image-size", default="auto")
    parser.add_argument("--output-format", default="png")
    parser.add_argument("--num-images", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--contact-sheet", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = tuple(args.row or DEFAULT_ROWS)
    fal_key = load_fal_key(args.env)
    if not fal_key:
        raise RuntimeError(
            "No FAL key found. Set one of FAL_KEY, FAL_API_KEY, FAL_AI_KEY, "
            "fal_key, fal_api_key, or fai_api_key in the environment or .env."
        )
    os.environ.setdefault("FAL_KEY", fal_key)

    job_dirs = select_job_dirs(args.input_root, rows)
    generated: list[Path] = []
    errors: list[str] = []
    for source_job_dir in job_dirs:
        output_job_dir = args.output_root / source_job_dir.name
        output_path = output_job_dir / "generated.png"
        if output_path.exists() and not args.force:
            print(f"Skipping existing {output_path}", flush=True)
            generated.append(output_path)
            continue

        copy_job_assets(source_job_dir, output_job_dir)
        try:
            result_path = generate_one(
                output_job_dir,
                model=args.model,
                quality=args.quality,
                image_size=args.image_size,
                output_format=args.output_format,
                num_images=args.num_images,
            )
        except Exception as exc:  # noqa: BLE001
            message = f"{source_job_dir.name}: {exc}"
            print(f"ERROR {message}", file=sys.stderr, flush=True)
            errors.append(message)
            continue
        generated.append(result_path)
        print(f"Generated {result_path}", flush=True)

    if generated:
        contact_sheet = args.contact_sheet or args.output_root / "review_contact_sheet.jpg"
        make_contact_sheet([path.parent for path in generated], contact_sheet)
        print(f"Review contact sheet written to {contact_sheet}", flush=True)

    if errors:
        print("Generation completed with errors:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    return 0


def load_fal_key(env_path: Path | None) -> str | None:
    for key in FAL_ENV_KEYS:
        value = os.environ.get(key)
        if value:
            return value
    if env_path is None or not env_path.exists():
        return None
    values: dict[str, str] = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    for key in FAL_ENV_KEYS:
        if values.get(key):
            return values[key]
    return None


def select_job_dirs(input_root: Path, rows: tuple[str, ...]) -> list[Path]:
    if not input_root.exists():
        raise FileNotFoundError(f"input root not found: {input_root}")
    selected: list[Path] = []
    for row in rows:
        matches = sorted(input_root.glob(f"{row}_*"))
        if len(matches) != 1:
            raise FileNotFoundError(f"expected one job directory for row {row}, found {len(matches)}")
        selected.append(matches[0])
    return selected


def copy_job_assets(source_job_dir: Path, output_job_dir: Path) -> None:
    output_job_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("style_1.png", "style_2.png", "prompt.txt", "metadata.json"):
        source = source_job_dir / filename
        if source.exists():
            shutil.copy2(source, output_job_dir / filename)
    for pattern in ("statue_reference.*", "control.*"):
        source = first_matching_file(source_job_dir, pattern)
        if source is None:
            raise FileNotFoundError(f"missing {pattern} in {source_job_dir}")
        shutil.copy2(source, output_job_dir / source.name)


def generate_one(
    job_dir: Path,
    *,
    model: str,
    quality: str,
    image_size: str,
    output_format: str,
    num_images: int,
) -> Path:
    import fal_client

    prompt = (job_dir / "prompt.txt").read_text(encoding="utf-8")
    image_paths = [
        job_dir / "style_1.png",
        job_dir / "style_2.png",
        required_matching_file(job_dir, "statue_reference.*"),
        required_matching_file(job_dir, "control.*"),
    ]
    image_urls = [fal_client.upload_file(path) for path in image_paths]
    result = fal_client.subscribe(
        model,
        arguments={
            "prompt": prompt,
            "image_urls": image_urls,
            "image_size": image_size,
            "quality": quality,
            "num_images": num_images,
            "output_format": output_format,
        },
        with_logs=True,
        client_timeout=600,
    )
    url = first_result_image_url(result)
    if not url:
        raise RuntimeError(f"no image URL returned; response keys: {sorted(result) if isinstance(result, dict) else type(result)}")
    output_path = job_dir / "generated.png"
    download_url(url, output_path)
    return output_path


def first_result_image_url(result: Any) -> str | None:
    if not isinstance(result, dict):
        return None
    images = result.get("images") or result.get("data", {}).get("images")
    if isinstance(images, list):
        for image in images:
            if isinstance(image, dict) and image.get("url"):
                return str(image["url"])
            if isinstance(image, str):
                return image
    image = result.get("image")
    if isinstance(image, dict) and image.get("url"):
        return str(image["url"])
    if isinstance(image, str):
        return image
    return None


def download_url(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "klein4b-gpt-image2-fal/1.0"})
    with urllib.request.urlopen(request, timeout=300) as response:
        output_path.write_bytes(response.read())


def make_contact_sheet(
    job_dirs: list[Path],
    output_path: Path,
    *,
    thumb_size: tuple[int, int] = (150, 150),
) -> Path:
    from PIL import Image, ImageDraw

    columns = (
        ("style_1", "style_1.*"),
        ("style_2", "style_2.*"),
        ("statue_ref", "statue_reference.*"),
        ("control", "control.*"),
        ("gpt_image_2", "generated.png"),
    )
    gap = 10
    row_header_height = 24
    label_height = 36
    row_height = row_header_height + thumb_size[1] + label_height + gap
    width = gap + len(columns) * (thumb_size[0] + gap)
    height = gap + max(1, len(job_dirs)) * row_height
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)

    y = gap
    for job_dir in job_dirs:
        draw.text((gap, y), job_dir.name, fill=(0, 0, 0))
        image_y = y + row_header_height
        for column_index, (label, pattern) in enumerate(columns):
            x = gap + column_index * (thumb_size[0] + gap)
            image_path = first_matching_file(job_dir, pattern)
            if image_path is None:
                draw.rectangle(
                    (x, image_y, x + thumb_size[0], image_y + thumb_size[1]),
                    outline=(180, 180, 180),
                    fill=(245, 245, 245),
                )
                draw.text((x + 8, image_y + 62), "missing", fill=(120, 120, 120))
            else:
                sheet.paste(thumbnail(image_path, thumb_size), (x, image_y))
            draw.text((x, image_y + thumb_size[1] + 4), label, fill=(0, 0, 0))
        y += row_height

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)
    return output_path


def thumbnail(path: Path, size: tuple[int, int]):
    from PIL import Image

    with Image.open(path) as image:
        thumb = image.convert("RGB")
        thumb.thumbnail(size)
        canvas = Image.new("RGB", size, "white")
        canvas.paste(thumb, ((size[0] - thumb.width) // 2, (size[1] - thumb.height) // 2))
        return canvas


def required_matching_file(directory: Path, pattern: str) -> Path:
    path = first_matching_file(directory, pattern)
    if path is None:
        raise FileNotFoundError(f"missing {pattern} in {directory}")
    return path


def first_matching_file(directory: Path, pattern: str) -> Path | None:
    return next(iter(sorted(directory.glob(pattern))), None)


if __name__ == "__main__":
    started = time.monotonic()
    try:
        raise SystemExit(main())
    finally:
        elapsed = time.monotonic() - started
        print(f"Elapsed {elapsed:.1f}s", flush=True)

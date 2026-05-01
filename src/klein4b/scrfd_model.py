from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from klein4b.paths import data_dir

DEFAULT_SCRFD_MODEL_NAME = "buffalo_l_det_10g"
DEFAULT_SCRFD_MODEL_URL = (
    "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
)
DEFAULT_SCRFD_ARCHIVE_MEMBER = "det_10g.onnx"

FetchModel = Callable[[str, Path], None]


class ScrfdModelDownloadError(RuntimeError):
    """Raised when the SCRFD model cannot be downloaded or validated."""


@dataclass(frozen=True)
class ScrfdModelDownload:
    path: Path
    manifest_path: Path
    bytes: int
    sha256: str
    source_url: str
    reused_existing: bool
    archive_member: str | None = None


def default_scrfd_model_path() -> Path:
    return data_dir() / "models" / "scrfd" / f"{DEFAULT_SCRFD_MODEL_NAME}.onnx"


def download_scrfd_model(
    *,
    output_path: Path | None = None,
    url: str = DEFAULT_SCRFD_MODEL_URL,
    force: bool = False,
    fetch: FetchModel | None = None,
) -> ScrfdModelDownload:
    fetch_model = fetch or _fetch_url_to_path
    model_path = output_path or default_scrfd_model_path()
    manifest_path = model_path.with_suffix(".manifest.json")
    reused_existing = model_path.exists() and not force

    if not reused_existing:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        download_path = model_path.with_suffix(".download")
        if download_path.exists():
            download_path.unlink()
        fetch_model(url, download_path)
        if not download_path.exists():
            raise ScrfdModelDownloadError(f"download did not create {download_path}")
        if download_path.stat().st_size == 0:
            download_path.unlink()
            raise ScrfdModelDownloadError("downloaded SCRFD model is empty")
        archive_member = _install_downloaded_model(download_path, model_path)
    else:
        archive_member = None

    byte_count, digest = _file_stats(model_path)
    if byte_count == 0:
        raise ScrfdModelDownloadError(f"cached SCRFD model is empty: {model_path}")

    result = ScrfdModelDownload(
        path=model_path,
        manifest_path=manifest_path,
        bytes=byte_count,
        sha256=digest,
        source_url=url,
        reused_existing=reused_existing,
        archive_member=archive_member,
    )
    _write_manifest(result)
    return result


def _fetch_url_to_path(url: str, destination: Path) -> None:
    with urllib.request.urlopen(url) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)


def _install_downloaded_model(download_path: Path, model_path: Path) -> str | None:
    temp_model_path = model_path.with_suffix(".tmp")
    if temp_model_path.exists():
        temp_model_path.unlink()

    if zipfile.is_zipfile(download_path):
        with zipfile.ZipFile(download_path) as archive:
            member_name = _find_scrfd_archive_member(archive.namelist())
            temp_model_path.write_bytes(archive.read(member_name))
        download_path.unlink()
        temp_model_path.replace(model_path)
        return member_name

    download_path.replace(model_path)
    return None


def _find_scrfd_archive_member(names: list[str]) -> str:
    for name in names:
        if name == DEFAULT_SCRFD_ARCHIVE_MEMBER or name.endswith(
            f"/{DEFAULT_SCRFD_ARCHIVE_MEMBER}"
        ):
            return name
    raise ScrfdModelDownloadError(f"SCRFD archive does not contain {DEFAULT_SCRFD_ARCHIVE_MEMBER}")


def _file_stats(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            byte_count += len(chunk)
            digest.update(chunk)
    return byte_count, digest.hexdigest()


def _write_manifest(result: ScrfdModelDownload) -> None:
    result.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model_name": result.path.stem,
        "source_url": result.source_url,
        "archive_member": result.archive_member,
        "path": str(result.path),
        "bytes": result.bytes,
        "sha256": result.sha256,
        "reused_existing": result.reused_existing,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "env": {
            "KLEIN4B_SCRFD_MODEL_PATH": str(result.path),
        },
    }
    result.manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

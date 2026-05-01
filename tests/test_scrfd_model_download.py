from __future__ import annotations

import hashlib
import importlib.util
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from klein4b import scrfd_model


def load_download_cli_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "download_scrfd_model.py"
    spec = importlib.util.spec_from_file_location("download_scrfd_model", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_download_scrfd_model_writes_default_cache_and_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    model_bytes = b"fake-face-detector-onnx-model"
    calls: list[tuple[str, Path]] = []
    monkeypatch.setattr(scrfd_model, "data_dir", lambda: data_root)

    def fake_fetch(url: str, destination: Path) -> None:
        calls.append((url, destination))
        with zipfile.ZipFile(destination, "w") as archive:
            archive.writestr("buffalo_l/det_10g.onnx", model_bytes)
            archive.writestr("buffalo_l/recognition.onnx", b"not-used")

    result = scrfd_model.download_scrfd_model(fetch=fake_fetch)

    expected_model_path = data_root / "models" / "scrfd" / "buffalo_l_det_10g.onnx"
    expected_manifest_path = data_root / "models" / "scrfd" / "buffalo_l_det_10g.manifest.json"
    assert result.path == expected_model_path
    assert result.manifest_path == expected_manifest_path
    assert result.bytes == len(model_bytes)
    assert result.sha256 == hashlib.sha256(model_bytes).hexdigest()
    assert result.reused_existing is False
    assert calls == [
        (scrfd_model.DEFAULT_SCRFD_MODEL_URL, expected_model_path.with_suffix(".download"))
    ]

    manifest = json.loads(expected_manifest_path.read_text(encoding="utf-8"))
    assert manifest["model_name"] == "buffalo_l_det_10g"
    assert manifest["source_url"] == scrfd_model.DEFAULT_SCRFD_MODEL_URL
    assert manifest["archive_member"] == "buffalo_l/det_10g.onnx"
    assert manifest["path"] == str(expected_model_path)
    assert manifest["bytes"] == len(model_bytes)
    assert manifest["sha256"] == hashlib.sha256(model_bytes).hexdigest()
    assert manifest["reused_existing"] is False
    assert manifest["env"] == {
        "KLEIN4B_SCRFD_MODEL_PATH": str(expected_model_path),
    }


def test_download_scrfd_model_reuses_existing_file_without_fetching(tmp_path: Path) -> None:
    model_path = tmp_path / "detectors" / "scrfd.onnx"
    model_path.parent.mkdir()
    model_path.write_bytes(b"cached-model")

    def fail_fetch(url: str, destination: Path) -> None:
        raise AssertionError(f"unexpected network fetch for {url} -> {destination}")

    result = scrfd_model.download_scrfd_model(output_path=model_path, fetch=fail_fetch)

    assert result.path == model_path
    assert result.bytes == len(b"cached-model")
    assert result.sha256 == hashlib.sha256(b"cached-model").hexdigest()
    assert result.reused_existing is True


def test_download_scrfd_model_rejects_empty_download(tmp_path: Path) -> None:
    def fake_fetch(url: str, destination: Path) -> None:
        destination.write_bytes(b"")

    with pytest.raises(scrfd_model.ScrfdModelDownloadError, match="empty"):
        scrfd_model.download_scrfd_model(
            output_path=tmp_path / "scrfd.onnx",
            fetch=fake_fetch,
        )


def test_download_scrfd_model_cli_prints_export_line(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = load_download_cli_module()
    model_path = tmp_path / "scrfd.onnx"
    manifest_path = tmp_path / "scrfd.manifest.json"
    calls: list[dict[str, object]] = []

    def fake_download(**kwargs: object) -> object:
        calls.append(dict(kwargs))
        return SimpleNamespace(
            path=model_path,
            manifest_path=manifest_path,
            bytes=12,
            sha256="abc123",
            source_url="https://example.invalid/scrfd.onnx",
            reused_existing=False,
        )

    monkeypatch.setattr(module, "download_scrfd_model", fake_download)

    module.main(
        [
            "--output",
            str(model_path),
            "--url",
            "https://example.invalid/scrfd.onnx",
        ]
    )

    output = capsys.readouterr().out
    assert calls == [
        {
            "output_path": model_path,
            "url": "https://example.invalid/scrfd.onnx",
            "force": False,
        }
    ]
    assert f"SCRFD model: {model_path}" in output
    assert f"Manifest: {manifest_path}" in output
    assert "SHA256: abc123" in output
    assert f"export KLEIN4B_SCRFD_MODEL_PATH={model_path}" in output

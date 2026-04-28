import importlib.util
import json
import zipfile
from pathlib import Path

import pytest

from klein4b.demo_dataset import (
    available_identities,
    build_split_map,
    demo_caption,
    parse_identity_from_name,
)


def load_bootstrap_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "bootstrap_demo_dataset.py"
    spec = importlib.util.spec_from_file_location("bootstrap_demo_dataset", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_demo_zip(path: Path, identities: list[str]) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for identity in identities:
            zf.writestr(f"images/{identity}_bare.jpg", f"bare-{identity}".encode())
            zf.writestr(
                f"images/{identity}_makeup_03.jpg",
                f"makeup-{identity}".encode(),
            )
    return path


def make_ffhq_makeup_zip(path: Path, identities: list[str]) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for identity in identities:
            zf.writestr(f"{identity}/", b"")
            zf.writestr(f"{identity}/bare.jpg", f"bare-{identity}".encode())
            zf.writestr(
                f"{identity}/makeup_03.jpg",
                f"makeup-{identity}".encode(),
            )
    return path


def test_parse_identity_from_name() -> None:
    assert parse_identity_from_name("images/002386_makeup_03.jpg") == "002386"
    assert parse_identity_from_name("images/002386_bare.jpg") == "002386"


def test_split_map_is_deterministic() -> None:
    identities = [f"{i:06d}" for i in range(60)]
    split_map = build_split_map(
        identities,
        train_count=40,
        val_count=10,
        test_count=10,
        seed=7,
    )
    assert split_map == {
        "train": [
            "000000",
            "000001",
            "000007",
            "000008",
            "000010",
            "000011",
            "000012",
            "000014",
            "000016",
            "000017",
            "000018",
            "000019",
            "000021",
            "000022",
            "000024",
            "000028",
            "000029",
            "000030",
            "000031",
            "000033",
            "000035",
            "000036",
            "000038",
            "000039",
            "000040",
            "000042",
            "000043",
            "000044",
            "000045",
            "000046",
            "000047",
            "000048",
            "000049",
            "000050",
            "000051",
            "000053",
            "000056",
            "000057",
            "000058",
            "000059",
        ],
        "val": [
            "000002",
            "000005",
            "000013",
            "000015",
            "000026",
            "000027",
            "000032",
            "000037",
            "000054",
            "000055",
        ],
        "test": [
            "000003",
            "000004",
            "000006",
            "000009",
            "000020",
            "000023",
            "000025",
            "000034",
            "000041",
            "000052",
        ],
    }


def test_build_split_map_requires_sixty_identities() -> None:
    identities = [f"{i:06d}" for i in range(59)]

    with pytest.raises(ValueError, match="Need at least 60 eligible identities"):
        build_split_map(
            identities,
            train_count=40,
            val_count=10,
            test_count=10,
            seed=7,
        )


def test_available_identities_supports_ffhq_makeup_archive_layout(
    tmp_path: Path,
) -> None:
    zip_path = make_ffhq_makeup_zip(
        tmp_path / "FFHQ-Makeup.zip",
        [f"{i:06d}" for i in range(60)],
    )

    identities = available_identities(zip_path)

    assert len(identities) == 60
    assert identities[:3] == ["000000", "000001", "000002"]
    assert identities[-1] == "000059"


def test_demo_caption_uses_trigger_without_style_words() -> None:
    caption = demo_caption("K4BMAKEUP03")
    assert caption.startswith("K4BMAKEUP03.")
    body = caption.partition(".")[2].lower()
    assert "makeup" not in body
    assert "lipstick" not in body


def test_bootstrap_demo_dataset_writes_clean_layout_and_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bootstrap = load_bootstrap_module()
    zip_path = make_demo_zip(
        tmp_path / "FFHQ-Makeup.zip",
        [f"{i:06d}" for i in range(60)],
    )
    data_root = tmp_path / "data_root"

    monkeypatch.setattr(bootstrap, "data_dir", lambda: data_root)
    monkeypatch.setattr(bootstrap, "download_demo_zip", lambda cache_dir: zip_path)

    bootstrap.main()

    demo_root = data_root / "demo_ffhq_makeup"
    manifest_rows = json.loads((demo_root / "manifest.json").read_text())
    assert len(manifest_rows) == 60

    split_map = build_split_map(
        available_identities(zip_path),
        train_count=40,
        val_count=10,
        test_count=10,
        seed=7,
    )
    split_sets = {split: set(ids) for split, ids in split_map.items()}
    assert split_sets["train"].isdisjoint(split_sets["val"])
    assert split_sets["train"].isdisjoint(split_sets["test"])
    assert split_sets["val"].isdisjoint(split_sets["test"])

    for row in manifest_rows:
        split = row["split"]
        identity = row["identity_id"]
        reference_path = Path(row["reference_path"])
        target_path = Path(row["target_path"])
        caption_path = row["caption_path"]

        assert identity in split_sets[split]
        assert reference_path == demo_root / "references" / split / f"{identity}.jpg"
        assert target_path == demo_root / split / f"{identity}.jpg"
        assert reference_path.exists()
        assert target_path.exists()

        if split == "train":
            assert caption_path == str(demo_root / split / f"{identity}.txt")
            assert Path(caption_path).exists()
            assert Path(caption_path).read_text() == demo_caption("K4BMAKEUP03")
        else:
            assert caption_path is None
            assert not (demo_root / split / f"{identity}.txt").exists()


def test_bootstrap_demo_dataset_supports_ffhq_makeup_archive_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bootstrap = load_bootstrap_module()
    zip_path = make_ffhq_makeup_zip(
        tmp_path / "FFHQ-Makeup.zip",
        [f"{i:06d}" for i in range(60)],
    )
    data_root = tmp_path / "data_root"

    monkeypatch.setattr(bootstrap, "data_dir", lambda: data_root)
    monkeypatch.setattr(bootstrap, "download_demo_zip", lambda cache_dir: zip_path)

    bootstrap.main()

    demo_root = data_root / "demo_ffhq_makeup"
    manifest_rows = json.loads((demo_root / "manifest.json").read_text())
    assert len(manifest_rows) == 60
    assert (demo_root / "train" / "000000.jpg").exists()
    assert (demo_root / "references" / "train" / "000000.jpg").exists()


def test_bootstrap_demo_dataset_rerun_removes_stale_generated_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bootstrap = load_bootstrap_module()
    zip_path = make_demo_zip(
        tmp_path / "FFHQ-Makeup.zip",
        [f"{i:06d}" for i in range(60)],
    )
    data_root = tmp_path / "data_root"
    cache_dir = data_root / "demo_ffhq_makeup" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_sentinel = cache_dir / "keep.txt"
    cache_sentinel.write_text("cache")

    monkeypatch.setattr(bootstrap, "data_dir", lambda: data_root)
    monkeypatch.setattr(bootstrap, "download_demo_zip", lambda cache_dir: zip_path)

    bootstrap.main()

    demo_root = data_root / "demo_ffhq_makeup"
    stale_paths = [
        demo_root / "train" / "stale.jpg",
        demo_root / "train" / "stale.txt",
        demo_root / "val" / "stale.jpg",
        demo_root / "test" / "stale.jpg",
        demo_root / "references" / "train" / "stale.jpg",
        demo_root / "references" / "val" / "stale.jpg",
        demo_root / "references" / "test" / "stale.jpg",
    ]
    for stale_path in stale_paths:
        stale_path.parent.mkdir(parents=True, exist_ok=True)
        stale_path.write_text("stale")

    bootstrap.main()

    for stale_path in stale_paths:
        assert not stale_path.exists()
    assert cache_sentinel.exists()

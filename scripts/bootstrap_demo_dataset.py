from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

from klein4b.demo_dataset import (
    available_identities,
    build_split_map,
    demo_caption,
    download_demo_zip,
    resolve_archive_members,
)
from klein4b.paths import data_dir


def clean_generated_dataset(root: Path) -> None:
    for path in [
        root / "train",
        root / "val",
        root / "test",
        root / "references",
    ]:
        if path.exists():
            shutil.rmtree(path)

    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest_path.unlink()


def main() -> None:
    root = data_dir() / "demo_ffhq_makeup"
    train_dir = root / "train"
    cache_dir = root / "cache"
    train_dir.mkdir(parents=True, exist_ok=True)

    zip_path = download_demo_zip(cache_dir)
    identities = available_identities(zip_path)
    split_map = build_split_map(
        identities,
        train_count=40,
        val_count=10,
        test_count=10,
        seed=7,
    )
    clean_generated_dataset(root)
    train_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    trigger = "K4BMAKEUP03"

    with zipfile.ZipFile(zip_path) as zf:
        archive_names = set(zf.namelist())
        for split, ids in split_map.items():
            split_dir = root / split
            refs_dir = root / "references" / split
            split_dir.mkdir(parents=True, exist_ok=True)
            refs_dir.mkdir(parents=True, exist_ok=True)
            for identity in ids:
                members = resolve_archive_members(archive_names, identity)
                if members is None:
                    raise FileNotFoundError(f"Missing archive members for identity {identity}.")
                bare_name, target_name = members
                ref_path = refs_dir / f"{identity}.jpg"
                tgt_path = split_dir / f"{identity}.jpg"
                txt_path = split_dir / f"{identity}.txt"

                ref_path.write_bytes(zf.read(bare_name))
                if split == "train":
                    tgt_path.write_bytes(zf.read(target_name))
                    txt_path.write_text(demo_caption(trigger))
                    caption_path = str(txt_path)
                else:
                    tgt_path.write_bytes(zf.read(target_name))
                    caption_path = None

                manifest_rows.append(
                    {
                        "split": split,
                        "identity_id": identity,
                        "reference_path": str(ref_path),
                        "target_path": str(tgt_path),
                        "caption_path": caption_path,
                    }
                )

    (root / "manifest.json").write_text(
        json.dumps(manifest_rows, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

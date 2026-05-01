from __future__ import annotations

import random
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

DATASET_REPO = "cyberagent/FFHQ-Makeup"


def parse_identity_from_name(path: str) -> str:
    path_obj = Path(path)
    parent = path_obj.parent.name
    if parent and parent != "." and path_obj.name in {"bare.jpg", "makeup_03.jpg"}:
        return parent
    return path_obj.name.split("_")[0]


def resolve_archive_members(
    names: set[str],
    identity: str,
    style_slot: str = "makeup_03",
) -> tuple[str, str] | None:
    candidates = [
        (
            f"images/{identity}_bare.jpg",
            f"images/{identity}_{style_slot}.jpg",
        ),
        (
            f"{identity}/bare.jpg",
            f"{identity}/{style_slot}.jpg",
        ),
    ]
    for bare_name, target_name in candidates:
        if bare_name in names and target_name in names:
            return bare_name, target_name
    return None


def build_split_map(
    identities: list[str],
    train_count: int,
    val_count: int,
    test_count: int,
    seed: int,
) -> dict[str, list[str]]:
    required = train_count + val_count + test_count
    if len(identities) < required:
        raise ValueError(f"Need at least {required} eligible identities, found {len(identities)}.")
    ordered = sorted(identities)
    random.Random(seed).shuffle(ordered)
    return {
        "train": sorted(ordered[:train_count]),
        "val": sorted(ordered[train_count : train_count + val_count]),
        "test": sorted(ordered[train_count + val_count : required]),
    }


def demo_caption(trigger_word: str) -> str:
    return (
        f"{trigger_word}. A centered close-up portrait of a person facing the camera, "
        "with shoulders visible, neutral framing, even studio lighting, and a simple background."
    )


def download_demo_zip(cache_dir: Path) -> Path:
    return Path(
        hf_hub_download(
            repo_id=DATASET_REPO,
            repo_type="dataset",
            filename="FFHQ-Makeup.zip",
            local_dir=cache_dir,
            local_dir_use_symlinks=False,
        )
    )


def available_identities(zip_path: Path, style_slot: str = "makeup_03") -> list[str]:
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
    identities: list[str] = []
    for name in sorted(names):
        if not (name.endswith(f"_{style_slot}.jpg") or name.endswith(f"/{style_slot}.jpg")):
            continue
        identity = parse_identity_from_name(name)
        if resolve_archive_members(names, identity, style_slot) is not None:
            identities.append(identity)
    return identities

import json
from dataclasses import dataclass
from pathlib import Path

from klein4b.paths import repo_root


@dataclass(frozen=True)
class AIToolkitLock:
    repo_url: str
    commit: str
    torch: dict[str, str]


def load_ai_toolkit_lock(path: Path) -> AIToolkitLock:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return AIToolkitLock(
        repo_url=payload["repo_url"],
        commit=payload["commit"],
        torch=payload["torch"],
    )


def ai_toolkit_lock_path() -> Path:
    return repo_root() / "configs" / "ai_toolkit.lock.json"


def torch_install_command() -> str:
    lock = load_ai_toolkit_lock(ai_toolkit_lock_path())
    torch_cfg = lock.torch
    return (
        "pip install --no-cache-dir "
        f"torch=={torch_cfg['version']} "
        f"torchvision=={torch_cfg['torchvision']} "
        f"torchaudio=={torch_cfg['torchaudio']} "
        f"--index-url {torch_cfg['index_url']}"
    )

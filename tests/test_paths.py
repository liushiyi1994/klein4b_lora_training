from pathlib import Path

from klein4b.paths import data_dir, outputs_dir, repo_root


def test_repo_root_contains_agents_md() -> None:
    root: Path = repo_root()
    assert root.name == "klein4b"
    assert (root / "AGENTS.md").exists()


def test_standard_runtime_directories_are_root_relative() -> None:
    root: Path = repo_root()
    assert data_dir() == root / "data"
    assert outputs_dir() == root / "outputs"

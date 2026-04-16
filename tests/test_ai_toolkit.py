from pathlib import Path

from klein4b.ai_toolkit import (
    AIToolkitLock,
    load_ai_toolkit_lock,
    torch_install_command,
)


def test_lock_file_has_expected_repo_and_commit() -> None:
    lock = load_ai_toolkit_lock(Path("configs/ai_toolkit.lock.json"))
    assert isinstance(lock, AIToolkitLock)
    assert lock.repo_url == "https://github.com/ostris/ai-toolkit.git"
    assert lock.commit == "8f67f5022e5ea39e8325cddb32fd6a0ab46a0813"
    assert lock.torch == {
        "version": "2.9.1",
        "torchvision": "0.24.1",
        "torchaudio": "2.9.1",
        "index_url": "https://download.pytorch.org/whl/cu128",
    }


def test_torch_install_command_targets_cu128() -> None:
    command = torch_install_command()
    assert command == (
        "pip install --no-cache-dir "
        "torch==2.9.1 "
        "torchvision==0.24.1 "
        "torchaudio==2.9.1 "
        "--index-url https://download.pytorch.org/whl/cu128"
    )


def test_shell_scripts_are_lock_driven() -> None:
    setup_script = Path("scripts/setup_local_env.sh").read_text(encoding="utf-8")
    bootstrap_script = Path("scripts/bootstrap_ai_toolkit.sh").read_text(encoding="utf-8")

    assert "configs/ai_toolkit.lock.json" in setup_script
    assert "configs/ai_toolkit.lock.json" in bootstrap_script
    assert "BASH_SOURCE" in setup_script
    assert "BASH_SOURCE" in bootstrap_script
    assert "Missing virtual environment" in bootstrap_script
    assert "python3 -m virtualenv" in setup_script
    assert "python3.12 -m venv" in setup_script
    assert "pip install virtualenv" in setup_script
    assert "torch==2.9.1" not in setup_script
    assert "https://github.com/ostris/ai-toolkit.git" not in setup_script
    assert "8f67f5022e5ea39e8325cddb32fd6a0ab46a0813" not in bootstrap_script

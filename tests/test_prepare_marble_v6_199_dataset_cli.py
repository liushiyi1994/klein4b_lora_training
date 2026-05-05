from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


def load_cli_module():
    module_path = (
        Path(__file__).resolve().parents[1] / "scripts" / ("prepare_marble_v6_199_dataset.py")
    )
    spec = importlib.util.spec_from_file_location("prepare_marble_v6_199_dataset", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_help_exits_without_building_dataset() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/prepare_marble_v6_199_dataset.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--source" in result.stdout
    assert "--target" in result.stdout


def test_cli_calls_builder_with_explicit_paths(monkeypatch, tmp_path: Path) -> None:
    module = load_cli_module()
    source = tmp_path / "charliestale-ml" / "data-synthesis"
    target = tmp_path / "v6_199"
    calls: list[dict[str, object]] = []

    class Report:
        row_count = 199
        archetype_counts = {"ctWinged + ctOrnateCuirass": 5}

    def fake_build_v6_199_dataset(**kwargs: object) -> object:
        calls.append(dict(kwargs))
        return Report()

    monkeypatch.setattr(module, "build_v6_199_dataset", fake_build_v6_199_dataset)

    exit_code = module.main(["--source", str(source), "--target", str(target)])

    assert exit_code == 0
    assert calls == [{"source_dir": source, "target_dir": target, "expected_count": 199}]


def test_cli_supports_expected_count_override_for_smoke_tests(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = load_cli_module()
    calls: list[dict[str, object]] = []

    class Report:
        row_count = 2
        archetype_counts = {}

    def fake_build_v6_199_dataset(**kwargs: object) -> object:
        calls.append(dict(kwargs))
        return Report()

    monkeypatch.setattr(module, "build_v6_199_dataset", fake_build_v6_199_dataset)

    exit_code = module.main(
        [
            "--source",
            str(tmp_path / "source"),
            "--target",
            str(tmp_path / "target"),
            "--expected-count",
            "2",
        ]
    )

    assert exit_code == 0
    assert calls[0]["expected_count"] == 2

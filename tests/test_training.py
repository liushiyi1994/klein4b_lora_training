import importlib.util
import subprocess
import sys
from pathlib import Path

from klein4b.training import build_training_command, render_training_config


def test_render_training_config_uses_flux2_klein_base_4b(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "demo_ffhq_makeup" / "train"
    dataset_dir.mkdir(parents=True)
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    template_path = tmp_path / "train.template.yaml"
    template_path.write_text(
        (
            "job: extension\n"
            "config:\n"
            '  dataset: "{{ dataset_dir }}"\n'
            '  output: "{{ training_folder }}"\n'
            '  model: "black-forest-labs/FLUX.2-klein-base-4B"\n'
            '  trigger: "K4BMAKEUP03"\n'
            "  steps: 1800\n"
            "  resolution:\n"
            "    - 512\n"
        ),
        encoding="utf-8",
    )

    config_text = render_training_config(
        template_path=template_path,
        dataset_dir=dataset_dir,
        output_dir=output_dir,
    )

    assert "black-forest-labs/FLUX.2-klein-base-4B" in config_text
    assert 'trigger: "K4BMAKEUP03"' in config_text
    assert "steps: 1800" in config_text
    assert "resolution:" in config_text
    assert str(dataset_dir) in config_text
    assert str(output_dir) in config_text


def test_render_training_config_sets_flux2_klein_arch_defaults(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "demo_ffhq_makeup" / "train"
    dataset_dir.mkdir(parents=True)
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    template_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "train_flux2_klein_makeup_demo.template.yaml"
    )

    config_text = render_training_config(
        template_path=template_path,
        dataset_dir=dataset_dir,
        output_dir=output_dir,
    )

    assert 'arch: "flux2_klein_4b"' in config_text
    assert 'noise_scheduler: "flowmatch"' in config_text
    assert 'sampler: "flowmatch"' in config_text
    assert "quantize_te: true" in config_text
    assert 'qtype: "qfloat8"' in config_text


def test_build_training_command_points_to_ai_toolkit_run_py() -> None:
    command = build_training_command(
        Path("vendor/ai-toolkit"),
        Path("outputs/runs/demo/train_config.yaml"),
    )
    assert command == [
        sys.executable,
        "vendor/ai-toolkit/run.py",
        "outputs/runs/demo/train_config.yaml",
    ]


def test_train_lora_cli_help_exits_without_starting_training() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/train_lora.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "usage:" in result.stdout
    assert "--config" in result.stdout


def test_train_lora_cli_writes_config_from_custom_inputs(monkeypatch, tmp_path: Path) -> None:
    run_calls: dict[str, object] = {}
    template_path = tmp_path / "train.template.yaml"
    template_path.write_text(
        (
            "job: extension\n"
            "config:\n"
            '  dataset: "{{ dataset_dir }}"\n'
            '  output: "{{ training_folder }}"\n'
        ),
        encoding="utf-8",
    )
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    output_root = tmp_path / "runs"
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "train_lora.py"
    spec = importlib.util.spec_from_file_location("task_train_lora", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def fake_run(command: list[str], check: bool) -> None:
        run_calls["command"] = command
        run_calls["check"] = check

    class FixedDatetime:
        @staticmethod
        def now(_tz):
            class FixedNow:
                @staticmethod
                def strftime(_fmt: str) -> str:
                    return "20260416-120000"

            return FixedNow()

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(module, "datetime", FixedDatetime)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_lora.py",
            "--config",
            str(template_path),
            "--dataset-dir",
            str(dataset_dir),
            "--output-root",
            str(output_root),
        ],
    )

    module.main()

    run_dir = output_root / "20260416-120000"
    config_path = run_dir / "train_config.yaml"
    assert config_path.exists()
    assert config_path.read_text(encoding="utf-8") == (
        f'job: extension\nconfig:\n  dataset: "{dataset_dir}"\n  output: "{run_dir}"\n'
    )
    assert run_calls["check"] is True
    assert run_calls["command"] == [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "vendor" / "ai-toolkit" / "run.py"),
        str(config_path),
    ]

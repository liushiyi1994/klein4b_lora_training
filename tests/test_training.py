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


def test_render_training_config_sets_marble_bust_defaults(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "marble-bust-data" / "v1" / "busts"
    dataset_dir.mkdir(parents=True)
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    template_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "train_flux2_klein_marble_bust.template.yaml"
    )

    config_text = render_training_config(
        template_path=template_path,
        dataset_dir=dataset_dir,
        output_dir=output_dir,
    )

    assert 'trigger_word: "<mrblbust>"' in config_text
    assert "steps: 2000" in config_text
    assert "linear: 32" in config_text
    assert "conv: 16" in config_text
    assert "width: 768" in config_text
    assert "height: 1024" in config_text
    assert "resolution:" in config_text
    assert "- 768" in config_text
    assert "marble statue bust shown chest-up" in config_text
    assert "visible shoulders and upper torso" in config_text
    assert "head-only crop" in config_text
    assert "stone-carved marble hair" in config_text
    assert "broad chiseled grooves" in config_text
    assert "individual hair strands" in config_text
    assert "natural hair color" in config_text
    assert str(dataset_dir) in config_text
    assert str(output_dir) in config_text


def test_render_training_config_sets_marble_v3_matte_lava_defaults(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "marble-bust-data" / "v3" / "busts"
    dataset_dir.mkdir(parents=True)
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    template_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "train_flux2_klein_marble_bust_v3_matte_lava.template.yaml"
    )

    config_text = render_training_config(
        template_path=template_path,
        dataset_dir=dataset_dir,
        output_dir=output_dir,
    )

    assert 'trigger_word: "<mrblbust>"' in config_text
    assert "marble_bust_v3_matte_lava_style" in config_text
    assert "steps: 2000" in config_text
    assert "matte weathered grey-white marble" in config_text
    assert "dry chalky unpolished stone" in config_text
    assert "small localized orange-red lava fissures" in config_text
    assert "lava glow does not illuminate face hair torso or background" in config_text
    assert "polished marble" in config_text
    assert "specular hotspots" in config_text
    assert "overexposed white face" in config_text
    assert "red or orange light on face" in config_text
    assert str(dataset_dir) in config_text
    assert str(output_dir) in config_text


def test_render_training_config_sets_marble_v4_weathered_face_defaults(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "marble-bust-data" / "v4_manual_weathered_face" / "busts"
    dataset_dir.mkdir(parents=True)
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    template_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "train_flux2_klein_marble_bust_v4_manual_weathered_face.template.yaml"
    )

    config_text = render_training_config(
        template_path=template_path,
        dataset_dir=dataset_dir,
        output_dir=output_dir,
    )

    assert 'trigger_word: "<mrblbust>"' in config_text
    assert "marble_bust_v4_manual_weathered_face_style" in config_text
    assert "dark ash-grey weathered marble" in config_text
    assert "dirty brown-grey mineral patina across the face" in config_text
    assert "rough pitted low-albedo face surface" in config_text
    assert "no frontal studio key light" in config_text
    assert "no perfect beauty lighting" in config_text
    assert "no separate rock pedestal" in config_text
    assert "no loose rocks below" in config_text
    assert "smooth porcelain face" in config_text
    assert "shiny forehead" in config_text
    assert str(dataset_dir) in config_text
    assert str(output_dir) in config_text


def test_render_training_config_sets_marble_v5_clothing_armor_defaults(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "ignored_for_paired_v5"
    dataset_dir.mkdir()
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    template_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "train_flux2_klein_marble_bust_v5_clothing_armor_rank64_unquantized.template.yaml"
    )

    config_text = render_training_config(
        template_path=template_path,
        dataset_dir=dataset_dir,
        output_dir=output_dir,
    )

    assert 'trigger_word: "<mrblbust>"' in config_text
    assert "marble_bust_v5_clothing_armor_rank64_unquantized_style" in config_text
    assert 'folder_path: "data/marble-bust-data/v5_new/bust_img"' in config_text
    assert 'control_path_1: "data/marble-bust-data/v5_new/ref_input"' in config_text
    assert "steps: 10000" in config_text
    assert "save_every: 200" in config_text
    assert "max_step_saves_to_keep: 50" in config_text
    assert "linear: 64" in config_text
    assert "conv: 32" in config_text
    assert "lr: 0.00006" in config_text
    assert "skip_first_sample: true" in config_text
    assert "train_text_encoder: false" in config_text
    assert "quantize: false" in config_text
    assert "quantize_te: false" in config_text
    assert "width: 768" in config_text
    assert "height: 1024" in config_text
    assert "guidance_scale: 2.5" in config_text
    assert "sample_steps: 24" in config_text
    assert "african_child_female_1.png" in config_text
    assert "eastasian_middleaged_male_1.png" in config_text
    assert "hispaniclatino_elderly_neutral_1.png" in config_text
    assert "middleeastern_youngadult_female_2.png" in config_text
    assert "southasian_middleaged_neutral_3.png" in config_text
    assert "head-only crop" in config_text
    assert "modern accessories" in config_text
    assert str(output_dir) in config_text


def test_render_training_config_sets_marble_v6_199_rank96_defaults(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "ignored_for_paired_v6"
    dataset_dir.mkdir()
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    template_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "train_flux2_klein_marble_bust_v6_199_rank96_unquantized.template.yaml"
    )

    config_text = render_training_config(
        template_path=template_path,
        dataset_dir=dataset_dir,
        output_dir=output_dir,
    )

    assert 'trigger_word: "<mrblbust>"' in config_text
    assert "marble_bust_v6_199_rank96_unquantized_style" in config_text
    assert 'folder_path: "data/marble-bust-data/v6_199/dataset"' in config_text
    assert 'control_path_1: "data/marble-bust-data/v6_199/control"' in config_text
    assert "steps: 16000" in config_text
    assert "save_every: 400" in config_text
    assert "sample_every: 400" in config_text
    assert "max_step_saves_to_keep: 50" in config_text
    assert "linear: 96" in config_text
    assert "linear_alpha: 96" in config_text
    assert "conv: 48" in config_text
    assert "conv_alpha: 48" in config_text
    assert "lr: 0.00005" in config_text
    assert "train_text_encoder: false" in config_text
    assert "quantize: false" in config_text
    assert "quantize_te: false" in config_text
    assert "guidance_scale: 2.5" in config_text
    assert "sample_steps: 24" in config_text
    assert "data/marble-bust-data/v6_199/control/001.png" in config_text
    assert "data/marble-bust-data/v6_199/control/100.png" in config_text
    assert "data/marble-bust-data/v6_199/control/199.png" in config_text
    assert "winged helmet with a tall crest" in config_text
    assert "ignore and remove any hat or modern headwear from the reference image" in config_text
    assert "tall crested Attic Athena helmet" in config_text
    assert "carved Corinthian helmet with cheek guards" in config_text
    assert str(output_dir) in config_text


def test_render_training_config_sets_marble_v7_30_klein9b_defaults(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "ignored_for_paired_v7"
    dataset_dir.mkdir()
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    template_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "train_flux2_klein9b_marble_bust_v7_30_rank64_unquantized.template.yaml"
    )

    config_text = render_training_config(
        template_path=template_path,
        dataset_dir=dataset_dir,
        output_dir=output_dir,
    )

    assert 'trigger_word: "<mrblbust>"' in config_text
    assert "marble_bust_v7_30_klein9b_rank64_unquantized_style" in config_text
    assert 'folder_path: "data/marble-bust-data/v7_30/dataset"' in config_text
    assert 'control_path_1: "data/marble-bust-data/v7_30/control"' in config_text
    assert "steps: 3600" in config_text
    assert "save_every: 200" in config_text
    assert "sample_every: 200" in config_text
    assert "max_step_saves_to_keep: 30" in config_text
    assert "linear: 64" in config_text
    assert "linear_alpha: 64" in config_text
    assert "conv: 32" in config_text
    assert "conv_alpha: 32" in config_text
    assert "lr: 0.00004" in config_text
    assert 'content_or_style: "style"' in config_text
    assert 'arch: "flux2_klein_9b"' in config_text
    assert 'name_or_path: "black-forest-labs/FLUX.2-klein-base-9B"' in config_text
    assert "quantize: false" in config_text
    assert "quantize_te: false" in config_text
    assert "low_vram: false" in config_text
    assert "qfloat8" not in config_text
    assert "guidance_scale: 2.5" in config_text
    assert "sample_steps: 24" in config_text
    assert "data/marble-bust-data/v7_30/control/001.png" in config_text
    assert "data/marble-bust-data/v7_30/control/015.png" in config_text
    assert "data/marble-bust-data/v7_30/control/030.png" in config_text
    assert "featureless blank marble eye surfaces" in config_text
    assert "identity-specific names or biographical details" in config_text
    assert "head-only crop" in config_text
    assert "modern accessories" in config_text
    assert str(output_dir) in config_text


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

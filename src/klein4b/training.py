import sys
from pathlib import Path

from klein4b.paths import repo_root


def default_training_template_path() -> Path:
    return repo_root() / "configs" / "train_flux2_klein_makeup_demo.template.yaml"


def render_training_config(template_path: Path, dataset_dir: Path, output_dir: Path) -> str:
    template = template_path.read_text(encoding="utf-8")
    return (
        template.replace("{{ training_folder }}", str(output_dir)).replace(
            "{{ dataset_dir }}", str(dataset_dir)
        )
    )


def build_training_command(ai_toolkit_dir: Path, config_path: Path) -> list[str]:
    return [sys.executable, str(ai_toolkit_dir / "run.py"), str(config_path)]

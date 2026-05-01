# Structured Marble Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a selfie-only VLM prompt-planning pipeline that renders a deterministic Klein marble prompt, runs AI Toolkit sample-style inference, and compares the result against the previous best output.

**Architecture:** Keep the feature in three small units: prompt-plan JSON/schema/rendering, AI Toolkit sample-style YAML/execution helpers, and a CLI that composes them. The OpenAI planner receives exactly one image input: the reference selfie. Pseudo target images are comparison-only artifacts and must never affect JSON planning or prompt rendering.

**Tech Stack:** Python 3.12, pytest, PyYAML, Pillow, OpenAI Python SDK Responses API, AI Toolkit `vendor/ai-toolkit/run.py`.

---

## File Structure

- Modify `requirements.txt`: add the OpenAI Python SDK.
- Modify `src/klein4b/inference.py`: restore `build_marble_pair_prompt` so existing committed sweep tests pass.
- Modify `tests/test_paths.py`: make the repo-root test work in global git worktrees.
- Create `src/klein4b/marble_prompt_planning.py`: prompt-plan schema, dataclasses, validation, deterministic prompt renderer, OpenAI request builder/caller.
- Create `src/klein4b/sample_style_inference.py`: best-setting constants, negative prompt, AI Toolkit YAML renderer, AI Toolkit command builder, sample discovery/copy, contact sheet creation.
- Create `scripts/run_structured_marble_inference.py`: CLI entry point.
- Create `tests/test_marble_prompt_planning.py`: unit tests for schema, validation, renderer, and OpenAI request shape.
- Create `tests/test_sample_style_inference.py`: unit tests for YAML rendering and output helper behavior.
- Create `tests/test_structured_marble_inference_cli.py`: unit tests for cached JSON replay, pseudo-target comparison-only handling, and run-config writing.

---

### Task 1: Restore Clean Baseline Tests

**Files:**
- Modify: `src/klein4b/inference.py`
- Modify: `tests/test_paths.py`
- Test: `tests/test_eval_marble_v4_checkpoint_sweep.py`
- Test: `tests/test_paths.py`

- [ ] **Step 1: Run the existing failing baseline tests**

Run:

```bash
pytest tests/test_eval_marble_v4_checkpoint_sweep.py tests/test_paths.py -q
```

Expected: FAIL because `build_marble_pair_prompt` is missing and `test_repo_root_contains_agents_md` expects the checkout directory name to be `klein4b`.

- [ ] **Step 2: Restore the marble pair prompt helper**

Add this function to `src/klein4b/inference.py` after `build_demo_prompt`:

```python
def build_marble_pair_prompt(pair_caption: str) -> str:
    prompt = pair_caption.strip()
    if prompt.startswith("transform into"):
        prompt = prompt.replace("transform into", "Change image 1 into", 1)
    return (
        f"{prompt}. Do not preserve skin material, selfie lighting, skin shine, "
        "makeup shine, wet lips, or catchlights from the reference. Use a rough "
        "pitted low-albedo face surface with dry chalky unpolished stone, uneven "
        "grey-brown mineral patina, subdued off-axis ambient lighting, and "
        "asymmetrical stone lighting with one side of the face slightly darker "
        "than the other, shadowed eye sockets, and shallow carved shadows under "
        "the brow ridge, nose, lower lip, and chin; no full-face even "
        "illumination, no frontal studio light, no head-on key light, no "
        "perfect portrait lighting, no smooth beauty-render face, no polished "
        "cheeks, no shiny forehead, no shiny nose, no shiny lips, no glossy "
        "polished marble, no wet shine, no specular hotspots. The eyes must be "
        "blank white sculpted stone eyes or closed carved eyelids; no pupils, "
        "no irises, no colored eyes, no catchlights. Hair must be carved from "
        "the same marble as the face. Use hairstyle only as sculptural shape "
        "with broad chiseled grooves, solid stone masses, and marble veins "
        "continuing through the hair; ignore natural hair color from the "
        "reference and preserve only the hairstyle silhouette, braid shapes, "
        "and carved lock structure; no individual hair strands, no natural "
        "hair color, no glossy hair, no soft hair strands."
    )
```

- [ ] **Step 3: Make the path test work in any worktree directory**

Change `tests/test_paths.py::test_repo_root_contains_agents_md` to:

```python
def test_repo_root_contains_agents_md() -> None:
    root: Path = repo_root()
    assert (root / "AGENTS.md").exists()
    assert (root / "pyproject.toml").exists()
```

- [ ] **Step 4: Verify baseline fixes**

Run:

```bash
pytest tests/test_eval_marble_v4_checkpoint_sweep.py tests/test_paths.py -q
```

Expected: all selected tests PASS.

- [ ] **Step 5: Commit baseline fixes**

Run:

```bash
git add src/klein4b/inference.py tests/test_paths.py
git commit -m "test: fix inference worktree baseline"
```

---

### Task 2: Add Prompt Plan Validation And Rendering

**Files:**
- Create: `src/klein4b/marble_prompt_planning.py`
- Create: `tests/test_marble_prompt_planning.py`

- [ ] **Step 1: Write failing tests for validation and rendering**

Create `tests/test_marble_prompt_planning.py` with:

```python
from __future__ import annotations

import pytest

from klein4b.marble_prompt_planning import (
    PromptPlanError,
    build_prompt_plan_schema,
    parse_prompt_plan,
    render_marble_prompt,
)


VALID_PLAN = {
    "reference_identity": {
        "age_band": "child",
        "gender_presentation": "feminine",
        "head_pose": "slight left-facing three-quarter head pose",
        "face_structure": ["rounded cheeks", "small compact jaw", "short nose"],
        "hair_or_headwear": ["short bob-like hair silhouette with soft bangs"],
        "broad_expression": "calm closed-mouth expression",
    },
    "target_style": {
        "bust_framing": "centered chest-up Greek marble statue bust",
        "statue_angle": "matching the reference head angle",
        "drapery_and_torso": ["simple classical drapery", "visible shoulders"],
        "headpiece_or_ornament": [],
        "stone_surface": ["matte weathered grey marble", "rough pitted low-albedo stone"],
        "weathering": ["grey-brown mineral patina", "grime in recesses"],
        "base_and_lava": ["broken lower bust base with localized lava only in cracks"],
        "background": "dark ember background",
    },
    "safety_overrides": {
        "identity_source_policy": "use only the reference portrait for identity",
        "eye_policy": "blank sculpted stone eyes or closed carved eyelids",
        "material_policy": "all hair and ornaments are carved marble",
        "banned_details": ["pupils", "irises", "catchlights"],
    },
}


def test_prompt_plan_schema_requires_selfie_only_sections() -> None:
    schema = build_prompt_plan_schema()

    assert schema["name"] == "marble_prompt_plan"
    assert schema["strict"] is True
    assert set(schema["schema"]["properties"]) == {
        "reference_identity",
        "target_style",
        "safety_overrides",
    }
    assert "pseudo_target" not in str(schema)


def test_parse_prompt_plan_rejects_race_or_ethnicity_descriptors() -> None:
    payload = {
        **VALID_PLAN,
        "reference_identity": {
            **VALID_PLAN["reference_identity"],
            "face_structure": ["East Asian face shape"],
        },
    }

    with pytest.raises(PromptPlanError, match="race or ethnicity"):
        parse_prompt_plan(payload)


def test_render_marble_prompt_is_final_image_prompt_with_fixed_constraints() -> None:
    plan = parse_prompt_plan(VALID_PLAN)

    prompt = render_marble_prompt(plan)

    assert prompt.startswith("Change image 1 into a <mrblbust> from the reference portrait,")
    assert "child" in prompt
    assert "rounded cheeks" in prompt
    assert "short bob-like hair silhouette" in prompt
    assert "centered chest-up Greek marble statue bust" in prompt
    assert "reference portrait as the only identity" in prompt
    assert "blank sculpted stone eyes or closed carved eyelids" in prompt
    assert "no pupils" in prompt
    assert "no irises" in prompt
    assert "no catchlights" in prompt
    assert "same marble as the face" in prompt
    assert "localized lava only in the broken lower base" in prompt
    assert "pseudo target" not in prompt.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_marble_prompt_planning.py -q
```

Expected: FAIL with `ModuleNotFoundError` for `klein4b.marble_prompt_planning`.

- [ ] **Step 3: Implement prompt plan schema, validation, and renderer**

Create `src/klein4b/marble_prompt_planning.py` with dataclasses `ReferenceIdentity`, `TargetStyle`, `SafetyOverrides`, `PromptPlan`; a `PromptPlanError`; `build_prompt_plan_schema`; `parse_prompt_plan`; and `render_marble_prompt`.

Use this validation rule:

```python
FORBIDDEN_IDENTITY_LABELS = (
    "african",
    "asian",
    "black",
    "caucasian",
    "east asian",
    "ethnicity",
    "european",
    "hispanic",
    "latino",
    "middle eastern",
    "race",
    "south asian",
    "white",
)
```

The renderer must join list fields with comma-separated phrases and always append fixed constraints:

```python
FIXED_PROMPT_CONSTRAINTS = (
    "Use the reference portrait as the only identity, pose, face-structure, "
    "hair silhouette, headwear silhouette, and broad-expression source. "
    "The eyes must be blank sculpted stone eyes or closed carved eyelids; "
    "no pupils, no irises, no colored eyes, no catchlights, no painted eyes, "
    "no realistic human eyes. Hair, eyebrows, facial hair, head coverings, "
    "and ornaments must be carved from the same marble as the face. Use matte "
    "weathered grey marble, rough pitted low-albedo stone, grey-brown mineral "
    "patina, grime in recesses, chipped edges, and localized lava only in the "
    "broken lower base. Avoid glossy marble, wet shine, specular hotspots, "
    "selfie lighting, beauty lighting, modern accessories, duplicate figures, "
    "side-by-side views, and collage."
)
```

- [ ] **Step 4: Verify prompt planning tests pass**

Run:

```bash
pytest tests/test_marble_prompt_planning.py -q
```

Expected: all tests PASS.

- [ ] **Step 5: Commit prompt planning core**

Run:

```bash
git add src/klein4b/marble_prompt_planning.py tests/test_marble_prompt_planning.py
git commit -m "feat: add marble prompt plan renderer"
```

---

### Task 3: Add Selfie-Only OpenAI Planner Request

**Files:**
- Modify: `src/klein4b/marble_prompt_planning.py`
- Modify: `tests/test_marble_prompt_planning.py`
- Modify: `requirements.txt`

- [ ] **Step 1: Write failing OpenAI request tests**

Append to `tests/test_marble_prompt_planning.py`:

```python
from pathlib import Path


def test_openai_planner_sends_exactly_one_image_input(tmp_path: Path) -> None:
    from klein4b.marble_prompt_planning import plan_prompt_with_openai

    calls: dict[str, object] = {}
    reference = tmp_path / "selfie.jpg"
    reference.write_bytes(b"fake-image")

    class FakeResponses:
        def create(self, **kwargs: object) -> object:
            calls["kwargs"] = kwargs

            class Response:
                output_text = __import__("json").dumps(VALID_PLAN)

            return Response()

    class FakeClient:
        responses = FakeResponses()

    plan = plan_prompt_with_openai(
        reference_path=reference,
        model="gpt-5.4-mini",
        client=FakeClient(),
    )

    kwargs = calls["kwargs"]
    assert kwargs["model"] == "gpt-5.4-mini"
    assert kwargs["text"]["format"]["type"] == "json_schema"
    assert kwargs["text"]["format"]["strict"] is True
    content = kwargs["input"][0]["content"]
    image_items = [item for item in content if item["type"] == "input_image"]
    assert len(image_items) == 1
    assert image_items[0]["image_url"].startswith("data:image/jpeg;base64,")
    assert "pseudo" not in str(kwargs).lower()
    assert plan.reference_identity.age_band == "child"
```

- [ ] **Step 2: Run the OpenAI request test to verify it fails**

Run:

```bash
pytest tests/test_marble_prompt_planning.py::test_openai_planner_sends_exactly_one_image_input -q
```

Expected: FAIL because `plan_prompt_with_openai` does not exist.

- [ ] **Step 3: Implement the OpenAI planner**

Add to `src/klein4b/marble_prompt_planning.py`:

```python
def plan_prompt_with_openai(reference_path: Path, model: str = "gpt-5.4-mini", client: object | None = None) -> PromptPlan:
    if client is None:
        from openai import OpenAI

        client = OpenAI()

    image_url = image_path_to_data_url(reference_path)
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": PLANNER_INSTRUCTIONS},
                    {"type": "input_image", "image_url": image_url},
                ],
            }
        ],
        text={"format": {"type": "json_schema", **build_prompt_plan_schema()}},
    )
    return parse_prompt_plan(json.loads(response.output_text))
```

Also add `image_path_to_data_url` using `mimetypes.guess_type` and base64 encoding. `PLANNER_INSTRUCTIONS` must explicitly say the model sees only the selfie/reference image and must not infer from any pseudo target image.

- [ ] **Step 4: Add OpenAI SDK requirement**

Append to `requirements.txt`:

```text
openai>=2.0.0
```

- [ ] **Step 5: Verify prompt planning tests pass**

Run:

```bash
pytest tests/test_marble_prompt_planning.py -q
```

Expected: all tests PASS.

- [ ] **Step 6: Commit OpenAI planner**

Run:

```bash
git add requirements.txt src/klein4b/marble_prompt_planning.py tests/test_marble_prompt_planning.py
git commit -m "feat: add selfie-only OpenAI prompt planner"
```

---

### Task 4: Add AI Toolkit Sample-Style YAML Helpers

**Files:**
- Create: `src/klein4b/sample_style_inference.py`
- Create: `tests/test_sample_style_inference.py`

- [ ] **Step 1: Write failing YAML rendering tests**

Create `tests/test_sample_style_inference.py` with:

```python
from __future__ import annotations

from pathlib import Path

import yaml

from klein4b.sample_style_inference import (
    DEFAULT_BEST_LORA_PATH,
    build_ai_toolkit_command,
    render_sample_style_config,
)


def test_render_sample_style_config_uses_best_klein_settings(tmp_path: Path) -> None:
    reference = tmp_path / "selfie.png"
    lora = tmp_path / "weights.safetensors"
    output_dir = tmp_path / "out"

    config_text = render_sample_style_config(
        name="structured_marble_test",
        training_folder=output_dir,
        reference_path=reference,
        prompt="Change image 1 into a <mrblbust> from the reference portrait, test prompt",
        lora_path=lora,
    )
    payload = yaml.safe_load(config_text)
    process = payload["config"]["process"][0]
    sample = process["sample"]

    assert process["network"]["pretrained_lora_path"] == str(lora)
    assert process["network"]["linear"] == 64
    assert process["network"]["conv"] == 32
    assert process["train"]["steps"] == 0
    assert process["train"]["disable_sampling"] is False
    assert process["model"]["arch"] == "flux2_klein_4b"
    assert sample["width"] == 768
    assert sample["height"] == 1024
    assert sample["seed"] == 7
    assert sample["guidance_scale"] == 2.5
    assert sample["sample_steps"] == 24
    assert sample["samples"][0]["ctrl_img_1"] == str(reference)
    assert sample["samples"][0]["prompt"].startswith("Change image 1 into a <mrblbust>")
    assert "head-only crop" in sample["neg"]
    assert "individual hair strands" in sample["neg"]


def test_default_best_lora_path_points_to_step_6500() -> None:
    assert DEFAULT_BEST_LORA_PATH.name.endswith("000006500.safetensors")


def test_build_ai_toolkit_command_points_to_run_py(tmp_path: Path) -> None:
    command = build_ai_toolkit_command(tmp_path / "vendor" / "ai-toolkit", tmp_path / "config.yaml")

    assert command == [
        str(tmp_path / "vendor" / "ai-toolkit" / "run.py"),
        str(tmp_path / "config.yaml"),
    ]
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_sample_style_inference.py -q
```

Expected: FAIL with `ModuleNotFoundError` for `klein4b.sample_style_inference`.

- [ ] **Step 3: Implement sample-style helpers**

Create `src/klein4b/sample_style_inference.py` with constants for the model name, best LoRA path, negative prompt, and `render_sample_style_config`. Use `yaml.safe_dump(..., sort_keys=False)` to avoid hand-written YAML escaping bugs.

The config structure must match the best-result doc:

```python
{
    "job": "extension",
    "config": {
        "name": name,
        "process": [
            {
                "type": "diffusion_trainer",
                "training_folder": str(training_folder),
                "device": "cuda",
                "trigger_word": "<mrblbust>",
                "network": {
                    "type": "lora",
                    "linear": 64,
                    "linear_alpha": 64,
                    "conv": 32,
                    "conv_alpha": 32,
                    "pretrained_lora_path": str(lora_path),
                },
                "train": {
                    "steps": 0,
                    "skip_first_sample": True,
                    "disable_sampling": False,
                    "batch_size": 1,
                    "train_unet": True,
                    "train_text_encoder": False,
                    "gradient_checkpointing": True,
                    "noise_scheduler": "flowmatch",
                    "optimizer": "adamw8bit",
                    "dtype": "bf16",
                },
                "model": {
                    "arch": "flux2_klein_4b",
                    "name_or_path": "black-forest-labs/FLUX.2-klein-base-4B",
                    "quantize": False,
                    "quantize_te": False,
                    "low_vram": True,
                    "model_kwargs": {"match_target_res": False},
                },
                "sample": {
                    "sampler": "flowmatch",
                    "width": 768,
                    "height": 1024,
                    "samples": [{"prompt": prompt, "ctrl_img_1": str(reference_path)}],
                    "neg": DEFAULT_NEGATIVE_PROMPT,
                    "seed": 7,
                    "guidance_scale": 2.5,
                    "sample_steps": 24,
                },
            }
        ],
    },
}
```

- [ ] **Step 4: Verify YAML tests pass**

Run:

```bash
pytest tests/test_sample_style_inference.py -q
```

Expected: all tests PASS.

- [ ] **Step 5: Commit YAML helpers**

Run:

```bash
git add src/klein4b/sample_style_inference.py tests/test_sample_style_inference.py
git commit -m "feat: render sample-style Klein inference config"
```

---

### Task 5: Add Structured Inference CLI

**Files:**
- Create: `scripts/run_structured_marble_inference.py`
- Create: `tests/test_structured_marble_inference_cli.py`

- [ ] **Step 1: Write failing CLI tests for cached replay and pseudo-target isolation**

Create `tests/test_structured_marble_inference_cli.py` with:

```python
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from PIL import Image

VALID_PLAN = {
    "reference_identity": {
        "age_band": "child",
        "gender_presentation": "feminine",
        "head_pose": "slight left-facing three-quarter head pose",
        "face_structure": ["rounded cheeks", "small compact jaw", "short nose"],
        "hair_or_headwear": ["short bob-like hair silhouette with soft bangs"],
        "broad_expression": "calm closed-mouth expression",
    },
    "target_style": {
        "bust_framing": "centered chest-up Greek marble statue bust",
        "statue_angle": "matching the reference head angle",
        "drapery_and_torso": ["simple classical drapery", "visible shoulders"],
        "headpiece_or_ornament": [],
        "stone_surface": ["matte weathered grey marble", "rough pitted low-albedo stone"],
        "weathering": ["grey-brown mineral patina", "grime in recesses"],
        "base_and_lava": ["broken lower bust base with localized lava only in cracks"],
        "background": "dark ember background",
    },
    "safety_overrides": {
        "identity_source_policy": "use only the reference portrait for identity",
        "eye_policy": "blank sculpted stone eyes or closed carved eyelids",
        "material_policy": "all hair and ornaments are carved marble",
        "banned_details": ["pupils", "irises", "catchlights"],
    },
}


def load_cli_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "run_structured_marble_inference.py"
    spec = importlib.util.spec_from_file_location("run_structured_marble_inference", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_replays_cached_prompt_plan_without_openai_or_ai_toolkit(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = load_cli_module()
    reference = tmp_path / "selfie.png"
    pseudo_target = tmp_path / "pseudo.png"
    previous_best = tmp_path / "previous.jpg"
    prompt_plan_json = tmp_path / "plan.json"
    output_dir = tmp_path / "out"
    Image.new("RGB", (12, 16), "red").save(reference)
    Image.new("RGB", (12, 16), "blue").save(pseudo_target)
    Image.new("RGB", (12, 16), "green").save(previous_best)
    prompt_plan_json.write_text(json.dumps(VALID_PLAN), encoding="utf-8")

    def fail_openai(*args: object, **kwargs: object) -> object:
        raise AssertionError("OpenAI planner must not run in cached replay")

    monkeypatch.setattr(module, "plan_prompt_with_openai", fail_openai)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_structured_marble_inference.py",
            "--reference",
            str(reference),
            "--pseudo-target",
            str(pseudo_target),
            "--previous-best",
            str(previous_best),
            "--prompt-plan-json",
            str(prompt_plan_json),
            "--skip-openai",
            "--no-run-ai-toolkit",
            "--output-dir",
            str(output_dir),
        ],
    )

    module.main()

    run_config = json.loads((output_dir / "run_config.json").read_text(encoding="utf-8"))
    assert run_config["reference"] == str(reference)
    assert run_config["pseudo_target"] == str(pseudo_target)
    assert run_config["pseudo_target_role"] == "comparison-only"
    assert run_config["used_openai"] is False
    assert (output_dir / "prompt_plan.json").exists()
    assert (output_dir / "prompt.txt").read_text(encoding="utf-8").startswith(
        "Change image 1 into a <mrblbust>"
    )
    assert (output_dir / "sample_style_inference.yaml").exists()
    assert (output_dir / "contact_sheet.jpg").exists()
```

- [ ] **Step 2: Run the CLI test to verify it fails**

Run:

```bash
pytest tests/test_structured_marble_inference_cli.py -q
```

Expected: FAIL because the CLI script does not exist.

- [ ] **Step 3: Implement the CLI**

Create `scripts/run_structured_marble_inference.py` with:

```python
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from klein4b.marble_prompt_planning import parse_prompt_plan, plan_prompt_with_openai, render_marble_prompt
from klein4b.sample_style_inference import (
    DEFAULT_BEST_LORA_PATH,
    build_ai_toolkit_command,
    make_comparison_contact_sheet,
    render_sample_style_config,
)
```

The parser must include `--reference`, `--output-dir`, `--pseudo-target`, `--previous-best`, `--lora`, `--model`, `--prompt-plan-json`, `--skip-openai`, `--no-run-ai-toolkit`, and `--ai-toolkit-dir`.

The CLI flow must:

1. create `output_dir`,
2. load cached JSON when `--skip-openai` is used,
3. call `plan_prompt_with_openai(reference_path=args.reference, model=args.model)` otherwise,
4. never pass `args.pseudo_target` to the planner,
5. render `prompt.txt`,
6. render `sample_style_inference.yaml`,
7. run AI Toolkit unless `--no-run-ai-toolkit`,
8. write `run_config.json`,
9. make `contact_sheet.jpg` from reference, pseudo target, previous best, and generated output when available.

- [ ] **Step 4: Add helper functions needed by the CLI**

In `src/klein4b/sample_style_inference.py`, add:

```python
def find_latest_sample(samples_dir: Path) -> Path:
    candidates = sorted(samples_dir.glob("*"))
    image_candidates = [
        path for path in candidates if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    ]
    if not image_candidates:
        raise FileNotFoundError(f"No sample images found in {samples_dir}")
    return image_candidates[-1]
```

and `make_comparison_contact_sheet(output_path, reference_path, pseudo_target_path, previous_best_path, generated_path)` using Pillow and the existing contact-sheet style from evaluation scripts.

- [ ] **Step 5: Verify CLI tests pass**

Run:

```bash
pytest tests/test_structured_marble_inference_cli.py -q
```

Expected: all tests PASS.

- [ ] **Step 6: Commit CLI**

Run:

```bash
git add scripts/run_structured_marble_inference.py src/klein4b/sample_style_inference.py tests/test_structured_marble_inference_cli.py
git commit -m "feat: add structured marble inference CLI"
```

---

### Task 6: Run Full Verification

**Files:**
- No code changes expected.

- [ ] **Step 1: Run focused unit tests**

Run:

```bash
pytest tests/test_marble_prompt_planning.py tests/test_sample_style_inference.py tests/test_structured_marble_inference_cli.py -q
```

Expected: all selected tests PASS.

- [ ] **Step 2: Run full test suite**

Run:

```bash
pytest -q
```

Expected: all tests PASS.

- [ ] **Step 3: Run lint**

Run:

```bash
ruff check .
```

Expected: no lint errors.

- [ ] **Step 4: Run formatting check or format**

Run:

```bash
ruff format .
```

Expected: no unintended unrelated changes.

- [ ] **Step 5: Commit any formatting-only changes**

Run only if `ruff format .` changed files:

```bash
git add requirements.txt src/klein4b/inference.py src/klein4b/marble_prompt_planning.py src/klein4b/sample_style_inference.py scripts/run_structured_marble_inference.py tests/test_paths.py tests/test_marble_prompt_planning.py tests/test_sample_style_inference.py tests/test_structured_marble_inference_cli.py
git commit -m "style: format structured inference pipeline"
```

---

### Task 7: Manual Pipeline Test Against Previous Best

**Files:**
- Runtime outputs under `outputs/eval/`; generated artifacts are intentionally untracked.

- [ ] **Step 1: Confirm runtime prerequisites**

Run from the feature worktree:

```bash
test -n "$OPENAI_API_KEY"
test -f /home/liush/projects/charlie_tale_ft/klein4b/vendor/ai-toolkit/run.py
test -f /home/liush/projects/charlie_tale_ft/klein4b/outputs/runs/marble_v4_pairs_rich_result_caption_rank64_unquantized/20260424-211309/marble_bust_v4_pairs_rich_result_caption_rank64_unquantized_style/marble_bust_v4_pairs_rich_result_caption_rank64_unquantized_style_000006500.safetensors
```

Expected: all commands exit with status 0.

- [ ] **Step 2: Run one live structured inference comparison**

Run:

```bash
.venv/bin/python scripts/run_structured_marble_inference.py \
  --reference /home/liush/projects/charlie_tale_ft/klein4b/additional_test_set/input/european_child_female_1.png \
  --pseudo-target /home/liush/projects/charlie_tale_ft/klein4b/additional_test_set/sudo_target/european_child_female_1.png \
  --previous-best /home/liush/projects/charlie_tale_ft/klein4b/outputs/eval/additional_test_set_klein6500_sample_style/generated_named/european_child_female_1.jpg \
  --lora /home/liush/projects/charlie_tale_ft/klein4b/outputs/runs/marble_v4_pairs_rich_result_caption_rank64_unquantized/20260424-211309/marble_bust_v4_pairs_rich_result_caption_rank64_unquantized_style/marble_bust_v4_pairs_rich_result_caption_rank64_unquantized_style_000006500.safetensors \
  --ai-toolkit-dir /home/liush/projects/charlie_tale_ft/klein4b/vendor/ai-toolkit \
  --output-dir /home/liush/projects/charlie_tale_ft/klein4b/outputs/eval/structured_json_klein6500/european_child_female_1
```

Expected:

- `prompt_plan.json` exists,
- `prompt.txt` exists and starts with `Change image 1 into a <mrblbust> from the reference portrait,`,
- `sample_style_inference.yaml` includes exactly one `ctrl_img_1`,
- `generated.*` or a stable copied generated image exists,
- `contact_sheet.jpg` includes reference, pseudo target, previous best, and new structured output.

- [ ] **Step 3: Inspect the comparison artifacts**

Open:

```text
/home/liush/projects/charlie_tale_ft/klein4b/outputs/eval/structured_json_klein6500/european_child_female_1/contact_sheet.jpg
```

Judge whether the structured prompt preserves child age, feminine presentation where visually clear, hair silhouette, bust framing, blank eyes, matte weathered stone, and avoids generic adult philosopher collapse.

- [ ] **Step 4: Commit code only**

Do not commit generated output artifacts. Confirm:

```bash
git status --short
```

Expected: only tracked source/test/docs changes are staged or committed; runtime outputs are outside the feature worktree or ignored.

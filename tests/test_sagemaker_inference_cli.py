from __future__ import annotations

import base64
import importlib.util
import json
from pathlib import Path

from PIL import Image


def load_cli_module():
    module_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "run_sagemaker_marble_inference_local.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_sagemaker_marble_inference_local", module_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_local_cli_runs_sagemaker_functions_and_writes_response(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = load_cli_module()
    reference = tmp_path / "selfie.png"
    model_dir = tmp_path / "model"
    response_path = tmp_path / "response.json"
    Image.new("RGB", (12, 16), "red").save(reference)
    model_dir.mkdir()
    calls: list[tuple[str, object]] = []

    def fake_load_model(**kwargs: object) -> str:
        calls.append(("load_model", kwargs))
        return "model"

    def fake_input_fn(request_body: bytes, request_content_type: str) -> str:
        calls.append(("input_fn", {"bytes": request_body, "content_type": request_content_type}))
        return "request"

    def fake_predict_fn(input_data: str, model: str) -> str:
        calls.append(("predict_fn", {"input": input_data, "model": model}))
        return "prediction"

    def fake_output_fn(prediction: str, response_content_type: str) -> str:
        calls.append(("output_fn", {"prediction": prediction, "accept": response_content_type}))
        return json.dumps({"ok": True})

    monkeypatch.setattr(module, "load_model", fake_load_model)
    monkeypatch.setattr(module, "input_fn", fake_input_fn)
    monkeypatch.setattr(module, "predict_fn", fake_predict_fn)
    monkeypatch.setattr(module, "output_fn", fake_output_fn)

    module.main(
        [
            "--reference",
            str(reference),
            "--model-dir",
            str(model_dir),
            "--output",
            str(response_path),
        ]
    )

    assert json.loads(response_path.read_text(encoding="utf-8")) == {"ok": True}
    assert calls[0] == (
        "load_model",
        {
            "model_dir": model_dir,
            "output_root": None,
            "lora_path": None,
            "ai_toolkit_dir": None,
            "planner_provider": None,
            "planner_model": None,
            "aws_region": None,
        },
    )
    assert calls[1][0] == "input_fn"
    assert calls[1][1]["content_type"] == "image/png"
    assert calls[2] == ("predict_fn", {"input": "request", "model": "model"})
    assert calls[3] == (
        "output_fn",
        {"prediction": "prediction", "accept": "application/json"},
    )


def test_local_cli_can_send_stable_request_id_as_json(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = load_cli_module()
    reference = tmp_path / "selfie.png"
    model_dir = tmp_path / "model"
    response_path = tmp_path / "response.json"
    Image.new("RGB", (12, 16), "red").save(reference)
    model_dir.mkdir()
    input_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        module,
        "load_model",
        lambda **_kwargs: "model",
    )

    def fake_input_fn(request_body: bytes, request_content_type: str) -> str:
        input_calls.append(
            {"body": json.loads(request_body.decode("utf-8")), "content_type": request_content_type}
        )
        return "request"

    monkeypatch.setattr(module, "input_fn", fake_input_fn)
    monkeypatch.setattr(module, "predict_fn", lambda _input_data, _model: "prediction")
    monkeypatch.setattr(module, "output_fn", lambda _prediction, _accept: json.dumps({"ok": True}))

    module.main(
        [
            "--reference",
            str(reference),
            "--model-dir",
            str(model_dir),
            "--request-id",
            "sample-5",
            "--output",
            str(response_path),
        ]
    )

    assert input_calls[0]["content_type"] == "application/json"
    assert input_calls[0]["body"]["request_id"] == "sample-5"
    assert base64.b64decode(input_calls[0]["body"]["image_base64"]) == reference.read_bytes()

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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


def test_parse_prompt_plan_rejects_invalid_age_band() -> None:
    payload = {
        **VALID_PLAN,
        "reference_identity": {
            **VALID_PLAN["reference_identity"],
            "age_band": "teenager",
        },
    }

    with pytest.raises(PromptPlanError, match="age_band.*allowed"):
        parse_prompt_plan(payload)


def test_parse_prompt_plan_rejects_invalid_gender_presentation() -> None:
    payload = {
        **VALID_PLAN,
        "reference_identity": {
            **VALID_PLAN["reference_identity"],
            "gender_presentation": "woman",
        },
    }

    with pytest.raises(PromptPlanError, match="gender_presentation.*allowed"):
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


def _input_image_items(payload: object) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        current = [payload] if payload.get("type") == "input_image" else []
        for value in payload.values():
            current.extend(_input_image_items(value))
        return current
    if isinstance(payload, list):
        image_items: list[dict[str, Any]] = []
        for item in payload:
            image_items.extend(_input_image_items(item))
        return image_items
    return []


def test_openai_planner_sends_exactly_one_image_input(tmp_path: Path) -> None:
    from klein4b.marble_prompt_planning import plan_prompt_with_openai

    calls: dict[str, object] = {}
    reference = tmp_path / "selfie.jpg"
    reference.write_bytes(b"fake-image")

    class FakeResponses:
        def create(self, **kwargs: object) -> object:
            calls["kwargs"] = kwargs

            class Response:
                output_text = json.dumps(VALID_PLAN)

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
    assert len(kwargs["input"]) == 1
    image_items = _input_image_items(kwargs["input"])
    assert len(image_items) == 1
    assert image_items[0]["image_url"].startswith("data:image/jpeg;base64,")
    assert "pseudo" not in str(kwargs).lower()
    assert plan.reference_identity.age_band == "child"


@pytest.mark.parametrize("file_name", ["selfie", "selfie.txt"])
def test_image_path_to_data_url_rejects_non_image_mime_type(
    tmp_path: Path,
    file_name: str,
) -> None:
    from klein4b.marble_prompt_planning import image_path_to_data_url

    reference = tmp_path / file_name
    reference.write_bytes(b"fake-image")

    with pytest.raises(PromptPlanError, match="Unsupported image MIME type"):
        image_path_to_data_url(reference)


def test_openai_planner_wraps_invalid_json_response(tmp_path: Path) -> None:
    from klein4b.marble_prompt_planning import plan_prompt_with_openai

    reference = tmp_path / "selfie.jpg"
    reference.write_bytes(b"fake-image")

    class FakeResponses:
        def create(self, **_kwargs: object) -> object:
            class Response:
                output_text = "{not json"

            return Response()

    class FakeClient:
        responses = FakeResponses()

    with pytest.raises(PromptPlanError, match="OpenAI planner returned invalid JSON"):
        plan_prompt_with_openai(reference_path=reference, client=FakeClient())

from __future__ import annotations

from pathlib import Path

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

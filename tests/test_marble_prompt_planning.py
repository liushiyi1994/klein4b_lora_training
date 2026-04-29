from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from klein4b.marble_prompt_planning import (
    PLANNER_INSTRUCTIONS,
    PromptPlanError,
    build_prompt_plan_schema,
    parse_prompt_plan,
    render_marble_prompt,
)

VALID_PLAN = {
    "reference_identity": {
        "age_band": "child",
        "gender_presentation": "feminine",
        "eye_state": "open",
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


def test_prompt_plan_schema_keeps_target_style_composition_only() -> None:
    schema = build_prompt_plan_schema()

    target_style_schema = schema["schema"]["properties"]["target_style"]

    assert set(target_style_schema["properties"]) == {
        "bust_framing",
        "statue_angle",
        "drapery_and_torso",
        "headpiece_or_ornament",
    }
    assert "stone_surface" not in str(target_style_schema)
    assert "weathering" not in str(target_style_schema)
    assert "base_and_lava" not in str(target_style_schema)
    assert "background" not in str(target_style_schema)


def test_planner_instructions_translate_selfie_cues_without_artifacts() -> None:
    instructions = PLANNER_INSTRUCTIONS.lower()

    assert "translate the person into a final marble bust" in instructions
    assert "do not describe the selfie" in instructions
    assert "identity-bearing face structure" in instructions
    assert "head direction" in instructions
    assert "hair silhouette" in instructions
    assert "hairstyle" in instructions
    assert "facial hair" in instructions
    assert "eye_state" in instructions
    assert "closed only when both eyes are clearly closed" in instructions
    assert "do not invent greek ornaments" in instructions
    assert "laurel wreaths" in instructions
    assert "modern clothing" in instructions
    assert "glasses" in instructions
    assert "hair accessories" in instructions
    assert "bows" in instructions
    assert "ribbons" in instructions
    assert "headscarves" in instructions
    assert "turbans" in instructions
    assert "do not reinterpret bows" in instructions
    assert "do not reinterpret" in instructions
    assert "only explicitly greek ornaments" in instructions
    assert "greek or classical ornaments" not in instructions
    assert "hand gestures" in instructions
    assert "race or ethnicity labels" in instructions


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


def test_parse_prompt_plan_allows_natural_hair_color_words() -> None:
    payload = {
        **VALID_PLAN,
        "reference_identity": {
            **VALID_PLAN["reference_identity"],
            "hair_or_headwear": ["short white hair", "black curls"],
        },
    }

    plan = parse_prompt_plan(payload)
    prompt = render_marble_prompt(plan)

    assert plan.reference_identity.hair_or_headwear == ("short white hair", "black curls")
    assert "short hair" in prompt
    assert "curls" in prompt
    assert "white hair" not in prompt
    assert "black curls" not in prompt


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
    assert "preserve the selfie head direction and head angle" in prompt
    assert "preserve reference yaw, pitch, roll, gaze direction, and neck orientation" in prompt
    assert "preserve the selfie hairstyle" in prompt
    assert "render open blank sculpted stone eyes" in prompt
    assert "closed carved eyelids" not in prompt
    assert "dry chalky unpolished stone" in prompt
    assert "subdued off-axis ambient lighting" in prompt
    assert "asymmetrical stone lighting" in prompt
    assert "dark ember background" in prompt
    assert "localized lava only in the broken lower base" in prompt
    assert "Preserve hair only; do not add a crown" in prompt
    assert "explicitly Greek headwear or ornament silhouette" not in prompt
    assert "Greek head covering or ornament" not in prompt
    assert "Greek or classical headwear" not in prompt
    assert "Greek or classical head covering" not in prompt
    assert "no pupils" in prompt
    assert "no irises" in prompt
    assert "no catchlights" in prompt
    assert "same marble as the face" in prompt
    assert "pseudo target" not in prompt.lower()


def test_render_marble_prompt_closes_eyes_only_when_reference_eyes_are_closed() -> None:
    closed_eye_plan = parse_prompt_plan(
        {
            **VALID_PLAN,
            "reference_identity": {
                **VALID_PLAN["reference_identity"],
                "eye_state": "closed",
            },
        }
    )

    closed_prompt = render_marble_prompt(closed_eye_plan)
    open_prompt = render_marble_prompt(parse_prompt_plan(VALID_PLAN))

    assert "Reference eyes are closed; render closed carved eyelids" in closed_prompt
    assert "render open blank sculpted stone eyes" in open_prompt
    assert "closed carved eyelids" not in open_prompt


def test_render_marble_prompt_forbids_lowered_eyelids_for_open_reference_eyes() -> None:
    prompt = render_marble_prompt(parse_prompt_plan(VALID_PLAN))

    assert "do not render closed eyes, lowered eyelids, sleeping eyelids" in prompt


def test_parse_prompt_plan_defaults_legacy_missing_eye_state_to_open() -> None:
    payload = {
        **VALID_PLAN,
        "reference_identity": {
            key: value
            for key, value in VALID_PLAN["reference_identity"].items()
            if key != "eye_state"
        },
    }

    plan = parse_prompt_plan(payload)

    assert plan.reference_identity.eye_state == "open"


def test_render_marble_prompt_uses_conditional_ornament_policy() -> None:
    no_ornament_prompt = render_marble_prompt(parse_prompt_plan(VALID_PLAN))
    with_ornament_plan = parse_prompt_plan(
        {
            **VALID_PLAN,
            "target_style": {
                **VALID_PLAN["target_style"],
                "headpiece_or_ornament": ["classical Greek laurel wreath carved as stone relief"],
            },
        }
    )

    with_ornament_prompt = render_marble_prompt(with_ornament_plan)

    assert "Preserve hair only; do not add a crown" in no_ornament_prompt
    assert "Allowed ornament" not in no_ornament_prompt
    assert "Allowed ornament: classical Greek laurel wreath carved as stone relief" in (
        with_ornament_prompt
    )
    assert "Preserve hair only; do not add a crown" not in with_ornament_prompt


def test_render_marble_prompt_excludes_planner_authored_material_style() -> None:
    payload = {
        **VALID_PLAN,
        "target_style": {
            **VALID_PLAN["target_style"],
            "stone_surface": ["white marble", "subtle polished highlights"],
            "weathering": ["light surface wear"],
            "base_and_lava": ["no lava or flame elements"],
            "background": "plain neutral museum-like backdrop",
        },
    }

    with pytest.raises(PromptPlanError, match="unexpected fields"):
        parse_prompt_plan(payload)


def test_parse_prompt_plan_rejects_material_language_inside_target_style() -> None:
    payload = {
        **VALID_PLAN,
        "target_style": {
            **VALID_PLAN["target_style"],
            "drapery_and_torso": ["simple classical drapery", "subtle polished highlights"],
        },
    }

    with pytest.raises(PromptPlanError, match="target_style must not include material"):
        parse_prompt_plan(payload)


@pytest.mark.parametrize(
    "target_phrase",
    [
        "smooth marble torso",
        "smooth marble finish",
        "clean bright stone surface",
    ],
)
def test_parse_prompt_plan_rejects_bright_smooth_finish_language_inside_target_style(
    target_phrase: str,
) -> None:
    payload = {
        **VALID_PLAN,
        "target_style": {
            **VALID_PLAN["target_style"],
            "drapery_and_torso": ["simple classical drapery", target_phrase],
        },
    }

    with pytest.raises(PromptPlanError, match="target_style must not include material"):
        parse_prompt_plan(payload)


def test_render_marble_prompt_hardens_against_bright_clean_marble() -> None:
    prompt = render_marble_prompt(parse_prompt_plan(VALID_PLAN))

    assert "mid-tone matte weathered grey marble" in prompt
    assert "dark grime in facial grooves, hair grooves, eye sockets, and drapery recesses" in prompt
    assert "Avoid bright white marble" in prompt
    assert "clean white stone" in prompt
    assert "overbright highlights" in prompt


def test_render_marble_prompt_suppresses_selfie_artifact_details() -> None:
    payload = {
        **VALID_PLAN,
        "reference_identity": {
            **VALID_PLAN["reference_identity"],
            "face_structure": [
                "large eyes with heavy eyeliner in the reference",
                "rounded cheeks",
                "slightly prominent upper teeth",
                "fair complexion",
            ],
            "hair_or_headwear": [
                "short wavy dark hair",
                "yellow headband or hair accessory",
                "dark eyeglass frames",
                "large bow-like headwear silhouette",
                "decorative circlet",
                "small tiara",
            ],
            "broad_expression": "open-mouthed laugh with visible upper teeth",
        },
    }
    plan = parse_prompt_plan(payload)

    prompt = render_marble_prompt(plan)

    assert "rounded cheeks" in prompt
    assert "short wavy hair" in prompt
    assert "subtle carved smile" in prompt
    assert "eyeliner" not in prompt
    assert "upper teeth" not in prompt
    assert "complexion" not in prompt
    assert "yellow" not in prompt
    assert "yellow headband" not in prompt
    assert "bow" not in prompt
    assert "decorative circlet" not in prompt
    assert "small tiara" not in prompt
    assert "eyeglass" not in prompt
    assert "open-mouthed" not in prompt
    assert "visible upper teeth" not in prompt


def test_render_marble_prompt_keeps_greek_style_ornaments_only() -> None:
    payload = {
        **VALID_PLAN,
        "target_style": {
            **VALID_PLAN["target_style"],
            "headpiece_or_ornament": [
                "classical Greek laurel wreath carved as stone relief",
                "modern bow-shaped ribbon ornament",
                "carved bow-like hair ornament retained in simplified classical form",
                "stylized carved headpiece inspired by classical motifs",
                "ancient ceremonial headpiece carved as stone",
            ],
        },
    }
    plan = parse_prompt_plan(payload)

    prompt = render_marble_prompt(plan)

    assert "classical Greek laurel wreath carved as stone relief" in prompt
    assert "modern bow-shaped ribbon ornament" not in prompt
    assert "carved bow-like hair ornament" not in prompt
    assert "classical motifs" not in prompt
    assert "ancient ceremonial headpiece" not in prompt


def test_render_marble_prompt_suppresses_non_greek_headwear() -> None:
    payload = {
        **VALID_PLAN,
        "reference_identity": {
            **VALID_PLAN["reference_identity"],
            "hair_or_headwear": [
                "closely cropped short hair",
                "patterned cap with rounded crown",
                "geometric patterned headwear",
                "wrapped headscarf with a turban-like silhouette",
            ],
        },
        "target_style": {
            **VALID_PLAN["target_style"],
            "headpiece_or_ornament": [
                "classical Greek laurel wreath carved as stone relief",
                "carved patterned cap simplified into a classical head covering shape",
                "carved wrapped head covering translated into a sculptural turban-like form",
                "simple carved geometric headpiece inspired by the reference silhouette",
                "none",
            ],
        },
    }
    plan = parse_prompt_plan(payload)

    prompt = render_marble_prompt(plan)

    assert "closely cropped short hair" in prompt
    assert "classical Greek laurel wreath carved as stone relief" in prompt
    assert "patterned cap" not in prompt
    assert "geometric patterned headwear" not in prompt
    assert "headscarf" not in prompt
    assert "turban" not in prompt
    assert "classical head covering shape" not in prompt
    assert "sculptural turban-like form" not in prompt
    assert "geometric headpiece" not in prompt
    assert "none" not in prompt


def test_render_marble_prompt_does_not_render_banned_detail_terms() -> None:
    payload = {
        **VALID_PLAN,
        "safety_overrides": {
            **VALID_PLAN["safety_overrides"],
            "banned_details": [
                "bows that read fabric-like or colorful",
                "ribbons",
                "modern glasses",
            ],
        },
    }
    plan = parse_prompt_plan(payload)

    prompt = render_marble_prompt(plan)

    assert "bows that read fabric-like or colorful" not in prompt
    assert "ribbons" not in prompt
    assert "modern glasses" not in prompt
    assert "no pupils" in prompt


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


def test_bedrock_nova_planner_sends_one_image_and_parses_json(tmp_path: Path) -> None:
    from klein4b.marble_prompt_planning import plan_prompt_with_bedrock_nova

    calls: dict[str, object] = {}
    reference = tmp_path / "selfie.png"
    reference.write_bytes(b"fake-image")

    class FakeBedrockClient:
        def converse(self, **kwargs: object) -> object:
            calls["kwargs"] = kwargs
            return {"output": {"message": {"content": [{"text": json.dumps(VALID_PLAN)}]}}}

    plan = plan_prompt_with_bedrock_nova(
        reference_path=reference,
        model="us.amazon.nova-2-lite-v1:0",
        client=FakeBedrockClient(),
    )

    kwargs = calls["kwargs"]
    assert kwargs["modelId"] == "us.amazon.nova-2-lite-v1:0"
    assert kwargs["inferenceConfig"]["maxTokens"] == 2048
    assert kwargs["inferenceConfig"]["temperature"] == 0
    assert len(kwargs["messages"]) == 1
    content = kwargs["messages"][0]["content"]
    image_blocks = [block["image"] for block in content if "image" in block]
    text_blocks = [block["text"] for block in content if "text" in block]
    assert len(image_blocks) == 1
    assert image_blocks[0]["format"] == "png"
    assert image_blocks[0]["source"]["bytes"] == b"fake-image"
    assert "pseudo" not in str(kwargs).lower()
    assert "Return only valid JSON" in text_blocks[0]
    assert plan.reference_identity.age_band == "child"


def test_bedrock_nova_planner_extracts_json_from_text_wrapper(tmp_path: Path) -> None:
    from klein4b.marble_prompt_planning import plan_prompt_with_bedrock_nova

    reference = tmp_path / "selfie.jpg"
    reference.write_bytes(b"fake-image")

    class FakeBedrockClient:
        def converse(self, **_kwargs: object) -> object:
            return {
                "output": {
                    "message": {
                        "content": [
                            {"text": f"Here is the JSON:\n```json\n{json.dumps(VALID_PLAN)}\n```"}
                        ]
                    }
                }
            }

    plan = plan_prompt_with_bedrock_nova(reference_path=reference, client=FakeBedrockClient())

    assert plan.reference_identity.age_band == "child"


def test_bedrock_nova_planner_drops_material_language_from_target_style(
    tmp_path: Path,
) -> None:
    from klein4b.marble_prompt_planning import plan_prompt_with_bedrock_nova

    reference = tmp_path / "selfie.png"
    reference.write_bytes(b"fake-image")
    payload = {
        **VALID_PLAN,
        "target_style": {
            **VALID_PLAN["target_style"],
            "drapery_and_torso": [
                "simple classical drapery",
                "smooth surface treatment with polished finish",
                "smooth marble torso",
                "smooth marble finish",
                "clean bright stone surface",
            ],
            "headpiece_or_ornament": [
                "classical Greek laurel wreath carved as stone relief",
                "background-like decorative surface glow",
            ],
        },
    }

    class FakeBedrockClient:
        def converse(self, **_kwargs: object) -> object:
            return {"output": {"message": {"content": [{"text": json.dumps(payload)}]}}}

    plan = plan_prompt_with_bedrock_nova(reference_path=reference, client=FakeBedrockClient())

    assert plan.target_style.drapery_and_torso == ("simple classical drapery",)
    assert plan.target_style.headpiece_or_ornament == (
        "classical Greek laurel wreath carved as stone relief",
    )

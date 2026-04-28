from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

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

_REFERENCE_IDENTITY_KEYS = (
    "age_band",
    "gender_presentation",
    "head_pose",
    "face_structure",
    "hair_or_headwear",
    "broad_expression",
)
_TARGET_STYLE_KEYS = (
    "bust_framing",
    "statue_angle",
    "drapery_and_torso",
    "headpiece_or_ornament",
    "stone_surface",
    "weathering",
    "base_and_lava",
    "background",
)
_SAFETY_OVERRIDE_KEYS = (
    "identity_source_policy",
    "eye_policy",
    "material_policy",
    "banned_details",
)
_PROMPT_PLAN_KEYS = ("reference_identity", "target_style", "safety_overrides")


class PromptPlanError(ValueError):
    """Raised when prompt-plan JSON is structurally invalid or unsafe."""


@dataclass(frozen=True)
class ReferenceIdentity:
    age_band: str
    gender_presentation: str
    head_pose: str
    face_structure: tuple[str, ...]
    hair_or_headwear: tuple[str, ...]
    broad_expression: str


@dataclass(frozen=True)
class TargetStyle:
    bust_framing: str
    statue_angle: str
    drapery_and_torso: tuple[str, ...]
    headpiece_or_ornament: tuple[str, ...]
    stone_surface: tuple[str, ...]
    weathering: tuple[str, ...]
    base_and_lava: tuple[str, ...]
    background: str


@dataclass(frozen=True)
class SafetyOverrides:
    identity_source_policy: str
    eye_policy: str
    material_policy: str
    banned_details: tuple[str, ...]


@dataclass(frozen=True)
class PromptPlan:
    reference_identity: ReferenceIdentity
    target_style: TargetStyle
    safety_overrides: SafetyOverrides


def build_prompt_plan_schema() -> dict[str, Any]:
    """Return the strict JSON schema expected from the selfie-only planner."""

    string_schema: dict[str, Any] = {"type": "string", "minLength": 1}
    string_list_schema: dict[str, Any] = {
        "type": "array",
        "items": string_schema,
    }

    reference_identity_schema = _object_schema(
        {
            "age_band": {
                "type": "string",
                "enum": [
                    "child",
                    "young adult",
                    "adult",
                    "middle-aged adult",
                    "elderly adult",
                    "unknown",
                ],
            },
            "gender_presentation": {
                "type": "string",
                "enum": ["masculine", "feminine", "androgynous", "unknown"],
            },
            "head_pose": string_schema,
            "face_structure": string_list_schema,
            "hair_or_headwear": string_list_schema,
            "broad_expression": string_schema,
        }
    )
    target_style_schema = _object_schema(
        {
            "bust_framing": string_schema,
            "statue_angle": string_schema,
            "drapery_and_torso": string_list_schema,
            "headpiece_or_ornament": string_list_schema,
            "stone_surface": string_list_schema,
            "weathering": string_list_schema,
            "base_and_lava": string_list_schema,
            "background": string_schema,
        }
    )
    safety_overrides_schema = _object_schema(
        {
            "identity_source_policy": string_schema,
            "eye_policy": string_schema,
            "material_policy": string_schema,
            "banned_details": string_list_schema,
        }
    )

    return {
        "name": "marble_prompt_plan",
        "strict": True,
        "schema": _object_schema(
            {
                "reference_identity": reference_identity_schema,
                "target_style": target_style_schema,
                "safety_overrides": safety_overrides_schema,
            }
        ),
    }


def parse_prompt_plan(payload: Mapping[str, Any]) -> PromptPlan:
    """Validate planner JSON and convert it to dataclasses."""

    if not isinstance(payload, Mapping):
        raise PromptPlanError("prompt plan must be a JSON object")

    _require_exact_keys(payload, _PROMPT_PLAN_KEYS, "prompt plan")

    reference_payload = _mapping_field(payload, "reference_identity")
    target_payload = _mapping_field(payload, "target_style")
    safety_payload = _mapping_field(payload, "safety_overrides")

    _require_exact_keys(reference_payload, _REFERENCE_IDENTITY_KEYS, "reference_identity")
    _require_exact_keys(target_payload, _TARGET_STYLE_KEYS, "target_style")
    _require_exact_keys(safety_payload, _SAFETY_OVERRIDE_KEYS, "safety_overrides")

    reference_identity = ReferenceIdentity(
        age_band=_string_field(reference_payload, "age_band"),
        gender_presentation=_string_field(reference_payload, "gender_presentation"),
        head_pose=_string_field(reference_payload, "head_pose"),
        face_structure=_string_list_field(reference_payload, "face_structure"),
        hair_or_headwear=_string_list_field(reference_payload, "hair_or_headwear"),
        broad_expression=_string_field(reference_payload, "broad_expression"),
    )
    _validate_reference_identity(reference_identity)

    target_style = TargetStyle(
        bust_framing=_string_field(target_payload, "bust_framing"),
        statue_angle=_string_field(target_payload, "statue_angle"),
        drapery_and_torso=_string_list_field(target_payload, "drapery_and_torso"),
        headpiece_or_ornament=_string_list_field(target_payload, "headpiece_or_ornament"),
        stone_surface=_string_list_field(target_payload, "stone_surface"),
        weathering=_string_list_field(target_payload, "weathering"),
        base_and_lava=_string_list_field(target_payload, "base_and_lava"),
        background=_string_field(target_payload, "background"),
    )
    safety_overrides = SafetyOverrides(
        identity_source_policy=_string_field(safety_payload, "identity_source_policy"),
        eye_policy=_string_field(safety_payload, "eye_policy"),
        material_policy=_string_field(safety_payload, "material_policy"),
        banned_details=_string_list_field(safety_payload, "banned_details"),
    )

    return PromptPlan(
        reference_identity=reference_identity,
        target_style=target_style,
        safety_overrides=safety_overrides,
    )


def render_marble_prompt(plan: PromptPlan) -> str:
    """Render a validated prompt plan into the final image prompt."""

    reference = plan.reference_identity
    target = plan.target_style
    safety = plan.safety_overrides

    reference_parts = [
        f"{reference.age_band} age band",
        f"{reference.gender_presentation} gender presentation",
        reference.head_pose,
        _join_phrases(reference.face_structure),
        _join_phrases(reference.hair_or_headwear),
        reference.broad_expression,
    ]
    target_parts = [
        target.bust_framing,
        target.statue_angle,
        _join_phrases(target.drapery_and_torso),
        _join_phrases(target.headpiece_or_ornament),
        _join_phrases(target.stone_surface),
        _join_phrases(target.weathering),
        _join_phrases(target.base_and_lava),
        target.background,
    ]
    safety_parts = [
        safety.identity_source_policy,
        safety.eye_policy,
        safety.material_policy,
        f"avoid {_join_phrases(safety.banned_details)}" if safety.banned_details else "",
    ]

    return (
        "Change image 1 into a <mrblbust> from the reference portrait, "
        f"preserving {_join_phrases(reference_parts)}. "
        f"Target style: {_join_phrases(target_parts)}. "
        f"Planner safety overrides: {_join_phrases(safety_parts)}. "
        f"{FIXED_PROMPT_CONSTRAINTS}"
    )


def _object_schema(properties: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": dict(properties),
        "required": list(properties),
    }


def _mapping_field(payload: Mapping[str, Any], field_name: str) -> Mapping[str, Any]:
    value = payload[field_name]
    if not isinstance(value, Mapping):
        raise PromptPlanError(f"{field_name} must be an object")
    return value


def _require_exact_keys(
    payload: Mapping[str, Any],
    expected_keys: tuple[str, ...],
    section_name: str,
) -> None:
    expected = set(expected_keys)
    actual = set(payload)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        raise PromptPlanError(f"{section_name} is missing required fields: {', '.join(missing)}")
    if extra:
        raise PromptPlanError(f"{section_name} has unexpected fields: {', '.join(extra)}")


def _string_field(payload: Mapping[str, Any], field_name: str) -> str:
    value = payload[field_name]
    if not isinstance(value, str):
        raise PromptPlanError(f"{field_name} must be a string")
    value = value.strip()
    if not value:
        raise PromptPlanError(f"{field_name} must not be empty")
    return value


def _string_list_field(payload: Mapping[str, Any], field_name: str) -> tuple[str, ...]:
    value = payload[field_name]
    if not isinstance(value, list):
        raise PromptPlanError(f"{field_name} must be a list of strings")

    items: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise PromptPlanError(f"{field_name}[{index}] must be a string")
        item = item.strip()
        if not item:
            raise PromptPlanError(f"{field_name}[{index}] must not be empty")
        items.append(item)
    return tuple(items)


def _validate_reference_identity(reference_identity: ReferenceIdentity) -> None:
    descriptors = (
        reference_identity.age_band,
        reference_identity.gender_presentation,
        reference_identity.head_pose,
        *reference_identity.face_structure,
        *reference_identity.hair_or_headwear,
        reference_identity.broad_expression,
    )
    for descriptor in descriptors:
        matched_label = _find_forbidden_identity_label(descriptor)
        if matched_label is not None:
            raise PromptPlanError(
                "reference_identity must not include race or ethnicity descriptors "
                f"(matched {matched_label!r})"
            )


def _find_forbidden_identity_label(descriptor: str) -> str | None:
    for label in FORBIDDEN_IDENTITY_LABELS:
        pattern = _identity_label_pattern(label)
        if pattern.search(descriptor):
            return label
    return None


def _identity_label_pattern(label: str) -> re.Pattern[str]:
    escaped_label = re.escape(label).replace(r"\ ", r"\s+")
    return re.compile(rf"(?<![a-z0-9]){escaped_label}(?![a-z0-9])", re.IGNORECASE)


def _join_phrases(phrases: tuple[str, ...] | list[str]) -> str:
    return ", ".join(phrase for phrase in phrases if phrase)


__all__ = [
    "FIXED_PROMPT_CONSTRAINTS",
    "FORBIDDEN_IDENTITY_LABELS",
    "PromptPlan",
    "PromptPlanError",
    "ReferenceIdentity",
    "SafetyOverrides",
    "TargetStyle",
    "build_prompt_plan_schema",
    "parse_prompt_plan",
    "render_marble_prompt",
]

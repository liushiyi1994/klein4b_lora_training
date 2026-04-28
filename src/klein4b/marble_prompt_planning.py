from __future__ import annotations

import base64
import json
import mimetypes
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ALLOWED_AGE_BANDS = (
    "child",
    "young adult",
    "adult",
    "middle-aged adult",
    "elderly adult",
    "unknown",
)

ALLOWED_GENDER_PRESENTATIONS = (
    "masculine",
    "feminine",
    "androgynous",
    "unknown",
)

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

SUPPRESSED_SELFIE_ARTIFACT_TERMS = (
    "background",
    "bow",
    "bow-shaped",
    "camera perspective",
    "catchlight",
    "circlet",
    "colored eye",
    "colored lip",
    "complexion",
    "earbud",
    "eyeglass",
    "eyeliner",
    "eyewear",
    "glasses",
    "hair accessory",
    "hand gesture",
    "headband",
    "jacket",
    "jewelry",
    "lipstick",
    "makeup",
    "peace sign",
    "phone",
    "ribbon",
    "selfie",
    "shirt",
    "skin texture",
    "t-shirt",
    "teeth",
    "thumbs up",
    "tiara",
    "watch",
)

SUPPRESSED_NON_GREEK_HEADWEAR_TERMS = (
    "baseball cap",
    "bonnet",
    "cap",
    "head scarf",
    "head wrap",
    "headscarf",
    "headwrap",
    "hat",
    "turban",
    "veil",
)

GENERIC_HEADWEAR_OR_ORNAMENT_TERMS = (
    "accessory",
    "head covering",
    "headpiece",
    "headwear",
    "ornament",
)

EMPTY_ORNAMENT_TERMS = (
    "none",
    "no headpiece",
    "no headwear",
    "no ornament",
    "no ornaments",
)

ALLOWED_GREEK_ORNAMENT_TERMS = (
    "greek",
    "laurel",
)

NATURAL_COLOR_WORDS = (
    "black",
    "blond",
    "blonde",
    "blue",
    "brown",
    "copper",
    "dark",
    "gold",
    "golden",
    "gray",
    "green",
    "grey",
    "light",
    "orange",
    "red",
    "reddish",
    "white",
    "yellow",
)

FORBIDDEN_TARGET_STYLE_TERMS = (
    "background",
    "dark ember",
    "glossy",
    "gray",
    "grey",
    "lava",
    "lighting",
    "low-albedo",
    "museum",
    "patina",
    "pitted",
    "polished",
    "shine",
    "specular",
    "surface",
    "weather",
    "weathered",
    "weathering",
    "white marble",
)

FIXED_PROMPT_CONSTRAINTS = (
    "Use the reference portrait as the only identity, pose, face-structure, "
    "hair silhouette, facial hair shape, and broad-expression source. "
    "Use the reference portrait as the only source for identity-bearing face "
    "structure, pose and head angle, age cues, gender presentation cues, hair "
    "silhouette, facial hair shape, and broad expression only when useful. "
    "Always preserve the selfie head direction and head angle; preserve "
    "reference yaw, pitch, roll, gaze direction, and neck orientation; do not "
    "normalize the bust to a frontal view or generic three-quarter view unless "
    "the selfie has that orientation. "
    "Always preserve the selfie hairstyle as carved marble, including length, part, "
    "bangs, volume, curl or straightness, updo, hairline, and overall hair "
    "mass; translate hair into carved stone hair, not decorative headwear. "
    "Translate the person into a final marble bust rather than describing "
    "selfie artifacts. Convert broad smiles or laughter into a subtle carved "
    "smile or soft neutral expression, and omit hand gestures. "
    "The eyes must be blank sculpted stone eyes or closed carved eyelids; "
    "no pupils, no irises, no colored eyes, no catchlights, no painted eyes, "
    "no realistic human eyes. Hair, eyebrows, facial hair, and any allowed "
    "ornament must be carved from the same marble as the face. Avoid modern "
    "design, modern decorative details, contemporary ornamentation, modern "
    "accessories, and modern clothing. Use matte weathered grey marble, rough pitted "
    "low-albedo face surface, dry chalky unpolished stone, uneven grey-brown "
    "mineral patina, grime in recesses, chipped edges, and localized lava only "
    "in the broken lower base against a dark ember background. Use subdued "
    "off-axis ambient lighting and "
    "asymmetrical stone lighting with one side of the face slightly darker "
    "than the other, shadowed eye sockets, and shallow carved shadows under "
    "the brow ridge, nose, lower lip, and chin. Avoid glossy marble, glossy "
    "polished marble, wet shine, specular hotspots, selfie lighting, beauty "
    "lighting, full-face even illumination, frontal studio light, head-on key "
    "light, perfect portrait lighting, smooth beauty-render face, polished "
    "cheeks, shiny forehead, shiny nose, shiny lips, modern design, modern "
    "decorative details, contemporary ornamentation, modern accessories, "
    "modern clothing, duplicate figures, side-by-side views, and collage."
)

NO_ORNAMENT_POLICY = (
    "Preserve hair only; do not add a crown, laurel wreath, tiara, circlet, "
    "headband, headdress, veil, cap, hat, head covering, or hair accessory."
)

PLANNER_INSTRUCTIONS = (
    "You are planning a prompt for a selfie-to-Greek-marble-statue-bust pipeline. "
    "You will receive exactly one image: the selfie/reference portrait. Treat that "
    "single image as the only source for identity-bearing face structure, pose "
    "and head angle, head direction, broad expression only if useful, hair "
    "silhouette, hairstyle, facial hair shape, age cues, "
    "gender presentation cues, and "
    "distinctive non-modern ornaments only if they can become carved stone. "
    "Preserve the selfie head direction and head angle, including yaw, pitch, "
    "roll, gaze direction, and neck orientation; do not normalize the person "
    "to a frontal or generic three-quarter bust unless the selfie has that "
    "orientation. Preserve the "
    "selfie hairstyle as carved marble, including length, part, bangs, volume, "
    "curl or straightness, updo, hairline, and overall hair mass. "
    "Translate the person into a final marble bust; do not describe the selfie. "
    "A smile can become a subtle carved smile or soft neutral expression. "
    "Suppress modern clothing, glasses, earbuds, watches, phones, jewelry that "
    "reads modern, modern hair accessories, bows, ribbons, clips, bands, hats, "
    "caps, headscarves, headwraps, turbans, veils, selfie lighting, camera "
    "perspective artifacts, skin texture, makeup, colored lips or eyes, hand "
    "gestures, peace signs, thumbs up, exaggerated facial expression, modern "
    "design details, contemporary ornamentation, and photographic background. "
    "Do not invent Greek ornaments, laurel wreaths, crowns, headpieces, or "
    "head coverings. Keep only explicitly Greek ornaments, such as a laurel "
    "wreath, when they are clearly visible in the reference and can become "
    "carved stone. Do not "
    "reinterpret bows, ribbons, clips, bands, caps, headscarves, headwraps, "
    "turbans, or veils as classical ornaments. Do not include "
    "race or ethnicity labels. Do not infer any identity, pose, or style detail "
    "from any target image, prior result, generated example, second image, or "
    "unstated visual source. Return only JSON matching the provided schema. Use "
    "target_style only for bust framing, statue angle, classical drapery or "
    "torso treatment, and non-modern carved ornament shape. Do not put marble "
    "material, polish, lighting, background, lava, weathering, or surface finish "
    "language in target_style; the pipeline supplies those final style rules."
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
                "enum": list(ALLOWED_AGE_BANDS),
            },
            "gender_presentation": {
                "type": "string",
                "enum": list(ALLOWED_GENDER_PRESENTATIONS),
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
        age_band=_enum_field(reference_payload, "age_band", ALLOWED_AGE_BANDS),
        gender_presentation=_enum_field(
            reference_payload,
            "gender_presentation",
            ALLOWED_GENDER_PRESENTATIONS,
        ),
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
    )
    _validate_target_style(target_style)
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
        _sanitize_reference_phrase(reference.head_pose),
        _join_phrases(_sanitize_reference_phrase(phrase) for phrase in reference.face_structure),
        _join_phrases(
            _sanitize_reference_phrase(phrase, strip_color_words=True)
            for phrase in reference.hair_or_headwear
        ),
        _translate_expression_phrase(reference.broad_expression),
    ]
    ornament_text = _join_phrases(
        _sanitize_ornament_phrase(phrase) for phrase in target.headpiece_or_ornament
    )
    target_parts = [
        target.bust_framing,
        target.statue_angle,
        _join_phrases(target.drapery_and_torso),
        ornament_text,
    ]
    safety_parts = [
        safety.identity_source_policy,
        safety.eye_policy,
        safety.material_policy,
    ]

    return (
        "Change image 1 into a <mrblbust> from the reference portrait, "
        f"preserving {_join_phrases(reference_parts)}. "
        f"Target style: {_join_phrases(target_parts)}. "
        f"Planner safety overrides: {_join_phrases(safety_parts)}. "
        f"{_render_ornament_policy(ornament_text)} "
        f"{FIXED_PROMPT_CONSTRAINTS}"
    )


def _render_ornament_policy(ornament_text: str) -> str:
    if ornament_text:
        return (
            f"Allowed ornament: {ornament_text}. Do not add any other crown, "
            "headpiece, head covering, hair accessory, modern design detail, "
            "or contemporary ornament."
        )
    return NO_ORNAMENT_POLICY


def image_path_to_data_url(reference_path: Path) -> str:
    """Encode an image path as a data URL for the OpenAI Responses API."""

    media_type, _encoding = mimetypes.guess_type(reference_path)
    if media_type is None or not media_type.startswith("image/"):
        raise PromptPlanError(f"Unsupported image MIME type for {reference_path}")
    encoded_image = base64.b64encode(reference_path.read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{encoded_image}"


def plan_prompt_with_openai(
    reference_path: Path,
    model: str = "gpt-5.4-mini",
    client: object | None = None,
) -> PromptPlan:
    """Request a strict prompt plan from OpenAI using only the reference selfie."""

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
    try:
        payload = json.loads(response.output_text)
    except json.JSONDecodeError as error:
        raise PromptPlanError("OpenAI planner returned invalid JSON") from error
    return parse_prompt_plan(payload)


def plan_prompt_with_bedrock_nova(
    reference_path: Path,
    model: str = "us.amazon.nova-2-lite-v1:0",
    region_name: str | None = None,
    client: object | None = None,
) -> PromptPlan:
    """Request a prompt plan from Amazon Nova using only the reference selfie."""

    if client is None:
        import boto3
        from botocore.config import Config

        client = boto3.client(
            "bedrock-runtime",
            region_name=region_name,
            config=Config(
                connect_timeout=3600,
                read_timeout=3600,
                retries={"max_attempts": 1},
            ),
        )

    prompt = (
        f"{PLANNER_INSTRUCTIONS}\n\n"
        "Return only valid JSON, with no Markdown fences or commentary. "
        "The JSON must validate against this schema:\n"
        f"{json.dumps(build_prompt_plan_schema()['schema'], indent=2)}"
    )
    response = client.converse(
        modelId=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"text": prompt},
                    {
                        "image": {
                            "format": _bedrock_image_format(reference_path),
                            "source": {"bytes": reference_path.read_bytes()},
                        }
                    },
                ],
            }
        ],
        inferenceConfig={"maxTokens": 2048, "temperature": 0},
    )
    text = _bedrock_response_text(response)
    payload = _json_object_from_model_text(text, "Bedrock Nova planner")
    payload = _sanitize_bedrock_prompt_plan_payload(payload)
    return parse_prompt_plan(payload)


def _object_schema(properties: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": dict(properties),
        "required": list(properties),
    }


def _bedrock_image_format(reference_path: Path) -> str:
    suffix = reference_path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "jpeg"
    if suffix == ".png":
        return "png"
    if suffix == ".webp":
        return "webp"
    if suffix == ".gif":
        return "gif"
    raise PromptPlanError(f"Unsupported image format for Bedrock Nova: {reference_path}")


def _bedrock_response_text(response: object) -> str:
    if not isinstance(response, Mapping):
        raise PromptPlanError("Bedrock Nova planner returned an invalid response shape")
    try:
        content = response["output"]["message"]["content"]
    except KeyError as error:
        raise PromptPlanError("Bedrock Nova planner returned an invalid response shape") from error
    if not isinstance(content, list):
        raise PromptPlanError("Bedrock Nova planner returned an invalid response shape")

    text_parts: list[str] = []
    for block in content:
        if isinstance(block, Mapping) and isinstance(block.get("text"), str):
            text_parts.append(block["text"])
    if not text_parts:
        raise PromptPlanError("Bedrock Nova planner returned no text")
    return "\n".join(text_parts)


def _json_object_from_model_text(text: str, source_name: str) -> Mapping[str, Any]:
    text = text.strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = _json_object_from_wrapped_text(text, source_name)
    if not isinstance(payload, Mapping):
        raise PromptPlanError(f"{source_name} returned JSON that is not an object")
    return payload


def _sanitize_bedrock_prompt_plan_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    target_style = payload.get("target_style")
    if not isinstance(target_style, Mapping):
        return payload

    sanitized_payload = dict(payload)
    sanitized_target = dict(target_style)
    for field_name in ("drapery_and_torso", "headpiece_or_ornament"):
        value = target_style.get(field_name)
        if isinstance(value, list):
            sanitized_target[field_name] = [
                item
                for item in value
                if not isinstance(item, str) or _find_forbidden_target_style_term(item) is None
            ]

    scalar_defaults = {
        "bust_framing": "classical bust cropped at upper chest",
        "statue_angle": "matching the reference head angle",
    }
    for field_name, default_value in scalar_defaults.items():
        value = target_style.get(field_name)
        if isinstance(value, str) and _find_forbidden_target_style_term(value) is not None:
            sanitized_target[field_name] = default_value

    sanitized_payload["target_style"] = sanitized_target
    return sanitized_payload


def _json_object_from_wrapped_text(text: str, source_name: str) -> Mapping[str, Any]:
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S | re.I)
    if fence_match is not None:
        candidate = fence_match.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise PromptPlanError(f"{source_name} returned invalid JSON")
        candidate = text[start : end + 1]
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as error:
        raise PromptPlanError(f"{source_name} returned invalid JSON") from error
    if not isinstance(payload, Mapping):
        raise PromptPlanError(f"{source_name} returned JSON that is not an object")
    return payload


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


def _enum_field(
    payload: Mapping[str, Any],
    field_name: str,
    allowed_values: tuple[str, ...],
) -> str:
    value = _string_field(payload, field_name)
    if value not in allowed_values:
        allowed = ", ".join(allowed_values)
        raise PromptPlanError(f"{field_name} must be one of the allowed values: {allowed}")
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


def _validate_target_style(target_style: TargetStyle) -> None:
    descriptors = (
        target_style.bust_framing,
        target_style.statue_angle,
        *target_style.drapery_and_torso,
        *target_style.headpiece_or_ornament,
    )
    for descriptor in descriptors:
        matched_term = _find_forbidden_target_style_term(descriptor)
        if matched_term is not None:
            raise PromptPlanError(
                "target_style must not include material, lighting, background, "
                f"weathering, lava, or finish language (matched {matched_term!r})"
            )


def _find_forbidden_identity_label(descriptor: str) -> str | None:
    for label in sorted(FORBIDDEN_IDENTITY_LABELS, key=len, reverse=True):
        pattern = _identity_label_pattern(label)
        if pattern.search(descriptor):
            return label
    return None


def _find_forbidden_target_style_term(descriptor: str) -> str | None:
    for term in sorted(FORBIDDEN_TARGET_STYLE_TERMS, key=len, reverse=True):
        pattern = _identity_label_pattern(term)
        if pattern.search(descriptor):
            return term
    return None


def _identity_label_pattern(label: str) -> re.Pattern[str]:
    escaped_label = re.escape(label).replace(r"\ ", r"\s+")
    return re.compile(rf"(?<![a-z0-9]){escaped_label}(?![a-z0-9])", re.IGNORECASE)


def _translate_expression_phrase(phrase: str) -> str:
    lowered = phrase.lower()
    if any(marker in lowered for marker in ("smile", "laugh", "grin")):
        return "subtle carved smile"
    if any(
        marker in lowered for marker in ("open-mouthed", "visible teeth", "pout", "exaggerated")
    ):
        return "soft neutral expression"
    return _sanitize_reference_phrase(phrase)


def _sanitize_reference_phrase(phrase: str, *, strip_color_words: bool = False) -> str:
    sanitized = phrase.strip()
    lowered = sanitized.lower()
    if any(term in lowered for term in SUPPRESSED_SELFIE_ARTIFACT_TERMS):
        return ""
    if strip_color_words and _contains_suppressed_non_greek_headwear(sanitized):
        return ""
    if strip_color_words and _contains_generic_non_greek_headwear_or_ornament(sanitized):
        return ""

    sanitized = re.sub(r"\bfacing\s+(?:the\s+)?camera\b", "facing forward", sanitized, flags=re.I)
    sanitized = re.sub(
        r"\bgaze\s+directed\s+at\s+(?:the\s+)?camera\b",
        "direct forward gaze",
        sanitized,
        flags=re.I,
    )
    if strip_color_words:
        sanitized = _strip_color_words(sanitized)
    return _normalize_phrase_whitespace(sanitized)


def _sanitize_ornament_phrase(phrase: str) -> str:
    lowered = phrase.lower()
    if _is_empty_ornament_phrase(phrase):
        return ""
    if any(term in lowered for term in SUPPRESSED_SELFIE_ARTIFACT_TERMS):
        return ""
    if _contains_suppressed_non_greek_headwear(phrase):
        return ""
    if _has_allowed_greek_ornament_term(phrase):
        return _normalize_phrase_whitespace(phrase)
    return ""


def _contains_suppressed_non_greek_headwear(phrase: str) -> bool:
    for term in sorted(SUPPRESSED_NON_GREEK_HEADWEAR_TERMS, key=len, reverse=True):
        if _identity_label_pattern(term).search(phrase):
            return True
    return False


def _contains_generic_non_greek_headwear_or_ornament(phrase: str) -> bool:
    if _has_allowed_greek_ornament_term(phrase):
        return False
    for term in sorted(GENERIC_HEADWEAR_OR_ORNAMENT_TERMS, key=len, reverse=True):
        if _identity_label_pattern(term).search(phrase):
            return True
    return False


def _has_allowed_greek_ornament_term(phrase: str) -> bool:
    return any(
        _identity_label_pattern(term).search(phrase) for term in ALLOWED_GREEK_ORNAMENT_TERMS
    )


def _is_empty_ornament_phrase(phrase: str) -> bool:
    normalized = _normalize_phrase_whitespace(phrase).lower()
    return normalized in EMPTY_ORNAMENT_TERMS


def _strip_color_words(phrase: str) -> str:
    color_pattern = "|".join(re.escape(word) for word in NATURAL_COLOR_WORDS)
    phrase = re.sub(rf"\b(?:{color_pattern})\b", "", phrase, flags=re.I)
    phrase = re.sub(r"\bwith\s+(?:lighter\s+)?tones\b", "", phrase, flags=re.I)
    phrase = re.sub(r"\btones\b", "", phrase, flags=re.I)
    return phrase


def _normalize_phrase_whitespace(phrase: str) -> str:
    phrase = re.sub(r"\s+", " ", phrase)
    phrase = re.sub(r"\s+([,;:.])", r"\1", phrase)
    phrase = re.sub(r"\(\s+", "(", phrase)
    phrase = re.sub(r"\s+\)", ")", phrase)
    phrase = re.sub(r"\s*/\s*", "/", phrase)
    return phrase.strip(" ,;")


def _join_phrases(phrases: Iterable[str]) -> str:
    return ", ".join(phrase for phrase in phrases if phrase)


__all__ = [
    "ALLOWED_AGE_BANDS",
    "ALLOWED_GENDER_PRESENTATIONS",
    "FIXED_PROMPT_CONSTRAINTS",
    "FORBIDDEN_IDENTITY_LABELS",
    "PLANNER_INSTRUCTIONS",
    "PromptPlan",
    "PromptPlanError",
    "ReferenceIdentity",
    "SafetyOverrides",
    "TargetStyle",
    "build_prompt_plan_schema",
    "image_path_to_data_url",
    "parse_prompt_plan",
    "plan_prompt_with_bedrock_nova",
    "plan_prompt_with_openai",
    "render_marble_prompt",
]

"""Typed pending-question construction, exact matching, and rendering."""

from __future__ import annotations

import json
import re
from typing import Any

from agent.knowledge.chain_identity import canonicalize_chain_scalar, repo_chain_names
from .contracts import ActionProposal, OptionContract, QuestionContract
from .input_values import extract_json_values, looks_like_wire_method_identity
from .localization import localized


def _question_validation(kind: str, validation: dict[str, Any] | None) -> dict[str, Any]:
    field_validation = dict(validation or {})
    if kind == "manual_value" and not field_validation:
        return {"value_type": "scalar_token", "max_length": 180}
    return field_validation


def manual_question(
    group: str,
    question_id: str,
    prompt: str,
    *,
    field: str,
    kind: str = "manual_value",
    accepted_action_types: tuple[str, ...] = (),
    queue_barrier: bool = False,
    validation: dict[str, Any] | None = None,
    requires_capabilities: tuple[str, ...] = (),
    help_text: str = "",
    completion_effect: str = "",
) -> dict[str, Any]:
    field_validation = _question_validation(kind, validation)
    return {
        "contract_version": 1,
        "id": question_id,
        "group": group,
        "kind": kind,
        "prompt": prompt,
        "field": field,
        "manual_input_allowed": True,
        "options": [],
        "accepted_action_types": sorted({"answer_pending", *accepted_action_types}),
        "queue_barrier": queue_barrier,
        "validation": field_validation,
        "requires_capabilities": list(requires_capabilities),
        "help_text": str(help_text or "").strip(),
        "completion_effect": str(completion_effect or "").strip(),
    }


def choice_question(
    group: str,
    question_id: str,
    prompt: str,
    *,
    field: str,
    options: list[dict[str, Any]],
    kind: str = "numbered_choice",
    manual_input_allowed: bool = False,
    accepted_action_types: tuple[str, ...] = (),
    queue_barrier: bool = False,
    validation: dict[str, Any] | None = None,
    requires_capabilities: tuple[str, ...] = (),
    help_text: str = "",
    completion_effect: str = "",
) -> dict[str, Any]:
    contracts: list[OptionContract] = []
    rendered: list[dict[str, Any]] = []
    for index, raw in enumerate(options, start=1):
        option_id = str(raw.get("id") or index)
        value = raw.get("value")
        expected = (
            dict(raw["expected_patch"])
            if "expected_patch" in raw
            else {f"confirmed_config.{field}": value}
        )
        raw_action = raw.get("action") if isinstance(raw.get("action"), dict) else {}
        action_type = str(raw_action.get("type") or "answer_pending")
        action_arguments = {
            key: item for key, item in raw_action.items() if key != "type"
        }
        action = ActionProposal(
            action_id=f"{question_id}:{option_id}",
            action_type=action_type,
            arguments=action_arguments,
            confidence="high",
        )
        contracts.append(
            OptionContract(
                option_id=option_id,
                value=value,
                action=action,
                expected_patch=expected,
                return_policy=str(raw.get("return_policy") or "fallback"),  # type: ignore[arg-type]
            )
        )
        rendered.append(
            {
                "id": option_id,
                "label": str(raw.get("label") or value),
                "value": value,
                "action": {"type": action.action_type, **dict(action.arguments)},
                "expected_patch": expected,
                "return_policy": str(raw.get("return_policy") or "fallback"),
                **(
                    {"semantic_action": str(raw.get("semantic_action"))}
                    if raw.get("semantic_action")
                    else {}
                ),
            }
        )
    QuestionContract(
        question_id=question_id,
        group=group,
        kind=kind,
        prompt_key=question_id,
        field=field,
        options=tuple(contracts),
        manual_input_allowed=manual_input_allowed,
        requires_capabilities=requires_capabilities,
    )
    return {
        "contract_version": 1,
        "id": question_id,
        "group": group,
        "kind": kind,
        "prompt": prompt,
        "field": field,
        "manual_input_allowed": manual_input_allowed,
        "options": rendered,
        "accepted_action_types": sorted(
            {
                "answer_pending",
                *accepted_action_types,
                *(str(option["action"].get("type") or "") for option in rendered),
            }
            - {""}
        ),
        "queue_barrier": queue_barrier,
        "validation": _question_validation(kind, validation),
        "requires_capabilities": list(requires_capabilities),
        "help_text": str(help_text or "").strip(),
        "completion_effect": str(completion_effect or "").strip(),
    }


def exact_answer(text: str, question: dict[str, Any]) -> tuple[bool, Any]:
    raw = str(text or "").strip()
    if not raw:
        return False, None
    options = list(question.get("options") or [])
    if options:
        normalized = raw.casefold()
        for index, option in enumerate(options, start=1):
            candidates = {
                str(index).casefold(),
                str(option.get("id") or "").strip().casefold(),
                str(option.get("label") or "").strip().casefold(),
                str(option.get("value") or "").strip().casefold(),
            }
            if normalized in candidates:
                return True, option.get("value")
        if question.get("kind") == "yes_no":
            if normalized in {"y", "yes"}:
                return True, options[0].get("value")
            if normalized in {"n", "no"} and len(options) > 1:
                return True, options[1].get("value")
    if question.get("manual_input_allowed") and answer_fits_pending(raw, question):
        return True, normalize_scalar(raw)
    return False, None


def expected_patch_for_value(question: dict[str, Any], value: Any) -> dict[str, Any]:
    """Return the declared postcondition for an exact option value."""

    for option in question.get("options") or []:
        if option.get("value") == value:
            return dict(option.get("expected_patch") or {})
    return {}


def action_for_value(question: dict[str, Any], value: Any) -> dict[str, Any]:
    """Return the executable action declared by an exact visible option."""

    for option in question.get("options") or []:
        if option.get("value") == value:
            return dict(option.get("action") or {})
    return {}


def normalize_scalar(value: str) -> str:
    return str(value or "").strip().rstrip(",，;；").strip()


def literal_matches_validation(value: str, validation: dict[str, Any] | None) -> bool:
    """Return whether a complete user turn is one typed field literal.

    This is deliberately structural.  It never tries to infer intent from
    words, and an undeclared manual field has no local fast path.
    """

    contract = validation if isinstance(validation, dict) else {}
    value_type = str(contract.get("value_type") or "").strip()
    raw = str(value or "").strip()
    if not value_type or not raw or "\n" in raw or "\r" in raw:
        return False
    if value_type == "scalar_token":
        return len(raw) <= int(contract.get("max_length") or 180) and bool(
            re.fullmatch(r"[A-Za-z0-9_./:@+\\=-]+", raw)
        )
    if value_type == "positive_number":
        return bool(re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", raw)) and float(raw) > 0
    if value_type == "positive_integer":
        return raw.isdigit() and int(raw) > 0
    if value_type == "json":
        try:
            return isinstance(json.loads(raw), (dict, list))
        except json.JSONDecodeError:
            return False
    if value_type == "enum":
        return raw.casefold() in {
            str(item).strip().casefold() for item in contract.get("values") or []
        }
    return False


def answer_fits_pending(text: str, question: dict[str, Any]) -> bool:
    """Decide whether one complete turn is a local typed answer.

    Semantic detours deliberately return ``False`` so the LLM planner can
    route them. Invalid scalar candidates return ``True`` when the owning
    domain must explain the field-level validation error.
    """

    raw = _strip_scalar(text)
    if not raw:
        return False
    if question.get("group") != "chain_identity" and canonicalize_chain_scalar(
        raw, known_chains=set(repo_chain_names())
    ):
        return False
    kind = str(question.get("kind") or "")
    validation = question.get("validation") or {}
    input_mode = str(validation.get("input_mode") or "")
    if question.get("manual_input_allowed") and literal_matches_validation(raw, validation):
        return True
    if kind == "yes_no":
        return raw.casefold() in {"y", "yes", "n", "no"} or matches_numbered_option(raw, question)
    if kind == "numbered_choice" and question.get("id") == "unknown_chain_identity_confirm":
        candidate = _chain_from_option_label(str((question.get("options") or [{}])[0].get("label") or ""))
        raw_chain = canonicalize_chain_scalar(raw, known_chains=set(repo_chain_names()))
        if (candidate and raw_chain and candidate == raw_chain) or raw.casefold() in {"y", "yes", "n", "no"}:
            return True
    if kind == "confirm_or_value":
        if raw.casefold() in {"y", "yes", "n", "no"} or matches_numbered_option(raw, question):
            return True
        field = str(question.get("field") or "")
        if field in {"DATA_VOL_SIZE", "ACCOUNTS_VOL_SIZE"}:
            return bool(re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", raw))
        if field == "MAINNET_RPC_URL_REVIEWED":
            return bool(_extract_url_candidate(raw))
        return _is_plain_scalar_answer(raw)
    if kind == "numbered_choice":
        return matches_numbered_option(raw, question)
    if kind == "device" and raw.casefold() in {"y", "yes"}:
        return len(question.get("options") or []) == 1
    if kind == "device" and raw.casefold() in {"n", "no"}:
        return False
    if kind == "device" and raw.isdigit():
        return matches_numbered_option(raw, question)
    if kind == "chain" and question.get("manual_input_allowed"):
        return _is_plain_chain_answer(raw)
    if kind == "url" and question.get("manual_input_allowed"):
        return _is_bare_endpoint_answer(raw)
    if kind == "evidence" and question.get("manual_input_allowed"):
        return _is_structured_evidence_literal(text, allow_params_only=True)
    if kind in {"manual_value", "positive_integer"} and question.get("manual_input_allowed"):
        if input_mode == "rpc_method_or_schema_evidence":
            return bool(
                looks_like_wire_method_identity(raw)
                or _is_structured_evidence_literal(text)
            )
        if input_mode == "rpc_weights":
            return bool(_parse_weight_spec(str(text or "")) or _first_number_text(raw))
        if raw.casefold() in {"y", "yes", "n", "no"}:
            return False
        if str(validation.get("value_type") or "") in {"positive_number", "positive_integer"}:
            # A complete scalar token belongs to the typed field contract even
            # when it is out of range or has the wrong numeric subtype. The
            # owning domain must reject it without giving semantic planning an
            # opportunity to substitute a different value. Multi-word prose
            # remains available to the semantic detour router.
            return bool(re.fullmatch(r"\S+", raw))
        return literal_matches_validation(raw, validation)
    if kind == "device" and question.get("manual_input_allowed"):
        return literal_matches_validation(raw, validation)
    return False


def _is_structured_evidence_literal(value: Any, *, allow_params_only: bool = False) -> bool:
    """Admit only wire syntax that is deterministic without intent routing."""

    text = str(value or "").strip()
    if not text:
        return False
    candidate = text
    if candidate.startswith("```") and candidate.endswith("```"):
        lines = candidate.splitlines()
        if len(lines) >= 3:
            candidate = "\n".join(lines[1:-1]).strip()
    try:
        parsed = json.loads(candidate)
    except (TypeError, ValueError, json.JSONDecodeError):
        parsed = None
    if isinstance(parsed, list):
        return allow_params_only
    if isinstance(parsed, dict) and (
        {"method", "params"}.issubset(parsed)
        or "result" in parsed
        or "error" in parsed
    ):
        return True
    # A pasted request is still deterministic wire evidence when it is
    # surrounded by explanatory prose. Parse the embedded value instead of
    # treating the prose itself as a pending-question answer.
    for embedded in extract_json_values(candidate):
        if isinstance(embedded, dict) and (
            {"method", "params"}.issubset(embedded)
            or "result" in embedded
            or "error" in embedded
        ):
            return True
    first_line = candidate.splitlines()[0].strip().casefold()
    return first_line == "curl" or first_line.startswith("curl ")


def coerce_pending_answer(text: str, question: dict[str, Any]) -> Any:
    """Convert an admitted literal to the exact option/domain value."""

    raw = _strip_scalar(text)
    options = question.get("options") or []
    if raw.isdigit() and options:
        index = int(raw) - 1
        if 0 <= index < len(options):
            return options[index].get("value")
    lowered = raw.casefold()
    if question.get("kind") == "device" and lowered in {"y", "yes"} and len(options) == 1:
        return options[0].get("value")
    if question.get("kind") == "numbered_choice" and question.get("id") == "unknown_chain_identity_confirm":
        candidate = _chain_from_option_label(str((options[0].get("label") if options else "") or ""))
        raw_chain = canonicalize_chain_scalar(raw, known_chains=set(repo_chain_names()))
        if candidate and raw_chain and candidate == raw_chain and options:
            return options[0].get("value")
        if lowered in {"y", "yes"} and options:
            return options[0].get("value")
        if lowered in {"n", "no"} and len(options) > 1:
            return options[1].get("value")
    if question.get("kind") == "yes_no":
        if lowered in {"y", "yes"}:
            return options[0].get("value") if options else True
        if lowered in {"n", "no"}:
            return options[1].get("value") if len(options) > 1 else False
    for option in options:
        candidates = {
            _stringify_option_field(option.get("value")).casefold(),
            _stringify_option_field(option.get("label")).casefold(),
            _stringify_option_field(option.get("id")).casefold(),
        }
        if lowered in candidates:
            return option.get("value")
    return raw


def pending_option_value_exists(value: Any, question: dict[str, Any]) -> bool:
    return any(option.get("value") == value for option in question.get("options") or [])


def render_question(question: dict[str, Any], language: str) -> str:
    lines = [str(question.get("prompt") or "").strip()]
    for index, option in enumerate(question.get("options") or [], start=1):
        lines.append(f"{index}. {option.get('label') or option.get('value')}")
    options = question.get("options") or []
    if question.get("manual_input_allowed"):
        lines.append(localized(
            language,
            "你可以回复编号，也可以直接输入自定义值。" if options else "请直接输入值。",
            "Reply with a number, or type a custom value directly." if options else "Type the value directly.",
        ))
    elif options:
        lines.append(localized(language, "请回复选项编号或选项名称。", "Reply with an option number or option name."))
    return "\n".join(line for line in lines if line)


def _stringify_option_field(value: Any) -> str:
    return "" if value is None else str(value)


def _chain_from_option_label(label: str) -> str:
    known = set(repo_chain_names())
    raw = str(label or "")
    for token in re.findall(r"`([^`]+)`|([A-Za-z0-9][A-Za-z0-9_-]{1,40})", raw):
        candidate_text = next((part for part in token if part), "")
        candidate = canonicalize_chain_scalar(candidate_text, known_chains=known)
        if candidate:
            return candidate
    return canonicalize_chain_scalar(raw, known_chains=known)


def matches_numbered_option(raw: str, question: dict[str, Any]) -> bool:
    options = question.get("options") or []
    if raw.isdigit():
        return 0 <= int(raw) - 1 < len(options)
    lowered = raw.casefold()
    for option in options:
        values = {
            str(option.get("label") or "").strip().casefold(),
            str(option.get("id") or "").strip().casefold(),
        }
        value = option.get("value")
        if value is False:
            values.add("false")
        elif value is not None:
            values.add(str(value).strip().casefold())
        if lowered in values:
            return True
    return False


def _first_number_text(value: Any) -> str:
    match = re.search(r"[0-9]+(?:\.[0-9]+)?", str(value or ""))
    if not match:
        return ""
    number = float(match.group(0))
    return str(int(number)) if number.is_integer() else str(number)


def _strip_scalar(value: str) -> str:
    text = str(value or "").strip()
    if len(text) >= 2 and text[0] in {"`", "'", '"'} and text[-1] in {"`", "'", '"'}:
        text = text[1:-1].strip()
    while text and text[-1] in {",", "，", ";", "；", "、"}:
        text = text[:-1].strip()
    return text


def _is_plain_scalar_answer(value: str) -> bool:
    text = str(value or "").strip()
    return bool(
        text
        and "\n" not in text
        and "\r" not in text
        and len(text) <= 180
        and re.fullmatch(r"[A-Za-z0-9_./:@+\\=-]+", text)
    )


def _extract_url_candidate(value: str) -> str:
    text = str(value or "").strip().strip("`'\"").rstrip(",，;；")
    match = re.search(
        r"(?P<url>(?:https?|wss?)://[^\s'\"`]+|(?:localhost|127\.0\.0\.1|\[[0-9a-fA-F:]+\]|[A-Za-z0-9.-]+):[0-9]{2,5}(?:/[^\s'\"`]*)?)",
        text,
        re.IGNORECASE,
    )
    return match.group("url").rstrip(".,;，；") if match else ""


def _is_bare_endpoint_answer(value: str) -> bool:
    text = str(value or "").strip()
    endpoint = _extract_url_candidate(text)
    return bool(endpoint and endpoint == text.strip("`'\"").rstrip(",，;；"))


def _is_plain_chain_answer(value: str) -> bool:
    text = str(value or "").strip()
    if not text or len(text) > 80:
        return False
    if canonicalize_chain_scalar(text, known_chains=set(repo_chain_names())):
        return True
    # Unknown multi-word names and prose require semantic identity planning.
    # Local admission is reserved for one atomic, source-exact candidate.
    return bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", text))


def _parse_weight_spec(value: str) -> dict[str, int]:
    text = str(value or "").strip()
    if not text:
        return {}
    try:
        candidate = json.loads(text)
    except json.JSONDecodeError:
        candidate = None
    if isinstance(candidate, dict):
        parsed_json: dict[str, int] = {}
        for key, item in candidate.items():
            method = str(key).strip()
            if not method:
                return {}
            try:
                parsed_json[method] = int(item)
            except (TypeError, ValueError):
                return {}
        return parsed_json
    parsed: dict[str, int] = {}
    for method, weight in re.findall(r"([A-Za-z][A-Za-z0-9_./:-]*)\s*(?:=|:)\s*([0-9]+)", text):
        parsed[method] = int(weight)
    return parsed

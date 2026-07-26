"""Typed pending-question construction, exact matching, and rendering."""

from __future__ import annotations

import json
import re
from decimal import Decimal, InvalidOperation
from typing import Any

from agent.knowledge.chain_identity import canonicalize_chain_scalar, repo_chain_names
from agent.workflows.group_registry import GROUP_OWNER
from .action_registry import (
    ACTION_BY_TYPE,
    semantic_value_domain_conflicts,
    validate_candidate_binding_contract,
)
from .contracts import (
    ActionProposal,
    OptionContract,
    QuestionContract,
    TextRef,
    text_ref_from_dict,
    text_ref_to_dict,
)
from .input_values import (
    extract_rpc_method_token_candidates,
    extract_url_candidates,
    has_rpc_wire_evidence,
    looks_like_wire_method_identity,
    parse_weight_spec,
)
from .response_catalog import render_text_ref


QUESTION_CONTRACT_VERSION = 3

_DOMAIN_CONTEXT_KEYS = {
    "config_field",
    "contract_type",
    "endpoint_role",
    "rpc_case",
}
_RPC_ENDPOINT_CONTRACTS = {
    ("final_benchmark", "runtime", "LOCAL_RPC_URL"),
    ("sync_observe", "runtime", "SYNC_OBSERVE_RPC_URL"),
    ("validation", "custom_rpc", ""),
    ("validation", "new_chain", ""),
}


def question_text(message_id: str, **arguments: Any) -> TextRef:
    """Build a language-independent reference to registered question text."""

    return TextRef(message_id=message_id, arguments=arguments)


def _optional_text_ref(value: TextRef | None, *, field: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, TextRef):
        raise TypeError(f"{field} must be a TextRef")
    return text_ref_to_dict(value)


def _validated_text_ref_mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"pending question {field} must be a typed mapping")
    return text_ref_to_dict(text_ref_from_dict(value))


def _question_validation(kind: str, validation: dict[str, Any] | None) -> dict[str, Any]:
    field_validation = dict(validation or {})
    if kind == "evidence" and not field_validation:
        return {"value_type": "evidence_contribution", "max_length": 65536}
    if kind == "manual_value" and not field_validation:
        return {"value_type": "scalar_token", "max_length": 180}
    if str(field_validation.get("value_type") or "") in {
        "positive_number",
        "positive_integer",
    }:
        field_validation.setdefault("normalization", "semantic_scalar")
    return field_validation


def _manual_owner_contract(
    *,
    manual_input_allowed: bool,
    manual_action: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return and validate the typed owner for free-form pending input."""

    if not manual_input_allowed:
        if manual_action:
            raise ValueError("manual_action requires manual_input_allowed")
        return {}
    declared = dict(
        manual_action
        or {"type": "answer_pending", "value_argument": "answer"}
    )
    action_type = str(declared.get("type") or "").strip()
    value_argument = str(declared.get("value_argument") or "").strip()
    spec = ACTION_BY_TYPE.get(action_type)
    if spec is None:
        raise ValueError(f"unknown manual_action type: {action_type or '<missing>'}")
    if not value_argument or value_argument not in spec.allowed_arguments:
        raise ValueError(
            f"manual_action {action_type} requires a writable value_argument"
        )
    return declared


def _candidate_binding_contract(
    *,
    manual_input_allowed: bool,
    candidate_bindings: tuple[dict[str, Any], ...],
) -> list[dict[str, Any]]:
    """Validate declarative domain-action sources for pending candidates."""

    if candidate_bindings and not manual_input_allowed:
        raise ValueError("candidate_bindings require manual_input_allowed")
    declared: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for raw in candidate_bindings:
        binding = validate_candidate_binding_contract(raw)
        action_type = binding["type"]
        value_argument = binding["value_argument"]
        mapping_key = binding.get("mapping_key", "")
        identity = (action_type, value_argument, mapping_key)
        if identity in seen:
            raise ValueError(f"duplicate candidate binding: {identity}")
        seen.add(identity)
        declared.append(binding)
    return declared


def validate_pending_question_contract(question: dict[str, Any]) -> dict[str, Any]:
    """Validate one complete pending-question contract at every trust boundary."""

    if int(question.get("contract_version") or 0) != QUESTION_CONTRACT_VERSION:
        raise ValueError(
            f"pending question requires contract_version {QUESTION_CONTRACT_VERSION}"
        )
    retired_text_keys = {"prompt", "help_text", "completion_effect"}
    present_retired = sorted(retired_text_keys & set(question))
    if present_retired:
        raise ValueError(
            "pending question contains retired rendered text fields: "
            + ", ".join(present_retired)
        )
    prompt_ref = _validated_text_ref_mapping(
        question.get("prompt_ref"),
        field="prompt_ref",
    )
    help_ref = (
        _validated_text_ref_mapping(question.get("help_ref"), field="help_ref")
        if question.get("help_ref") is not None
        else None
    )
    completion_effect_ref = (
        _validated_text_ref_mapping(
            question.get("completion_effect_ref"),
            field="completion_effect_ref",
        )
        if question.get("completion_effect_ref") is not None
        else None
    )
    group = str(question.get("group") or "").strip()
    owner = str(question.get("owner") or "").strip()
    if group not in GROUP_OWNER:
        raise ValueError(f"pending question has unknown group: {group or '<missing>'}")
    if not owner:
        raise ValueError("pending question requires an explicit owner")
    if owner not in set(GROUP_OWNER.values()):
        raise ValueError(f"pending question has unknown owner: {owner!r}")
    domain_context = question.get("domain_context") or {}
    if not isinstance(domain_context, dict):
        raise ValueError("pending question domain_context must be an object")
    unknown_context = set(domain_context) - _DOMAIN_CONTEXT_KEYS
    if unknown_context:
        raise ValueError(
            "pending question has unknown domain_context keys: "
            + ", ".join(sorted(unknown_context))
        )
    contract_type = str(domain_context.get("contract_type") or "")
    endpoint_role = str(domain_context.get("endpoint_role") or "")
    config_field = str(domain_context.get("config_field") or "")
    rpc_case = str(domain_context.get("rpc_case") or "")
    allowed_rpc_cases = (
        {"runtime", "custom_rpc", "new_chain"}
        if contract_type == "rpc_endpoint"
        else {"custom_rpc", "new_chain"}
    )
    if rpc_case and (owner != "chain_rpc" or rpc_case not in allowed_rpc_cases):
        raise ValueError(
            "pending question rpc_case is not valid for its declared domain contract"
        )
    if contract_type:
        if contract_type != "rpc_endpoint":
            raise ValueError(
                f"pending question has unknown domain contract type: {contract_type!r}"
            )
        if owner != "chain_rpc" or group != "endpoint_process":
            raise ValueError(
                "rpc_endpoint domain contract requires the Chain/RPC endpoint owner"
            )
        if (endpoint_role, rpc_case, config_field) not in _RPC_ENDPOINT_CONTRACTS:
            raise ValueError("pending question has an invalid rpc_endpoint domain contract")
    elif endpoint_role or config_field:
        raise ValueError(
            "endpoint_role and config_field require contract_type=rpc_endpoint"
        )
    candidate_bindings = tuple(
        dict(value)
        for value in question.get("candidate_bindings") or []
        if isinstance(value, dict)
    )
    if len(candidate_bindings) != len(question.get("candidate_bindings") or []):
        raise ValueError("candidate_bindings must contain only objects")
    declared_bindings = _candidate_binding_contract(
        manual_input_allowed=question.get("manual_input_allowed") is True,
        candidate_bindings=candidate_bindings,
    )
    manual = question.get("manual_action")
    declared_manual = _manual_owner_contract(
        manual_input_allowed=question.get("manual_input_allowed") is True,
        manual_action=dict(manual) if isinstance(manual, dict) else None,
    )
    structured_key = str(question.get("structured_config_key") or "").strip()
    if structured_key and question.get("manual_input_allowed") is not True:
        raise ValueError("structured_config_key requires manual_input_allowed")
    declared_types = {
        "answer_pending",
        *(str(value.get("type") or "") for value in declared_bindings),
        *(
            [str(declared_manual.get("type") or "")]
            if declared_manual
            else []
        ),
        *(["propose_config_values"] if structured_key else []),
    }
    for option in question.get("options") or []:
        if not isinstance(option, dict):
            raise ValueError("question options must contain only objects")
        option_retired = sorted(
            {"label", "description", "completion_effect"} & set(option)
        )
        if option_retired:
            raise ValueError(
                "question option contains retired rendered text fields: "
                + ", ".join(option_retired)
            )
        option["label_ref"] = _validated_text_ref_mapping(
            option.get("label_ref"),
            field="option.label_ref",
        )
        for ref_key in ("description_ref", "completion_effect_ref"):
            if option.get(ref_key) is not None:
                option[ref_key] = _validated_text_ref_mapping(
                    option.get(ref_key),
                    field=f"option.{ref_key}",
                )
        action = option.get("action")
        if isinstance(action, dict):
            declared_types.add(str(action.get("type") or ""))
    accepted = {
        str(value)
        for value in question.get("accepted_action_types") or []
        if str(value)
    }
    unknown = sorted(
        action_type
        for action_type in accepted | declared_types
        if action_type and action_type not in ACTION_BY_TYPE
    )
    if unknown:
        raise ValueError(
            "question contract declares unknown action types: "
            + ", ".join(unknown)
        )
    missing = sorted(declared_types - accepted - {""})
    if missing:
        raise ValueError(
            "question contract omits accepted action types: "
            + ", ".join(missing)
        )
    normalized = dict(question)
    normalized["prompt_ref"] = prompt_ref
    if help_ref is not None:
        normalized["help_ref"] = help_ref
    if completion_effect_ref is not None:
        normalized["completion_effect_ref"] = completion_effect_ref
    normalized["owner"] = owner
    normalized["domain_context"] = dict(domain_context)
    options = list(question.get("options") or [])
    if options and question.get("manual_input_allowed") is not True:
        value_domain = "closed_options"
    elif options:
        value_domain = "options_or_typed_value"
    elif str(declared_manual.get("type") or "") in {
        "choose_chain",
        "change_chain",
    }:
        value_domain = "researched_identity"
    else:
        value_domain = "typed_value"
    declared_value_domain = str(question.get("value_domain") or value_domain)
    if declared_value_domain != value_domain:
        raise ValueError(
            "pending question value_domain conflicts with its typed contract"
        )
    normalized["value_domain"] = value_domain
    if declared_bindings:
        normalized["candidate_bindings"] = declared_bindings
    if declared_manual:
        normalized["manual_action"] = declared_manual
    return normalized


def manual_question(
    group: str,
    question_id: str,
    prompt: TextRef,
    *,
    owner: str,
    field: str,
    kind: str = "manual_value",
    accepted_action_types: tuple[str, ...] = (),
    manual_action: dict[str, Any] | None = None,
    queue_barrier: bool = False,
    barrier_policy: str = "",
    validation: dict[str, Any] | None = None,
    requires_capabilities: tuple[str, ...] = (),
    help_text: TextRef | None = None,
    completion_effect: TextRef | None = None,
    domain_context: dict[str, Any] | None = None,
    evidence_path: str = "",
    rejection_evidence_value: Any = None,
    structured_input_owner: bool = False,
    structured_config_key: str = "",
    candidate_bindings: tuple[dict[str, Any], ...] = (),
) -> dict[str, Any]:
    if not isinstance(prompt, TextRef):
        raise TypeError("manual_question prompt must be a TextRef")
    field_validation = _question_validation(kind, validation)
    effective_barrier_policy = (
        str(barrier_policy or "").strip()
        or ("exclusive_owner" if queue_barrier else "")
    )
    declared_manual_action = _manual_owner_contract(
        manual_input_allowed=True,
        manual_action=manual_action,
    )
    declared_action_type = str(declared_manual_action.get("type") or "").strip()
    declared_bindings = _candidate_binding_contract(
        manual_input_allowed=True,
        candidate_bindings=candidate_bindings,
    )
    declared_binding_types = {
        str(binding["type"]) for binding in declared_bindings
    }
    structured_key = str(structured_config_key or "").strip()
    return validate_pending_question_contract({
        "contract_version": QUESTION_CONTRACT_VERSION,
        "id": question_id,
        "group": group,
        "owner": owner,
        "kind": kind,
        "prompt_ref": text_ref_to_dict(prompt),
        "field": field,
        "manual_input_allowed": True,
        **({"structured_input_owner": True} if structured_input_owner else {}),
        **({"structured_config_key": structured_key} if structured_key else {}),
        **({"candidate_bindings": declared_bindings} if declared_bindings else {}),
        "options": [],
        "accepted_action_types": sorted({
            "answer_pending",
            *accepted_action_types,
            *([declared_action_type] if declared_action_type else []),
            *declared_binding_types,
            *(["propose_config_values"] if structured_key else []),
            *(["start_evidence_collection"] if kind == "evidence" else []),
        }),
        **({"manual_action": declared_manual_action} if declared_manual_action else {}),
        "queue_barrier": queue_barrier,
        **(
            {"barrier_policy": effective_barrier_policy}
            if effective_barrier_policy
            else {}
        ),
        "validation": field_validation,
        "requires_capabilities": list(requires_capabilities),
        "domain_context": dict(domain_context or {}),
        **(
            {"help_ref": _optional_text_ref(help_text, field="help_text")}
            if help_text is not None
            else {}
        ),
        **(
            {
                "completion_effect_ref": _optional_text_ref(
                    completion_effect,
                    field="completion_effect",
                )
            }
            if completion_effect is not None
            else {}
        ),
        "evidence_path": str(evidence_path or "").strip(),
        **(
            {"rejection_evidence_value": rejection_evidence_value}
            if rejection_evidence_value is not None
            else {}
        ),
    })


def choice_question(
    group: str,
    question_id: str,
    prompt: TextRef,
    *,
    owner: str,
    field: str,
    options: list[dict[str, Any]],
    kind: str = "numbered_choice",
    manual_input_allowed: bool = False,
    accepted_action_types: tuple[str, ...] = (),
    manual_action: dict[str, Any] | None = None,
    queue_barrier: bool = False,
    barrier_policy: str = "",
    validation: dict[str, Any] | None = None,
    requires_capabilities: tuple[str, ...] = (),
    help_text: TextRef | None = None,
    completion_effect: TextRef | None = None,
    domain_context: dict[str, Any] | None = None,
    evidence_path: str = "",
    rejection_evidence_value: Any = None,
    structured_config_key: str = "",
    candidate_bindings: tuple[dict[str, Any], ...] = (),
) -> dict[str, Any]:
    if not isinstance(prompt, TextRef):
        raise TypeError("choice_question prompt must be a TextRef")
    effective_barrier_policy = (
        str(barrier_policy or "").strip()
        or ("exclusive_owner" if queue_barrier else "")
    )
    declared_manual_action = _manual_owner_contract(
        manual_input_allowed=manual_input_allowed,
        manual_action=manual_action,
    )
    declared_action_type = str(declared_manual_action.get("type") or "").strip()
    declared_bindings = _candidate_binding_contract(
        manual_input_allowed=manual_input_allowed,
        candidate_bindings=candidate_bindings,
    )
    declared_binding_types = {
        str(binding["type"]) for binding in declared_bindings
    }
    structured_key = str(structured_config_key or "").strip()
    if structured_key and not manual_input_allowed:
        raise ValueError("structured_config_key requires manual_input_allowed")
    contracts: list[OptionContract] = []
    rendered: list[dict[str, Any]] = []
    for index, raw in enumerate(options, start=1):
        option_id = str(raw.get("id") or index)
        value = raw.get("value")
        label_ref = raw.get("label")
        if not isinstance(label_ref, TextRef):
            raise TypeError(
                f"choice_question option {option_id} label must be a TextRef"
            )
        description_ref = raw.get("description")
        if description_ref is not None and not isinstance(description_ref, TextRef):
            raise TypeError(
                f"choice_question option {option_id} description must be a TextRef"
            )
        option_completion_ref = raw.get("completion_effect")
        if (
            option_completion_ref is not None
            and not isinstance(option_completion_ref, TextRef)
        ):
            raise TypeError(
                f"choice_question option {option_id} completion_effect must be a TextRef"
            )
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
                label=label_ref,
                description=description_ref,
                completion_effect=option_completion_ref,
                manual_entry=raw.get("manual_entry") is True,
                return_policy=str(raw.get("return_policy") or "fallback"),  # type: ignore[arg-type]
            )
        )
        rendered.append(
            {
                "id": option_id,
                "label_ref": text_ref_to_dict(label_ref),
                **(
                    {"description_ref": text_ref_to_dict(description_ref)}
                    if description_ref is not None
                    else {}
                ),
                **(
                    {"completion_effect_ref": text_ref_to_dict(option_completion_ref)}
                    if option_completion_ref is not None
                    else {}
                ),
                "manual_entry": raw.get("manual_entry") is True,
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
        prompt=prompt,
        owner=owner,
        field=field,
        options=tuple(contracts),
        manual_input_allowed=manual_input_allowed,
        requires_capabilities=requires_capabilities,
        domain_context=dict(domain_context or {}),
        help_text=help_text,
        completion_effect=completion_effect,
    )
    return validate_pending_question_contract({
        "contract_version": QUESTION_CONTRACT_VERSION,
        "id": question_id,
        "group": group,
        "owner": owner,
        "kind": kind,
        "prompt_ref": text_ref_to_dict(prompt),
        "field": field,
        "manual_input_allowed": manual_input_allowed,
        **({"structured_config_key": structured_key} if structured_key else {}),
        **({"candidate_bindings": declared_bindings} if declared_bindings else {}),
        "options": rendered,
        "accepted_action_types": sorted(
            {
                "answer_pending",
                *accepted_action_types,
                *([declared_action_type] if declared_action_type else []),
                *declared_binding_types,
                *(["propose_config_values"] if structured_key else []),
                *(str(option["action"].get("type") or "") for option in rendered),
            }
            - {""}
        ),
        **({"manual_action": declared_manual_action} if declared_manual_action else {}),
        "queue_barrier": queue_barrier,
        **(
            {"barrier_policy": effective_barrier_policy}
            if effective_barrier_policy
            else {}
        ),
        "validation": _question_validation(kind, validation),
        "requires_capabilities": list(requires_capabilities),
        "domain_context": dict(domain_context or {}),
        **(
            {"help_ref": _optional_text_ref(help_text, field="help_text")}
            if help_text is not None
            else {}
        ),
        **(
            {
                "completion_effect_ref": _optional_text_ref(
                    completion_effect,
                    field="completion_effect",
                )
            }
            if completion_effect is not None
            else {}
        ),
        "evidence_path": str(evidence_path or "").strip(),
        **(
            {"rejection_evidence_value": rejection_evidence_value}
            if rejection_evidence_value is not None
            else {}
        ),
    })


def exact_option_answer(text: str, question: dict[str, Any]) -> tuple[bool, Any]:
    """Match only an option explicitly declared by the active question."""

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
                str(option.get("value") or "").strip().casefold(),
            }
            label_ref = option.get("label_ref")
            if isinstance(label_ref, dict):
                reference = text_ref_from_dict(label_ref)
                candidates.update(
                    render_text_ref(
                        reference,
                        language,
                        kind="option_label",
                    ).strip().casefold()
                    for language in ("en", "zh")
                )
            if normalized in candidates:
                return True, option.get("value")
        if question.get("kind") == "yes_no":
            if normalized in {"y", "yes"}:
                return True, options[0].get("value")
            if normalized in {"n", "no"} and len(options) > 1:
                return True, options[1].get("value")
    return False, None


def exact_answer(text: str, question: dict[str, Any]) -> tuple[bool, Any]:
    """Match a declared option or a complete typed manual contract value."""

    matched, value = exact_option_answer(text, question)
    if matched:
        return True, value
    raw = str(text or "").strip()
    if question.get("manual_input_allowed") and answer_fits_pending(raw, question):
        return True, coerce_pending_answer(raw, question)
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
    if value_type == "bounded_text":
        # Multi-word text requires semantic ownership before it becomes a
        # value. A single token can still use the deterministic fast path.
        return len(raw) <= int(contract.get("max_length") or 512) and bool(
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
    if value_type == "url":
        return bool(_extract_url_candidate(raw))
    if value_type == "enum":
        return raw.casefold() in {
            str(item).strip().casefold() for item in contract.get("values") or []
        }
    return False


def value_satisfies_pending_contract(value: Any, question: dict[str, Any]) -> bool:
    """Validate one value already extracted for the active typed owner.

    This is deliberately separate from ``answer_fits_pending``. The latter
    decides whether an entire raw terminal turn is safe for deterministic
    dispatch; this function validates a source-grounded value after semantic
    planning has isolated it from surrounding prose.
    """

    if pending_option_value_exists(value, question):
        return True
    if question.get("manual_input_allowed") is not True:
        return False
    validation = question.get("validation") or {}
    if str(validation.get("input_mode") or "") == "rpc_weights":
        return bool(parse_weight_spec(value))
    raw = _strip_scalar(str(value or ""))
    if not raw:
        return False
    value_type = str(validation.get("value_type") or "")
    if value_type == "evidence_contribution":
        max_length = int(validation.get("max_length") or 65536)
        return bool(
            len(raw) <= max_length
            and all(
                character.isprintable() or character in {"\n", "\r", "\t"}
                for character in raw
            )
        )
    if value_type == "bounded_text":
        max_length = int(validation.get("max_length") or 512)
        return bool(
            len(raw) <= max_length
            and "\n" not in raw
            and "\r" not in raw
            and all(character.isprintable() for character in raw)
        )
    if value_type in {
        "scalar_token",
        "positive_number",
        "positive_integer",
        "json",
        "url",
        "enum",
    }:
        return literal_matches_validation(raw, validation)
    return answer_fits_pending(raw, question)


def pending_value_identity(value: Any, question: dict[str, Any]) -> str:
    """Return one contract-owned identity for an already valid value."""

    if not value_satisfies_pending_contract(value, question):
        return ""
    for option in question.get("options") or []:
        if isinstance(option, dict) and option.get("value") == value:
            return "option:" + json.dumps(
                option.get("value"),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
    validation = dict(question.get("validation") or {})
    value_type = str(validation.get("value_type") or "")
    raw = _strip_scalar(str(value))
    if value_type == "positive_integer":
        return f"integer:{int(raw)}"
    if value_type == "positive_number":
        try:
            number = Decimal(raw).normalize()
        except InvalidOperation:
            return ""
        return f"number:{format(number, 'f')}"
    if value_type == "enum":
        return f"enum:{raw.casefold()}"
    if value_type == "json":
        parsed = value if isinstance(value, (dict, list)) else json.loads(raw)
        return "json:" + json.dumps(
            parsed,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    if str(validation.get("input_mode") or "") == "rpc_weights":
        parsed = parse_weight_spec(value)
        return "rpc_weights:" + json.dumps(
            parsed,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    return f"scalar:{raw}"


def typed_pending_value_candidates(
    text: str,
    question: dict[str, Any],
) -> tuple[str, ...]:
    """Expose deterministic syntax candidates without deciding user intent."""

    if question.get("manual_input_allowed") is not True:
        return ()
    validation = question.get("validation") or {}
    if str(question.get("kind") or "") == "url" or str(
        validation.get("value_type") or ""
    ) == "url":
        return tuple(
            candidate
            for candidate in extract_url_candidates(text)
            if value_satisfies_pending_contract(candidate, question)
        )
    input_mode = str(validation.get("input_mode") or "")
    if input_mode == "rpc_method_or_schema_evidence":
        candidates = extract_rpc_method_token_candidates(text)
        if has_rpc_wire_evidence(text):
            candidates.append(str(text or "").strip())
        return tuple(dict.fromkeys(
            candidate
            for candidate in candidates
            if value_satisfies_pending_contract(candidate, question)
        ))
    if str(validation.get("value_type") or "") == "evidence_contribution":
        return (str(text or "").strip(),) if has_rpc_wire_evidence(text) else ()
    literal = _strip_scalar(text)
    if literal_matches_validation(literal, validation):
        return (literal,)
    return ()


def pending_contract_allows_semantic_scalar_normalization(
    question: dict[str, Any],
) -> bool:
    """Return whether the typed question declares semantic scalar coercion."""

    validation = question.get("validation") or {}
    declared = str(validation.get("normalization") or "")
    if declared:
        return declared == "semantic_scalar"
    # Checkpoints created before this capability was made explicit still carry
    # the typed numeric schema. Preserve their semantics without relying on a
    # question id, field name, language, or value vocabulary.
    return str(validation.get("value_type") or "") in {
        "positive_number",
        "positive_integer",
    }


def manual_literal_violation(value: str, question: dict[str, Any]) -> dict[str, Any]:
    """Return a deterministic field-contract violation, if one is certain.

    Ambiguous prose is intentionally not classified here. The semantic planner
    remains responsible for detours; this boundary only rejects input shapes
    that cannot be legal under the declared pending contract.
    """

    if question.get("manual_input_allowed") is not True:
        return {}
    validation = question.get("validation") or {}
    raw = _strip_scalar(value)
    normalized = raw.casefold()
    for index, option in enumerate(question.get("options") or [], start=1):
        candidates = {
            str(index).casefold(),
            str(option.get("id") or "").strip().casefold(),
            str(option.get("value") or "").strip().casefold(),
        }
        if normalized in candidates:
            return {}
    if str(question.get("kind") or "") == "yes_no" and normalized in {"y", "yes", "n", "no"}:
        return {}
    value_type = str(validation.get("value_type") or "")
    max_length = int(validation.get("max_length") or 180)
    scalar_shape = bool(raw and "\n" not in raw and "\r" not in raw and re.fullmatch(r"\S+", raw))
    if value_type == "scalar_token" and scalar_shape and len(raw) > max_length:
        return {"code": "max_length", "max_length": max_length}
    if value_type in {"positive_number", "positive_integer"} and scalar_shape:
        if not literal_matches_validation(raw, validation):
            return {"code": value_type}
    return {}


def answer_fits_pending(text: str, question: dict[str, Any]) -> bool:
    """Decide whether one complete turn is a local typed answer.

    Semantic detours deliberately return ``False`` so the LLM planner can
    route them. Invalid scalar candidates return ``True`` when the owning
    domain must explain the field-level validation error.
    """

    raw = _strip_scalar(text)
    if not raw:
        return False
    if semantic_value_domain_conflicts(
        text,
        owning_group=str(question.get("group") or ""),
        pending_question=question,
    ):
        return False
    kind = str(question.get("kind") or "")
    validation = question.get("validation") or {}
    input_mode = str(validation.get("input_mode") or "")
    if question.get("manual_input_allowed") and literal_matches_validation(raw, validation):
        return True
    if kind == "yes_no":
        return raw.casefold() in {"y", "yes", "n", "no"} or matches_numbered_option(raw, question)
    if kind == "confirm_or_value":
        if raw.casefold() in {"y", "yes", "n", "no"} or matches_numbered_option(raw, question):
            return True
        return bool(
            question.get("manual_input_allowed")
            and literal_matches_validation(raw, validation)
        )
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
            return bool(parse_weight_spec(text))
        if raw.casefold() in {"y", "yes", "n", "no"}:
            return False
        if manual_literal_violation(raw, question):
            return True
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
    if has_rpc_wire_evidence(candidate, allow_params_only=allow_params_only):
        return True
    first_line = candidate.splitlines()[0].strip().casefold()
    return first_line == "curl" or first_line.startswith("curl ")


def coerce_pending_answer(text: str, question: dict[str, Any]) -> Any:
    """Convert an admitted literal to the exact option/domain value."""

    raw = _strip_scalar(text)
    validation = question.get("validation") or {}
    if str(validation.get("input_mode") or "") == "rpc_weights":
        weights = parse_weight_spec(text)
        if weights:
            return weights
    options = question.get("options") or []
    if raw.isdigit() and options:
        index = int(raw) - 1
        if 0 <= index < len(options):
            return options[index].get("value")
    lowered = raw.casefold()
    if question.get("kind") == "device" and lowered in {"y", "yes"} and len(options) == 1:
        return options[0].get("value")
    if question.get("kind") == "yes_no":
        if lowered in {"y", "yes"}:
            return options[0].get("value") if options else True
        if lowered in {"n", "no"}:
            return options[1].get("value") if len(options) > 1 else False
    for option in options:
        candidates = {
            _stringify_option_field(option.get("value")).casefold(),
            _stringify_option_field(option.get("id")).casefold(),
        }
        if lowered in candidates:
            return option.get("value")
    return raw


def pending_option_value_exists(value: Any, question: dict[str, Any]) -> bool:
    return any(option.get("value") == value for option in question.get("options") or [])


def render_question(question: dict[str, Any], language: str) -> str:
    validated = validate_pending_question_contract(question)
    lines = [
        render_text_ref(
            text_ref_from_dict(validated["prompt_ref"]),
            language,
            kind="question_prompt",
        )
    ]
    for index, option in enumerate(validated.get("options") or [], start=1):
        label = render_text_ref(
            text_ref_from_dict(option["label_ref"]),
            language,
            kind="option_label",
        )
        description = (
            render_text_ref(
                text_ref_from_dict(option["description_ref"]),
                language,
                kind="option_description",
            )
            if option.get("description_ref")
            else ""
        )
        lines.append(f"{index}. {label}{f' — {description}' if description else ''}")
    options = validated.get("options") or []
    if validated.get("manual_input_allowed"):
        instruction_id = (
            "question.instruction.options_or_value"
            if options
            else "question.instruction.value"
        )
        lines.append(render_text_ref(
            TextRef(instruction_id),
            language,
            kind="question_instruction",
        ))
    elif options:
        lines.append(render_text_ref(
            TextRef("question.instruction.option"),
            language,
            kind="question_instruction",
        ))
    return "\n".join(line for line in lines if line)


def render_question_context(question: dict[str, Any], language: str) -> str:
    """Render registered help and completion text for consultation responses."""

    validated = validate_pending_question_contract(question)
    lines: list[str] = []
    for key, kind in (
        ("help_ref", "question_help"),
        ("completion_effect_ref", "completion_effect"),
    ):
        if validated.get(key):
            lines.append(render_text_ref(
                text_ref_from_dict(validated[key]),
                language,
                kind=kind,
            ))
    return "\n".join(lines)


def _stringify_option_field(value: Any) -> str:
    return "" if value is None else str(value)


def matches_numbered_option(raw: str, question: dict[str, Any]) -> bool:
    options = question.get("options") or []
    if raw.isdigit():
        return 0 <= int(raw) - 1 < len(options)
    lowered = raw.casefold()
    for option in options:
        values = {
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

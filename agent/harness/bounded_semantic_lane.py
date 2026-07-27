"""Generic bounded semantic-value mapping without product-state mutation.

This optional lane recognizes one value from immutable question and action
registries.  Models may select and independently review a candidate, but only
deterministic code may materialize the corresponding action.  Any ambiguity or
contract failure escalates to the existing hierarchical planner.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from typing import Any, Mapping, Sequence

from ..llm.providers import provider_from_config
from .action_registry import (
    ACTION_ARGUMENT_SCHEMAS,
    ACTION_BY_TYPE,
    action_registry_contract_hash,
    registered_semantic_value_domains,
    validate_action_contract,
)
from .plan_coverage import TurnClause, segment_user_turn
from .questions import (
    pending_value_identity,
    typed_pending_value_candidates,
    value_satisfies_pending_contract,
)
from .semantic_compiler import request_semantic_compilation_result
from .state import AgentGraphState


_MAPPER_KEYS = frozenset({
    "decision",
    "candidate_id",
    "source_quote",
    "clause_verdicts",
    "sibling_verdict",
    "reason",
})
_CLAUSE_VERDICT_KEYS = frozenset({"clause_id", "verdict", "reason"})
_ADMISSION_KEYS = frozenset({
    "mapping_hash",
    "candidate_id",
    "verdict",
    "checks",
    "reason",
})
_ADMISSION_CHECK_KEYS = frozenset({
    "candidate_immutable",
    "quote_is_exact",
    "all_clauses_covered",
    "no_sibling_demand",
    "no_ambiguity",
})
_MAPPER_DECISIONS = frozenset({"selected", "escalate"})
_CLAUSE_VERDICTS = frozenset({"direct", "support", "unrelated"})
_SIBLING_VERDICTS = frozenset({"none", "present", "ambiguous"})
_ADMISSION_VERDICTS = frozenset({"accept", "reject"})
_IDENTIFIER_CHARACTER = r"\w.-"
_BOUNDED_RECEIPT_VERSION = 2


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _content_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _strict_object(raw: str) -> dict[str, Any] | None:
    try:
        value = json.loads(str(raw or ""))
    except (TypeError, json.JSONDecodeError):
        return None
    return dict(value) if isinstance(value, Mapping) else None


def _question_hash(question: Mapping[str, Any]) -> str:
    return _content_hash(dict(question)) if question else ""


def _candidate_id(payload: Mapping[str, Any]) -> str:
    return "semantic-" + _content_hash(payload)[:24]


def _candidate(
    *,
    source_kind: str,
    canonical_value: Any,
    matched_value: Any,
    action_type: str,
    action_argument: str,
    owner: str,
    group: str,
    source_hash: str,
    question_hash: str,
    registry_hash: str,
    value_identity: str,
) -> dict[str, Any]:
    binding = {
        "source_kind": source_kind,
        "canonical_value": canonical_value,
        "matched_value": matched_value,
        "action_type": action_type,
        "action_argument": action_argument,
        "owner": owner,
        "group": group,
        "source_hash": source_hash,
        "question_hash": question_hash,
        "registry_hash": registry_hash,
        "value_identity": value_identity,
    }
    return {"candidate_id": _candidate_id(binding), **binding}


def _registered_occurrences(text: str, value: str) -> tuple[str, ...]:
    if not value:
        return ()
    pattern = (
        rf"(?<![{_IDENTIFIER_CHARACTER}])"
        rf"{re.escape(value)}"
        rf"(?![{_IDENTIFIER_CHARACTER}])"
    )
    return tuple(
        match.group(0)
        for match in re.finditer(pattern, text, flags=re.IGNORECASE)
    )


def _pending_option_occurrences(
    text: str,
    option: Mapping[str, Any],
) -> tuple[str, ...]:
    """Return exact source anchors that identify one pending option."""

    anchors = {
        str(option.get(key) or "").strip()
        for key in ("id", "value")
    } - {""}
    return tuple(dict.fromkeys(
        occurrence
        for anchor in anchors
        for occurrence in _registered_occurrences(text, anchor)
    ))


def build_candidate_catalog(
    state: Mapping[str, Any],
    text: str,
) -> tuple[dict[str, Any], ...]:
    """Build immutable candidates only from active product registries."""

    source = str(text or "")
    if not source.strip():
        return ()
    pending = (
        dict(state.get("pending_question") or {})
        if isinstance(state.get("pending_question"), Mapping)
        else {}
    )
    source_hash = _content_hash(source)
    question_hash = _question_hash(pending)
    registry_hash = action_registry_contract_hash()
    candidates: list[dict[str, Any]] = []

    for option in pending.get("options") or []:
        if not isinstance(option, Mapping) or "value" not in option:
            continue
        value = option.get("value")
        identity = pending_value_identity(value, pending)
        if not identity:
            continue
        for occurrence in _pending_option_occurrences(source, option):
            candidates.append(_candidate(
                source_kind="pending_option",
                canonical_value=value,
                matched_value=occurrence,
                action_type="answer_pending",
                action_argument="selected_value",
                owner="coordinator",
                group=str(pending.get("group") or ""),
                source_hash=source_hash,
                question_hash=question_hash,
                registry_hash=registry_hash,
                value_identity=identity,
            ))

    for value in typed_pending_value_candidates(source, pending):
        if not value_satisfies_pending_contract(value, pending):
            continue
        identity = pending_value_identity(value, pending)
        if not identity:
            continue
        candidates.append(_candidate(
            source_kind="pending_manual",
            canonical_value=value,
            matched_value=value,
            action_type="answer_pending",
            action_argument="answer",
            owner="coordinator",
            group=str(pending.get("group") or ""),
            source_hash=source_hash,
            question_hash=question_hash,
            registry_hash=registry_hash,
            value_identity=identity,
        ))

    for record in registered_semantic_value_domains():
        action_type = str(record.get("action_type") or "")
        argument = str(record.get("argument") or "")
        target_group = str(record.get("target_group") or "")
        spec = ACTION_BY_TYPE.get(action_type)
        if (
            spec is None
            or not target_group
            or argument not in spec.allowed_arguments
        ):
            continue
        registered_value = str(record.get("value") or "")
        for occurrence in _registered_occurrences(source, registered_value):
            candidates.append(_candidate(
                source_kind="registered_value",
                canonical_value=record.get("canonical_value"),
                matched_value=occurrence,
                action_type=action_type,
                action_argument=argument,
                owner=spec.owner,
                group=target_group,
                source_hash=source_hash,
                question_hash=question_hash,
                registry_hash=registry_hash,
                value_identity=(
                    f"registry:{record.get('semantic_owner')}:{registered_value.casefold()}"
                ),
            ))

    unique: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        unique.setdefault(str(candidate["candidate_id"]), candidate)
    return tuple(unique.values())


def _mapper_prompt() -> str:
    return (
        "Select at most one immutable candidate from candidate_catalog. Return one "
        "strict JSON object with exactly decision, candidate_id, source_quote, "
        "clause_verdicts, sibling_verdict, and reason. decision is selected or "
        "escalate. candidate_id must be copied exactly or empty when escalating. "
        "source_quote must be one exact substring of the original turn and directly "
        "express the selected value. Include exactly one verdict for every supplied "
        "clause_id, in order. Every clause_verdicts item must contain exactly "
        "clause_id, verdict, and reason; verdict is direct, support, or unrelated. Select only "
        "when exactly one candidate represents the complete turn and every clause is "
        "direct or support. Set sibling_verdict to present or ambiguous when any "
        "independent request, comparison, negation, example-only mention, conflicting "
        "value, or uncertainty remains. Otherwise use none. Never create or edit a "
        "candidate, action, value, owner, or group."
    )


def _admission_prompt() -> str:
    return (
        "Independently review one immutable bounded semantic mapping. Return one "
        "strict JSON object with exactly mapping_hash, candidate_id, verdict, checks, "
        "and reason. Copy mapping_hash and candidate_id exactly. verdict is accept or "
        "reject. checks must contain exactly candidate_immutable, quote_is_exact, "
        "all_clauses_covered, no_sibling_demand, and no_ambiguity, each boolean. "
        "Accept only when every check is true. Do not edit the mapping or propose an "
        "action."
    )


def _validate_mapping(
    raw: str,
    *,
    text: str,
    clauses: Sequence[TurnClause],
    catalog: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    document = _strict_object(raw)
    if document is None or frozenset(document) != _MAPPER_KEYS:
        return None
    if document.get("decision") not in _MAPPER_DECISIONS:
        return None
    if document.get("decision") != "selected":
        return None
    candidate_id = str(document.get("candidate_id") or "")
    matches = [
        dict(candidate)
        for candidate in catalog
        if str(candidate.get("candidate_id") or "") == candidate_id
    ]
    if len(matches) != 1:
        return None
    quote = str(document.get("source_quote") or "")
    if not quote or quote not in text:
        return None
    if document.get("sibling_verdict") != "none":
        return None
    verdicts = document.get("clause_verdicts")
    if not isinstance(verdicts, list) or len(verdicts) != len(clauses):
        return None
    expected_ids = [clause.clause_id for clause in clauses]
    actual_ids: list[str] = []
    for verdict in verdicts:
        if not isinstance(verdict, Mapping):
            return None
        if frozenset(verdict) != _CLAUSE_VERDICT_KEYS:
            return None
        actual_ids.append(str(verdict.get("clause_id") or ""))
        if verdict.get("verdict") not in _CLAUSE_VERDICTS:
            return None
        if verdict.get("verdict") == "unrelated":
            return None
    if actual_ids != expected_ids:
        return None
    direct_clause_texts = [
        clause.text
        for clause, verdict in zip(clauses, verdicts)
        if verdict.get("verdict") == "direct"
    ]
    if not direct_clause_texts or not any(
        quote in clause_text for clause_text in direct_clause_texts
    ):
        return None
    selected = matches[0]
    if selected.get("source_kind") in {
        "pending_option",
        "pending_manual",
        "registered_value",
    }:
        matched_value = str(selected.get("matched_value") or "")
        if not matched_value or matched_value.casefold() not in quote.casefold():
            return None
    selected_identity = str(selected.get("value_identity") or "")
    competing = {
        str(candidate.get("value_identity") or "")
        for candidate in catalog
        if (
            str(candidate.get("candidate_id") or "") != candidate_id
            and str(candidate.get("matched_value") or "").casefold()
            in text.casefold()
        )
    }
    if competing - {selected_identity}:
        return None
    return {
        **document,
        "selected_candidate": selected,
    }


def _materialize_action(
    candidate: Mapping[str, Any],
    source_quote: str,
) -> dict[str, Any] | None:
    action_type = str(candidate.get("action_type") or "")
    argument = str(candidate.get("action_argument") or "")
    spec = ACTION_BY_TYPE.get(action_type)
    if spec is None or argument not in spec.allowed_arguments:
        return None
    value = (
        candidate.get("matched_value")
        if candidate.get("source_kind") == "registered_value"
        else candidate.get("canonical_value")
    )
    action: dict[str, Any] = {
        "type": action_type,
        argument: value,
    }
    if "source_evidence" in spec.allowed_arguments:
        action["source_evidence"] = source_quote
    for key, fixed_value in spec.entry_intake_fixed_arguments:
        action[key] = fixed_value
    for key in spec.required_arguments:
        if key in action:
            continue
        schema = ACTION_ARGUMENT_SCHEMAS.get(key) or {}
        if schema.get("type") == "boolean" and key in spec.allowed_arguments:
            action[key] = True
    try:
        return validate_action_contract(action)
    except ValueError:
        return None


def _validate_admission(
    raw: str,
    *,
    mapping_hash: str,
    candidate_id: str,
) -> dict[str, Any] | None:
    document = _strict_object(raw)
    if document is None or frozenset(document) != _ADMISSION_KEYS:
        return None
    if document.get("mapping_hash") != mapping_hash:
        return None
    if document.get("candidate_id") != candidate_id:
        return None
    if document.get("verdict") not in _ADMISSION_VERDICTS:
        return None
    checks = document.get("checks")
    if not isinstance(checks, Mapping):
        return None
    if frozenset(checks) != _ADMISSION_CHECK_KEYS:
        return None
    if (
        document.get("verdict") != "accept"
        or not all(checks.get(key) is True for key in _ADMISSION_CHECK_KEYS)
    ):
        return None
    return document


def _review_ready_document(
    *,
    started_monotonic: float,
    clauses: Sequence[TurnClause],
    candidate: Mapping[str, Any],
    mapping: Mapping[str, Any],
    action: Mapping[str, Any],
    request_sizes: Sequence[int],
    admission_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    owner = str(candidate.get("owner") or "")
    group = str(candidate.get("group") or "")
    operation = (
        "pending_answer"
        if str(candidate.get("source_kind") or "").startswith("pending_")
        else "domain_request"
    )
    quote = str(mapping.get("source_quote") or "")
    source_partition = [
        {
            "unit_id": f"unit-{index}",
            "clause_id": clause.clause_id,
            "source_text": clause.text,
            "operation": operation,
            "owner_routes": [{"owner": owner, "group": group}],
            "reason": "bounded semantic-value candidate admitted",
        }
        for index, clause in enumerate(clauses, start=1)
    ]
    unit_ids = [str(unit["unit_id"]) for unit in source_partition]
    owner_documents = {
        owner: {
            "actions": [dict(action)],
            "bindings": [
                {
                    "unit_id": unit_id,
                    "action_indexes": [0],
                    "disposition": "action",
                    "reason": "deterministic registry materialization",
                }
                for unit_id in unit_ids
            ],
        },
    }
    return {
        "contract_version": 1,
        "planner_kind": "bounded_semantic_value",
        "status": "review_plan",
        "started_monotonic": started_monotonic,
        "clauses": [clause.as_dict() for clause in clauses],
        "source_partition": source_partition,
        "routed_partition": list(source_partition),
        "owner_requests": [{
            "owner": owner,
            "unit_ids": unit_ids,
            "groups": [group] if group else [],
        }],
        "owner_cursor": 1,
        "owner_documents": owner_documents,
        "request_sizes": [int(value) for value in request_sizes],
        "stage_a_calls": 1,
        "stage_b_calls": 0,
        "admission_calls": 1,
        "errors": [],
        "owner_count": 1,
        "unit_count": len(source_partition),
        "bounded_mapping_hash": _content_hash({
            key: value
            for key, value in mapping.items()
            if key != "selected_candidate"
        }),
        "bounded_source_kind": str(candidate.get("source_kind") or ""),
        "bounded_clause_verdicts_hash": str(
            admission_receipt.get("clause_verdicts_hash") or ""
        ),
        "bounded_sibling_verdict": str(
            admission_receipt.get("sibling_verdict") or ""
        ),
        "bounded_checks_hash": str(
            admission_receipt.get("checks_hash") or ""
        ),
        "bounded_admission_receipt": dict(admission_receipt),
    }


def compile_bounded_semantic_value(
    state: AgentGraphState,
    text: str,
    *,
    provider: Any | None = None,
) -> dict[str, Any] | None:
    """Return a review-ready semantic plan, or ``None`` to use hierarchy."""

    started_monotonic = time.monotonic()
    clauses = segment_user_turn(text)
    catalog = build_candidate_catalog(state, text)
    if not clauses or not catalog:
        return None
    active_provider = provider if provider is not None else provider_from_config()
    mapper_payload = {
        "original_turn": str(text),
        "clauses": [clause.as_dict() for clause in clauses],
        "candidate_catalog": [dict(candidate) for candidate in catalog],
        "candidate_catalog_hash": _content_hash(catalog),
    }
    mapper_prompt = _mapper_prompt()
    mapper_result = request_semantic_compilation_result(
        active_provider,
        system_prompt=mapper_prompt,
        request_payload=mapper_payload,
        max_tokens=1400,
        reasoning_mode="disabled",
    )
    mapping = _validate_mapping(
        mapper_result.text,
        text=str(text),
        clauses=clauses,
        catalog=catalog,
    )
    if mapping is None:
        return None
    candidate = dict(mapping["selected_candidate"])
    action = _materialize_action(candidate, str(mapping["source_quote"]))
    if action is None:
        return None
    immutable_mapping = {
        key: value
        for key, value in mapping.items()
        if key != "selected_candidate"
    }
    mapping_hash = _content_hash(immutable_mapping)
    admission_payload = {
        "mapping_hash": mapping_hash,
        "mapping": immutable_mapping,
        "selected_candidate": candidate,
        "materialized_action": action,
        "original_turn_hash": _content_hash(str(text)),
        "candidate_catalog_hash": _content_hash(catalog),
    }
    admission_prompt = _admission_prompt()
    admission_result = request_semantic_compilation_result(
        active_provider,
        system_prompt=admission_prompt,
        request_payload=admission_payload,
        max_tokens=700,
        reasoning_mode="disabled",
    )
    admission = _validate_admission(
        admission_result.text,
        mapping_hash=mapping_hash,
        candidate_id=str(candidate["candidate_id"]),
    )
    if admission is None:
        return None
    candidate_catalog_hash = _content_hash(catalog)
    receipt = {
        "version": _BOUNDED_RECEIPT_VERSION,
        "mapping_hash": mapping_hash,
        "candidate_id": str(candidate["candidate_id"]),
        "source_kind": str(candidate.get("source_kind") or ""),
        "value_identity": str(candidate.get("value_identity") or ""),
        "source_hash": str(candidate.get("source_hash") or ""),
        "question_hash": str(candidate.get("question_hash") or ""),
        "registry_hash": str(candidate.get("registry_hash") or ""),
        "candidate_catalog_hash": candidate_catalog_hash,
        "action_hash": _content_hash(action),
        "checks_hash": _content_hash(admission.get("checks") or {}),
        "clause_verdicts_hash": _content_hash(
            immutable_mapping.get("clause_verdicts") or []
        ),
        "sibling_verdict": str(
            immutable_mapping.get("sibling_verdict") or ""
        ),
        "mapper_provider": mapper_result.provider,
        "mapper_model": mapper_result.model,
        "mapper_response_hash": mapper_result.response_hash,
        "mapper_contract_hash": _content_hash(_mapper_prompt()),
        "admission_provider": admission_result.provider,
        "admission_model": admission_result.model,
        "admission_response_hash": admission_result.response_hash,
        "admission_contract_hash": _content_hash(_admission_prompt()),
    }
    receipt["receipt_hash"] = _content_hash(receipt)
    request_sizes = (
        len(mapper_prompt.encode("utf-8"))
        + len(_canonical_json(mapper_payload).encode("utf-8")),
        len(admission_prompt.encode("utf-8"))
        + len(_canonical_json(admission_payload).encode("utf-8")),
    )
    return _review_ready_document(
        started_monotonic=started_monotonic,
        clauses=clauses,
        candidate=candidate,
        mapping=mapping,
        action=action,
        request_sizes=request_sizes,
        admission_receipt=receipt,
    )


def review_bounded_semantic_plan(
    state: Mapping[str, Any],
    document: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Project one admitted bounded plan without another semantic reviewer."""

    if document.get("planner_kind") != "bounded_semantic_value":
        return None
    receipt = dict(document.get("bounded_admission_receipt") or {})
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_hash"}
    source_text = str((state.get("turn_context") or {}).get("text") or "")
    current_catalog = build_candidate_catalog(state, source_text)
    current_candidates = [
        candidate
        for candidate in current_catalog
        if candidate.get("candidate_id") == receipt.get("candidate_id")
    ]
    if (
        receipt.get("version") != _BOUNDED_RECEIPT_VERSION
        or receipt.get("receipt_hash") != _content_hash(unsigned)
        or receipt.get("source_hash") != _content_hash(source_text)
        or receipt.get("question_hash") != _question_hash(
            dict(state.get("pending_question") or {})
        )
        or receipt.get("registry_hash") != action_registry_contract_hash()
        or receipt.get("candidate_catalog_hash") != _content_hash(current_catalog)
        or len(current_candidates) != 1
        or receipt.get("source_kind") != document.get("bounded_source_kind")
        or receipt.get("source_kind") != current_candidates[0].get("source_kind")
        or receipt.get("value_identity") != current_candidates[0].get(
            "value_identity"
        )
        or receipt.get("mapping_hash") != document.get("bounded_mapping_hash")
        or receipt.get("clause_verdicts_hash")
        != document.get("bounded_clause_verdicts_hash")
        or receipt.get("sibling_verdict")
        != document.get("bounded_sibling_verdict")
        or receipt.get("checks_hash") != document.get("bounded_checks_hash")
        or receipt.get("sibling_verdict") != "none"
        or receipt.get("mapper_contract_hash") != _content_hash(_mapper_prompt())
        or receipt.get("admission_contract_hash") != _content_hash(
            _admission_prompt()
        )
        or not receipt.get("mapper_provider")
        or not receipt.get("mapper_model")
        or receipt.get("mapper_provider") != receipt.get("admission_provider")
        or receipt.get("mapper_model") != receipt.get("admission_model")
        or any(
            len(str(receipt.get(key) or "")) != 64
            for key in (
                "mapper_response_hash",
                "admission_response_hash",
                "clause_verdicts_hash",
                "checks_hash",
            )
        )
    ):
        raise ValueError("bounded semantic admission receipt is stale or invalid")
    owner_documents = dict(document.get("owner_documents") or {})
    if len(owner_documents) != 1:
        raise ValueError("bounded semantic plan must have exactly one owner")
    owner_document = dict(next(iter(owner_documents.values())) or {})
    actions = [
        dict(item)
        for item in owner_document.get("actions") or []
        if isinstance(item, Mapping)
    ]
    bindings = [
        dict(item)
        for item in owner_document.get("bindings") or []
        if isinstance(item, Mapping)
    ]
    if len(actions) != 1 or not bindings:
        raise ValueError("bounded semantic plan cardinality is invalid")
    action = validate_action_contract(actions[0])
    if receipt.get("action_hash") != _content_hash(action):
        raise ValueError("bounded semantic action differs from admitted action")
    source_partition = [
        dict(item)
        for item in document.get("source_partition") or []
        if isinstance(item, Mapping)
    ]
    if not source_partition or len(bindings) != len(source_partition):
        raise ValueError("bounded semantic source partition is invalid")
    units_by_id = {
        str(unit.get("unit_id") or ""): unit
        for unit in source_partition
    }
    if (
        "" in units_by_id
        or len(units_by_id) != len(source_partition)
        or {
            str(binding.get("unit_id") or "")
            for binding in bindings
        } != set(units_by_id)
        or any(
            binding.get("action_indexes") != [0]
            or binding.get("disposition") != "action"
            for binding in bindings
        )
    ):
        raise ValueError("bounded semantic binding is invalid")
    return {
        "actions": [action],
        "semantic_units": [
            {
                "unit_id": unit_id,
                "clause_id": str(unit.get("clause_id") or ""),
                "source_text": str(unit.get("source_text") or ""),
                "disposition": "action",
                "action_indexes": [0],
                "reason": "bounded semantic admission authority accepted",
            }
            for unit_id, unit in units_by_id.items()
        ],
        "pending_choice_contracts": [],
        "reason": "bounded semantic mapping admitted once",
        "planner_metrics": {
            "stage_a_calls": int(document.get("stage_a_calls") or 0),
            "stage_b_calls": 0,
            "admission_calls": 1,
            "model_calls": 2,
            "prompt_bytes": sum(
                int(value) for value in document.get("request_sizes") or []
            ),
            "largest_request_bytes": max(
                (
                    int(value)
                    for value in document.get("request_sizes") or []
                ),
                default=0,
            ),
            "elapsed_ms": int(
                (time.monotonic() - float(document.get("started_monotonic") or time.monotonic()))
                * 1000
            ),
            "owner_count": 1,
            "unit_count": len(source_partition),
        },
    }

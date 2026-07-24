"""Immutable G3 variant contracts and response-bound Codex attestations."""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from tests.agent_live.coverage_evidence import content_hash


VARIANT_ATTESTATION_SCHEMA_VERSION = 2
SOURCE_CONTRACT_SCHEMA_VERSION = 1
VARIANT_CONTRACT_SCHEMA_VERSION = 1
RELATION_BY_VARIANT = {
    "exact": "exact_fixture",
    "isomorphic": "isomorphic_meaning",
    "negative": "adjacent_non_trigger",
    "neighboring": "neighboring_transition",
}
_SHA256 = re.compile(r"[0-9a-f]{64}")


def build_source_contract(case_contract: Mapping[str, Any]) -> dict[str, Any]:
    source_steps = [
        {
            "step_id": str(step.get("step_id") or ""),
            "turn_index": int(step.get("turn_index") or 0),
            "semantic_role": str(step.get("semantic_role") or ""),
        }
        for step in case_contract.get("source_steps") or ()
        if isinstance(step, Mapping)
    ]
    contract = {
        "schema_version": SOURCE_CONTRACT_SCHEMA_VERSION,
        "case_id": str(case_contract.get("case_id") or ""),
        "source_turns_hash": str(case_contract.get("source_turns_hash") or ""),
        "source_steps": source_steps,
    }
    validate_source_contract(contract)
    return contract


def build_variant_contract(
    *,
    variant: str,
    source_contract: Mapping[str, Any],
    declaration: Mapping[str, Any],
) -> dict[str, Any]:
    contract = {
        "schema_version": VARIANT_CONTRACT_SCHEMA_VERSION,
        "variant": str(variant),
        "relation": str(declaration.get("relation") or ""),
        "minimum_attestations": int(
            declaration.get("minimum_attestations") or 0
        ),
        "source_contract_hash": content_hash(source_contract),
        "allowed_source_step_ids": [
            str(step.get("step_id") or "")
            for step in source_contract.get("source_steps") or ()
            if isinstance(step, Mapping)
        ],
        "required_source_step_ids": [
            str(step.get("step_id") or "")
            for step in source_contract.get("source_steps") or ()
            if isinstance(step, Mapping)
        ],
    }
    validate_variant_contract(contract, source_contract=source_contract)
    return contract


def validate_source_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version",
        "case_id",
        "source_turns_hash",
        "source_steps",
    }
    if set(contract) != required:
        raise ValueError("retained source contract fields are invalid")
    if contract.get("schema_version") != SOURCE_CONTRACT_SCHEMA_VERSION:
        raise ValueError("retained source contract schema is unsupported")
    if (
        not str(contract.get("case_id") or "")
        or _SHA256.fullmatch(str(contract.get("source_turns_hash") or ""))
        is None
    ):
        raise ValueError("retained source contract identity is invalid")
    steps = contract.get("source_steps")
    if (
        not isinstance(steps, Sequence)
        or isinstance(steps, (str, bytes))
        or not steps
    ):
        raise ValueError("retained source contract has no semantic steps")
    step_ids: list[str] = []
    turn_indexes: list[int] = []
    for step in steps:
        if (
            not isinstance(step, Mapping)
            or set(step) != {"step_id", "turn_index", "semantic_role"}
            or not str(step.get("step_id") or "")
            or not str(step.get("semantic_role") or "")
            or isinstance(step.get("turn_index"), bool)
            or not isinstance(step.get("turn_index"), int)
            or int(step["turn_index"]) <= 0
        ):
            raise ValueError("retained source semantic step is invalid")
        step_ids.append(str(step["step_id"]))
        turn_indexes.append(int(step["turn_index"]))
    if (
        len(step_ids) != len(set(step_ids))
        or len(turn_indexes) != len(set(turn_indexes))
        or turn_indexes != list(range(1, len(turn_indexes) + 1))
    ):
        raise ValueError("retained source semantic steps are incomplete")
    return dict(contract)


def validate_variant_contract(
    contract: Mapping[str, Any],
    *,
    source_contract: Mapping[str, Any],
) -> dict[str, Any]:
    validate_source_contract(source_contract)
    required = {
        "schema_version",
        "variant",
        "relation",
        "minimum_attestations",
        "source_contract_hash",
        "allowed_source_step_ids",
        "required_source_step_ids",
    }
    if set(contract) != required:
        raise ValueError("retained variant contract fields are invalid")
    variant = str(contract.get("variant") or "")
    if (
        contract.get("schema_version") != VARIANT_CONTRACT_SCHEMA_VERSION
        or RELATION_BY_VARIANT.get(variant) != contract.get("relation")
        or contract.get("source_contract_hash") != content_hash(source_contract)
        or isinstance(contract.get("minimum_attestations"), bool)
        or not isinstance(contract.get("minimum_attestations"), int)
        or int(contract["minimum_attestations"]) < 0
        or (
            variant == "exact"
            and int(contract["minimum_attestations"]) != 0
        )
        or (
            variant != "exact"
            and int(contract["minimum_attestations"]) <= 0
        )
    ):
        raise ValueError("retained variant contract semantics are invalid")
    allowed = contract.get("allowed_source_step_ids")
    required_steps = contract.get("required_source_step_ids")
    source_ids = [
        str(step.get("step_id") or "")
        for step in source_contract.get("source_steps") or ()
    ]
    if (
        not isinstance(allowed, list)
        or allowed != source_ids
        or len(allowed) != len(set(allowed))
        or not isinstance(required_steps, list)
        or required_steps != source_ids
        or len(required_steps) != len(set(required_steps))
        or (
            variant != "exact"
            and int(contract["minimum_attestations"]) > len(required_steps)
        )
    ):
        raise ValueError("retained variant source-step binding is invalid")
    return dict(contract)


def validate_verifier_input_contract(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("retained verifier input contract is missing")
    required = {
        "mode",
        "source_contract",
        "source_contract_hash",
        "variant_contract",
        "variant_contract_hash",
    }
    if not required.issubset(value):
        raise ValueError("retained verifier input contract is incomplete")
    source = validate_source_contract(dict(value["source_contract"]))
    variant = validate_variant_contract(
        dict(value["variant_contract"]),
        source_contract=source,
    )
    if (
        value.get("source_contract_hash") != content_hash(source)
        or value.get("variant_contract_hash") != content_hash(variant)
    ):
        raise ValueError("retained verifier input contract hash is stale")
    return {
        "mode": str(value.get("mode") or ""),
        "source_contract": source,
        "source_contract_hash": str(value["source_contract_hash"]),
        "variant_contract": variant,
        "variant_contract_hash": str(value["variant_contract_hash"]),
    }


def build_variant_attestation(
    *,
    actor: Mapping[str, str],
    verifier_input_contract: Mapping[str, Any],
    execution_binding: Mapping[str, str],
    source_step_id: str,
    semantic_role: str,
    turn_index: int,
    previous_response_hash: str,
    user_message_hash: str,
    selected_at_ns: int,
    declared_at_ns: int,
) -> dict[str, Any]:
    contract = validate_verifier_input_contract(verifier_input_contract)
    unsigned = {
        "schema_version": VARIANT_ATTESTATION_SCHEMA_VERSION,
        "attestation_type": "retained_regression_variant",
        "identity_strength": "auditable_declaration_only",
        "cryptographic_identity_claimed": False,
        "actor": {
            "actor_kind": str(actor.get("actor_kind") or ""),
            "task_id": str(actor.get("task_id") or ""),
            "model": str(actor.get("model") or ""),
        },
        "execution_binding": {
            field: str(execution_binding.get(field) or "")
            for field in (
                "execution_id",
                "session_id",
                "schedule_id",
                "obligation_id",
                "broker_request_id",
                "simulator_attestation_id",
            )
        },
        "source_contract_hash": contract["source_contract_hash"],
        "variant_contract_hash": contract["variant_contract_hash"],
        "relation": contract["variant_contract"]["relation"],
        "source_step_id": str(source_step_id),
        "semantic_role": str(semantic_role),
        "turn_index": int(turn_index),
        "previous_response_hash": str(previous_response_hash),
        "user_message_hash": str(user_message_hash),
        "selected_at_ns": int(selected_at_ns),
        "declared_at_ns": int(declared_at_ns),
    }
    return {**unsigned, "attestation_id": content_hash(unsigned)}


def validate_variant_attestation(
    attestation: Mapping[str, Any],
    *,
    verifier_input_contract: Mapping[str, Any],
    turn: Any,
    decision: Any,
) -> dict[str, Any]:
    contract = validate_verifier_input_contract(verifier_input_contract)
    required = {
        "schema_version",
        "attestation_type",
        "identity_strength",
        "cryptographic_identity_claimed",
        "actor",
        "execution_binding",
        "source_contract_hash",
        "variant_contract_hash",
        "relation",
        "source_step_id",
        "semantic_role",
        "turn_index",
        "previous_response_hash",
        "user_message_hash",
        "selected_at_ns",
        "declared_at_ns",
        "attestation_id",
    }
    if set(attestation) != required:
        raise ValueError("retained variant attestation fields are invalid")
    actor = attestation.get("actor")
    execution_binding = attestation.get("execution_binding")
    if (
        attestation.get("schema_version")
        != VARIANT_ATTESTATION_SCHEMA_VERSION
        or attestation.get("attestation_type")
        != "retained_regression_variant"
        or attestation.get("identity_strength")
        != "auditable_declaration_only"
        or attestation.get("cryptographic_identity_claimed") is not False
        or not isinstance(actor, Mapping)
        or set(actor) != {"actor_kind", "task_id", "model"}
        or actor.get("actor_kind") != "codex"
        or any(
            not str(actor.get(field) or "")
            for field in ("task_id", "model")
        )
        or not isinstance(execution_binding, Mapping)
        or set(execution_binding) != {
            "execution_id",
            "session_id",
            "schedule_id",
            "obligation_id",
            "broker_request_id",
            "simulator_attestation_id",
        }
        or any(not str(value or "") for value in execution_binding.values())
    ):
        raise ValueError("retained variant attestation actor is invalid")
    source = contract["source_contract"]
    variant = contract["variant_contract"]
    source_step = next(
        (
            dict(step)
            for step in source["source_steps"]
            if step.get("step_id") == attestation.get("source_step_id")
        ),
        None,
    )
    if (
        source_step is None
        or attestation.get("source_step_id")
        not in variant["allowed_source_step_ids"]
        or attestation.get("semantic_role")
        != source_step["semantic_role"]
        or attestation.get("source_contract_hash")
        != contract["source_contract_hash"]
        or attestation.get("variant_contract_hash")
        != contract["variant_contract_hash"]
        or attestation.get("relation") != variant["relation"]
    ):
        raise ValueError("retained variant attestation contract binding is invalid")
    if (
        attestation.get("turn_index") != getattr(turn, "turn_index", None)
        or attestation.get("turn_index") != getattr(decision, "turn_index", None)
        or attestation.get("previous_response_hash")
        != getattr(decision, "previous_response_hash", None)
        or attestation.get("user_message_hash")
        != getattr(decision, "user_message_hash", None)
        or attestation.get("selected_at_ns")
        != getattr(decision, "selected_at_ns", None)
        or attestation.get("previous_response_hash")
        != content_hash(getattr(turn, "previous_agent_response", ""))
        or attestation.get("user_message_hash")
        != content_hash(getattr(turn, "user_message", ""))
        or execution_binding.get("execution_id")
        != getattr(decision, "execution_id", None)
        or execution_binding.get("broker_request_id")
        != getattr(decision, "broker_request_id", None)
        or execution_binding.get("simulator_attestation_id")
        != (
            getattr(decision, "simulator_attestation", {}) or {}
        ).get("attestation_id")
        or dict(actor)
        != dict(
            (
                getattr(decision, "simulator_attestation", {}) or {}
            ).get("actor") or {}
        )
        or execution_binding.get("session_id")
        != (
            getattr(decision, "simulator_context_binding", {}) or {}
        ).get("session_id")
        or execution_binding.get("schedule_id")
        != (
            (
                getattr(decision, "simulator_context_binding", {}) or {}
            ).get("control_identity") or {}
        ).get("schedule_id")
        or execution_binding.get("obligation_id")
        != getattr(decision, "obligation_id", None)
    ):
        raise ValueError("retained variant attestation turn binding is stale")
    declared_at_ns = attestation.get("declared_at_ns")
    if (
        isinstance(declared_at_ns, bool)
        or not isinstance(declared_at_ns, int)
        or declared_at_ns < int(getattr(decision, "selected_at_ns", 0))
        or declared_at_ns > int(getattr(decision, "submitted_at_ns", 0))
    ):
        raise ValueError("retained variant attestation timing is invalid")
    unsigned = {
        key: value
        for key, value in attestation.items()
        if key != "attestation_id"
    }
    if attestation.get("attestation_id") != content_hash(unsigned):
        raise ValueError("retained variant attestation identity is stale")
    return dict(attestation)


def collect_valid_variant_attestations(
    context: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    provenance = tuple(
        getattr(decision, "variant_attestation", {}) or {}
        for decision in tuple(getattr(context, "completed_decisions", ()) or ())
        if getattr(decision, "variant_attestation", {}) or {}
    )
    indexed: list[tuple[str, Mapping[str, Any]]] = []
    seen_ids: set[str] = set()
    duplicate_ids: list[str] = []
    for raw in provenance:
        if isinstance(raw, Mapping):
            identity = str(raw.get("attestation_id") or content_hash(raw))
            if identity in seen_ids:
                duplicate_ids.append(identity)
                continue
            seen_ids.add(identity)
            indexed.append((identity, raw))
    turns = {
        int(getattr(turn, "turn_index", -1)): turn
        for turn in tuple(getattr(context, "completed_turns", ()) or ())
    }
    decisions = {
        int(getattr(decision, "turn_index", -1)): decision
        for decision in tuple(getattr(context, "completed_decisions", ()) or ())
    }
    valid: list[dict[str, Any]] = []
    invalid: list[dict[str, str]] = [
        {
            "attestation_id": identity,
            "reason": "retained variant attestation is duplicated",
        }
        for identity in duplicate_ids
    ]
    verifier_input = getattr(context, "verifier_input_contract", {}) or {}
    for identity, attestation in indexed:
        turn_index = int(attestation.get("turn_index") or -1)
        try:
            turn = turns[turn_index]
            decision = decisions[turn_index]
            valid.append(validate_variant_attestation(
                attestation,
                verifier_input_contract=verifier_input,
                turn=turn,
                decision=decision,
            ))
        except (KeyError, TypeError, ValueError) as exc:
            invalid.append({
                "attestation_id": identity,
                "reason": str(exc),
            })
    ordered = sorted(
        valid,
        key=lambda item: int(item["turn_index"]),
    )
    source_steps = {
        str(item["step_id"]): int(item["turn_index"])
        for item in (
            (verifier_input.get("source_contract") or {}).get(
                "source_steps", ()
            )
        )
        if isinstance(item, Mapping)
    }
    observed_source_order = [
        source_steps.get(str(item.get("source_step_id") or ""), -1)
        for item in ordered
    ]
    observed_turns = [int(item["turn_index"]) for item in ordered]
    if (
        len(observed_turns) != len(set(observed_turns))
        or observed_source_order != sorted(observed_source_order)
        or len(observed_source_order) != len(set(observed_source_order))
    ):
        invalid.append({
            "attestation_id": "variant-sequence",
            "reason": (
                "retained variant attestations are not one-to-one ordered "
                "semantic steps"
            ),
        })
    return valid, invalid

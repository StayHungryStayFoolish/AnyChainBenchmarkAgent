"""Executable-provider contracts for Phase 8 G3 retained regressions.

This module compiles the frozen 60-row obligation catalog into execution
contracts and an admission-compatible evidence adapter.  It deliberately does
not execute the CLI, choose future response-driven turns, or claim outcomes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from tests.agent_live.coverage_evidence import content_hash, pty_transcript_hash
from tests.agent_live.chaos_scheduler import build_journey_schedule
from tests.agent_live.dynamic_dual_ai_chaos import (
    JourneyPostconditionResult,
    JourneyPostconditionVerifierDefinition,
    JourneyVerifierContext,
    build_journey_outcome_verifier_registry,
)
from tests.agent_live.product_obligation_evidence import (
    PRODUCT_OBLIGATION_EVIDENCE_SCHEMA_VERSION,
)
from tests.agent_live.retained_regression_obligations import (
    KNOWN_POSTCONDITION_IDS,
    POSTCONDITION_REGISTRY_ID,
    RETAINED_REGRESSION_OBLIGATION_COUNT,
    validate_retained_regression_obligations,
)


RETAINED_REGRESSION_PROVIDER_SCHEMA_VERSION = 1
RETAINED_REGRESSION_PROVIDER_ID = "retained-regression-runner-v1"
RETAINED_REGRESSION_VERIFIER_REGISTRY_ID = (
    "retained-regression-declarative-verifiers-v1"
)
RETAINED_REGRESSION_REGISTRY_IMPORT = (
    "tests.agent_live.retained_regression_runner:"
    "RETAINED_REGRESSION_JOURNEY_VERIFIER_REGISTRY"
)
RETAINED_REGRESSION_SEED = 20260724
REQUIRED_ARTIFACT_ROLES = ("transcript", "runtime_events", "checkpoint_diff")

_RULE_CLASSES: Mapping[str, tuple[str, ...]] = {
    "response_visible": (
        "current_turn_language_preserved",
        "orientation_answered_read_only",
        "consultation_answered_read_only",
        "retained_state_described",
        "resume_action_contract_exposed",
        "execution_stage_explained",
        "evidence_analysis_returned",
    ),
    "response_bound": (
        "response_driven_selection_observed",
        "isomorphic_meaning_attested",
        "adjacent_non_trigger_attested",
        "neighboring_transition_attested",
        "semantic_units_partitioned_in_order",
    ),
    "pending_preserved": (
        "blocking_question_preserved",
        "current_menu_binding_preserved",
        "consultation_preserved_pending_work",
        "unresolved_units_preserved",
    ),
    "pending_advanced": (
        "detected_size_confirmed",
        "typed_detected_value_confirmed",
        "owned_group_backtrack_completed",
        "chain_confirmation_required",
        "custom_method_collection_exited",
    ),
    "action_provenance": (
        "visible_option_action_executed",
        "unknown_chain_identity_resolution_started",
        "chain_mode_change_confirmed",
        "new_chain_request_routed",
        "admitted_mutations_only",
        "approved_execution_submitted_once",
    ),
    "state_changed": (
        "incompatible_state_invalidated",
        "fallback_resumed",
        "compatible_environment_retained",
        "mode_specific_state_invalidated",
        "rpc_groups_rerequired",
        "effective_runtime_method_set_materialized",
        "copied_scalar_normalized",
        "rpc_schema_evidence_extracted",
        "example_endpoint_scope_preserved",
    ),
    "bounded_sequence": (
        "disk_limits_collected_once",
        "multiline_evidence_block_collected_once",
        "status_uses_job_evidence",
        "exact_fixture_turns_observed",
    ),
    "forbidden_absent": (
        "workflow_state_mutated_by_consultation",
        "stale_fallback_emitted",
        "consultation_consumed_as_pending_value",
        "unhandled_visible_option",
        "stale_menu_choice_applied",
        "unknown_chain_silently_configured",
        "environment_text_consumed_as_region",
        "disk_subgroup_repeated",
        "typed_confirmation_rejected_by_side_channel",
        "backtrack_lost_configuration_state",
        "stale_preflight_executed",
        "removed_default_method_materialized",
        "custom_method_collection_looped",
        "example_endpoint_replaced_runtime_endpoint",
        "schema_intake_looped",
        "ordinary_question_counted_as_evidence",
        "blank_prompt_counted_as_evidence",
        "duplicate_job_submission",
        "invented_job_status",
        "sync_observe_ran_rpc_load",
        "ambiguous_change_silently_committed",
        "semantic_unit_dropped",
    ),
}

_DECLARATIVE_ASSERTIONS: Mapping[str, tuple[Mapping[str, Any], ...]] = {
    "response_visible": (
        {
            "fact": "turn.agent_response",
            "operator": "visible_nonempty",
            "scope": "relevant_turns",
        },
        {
            "fact": "turn.agent_response_received_at_ns",
            "operator": "at_or_after",
            "other_fact": "turn.user_message_submitted_at_ns",
            "scope": "relevant_turns",
        },
    ),
    "response_bound": (
        {
            "fact": "turn.user_message_submitted_at_ns",
            "operator": "at_or_after",
            "other_fact": "turn.previous_response_received_at_ns",
            "scope": "relevant_turns",
        },
        {
            "fact": "turn.decision_provenance.previous_response_hash",
            "operator": "equals",
            "other_fact": "turn.previous_agent_response.sha256",
            "scope": "response_driven_turns",
        },
    ),
    "pending_preserved": (
        {
            "fact": "runtime_event.pending_contract",
            "operator": "lineage_preserved",
            "scope": "consultation_turns",
        },
        {
            "fact": "checkpoint_diff.material_paths",
            "operator": "excludes_roots",
            "expected": ["confirmed", "groups", "pending_question"],
            "scope": "consultation_turns",
        },
    ),
    "pending_advanced": (
        {
            "fact": "runtime_event.pending_contract",
            "operator": "consumed_by_accepted_action",
            "scope": "relevant_turns",
        },
        {
            "fact": "runtime_event.admitted_action_provenance",
            "operator": "registered_and_complete",
            "scope": "relevant_turns",
        },
    ),
    "action_provenance": (
        {
            "fact": "runtime_event.admitted_action_provenance",
            "operator": "registered_and_complete",
            "scope": "relevant_turns",
        },
        {
            "fact": "runtime_event.state_diff_hashes",
            "operator": "consistent_with_action_effects",
            "scope": "relevant_turns",
        },
    ),
    "state_changed": (
        {
            "fact": "checkpoint_diff.material_paths",
            "operator": "nonempty",
            "scope": "relevant_turns",
        },
        {
            "fact": "runtime_event.fingerprint_chain",
            "operator": "contiguous",
            "scope": "all_turns",
        },
        {
            "fact": "runtime_event.state_diff_hashes",
            "operator": "consistent_with_checkpoint_diff",
            "scope": "relevant_turns",
        },
    ),
    "bounded_sequence": (
        {
            "fact": "runtime_event.fingerprint_chain",
            "operator": "contiguous",
            "scope": "all_turns",
        },
        {
            "fact": "turn.transcript_hash",
            "operator": "unique_and_complete",
            "scope": "all_turns",
        },
        {
            "fact": "runtime_event.commit_count",
            "operator": "equals_submitted_turn_count",
            "scope": "execution",
        },
    ),
    "forbidden_absent": (
        {
            "fact": "runtime_event.forbidden_observation",
            "operator": "absent",
            "scope": "all_turns",
        },
        {
            "fact": "runtime_event.fingerprint_chain",
            "operator": "contiguous_without_regression",
            "scope": "all_turns",
        },
    ),
}


def retained_regression_postcondition(
    context: JourneyVerifierContext,
) -> JourneyPostconditionResult:
    """Evaluate only reviewed machine predicates over response-bound lineage."""

    postcondition_id = str(context.evaluating_postcondition_id or "")
    lineage_valid, lineage = _validate_journey_lineage(context)
    if not lineage_valid:
        return JourneyPostconditionResult(
            postcondition_id=postcondition_id,
            satisfied=False,
            details={
                **lineage,
                "reason": "journey lineage is incomplete or inconsistent",
                "fail_closed": True,
            },
        )
    evaluator = _IMPLEMENTED_POSTCONDITION_EVALUATORS.get(postcondition_id)
    if evaluator is None:
        return JourneyPostconditionResult(
            postcondition_id=postcondition_id,
            satisfied=False,
            details={
                **lineage,
                "reason": "retained semantic verifier is not implemented",
                "fail_closed": True,
            },
        )
    satisfied, details = evaluator(context)
    return JourneyPostconditionResult(
        postcondition_id=postcondition_id,
        satisfied=satisfied,
        details={
            **lineage,
            **details,
            "fail_closed": False,
        },
    )


def _validate_journey_lineage(
    context: JourneyVerifierContext,
) -> tuple[bool, dict[str, Any]]:
    turns = tuple(context.completed_turns)
    events = tuple(context.completed_events)
    decisions = tuple(context.completed_decisions)
    response_driven = bool(decisions)
    valid = bool(
        turns
        and len(turns) == len(events)
        and (not response_driven or len(turns) == len(decisions))
        and context.latest_turn == turns[-1]
        and context.current_event == events[-1]
    )
    failures: list[str] = []
    previous_event = context.initial_event
    for index, (turn, event) in enumerate(zip(turns, events)):
        if event.schema_version != 3:
            failures.append(f"turn-{index}:schema")
        if event.before_fingerprint != previous_event.after_fingerprint:
            failures.append(f"turn-{index}:fingerprint-chain")
        if (
            turn.before_fingerprint != event.before_fingerprint
            or turn.after_fingerprint != event.after_fingerprint
            or turn.turn_index != event.turn_index
        ):
            failures.append(f"turn-{index}:turn-event-binding")
        receipt = dict(event.turn_receipt_summary or {})
        if receipt.get("input_hash") != hashlib.sha256(
            turn.user_message.encode("utf-8")
        ).hexdigest():
            failures.append(f"turn-{index}:input-hash")
        if turn.transcript_hash != pty_transcript_hash(
            session_id=event.thread_id,
            turn_index=turn.turn_index,
            previous_agent_response=turn.previous_agent_response,
            user_message=turn.user_message,
            agent_response=turn.agent_response,
        ):
            failures.append(f"turn-{index}:transcript-hash")
        if response_driven:
            decision = decisions[index]
            if (
                decision.turn_index != turn.turn_index
                or decision.previous_response_hash
                != content_hash(turn.previous_agent_response)
                or decision.user_message_hash != content_hash(turn.user_message)
                or decision.selected_at_ns < turn.previous_response_received_at_ns
                or decision.submitted_at_ns < decision.selected_at_ns
                or turn.user_message_submitted_at_ns < decision.submitted_at_ns
            ):
                failures.append(f"turn-{index}:decision-binding")
        previous_event = event
    valid = valid and not failures
    return valid, {
        "complete_runtime_lineage_observed": valid,
        "completed_turn_count": len(turns),
        "completed_event_count": len(events),
        "completed_decision_count": len(decisions),
        "lineage_failures": failures,
    }


def _response_driven_selection_observed(
    context: JourneyVerifierContext,
) -> tuple[bool, dict[str, Any]]:
    decision_count = len(context.completed_decisions)
    return decision_count == len(context.completed_turns), {
        "response_bound_decision_count": decision_count,
    }


def _duplicate_job_submission(
    context: JourneyVerifierContext,
) -> tuple[bool, dict[str, Any]]:
    submissions: dict[str, set[tuple[str, str]]] = {}
    for event in context.completed_events:
        receipt = dict(event.execution_receipt_summary or {})
        key = str(
            receipt.get("receipt_idempotency_key")
            or receipt.get("intent_idempotency_key")
            or ""
        )
        receipt_id = str(receipt.get("receipt_id") or "")
        job_id = str(receipt.get("job_id") or "")
        if key and (receipt_id or job_id):
            submissions.setdefault(key, set()).add((receipt_id, job_id))
    duplicates = {
        key: len(identities)
        for key, identities in submissions.items()
        if len(identities) > 1
    }
    return bool(duplicates), {
        "submission_key_count": len(submissions),
        "duplicate_submission_counts": duplicates,
    }


def _typed_confirmation_rejected_by_side_channel(
    context: JourneyVerifierContext,
) -> tuple[bool, dict[str, Any]]:
    rejected_after_acceptance: list[str] = []
    accepted_pending_resolution_count = 0
    for event in context.completed_events:
        accepted_in_turn = any(
            receipt.get("receipt_type") == "pending_resolution"
            and receipt.get("verdict") == "accepted"
            for receipt in event.control_receipts
        )
        if not accepted_in_turn:
            continue
        accepted_pending_resolution_count += 1
        rejected_after_acceptance.extend(
            str(receipt.get("receipt_id") or "")
            for receipt in event.control_receipts
            if receipt.get("receipt_type") == "domain_commit"
            and receipt.get("completion") == "rejected"
        )
    return bool(rejected_after_acceptance), {
        "accepted_pending_resolution_count": accepted_pending_resolution_count,
        "rejected_domain_receipt_ids": rejected_after_acceptance,
    }


_IMPLEMENTED_POSTCONDITION_EVALUATORS = {
    "response_driven_selection_observed": _response_driven_selection_observed,
    "duplicate_job_submission": _duplicate_job_submission,
    "typed_confirmation_rejected_by_side_channel": (
        _typed_confirmation_rejected_by_side_channel
    ),
}


RETAINED_REGRESSION_JOURNEY_VERIFIER_REGISTRY = (
    build_journey_outcome_verifier_registry(tuple(
        JourneyPostconditionVerifierDefinition(
            postcondition_id=postcondition_id,
            verifier_id=f"retained-regression-{postcondition_id}",
            verifier_version=1,
            description=(
                "Reviewed machine evaluator over response-bound runtime "
                "lineage."
                if postcondition_id in _IMPLEMENTED_POSTCONDITION_EVALUATORS
                else
                "Fail-closed retained-regression semantic awaiting a reviewed "
                "machine evaluator."
            ),
            verifier=retained_regression_postcondition,
        )
        for postcondition_id in sorted(KNOWN_POSTCONDITION_IDS)
    ))
)


def build_retained_regression_runner_provider(
    *,
    obligations: Sequence[Mapping[str, Any]],
    revision: Mapping[str, str],
) -> dict[str, Any]:
    """Compile the frozen G3 denominator into execution-provider contracts."""

    active_revision = _validated_revision(revision)
    validate_retained_regression_obligations(
        obligations,
        revision=active_revision,
    )
    verifier_rules = _build_verifier_rules()
    targets = tuple(
        _build_target(dict(obligation), verifier_rules=verifier_rules)
        for obligation in obligations
    )
    unsigned = {
        "schema_version": RETAINED_REGRESSION_PROVIDER_SCHEMA_VERSION,
        "provider_id": RETAINED_REGRESSION_PROVIDER_ID,
        "revision_binding": active_revision,
        "seed": RETAINED_REGRESSION_SEED,
        "obligation_count": len(targets),
        "generation_is_execution": False,
        "execution_status": "not_run",
        "verifier_registry": {
            "registry_id": RETAINED_REGRESSION_VERIFIER_REGISTRY_ID,
            "registry_import": RETAINED_REGRESSION_REGISTRY_IMPORT,
            "journey_registry_id": (
                RETAINED_REGRESSION_JOURNEY_VERIFIER_REGISTRY.registry_id
            ),
            "source_contract_registry_id": POSTCONDITION_REGISTRY_ID,
            "execution_readiness": "blocked",
            "unsupported_postcondition_ids": sorted(
                set(KNOWN_POSTCONDITION_IDS)
                - set(_IMPLEMENTED_POSTCONDITION_EVALUATORS)
            ),
            "rules": [
                verifier_rules[key] for key in sorted(verifier_rules)
            ],
        },
        "targets": list(targets),
    }
    provider = {**unsigned, "provider_hash": content_hash(unsigned)}
    validate_retained_regression_runner_provider(
        provider,
        obligations=obligations,
        revision=active_revision,
    )
    return provider


def retained_regression_journey_definitions(
    provider: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Return only open response-driven definitions; exact replay is excluded."""

    return tuple(
        dict(target["journey_definition"])
        for target in provider.get("targets") or ()
        if target.get("variant") != "exact"
    )


def validate_retained_regression_runner_provider(
    provider: Mapping[str, Any],
    *,
    obligations: Sequence[Mapping[str, Any]],
    revision: Mapping[str, str],
) -> None:
    """Fail closed on stale contracts, prewritten journeys, or evidence claims."""

    active_revision = _validated_revision(revision)
    validate_retained_regression_obligations(
        obligations,
        revision=active_revision,
    )
    expected = {
        str(row["obligation_id"]): dict(row)
        for row in obligations
    }
    allowed_provider_fields = {
        "schema_version",
        "provider_id",
        "revision_binding",
        "seed",
        "obligation_count",
        "generation_is_execution",
        "execution_status",
        "verifier_registry",
        "targets",
        "provider_hash",
    }
    if set(provider) != allowed_provider_fields:
        raise ValueError("retained regression provider shape is incomplete")
    if provider.get("schema_version") != RETAINED_REGRESSION_PROVIDER_SCHEMA_VERSION:
        raise ValueError("unsupported retained regression provider schema")
    if provider.get("provider_id") != RETAINED_REGRESSION_PROVIDER_ID:
        raise ValueError("unknown retained regression provider")
    if provider.get("revision_binding") != active_revision:
        raise ValueError("stale retained regression provider revision")
    if provider.get("seed") != RETAINED_REGRESSION_SEED:
        raise ValueError("retained regression provider seed drifted")
    if (
        provider.get("generation_is_execution") is not False
        or provider.get("execution_status") != "not_run"
    ):
        raise ValueError("retained regression provider generation claimed execution")

    registry = dict(provider.get("verifier_registry") or {})
    rules = registry.get("rules")
    if (
        registry.get("registry_id") != RETAINED_REGRESSION_VERIFIER_REGISTRY_ID
        or registry.get("registry_import") != RETAINED_REGRESSION_REGISTRY_IMPORT
        or registry.get("journey_registry_id")
        != RETAINED_REGRESSION_JOURNEY_VERIFIER_REGISTRY.registry_id
        or registry.get("source_contract_registry_id") != POSTCONDITION_REGISTRY_ID
        or registry.get("execution_readiness") != "blocked"
        or set(registry.get("unsupported_postcondition_ids") or ())
        != (
            set(KNOWN_POSTCONDITION_IDS)
            - set(_IMPLEMENTED_POSTCONDITION_EVALUATORS)
        )
        or not isinstance(rules, Sequence)
        or isinstance(rules, (str, bytes))
    ):
        raise ValueError("retained regression verifier registry is incomplete")
    rule_index = _validate_verifier_rules(rules)

    targets = provider.get("targets")
    if (
        not isinstance(targets, Sequence)
        or isinstance(targets, (str, bytes))
        or len(targets) != RETAINED_REGRESSION_OBLIGATION_COUNT
        or provider.get("obligation_count") != len(targets)
    ):
        raise ValueError("retained regression provider must contain exactly 60 targets")
    seen: set[str] = set()
    for raw_target in targets:
        if not isinstance(raw_target, Mapping):
            raise ValueError("retained regression target must be an object")
        target = dict(raw_target)
        obligation_id = str(target.get("obligation_id") or "")
        obligation = expected.get(obligation_id)
        if obligation is None or obligation_id in seen:
            raise ValueError("unknown or duplicate retained regression target")
        seen.add(obligation_id)
        _validate_target(target, obligation=obligation, rule_index=rule_index)
    if seen != set(expected):
        raise ValueError("retained regression provider is missing obligations")

    unsigned = dict(provider)
    recorded_hash = str(unsigned.pop("provider_hash", "") or "")
    if recorded_hash != content_hash(unsigned):
        raise ValueError("retained regression provider hash is stale")


def build_product_obligation_evidence_artifact(
    *,
    obligation: Mapping[str, Any],
    target: Mapping[str, Any],
    revision: Mapping[str, str],
    execution: Mapping[str, Any],
    artifact_paths: Mapping[str, str | Path],
    verifier_context: JourneyVerifierContext,
) -> dict[str, Any]:
    """Run the authoritative evaluator and adapt its results to admission."""

    active_revision = _validated_revision(revision)
    obligation_id = str(obligation.get("obligation_id") or "")
    if target.get("obligation_id") != obligation_id:
        raise ValueError("evidence target does not match the frozen obligation")
    if target.get("obligation_contract_hash") != obligation.get("contract_hash"):
        raise ValueError("evidence target has a stale obligation contract")

    execution_payload = _validated_execution_identity(
        execution,
        variant=str(obligation.get("variant") or ""),
    )
    artifacts = _artifact_references(artifact_paths)
    artifact_hashes = [item["sha256"] for item in artifacts]
    expected_ids = _expected_verifier_ids(obligation)
    normalized_results = []
    statuses: list[str] = []
    required_ids = set(
        dict(obligation.get("verifier_contract") or {}).get(
            "required_postcondition_ids", ()
        )
    )
    for verifier_id in sorted(expected_ids):
        definition = RETAINED_REGRESSION_JOURNEY_VERIFIER_REGISTRY.definitions[
            verifier_id
        ]
        result = definition.verifier(replace(
            verifier_context,
            evaluating_postcondition_id=verifier_id,
        ))
        if (
            not isinstance(result, JourneyPostconditionResult)
            or result.postcondition_id != verifier_id
        ):
            raise ValueError("retained regression evaluator returned an invalid result")
        satisfied = (
            result.satisfied if verifier_id in required_ids else not result.satisfied
        )
        status = "passed" if satisfied else "failed"
        details = json.dumps(
            {
                **dict(result.details),
                "evaluator": {
                    "registry_id": (
                        RETAINED_REGRESSION_JOURNEY_VERIFIER_REGISTRY.registry_id
                    ),
                    "implementation_hash": definition.implementation_hash,
                    "verifier_version": definition.verifier_version,
                },
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        statuses.append(status)
        normalized_results.append({
            "verifier_id": verifier_id,
            "status": status,
            "details": details,
            "evidence_sha256s": list(artifact_hashes),
        })
    outcome = (
        "failed"
        if "failed" in statuses
        else "externally_blocked"
        if "externally_blocked" in statuses
        else "passed"
    )
    identity = {
        "obligation_id": obligation_id,
        "obligation_contract_hash": obligation["contract_hash"],
        "revision_binding": active_revision,
        "execution_id": execution_payload["execution_id"],
        "artifact_sha256s": sorted(artifact_hashes),
    }
    unsigned = {
        "schema_version": PRODUCT_OBLIGATION_EVIDENCE_SCHEMA_VERSION,
        "evidence_id": content_hash(identity),
        "obligation_id": obligation_id,
        "obligation_contract_hash": obligation["contract_hash"],
        "revision_binding": active_revision,
        "outcome": outcome,
        "execution": execution_payload,
        "artifacts": artifacts,
        "verifier_results": normalized_results,
    }
    return {**unsigned, "evidence_hash": content_hash(unsigned)}


def _build_target(
    obligation: Mapping[str, Any],
    *,
    verifier_rules: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    variant = str(obligation["variant"])
    verifier = dict(obligation["verifier_contract"])
    verifier_ids = [
        *verifier["required_postcondition_ids"],
        *verifier["forbidden_postcondition_ids"],
    ]
    base = {
        "obligation_id": obligation["obligation_id"],
        "obligation_contract_hash": obligation["contract_hash"],
        "case_id": obligation["case_id"],
        "variant": variant,
        "severity": obligation["severity"],
        "seed_contract": dict(obligation["seed_contract"]),
        "verifier_rule_bindings": [
            {
                "postcondition_id": verifier_id,
                "rule_id": verifier_rules[verifier_id]["rule_id"],
                "rule_hash": verifier_rules[verifier_id]["rule_hash"],
            }
            for verifier_id in verifier_ids
        ],
    }
    if variant == "exact":
        stimulus = dict(obligation["stimulus_contract"])
        base["execution_contract"] = {
            "lane": "real_cli",
            "transport": "real_pty",
            "actor": "fixture_replay",
            "evidence_class": "real_cli",
            "qualifies_as_dynamic": False,
            "turns": list(stimulus["turns"]),
            "turns_hash": stimulus["turns_hash"],
        }
        return base

    base["execution_contract"] = {
        "lane": "journey",
        "transport": "real_pty",
        "actor": "codex_as_user",
        "evidence_class": "response_driven_journey",
        "selection_mode": "response_driven",
        "qualifies_as_dynamic": True,
        "read_complete_agent_response_before_each_turn": True,
        "prewritten_future_turns_forbidden": True,
    }
    base["journey_definition"] = {
        "lane": "journey",
        "verifier_registry": RETAINED_REGRESSION_REGISTRY_IMPORT,
        "journey": {
            "journey_id": str(obligation["obligation_id"]),
            "start_scenario": obligation["seed_contract"]["scenario_id"],
            "persona": _persona_for_variant(variant),
            "mission": (
                "Exercise the frozen retained-regression contract "
                f"{obligation['case_id']}/{variant}; choose each user turn only "
                "after observing the complete preceding Agent response. "
                "The source-turn hash is "
                f"{obligation['stimulus_contract']['source_turns_hash']}."
            ),
            "allowed_risk_factors": [f"retained_regression_{variant}"],
            "max_turns": 12,
            "terminal_outcome": {
                "outcome_id": f"{obligation['obligation_id']}-complete",
                "required_postcondition_ids": list(
                    verifier["required_postcondition_ids"]
                ),
            },
            "forbidden_outcomes": [
                {
                    "outcome_id": (
                        f"{obligation['obligation_id']}-forbidden-{index}"
                    ),
                    "required_postcondition_ids": [postcondition_id],
                }
                for index, postcondition_id in enumerate(
                    verifier["forbidden_postcondition_ids"],
                    start=1,
                )
            ],
        },
    }
    return base


def _validate_target(
    target: Mapping[str, Any],
    *,
    obligation: Mapping[str, Any],
    rule_index: Mapping[str, Mapping[str, Any]],
) -> None:
    variant = str(obligation["variant"])
    expected_fields = {
        "obligation_id",
        "obligation_contract_hash",
        "case_id",
        "variant",
        "severity",
        "seed_contract",
        "verifier_rule_bindings",
        "execution_contract",
    }
    if variant != "exact":
        expected_fields.add("journey_definition")
    if set(target) != expected_fields:
        raise ValueError("retained regression target shape is incomplete")
    for field in (
        "obligation_id",
        "obligation_contract_hash",
        "case_id",
        "variant",
        "severity",
        "seed_contract",
    ):
        expected = obligation.get(field)
        if field == "obligation_contract_hash":
            expected = obligation.get("contract_hash")
        if target.get(field) != expected:
            raise ValueError(f"retained regression target drifted: {field}")

    verifier = dict(obligation["verifier_contract"])
    expected_ids = {
        *verifier["required_postcondition_ids"],
        *verifier["forbidden_postcondition_ids"],
    }
    bindings = target.get("verifier_rule_bindings")
    if not isinstance(bindings, Sequence) or isinstance(bindings, (str, bytes)):
        raise ValueError("retained regression verifier bindings are missing")
    observed_ids: set[str] = set()
    for binding in bindings:
        if not isinstance(binding, Mapping) or set(binding) != {
            "postcondition_id",
            "rule_id",
            "rule_hash",
        }:
            raise ValueError("retained regression verifier binding is invalid")
        postcondition_id = str(binding.get("postcondition_id") or "")
        rule = rule_index.get(postcondition_id)
        if (
            rule is None
            or postcondition_id in observed_ids
            or binding.get("rule_id") != rule["rule_id"]
            or binding.get("rule_hash") != rule["rule_hash"]
        ):
            raise ValueError("retained regression verifier binding is stale")
        observed_ids.add(postcondition_id)
    if observed_ids != expected_ids:
        raise ValueError("retained regression verifier bindings are incomplete")

    execution = dict(target.get("execution_contract") or {})
    if execution.get("transport") != "real_pty":
        raise ValueError("retained regression execution must use a real PTY")
    if variant == "exact":
        if (
            execution.get("lane") != "real_cli"
            or execution.get("evidence_class") != "real_cli"
            or execution.get("qualifies_as_dynamic") is not False
            or execution.get("turns")
            != obligation["stimulus_contract"]["turns"]
            or execution.get("turns_hash")
            != obligation["stimulus_contract"]["turns_hash"]
            or "journey_definition" in target
        ):
            raise ValueError("exact retained regression cannot claim dynamic evidence")
        return

    if (
        execution.get("lane") != "journey"
        or execution.get("actor") != "codex_as_user"
        or execution.get("selection_mode") != "response_driven"
        or execution.get("qualifies_as_dynamic") is not True
        or execution.get("read_complete_agent_response_before_each_turn") is not True
        or execution.get("prewritten_future_turns_forbidden") is not True
    ):
        raise ValueError("response-driven retained regression contract is incomplete")
    journey_definition = dict(target.get("journey_definition") or {})
    journey = dict(journey_definition.get("journey") or {})
    serialized = repr(journey_definition)
    if (
        journey_definition.get("lane") != "journey"
        or journey_definition.get("verifier_registry")
        != RETAINED_REGRESSION_REGISTRY_IMPORT
    ):
        raise ValueError("retained regression Journey lane is invalid")
    if any(
        field in journey_definition or field in journey
        for field in ("turns", "messages", "future_user_turns", "dialogue")
    ):
        raise ValueError("response-driven retained regression contains prewritten turns")
    if str(obligation["stimulus_contract"]["source_turns_hash"]) not in serialized:
        raise ValueError("retained regression Journey definition is incomplete")
    build_journey_schedule(
        revision=dict(
            obligation["revision_binding"]["revision"]
        ),
        seed=RETAINED_REGRESSION_SEED,
        journey=journey,
    )


def _build_verifier_rules() -> dict[str, dict[str, Any]]:
    class_by_id: dict[str, str] = {}
    for class_name, postcondition_ids in _RULE_CLASSES.items():
        for postcondition_id in postcondition_ids:
            if postcondition_id in class_by_id:
                raise ValueError(
                    f"duplicate retained regression verifier rule: {postcondition_id}"
                )
            class_by_id[postcondition_id] = class_name
    if set(class_by_id) != set(KNOWN_POSTCONDITION_IDS):
        missing = sorted(set(KNOWN_POSTCONDITION_IDS) - set(class_by_id))
        extra = sorted(set(class_by_id) - set(KNOWN_POSTCONDITION_IDS))
        raise ValueError(
            f"retained regression verifier rule coverage drifted: missing={missing}, extra={extra}"
        )
    rules: dict[str, dict[str, Any]] = {}
    for postcondition_id, class_name in class_by_id.items():
        unsigned = {
            "postcondition_id": postcondition_id,
            "rule_id": f"retained-regression/{postcondition_id}/v1",
            "rule_version": 1,
            "rule_class": class_name,
            "evaluation": "all",
            "required_artifact_roles": list(REQUIRED_ARTIFACT_ROLES),
            "assertions": [
                dict(assertion)
                for assertion in _DECLARATIVE_ASSERTIONS[class_name]
            ],
            "natural_language_expected_is_verifier": False,
            "keyword_matching_forbidden": True,
        }
        rules[postcondition_id] = {
            **unsigned,
            "rule_hash": content_hash(unsigned),
        }
    return rules


def _validate_verifier_rules(
    rules: Sequence[Any],
) -> dict[str, dict[str, Any]]:
    allowed_operators = {
        "visible_nonempty",
        "at_or_after",
        "equals",
        "lineage_preserved",
        "excludes_roots",
        "consumed_by_accepted_action",
        "registered_and_complete",
        "consistent_with_action_effects",
        "nonempty",
        "contiguous",
        "consistent_with_checkpoint_diff",
        "unique_and_complete",
        "equals_submitted_turn_count",
        "absent",
        "contiguous_without_regression",
    }
    index: dict[str, dict[str, Any]] = {}
    for raw in rules:
        if not isinstance(raw, Mapping):
            raise ValueError("retained regression verifier rule must be an object")
        rule = dict(raw)
        postcondition_id = str(rule.get("postcondition_id") or "")
        if postcondition_id not in KNOWN_POSTCONDITION_IDS or postcondition_id in index:
            raise ValueError("unknown or duplicate retained regression verifier rule")
        assertions = rule.get("assertions")
        if (
            rule.get("rule_version") != 1
            or rule.get("evaluation") != "all"
            or rule.get("required_artifact_roles") != list(REQUIRED_ARTIFACT_ROLES)
            or rule.get("natural_language_expected_is_verifier") is not False
            or rule.get("keyword_matching_forbidden") is not True
            or not isinstance(assertions, Sequence)
            or isinstance(assertions, (str, bytes))
            or not assertions
        ):
            raise ValueError("retained regression verifier rule contract is incomplete")
        for assertion in assertions:
            if (
                not isinstance(assertion, Mapping)
                or not str(assertion.get("fact") or "")
                or assertion.get("operator") not in allowed_operators
                or not str(assertion.get("scope") or "")
            ):
                raise ValueError("retained regression declarative assertion is invalid")
        unsigned = dict(rule)
        recorded_hash = str(unsigned.pop("rule_hash", "") or "")
        if recorded_hash != content_hash(unsigned):
            raise ValueError("retained regression verifier rule hash is stale")
        index[postcondition_id] = rule
    if set(index) != set(KNOWN_POSTCONDITION_IDS):
        raise ValueError("retained regression verifier registry is incomplete")
    return index


def _artifact_references(
    artifact_paths: Mapping[str, str | Path],
) -> list[dict[str, str]]:
    if set(artifact_paths) != set(REQUIRED_ARTIFACT_ROLES):
        raise ValueError("retained regression evidence requires exactly three artifacts")
    references = []
    for role in REQUIRED_ARTIFACT_ROLES:
        path = Path(artifact_paths[role]).expanduser().resolve()
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError(f"retained regression evidence artifact is missing: {path}")
        references.append({
            "role": role,
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    return references


def _validated_execution_identity(
    execution: Mapping[str, Any],
    *,
    variant: str,
) -> dict[str, str]:
    expected_fields = {
        "execution_id",
        "runner",
        "transport",
        "provider",
        "model",
        "started_at",
        "finished_at",
    }
    payload = {key: str(value or "").strip() for key, value in execution.items()}
    if set(payload) != expected_fields or any(not payload[key] for key in expected_fields):
        raise ValueError("retained regression execution identity is incomplete")
    if payload["transport"] != "real_pty":
        raise ValueError("retained regression evidence must come from a real PTY")
    expected_runner = (
        "retained-regression-real-cli-v1"
        if variant == "exact"
        else "retained-regression-response-driven-journey-v1"
    )
    if payload["runner"] != expected_runner:
        raise ValueError("retained regression evidence class does not match its variant")
    if payload["started_at"] == payload["finished_at"]:
        raise ValueError("retained regression execution interval is empty")
    return payload


def _expected_verifier_ids(obligation: Mapping[str, Any]) -> set[str]:
    verifier = dict(obligation.get("verifier_contract") or {})
    return {
        *verifier.get("required_postcondition_ids", ()),
        *verifier.get("forbidden_postcondition_ids", ()),
    }


def _persona_for_variant(variant: str) -> str:
    return {
        "isomorphic": "operator expressing the same goal in a different surface form",
        "negative": "operator exercising an adjacent non-triggering intent",
        "neighboring": "operator exercising a neighboring authoritative transition",
    }[variant]


def _validated_revision(revision: Mapping[str, str]) -> dict[str, str]:
    normalized = {
        "commit": str(revision.get("commit") or "").strip(),
        "worktree_hash": str(revision.get("worktree_hash") or "").strip(),
    }
    if not normalized["commit"] or not normalized["worktree_hash"]:
        raise ValueError("retained regression runner requires a revision binding")
    return normalized

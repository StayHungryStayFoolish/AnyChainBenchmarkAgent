"""Executable-provider contracts for Phase 8 G3 retained regressions.

This module compiles the frozen 60-row obligation catalog into execution
contracts and an admission-compatible evidence adapter.  It deliberately does
not execute the CLI, choose future response-driven turns, or claim outcomes.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import inspect
import json
import os
import tempfile
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from agent.harness.runtime_identity import repository_revision
from agent.llm.config import load_llm_config
from agent.utils.redaction import redact
from tests.agent_live.batch_orchestrator import (
    TimeoutPolicy,
    freeze_batch_manifest,
)
from tests.agent_live.coverage_evidence import (
    PtyCliTurnRecord,
    RuntimeTurnEvent,
    content_hash,
    pty_transcript_hash,
    validate_runtime_turn_event,
)
from tests.agent_live.chaos_scheduler import (
    build_journey_schedule,
    journey_schedule_payload,
)
from tests.agent_live.codex_simulator_bridge import validate_simulator_attestation
from tests.agent_live.completed_journey_batch import (
    CompletedJourneySource,
    G3_ARTIFACT_TYPE,
    convert_completed_journey_batch,
)
from tests.agent_live.dynamic_dual_ai_chaos import (
    ChaosRunConfig,
    CompletionExpectation,
    ContainerPtyBridgeTransport,
    JsonlRuntimeEventStream,
    JsonlTerminalOutcomeStream,
    JourneyDecisionProvenance,
    JourneyPostconditionResult,
    JourneyPostconditionVerifierDefinition,
    JourneyVerifierContext,
    TerminalCompletionError,
    WorkflowCompletion,
    _runtime_event_from_mapping,
    _startup_resume_submission,
    build_journey_outcome_verifier_registry,
    validate_journey_evidence_artifact,
    validate_startup_session_event,
    validate_startup_terminal_protocol,
    transport_for_config,
    wait_for_turn_completion,
    write_terminal_completion_diagnostic,
)
from tests.agent_live.product_obligation_evidence import (
    PRODUCT_OBLIGATION_EVIDENCE_SCHEMA_VERSION,
    admit_product_obligation_evidence,
)
from tests.agent_live.retained_regression_obligations import (
    KNOWN_POSTCONDITION_IDS,
    POSTCONDITION_REGISTRY_ID,
    RETAINED_REGRESSION_OBLIGATION_COUNT,
    build_retained_regression_obligations,
    validate_retained_regression_obligations,
)
from tests.agent_live.retained_regression_predicates import (
    POSTCONDITION_EVALUATORS,
)
from tests.agent_live.retained_regression_attestations import (
    validate_variant_attestation,
    validate_verifier_input_contract,
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
EXACT_RUNTIME_ARTIFACT_ROLES = (
    *REQUIRED_ARTIFACT_ROLES,
    "process_guard_receipt",
)
RETAINED_REGRESSION_DEFINITION_MANIFEST_SCHEMA_VERSION = 1
RETAINED_REGRESSION_TARGET_SET_SCHEMA_VERSION = 1
RETAINED_REGRESSION_EXACT_COUNT = 15
RETAINED_REGRESSION_OPEN_COUNT = 45
DEFAULT_PROVIDER = "deepseek"


def _is_sha256(value: Any) -> bool:
    text = str(value or "")
    return (
        len(text) == 64
        and text == text.lower()
        and all(character in "0123456789abcdef" for character in text)
    )


_AUDITABLE_ACTOR_ATTESTATION_POLICY = {
    "required": True,
    "identity_strength": "auditable_declaration_only",
    "cryptographic_identity_claimed": False,
    "selection_mode": "response_driven",
    "prewritten_future_turns": False,
}

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
        "mode_consultation_preserves_chain_pending",
        "unresolved_units_preserved",
        "declined_mode_change_resumes_chain_pending",
    ),
    "pending_advanced": (
        "disk_size_resolved",
        "typed_detected_value_confirmed",
        "owned_group_backtrack_completed",
        "chain_confirmation_required",
        "custom_method_collection_exited",
    ),
    "action_provenance": (
        "visible_option_action_executed",
        "real_node_selection_executed_at_source",
        "mode_change_request_routed_from_chain_pending",
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
        "effective_workload_commit_replaces_defaults",
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
        "removed_default_method_committed",
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
        "mode_request_consumed_as_chain_identity",
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
        try:
            validate_runtime_turn_event(event)
        except ValueError as exc:
            failures.append(f"turn-{index}:event:{exc}")
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


_IMPLEMENTED_POSTCONDITION_EVALUATORS = dict(POSTCONDITION_EVALUATORS)


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
    unsupported = sorted(
        set(KNOWN_POSTCONDITION_IDS)
        - set(_IMPLEMENTED_POSTCONDITION_EVALUATORS)
    )
    execution_blockers = [
        f"missing_postcondition_evaluator:{postcondition_id}"
        for postcondition_id in unsupported
    ]
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
            "execution_readiness": "ready" if not execution_blockers else "blocked",
            "execution_blockers": execution_blockers,
            "unsupported_postcondition_ids": unsupported,
            "simulator_attestation_policy": dict(
                _AUDITABLE_ACTOR_ATTESTATION_POLICY
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


def write_retained_regression_provider(
    *,
    repo_root: str | Path,
    output_path: str | Path,
) -> Path:
    """Build and immutably write the authoritative G3 provider."""

    root = Path(repo_root).resolve()
    revision = repository_revision(root)
    obligations = build_retained_regression_obligations(
        repo_root=root,
        revision=revision,
    )
    provider = build_retained_regression_runner_provider(
        obligations=obligations,
        revision=revision,
    )
    return _write_json_once(Path(output_path).resolve(), provider)


def load_retained_regression_provider(
    *,
    repo_root: str | Path,
    provider_path: str | Path,
    require_current_revision: bool = True,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...], dict[str, str]]:
    """Load a provider and recompile its complete authoritative catalog."""

    root = Path(repo_root).resolve()
    provider = _load_mapping(
        Path(provider_path).resolve(),
        "retained regression provider",
    )
    revision = _validated_revision(provider.get("revision_binding") or {})
    if require_current_revision and repository_revision(root) != revision:
        raise ValueError("retained regression provider revision is not current")
    obligations = build_retained_regression_obligations(
        repo_root=root,
        revision=revision,
    )
    validate_retained_regression_runner_provider(
        provider,
        obligations=obligations,
        revision=revision,
    )
    return provider, obligations, revision


def build_retained_regression_definition_manifest(
    provider: Mapping[str, Any],
    *,
    obligation_id: str | None = None,
) -> dict[str, Any]:
    """Freeze all or one of the 45 open G3 Journey definitions."""

    if (
        provider.get("execution_status") != "not_run"
        or dict(provider.get("verifier_registry") or {}).get(
            "execution_readiness"
        )
        != "ready"
    ):
        raise ValueError("retained regression provider is not execution ready")
    selected = [
        dict(target)
        for target in provider.get("targets") or ()
        if target.get("variant") != "exact"
        and (
            obligation_id is None
            or target.get("obligation_id") == obligation_id
        )
    ]
    if obligation_id is not None and len(selected) != 1:
        raise ValueError(f"unknown open G3 obligation: {obligation_id}")
    if obligation_id is None and len(selected) != RETAINED_REGRESSION_OPEN_COUNT:
        raise ValueError("G3 provider does not contain all 45 open obligations")
    definitions = []
    for target in selected:
        definition = dict(target["journey_definition"])
        _reject_future_turns(definition)
        definitions.append({
            "obligation_id": target["obligation_id"],
            "obligation_contract_hash": target["obligation_contract_hash"],
            "variant": target["variant"],
            "definition": definition,
            "definition_hash": content_hash(definition),
        })
    unsigned = {
        "schema_version": (
            RETAINED_REGRESSION_DEFINITION_MANIFEST_SCHEMA_VERSION
        ),
        "artifact_type": "retained_regression_journey_definition_manifest",
        "provider_hash": provider["provider_hash"],
        "revision_binding": dict(provider["revision_binding"]),
        "selection": "single" if obligation_id else "all",
        "definition_count": len(definitions),
        "generation_is_execution": False,
        "prewritten_future_turns": False,
        "actor_attestation_policy": dict(
            _AUDITABLE_ACTOR_ATTESTATION_POLICY
        ),
        "definitions": definitions,
    }
    manifest = {**unsigned, "manifest_hash": content_hash(unsigned)}
    _reject_future_turns(manifest)
    return manifest


def write_retained_regression_definition_manifest(
    manifest: Mapping[str, Any],
    output_path: str | Path,
) -> Path:
    """Write a G3 definition manifest without replacing an existing artifact."""

    return _write_json_once(Path(output_path).resolve(), dict(manifest))


def write_retained_regression_target_set(
    *,
    provider: Mapping[str, Any],
    definition_manifest: Mapping[str, Any],
    output_dir: str | Path,
) -> Path:
    """Materialize immutable numbered targets for the shared batch runtime."""

    definitions = _validated_definition_manifest(
        provider=provider,
        manifest=definition_manifest,
        require_all=True,
    )
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"G3 target directory is immutable: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}-staging-",
        dir=destination.parent,
    ) as temporary:
        staging = Path(temporary)
        target_rows: list[dict[str, Any]] = []
        for index, row in enumerate(definitions, start=1):
            payload = _batch_target_payload(row["definition"])
            name = f"{index:02d}.json"
            encoded = _json_bytes(payload)
            (staging / name).write_bytes(encoded)
            frozen = dict(payload["frozen_execution"])
            target_rows.append({
                "index": index,
                "obligation_id": row["obligation_id"],
                "obligation_contract_hash": row[
                    "obligation_contract_hash"
                ],
                "variant": row["variant"],
                "seed": frozen["seed"],
                "schedule_id": frozen["schedule_id"],
                "subject_group": frozen["subject_group"],
                "path": name,
                "sha256": hashlib.sha256(encoded).hexdigest(),
            })
        obligation_set = [
            {
                "obligation_id": row["obligation_id"],
                "schedule_id": row["schedule_id"],
                "seed": row["seed"],
                "subject_group": row["subject_group"],
            }
            for row in target_rows
        ]
        unsigned = {
            "schema_version": RETAINED_REGRESSION_TARGET_SET_SCHEMA_VERSION,
            "artifact_type": "retained_regression_target_set",
            "provider_hash": provider["provider_hash"],
            "definition_manifest_hash": definition_manifest["manifest_hash"],
            "revision_binding": dict(provider["revision_binding"]),
            "expected_obligation_set_hash": content_hash(obligation_set),
            "target_count": len(target_rows),
            "targets": target_rows,
        }
        target_manifest = {
            **unsigned,
            "manifest_hash": content_hash(unsigned),
        }
        (staging / "manifest.json").write_bytes(_json_bytes(target_manifest))
        os.replace(staging, destination)
    return destination / "manifest.json"


def freeze_retained_regression_open_batch(
    *,
    repo_root: str | Path,
    provider: Mapping[str, Any],
    targets_dir: str | Path,
    manifest_path: str | Path,
    runtime_base: str | Path,
    max_concurrency: int | None = None,
    worker_runtime: str = "linux",
    timeout_policy: TimeoutPolicy = TimeoutPolicy(),
) -> Any:
    """Validate all 45 open targets and freeze one bounded shared batch."""

    target_root = Path(targets_dir).resolve()
    target_manifest = _load_mapping(
        target_root / "manifest.json",
        "retained regression target manifest",
    )
    target_rows = _validated_target_set(
        provider=provider,
        target_manifest=target_manifest,
        targets_dir=target_root,
    )
    return freeze_batch_manifest(
        repo_root=repo_root,
        targets_dir=target_root,
        manifest_path=manifest_path,
        runtime_base=runtime_base,
        shard_count=len(target_rows),
        max_concurrency=max_concurrency,
        seed_base=RETAINED_REGRESSION_SEED,
        expected_revision=dict(provider["revision_binding"]),
        timeout_policy=timeout_policy,
        worker_runtime=worker_runtime,
        formal_profile=False,
        expected_obligation_set_hash=str(
            target_manifest["expected_obligation_set_hash"]
        ),
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
    unsupported = sorted(
        set(KNOWN_POSTCONDITION_IDS)
        - set(_IMPLEMENTED_POSTCONDITION_EVALUATORS)
    )
    execution_blockers = [
        f"missing_postcondition_evaluator:{postcondition_id}"
        for postcondition_id in unsupported
    ]
    if (
        set(registry)
        != {
            "registry_id",
            "registry_import",
            "journey_registry_id",
            "source_contract_registry_id",
            "execution_readiness",
            "execution_blockers",
            "unsupported_postcondition_ids",
            "simulator_attestation_policy",
            "rules",
        }
        or registry.get("registry_id")
        != RETAINED_REGRESSION_VERIFIER_REGISTRY_ID
        or registry.get("registry_import") != RETAINED_REGRESSION_REGISTRY_IMPORT
        or registry.get("journey_registry_id")
        != RETAINED_REGRESSION_JOURNEY_VERIFIER_REGISTRY.registry_id
        or registry.get("source_contract_registry_id") != POSTCONDITION_REGISTRY_ID
        or registry.get("execution_readiness")
        != ("ready" if not execution_blockers else "blocked")
        or registry.get("execution_blockers") != execution_blockers
        or set(registry.get("unsupported_postcondition_ids") or ())
        != set(unsupported)
        or registry.get("simulator_attestation_policy")
        != _AUDITABLE_ACTOR_ATTESTATION_POLICY
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
    artifact_display_paths: Mapping[str, str | Path] | None = None,
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
    artifacts, bound_context = _reconstruct_verifier_context(
        artifact_paths,
        obligation=obligation,
        target=target,
        revision=active_revision,
        execution=execution_payload,
        artifact_display_paths=artifact_display_paths,
    )
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
            bound_context,
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
            "verifier_version": 1,
            "implementation_hash": (
                _current_evaluator_implementation_hash(verifier_id)
            ),
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
    round_id = "g3-retained-regression"
    session_id = execution_payload["execution_id"]
    request_ids = [
        f"retained-regression:{obligation_id}:{execution_payload['execution_id']}"
    ]
    identity = {
        "obligation_id": obligation_id,
        "obligation_contract_hash": obligation["contract_hash"],
        "revision_binding": active_revision,
        "round_id": round_id,
        "session_id": session_id,
        "request_ids": request_ids,
        "execution_id": execution_payload["execution_id"],
        "artifact_sha256s": sorted(artifact_hashes),
    }
    unsigned = {
        "schema_version": PRODUCT_OBLIGATION_EVIDENCE_SCHEMA_VERSION,
        "evidence_id": content_hash(identity),
        "obligation_id": obligation_id,
        "obligation_contract_hash": obligation["contract_hash"],
        "revision_binding": active_revision,
        "round_id": round_id,
        "session_id": session_id,
        "request_ids": request_ids,
        "outcome": outcome,
        "execution": execution_payload,
        "artifacts": artifacts,
        "verifier_results": normalized_results,
    }
    return {**unsigned, "evidence_hash": content_hash(unsigned)}


def _validate_exact_terminal_revision(
    terminal_outcome: Any,
    *,
    active_revision: Mapping[str, str],
    stage: str,
) -> None:
    if dict(terminal_outcome.origin_revision) != dict(active_revision):
        raise RuntimeError(
            f"exact retained-regression {stage} terminal revision is stale"
        )


def execute_exact_retained_regression(
    *,
    repo_root: str | Path,
    obligation: Mapping[str, Any],
    target: Mapping[str, Any],
    revision: Mapping[str, str],
    output_root: str | Path,
    timeout_seconds: float = 180.0,
    worker_runtime: str = "docker",
) -> Path:
    """Replay one immutable exact fixture through the real Linux CLI/PTY."""

    root = Path(repo_root).resolve()
    active_revision = _validated_revision(revision)
    if (
        str(obligation.get("variant") or "") != "exact"
        or str(target.get("variant") or "") != "exact"
    ):
        raise ValueError("exact retained-regression executor requires an exact target")
    if target.get("obligation_id") != obligation.get("obligation_id"):
        raise ValueError("exact retained-regression target identity is stale")
    turns_to_submit = tuple(
        str(item)
        for item in dict(obligation.get("stimulus_contract") or {}).get(
            "turns", ()
        )
    )
    if not turns_to_submit:
        raise ValueError("exact retained-regression fixture has no turns")

    obligation_id = str(obligation["obligation_id"])
    execution_id = (
        f"g3-exact-{content_hash({'obligation_id': obligation_id, 'revision': active_revision})[:20]}"
    )
    runtime_root = Path(output_root).resolve() / execution_id
    if runtime_root.exists():
        raise FileExistsError(
            f"exact retained-regression runtime is immutable: {runtime_root}"
        )
    runtime_root.mkdir(parents=True)
    session_id = execution_id
    checkpoint_path = runtime_root / "checkpoints.sqlite"
    event_path = runtime_root / "turn-events.jsonl"
    terminal_outcome_path = runtime_root / "terminal-outcomes.jsonl"
    try:
        container_runtime = (
            Path("/workspace") / runtime_root.relative_to(root)
        )
    except ValueError as exc:
        raise ValueError(
            "exact retained-regression output must be inside the repository"
        ) from exc
    config_factory = {
        "docker": ChaosRunConfig.docker,
        "linux": ChaosRunConfig.linux,
    }.get(worker_runtime)
    if config_factory is None:
        raise ValueError(f"unsupported exact worker runtime: {worker_runtime}")
    config = config_factory(
        root,
        session_id=session_id,
        execution_id=execution_id,
        session_purpose="retained-regression-exact",
        runtime_root=runtime_root,
        runtime_root_in_process=container_runtime,
        response_timeout_seconds=timeout_seconds,
    )
    from tests.agent_live.runtime_checkpoint import (
        reviewed_scenario,
        seed_runtime_checkpoint,
    )

    scenario = reviewed_scenario(
        str(dict(obligation["seed_contract"])["scenario_id"])
    )
    seed_runtime_checkpoint(
        scenario.seed_state,
        checkpoint_path=checkpoint_path,
        session_id=session_id,
        session_purpose="retained-regression-exact",
        scenario_id=scenario.scenario_id,
        scenario_state_fingerprint=scenario.state_fingerprint,
    )
    env = os.environ.copy()
    env.update({
        "ANYCHAIN_AGENT_CHECKPOINT_PATH": str(
            container_runtime / "checkpoints.sqlite"
        ),
        "ANYCHAIN_AGENT_SESSION_ID": session_id,
        "ANYCHAIN_AGENT_SESSION_PURPOSE": "retained-regression-exact",
        "ANYCHAIN_AGENT_JOBS_DIR": str(container_runtime / "jobs"),
        "ANYCHAIN_AGENT_TURN_EVENT_FILE": str(
            container_runtime / "turn-events.jsonl"
        ),
        "ANYCHAIN_AGENT_TERMINAL_OUTCOME_FILE": str(
            container_runtime / "terminal-outcomes.jsonl"
        ),
    })
    env = config.isolated_environment(env)
    transport = transport_for_config(config)
    event_stream = JsonlRuntimeEventStream(event_path)
    terminal_outcome_stream = JsonlTerminalOutcomeStream(
        terminal_outcome_path,
        poll_interval_seconds=config.poll_interval_seconds,
    )
    observed_turns: list[PtyCliTurnRecord] = []
    observed_events: list[RuntimeTurnEvent] = []
    transcript_lines: list[str] = []
    started_at = str(time.time_ns())
    initial_event: RuntimeTurnEvent | None = None
    active_user_message = ""
    process_guard_proof: Mapping[str, Any] | None = None
    terminal_outcome_stream.mark_process_start()
    event_stream.mark_process_start()
    transport.start(env=env)
    try:
        previous_response = transport.read_complete_agent_response(
            timeout_seconds=timeout_seconds
        )
        previous_received_at_ns = time.time_ns()
        startup_session = validate_startup_session_event(
            terminal_outcome_stream,
            expected_revision=active_revision,
            expected_provider=config.provider,
            expected_model=config.model,
            expected_session_id=session_id,
            expected_session_purpose="retained-regression-exact",
            startup_response=previous_response,
        )
        provider = startup_session.provider
        model = startup_session.model
        runtime_history, startup_events = event_stream.capture_startup_snapshot()
        combined_events = (*runtime_history, *startup_events)
        if not combined_events:
            raise RuntimeError("retained regression requires a baseline runtime event")
        for historical_event in runtime_history:
            validate_runtime_turn_event(historical_event)
        initial_event = combined_events[-1]
        startup_outcomes, startup_detours = (
            terminal_outcome_stream.capture_startup_snapshot()
        )
        validate_startup_terminal_protocol(
            startup_events,
            startup_outcomes,
            expected_revision=active_revision,
            session_event=startup_session,
            detours=startup_detours,
            baseline_event=initial_event,
        )
        validate_runtime_turn_event(initial_event)
        if dict(initial_event.revision) != active_revision:
            raise RuntimeError(
                "exact retained-regression startup revision is stale"
            )
        transcript_lines.append(previous_response)
        resume_submission = _startup_resume_submission(
            initial_event.pending_contract,
            desired="continue",
        )
        if resume_submission:
            active_user_message = resume_submission
            transport.submit_bracketed_paste(resume_submission)
            resume_completion = wait_for_turn_completion(
                transport,
                event_stream,
                terminal_outcome_stream,
                timeout_seconds=timeout_seconds,
                expectation=CompletionExpectation(
                    submitted_input=resume_submission,
                    session_id=session_id,
                    session_purpose=config.session_purpose,
                    product_authority_id=startup_session.product_authority_id,
                    process_instance_id=startup_session.process_instance_id,
                    product_revision=int(initial_event.product_revision),
                    product_checkpoint_thread_id=(
                        initial_event.product_checkpoint_thread_id
                    ),
                    product_checkpoint_id=initial_event.product_checkpoint_id,
                    product_fingerprint=initial_event.after_fingerprint,
                ),
            )
            if not isinstance(resume_completion, WorkflowCompletion):
                raise RuntimeError(
                    "retained resume did not produce a workflow completion"
                )
            previous_response = resume_completion.response
            initial_event = resume_completion.event
            validate_runtime_turn_event(initial_event)
            _validate_exact_terminal_revision(
                resume_completion.terminal_outcome,
                active_revision=active_revision,
                stage="resume",
            )
            previous_received_at_ns = time.time_ns()
            transcript_lines.extend((
                f"User> {resume_submission}",
                previous_response,
            ))
            active_user_message = ""

        baseline = initial_event
        for user_message in turns_to_submit:
            submitted_at_ns = time.time_ns()
            active_user_message = user_message
            transport.submit_bracketed_paste(user_message)
            completion = wait_for_turn_completion(
                transport,
                event_stream,
                terminal_outcome_stream,
                timeout_seconds=timeout_seconds,
                expectation=CompletionExpectation(
                    submitted_input=user_message,
                    session_id=session_id,
                    session_purpose=config.session_purpose,
                    product_authority_id=startup_session.product_authority_id,
                    process_instance_id=startup_session.process_instance_id,
                    product_revision=int(baseline.product_revision),
                    product_checkpoint_thread_id=(
                        baseline.product_checkpoint_thread_id
                    ),
                    product_checkpoint_id=baseline.product_checkpoint_id,
                    product_fingerprint=baseline.after_fingerprint,
                ),
            )
            if not isinstance(completion, WorkflowCompletion):
                raise RuntimeError(
                    "retained turn did not produce a workflow completion"
                )
            committed = completion.event
            response = completion.response
            response_received_at_ns = time.time_ns()
            validate_runtime_turn_event(committed)
            _validate_exact_terminal_revision(
                completion.terminal_outcome,
                active_revision=active_revision,
                stage="turn",
            )
            if (
                dict(committed.revision) != active_revision
                or committed.before_fingerprint
                != baseline.after_fingerprint
                or committed.turn_index != baseline.turn_index + 1
            ):
                raise RuntimeError(
                    "exact retained-regression runtime lineage is stale"
                )
            turn = PtyCliTurnRecord(
                session_id=session_id,
                turn_index=committed.turn_index,
                previous_agent_response=previous_response,
                user_message=user_message,
                agent_response=response,
                provider=provider,
                model=model,
                before_fingerprint=committed.before_fingerprint,
                after_fingerprint=committed.after_fingerprint,
                transcript_hash=pty_transcript_hash(
                    session_id=session_id,
                    turn_index=committed.turn_index,
                    previous_agent_response=previous_response,
                    user_message=user_message,
                    agent_response=response,
                ),
                previous_response_received_at_ns=previous_received_at_ns,
                user_message_submitted_at_ns=submitted_at_ns,
                agent_response_received_at_ns=response_received_at_ns,
            )
            observed_turns.append(turn)
            observed_events.append(committed)
            transcript_lines.extend((f"User> {user_message}", response))
            previous_response = response
            previous_received_at_ns = response_received_at_ns
            baseline = committed
            active_user_message = ""
    except TerminalCompletionError as exc:
        transcript_lines.append(f"User> {active_user_message}")
        failed_output = exc.response or (
            str(exc.response_error)
            if exc.response_error is not None
            else ""
        )
        if failed_output:
            transcript_lines.append(failed_output)
        write_terminal_completion_diagnostic(
            exc,
            runtime_root / "diagnostics",
            session_id=session_id,
            revision=active_revision,
            user_message=active_user_message,
            last_complete_event=observed_events[-1] if observed_events else initial_event,
        )
        raise
    finally:
        try:
            transport.close()
            if not isinstance(transport, ContainerPtyBridgeTransport):
                raise RuntimeError(
                    "exact retained-regression requires the container PTY bridge"
                )
            process_guard_proof = transport.validated_execution_proof()
        finally:
            (runtime_root / "transcript.txt").write_text(
                str(redact("\n".join(transcript_lines).rstrip() + "\n")),
                encoding="utf-8",
            )

    if initial_event is None:
        raise RuntimeError("exact retained-regression did not observe a startup event")
    if process_guard_proof is None:
        raise RuntimeError(
            "exact retained-regression has no validated process cleanup proof"
        )

    artifact_paths = _write_exact_retained_artifacts(
        runtime_root=runtime_root,
        obligation=obligation,
        target=target,
        revision=active_revision,
        execution_id=execution_id,
        initial_event=initial_event,
        events=observed_events,
        turns=observed_turns,
    )
    artifact_paths["process_guard_receipt"] = Path(
        str(process_guard_proof["path"])
    ).resolve()
    execution = {
        "execution_id": execution_id,
        "runner": "retained-regression-real-cli-v1",
        "transport": "real_pty",
        "provider": provider,
        "model": model,
        "started_at": started_at,
        "finished_at": str(time.time_ns()),
    }
    evidence = build_product_obligation_evidence_artifact(
        obligation=obligation,
        target=target,
        revision=active_revision,
        execution=execution,
        artifact_paths=artifact_paths,
    )
    evidence_path = runtime_root / "product-obligation-evidence.json"
    evidence_path.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return evidence_path


def execute_exact_retained_regressions(
    *,
    repo_root: str | Path,
    provider: Mapping[str, Any],
    obligations: Sequence[Mapping[str, Any]],
    output_root: str | Path,
    obligation_id: str | None = None,
    timeout_seconds: float = 180.0,
    worker_runtime: str = "docker",
) -> tuple[Path, ...]:
    """Execute all 15 exact targets, or one explicitly selected exact target."""

    obligation_index = {
        str(row["obligation_id"]): dict(row)
        for row in obligations
    }
    selected = [
        dict(target)
        for target in provider.get("targets") or ()
        if target.get("variant") == "exact"
        and (
            obligation_id is None
            or target.get("obligation_id") == obligation_id
        )
    ]
    if obligation_id is not None and len(selected) != 1:
        raise ValueError(f"unknown exact G3 obligation: {obligation_id}")
    if obligation_id is None and len(selected) != RETAINED_REGRESSION_EXACT_COUNT:
        raise ValueError("G3 provider does not contain all 15 exact obligations")
    output = Path(output_root).resolve()
    if output.exists():
        raise FileExistsError(f"G3 exact output is immutable: {output}")
    output.mkdir(parents=True)
    evidence_paths: list[Path] = []
    admitted_evidence: list[dict[str, str]] = []
    for target in selected:
        obligation = obligation_index.get(str(target["obligation_id"]))
        if obligation is None:
            raise ValueError("G3 exact target has no authoritative obligation")
        evidence_path = execute_exact_retained_regression(
            repo_root=repo_root,
            obligation=obligation,
            target=target,
            revision=dict(provider["revision_binding"]),
            output_root=output,
            timeout_seconds=timeout_seconds,
            worker_runtime=worker_runtime,
        )
        admission = admit_product_obligation_evidence(
            obligations=(obligation,),
            evidence_paths=(evidence_path,),
            revision=dict(provider["revision_binding"]),
        )
        if not admission["complete"]:
            outcome = admission["outcomes"][str(target["obligation_id"])]
            raise RuntimeError(
                "G3 exact evidence did not pass: "
                f"{target['obligation_id']} outcome={outcome}"
            )
        admitted = dict(
            dict(admission.get("admitted_evidence") or {}).get(
                str(target["obligation_id"])
            )
            or {}
        )
        if (
            Path(str(admitted.get("evidence_path") or "")).resolve()
            != evidence_path.resolve()
            or not _is_sha256(admitted.get("evidence_sha256"))
        ):
            raise RuntimeError(
                "G3 exact admission did not return its immutable evidence binding"
            )
        evidence_paths.append(evidence_path)
        admitted_evidence.append({
            "obligation_id": str(target["obligation_id"]),
            "path": str(evidence_path),
            "sha256": str(admitted["evidence_sha256"]),
        })
    summary_unsigned = {
        "schema_version": 1,
        "artifact_type": "retained_regression_exact_execution_index",
        "provider_hash": provider["provider_hash"],
        "revision_binding": dict(provider["revision_binding"]),
        "selection": "single" if obligation_id else "all",
        "scheduled": len(selected),
        "completed": len(evidence_paths),
        "evidence": admitted_evidence,
    }
    _write_json_once(
        output / "execution-index.json",
        {
            **summary_unsigned,
            "index_hash": content_hash(summary_unsigned),
        },
    )
    return tuple(evidence_paths)


def convert_completed_retained_journey_to_product_evidence(
    *,
    obligation: Mapping[str, Any],
    target: Mapping[str, Any],
    revision: Mapping[str, str],
    source: CompletedJourneySource,
    evidence_path: str | Path,
    provider: str | None = None,
    model: str | None = None,
) -> Path:
    """Convert only a controller-admitted completed-batch source."""

    obligation_id = str(obligation.get("obligation_id") or "")
    source.require_authority(obligation_id=obligation_id)
    with tempfile.TemporaryDirectory(
        prefix="anychain-completed-retained-"
    ) as temporary:
        snapshot_root = source.materialize_runtime(Path(temporary))
        return _convert_retained_journey_runtime_to_product_evidence(
            obligation=obligation,
            target=target,
            revision=revision,
            runtime_root=snapshot_root,
            source_runtime_root=source.runtime_root,
            controller_candidate=(
                source.controller_snapshot.candidate_payload()
            ),
            evidence_path=evidence_path,
            provider=provider,
            model=model,
        )


def _convert_retained_journey_runtime_to_product_evidence(
    *,
    obligation: Mapping[str, Any],
    target: Mapping[str, Any],
    revision: Mapping[str, str],
    runtime_root: str | Path,
    source_runtime_root: str | Path | None = None,
    controller_candidate: Mapping[str, Any] | None = None,
    evidence_path: str | Path,
    provider: str | None = None,
    model: str | None = None,
) -> Path:
    """Validate one completed open Journey and adapt its retained artifacts."""

    if str(obligation.get("variant") or "") == "exact":
        raise ValueError("Journey conversion requires an open G3 obligation")
    configured_identity = load_llm_config()
    provider = str(provider or configured_identity.provider).strip()
    model = str(model or configured_identity.model).strip()
    if provider != DEFAULT_PROVIDER or not model:
        raise ValueError(
            "G3 Journey evidence requires an explicit DeepSeek provider/model"
        )
    root = Path(runtime_root).resolve()
    artifact_root = Path(source_runtime_root or root).resolve()
    result = _load_mapping(root / "journey-result.json", "Journey result")
    schedule_payload = _load_mapping(
        root / "journey-schedule.json",
        "Journey schedule",
    )
    expected_schedule = _schedule_for_evidence(
        obligation=obligation,
        target=target,
        revision=revision,
    )
    if schedule_payload != journey_schedule_payload(expected_schedule):
        raise ValueError("completed Journey schedule does not match G3")
    evidence_source = (
        root / "journey-controller-admission" / "candidate.json"
        if controller_candidate is not None
        else _resolve_runtime_artifact(
            root,
            str(result.get("evidence_path") or ""),
            label="Journey evidence",
        )
    )
    source_evidence = (
        dict(controller_candidate)
        if controller_candidate is not None
        else _load_mapping(evidence_source, "Journey evidence")
    )
    validate_journey_evidence_artifact(
        source_evidence,
        schedule=expected_schedule,
        verifier_registry=RETAINED_REGRESSION_JOURNEY_VERIFIER_REGISTRY,
        revision=revision,
    )
    if (
        result.get("schema_version") != 1
        or result.get("journey_id") != obligation.get("obligation_id")
        or result.get("schedule_id") != expected_schedule.schedule_id
        or dict(result.get("revision") or {}) != dict(revision)
        or result.get("terminal_classification") != "passed"
        or result.get("execution_status") != "passed"
        or result.get("qualifying_evidence") is not True
        or result.get("evidence_id") != source_evidence.get("evidence_id")
        or result.get("turns") != source_evidence.get("turns")
        or source_evidence.get("provider") != provider
        or source_evidence.get("model") != model
    ):
        raise ValueError("completed Journey result is not qualifying G3 evidence")
    retained = source_evidence.get("retained_artifacts")
    if not isinstance(retained, Mapping) or set(retained) != set(
        REQUIRED_ARTIFACT_ROLES
    ):
        raise ValueError("completed Journey lacks retained G3 artifacts")
    artifact_paths: dict[str, Path] = {}
    for role in REQUIRED_ARTIFACT_ROLES:
        reference = retained.get(role)
        if not isinstance(reference, Mapping):
            raise ValueError(f"retained G3 {role} reference is invalid")
        path = _resolve_runtime_artifact(
            root,
            str(reference.get("path") or ""),
            label=f"retained G3 {role}",
            source_root=artifact_root,
        )
        if hashlib.sha256(path.read_bytes()).hexdigest() != reference.get(
            "sha256"
        ):
            raise ValueError(f"retained G3 {role} hash is stale")
        artifact_paths[role] = path
    turns = tuple(source_evidence.get("turns") or ())
    if not turns:
        raise ValueError("completed Journey has no response-bound turns")
    started_at = min(
        int(dict(turn.get("turn_identity") or {})[
            "previous_response_received_at_ns"
        ])
        for turn in turns
        if isinstance(turn, Mapping)
    )
    finished_at = max(
        int(dict(turn.get("turn_identity") or {})[
            "agent_response_received_at_ns"
        ])
        for turn in turns
        if isinstance(turn, Mapping)
    )
    display_paths = {
        role: artifact_root / path.relative_to(root)
        for role, path in artifact_paths.items()
    }
    payload = build_product_obligation_evidence_artifact(
        obligation=obligation,
        target=target,
        revision=revision,
        execution={
            "execution_id": source_evidence["execution_id"],
            "runner": "retained-regression-response-driven-journey-v1",
            "transport": "real_pty",
            "provider": provider,
            "model": model,
            "started_at": str(started_at),
            "finished_at": str(finished_at),
        },
        artifact_paths=artifact_paths,
        artifact_display_paths=display_paths,
    )
    return _write_json_once(Path(evidence_path).resolve(), payload)


def _write_exact_retained_artifacts(
    *,
    runtime_root: Path,
    obligation: Mapping[str, Any],
    target: Mapping[str, Any],
    revision: Mapping[str, str],
    execution_id: str,
    initial_event: RuntimeTurnEvent,
    events: Sequence[RuntimeTurnEvent],
    turns: Sequence[PtyCliTurnRecord],
) -> dict[str, Path]:
    if not turns or len(events) != len(turns):
        raise ValueError("exact retained-regression runtime lineage is incomplete")
    schedule = _schedule_for_evidence(
        obligation=obligation,
        target=target,
        revision=revision,
    )
    identity = {
        "obligation_id": str(obligation["obligation_id"]),
        "execution_id": execution_id,
        "revision": dict(revision),
        "schedule_id": schedule.schedule_id,
        "verifier_input_contract_hash": content_hash(
            _verifier_input_contract(obligation)
        ),
    }
    payloads = {
        "transcript": {
            "turns": [{"turn": asdict(turn)} for turn in turns],
        },
        "runtime_events": {
            "initial_event": asdict(initial_event),
            "events": [asdict(event) for event in events],
        },
        "checkpoint_diff": {
            "before_fingerprint": initial_event.after_fingerprint,
            "after_fingerprint": events[-1].after_fingerprint,
            "material_state_diff_hashes": [
                dict(event.material_state_diff_hashes)
                for event in events
            ],
        },
    }
    paths: dict[str, Path] = {}
    artifact_root = runtime_root / "retained-artifacts"
    artifact_root.mkdir()
    for role, payload in payloads.items():
        unsigned = {
            "schema_version": 1,
            "artifact_type": f"retained_regression_{role}",
            "identity": identity,
            **payload,
        }
        artifact = {**unsigned, "artifact_hash": content_hash(unsigned)}
        path = artifact_root / f"{role}.json"
        path.write_text(
            json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        paths[role] = path
    return paths


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
    journey = {
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
        "verifier_input_contract": _verifier_input_contract(obligation),
    }
    revision = dict(obligation["revision_binding"]["revision"])
    schedule = build_journey_schedule(
        revision=revision,
        seed=RETAINED_REGRESSION_SEED,
        journey=journey,
    )
    base["journey_definition"] = {
        "lane": "journey",
        "verifier_registry": RETAINED_REGRESSION_REGISTRY_IMPORT,
        "verifier_input_contract": _verifier_input_contract(obligation),
        "journey": journey,
        "frozen_execution": {
            "obligation_id": str(obligation["obligation_id"]),
            "seed": RETAINED_REGRESSION_SEED,
            "schedule_id": schedule.schedule_id,
            "schedule_hash": content_hash(
                journey_schedule_payload(schedule)
            ),
            "subject_group": schedule.subject_group,
            "revision_binding": revision,
        },
        "simulator_attestation_contract": {
            "required": True,
            "identity_strength": "auditable_declaration_only",
            "scripted_actor_qualifies": False,
            "cryptographic_identity_claimed": False,
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
    frozen_execution = dict(
        journey_definition.get("frozen_execution") or {}
    )
    simulator_attestation_contract = dict(
        journey_definition.get("simulator_attestation_contract") or {}
    )
    serialized = repr(journey_definition)
    if (
        journey_definition.get("lane") != "journey"
        or journey_definition.get("verifier_registry")
        != RETAINED_REGRESSION_REGISTRY_IMPORT
        or journey_definition.get("verifier_input_contract")
        != _verifier_input_contract(obligation)
        or journey.get("verifier_input_contract")
        != _verifier_input_contract(obligation)
    ):
        raise ValueError("retained regression Journey lane is invalid")
    if any(
        field in journey_definition or field in journey
        for field in ("turns", "messages", "future_user_turns", "dialogue")
    ):
        raise ValueError("response-driven retained regression contains prewritten turns")
    if str(obligation["stimulus_contract"]["source_turns_hash"]) not in serialized:
        raise ValueError("retained regression Journey definition is incomplete")
    revision = dict(obligation["revision_binding"]["revision"])
    schedule = build_journey_schedule(
        revision=revision,
        seed=RETAINED_REGRESSION_SEED,
        journey=journey,
    )
    if frozen_execution != {
        "obligation_id": str(obligation["obligation_id"]),
        "seed": RETAINED_REGRESSION_SEED,
        "schedule_id": schedule.schedule_id,
        "schedule_hash": content_hash(journey_schedule_payload(schedule)),
        "subject_group": schedule.subject_group,
        "revision_binding": revision,
    }:
        raise ValueError(
            "retained regression frozen Journey execution is stale"
        )
    if simulator_attestation_contract != {
        "required": True,
        "identity_strength": "auditable_declaration_only",
        "scripted_actor_qualifies": False,
        "cryptographic_identity_claimed": False,
    }:
        raise ValueError(
            "retained regression simulator attestation policy is invalid"
        )


def _verifier_input_contract(
    obligation: Mapping[str, Any],
) -> dict[str, Any]:
    stimulus = dict(obligation.get("stimulus_contract") or {})
    return {
        "mode": str(stimulus.get("mode") or ""),
        "source_contract": dict(stimulus.get("source_contract") or {}),
        "source_contract_hash": str(
            stimulus.get("source_contract_hash") or ""
        ),
        "variant_contract": dict(stimulus.get("variant_contract") or {}),
        "variant_contract_hash": str(
            stimulus.get("variant_contract_hash") or ""
        ),
    }


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
            "evaluator_implementation_hash": (
                _current_evaluator_implementation_hash(postcondition_id)
            ),
        }
        rules[postcondition_id] = {
            **unsigned,
            "rule_hash": content_hash(unsigned),
        }
    return rules


def _current_evaluator_implementation_hash(postcondition_id: str) -> str:
    evaluator = _IMPLEMENTED_POSTCONDITION_EVALUATORS.get(postcondition_id)
    if evaluator is None:
        identity: dict[str, Any] = {"status": "not_implemented"}
    elif inspect.isfunction(evaluator):
        identity = {
            "module": evaluator.__module__,
            "qualname": evaluator.__qualname__,
            "source": inspect.getsource(evaluator),
        }
    else:
        identity = {
            "status": "invalid_evaluator_object",
            "type": type(evaluator).__qualname__,
        }
    return content_hash(identity)


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
            or not _is_sha256(rule.get("evaluator_implementation_hash"))
            or rule.get("evaluator_implementation_hash")
            != _current_evaluator_implementation_hash(postcondition_id)
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


def _reconstruct_verifier_context(
    artifact_paths: Mapping[str, str | Path],
    *,
    obligation: Mapping[str, Any],
    target: Mapping[str, Any],
    revision: Mapping[str, str],
    execution: Mapping[str, str],
    artifact_display_paths: Mapping[str, str | Path] | None = None,
) -> tuple[list[dict[str, str]], JourneyVerifierContext]:
    variant = str(target.get("variant") or "")
    required_roles = (
        EXACT_RUNTIME_ARTIFACT_ROLES
        if variant == "exact"
        else REQUIRED_ARTIFACT_ROLES
    )
    if set(artifact_paths) != set(required_roles):
        raise ValueError(
            "retained regression evidence has an invalid runtime artifact set"
        )
    expected_schedule = _schedule_for_evidence(
        obligation=obligation,
        target=target,
        revision=revision,
    )
    expected_identity = {
        "obligation_id": str(obligation.get("obligation_id") or ""),
        "execution_id": str(execution.get("execution_id") or ""),
        "revision": dict(revision),
        "schedule_id": expected_schedule.schedule_id,
        "verifier_input_contract_hash": content_hash(
            _verifier_input_contract(obligation)
        ),
    }
    references: list[dict[str, str]] = []
    payloads: dict[str, Mapping[str, Any]] = {}
    for role in REQUIRED_ARTIFACT_ROLES:
        path = Path(artifact_paths[role]).expanduser().resolve()
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError(f"retained regression evidence artifact is missing: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"retained regression {role} artifact is not JSON"
            ) from exc
        if not isinstance(payload, Mapping):
            raise ValueError(f"retained regression {role} artifact is invalid")
        unsigned = {
            key: value for key, value in payload.items()
            if key != "artifact_hash"
        }
        if (
            payload.get("schema_version") != 1
            or payload.get("artifact_type")
            != f"retained_regression_{role}"
            or payload.get("identity") != expected_identity
            or payload.get("artifact_hash") != content_hash(unsigned)
        ):
            raise ValueError(
                f"retained regression {role} artifact binding is stale"
            )
        payloads[role] = payload
        references.append({
            "role": role,
            "path": str(Path(
                (artifact_display_paths or {}).get(role, path)
            ).expanduser().resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    if variant == "exact":
        process_path = Path(
            artifact_paths["process_guard_receipt"]
        ).expanduser().resolve()
        if not process_path.is_file() or process_path.stat().st_size <= 0:
            raise ValueError(
                "retained regression process guard receipt is missing"
            )
        references.append({
            "role": "process_guard_receipt",
            "path": str(Path(
                (artifact_display_paths or {}).get(
                    "process_guard_receipt",
                    process_path,
                )
            ).expanduser().resolve()),
            "sha256": hashlib.sha256(process_path.read_bytes()).hexdigest(),
        })
    transcript_rows = payloads["transcript"].get("turns")
    runtime_rows = payloads["runtime_events"].get("events")
    initial_raw = payloads["runtime_events"].get("initial_event")
    if (
        not isinstance(transcript_rows, list)
        or not transcript_rows
        or not isinstance(runtime_rows, list)
        or len(runtime_rows) != len(transcript_rows)
        or not isinstance(initial_raw, Mapping)
    ):
        raise ValueError("retained regression artifact turn lineage is incomplete")
    initial_event = _runtime_event_from_mapping(initial_raw)
    validate_runtime_turn_event(initial_event)
    events = tuple(
        _runtime_event_from_mapping(item)
        for item in runtime_rows
        if isinstance(item, Mapping)
    )
    if len(events) != len(runtime_rows):
        raise ValueError("retained regression runtime event is invalid")
    turns: list[PtyCliTurnRecord] = []
    decisions: list[JourneyDecisionProvenance] = []
    transcript: list[tuple[str, str]] = []
    seen_broker_request_ids: set[str] = set()
    verifier_input = validate_verifier_input_contract(
        _verifier_input_contract(obligation)
    )
    for index, row in enumerate(transcript_rows):
        expected_row_fields = (
            {"turn"} if variant == "exact" else {"turn", "decision"}
        )
        if not isinstance(row, Mapping) or set(row) != expected_row_fields:
            raise ValueError("retained regression transcript row is invalid")
        turn_raw = row.get("turn")
        decision_raw = row.get("decision")
        if not isinstance(turn_raw, Mapping) or (
            variant != "exact"
            and not isinstance(decision_raw, Mapping)
        ):
            raise ValueError("retained regression transcript lineage is invalid")
        try:
            turn = PtyCliTurnRecord(**dict(turn_raw))
            decision = (
                JourneyDecisionProvenance(**dict(decision_raw))
                if variant != "exact"
                else None
            )
        except TypeError as exc:
            raise ValueError(
                "retained regression transcript schema is invalid"
            ) from exc
        event = events[index]
        validate_runtime_turn_event(event)
        expected_before = (
            initial_event.after_fingerprint
            if index == 0
            else events[index - 1].after_fingerprint
        )
        expected_turn_index = (
            initial_event.turn_index + 1
            if index == 0
            else events[index - 1].turn_index + 1
        )
        expected_context_binding = {
            "session_id": turn.session_id,
            "turn_index": turn.turn_index,
            "previous_response_hash": content_hash(
                turn.previous_agent_response
            ),
            "previous_response_received_at_ns": (
                turn.previous_response_received_at_ns
            ),
            "control_identity": {
                "lane": "journey",
                "schedule_id": expected_schedule.schedule_id,
            },
            "observed_edge_keys": list(
                (decision.simulator_context_binding if decision else {}).get(
                    "observed_edge_keys", ()
                )
            ),
        }
        if (
            turn.session_id != initial_event.thread_id
            or event.thread_id != initial_event.thread_id
            or event.turn_index != expected_turn_index
            or turn.turn_index != event.turn_index
            or (
                decision is not None
                and decision.turn_index != event.turn_index
            )
            or event.before_fingerprint != expected_before
            or turn.before_fingerprint != event.before_fingerprint
            or turn.after_fingerprint != event.after_fingerprint
            or turn.transcript_hash
            != pty_transcript_hash(
                session_id=turn.session_id,
                turn_index=turn.turn_index,
                previous_agent_response=turn.previous_agent_response,
                user_message=turn.user_message,
                agent_response=turn.agent_response,
            )
            or (
                decision is not None
                and (
                    decision.previous_response_hash
                    != content_hash(turn.previous_agent_response)
                    or decision.user_message_hash
                    != content_hash(turn.user_message)
                    or decision.submitted_at_ns
                    != turn.user_message_submitted_at_ns
                    or dict(decision.simulator_context_binding)
                    != expected_context_binding
                )
            )
            or turn.provider != execution["provider"]
            or turn.model != execution["model"]
            or turn.previous_response_received_at_ns
            < int(execution["started_at"])
            or turn.agent_response_received_at_ns
            > int(execution["finished_at"])
        ):
            raise ValueError(
                "retained regression transcript/runtime binding is stale"
            )
        if decision is not None:
            request_id = str(decision.broker_request_id or "")
            if not request_id or request_id in seen_broker_request_ids:
                raise ValueError(
                    "retained regression broker request identity is missing or replayed"
                )
            seen_broker_request_ids.add(request_id)
            unsigned_decision = {
                "previous_response_hash": decision.previous_response_hash,
                "broker_request_id": request_id,
                "user_message": turn.user_message,
                "persona": decision.persona,
                "mission": decision.mission,
                "rationale": decision.rationale,
                "risk_factor_ids": list(decision.risk_factor_ids),
                "variant_binding": {
                    "source_step_id": str(
                        decision.variant_attestation.get(
                            "source_step_id", ""
                        )
                    ),
                    "semantic_role": str(
                        decision.variant_attestation.get(
                            "semantic_role", ""
                        )
                    ),
                },
            }
            validate_simulator_attestation(
                decision.simulator_attestation,
                previous_response_hash=decision.previous_response_hash,
                decision_hash=content_hash(unsigned_decision),
                request_id=request_id,
                context_hash=content_hash(
                    decision.simulator_context_binding
                ),
                user_message_hash=decision.user_message_hash,
                turn_index=turn.turn_index,
                submitted_at_ns=turn.user_message_submitted_at_ns,
            )
            validate_variant_attestation(
                decision.variant_attestation,
                verifier_input_contract=verifier_input,
                turn=turn,
                decision=decision,
            )
        turns.append(turn)
        if decision is not None:
            decisions.append(decision)
        transcript.append((turn.user_message, turn.agent_response))
    checkpoint = payloads["checkpoint_diff"]
    if (
        checkpoint.get("before_fingerprint") != initial_event.after_fingerprint
        or checkpoint.get("after_fingerprint") != events[-1].after_fingerprint
        or checkpoint.get("material_state_diff_hashes")
        != [dict(event.material_state_diff_hashes) for event in events]
    ):
        raise ValueError("retained regression checkpoint diff is stale")
    if dict(initial_event.revision) != dict(revision) or any(
        dict(event.revision) != dict(revision) for event in events
    ):
        raise ValueError("retained regression runtime revision is stale")
    context = JourneyVerifierContext(
        schedule=expected_schedule,
        initial_event=initial_event,
        current_event=events[-1],
        completed_turns=tuple(turns),
        transcript=tuple(transcript),
        observed_edge_keys=(),
        latest_turn=turns[-1],
        completed_events=events,
        completed_decisions=tuple(decisions),
        verifier_input_contract=verifier_input,
    )
    return references, context


def _schedule_for_evidence(
    *,
    obligation: Mapping[str, Any],
    target: Mapping[str, Any],
    revision: Mapping[str, str],
) -> Any:
    if str(target.get("variant") or "") != "exact":
        journey_definition = dict(target.get("journey_definition") or {})
        journey = dict(journey_definition.get("journey") or {})
    else:
        verifier = dict(obligation.get("verifier_contract") or {})
        journey = {
            "journey_id": str(obligation.get("obligation_id") or ""),
            "start_scenario": str(
                dict(obligation.get("seed_contract") or {}).get(
                    "scenario_id", ""
                )
            ),
            "persona": "exact retained regression fixture replay",
            "mission": "Replay the immutable retained regression turns.",
            "allowed_risk_factors": [],
            "max_turns": max(
                1,
                len(
                    dict(obligation.get("stimulus_contract") or {}).get(
                        "turns", ()
                    )
                ),
            ),
            "terminal_outcome": {
                "outcome_id": (
                    f"{obligation.get('obligation_id')}-exact-complete"
                ),
                "required_postcondition_ids": list(
                    verifier.get("required_postcondition_ids") or ()
                ),
            },
            "forbidden_outcomes": [
                {
                    "outcome_id": (
                        f"{obligation.get('obligation_id')}-exact-forbidden-"
                        f"{index}"
                    ),
                    "required_postcondition_ids": [postcondition_id],
                }
                for index, postcondition_id in enumerate(
                    verifier.get("forbidden_postcondition_ids") or (),
                    start=1,
                )
            ],
            "verifier_input_contract": _verifier_input_contract(obligation),
        }
    return build_journey_schedule(
        revision=revision,
        seed=RETAINED_REGRESSION_SEED,
        journey=journey,
    )


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
    try:
        started_at_ns = int(payload["started_at"])
        finished_at_ns = int(payload["finished_at"])
    except ValueError as exc:
        raise ValueError(
            "retained regression execution interval is invalid"
        ) from exc
    if started_at_ns <= 0 or finished_at_ns <= started_at_ns:
        raise ValueError(
            "retained regression execution interval is invalid"
        )
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


def _validated_definition_manifest(
    *,
    provider: Mapping[str, Any],
    manifest: Mapping[str, Any],
    require_all: bool,
) -> tuple[dict[str, Any], ...]:
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    definitions = manifest.get("definitions")
    if (
        manifest.get("schema_version")
        != RETAINED_REGRESSION_DEFINITION_MANIFEST_SCHEMA_VERSION
        or manifest.get("artifact_type")
        != "retained_regression_journey_definition_manifest"
        or manifest.get("provider_hash") != provider.get("provider_hash")
        or manifest.get("revision_binding") != provider.get("revision_binding")
        or manifest.get("generation_is_execution") is not False
        or manifest.get("prewritten_future_turns") is not False
        or manifest.get("actor_attestation_policy")
        != _AUDITABLE_ACTOR_ATTESTATION_POLICY
        or manifest.get("manifest_hash") != content_hash(unsigned)
        or not isinstance(definitions, Sequence)
        or isinstance(definitions, (str, bytes))
        or manifest.get("definition_count") != len(definitions)
    ):
        raise ValueError("retained regression definition manifest is invalid")
    authoritative = {
        str(target["obligation_id"]): dict(target)
        for target in provider.get("targets") or ()
        if target.get("variant") != "exact"
    }
    if require_all and (
        manifest.get("selection") != "all"
        or len(definitions) != RETAINED_REGRESSION_OPEN_COUNT
    ):
        raise ValueError("G3 execution requires all 45 open definitions")
    if not require_all and manifest.get("selection") not in {"all", "single"}:
        raise ValueError("G3 definition selection is invalid")
    validated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in definitions:
        if not isinstance(raw, Mapping):
            raise ValueError("G3 definition row is invalid")
        row = dict(raw)
        obligation_id = str(row.get("obligation_id") or "")
        target = authoritative.get(obligation_id)
        definition = row.get("definition")
        if (
            set(row)
            != {
                "obligation_id",
                "obligation_contract_hash",
                "variant",
                "definition",
                "definition_hash",
            }
            or target is None
            or obligation_id in seen
            or row.get("obligation_contract_hash")
            != target["obligation_contract_hash"]
            or row.get("variant") != target["variant"]
            or definition != target["journey_definition"]
            or row.get("definition_hash") != content_hash(definition)
        ):
            raise ValueError("G3 definition differs from its provider")
        _reject_future_turns(definition)
        seen.add(obligation_id)
        validated.append(row)
    expected_ids = set(authoritative)
    if require_all and seen != expected_ids:
        raise ValueError("G3 definition manifest is incomplete")
    return tuple(validated)


def _validated_target_set(
    *,
    provider: Mapping[str, Any],
    target_manifest: Mapping[str, Any],
    targets_dir: Path,
) -> tuple[dict[str, Any], ...]:
    unsigned = {
        key: value
        for key, value in target_manifest.items()
        if key != "manifest_hash"
    }
    rows = target_manifest.get("targets")
    canonical_definitions = build_retained_regression_definition_manifest(
        provider
    )
    if (
        target_manifest.get("schema_version")
        != RETAINED_REGRESSION_TARGET_SET_SCHEMA_VERSION
        or target_manifest.get("artifact_type")
        != "retained_regression_target_set"
        or target_manifest.get("provider_hash") != provider.get("provider_hash")
        or target_manifest.get("definition_manifest_hash")
        != canonical_definitions["manifest_hash"]
        or target_manifest.get("revision_binding")
        != provider.get("revision_binding")
        or target_manifest.get("manifest_hash") != content_hash(unsigned)
        or not isinstance(rows, Sequence)
        or isinstance(rows, (str, bytes))
        or len(rows) != RETAINED_REGRESSION_OPEN_COUNT
        or target_manifest.get("target_count") != len(rows)
    ):
        raise ValueError("retained regression target set is invalid")
    definitions = _validated_definition_manifest(
        provider=provider,
        manifest=canonical_definitions,
        require_all=True,
    )
    obligation_set: list[dict[str, Any]] = []
    validated: list[dict[str, Any]] = []
    for index, (raw, definition_row) in enumerate(
        zip(rows, definitions, strict=True),
        start=1,
    ):
        if not isinstance(raw, Mapping):
            raise ValueError("G3 target row is invalid")
        row = dict(raw)
        expected_name = f"{index:02d}.json"
        target_path = (targets_dir / expected_name).resolve()
        try:
            target_path.relative_to(targets_dir)
        except ValueError as exc:
            raise ValueError("G3 target escapes its immutable directory") from exc
        if (
            row.get("index") != index
            or row.get("path") != expected_name
            or row.get("obligation_id") != definition_row["obligation_id"]
            or row.get("obligation_contract_hash")
            != definition_row["obligation_contract_hash"]
            or row.get("variant") != definition_row["variant"]
            or not target_path.is_file()
            or hashlib.sha256(target_path.read_bytes()).hexdigest()
            != row.get("sha256")
            or _load_mapping(target_path, "G3 target")
            != _batch_target_payload(definition_row["definition"])
        ):
            raise ValueError("G3 target file differs from its definition")
        frozen = dict(definition_row["definition"]["frozen_execution"])
        if {
            "seed": row.get("seed"),
            "schedule_id": row.get("schedule_id"),
            "subject_group": row.get("subject_group"),
        } != {
            "seed": frozen["seed"],
            "schedule_id": frozen["schedule_id"],
            "subject_group": frozen["subject_group"],
        }:
            raise ValueError("G3 target frozen execution is stale")
        obligation_set.append({
            "obligation_id": row["obligation_id"],
            "schedule_id": row["schedule_id"],
            "seed": row["seed"],
            "subject_group": row["subject_group"],
        })
        validated.append(row)
    if target_manifest.get("expected_obligation_set_hash") != content_hash(
        obligation_set
    ):
        raise ValueError("G3 target obligation set hash is stale")
    return tuple(validated)


def _batch_target_payload(definition: Mapping[str, Any]) -> dict[str, Any]:
    """Project provider metadata onto the shared Journey target schema."""

    payload = {
        key: value
        for key, value in definition.items()
        if key != "verifier_input_contract"
    }
    if set(payload) != {
        "lane",
        "verifier_registry",
        "journey",
        "frozen_execution",
        "simulator_attestation_contract",
    }:
        raise ValueError("G3 Journey cannot be projected onto the batch schema")
    return payload


def _reject_future_turns(value: Any) -> None:
    forbidden = {"turns", "messages", "future_user_turns", "dialogue"}
    if isinstance(value, Mapping):
        leaked = forbidden & set(value)
        if leaked:
            raise ValueError(
                "response-driven G3 definition contains prewritten turns: "
                + ", ".join(sorted(leaked))
            )
        for nested in value.values():
            _reject_future_turns(nested)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for nested in value:
            _reject_future_turns(nested)


def _resolve_runtime_artifact(
    root: Path,
    raw_path: str,
    *,
    label: str,
    source_root: Path | None = None,
) -> Path:
    declared = Path(raw_path).expanduser()
    translated = None
    if source_root is not None and declared.is_absolute():
        try:
            translated = root / declared.resolve().relative_to(
                source_root.resolve()
            )
        except ValueError as exc:
            raise ValueError(f"{label} is outside its runtime root") from exc
    candidates = tuple(path for path in (
        translated,
        declared if source_root is None else None,
        root / "evidence" / declared.name,
        root / "retained-artifacts" / declared.name,
    ) if path is not None)
    matches = tuple(
        path.resolve()
        for path in candidates
        if path.is_file()
    )
    unique = tuple(dict.fromkeys(matches))
    if len(unique) != 1:
        raise ValueError(f"{label} path cannot be resolved unambiguously")
    try:
        unique[0].relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} is outside its runtime root") from exc
    return unique[0]


def _load_mapping(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not readable JSON: {path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return dict(payload)


def _write_json_once(path: Path, payload: Mapping[str, Any]) -> Path:
    if path.exists():
        raise FileExistsError(f"immutable output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_json_bytes(payload))
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return path


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _validated_revision(revision: Mapping[str, str]) -> dict[str, str]:
    normalized = {
        "commit": str(revision.get("commit") or "").strip(),
        "worktree_hash": str(revision.get("worktree_hash") or "").strip(),
    }
    if not normalized["commit"] or not normalized["worktree_hash"]:
        raise ValueError("retained regression runner requires a revision binding")
    return normalized


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Execute and adapt the complete Phase 8 G3 catalog.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    provider = commands.add_parser("provider")
    provider.add_argument("--repo-root", required=True, type=Path)
    provider.add_argument("--output", required=True, type=Path)

    validate = commands.add_parser("validate-provider")
    validate.add_argument("--repo-root", required=True, type=Path)
    validate.add_argument("--provider", required=True, type=Path)

    exact = commands.add_parser("exact")
    exact.add_argument("--repo-root", required=True, type=Path)
    exact.add_argument("--provider", required=True, type=Path)
    exact.add_argument("--output-root", required=True, type=Path)
    exact.add_argument("--obligation-id")
    exact.add_argument("--timeout-seconds", type=float, default=180.0)
    exact.add_argument(
        "--worker-runtime",
        choices=("linux", "docker"),
        default="docker",
    )

    definitions = commands.add_parser("definitions")
    definitions.add_argument("--repo-root", required=True, type=Path)
    definitions.add_argument("--provider", required=True, type=Path)
    definitions.add_argument("--output", required=True, type=Path)
    definitions.add_argument("--obligation-id")

    targets = commands.add_parser("targets")
    targets.add_argument("--repo-root", required=True, type=Path)
    targets.add_argument("--provider", required=True, type=Path)
    targets.add_argument("--definitions", required=True, type=Path)
    targets.add_argument("--output-dir", required=True, type=Path)

    batch = commands.add_parser("batch")
    batch.add_argument("--repo-root", required=True, type=Path)
    batch.add_argument("--provider", required=True, type=Path)
    batch.add_argument("--targets-dir", required=True, type=Path)
    batch.add_argument("--output", required=True, type=Path)
    batch.add_argument("--runtime-base", required=True, type=Path)
    batch.add_argument("--broker-root", type=Path)
    batch.add_argument("--result-index", type=Path)
    batch.add_argument("--decision-timeout-seconds", type=float, default=120.0)
    batch.add_argument("--max-concurrency", type=int)
    batch.add_argument(
        "--worker-runtime",
        choices=("linux", "docker"),
        default="linux",
    )

    evidence_batch = commands.add_parser("evidence-batch")
    evidence_batch.add_argument("--repo-root", required=True, type=Path)
    evidence_batch.add_argument("--provider", required=True, type=Path)
    evidence_batch.add_argument("--manifest", required=True, type=Path)
    evidence_batch.add_argument("--result-index", required=True, type=Path)
    evidence_batch.add_argument("--authority-trust-root-id", required=True)
    evidence_batch.add_argument("--output-dir", required=True, type=Path)
    evidence_batch.add_argument("--provider-name")
    evidence_batch.add_argument("--model")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "provider":
        write_retained_regression_provider(
            repo_root=args.repo_root,
            output_path=args.output,
        )
        return 0

    provider, obligations, revision = load_retained_regression_provider(
        repo_root=args.repo_root,
        provider_path=args.provider,
    )
    if args.command == "validate-provider":
        return 0
    configured_identity = load_llm_config()
    evidence_provider = (
        getattr(args, "provider_name", None) or configured_identity.provider
    )
    evidence_model = getattr(args, "model", None) or configured_identity.model
    if args.command == "exact":
        execute_exact_retained_regressions(
            repo_root=args.repo_root,
            provider=provider,
            obligations=obligations,
            output_root=args.output_root,
            obligation_id=args.obligation_id,
            timeout_seconds=args.timeout_seconds,
            worker_runtime=args.worker_runtime,
        )
        return 0
    if args.command == "definitions":
        manifest = build_retained_regression_definition_manifest(
            provider,
            obligation_id=args.obligation_id,
        )
        write_retained_regression_definition_manifest(
            manifest,
            args.output,
        )
        return 0
    if args.command == "targets":
        manifest = _load_mapping(
            args.definitions.resolve(),
            "retained regression definition manifest",
        )
        write_retained_regression_target_set(
            provider=provider,
            definition_manifest=manifest,
            output_dir=args.output_dir,
        )
        return 0
    if args.command == "batch":
        manifest = freeze_retained_regression_open_batch(
            repo_root=args.repo_root,
            provider=provider,
            targets_dir=args.targets_dir,
            manifest_path=args.output,
            runtime_base=args.runtime_base,
            max_concurrency=args.max_concurrency,
            worker_runtime=args.worker_runtime,
        )
        print(json.dumps({
            "manifest_id": manifest.manifest_id,
            "authority_trust_root_id": manifest.pty_authority_trust_root_id,
        }, sort_keys=True))
        if bool(args.broker_root) != bool(args.result_index):
            raise ValueError(
                "--broker-root and --result-index must be supplied together"
            )
        if args.broker_root and args.result_index:
            from tests.agent_live.filesystem_decision_broker import (
                FilesystemDecisionBroker,
                _run_with_signal_cleanup,
            )

            broker = FilesystemDecisionBroker(
                args.broker_root,
                batch_id=manifest.batch_id,
                timeout_seconds=args.decision_timeout_seconds,
            )
            asyncio.run(_run_with_signal_cleanup(
                manifest,
                broker=broker,
                result_index_path=args.result_index,
                authority_signer=manifest._authority_signer,
            ))
        return 0

    if args.command == "evidence-batch":
        open_obligations = tuple(
            row for row in obligations if row.get("variant") != "exact"
        )
        target_index = {
            str(row["obligation_id"]): dict(row)
            for row in provider["targets"]
        }

        def convert_one(
            obligation: Mapping[str, Any],
            source: CompletedJourneySource,
            evidence_path: Path,
            _checkpoint_diff: Path | None,
        ) -> Path:
            return convert_completed_retained_journey_to_product_evidence(
                obligation=obligation,
                target=target_index[str(obligation["obligation_id"])],
                revision=revision,
                source=source,
                evidence_path=evidence_path,
                provider=evidence_provider,
                model=evidence_model,
            )

        convert_completed_journey_batch(
            manifest_path=args.manifest,
            result_index_path=args.result_index,
            output_dir=args.output_dir,
            obligations=open_obligations,
            revision=revision,
            artifact_type=G3_ARTIFACT_TYPE,
            round_id="",
            expected_authority_trust_root_id=args.authority_trust_root_id,
            convert_one=convert_one,
        )
        return 0

    raise ValueError(f"unsupported retained-regression command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())

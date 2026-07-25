"""Tamper-evident artifacts for compiled-graph and real PTY evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import struct
import time
import urllib.parse
import zlib
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from tests.agent_live.coverage_events import state_diff_between
from agent.harness.runtime_identity import repository_revision
from agent.runners.artifact_ownership import (
    ARTIFACT_MANIFEST_SCHEMA_VERSION,
    derive_artifact_owner_roots,
    required_artifact_owner,
    validate_owned_artifact_path,
)
from agent.harness.control_receipts import (
    validate_persisted_domain_control_receipt,
)
from agent.runners.execution_scenarios import (
    EXECUTION_SCENARIOS,
    scenario_by_id,
    workflow_type_from_plan,
)
from agent.runners.plan_projection import validate_execution_plan_projection
from agent.runners.runtime_env_projection import validate_runtime_env_projection
from agent.utils.redaction import redact
from tests.agent_live.real_execution_host_attestation import (
    validate_host_attestation_file,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_SCHEMA_VERSION = 4
CLI_ARTIFACT_SCHEMA_VERSION = 6
REAL_EXECUTION_ARTIFACT_SCHEMA_VERSION = 4
TURN_OBSERVATION_SCHEMA_VERSION = 2
PTY_DIAGNOSTIC_SCHEMA_VERSION = 1
PTY_DIAGNOSTIC_STATUSES = {
    "failed_attempt": frozenset({
        "verification_pending", "verification_error", "postcondition_failed",
    }),
    "interruption": frozenset({"interrupted"}),
}


@dataclass(frozen=True)
class G5RuntimeContract:
    schema_version: int
    contract_id: str
    chain_id: str
    image_digest: str
    image_reference: str
    rpc_url: str
    metrics_url: str
    compose_service: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


G5_RUNTIME_CONTRACT = G5RuntimeContract(
    schema_version=1,
    contract_id="phase-e-g5-local-geth-v1",
    chain_id="0x539",
    image_digest="sha256:746d19134a870e0c83760eee5897696678b0e57c1f73e3392254fcd8426e5782",
    image_reference="ethereum/client-go@sha256:746d19134a870e0c83760eee5897696678b0e57c1f73e3392254fcd8426e5782",
    rpc_url="http://geth-dev:8545",
    metrics_url="http://geth-dev:6060/debug/metrics/prometheus",
    compose_service="geth-dev",
)


@dataclass(frozen=True)
class G5ScenarioAdmission:
    sequence_index: int
    predecessor_scenario_id: str = ""
    endpoint_env_var: str = ""


G5_SCENARIO_ADMISSION = {
    "rpc_fake_node_smoke": G5ScenarioAdmission(sequence_index=1),
    "rpc_real_node_smoke": G5ScenarioAdmission(
        sequence_index=2,
        predecessor_scenario_id="rpc_fake_node_smoke",
        endpoint_env_var="LOCAL_RPC_URL",
    ),
    "rpc_real_node_final": G5ScenarioAdmission(
        sequence_index=3,
        predecessor_scenario_id="rpc_real_node_smoke",
        endpoint_env_var="LOCAL_RPC_URL",
    ),
    "sync_observe_bounded": G5ScenarioAdmission(
        sequence_index=4,
        predecessor_scenario_id="rpc_real_node_final",
        endpoint_env_var="SYNC_OBSERVE_RPC_URL",
    ),
}


def g5_scenario_admission(scenario_id: str) -> G5ScenarioAdmission:
    try:
        return G5_SCENARIO_ADMISSION[scenario_id]
    except KeyError as exc:
        raise ValueError(f"scenario has no G5 admission contract: {scenario_id}") from exc
EXECUTION_EVIDENCE_CLASSES = frozenset(
    {"deterministic", "real_cli", "dynamic_dual_ai", "real_execution"}
)
COMPILED_GRAPH_RUNNER = "tests.agent_live.graph_turn.invoke_product_graph_turn"
PTY_REAL_CLI_RUNNER = "tests.agent_live.pty.real_cli"
PTY_DYNAMIC_DUAL_AI_RUNNER = "tests.agent_live.pty.dynamic_dual_ai"
REAL_EXECUTION_RUNNER = "tests.agent_live.real_execution.integration"


@dataclass(frozen=True)
class PtyCliTurnRecord:
    """One observed PTY turn, including state and transcript fingerprints."""

    session_id: str
    turn_index: int
    previous_agent_response: str
    user_message: str
    agent_response: str
    provider: str
    model: str
    before_fingerprint: str
    after_fingerprint: str
    transcript_hash: str
    previous_response_received_at_ns: int
    user_message_submitted_at_ns: int
    agent_response_received_at_ns: int


@dataclass(frozen=True)
class RuntimeTurnEvent:
    """One product-process event read from ``ANYCHAIN_AGENT_TURN_EVENT_FILE``."""

    schema_version: int
    event_type: str
    thread_id: str
    session_purpose: str
    before_fingerprint: str
    after_fingerprint: str
    turn_index: int
    active_group: str
    pending_question_id: str
    action_queue_types: tuple[str, ...]
    pending_contract: Mapping[str, Any] = field(default_factory=dict)
    revision: Mapping[str, str] = field(default_factory=dict)
    admitted_action_types: tuple[str, ...] = ()
    admitted_action_targets: tuple[Mapping[str, str], ...] = ()
    admitted_action_provenance: tuple[Mapping[str, Any], ...] = ()
    turn_receipt_summary: Mapping[str, Any] = field(default_factory=dict)
    pending_transition: Mapping[str, Any] = field(default_factory=dict)
    render_manifest: Mapping[str, Any] = field(default_factory=dict)
    execution_receipt_summary: Mapping[str, Any] = field(default_factory=dict)
    control_receipts: tuple[Mapping[str, Any], ...] = ()
    state_diff_hashes: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    material_state_diff_hashes: Mapping[str, Mapping[str, str]] = field(
        default_factory=dict
    )
    after_value_hashes: Mapping[str, str] = field(default_factory=dict)
    next_result: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VerifiedPostcondition:
    """Facts independently checked after a PTY turn.

    The PTY transport cannot produce this record. A Linux/Docker integration
    must inspect the committed checkpoint or resulting job artifacts and name
    the verifier it used.
    """

    verifier_id: str
    passed: bool
    observed_coverage_ids: tuple[str, ...]
    admitted_typed_actions: tuple[str, ...]
    state_diff: Mapping[str, Any]
    next_question_or_result: Mapping[str, Any]
    details: Mapping[str, Any]
    job_artifacts: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True)
class TurnObservation:
    """Revision-bound observed turn shared by PTY coverage consumers."""

    seed: int
    revision: Mapping[str, str]
    target_edge_key: str
    target_contract_hash: str
    target_variant_hash: str
    prior_agent_response: str
    simulator_decision: Mapping[str, Any]
    exact_user_turn: str
    provider: str
    model: str
    before_turn_index: int
    after_turn_index: int
    before_state_fingerprint: str
    after_state_fingerprint: str
    pending_contract: Mapping[str, Any]
    runtime_events: tuple[RuntimeTurnEvent, ...]
    verified_postcondition: VerifiedPostcondition
    continuation_turns: tuple[PtyCliTurnRecord, ...] = ()
    continuation_simulator_decisions: tuple[Mapping[str, Any], ...] = ()
    schema_version: int = TURN_OBSERVATION_SCHEMA_VERSION


@dataclass(frozen=True)
class DynamicTurnSelection:
    """Auditable response-driven decision made by the Codex user simulator."""

    persona: str
    goal: str
    selected_message: str
    rationale: str
    target_coverage_ids: tuple[str, ...]
    selected_at_ns: int
    simulator: str = "codex"
    selection_mode: str = "response_driven"


@dataclass(frozen=True)
class PtyDiagnosticRecord:
    """Non-qualifying diagnostic for one completed PTY boundary.

    Diagnostics are intentionally separate from coverage evidence.  A failed
    attempt may contain the just-completed turn; an interruption may contain
    only the last boundary that was fully observed before transport failed.
    """

    diagnostic_kind: str
    verification_status: str
    target_id: str
    target_edge_key: str
    revision: Mapping[str, str]
    session_id: str
    reason: str
    last_complete_response: str
    last_complete_event: RuntimeTurnEvent
    completed_turn: PtyCliTurnRecord | None = None
    dynamic_selection: DynamicTurnSelection | None = None
    verified_postcondition: VerifiedPostcondition | None = None
    schema_version: int = PTY_DIAGNOSTIC_SCHEMA_VERSION


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def pty_transcript_hash(
    *,
    session_id: str,
    turn_index: int,
    previous_agent_response: str,
    user_message: str,
    agent_response: str,
) -> str:
    """Fingerprint the exact transcript context represented by one PTY turn."""

    return content_hash({
        "session_id": session_id,
        "turn_index": turn_index,
        "previous_agent_response": previous_agent_response,
        "user_message": user_message,
        "agent_response": agent_response,
    })


def build_evidence_artifact(
    *,
    edge: Mapping[str, Any],
    evidence_class: str,
    scenario_id: str,
    runner_type: str,
    revision: Mapping[str, str],
    input_value: Any,
    seed_state: Mapping[str, Any],
    before_state: Mapping[str, Any],
    after_state: Mapping[str, Any],
    events: Iterable[Mapping[str, Any]],
    exit_status: int,
    outcome: str,
    error: str = "",
    started_at: str | None = None,
    finished_at: str | None = None,
    admitted: bool | None = None,
) -> dict[str, Any]:
    if evidence_class != "deterministic":
        raise ValueError(
            "compiled-graph evidence only supports deterministic turns; "
            "use build_pty_cli_evidence_artifact for real_cli or dynamic_dual_ai"
        )
    _require_lane(edge, evidence_class)
    required_text = {
        "edge_key": edge.get("edge_key"),
        "contract_hash": edge.get("contract_hash"),
        "runtime_variant_hash": edge.get("contract_variant_hash"),
        "scenario_id": scenario_id,
        "runner_type": runner_type,
        "revision.commit": revision.get("commit"),
        "revision.worktree_hash": revision.get("worktree_hash"),
    }
    missing = sorted(name for name, value in required_text.items() if not str(value or "").strip())
    if missing:
        raise ValueError(f"missing evidence identity: {', '.join(missing)}")
    if runner_type != COMPILED_GRAPH_RUNNER:
        raise ValueError(f"deterministic turn evidence requires {COMPILED_GRAPH_RUNNER}")
    if outcome not in {"passed", "failed"}:
        raise ValueError(f"invalid evidence outcome: {outcome}")
    if (outcome == "passed") != (int(exit_status) == 0):
        raise ValueError("outcome and exit status disagree")
    if outcome == "failed" and not str(error or "").strip():
        raise ValueError("failed artifact requires an error")

    raw_seed = deepcopy(dict(seed_state))
    raw_before = deepcopy(dict(before_state))
    raw_after = deepcopy(dict(after_state))
    if raw_before.get("last_user_input") != input_value:
        raise ValueError("input does not match the compiled-graph invocation state")
    seed = redact(raw_seed)
    before = redact(raw_before)
    after = redact(raw_after)
    safe_input = redact(deepcopy(input_value))
    question = deepcopy(dict(before.get("pending_question") or {}))
    event_trace = []
    for event in events:
        safe_event = deepcopy(dict(event))
        safe_event["details"] = redact(dict(safe_event.get("details") or {}))
        event_trace.append(safe_event)
    returned = _has_event(event_trace, "compiled_graph_turn_returned", edge)
    raised = _has_event(event_trace, "compiled_graph_turn_raised", edge)
    if outcome == "passed" and not returned:
        raise ValueError("passed artifact has no compiled graph return")
    if outcome == "failed" and not (returned or raised):
        raise ValueError("failed artifact has no compiled graph observation")
    if any(
        str(event.get("event_type") or "") in {"edge_executed", "edge_execution_failed"}
        for event in event_trace
    ):
        raise ValueError("runner-declared edge outcome events are forbidden")

    raw_question = deepcopy(dict(raw_before.get("pending_question") or {}))
    raw_state_diff = state_diff_between(raw_before, raw_after)
    raw_response = deepcopy(list(raw_after.get("visible_response") or []))
    raw_next_question = deepcopy(dict(raw_after.get("pending_question") or {}))
    state_diff = state_diff_between(before, after)
    response = deepcopy(list(after.get("visible_response") or []))
    next_question = deepcopy(dict(after.get("pending_question") or {}))
    admitted_action = _action_admission(
        edge,
        question,
        safe_input,
        admitted=(returned if admitted is None else bool(admitted)),
    )
    turn_evidence = {
        "compiled_graph_helper": COMPILED_GRAPH_RUNNER,
        "seed": seed,
        "before": before,
        "input": safe_input,
        "admitted_action": admitted_action,
        "runtime_actions": {
            "proposed": deepcopy(list(after.get("proposed_actions") or [])),
            "queued": deepcopy(list(after.get("action_queue") or [])),
            "completed": deepcopy(list(after.get("completed_actions") or [])),
            "applied_action_ids": deepcopy(list(after.get("applied_action_ids") or [])),
        },
        "state_diff": state_diff,
        "after": after,
        "question": question,
        "next_question": next_question,
        "response": response,
        "turn_context": deepcopy(dict(after.get("turn_context") or {})),
    }
    payload: dict[str, Any] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "edge_key": str(edge.get("edge_key") or ""),
        "contract_hash": str(edge.get("contract_hash") or ""),
        "runtime_variant_hash": str(edge.get("contract_variant_hash") or ""),
        "evidence_class": evidence_class,
        "scenario_id": str(scenario_id),
        "runner_type": str(runner_type),
        "revision": {
            "commit": str(revision.get("commit") or ""),
            "worktree_hash": str(revision.get("worktree_hash") or ""),
        },
        "input_hash": content_hash(safe_input),
        "source_input_hash": content_hash(input_value),
        "source_boundary_hashes": {
            "before_hash": content_hash(raw_before),
            "input_hash": content_hash(input_value),
            "question_hash": content_hash(raw_question),
            "after_hash": content_hash(raw_after),
            "state_diff_hash": content_hash(raw_state_diff),
            "response_hash": content_hash(raw_response),
            "next_question_hash": content_hash(raw_next_question),
        },
        "seed_state_hash": content_hash(seed),
        "before_state_hash": content_hash(before),
        "after_state_hash": content_hash(after),
        "turn_evidence": turn_evidence,
        "turn_evidence_hash": content_hash(turn_evidence),
        "event_trace": event_trace,
        "event_trace_hash": content_hash(event_trace),
        "exit_status": int(exit_status),
        "outcome": str(outcome),
        "error": str(redact(error or "")),
        "content_redacted": True,
        "started_at": started_at or _utc_timestamp(),
        "finished_at": finished_at or _utc_timestamp(),
    }
    payload["evidence_id"] = content_hash(payload)
    payload["artifact_hash"] = content_hash(payload)
    return payload


def write_evidence_artifact(artifact: Mapping[str, Any], directory: str | Path) -> Path:
    target_dir = Path(directory)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{artifact['evidence_id']}.json"
    target.write_text(
        json.dumps(dict(artifact), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def build_pty_diagnostic_artifact(record: PtyDiagnosticRecord) -> dict[str, Any]:
    """Build a redacted, tamper-evident record that can never satisfy a lane.

    The caller writes a ``verification_pending`` failed-attempt record before
    invoking the postcondition verifier.  It removes that record after a pass
    or replaces it with ``postcondition_failed`` details after a failure.
    """

    if record.diagnostic_kind not in {"failed_attempt", "interruption"}:
        raise ValueError("PTY diagnostic kind is invalid")
    if record.verification_status not in PTY_DIAGNOSTIC_STATUSES[record.diagnostic_kind]:
        raise ValueError("PTY diagnostic verification status is invalid")
    required = {
        "target_id": record.target_id,
        "target_edge_key": record.target_edge_key,
        "revision.commit": record.revision.get("commit"),
        "revision.worktree_hash": record.revision.get("worktree_hash"),
        "session_id": record.session_id,
        "reason": record.reason,
        "last_complete_response": record.last_complete_response,
    }
    missing = sorted(name for name, value in required.items() if not str(value or "").strip())
    if missing:
        raise ValueError(f"PTY diagnostic identity is missing: {', '.join(missing)}")
    _validate_runtime_event(record.last_complete_event)
    if dict(record.last_complete_event.revision) != dict(record.revision):
        raise ValueError("PTY diagnostic runtime revision mismatch")
    if record.last_complete_event.thread_id != record.session_id:
        raise ValueError("PTY diagnostic runtime event belongs to another session")
    if record.diagnostic_kind == "failed_attempt" and record.completed_turn is None:
        raise ValueError("failed-attempt diagnostic requires a completed turn")
    if record.dynamic_selection is not None and record.completed_turn is None:
        raise ValueError("PTY diagnostic selection requires a completed turn")
    if record.completed_turn is not None:
        _validate_pty_turn_record(record.completed_turn)
        if record.completed_turn.session_id != record.session_id:
            raise ValueError("PTY diagnostic turn belongs to another session")
    if record.dynamic_selection is not None:
        _validate_dynamic_selection(record.completed_turn, record.dynamic_selection)  # type: ignore[arg-type]
    if record.verification_status == "postcondition_failed":
        if record.verified_postcondition is None or record.verified_postcondition.passed:
            raise ValueError("postcondition-failed diagnostic requires a failed verification")
    elif record.verified_postcondition is not None:
        raise ValueError("PTY diagnostic carries a postcondition before verification failed")

    safe_response = str(redact(record.last_complete_response))
    safe_event = redact(_runtime_event_payload(record.last_complete_event))
    safe_turn = (
        _redacted_pty_turn_payload(record.completed_turn)
        if record.completed_turn is not None
        else None
    )
    safe_selection = (
        _redacted_dynamic_selection_payload(record.dynamic_selection)
        if record.dynamic_selection is not None
        else None
    )
    safe_postcondition = (
        _redacted_verified_postcondition_payload(record.verified_postcondition)
        if record.verified_postcondition is not None
        else None
    )
    body: dict[str, Any] = {
        "artifact_type": "pty_diagnostic",
        "schema_version": record.schema_version,
        "diagnostic_kind": record.diagnostic_kind,
        "verification_status": record.verification_status,
        "qualifying_evidence": False,
        "target_id": record.target_id,
        "target_edge_key": record.target_edge_key,
        "revision": dict(record.revision),
        "session_id": record.session_id,
        "reason": str(redact(record.reason)),
        "last_complete_boundary": {
            "agent_response": safe_response,
            "agent_response_hash": content_hash(safe_response),
            "runtime_event": safe_event,
            "runtime_event_hash": content_hash(safe_event),
            "completed_turn": safe_turn,
            "completed_turn_hash": content_hash(safe_turn) if safe_turn is not None else "",
            "dynamic_selection": safe_selection,
            "dynamic_selection_hash": (
                content_hash(safe_selection) if safe_selection is not None else ""
            ),
            "verified_postcondition": safe_postcondition,
            "verified_postcondition_hash": (
                content_hash(safe_postcondition) if safe_postcondition is not None else ""
            ),
        },
        "content_redacted": True,
        "created_at": _utc_timestamp(),
    }
    body["diagnostic_id"] = content_hash(body)
    body["artifact_hash"] = content_hash(body)
    return body


def write_pty_diagnostic_artifact(
    artifact: Mapping[str, Any],
    directory: str | Path,
) -> Path:
    """Atomically persist a validated diagnostic outside evidence lanes."""

    valid, reason = validate_pty_diagnostic_artifact(artifact)
    if not valid:
        raise ValueError(f"invalid PTY diagnostic artifact: {reason}")
    target_dir = Path(directory)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{artifact['diagnostic_id']}.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(dict(artifact), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)
    return target


def validate_pty_diagnostic_artifact(artifact: Mapping[str, Any]) -> tuple[bool, str]:
    """Validate diagnostics while keeping them ineligible for pass evidence."""

    if artifact.get("artifact_type") != "pty_diagnostic":
        return False, "artifact is not a PTY diagnostic"
    if artifact.get("schema_version") != PTY_DIAGNOSTIC_SCHEMA_VERSION:
        return False, "unsupported PTY diagnostic schema"
    if artifact.get("qualifying_evidence") is not False:
        return False, "PTY diagnostic must not qualify as evidence"
    kind = str(artifact.get("diagnostic_kind") or "")
    status = str(artifact.get("verification_status") or "")
    if kind not in PTY_DIAGNOSTIC_STATUSES or status not in PTY_DIAGNOSTIC_STATUSES[kind]:
        return False, "PTY diagnostic kind or status is invalid"
    required_text = (
        "target_id", "target_edge_key", "session_id", "reason", "created_at",
        "diagnostic_id", "artifact_hash",
    )
    if any(not str(artifact.get(name) or "").strip() for name in required_text):
        return False, "PTY diagnostic identity is incomplete"
    revision = artifact.get("revision")
    if not isinstance(revision, Mapping):
        return False, "PTY diagnostic revision is invalid"
    if not str(revision.get("commit") or "").strip() or not _is_sha256(
        str(revision.get("worktree_hash") or "")
    ):
        return False, "PTY diagnostic revision identity is invalid"
    boundary = artifact.get("last_complete_boundary")
    if not isinstance(boundary, Mapping):
        return False, "PTY diagnostic has no complete boundary"
    response = boundary.get("agent_response")
    event = boundary.get("runtime_event")
    if not isinstance(response, str) or not response.strip() or not isinstance(event, Mapping):
        return False, "PTY diagnostic complete boundary is invalid"
    if content_hash(response) != str(boundary.get("agent_response_hash") or ""):
        return False, "PTY diagnostic response hash mismatch"
    if content_hash(event) != str(boundary.get("runtime_event_hash") or ""):
        return False, "PTY diagnostic runtime-event hash mismatch"
    try:
        runtime_event_values = dict(event)
        runtime_event_values["action_queue_types"] = tuple(
            runtime_event_values.get("action_queue_types") or ()
        )
        runtime_event_values["admitted_action_types"] = tuple(
            runtime_event_values.get("admitted_action_types") or ()
        )
        runtime_event_values["admitted_action_targets"] = tuple(
            dict(item) for item in runtime_event_values.get("admitted_action_targets") or ()
        )
        runtime_event_values["admitted_action_provenance"] = tuple(
            dict(item)
            for item in runtime_event_values.get("admitted_action_provenance") or ()
        )
        runtime_event_values["control_receipts"] = tuple(
            dict(item)
            for item in runtime_event_values.get("control_receipts") or ()
        )
        runtime_event = RuntimeTurnEvent(**runtime_event_values)
        _validate_runtime_event(runtime_event)
    except (TypeError, ValueError) as exc:
        return False, f"PTY diagnostic runtime event is invalid: {exc}"
    if runtime_event.thread_id != str(artifact.get("session_id") or ""):
        return False, "PTY diagnostic runtime event belongs to another session"
    if dict(runtime_event.revision) != dict(revision):
        return False, "PTY diagnostic runtime revision mismatch"
    completed_turn = boundary.get("completed_turn")
    if kind == "failed_attempt" and not isinstance(completed_turn, Mapping):
        return False, "failed-attempt diagnostic has no completed turn"
    for value_name, hash_name in (
        ("completed_turn", "completed_turn_hash"),
        ("dynamic_selection", "dynamic_selection_hash"),
        ("verified_postcondition", "verified_postcondition_hash"),
    ):
        value = boundary.get(value_name)
        observed_hash = str(boundary.get(hash_name) or "")
        if value is None:
            if observed_hash:
                return False, f"PTY diagnostic {value_name} hash is unexpected"
        elif content_hash(value) != observed_hash:
            return False, f"PTY diagnostic {value_name} hash mismatch"
    typed_turn: PtyCliTurnRecord | None = None
    if isinstance(completed_turn, Mapping):
        try:
            turn_values = dict(completed_turn)
            for derived in (
                "previous_response_hash", "user_message_hash", "agent_response_hash",
            ):
                turn_values.pop(derived, None)
            typed_turn = PtyCliTurnRecord(**turn_values)
            _validate_pty_turn_record(typed_turn)
        except (TypeError, ValueError) as exc:
            return False, f"PTY diagnostic completed turn is invalid: {exc}"
        if typed_turn.session_id != str(artifact.get("session_id") or ""):
            return False, "PTY diagnostic completed turn belongs to another session"
        if typed_turn.turn_index != runtime_event.turn_index:
            return False, "PTY diagnostic turn and runtime boundary disagree"
        if typed_turn.agent_response != response:
            return False, "PTY diagnostic response and completed turn disagree"
    raw_selection = boundary.get("dynamic_selection")
    if raw_selection is not None:
        if typed_turn is None or not isinstance(raw_selection, Mapping):
            return False, "PTY diagnostic dynamic selection has no completed turn"
        try:
            selection_values = dict(raw_selection)
            selection_values["target_coverage_ids"] = tuple(
                selection_values.get("target_coverage_ids") or ()
            )
            _validate_dynamic_selection(
                typed_turn,
                DynamicTurnSelection(**selection_values),
            )
        except (TypeError, ValueError) as exc:
            return False, f"PTY diagnostic dynamic selection is invalid: {exc}"
    if status == "postcondition_failed":
        postcondition = boundary.get("verified_postcondition")
        if not isinstance(postcondition, Mapping) or postcondition.get("passed") is not False:
            return False, "failed diagnostic has no failed postcondition"
    elif boundary.get("verified_postcondition") is not None:
        return False, "PTY diagnostic has an unexpected postcondition"
    if artifact.get("content_redacted") is not True:
        return False, "PTY diagnostic does not declare redaction"
    unsigned = dict(artifact)
    artifact_hash = str(unsigned.pop("artifact_hash", ""))
    if content_hash(unsigned) != artifact_hash:
        return False, "PTY diagnostic artifact hash mismatch"
    identity = dict(unsigned)
    diagnostic_id = str(identity.pop("diagnostic_id", ""))
    if content_hash(identity) != diagnostic_id:
        return False, "PTY diagnostic id mismatch"
    return True, ""


def _runtime_event_payload(event: RuntimeTurnEvent) -> dict[str, Any]:
    payload = asdict(event)
    payload["action_queue_types"] = list(event.action_queue_types)
    payload["admitted_action_types"] = list(event.admitted_action_types)
    payload["admitted_action_targets"] = [dict(item) for item in event.admitted_action_targets]
    payload["admitted_action_provenance"] = [
        dict(item) for item in event.admitted_action_provenance
    ]
    payload["control_receipts"] = [
        dict(item) for item in event.control_receipts
    ]
    return payload


def _redacted_pty_turn_payload(turn: PtyCliTurnRecord) -> dict[str, Any]:
    previous_response = str(redact(turn.previous_agent_response))
    user_message = str(redact(turn.user_message))
    agent_response = str(redact(turn.agent_response))
    payload = asdict(turn)
    payload.update({
        "previous_agent_response": previous_response,
        "user_message": user_message,
        "agent_response": agent_response,
        "transcript_hash": pty_transcript_hash(
            session_id=turn.session_id,
            turn_index=turn.turn_index,
            previous_agent_response=previous_response,
            user_message=user_message,
            agent_response=agent_response,
        ),
    })
    payload["previous_response_hash"] = content_hash(previous_response)
    payload["user_message_hash"] = content_hash(user_message)
    payload["agent_response_hash"] = content_hash(agent_response)
    return payload


def _redacted_dynamic_selection_payload(selection: DynamicTurnSelection) -> dict[str, Any]:
    payload = redact(asdict(selection))
    payload["target_coverage_ids"] = list(selection.target_coverage_ids)
    return payload


def _verified_postcondition_payload(postcondition: VerifiedPostcondition) -> dict[str, Any]:
    payload = asdict(postcondition)
    payload["observed_coverage_ids"] = list(postcondition.observed_coverage_ids)
    payload["admitted_typed_actions"] = list(postcondition.admitted_typed_actions)
    payload["job_artifacts"] = [dict(item) for item in postcondition.job_artifacts]
    return payload


def _redacted_verified_postcondition_payload(
    postcondition: VerifiedPostcondition,
) -> dict[str, Any]:
    """Redact business content while preserving immutable coverage identity."""

    payload = _verified_postcondition_payload(postcondition)
    payload["state_diff"] = redact(dict(postcondition.state_diff))
    payload["next_question_or_result"] = redact(
        dict(postcondition.next_question_or_result)
    )
    details = dict(postcondition.details)
    declared = details.pop("declared_target_results", None)
    safe_details = redact(details)
    if isinstance(declared, Mapping):
        safe_details["declared_target_results"] = {
            str(coverage_id): redact(dict(result))
            if isinstance(result, Mapping)
            else redact(result)
            for coverage_id, result in declared.items()
        }
    payload["details"] = safe_details
    payload["job_artifacts"] = [redact(dict(item)) for item in postcondition.job_artifacts]
    return payload


def build_pty_cli_evidence_artifact(
    *,
    edge: Mapping[str, Any],
    evidence_class: str,
    revision: Mapping[str, str],
    turn: PtyCliTurnRecord,
    observation: TurnObservation,
    dynamic_selection: DynamicTurnSelection | None = None,
    execution_case: Mapping[str, Any] | None = None,
    seed_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build tamper-evident evidence from a real PTY CLI turn.

    ``real_cli`` proves that the product terminal executed an observed turn.
    ``dynamic_dual_ai`` additionally proves that the selected user message was
    chosen after the previous Agent response by a response-driven Codex actor.
    """

    if evidence_class not in {"real_cli", "dynamic_dual_ai"}:
        raise ValueError("PTY evidence class must be real_cli or dynamic_dual_ai")
    _require_lane(edge, evidence_class)
    runner_type = (
        PTY_DYNAMIC_DUAL_AI_RUNNER
        if evidence_class == "dynamic_dual_ai"
        else PTY_REAL_CLI_RUNNER
    )
    _validate_pty_turn_record(turn)
    selection_payload: dict[str, Any] | None = None
    if evidence_class == "dynamic_dual_ai":
        if dynamic_selection is None:
            raise ValueError("dynamic_dual_ai evidence requires a dynamic selection record")
        _validate_dynamic_selection(turn, dynamic_selection)
        selection_payload = asdict(dynamic_selection)
        selection_payload["target_coverage_ids"] = list(dynamic_selection.target_coverage_ids)
        selection_payload["previous_response_hash"] = content_hash(turn.previous_agent_response)
    elif dynamic_selection is not None:
        raise ValueError("real_cli evidence must not carry a dynamic selection record")
    fixed_real_cli = (
        evidence_class == "real_cli"
        and not bool(((edge.get("evidence") or {}).get("dynamic_dual_ai") or {}).get("required"))
    )
    if fixed_real_cli:
        if not isinstance(execution_case, Mapping) or not isinstance(seed_receipt, Mapping):
            raise ValueError("real_cli evidence requires execution-case and seed provenance")
        _validate_real_cli_provenance(
            edge=edge,
            turn=turn,
            observation=observation,
            execution_case=execution_case,
            seed_receipt=seed_receipt,
        )
    elif evidence_class != "real_cli" and (execution_case is not None or seed_receipt is not None):
        raise ValueError("dynamic evidence must not claim fixed execution-case provenance")

    required_identity = {
        "edge_key": edge.get("edge_key"),
        "contract_hash": edge.get("contract_hash"),
        "runtime_variant_hash": edge.get("contract_variant_hash"),
        "revision.commit": revision.get("commit"),
        "revision.worktree_hash": revision.get("worktree_hash"),
    }
    missing = sorted(
        name for name, value in required_identity.items()
        if not str(value or "").strip()
    )
    if missing:
        raise ValueError(f"missing evidence identity: {', '.join(missing)}")
    _validate_turn_observation(
        observation,
        edge=edge,
        revision=revision,
        turn=turn,
        dynamic_selection=dynamic_selection,
    )

    safe_turn = _redacted_pty_turn(turn)
    safe_selection = (
        _redacted_dynamic_selection(dynamic_selection)
        if dynamic_selection is not None
        else None
    )
    safe_observation = _redacted_turn_observation(observation)
    turn_payload = asdict(safe_turn)
    turn_payload.update({
        "previous_response_hash": content_hash(safe_turn.previous_agent_response),
        "user_message_hash": content_hash(safe_turn.user_message),
        "agent_response_hash": content_hash(safe_turn.agent_response),
        "source_previous_response_hash": content_hash(turn.previous_agent_response),
        "source_user_message_hash": content_hash(turn.user_message),
        "source_agent_response_hash": content_hash(turn.agent_response),
    })
    selection_payload = None
    if safe_selection is not None:
        selection_payload = asdict(safe_selection)
        selection_payload["target_coverage_ids"] = list(safe_selection.target_coverage_ids)
        selection_payload["previous_response_hash"] = content_hash(
            safe_turn.previous_agent_response
        )
    observation_payload = _turn_observation_payload(safe_observation)
    payload: dict[str, Any] = {
        "artifact_type": "pty_cli_turn",
        "schema_version": CLI_ARTIFACT_SCHEMA_VERSION,
        "edge_key": str(edge.get("edge_key") or ""),
        "contract_hash": str(edge.get("contract_hash") or ""),
        "runtime_variant_hash": str(edge.get("contract_variant_hash") or ""),
        "evidence_class": evidence_class,
        "runner_type": runner_type,
        "revision": {
            "commit": str(revision.get("commit") or ""),
            "worktree_hash": str(revision.get("worktree_hash") or ""),
        },
        "turn": turn_payload,
        "turn_hash": content_hash(turn_payload),
        "dynamic_selection": selection_payload,
        "execution_case": dict(execution_case or {}),
        "execution_case_hash": content_hash(dict(execution_case or {})) if execution_case else "",
        "seed_receipt": dict(seed_receipt or {}),
        "seed_receipt_hash": content_hash(dict(seed_receipt or {})) if seed_receipt else "",
        "turn_observation": observation_payload,
        "turn_observation_hash": content_hash(observation_payload),
        "outcome": "passed",
        "exit_status": 0,
        "error": "",
        "content_redacted": True,
        "started_at": _utc_timestamp(),
        "finished_at": _utc_timestamp(),
    }
    payload["evidence_id"] = content_hash(payload)
    payload["artifact_hash"] = content_hash(payload)
    return payload


def _redacted_pty_turn(turn: PtyCliTurnRecord) -> PtyCliTurnRecord:
    previous = str(redact(turn.previous_agent_response))
    user_message = str(redact(turn.user_message))
    response = str(redact(turn.agent_response))
    return replace(
        turn,
        previous_agent_response=previous,
        user_message=user_message,
        agent_response=response,
        transcript_hash=pty_transcript_hash(
            session_id=turn.session_id,
            turn_index=turn.turn_index,
            previous_agent_response=previous,
            user_message=user_message,
            agent_response=response,
        ),
    )


def _redacted_dynamic_selection(
    selection: DynamicTurnSelection,
) -> DynamicTurnSelection:
    return replace(
        selection,
        selected_message=str(redact(selection.selected_message)),
        persona=str(redact(selection.persona)),
        goal=str(redact(selection.goal)),
        rationale=str(redact(selection.rationale)),
    )


def _redacted_turn_observation(observation: TurnObservation) -> TurnObservation:
    payload = _turn_observation_payload(observation)
    payload["prior_agent_response"] = str(redact(observation.prior_agent_response))
    payload["exact_user_turn"] = str(redact(observation.exact_user_turn))
    payload["simulator_decision"] = _redacted_simulator_decision(
        observation.simulator_decision
    )
    payload["continuation_turns"] = [
        asdict(_redacted_pty_turn(turn))
        for turn in observation.continuation_turns
    ]
    payload["continuation_simulator_decisions"] = [
        _redacted_simulator_decision(decision)
        for decision in observation.continuation_simulator_decisions
    ]
    payload["verified_postcondition"] = _redacted_verified_postcondition_payload(
        observation.verified_postcondition
    )
    return _turn_observation_from_payload(payload)


def _redacted_simulator_decision(decision: Mapping[str, Any]) -> dict[str, Any]:
    """Project simulator content separately from coverage identity."""

    payload = {
        str(key): redact(value)
        for key, value in decision.items()
        if str(key) != "target_coverage_ids"
    }
    if "target_coverage_ids" in decision:
        payload["target_coverage_ids"] = [
            str(value) for value in decision.get("target_coverage_ids") or ()
        ]
    return payload


def validate_pty_cli_evidence_artifact(
    artifact: Mapping[str, Any],
    *,
    edge: Mapping[str, Any],
    revision: Mapping[str, str],
) -> tuple[bool, str]:
    """Validate PTY evidence without confusing scripted and dynamic turns."""

    if artifact.get("artifact_type") != "pty_cli_turn":
        return False, "artifact is not PTY CLI evidence"
    if artifact.get("schema_version") != CLI_ARTIFACT_SCHEMA_VERSION:
        return False, "unsupported PTY CLI evidence schema"
    if artifact.get("content_redacted") is not True:
        return False, "PTY CLI evidence does not declare redaction"
    turn_payload = artifact.get("turn")
    if not isinstance(turn_payload, Mapping):
        return False, "PTY evidence has no turn payload"
    for name in (
        "source_previous_response_hash", "source_user_message_hash",
        "source_agent_response_hash",
    ):
        if not _is_sha256(str(turn_payload.get(name) or "")):
            return False, f"PTY source boundary hash is invalid: {name}"
    evidence_class = str(artifact.get("evidence_class") or "")
    lane_error = _lane_error(edge, evidence_class)
    if lane_error:
        return False, lane_error
    expected_runner = {
        "real_cli": PTY_REAL_CLI_RUNNER,
        "dynamic_dual_ai": PTY_DYNAMIC_DUAL_AI_RUNNER,
    }.get(evidence_class)
    if expected_runner is None:
        return False, "PTY evidence class is invalid"
    if str(artifact.get("runner_type") or "") != expected_runner:
        return False, "PTY runner and evidence class disagree"
    if str(artifact.get("edge_key") or "") != str(edge.get("edge_key") or ""):
        return False, "edge key mismatch"
    if str(artifact.get("contract_hash") or "") != str(edge.get("contract_hash") or ""):
        return False, "contract hash mismatch"
    if str(artifact.get("runtime_variant_hash") or "") != str(
        edge.get("contract_variant_hash") or ""
    ):
        return False, "runtime variant hash mismatch"
    if dict(artifact.get("revision") or {}) != dict(revision):
        return False, "repository revision mismatch"

    raw_turn = dict(artifact.get("turn") or {})
    for derived in (
        "previous_response_hash", "user_message_hash", "agent_response_hash",
        "source_previous_response_hash", "source_user_message_hash",
        "source_agent_response_hash",
    ):
        raw_turn.pop(derived, None)
    try:
        turn = PtyCliTurnRecord(**raw_turn)
    except (TypeError, ValueError) as exc:
        return False, f"invalid PTY turn record: {exc}"
    try:
        _validate_pty_turn_record(turn)
    except ValueError as exc:
        return False, str(exc)

    raw_case = artifact.get("execution_case")
    raw_receipt = artifact.get("seed_receipt")
    fixed_real_cli = (
        evidence_class == "real_cli"
        and not bool(((edge.get("evidence") or {}).get("dynamic_dual_ai") or {}).get("required"))
    )
    if fixed_real_cli:
        if not isinstance(raw_case, Mapping) or not isinstance(raw_receipt, Mapping):
            return False, "real CLI artifact has no execution-case seed provenance"
        if content_hash(raw_case) != str(artifact.get("execution_case_hash") or ""):
            return False, "execution-case hash mismatch"
        if content_hash(raw_receipt) != str(artifact.get("seed_receipt_hash") or ""):
            return False, "seed receipt hash mismatch"
        try:
            _validate_real_cli_provenance(
                edge=edge,
                turn=turn,
                observation=_turn_observation_from_payload(
                    dict(artifact.get("turn_observation") or {})
                ),
                execution_case=raw_case,
                seed_receipt=raw_receipt,
            )
        except (TypeError, ValueError) as exc:
            return False, f"invalid real CLI provenance: {exc}"
    elif evidence_class != "real_cli" and (
        raw_case or raw_receipt or artifact.get("execution_case_hash") or artifact.get("seed_receipt_hash")
    ):
        return False, "dynamic artifact carries fixed execution-case provenance"

    turn_payload = dict(artifact.get("turn") or {})
    expected_hashes = {
        "previous_response_hash": content_hash(turn.previous_agent_response),
        "user_message_hash": content_hash(turn.user_message),
        "agent_response_hash": content_hash(turn.agent_response),
    }
    if any(turn_payload.get(name) != value for name, value in expected_hashes.items()):
        return False, "PTY turn content hash mismatch"
    if content_hash(turn_payload) != str(artifact.get("turn_hash") or ""):
        return False, "PTY turn hash mismatch"

    selection_payload = artifact.get("dynamic_selection")
    if evidence_class == "dynamic_dual_ai":
        if not isinstance(selection_payload, Mapping):
            return False, "dynamic evidence has no selection record"
        raw_selection = dict(selection_payload)
        previous_response_hash = raw_selection.pop("previous_response_hash", "")
        raw_selection["target_coverage_ids"] = tuple(raw_selection.get("target_coverage_ids") or ())
        try:
            selection = DynamicTurnSelection(**raw_selection)
            _validate_dynamic_selection(turn, selection)
        except (TypeError, ValueError) as exc:
            return False, f"invalid dynamic selection: {exc}"
        if previous_response_hash != content_hash(turn.previous_agent_response):
            return False, "dynamic selection previous response hash mismatch"
    elif selection_payload is not None:
        return False, "scripted real_cli evidence carries a dynamic selection"

    raw_observation = artifact.get("turn_observation")
    if not isinstance(raw_observation, Mapping):
        return False, "PTY evidence has no turn observation"
    try:
        observation = _turn_observation_from_payload(raw_observation)
        _validate_turn_observation(
            observation,
            edge=edge,
            revision=revision,
            turn=turn,
            dynamic_selection=(selection if evidence_class == "dynamic_dual_ai" else None),
        )
    except (TypeError, ValueError) as exc:
        return False, f"invalid turn observation: {exc}"
    if content_hash(raw_observation) != str(artifact.get("turn_observation_hash") or ""):
        return False, "turn observation hash mismatch"

    outcome = str(artifact.get("outcome") or "")
    try:
        exit_status = int(artifact.get("exit_status"))
    except (TypeError, ValueError):
        return False, "PTY exit status is invalid"
    if outcome != "passed" or exit_status != 0 or str(artifact.get("error") or ""):
        return False, "qualifying PTY evidence must be an observed pass"

    unsigned = dict(artifact)
    artifact_hash = str(unsigned.pop("artifact_hash", ""))
    if content_hash(unsigned) != artifact_hash:
        return False, "artifact hash mismatch"
    evidence_payload = dict(unsigned)
    evidence_id = str(evidence_payload.pop("evidence_id", ""))
    if content_hash(evidence_payload) != evidence_id:
        return False, "evidence id mismatch"
    return True, ""


def _validate_real_cli_provenance(
    *,
    edge: Mapping[str, Any],
    turn: PtyCliTurnRecord,
    observation: TurnObservation,
    execution_case: Mapping[str, Any],
    seed_receipt: Mapping[str, Any],
) -> None:
    from tests.agent_live.harness_contract_scenarios import (
        canonical_question_contract,
        canonical_scenario_state,
    )
    from tests.agent_live.reviewed_execution_cases import reviewed_execution_case

    resolved = reviewed_execution_case(edge)
    if resolved is None:
        raise ValueError("ledger edge has no reviewed execution case")
    scenario, authoritative_case = resolved
    if dict(execution_case) != authoritative_case.descriptor:
        raise ValueError("execution case is not the authoritative reviewed descriptor")
    if authoritative_case.case_id not in set(edge.get("execution_case_ids") or ()):
        raise ValueError("execution case is not bound to the ledger edge")
    if authoritative_case.descriptor_hash != str(edge.get("execution_case_hash") or ""):
        raise ValueError("ledger execution-case hash is stale")
    receipt = dict(seed_receipt)
    receipt_hash = str(receipt.pop("receipt_hash", ""))
    if content_hash(receipt) != receipt_hash:
        raise ValueError("seed receipt self-hash mismatch")
    required = {
        "scenario_id": scenario.scenario_id,
        "scenario_state_fingerprint": scenario.state_fingerprint,
        "seed_state_hash": content_hash(canonical_scenario_state(scenario.seed_state or {})),
        "session_id": turn.session_id,
        "session_purpose": str(observation.runtime_events[0].session_purpose),
        "pending_question_id": str(scenario.question.get("id") or ""),
        "pending_contract_hash": content_hash(canonical_question_contract(scenario.question)),
    }
    mismatches = {
        key: {"expected": value, "actual": receipt.get(key)}
        for key, value in required.items()
        if receipt.get(key) != value
    }
    if mismatches:
        raise ValueError(f"seed receipt does not match reviewed scenario: {mismatches}")
    for key in ("projected_state_hash", "checkpoint_sha256"):
        value = str(receipt.get(key) or "")
        if not _is_sha256(value):
            raise ValueError(f"seed receipt has invalid {key}")
    if str(receipt.get("checkpoint_path") or "").strip() == "":
        raise ValueError("seed receipt has no checkpoint path")
    if not authoritative_case.admits_recorded_input(turn.user_message):
        raise ValueError("PTY input does not match the reviewed execution case")


def _validate_pty_turn_record(turn: PtyCliTurnRecord) -> None:
    required = {
        "session_id": turn.session_id,
        "previous_agent_response": turn.previous_agent_response,
        "user_message": turn.user_message,
        "agent_response": turn.agent_response,
        "provider": turn.provider,
        "model": turn.model,
        "before_fingerprint": turn.before_fingerprint,
        "after_fingerprint": turn.after_fingerprint,
        "transcript_hash": turn.transcript_hash,
    }
    missing = sorted(name for name, value in required.items() if not str(value or "").strip())
    if missing:
        raise ValueError(f"PTY turn is missing: {', '.join(missing)}")
    if turn.turn_index < 1:
        raise ValueError("PTY turn index must be positive")
    for name in ("before_fingerprint", "after_fingerprint", "transcript_hash"):
        if not _is_sha256(str(getattr(turn, name))):
            raise ValueError(f"PTY turn fingerprint is invalid: {name}")
    expected_transcript_hash = pty_transcript_hash(
        session_id=turn.session_id,
        turn_index=turn.turn_index,
        previous_agent_response=turn.previous_agent_response,
        user_message=turn.user_message,
        agent_response=turn.agent_response,
    )
    if turn.transcript_hash != expected_transcript_hash:
        raise ValueError("PTY transcript hash does not match the recorded turn")
    if not (
        turn.previous_response_received_at_ns
        <= turn.user_message_submitted_at_ns
        <= turn.agent_response_received_at_ns
    ):
        raise ValueError("PTY turn timestamps are not ordered")


def _validate_dynamic_selection(
    turn: PtyCliTurnRecord,
    selection: DynamicTurnSelection,
) -> None:
    if selection.selection_mode != "response_driven":
        raise ValueError("fixed or scripted prompts cannot be labeled dynamic_dual_ai")
    if selection.simulator.casefold() != "codex":
        raise ValueError("dynamic_dual_ai requires the Codex user simulator")
    if selection.selected_message != turn.user_message:
        raise ValueError("dynamic selected message differs from the PTY user message")
    if not all((selection.persona.strip(), selection.goal.strip(), selection.rationale.strip())):
        raise ValueError("dynamic selection requires persona, goal, and rationale")
    if not selection.target_coverage_ids or any(
        not str(item or "").strip() for item in selection.target_coverage_ids
    ):
        raise ValueError("dynamic selection requires target coverage ids")
    if not (
        turn.previous_response_received_at_ns
        <= selection.selected_at_ns
        <= turn.user_message_submitted_at_ns
    ):
        raise ValueError("dynamic message was not selected after the previous response")


def _turn_observation_payload(observation: TurnObservation) -> dict[str, Any]:
    payload = asdict(observation)
    payload["runtime_events"] = [
        {**asdict(event), "action_queue_types": list(event.action_queue_types)}
        for event in observation.runtime_events
    ]
    postcondition = asdict(observation.verified_postcondition)
    postcondition["observed_coverage_ids"] = list(
        observation.verified_postcondition.observed_coverage_ids
    )
    postcondition["admitted_typed_actions"] = list(
        observation.verified_postcondition.admitted_typed_actions
    )
    postcondition["job_artifacts"] = [
        dict(item) for item in observation.verified_postcondition.job_artifacts
    ]
    payload["verified_postcondition"] = postcondition
    return payload


def _turn_observation_from_payload(payload: Mapping[str, Any]) -> TurnObservation:
    values = dict(payload)
    values["runtime_events"] = tuple(
        RuntimeTurnEvent(
            **{
                **dict(item),
                "action_queue_types": tuple(dict(item).get("action_queue_types") or ()),
                "admitted_action_types": tuple(
                    dict(item).get("admitted_action_types") or ()
                ),
            }
        )
        for item in values.get("runtime_events") or ()
    )
    values["continuation_turns"] = tuple(
        PtyCliTurnRecord(**dict(item))
        for item in values.get("continuation_turns") or ()
    )
    values["continuation_simulator_decisions"] = tuple(
        dict(item)
        for item in values.get("continuation_simulator_decisions") or ()
    )
    raw_postcondition = dict(values.get("verified_postcondition") or {})
    raw_postcondition["observed_coverage_ids"] = tuple(
        raw_postcondition.get("observed_coverage_ids") or ()
    )
    raw_postcondition["admitted_typed_actions"] = tuple(
        raw_postcondition.get("admitted_typed_actions") or ()
    )
    raw_postcondition["job_artifacts"] = tuple(
        dict(item) for item in raw_postcondition.get("job_artifacts") or ()
    )
    values["verified_postcondition"] = VerifiedPostcondition(**raw_postcondition)
    return TurnObservation(**values)


def _validate_turn_observation(
    observation: TurnObservation,
    *,
    edge: Mapping[str, Any],
    revision: Mapping[str, str],
    turn: PtyCliTurnRecord,
    dynamic_selection: DynamicTurnSelection | None,
) -> None:
    if observation.schema_version != TURN_OBSERVATION_SCHEMA_VERSION:
        raise ValueError("unsupported turn observation schema")
    if dict(observation.revision) != dict(revision):
        raise ValueError("turn observation repository revision mismatch")
    expected_identity = (
        str(edge.get("edge_key") or ""),
        str(edge.get("contract_hash") or ""),
        str(edge.get("contract_variant_hash") or ""),
    )
    observed_identity = (
        observation.target_edge_key,
        observation.target_contract_hash,
        observation.target_variant_hash,
    )
    if observed_identity != expected_identity:
        raise ValueError("turn observation target is not the authoritative ledger edge")
    if observation.prior_agent_response != turn.previous_agent_response:
        raise ValueError("turn observation prior response mismatch")
    if observation.exact_user_turn != turn.user_message:
        raise ValueError("turn observation user turn mismatch")
    if observation.provider != turn.provider or observation.model != turn.model:
        raise ValueError("turn observation provider identity mismatch")
    if not observation.provider.strip() or not observation.model.strip():
        raise ValueError("turn observation provider identity is missing")
    if observation.before_turn_index < 0:
        raise ValueError("turn observation before turn index is invalid")
    if len(observation.runtime_events) < 2:
        raise ValueError("turn observation requires a baseline and at least one committed event")
    baseline, committed, *continuation_events = observation.runtime_events
    expected_after_index = observation.before_turn_index + len(observation.runtime_events) - 1
    if observation.after_turn_index != expected_after_index:
        raise ValueError("turn observation indexes do not cover its complete event chain")
    if turn.turn_index != observation.before_turn_index + 1:
        raise ValueError("root PTY turn does not immediately follow the baseline")
    if len(observation.continuation_turns) != len(continuation_events):
        raise ValueError("continuation PTY turns do not match continuation runtime events")
    for event in observation.runtime_events:
        _validate_runtime_event(event)
        if dict(event.revision) != dict(observation.revision):
            raise ValueError("runtime event revision mismatch within linked journey")
        if event.thread_id != turn.session_id:
            raise ValueError("runtime event thread does not match the PTY session")
        if event.session_purpose != baseline.session_purpose:
            raise ValueError("runtime event session purpose changed")
    for previous, current in zip(
        observation.runtime_events,
        observation.runtime_events[1:],
    ):
        if current.event_type not in {"turn_committed", "turn_recovered"}:
            raise ValueError("linked runtime event is not a committed turn")
        if current.turn_index != previous.turn_index + 1:
            raise ValueError("linked runtime event turn indexes are not contiguous")
        if current.before_fingerprint != previous.after_fingerprint:
            raise ValueError("linked runtime event fingerprint chain is stale")
    if baseline.turn_index != observation.before_turn_index:
        raise ValueError("baseline runtime event turn index is stale")
    terminal_event = observation.runtime_events[-1]
    if terminal_event.turn_index != observation.after_turn_index:
        raise ValueError("terminal runtime event turn index is stale")
    if observation.before_state_fingerprint != committed.before_fingerprint:
        raise ValueError("turn observation before fingerprint mismatch")
    if observation.after_state_fingerprint != terminal_event.after_fingerprint:
        raise ValueError("turn observation after fingerprint mismatch")
    if turn.before_fingerprint != committed.before_fingerprint:
        raise ValueError("PTY before fingerprint was not observed from the committed event")
    if turn.after_fingerprint != committed.after_fingerprint:
        raise ValueError("PTY after fingerprint was not observed from the committed event")
    for continuation_turn, continuation_event in zip(
        observation.continuation_turns,
        continuation_events,
    ):
        _validate_pty_turn_record(continuation_turn)
        if continuation_turn.session_id != turn.session_id:
            raise ValueError("continuation PTY turn changed session identity")
        if continuation_turn.provider != turn.provider or continuation_turn.model != turn.model:
            raise ValueError("continuation PTY turn changed provider identity")
        if continuation_turn.turn_index != continuation_event.turn_index:
            raise ValueError("continuation PTY and runtime turn indexes disagree")
        if continuation_turn.before_fingerprint != continuation_event.before_fingerprint:
            raise ValueError("continuation PTY before fingerprint mismatch")
        if continuation_turn.after_fingerprint != continuation_event.after_fingerprint:
            raise ValueError("continuation PTY after fingerprint mismatch")
    if dict(observation.pending_contract) != dict(baseline.pending_contract):
        raise ValueError("pending contract is not bound to the baseline runtime event")
    edge_type = str(edge.get("edge_type") or "")
    if edge_type != "action_transition" and not baseline.pending_question_id:
        raise ValueError("baseline runtime event has no pending contract")
    elif edge_type != "action_transition":
        expected_question = str(edge.get("question_id") or "")
        if baseline.pending_question_id != expected_question:
            raise ValueError("baseline pending question does not match the target edge")
        from tests.agent_live.generate_harness_coverage_ledger import contract_variant_hash

        if contract_variant_hash(baseline.pending_contract) != str(
            edge.get("contract_hash") or ""
        ):
            raise ValueError("baseline pending contract hash does not match the target edge")
    if dynamic_selection is not None:
        if dict(observation.simulator_decision) != {
            **asdict(dynamic_selection),
            "target_coverage_ids": list(dynamic_selection.target_coverage_ids),
        }:
            raise ValueError("turn observation simulator decision mismatch")
        if observation.target_edge_key not in dynamic_selection.target_coverage_ids:
            raise ValueError("dynamic selection did not target the authoritative edge key")
        if len(observation.continuation_simulator_decisions) != len(
            observation.continuation_turns
        ):
            raise ValueError("continuation simulator decisions do not match continuation turns")
        for continuation_turn, raw_selection in zip(
            observation.continuation_turns,
            observation.continuation_simulator_decisions,
        ):
            selection_values = dict(raw_selection)
            selection_values["target_coverage_ids"] = tuple(
                selection_values.get("target_coverage_ids") or ()
            )
            continuation_selection = DynamicTurnSelection(**selection_values)
            _validate_dynamic_selection(continuation_turn, continuation_selection)
            if observation.target_edge_key not in continuation_selection.target_coverage_ids:
                raise ValueError("continuation selection lost the authoritative edge target")
    elif observation.simulator_decision:
        raise ValueError("real CLI observation must not claim a simulator decision")
    elif observation.continuation_simulator_decisions:
        raise ValueError("real CLI observation must not claim continuation simulator decisions")

    postcondition = observation.verified_postcondition
    if not postcondition.verifier_id.strip():
        raise ValueError("turn observation postcondition verifier is missing")
    if not postcondition.passed:
        raise ValueError("turn observation postcondition did not pass")
    if observation.target_edge_key not in postcondition.observed_coverage_ids:
        raise ValueError("verified postcondition did not observe the target edge")
    rejection_expected = (
        str(edge.get("edge_type") or "") == "manual_input"
        and edge.get("expected_admitted") is False
    )
    if not postcondition.admitted_typed_actions and not rejection_expected:
        raise ValueError("verified postcondition has no admitted typed actions")
    if not postcondition.state_diff:
        raise ValueError("verified postcondition has no state diff")
    if not postcondition.next_question_or_result:
        raise ValueError("verified postcondition has no next question or result")
    if not postcondition.details:
        raise ValueError("verified postcondition has no assertion details")
    if edge.get("real_execution_required") and not postcondition.job_artifacts:
        raise ValueError("real execution edge has no verified job artifact")
    for artifact in postcondition.job_artifacts:
        if not all(str(artifact.get(name) or "").strip() for name in ("path", "sha256")):
            raise ValueError("verified job artifact identity is incomplete")
        if not _is_sha256(str(artifact.get("sha256") or "")):
            raise ValueError("verified job artifact hash is invalid")


def _validate_runtime_event(event: RuntimeTurnEvent) -> None:
    if event.schema_version not in {2, 3}:
        raise ValueError("unsupported runtime turn event schema")
    if event.event_type not in {"startup_snapshot", "turn_committed", "turn_recovered"}:
        raise ValueError("runtime event type is not observable")
    if not event.thread_id.strip() or not event.session_purpose.strip():
        raise ValueError("runtime event identity is missing")
    if event.turn_index < 0:
        raise ValueError("runtime event turn index is invalid")
    for value in (event.before_fingerprint, event.after_fingerprint):
        if not _is_sha256(value):
            raise ValueError("runtime event fingerprint is invalid")
    if not event.revision.get("commit") or not _is_sha256(
        str(event.revision.get("worktree_hash") or "")
    ):
        raise ValueError("runtime event revision is invalid")
    if event.schema_version == 3:
        turn_receipt = dict(event.turn_receipt_summary or {})
        if turn_receipt:
            if (
                not str(turn_receipt.get("turn_id") or "")
                or not _is_sha256(str(turn_receipt.get("input_hash") or ""))
            ):
                raise ValueError("runtime turn receipt identity is invalid")
            admitted_ids = [
                str(item)
                for item in turn_receipt.get("admitted_action_ids") or ()
                if str(item)
            ]
            provenance_ids = [
                str(item.get("action_id") or "")
                for item in event.admitted_action_provenance
                if str(item.get("action_id") or "")
            ]
            if provenance_ids != admitted_ids:
                raise ValueError(
                    "runtime action provenance differs from turn admission order"
                )
            execution_order = [
                str(item)
                for item in turn_receipt.get("execution_order") or ()
                if str(item)
            ]
            if any(item not in admitted_ids for item in execution_order):
                raise ValueError("runtime turn receipt executed an unadmitted action")
        for action in event.admitted_action_provenance:
            if not isinstance(action, Mapping) or not str(action.get("type") or ""):
                raise ValueError("runtime action provenance is invalid")
            for field_name in ("arguments_hash", "source_hash"):
                if not _is_sha256(str(action.get(field_name) or "")):
                    raise ValueError("runtime action provenance hash is invalid")
            value_hashes = action.get("argument_value_hashes")
            if (
                not isinstance(value_hashes, Mapping)
                or set(value_hashes)
                != set(action.get("argument_names") or ())
                or any(
                    not str(key)
                    or not _is_sha256(str(value))
                    for key, value in value_hashes.items()
                )
            ):
                raise ValueError(
                    "runtime action argument-value provenance is invalid"
                )
        transition = dict(event.pending_transition or {})
        for field_name in ("before_hash", "after_hash"):
            if not _is_sha256(str(transition.get(field_name) or "")):
                raise ValueError("runtime pending transition hash is invalid")
        manifest = dict(event.render_manifest or {})
        if any(
            not _is_sha256(str(item))
            for item in manifest.get("fragment_hashes") or ()
        ):
            raise ValueError("runtime render manifest hash is invalid")
        for receipt in event.control_receipts:
            if (
                not isinstance(receipt, Mapping)
                or not str(receipt.get("receipt_type") or "")
                or not _is_sha256(str(receipt.get("receipt_id") or ""))
            ):
                raise ValueError("runtime control receipt is invalid")
            unsigned = dict(receipt)
            receipt_id = str(unsigned.pop("receipt_id"))
            if content_hash(unsigned) != receipt_id:
                raise ValueError("runtime control receipt hash is stale")
            valid, reason = validate_persisted_domain_control_receipt(
                receipt,
                turn_index=event.turn_index,
            )
            if not valid:
                raise ValueError(
                    f"runtime domain control receipt is invalid: {reason}"
                )
        for path, hashes in event.material_state_diff_hashes.items():
            if (
                not str(path)
                or not isinstance(hashes, Mapping)
                or any(
                    value and not _is_sha256(str(value))
                    for value in (
                        hashes.get("before"),
                        hashes.get("after"),
                    )
                )
            ):
                raise ValueError("runtime material state diff is invalid")
    if str(event.pending_contract.get("id") or "") != event.pending_question_id:
        raise ValueError("runtime event pending contract identity is inconsistent")


def validate_runtime_turn_event(event: RuntimeTurnEvent) -> None:
    """Validate one complete runtime event at an external evidence boundary."""

    _validate_runtime_event(event)


def verify_runtime_postcondition(
    edge: Mapping[str, Any],
    baseline: RuntimeTurnEvent,
    committed: RuntimeTurnEvent,
    turn: PtyCliTurnRecord,
    *,
    continuation_events: Iterable[RuntimeTurnEvent] = (),
) -> VerifiedPostcondition:
    """Verify an immediate turn or a linked deferred-transition journey.

    This verifier is fixed by the evidence module. Callers cannot replace it
    with a callback that simply declares the scheduled edge successful. A
    prerequisite-deferred first turn never qualifies by itself.
    """

    errors: list[str] = []
    linked_events = tuple(continuation_events)
    if committed.before_fingerprint != baseline.after_fingerprint:
        errors.append("runtime fingerprint chain did not advance from the baseline")
    if committed.turn_index != baseline.turn_index + 1:
        errors.append("runtime turn index did not advance exactly once")
    edge_type = str(edge.get("edge_type") or "")
    admitted = tuple(item for item in committed.admitted_action_types if item)
    rejection_expected = edge_type == "manual_input" and edge.get("expected_admitted") is False
    if not admitted and not rejection_expected:
        errors.append("runtime emitted no admitted typed action")
    scheduled_action = str(edge.get("action_type") or "")
    navigation_transition_outcome = ""
    expected_admitted_actions = {
        str(item)
        for item in edge.get("expected_admitted_action_types") or ()
        if str(item)
    }
    if edge_type == "question_option" and expected_admitted_actions:
        declared_by_pending = {
            str(item)
            for item in baseline.pending_contract.get("accepted_action_types") or ()
            if str(item)
        }
        undeclared_equivalents = sorted(expected_admitted_actions - declared_by_pending)
        if undeclared_equivalents:
            errors.append(
                "option-owner equivalence was not declared by the pending contract: "
                + ", ".join(undeclared_equivalents)
            )
        if not expected_admitted_actions.intersection(admitted):
            errors.append(
                "no declared option-owner action was admitted: expected one of "
                f"{sorted(expected_admitted_actions)}"
            )
    elif (
        edge_type in {"action_transition", "question_option"}
        and scheduled_action
        and scheduled_action not in admitted
    ):
        errors.append(f"scheduled action was not admitted: {scheduled_action}")
    if edge_type == "action_transition" and scheduled_action == "change_group":
        navigation_targets = {
            str(item.get("group") or "")
            for item in committed.admitted_action_targets
            if str(item.get("type") or "") == "change_group"
            and str(item.get("group") or "")
        }
        visible_group = str((committed.next_result or {}).get("group") or committed.active_group or "")
        if not navigation_targets:
            errors.append("change_group emitted no admitted destination")
        expected_target = str((edge.get("expected_postcondition") or {}).get("target_group") or "")
        if expected_target and expected_target not in navigation_targets:
            errors.append(
                "change_group admitted the wrong destination: "
                f"expected={expected_target}, admitted={sorted(navigation_targets)}"
            )
        else:
            navigation_contract = dict(
                (edge.get("expected_postcondition") or {}).get("navigation_transition") or {}
            )
            admitted_target = next(iter(navigation_targets)) if len(navigation_targets) == 1 else ""
            immediate_group = str(
                navigation_contract.get("immediate_group")
                or expected_target
                or admitted_target
                or ""
            )
            deferred_groups = {
                str(item)
                for item in navigation_contract.get("prerequisite_deferred_groups") or ()
                if str(item)
            }
            deferred_target_observed = bool(
                expected_target
                and _postcondition_value_matches(
                    committed.after_value_hashes,
                    "control.deferred_group",
                    expected_target,
                )
            )
            if visible_group == immediate_group and not deferred_target_observed:
                navigation_transition_outcome = "immediate"
            elif deferred_target_observed and visible_group in deferred_groups:
                navigation_transition_outcome = "prerequisite_deferred"
            else:
                errors.append(
                    "change_group destination was not observed as either immediate or "
                    "prerequisite-deferred: "
                    f"visible={visible_group or '<none>'}, target={expected_target or '<none>'}, "
                    f"prerequisites={sorted(deferred_groups)}"
                )
    if edge_type == "manual_input":
        expected_question = str(edge.get("question_id") or "")
        if expected_question and baseline.pending_question_id != expected_question:
            errors.append(f"manual input baseline question mismatch: {expected_question}")
        accepted = {
            str(item).strip()
            for item in baseline.pending_contract.get("accepted_action_types") or []
            if str(item).strip()
        }
        interrupts_pending = bool(edge.get("interrupts_pending_contract"))
        if (
            not rejection_expected
            and not interrupts_pending
            and not accepted.intersection(admitted)
        ):
            errors.append("manual input admitted no action declared by the pending contract")
        if rejection_expected and admitted and not accepted.intersection(admitted):
            errors.append("rejected manual input admitted an unrelated action")

    expected = dict(edge.get("expected_postcondition") or {})
    expected_paths: list[str] = []
    structured_review_keys: list[str] = []
    deferred_option_action = ""
    deferred_option_prerequisite = ""
    if edge_type == "question_option":
        deferred_contract = {
            str(action_type): tuple(str(item) for item in prerequisites if str(item))
            for action_type, prerequisites in (
                edge.get("prerequisite_deferred_actions") or {}
            ).items()
        }
        visible_group = str(
            (committed.next_result or {}).get("group") or committed.active_group or ""
        )
        for action_type in admitted:
            prerequisites = deferred_contract.get(action_type, ())
            if (
                action_type in committed.action_queue_types
                and visible_group in prerequisites
            ):
                deferred_option_action = action_type
                deferred_option_prerequisite = visible_group
                if committed.pending_contract.get("resume_action_queue") is not True:
                    errors.append(
                        "prerequisite-deferred option did not preserve the queue barrier"
                    )
                break
    if scheduled_action == "propose_config_values" and turn is not None:
        from agent.harness.domains.environment import extract_structured_input_candidates

        candidates = extract_structured_input_candidates(turn.user_message) or {}
        structured_paths = {
            **{
                str(key): f"inferred_config.pending_review.config_values.{key}"
                for key in (candidates.get("config_values") or {})
            },
            **{
                str(key): f"inferred_config.pending_review.unmapped_values.{key}"
                for key in (candidates.get("unmapped_values") or {})
            },
        }
        for key, path in structured_paths.items():
            structured_review_keys.append(key)
            if not _path_value_hashes(committed.after_value_hashes, path):
                errors.append(f"structured review silently lost source key: {key}")
    if edge_type == "question_option":
        for path, value in expected.items():
            expected_paths.append(str(path))
            if deferred_option_action:
                if _path_value_hashes(committed.after_value_hashes, str(path)) != _path_value_hashes(
                    baseline.after_value_hashes, str(path)
                ):
                    errors.append(
                        f"prerequisite-deferred option mutated its destination early: {path}"
                    )
            elif not _postcondition_value_matches(
                committed.after_value_hashes,
                str(path),
                value,
            ):
                errors.append(f"expected postcondition was not observed: {path}")
        if not deferred_option_action:
            _verify_runtime_state_relations(
                tuple(edge.get("expected_state_relations") or ()),
                baseline,
                committed,
                errors,
            )
    elif edge_type == "manual_input":
        path = str(expected.get("path") or "").strip()
        if path:
            expected_paths.append(path)
            after_hashes = _path_value_hashes(committed.after_value_hashes, path)
            before_hashes = _path_value_hashes(baseline.after_value_hashes, path)
            if rejection_expected:
                if "rejection_value" in expected and not _postcondition_value_matches(
                    committed.after_value_hashes,
                    path,
                    expected["rejection_value"],
                ):
                    errors.append(
                        f"rejected manual input did not record declared evidence: {path}"
                    )
                elif "rejection_value" not in expected and after_hashes != before_hashes:
                    errors.append(f"rejected manual input changed the destination field: {path}")
                if committed.pending_question_id != baseline.pending_question_id:
                    errors.append("rejected manual input did not preserve the pending question")
                from tests.agent_live.generate_harness_coverage_ledger import contract_variant_hash

                if contract_variant_hash(committed.pending_contract) != contract_variant_hash(
                    baseline.pending_contract
                ):
                    errors.append("rejected manual input changed the pending contract")
            elif not after_hashes:
                errors.append(f"manual-input postcondition was not observed: {path}")
            elif after_hashes == before_hashes:
                errors.append(f"manual-input postcondition did not change: {path}")
            elif "value" in expected and not _postcondition_value_matches(
                committed.after_value_hashes,
                path,
                expected["value"],
            ):
                errors.append(f"manual-input postcondition value mismatch: {path}")
        expected_next_ids = {
            str(item).strip()
            for item in expected.get("next_question_ids") or []
            if str(item).strip()
        }
        if (
            not rejection_expected
            and expected_next_ids
            and committed.pending_question_id not in expected_next_ids
        ):
            errors.append(
                "manual input reached an unexpected next question: "
                f"{committed.pending_question_id or '<none>'}; "
                f"expected one of {sorted(expected_next_ids)}"
            )

    next_result = dict(committed.next_result or {})
    if not next_result:
        errors.append("runtime emitted no next question or terminal result")
    state_diff = dict(committed.state_diff_hashes or {})
    if not state_diff:
        errors.append("runtime emitted no state transition")

    deferred_transition = bool(
        navigation_transition_outcome == "prerequisite_deferred"
        or deferred_option_action
    )
    terminal_event = committed
    journey_status = "not_required"
    pre_journey_errors = tuple(errors)
    linkage_errors: tuple[str, ...] = ()
    if deferred_transition:
        journey_status = "awaiting_terminal_postcondition"
        if not linked_events:
            errors.append(
                "prerequisite-deferred transition requires linked multi-turn "
                "terminal postcondition evidence"
            )
        else:
            previous = committed
            linkage_error_start = len(errors)
            for index, event in enumerate(linked_events, start=1):
                if event.thread_id != committed.thread_id:
                    errors.append(f"linked journey event {index} changed thread identity")
                if event.session_purpose != committed.session_purpose:
                    errors.append(f"linked journey event {index} changed session purpose")
                if dict(event.revision) != dict(committed.revision):
                    errors.append(f"linked journey event {index} changed repository revision")
                if event.turn_index != previous.turn_index + 1:
                    errors.append(f"linked journey event {index} is not the next turn")
                if event.before_fingerprint != previous.after_fingerprint:
                    errors.append(
                        f"linked journey event {index} broke the fingerprint chain"
                    )
                previous = event
            linkage_errors = tuple(errors[linkage_error_start:])
            terminal_event = linked_events[-1]
            terminal_visible_group = str(
                (terminal_event.next_result or {}).get("group")
                or terminal_event.active_group
                or ""
            )
            if navigation_transition_outcome == "prerequisite_deferred":
                target_group = str(expected.get("target_group") or "")
                if terminal_visible_group != target_group:
                    errors.append(
                        "linked navigation journey did not reach its terminal group: "
                        f"expected={target_group or '<none>'}, "
                        f"visible={terminal_visible_group or '<none>'}"
                    )
                if _path_value_hashes(
                    terminal_event.after_value_hashes,
                    "control.deferred_group",
                ):
                    errors.append("linked navigation journey did not clear its deferred target")
            if deferred_option_action:
                for path, value in expected.items():
                    if not _postcondition_value_matches(
                        terminal_event.after_value_hashes,
                        str(path),
                        value,
                    ):
                        errors.append(
                            f"linked option journey did not reach terminal postcondition: {path}"
                        )
                if deferred_option_action in terminal_event.action_queue_types:
                    errors.append("linked option journey did not drain its deferred action")
                _verify_runtime_state_relations(
                    tuple(edge.get("expected_state_relations") or ()),
                    baseline,
                    terminal_event,
                    errors,
                )
            if not terminal_event.next_result:
                errors.append("linked journey emitted no terminal question or result")
            if not errors:
                journey_status = "terminal_postcondition_observed"

    continuation_eligible = bool(
        deferred_transition
        and journey_status == "awaiting_terminal_postcondition"
        and not pre_journey_errors
        and not linkage_errors
    )

    edge_key = str(edge.get("edge_key") or "")
    observed_actions = tuple(dict.fromkeys(
        action_type
        for event in (committed, *linked_events)
        for action_type in event.admitted_action_types
        if action_type
    ))
    combined_state_diff = {
        path: hashes
        for event in (committed, *linked_events)
        for path, hashes in event.state_diff_hashes.items()
    }
    details = {
        "baseline_question_id": baseline.pending_question_id,
        "committed_question_id": committed.pending_question_id,
        "expected_postcondition_paths": sorted(expected_paths),
        "expected_admitted": edge.get("expected_admitted"),
        "transition_outcome": (
            navigation_transition_outcome
            or ("prerequisite_deferred" if deferred_option_action else "immediate")
        ),
        "deferred_action": deferred_option_action,
        "deferred_prerequisite": deferred_option_prerequisite,
        "journey_status": journey_status,
        "continuation_eligible": continuation_eligible,
        "pre_journey_errors": list(pre_journey_errors),
        "linkage_errors": list(linkage_errors),
        "linked_turn_count": len(linked_events),
        "structured_review_keys": sorted(structured_review_keys),
        "rejection_observed": bool(rejection_expected and not errors),
        "errors": errors,
    }
    return VerifiedPostcondition(
        verifier_id="anychain.runtime-transition-proof.v1",
        passed=not errors,
        observed_coverage_ids=(edge_key,) if not errors and edge_key else (),
        admitted_typed_actions=observed_actions,
        state_diff=combined_state_diff,
        next_question_or_result=dict(terminal_event.next_result or {}),
        details=details,
    )


def _path_value_hashes(values: Mapping[str, str], path: str) -> dict[str, str]:
    """Return one scalar hash or all leaf hashes owned by a structured path."""

    prefix = f"{path}."
    return {
        candidate[len(prefix):] if candidate.startswith(prefix) else "": value
        for candidate, value in values.items()
        if candidate == path or candidate.startswith(prefix)
    }


def _postcondition_value_matches(
    observed: Mapping[str, str],
    path: str,
    expected: Any,
) -> bool:
    expected_hashes = _leaf_hashes_for_expected(expected)
    return _path_value_hashes(observed, path) == expected_hashes


def _leaf_hashes_for_expected(value: Any, prefix: tuple[str, ...] = ()) -> dict[str, str]:
    if isinstance(value, Mapping) and value:
        output: dict[str, str] = {}
        for key in sorted(value, key=str):
            output.update(_leaf_hashes_for_expected(value[key], (*prefix, str(key))))
        return output
    return {".".join(prefix): content_hash(value)}


def _verify_runtime_state_relations(
    relations: tuple[Mapping[str, Any], ...],
    baseline: RuntimeTurnEvent,
    committed: RuntimeTurnEvent,
    errors: list[str],
) -> None:
    for relation in relations:
        kind = str(relation.get("kind") or "")
        if kind == "path_absent_after":
            path = str(relation.get("path") or "")
            observed = any(
                candidate == path or candidate.startswith(f"{path}.")
                for candidate in committed.after_value_hashes
            )
            if not path or observed:
                errors.append(f"state relation path remains present: {path or '<empty>'}")
            continue
        if kind == "path_equals_before_path":
            after_path = str(relation.get("after_path") or "")
            before_path = str(relation.get("before_path") or "")
            if (
                not before_path
                or not after_path
                or before_path not in baseline.after_value_hashes
                or after_path not in committed.after_value_hashes
                or baseline.after_value_hashes[before_path] != committed.after_value_hashes[after_path]
            ):
                errors.append(f"state relation was not observed: {after_path} <- {before_path}")
            continue
        if kind == "prefix_equals_before_prefix":
            after_prefix = str(relation.get("after_prefix") or "")
            before_prefix = str(relation.get("before_prefix") or "")
            ignored = {str(item) for item in relation.get("ignored_suffixes") or ()}
            before_values = {
                path[len(before_prefix):]: value
                for path, value in baseline.after_value_hashes.items()
                if path == before_prefix or path.startswith(f"{before_prefix}.")
                if path.rsplit(".", 1)[-1] not in ignored
            }
            after_values = {
                path[len(after_prefix):]: value
                for path, value in committed.after_value_hashes.items()
                if path == after_prefix or path.startswith(f"{after_prefix}.")
                if path.rsplit(".", 1)[-1] not in ignored
            }
            if not before_values or not after_values or before_values != after_values:
                errors.append(f"state prefix relation was not observed: {after_prefix} <- {before_prefix}")
            continue
        errors.append(f"unsupported state relation: {kind or '<empty>'}")


def validate_evidence_artifact(
    artifact: Mapping[str, Any],
    *,
    edge: Mapping[str, Any],
    revision: Mapping[str, str],
) -> tuple[bool, str]:
    required = {
        "schema_version", "evidence_id", "artifact_hash", "edge_key",
        "contract_hash", "runtime_variant_hash", "evidence_class",
        "scenario_id", "runner_type", "revision", "input_hash",
        "seed_state_hash", "before_state_hash", "after_state_hash",
        "turn_evidence", "turn_evidence_hash", "event_trace", "event_trace_hash",
        "exit_status", "outcome", "error", "started_at", "finished_at",
        "content_redacted", "source_input_hash", "source_boundary_hashes",
    }
    missing = sorted(required - set(artifact))
    if missing:
        return False, f"missing artifact fields: {', '.join(missing)}"
    try:
        schema_version = int(artifact.get("schema_version"))
        exit_status = int(artifact.get("exit_status"))
    except (TypeError, ValueError):
        return False, "schema version or exit status is not an integer"
    if schema_version != ARTIFACT_SCHEMA_VERSION:
        return False, "unsupported artifact schema"
    if str(artifact.get("runner_type") or "") != COMPILED_GRAPH_RUNNER:
        return False, "artifact did not use the compiled LangGraph helper"
    for field in (
        "edge_key", "contract_hash", "runtime_variant_hash", "evidence_class",
        "scenario_id", "runner_type", "input_hash", "seed_state_hash",
        "before_state_hash", "after_state_hash", "turn_evidence_hash",
        "event_trace_hash", "evidence_id", "artifact_hash", "started_at", "finished_at",
    ):
        if not str(artifact.get(field) or "").strip():
            return False, f"artifact field is empty: {field}"
    if str(artifact.get("evidence_class")) != "deterministic":
        return False, "compiled-graph artifact must use deterministic evidence"
    if artifact.get("content_redacted") is not True:
        return False, "compiled-graph artifact does not declare redaction"
    source_boundary_hashes = artifact.get("source_boundary_hashes")
    if not isinstance(source_boundary_hashes, Mapping) or any(
        not _is_sha256(str(source_boundary_hashes.get(name) or ""))
        for name in (
            "before_hash", "input_hash", "question_hash", "after_hash",
            "state_diff_hash", "response_hash", "next_question_hash",
        )
    ):
        return False, "compiled-graph source boundary hashes are invalid"
    lane_error = _lane_error(edge, "deterministic")
    if lane_error:
        return False, lane_error
    artifact_revision = artifact.get("revision")
    if not isinstance(artifact_revision, Mapping):
        return False, "artifact revision is not an object"
    if not str(artifact_revision.get("commit") or "").strip():
        return False, "artifact revision commit is empty"
    if not _is_sha256(str(artifact_revision.get("worktree_hash") or "")):
        return False, "artifact worktree hash is invalid"
    for field in (
        "input_hash", "seed_state_hash", "before_state_hash", "after_state_hash",
        "turn_evidence_hash", "event_trace_hash", "evidence_id", "artifact_hash",
        "source_input_hash",
    ):
        if not _is_sha256(str(artifact.get(field) or "")):
            return False, f"artifact hash is invalid: {field}"
    if str(artifact.get("edge_key") or "") != str(edge.get("edge_key") or ""):
        return False, "edge key mismatch"
    if str(artifact.get("contract_hash") or "") != str(edge.get("contract_hash") or ""):
        return False, "contract hash mismatch"
    if str(artifact.get("runtime_variant_hash") or "") != str(edge.get("contract_variant_hash") or ""):
        return False, "runtime variant hash mismatch"
    if dict(artifact_revision) != dict(revision):
        return False, "repository revision mismatch"
    outcome = str(artifact.get("outcome") or "")
    if outcome not in {"passed", "failed"}:
        return False, "runner outcome is invalid"
    if outcome == "passed" and exit_status != 0:
        return False, "passed artifact has nonzero exit status"
    if outcome == "failed" and exit_status == 0:
        return False, "failed artifact has zero exit status"
    if outcome == "failed" and not str(artifact.get("error") or "").strip():
        return False, "failed artifact has no error"

    raw_events = artifact.get("event_trace")
    if not isinstance(raw_events, list) or not all(isinstance(event, Mapping) for event in raw_events):
        return False, "event trace is not a list of objects"
    events = list(raw_events)
    if any(
        str(event.get("event_type") or "") in {"edge_executed", "edge_execution_failed"}
        for event in events
    ):
        return False, "runner-declared edge outcome event is forbidden"
    if content_hash(events) != str(artifact.get("event_trace_hash") or ""):
        return False, "event trace hash mismatch"
    returned = _event(events, "compiled_graph_turn_returned", edge)
    raised = _event(events, "compiled_graph_turn_raised", edge)
    if outcome == "passed" and returned is None:
        return False, "compiled graph return event is missing"
    if outcome == "failed" and returned is None and raised is None:
        return False, "compiled graph observation is missing"

    turn = artifact.get("turn_evidence")
    if not isinstance(turn, Mapping):
        return False, "turn evidence is not an object"
    required_turn = {
        "compiled_graph_helper", "seed", "before", "input", "admitted_action",
        "runtime_actions",
        "state_diff", "after", "question", "next_question", "response", "turn_context",
    }
    missing_turn = sorted(required_turn - set(turn))
    if missing_turn:
        return False, f"missing turn evidence fields: {', '.join(missing_turn)}"
    if str(turn.get("compiled_graph_helper") or "") != COMPILED_GRAPH_RUNNER:
        return False, "turn evidence helper mismatch"
    seed = turn.get("seed")
    before = turn.get("before")
    after = turn.get("after")
    question = turn.get("question")
    if not all(isinstance(value, Mapping) for value in (seed, before, after, question)):
        return False, "turn state or question evidence is not an object"
    if content_hash(turn) != str(artifact.get("turn_evidence_hash") or ""):
        return False, "turn evidence hash mismatch"
    if content_hash(seed) != str(artifact.get("seed_state_hash") or ""):
        return False, "seed state hash mismatch"
    if content_hash(before) != str(artifact.get("before_state_hash") or ""):
        return False, "before state hash mismatch"
    if content_hash(after) != str(artifact.get("after_state_hash") or ""):
        return False, "after state hash mismatch"
    if content_hash(turn.get("input")) != str(artifact.get("input_hash") or ""):
        return False, "input hash mismatch"
    if before.get("last_user_input") != turn.get("input"):
        return False, "recorded input differs from invocation state"
    if dict(before.get("pending_question") or {}) != dict(question):
        return False, "recorded question differs from invocation state"
    if dict(after.get("pending_question") or {}) != dict(turn.get("next_question") or {}):
        return False, "recorded next question differs from graph result"
    if list(after.get("visible_response") or []) != list(turn.get("response") or []):
        return False, "recorded response differs from graph result"
    if dict(after.get("turn_context") or {}) != dict(turn.get("turn_context") or {}):
        return False, "recorded turn context differs from graph result"
    expected_diff = state_diff_between(before, after)
    if dict(turn.get("state_diff") or {}) != expected_diff:
        return False, "recorded state diff is not reproducible"
    raw_expected_admitted = edge.get("expected_admitted")
    expected_admitted = True if raw_expected_admitted is None else bool(raw_expected_admitted)
    expected_action = _action_admission(
        edge,
        question,
        turn.get("input"),
        admitted=expected_admitted,
    )
    if not expected_action:
        return False, "compiled turn has no derivable action admission evidence"
    if dict(turn.get("admitted_action") or {}) != expected_action:
        return False, "recorded admitted action is not derived from the question contract"
    runtime_actions = turn.get("runtime_actions")
    if not isinstance(runtime_actions, Mapping):
        return False, "runtime action evidence is not an object"
    expected_runtime_actions = {
        "proposed": list(after.get("proposed_actions") or []),
        "queued": list(after.get("action_queue") or []),
        "completed": list(after.get("completed_actions") or []),
        "applied_action_ids": list(after.get("applied_action_ids") or []),
    }
    if dict(runtime_actions) != expected_runtime_actions:
        return False, "runtime action evidence differs from graph result"
    if returned is not None:
        reason = _validate_return_event(returned, artifact, turn)
        if reason:
            return False, reason
        context = dict(turn.get("turn_context") or {})
        # A reset action intentionally returns a fresh state and therefore
        # clears turn_context. For all other turns that retain it, verify the
        # graph's normalized input and exact pending snapshot.
        if context and expected_admitted:
            if context.get("kind") != "pending":
                return False, "compiled turn did not admit the pending-question path"
            if context.get("text") != str(turn.get("input") or "").strip():
                return False, "turn context input mismatch"
            if dict(context.get("pending_snapshot") or {}) != dict(question):
                return False, "turn context question snapshot mismatch"
        if not expected_admitted:
            if str((after.get("pending_question") or {}).get("id") or "") != str(question.get("id") or ""):
                return False, "rejected input did not preserve the pending question"
            protected_path = str((edge.get("expected_postcondition") or {}).get("path") or "")
            if protected_path and _read_path(before, protected_path) != _read_path(after, protected_path):
                return False, "rejected input mutated the protected field"
            if not any(str(item).strip() for item in after.get("visible_response") or []):
                return False, "rejected input has no explicit response"

    unsigned = dict(artifact)
    artifact_hash = str(unsigned.pop("artifact_hash", ""))
    if content_hash(unsigned) != artifact_hash:
        return False, "artifact hash mismatch"
    evidence_id_payload = dict(unsigned)
    evidence_id = str(evidence_id_payload.pop("evidence_id", ""))
    if content_hash(evidence_id_payload) != evidence_id:
        return False, "evidence id mismatch"
    return True, ""


def load_valid_evidence_reference(
    reference: str,
    *,
    edge: Mapping[str, Any],
    revision: Mapping[str, str],
) -> tuple[dict[str, Any] | None, str]:
    path = Path(reference)
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"cannot load evidence artifact: {exc}"
    if not isinstance(artifact, dict):
        return None, "evidence artifact is not an object"
    if artifact.get("artifact_type") == "pty_cli_turn":
        valid, reason = validate_pty_cli_evidence_artifact(
            artifact,
            edge=edge,
            revision=revision,
        )
    elif artifact.get("artifact_type") == "real_execution":
        valid, reason = validate_real_execution_evidence_artifact(
            artifact,
            edge=edge,
            revision=revision,
        )
    elif artifact.get("artifact_type") == "pty_diagnostic":
        return None, "PTY diagnostic artifacts never qualify as coverage evidence"
    else:
        valid, reason = validate_evidence_artifact(artifact, edge=edge, revision=revision)
    return (artifact, "") if valid else (None, reason)


def build_real_execution_evidence_artifact(
    *,
    edge: Mapping[str, Any],
    revision: Mapping[str, str],
    scenario_id: str,
    operation_kind: str,
    request: Mapping[str, Any],
    result: Mapping[str, Any],
    job_id: str,
    job_artifacts: Iterable[Mapping[str, Any]],
    log_artifacts: Iterable[Mapping[str, Any]],
    outcome: str = "passed",
    exit_status: int = 0,
    error: str = "",
    started_at: str | None = None,
    finished_at: str | None = None,
) -> dict[str, Any]:
    """Build the independent proof contract for a real side-effecting job.

    This producer does not run a benchmark.  A Linux/Docker integration runner
    must supply the observed request, result, job identity, logs, and hashed
    artifacts.  That separation prevents a PTY acknowledgement from being
    mistaken for execution evidence.
    """

    _require_lane(edge, "real_execution")
    if operation_kind not in {"preflight_smoke", "final_benchmark", "sync_observe"}:
        raise ValueError(f"unsupported real execution operation: {operation_kind}")
    scenario = scenario_by_id(scenario_id)
    if scenario.action_type != str(edge.get("action_type") or ""):
        raise ValueError(
            f"real execution scenario disagrees with {edge.get('action_type')}: {scenario_id}"
        )
    if scenario.operation_kind != operation_kind:
        raise ValueError(
            f"real execution operation disagrees with {scenario_id}: {operation_kind}"
        )
    required_identity = {
        "edge_key": edge.get("edge_key"),
        "contract_hash": edge.get("contract_hash"),
        "runtime_variant_hash": edge.get("contract_variant_hash"),
        "revision.commit": revision.get("commit"),
        "revision.worktree_hash": revision.get("worktree_hash"),
    }
    missing = [name for name, value in required_identity.items() if not str(value or "").strip()]
    if missing or not _is_sha256(str(revision.get("worktree_hash") or "")):
        raise ValueError(f"invalid real execution identity: {', '.join(missing) or 'worktree hash'}")
    if not str(job_id or "").strip():
        raise ValueError("real execution evidence requires a job id")
    if outcome not in {"passed", "observed-fail"}:
        raise ValueError(f"invalid real execution outcome: {outcome}")
    if outcome == "passed" and (int(exit_status) != 0 or str(error or "").strip()):
        raise ValueError("passed real execution evidence cannot contain a failure")
    if outcome == "observed-fail" and int(exit_status) == 0:
        raise ValueError("observed-fail real execution evidence requires non-zero exit status")
    if outcome == "observed-fail" and not str(error or "").strip():
        raise ValueError("observed-fail real execution evidence requires an error")
    if not request or not result:
        raise ValueError("real execution evidence requires request and result objects")
    raw_artifacts = [dict(item) for item in job_artifacts]
    raw_logs = [dict(item) for item in log_artifacts]
    if not raw_artifacts or not raw_logs:
        raise ValueError("real execution evidence requires hashed job and log artifacts")
    observed_job = dict(result.get("observed_job") or {})
    run_dir = Path(str(observed_job.get("run_dir") or ""))
    if not run_dir.is_dir() or run_dir.is_symlink() or run_dir.name != str(job_id):
        raise ValueError("real execution evidence requires one fresh non-symlink run_dir")
    for item in (*raw_artifacts, *raw_logs):
        _validate_hashed_artifact_identity(
            item,
            expected_job_id=str(job_id),
            expected_run_dir=run_dir,
        )
    safe_request = redact(deepcopy(dict(request)))
    safe_result = redact(deepcopy(dict(result)))
    artifacts = redact(raw_artifacts)
    logs = redact(raw_logs)
    payload: dict[str, Any] = {
        "artifact_type": "real_execution",
        "schema_version": REAL_EXECUTION_ARTIFACT_SCHEMA_VERSION,
        "edge_key": str(edge.get("edge_key") or ""),
        "contract_hash": str(edge.get("contract_hash") or ""),
        "runtime_variant_hash": str(edge.get("contract_variant_hash") or ""),
        "evidence_class": "real_execution",
        "runner_type": REAL_EXECUTION_RUNNER,
        "revision": dict(revision),
        "scenario_id": scenario_id,
        "operation_kind": operation_kind,
        "request": safe_request,
        "request_hash": content_hash(safe_request),
        "source_request_hash": content_hash(request),
        "result": safe_result,
        "result_hash": content_hash(safe_result),
        "source_result_hash": content_hash(result),
        "job_id": str(job_id),
        "job_artifacts": artifacts,
        "job_artifacts_hash": content_hash(artifacts),
        "log_artifacts": logs,
        "log_artifacts_hash": content_hash(logs),
        "outcome": outcome,
        "exit_status": int(exit_status),
        "error": str(redact(str(error or ""))),
        "content_redacted": True,
        "started_at": started_at or _utc_timestamp(),
        "finished_at": finished_at or _utc_timestamp(),
    }
    payload["evidence_id"] = content_hash(payload)
    payload["artifact_hash"] = content_hash(payload)
    return payload


def validate_real_execution_evidence_artifact(
    artifact: Mapping[str, Any],
    *,
    edge: Mapping[str, Any],
    revision: Mapping[str, str],
    allow_observed_failure: bool = False,
) -> tuple[bool, str]:
    lane_error = _lane_error(edge, "real_execution")
    if lane_error:
        return False, lane_error
    if artifact.get("artifact_type") != "real_execution":
        return False, "artifact is not real execution evidence"
    if artifact.get("schema_version") != REAL_EXECUTION_ARTIFACT_SCHEMA_VERSION:
        return False, "unsupported real execution evidence schema"
    if artifact.get("content_redacted") is not True:
        return False, "real execution evidence does not declare redaction"
    for name in ("source_request_hash", "source_result_hash"):
        if not _is_sha256(str(artifact.get(name) or "")):
            return False, f"real execution source hash is invalid: {name}"
    if artifact.get("evidence_class") != "real_execution":
        return False, "real execution evidence class is invalid"
    if artifact.get("runner_type") != REAL_EXECUTION_RUNNER:
        return False, "real execution runner identity is invalid"
    identity = (
        str(artifact.get("edge_key") or ""),
        str(artifact.get("contract_hash") or ""),
        str(artifact.get("runtime_variant_hash") or ""),
    )
    expected = (
        str(edge.get("edge_key") or ""),
        str(edge.get("contract_hash") or ""),
        str(edge.get("contract_variant_hash") or ""),
    )
    if identity != expected:
        return False, "real execution edge identity mismatch"
    if dict(artifact.get("revision") or {}) != dict(revision):
        return False, "repository revision mismatch"
    if artifact.get("operation_kind") not in {"preflight_smoke", "final_benchmark", "sync_observe"}:
        return False, "real execution operation is invalid"
    try:
        scenario = scenario_by_id(str(artifact.get("scenario_id") or ""))
    except ValueError as exc:
        return False, str(exc)
    if scenario.action_type != str(edge.get("action_type") or ""):
        return False, "real execution scenario disagrees with action"
    if artifact.get("operation_kind") != scenario.operation_kind:
        return False, "real execution operation disagrees with scenario"
    if not str(artifact.get("job_id") or "").strip():
        return False, "real execution job id is missing"
    outcome = str(artifact.get("outcome") or "")
    exit_status = artifact.get("exit_status")
    error = str(artifact.get("error") or "")
    if outcome == "passed":
        if exit_status != 0 or error:
            return False, "passed real execution evidence contains a failure"
    elif outcome == "observed-fail":
        if not allow_observed_failure:
            return False, "observed-fail evidence does not qualify as a passing execution edge"
        if not isinstance(exit_status, int) or exit_status == 0 or not error:
            return False, "observed-fail evidence has no observed failure"
    else:
        return False, "real execution outcome is invalid"
    try:
        started_at = datetime.fromisoformat(
            str(artifact.get("started_at") or "").replace("Z", "+00:00")
        )
        finished_at = datetime.fromisoformat(
            str(artifact.get("finished_at") or "").replace("Z", "+00:00")
        )
    except ValueError:
        return False, "real execution observation timestamp is invalid"
    if (
        started_at.tzinfo is None
        or finished_at.tzinfo is None
        or finished_at < started_at
    ):
        return False, "real execution observation time range is invalid"
    request = artifact.get("request")
    result = artifact.get("result")
    if not isinstance(request, Mapping) or not request:
        return False, "real execution request is missing"
    if not isinstance(result, Mapping) or not result:
        return False, "real execution result is missing"
    if content_hash(request) != artifact.get("request_hash"):
        return False, "real execution request hash mismatch"
    if content_hash(result) != artifact.get("result_hash"):
        return False, "real execution result hash mismatch"
    observed_job = dict(result.get("observed_job") or {})
    run_dir = Path(str(observed_job.get("run_dir") or ""))
    if not run_dir.is_dir() or run_dir.is_symlink():
        return False, "real execution run_dir is missing or symlinked"
    observed_status = str(observed_job.get("status") or "")
    observed_exit_status = observed_job.get("exit_code")
    observed_error = str(observed_job.get("error") or "")
    expected_outcome = (
        "passed"
        if observed_status == "completed" and observed_exit_status == 0
        else "observed-fail"
    )
    expected_error = "" if expected_outcome == "passed" else observed_error
    if (
        not isinstance(observed_exit_status, int)
        or artifact.get("outcome") != expected_outcome
        or artifact.get("exit_status") != observed_exit_status
        or str(artifact.get("error") or "") != expected_error
    ):
        return False, "real execution outcome is not derived from persisted job state"
    if expected_outcome == "observed-fail" and (
        observed_exit_status == 0 or not observed_error
    ):
        return False, "persisted observed failure is incomplete"
    for field, hash_field in (
        ("job_artifacts", "job_artifacts_hash"),
        ("log_artifacts", "log_artifacts_hash"),
    ):
        items = artifact.get(field)
        if not isinstance(items, list) or not items:
            return False, f"{field} is missing"
        try:
            for item in items:
                _validate_hashed_artifact_identity(
                    item,
                    expected_job_id=str(artifact.get("job_id") or ""),
                    expected_run_dir=run_dir,
                )
        except ValueError as exc:
            return False, str(exc)
        if content_hash(items) != artifact.get(hash_field):
            return False, f"{field} hash mismatch"
    scenario_error = _real_execution_scenario_error(artifact, scenario)
    if scenario_error:
        return False, scenario_error
    unsigned = dict(artifact)
    artifact_hash = str(unsigned.pop("artifact_hash", ""))
    if content_hash(unsigned) != artifact_hash:
        return False, "artifact hash mismatch"
    evidence_payload = dict(unsigned)
    evidence_id = str(evidence_payload.pop("evidence_id", ""))
    if content_hash(evidence_payload) != evidence_id:
        return False, "evidence id mismatch"
    return True, ""


def _real_execution_scenario_error(
    artifact: Mapping[str, Any],
    scenario: Any,
) -> str:
    request = dict(artifact.get("request") or {})
    if str(request.get("scenario_id") or "") != scenario.scenario_id:
        return "real execution request scenario mismatch"
    if str(request.get("service_operation") or "") != scenario.operation:
        return "real execution request operation mismatch"
    approved_plan_file = Path(str(request.get("approved_plan_file") or ""))
    try:
        approved_plan_sha256 = hashlib.sha256(approved_plan_file.read_bytes()).hexdigest()
    except OSError as exc:
        return f"approved plan cannot be rehashed: {exc}"
    if approved_plan_sha256 != str(request.get("approved_plan_sha256") or ""):
        return "approved plan hash mismatch"
    if dict(request.get("approved_plan_revision") or {}) != dict(artifact.get("revision") or {}):
        return "approved plan revision binding mismatch"
    try:
        approved_plan = json.loads(approved_plan_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return f"approved plan is invalid: {exc}"
    envelope_file = Path(str(request.get("admission_envelope_file") or ""))
    try:
        envelope_sha256 = hashlib.sha256(envelope_file.read_bytes()).hexdigest()
        envelope = json.loads(envelope_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return f"G5 admission envelope is invalid: {exc}"
    if envelope_sha256 != str(request.get("admission_envelope_sha256") or ""):
        return "G5 admission envelope hash mismatch"
    envelope_error = _g5_admission_envelope_error(
        envelope=envelope,
        request=request,
        scenario=scenario,
        artifact=artifact,
        approved_plan=approved_plan,
    )
    if envelope_error:
        return envelope_error
    result = dict(artifact.get("result") or {})
    observed_job = dict(result.get("observed_job") or {})
    artifacts = dict(observed_job.get("artifacts") or {})
    plan_paths = [
        Path(str(item.get("path") or ""))
        for item in artifact.get("job_artifacts") or ()
        if Path(str(item.get("path") or "")).name == "plan.json"
    ]
    if len(plan_paths) != 1:
        return "real execution evidence must bind one job plan"
    job_paths = [
        Path(str(item.get("path") or ""))
        for item in artifact.get("job_artifacts") or ()
        if Path(str(item.get("path") or "")).name == "job.json"
    ]
    if len(job_paths) != 1:
        return "real execution evidence must bind one persisted job"
    try:
        plan = json.loads(plan_paths[0].read_text(encoding="utf-8"))
        persisted_job = redact(
            json.loads(job_paths[0].read_text(encoding="utf-8"))
        )
    except (OSError, json.JSONDecodeError) as exc:
        return f"real execution job plan or persisted job is invalid: {exc}"
    if workflow_type_from_plan(plan) != scenario.workflow_type:
        return "real execution workflow mismatch"
    provenance = dict(plan.get("execution_provenance") or {})
    if str(provenance.get("scenario_id") or "") != scenario.scenario_id:
        return "real execution provenance scenario mismatch"
    if str(provenance.get("operation") or "") != scenario.operation:
        return "real execution provenance operation mismatch"
    if str(provenance.get("approved_plan_file") or "") != str(approved_plan_file.resolve()):
        return "job plan approved-plan path mismatch"
    if str(provenance.get("approved_plan_sha256") or "") != approved_plan_sha256:
        return "job plan approved-plan hash mismatch"
    if str(provenance.get("job_id") or "") != str(artifact.get("job_id") or ""):
        return "job plan provenance job mismatch"
    if str(provenance.get("job_plan_file") or "") != str(plan_paths[0].resolve()):
        return "job plan provenance path mismatch"
    if "g5_admission_provenance" in plan:
        return "G5 admission metadata leaked into the product job plan"
    try:
        validate_execution_plan_projection(
            approved_plan,
            plan,
            approved_plan_file=approved_plan_file,
            operation=scenario.operation,
            scenario_id=scenario.scenario_id,
            job_id=str(artifact.get("job_id") or ""),
            job_plan_file=plan_paths[0],
        )
    except (OSError, ValueError) as exc:
        return f"job plan projection is invalid: {exc}"
    observed_job = dict(result.get("observed_job") or {})
    if str(observed_job.get("job_id") or "") != str(artifact.get("job_id") or ""):
        return "observed job identity mismatch"
    for field in ("job_id", "status", "exit_code", "error", "created_at"):
        observed_value = observed_job.get(field)
        persisted_value = persisted_job.get(field)
        if field == "error":
            observed_value = str(observed_value or "")
            persisted_value = str(persisted_value or "")
        if observed_value != persisted_value:
            return f"observed job disagrees with persisted job: {field}"
    jobs_dir = Path(str(request.get("jobs_dir") or "")).resolve()
    run_dir = Path(str(observed_job.get("run_dir") or "")).resolve()
    if run_dir.parent != jobs_dir or run_dir.name != str(artifact.get("job_id") or ""):
        return "observed job is outside the admitted jobs root"
    try:
        owner_roots = derive_artifact_owner_roots(observed_job)
    except (OSError, ValueError) as exc:
        return f"real execution artifact owner roots are invalid: {exc}"
    try:
        runtime_projection = validate_runtime_env_projection(observed_job)
    except (OSError, ValueError) as exc:
        return f"real execution runtime.env projection is invalid: {exc}"
    if request.get("runtime_env_projection") != runtime_projection:
        return "real execution runtime.env projection receipt mismatch"
    try:
        admitted_at = datetime.fromisoformat(
            str(request.get("jobs_root_admitted_at") or "").replace("Z", "+00:00")
        )
        created_at = datetime.fromisoformat(
            str(observed_job.get("created_at") or "").replace("Z", "+00:00")
        )
        submitted_at = datetime.fromisoformat(
            str(artifact.get("started_at") or "").replace("Z", "+00:00")
        )
    except ValueError:
        return "real execution freshness timestamp is invalid"
    if admitted_at.tzinfo is None or created_at.tzinfo is None or created_at < admitted_at:
        return "observed job predates the admitted jobs root"
    if submitted_at.tzinfo is None or created_at < submitted_at:
        return "observed job predates its execution submission"
    endpoint_error = _real_execution_endpoint_error(
        request=request,
        plan=approved_plan,
        scenario=scenario,
        artifact=artifact,
        envelope=envelope,
    )
    if endpoint_error:
        return endpoint_error
    runtime_error = _real_execution_runtime_attestation_error(
        request=request,
        plan=approved_plan,
        scenario=scenario,
        artifact=artifact,
        envelope=envelope,
    )
    if runtime_error:
        return runtime_error
    command = [str(item) for item in (plan.get("execution") or {}).get("command") or ()]
    for token in scenario.required_command_tokens:
        if token not in command:
            return f"real execution command is missing required token: {token}"
    for token in scenario.forbidden_command_tokens:
        if token in command:
            return f"real execution command contains forbidden token: {token}"
    if scenario.operation_kind == "sync_observe":
        try:
            duration_index = command.index("--duration")
            duration_seconds = int(command[duration_index + 1])
        except (ValueError, IndexError):
            return "bounded sync-observe command has no valid duration"
        if duration_seconds <= 0:
            return "bounded sync-observe duration must be positive"
        execution_env = dict((plan.get("execution") or {}).get("environment") or {})
        if str(plan.get("rpc_mode") or "") != "sync_observe":
            return "sync-observe plan has an RPC workload mode"
        if plan.get("use_fake_node") is not False:
            return "sync-observe plan is not bound to a real-node source"
        if str(execution_env.get("LOCAL_RPC_URL") or ""):
            return "sync-observe plan contains an RPC benchmark endpoint"
        if not str(execution_env.get("SYNC_OBSERVE_RPC_URL") or ""):
            return "sync-observe plan has no observation endpoint"
        if any(
            str(execution_env.get(name) or "")
            for name in (
                "SYNC_OBSERVE_INITIAL_QPS",
                "SYNC_OBSERVE_MAX_QPS",
                "SYNC_OBSERVE_QPS_STEP",
            )
        ):
            return "sync-observe plan contains a QPS profile"
    if str(artifact.get("outcome") or "") == "observed-fail":
        return ""
    manifest_error = _required_artifact_manifest_error(
        artifact=artifact,
        request=request,
        observed_artifacts=artifacts,
        scenario=scenario,
    )
    if manifest_error:
        return manifest_error
    for name in scenario.required_artifacts:
        raw = str(artifacts.get(name) or "")
        if not raw or not Path(raw).is_file():
            return f"real execution required artifact is missing: {name}"
    for name in scenario.forbidden_artifacts:
        if str(artifacts.get(name) or ""):
            return f"real execution contains forbidden artifact: {name}"
    content_error = _real_execution_content_error(
        artifacts=artifacts,
        plan=plan,
        scenario=scenario,
    )
    if content_error:
        return content_error
    return ""


def _real_execution_content_error(
    *,
    artifacts: Mapping[str, Any],
    plan: Mapping[str, Any],
    scenario: Any,
) -> str:
    summary_error = _summary_artifact_error(
        Path(str(artifacts.get("summary_json") or "")),
        plan=plan,
        sync_observe=scenario.operation_kind == "sync_observe",
    )
    if summary_error:
        return summary_error
    report_names = (
        ("html_report_en", "html_report_zh")
        if scenario.operation_kind == "sync_observe"
        else ("html_report",)
    )
    for name in report_names:
        report_error = _html_report_error(
            Path(str(artifacts.get(name) or "")),
            label=name,
        )
        if report_error:
            return report_error
    if scenario.operation_kind == "sync_observe":
        return _sync_observe_content_error(artifacts)
    return _rpc_benchmark_content_error(artifacts, plan=plan)


def _summary_artifact_error(
    path: Path,
    *,
    plan: Mapping[str, Any],
    sync_observe: bool,
) -> str:
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return f"real execution summary JSON is invalid: {exc}"
    if not isinstance(summary, Mapping) or not str(summary.get("run_id") or ""):
        return "real execution summary has no run identity"
    expected_mode = (
        "sync_observe"
        if sync_observe
        else str(plan.get("benchmark_mode") or "").strip().lower()
    )
    if str(summary.get("benchmark_mode") or "").strip().lower() != expected_mode:
        return "real execution summary benchmark mode mismatch"
    try:
        start_epoch = _observation_epoch(summary.get("start_time"))
        end_epoch = _observation_epoch(summary.get("end_time"))
    except ValueError as exc:
        return f"real execution summary time range is invalid: {exc}"
    if end_epoch < start_epoch:
        return "real execution summary time range is reversed"
    parameters = dict(summary.get("test_parameters") or {})
    parameter_names = {
        "initial_qps",
        "max_qps",
        "qps_step",
        "duration_per_level",
    }
    if set(parameters) != parameter_names or any(
        not isinstance(parameters.get(name), int) for name in parameter_names
    ):
        return "real execution summary test parameters are invalid"
    max_successful_qps = summary.get("max_successful_qps")
    if not isinstance(max_successful_qps, int):
        return "real execution summary successful QPS is invalid"
    if sync_observe:
        if max_successful_qps != 0 or any(parameters.values()):
            return "sync-observe summary contains an RPC workload"
    elif (
        max_successful_qps <= 0
        or parameters["initial_qps"] <= 0
        or parameters["max_qps"] < parameters["initial_qps"]
        or parameters["qps_step"] <= 0
        or parameters["duration_per_level"] <= 0
    ):
        return "RPC benchmark summary has no successful workload"
    return ""


def _html_report_error(path: Path, *, label: str) -> str:
    try:
        raw = path.read_bytes()
        document = raw.decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return f"{label} is not a valid UTF-8 HTML report: {exc}"
    lowered = document.lower()
    if (
        len(raw) < 256
        or "<html" not in lowered
        or "</html>" not in lowered
        or "<title" not in lowered
        or "report" not in lowered
    ):
        return f"{label} has no complete report document"
    return ""


def _rpc_benchmark_content_error(
    artifacts: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
) -> str:
    summary_path = Path(str(artifacts.get("summary_json") or ""))
    performance_path = Path(str(artifacts.get("performance_csv") or ""))
    try:
        with performance_path.open(newline="", encoding="utf-8") as handle:
            performance_rows = list(csv.DictReader(handle))
    except (OSError, csv.Error) as exc:
        return f"RPC performance CSV is invalid: {exc}"
    required_performance_columns = {
        "timestamp",
        "current_qps",
        "rpc_latency_ms",
        "qps_data_available",
        "cpu_usage",
        "mem_usage",
    }
    if not performance_rows:
        return "RPC performance CSV has no observed rows"
    if missing := sorted(required_performance_columns - set(performance_rows[0])):
        return "RPC performance CSV is missing columns: " + ", ".join(missing)
    has_workload_observation = False
    for row in performance_rows:
        try:
            current_qps = float(str(row.get("current_qps") or "0"))
            latency = float(str(row.get("rpc_latency_ms") or "0"))
        except ValueError:
            return "RPC performance CSV contains a non-numeric workload observation"
        if current_qps < 0 or latency < 0:
            return "RPC performance CSV contains a negative workload observation"
        if (
            current_qps > 0
            and str(row.get("qps_data_available") or "").strip().lower()
            in {"true", "1"}
        ):
            has_workload_observation = True
    if not has_workload_observation:
        return "RPC performance CSV has no available positive-QPS observation"
    range_error = _observation_range_error(
        summary_path,
        performance_rows,
        label="RPC performance CSV",
    )
    if range_error:
        return range_error

    proxy_path = Path(str(artifacts.get("proxy_method_csv") or ""))
    try:
        with proxy_path.open(newline="", encoding="utf-8") as handle:
            proxy_rows = list(csv.DictReader(handle))
    except (OSError, csv.Error) as exc:
        return f"RPC proxy method CSV is invalid: {exc}"
    proxy_columns = {"timestamp_ns", "method_name", "status_code", "latency_ms"}
    if not proxy_rows or not proxy_columns.issubset(proxy_rows[0]):
        return "RPC proxy method CSV has no typed method observations"
    for row in proxy_rows:
        try:
            status_code = int(str(row.get("status_code") or ""))
            latency = float(str(row.get("latency_ms") or ""))
            timestamp_ns = int(str(row.get("timestamp_ns") or ""))
        except ValueError:
            return "RPC proxy method CSV contains an invalid observation"
        if (
            not str(row.get("method_name") or "").strip()
            or not 100 <= status_code <= 599
            or latency < 0
            or timestamp_ns <= 0
        ):
            return "RPC proxy method CSV contains an invalid observation"

    vegeta_path = Path(str(artifacts.get("vegeta_json") or ""))
    try:
        vegeta = json.loads(vegeta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return f"Vegeta result JSON is invalid: {exc}"
    if not isinstance(vegeta, Mapping):
        return "Vegeta result is not a JSON object"
    requests = vegeta.get("requests")
    success = vegeta.get("success")
    status_codes = dict(vegeta.get("status_codes") or {})
    if (
        not isinstance(requests, int)
        or requests <= 0
        or not isinstance(success, (int, float))
        or not 0 <= float(success) <= 1
        or not isinstance(vegeta.get("throughput"), (int, float))
        or float(vegeta["throughput"]) <= 0
    ):
        return "Vegeta result has no valid request statistics"
    try:
        observed_status_total = sum(int(value) for value in status_codes.values())
    except (TypeError, ValueError):
        return "Vegeta result status counts are invalid"
    if observed_status_total != requests or not any(
        str(code).startswith("2") and int(count) > 0
        for code, count in status_codes.items()
    ):
        return "Vegeta result status counts do not prove successful requests"
    expected_methods, workload_error = _approved_rpc_methods(plan)
    if workload_error:
        return workload_error
    workload_rows = [
        row
        for row in proxy_rows
        if str(row.get("method_name") or "").strip() in expected_methods
    ]
    if len(workload_rows) != requests:
        return "RPC proxy workload count does not match Vegeta requests"
    observed_method_names = {
        str(row.get("method_name") or "").strip() for row in workload_rows
    }
    if observed_method_names != expected_methods:
        return "RPC proxy workload methods do not match the approved plan"
    proxy_status_counts: dict[str, int] = {}
    for row in workload_rows:
        status = str(int(str(row.get("status_code") or "")))
        proxy_status_counts[status] = proxy_status_counts.get(status, 0) + 1
    normalized_vegeta_status = {
        str(int(str(code))): int(count) for code, count in status_codes.items()
    }
    if proxy_status_counts != normalized_vegeta_status:
        return "RPC proxy workload status counts do not match Vegeta results"
    try:
        summary_start, summary_end = _summary_time_range(summary_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return f"RPC summary time range cannot be loaded: {exc}"
    for row in workload_rows:
        observed = int(str(row.get("timestamp_ns") or "")) / 1_000_000_000
        if not summary_start <= observed <= summary_end:
            return "RPC proxy workload observation is outside the summary time range"
    return ""


def _sync_observe_content_error(artifacts: Mapping[str, Any]) -> str:
    summary_path = Path(str(artifacts.get("summary_json") or ""))
    csv_path = Path(str(artifacts.get("performance_csv") or ""))
    try:
        with csv_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error) as exc:
        return f"sync-observe performance CSV is invalid: {exc}"
    if not rows:
        return "sync-observe performance CSV has no observed rows"
    required_columns = {
        "timestamp",
        "cpu_usage",
        "mem_usage",
        "net_total_mbps",
        "local_block_height",
        "sync_status",
        "execution_mgas_per_sec",
        "execution_metric_source",
        "execution_metric_status",
        "current_qps",
        "qps_data_available",
    }
    columns = set(rows[0])
    if missing := sorted(required_columns - columns):
        return "sync-observe performance CSV is missing columns: " + ", ".join(missing)
    if not any(column.endswith("_total_iops") for column in columns):
        return "sync-observe performance CSV has no disk IOPS column"
    if not any(column.endswith("_avg_await") for column in columns):
        return "sync-observe performance CSV has no disk latency column"
    has_healthy_node_observation = False
    for row in rows:
        if str(row.get("current_qps") or "").strip() not in {"0", "0.0", "0.00"}:
            return "sync-observe reported a non-zero QPS workload"
        if str(row.get("qps_data_available") or "").strip().lower() not in {"false", "0"}:
            return "sync-observe incorrectly reports QPS data as available"
        status = str(row.get("execution_metric_status") or "").strip().lower()
        source = str(row.get("execution_metric_source") or "").strip()
        if status not in {"available", "unavailable"} or not source:
            return "sync-observe MGas provenance is ambiguous"
        if status == "available":
            try:
                mgas = float(str(row.get("execution_mgas_per_sec") or ""))
                gas = float(str(row.get("execution_gas_per_sec") or ""))
            except ValueError:
                return "sync-observe available MGas row has no numeric value"
            if mgas < 0 or gas < 0 or abs(gas - (mgas * 1_000_000)) > 0.5:
                return "sync-observe MGas and gas/s observations disagree"
        local_height = str(row.get("local_block_height") or "").strip().lower()
        local_health = str(row.get("local_health") or "").strip().lower()
        sync_status = str(row.get("sync_status") or "").strip().lower()
        probe_error = str(row.get("probe_error") or "").strip().lower()
        if (
            local_height not in {"", "null", "n/a"}
            and local_health in {"1", "true"}
            and sync_status not in {"", "unknown", "unhealthy"}
            and probe_error in {"", "null", "none"}
        ):
            try:
                has_healthy_node_observation = float(local_height) >= 0
            except ValueError:
                return "sync-observe local height observation is invalid"
    if not has_healthy_node_observation:
        return "sync-observe has no healthy observed node sample"
    range_error = _observation_range_error(
        summary_path,
        rows,
        label="sync-observe performance CSV",
    )
    if range_error:
        return range_error
    health_path = Path(str(artifacts.get("sync_health_csv") or ""))
    try:
        with health_path.open(newline="", encoding="utf-8") as handle:
            health_rows = list(csv.DictReader(handle))
    except (OSError, csv.Error) as exc:
        return f"sync-observe health CSV is invalid: {exc}"
    health_columns = {
        "timestamp",
        "local_block_height",
        "sync_status",
        "probe_error",
    }
    if not health_rows or not health_columns.issubset(health_rows[0]):
        return "sync-observe health CSV has no typed observations"
    if not any(
        str(row.get("local_block_height") or "").strip().lower()
        not in {"", "null", "n/a"}
        and str(row.get("sync_status") or "").strip().lower()
        not in {"", "unknown", "unhealthy"}
        and str(row.get("probe_error") or "").strip().lower()
        in {"", "null", "none"}
        for row in health_rows
    ):
        return "sync-observe health CSV has no successful node probe"
    range_error = _observation_range_error(
        summary_path,
        health_rows,
        label="sync-observe health CSV",
        require_all_within=False,
    )
    if range_error:
        return range_error
    chart_path = Path(str(artifacts.get("sync_timeline_chart") or ""))
    try:
        chart = chart_path.read_bytes()
    except OSError as exc:
        return f"sync-observe timeline chart is unreadable: {exc}"
    return _png_error(chart)


def _approved_rpc_methods(plan: Mapping[str, Any]) -> tuple[set[str], str]:
    mode = str(plan.get("rpc_mode") or "").strip().lower()
    requirements = dict(plan.get("chain_template_requirements") or {})
    if mode == "single":
        method = str(requirements.get("single_method") or "").strip()
        return ({method}, "") if method else (set(), "approved single RPC method is missing")
    if mode == "mixed":
        weighted = requirements.get("mixed_weighted")
        if not isinstance(weighted, list):
            return set(), "approved mixed RPC workload is missing"
        methods = {
            str(item.get("method") or "").strip()
            for item in weighted
            if isinstance(item, Mapping)
            and isinstance(item.get("weight"), int)
            and item["weight"] > 0
        }
        if not methods:
            return set(), "approved mixed RPC workload has no positive-weight methods"
        return methods, ""
    return set(), "approved RPC mode is invalid"


def _observation_epoch(value: Any) -> float:
    raw = str(value or "").strip()
    if not raw:
        raise ValueError("timestamp is missing")
    parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _summary_time_range(path: Path) -> tuple[float, float]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(summary, Mapping):
        raise ValueError("summary is not a JSON object")
    start = _observation_epoch(summary.get("start_time"))
    end = _observation_epoch(summary.get("end_time"))
    if end < start:
        raise ValueError("summary time range is reversed")
    return start, end


def _observation_range_error(
    summary_path: Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    label: str,
    require_all_within: bool = True,
) -> str:
    try:
        start, end = _summary_time_range(summary_path)
        observations = [_observation_epoch(row.get("timestamp")) for row in rows]
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return f"{label} time range is invalid: {exc}"
    if not observations:
        return f"{label} has no timestamped observations"
    if require_all_within:
        if min(observations) < start or max(observations) > end:
            return f"{label} observations are outside the summary time range"
    elif max(observations) < start or min(observations) > end:
        return f"{label} does not overlap the summary time range"
    return ""


def _png_error(data: bytes) -> str:
    if len(data) < 45 or data[:8] != b"\x89PNG\r\n\x1a\n":
        return "sync-observe timeline chart is not a valid PNG"
    offset = 8
    seen_ihdr = False
    seen_iend = False
    width = height = bit_depth = color_type = interlace = 0
    decompressor = zlib.decompressobj()
    decoded_bytes = 0
    try:
        while offset < len(data):
            if offset + 12 > len(data):
                raise ValueError("truncated PNG chunk")
            length = struct.unpack(">I", data[offset : offset + 4])[0]
            chunk_type = data[offset + 4 : offset + 8]
            end = offset + 12 + length
            if end > len(data):
                raise ValueError("truncated PNG chunk payload")
            payload = data[offset + 8 : offset + 8 + length]
            expected_crc = struct.unpack(">I", data[offset + 8 + length : end])[0]
            if zlib.crc32(chunk_type + payload) & 0xFFFFFFFF != expected_crc:
                raise ValueError("PNG chunk CRC mismatch")
            if chunk_type == b"IHDR":
                if seen_ihdr or offset != 8 or length != 13:
                    raise ValueError("invalid PNG IHDR")
                width, height, bit_depth, color_type, compression, filtering, interlace = struct.unpack(
                    ">IIBBBBB", payload
                )
                if (
                    width <= 0
                    or height <= 0
                    or bit_depth != 8
                    or color_type != 6
                    or compression != 0
                    or filtering != 0
                    or interlace != 0
                ):
                    raise ValueError("unsupported PNG image format")
                seen_ihdr = True
            elif chunk_type == b"IDAT":
                if not seen_ihdr or seen_iend:
                    raise ValueError("invalid PNG IDAT ordering")
                decoded_bytes += len(decompressor.decompress(payload))
            elif chunk_type == b"IEND":
                if length != 0 or not seen_ihdr or seen_iend:
                    raise ValueError("invalid PNG IEND")
                decoded_bytes += len(decompressor.flush())
                seen_iend = True
                if end != len(data):
                    raise ValueError("data follows PNG IEND")
            offset = end
    except (struct.error, zlib.error, ValueError) as exc:
        return f"sync-observe timeline chart is invalid: {exc}"
    expected_decoded = height * (1 + width * 4)
    if not seen_ihdr or not seen_iend or decoded_bytes != expected_decoded:
        return "sync-observe timeline chart has incomplete image data"
    return ""


def _g5_admission_envelope_error(
    *,
    envelope: Mapping[str, Any],
    request: Mapping[str, Any],
    scenario: Any,
    artifact: Mapping[str, Any],
    approved_plan: Mapping[str, Any],
) -> str:
    admission = g5_scenario_admission(scenario.scenario_id)
    if envelope.get("schema_version") != 1:
        return "G5 admission envelope schema is invalid"
    if dict(envelope.get("repository_revision") or {}) != dict(
        artifact.get("revision") or {}
    ):
        return "G5 admission envelope revision mismatch"
    if (
        str(envelope.get("scenario_id") or "") != scenario.scenario_id
        or envelope.get("sequence_index") != admission.sequence_index
    ):
        return "G5 admission envelope scenario mismatch"
    approved_path = Path(str(request.get("approved_plan_file") or "")).resolve()
    if (
        str(envelope.get("approved_plan_file") or "") != str(approved_path)
        or str(envelope.get("approved_plan_sha256") or "")
        != str(request.get("approved_plan_sha256") or "")
    ):
        return "G5 admission envelope approved-plan binding mismatch"
    endpoint_contract = dict(envelope.get("endpoint_identity_contract") or {})
    container_requirements = dict(envelope.get("container_metrics_requirements") or {})
    runtime_contract = dict(envelope.get("g5_runtime_contract") or {})
    runtime_contract_sha256 = str(envelope.get("g5_runtime_contract_sha256") or "")
    if not admission.endpoint_env_var:
        if (
            endpoint_contract
            or container_requirements
            or runtime_contract
            or runtime_contract_sha256
        ):
            return "fake-node G5 envelope contains real-node requirements"
        return ""
    expected_runtime_contract = G5_RUNTIME_CONTRACT.to_dict()
    if (
        runtime_contract != expected_runtime_contract
        or runtime_contract_sha256 != content_hash(runtime_contract)
    ):
        return "G5 runtime contract binding mismatch"
    execution_env = dict((approved_plan.get("execution") or {}).get("environment") or {})
    endpoint = str(execution_env.get(admission.endpoint_env_var) or "")
    metrics_url = str(execution_env.get("NODE_PROMETHEUS_METRICS_URL") or "")
    expected_endpoint = {
        "chain": str(approved_plan.get("chain") or ""),
        "env_var": admission.endpoint_env_var,
        "endpoint_sha256": hashlib.sha256(endpoint.encode("utf-8")).hexdigest(),
        "probe_method": "eth_chainId",
        "params_sha256": content_hash([]),
        "expected_identity": G5_RUNTIME_CONTRACT.chain_id,
    }
    if (
        endpoint != G5_RUNTIME_CONTRACT.rpc_url
        or metrics_url != G5_RUNTIME_CONTRACT.metrics_url
        or endpoint_contract != expected_endpoint
    ):
        return "G5 admission endpoint identity contract mismatch"
    expected_container = {
        "compose_service": G5_RUNTIME_CONTRACT.compose_service,
        "image_digest": G5_RUNTIME_CONTRACT.image_digest,
        "image_reference": G5_RUNTIME_CONTRACT.image_reference,
        "rpc_url_sha256": hashlib.sha256(
            G5_RUNTIME_CONTRACT.rpc_url.encode("utf-8")
        ).hexdigest(),
        "metrics_env_var": "NODE_PROMETHEUS_METRICS_URL",
        "metrics_url_sha256": hashlib.sha256(metrics_url.encode("utf-8")).hexdigest(),
        "metrics_required": True,
    }
    if container_requirements != expected_container:
        return "G5 admission container/metrics requirements mismatch"
    try:
        created_at = datetime.fromisoformat(
            str(envelope.get("created_at") or "").replace("Z", "+00:00")
        )
        admitted_at = datetime.fromisoformat(
            str(request.get("jobs_root_admitted_at") or "").replace("Z", "+00:00")
        )
        started_at = datetime.fromisoformat(
            str(artifact.get("started_at") or "").replace("Z", "+00:00")
        )
    except ValueError:
        return "G5 admission envelope timestamp is invalid"
    if (
        any(value.tzinfo is None for value in (created_at, admitted_at, started_at))
        or not admitted_at <= created_at <= started_at
    ):
        return "G5 admission envelope timestamp is out of bounds"
    return ""


def _real_execution_endpoint_error(
    *,
    request: Mapping[str, Any],
    plan: Mapping[str, Any],
    scenario: Any,
    artifact: Mapping[str, Any],
    envelope: Mapping[str, Any],
) -> str:
    admission = g5_scenario_admission(scenario.scenario_id)
    probe = dict(request.get("endpoint_probe") or {})
    if not admission.endpoint_env_var:
        return "fake-node scenario unexpectedly contains endpoint probe evidence" if probe else ""
    identity = dict(envelope.get("endpoint_identity_contract") or {})
    execution_env = dict((plan.get("execution") or {}).get("environment") or {})
    endpoint = str(execution_env.get(admission.endpoint_env_var) or "").strip()
    expected = str(identity.get("expected_identity") or "").strip()
    method = str(identity.get("probe_method") or "").strip()
    required = {
        "chain": str(plan.get("chain") or ""),
        "endpoint_sha256": hashlib.sha256(endpoint.encode("utf-8")).hexdigest() if endpoint else "",
        "endpoint_env_var": admission.endpoint_env_var,
        "probe_method": method,
        "expected_identity": expected,
        "verified": True,
    }
    for name, value in required.items():
        if probe.get(name) != value:
            return f"real execution endpoint probe mismatch: {name}"
    if not isinstance(probe.get("http_status"), int) or not 200 <= probe["http_status"] < 300:
        return "real execution endpoint probe HTTP status is invalid"
    request_contract = dict(probe.get("request_contract") or {})
    response_contract = dict(probe.get("response_contract") or {})
    expected_request_contract = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": method,
        "params_sha256": str(identity.get("params_sha256") or ""),
    }
    if request_contract != expected_request_contract:
        return "real execution endpoint request extraction contract mismatch"
    if content_hash(request_contract) != str(probe.get("request_contract_sha256") or ""):
        return "real execution endpoint request contract hash mismatch"
    if (
        response_contract.get("jsonrpc") != "2.0"
        or response_contract.get("id") != request_contract.get("id")
        or response_contract.get("result_path") != "$.result"
        or response_contract.get("result_type") != "hex_quantity"
        or not isinstance(response_contract.get("result"), str)
        or re.fullmatch(r"0x[0-9a-fA-F]+", response_contract["result"]) is None
        or set(response_contract)
        != {"jsonrpc", "id", "result_path", "result_type", "result"}
    ):
        return "real execution endpoint response extraction contract is invalid"
    if content_hash(response_contract) != str(probe.get("response_contract_sha256") or ""):
        return "real execution endpoint response contract hash mismatch"
    observed = str(response_contract.get("result") or "").strip()
    if str(probe.get("observed_identity") or "") != observed:
        return "real execution endpoint observed identity is not probe-derived"
    if observed.lower() != expected.lower():
        return "real execution endpoint observed identity does not match approved identity"
    if content_hash(response_contract) != str(probe.get("response_sha256") or ""):
        return "real execution endpoint typed response hash mismatch"
    try:
        probed_at = datetime.fromisoformat(str(probe.get("probed_at") or "").replace("Z", "+00:00"))
    except ValueError:
        return "real execution endpoint probe timestamp is invalid"
    if probed_at.tzinfo is None:
        return "real execution endpoint probe timestamp has no timezone"
    try:
        admitted_at = datetime.fromisoformat(
            str(request.get("jobs_root_admitted_at") or "").replace("Z", "+00:00")
        )
    except ValueError:
        return "real execution endpoint admission timestamp is invalid"
    if probed_at < admitted_at:
        return "real execution endpoint probe predates jobs-root admission"
    try:
        started_at = datetime.fromisoformat(
            str(artifact.get("started_at") or "").replace("Z", "+00:00")
        )
    except ValueError:
        return "real execution endpoint execution timestamp is invalid"
    if probed_at > started_at:
        return "real execution endpoint probe occurred after submission"
    return ""


def _real_execution_runtime_attestation_error(
    *,
    request: Mapping[str, Any],
    plan: Mapping[str, Any],
    scenario: Any,
    artifact: Mapping[str, Any],
    envelope: Mapping[str, Any],
) -> str:
    admission = g5_scenario_admission(scenario.scenario_id)
    attestation = dict(request.get("runtime_attestation") or {})
    if not admission.endpoint_env_var:
        return (
            "fake-node scenario unexpectedly contains runtime attestation"
            if attestation
            else ""
        )
    if attestation.get("schema_version") != 1:
        return "geth-dev runtime attestation schema is invalid"
    if dict(attestation.get("repository_revision") or {}) != dict(
        artifact.get("revision") or {}
    ):
        return "geth-dev runtime attestation revision mismatch"
    unsigned = dict(attestation)
    attestation_hash = str(unsigned.pop("attestation_sha256", ""))
    if content_hash(unsigned) != attestation_hash:
        return "geth-dev runtime attestation hash mismatch"
    host_attestation_file = str(attestation.get("host_attestation_file") or "")
    host_attestation_hash = str(attestation.get("host_attestation_sha256") or "")
    try:
        host_attestation = validate_host_attestation_file(
            host_attestation_file,
            repo_root=REPO_ROOT,
            revision=dict(artifact.get("revision") or {}),
        )
    except (OSError, ValueError) as exc:
        return f"G5 host attestation is invalid: {exc}"
    if host_attestation.get("attestation_file_sha256") != host_attestation_hash:
        return "G5 host attestation hash binding mismatch"
    container = dict(attestation.get("container") or {})
    container_id = str(container.get("container_id") or "")
    image_digest = str(container.get("image_digest") or "")
    requirements = dict(envelope.get("container_metrics_requirements") or {})
    if (
        len(container_id) < 12
        or any(character not in "0123456789abcdef" for character in container_id.lower())
        or image_digest != G5_RUNTIME_CONTRACT.image_digest
        or container.get("image_reference") != G5_RUNTIME_CONTRACT.image_reference
        or container.get("compose_service") != G5_RUNTIME_CONTRACT.compose_service
        or image_digest != requirements.get("image_digest")
        or container.get("image_reference") != requirements.get("image_reference")
        or G5_RUNTIME_CONTRACT.compose_service
        not in (container.get("network_aliases") or ())
        or not container.get("network_addresses")
    ):
        return "geth-dev container identity is invalid"
    if content_hash(container) != str(attestation.get("container_contract_sha256") or ""):
        return "geth-dev container contract hash mismatch"
    target_container = dict(host_attestation.get("target_container") or {})
    target_networks = dict(target_container.get("networks") or {})
    target_addresses = sorted({
        str(details.get("ip_address") or "")
        for details in target_networks.values()
        if str(details.get("ip_address") or "")
    })
    target_aliases = sorted({
        str(alias)
        for details in target_networks.values()
        for alias in (details.get("aliases") or ())
        if str(alias)
    })
    expected_container = {
        "container_id": str(target_container.get("container_id") or ""),
        "image_digest": str(target_container.get("image_digest") or ""),
        "image_reference": str(target_container.get("image_reference") or ""),
        "compose_service": str(target_container.get("compose_service") or ""),
        "compose_project": str(target_container.get("compose_project") or ""),
        "network_addresses": target_addresses,
        "network_aliases": target_aliases,
    }
    inspect_hash = str(attestation.get("docker_inspect_sha256") or "")
    if container != expected_container:
        return "geth-dev runtime container disagrees with host attestation"
    if (
        not _is_sha256(inspect_hash)
        or inspect_hash
        != str(
            (host_attestation.get("docker_inspect_sha256") or {}).get(
                G5_RUNTIME_CONTRACT.compose_service
            )
            or ""
        )
    ):
        return "geth-dev docker inspect hash is invalid"
    endpoint_probe = dict(request.get("endpoint_probe") or {})
    if (
        attestation.get("rpc_endpoint_sha256") != endpoint_probe.get("endpoint_sha256")
        or attestation.get("rpc_chain_id") != endpoint_probe.get("observed_identity")
        or attestation.get("rpc_response_sha256") != endpoint_probe.get("response_sha256")
    ):
        return "geth-dev RPC attestation is not bound to the endpoint probe"
    rpc_route = dict(attestation.get("rpc_route") or {})
    metrics_route = dict(attestation.get("metrics_route") or {})
    container_addresses = set(container.get("network_addresses") or ())
    for label, route, expected_url in (
        ("RPC", rpc_route, G5_RUNTIME_CONTRACT.rpc_url),
        ("metrics", metrics_route, G5_RUNTIME_CONTRACT.metrics_url),
    ):
        parsed = urllib.parse.urlsplit(expected_url)
        if (
            route.get("scheme") != parsed.scheme
            or route.get("hostname") != G5_RUNTIME_CONTRACT.compose_service
            or route.get("port") != parsed.port
            or route.get("path") != (parsed.path or "/")
            or not set(route.get("resolved_addresses") or ()) & container_addresses
        ):
            return f"geth-dev {label} route is not bound to the inspected container"
    execution_env = dict((plan.get("execution") or {}).get("environment") or {})
    metrics_env_var = str(requirements.get("metrics_env_var") or "")
    metrics_url = str(execution_env.get(metrics_env_var) or "").strip()
    metrics = dict(attestation.get("metrics_probe") or {})
    if not metrics_url:
        return "geth-dev metrics endpoint is absent from the approved plan"
    if metrics.get("metrics_url_sha256") != hashlib.sha256(
        metrics_url.encode("utf-8")
    ).hexdigest():
        return "geth-dev metrics URL hash mismatch"
    if metrics.get("metrics_url_sha256") != requirements.get("metrics_url_sha256"):
        return "geth-dev metrics probe does not satisfy the G5 envelope"
    if (
        not isinstance(metrics.get("http_status"), int)
        or not 200 <= metrics["http_status"] < 300
        or not _is_sha256(str(metrics.get("body_sha256") or ""))
        or not isinstance(metrics.get("body_size_bytes"), int)
        or metrics["body_size_bytes"] <= 0
        or not isinstance(metrics.get("non_comment_sample_count"), int)
        or metrics["non_comment_sample_count"] <= 0
        or not isinstance(metrics.get("metric_family_count"), int)
        or metrics["metric_family_count"] <= 0
        or metrics.get("parser") != "prometheus_text_v0.0.4"
    ):
        return "geth-dev metrics probe contract is invalid"
    try:
        admitted_at = datetime.fromisoformat(
            str(request.get("jobs_root_admitted_at") or "").replace("Z", "+00:00")
        )
        metrics_at = datetime.fromisoformat(
            str(metrics.get("probed_at") or "").replace("Z", "+00:00")
        )
        attested_at = datetime.fromisoformat(
            str(attestation.get("attested_at") or "").replace("Z", "+00:00")
        )
        started_at = datetime.fromisoformat(
            str(artifact.get("started_at") or "").replace("Z", "+00:00")
        )
    except ValueError:
        return "geth-dev runtime attestation timestamp is invalid"
    if (
        any(
            value.tzinfo is None
            for value in (admitted_at, metrics_at, attested_at, started_at)
        )
        or not admitted_at <= metrics_at <= attested_at <= started_at
    ):
        return "geth-dev runtime attestation timestamps are out of bounds"
    try:
        host_attested_at = datetime.fromisoformat(
            str(host_attestation.get("attested_at") or "").replace("Z", "+00:00")
        )
    except ValueError:
        return "G5 host attestation timestamp is invalid"
    if host_attested_at.tzinfo is None or host_attested_at > admitted_at:
        return "G5 host attestation postdates jobs-root admission"
    return ""


def _required_artifact_manifest_error(
    *,
    artifact: Mapping[str, Any],
    request: Mapping[str, Any],
    observed_artifacts: Mapping[str, Any],
    scenario: Any,
) -> str:
    manifests = [
        item for item in artifact.get("job_artifacts") or ()
        if Path(str(item.get("path") or "")).name == "required-artifacts.sha256.json"
    ]
    if len(manifests) != 1:
        return "real execution evidence must bind one required-artifact manifest"
    manifest_record = manifests[0]
    observed_job = dict(
        ((artifact.get("result") or {}).get("observed_job") or {})
    )
    try:
        owner_roots = derive_artifact_owner_roots(observed_job)
    except (OSError, ValueError) as exc:
        return f"required-artifact owner roots are invalid: {exc}"
    try:
        _validate_hashed_artifact_identity(
            manifest_record,
            expected_job_id=str(artifact.get("job_id") or ""),
            expected_run_dir=Path(
                str(((artifact.get("result") or {}).get("observed_job") or {}).get("run_dir") or "")
            ),
        )
        manifest = json.loads(Path(str(manifest_record["path"])).read_text(encoding="utf-8"))
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        return f"required-artifact manifest is invalid: {exc}"
    if manifest.get("schema_version") != ARTIFACT_MANIFEST_SCHEMA_VERSION:
        return "required-artifact manifest schema mismatch"
    if str(manifest.get("job_id") or "") != str(artifact.get("job_id") or ""):
        return "required-artifact manifest job mismatch"
    if str(manifest.get("scenario_id") or "") != scenario.scenario_id:
        return "required-artifact manifest scenario mismatch"
    expected_roots = {
        name: str(path)
        for name, path in sorted(owner_roots.items())
    }
    if manifest.get("owner_roots") != expected_roots:
        return "required-artifact manifest owner roots mismatch"
    expected_roots_hash = hashlib.sha256(
        json.dumps(
            expected_roots,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if manifest.get("owner_roots_sha256") != expected_roots_hash:
        return "required-artifact manifest owner roots hash mismatch"
    job_plan_file = Path(str(observed_job.get("plan_file") or ""))
    try:
        expected_job_plan_hash = hashlib.sha256(job_plan_file.read_bytes()).hexdigest()
    except OSError as exc:
        return f"required-artifact job plan cannot be read: {exc}"
    if manifest.get("job_plan_sha256") != expected_job_plan_hash:
        return "required-artifact manifest job plan hash mismatch"
    records = list(manifest.get("artifacts") or ())
    names = [str(item.get("name") or "") for item in records]
    if len(names) != len(set(names)) or set(names) != set(scenario.required_artifacts):
        return "required-artifact manifest names mismatch"
    if records != list(request.get("required_artifact_hashes") or ()):
        return "required-artifact request binding mismatch"
    for record in records:
        name = str(record.get("name") or "")
        path = Path(str(record.get("path") or ""))
        if str(observed_artifacts.get(name) or "") != str(path):
            return f"required-artifact observed path mismatch: {name}"
        expected_owner = required_artifact_owner(scenario, name)
        expected_root = owner_roots[expected_owner].resolve()
        if (
            record.get("owner") != expected_owner
            or record.get("owner_root") != str(expected_root)
        ):
            return f"required-artifact owner mismatch: {name}"
        try:
            admitted_path = validate_owned_artifact_path(
                path,
                expected_owner=expected_owner,
                owner_roots=owner_roots,
            )
            if record.get("relative_path") != str(
                path.relative_to(expected_root)
            ):
                return f"required-artifact relative path mismatch: {name}"
            if record.get("resolved_path") != str(admitted_path):
                return f"required-artifact resolved path mismatch: {name}"
            observed_link_target = os.readlink(path) if path.is_symlink() else ""
            if record.get("link_target") != observed_link_target:
                return f"required-artifact link target mismatch: {name}"
            if hashlib.sha256(admitted_path.read_bytes()).hexdigest() != str(record.get("sha256") or ""):
                return f"required-artifact hash mismatch: {name}"
            if admitted_path.stat().st_size != int(record.get("size_bytes", -1)):
                return f"required-artifact size mismatch: {name}"
        except (OSError, TypeError, ValueError) as exc:
            return f"required-artifact cannot be admitted: {name}: {exc}"
    return ""


def validate_real_execution_ledger_artifacts(
    artifacts: Iterable[Mapping[str, Any]],
    *,
    revision: Mapping[str, str],
) -> tuple[bool, str]:
    observed = list(artifacts)
    expected = sorted(
        (scenario for scenario in EXECUTION_SCENARIOS if scenario.real_evidence_required),
        key=lambda scenario: g5_scenario_admission(scenario.scenario_id).sequence_index,
    )
    if [item.get("scenario_id") for item in observed] != [
        scenario.scenario_id for scenario in expected
    ]:
        return False, "real execution ledger scenario sequence mismatch"
    job_ids = [str(item.get("job_id") or "") for item in observed]
    if not all(job_ids) or len(set(job_ids)) != len(job_ids):
        return False, "real execution ledger job identities are not unique"
    if any(item.get("outcome") != "passed" for item in observed):
        return False, "real execution ledger requires four passing scenarios"
    for artifact, scenario in zip(observed, expected):
        admission = g5_scenario_admission(scenario.scenario_id)
        if dict(artifact.get("revision") or {}) != dict(revision):
            return False, "real execution ledger revision mismatch"
        request = dict(artifact.get("request") or {})
        if request.get("ledger_sequence") != admission.sequence_index:
            return False, "real execution ledger sequence binding mismatch"
        if (
            str(request.get("predecessor_scenario_id") or "")
            != admission.predecessor_scenario_id
        ):
            return False, "real execution ledger predecessor binding mismatch"
    for predecessor, current in zip(observed, observed[1:]):
        try:
            predecessor_finished = datetime.fromisoformat(
                str(predecessor.get("finished_at") or "").replace("Z", "+00:00")
            )
            current_started = datetime.fromisoformat(
                str(current.get("started_at") or "").replace("Z", "+00:00")
            )
        except ValueError:
            return False, "real execution ledger ordering timestamp is invalid"
        if (
            predecessor.get("outcome") != "passed"
            or predecessor_finished.tzinfo is None
            or current_started.tzinfo is None
            or predecessor_finished > current_started
        ):
            return False, "real execution ledger is not a fully serial passing sequence"
    smoke = observed[1]
    final = observed[2]
    smoke_probe = dict((smoke.get("request") or {}).get("endpoint_probe") or {})
    final_probe = dict((final.get("request") or {}).get("endpoint_probe") or {})
    if smoke_probe.get("endpoint_sha256") != final_probe.get("endpoint_sha256"):
        return False, "real-node smoke and final target different endpoints"
    smoke_profile = dict(
        ((smoke.get("request") or {}).get("runtime_env_projection") or {}).get(
            "execution_profile"
        )
        or {}
    )
    final_profile = dict(
        ((final.get("request") or {}).get("runtime_env_projection") or {}).get(
            "execution_profile"
        )
        or {}
    )
    smoke_units = smoke_profile.get("minimum_request_seconds")
    final_units = final_profile.get("minimum_request_seconds")
    if (
        not isinstance(smoke_units, int)
        or not isinstance(final_units, int)
        or smoke_units <= 0
        or final_units <= smoke_units
        or smoke_profile.get("output_root_sha256")
        == final_profile.get("output_root_sha256")
    ):
        return False, "real-node final profile is not materially stronger than smoke"
    runtime_artifacts = (observed[1], observed[2], observed[3])
    runtime_identities = {
        (
            str(
                ((item.get("request") or {}).get("runtime_attestation") or {})
                .get("container", {})
                .get("container_id")
                or ""
            ),
            str(
                ((item.get("request") or {}).get("runtime_attestation") or {})
                .get("container", {})
                .get("image_digest")
                or ""
            ),
        )
        for item in runtime_artifacts
    }
    if len(runtime_identities) != 1 or not all(next(iter(runtime_identities), ())):
        return False, "real/sync execution did not attest one geth-dev container image"
    host_identities = {
        (
            str(
                ((item.get("request") or {}).get("runtime_attestation") or {})
                .get("host_attestation_file")
                or ""
            ),
            str(
                ((item.get("request") or {}).get("runtime_attestation") or {})
                .get("host_attestation_sha256")
                or ""
            ),
            str(
                ((item.get("request") or {}).get("runtime_attestation") or {})
                .get("docker_inspect_sha256")
                or ""
            ),
        )
        for item in runtime_artifacts
    }
    if len(host_identities) != 1 or not all(next(iter(host_identities), ())):
        return False, "real/sync execution did not share one host attestation"
    return True, ""


def validate_real_execution_predecessor(
    artifacts: Iterable[Mapping[str, Any]],
    *,
    scenario: Any,
    started_at: str,
    endpoint_probe: Mapping[str, Any],
    runtime_attestation: Mapping[str, Any],
) -> tuple[bool, str]:
    predecessor_id = g5_scenario_admission(
        scenario.scenario_id
    ).predecessor_scenario_id
    if not predecessor_id:
        return True, ""
    matches = [
        artifact
        for artifact in artifacts
        if str(artifact.get("scenario_id") or "") == predecessor_id
    ]
    if len(matches) != 1:
        return False, f"required predecessor evidence is missing: {predecessor_id}"
    predecessor = matches[0]
    if predecessor.get("outcome") != "passed":
        return False, f"required predecessor did not pass: {predecessor_id}"
    try:
        predecessor_finished = datetime.fromisoformat(
            str(predecessor.get("finished_at") or "").replace("Z", "+00:00")
        )
        current_started = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
    except ValueError:
        return False, "predecessor ordering timestamp is invalid"
    if (
        predecessor_finished.tzinfo is None
        or current_started.tzinfo is None
        or predecessor_finished > current_started
    ):
        return False, f"required predecessor is not serial: {predecessor_id}"
    predecessor_probe = dict(
        (predecessor.get("request") or {}).get("endpoint_probe") or {}
    )
    if (
        predecessor_probe
        and endpoint_probe
        and predecessor_probe.get("endpoint_sha256")
        != endpoint_probe.get("endpoint_sha256")
    ):
        return False, f"required predecessor targeted a different endpoint: {predecessor_id}"
    predecessor_runtime = dict(
        (predecessor.get("request") or {}).get("runtime_attestation") or {}
    )
    predecessor_container = dict(predecessor_runtime.get("container") or {})
    current_container = dict(runtime_attestation.get("container") or {})
    if predecessor_container and current_container and (
        predecessor_container.get("container_id") != current_container.get("container_id")
        or predecessor_container.get("image_digest") != current_container.get("image_digest")
    ):
        return False, f"required predecessor used a different runtime: {predecessor_id}"
    return True, ""


def _validate_hashed_artifact_identity(
    item: Mapping[str, Any],
    *,
    expected_job_id: str = "",
    expected_run_dir: Path | None = None,
) -> None:
    if not isinstance(item, Mapping):
        raise ValueError("hashed artifact identity is not an object")
    raw_path = str(item.get("path") or "").strip()
    expected_hash = str(item.get("sha256") or "")
    if not raw_path or not _is_sha256(expected_hash):
        raise ValueError("hashed artifact identity requires path and sha256")
    path = Path(raw_path)
    if not path.is_file():
        raise ValueError(f"hashed artifact does not exist: {path}")
    if expected_run_dir is not None:
        _validate_exact_descendant(path, expected_run_dir=expected_run_dir)
    elif expected_job_id and expected_job_id not in path.parts:
        raise ValueError(f"hashed artifact is not owned by job {expected_job_id}: {path}")
    observed_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed_hash != expected_hash:
        raise ValueError(f"hashed artifact content mismatch: {path}")


def _validate_exact_descendant(path: Path, *, expected_run_dir: Path) -> None:
    if expected_run_dir.is_symlink() or not expected_run_dir.is_dir():
        raise ValueError("fresh run_dir is missing or symlinked")
    root = expected_run_dir.resolve(strict=True)
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise ValueError(f"artifact escapes fresh run_dir: {path}") from exc
    current = path if path.is_absolute() else Path.cwd() / path
    while current != root:
        if current.is_symlink():
            raise ValueError(f"artifact path contains a symlink: {current}")
        parent = current.parent
        if parent == current:
            raise ValueError(f"artifact is not rooted in fresh run_dir: {path}")
        current = parent
    if resolved == root:
        raise ValueError("artifact cannot be the run_dir itself")


def _lane_error(edge: Mapping[str, Any], evidence_class: str) -> str:
    lane = (edge.get("evidence") or {}).get(evidence_class)
    if lane is None:
        return "edge has no lane applicability declaration"
    if not bool(lane.get("required")):
        return f"edge is not required for {evidence_class}: {lane.get('applicability_reason') or 'unspecified'}"
    return ""


def _require_lane(edge: Mapping[str, Any], evidence_class: str) -> None:
    reason = _lane_error(edge, evidence_class)
    if reason:
        raise ValueError(reason)


def _action_admission(
    edge: Mapping[str, Any],
    question: Mapping[str, Any],
    input_value: Any,
    *,
    admitted: bool,
) -> dict[str, Any]:
    if str(edge.get("edge_type") or "") == "question_option":
        option_id = str(edge.get("option_id") or "")
        for option in question.get("options") or []:
            if str(option.get("id") or "") == option_id:
                candidates = {
                    option_id.casefold(),
                    str(option.get("label") or "").strip().casefold(),
                    str(option.get("value") or "").strip().casefold(),
                }
                if str(input_value or "").strip().casefold() not in candidates:
                    return {}
                return {
                    "admitted": admitted,
                    "source": "pending_question_option",
                    "option_id": option_id,
                    "input": deepcopy(input_value),
                    "action": deepcopy(dict(option.get("action") or {})),
                }
        return {}
    if str(edge.get("edge_type") or "") == "manual_input":
        return {
            "admitted": admitted,
            "source": "pending_question_manual_input",
            "input": deepcopy(input_value),
            "action": {
                "type": "answer_pending",
                "question_id": str(question.get("id") or ""),
                "field": str(question.get("field") or ""),
                "value": _normalize_scalar(input_value),
            },
        }
    return {}


def _validate_return_event(
    event: Mapping[str, Any],
    artifact: Mapping[str, Any],
    turn: Mapping[str, Any],
) -> str:
    details = event.get("details")
    if not isinstance(details, Mapping):
        return "compiled graph return event has no details"
    expected = dict(artifact.get("source_boundary_hashes") or {})
    mismatches = {key: (details.get(key), value) for key, value in expected.items() if details.get(key) != value}
    return f"compiled graph return hashes mismatch: {mismatches}" if mismatches else ""


def _has_event(events: Iterable[Mapping[str, Any]], event_type: str, edge: Mapping[str, Any]) -> bool:
    return _event(events, event_type, edge) is not None


def _event(
    events: Iterable[Mapping[str, Any]],
    event_type: str,
    edge: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    edge_key = str(edge.get("edge_key") or "")
    return next(
        (
            event for event in events
            if str(event.get("event_type") or "") == event_type
            and str(event.get("edge_key") or "") == edge_key
        ),
        None,
    )


def _normalize_scalar(value: Any) -> str:
    return str(value or "").strip().rstrip(",，;；").strip()


def _read_path(value: Mapping[str, Any], path: str) -> Any:
    current: Any = value
    for part in str(path).split("."):
        if not isinstance(current, Mapping):
            return None
        current = current.get(part)
    return current


def _utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)

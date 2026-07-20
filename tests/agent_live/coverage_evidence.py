"""Tamper-evident artifacts for compiled-graph and real PTY evidence."""

from __future__ import annotations

import hashlib
import json
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from agent.harness.coverage_events import state_diff_between
from agent.harness.runtime_identity import repository_revision
from agent.utils.redaction import redact


ARTIFACT_SCHEMA_VERSION = 3
CLI_ARTIFACT_SCHEMA_VERSION = 4
REAL_EXECUTION_ARTIFACT_SCHEMA_VERSION = 1
TURN_OBSERVATION_SCHEMA_VERSION = 1
PTY_DIAGNOSTIC_SCHEMA_VERSION = 1
PTY_DIAGNOSTIC_STATUSES = {
    "failed_attempt": frozenset({
        "verification_pending", "verification_error", "postcondition_failed",
    }),
    "interruption": frozenset({"interrupted"}),
}
EXECUTION_EVIDENCE_CLASSES = frozenset(
    {"deterministic", "real_cli", "dynamic_dual_ai", "real_execution"}
)
COMPILED_GRAPH_RUNNER = "tests.agent_live.graph_turn.invoke_product_graph_turn"
PTY_REAL_CLI_RUNNER = "tests.agent_live.pty.real_cli"
PTY_DYNAMIC_DUAL_AI_RUNNER = "tests.agent_live.pty.dynamic_dual_ai"
REAL_EXECUTION_RUNNER = "tests.agent_live.real_execution.integration"
REAL_EXECUTION_OPERATION_BY_ACTION = {
    "approve_preflight_smoke": "preflight_smoke",
    "approve_final_benchmark": "final_benchmark",
}


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
    state_diff_hashes: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
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

    seed = deepcopy(dict(seed_state))
    before = deepcopy(dict(before_state))
    after = deepcopy(dict(after_state))
    question = deepcopy(dict(before.get("pending_question") or {}))
    if before.get("last_user_input") != input_value:
        raise ValueError("input does not match the compiled-graph invocation state")
    event_trace = [deepcopy(dict(event)) for event in events]
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

    state_diff = state_diff_between(before, after)
    response = deepcopy(list(after.get("visible_response") or []))
    next_question = deepcopy(dict(after.get("pending_question") or {}))
    admitted_action = _action_admission(
        edge,
        question,
        input_value,
        admitted=(returned if admitted is None else bool(admitted)),
    )
    turn_evidence = {
        "compiled_graph_helper": COMPILED_GRAPH_RUNNER,
        "seed": seed,
        "before": before,
        "input": deepcopy(input_value),
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
        "input_hash": content_hash(input_value),
        "seed_state_hash": content_hash(seed),
        "before_state_hash": content_hash(before),
        "after_state_hash": content_hash(after),
        "turn_evidence": turn_evidence,
        "turn_evidence_hash": content_hash(turn_evidence),
        "event_trace": event_trace,
        "event_trace_hash": content_hash(event_trace),
        "exit_status": int(exit_status),
        "outcome": str(outcome),
        "error": str(error or ""),
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
        redact(_verified_postcondition_payload(record.verified_postcondition))
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

    turn_payload = asdict(turn)
    turn_payload.update({
        "previous_response_hash": content_hash(turn.previous_agent_response),
        "user_message_hash": content_hash(turn.user_message),
        "agent_response_hash": content_hash(turn.agent_response),
    })
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
        "turn_observation": _turn_observation_payload(observation),
        "turn_observation_hash": content_hash(_turn_observation_payload(observation)),
        "outcome": "passed",
        "exit_status": 0,
        "error": "",
        "started_at": _utc_timestamp(),
        "finished_at": _utc_timestamp(),
    }
    payload["evidence_id"] = content_hash(payload)
    payload["artifact_hash"] = content_hash(payload)
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
    for derived in ("previous_response_hash", "user_message_hash", "agent_response_hash"):
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
        "seed_state_hash": content_hash(scenario.seed_state or {}),
        "session_id": turn.session_id,
        "session_purpose": str(observation.runtime_events[0].session_purpose),
        "pending_question_id": str(scenario.question.get("id") or ""),
        "pending_contract_hash": content_hash(scenario.question),
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
    if content_hash(turn.user_message) != content_hash(authoritative_case.resolve_input()):
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
    if observation.after_turn_index != observation.before_turn_index + 1:
        raise ValueError("turn observation event did not advance exactly one turn")
    if turn.turn_index != observation.after_turn_index:
        raise ValueError("PTY and runtime turn indexes disagree")
    if len(observation.runtime_events) != 2:
        raise ValueError("turn observation requires baseline and committed runtime events")
    baseline, committed = observation.runtime_events
    _validate_runtime_event(baseline)
    _validate_runtime_event(committed)
    if dict(baseline.revision) != dict(observation.revision):
        raise ValueError("baseline runtime event revision mismatch")
    if dict(committed.revision) != dict(observation.revision):
        raise ValueError("committed runtime event revision mismatch")
    if committed.event_type not in {"turn_committed", "turn_recovered"}:
        raise ValueError("new runtime event is not a committed turn")
    if baseline.thread_id != turn.session_id or committed.thread_id != turn.session_id:
        raise ValueError("runtime event thread does not match the PTY session")
    if baseline.session_purpose != committed.session_purpose:
        raise ValueError("runtime event session purpose changed")
    if baseline.turn_index != observation.before_turn_index:
        raise ValueError("baseline runtime event turn index is stale")
    if committed.turn_index != observation.after_turn_index:
        raise ValueError("committed runtime event turn index is stale")
    if committed.before_fingerprint != baseline.after_fingerprint:
        raise ValueError("runtime event fingerprint chain is stale")
    if observation.before_state_fingerprint != committed.before_fingerprint:
        raise ValueError("turn observation before fingerprint mismatch")
    if observation.after_state_fingerprint != committed.after_fingerprint:
        raise ValueError("turn observation after fingerprint mismatch")
    if turn.before_fingerprint != committed.before_fingerprint:
        raise ValueError("PTY before fingerprint was not observed from the committed event")
    if turn.after_fingerprint != committed.after_fingerprint:
        raise ValueError("PTY after fingerprint was not observed from the committed event")
    if dict(observation.pending_contract) != dict(baseline.pending_contract):
        raise ValueError("pending contract is not bound to the baseline runtime event")
    edge_type = str(edge.get("edge_type") or "")
    if edge_type == "action_transition":
        if baseline.pending_question_id or baseline.pending_contract:
            raise ValueError("action-only baseline unexpectedly has a pending contract")
    elif not baseline.pending_question_id:
        raise ValueError("baseline runtime event has no pending contract")
    if dynamic_selection is not None:
        if dict(observation.simulator_decision) != {
            **asdict(dynamic_selection),
            "target_coverage_ids": list(dynamic_selection.target_coverage_ids),
        }:
            raise ValueError("turn observation simulator decision mismatch")
        if observation.target_edge_key not in dynamic_selection.target_coverage_ids:
            raise ValueError("dynamic selection did not target the authoritative edge key")
    elif observation.simulator_decision:
        raise ValueError("real CLI observation must not claim a simulator decision")

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
    if event.schema_version != 2:
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
    if str(event.pending_contract.get("id") or "") != event.pending_question_id:
        raise ValueError("runtime event pending contract identity is inconsistent")


def verify_runtime_postcondition(
    edge: Mapping[str, Any],
    baseline: RuntimeTurnEvent,
    committed: RuntimeTurnEvent,
    turn: PtyCliTurnRecord,
) -> VerifiedPostcondition:
    """Verify one live turn from product-emitted transition facts.

    This verifier is fixed by the evidence module. Callers cannot replace it
    with a callback that simply declares the scheduled edge successful.
    """

    del turn
    errors: list[str] = []
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
    if edge_type in {"action_transition", "question_option"} and scheduled_action and scheduled_action not in admitted:
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
        elif visible_group not in navigation_targets:
            errors.append(
                "change_group destination was not observed: "
                f"visible={visible_group or '<none>'}, admitted={sorted(navigation_targets)}"
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
        if (
            not rejection_expected
            and scheduled_action
            and scheduled_action not in admitted
        ):
            errors.append(f"scheduled manual-input action was not admitted: {scheduled_action}")
        if rejection_expected and admitted and not accepted.intersection(admitted):
            errors.append("rejected manual input admitted an unrelated action")

    expected = dict(edge.get("expected_postcondition") or {})
    expected_paths: list[str] = []
    if edge_type == "question_option":
        for path, value in expected.items():
            expected_paths.append(str(path))
            if not _postcondition_value_matches(
                committed.after_value_hashes,
                str(path),
                value,
            ):
                errors.append(f"expected postcondition was not observed: {path}")
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
                if after_hashes != before_hashes:
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

    edge_key = str(edge.get("edge_key") or "")
    details = {
        "baseline_question_id": baseline.pending_question_id,
        "committed_question_id": committed.pending_question_id,
        "expected_postcondition_paths": sorted(expected_paths),
        "expected_admitted": edge.get("expected_admitted"),
        "rejection_observed": bool(rejection_expected and not errors),
        "errors": errors,
    }
    return VerifiedPostcondition(
        verifier_id="anychain.runtime-transition-proof.v1",
        passed=not errors,
        observed_coverage_ids=(edge_key,) if not errors and edge_key else (),
        admitted_typed_actions=admitted,
        state_diff=state_diff,
        next_question_or_result=next_result,
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
    operation_kind: str,
    request: Mapping[str, Any],
    result: Mapping[str, Any],
    job_id: str,
    job_artifacts: Iterable[Mapping[str, Any]],
    log_artifacts: Iterable[Mapping[str, Any]],
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
    expected_operation = REAL_EXECUTION_OPERATION_BY_ACTION.get(str(edge.get("action_type") or ""))
    if expected_operation and operation_kind != expected_operation:
        raise ValueError(
            f"real execution operation disagrees with {edge.get('action_type')}: {operation_kind}"
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
    if not request or not result:
        raise ValueError("real execution evidence requires request and result objects")
    artifacts = [dict(item) for item in job_artifacts]
    logs = [dict(item) for item in log_artifacts]
    if not artifacts or not logs:
        raise ValueError("real execution evidence requires hashed job and log artifacts")
    for item in (*artifacts, *logs):
        _validate_hashed_artifact_identity(item, expected_job_id=str(job_id))
    payload: dict[str, Any] = {
        "artifact_type": "real_execution",
        "schema_version": REAL_EXECUTION_ARTIFACT_SCHEMA_VERSION,
        "edge_key": str(edge.get("edge_key") or ""),
        "contract_hash": str(edge.get("contract_hash") or ""),
        "runtime_variant_hash": str(edge.get("contract_variant_hash") or ""),
        "evidence_class": "real_execution",
        "runner_type": REAL_EXECUTION_RUNNER,
        "revision": dict(revision),
        "operation_kind": operation_kind,
        "request": deepcopy(dict(request)),
        "request_hash": content_hash(request),
        "result": deepcopy(dict(result)),
        "result_hash": content_hash(result),
        "job_id": str(job_id),
        "job_artifacts": artifacts,
        "job_artifacts_hash": content_hash(artifacts),
        "log_artifacts": logs,
        "log_artifacts_hash": content_hash(logs),
        "outcome": "passed",
        "exit_status": 0,
        "error": "",
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
) -> tuple[bool, str]:
    lane_error = _lane_error(edge, "real_execution")
    if lane_error:
        return False, lane_error
    if artifact.get("artifact_type") != "real_execution":
        return False, "artifact is not real execution evidence"
    if artifact.get("schema_version") != REAL_EXECUTION_ARTIFACT_SCHEMA_VERSION:
        return False, "unsupported real execution evidence schema"
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
    expected_operation = REAL_EXECUTION_OPERATION_BY_ACTION.get(str(edge.get("action_type") or ""))
    if expected_operation and artifact.get("operation_kind") != expected_operation:
        return False, "real execution operation disagrees with action"
    if not str(artifact.get("job_id") or "").strip():
        return False, "real execution job id is missing"
    if artifact.get("outcome") != "passed" or artifact.get("exit_status") != 0 or artifact.get("error"):
        return False, "qualifying real execution evidence must be an observed pass"
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
                )
        except ValueError as exc:
            return False, str(exc)
        if content_hash(items) != artifact.get(hash_field):
            return False, f"{field} hash mismatch"
    unsigned = dict(artifact)
    artifact_hash = str(unsigned.pop("artifact_hash", ""))
    if content_hash(unsigned) != artifact_hash:
        return False, "artifact hash mismatch"
    evidence_payload = dict(unsigned)
    evidence_id = str(evidence_payload.pop("evidence_id", ""))
    if content_hash(evidence_payload) != evidence_id:
        return False, "evidence id mismatch"
    return True, ""


def _validate_hashed_artifact_identity(
    item: Mapping[str, Any],
    *,
    expected_job_id: str = "",
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
    if expected_job_id and expected_job_id not in path.parts:
        raise ValueError(f"hashed artifact is not owned by job {expected_job_id}: {path}")
    observed_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed_hash != expected_hash:
        raise ValueError(f"hashed artifact content mismatch: {path}")


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
    expected = {
        "before_hash": artifact.get("before_state_hash"),
        "input_hash": artifact.get("input_hash"),
        "question_hash": content_hash(turn.get("question")),
        "after_hash": artifact.get("after_state_hash"),
        "state_diff_hash": content_hash(turn.get("state_diff")),
        "response_hash": content_hash(turn.get("response")),
        "next_question_hash": content_hash(turn.get("next_question")),
    }
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

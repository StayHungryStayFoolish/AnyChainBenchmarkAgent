"""Immutable batch orchestration for response-driven Agent Chaos shards.

The orchestrator owns process isolation, terminal classification, and artifact
indexing.  It deliberately does not generate simulator messages or change the
coverage, discovery, or bridge protocols.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import fcntl
import hashlib
import hmac
import inspect
import json
import os
import shutil
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Protocol, Sequence

from agent.harness.runtime_identity import repository_revision
from agent.harness.terminal_protocol import presentation_hash
from agent.utils.redaction import redact
from tests.agent_live.chaos_scheduler import (
    CHAOS_SCHEDULE_SCHEMA_VERSION,
    build_chaos_schedule,
    build_journey_schedule,
    journey_schedule_payload,
    schedule_payload,
    validate_chaos_schedule,
    validate_journey_schedule,
)
from tests.agent_live.codex_simulator_bridge import (
    CONTEXT_FRAME,
    DECISION_FRAME,
    RESULT_FRAME,
    simulator_context_binding,
    simulator_context_hash,
    validate_simulator_attestation,
)
from tests.agent_live.journey_simulator_bridge import (
    CONTEXT_FRAME as JOURNEY_CONTEXT_FRAME,
    DECISION_FRAME as JOURNEY_DECISION_FRAME,
    RESULT_FRAME as JOURNEY_RESULT_FRAME,
    load_verifier_registry,
)
from tests.agent_live.dynamic_dual_ai_chaos import (
    ChaosRunConfig,
    ContainerPtyBridgeTransport,
    DynamicDualAiChaosRunner,
    DynamicDualAiJourneyRunner,
    JourneyDecisionProvenance,
    JourneyExternallyBlockedError,
    JourneyObservedEdge,
    JourneyOutcomeVerifierRegistry,
    JourneyRunError,
    JourneySimulatorInvalidError,
    JourneySimulatorContext,
    JourneySimulatorDecision,
    JourneyTerminalClassification,
    SimulatorContext,
    SimulatorDecision,
    SimulatorDecisionInvalid,
    SimulatorTerminalClassification,
    TerminalOutcomeObservation,
    _runtime_event_from_mapping,
    _terminal_outcome_from_mapping,
    _journey_outcome_verification_payload,
    build_journey_turn_result,
    build_journey_verifier_context,
    observe_journey_edges,
    validate_journey_evidence_artifact,
    verify_journey_forbidden_outcomes,
    verify_journey_outcome,
    verify_declared_postconditions,
)
from tests.agent_live.container_process_guard import (
    EXECUTION_ID_ENV,
    RECEIPT_DIR_ENV,
    CleanupReceiptArtifact,
    ContainerProcessGuard,
)
from tests.agent_live.discovery_ledger import (
    DISCOVERY_LEDGER_SCHEMA_VERSION,
    RESULT_CLASSIFICATIONS,
    append_discovery_attempt,
    build_discovery_attempt,
)
from tests.agent_live.coverage_evidence import (
    DynamicTurnSelection,
    PtyCliTurnRecord,
    PtyAuthoritySigner,
    TurnObservation,
    _evidence_projection,
    _project_runtime_event_payload,
    _turn_observation_from_payload,
    admit_pty_artifact_bundle,
    canonical_controller_turn_facts,
    create_pty_authority_signer,
    load_pty_admitted_artifact,
    load_validated_pty_artifact_bundle,
    load_pty_authority_receipt,
    load_valid_evidence_reference,
    pty_admitted_bundle_digest,
    pty_artifact_bundle_digest,
    rollback_pty_artifact_bundle,
    rollback_pty_artifact_admission,
    turn_observation_payload,
    sign_controller_payload,
    pty_transcript_hash,
    validate_pty_cli_candidate_artifact,
    validate_pty_cli_evidence_artifact,
    validate_pty_diagnostic_candidate,
    validate_pty_diagnostic_artifact,
    verify_controller_payload_signature,
    validate_runtime_turn_event,
    verify_runtime_postcondition,
)
from tests.agent_live.generate_harness_coverage_ledger import build_ledger


BATCH_MANIFEST_SCHEMA_VERSION = 7
BATCH_RESULT_SCHEMA_VERSION = 5
CLEANUP_RECEIPT_SCHEMA_VERSION = 2
BATCH_SURVIVOR_PROOF_SCHEMA_VERSION = 1
DEFAULT_SHARD_COUNT = 32
DEFAULT_MAX_CONCURRENCY = 4
FORMAL_EDGE_SHARD_COUNT = 24
FORMAL_JOURNEY_SHARD_COUNT = 8
SHARD_LANES = frozenset({"edge", "journey"})
JOURNEY_CONTROLLER_OBSERVATION_FIELDS = frozenset({
    "shard_turn_ordinal",
    "turn_index",
    "previous_response_hash",
    "user_message_hash",
    "approved_decision_hash",
    "simulator_context_hash",
    "approved_decision",
    "simulator_context",
    "submission_sequence",
    "submitted_input_commitment",
    "agent_response_hash",
    "baseline_runtime_event_hash",
    "committed_runtime_event_hash",
    "terminal_runtime_event_id",
    "terminal_runtime_event_sequence",
    "terminal_outcome_hash",
    "turn_result_hash",
    "previous_controller_observation_hash",
    "controller_observation_hash",
})


class ExternalDecisionBlocked(RuntimeError):
    """The external simulator could not provide a decision for this response."""


class DecisionBroker(Protocol):
    def __call__(
        self, shard_id: str, context: Mapping[str, Any]
    ) -> Mapping[str, Any] | Awaitable[Mapping[str, Any]]:
        """Return exactly one decision bound to the supplied response context."""


CommandFactory = Callable[[int, Path, Path, str, int, str], Sequence[str]]


@dataclass(frozen=True)
class TimeoutPolicy:
    shard_seconds: float = 900.0
    decision_seconds: float = 240.0
    cleanup_seconds: float = 5.0


@dataclass(frozen=True)
class FrozenShardSpec:
    shard_id: str
    index: int
    seed: int
    session_id: str
    target_path: str
    target_hash: str
    schedule_id: str
    runtime_root: str
    execution_id: str
    container_cleanup_receipt_dir: str
    command: tuple[str, ...]
    target_ids: tuple[str, ...]
    edge_keys: tuple[str, ...]
    personas: tuple[str, ...]
    goals: tuple[str, ...]
    input_classes: tuple[str, ...]
    sequence_ids: tuple[str, ...]
    factor_ids: tuple[str, ...]
    lane: str = "edge"
    verifier_registry_import: str = ""
    verifier_registry_id: str = ""
    obligation_id: str = ""
    subject_group: str = ""
    simulator_attestation_required: bool = False


@dataclass(frozen=True)
class FrozenBatchManifest:
    manifest_id: str
    batch_id: str
    created_at_ns: int
    manifest_path: str
    discovery_ledger_path: str
    repo_root: str
    revision: Mapping[str, str]
    shard_count: int
    max_concurrency: int
    pty_authority_trust_root_id: str
    pty_authority_public_key_b64: str
    shards: tuple[FrozenShardSpec, ...]
    required_env_names: tuple[str, ...]
    timeout_policy: TimeoutPolicy
    stderr_cap_bytes: int
    expected_obligation_set_hash: str = ""
    worker_runtime: str = "docker"
    formal_profile: bool = False
    controller_owned_execution: bool = True
    scheduler_schema_version: int = CHAOS_SCHEDULE_SCHEMA_VERSION
    discovery_schema_version: int = DISCOVERY_LEDGER_SCHEMA_VERSION
    schema_version: int = BATCH_MANIFEST_SCHEMA_VERSION
    _authority_signer: PtyAuthoritySigner | None = field(
        default=None,
        compare=False,
        repr=False,
    )


@dataclass(frozen=True)
class CleanupReceipt:
    receipt_id: str
    schema_version: int
    shard_id: str
    execution_id: str
    host_proof: Mapping[str, Any]
    container_proof: Mapping[str, Any]
    worker_identity: Mapping[str, Any]
    exit_code: int | None
    actions: tuple[str, ...]
    errors: tuple[str, ...]
    cleaned: bool
    started_at_ns: int
    finished_at_ns: int


@dataclass(frozen=True)
class ShardResult:
    shard_id: str
    classification: str
    attempt_count: int
    started_at_ns: int
    finished_at_ns: int
    exit_code: int | None
    target_hash: str
    response_hashes: tuple[str, ...]
    decision_hashes: tuple[str, ...]
    transcript_hash: str
    schedule_result_hash: str
    evidence_hashes: tuple[str, ...]
    diagnostic_hashes: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    diagnostic_ids: tuple[str, ...]
    stderr_hash: str
    stderr_path: str
    stderr_truncated: bool
    cleanup_receipt_path: str
    cleanup_receipt_hash: str
    reason: str


@dataclass(frozen=True)
class BatchResultIndex:
    index_id: str
    batch_id: str
    manifest_id: str
    revision: Mapping[str, str]
    execution_authority_mode: str
    execution_status: str
    release_status: str
    scheduled: int
    started: int
    completed: int
    discovery_attempt_ids: tuple[str, ...]
    classification_counts: Mapping[str, int]
    batch_survivor_proof: Mapping[str, Any]
    pty_authority_trust_root_id: str
    pty_authority_public_key_b64: str
    controller_signature_b64: str
    shards: tuple[ShardResult, ...]
    schema_version: int = BATCH_RESULT_SCHEMA_VERSION


@dataclass(frozen=True)
class JourneyControllerAuthoritySnapshot:
    """Immutable bytes admitted by one controller-owned Journey bundle."""

    candidate_bytes: bytes
    authority_bytes: bytes
    bundle_digest: str

    def candidate_payload(self) -> dict[str, Any]:
        payload = json.loads(self.candidate_bytes)
        if not isinstance(payload, dict):
            raise ValueError("Journey controller candidate must be an object")
        return payload

    def authority_payload(self) -> dict[str, Any]:
        payload = json.loads(self.authority_bytes)
        if not isinstance(payload, dict):
            raise ValueError("Journey controller authority must be an object")
        return payload


@dataclass
class _RunState:
    response_hashes: list[str] = field(default_factory=list)
    decision_hashes: list[str] = field(default_factory=list)
    controller_turn_observations: list[Mapping[str, Any]] = field(
        default_factory=list
    )
    controller_turn_facts: list[bytes] = field(
        default_factory=list
    )
    approved_decisions: list[Mapping[str, Any]] = field(default_factory=list)
    submitted_inputs: list[Mapping[str, Any]] = field(default_factory=list)
    observation_submission_cursor: int = 0
    context_turn_indices: list[int] = field(default_factory=list)
    result_payload: Mapping[str, Any] | None = None
    forced_classification: str = ""
    reason: str = ""


@dataclass(frozen=True)
class _HostCleanupOutcome:
    artifact: CleanupReceiptArtifact | None
    cleaned: bool
    exit_code: int | None
    actions: tuple[str, ...]
    errors: tuple[str, ...]


class _ControllerOwnedPtyTransport(ContainerPtyBridgeTransport):
    """Expose the controller-owned PTY bridge to the outer process guard."""

    def __init__(
        self,
        *args: Any,
        process_guard: ContainerProcessGuard,
        state: _RunState,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._controller_process_guard = process_guard
        self._controller_state = state
        self.controller_worker_identity: Mapping[str, Any] = {}

    def start(self, *, env: Mapping[str, str]) -> None:
        super().start(env=env)
        process = self._process
        if process is None:
            raise RuntimeError("controller-owned PTY bridge did not start")
        identity = self._controller_process_guard.register_pid(
            process.pid,
            role="controller_pty_bridge",
        )
        self._controller_process_guard.register_pid(
            process.pid,
            role="host_worker",
        )
        self.controller_worker_identity = identity.payload()

    def submit_bracketed_paste(self, message: str) -> None:
        bound_approval: Mapping[str, Any] | None = None
        bound_sequences = {
            int(item.get("approval_sequence") or 0)
            for item in self._controller_state.submitted_inputs
            if int(item.get("approval_sequence") or 0) > 0
        }
        for approved in self._controller_state.approved_decisions:
            approval_sequence = int(approved.get("sequence") or 0)
            if approval_sequence in bound_sequences:
                continue
            decision = dict(approved.get("decision") or {})
            if str(decision.get("user_message") or "") != message:
                raise ValueError(
                    "PTY submission differs from the pending approved "
                    "simulator decision"
                )
            bound_approval = approved
            break
        self._controller_state.submitted_inputs.append({
            "sequence": len(self._controller_state.submitted_inputs) + 1,
            "message": message,
            "message_hash": _content_hash(message),
            "source": (
                "simulator_decision"
                if bound_approval is not None
                else "controller_runtime"
            ),
            "approval_sequence": (
                int(bound_approval.get("sequence") or 0)
                if bound_approval is not None
                else 0
            ),
            "submitted_at_ns": time.time_ns(),
        })
        super().submit_bracketed_paste(message)


class _ControllerDecisionAdapter:
    """Synchronously bridge a controller-owned runner to the async broker."""

    def __init__(
        self,
        *,
        loop: asyncio.AbstractEventLoop,
        broker: DecisionBroker,
        manifest: FrozenBatchManifest,
        shard: FrozenShardSpec,
        state: _RunState,
    ) -> None:
        self.loop = loop
        self.broker = broker
        self.manifest = manifest
        self.shard = shard
        self.state = state

    def __call__(
        self,
        context: SimulatorContext | JourneySimulatorContext,
    ) -> SimulatorDecision | JourneySimulatorDecision:
        payload = self._context_payload(context)
        _validate_context_frame(
            self.shard,
            payload,
            len(self.state.response_hashes),
            previous_turn_index=(
                self.state.context_turn_indices[-1]
                if self.state.context_turn_indices
                else None
            ),
        )
        response_hash = str(payload["previous_response_hash"])
        self.state.response_hashes.append(response_hash)
        if self.shard.lane == "journey":
            self.state.context_turn_indices.append(int(payload["turn_index"]))
        future = asyncio.run_coroutine_threadsafe(
            _call_broker_with_timeout(
                self.broker,
                self.shard.shard_id,
                payload,
                timeout_seconds=self.manifest.timeout_policy.decision_seconds,
            ),
            self.loop,
        )
        try:
            decision = future.result(
                timeout=(
                    self.manifest.timeout_policy.decision_seconds
                    + max(self.manifest.timeout_policy.cleanup_seconds, 1.0)
                )
            )
        except concurrent.futures.TimeoutError as exc:
            future.cancel()
            raise ExternalDecisionBlocked(
                "simulator decision timed out"
            ) from exc
        try:
            normalized = _validate_decision(
                payload,
                decision,
                lane=self.shard.lane,
                simulator_attestation_required=(
                    self.shard.simulator_attestation_required
                ),
            )
        except _SimulatorInvalid as exc:
            if self.shard.lane == "journey":
                raise JourneySimulatorInvalidError(str(exc)) from exc
            raise SimulatorDecisionInvalid(str(exc)) from exc
        self.state.decision_hashes.append(_content_hash(normalized))
        self.state.approved_decisions.append({
            "sequence": len(self.state.approved_decisions) + 1,
            "decision": dict(normalized),
            "decision_hash": _content_hash(normalized),
            "context_hash": simulator_context_hash(payload),
            "context": json.loads(
                _canonical_json(redact(payload))
            ),
            "approved_at_ns": time.time_ns(),
        })
        if self.shard.lane == "journey":
            return JourneySimulatorDecision(
                user_message=str(normalized["user_message"]),
                persona=str(normalized["persona"]),
                mission=str(normalized["mission"]),
                rationale=str(normalized["rationale"]),
                risk_factor_ids=tuple(
                    str(item)
                    for item in normalized.get("risk_factor_ids") or ()
                ),
                broker_request_id=(
                    str(normalized.get("broker_request_id") or "")
                    if normalized.get("simulator_attestation")
                    else ""
                ),
                simulator_attestation=dict(
                    normalized.get("simulator_attestation") or {}
                ),
                variant_binding={
                    str(key): str(value)
                    for key, value in dict(
                        normalized.get("variant_binding") or {}
                    ).items()
                },
            )
        return SimulatorDecision(
            user_message=str(normalized["user_message"]),
            persona=str(normalized["persona"]),
            goal=str(normalized["goal"]),
            rationale=str(normalized["rationale"]),
            target_coverage_ids=tuple(
                str(item)
                for item in normalized.get("target_coverage_ids") or ()
            ),
        )

    def _context_payload(
        self,
        context: SimulatorContext | JourneySimulatorContext,
    ) -> dict[str, Any]:
        response_hash = _content_hash(context.previous_agent_response)
        common = {
            "schema_version": 1,
            "session_id": context.session_id,
            "turn_index": context.turn_index,
            "previous_agent_response": str(
                redact(context.previous_agent_response)
            ),
            "previous_response_hash": response_hash,
            "previous_response_received_at_ns": (
                context.previous_response_received_at_ns
            ),
            "transcript": [
                [str(redact(user)), str(redact(agent))]
                for user, agent in context.transcript
            ],
        }
        if isinstance(context, JourneySimulatorContext):
            return {
                **common,
                "schedule": journey_schedule_payload(context.schedule),
                "observed_edge_keys": list(context.observed_edge_keys),
            }
        return {
            **common,
            "scheduled_target": asdict(context.scheduled_target),
            "coverage_contract": dict(context.coverage_contract),
        }


def _append_controller_fact(
    state: _RunState,
    *,
    lane: str,
    fact: Mapping[str, Any],
) -> None:
    """Deep-freeze one callback fact as canonical bytes and a hash-chain link."""

    previous_hash = ""
    if state.controller_turn_facts:
        previous = json.loads(state.controller_turn_facts[-1])
        previous_hash = str(previous.get("controller_fact_hash") or "")
    unsigned = {
        "ordinal": len(state.controller_turn_facts) + 1,
        "lane": lane,
        "previous_controller_fact_hash": previous_hash,
        "fact": json.loads(_canonical_json(fact)),
    }
    frozen = {
        **unsigned,
        "controller_fact_hash": _content_hash(unsigned),
    }
    state.controller_turn_facts.append(
        _canonical_json(frozen).encode("utf-8")
    )


def _controller_fact_payloads(
    state: _RunState,
    *,
    lane: str,
) -> tuple[Mapping[str, Any], ...]:
    """Decode and validate the immutable controller callback ledger."""

    decoded: list[Mapping[str, Any]] = []
    previous_hash = ""
    for ordinal, raw in enumerate(state.controller_turn_facts, start=1):
        if not isinstance(raw, bytes):
            raise ValueError("controller turn ledger retained a mutable record")
        try:
            envelope = json.loads(raw)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("controller turn ledger is not canonical") from exc
        if not isinstance(envelope, Mapping):
            raise ValueError("controller turn ledger record is not an object")
        unsigned = {
            key: value
            for key, value in envelope.items()
            if key != "controller_fact_hash"
        }
        if (
            int(envelope.get("ordinal") or 0) != ordinal
            or envelope.get("previous_controller_fact_hash") != previous_hash
            or envelope.get("controller_fact_hash") != _content_hash(unsigned)
            or _canonical_json(envelope).encode("utf-8") != raw
        ):
            raise ValueError("controller turn ledger hash chain is invalid")
        previous_hash = str(envelope["controller_fact_hash"])
        if envelope.get("lane") == lane:
            fact = envelope.get("fact")
            if not isinstance(fact, Mapping):
                raise ValueError("controller turn ledger fact is malformed")
            decoded.append(dict(fact))
    return tuple(decoded)


def _turn_from_payload(payload: Mapping[str, Any]) -> PtyCliTurnRecord:
    return PtyCliTurnRecord(**dict(payload))


def _selection_from_payload(
    payload: Mapping[str, Any],
) -> DynamicTurnSelection:
    values = dict(payload)
    values["target_coverage_ids"] = tuple(
        values.get("target_coverage_ids") or ()
    )
    return DynamicTurnSelection(**values)


def _journey_decision_from_payload(
    payload: Mapping[str, Any],
) -> JourneyDecisionProvenance:
    values = dict(payload)
    values["risk_factor_ids"] = tuple(values.get("risk_factor_ids") or ())
    for field_name in (
        "simulator_context_binding",
        "simulator_attestation",
        "variant_attestation",
    ):
        values[field_name] = dict(values.get(field_name) or {})
    return JourneyDecisionProvenance(**values)


def _safe_terminal_outcome_projection(
    outcome: TerminalOutcomeObservation,
    *,
    runtime_event_payload_hash: str,
) -> TerminalOutcomeObservation:
    payload = dict(_evidence_projection(asdict(outcome)))
    payload["runtime_event_payload_hash"] = runtime_event_payload_hash
    unsigned = {
        key: value
        for key, value in payload.items()
        if key != "record_hash"
    }
    payload["record_hash"] = _content_hash(unsigned)
    return _terminal_outcome_from_mapping(payload)


class _ControllerTurnLedger:
    """Capture canonical Edge facts before candidate serialization."""

    def __init__(self, state: _RunState) -> None:
        self.state = state

    def __call__(
        self,
        *,
        edge: Mapping[str, Any],
        turn: PtyCliTurnRecord,
        observation: TurnObservation,
        selection: DynamicTurnSelection,
        terminal_outcomes: tuple[TerminalOutcomeObservation, ...],
    ) -> None:
        _append_controller_fact(
            self.state,
            lane="edge",
            fact={
                "edge": dict(edge),
                "turn": asdict(turn),
                "observation": turn_observation_payload(observation),
                "selection": asdict(selection),
                "terminal_outcomes": [
                    asdict(item) for item in terminal_outcomes
                ],
            },
        )


class _ControllerJourneyLedger:
    """Independently verify and freeze one Journey callback boundary."""

    def __init__(
        self,
        state: _RunState,
        *,
        schedule: Any,
        registry: JourneyOutcomeVerifierRegistry,
        edge_index: Mapping[str, Mapping[str, Any]],
    ) -> None:
        self.state = state
        self.schedule = schedule
        self.registry = registry
        self.edge_index = edge_index

    def observe_initial(
        self,
        *,
        initial_event: Any,
        terminal: Any,
        forbidden: Sequence[Any],
    ) -> None:
        del terminal, forbidden
        safe_initial = _runtime_event_from_mapping(
            _project_runtime_event_payload(initial_event)
        )
        validate_runtime_turn_event(safe_initial)
        context = build_journey_verifier_context(
            schedule=self.schedule,
            initial_event=safe_initial,
            current_event=safe_initial,
            turns=(),
            events=(),
            decisions=(),
            transcript=(),
            observed_edge_keys=(),
            latest_turn=None,
        )
        safe_forbidden = verify_journey_forbidden_outcomes(
            self.schedule,
            context,
            self.registry,
        )
        safe_terminal = verify_journey_outcome(
            self.schedule.terminal_outcome,
            context,
            self.registry,
        )
        _append_controller_fact(
            self.state,
            lane="journey_initial",
            fact={
                "initial_event": asdict(safe_initial),
                "initial_verification": {
                    "terminal_outcome": (
                        _journey_outcome_verification_payload(safe_terminal)
                    ),
                    "forbidden_outcomes": [
                        _journey_outcome_verification_payload(item)
                        for item in safe_forbidden
                    ],
                },
            },
        )

    def __call__(
        self,
        *,
        initial_event: Any,
        turn: PtyCliTurnRecord,
        baseline_event: Any,
        committed_event: Any,
        terminal_outcome: TerminalOutcomeObservation,
        decision: JourneyDecisionProvenance,
    ) -> None:
        protected_values = (
            (turn.user_message,)
            if turn.user_message
            else ()
        )
        safe_initial = _runtime_event_from_mapping(
            _project_runtime_event_payload(
                initial_event,
                protected_values=protected_values,
            )
        )
        safe_baseline = _runtime_event_from_mapping(
            _project_runtime_event_payload(
                baseline_event,
                protected_values=protected_values,
            )
        )
        safe_committed = _runtime_event_from_mapping(
            _project_runtime_event_payload(
                committed_event,
                protected_values=protected_values,
            )
        )
        for event in (safe_initial, safe_baseline, safe_committed):
            validate_runtime_turn_event(event)
        frozen_turn = replace(
            turn,
            previous_agent_response=str(
                _evidence_projection(turn.previous_agent_response)
            ),
            user_message=str(_evidence_projection(turn.user_message)),
            agent_response=str(_evidence_projection(turn.agent_response)),
        )
        frozen_turn = replace(
            frozen_turn,
            transcript_hash=pty_transcript_hash(
                session_id=frozen_turn.session_id,
                turn_index=frozen_turn.turn_index,
                previous_agent_response=(
                    frozen_turn.previous_agent_response
                ),
                user_message=frozen_turn.user_message,
                agent_response=frozen_turn.agent_response,
            ),
        )
        frozen_decision = _journey_decision_from_payload(
            _evidence_projection(asdict(decision))
        )
        frozen_terminal_outcome = _safe_terminal_outcome_projection(
            terminal_outcome,
            runtime_event_payload_hash=(
                safe_committed.runtime_event_payload_hash
            ),
        )
        prior = _controller_fact_payloads(self.state, lane="journey")
        turns = [
            _turn_from_payload(item["turn"])
            for item in prior
        ] + [frozen_turn]
        events = [
            _runtime_event_from_mapping(item["committed_event"])
            for item in prior
        ] + [safe_committed]
        decisions = [
            _journey_decision_from_payload(item["decision"])
            for item in prior
        ] + [frozen_decision]
        transcript = [
            (item.user_message, item.agent_response)
            for item in turns
        ]
        observed_edge_keys: list[str] = []
        observed_current = observe_journey_edges(
            self.edge_index,
            safe_baseline,
            safe_committed,
            frozen_turn,
        )
        for item in prior:
            for edge_key in item.get("observed_edge_keys") or ():
                if edge_key not in observed_edge_keys:
                    observed_edge_keys.append(str(edge_key))
        for item in observed_current:
            if item.edge_key not in observed_edge_keys:
                observed_edge_keys.append(item.edge_key)
        context = build_journey_verifier_context(
            schedule=self.schedule,
            initial_event=safe_initial,
            current_event=safe_committed,
            turns=turns,
            events=events,
            decisions=decisions,
            transcript=transcript,
            observed_edge_keys=observed_edge_keys,
            latest_turn=frozen_turn,
        )
        forbidden = verify_journey_forbidden_outcomes(
            self.schedule,
            context,
            self.registry,
        )
        terminal = verify_journey_outcome(
            self.schedule.terminal_outcome,
            context,
            self.registry,
        )
        turn_result = build_journey_turn_result(
            turn=frozen_turn,
            decision=frozen_decision,
            observed_edges=observed_current,
            terminal=terminal,
            forbidden=forbidden,
        )
        _append_controller_fact(
            self.state,
            lane="journey",
            fact={
                "initial_event": asdict(safe_initial),
                "turn": asdict(frozen_turn),
                "baseline_event": asdict(safe_baseline),
                "committed_event": asdict(safe_committed),
                "terminal_outcome": asdict(frozen_terminal_outcome),
                "decision": asdict(frozen_decision),
                "observed_edge_keys": observed_edge_keys,
                "turn_result": turn_result,
            },
        )


def _approved_submission_bindings(
    state: _RunState,
    messages: Sequence[str],
    *,
    input_commitment_key: bytes | None = None,
) -> tuple[Mapping[str, Any], ...]:
    """Resolve qualifying messages against ordered approval/submission facts."""

    approval_rows = tuple(state.approved_decisions)
    approvals = {
        int(item.get("sequence") or 0): item
        for item in approval_rows
    }
    submitted_sequences = [
        int(item.get("sequence") or 0)
        for item in state.submitted_inputs
    ]
    if (
        len(approvals) != len(approval_rows)
        or any(sequence < 1 for sequence in approvals)
        or submitted_sequences != sorted(set(submitted_sequences))
        or any(sequence < 1 for sequence in submitted_sequences)
    ):
        raise ValueError(
            "controller approval/submission ledger order is invalid"
        )
    bound_approval_sequences = {
        int(item.get("approval_sequence") or 0)
        for item in state.submitted_inputs
        if int(item.get("approval_sequence") or 0) > 0
    }
    expected_approval_sequences = set(approvals)
    if bound_approval_sequences != expected_approval_sequences:
        raise ValueError(
            "controller approval/submission ledger has an unbound decision"
        )
    bindings: list[Mapping[str, Any]] = []
    cursor = state.observation_submission_cursor
    for message in messages:
        matched: Mapping[str, Any] | None = None
        while cursor < len(state.submitted_inputs):
            submitted = state.submitted_inputs[cursor]
            cursor += 1
            approval_sequence = int(
                submitted.get("approval_sequence") or 0
            )
            if approval_sequence <= 0:
                continue
            approved = approvals.get(approval_sequence)
            if approved is None:
                raise ValueError(
                    "PTY submission references an unknown simulator approval"
                )
            decision = dict(approved.get("decision") or {})
            if (
                submitted.get("source") != "simulator_decision"
                or
                submitted.get("message") != decision.get("user_message")
                or submitted.get("message_hash")
                != _content_hash(str(decision.get("user_message") or ""))
            ):
                raise ValueError(
                    "approved simulator decision differs from PTY submission"
                )
            projected_message = str(
                _evidence_projection(submitted.get("message") or "")
            )
            if projected_message != message:
                raise ValueError(
                    "next approved PTY submission differs from the "
                    "controller-observed qualifying turn"
                )
            frozen_decision = json.loads(
                _canonical_json(
                    _evidence_projection(
                        approved.get("decision") or {}
                    )
                )
            )
            frozen_context = json.loads(
                _canonical_json(
                    _evidence_projection(
                        approved.get("context") or {}
                    )
                )
            )
            raw_input_commitment = (
                hmac.new(
                    input_commitment_key,
                    _canonical_json({
                        "submission_sequence": int(
                            submitted.get("sequence") or 0
                        ),
                        "user_message": str(
                            submitted.get("message") or ""
                        ),
                    }).encode("utf-8"),
                    hashlib.sha256,
                ).hexdigest()
                if input_commitment_key is not None
                else ""
            )
            if raw_input_commitment:
                frozen_decision[
                    "_controller_submitted_input_commitment"
                ] = raw_input_commitment
                frozen_decision[
                    "_controller_source_user_message_hash"
                ] = _content_hash(
                    str(submitted.get("message") or "")
                )
            matched = {
                "decision_hash": _content_hash(frozen_decision),
                "context_hash": simulator_context_hash(
                    frozen_context
                ),
                "decision": frozen_decision,
                "context": frozen_context,
                "submission_sequence": int(
                    submitted.get("sequence") or 0
                ),
                "submitted_input_commitment": (
                    raw_input_commitment
                ),
            }
            break
        if matched is None:
            raise ValueError(
                "qualifying turn has no controller-observed PTY submission"
            )
        bindings.append(matched)
    state.observation_submission_cursor = cursor
    return tuple(bindings)


def freeze_batch_manifest(
    *,
    repo_root: str | Path,
    targets_dir: str | Path,
    manifest_path: str | Path,
    runtime_base: str | Path,
    shard_count: int = DEFAULT_SHARD_COUNT,
    max_concurrency: int | None = None,
    authority_signer: PtyAuthoritySigner | None = None,
    seed_base: int = 10_000,
    command_factory: CommandFactory | None = None,
    expected_revision: Mapping[str, str] | None = None,
    required_env_names: Sequence[str] = ("DEEPSEEK_API_KEY",),
    timeout_policy: TimeoutPolicy = TimeoutPolicy(),
    stderr_cap_bytes: int = 65_536,
    discovery_ledger_path: str | Path | None = None,
    formal_profile: bool = False,
    worker_runtime: str = "docker",
    expected_obligation_set_hash: str = "",
) -> FrozenBatchManifest:
    """Validate every target and atomically freeze the complete spawn manifest."""

    root = Path(repo_root).resolve()
    target_root = Path(targets_dir).resolve()
    output = Path(manifest_path).resolve()
    runtime = Path(runtime_base).resolve()
    discovery_path = Path(
        discovery_ledger_path
        or (root / ".agent" / "discovery" / "attempts.jsonl")
    ).resolve()
    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    resolved_max_concurrency = (
        min(DEFAULT_MAX_CONCURRENCY, shard_count)
        if max_concurrency is None
        else max_concurrency
    )
    if (
        not isinstance(resolved_max_concurrency, int)
        or isinstance(resolved_max_concurrency, bool)
        or resolved_max_concurrency < 1
    ):
        raise ValueError("max_concurrency must be a positive integer")
    if resolved_max_concurrency > shard_count:
        raise ValueError("max_concurrency cannot exceed shard_count")
    if stderr_cap_bytes < 1:
        raise ValueError("stderr_cap_bytes must be positive")
    if worker_runtime not in {"docker", "linux"}:
        raise ValueError("worker_runtime must be docker or linux")
    if formal_profile and worker_runtime != "linux":
        raise ValueError("formal batches must run inside the Linux control plane")
    if output.exists():
        raise FileExistsError(f"batch manifest is immutable: {output}")
    revision = repository_revision(root)
    if expected_revision is not None and dict(expected_revision) != revision:
        raise ValueError("repository revision does not match the requested batch revision")

    ledger = build_ledger(revision=revision)
    edge_index = {
        str(edge.get("edge_key") or ""): edge
        for edge in ledger.get("edges") or ()
    }
    controller_owned_execution = command_factory is None
    factory = command_factory or _default_command_factory(
        root,
        worker_runtime=worker_runtime,
        decision_timeout_seconds=(
            timeout_policy.decision_seconds + max(timeout_policy.cleanup_seconds, 1.0)
        ),
    )
    batch_nonce = hashlib.sha256(
        f"{revision}|{seed_base}|{shard_count}|{time.time_ns()}".encode()
    ).hexdigest()[:16]
    shards: list[FrozenShardSpec] = []
    for index in range(1, shard_count + 1):
        target_path = target_root / f"{index:02d}.json"
        try:
            target_path = target_path.resolve(strict=True)
            target_path.relative_to(target_root)
        except (FileNotFoundError, ValueError) as exc:
            raise ValueError(
                f"target file must be contained by the frozen target directory: {target_path}"
            ) from exc
        payload = _load_json(target_path)
        lane = _target_lane(payload)
        target_seed = seed_base + index
        frozen_execution: Mapping[str, Any] = {}
        simulator_attestation_required = False
        obligation_id = ""
        subject_group = ""
        if lane == "journey":
            frozen_execution = _frozen_journey_execution(payload)
            if frozen_execution:
                target_seed = int(frozen_execution["seed"])
                obligation_id = str(frozen_execution["obligation_id"])
            simulator_attestation_required = _simulator_attestation_required(payload)
        shard_id = f"{batch_nonce}-{index:02d}"
        session_id = f"batch-{shard_id}"
        shard_runtime = runtime / shard_id
        target_hash = _sha256_file(target_path)
        execution_id = "anychain-chaos-" + hashlib.sha256(
            f"{batch_nonce}|{index}|{session_id}|{target_hash}".encode("utf-8")
        ).hexdigest()
        container_receipt_dir = shard_runtime / "container-cleanup-receipts"
        verifier_registry_import = ""
        verifier_registry_id = ""
        if lane == "edge":
            targets = _target_rows(payload)
            if not targets:
                raise ValueError(f"target file has no scheduled targets: {target_path}")
            schedule = build_chaos_schedule(
                ledger,
                revision=revision,
                seed=target_seed,
                targets=targets,
            )
            validate_chaos_schedule(schedule, ledger=ledger, revision=revision)
            target_ids = tuple(target.target_id for target in schedule.targets)
            edge_keys = tuple(target.edge_key for target in schedule.targets)
            personas = tuple(target.persona for target in schedule.targets)
            goals = tuple(target.goal for target in schedule.targets)
            edges = [edge_index[target.edge_key] for target in schedule.targets]
            input_classes = tuple(str(edge.get("input_class") or "") for edge in edges)
            sequence_ids = tuple(
                target.sequence_id or f"target:{target.target_id}"
                for target in schedule.targets
            )
            factor_ids = tuple(
                "|".join((*target.tuple_ids, target.scenario_id))
                or f"edge:{target.edge_key}"
                for target in schedule.targets
            )
        else:
            journey, verifier_registry_import = _journey_target(payload)
            schedule = build_journey_schedule(
                revision=revision,
                seed=target_seed,
                journey=journey,
            )
            validate_journey_schedule(schedule, revision=revision)
            subject_group = schedule.subject_group
            if frozen_execution:
                if dict(frozen_execution["revision_binding"]) != revision:
                    raise ValueError("frozen Journey execution revision is stale")
                if str(frozen_execution["schedule_id"]) != schedule.schedule_id:
                    raise ValueError("frozen Journey schedule identity drifted")
                if str(frozen_execution["schedule_hash"]) != _content_hash(
                    journey_schedule_payload(schedule)
                ):
                    raise ValueError("frozen Journey schedule hash drifted")
                if obligation_id != schedule.journey_id:
                    raise ValueError("frozen Journey obligation identity drifted")
                if (
                    str(frozen_execution["subject_group"])
                    != schedule.subject_group
                ):
                    raise ValueError(
                        "frozen Journey subject group drifted"
                    )
            registry = load_verifier_registry(verifier_registry_import)
            verifier_registry_id = registry.registry_id
            target_ids = (schedule.journey_id,)
            edge_keys = ()
            personas = (schedule.persona,)
            goals = (schedule.mission,)
            input_classes = ("response_driven_journey",)
            sequence_ids = (f"journey:{schedule.journey_id}",)
            factor_ids = tuple(schedule.allowed_risk_factors) or (
                f"journey:{schedule.journey_id}",
            )
        command = tuple(str(item) for item in factory(
            index, target_path, shard_runtime, session_id, target_seed,
            schedule.schedule_id,
        ))
        command = _bind_docker_execution_environment(
            command,
            execution_id=execution_id,
            container_receipt_dir=container_receipt_dir,
            repo_root=root,
        )
        if not command or any(not item for item in command):
            raise ValueError(f"shard {index:02d} has an invalid command")
        _reject_secret_bearing_command(command)
        shards.append(FrozenShardSpec(
            shard_id=shard_id,
            index=index,
            seed=target_seed,
            session_id=session_id,
            target_path=str(target_path),
            target_hash=target_hash,
            schedule_id=schedule.schedule_id,
            runtime_root=str(shard_runtime),
            execution_id=execution_id,
            container_cleanup_receipt_dir=str(container_receipt_dir),
            command=command,
            target_ids=target_ids,
            edge_keys=edge_keys,
            personas=personas,
            goals=goals,
            input_classes=input_classes,
            sequence_ids=sequence_ids,
            factor_ids=factor_ids,
            lane=lane,
            verifier_registry_import=verifier_registry_import,
            verifier_registry_id=verifier_registry_id,
            obligation_id=obligation_id,
            subject_group=subject_group,
            simulator_attestation_required=simulator_attestation_required,
        ))

    lane_counts = {lane: sum(item.lane == lane for item in shards) for lane in SHARD_LANES}
    if formal_profile and lane_counts != {
        "edge": FORMAL_EDGE_SHARD_COUNT,
        "journey": FORMAL_JOURNEY_SHARD_COUNT,
    }:
        raise ValueError(
            "formal batch requires exactly 24 edge shards and 8 journey shards"
        )
    journey_shards = tuple(
        shard for shard in shards if shard.lane == "journey"
    )
    obligation_ids = [
        shard.obligation_id for shard in journey_shards if shard.obligation_id
    ]
    schedule_ids = [shard.schedule_id for shard in journey_shards]
    if (
        len(obligation_ids) != len(set(obligation_ids))
        or len(schedule_ids) != len(set(schedule_ids))
    ):
        raise ValueError(
            "batch has duplicate Journey obligation or schedule identity"
        )
    observed_obligation_set_hash = _content_hash([
        {
            "obligation_id": shard.obligation_id,
            "schedule_id": shard.schedule_id,
            "seed": shard.seed,
            "subject_group": shard.subject_group,
        }
        for shard in sorted(
            journey_shards,
            key=lambda item: (
                item.obligation_id,
                item.schedule_id,
            ),
        )
    ])
    if (
        expected_obligation_set_hash
        and observed_obligation_set_hash
        != expected_obligation_set_hash
    ):
        raise ValueError(
            "batch Journey obligation set differs from the expected set"
        )

    frozen_signer = authority_signer or create_pty_authority_signer()
    unsigned = {
        "created_at_ns": time.time_ns(),
        "manifest_path": str(output),
        "discovery_ledger_path": str(discovery_path),
        "repo_root": str(root),
        "revision": revision,
        "shard_count": shard_count,
        "max_concurrency": resolved_max_concurrency,
        "pty_authority_trust_root_id": frozen_signer.trust_root_id,
        "pty_authority_public_key_b64": frozen_signer.public_key_b64,
        "shards": [_shard_payload(item) for item in shards],
        "required_env_names": sorted({str(item) for item in required_env_names}),
        "timeout_policy": asdict(timeout_policy),
        "stderr_cap_bytes": stderr_cap_bytes,
        "expected_obligation_set_hash": expected_obligation_set_hash,
        "worker_runtime": worker_runtime,
        "formal_profile": formal_profile,
        "controller_owned_execution": controller_owned_execution,
        "scheduler_schema_version": CHAOS_SCHEDULE_SCHEMA_VERSION,
        "discovery_schema_version": DISCOVERY_LEDGER_SCHEMA_VERSION,
        "schema_version": BATCH_MANIFEST_SCHEMA_VERSION,
    }
    batch_id = _content_hash(unsigned)
    identity = {**unsigned, "batch_id": batch_id}
    manifest = FrozenBatchManifest(
        manifest_id=_content_hash(identity),
        batch_id=batch_id,
        created_at_ns=int(unsigned["created_at_ns"]),
        manifest_path=str(output),
        discovery_ledger_path=str(discovery_path),
        repo_root=str(root),
        revision=revision,
        shard_count=shard_count,
        max_concurrency=resolved_max_concurrency,
        pty_authority_trust_root_id=frozen_signer.trust_root_id,
        pty_authority_public_key_b64=frozen_signer.public_key_b64,
        shards=tuple(shards),
        required_env_names=tuple(unsigned["required_env_names"]),
        timeout_policy=timeout_policy,
        stderr_cap_bytes=stderr_cap_bytes,
        expected_obligation_set_hash=expected_obligation_set_hash,
        worker_runtime=worker_runtime,
        formal_profile=formal_profile,
        controller_owned_execution=controller_owned_execution,
        _authority_signer=frozen_signer,
    )
    _write_immutable_json(output, manifest_payload(manifest))
    validate_frozen_manifest(manifest, manifest_path=output)
    return manifest


def load_frozen_manifest(path: str | Path) -> FrozenBatchManifest:
    payload = _load_json(Path(path))
    manifest = FrozenBatchManifest(
        manifest_id=str(payload["manifest_id"]),
        batch_id=str(payload["batch_id"]),
        created_at_ns=int(payload["created_at_ns"]),
        manifest_path=str(payload["manifest_path"]),
        discovery_ledger_path=str(payload["discovery_ledger_path"]),
        repo_root=str(payload["repo_root"]),
        revision=dict(payload["revision"]),
        shard_count=int(payload["shard_count"]),
        max_concurrency=int(payload["max_concurrency"]),
        pty_authority_trust_root_id=str(
            payload.get("pty_authority_trust_root_id") or ""
        ),
        pty_authority_public_key_b64=str(
            payload.get("pty_authority_public_key_b64") or ""
        ),
        shards=tuple(FrozenShardSpec(
            **{**row, "command": tuple(row["command"]),
               "target_ids": tuple(row["target_ids"]),
               "edge_keys": tuple(row["edge_keys"]),
               "personas": tuple(row["personas"]),
               "goals": tuple(row["goals"]),
               "input_classes": tuple(row["input_classes"]),
               "sequence_ids": tuple(row["sequence_ids"]),
               "factor_ids": tuple(row["factor_ids"])}
        ) for row in payload["shards"]),
        required_env_names=tuple(payload["required_env_names"]),
        timeout_policy=TimeoutPolicy(**payload["timeout_policy"]),
        stderr_cap_bytes=int(payload["stderr_cap_bytes"]),
        expected_obligation_set_hash=str(
            payload.get("expected_obligation_set_hash") or ""
        ),
        worker_runtime=str(payload.get("worker_runtime") or "docker"),
        formal_profile=bool(payload.get("formal_profile", False)),
        controller_owned_execution=bool(
            payload.get("controller_owned_execution", False)
        ),
        scheduler_schema_version=int(payload["scheduler_schema_version"]),
        discovery_schema_version=int(payload["discovery_schema_version"]),
        schema_version=int(payload["schema_version"]),
    )
    validate_frozen_manifest(manifest, manifest_path=Path(path))
    return manifest


def validate_frozen_manifest(
    manifest: FrozenBatchManifest,
    *,
    manifest_path: str | Path | None = None,
    verify_revision: bool = True,
) -> None:
    if manifest.schema_version != BATCH_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported batch manifest schema")
    if manifest.scheduler_schema_version != CHAOS_SCHEDULE_SCHEMA_VERSION:
        raise ValueError("scheduler schema changed after manifest creation")
    if manifest.discovery_schema_version != DISCOVERY_LEDGER_SCHEMA_VERSION:
        raise ValueError("discovery schema changed after manifest creation")
    if manifest.shard_count != len(manifest.shards):
        raise ValueError("manifest shard_count does not match its shard records")
    if (
        not isinstance(manifest.max_concurrency, int)
        or isinstance(manifest.max_concurrency, bool)
        or manifest.max_concurrency < 1
    ):
        raise ValueError("manifest max_concurrency must be a positive integer")
    if manifest.max_concurrency > manifest.shard_count:
        raise ValueError("manifest max_concurrency exceeds shard_count")
    if (
        not manifest.pty_authority_public_key_b64
        or _content_hash({
            "ed25519_public_key": manifest.pty_authority_public_key_b64,
        })
        != manifest.pty_authority_trust_root_id
    ):
        raise ValueError("manifest PTY authority trust root is invalid")
    if manifest.worker_runtime not in {"docker", "linux"}:
        raise ValueError("manifest worker runtime is invalid")
    if manifest.formal_profile and manifest.worker_runtime != "linux":
        raise ValueError("formal manifest escaped the Linux control plane")
    if manifest.formal_profile and not manifest.controller_owned_execution:
        raise ValueError(
            "formal manifest requires controller-owned execution"
        )
    if any(shard.lane not in SHARD_LANES for shard in manifest.shards):
        raise ValueError("manifest contains an unsupported shard lane")
    lane_counts = {
        lane: sum(shard.lane == lane for shard in manifest.shards)
        for lane in SHARD_LANES
    }
    if manifest.formal_profile and lane_counts != {
        "edge": FORMAL_EDGE_SHARD_COUNT,
        "journey": FORMAL_JOURNEY_SHARD_COUNT,
    }:
        raise ValueError("formal batch lane cardinality changed after manifest creation")
    indexes = [item.index for item in manifest.shards]
    if indexes != list(range(1, manifest.shard_count + 1)):
        raise ValueError("manifest shard indexes are not contiguous")
    for attribute in (
        "shard_id", "session_id", "runtime_root", "target_path", "execution_id",
        "container_cleanup_receipt_dir",
    ):
        values = [getattr(item, attribute) for item in manifest.shards]
        if len(values) != len(set(values)):
            raise ValueError(f"manifest has duplicate {attribute}")
    journey_shards = tuple(
        shard for shard in manifest.shards if shard.lane == "journey"
    )
    obligation_ids = [
        shard.obligation_id for shard in journey_shards if shard.obligation_id
    ]
    schedule_ids = [shard.schedule_id for shard in journey_shards]
    if (
        len(obligation_ids) != len(set(obligation_ids))
        or len(schedule_ids) != len(set(schedule_ids))
    ):
        raise ValueError(
            "manifest has duplicate Journey obligation or schedule identity"
        )
    observed_obligation_set_hash = _content_hash([
        {
            "obligation_id": shard.obligation_id,
            "schedule_id": shard.schedule_id,
            "seed": shard.seed,
            "subject_group": shard.subject_group,
        }
        for shard in sorted(
            journey_shards,
            key=lambda item: (
                item.obligation_id,
                item.schedule_id,
            ),
        )
    ])
    if (
        manifest.expected_obligation_set_hash
        and observed_obligation_set_hash
        != manifest.expected_obligation_set_hash
    ):
        raise ValueError(
            "manifest Journey obligation set differs from the expected set"
        )
    for shard in manifest.shards:
        target = Path(shard.target_path)
        if _sha256_file(target) != shard.target_hash:
            raise ValueError(f"frozen target changed: {target}")
        if not shard.execution_id.startswith("anychain-chaos-"):
            raise ValueError(f"shard execution id is invalid: {shard.shard_id}")
        expected_receipt_dir = Path(shard.runtime_root) / "container-cleanup-receipts"
        if Path(shard.container_cleanup_receipt_dir) != expected_receipt_dir:
            raise ValueError(f"container cleanup receipt directory changed: {shard.shard_id}")
        _validate_bound_docker_environment(shard, Path(manifest.repo_root))
        if manifest.formal_profile:
            _validate_formal_linux_command(
                shard,
                Path(manifest.repo_root),
                broker_decision_timeout_seconds=manifest.timeout_policy.decision_seconds,
            )
    ledger = build_ledger(revision=manifest.revision)
    for shard in manifest.shards:
        target_payload = _load_json(Path(shard.target_path))
        if _target_lane(target_payload) != shard.lane:
            raise ValueError(f"frozen shard lane changed: {shard.shard_id}")
        if shard.lane == "edge":
            schedule = build_chaos_schedule(
                ledger,
                revision=manifest.revision,
                seed=shard.seed,
                targets=_target_rows(target_payload),
            )
            validate_chaos_schedule(schedule, ledger=ledger, revision=manifest.revision)
            if shard.verifier_registry_import or shard.verifier_registry_id:
                raise ValueError("edge shard cannot bind a Journey verifier registry")
        else:
            journey, registry_import = _journey_target(target_payload)
            frozen_execution = _frozen_journey_execution(target_payload)
            schedule = build_journey_schedule(
                revision=manifest.revision,
                seed=shard.seed,
                journey=journey,
            )
            validate_journey_schedule(schedule, revision=manifest.revision)
            if schedule.subject_group != shard.subject_group:
                raise ValueError("frozen Journey subject group changed")
            if frozen_execution:
                if int(frozen_execution["seed"]) != shard.seed:
                    raise ValueError("frozen Journey seed changed")
                if str(frozen_execution["obligation_id"]) != shard.obligation_id:
                    raise ValueError("frozen Journey obligation changed")
                if str(frozen_execution["schedule_id"]) != shard.schedule_id:
                    raise ValueError("frozen Journey schedule binding changed")
                if str(frozen_execution["schedule_hash"]) != _content_hash(
                    journey_schedule_payload(schedule)
                ):
                    raise ValueError("frozen Journey schedule hash changed")
                if str(
                    frozen_execution["subject_group"]
                ) != shard.subject_group:
                    raise ValueError(
                        "frozen Journey subject group changed"
                    )
            elif shard.obligation_id:
                raise ValueError("Journey shard retained an unbound obligation id")
            if _simulator_attestation_required(
                target_payload
            ) != shard.simulator_attestation_required:
                raise ValueError("Journey simulator attestation policy changed")
            if registry_import != shard.verifier_registry_import:
                raise ValueError("frozen Journey verifier import changed")
            if load_verifier_registry(registry_import).registry_id != shard.verifier_registry_id:
                raise ValueError("frozen Journey verifier registry identity changed")
        if schedule.schedule_id != shard.schedule_id:
            raise ValueError(f"frozen scheduler identity changed: {shard.shard_id}")
    unsigned = _manifest_unsigned_payload(manifest)
    if _content_hash(unsigned) != manifest.batch_id:
        raise ValueError("batch manifest content hash is stale")
    if _content_hash({**unsigned, "batch_id": manifest.batch_id}) != manifest.manifest_id:
        raise ValueError("batch manifest identity is stale")
    if verify_revision and repository_revision(manifest.repo_root) != dict(manifest.revision):
        raise ValueError("repository revision changed after manifest freeze")
    path = Path(manifest_path or manifest.manifest_path).resolve()
    if path != Path(manifest.manifest_path).resolve():
        raise ValueError("manifest path does not match its frozen identity")
    if not path.is_file() or path.stat().st_mode & 0o222:
        raise ValueError("frozen manifest must exist and be read-only")
    if _load_json(path) != manifest_payload(manifest):
        raise ValueError("on-disk manifest does not match the in-memory manifest")


async def run_batch(
    manifest: FrozenBatchManifest | str | Path,
    *,
    broker: DecisionBroker,
    result_index_path: str | Path,
    verify_revision: bool = True,
    interruption_event: asyncio.Event | None = None,
    authority_signer: PtyAuthoritySigner | None = None,
) -> BatchResultIndex:
    """Run every frozen shard once and wait for every shard to terminate."""

    frozen = load_frozen_manifest(manifest) if isinstance(manifest, (str, Path)) else manifest
    _require_linux_control_plane()
    validate_frozen_manifest(frozen, verify_revision=verify_revision)
    authority_signer = authority_signer or frozen._authority_signer
    if authority_signer is None:
        raise ValueError(
            "a loaded batch manifest requires its controller authority signer"
        )
    if (
        authority_signer.trust_root_id != frozen.pty_authority_trust_root_id
        or authority_signer.public_key_b64
        != frozen.pty_authority_public_key_b64
    ):
        raise ValueError("controller signer differs from the frozen trust root")
    index_path = Path(result_index_path).resolve()
    if index_path.exists():
        raise FileExistsError(f"batch result index is immutable: {index_path}")

    semaphore = asyncio.Semaphore(frozen.max_concurrency)

    async def run_bounded(shard: FrozenShardSpec) -> ShardResult:
        try:
            async with semaphore:
                if (
                    interruption_event is not None
                    and interruption_event.is_set()
                ):
                    return _not_started_interruption_result(
                        frozen,
                        shard,
                        asyncio.CancelledError(
                            "batch interruption preceded shard admission"
                        ),
                    )
                runner = (
                    _run_controller_owned_shard
                    if frozen.controller_owned_execution
                    else _run_shard
                )
                return await runner(
                    frozen, shard, broker, authority_signer
                )
        except asyncio.CancelledError as exc:
            return _not_started_interruption_result(frozen, shard, exc)

    tasks = [
        asyncio.create_task(run_bounded(shard))
        for shard in frozen.shards
    ]
    batch_interrupted = False
    try:
        gather = asyncio.gather(*tasks, return_exceptions=True)
        if interruption_event is None:
            raw_results = await gather
        else:
            interruption_wait = asyncio.create_task(interruption_event.wait())
            try:
                done, _pending = await asyncio.wait(
                    (gather, interruption_wait),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if interruption_wait in done and interruption_event.is_set():
                    batch_interrupted = True
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                raw_results = await gather
            finally:
                if not interruption_wait.done():
                    interruption_wait.cancel()
                await asyncio.gather(interruption_wait, return_exceptions=True)
    except asyncio.CancelledError:
        batch_interrupted = True
        for task in tasks:
            if not task.done():
                task.cancel()
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)
    results: list[ShardResult] = []
    for shard, raw in zip(frozen.shards, raw_results, strict=True):
        if isinstance(raw, BaseException):
            results.append(_synthetic_interruption_result(frozen, shard, raw))
        else:
            results.append(raw)
    validated_results: list[ShardResult] = []
    for shard, result in zip(frozen.shards, results, strict=True):
        try:
            _validate_composite_cleanup_receipt(shard, result)
        except Exception as exc:
            result = replace(
                result,
                classification="infrastructure_interrupted",
                reason=f"cleanup receipt validation failed: {type(exc).__name__}: {exc}",
            )
        validated_results.append(result)
    results = validated_results

    batch_survivor_proof = await _final_batch_survivor_proof(frozen)
    if not batch_survivor_proof["cleaned"]:
        failed_execution_ids = {
            str(row.get("execution_id") or "")
            for scan in batch_survivor_proof["scans"]
            for row in scan["survivors"]
        }
        fail_all = bool(batch_survivor_proof["errors"])
        results = [
            replace(
                result,
                classification="infrastructure_interrupted",
                reason="final batch-wide execution-id survivor proof failed",
            )
            if fail_all or shard.execution_id in failed_execution_ids
            else result
            for shard, result in zip(frozen.shards, results, strict=True)
        ]
    validate_frozen_manifest(frozen, verify_revision=verify_revision)
    attempt_ids = _append_discovery_results(frozen, results)
    counts = {name: 0 for name in sorted(RESULT_CLASSIFICATIONS)}
    for result in results:
        counts[result.classification] += 1
    cleanup_complete = bool(batch_survivor_proof["cleaned"]) and all(
        _load_json(Path(result.cleanup_receipt_path)).get("cleaned") is True
        for result in results
    )
    execution_status = (
        "discovery_complete"
        if cleanup_complete and not batch_interrupted
        else "infrastructure_interrupted"
    )
    unsigned = {
        "batch_id": frozen.batch_id,
        "manifest_id": frozen.manifest_id,
        "revision": dict(frozen.revision),
        "execution_authority_mode": (
            "controller_owned_v1"
            if frozen.controller_owned_execution
            else "diagnostic_worker_v1"
        ),
        "execution_status": execution_status,
        "release_status": "not_evaluated",
        "scheduled": frozen.shard_count,
        "started": frozen.shard_count,
        "completed": len(results),
        "discovery_attempt_ids": list(attempt_ids),
        "classification_counts": counts,
        "batch_survivor_proof": batch_survivor_proof,
        "pty_authority_trust_root_id": authority_signer.trust_root_id,
        "pty_authority_public_key_b64": authority_signer.public_key_b64,
        "shards": [_result_payload(item) for item in results],
        "schema_version": BATCH_RESULT_SCHEMA_VERSION,
    }
    controller_signature_b64 = sign_controller_payload(
        authority_signer,
        unsigned,
    )
    index = BatchResultIndex(
        index_id=_content_hash({
            **unsigned,
            "controller_signature_b64": controller_signature_b64,
        }),
        batch_id=frozen.batch_id,
        manifest_id=frozen.manifest_id,
        revision=dict(frozen.revision),
        execution_authority_mode=unsigned["execution_authority_mode"],
        execution_status=execution_status,
        release_status="not_evaluated",
        scheduled=frozen.shard_count,
        started=frozen.shard_count,
        completed=len(results),
        discovery_attempt_ids=tuple(attempt_ids),
        classification_counts=counts,
        batch_survivor_proof=batch_survivor_proof,
        pty_authority_trust_root_id=authority_signer.trust_root_id,
        pty_authority_public_key_b64=authority_signer.public_key_b64,
        controller_signature_b64=controller_signature_b64,
        shards=tuple(results),
    )
    _write_immutable_json(index_path, result_index_payload(index))
    return index


async def _run_shard(
    manifest: FrozenBatchManifest,
    shard: FrozenShardSpec,
    broker: DecisionBroker,
    authority_signer: PtyAuthoritySigner,
) -> ShardResult:
    started = time.time_ns()
    runtime_root = Path(shard.runtime_root)
    runtime_root.mkdir(parents=True, exist_ok=False)
    stderr_path = runtime_root / "worker.stderr.redacted.txt"
    receipt_path = runtime_root / "cleanup-receipt.json"
    host_receipt_dir = runtime_root / "host-cleanup-receipts"
    host_guard = ContainerProcessGuard(
        shard.execution_id,
        receipt_dir=host_receipt_dir,
        term_grace_seconds=max(manifest.timeout_policy.cleanup_seconds / 2, 0.01),
        kill_grace_seconds=max(manifest.timeout_policy.cleanup_seconds / 2, 0.01),
        scan_interval_seconds=min(
            0.05, max(manifest.timeout_policy.cleanup_seconds / 20, 0.005)
        ),
    )
    state = _RunState()
    process: asyncio.subprocess.Process | None = None
    worker_identity: Mapping[str, Any] = {}
    stderr_task: asyncio.Task[tuple[bytes, bool]] | None = None
    exit_code: int | None = None
    cleanup_outcome = _HostCleanupOutcome(None, False, None, (), ())
    stderr_bytes = b""
    stderr_truncated = False
    try:
        worker_env = os.environ.copy()
        worker_env[EXECUTION_ID_ENV] = shard.execution_id
        worker_env[RECEIPT_DIR_ENV] = shard.container_cleanup_receipt_dir
        process = await asyncio.create_subprocess_exec(
            *shard.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=manifest.repo_root,
            start_new_session=True,
            env=worker_env,
        )
        identity = host_guard.register_pid(process.pid, role="host_worker")
        if _is_docker_exec_command(shard.command):
            host_guard.register_pid(process.pid, role="docker_exec")
        worker_identity = identity.payload()
        assert process.stderr is not None
        stderr_task = asyncio.create_task(
            _read_capped(process.stderr, manifest.stderr_cap_bytes)
        )
        await asyncio.wait_for(
            _exchange_frames(
                process, manifest, shard, broker, state, manifest.timeout_policy
            ),
            timeout=manifest.timeout_policy.shard_seconds,
        )
        exit_code = await asyncio.wait_for(
            process.wait(), timeout=manifest.timeout_policy.shard_seconds
        )
    except ExternalDecisionBlocked as exc:
        state.forced_classification = "externally_blocked"
        state.reason = str(exc)
    except _SimulatorInvalid as exc:
        state.forced_classification = "simulator_invalid"
        state.reason = str(exc)
    except asyncio.TimeoutError:
        state.forced_classification = "infrastructure_interrupted"
        state.reason = "shard wall-clock timeout"
    except asyncio.CancelledError:
        state.forced_classification = "infrastructure_interrupted"
        state.reason = "batch cancellation requested"
    except Exception as exc:
        state.forced_classification = "infrastructure_interrupted"
        state.reason = f"{type(exc).__name__}: {exc}"
    finally:
        cleanup_outcome = await _cleanup_process(
            process,
            host_guard,
            manifest.timeout_policy.cleanup_seconds,
        )
        exit_code = cleanup_outcome.exit_code
        if stderr_task is not None:
            try:
                stderr_bytes, stderr_truncated = await stderr_task
            except Exception as exc:
                state.forced_classification = "infrastructure_interrupted"
                state.reason = f"stderr collection failed: {type(exc).__name__}: {exc}"
        if not cleanup_outcome.cleaned:
            state.forced_classification = "infrastructure_interrupted"
            state.reason = "host worker cleanup could not prove process termination"

    redacted_stderr = str(redact(stderr_bytes.decode("utf-8", errors="replace")))
    persisted_stderr = redacted_stderr.encode("utf-8")
    if len(persisted_stderr) > manifest.stderr_cap_bytes:
        persisted_stderr = persisted_stderr[:manifest.stderr_cap_bytes]
        stderr_truncated = True
    _atomic_write(stderr_path, persisted_stderr, mode=0o600)
    cleanup_errors = list(cleanup_outcome.errors)
    host_proof: Mapping[str, Any] = {}
    container_proof: Mapping[str, Any] = {}
    if cleanup_outcome.artifact is not None:
        try:
            host_proof = _validated_guard_proof(
                cleanup_outcome.artifact.path,
                execution_id=shard.execution_id,
                required_roles=(
                    ("host_worker", "docker_exec")
                    if _is_docker_exec_command(shard.command)
                    else ("host_worker",)
                ),
                allowed_roots=(host_receipt_dir,),
            )
        except Exception as exc:
            cleanup_errors.append(f"host proof invalid: {type(exc).__name__}: {exc}")
    else:
        cleanup_errors.append("host process guard did not produce a receipt")
    try:
        inner_path = _single_guard_receipt_path(
            Path(shard.container_cleanup_receipt_dir)
        )
        container_proof = _validated_guard_proof(
            inner_path,
            execution_id=shard.execution_id,
            required_roles=("container_bridge", "agent_process_group_leader"),
            allowed_roots=(Path(shard.container_cleanup_receipt_dir),),
        )
    except Exception as exc:
        cleanup_errors.append(f"container proof invalid: {type(exc).__name__}: {exc}")

    composite_cleaned = (
        cleanup_outcome.cleaned
        and bool(host_proof.get("cleaned"))
        and bool(container_proof.get("cleaned"))
        and not cleanup_errors
    )
    receipt_unsigned = {
        "schema_version": CLEANUP_RECEIPT_SCHEMA_VERSION,
        "shard_id": shard.shard_id,
        "execution_id": shard.execution_id,
        "host_proof": host_proof,
        "container_proof": container_proof,
        "worker_identity": dict(worker_identity),
        "exit_code": exit_code,
        "actions": list(cleanup_outcome.actions),
        "errors": cleanup_errors,
        "cleaned": composite_cleaned,
        "started_at_ns": started,
        "finished_at_ns": time.time_ns(),
    }
    receipt = CleanupReceipt(
        receipt_id=_content_hash(receipt_unsigned),
        schema_version=CLEANUP_RECEIPT_SCHEMA_VERSION,
        shard_id=shard.shard_id,
        execution_id=shard.execution_id,
        host_proof=host_proof,
        container_proof=container_proof,
        worker_identity=worker_identity,
        exit_code=exit_code,
        actions=cleanup_outcome.actions,
        errors=tuple(cleanup_errors),
        cleaned=receipt_unsigned["cleaned"],
        started_at_ns=started,
        finished_at_ns=receipt_unsigned["finished_at_ns"],
    )
    _write_immutable_json(receipt_path, asdict(receipt))
    artifacts = _artifact_hashes(
        manifest,
        state.result_payload,
        shard,
        authority_signer=authority_signer,
        observed_response_hashes=tuple(state.response_hashes),
        observed_decision_hashes=tuple(state.decision_hashes),
        controller_turn_observations=tuple(
            state.controller_turn_observations
        ),
        controller_turn_facts=_controller_fact_payloads(
            state,
            lane=shard.lane,
        ),
        controller_initial_facts=_controller_fact_payloads(
            state,
            lane="journey_initial",
        ),
    )
    classification, reason = _classify(
        manifest, shard, state, exit_code, receipt.cleaned, artifacts
    )
    diagnostic_ids = tuple(artifacts["diagnostic_ids"]) or (receipt.receipt_id,)
    return ShardResult(
        shard_id=shard.shard_id,
        classification=classification,
        attempt_count=1,
        started_at_ns=started,
        finished_at_ns=time.time_ns(),
        exit_code=exit_code,
        target_hash=shard.target_hash,
        response_hashes=tuple(state.response_hashes),
        decision_hashes=tuple(state.decision_hashes),
        transcript_hash=artifacts["transcript_hash"],
        schedule_result_hash=artifacts["schedule_result_hash"],
        evidence_hashes=tuple(artifacts["evidence_hashes"]),
        diagnostic_hashes=tuple(artifacts["diagnostic_hashes"]),
        evidence_ids=tuple(artifacts["evidence_ids"]),
        diagnostic_ids=diagnostic_ids,
        stderr_hash=_sha256_file(stderr_path),
        stderr_path=str(stderr_path),
        stderr_truncated=stderr_truncated,
        cleanup_receipt_path=str(receipt_path),
        cleanup_receipt_hash=_sha256_file(receipt_path),
        reason=str(redact(reason))[:1000],
    )


def _controller_observations_from_edge_result(
    *,
    result: Any,
    state: _RunState,
    edge_index: Mapping[str, Mapping[str, Any]],
    input_commitment_key: bytes,
) -> tuple[Mapping[str, Any], ...]:
    """Freeze exact in-process PTY facts before any artifact admission."""

    observations: list[Mapping[str, Any]] = []
    dynamic_paths = []
    for path in tuple(result.evidence_paths):
        artifact = _load_json(Path(path))
        if (
            artifact.get("artifact_type") == "pty_cli_turn"
            and artifact.get("evidence_class") == "dynamic_dual_ai"
        ):
            dynamic_paths.append((Path(path), artifact))
    trusted_facts = _controller_fact_payloads(state, lane="edge")
    if (
        len(dynamic_paths) != len(trusted_facts)
    ):
        raise ValueError(
            "controller-owned PTY run did not produce one observation and "
            "candidate per approved simulator decision"
        )
    previous_receipt_hash = ""
    for ordinal, (_path, artifact) in enumerate(dynamic_paths):
        trusted = trusted_facts[ordinal]
        trusted_turn = _turn_from_payload(trusted["turn"])
        trusted_observation = _turn_observation_from_payload(
            trusted["observation"]
        )
        trusted_edge = dict(trusted["edge"])
        trusted_selection = _selection_from_payload(trusted["selection"])
        terminal_outcomes = tuple(
            _terminal_outcome_from_mapping(item)
            for item in trusted["terminal_outcomes"]
        )
        submission_bindings = _approved_submission_bindings(
            state,
            (
                trusted_turn.user_message,
                *(
                    item.user_message
                    for item in trusted_observation.continuation_turns
                ),
            ),
            input_commitment_key=input_commitment_key,
        )
        canonical_facts = canonical_controller_turn_facts(
            edge=trusted_edge,
            turn=trusted_turn,
            observation=trusted_observation,
            dynamic_selection=trusted_selection,
        )
        trusted_turn_payload = dict(canonical_facts["turn"])
        trusted_observation_payload = dict(
            canonical_facts["turn_observation"]
        )
        turn_payload = dict(artifact.get("turn") or {})
        turn_index = int(trusted_turn.turn_index)
        if (
            int(turn_payload.get("turn_index") or 0) != turn_index
            or str(
                trusted_turn_payload.get("source_previous_response_hash") or ""
            )
            != str(turn_payload.get("source_previous_response_hash") or "")
            or str(
                trusted_turn_payload.get("source_agent_response_hash") or ""
            )
            != str(turn_payload.get("source_agent_response_hash") or "")
        ):
            raise ValueError(
                "candidate response differs from the controller-owned PTY"
            )
        raw_observation = dict(artifact.get("turn_observation") or {})
        if _content_hash(raw_observation) != _content_hash(
            trusted_observation_payload
        ):
            raise ValueError(
                "candidate turn observation differs from controller memory"
            )
        edge_key = str(trusted_edge.get("edge_key") or "")
        if str(artifact.get("edge_key") or "") != edge_key:
            raise ValueError("candidate edge differs from controller memory")
        edge = edge_index.get(edge_key)
        if edge is None:
            raise ValueError(
                "controller-owned candidate references an unknown edge"
            )
        baseline, committed, *continuations = trusted_observation.runtime_events
        recomputed = verify_declared_postconditions(
            baseline,
            committed,
            trusted_turn,
            edge_index=edge_index,
            target_coverage_ids=(
                trusted_selection.target_coverage_ids
            ),
            continuation_events=continuations,
        )
        if _content_hash(asdict(recomputed)) != _content_hash(
            trusted_observation_payload.get("verified_postcondition") or {}
        ):
            raise ValueError(
                "candidate postcondition differs from controller recomputation"
            )
        runtime_hashes = [
            str(event.runtime_event_payload_hash or "")
            for event in trusted_observation.runtime_events
        ]
        if not runtime_hashes or any(
            len(value) != 64 for value in runtime_hashes
        ):
            raise ValueError(
                "controller-owned runtime observation is incomplete"
            )
        unsigned_receipt = {
            "shard_turn_ordinal": ordinal + 1,
            "turn_index": turn_index,
            "edge_key": edge_key,
            "previous_response_hash": str(
                trusted_turn_payload.get("source_previous_response_hash") or ""
            ),
            "simulator_decision_hash": _content_hash(
                trusted_observation_payload.get("simulator_decision") or {}
            ),
            "approved_decision_hashes": [
                str(item["decision_hash"])
                for item in submission_bindings
            ],
            "simulator_context_hashes": [
                str(item["context_hash"])
                for item in submission_bindings
            ],
            "submission_sequences": [
                int(item["submission_sequence"])
                for item in submission_bindings
            ],
            "user_message_hash": str(
                trusted_turn_payload.get("source_user_message_hash") or ""
            ),
            "submitted_input_commitment": str(
                submission_bindings[0][
                    "submitted_input_commitment"
                ]
            ),
            "agent_response_hash": str(
                trusted_turn_payload.get("source_agent_response_hash") or ""
            ),
            "runtime_event_payload_hashes": runtime_hashes,
            "terminal_runtime_event_id": str(
                trusted_observation.runtime_events[-1].runtime_event_id or ""
            ),
            "terminal_runtime_event_sequence": (
                trusted_observation.runtime_events[-1].runtime_event_sequence
            ),
            "terminal_outcome_hashes": [
                _content_hash(asdict(outcome))
                for outcome in terminal_outcomes
            ],
            "verified_postcondition_hash": _content_hash(
                asdict(recomputed)
            ),
            "turn_observation_hash": _content_hash(
                trusted_observation_payload
            ),
            "candidate_artifact_hash": str(
                artifact.get("artifact_hash") or ""
            ),
            "previous_controller_observation_hash": (
                previous_receipt_hash
            ),
        }
        receipt_hash = _content_hash(unsigned_receipt)
        observations.append({
            **unsigned_receipt,
            "controller_observation_hash": receipt_hash,
        })
        previous_receipt_hash = receipt_hash
    return tuple(observations)


def _controller_observations_from_journey_result(
    *,
    result: Any,
    state: _RunState,
    input_commitment_key: bytes,
) -> tuple[Mapping[str, Any], ...]:
    """Bind Journey candidate rows to controller-owned in-memory boundaries."""

    artifact = _load_json(Path(result.evidence_path))
    candidate_turns = list(artifact.get("turns") or ())
    trusted_facts = list(
        _controller_fact_payloads(state, lane="journey")
    )
    if len(candidate_turns) != len(trusted_facts):
        raise ValueError(
            "controller-owned Journey has an incomplete observation ledger"
        )
    observations: list[Mapping[str, Any]] = []
    submission_bindings = _approved_submission_bindings(
        state,
        [str(item["turn"]["user_message"]) for item in trusted_facts],
        input_commitment_key=input_commitment_key,
    )
    previous_receipt_hash = ""
    for ordinal, (candidate, trusted, binding) in enumerate(
        zip(
            candidate_turns,
            trusted_facts,
            submission_bindings,
            strict=True,
        )
    ):
        trusted_result = dict(trusted["turn_result"])
        if _content_hash(candidate) != _content_hash(trusted_result):
            raise ValueError(
                "Journey candidate differs from controller memory"
            )
        turn = _turn_from_payload(trusted["turn"])
        baseline = _runtime_event_from_mapping(trusted["baseline_event"])
        committed = _runtime_event_from_mapping(trusted["committed_event"])
        terminal_outcome = _terminal_outcome_from_mapping(
            trusted["terminal_outcome"]
        )
        unsigned = {
            "shard_turn_ordinal": ordinal + 1,
            "turn_index": int(turn.turn_index),
            "previous_response_hash": _content_hash(
                turn.previous_agent_response
            ),
            "user_message_hash": _content_hash(turn.user_message),
            "approved_decision_hash": str(binding["decision_hash"]),
            "simulator_context_hash": str(binding["context_hash"]),
            "approved_decision": dict(binding["decision"]),
            "simulator_context": dict(binding["context"]),
            "submission_sequence": int(binding["submission_sequence"]),
            "submitted_input_commitment": str(
                binding["submitted_input_commitment"]
            ),
            "agent_response_hash": _content_hash(turn.agent_response),
            "baseline_runtime_event_hash": str(
                baseline.runtime_event_payload_hash or ""
            ),
            "committed_runtime_event_hash": str(
                committed.runtime_event_payload_hash or ""
            ),
            "terminal_runtime_event_id": str(
                committed.runtime_event_id or ""
            ),
            "terminal_runtime_event_sequence": (
                committed.runtime_event_sequence
            ),
            "terminal_outcome_hash": _content_hash(
                asdict(terminal_outcome)
            ),
            "turn_result_hash": _content_hash(trusted_result),
            "previous_controller_observation_hash": (
                previous_receipt_hash
            ),
        }
        receipt_hash = _content_hash(unsigned)
        observations.append({
            **unsigned,
            "controller_observation_hash": receipt_hash,
        })
        previous_receipt_hash = receipt_hash
    return tuple(observations)


def _execute_controller_owned_runner(
    *,
    manifest: FrozenBatchManifest,
    shard: FrozenShardSpec,
    broker: DecisionBroker,
    state: _RunState,
    loop: asyncio.AbstractEventLoop,
    transport: _ControllerOwnedPtyTransport,
    input_commitment_key: bytes,
) -> None:
    """Run the real Agent PTY inside the controller process."""

    target_payload = _load_json(Path(shard.target_path))
    ledger = build_ledger(revision=manifest.revision)
    adapter = _ControllerDecisionAdapter(
        loop=loop,
        broker=broker,
        manifest=manifest,
        shard=shard,
        state=state,
    )
    runtime_root = Path(shard.runtime_root)
    extra_env = {
        EXECUTION_ID_ENV: shard.execution_id,
        RECEIPT_DIR_ENV: shard.container_cleanup_receipt_dir,
    }
    if shard.lane == "edge":
        schedule = build_chaos_schedule(
            ledger,
            revision=manifest.revision,
            seed=shard.seed,
            targets=_target_rows(target_payload),
        )
        max_turns = sum(
            1 + int(target.continuation_turn_budget)
            for target in schedule.targets
        )
        config = ChaosRunConfig.linux(
            manifest.repo_root,
            session_id=shard.session_id,
            execution_id=shard.execution_id,
            max_turns=max_turns,
            response_timeout_seconds=manifest.timeout_policy.shard_seconds,
            runtime_root=runtime_root,
            runtime_root_in_process=runtime_root,
            extra_env=extra_env,
        )
        runner = DynamicDualAiChaosRunner(
            config,
            adapter,
            ledger=ledger,
            schedule=schedule,
            transport=transport,
            controller_turn_observer=_ControllerTurnLedger(state),
            revision=manifest.revision,
        )
        result = runner.run()
        state.controller_turn_observations.extend(
            _controller_observations_from_edge_result(
                result=result,
                state=state,
                edge_index={
                    str(edge.get("edge_key") or ""): edge
                    for edge in ledger.get("edges") or ()
                },
                input_commitment_key=input_commitment_key,
            )
        )
        state.result_payload = {
            "session_id": result.session_id,
            "execution_status": result.execution_status,
            "terminal_classification": (
                SimulatorTerminalClassification.PASSED.value
            ),
            "failure_reason": "",
            "schedule_path": str(result.schedule_path),
            "schedule_result_path": str(result.schedule_result_path),
            "transcript_path": str(result.transcript_path),
            "evidence_paths": [
                str(path) for path in result.evidence_paths
            ],
        }
        return

    journey, registry_import = _journey_target(target_payload)
    schedule = build_journey_schedule(
        revision=manifest.revision,
        seed=shard.seed,
        journey=journey,
    )
    registry = load_verifier_registry(registry_import)
    config = ChaosRunConfig.linux(
        manifest.repo_root,
        session_id=shard.session_id,
        execution_id=shard.execution_id,
        max_turns=schedule.max_turns,
        response_timeout_seconds=manifest.timeout_policy.shard_seconds,
        runtime_root=runtime_root,
        runtime_root_in_process=runtime_root,
        extra_env=extra_env,
    )
    runner = DynamicDualAiJourneyRunner(
        config,
        adapter,
        ledger=ledger,
        schedule=schedule,
        postcondition_verifier_registry=registry,
        transport=transport,
        controller_turn_observer=_ControllerJourneyLedger(
            state,
            schedule=schedule,
            registry=registry,
            edge_index={
                str(edge.get("edge_key") or ""): edge
                for edge in ledger.get("edges") or ()
            },
        ),
        revision=manifest.revision,
    )
    result = runner.run()
    state.controller_turn_observations.extend(
        _controller_observations_from_journey_result(
            result=result,
            state=state,
            input_commitment_key=input_commitment_key,
        )
    )
    state.result_payload = {
        "session_id": result.session_id,
        "terminal_classification": result.terminal_classification.value,
        "schedule_id": schedule.schedule_id,
        "schedule_path": str(result.schedule_path),
        "journey_result_path": str(result.journey_result_path),
        "transcript_path": str(result.transcript_path),
        "evidence_id": result.evidence_id,
        "evidence_path": str(result.evidence_path),
    }


async def _run_controller_owned_shard(
    manifest: FrozenBatchManifest,
    shard: FrozenShardSpec,
    broker: DecisionBroker,
    authority_signer: PtyAuthoritySigner,
) -> ShardResult:
    """Produce qualifying evidence without a worker-owned observation path."""

    started = time.time_ns()
    runtime_root = Path(shard.runtime_root)
    runtime_root.mkdir(parents=True, exist_ok=False)
    stderr_path = runtime_root / "controller-runner.stderr.redacted.txt"
    receipt_path = runtime_root / "cleanup-receipt.json"
    host_receipt_dir = runtime_root / "host-cleanup-receipts"
    host_guard = ContainerProcessGuard(
        shard.execution_id,
        receipt_dir=host_receipt_dir,
        term_grace_seconds=max(
            manifest.timeout_policy.cleanup_seconds / 2,
            0.01,
        ),
        kill_grace_seconds=max(
            manifest.timeout_policy.cleanup_seconds / 2,
            0.01,
        ),
        scan_interval_seconds=min(
            0.05,
            max(manifest.timeout_policy.cleanup_seconds / 20, 0.005),
        ),
    )
    config = ChaosRunConfig.linux(
        manifest.repo_root,
        session_id=shard.session_id,
        execution_id=shard.execution_id,
        runtime_root=runtime_root,
        runtime_root_in_process=runtime_root,
        extra_env={
            EXECUTION_ID_ENV: shard.execution_id,
            RECEIPT_DIR_ENV: shard.container_cleanup_receipt_dir,
        },
    )
    state = _RunState()
    transport = _ControllerOwnedPtyTransport(
        config.command,
        cwd=config.repo_root,
        poll_interval_seconds=config.poll_interval_seconds,
        execution_id=shard.execution_id,
        cleanup_receipt_dir=shard.container_cleanup_receipt_dir,
        process_guard=host_guard,
        state=state,
    )
    input_commitment_key = os.urandom(32)
    loop = asyncio.get_running_loop()
    runner_task = asyncio.create_task(asyncio.to_thread(
        _execute_controller_owned_runner,
        manifest=manifest,
        shard=shard,
        broker=broker,
        state=state,
        loop=loop,
        transport=transport,
        input_commitment_key=input_commitment_key,
    ))
    exit_code: int | None = None
    try:
        await asyncio.wait_for(
            asyncio.shield(runner_task),
            timeout=manifest.timeout_policy.shard_seconds,
        )
        exit_code = 0
    except ExternalDecisionBlocked as exc:
        state.forced_classification = "externally_blocked"
        state.reason = str(exc)
    except (SimulatorDecisionInvalid, JourneySimulatorInvalidError) as exc:
        state.forced_classification = "simulator_invalid"
        state.reason = str(exc)
    except JourneyRunError as exc:
        state.forced_classification = exc.classification.value
        state.reason = str(exc)
    except asyncio.TimeoutError:
        state.forced_classification = "infrastructure_interrupted"
        state.reason = "controller-owned shard wall-clock timeout"
    except BaseException as exc:
        state.forced_classification = "infrastructure_interrupted"
        state.reason = f"{type(exc).__name__}: {exc}"
    finally:
        if not runner_task.done():
            try:
                transport.close()
            except BaseException:
                pass
            try:
                await asyncio.wait_for(
                    asyncio.shield(runner_task),
                    timeout=manifest.timeout_policy.cleanup_seconds,
                )
            except BaseException:
                runner_task.cancel()

    process = transport._process
    reapers = (
        {process.pid: process.poll}
        if process is not None
        else {}
    )
    cleanup_errors: list[str] = []
    host_proof: Mapping[str, Any] = {}
    container_proof: Mapping[str, Any] = {}
    try:
        host_artifact = await asyncio.to_thread(
            host_guard.cleanup,
            reapers=reapers,
        )
        host_proof = _validated_guard_proof(
            host_artifact.path,
            execution_id=shard.execution_id,
            required_roles=("host_worker", "controller_pty_bridge"),
            allowed_roots=(host_receipt_dir,),
        )
    except BaseException as exc:
        cleanup_errors.append(
            f"controller process guard failed: {type(exc).__name__}: {exc}"
        )
    try:
        inner_path = _single_guard_receipt_path(
            Path(shard.container_cleanup_receipt_dir)
        )
        container_proof = _validated_guard_proof(
            inner_path,
            execution_id=shard.execution_id,
            required_roles=(
                "container_bridge",
                "agent_process_group_leader",
            ),
            allowed_roots=(
                Path(shard.container_cleanup_receipt_dir),
            ),
        )
    except BaseException as exc:
        cleanup_errors.append(
            f"Agent process guard failed: {type(exc).__name__}: {exc}"
        )
    cleaned = bool(
        host_proof.get("cleaned")
        and container_proof.get("cleaned")
        and not cleanup_errors
    )
    receipt_unsigned = {
        "schema_version": CLEANUP_RECEIPT_SCHEMA_VERSION,
        "shard_id": shard.shard_id,
        "execution_id": shard.execution_id,
        "host_proof": host_proof,
        "container_proof": container_proof,
        "worker_identity": dict(transport.controller_worker_identity),
        "exit_code": exit_code,
        "actions": ["controller_owned_runner_reaped"],
        "errors": cleanup_errors,
        "cleaned": cleaned,
        "started_at_ns": started,
        "finished_at_ns": time.time_ns(),
    }
    receipt = {
        "receipt_id": _content_hash(receipt_unsigned),
        **receipt_unsigned,
    }
    _write_immutable_json(receipt_path, receipt)
    _atomic_write(stderr_path, b"", mode=0o600)
    artifacts = _artifact_hashes(
        manifest,
        state.result_payload,
        shard,
        authority_signer=authority_signer,
        observed_response_hashes=tuple(state.response_hashes),
        observed_decision_hashes=tuple(state.decision_hashes),
        controller_turn_observations=tuple(
            state.controller_turn_observations
        ),
        controller_turn_facts=_controller_fact_payloads(
            state,
            lane=shard.lane,
        ),
        controller_initial_facts=_controller_fact_payloads(
            state,
            lane="journey_initial",
        ),
    )
    classification, reason = _classify(
        manifest,
        shard,
        state,
        exit_code,
        cleaned,
        artifacts,
    )
    diagnostic_ids = tuple(artifacts["diagnostic_ids"]) or (
        str(receipt["receipt_id"]),
    )
    return ShardResult(
        shard_id=shard.shard_id,
        classification=classification,
        attempt_count=1,
        started_at_ns=started,
        finished_at_ns=time.time_ns(),
        exit_code=exit_code,
        target_hash=shard.target_hash,
        response_hashes=tuple(state.response_hashes),
        decision_hashes=tuple(state.decision_hashes),
        transcript_hash=artifacts["transcript_hash"],
        schedule_result_hash=artifacts["schedule_result_hash"],
        evidence_hashes=tuple(artifacts["evidence_hashes"]),
        diagnostic_hashes=tuple(artifacts["diagnostic_hashes"]),
        evidence_ids=tuple(artifacts["evidence_ids"]),
        diagnostic_ids=diagnostic_ids,
        stderr_hash=_sha256_file(stderr_path),
        stderr_path=str(stderr_path),
        stderr_truncated=False,
        cleanup_receipt_path=str(receipt_path),
        cleanup_receipt_hash=_sha256_file(receipt_path),
        reason=str(redact(reason))[:1000],
    )


async def _exchange_frames(
    process: asyncio.subprocess.Process,
    manifest: FrozenBatchManifest,
    shard: FrozenShardSpec,
    broker: DecisionBroker,
    state: _RunState,
    policy: TimeoutPolicy,
) -> None:
    assert process.stdout is not None
    assert process.stdin is not None
    while True:
        line = await process.stdout.readline()
        if not line:
            return
        text = line.decode("utf-8", errors="replace").rstrip("\r\n")
        context_frame = JOURNEY_CONTEXT_FRAME if shard.lane == "journey" else CONTEXT_FRAME
        decision_frame = JOURNEY_DECISION_FRAME if shard.lane == "journey" else DECISION_FRAME
        result_frame = JOURNEY_RESULT_FRAME if shard.lane == "journey" else RESULT_FRAME
        if text.startswith(context_frame):
            context = _parse_frame(text, context_frame)
            _validate_context_frame(
                shard,
                context,
                len(state.response_hashes),
                previous_turn_index=(
                    state.context_turn_indices[-1]
                    if state.context_turn_indices
                    else None
                ),
            )
            response_hash = str(context.get("previous_response_hash") or "")
            if not response_hash:
                raise RuntimeError("simulator context has no response hash")
            state.response_hashes.append(response_hash)
            if shard.lane == "journey":
                state.context_turn_indices.append(int(context["turn_index"]))
            try:
                decision = await _call_broker_with_timeout(
                    broker,
                    shard.shard_id,
                    context,
                    timeout_seconds=policy.decision_seconds,
                )
            except ExternalDecisionBlocked:
                raise
            except asyncio.TimeoutError as exc:
                raise ExternalDecisionBlocked("simulator decision timed out") from exc
            if decision is None:
                raise ExternalDecisionBlocked("simulator returned no decision")
            normalized = _validate_decision(
                context,
                decision,
                lane=shard.lane,
                simulator_attestation_required=shard.simulator_attestation_required,
            )
            state.decision_hashes.append(_content_hash(normalized))
            process.stdin.write(
                (decision_frame + _canonical_json(normalized) + "\n").encode("utf-8")
            )
            await process.stdin.drain()
        elif text.startswith(result_frame):
            if state.result_payload is not None:
                raise RuntimeError("worker emitted more than one terminal result frame")
            result = _parse_frame(text, result_frame)
            _validate_result_frame(manifest, shard, result)
            state.result_payload = result


async def _call_broker_with_timeout(
    broker: DecisionBroker,
    shard_id: str,
    context: Mapping[str, Any],
    *,
    timeout_seconds: float,
) -> Mapping[str, Any]:
    async_call = inspect.iscoroutinefunction(broker) or inspect.iscoroutinefunction(
        getattr(broker, "__call__", None)
    )
    if async_call:
        return await asyncio.wait_for(
            broker(shard_id, context), timeout=timeout_seconds
        )
    # A synchronous Codex broker can block on external input. Running it on the
    # event-loop thread would stall every shard and defeat independent timeouts.
    result = await asyncio.wait_for(
        asyncio.to_thread(broker, shard_id, context),
        timeout=timeout_seconds,
    )
    if inspect.isawaitable(result):
        return await asyncio.wait_for(result, timeout=timeout_seconds)
    return result


class _SimulatorInvalid(RuntimeError):
    pass


def _validate_decision(
    context: Mapping[str, Any],
    decision: Mapping[str, Any],
    *,
    lane: str = "edge",
    simulator_attestation_required: bool = False,
) -> dict[str, Any]:
    if not isinstance(decision, Mapping):
        raise _SimulatorInvalid("simulator decision is not an object")
    if lane == "journey":
        required = {
            "previous_response_hash", "user_message", "persona", "mission",
            "rationale", "risk_factor_ids",
        }
    else:
        required = {
            "previous_response_hash", "user_message", "persona", "goal",
            "rationale", "target_coverage_ids",
        }
    missing = required.difference(decision)
    if missing:
        raise _SimulatorInvalid(f"simulator decision is missing: {sorted(missing)}")
    expected_hash = str(context.get("previous_response_hash") or "")
    if str(decision.get("previous_response_hash") or "") != expected_hash:
        raise _SimulatorInvalid("simulator decision is bound to a stale response")
    if lane == "journey":
        schedule = context.get("schedule") or {}
        if str(decision.get("persona") or "") != str(schedule.get("persona") or ""):
            raise _SimulatorInvalid("simulator changed the Journey persona")
        if str(decision.get("mission") or "") != str(schedule.get("mission") or ""):
            raise _SimulatorInvalid("simulator changed the Journey mission")
        risks = [str(item) for item in decision.get("risk_factor_ids") or ()]
        if len(risks) != len(set(risks)) or set(risks) - set(
            schedule.get("allowed_risk_factors") or ()
        ):
            raise _SimulatorInvalid("simulator selected invalid Journey risk factors")
        verifier_input = schedule.get("verifier_input_contract") or {}
        binding = decision.get("variant_binding") or {}
        if verifier_input:
            try:
                from tests.agent_live.retained_regression_attestations import (
                    validate_verifier_input_contract,
                )

                contract = validate_verifier_input_contract(verifier_input)
            except ValueError as exc:
                raise _SimulatorInvalid(str(exc)) from exc
            if (
                not isinstance(binding, Mapping)
                or set(binding) != {"source_step_id", "semantic_role"}
            ):
                raise _SimulatorInvalid(
                    "retained regression decision has no exact variant binding"
                )
            source_step = next(
                (
                    item
                    for item in contract["source_contract"]["source_steps"]
                    if item.get("step_id") == binding.get("source_step_id")
                ),
                None,
            )
            if (
                source_step is None
                or binding.get("source_step_id")
                not in contract["variant_contract"]["allowed_source_step_ids"]
                or binding.get("semantic_role")
                != source_step.get("semantic_role")
            ):
                raise _SimulatorInvalid(
                    "retained regression variant binding is outside the frozen contract"
                )
        elif binding:
            raise _SimulatorInvalid(
                "generic Journey decision declared a retained variant binding"
            )
    else:
        target = context.get("scheduled_target") or {}
        if str(decision.get("persona") or "") != str(target.get("persona") or ""):
            raise _SimulatorInvalid("simulator changed the scheduled persona")
        if str(decision.get("goal") or "") != str(target.get("goal") or ""):
            raise _SimulatorInvalid("simulator changed the scheduled goal")
        coverage_ids = [str(item) for item in decision.get("target_coverage_ids") or ()]
        if str(target.get("edge_key") or "") not in coverage_ids:
            raise _SimulatorInvalid("simulator dropped the scheduled coverage edge")
    if not str(decision.get("user_message") or ""):
        raise _SimulatorInvalid("simulator decision has an empty user message")
    if simulator_attestation_required:
        _validate_simulator_attestation(context, decision)
    return dict(decision)


def _validate_simulator_attestation(
    context: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> None:
    attestation = decision.get("simulator_attestation")
    if not isinstance(attestation, Mapping):
        raise _SimulatorInvalid("product Journey decision has no simulator attestation")
    unsigned_decision = {
        key: value for key, value in decision.items()
        if key != "simulator_attestation"
    }
    try:
        request_id = str(decision.get("broker_request_id") or "")
        if not request_id:
            raise ValueError(
                "attested simulator decision has no broker request identity"
            )
        validate_simulator_attestation(
            attestation,
            previous_response_hash=str(context.get("previous_response_hash") or ""),
            decision_hash=_content_hash(unsigned_decision),
            request_id=request_id,
            context_hash=simulator_context_hash(context),
            user_message_hash=_content_hash(
                str(decision.get("user_message") or "")
            ),
            turn_index=int(context.get("turn_index") or 0),
        )
    except ValueError as exc:
        raise _SimulatorInvalid(str(exc)) from exc


def _validate_context_frame(
    shard: FrozenShardSpec,
    context: Mapping[str, Any],
    target_index: int,
    *,
    previous_turn_index: int | None = None,
) -> None:
    if str(context.get("session_id") or "") != shard.session_id:
        raise RuntimeError("simulator context belongs to another session")
    if shard.lane == "journey":
        schedule = context.get("schedule")
        if not isinstance(schedule, Mapping):
            raise RuntimeError("Journey simulator context has no frozen schedule")
        if str(schedule.get("schedule_id") or "") != shard.schedule_id:
            raise RuntimeError("Journey simulator context schedule identity changed")
        if str(schedule.get("persona") or "") != shard.personas[0]:
            raise RuntimeError("Journey simulator context persona changed")
        if str(schedule.get("mission") or "") != shard.goals[0]:
            raise RuntimeError("Journey simulator context mission changed")
        raw_turn_index = context.get("turn_index")
        if (
            isinstance(raw_turn_index, bool)
            or not isinstance(raw_turn_index, int)
            or raw_turn_index < 1
        ):
            raise RuntimeError("Journey simulator context turn identity is invalid")
        turn_index = raw_turn_index
        if previous_turn_index is not None and turn_index not in {
            previous_turn_index,
            previous_turn_index + 1,
        }:
            raise RuntimeError("Journey simulator context turn order is invalid")
        return
    if target_index >= len(shard.target_ids):
        raise RuntimeError("worker requested more decisions than the frozen schedule")
    target = context.get("scheduled_target")
    if not isinstance(target, Mapping):
        raise RuntimeError("simulator context has no scheduled target")
    expected = {
        "target_id": shard.target_ids[target_index],
        "edge_key": shard.edge_keys[target_index],
        "persona": shard.personas[target_index],
        "goal": shard.goals[target_index],
    }
    for key, value in expected.items():
        if str(target.get(key) or "") != value:
            raise RuntimeError(f"simulator context {key} differs from the frozen target")


async def _cleanup_process(
    process: asyncio.subprocess.Process | None,
    guard: ContainerProcessGuard,
    cleanup_seconds: float,
) -> _HostCleanupOutcome:
    actions: list[str] = []
    errors: list[str] = []
    if process is not None and process.stdin is not None and not process.stdin.is_closing():
        process.stdin.close()
        actions.append("stdin_closed")
    if process is None:
        actions.append("spawn_failed_no_process")
    elif process.returncode is None:
        try:
            await asyncio.wait_for(
                process.wait(),
                timeout=max(cleanup_seconds / 2, 0.01),
            )
            actions.append("worker_exited_after_stdin_close")
        except asyncio.TimeoutError:
            actions.append("worker_graceful_exit_timeout")
    reapers = (
        {process.pid: lambda: process.returncode}
        if process is not None
        else {}
    )
    artifact: CleanupReceiptArtifact | None = None
    try:
        artifact = await asyncio.to_thread(guard.cleanup, reapers=reapers)
        actions.append("host_process_guard_completed")
    except Exception as exc:
        errors.append(f"host process guard failed: {type(exc).__name__}: {exc}")
    if process is not None and process.returncode is None:
        try:
            await asyncio.wait_for(process.wait(), timeout=cleanup_seconds)
        except asyncio.TimeoutError:
            actions.append("worker_wait_timeout_after_guard")
            errors.append("worker process was not reaped after host guard cleanup")
    worker_reaped = process is not None and process.returncode is not None
    if worker_reaped:
        actions.append("worker_reaped")
    cleaned = bool(artifact and artifact.cleaned and worker_reaped and not errors)
    return _HostCleanupOutcome(
        artifact=artifact,
        cleaned=cleaned,
        exit_code=process.returncode if process is not None else None,
        actions=tuple(actions),
        errors=tuple(errors),
    )


async def _read_capped(
    stream: asyncio.StreamReader, cap: int
) -> tuple[bytes, bool]:
    retained = bytearray()
    truncated = False
    while True:
        chunk = await stream.read(8192)
        if not chunk:
            break
        remaining = cap - len(retained)
        if remaining > 0:
            retained.extend(chunk[:remaining])
        if len(chunk) > max(remaining, 0):
            truncated = True
    return bytes(retained), truncated


def _fsync_parent_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _classify(
    manifest: FrozenBatchManifest,
    shard: FrozenShardSpec,
    state: _RunState,
    exit_code: int | None,
    cleaned: bool,
    artifacts: Mapping[str, Any],
) -> tuple[str, str]:
    if not cleaned:
        return "infrastructure_interrupted", "worker process cleanup was incomplete"
    if state.forced_classification:
        return state.forced_classification, state.reason
    if artifacts.get("validation_error"):
        return "infrastructure_interrupted", str(artifacts["validation_error"])
    if artifacts.get("product_failure"):
        return "product_failed", str(artifacts["product_failure"])
    if artifacts.get("terminal_classification") == SimulatorTerminalClassification.SIMULATOR_INVALID.value:
        return "simulator_invalid", str(artifacts.get("failure_reason") or "simulator decision invalid")
    if shard.lane == "journey":
        status = str(artifacts.get("execution_status") or "")
        if status == JourneyTerminalClassification.PASSED.value and exit_code == 0:
            has_zero_turn_startup_proof = (
                not state.response_hashes
                and not state.decision_hashes
                and len(
                    _controller_fact_payloads(
                        state,
                        lane="journey_initial",
                    )
                )
                == 1
            )
            if (
                len(state.response_hashes) != len(state.decision_hashes)
                or (
                    not state.response_hashes
                    and not has_zero_turn_startup_proof
                )
            ):
                return "infrastructure_interrupted", "Journey decision ledger is incomplete"
            return "passed", "Journey terminal postconditions passed"
        if status == JourneyTerminalClassification.PRODUCT_FAILED.value:
            return "product_failed", str(artifacts.get("product_failure") or status)
        if status == JourneyTerminalClassification.SIMULATOR_INVALID.value:
            return "simulator_invalid", status
        if status == JourneyTerminalClassification.EXTERNALLY_BLOCKED.value:
            return "externally_blocked", status
        return "infrastructure_interrupted", state.reason or status or "Journey did not finish"
    if (
        exit_code == 0
        and artifacts.get("execution_status") == "complete"
        and (
            (
                manifest.controller_owned_execution
                and len(state.controller_turn_observations)
                == len(shard.target_ids)
            )
            or (
                not manifest.controller_owned_execution
                and len(state.response_hashes) == len(shard.target_ids)
                and len(state.decision_hashes) == len(shard.target_ids)
            )
        )
    ):
        return "passed", "all scheduled targets passed"
    return "infrastructure_interrupted", state.reason or "worker did not complete cleanly"


def _artifact_hashes(
    manifest: FrozenBatchManifest,
    payload: Mapping[str, Any] | None,
    shard: FrozenShardSpec | None = None,
    *,
    authority_signer: PtyAuthoritySigner | None = None,
    observed_response_hashes: tuple[str, ...] = (),
    observed_decision_hashes: tuple[str, ...] = (),
    controller_turn_observations: tuple[Mapping[str, Any], ...] = (),
    controller_turn_facts: tuple[Mapping[str, Any], ...] = (),
    controller_initial_facts: tuple[Mapping[str, Any], ...] = (),
) -> dict[str, Any]:
    data = payload if isinstance(payload, Mapping) else {}
    empty = {
        "transcript_hash": "", "schedule_result_hash": "",
        "evidence_hashes": (), "diagnostic_hashes": (),
        "evidence_ids": (), "diagnostic_ids": (),
        "execution_status": "", "terminal_classification": "", "failure_reason": "",
        "product_failure": "", "validation_error": "",
    }
    if shard is None:
        return empty
    if authority_signer is None:
        raise ValueError("batch controller authority signer is required")
    if (
        manifest.controller_owned_execution
        and shard.lane != "journey"
        and not controller_turn_observations
    ):
        return {
            **empty,
            "validation_error": (
                "controller-owned execution has no authoritative turn "
                "observation ledger"
            ),
        }
    observed_turns = (
        [dict(item) for item in controller_turn_observations]
        if controller_turn_observations
        else [
            {
                "previous_response_hash": response_hash,
                "simulator_decision_hash": decision_hash,
            }
            for response_hash, decision_hash in zip(
                observed_response_hashes,
                observed_decision_hashes,
            )
        ]
    )
    authority_context = {
        "batch_id": manifest.batch_id,
        "manifest_id": manifest.manifest_id,
        "shard_id": shard.shard_id,
        "execution_id": shard.execution_id,
        "schedule_id": shard.schedule_id,
        "target_hash": shard.target_hash,
        "observed_turns": observed_turns,
    }
    if shard.lane == "journey":
        return _journey_artifact_hashes(
            manifest,
            data,
            shard,
            empty,
            authority_signer=authority_signer,
            controller_turn_observations=controller_turn_observations,
            controller_turn_facts=controller_turn_facts,
            controller_initial_facts=controller_initial_facts,
        )
    artifact_bundle_path = (
        Path(shard.runtime_root) / "controller-pty-artifacts.admitted"
    )
    try:
        runtime_roots = _worker_runtime_roots(manifest, shard)
        schedule_path = _known_artifact_path(
            manifest, shard, data, "schedule_path", "schedule.json"
        )
        result_path = _known_artifact_path(
            manifest, shard, data, "schedule_result_path", "schedule-result.json"
        )
        transcript = _known_artifact_path(
            manifest, shard, data, "transcript_path", "transcript.txt"
        )
        for name, path in (
            ("schedule", schedule_path),
            ("schedule result", result_path),
            ("transcript", transcript),
        ):
            if path is None or not path.is_file():
                raise ValueError(f"worker {name} is missing")
            _require_contained(path, runtime_roots, label=name)

        expected_schedule = build_chaos_schedule(
            build_ledger(revision=manifest.revision),
            revision=manifest.revision,
            seed=shard.seed,
            targets=_target_rows(_load_json(Path(shard.target_path))),
        )
        if _load_json(schedule_path) != schedule_payload(expected_schedule):
            raise ValueError("worker schedule does not match the frozen authoritative schedule")
        schedule_result = _load_json(result_path)
        _validate_schedule_result(manifest, shard, schedule_result, data, runtime_roots)

        ledger = build_ledger(revision=manifest.revision)
        edge_index = {
            str(edge.get("edge_key") or ""): edge
            for edge in ledger.get("edges") or ()
        }
        evidence_paths = [
            _resolve_required_path(manifest, item, runtime_roots, label="evidence")
            for item in data.get("evidence_paths") or ()
        ]
        evidence_candidates: list[
            tuple[Path, Mapping[str, Any], Mapping[str, Any]]
        ] = []
        for path in evidence_paths:
            artifact = _load_json(path)
            edge_key = str(artifact.get("edge_key") or "")
            if edge_key not in shard.edge_keys:
                raise ValueError("evidence references a target outside this shard")
            edge = edge_index.get(edge_key)
            if edge is None:
                raise ValueError("evidence references an unknown authoritative edge")
            if artifact.get("artifact_type") == "pty_cli_turn":
                candidate_valid, candidate_reason = (
                    validate_pty_cli_candidate_artifact(
                        artifact,
                        edge=edge,
                        revision=manifest.revision,
                        controller_context=authority_context,
                    )
                )
                if not candidate_valid:
                    raise ValueError(
                        f"invalid unsigned PTY candidate: {candidate_reason}"
                    )
            evidence_candidates.append((path, artifact, edge))

        diagnostic_paths = _diagnostic_paths_from_result(
            manifest, schedule_result, runtime_roots
        )
        diagnostic_candidates: list[
            tuple[Mapping[str, Any], Path, Mapping[str, Any]]
        ] = []
        product_failure = ""
        for row, path in diagnostic_paths:
            artifact = _load_json(path)
            if artifact.get("artifact_type") == "pty_diagnostic":
                candidate_valid, candidate_reason = (
                    validate_pty_diagnostic_candidate(
                        artifact,
                        expected_target_edge_key=str(
                            row.get("edge_key") or ""
                        ),
                    )
                )
                if not candidate_valid:
                    raise ValueError(
                        f"invalid unsigned PTY diagnostic: {candidate_reason}"
                    )
            if dict(artifact.get("revision") or {}) != dict(manifest.revision):
                raise ValueError("diagnostic repository revision mismatch")
            if str(artifact.get("target_id") or "") != str(row.get("target_id") or ""):
                raise ValueError("diagnostic target identity mismatch")
            if str(artifact.get("target_edge_key") or "") != str(row.get("edge_key") or ""):
                raise ValueError("diagnostic target edge mismatch")
            diagnostic_candidates.append((row, path, artifact))
            if artifact.get("verification_status") == "postcondition_failed":
                product_failure = str(row.get("reason") or "postcondition failed")

        pty_paths = [
            path
            for path, artifact, _edge in evidence_candidates
            if artifact.get("artifact_type") == "pty_cli_turn"
        ] + [
            path
            for _row, path, artifact in diagnostic_candidates
            if artifact.get("artifact_type") == "pty_diagnostic"
        ]
        for path in pty_paths:
            if path.with_suffix(".admitted").exists():
                rollback_pty_artifact_admission(path)
        if pty_paths:
            admit_pty_artifact_bundle(
                artifact_bundle_path,
                pty_paths,
                signer=authority_signer,
                controller_context=authority_context,
            )
            bundle_items, bundle_digest = (
                load_validated_pty_artifact_bundle(
                    artifact_bundle_path,
                    trusted_public_key_b64=(
                        authority_signer.public_key_b64
                    ),
                    expected_controller_context=authority_context,
                )
            )
            admitted_by_id = {
                str(
                    item["artifact"].get("evidence_id")
                    or item["artifact"].get("diagnostic_id")
                    or ""
                ): item
                for item in bundle_items
            }
        else:
            bundle_digest = ""
            admitted_by_id = {}

        evidence_hashes: list[str] = []
        evidence_ids: list[str] = []
        for path, candidate, edge in evidence_candidates:
            if candidate.get("artifact_type") == "pty_cli_turn":
                evidence_id = str(candidate.get("evidence_id") or "")
                item = admitted_by_id.get(evidence_id)
                if item is None:
                    raise ValueError(
                        "PTY evidence is absent from the shard bundle"
                    )
                validated = item["artifact"]
                valid, reason = validate_pty_cli_evidence_artifact(
                    validated,
                    edge=edge,
                    revision=manifest.revision,
                    authority=item["authority"],
                    trusted_public_key_b64=(
                        authority_signer.public_key_b64
                    ),
                    expected_controller_context=authority_context,
                )
                if not valid:
                    raise ValueError(
                        f"invalid evidence artifact: {reason}"
                    )
                evidence_hashes.append(_content_hash({
                    "artifact_bundle_digest": bundle_digest,
                    "artifact_id": evidence_id,
                }))
            else:
                validated, reason = load_valid_evidence_reference(
                    str(path),
                    edge=edge,
                    revision=manifest.revision,
                    trusted_public_key_b64=(
                        authority_signer.public_key_b64
                    ),
                    expected_controller_context=authority_context,
                )
                if validated is None:
                    raise ValueError(
                        f"invalid evidence artifact: {reason}"
                    )
                evidence_id = str(validated.get("evidence_id") or "")
                evidence_hashes.append(_sha256_file(path))
            if not evidence_id:
                raise ValueError("validated evidence has no evidence_id")
            evidence_ids.append(evidence_id)

        diagnostic_hashes: list[str] = []
        diagnostic_ids: list[str] = []
        for row, path, candidate in diagnostic_candidates:
            if candidate.get("artifact_type") == "pty_diagnostic":
                diagnostic_id = str(candidate.get("diagnostic_id") or "")
                item = admitted_by_id.get(diagnostic_id)
                if item is None:
                    raise ValueError(
                        "PTY diagnostic is absent from the shard bundle"
                    )
                artifact = item["artifact"]
                authority = item["authority"]
                diagnostic_hashes.append(_content_hash({
                    "artifact_bundle_digest": bundle_digest,
                    "artifact_id": diagnostic_id,
                }))
            else:
                diagnostic_id = str(candidate.get("diagnostic_id") or "")
                artifact = candidate
                authority = {}
                diagnostic_hashes.append(_sha256_file(path))
            valid, reason = validate_pty_diagnostic_artifact(
                artifact,
                authority=authority,
                trusted_public_key_b64=authority_signer.public_key_b64,
                expected_controller_context=authority_context,
                expected_target_edge_key=str(row.get("edge_key") or ""),
            )
            if not valid:
                raise ValueError(f"invalid diagnostic artifact: {reason}")
            diagnostic_ids.append(diagnostic_id)

        status = str(schedule_result.get("execution_status") or "")
        if status == "complete" and not evidence_ids:
            raise ValueError("completed schedule has no validated evidence")
        return {
            "transcript_hash": _sha256_file(transcript),
            "schedule_result_hash": _sha256_file(result_path),
            "evidence_hashes": tuple(evidence_hashes),
            "diagnostic_hashes": tuple(diagnostic_hashes),
            "evidence_ids": tuple(evidence_ids),
            "diagnostic_ids": tuple(diagnostic_ids),
            "execution_status": status,
            "terminal_classification": str(data.get("terminal_classification") or ""),
            "failure_reason": str(data.get("failure_reason") or ""),
            "product_failure": product_failure,
            "validation_error": "",
        }
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        rollback_pty_artifact_bundle(artifact_bundle_path)
        return {**empty, "validation_error": f"artifact validation failed: {exc}"}


def _journey_artifact_hashes(
    manifest: FrozenBatchManifest,
    data: Mapping[str, Any],
    shard: FrozenShardSpec,
    empty: Mapping[str, Any],
    *,
    authority_signer: PtyAuthoritySigner,
    controller_turn_observations: tuple[Mapping[str, Any], ...],
    controller_turn_facts: tuple[Mapping[str, Any], ...],
    controller_initial_facts: tuple[Mapping[str, Any], ...],
) -> dict[str, Any]:
    authority_bundle_path = (
        Path(shard.runtime_root) / "journey-controller-admission"
    )
    try:
        roots = _worker_runtime_roots(manifest, shard)
        schedule_path = _resolve_required_path(
            manifest, data.get("schedule_path"), roots, label="Journey schedule"
        )
        result_path = _resolve_required_path(
            manifest, data.get("journey_result_path"), roots, label="Journey result"
        )
        transcript_path = _resolve_required_path(
            manifest, data.get("transcript_path"), roots, label="Journey transcript"
        )
        evidence_path = _resolve_required_path(
            manifest, data.get("evidence_path"), roots, label="Journey evidence"
        )
        for label, path in (
            ("schedule", schedule_path), ("result", result_path),
            ("transcript", transcript_path), ("evidence", evidence_path),
        ):
            if not path.is_file():
                raise ValueError(f"Journey {label} is missing")
        journey, registry_import = _journey_target(
            _load_json(Path(shard.target_path))
        )
        schedule = build_journey_schedule(
            revision=manifest.revision,
            seed=shard.seed,
            journey=journey,
        )
        if _load_json(schedule_path) != journey_schedule_payload(schedule):
            raise ValueError("worker Journey schedule differs from the frozen schedule")
        registry = load_verifier_registry(registry_import)
        if registry.registry_id != shard.verifier_registry_id:
            raise ValueError("worker Journey verifier registry identity changed")
        evidence = _load_json(evidence_path)
        result = _load_json(result_path)
        status = str(result.get("terminal_classification") or "")
        if str(data.get("terminal_classification") or "") != status:
            raise ValueError("Journey frame and result classification disagree")
        if str(data.get("schedule_id") or "") != shard.schedule_id:
            raise ValueError("Journey result frame schedule identity changed")
        if str(data.get("evidence_id") or "") != str(result.get("evidence_id") or ""):
            raise ValueError("Journey result frame evidence identity changed")
        product_failure = ""
        evidence_ids: tuple[str, ...] = ()
        evidence_hashes: tuple[str, ...] = ()
        if status == JourneyTerminalClassification.PASSED.value:
            candidate_turns = list(evidence.get("turns") or ())
            if manifest.controller_owned_execution:
                if (
                    len(candidate_turns)
                    != len(controller_turn_observations)
                    or len(controller_initial_facts) != 1
                ):
                    raise ValueError(
                        "Journey evidence has no complete controller observation ledger"
                    )
                previous_receipt_hash = ""
                for candidate, observation in zip(
                    candidate_turns,
                    controller_turn_observations,
                    strict=True,
                ):
                    unsigned_observation = dict(observation)
                    receipt_hash = str(
                        unsigned_observation.pop(
                            "controller_observation_hash",
                            "",
                        )
                        or ""
                    )
                    if (
                        _content_hash(unsigned_observation) != receipt_hash
                        or observation.get(
                            "previous_controller_observation_hash"
                        ) != previous_receipt_hash
                        or observation.get("turn_result_hash")
                        != _content_hash(candidate)
                    ):
                        raise ValueError(
                            "Journey controller observation chain is invalid"
                        )
                    previous_receipt_hash = receipt_hash
            validate_journey_evidence_artifact(
                evidence,
                schedule=schedule,
                verifier_registry=registry,
                revision=manifest.revision,
            )
            evidence_ids = (str(evidence["evidence_id"]),)
            if not manifest.controller_owned_execution:
                evidence_hashes = (_sha256_file(evidence_path),)
                return {
                    "transcript_hash": _sha256_file(transcript_path),
                    "schedule_result_hash": _sha256_file(result_path),
                    "evidence_hashes": evidence_hashes,
                    "diagnostic_hashes": (),
                    "evidence_ids": evidence_ids,
                    "diagnostic_ids": (),
                    "execution_status": status,
                    "product_failure": product_failure,
                    "validation_error": "",
                }
            authority_unsigned = {
                "batch_id": manifest.batch_id,
                "manifest_id": manifest.manifest_id,
                "shard_id": shard.shard_id,
                "execution_id": shard.execution_id,
                "schedule_id": shard.schedule_id,
                "candidate_artifact_sha256": _sha256_file(evidence_path),
                "controller_turn_observations": [
                    dict(item)
                    for item in controller_turn_observations
                ],
                "controller_turn_facts": [
                    dict(item)
                    for item in controller_turn_facts
                ],
                "controller_initial_fact": (
                    dict(controller_initial_facts[0])
                    if len(controller_initial_facts) == 1
                    else {}
                ),
            }
            authority_payload = {
                **authority_unsigned,
                "controller_signature_b64": sign_controller_payload(
                    authority_signer,
                    authority_unsigned,
                ),
            }
            authority_payload["authority_id"] = _content_hash(
                authority_payload
            )
            _publish_journey_controller_bundle(
                evidence_path=evidence_path,
                authority=authority_payload,
                bundle_path=authority_bundle_path,
            )
            evidence_hashes = (
                validate_journey_controller_authority(
                    manifest=manifest,
                    shard=shard,
                    bundle_path=authority_bundle_path,
                ),
            )
        elif status == JourneyTerminalClassification.PRODUCT_FAILED.value:
            product_failure = str(result.get("failure_reason") or "Journey product failure")
        return {
            "transcript_hash": _sha256_file(transcript_path),
            "schedule_result_hash": _sha256_file(result_path),
            "evidence_hashes": evidence_hashes,
            "diagnostic_hashes": (),
            "evidence_ids": evidence_ids,
            "diagnostic_ids": (),
            "execution_status": status,
            "product_failure": product_failure,
            "validation_error": "",
        }
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        return {**empty, "validation_error": f"artifact validation failed: {exc}"}


@contextmanager
def _journey_controller_bundle_lock(bundle_path: Path):
    lock_path = bundle_path.with_name(f".{bundle_path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _remove_journey_controller_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        path.chmod(0o700)
        for child in path.iterdir():
            if not child.is_symlink():
                child.chmod(0o600)
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _journey_controller_staging_paths(
    bundle_path: Path,
) -> tuple[Path, ...]:
    return tuple(
        bundle_path.parent.glob(f"{bundle_path.name}.tmp.*")
    )


def _journey_controller_recovery_marker(
    bundle_path: Path,
) -> Path:
    return bundle_path.with_name(
        f".{bundle_path.name}.recovery-required"
    )


def _create_journey_controller_recovery_marker(
    bundle_path: Path,
) -> None:
    marker = _journey_controller_recovery_marker(bundle_path)
    descriptor = os.open(
        marker,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY,
        0o400,
    )
    try:
        payload = (
            json.dumps(
                {
                    "schema_version": 1,
                    "bundle_path": bundle_path.name,
                    "created_at_ns": time.time_ns(),
                },
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        written = 0
        while written < len(payload):
            written += os.write(descriptor, payload[written:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_parent_directory(marker.parent)


def _clear_journey_controller_recovery_marker(
    bundle_path: Path,
) -> None:
    marker = _journey_controller_recovery_marker(bundle_path)
    if not marker.exists():
        return
    marker.unlink()
    try:
        _fsync_parent_directory(marker.parent)
    except Exception as clear_error:
        try:
            _create_journey_controller_recovery_marker(bundle_path)
        except Exception as restore_error:
            raise RuntimeError(
                "Journey controller recovery marker clear failed and "
                "required-state restoration failed"
            ) from restore_error
        raise RuntimeError(
            "Journey controller recovery marker clear was not durable; "
            "recovery remains required"
        ) from clear_error


def _load_journey_controller_bundle(
    bundle_path: Path,
    *,
    allow_recovery_marker: bool = False,
) -> tuple[Path, Path, JourneyControllerAuthoritySnapshot, str]:
    if (
        not allow_recovery_marker
        and _journey_controller_recovery_marker(bundle_path).exists()
    ):
        raise ValueError(
            "Journey controller bundle requires durability reconciliation"
        )
    path = bundle_path.resolve()
    if (
        path.is_symlink()
        or not path.is_dir()
        or path.stat().st_mode & 0o222
        or {entry.name for entry in path.iterdir()}
        != {"candidate.json", "authority.json", "COMMIT.json"}
    ):
        raise ValueError(
            "Journey controller admission bundle is missing or mutable"
        )
    candidate = path / "candidate.json"
    authority_file = path / "authority.json"
    commit_file = path / "COMMIT.json"
    if any(
        item.is_symlink()
        or not item.is_file()
        or item.stat().st_mode & 0o222
        for item in (candidate, authority_file, commit_file)
    ):
        raise ValueError(
            "Journey controller admission bundle contains an invalid file"
        )
    try:
        candidate_bytes = candidate.read_bytes()
        authority_bytes = authority_file.read_bytes()
        commit_bytes = commit_file.read_bytes()
        authority = json.loads(authority_bytes)
        commit = json.loads(commit_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            "Journey controller admission bundle is incomplete"
        ) from exc
    if not isinstance(authority, Mapping) or not isinstance(commit, Mapping):
        raise ValueError(
            "Journey controller admission bundle is malformed"
        )
    expected_commit = {
        "schema_version": 1,
        "candidate_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
        "authority_sha256": hashlib.sha256(authority_bytes).hexdigest(),
        "authority_id": str(authority.get("authority_id") or ""),
    }
    if dict(commit) != expected_commit:
        raise ValueError(
            "Journey controller admission commit marker is stale"
        )
    snapshot = JourneyControllerAuthoritySnapshot(
        candidate_bytes=candidate_bytes,
        authority_bytes=authority_bytes,
        bundle_digest=_content_hash({
            **expected_commit,
            "commit_sha256": hashlib.sha256(commit_bytes).hexdigest(),
        }),
    )
    return (
        candidate,
        authority_file,
        snapshot,
        snapshot.bundle_digest,
    )


def _publish_journey_controller_bundle(
    *,
    evidence_path: Path,
    authority: Mapping[str, Any],
    bundle_path: Path,
) -> Path:
    candidate_bytes = evidence_path.read_bytes()
    authority_bytes = (
        json.dumps(
            dict(authority),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    with _journey_controller_bundle_lock(bundle_path):
        for staging in _journey_controller_staging_paths(bundle_path):
            _remove_journey_controller_path(staging)
        if _journey_controller_recovery_marker(bundle_path).exists():
            raise RuntimeError(
                "Journey controller bundle requires explicit "
                "durability recovery"
            )
        if bundle_path.exists():
            candidate, authority_file, existing_snapshot, _digest = (
                _load_journey_controller_bundle(bundle_path)
            )
            existing = existing_snapshot.authority_payload()
            if (
                candidate.read_bytes() == candidate_bytes
                and authority_file.read_bytes() == authority_bytes
                and existing == dict(authority)
            ):
                _fsync_parent_directory(bundle_path)
                _fsync_parent_directory(bundle_path.parent)
                return bundle_path
            raise FileExistsError(
                "Journey controller bundle belongs to another "
                "candidate or authority"
            )
        staging = bundle_path.with_name(
            f"{bundle_path.name}.tmp.{os.getpid()}.{time.time_ns()}"
        )
        published = False
        try:
            staging.mkdir(mode=0o700)
            candidate = staging / "candidate.json"
            authority_file = staging / "authority.json"
            commit_file = staging / "COMMIT.json"
            candidate.write_bytes(candidate_bytes)
            authority_file.write_bytes(authority_bytes)
            commit_file.write_text(
                json.dumps({
                    "schema_version": 1,
                    "candidate_sha256": hashlib.sha256(
                        candidate_bytes
                    ).hexdigest(),
                    "authority_sha256": hashlib.sha256(
                        authority_bytes
                    ).hexdigest(),
                    "authority_id": str(
                        authority.get("authority_id") or ""
                    ),
                }, indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            for item in (candidate, authority_file, commit_file):
                item.chmod(0o400)
                with item.open("rb") as handle:
                    os.fsync(handle.fileno())
            _fsync_parent_directory(staging)
            staging.chmod(0o500)
            _fsync_parent_directory(staging)
            _create_journey_controller_recovery_marker(bundle_path)
            if bundle_path.exists():
                raise FileExistsError(
                    f"Journey controller bundle exists: {bundle_path}"
                )
            staging.rename(bundle_path)
            published = True
            _fsync_parent_directory(bundle_path)
            _fsync_parent_directory(bundle_path.parent)
            _clear_journey_controller_recovery_marker(bundle_path)
        except Exception:
            if published and bundle_path.exists():
                raise RuntimeError(
                    "Journey controller bundle durability is uncertain; "
                    "retry reconciliation is required"
                )
            if staging.exists():
                _remove_journey_controller_path(staging)
                _fsync_parent_directory(staging.parent)
            _clear_journey_controller_recovery_marker(bundle_path)
            raise
        return bundle_path


def recover_journey_controller_bundle(
    bundle_path: str | Path,
    *,
    expected_candidate: Mapping[str, Any] | None = None,
    expected_authority: Mapping[str, Any] | None = None,
) -> Path | None:
    """Reconcile one uncertain Journey publication without replacing it."""

    path = Path(bundle_path)
    with _journey_controller_bundle_lock(path):
        for staging in _journey_controller_staging_paths(path):
            _remove_journey_controller_path(staging)
        if not path.exists():
            _clear_journey_controller_recovery_marker(path)
            return None
        _fsync_parent_directory(path)
        _fsync_parent_directory(path.parent)
        candidate, _authority_path, snapshot, _digest = (
            _load_journey_controller_bundle(
                path,
                allow_recovery_marker=True,
            )
        )
        authority = snapshot.authority_payload()
        if (
            expected_candidate is not None
            and snapshot.candidate_payload() != dict(expected_candidate)
        ):
            raise FileExistsError(
                "Journey controller bundle candidate changed"
            )
        if (
            expected_authority is not None
            and dict(authority) != dict(expected_authority)
        ):
            raise FileExistsError(
                "Journey controller bundle authority changed"
            )
        _clear_journey_controller_recovery_marker(path)
        return path


def _rollback_journey_controller_bundle(bundle_path: Path) -> None:
    with _journey_controller_bundle_lock(bundle_path):
        for staging in _journey_controller_staging_paths(bundle_path):
            _remove_journey_controller_path(staging)
        _remove_journey_controller_path(bundle_path)
        _clear_journey_controller_recovery_marker(bundle_path)
        _fsync_parent_directory(bundle_path.parent)


def validate_journey_controller_authority(
    *,
    manifest: FrozenBatchManifest,
    shard: FrozenShardSpec,
    bundle_path: str | Path,
) -> str:
    """Validate one Journey candidate against its controller-owned receipt."""

    return load_validated_journey_controller_authority(
        manifest=manifest,
        shard=shard,
        bundle_path=bundle_path,
    ).bundle_digest


def load_validated_journey_controller_authority(
    *,
    manifest: FrozenBatchManifest,
    shard: FrozenShardSpec,
    bundle_path: str | Path,
) -> JourneyControllerAuthoritySnapshot:
    """Validate and return the same immutable Journey bundle snapshot."""

    _candidate, _authority_file, snapshot, _digest = _load_journey_controller_bundle(
        Path(bundle_path)
    )
    authority = snapshot.authority_payload()
    candidate_payload = snapshot.candidate_payload()
    expected_fields = {
        "batch_id",
        "manifest_id",
        "shard_id",
        "execution_id",
        "schedule_id",
        "candidate_artifact_sha256",
        "controller_turn_observations",
        "controller_turn_facts",
        "controller_initial_fact",
        "controller_signature_b64",
        "authority_id",
    }
    if not isinstance(authority, Mapping) or set(authority) != expected_fields:
        raise ValueError("Journey controller authority schema is invalid")
    signed_payload = {
        key: value
        for key, value in authority.items()
        if key not in {"controller_signature_b64", "authority_id"}
    }
    authority_identity_payload = {
        key: value
        for key, value in authority.items()
        if key != "authority_id"
    }
    if (
        authority.get("batch_id") != manifest.batch_id
        or authority.get("manifest_id") != manifest.manifest_id
        or authority.get("shard_id") != shard.shard_id
        or authority.get("execution_id") != shard.execution_id
        or authority.get("schedule_id") != shard.schedule_id
        or authority.get("candidate_artifact_sha256")
        != hashlib.sha256(snapshot.candidate_bytes).hexdigest()
        or authority.get("authority_id")
        != _content_hash(authority_identity_payload)
        or not verify_controller_payload_signature(
            signed_payload,
            signature_b64=str(
                authority.get("controller_signature_b64") or ""
            ),
            trusted_public_key_b64=(
                manifest.pty_authority_public_key_b64
            ),
        )
    ):
        raise ValueError("Journey controller authority binding is invalid")
    observations = authority.get("controller_turn_observations")
    controller_facts = authority.get("controller_turn_facts")
    controller_initial_fact = authority.get("controller_initial_fact")
    journey, registry_import = _journey_target(
        _load_json(Path(shard.target_path))
    )
    schedule = build_journey_schedule(
        revision=manifest.revision,
        seed=shard.seed,
        journey=journey,
    )
    validate_journey_schedule(schedule, revision=manifest.revision)
    registry = load_verifier_registry(registry_import)
    if (
        schedule.schedule_id != shard.schedule_id
        or registry_import != shard.verifier_registry_import
        or registry.registry_id != shard.verifier_registry_id
    ):
        raise ValueError(
            "Journey controller candidate authority changed"
        )
    validate_journey_evidence_artifact(
        candidate_payload,
        schedule=schedule,
        verifier_registry=registry,
        revision=manifest.revision,
    )
    candidate_turns = candidate_payload.get("turns")
    if (
        not isinstance(observations, Sequence)
        or isinstance(observations, (str, bytes))
        or not isinstance(candidate_turns, list)
        or len(candidate_turns) != len(observations)
        or not isinstance(controller_facts, Sequence)
        or isinstance(controller_facts, (str, bytes))
        or len(controller_facts) != len(observations)
        or not isinstance(controller_initial_fact, Mapping)
        or not controller_initial_fact
    ):
        raise ValueError(
            "Journey controller authority observation cardinality is invalid"
        )
    previous_receipt_hash = ""
    previous_turn_index = 0
    previous_submission_sequence = 0
    previous_runtime_sequence = 0
    for ordinal, (raw, candidate_turn) in enumerate(
        zip(observations, candidate_turns, strict=True),
        start=1,
    ):
        if not isinstance(raw, Mapping):
            raise ValueError(
                "Journey controller authority observation is invalid"
            )
        if set(raw) != JOURNEY_CONTROLLER_OBSERVATION_FIELDS:
            raise ValueError(
                "Journey controller authority observation schema is invalid"
            )
        observation = dict(raw)
        receipt_hash = str(
            observation.pop("controller_observation_hash", "") or ""
        )
        turn_index = int(observation.get("turn_index") or 0)
        submission_sequence = int(
            observation.get("submission_sequence") or 0
        )
        runtime_sequence = int(
            observation.get("terminal_runtime_event_sequence") or 0
        )
        digest_fields = (
            "previous_response_hash",
            "user_message_hash",
            "approved_decision_hash",
            "simulator_context_hash",
            "submitted_input_commitment",
            "agent_response_hash",
            "baseline_runtime_event_hash",
            "committed_runtime_event_hash",
            "terminal_outcome_hash",
            "turn_result_hash",
        )
        if (
            int(observation.get("shard_turn_ordinal") or 0) != ordinal
            or turn_index <= previous_turn_index
            or submission_sequence <= previous_submission_sequence
            or runtime_sequence <= previous_runtime_sequence
            or (
                ordinal > 1
                and turn_index != previous_turn_index + 1
            )
            or (
                ordinal > 1
                and runtime_sequence != previous_runtime_sequence + 1
            )
            or not str(
                observation.get("terminal_runtime_event_id") or ""
            )
            or any(
                len(str(observation.get(field) or "")) != 64
                for field in digest_fields
            )
            or len(receipt_hash) != 64
            or _content_hash(observation) != receipt_hash
            or observation.get("previous_controller_observation_hash")
            != previous_receipt_hash
            or not isinstance(candidate_turn, Mapping)
            or int(candidate_turn.get("turn_index") or 0) != turn_index
            or observation.get("turn_result_hash")
            != _content_hash(candidate_turn)
        ):
            raise ValueError(
                "Journey controller authority observation chain is invalid"
            )
        previous_receipt_hash = receipt_hash
        previous_turn_index = turn_index
        previous_submission_sequence = submission_sequence
        previous_runtime_sequence = runtime_sequence
    _validate_and_recompute_journey_controller_facts(
        manifest=manifest,
        shard=shard,
        candidate_turns=candidate_turns,
        observations=observations,
        controller_facts=controller_facts,
        controller_initial_fact=controller_initial_fact,
        candidate_initial_verification=(
            candidate_payload.get("initial_verification")
        ),
    )
    return snapshot


def _validate_and_recompute_journey_controller_facts(
    *,
    manifest: FrozenBatchManifest,
    shard: FrozenShardSpec,
    candidate_turns: Sequence[Mapping[str, Any]],
    observations: Sequence[Mapping[str, Any]],
    controller_facts: Sequence[Mapping[str, Any]],
    controller_initial_fact: Mapping[str, Any],
    candidate_initial_verification: Any,
) -> None:
    """Re-run Journey verifiers from signed, persisted controller facts."""

    target_path = Path(shard.target_path).resolve()
    if (
        not target_path.is_file()
        or _sha256_file(target_path) != shard.target_hash
    ):
        raise ValueError("Journey frozen target binding is invalid")
    journey, registry_import = _journey_target(_load_json(target_path))
    if registry_import != shard.verifier_registry_import:
        raise ValueError("Journey verifier registry import changed")
    schedule = build_journey_schedule(
        revision=manifest.revision,
        seed=shard.seed,
        journey=journey,
    )
    validate_journey_schedule(schedule, revision=manifest.revision)
    if schedule.schedule_id != shard.schedule_id:
        raise ValueError("Journey frozen schedule identity changed")
    registry = load_verifier_registry(registry_import)
    if registry.registry_id != shard.verifier_registry_id:
        raise ValueError("Journey verifier registry identity changed")
    edge_index = {
        str(edge.get("edge_key") or ""): edge
        for edge in build_ledger(revision=manifest.revision).get("edges") or ()
    }

    if set(controller_initial_fact) != {
        "initial_event",
        "initial_verification",
    }:
        raise ValueError("Journey initial controller fact is invalid")
    initial_event_payload = controller_initial_fact.get("initial_event")
    persisted_initial_verification = controller_initial_fact.get(
        "initial_verification"
    )
    if (
        not isinstance(initial_event_payload, Mapping)
        or not isinstance(persisted_initial_verification, Mapping)
        or not isinstance(candidate_initial_verification, Mapping)
    ):
        raise ValueError("Journey initial verification is incomplete")
    initial_event = _runtime_event_from_mapping(initial_event_payload)
    validate_runtime_turn_event(initial_event)
    if (
        initial_event.thread_id != shard.session_id
        or dict(initial_event.revision) != dict(manifest.revision)
    ):
        raise ValueError(
            "Journey initial runtime event authority is invalid"
        )
    initial_context = build_journey_verifier_context(
        schedule=schedule,
        initial_event=initial_event,
        current_event=initial_event,
        turns=(),
        events=(),
        decisions=(),
        transcript=(),
        observed_edge_keys=(),
        latest_turn=None,
    )
    recomputed_initial = {
        "terminal_outcome": _journey_outcome_verification_payload(
            verify_journey_outcome(
                schedule.terminal_outcome,
                initial_context,
                registry,
            )
        ),
        "forbidden_outcomes": [
            _journey_outcome_verification_payload(item)
            for item in verify_journey_forbidden_outcomes(
                schedule,
                initial_context,
                registry,
            )
        ],
    }
    if (
        recomputed_initial != persisted_initial_verification
        or recomputed_initial != candidate_initial_verification
    ):
        raise ValueError(
            "Journey initial verification failed recomputation"
        )

    turns: list[PtyCliTurnRecord] = []
    events: list[Any] = []
    decisions: list[JourneyDecisionProvenance] = []
    observed_edge_keys: list[str] = []
    previous_committed_payload: Mapping[str, Any] | None = None
    runtime_event_ids: set[str] = {
        str(initial_event.runtime_event_id or "")
    }
    terminal_event_ids: set[str] = {
        str(initial_event.terminal_event_id or "")
    }
    observed_provider = ""
    observed_model = ""
    for ordinal, (raw_fact, candidate_turn, observation) in enumerate(
        zip(
            controller_facts,
            candidate_turns,
            observations,
            strict=True,
        ),
        start=1,
    ):
        if not isinstance(raw_fact, Mapping):
            raise ValueError("Journey controller fact is not an object")
        required = {
            "initial_event",
            "turn",
            "baseline_event",
            "committed_event",
            "terminal_outcome",
            "decision",
            "observed_edge_keys",
            "turn_result",
        }
        if set(raw_fact) != required:
            raise ValueError("Journey controller fact schema is invalid")
        fact = json.loads(_canonical_json(raw_fact))
        initial_payload = fact["initial_event"]
        baseline_payload = fact["baseline_event"]
        committed_payload = fact["committed_event"]
        if not all(
            isinstance(item, Mapping)
            for item in (
                initial_payload,
                baseline_payload,
                committed_payload,
                fact["turn"],
                fact["decision"],
                fact["terminal_outcome"],
                fact["turn_result"],
            )
        ):
            raise ValueError("Journey controller fact payload is malformed")
        if initial_payload != initial_event_payload:
            raise ValueError("Journey initial runtime event changed")
        if ordinal == 1 and baseline_payload != initial_event_payload:
            raise ValueError("Journey first baseline is not the initial event")
        if (
            previous_committed_payload is not None
            and baseline_payload != previous_committed_payload
        ):
            raise ValueError("Journey runtime event adjacency is invalid")

        turn = _turn_from_payload(fact["turn"])
        baseline = _runtime_event_from_mapping(baseline_payload)
        committed = _runtime_event_from_mapping(committed_payload)
        validate_runtime_turn_event(baseline)
        validate_runtime_turn_event(committed)
        decision = _journey_decision_from_payload(fact["decision"])
        terminal_outcome = _terminal_outcome_from_mapping(
            fact["terminal_outcome"]
        )
        approved_decision = observation.get("approved_decision")
        simulator_context = observation.get("simulator_context")
        try:
            expected_context_binding = simulator_context_binding(
                simulator_context
                if isinstance(simulator_context, Mapping)
                else {}
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Journey simulator context binding is invalid"
            ) from exc
        approved_commitment = (
            str(
                approved_decision.get(
                    "_controller_submitted_input_commitment"
                )
                or ""
            )
            if isinstance(approved_decision, Mapping)
            else ""
        )
        approved_risk_factors = tuple(
            str(item)
            for item in (
                approved_decision.get("risk_factor_ids") or ()
                if isinstance(approved_decision, Mapping)
                else ()
            )
        )
        if (
            turn.turn_index
            != int(observation.get("turn_index") or 0)
            or turn.session_id != shard.session_id
            or not turn.provider
            or not turn.model
            or (
                observed_provider
                and turn.provider != observed_provider
            )
            or (
                observed_model
                and turn.model != observed_model
            )
            or turn.transcript_hash
            != pty_transcript_hash(
                session_id=turn.session_id,
                turn_index=turn.turn_index,
                previous_agent_response=turn.previous_agent_response,
                user_message=turn.user_message,
                agent_response=turn.agent_response,
            )
            or any(
                event.thread_id != shard.session_id
                or dict(event.revision) != dict(manifest.revision)
                or event.session_purpose
                != initial_event.session_purpose
                for event in (baseline, committed)
            )
            or committed.base_revision != baseline.product_revision
            or committed.base_checkpoint_thread_id
            != baseline.product_checkpoint_thread_id
            or committed.base_checkpoint_id
            != baseline.product_checkpoint_id
            or decision.turn_index != turn.turn_index
            or decision.execution_id != shard.execution_id
            or decision.obligation_id != schedule.journey_id
            or decision.previous_response_hash
            != str(
                expected_context_binding.get(
                    "previous_response_hash"
                )
                or ""
            )
            or decision.user_message_hash
            != str(
                approved_decision.get(
                    "_controller_source_user_message_hash"
                )
                or ""
            )
            or not isinstance(approved_decision, Mapping)
            or _content_hash(approved_decision)
            != observation.get("approved_decision_hash")
            or not isinstance(simulator_context, Mapping)
            or simulator_context_hash(simulator_context)
            != observation.get("simulator_context_hash")
            or dict(decision.simulator_context_binding)
            != expected_context_binding
            or str(approved_decision.get("user_message") or "")
            != turn.user_message
            or str(approved_decision.get("persona") or "")
            != decision.persona
            or str(approved_decision.get("mission") or "")
            != decision.mission
            or str(approved_decision.get("rationale") or "")
            != decision.rationale
            or approved_risk_factors != tuple(decision.risk_factor_ids)
            or dict(
                approved_decision.get("simulator_attestation") or {}
            )
            != dict(decision.simulator_attestation)
            or (
                dict(approved_decision.get("variant_binding") or {})
                and (
                    str(
                        decision.variant_attestation.get(
                            "source_step_id"
                        )
                        or ""
                    )
                    != str(
                        dict(
                            approved_decision.get(
                                "variant_binding"
                            )
                            or {}
                        ).get("source_step_id")
                        or ""
                    )
                    or str(
                        decision.variant_attestation.get(
                            "semantic_role"
                        )
                        or ""
                    )
                    != str(
                        dict(
                            approved_decision.get(
                                "variant_binding"
                            )
                            or {}
                        ).get("semantic_role")
                        or ""
                    )
                )
            )
            or approved_commitment
            != str(
                observation.get("submitted_input_commitment") or ""
            )
            or observation.get("previous_response_hash")
            != _content_hash(turn.previous_agent_response)
            or observation.get("user_message_hash")
            != _content_hash(turn.user_message)
            or observation.get("agent_response_hash")
            != _content_hash(turn.agent_response)
            or observation.get("baseline_runtime_event_hash")
            != baseline.runtime_event_payload_hash
            or observation.get("committed_runtime_event_hash")
            != committed.runtime_event_payload_hash
            or turn.before_fingerprint != baseline.after_fingerprint
            or turn.after_fingerprint != committed.after_fingerprint
            or turn.user_message_submitted_at_ns != decision.submitted_at_ns
            or not (
                turn.previous_response_received_at_ns
                <= decision.selected_at_ns
                <= decision.submitted_at_ns
                <= turn.agent_response_received_at_ns
            )
            or str(committed.runtime_event_id or "")
            != str(observation.get("terminal_runtime_event_id") or "")
            or committed.runtime_event_sequence
            != int(
                observation.get("terminal_runtime_event_sequence") or 0
            )
            or _content_hash(asdict(terminal_outcome))
            != observation.get("terminal_outcome_hash")
            or terminal_outcome.transaction_id
            != committed.transaction_id
            or terminal_outcome.runtime_event_id
            != committed.runtime_event_id
            or terminal_outcome.runtime_event_sequence
            != committed.runtime_event_sequence
            or terminal_outcome.runtime_event_payload_hash
            != committed.runtime_event_payload_hash
            or terminal_outcome.product_authority_id
            != committed.product_authority_id
            or terminal_outcome.product_fingerprint
            != committed.after_fingerprint
            or terminal_outcome.event_id != committed.terminal_event_id
            or committed.runtime_event_id in runtime_event_ids
            or committed.terminal_event_id in terminal_event_ids
        ):
            raise ValueError(
                "Journey controller fact differs from its observation"
            )
        if ordinal == 1 and (
            committed.runtime_event_sequence
            != initial_event.runtime_event_sequence + 1
            or turn.turn_index != initial_event.turn_index + 1
        ):
            raise ValueError(
                "Journey first committed event is not adjacent to startup"
            )
        runtime_event_ids.add(committed.runtime_event_id)
        terminal_event_ids.add(committed.terminal_event_id)
        observed_provider = turn.provider
        observed_model = turn.model

        turns.append(turn)
        events.append(committed)
        decisions.append(decision)
        observed_current = observe_journey_edges(
            edge_index,
            baseline,
            committed,
            turn,
        )
        for item in observed_current:
            if item.edge_key not in observed_edge_keys:
                observed_edge_keys.append(item.edge_key)
        if list(fact["observed_edge_keys"]) != observed_edge_keys:
            raise ValueError("Journey observed edge history is invalid")
        transcript = [
            (item.user_message, item.agent_response)
            for item in turns
        ]
        context = build_journey_verifier_context(
            schedule=schedule,
            initial_event=initial_event,
            current_event=committed,
            turns=turns,
            events=events,
            decisions=decisions,
            transcript=transcript,
            observed_edge_keys=observed_edge_keys,
            latest_turn=turn,
        )
        forbidden = verify_journey_forbidden_outcomes(
            schedule,
            context,
            registry,
        )
        terminal = verify_journey_outcome(
            schedule.terminal_outcome,
            context,
            registry,
        )
        recomputed = build_journey_turn_result(
            turn=turn,
            decision=decision,
            observed_edges=observed_current,
            terminal=terminal,
            forbidden=forbidden,
        )
        if (
            recomputed != fact["turn_result"]
            or recomputed != candidate_turn
            or _content_hash(recomputed)
            != observation.get("turn_result_hash")
        ):
            raise ValueError(
                f"Journey controller fact {ordinal} failed recomputation"
            )
        previous_committed_payload = committed_payload


def _validate_result_frame(
    manifest: FrozenBatchManifest,
    shard: FrozenShardSpec,
    payload: Mapping[str, Any],
) -> None:
    if shard.lane == "journey":
        required = {
            "session_id", "terminal_classification", "schedule_id", "schedule_path",
            "journey_result_path", "transcript_path", "evidence_id", "evidence_path",
        }
        if set(payload) != required:
            raise RuntimeError("Journey worker result frame fields mismatch")
        if str(payload.get("session_id") or "") != shard.session_id:
            raise RuntimeError("Journey worker result belongs to another session")
        try:
            JourneyTerminalClassification(str(payload.get("terminal_classification") or ""))
        except ValueError as exc:
            raise RuntimeError("Journey worker terminal classification is invalid") from exc
        roots = _worker_runtime_roots(manifest, shard)
        for key in (
            "schedule_path", "journey_result_path", "transcript_path", "evidence_path"
        ):
            _resolve_required_path(manifest, payload[key], roots, label=key)
        return
    required = {
        "session_id", "execution_status", "schedule_path", "schedule_result_path",
        "transcript_path", "evidence_paths", "terminal_classification", "failure_reason",
    }
    if set(payload) != required:
        raise RuntimeError(
            "worker result frame fields mismatch: "
            f"missing={sorted(required - set(payload))}, extra={sorted(set(payload) - required)}"
        )
    if str(payload.get("session_id") or "") != shard.session_id:
        raise RuntimeError("worker result belongs to another session")
    if payload.get("execution_status") not in {"complete", "incomplete"}:
        raise RuntimeError("worker result has an invalid execution status")
    try:
        terminal = SimulatorTerminalClassification(
            str(payload.get("terminal_classification") or "")
        )
    except ValueError as exc:
        raise RuntimeError("worker terminal classification is invalid") from exc
    if (terminal is SimulatorTerminalClassification.PASSED) != (
        payload.get("execution_status") == "complete"
    ):
        raise RuntimeError("worker terminal classification contradicts execution status")
    if terminal is SimulatorTerminalClassification.PASSED and payload.get("failure_reason"):
        raise RuntimeError("passed worker result cannot include a failure reason")
    if terminal is SimulatorTerminalClassification.SIMULATOR_INVALID and not str(
        payload.get("failure_reason") or ""
    ).strip():
        raise RuntimeError("simulator_invalid worker result requires a failure reason")
    if not isinstance(payload.get("evidence_paths"), list):
        raise RuntimeError("worker result evidence_paths must be a list")
    roots = _worker_runtime_roots(manifest, shard)
    for key in ("schedule_path", "schedule_result_path", "transcript_path"):
        _resolve_required_path(manifest, payload[key], roots, label=key)
    for value in payload["evidence_paths"]:
        _resolve_required_path(manifest, value, roots, label="evidence")


def _validate_schedule_result(
    manifest: FrozenBatchManifest,
    shard: FrozenShardSpec,
    result: Mapping[str, Any],
    frame: Mapping[str, Any],
    runtime_roots: Sequence[Path],
) -> None:
    required = {
        "schema_version", "schedule_id", "revision", "execution_status",
        "required_target_count", "passed_target_count", "targets",
    }
    if set(result) != required:
        raise ValueError("schedule result fields do not match the authoritative schema")
    if result.get("schema_version") != 1:
        raise ValueError("unsupported schedule result schema")
    if str(result.get("schedule_id") or "") != shard.schedule_id:
        raise ValueError("schedule result identity mismatch")
    if dict(result.get("revision") or {}) != dict(manifest.revision):
        raise ValueError("schedule result repository revision mismatch")
    if frame and result.get("execution_status") != frame.get("execution_status"):
        raise ValueError("result frame and schedule result status disagree")
    rows = result.get("targets")
    if not isinstance(rows, list) or len(rows) != len(shard.target_ids):
        raise ValueError("schedule result target cardinality mismatch")
    observed = []
    passed = 0
    lane_paths: list[Path] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("schedule result target row is not an object")
        identity = (str(row.get("target_id") or ""), str(row.get("edge_key") or ""))
        observed.append(identity)
        status = str(row.get("status") or "")
        if status not in {"passed", "failed"}:
            raise ValueError("schedule result target status is invalid")
        passed += status == "passed"
        if status == "passed":
            lane = row.get("lane_evidence")
            if not isinstance(lane, Mapping) or "dynamic_dual_ai" not in lane:
                raise ValueError("passed target has no dynamic_dual_ai evidence binding")
            for lane_name, lane_item in lane.items():
                if lane_name not in {"dynamic_dual_ai", "real_cli"} or not isinstance(
                    lane_item, Mapping
                ):
                    raise ValueError("passed target has an invalid evidence lane binding")
                lane_paths.append(_resolve_required_path(
                    manifest,
                    lane_item.get("evidence_path"),
                    runtime_roots,
                    label="lane evidence",
                ))
        elif not str(row.get("diagnostic_path") or ""):
            raise ValueError("failed target has no diagnostic path")
    if observed != list(zip(shard.target_ids, shard.edge_keys, strict=True)):
        raise ValueError("schedule result targets differ from the frozen target order")
    if result.get("required_target_count") != len(rows):
        raise ValueError("schedule result required target count is false")
    if result.get("passed_target_count") != passed:
        raise ValueError("schedule result passed target count is false")
    status = result.get("execution_status")
    if (status == "complete") != (passed == len(rows)):
        raise ValueError("schedule completion status contradicts target outcomes")
    declared_paths = [
        _resolve_required_path(manifest, item, runtime_roots, label="evidence")
        for item in frame.get("evidence_paths") or ()
    ]
    if sorted(map(str, declared_paths)) != sorted(map(str, lane_paths)):
        raise ValueError("result evidence paths differ from target lane bindings")


def _diagnostic_paths_from_result(
    manifest: FrozenBatchManifest,
    schedule_result: Mapping[str, Any],
    runtime_roots: Sequence[Path],
) -> list[tuple[Mapping[str, Any], Path]]:
    diagnostics: list[tuple[Mapping[str, Any], Path]] = []
    for row in schedule_result.get("targets") or ():
        if isinstance(row, Mapping) and row.get("status") == "failed":
            diagnostics.append((row, _resolve_required_path(
                manifest, row.get("diagnostic_path"), runtime_roots, label="diagnostic"
            )))
    return diagnostics


def _known_artifact_path(
    manifest: FrozenBatchManifest,
    shard: FrozenShardSpec | None,
    payload: Mapping[str, Any],
    payload_key: str,
    filename: str,
) -> Path | None:
    declared = _resolve_artifact_path(manifest, payload.get(payload_key))
    if declared is not None:
        return declared
    if shard is None:
        return None
    candidates = (
        Path(shard.runtime_root) / filename,
        Path(manifest.repo_root) / ".agent" / "dynamic-chaos" / shard.session_id / filename,
    )
    return next((path.resolve() for path in candidates if path.is_file()), None)


def _worker_runtime_roots(
    manifest: FrozenBatchManifest, shard: FrozenShardSpec
) -> tuple[Path, ...]:
    return (
        Path(shard.runtime_root).resolve(),
        (
            Path(manifest.repo_root)
            / ".agent" / "dynamic-chaos" / shard.session_id
        ).resolve(),
    )


def _resolve_required_path(
    manifest: FrozenBatchManifest,
    value: Any,
    roots: Sequence[Path],
    *,
    label: str,
) -> Path:
    path = _resolve_artifact_path(manifest, value)
    if path is None or not path.is_file():
        raise ValueError(f"{label} path is missing")
    _require_contained(path, roots, label=label)
    return path


def _require_contained(path: Path, roots: Sequence[Path], *, label: str) -> None:
    resolved = path.resolve(strict=True)
    for root in roots:
        try:
            resolved.relative_to(root.resolve())
            return
        except ValueError:
            continue
    raise ValueError(f"{label} path escapes the shard runtime roots")


def _resolve_artifact_path(
    manifest: FrozenBatchManifest, value: Any
) -> Path | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    path = Path(raw)
    if raw == "/workspace" or raw.startswith("/workspace/"):
        path = Path(manifest.repo_root) / Path(raw).relative_to("/workspace")
    elif not path.is_absolute():
        path = Path(manifest.repo_root) / path
    return path.resolve()


def _synthetic_interruption_result(
    manifest: FrozenBatchManifest, shard: FrozenShardSpec, exc: BaseException
) -> ShardResult:
    now = time.time_ns()
    runtime = Path(shard.runtime_root)
    runtime.mkdir(parents=True, exist_ok=True)
    stderr_path = runtime / "worker.stderr.redacted.txt"
    receipt_path = runtime / "synthetic-cleanup-receipt.json"
    if not stderr_path.exists():
        _atomic_write(stderr_path, b"", mode=0o600)
    receipt = {
        "receipt_id": "",
        "schema_version": CLEANUP_RECEIPT_SCHEMA_VERSION,
        "shard_id": shard.shard_id,
        "execution_id": shard.execution_id,
        "host_proof": {},
        "container_proof": {},
        "worker_identity": {},
        "exit_code": None,
        "actions": ["orchestrator_task_exception"],
        "errors": ["synthetic interruption has no process cleanup proof"],
        "cleaned": False,
        "started_at_ns": now,
        "finished_at_ns": now,
    }
    receipt["receipt_id"] = _content_hash({k: v for k, v in receipt.items() if k != "receipt_id"})
    _write_immutable_json(receipt_path, receipt)
    return ShardResult(
        shard_id=shard.shard_id, classification="infrastructure_interrupted",
        attempt_count=1, started_at_ns=now, finished_at_ns=now, exit_code=None,
        target_hash=shard.target_hash, response_hashes=(), decision_hashes=(),
        transcript_hash="", schedule_result_hash="", evidence_hashes=(),
        diagnostic_hashes=(), evidence_ids=(),
        diagnostic_ids=(str(receipt["receipt_id"]),),
        stderr_hash=_sha256_file(stderr_path),
        stderr_path=str(stderr_path), stderr_truncated=False,
        cleanup_receipt_path=str(receipt_path),
        cleanup_receipt_hash=_sha256_file(receipt_path),
        reason=str(redact(f"{type(exc).__name__}: {exc}"))[:1000],
    )


def _not_started_interruption_result(
    manifest: FrozenBatchManifest, shard: FrozenShardSpec, exc: BaseException
) -> ShardResult:
    now = time.time_ns()
    runtime = Path(shard.runtime_root)
    runtime.mkdir(parents=True, exist_ok=True)
    stderr_path = runtime / "worker.stderr.redacted.txt"
    receipt_path = runtime / "not-started-cleanup-receipt.json"
    _atomic_write(stderr_path, b"", mode=0o600)
    receipt = {
        "receipt_id": "",
        "schema_version": CLEANUP_RECEIPT_SCHEMA_VERSION,
        "shard_id": shard.shard_id,
        "execution_id": shard.execution_id,
        "host_proof": {
            "registered_processes": [],
            "survivor_scans": [],
            "errors": [],
            "cleaned": True,
        },
        "container_proof": {
            "registered_processes": [],
            "zero_survivor_scans": [],
            "errors": [],
            "cleaned": True,
        },
        "worker_identity": {},
        "exit_code": None,
        "actions": ["worker_not_started"],
        "errors": [],
        "cleaned": True,
        "started_at_ns": now,
        "finished_at_ns": now,
    }
    receipt["receipt_id"] = _content_hash(
        {key: value for key, value in receipt.items() if key != "receipt_id"}
    )
    _write_immutable_json(receipt_path, receipt)
    return ShardResult(
        shard_id=shard.shard_id,
        classification="infrastructure_interrupted",
        attempt_count=1,
        started_at_ns=now,
        finished_at_ns=now,
        exit_code=None,
        target_hash=shard.target_hash,
        response_hashes=(),
        decision_hashes=(),
        transcript_hash="",
        schedule_result_hash="",
        evidence_hashes=(),
        diagnostic_hashes=(),
        evidence_ids=(),
        diagnostic_ids=(str(receipt["receipt_id"]),),
        stderr_hash=_sha256_file(stderr_path),
        stderr_path=str(stderr_path),
        stderr_truncated=False,
        cleanup_receipt_path=str(receipt_path),
        cleanup_receipt_hash=_sha256_file(receipt_path),
        reason=str(redact(f"{type(exc).__name__}: worker not started"))[:1000],
    )


def _append_discovery_results(
    manifest: FrozenBatchManifest,
    results: Sequence[ShardResult],
) -> tuple[str, ...]:
    by_shard = {shard.shard_id: shard for shard in manifest.shards}
    attempt_ids: list[str] = []
    empty_hash = hashlib.sha256(b"").hexdigest()
    for result in results:
        shard = by_shard[result.shard_id]
        diagnostic_ids = result.diagnostic_ids or (result.cleanup_receipt_hash,)
        stable_coverage_ids = shard.edge_keys if shard.lane == "edge" else ()
        journey_ids = shard.target_ids if shard.lane == "journey" else ()
        attempt = build_discovery_attempt(
            batch_id=manifest.batch_id,
            shard_id=result.shard_id,
            stable_coverage_ids=stable_coverage_ids,
            journey_ids=journey_ids,
            revision=manifest.revision,
            schedule_hash=shard.schedule_id,
            response_hashes=result.response_hashes,
            decision_hashes=result.decision_hashes,
            transcript_hash=result.transcript_hash or empty_hash,
            persona=" | ".join(shard.personas),
            input_classes=shard.input_classes,
            sequence_ids=shard.sequence_ids,
            factor_ids=shard.factor_ids,
            started_at=_timestamp_from_ns(result.started_at_ns),
            finished_at=_timestamp_from_ns(result.finished_at_ns),
            classification=result.classification,
            diagnostic_ids=diagnostic_ids,
            evidence_ids=result.evidence_ids,
            defect_id=(
                diagnostic_ids[0] if result.classification == "product_failed" else ""
            ),
        )
        append_discovery_attempt(manifest.discovery_ledger_path, attempt)
        attempt_ids.append(attempt.attempt_id)
    return tuple(attempt_ids)


def _timestamp_from_ns(value: int) -> str:
    return datetime.fromtimestamp(value / 1_000_000_000, tz=timezone.utc).isoformat()


def manifest_payload(manifest: FrozenBatchManifest) -> dict[str, Any]:
    return {
        "manifest_id": manifest.manifest_id,
        "batch_id": manifest.batch_id,
        **_manifest_unsigned_payload(manifest),
    }


def result_index_payload(index: BatchResultIndex) -> dict[str, Any]:
    return {
        "index_id": index.index_id,
        "batch_id": index.batch_id,
        "manifest_id": index.manifest_id,
        "revision": dict(index.revision),
        "execution_authority_mode": index.execution_authority_mode,
        "execution_status": index.execution_status,
        "release_status": index.release_status,
        "scheduled": index.scheduled,
        "started": index.started,
        "completed": index.completed,
        "discovery_attempt_ids": list(index.discovery_attempt_ids),
        "classification_counts": dict(index.classification_counts),
        "batch_survivor_proof": dict(index.batch_survivor_proof),
        "pty_authority_trust_root_id": index.pty_authority_trust_root_id,
        "pty_authority_public_key_b64": index.pty_authority_public_key_b64,
        "controller_signature_b64": index.controller_signature_b64,
        "shards": [_result_payload(item) for item in index.shards],
        "schema_version": index.schema_version,
    }


def validate_completed_shard_result(
    shard: FrozenShardSpec,
    payload: Mapping[str, Any],
) -> ShardResult:
    """Rebuild and validate one persisted shard result and its cleanup proof."""

    expected_fields = {item.name for item in fields(ShardResult)}
    if set(payload) != expected_fields:
        raise ValueError("persisted shard result fields do not match the schema")
    normalized = dict(payload)
    for key in (
        "response_hashes",
        "decision_hashes",
        "evidence_hashes",
        "diagnostic_hashes",
        "evidence_ids",
        "diagnostic_ids",
    ):
        value = normalized.get(key)
        if not isinstance(value, list) or not all(
            isinstance(item, str) for item in value
        ):
            raise ValueError(f"persisted shard result {key} is invalid")
        normalized[key] = tuple(value)
    result = ShardResult(**normalized)
    if result.shard_id != shard.shard_id:
        raise ValueError("persisted shard result identity mismatch")
    _validate_composite_cleanup_receipt(shard, result)
    return result


def _manifest_unsigned_payload(manifest: FrozenBatchManifest) -> dict[str, Any]:
    return {
        "created_at_ns": manifest.created_at_ns,
        "manifest_path": manifest.manifest_path,
        "discovery_ledger_path": manifest.discovery_ledger_path,
        "repo_root": manifest.repo_root,
        "revision": dict(manifest.revision),
        "shard_count": manifest.shard_count,
        "max_concurrency": manifest.max_concurrency,
        "pty_authority_trust_root_id": manifest.pty_authority_trust_root_id,
        "pty_authority_public_key_b64": manifest.pty_authority_public_key_b64,
        "shards": [_shard_payload(item) for item in manifest.shards],
        "required_env_names": list(manifest.required_env_names),
        "timeout_policy": asdict(manifest.timeout_policy),
        "stderr_cap_bytes": manifest.stderr_cap_bytes,
        "expected_obligation_set_hash": (
            manifest.expected_obligation_set_hash
        ),
        "worker_runtime": manifest.worker_runtime,
        "formal_profile": manifest.formal_profile,
        "controller_owned_execution": (
            manifest.controller_owned_execution
        ),
        "scheduler_schema_version": manifest.scheduler_schema_version,
        "discovery_schema_version": manifest.discovery_schema_version,
        "schema_version": manifest.schema_version,
    }


def _shard_payload(shard: FrozenShardSpec) -> dict[str, Any]:
    payload = asdict(shard)
    payload["command"] = list(shard.command)
    payload["target_ids"] = list(shard.target_ids)
    payload["edge_keys"] = list(shard.edge_keys)
    payload["personas"] = list(shard.personas)
    payload["goals"] = list(shard.goals)
    payload["input_classes"] = list(shard.input_classes)
    payload["sequence_ids"] = list(shard.sequence_ids)
    payload["factor_ids"] = list(shard.factor_ids)
    return payload


def _result_payload(result: ShardResult) -> dict[str, Any]:
    payload = asdict(result)
    for key in (
        "response_hashes", "decision_hashes", "evidence_hashes", "diagnostic_hashes",
        "evidence_ids", "diagnostic_ids",
    ):
        payload[key] = list(payload[key])
    return payload


def _bind_docker_execution_environment(
    command: Sequence[str],
    *,
    execution_id: str,
    container_receipt_dir: Path,
    repo_root: Path,
) -> tuple[str, ...]:
    frozen = tuple(command)
    insert_at = _docker_exec_insert_index(frozen)
    if insert_at is None:
        return frozen
    try:
        receipt_in_container = Path("/workspace") / container_receipt_dir.relative_to(
            repo_root
        )
    except ValueError as exc:
        raise ValueError("Docker cleanup receipts must be contained by the repository") from exc
    bindings = (
        "-e", f"{EXECUTION_ID_ENV}={execution_id}",
        "-e", f"{RECEIPT_DIR_ENV}={receipt_in_container}",
    )
    return (*frozen[:insert_at], *bindings, *frozen[insert_at:])


def _validate_bound_docker_environment(
    shard: FrozenShardSpec,
    repo_root: Path,
) -> None:
    insert_at = _docker_exec_insert_index(shard.command)
    if insert_at is None:
        return
    expected_execution = f"{EXECUTION_ID_ENV}={shard.execution_id}"
    try:
        expected_receipt = Path("/workspace") / Path(
            shard.container_cleanup_receipt_dir
        ).relative_to(repo_root)
    except ValueError as exc:
        raise ValueError("Docker cleanup receipt directory escapes the repository") from exc
    values = tuple(shard.command)
    if expected_execution not in values:
        raise ValueError(f"Docker command lost its execution id: {shard.shard_id}")
    receipt_values = [
        item for item in values if item.startswith(f"{RECEIPT_DIR_ENV}=")
    ]
    if receipt_values != [f"{RECEIPT_DIR_ENV}={expected_receipt}"]:
        raise ValueError(f"Docker command lost its cleanup receipt directory: {shard.shard_id}")


def _validate_formal_linux_command(
    shard: FrozenShardSpec,
    repo_root: Path,
    *,
    broker_decision_timeout_seconds: float,
) -> None:
    """Keep formal workers on the frozen in-container Linux implementation."""

    expected_module = (
        "tests.agent_live.journey_simulator_bridge"
        if shard.lane == "journey"
        else "tests.agent_live.codex_simulator_bridge"
    )
    command = tuple(shard.command)
    expected_prefix = (
        str(repo_root / ".venv-adk" / "bin" / "python"),
        "-m",
        expected_module,
    )
    if command[:3] != expected_prefix:
        raise ValueError(
            f"formal shard escaped its Linux worker implementation: {shard.shard_id}"
        )
    required_pairs = {
        "--repo-root": str(repo_root),
        "--runtime": "linux",
        "--session-id": shard.session_id,
    }
    for flag, expected in required_pairs.items():
        positions = [index for index, value in enumerate(command) if value == flag]
        if len(positions) != 1 or positions[0] + 1 >= len(command):
            raise ValueError(f"formal shard command lost {flag}: {shard.shard_id}")
        if command[positions[0] + 1] != expected:
            raise ValueError(f"formal shard command changed {flag}: {shard.shard_id}")
    timeout_positions = [
        index for index, value in enumerate(command)
        if value == "--decision-timeout-seconds"
    ]
    if len(timeout_positions) != 1 or timeout_positions[0] + 1 >= len(command):
        raise ValueError(
            f"formal shard command lost --decision-timeout-seconds: {shard.shard_id}"
        )
    try:
        worker_timeout = float(command[timeout_positions[0] + 1])
    except ValueError as exc:
        raise ValueError(
            f"formal shard command has an invalid decision timeout: {shard.shard_id}"
        ) from exc
    if worker_timeout <= broker_decision_timeout_seconds:
        raise ValueError(
            "formal worker decision timeout must exceed the broker timeout: "
            f"{shard.shard_id}"
        )


def _docker_exec_insert_index(command: Sequence[str]) -> int | None:
    if tuple(command[:2]) == ("docker", "exec"):
        return 2
    if tuple(command[:3]) == ("docker", "compose", "exec"):
        return 3
    return None


def _is_docker_exec_command(command: Sequence[str]) -> bool:
    return _docker_exec_insert_index(command) is not None


def _single_guard_receipt_path(receipt_dir: Path) -> Path:
    try:
        paths = tuple(receipt_dir.glob("container-cleanup-receipt-*.json"))
    except OSError as exc:
        raise ValueError(f"cannot enumerate process guard receipts: {exc}") from exc
    if len(paths) != 1:
        raise ValueError(f"expected exactly one process guard receipt, found {len(paths)}")
    return paths[0]


def _validated_guard_proof(
    path: Path,
    *,
    execution_id: str,
    required_roles: Sequence[str],
    allowed_roots: Sequence[Path],
) -> dict[str, Any]:
    _require_contained(path, allowed_roots, label="process guard receipt")
    payload = _load_json(path)
    required = {
        "receipt_id", "schema_version", "execution_id", "container_id",
        "pid_namespace_inode", "bridge_identity", "registered_processes",
        "term_decisions", "kill_decisions", "reap_results", "survivor_scans",
        "zero_survivor_scans", "errors", "cleaned", "started_at_ns",
        "finished_at_ns",
    }
    if not isinstance(payload, Mapping) or set(payload) != required:
        raise ValueError("process guard receipt fields do not match the authoritative schema")
    if payload["schema_version"] != 1:
        raise ValueError("unsupported process guard receipt schema")
    if payload["execution_id"] != execution_id:
        raise ValueError("process guard receipt execution id mismatch")
    unsigned = {key: value for key, value in payload.items() if key != "receipt_id"}
    receipt_id = hashlib.sha256(_guard_canonical_json(unsigned)).hexdigest()
    if payload["receipt_id"] != receipt_id:
        raise ValueError("process guard receipt content hash is stale")
    if path.name != f"container-cleanup-receipt-{receipt_id}.json":
        raise ValueError("process guard receipt filename is not content-addressed")
    if not str(payload["container_id"]):
        raise ValueError("process guard receipt has no container/host identity")
    if int(payload["pid_namespace_inode"]) <= 0:
        raise ValueError("process guard receipt has no PID namespace identity")

    registered_rows = payload["registered_processes"]
    if not isinstance(registered_rows, list) or not registered_rows:
        raise ValueError("process guard receipt has no registered processes")
    registered: set[tuple[int, int, int, int]] = set()
    roles: set[str] = set()
    for row in registered_rows:
        identity = _identity_tuple(row)
        if identity in registered:
            raise ValueError("process guard receipt repeats a registered identity")
        registered.add(identity)
        row_roles = row.get("roles") if isinstance(row, Mapping) else None
        if not isinstance(row_roles, list) or not all(
            isinstance(role, str) and role for role in row_roles
        ):
            raise ValueError("registered process roles are invalid")
        roles.update(row_roles)
    if not set(required_roles).issubset(roles):
        raise ValueError("process guard receipt is missing required registered roles")
    if _identity_tuple(payload["bridge_identity"]) not in registered:
        raise ValueError("process guard bridge/observer identity was not registered")

    term_identities = _validate_signal_decisions(
        payload["term_decisions"], registered, expected_signal="SIGTERM"
    )
    kill_identities = _validate_signal_decisions(
        payload["kill_decisions"], registered, expected_signal="SIGKILL"
    )
    if not kill_identities.issubset(term_identities):
        raise ValueError("process guard issued KILL without a TERM decision")

    zero_scans = payload["zero_survivor_scans"]
    if not isinstance(zero_scans, list) or len(zero_scans) != 2:
        raise ValueError("process guard receipt lacks two stable zero-survivor scans")
    for scan in zero_scans:
        if not isinstance(scan, Mapping) or set(scan) != {
            "scan_index", "purpose", "scanned_at_ns", "survivors", "scan_errors"
        }:
            raise ValueError("process guard zero-survivor scan is malformed")
        if (
            scan["purpose"] != "zero_survivor_verification"
            or scan["survivors"]
            or scan["scan_errors"]
        ):
            raise ValueError("process guard zero-survivor scans are not clean")
    if (
        int(zero_scans[1]["scan_index"]) <= int(zero_scans[0]["scan_index"])
        or int(zero_scans[1]["scanned_at_ns"]) <= int(zero_scans[0]["scanned_at_ns"])
    ):
        raise ValueError("process guard zero-survivor scans are not stable and ordered")
    survivor_scans = payload["survivor_scans"]
    if not isinstance(survivor_scans, list) or survivor_scans[-2:] != zero_scans:
        raise ValueError("process guard final scans differ from its zero-survivor proof")

    reap_results = payload["reap_results"]
    if not isinstance(reap_results, list) or not all(
        isinstance(row, Mapping)
        and set(row) == {"pid", "reaped", "return_code", "error"}
        and int(row.get("pid") or 0) > 0
        and isinstance(row.get("reaped"), bool)
        for row in reap_results
    ):
        raise ValueError("process guard reap results are malformed")
    if payload["cleaned"]:
        if payload["errors"]:
            raise ValueError("clean process guard receipt contains errors")
        if any(not bool(row.get("reaped")) for row in reap_results):
            raise ValueError("clean process guard receipt contains an unreaped process")

    return {
        "receipt_id": receipt_id,
        "path": str(path.resolve()),
        "sha256": _sha256_file(path),
        "execution_id": execution_id,
        "container_id": str(payload["container_id"]),
        "pid_namespace_inode": int(payload["pid_namespace_inode"]),
        "registered_processes": registered_rows,
        "term_decisions": payload["term_decisions"],
        "kill_decisions": payload["kill_decisions"],
        "reap_results": payload["reap_results"],
        "zero_survivor_scans": zero_scans,
        "errors": payload["errors"],
        "cleaned": bool(payload["cleaned"]),
    }


def _validate_signal_decisions(
    rows: Any,
    registered: set[tuple[int, int, int, int]],
    *,
    expected_signal: str,
) -> set[tuple[int, int, int, int]]:
    if not isinstance(rows, list):
        raise ValueError("process guard signal decisions are not a list")
    identities: set[tuple[int, int, int, int]] = set()
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {
            "identity", "signal", "sent", "outcome", "decided_at_ns"
        } or row.get("signal") != expected_signal:
            raise ValueError("process guard signal decision is malformed")
        if not isinstance(row.get("sent"), bool) or int(row.get("decided_at_ns") or 0) <= 0:
            raise ValueError("process guard signal decision metadata is malformed")
        if row["sent"] != (row.get("outcome") == "sent"):
            raise ValueError("process guard signal outcome contradicts its sent flag")
        identity = _identity_tuple(row.get("identity"))
        if identity not in registered:
            raise ValueError("process guard signaled an unregistered identity")
        identities.add(identity)
    return identities


def _identity_tuple(value: Any) -> tuple[int, int, int, int]:
    if not isinstance(value, Mapping):
        raise ValueError("process identity is not an object")
    try:
        identity = (
            int(value["pid"]), int(value["pgid"]), int(value["start_ticks"]),
            int(value["pid_namespace_inode"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("process identity fields are invalid") from exc
    if any(item <= 0 for item in identity):
        raise ValueError("process identity fields must be positive")
    return identity


def _validate_composite_cleanup_receipt(
    shard: FrozenShardSpec,
    result: ShardResult,
) -> None:
    path = Path(result.cleanup_receipt_path)
    _require_contained(path, (Path(shard.runtime_root),), label="composite cleanup receipt")
    if _sha256_file(path) != result.cleanup_receipt_hash:
        raise ValueError("composite cleanup receipt file hash mismatch")
    payload = _load_json(path)
    required = {
        "receipt_id", "schema_version", "shard_id", "execution_id", "host_proof",
        "container_proof", "worker_identity", "exit_code", "actions", "errors",
        "cleaned", "started_at_ns", "finished_at_ns",
    }
    if not isinstance(payload, Mapping) or set(payload) != required:
        raise ValueError("composite cleanup receipt fields do not match the schema")
    if payload["schema_version"] != CLEANUP_RECEIPT_SCHEMA_VERSION:
        raise ValueError("unsupported composite cleanup receipt schema")
    if payload["shard_id"] != shard.shard_id or payload["execution_id"] != shard.execution_id:
        raise ValueError("composite cleanup receipt identity mismatch")
    unsigned = {key: value for key, value in payload.items() if key != "receipt_id"}
    if payload["receipt_id"] != _content_hash(unsigned):
        raise ValueError("composite cleanup receipt content hash is stale")
    if not payload["cleaned"]:
        details = "; ".join(str(item) for item in payload["errors"])
        raise ValueError(
            "composite cleanup receipt does not prove cleanup"
            + (f": {details}" if details else "")
        )
    if payload["errors"]:
        raise ValueError("clean composite cleanup receipt contains errors")
    if payload["actions"] == ["worker_not_started"]:
        expected_host_proof = {
            "registered_processes": [],
            "survivor_scans": [],
            "errors": [],
            "cleaned": True,
        }
        expected_container_proof = {
            "registered_processes": [],
            "zero_survivor_scans": [],
            "errors": [],
            "cleaned": True,
        }
        if (
            payload["worker_identity"] != {}
            or payload["exit_code"] is not None
            or payload["host_proof"] != expected_host_proof
            or payload["container_proof"] != expected_container_proof
        ):
            raise ValueError("not-started cleanup receipt contains process state")
        return
    proof_contracts = {
        "host_proof": (
            ("host_worker", "docker_exec")
            if _is_docker_exec_command(shard.command)
            else ("host_worker",),
            (Path(shard.runtime_root) / "host-cleanup-receipts",),
        ),
        "container_proof": (
            ("container_bridge", "agent_process_group_leader"),
            (Path(shard.container_cleanup_receipt_dir),),
        ),
    }
    for key, (roles, roots) in proof_contracts.items():
        proof = payload[key]
        if not isinstance(proof, Mapping) or proof.get("execution_id") != shard.execution_id:
            raise ValueError(f"composite {key} execution id mismatch")
        if proof.get("cleaned") is not True:
            raise ValueError(f"composite {key} is incomplete")
        proof_path = Path(str(proof.get("path") or ""))
        if not proof_path.is_file() or _sha256_file(proof_path) != proof.get("sha256"):
            raise ValueError(f"composite {key} file binding is invalid")
        revalidated = _validated_guard_proof(
            proof_path,
            execution_id=shard.execution_id,
            required_roles=roles,
            allowed_roots=roots,
        )
        if revalidated != proof:
            raise ValueError(f"composite {key} summary differs from its guard receipt")
    worker = _identity_tuple(payload["worker_identity"])
    host_registered = {
        _identity_tuple(row) for row in payload["host_proof"]["registered_processes"]
    }
    if worker not in host_registered:
        raise ValueError("composite worker identity is not in the host proof")


async def _final_batch_survivor_proof(
    manifest: FrozenBatchManifest,
) -> dict[str, Any]:
    started_at_ns = time.time_ns()
    expected = tuple(shard.execution_id for shard in manifest.shards)
    first = await asyncio.to_thread(_scan_batch_execution_ids, expected)
    await asyncio.sleep(0.05)
    second = await asyncio.to_thread(_scan_batch_execution_ids, expected)
    errors = sorted(set((*first["scan_errors"], *second["scan_errors"])))
    cleaned = not errors and not first["survivors"] and not second["survivors"]
    unsigned = {
        "schema_version": BATCH_SURVIVOR_PROOF_SCHEMA_VERSION,
        "execution_ids": list(expected),
        "scans": [first, second],
        "errors": errors,
        "cleaned": cleaned,
        "started_at_ns": started_at_ns,
        "finished_at_ns": time.time_ns(),
    }
    return {"proof_id": _content_hash(unsigned), **unsigned}


def _scan_batch_execution_ids(execution_ids: Sequence[str]) -> dict[str, Any]:
    expected = {value.encode("utf-8"): value for value in execution_ids}
    token_name = EXECUTION_ID_ENV.encode("ascii")
    survivors: list[dict[str, Any]] = []
    errors: list[str] = []
    try:
        entries = tuple(Path("/proc").iterdir())
    except OSError as exc:
        return {"scanned_at_ns": time.time_ns(), "survivors": [], "scan_errors": [str(exc)]}
    for entry in entries:
        if not entry.name.isdecimal():
            continue
        pid = int(entry.name)
        try:
            before = _read_host_identity(pid)
            environ = (entry / "environ").read_bytes()
            after = _read_host_identity(pid)
        except (FileNotFoundError, ProcessLookupError):
            continue
        except OSError as exc:
            errors.append(f"cannot inspect host process {pid}: {exc}")
            continue
        if before != after:
            errors.append(f"host pid {pid} changed identity during batch scan")
            continue
        for token in environ.split(b"\0"):
            name, separator, value = token.partition(b"=")
            if separator and name == token_name and value in expected:
                survivors.append({
                    "execution_id": expected[value],
                    "identity": dict(after),
                })
                break
    survivors.sort(key=lambda row: (row["execution_id"], row["identity"]["pid"]))
    return {
        "scanned_at_ns": time.time_ns(),
        "survivors": survivors,
        "scan_errors": errors,
    }


def _read_host_identity(pid: int) -> dict[str, int]:
    root = Path("/proc") / str(pid)
    stat_text = (root / "stat").read_text(encoding="utf-8")
    namespace_inode = (root / "ns" / "pid").stat().st_ino
    closing_paren = stat_text.rfind(")")
    fields = stat_text[closing_paren + 2:].split() if closing_paren >= 0 else []
    if len(fields) <= 19:
        raise OSError(f"malformed /proc/{pid}/stat")
    try:
        return {
            "pid": pid,
            "pgid": int(fields[2]),
            "start_ticks": int(fields[19]),
            "pid_namespace_inode": namespace_inode,
        }
    except ValueError as exc:
        raise OSError(f"malformed /proc/{pid}/stat identity fields") from exc


def _guard_canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _default_command_factory(
    repo_root: Path,
    *,
    worker_runtime: str,
    decision_timeout_seconds: float,
) -> CommandFactory:
    def factory(
        index: int,
        target: Path,
        runtime: Path,
        session: str,
        seed: int,
        schedule_id: str,
    ) -> Sequence[str]:
        del index, runtime
        payload = _load_json(target)
        lane = _target_lane(payload)
        if worker_runtime == "docker":
            try:
                target_for_worker = Path("/workspace") / target.relative_to(repo_root)
            except ValueError as exc:
                raise ValueError("Docker target must be contained by the repository") from exc
            base = (
                "docker", "exec", "-i", "-w", "/workspace",
                "blockchain-node-benchmark-bench-1",
            )
            worker_root = "/workspace"
            python = ".venv-adk/bin/python"
        else:
            target_for_worker = target
            base = ()
            worker_root = str(repo_root)
            python = str(repo_root / ".venv-adk" / "bin" / "python")
        if lane == "edge":
            return (*base,
                python, "-m", "tests.agent_live.codex_simulator_bridge",
                "--targets", str(target_for_worker), "--repo-root", worker_root,
                "--seed", str(seed), "--session-id", session, "--runtime", "linux",
                "--decision-timeout-seconds", str(decision_timeout_seconds),
            )
        _, registry_import = _journey_target(payload)
        return (*base,
            python, "-m", "tests.agent_live.journey_simulator_bridge",
            "--journey-definition", str(target_for_worker),
            "--verifier-registry", registry_import,
            "--repo-root", worker_root, "--seed", str(seed),
            "--expected-schedule-id", schedule_id,
            "--session-id", session, "--runtime", "linux",
            "--decision-timeout-seconds", str(decision_timeout_seconds),
        )
    return factory


def _require_linux_control_plane() -> None:
    if sys.platform != "linux" or not Path("/proc/self/stat").is_file():
        raise RuntimeError(
            "batch execution requires a Linux control plane with /proc; "
            "macOS can freeze or inspect manifests but cannot produce qualifying evidence"
        )


def _target_rows(payload: Any) -> list[Mapping[str, Any]]:
    rows = payload.get("targets") if isinstance(payload, Mapping) else payload
    if not isinstance(rows, list):
        raise ValueError("target file must be a list or contain a targets list")
    if not all(isinstance(item, Mapping) for item in rows):
        raise ValueError("target rows must be JSON objects")
    allowed = {
        "target_id", "edge_key", "persona", "goal", "sequence_id",
        "tuple_ids", "scenario_id",
    }
    for index, row in enumerate(rows):
        unknown = sorted(set(row) - allowed)
        if unknown:
            raise ValueError(
                f"target row {index} contains unsupported or future-dialogue fields: "
                + ", ".join(unknown)
            )
    return list(rows)


def _target_lane(payload: Any) -> str:
    if isinstance(payload, Mapping) and "lane" in payload:
        lane = str(payload.get("lane") or "").strip()
    else:
        lane = "edge"
    if lane not in SHARD_LANES:
        raise ValueError(f"unsupported batch target lane: {lane or '<empty>'}")
    return lane


def _journey_target(payload: Any) -> tuple[Mapping[str, Any], str]:
    if not isinstance(payload, Mapping) or _target_lane(payload) != "journey":
        raise ValueError("Journey target must be an object with lane=journey")
    allowed = {
        "lane",
        "journey",
        "verifier_registry",
        "frozen_execution",
        "simulator_attestation_contract",
    }
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(
            "Journey target contains unsupported or future-dialogue fields: "
            + ", ".join(unknown)
        )
    journey = payload.get("journey")
    if not isinstance(journey, Mapping):
        raise ValueError("Journey target requires a journey object")
    registry_import = str(payload.get("verifier_registry") or "").strip()
    if not registry_import:
        raise ValueError("Journey target requires verifier_registry")
    return journey, registry_import


def _frozen_journey_execution(payload: Any) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        return {}
    raw = payload.get("frozen_execution")
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError("frozen Journey execution contract must be an object")
    required = {
        "obligation_id",
        "seed",
        "schedule_id",
        "schedule_hash",
        "subject_group",
        "revision_binding",
    }
    if set(raw) != required:
        raise ValueError("frozen Journey execution contract fields mismatch")
    if not str(raw.get("obligation_id") or "").strip():
        raise ValueError("frozen Journey execution has no obligation id")
    seed = raw.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("frozen Journey execution seed is invalid")
    for field in ("schedule_id", "schedule_hash"):
        value = str(raw.get(field) or "")
        if len(value) != 64:
            raise ValueError(f"frozen Journey execution {field} is invalid")
    revision = raw.get("revision_binding")
    if not isinstance(revision, Mapping) or not all(
        str(revision.get(field) or "").strip()
        for field in ("commit", "worktree_hash")
    ):
        raise ValueError("frozen Journey execution revision is incomplete")
    return dict(raw)


def _simulator_attestation_required(payload: Any) -> bool:
    if not isinstance(payload, Mapping):
        return False
    raw = payload.get("simulator_attestation_contract")
    if raw is None:
        return False
    expected = {
        "required": True,
        "identity_strength": "auditable_declaration_only",
        "scripted_actor_qualifies": False,
        "cryptographic_identity_claimed": False,
    }
    if raw != expected:
        raise ValueError("Journey simulator attestation contract is invalid")
    return True


def _reject_secret_bearing_command(command: Sequence[str]) -> None:
    markers = ("api_key=", "token=", "password=", "authorization=", "bearer ")
    lowered = " ".join(command).lower()
    if any(marker in lowered for marker in markers):
        raise ValueError("worker command must reference environment names, not secret values")


def _parse_frame(line: str, prefix: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(line[len(prefix):])
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid {prefix.strip()} frame") from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"{prefix.strip()} frame is not an object")
    return payload


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _content_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(65_536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, content: bytes, *, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        os.chmod(path, mode)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _write_immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"immutable artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (json.dumps(
        payload, ensure_ascii=False, indent=2, sort_keys=True
    ) + "\n").encode("utf-8")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    published = False
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(f"immutable artifact already exists: {path}") from exc
        published = True
        os.chmod(path, 0o444)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except Exception:
        if published and path.exists():
            os.chmod(path, 0o600)
            path.unlink()
            try:
                _fsync_parent_directory(path.parent)
            except OSError:
                pass
        raise
    finally:
        temporary.unlink(missing_ok=True)

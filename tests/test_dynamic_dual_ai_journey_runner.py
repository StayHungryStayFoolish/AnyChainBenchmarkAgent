"""Focused contracts for the response-driven open-journey runtime."""

from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping
from unittest.mock import patch

from agent.harness.input_identity import user_input_hash
from agent.harness.terminal_protocol import (
    TerminalProtocolError,
    TerminalSessionEvent,
    build_terminal_session_event,
    presentation_hash,
    validate_terminal_outcome_projection,
)
from tests.agent_live.chaos_scheduler import (
    build_journey_schedule,
    journey_schedule_payload,
)
from tests.agent_live.coverage_evidence import (
    PtyCliTurnRecord,
    RuntimeTurnEvent,
    VerifiedPostcondition,
    content_hash,
    pty_transcript_hash,
)
from tests.agent_live.codex_simulator_bridge import (
    build_simulator_attestation,
    simulator_context_binding,
    simulator_context_hash,
)
from tests.agent_live.dynamic_dual_ai_chaos import (
    ChaosRunConfig,
    DynamicDualAiJourneyRunner,
    JourneyDecisionProvenance,
    JourneyExternallyBlockedError,
    JourneyInfrastructureInterruptedError,
    JourneyPostconditionResult,
    JourneyPostconditionVerifierDefinition,
    JourneyProductFailure,
    JourneySimulatorContext,
    JourneySimulatorDecision,
    JourneySimulatorInvalidError,
    JourneyTerminalClassification,
    TerminalOutcomeObservation,
    build_journey_outcome_verifier_registry,
    build_journey_verifier_context,
    validate_journey_evidence_artifact,
    verify_journey_forbidden_outcomes,
    verify_journey_outcome,
)
from tests.agent_live.batch_orchestrator import (
    _ControllerJourneyLedger,
    _RunState,
    _controller_fact_payloads,
    _validate_and_recompute_journey_controller_facts,
)
from tests.agent_live.generate_harness_coverage_ledger import contract_variant_hash
from tests.agent_live.retained_regression_attestations import (
    build_source_contract,
    build_variant_contract,
)


REVISION = {"commit": "journey-test", "worktree_hash": "a" * 64}
EDGE_KEY = "opening::opening_next_action::fake_node::option"
EDGE = {
    "edge_key": EDGE_KEY,
    "edge_type": "question_option",
    "question_id": "opening_next_action",
    "contract_hash": contract_variant_hash({"id": "opening_next_action"}),
    "contract_variant_hash": "variant-hash",
    "action_type": "answer_pending",
    "applicable": True,
    "expected_postcondition": {"target_mode": "fake-node"},
}


def _independent_journey_outcome(
    turn_index: int,
    *,
    response: str = "Agent> fixture response",
) -> TerminalOutcomeObservation:
    transaction_id = f"00000000-0000-0000-0000-{turn_index:012d}"
    physical_thread_id = f"attempt:{transaction_id}"
    outcome = TerminalOutcomeObservation(
        schema_version=3,
        record_type="terminal_outcome_projection",
        projection_id=f"projection-{turn_index}",
        event_id=f"terminal-{turn_index}",
        transaction_id=transaction_id,
        product_authority_id="dynamic-dual-ai-chaos:journey-session",
        logical_thread_id="dynamic-dual-ai-chaos:journey-session",
        physical_thread_id=physical_thread_id,
        outcome="committed",
        failure_category="",
        diagnostic_hash="",
        base_revision=turn_index - 1,
        base_checkpoint_thread_id=(
            "head"
            if turn_index == 1
            else "attempt:"
            f"00000000-0000-0000-0000-{turn_index - 1:012d}"
        ),
        base_checkpoint_id=f"checkpoint-{turn_index - 1}",
        base_fingerprint=chr(96 + turn_index) * 64,
        attempt_checkpoint_id=f"checkpoint-{turn_index}",
        attempt_fingerprint=chr(97 + turn_index) * 64,
        product_revision=turn_index,
        product_checkpoint_thread_id=physical_thread_id,
        product_checkpoint_id=f"checkpoint-{turn_index}",
        product_fingerprint=chr(97 + turn_index) * 64,
        render_hash="f" * 64,
        runtime_event_id=f"runtime-{turn_index}",
        runtime_event_sequence=turn_index,
        runtime_event_payload_hash="9" * 64,
        presentation_hash=presentation_hash(response),
        origin_revision=REVISION,
        delivery_phase="live",
        projected_at="2026-07-27T00:00:00+00:00",
        record_hash="0" * 64,
    )
    payload = asdict(outcome)
    payload["record_hash"] = hashlib.sha256(
        json.dumps(
            {
                key: value
                for key, value in payload.items()
                if key != "record_hash"
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return validate_terminal_outcome_projection(payload)


def ready_at_turn_three(context):
    return JourneyPostconditionResult(
        "ready_state",
        context.current_event.turn_index >= 3,
        {"turn_index": context.current_event.turn_index},
    )


def ready_at_turn_two(context):
    return JourneyPostconditionResult(
        "ready_state",
        context.current_event.turn_index >= 2,
        {"turn_index": context.current_event.turn_index},
    )


def ready_at_turn_ninety_nine(context):
    return JourneyPostconditionResult(
        "ready_state",
        context.current_event.turn_index >= 99,
        {"turn_index": context.current_event.turn_index},
    )


def state_lost_at_turn_two(context):
    return JourneyPostconditionResult(
        "state_lost",
        context.current_event.turn_index >= 2,
        {"turn_index": context.current_event.turn_index},
    )


def ready_at_start(context):
    return JourneyPostconditionResult(
        postcondition_id="ready_state",
        satisfied=context.latest_turn is None,
        details={"source": "startup"},
    )


def state_is_not_lost(context):
    return JourneyPostconditionResult(
        "state_lost",
        False,
        {"turn_index": context.current_event.turn_index},
    )


def invalid_verifier_result(context):
    del context
    return False


class FakeTransport:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.submitted: list[str] = []
        self.closed = False

    def start(self, *, env: Mapping[str, str]) -> None:
        del env

    def read_complete_agent_response(self, *, timeout_seconds: float) -> str:
        del timeout_seconds
        if not self.responses:
            raise AssertionError("journey runner read more responses than expected")
        return self.responses.pop(0)

    def submit_bracketed_paste(self, message: str) -> None:
        self.submitted.append(message)

    def send_interrupt(self) -> None:
        raise AssertionError("journey runner unexpectedly interrupted the PTY")

    def close(self) -> None:
        self.closed = True


class FakeRuntimeEventStream:
    def __init__(self, events: list[RuntimeTurnEvent]) -> None:
        self.events = list(events)

    def mark_process_start(self) -> None:
        return None

    def capture_startup_snapshot(self):
        return (), ()

    def baseline(self) -> RuntimeTurnEvent:
        return self.events.pop(0)

    def mark_baseline(self) -> None:
        return None

    def next_event(self, *, timeout_seconds: float) -> RuntimeTurnEvent:
        del timeout_seconds
        if not self.events:
            raise RuntimeError("missing committed event")
        return self.events.pop(0)

    def assert_publication_fence(self, detour) -> None:
        if self.events:
            raise RuntimeError("unexpected runtime event at detour fence")
        if detour.runtime_event_fence_sequence != 0:
            raise RuntimeError("fake stream does not contain the detour fence")


class FakeTerminalOutcomeStream:
    def __init__(
        self,
        *,
        turn_count: int,
        startup_product_revision: int = 1,
        startup_product_fingerprint: str = "b" * 64,
        responses: list[str] | None = None,
    ) -> None:
        response_frames = list(responses or ())
        self.outcomes = [
            _independent_journey_outcome(
                index,
                response=(
                    response_frames[index - 1]
                    if index <= len(response_frames)
                    else f"Agent> fixture response {index}"
                ),
            )
            for index in range(1, turn_count + 1)
        ]
        self._outcome_cursor = 0
        self._startup_product_revision = startup_product_revision
        self._startup_product_fingerprint = startup_product_fingerprint

    def baseline_session_events(
        self,
        *,
        startup_response: str,
        session_id: str,
        session_purpose: str,
    ) -> tuple[TerminalSessionEvent, ...]:
        transaction_id = (
            "00000000-0000-0000-0000-"
            f"{self._startup_product_revision:012d}"
        )
        physical_thread_id = f"attempt:{transaction_id}"
        return (
            build_terminal_session_event(
                process_instance_id="test-process",
                session_id=session_id,
                session_purpose=session_purpose,
                provider="deepseek",
                model="deepseek-chat",
                auth_mode="api_key",
                provider_ready=True,
                startup_status="ready",
                failure_category="",
                product_authority_id=f"{session_purpose}:{session_id}",
                product_revision=self._startup_product_revision,
                product_checkpoint_thread_id=physical_thread_id,
                product_checkpoint_id=(
                    f"checkpoint-{self._startup_product_revision}"
                ),
                product_fingerprint=self._startup_product_fingerprint,
                runtime_event_fence_sequence=self._startup_product_revision,
                runtime_event_fence_terminal_event_id=(
                    f"terminal-{self._startup_product_revision}"
                ),
                runtime_event_fence_id=(
                    f"runtime-{self._startup_product_revision}"
                ),
                runtime_event_fence_hash="9" * 64,
                rendered_frame=startup_response,
                origin_revision=REVISION,
            ),
        )

    def mark_process_start(self) -> None:
        return None

    def capture_startup_snapshot(self):
        self._outcome_cursor = self._startup_product_revision
        return (), ()

    def mark_baseline(self) -> None:
        self._outcome_cursor = self._startup_product_revision

    def next_outcome(
        self,
        *,
        timeout_seconds: float,
    ) -> TerminalOutcomeObservation:
        del timeout_seconds
        if self._outcome_cursor >= len(self.outcomes):
            raise RuntimeError("missing terminal outcome")
        outcome = self.outcomes[self._outcome_cursor]
        self._outcome_cursor += 1
        return outcome

    def next_completion(
        self,
        *,
        timeout_seconds: float,
    ):
        return self.next_outcome(timeout_seconds=timeout_seconds)


class OrderedClock:
    def __init__(self) -> None:
        self.value = 100

    def __call__(self) -> int:
        self.value += 10
        return self.value


class DynamicDualAiJourneyRunnerTest(unittest.TestCase):
    def setUp(self) -> None:
        self._runtime_identity = patch(
            "tests.agent_live.dynamic_dual_ai_chaos.load_llm_config",
            return_value=SimpleNamespace(
                provider="deepseek",
                model="deepseek-chat",
            ),
        )
        self._runtime_identity.start()
        self.addCleanup(self._runtime_identity.stop)

    def _event(
        self,
        turn_index: int,
        submitted_input: str = "<fixture-input-not-specified>",
    ) -> RuntimeTurnEvent:
        transaction_id = f"00000000-0000-0000-0000-{turn_index:012d}"
        physical_thread_id = f"attempt:{transaction_id}"
        return RuntimeTurnEvent(
            schema_version=6,
            event_type="turn_committed",
            thread_id="journey-session",
            session_purpose="dynamic-dual-ai-chaos",
            before_fingerprint=chr(96 + turn_index) * 64,
            after_fingerprint=chr(97 + turn_index) * 64,
            turn_index=turn_index,
            active_group="opening",
            pending_question_id="opening_next_action",
            pending_contract={"id": "opening_next_action"},
            revision=REVISION,
            action_queue_types=(),
            observation="turn_committed",
            admitted_action_types=("answer_pending",) if turn_index > 1 else (),
            turn_receipt_summary={
                "turn_id": f"test:{turn_index}",
                "input_hash": user_input_hash(submitted_input),
                "admitted_action_ids": [],
                "execution_order": [],
            },
            pending_transition={
                "before_hash": hashlib.sha256(b"pending-before").hexdigest(),
                "after_hash": hashlib.sha256(b"pending-after").hexdigest(),
            },
            state_diff_hashes=(
                {"target_mode": {"before": "", "after": "f" * 64}}
                if turn_index > 1 else {}
            ),
            after_value_hashes={},
            next_result={"kind": "question", "question_id": "opening_next_action"},
            runtime_event_id=f"runtime-{turn_index}",
            runtime_event_sequence=turn_index,
            runtime_event_payload_hash="9" * 64,
            terminal_event_id=f"terminal-{turn_index}",
            transaction_id=transaction_id,
            terminal_outcome="committed",
            render_hash="f" * 64,
            base_revision=turn_index - 1,
            base_checkpoint_thread_id=(
                "head"
                if turn_index == 1
                else "attempt:"
                f"00000000-0000-0000-0000-{turn_index - 1:012d}"
            ),
            base_checkpoint_id=f"checkpoint-{turn_index - 1}",
            product_revision=turn_index,
            product_checkpoint_thread_id=physical_thread_id,
            product_checkpoint_id=f"checkpoint-{turn_index}",
            product_authority_id="dynamic-dual-ai-chaos:journey-session",
            physical_thread_id=physical_thread_id,
            attempt_checkpoint_id=f"checkpoint-{turn_index}",
        )

    def _schedule(
        self,
        *,
        max_turns: int = 3,
        verifier_input_contract=None,
        start_scenario: str = "opening",
    ):
        return build_journey_schedule(
            revision=REVISION,
            seed=271,
            journey={
                "journey_id": "response-driven-journey",
                "start_scenario": start_scenario,
                "persona": "operator who reacts to the current response",
                "mission": "reach a verified ready state without scripted turns",
                "allowed_risk_factors": ["change_direction"],
                "max_turns": max_turns,
                "terminal_outcome": {
                    "outcome_id": "ready",
                    "required_postcondition_ids": ["ready_state"],
                },
                "forbidden_outcomes": [{
                    "outcome_id": "lost_state",
                    "required_postcondition_ids": ["state_lost"],
                }],
                "verifier_input_contract": verifier_input_contract or {},
            },
        )

    def _retained_verifier_input(self):
        source = build_source_contract({
            "case_id": "journey-runner-retained",
            "source_turns_hash": content_hash(("first", "second")),
            "source_steps": [
                {
                    "step_id": "source-1",
                    "turn_index": 1,
                    "semantic_role": "capability_consultation",
                },
                {
                    "step_id": "source-2",
                    "turn_index": 2,
                    "semantic_role": "select_real_node",
                },
            ],
        })
        variant = build_variant_contract(
            variant="isomorphic",
            source_contract=source,
            declaration={
                "relation": "isomorphic_meaning",
                "minimum_attestations": 2,
            },
        )
        return {
            "mode": "response_driven_isomorphic",
            "source_contract": source,
            "source_contract_hash": content_hash(source),
            "variant_contract": variant,
            "variant_contract_hash": content_hash(variant),
        }

    def _decision(self, context: JourneySimulatorContext) -> JourneySimulatorDecision:
        return JourneySimulatorDecision(
            user_message=f"Responding to turn {context.turn_index}: continue safely.",
            persona=context.schedule.persona,
            mission=context.schedule.mission,
            rationale="Selected from the complete response just observed.",
        )

    def _registry(self, ready=ready_at_turn_three, lost=state_is_not_lost):
        return build_journey_outcome_verifier_registry((
            JourneyPostconditionVerifierDefinition(
                "ready_state", "ready-state", 1, "Proves the terminal ready state.", ready
            ),
            JourneyPostconditionVerifierDefinition(
                "state_lost", "state-lost", 1, "Detects forbidden state loss.", lost
            ),
        ))

    def test_controller_journey_callback_recomputes_and_deep_freezes(
        self,
    ) -> None:
        schedule = self._schedule()
        state = _RunState()
        ledger = _ControllerJourneyLedger(
            state,
            schedule=schedule,
            registry=self._registry(),
            edge_index={EDGE_KEY: dict(EDGE)},
        )
        initial = self._event(1)
        committed = self._event(2, "continue")
        mutable_context = {"nested": {"value": "original"}}
        turn = PtyCliTurnRecord(
            session_id="journey-session",
            turn_index=2,
            previous_agent_response="Agent> choose",
            user_message="continue",
            agent_response="Agent> not ready",
            provider="deepseek",
            model="deepseek-chat",
            before_fingerprint=initial.after_fingerprint,
            after_fingerprint=committed.after_fingerprint,
            transcript_hash=pty_transcript_hash(
                session_id="journey-session",
                turn_index=2,
                previous_agent_response="Agent> choose",
                user_message="continue",
                agent_response="Agent> not ready",
            ),
            previous_response_received_at_ns=100,
            user_message_submitted_at_ns=110,
            agent_response_received_at_ns=120,
        )
        decision = JourneyDecisionProvenance(
            turn_index=2,
            previous_response_hash=content_hash(
                turn.previous_agent_response
            ),
            selected_at_ns=105,
            submitted_at_ns=110,
            user_message_hash=content_hash(turn.user_message),
            persona=schedule.persona,
            mission=schedule.mission,
            rationale="response-bound",
            risk_factor_ids=(),
            execution_id="e" * 64,
            obligation_id=schedule.journey_id,
            broker_request_id="request-1",
            simulator_context_binding=mutable_context,
        )

        ledger(
            initial_event=initial,
            turn=turn,
            baseline_event=initial,
            committed_event=committed,
            terminal_outcome=_independent_journey_outcome(
                2,
                response=turn.agent_response,
            ),
            decision=decision,
        )
        mutable_context["nested"]["value"] = "mutated"

        frozen = _controller_fact_payloads(state, lane="journey")[0]
        self.assertFalse(
            frozen["turn_result"]["terminal_outcome"]["satisfied"]
        )
        self.assertEqual(
            frozen["decision"]["simulator_context_binding"]["nested"][
                "value"
            ],
            "original",
        )
        with self.assertRaises(TypeError):
            ledger(
                initial_event=initial,
                turn=turn,
                baseline_event=initial,
                committed_event=committed,
                terminal_outcome=_independent_journey_outcome(
                    2,
                    response=turn.agent_response,
                ),
                decision=decision,
                turn_result={
                    "terminal_outcome": {"satisfied": True},
                },
            )

    def test_persisted_controller_facts_are_independently_recomputed(
        self,
    ) -> None:
        schedule = self._schedule()
        registry = self._registry()
        state = _RunState()
        ledger = _ControllerJourneyLedger(
            state,
            schedule=schedule,
            registry=registry,
            edge_index={EDGE_KEY: dict(EDGE)},
        )
        initial = self._event(1)
        committed = self._event(2, "continue")
        turn = PtyCliTurnRecord(
            session_id="journey-session",
            turn_index=2,
            previous_agent_response="Agent> choose",
            user_message="continue",
            agent_response="Agent> not ready",
            provider="deepseek",
            model="deepseek-chat",
            before_fingerprint=initial.after_fingerprint,
            after_fingerprint=committed.after_fingerprint,
            transcript_hash=pty_transcript_hash(
                session_id="journey-session",
                turn_index=2,
                previous_agent_response="Agent> choose",
                user_message="continue",
                agent_response="Agent> not ready",
            ),
            previous_response_received_at_ns=100,
            user_message_submitted_at_ns=110,
            agent_response_received_at_ns=120,
        )
        simulator_context = {
            "session_id": turn.session_id,
            "turn_index": turn.turn_index,
            "previous_response_hash": content_hash(
                turn.previous_agent_response
            ),
            "previous_response_received_at_ns": (
                turn.previous_response_received_at_ns
            ),
            "schedule": {"schedule_id": schedule.schedule_id},
            "observed_edge_keys": [],
        }
        decision = JourneyDecisionProvenance(
            turn_index=2,
            previous_response_hash=content_hash(
                turn.previous_agent_response
            ),
            selected_at_ns=105,
            submitted_at_ns=110,
            user_message_hash=content_hash(turn.user_message),
            persona=schedule.persona,
            mission=schedule.mission,
            rationale="response-bound",
            risk_factor_ids=(),
            execution_id="e" * 64,
            obligation_id=schedule.journey_id,
            broker_request_id="request-1",
            simulator_context_binding=simulator_context_binding(
                simulator_context
            ),
        )
        terminal_outcome = _independent_journey_outcome(
            2,
            response=turn.agent_response,
        )
        initial_context = build_journey_verifier_context(
            schedule=schedule,
            initial_event=initial,
            current_event=initial,
            turns=(),
            events=(),
            decisions=(),
            transcript=(),
            observed_edge_keys=(),
            latest_turn=None,
        )
        ledger.observe_initial(
            initial_event=initial,
            terminal=verify_journey_outcome(
                schedule.terminal_outcome,
                initial_context,
                registry,
            ),
            forbidden=verify_journey_forbidden_outcomes(
                schedule,
                initial_context,
                registry,
            ),
        )
        ledger(
            initial_event=initial,
            turn=turn,
            baseline_event=initial,
            committed_event=committed,
            terminal_outcome=terminal_outcome,
            decision=decision,
        )
        fact = dict(_controller_fact_payloads(state, lane="journey")[0])
        initial_fact = dict(
            _controller_fact_payloads(
                state,
                lane="journey_initial",
            )[0]
        )
        for field_name in (
            "initial_event",
            "baseline_event",
            "committed_event",
        ):
            event_payload = dict(fact[field_name])
            event_payload.pop("runtime_event_payload_hash", None)
            event_payload["runtime_event_payload_hash"] = content_hash(
                event_payload
            )
            fact[field_name] = event_payload
        initial_event_payload = dict(initial_fact["initial_event"])
        initial_event_payload.pop("runtime_event_payload_hash", None)
        initial_event_payload["runtime_event_payload_hash"] = content_hash(
            initial_event_payload
        )
        initial_fact["initial_event"] = initial_event_payload
        fact["initial_event"] = dict(initial_event_payload)
        fact["baseline_event"] = dict(initial_event_payload)
        terminal_payload = dict(fact["terminal_outcome"])
        terminal_payload["runtime_event_payload_hash"] = fact[
            "committed_event"
        ]["runtime_event_payload_hash"]
        terminal_payload.pop("record_hash", None)
        terminal_payload["record_hash"] = content_hash(terminal_payload)
        fact["terminal_outcome"] = terminal_payload
        candidate = dict(fact["turn_result"])
        submitted_input_commitment = content_hash(
            "raw-submitted-input"
        )
        approved_decision = {
            "user_message": turn.user_message,
            "persona": decision.persona,
            "mission": decision.mission,
            "rationale": decision.rationale,
            "risk_factor_ids": [],
            "simulator_attestation": {},
            "variant_binding": {},
            "_controller_submitted_input_commitment": (
                submitted_input_commitment
            ),
            "_controller_source_user_message_hash": content_hash(
                turn.user_message
            ),
        }
        observation = {
            "turn_index": turn.turn_index,
            "previous_response_hash": content_hash(
                fact["turn"]["previous_agent_response"]
            ),
            "user_message_hash": content_hash(
                fact["turn"]["user_message"]
            ),
            "agent_response_hash": content_hash(
                fact["turn"]["agent_response"]
            ),
            "approved_decision": approved_decision,
            "approved_decision_hash": content_hash(approved_decision),
            "simulator_context": simulator_context,
            "simulator_context_hash": simulator_context_hash(
                simulator_context
            ),
            "submitted_input_commitment": (
                submitted_input_commitment
            ),
            "baseline_runtime_event_hash": fact["baseline_event"][
                "runtime_event_payload_hash"
            ],
            "committed_runtime_event_hash": fact["committed_event"][
                "runtime_event_payload_hash"
            ],
            "terminal_runtime_event_id": fact["committed_event"][
                "runtime_event_id"
            ],
            "terminal_runtime_event_sequence": (
                fact["committed_event"]["runtime_event_sequence"]
            ),
            "terminal_outcome_hash": content_hash(
                terminal_payload
            ),
            "turn_result_hash": content_hash(candidate),
        }
        with tempfile.TemporaryDirectory() as temporary:
            target_path = Path(temporary) / "target.json"
            target_path.write_text("{}\n", encoding="utf-8")
            shard = SimpleNamespace(
                target_path=str(target_path),
                target_hash=hashlib.sha256(
                    target_path.read_bytes()
                ).hexdigest(),
                verifier_registry_import="tests.fixture.registry",
                verifier_registry_id=registry.registry_id,
                schedule_id=schedule.schedule_id,
                seed=31,
                session_id="journey-session",
                execution_id="e" * 64,
            )
            manifest = SimpleNamespace(revision=REVISION)
            with (
                patch(
                    "tests.agent_live.batch_orchestrator._journey_target",
                    return_value=({}, shard.verifier_registry_import),
                ),
                patch(
                    "tests.agent_live.batch_orchestrator.build_journey_schedule",
                    return_value=schedule,
                ),
                patch(
                    "tests.agent_live.batch_orchestrator."
                    "validate_journey_schedule"
                ),
                patch(
                    "tests.agent_live.batch_orchestrator."
                    "load_verifier_registry",
                    return_value=registry,
                ),
                patch(
                    "tests.agent_live.batch_orchestrator.build_ledger",
                    return_value={
                        "revision": REVISION,
                        "edges": [dict(EDGE)],
                    },
                ),
            ):
                _validate_and_recompute_journey_controller_facts(
                    manifest=manifest,
                    shard=shard,
                    candidate_turns=[candidate],
                    observations=[observation],
                    controller_facts=[fact],
                    controller_initial_fact=initial_fact,
                    candidate_initial_verification=initial_fact[
                        "initial_verification"
                    ],
                )
                for field_name in (
                    "previous_response_hash",
                    "user_message_hash",
                    "agent_response_hash",
                    "baseline_runtime_event_hash",
                    "committed_runtime_event_hash",
                    "approved_decision_hash",
                    "simulator_context_hash",
                ):
                    forged_observation = dict(observation)
                    forged_observation[field_name] = "0" * 64
                    with self.assertRaisesRegex(
                        ValueError,
                        "differs from its observation",
                    ):
                        _validate_and_recompute_journey_controller_facts(
                            manifest=manifest,
                            shard=shard,
                            candidate_turns=[candidate],
                            observations=[forged_observation],
                            controller_facts=[fact],
                            controller_initial_fact=initial_fact,
                            candidate_initial_verification=initial_fact[
                                "initial_verification"
                            ],
                        )
                forged = json.loads(json.dumps(fact))
                forged["turn_result"]["terminal_outcome"][
                    "satisfied"
                ] = True
                with self.assertRaisesRegex(
                    ValueError,
                    "failed recomputation",
                ):
                    _validate_and_recompute_journey_controller_facts(
                        manifest=manifest,
                        shard=shard,
                        candidate_turns=[forged["turn_result"]],
                        observations=[{
                            **observation,
                            "turn_result_hash": content_hash(
                                forged["turn_result"]
                            ),
                        }],
                        controller_facts=[forged],
                        controller_initial_fact=initial_fact,
                        candidate_initial_verification=initial_fact[
                            "initial_verification"
                        ],
                    )

                forged_session = json.loads(json.dumps(fact))
                forged_committed = forged_session["committed_event"]
                forged_committed["thread_id"] = "another-session"
                forged_committed["product_authority_id"] = (
                    f"{forged_committed['session_purpose']}:"
                    "another-session"
                )
                forged_committed.pop(
                    "runtime_event_payload_hash",
                    None,
                )
                forged_committed[
                    "runtime_event_payload_hash"
                ] = content_hash(forged_committed)
                forged_terminal = forged_session["terminal_outcome"]
                forged_terminal["logical_thread_id"] = (
                    "another-session"
                )
                forged_terminal["product_authority_id"] = (
                    forged_committed["product_authority_id"]
                )
                forged_terminal["runtime_event_payload_hash"] = (
                    forged_committed["runtime_event_payload_hash"]
                )
                forged_terminal.pop("record_hash", None)
                forged_terminal["record_hash"] = content_hash(
                    forged_terminal
                )
                forged_observation = {
                    **observation,
                    "committed_runtime_event_hash": (
                        forged_committed[
                            "runtime_event_payload_hash"
                        ]
                    ),
                    "terminal_outcome_hash": content_hash(
                        forged_terminal
                    ),
                }
                with self.assertRaises(
                    (ValueError, TerminalProtocolError),
                ):
                    _validate_and_recompute_journey_controller_facts(
                        manifest=manifest,
                        shard=shard,
                        candidate_turns=[candidate],
                        observations=[forged_observation],
                        controller_facts=[forged_session],
                        controller_initial_fact=initial_fact,
                        candidate_initial_verification=initial_fact[
                            "initial_verification"
                        ],
                    )

    def _runner(
        self,
        root,
        *,
        schedule,
        simulator,
        transport,
        events,
        registry=None,
        startup_product_revision=1,
        startup_product_fingerprint="b" * 64,
        controller_turn_observer=None,
    ):
        return DynamicDualAiJourneyRunner(
            ChaosRunConfig.linux(root, session_id="journey-session"),
            simulator,
            ledger={"revision": REVISION, "edges": [dict(EDGE)]},
            schedule=schedule,
            postcondition_verifier_registry=registry or self._registry(),
            transport=transport,
            controller_turn_observer=controller_turn_observer,
            event_stream=FakeRuntimeEventStream(events),
            terminal_outcome_stream=FakeTerminalOutcomeStream(
                turn_count=max(len(events), startup_product_revision),
                startup_product_revision=startup_product_revision,
                startup_product_fingerprint=startup_product_fingerprint,
                responses=list(transport.responses),
            ),
            revision=REVISION,
            clock_ns=OrderedClock(),
        )

    def test_startup_satisfied_journey_persists_controller_initial_fact(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = _RunState()
            schedule = self._schedule(max_turns=1)
            registry = self._registry(ready=ready_at_start)
            observer = _ControllerJourneyLedger(
                state,
                schedule=schedule,
                registry=registry,
                edge_index={EDGE_KEY: dict(EDGE)},
            )
            runner = self._runner(
                Path(tmpdir),
                schedule=schedule,
                simulator=lambda context: self._decision(context),
                transport=FakeTransport(["Agent> Ready at startup."]),
                events=[self._event(1)],
                registry=registry,
                controller_turn_observer=observer,
            )

            result = self._run(runner)

            self.assertEqual(
                result.terminal_classification,
                JourneyTerminalClassification.PASSED,
            )
            self.assertEqual(len(result.turns), 0)
            self.assertEqual(
                len(_controller_fact_payloads(
                    state,
                    lane="journey_initial",
                )),
                1,
            )
            self.assertEqual(
                _controller_fact_payloads(
                    state,
                    lane="journey",
                ),
                (),
            )

    def test_initial_controller_fact_never_persists_raw_secrets(
        self,
    ) -> None:
        schedule = self._schedule(max_turns=1)
        registry = self._registry(ready=ready_at_start)
        state = _RunState()
        ledger = _ControllerJourneyLedger(
            state,
            schedule=schedule,
            registry=registry,
            edge_index={EDGE_KEY: dict(EDGE)},
        )
        secret = "https://user:secret-value@example.com"
        initial = replace(
            self._event(1),
            pending_contract={
                "id": "opening_next_action",
                "api_key": secret,
                "secret_ref": "semantic-secret:owning-capability",
            },
        )

        ledger.observe_initial(
            initial_event=initial,
            terminal=None,
            forbidden=(),
        )

        serialized = json.dumps(
            _controller_fact_payloads(
                state,
                lane="journey_initial",
            ),
            sort_keys=True,
        )
        self.assertNotIn(secret, serialized)
        self.assertNotIn("semantic-secret:", serialized)

    def _run(self, runner, *, reviewed_question=None):
        seed_state = {}
        question = {}
        if str(runner.schedule.start_scenario).startswith("resume"):
            question = {
                "id": "resume_harness_session",
                "options": [
                    {"value": "continue"},
                    {"value": "modify"},
                    {"value": "reset"},
                ],
            }
            seed_state = {"pending_question": dict(question)}
        if reviewed_question is not None:
            question = dict(reviewed_question)
            seed_state = {"pending_question": dict(question)}
        scenario = SimpleNamespace(
            scenario_id=runner.schedule.start_scenario,
            state_fingerprint="scenario-fingerprint",
            seed_state=seed_state,
            question=question,
        )
        with patch(
            "tests.agent_live.runtime_checkpoint.reviewed_scenario", return_value=scenario
        ), patch("tests.agent_live.runtime_checkpoint.seed_runtime_checkpoint"):
            return runner.run()

    def _verified_edge(self):
        return VerifiedPostcondition(
            verifier_id="normal-edge-verifier",
            passed=True,
            observed_coverage_ids=(EDGE_KEY,),
            admitted_typed_actions=("answer_pending",),
            state_diff={"target_mode": {"changed": True}},
            next_question_or_result={"question_id": "opening_next_action"},
            details={"errors": []},
        )

    def test_response_driven_test_transport_cannot_emit_qualifying_evidence(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            transport = FakeTransport([
                "Agent> Model config: provider=deepseek, model=deepseek-chat, auth=api_key\nFirst.",
                "Agent> Second.",
                "Agent> Terminal.",
            ])
            contexts = []

            def simulator(context):
                contexts.append(context)
                return self._decision(context)

            schedule = self._schedule()
            registry = self._registry()
            runner = self._runner(
                Path(tmpdir), schedule=schedule, simulator=simulator, transport=transport,
                events=[
                    self._event(1),
                    self._event(2, "Responding to turn 2: continue safely."),
                    self._event(3, "Responding to turn 3: continue safely."),
                ],
                registry=registry,
            )
            with patch(
                "tests.agent_live.dynamic_dual_ai_chaos.verify_runtime_postcondition",
                return_value=self._verified_edge(),
            ) as edge_verifier:
                result = self._run(runner)

            self.assertEqual(result.terminal_classification, JourneyTerminalClassification.PASSED)
            self.assertEqual(len(contexts), 2)
            self.assertEqual(edge_verifier.call_count, 2)
            self.assertEqual(result.observed_edge_keys, (EDGE_KEY,))
            artifact = json.loads(result.evidence_path.read_text())
            self.assertEqual(result.evidence_path.name, f"journey-{result.evidence_id}.json")
            self.assertFalse(artifact["qualifying_evidence"])
            self.assertEqual(
                artifact["qualification_reason"],
                "untrusted_test_transport",
            )
            with self.assertRaisesRegex(ValueError, "not marked as qualifying"):
                validate_journey_evidence_artifact(
                    artifact,
                    schedule=schedule,
                    verifier_registry=registry,
                    revision=REVISION,
                )
            self.assertNotIn("target_coverage_ids", artifact["turns"][0]["decision"])
            provenance = artifact["turns"][0]["decision_provenance"]
            self.assertRegex(provenance["previous_response_hash"], r"^[0-9a-f]{64}$")
            self.assertRegex(provenance["user_message_hash"], r"^[0-9a-f]{64}$")
            self.assertEqual(
                provenance["submitted_at_ns"],
                artifact["turns"][0]["turn_identity"]["user_message_submitted_at_ns"],
            )

    def test_startup_resume_is_bootstrap_not_a_simulator_source_step(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            resume_contract = {
                "id": "resume_harness_session",
                "options": [
                    {"value": "continue"},
                    {"value": "modify"},
                    {"value": "reset"},
                ],
            }
            baseline = replace(
                self._event(1),
                pending_question_id="resume_harness_session",
                pending_contract=resume_contract,
                next_result={
                    "kind": "question",
                    "question_id": "resume_harness_session",
                },
            )
            resumed = replace(
                self._event(2, "continue"),
                admitted_action_types=("resume_session",),
                pending_question_id="opening_next_action",
                pending_contract={"id": "opening_next_action"},
                next_result={
                    "kind": "question",
                    "question_id": "opening_next_action",
                },
            )
            final_message = "Responding to turn 3: continue safely."
            final = self._event(3, final_message)
            transport = FakeTransport([
                "Agent> Resume the retained configuration?",
                "Agent> Retained configuration restored. What next?",
                "Agent> Ready.",
            ])
            contexts: list[JourneySimulatorContext] = []

            def simulator(context):
                contexts.append(context)
                return self._decision(context)

            runner = self._runner(
                Path(tmpdir),
                schedule=self._schedule(max_turns=1),
                simulator=simulator,
                transport=transport,
                events=[baseline, resumed, final],
            )

            result = self._run(runner)

            self.assertEqual(
                result.terminal_classification,
                JourneyTerminalClassification.PASSED,
            )
            self.assertEqual(transport.submitted, ["continue", final_message])
            self.assertEqual(len(contexts), 1)
            self.assertEqual(contexts[0].turn_index, 3)
            self.assertIn(
                "Retained configuration restored",
                contexts[0].previous_agent_response,
            )
            self.assertEqual(len(result.turns), 1)
            self.assertEqual(
                result.turns[0].turn_index,
                3,
            )

    def test_resume_journey_leaves_resume_contract_to_simulator(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            resume_contract = {
                "id": "resume_harness_session",
                "options": [
                    {"value": "continue"},
                    {"value": "modify"},
                    {"value": "reset"},
                ],
            }
            baseline = replace(
                self._event(1),
                pending_question_id="resume_harness_session",
                pending_contract=resume_contract,
                next_result={
                    "kind": "question",
                    "question_id": "resume_harness_session",
                },
            )
            final_message = "continue"
            final = replace(
                self._event(2, final_message),
                admitted_action_types=("resume_session",),
            )
            transport = FakeTransport([
                "Agent> Resume the retained configuration?",
                "Agent> Retained configuration restored.",
            ])
            contexts: list[JourneySimulatorContext] = []

            def simulator(context):
                contexts.append(context)
                return replace(
                    self._decision(context),
                    user_message=final_message,
                )

            runner = self._runner(
                Path(tmpdir),
                schedule=self._schedule(
                    max_turns=1,
                    start_scenario="resume",
                ),
                simulator=simulator,
                transport=transport,
                events=[baseline, final],
                registry=self._registry(ready_at_turn_two),
            )

            result = self._run(runner)

            self.assertEqual(
                result.terminal_classification,
                JourneyTerminalClassification.PASSED,
            )
            self.assertEqual(transport.submitted, [final_message])
            self.assertEqual(len(contexts), 1)
            self.assertEqual(
                contexts[0].previous_agent_response,
                "Agent> Resume the retained configuration?",
            )

    def test_changed_resume_action_is_bootstrap_not_owned_by_journey(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            live_contract = {
                "id": "resume_harness_session",
                "options": [
                    {"value": "continue", "semantic_action": "resume_session"},
                    {"value": "modify", "semantic_action": "modify_session"},
                    {"value": "reset", "semantic_action": "reset_session"},
                ],
            }
            reviewed_contract = {
                "id": "resume_harness_session",
                "options": [
                    {"value": "continue", "semantic_action": "reset_session"},
                    {"value": "modify", "semantic_action": "modify_session"},
                    {"value": "reset", "semantic_action": "reset_session"},
                ],
            }
            baseline = replace(
                self._event(1),
                pending_question_id="resume_harness_session",
                pending_contract=live_contract,
                next_result={
                    "kind": "question",
                    "question_id": "resume_harness_session",
                },
            )
            resumed = replace(
                self._event(2, "continue"),
                admitted_action_types=("resume_session",),
                pending_question_id="opening_next_action",
                pending_contract={"id": "opening_next_action"},
                next_result={
                    "kind": "question",
                    "question_id": "opening_next_action",
                },
            )
            final_message = "Responding to turn 3: continue safely."
            transport = FakeTransport([
                "Agent> Resume the retained configuration?",
                "Agent> Retained configuration restored. What next?",
                "Agent> Ready.",
            ])
            contexts: list[JourneySimulatorContext] = []

            def simulator(context):
                contexts.append(context)
                return self._decision(context)

            runner = self._runner(
                Path(tmpdir),
                schedule=self._schedule(max_turns=1, start_scenario="resume"),
                simulator=simulator,
                transport=transport,
                events=[baseline, resumed, self._event(3, final_message)],
            )

            result = self._run(
                runner,
                reviewed_question=reviewed_contract,
            )

            self.assertEqual(
                result.terminal_classification,
                JourneyTerminalClassification.PASSED,
            )
            self.assertEqual(transport.submitted, ["continue", final_message])
            self.assertEqual(len(contexts), 1)

    def test_retained_journey_binds_live_codex_decisions_to_semantic_steps(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            verifier_input = self._retained_verifier_input()
            schedule = self._schedule(verifier_input_contract=verifier_input)

            def simulator(context):
                step_index = context.turn_index - 1
                binding = {
                    "source_step_id": f"source-{step_index}",
                    "semantic_role": (
                        "capability_consultation"
                        if step_index == 1
                        else "select_real_node"
                    ),
                }
                message = f"response-driven retained turn {step_index}"
                broker_request_id = f"request-{step_index}"
                unsigned = {
                    "previous_response_hash": content_hash(
                        context.previous_agent_response
                    ),
                    "broker_request_id": broker_request_id,
                    "user_message": message,
                    "persona": context.schedule.persona,
                    "mission": context.schedule.mission,
                    "rationale": "Selected from the complete response.",
                    "risk_factor_ids": [],
                    "variant_binding": binding,
                }
                attestation = build_simulator_attestation(
                    actor_kind="codex",
                    task_id="task-retained",
                    model="gpt-test",
                    request_id=broker_request_id,
                    previous_response_hash=unsigned["previous_response_hash"],
                    context_hash=simulator_context_hash({
                        "session_id": context.session_id,
                        "turn_index": context.turn_index,
                        "previous_response_hash": unsigned[
                            "previous_response_hash"
                        ],
                        "previous_response_received_at_ns": (
                            context.previous_response_received_at_ns
                        ),
                        "schedule": journey_schedule_payload(
                            context.schedule
                        ),
                        "observed_edge_keys": list(
                            context.observed_edge_keys
                        ),
                    }),
                    decision_hash=content_hash(unsigned),
                    user_message_hash=content_hash(message),
                    turn_index=context.turn_index,
                    declared_at_ns=115 if step_index == 1 else 145,
                )
                return JourneySimulatorDecision(
                    user_message=message,
                    persona=context.schedule.persona,
                    mission=context.schedule.mission,
                    rationale="Selected from the complete response.",
                    broker_request_id=broker_request_id,
                    simulator_attestation=attestation,
                    variant_binding=binding,
                )

            runner = self._runner(
                Path(tmpdir),
                schedule=schedule,
                simulator=simulator,
                transport=FakeTransport([
                    "Agent> Model config: provider=deepseek, model=deepseek-chat, auth=api_key\nFirst.",
                    "Agent> Second.",
                    "Agent> Terminal.",
                ]),
                events=[
                    self._event(1),
                    self._event(2, "response-driven retained turn 1"),
                    self._event(3, "response-driven retained turn 2"),
                ],
            )
            with patch(
                "tests.agent_live.dynamic_dual_ai_chaos.verify_runtime_postcondition",
                return_value=self._verified_edge(),
            ):
                result = self._run(runner)
            artifact = json.loads(result.evidence_path.read_text())
            attestations = [
                row["decision_provenance"]["variant_attestation"]
                for row in artifact["turns"]
            ]
            self.assertEqual(
                [item["source_step_id"] for item in attestations],
                ["source-1", "source-2"],
            )
            self.assertTrue(
                all(item["actor"]["actor_kind"] == "codex" for item in attestations)
            )

    def test_journey_transcript_is_redacted_at_the_persistence_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            secret = "abcdefghijklmnopqrstuvwxyz123456"
            transport = FakeTransport([
                "Agent> Model config: provider=deepseek, model=deepseek-chat, auth=api_key\nStart.",
                "Agent> Authorization: Bearer journey-secret-token. Done.",
            ])

            def simulator(context):
                return JourneySimulatorDecision(
                    user_message=f"https://rpc.example/{secret}",
                    persona=context.schedule.persona,
                    mission=context.schedule.mission,
                    rationale="Exercise the persisted redaction boundary.",
                )

            runner = self._runner(
                root,
                schedule=self._schedule(max_turns=1),
                simulator=simulator,
                transport=transport,
                events=[
                    self._event(1),
                    self._event(2, f"https://rpc.example/{secret}"),
                ],
                registry=self._registry(ready=ready_at_turn_two),
            )
            result = self._run(runner)

            transcript = result.transcript_path.read_text(encoding="utf-8")
            self.assertNotIn(secret, transcript)
            self.assertNotIn("journey-secret-token", transcript)
            self.assertIn("https://rpc.example/***REDACTED***", transcript)
            self.assertIn("Bearer ***REDACTED***", transcript)
            evidence = result.evidence_path.read_text(encoding="utf-8")
            journey_result = result.journey_result_path.read_text(encoding="utf-8")
            for persisted in (evidence, journey_result):
                self.assertNotIn(secret, persisted)
                self.assertNotIn("journey-secret-token", persisted)
            artifact = json.loads(evidence)
            self.assertTrue(artifact["content_redacted"])
            self.assertRegex(artifact["source_transcript_hash"], r"^[0-9a-f]{64}$")

    def test_registry_is_versioned_deterministic_and_rejects_lambda(self) -> None:
        first = self._registry()
        second = build_journey_outcome_verifier_registry(tuple(
            reversed(tuple(first.definitions.values()))
        ))
        self.assertEqual(first.registry_id, second.registry_id)
        with self.assertRaisesRegex(TypeError, "lambda"):
            build_journey_outcome_verifier_registry((
                JourneyPostconditionVerifierDefinition(
                    "ready_state", "ready-state", 1, "Invalid lambda.",
                    lambda context: JourneyPostconditionResult("ready_state", True),
                ),
            ))

    def test_tampered_or_revision_mismatched_evidence_cannot_qualify(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            schedule = self._schedule()
            registry = self._registry()
            runner = self._runner(
                Path(tmpdir), schedule=schedule, simulator=self._decision,
                transport=FakeTransport([
                    "Agent> Model config: provider=deepseek, model=deepseek-chat, auth=api_key\nDone."
                ]),
                events=[self._event(3)],
                registry=registry,
                startup_product_revision=3,
                startup_product_fingerprint="d" * 64,
            )
            result = self._run(runner)
            artifact = json.loads(result.evidence_path.read_text())
            artifact["completed_turn_count"] = 99
            with self.assertRaisesRegex(ValueError, "artifact hash"):
                validate_journey_evidence_artifact(
                    artifact, schedule=schedule, verifier_registry=registry, revision=REVISION
                )
            with self.assertRaisesRegex(ValueError, "revision"):
                validate_journey_evidence_artifact(
                    json.loads(result.evidence_path.read_text()),
                    schedule=schedule,
                    verifier_registry=registry,
                    revision={"commit": "other", "worktree_hash": "b" * 64},
                )

    def test_max_turns_and_forbidden_outcomes_are_product_failures(self) -> None:
        cases = (
            (ready_at_turn_ninety_nine, state_is_not_lost, "exhausted max_turns"),
            (ready_at_turn_three, state_lost_at_turn_two, "forbidden outcome"),
        )
        for index, (ready, lost, message) in enumerate(cases):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as tmpdir:
                runner = self._runner(
                    Path(tmpdir), schedule=self._schedule(max_turns=1), simulator=self._decision,
                    transport=FakeTransport([
                        "Agent> Model config: provider=deepseek, model=deepseek-chat, auth=api_key\nStart.",
                        "Agent> Next.",
                    ]),
                    events=[
                        self._event(1),
                        self._event(
                            2,
                            "Responding to turn 2: continue safely.",
                        ),
                    ],
                    registry=self._registry(ready, lost),
                )
                with patch(
                    "tests.agent_live.dynamic_dual_ai_chaos.verify_runtime_postcondition",
                    return_value=self._verified_edge(),
                ), self.assertRaisesRegex(JourneyProductFailure, message):
                    self._run(runner)
                payload = json.loads(
                    (Path(tmpdir) / ".agent/dynamic-chaos/journey-session/journey-result.json").read_text()
                )
                self.assertEqual(payload["terminal_classification"], "product_failed")
                self.assertFalse(payload["qualifying_evidence"])

    def test_invalid_simulator_and_external_block_are_distinct(self) -> None:
        decisions = (
            (lambda context: None, JourneyExternallyBlockedError, "externally_blocked"),
            (
                lambda context: JourneySimulatorDecision(
                    "continue", "wrong persona", context.schedule.mission, "reason"
                ),
                JourneySimulatorInvalidError,
                "simulator_invalid",
            ),
        )
        for index, (simulator, error_type, classification) in enumerate(decisions):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as tmpdir:
                runner = self._runner(
                    Path(tmpdir), schedule=self._schedule(), simulator=simulator,
                    transport=FakeTransport([
                        "Agent> Model config: provider=deepseek, model=deepseek-chat, auth=api_key\nStart."
                    ]),
                    events=[self._event(1)], registry=self._registry(ready_at_turn_ninety_nine),
                )
                with self.assertRaises(error_type):
                    self._run(runner)
                payload = json.loads(
                    (Path(tmpdir) / ".agent/dynamic-chaos/journey-session/journey-result.json").read_text()
                )
                self.assertEqual(payload["terminal_classification"], classification)

    def test_bad_verifier_contract_is_infrastructure_not_product_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = self._runner(
                Path(tmpdir), schedule=self._schedule(), simulator=self._decision,
                transport=FakeTransport([
                    "Agent> Model config: provider=deepseek, model=deepseek-chat, auth=api_key\nStart."
                ]),
                events=[self._event(1)],
                registry=self._registry(invalid_verifier_result),
            )
            with self.assertRaisesRegex(
                JourneyInfrastructureInterruptedError, "JourneyPostconditionResult"
            ):
                self._run(runner)


if __name__ == "__main__":
    unittest.main()

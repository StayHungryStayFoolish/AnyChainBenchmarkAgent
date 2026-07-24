"""Focused contracts for the response-driven open-journey runtime."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping
from unittest.mock import patch

from tests.agent_live.chaos_scheduler import build_journey_schedule
from tests.agent_live.coverage_evidence import RuntimeTurnEvent, VerifiedPostcondition
from tests.agent_live.dynamic_dual_ai_chaos import (
    ChaosRunConfig,
    DynamicDualAiJourneyRunner,
    JourneyExternallyBlockedError,
    JourneyInfrastructureInterruptedError,
    JourneyPostconditionResult,
    JourneyPostconditionVerifierDefinition,
    JourneyProductFailure,
    JourneySimulatorContext,
    JourneySimulatorDecision,
    JourneySimulatorInvalidError,
    JourneyTerminalClassification,
    build_journey_outcome_verifier_registry,
    validate_journey_evidence_artifact,
)
from tests.agent_live.generate_harness_coverage_ledger import contract_variant_hash


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


def ready_at_turn_three(context):
    return JourneyPostconditionResult(
        "ready_state",
        context.current_event.turn_index >= 3,
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


class FakeEventStream:
    def __init__(self, events: list[RuntimeTurnEvent]) -> None:
        self.events = list(events)

    def baseline(self) -> RuntimeTurnEvent:
        return self.events.pop(0)

    def next_event(self, *, timeout_seconds: float) -> RuntimeTurnEvent:
        del timeout_seconds
        if not self.events:
            raise RuntimeError("missing committed event")
        return self.events.pop(0)


class OrderedClock:
    def __init__(self) -> None:
        self.value = 100

    def __call__(self) -> int:
        self.value += 10
        return self.value


class DynamicDualAiJourneyRunnerTest(unittest.TestCase):
    def _event(self, turn_index: int) -> RuntimeTurnEvent:
        return RuntimeTurnEvent(
            schema_version=2,
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
            admitted_action_types=("answer_pending",) if turn_index > 1 else (),
            state_diff_hashes=(
                {"target_mode": {"before": "", "after": "f" * 64}}
                if turn_index > 1 else {}
            ),
            after_value_hashes={},
            next_result={"kind": "question", "question_id": "opening_next_action"},
        )

    def _schedule(self, *, max_turns: int = 3):
        return build_journey_schedule(
            revision=REVISION,
            seed=271,
            journey={
                "journey_id": "response-driven-journey",
                "start_scenario": "opening",
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
            },
        )

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

    def _runner(self, root, *, schedule, simulator, transport, events, registry=None):
        return DynamicDualAiJourneyRunner(
            ChaosRunConfig.linux(root, session_id="journey-session"),
            simulator,
            ledger={"revision": REVISION, "edges": [dict(EDGE)]},
            schedule=schedule,
            postcondition_verifier_registry=registry or self._registry(),
            transport=transport,
            event_stream=FakeEventStream(events),
            revision=REVISION,
            clock_ns=OrderedClock(),
        )

    def _run(self, runner):
        scenario = SimpleNamespace(
            scenario_id="opening", state_fingerprint="scenario-fingerprint", seed_state={}
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

    def test_response_driven_turns_record_edges_and_qualifying_evidence(self) -> None:
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
                events=[self._event(1), self._event(2), self._event(3)], registry=registry,
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
            self.assertTrue(artifact["qualifying_evidence"])
            validate_journey_evidence_artifact(
                artifact, schedule=schedule, verifier_registry=registry, revision=REVISION
            )
            self.assertNotIn("target_coverage_ids", artifact["turns"][0]["decision"])
            provenance = artifact["turns"][0]["decision_provenance"]
            self.assertRegex(provenance["previous_response_hash"], r"^[0-9a-f]{64}$")
            self.assertRegex(provenance["user_message_hash"], r"^[0-9a-f]{64}$")
            self.assertEqual(
                provenance["submitted_at_ns"],
                artifact["turns"][0]["turn_identity"]["user_message_submitted_at_ns"],
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
                events=[self._event(1), self._event(3)],
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
                events=[self._event(3)], registry=registry,
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
                    events=[self._event(1), self._event(2)],
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

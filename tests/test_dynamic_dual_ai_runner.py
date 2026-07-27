"""Contract tests for the response-driven PTY chaos runner (no real API)."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping
from unittest.mock import patch

from agent.harness.input_identity import user_input_hash
from agent.harness.terminal_protocol import (
    TerminalDetourProjection,
    TerminalSessionEvent,
    build_terminal_session_event,
    presentation_hash,
    validate_terminal_detour_projection,
    validate_terminal_outcome_projection,
)
from agent.terminal.language import t
from tests.agent_live.chaos_scheduler import ScheduledCoverageTarget, build_chaos_schedule
from tests.agent_live.coverage_evidence import (
    RuntimeTurnEvent,
    VerifiedPostcondition,
    content_hash,
    load_valid_evidence_reference,
    validate_pty_diagnostic_artifact,
    validate_pty_cli_evidence_artifact,
    verify_runtime_postcondition,
)
from tests.agent_live.dynamic_dual_ai_chaos import (
    ChaosRunConfig,
    CompletionExpectation,
    DynamicDualAiChaosRunner,
    ContainerPtyBridgeTransport,
    NoRuntimeEventBeforeFence,
    SimulatorContext,
    SimulatorDecision,
    SimulatorDecisionInvalid,
    SessionTerminationCompletion,
    SubprocessPtyTransport,
    TerminalPresentationViolation,
    TerminalDetourCompletion,
    TerminalProtocolViolation,
    TerminalOutcomeObservation,
    TerminalTurnFailure,
    _complete_agent_response,
    _validate_decision,
    _verify_declared_postconditions,
    encode_bracketed_paste,
    transport_for_config,
    validate_startup_session_event,
    validate_startup_terminal_protocol,
    wait_for_turn_completion,
)
from tests.agent_live.generate_harness_coverage_ledger import contract_variant_hash


class SimulatorInputClassAdmissionTest(unittest.TestCase):
    def _decision(self, message: str) -> SimulatorDecision:
        return SimulatorDecision(
            user_message=message,
            persona="operator",
            goal="exercise the scheduled input class",
            rationale="selected after reading the Agent response",
            target_coverage_ids=("edge-1",),
        )

    def _target(self) -> ScheduledCoverageTarget:
        return ScheduledCoverageTarget(
            target_id="target-1",
            edge_key="edge-1",
            persona="operator",
            goal="exercise the scheduled input class",
        )

    def test_multiline_prose_rejects_structured_assignment(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not exercise multiline_prose"):
            _validate_decision(
                self._decision("CLOUD_REGION=us-central1\nCLOUD_ZONE=us-central1-a"),
                self._target(),
                {"input_class": "multiline_prose"},
            )

    def test_multiline_prose_accepts_multiple_prose_lines(self) -> None:
        _validate_decision(
            self._decision("The region should stay unchanged.\nPlease explain what the next step needs."),
            self._target(),
            {"input_class": "multiline_prose"},
        )

    def test_multiline_prose_accepts_a_standalone_url_line(self) -> None:
        _validate_decision(
            self._decision(
                "Use this endpoint only for schema validation:\nhttp://fake-node:19000"
            ),
            self._target(),
            {"input_class": "multiline_prose"},
        )

    def test_input_class_violation_has_a_typed_simulator_error(self) -> None:
        with self.assertRaises(SimulatorDecisionInvalid):
            _validate_decision(
                self._decision("NETWORK_INTERFACE=eth0\nUse the primary interface."),
                self._target(),
                {"input_class": "multiline_prose"},
            )

    def test_structured_lane_requires_structured_region(self) -> None:
        with self.assertRaisesRegex(ValueError, "structured_json_yaml_env_curl"):
            _validate_decision(
                self._decision("Please configure the region for me."),
                self._target(),
                {"input_class": "structured_json_yaml_env_curl"},
            )

        _validate_decision(
            self._decision("CLOUD_REGION=us-central1\nCLOUD_ZONE=us-central1-a"),
            self._target(),
            {"input_class": "structured_json_yaml_env_curl"},
        )


class DeclaredTargetSetVerificationTest(unittest.TestCase):
    def test_every_declared_sibling_target_must_pass(self) -> None:
        passed = VerifiedPostcondition(
            verifier_id="fixture",
            passed=True,
            observed_coverage_ids=("edge-a",),
            admitted_typed_actions=("answer_pending",),
            state_diff={"changed": True},
            next_question_or_result={"pending": "next"},
            details={"errors": []},
        )
        failed = VerifiedPostcondition(
            verifier_id="fixture",
            passed=False,
            observed_coverage_ids=(),
            admitted_typed_actions=(),
            state_diff={"changed": True},
            next_question_or_result={"pending": "next"},
            details={"errors": ["sibling demand was not preserved"]},
        )
        with patch(
            "tests.agent_live.dynamic_dual_ai_chaos.verify_runtime_postcondition",
            side_effect=[passed, failed],
        ) as verifier:
            result = _verify_declared_postconditions(
                SimpleNamespace(),
                SimpleNamespace(),
                SimpleNamespace(),
                edge_index={"edge-a": {}, "edge-b": {}},
                target_coverage_ids=("edge-a", "edge-b"),
            )

        self.assertFalse(result.passed)
        self.assertEqual(verifier.call_count, 2)
        self.assertIn("edge-b: sibling demand was not preserved", result.details["errors"])

    def test_unknown_declared_sibling_target_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown coverage ids: edge-missing"):
            _verify_declared_postconditions(
                SimpleNamespace(),
                SimpleNamespace(),
                SimpleNamespace(),
                edge_index={"edge-a": {}},
                target_coverage_ids=("edge-a", "edge-missing"),
            )


EDGE = {
    "edge_key": "opening::opening_next_action::fake_node::option",
    "contract_hash": contract_variant_hash({"id": "opening_next_action"}),
    "contract_variant_hash": "variant-hash",
    "question_id": "opening_next_action",
    "applicable": True,
    "real_execution_required": False,
    "executable_scenario_ids": ["opening"],
    "evidence": {
        "dynamic_dual_ai": {
            "required": True,
            "applicability_reason": "response-driven semantic test fixture",
        },
        "real_cli": {
            "required": True,
            "applicability_reason": "same PTY turn proves terminal transport",
        },
    },
}
REVISION = {"commit": "test-commit", "worktree_hash": "a" * 64}


def _independent_workflow_outcome(
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
        product_authority_id="dynamic-dual-ai-chaos:contract-session",
        logical_thread_id="dynamic-dual-ai-chaos:contract-session",
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
    return _seal_terminal_outcome(outcome)


def _seal_terminal_outcome(
    outcome: TerminalOutcomeObservation,
) -> TerminalOutcomeObservation:
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


class FakeTransport:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.started_env: dict[str, str] = {}
        self.submitted: list[str] = []
        self.closed = False

    def start(self, *, env: Mapping[str, str]) -> None:
        self.started_env = dict(env)

    def read_complete_agent_response(self, *, timeout_seconds: float) -> str:
        del timeout_seconds
        if not self.responses:
            raise AssertionError("runner read more responses than scheduled")
        return self.responses.pop(0)

    def submit_bracketed_paste(self, message: str) -> None:
        self.submitted.append(message)

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
            raise NoRuntimeEventBeforeFence(
                "CLI returned without a new committed runtime turn event"
            )
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
        outcomes: list[TerminalOutcomeObservation] | None = None,
        publish_session_event: bool = True,
        startup_product_revision: int = 1,
        startup_outcome_count: int | None = None,
        responses: list[str] | None = None,
    ) -> None:
        response_frames = list(responses or ())
        self.outcomes = list(outcomes) if outcomes is not None else [
            _independent_workflow_outcome(
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
        self._publish_session_event = publish_session_event
        self._startup_product_revision = startup_product_revision
        self._startup_outcome_count = (
            startup_product_revision
            if startup_outcome_count is None
            else startup_outcome_count
        )

    def baseline_session_events(
        self,
        *,
        startup_response: str,
        session_id: str,
        session_purpose: str,
    ) -> tuple[TerminalSessionEvent, ...]:
        if not self._publish_session_event:
            return ()
        return (
            _ready_session_event(
                REVISION,
                startup_response=startup_response,
                session_id=session_id,
                session_purpose=session_purpose,
                product_revision=self._startup_product_revision,
            ),
        )

    def mark_process_start(self) -> None:
        return None

    def capture_startup_snapshot(self):
        self._outcome_cursor = self._startup_outcome_count
        return (), ()

    def mark_baseline(self) -> None:
        self._outcome_cursor = self._startup_outcome_count

    def next_outcome(
        self,
        *,
        timeout_seconds: float,
    ) -> TerminalOutcomeObservation:
        del timeout_seconds
        if self._outcome_cursor >= len(self.outcomes):
            raise RuntimeError("CLI returned without a typed terminal outcome")
        outcome = self.outcomes[self._outcome_cursor]
        self._outcome_cursor += 1
        return outcome

    def next_completion(
        self,
        *,
        timeout_seconds: float,
    ):
        return self.next_outcome(timeout_seconds=timeout_seconds)


def _ready_session_event(
    revision: Mapping[str, str],
    *,
    startup_response: str,
    session_id: str,
    session_purpose: str,
    product_revision: int,
    product_fingerprint: str | None = None,
    product_checkpoint_thread_id: str | None = None,
    product_checkpoint_id: str | None = None,
    runtime_event_fence_terminal_event_id: str | None = None,
    runtime_event_fence_id: str | None = None,
) -> TerminalSessionEvent:
    transaction_id = (
        f"00000000-0000-0000-0000-{product_revision:012d}"
    )
    physical_thread_id = f"attempt:{transaction_id}"
    return build_terminal_session_event(
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
        product_revision=product_revision,
        product_checkpoint_thread_id=(
            product_checkpoint_thread_id or physical_thread_id
        ),
        product_checkpoint_id=(
            product_checkpoint_id or f"checkpoint-{product_revision}"
        ),
        product_fingerprint=(
            product_fingerprint
            or chr(96 + product_revision + 1) * 64
        ),
        runtime_event_fence_sequence=product_revision,
        runtime_event_fence_terminal_event_id=(
            runtime_event_fence_terminal_event_id
            or f"terminal-{product_revision}"
        ),
        runtime_event_fence_id=(
            runtime_event_fence_id or f"runtime-{product_revision}"
        ),
        runtime_event_fence_hash="9" * 64,
        rendered_frame=startup_response,
        origin_revision=revision,
    )


def _terminal_outcome(
    outcome: str,
    *,
    failure_category: str = "",
    product_fingerprint: str = "c" * 64,
    product_authority: str = "chaos:test-session",
) -> TerminalOutcomeObservation:
    return _seal_terminal_outcome(TerminalOutcomeObservation(
        schema_version=3,
        record_type="terminal_outcome_projection",
        projection_id=f"projection-{outcome}-{failure_category or 'ok'}",
        event_id=f"terminal-{outcome}-{failure_category or 'ok'}",
        transaction_id="00000000-0000-0000-0000-000000000099",
        product_authority_id=product_authority,
        logical_thread_id=product_authority,
        physical_thread_id="attempt:00000000-0000-0000-0000-000000000099",
        outcome=outcome,
        failure_category=failure_category,
        diagnostic_hash="d" * 64 if outcome != "committed" else "",
        base_revision=2,
        base_checkpoint_thread_id="head",
        base_checkpoint_id="checkpoint-2",
        base_fingerprint=product_fingerprint,
        attempt_checkpoint_id=(
            "checkpoint-3" if outcome == "committed" else ""
        ),
        attempt_fingerprint=product_fingerprint if outcome == "committed" else "",
        product_revision=3 if outcome == "committed" else 2,
        product_checkpoint_thread_id=(
            "attempt:00000000-0000-0000-0000-000000000099"
            if outcome == "committed"
            else "head"
        ),
        product_checkpoint_id=(
            "checkpoint-3" if outcome == "committed" else "checkpoint-2"
        ),
        product_fingerprint=product_fingerprint,
        render_hash="f" * 64 if outcome == "committed" else "",
        runtime_event_id=(
            "runtime-terminal-committed-ok"
            if outcome == "committed"
            else ""
        ),
        runtime_event_sequence=3 if outcome == "committed" else 2,
        runtime_event_payload_hash="9" * 64 if outcome == "committed" else "",
        presentation_hash=presentation_hash("Agent> fixture response"),
        origin_revision=REVISION,
        delivery_phase="live",
        projected_at="2026-07-27T00:00:00+00:00",
        record_hash="0" * 64,
    ))


def _terminal_detour(response: str) -> TerminalDetourProjection:
    detour = TerminalDetourProjection(
        schema_version=3,
        record_type="terminal_detour_projection",
        projection_id="detour-projection-test",
        detour_id="00000000-0000-0000-0000-000000000088",
        product_authority_id="chaos:test-session",
        logical_thread_id="chaos:test-session",
        process_instance_id="process-1",
        session_id="test-session",
        session_purpose="chaos",
        command_name="help",
        input_hash=hashlib.sha256(b"help").hexdigest(),
        result_kind="command",
        interruption_kind="",
        termination_reason="",
        exit_code=None,
        effect_class="read_only",
        effect_status="not_applicable",
        shell_state_before_hash="4" * 64,
        shell_state_after_hash="4" * 64,
        product_revision_before=2,
        product_checkpoint_thread_id_before="head",
        product_checkpoint_id_before="checkpoint-2",
        product_fingerprint_before="c" * 64,
        product_revision_after=2,
        product_checkpoint_thread_id_after="head",
        product_checkpoint_id_after="checkpoint-2",
        product_fingerprint_after="c" * 64,
        runtime_event_fence_sequence=0,
        runtime_event_fence_terminal_event_id="",
        runtime_event_fence_id="",
        runtime_event_fence_hash="",
        response_hash="6" * 64,
        stream_chunk_count=0,
        stream_hash="9" * 64,
        stream_stop_reason="not_streaming",
        identity_trust="trusted",
        presentation_hash=presentation_hash(response),
        origin_revision=REVISION,
        delivery_phase="live",
        projected_at="2026-07-27T00:00:00+00:00",
        record_hash="0" * 64,
    )
    return _seal_terminal_detour(detour)


def _seal_terminal_detour(
    detour: TerminalDetourProjection,
) -> TerminalDetourProjection:
    payload = asdict(detour)
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
    return validate_terminal_detour_projection(payload)


def _completion_expectation(
    submitted_input: str = "test input",
    *,
    product_fingerprint: str = "c" * 64,
) -> CompletionExpectation:
    return CompletionExpectation(
        submitted_input=submitted_input,
        session_id="test-session",
        session_purpose="chaos",
        product_authority_id="chaos:test-session",
        process_instance_id="process-1",
        product_revision=2,
        product_checkpoint_thread_id="head",
        product_checkpoint_id="checkpoint-2",
        product_fingerprint=product_fingerprint,
    )


class OrderedClock:
    def __init__(self) -> None:
        self.value = 100

    def __call__(self) -> int:
        self.value += 10
        return self.value


class DynamicDualAiRunnerTest(unittest.TestCase):
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

    def test_config_rejects_extra_environment_identity_override(self) -> None:
        for key in ("LLM_PROVIDER", "LLM_MODEL", "AGENT_CONFIG_LOCAL"):
            with self.subTest(key=key):
                with self.assertRaisesRegex(
                    ValueError,
                    "cannot override frozen identity keys",
                ):
                    ChaosRunConfig(
                        repo_root=Path("/workspace"),
                        command=("./bin/anychain-agent",),
                        provider="deepseek",
                        model="deepseek-v4-pro",
                        extra_env={key: "override"},
                    )

    def test_config_freezes_extra_environment_snapshot(self) -> None:
        source = {"CUSTOM_CHAOS_VALUE": "before"}
        config = ChaosRunConfig(
            repo_root=Path("/workspace"),
            command=("./bin/anychain-agent",),
            provider="deepseek",
            model="deepseek-v4-pro",
            extra_env=source,
        )

        source["CUSTOM_CHAOS_VALUE"] = "after"
        source["LLM_MODEL"] = "another-model"
        environment = config.isolated_environment({
            "LLM_PROVIDER": "another-provider",
            "LLM_MODEL": "another-model",
            "AGENT_CONFIG_LOCAL": "/tmp/private.sh",
        })

        self.assertEqual(config.extra_env["CUSTOM_CHAOS_VALUE"], "before")
        self.assertEqual(environment["CUSTOM_CHAOS_VALUE"], "before")
        self.assertEqual(environment["LLM_PROVIDER"], "deepseek")
        self.assertEqual(environment["LLM_MODEL"], "deepseek-v4-pro")
        self.assertEqual(environment["AGENT_CONFIG_LOCAL"], "/dev/null")
        with self.assertRaises(TypeError):
            config.extra_env["CUSTOM_CHAOS_VALUE"] = "mutated"

    def test_complete_response_uses_the_newest_prompt_delimited_agent_frame(self) -> None:
        startup = "Agent> Ready.\nAgent> Choose a mode.\nUser> "
        self.assertEqual(
            _complete_agent_response(startup),
            "Agent> Ready.\nAgent> Choose a mode.",
        )

        redrawn = (
            "terminal capability warning\nUser> \n"
            "Agent> Mode selected.\nAgent> Which chain?\nUser> "
        )
        self.assertEqual(
            _complete_agent_response(redrawn),
            "Agent> Mode selected.\nAgent> Which chain?",
        )

    def test_prompt_redraw_without_agent_output_is_not_a_complete_response(self) -> None:
        self.assertIsNone(_complete_agent_response("terminal capability warning\nUser> "))
        self.assertIsNone(
            _complete_agent_response(
                "terminal capability warning\nUser> \ninput echo only\nUser> "
            )
        )

    def _ledger(self) -> dict:
        return {"revision": REVISION, "edges": [dict(EDGE)]}

    def _schedule(self, ledger: Mapping[str, object]) -> object:
        return build_chaos_schedule(
            ledger,
            revision=REVISION,
            seed=17,
            targets=[{
                "target_id": "mode-turn",
                "edge_key": EDGE["edge_key"],
                "persona": "impatient operator",
                "goal": "exercise response-dependent navigation",
                "tuple_ids": ["language=en|workflow_mode=fake"],
            }],
        )

    def _event(
        self,
        turn_index: int,
        before: str,
        after: str,
        pending: str,
        submitted_input: str = "<fixture-input-not-specified>",
    ) -> RuntimeTurnEvent:
        transaction_id = f"00000000-0000-0000-0000-{turn_index:012d}"
        physical_thread_id = f"attempt:{transaction_id}"
        return RuntimeTurnEvent(
            schema_version=6,
            event_type="turn_committed",
            thread_id="contract-session",
            session_purpose="dynamic-dual-ai-chaos",
            before_fingerprint=before,
            after_fingerprint=after,
            turn_index=turn_index,
            active_group="opening",
            pending_question_id=pending,
            action_queue_types=(),
            pending_contract=(
                {
                    "id": pending,
                    "options": [
                        {"id": "1", "value": "continue"},
                        {"id": "2", "value": "modify"},
                        {"id": "3", "value": "reset"},
                    ],
                }
                if pending == "resume_harness_session"
                else ({"id": pending} if pending else {})
            ),
            revision=REVISION,
            observation="turn_committed",
            admitted_action_types=("choose_target_mode",) if turn_index > 1 else (),
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
            next_result={"kind": "question", "question_id": pending},
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
            product_authority_id="dynamic-dual-ai-chaos:contract-session",
            physical_thread_id=physical_thread_id,
            attempt_checkpoint_id=f"checkpoint-{turn_index}",
        )

    def test_runner_requires_independent_runtime_and_terminal_producers(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ledger = self._ledger()
            runtime_stream = FakeRuntimeEventStream([
                self._event(
                    1,
                    "a" * 64,
                    "b" * 64,
                    "opening_next_action",
                )
            ])
            with self.assertRaisesRegex(
                ValueError,
                "independent terminal outcome stream",
            ):
                DynamicDualAiChaosRunner(
                    ChaosRunConfig.linux(
                        root,
                        session_id="contract-session",
                    ),
                    lambda context: None,
                    ledger=ledger,
                    schedule=self._schedule(ledger),
                    event_stream=runtime_stream,
                    revision=REVISION,
                )
            with self.assertRaisesRegex(
                ValueError,
                "independent producers",
            ):
                DynamicDualAiChaosRunner(
                    ChaosRunConfig.linux(
                        root,
                        session_id="contract-session",
                    ),
                    lambda context: None,
                    ledger=ledger,
                    schedule=self._schedule(ledger),
                    event_stream=runtime_stream,
                    terminal_outcome_stream=runtime_stream,
                    revision=REVISION,
                )

    def test_each_turn_is_selected_from_schedule_after_full_previous_response(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ledger = self._ledger()
            transport = FakeTransport([
                "Agent> Model config: provider=deepseek, model=deepseek-chat, auth=api_key\n"
                "Agent> Authorization: Bearer abcdefghijklmnopqrstuvwxyz123456\n"
                "Agent> Choose a mode.\n1. fake-node\n2. real-node",
                "Agent> Endpoint https://rpc.example/abcdefghijklmnopqrstuvwxyz123456 accepted.\n"
                "Agent> Which chain do you want to test?",
            ])
            seen: list[SimulatorContext] = []

            def simulator(context: SimulatorContext) -> SimulatorDecision:
                seen.append(context)
                self.assertIn("Choose a mode", context.previous_agent_response)
                self.assertEqual(context.scheduled_target.edge_key, EDGE["edge_key"])
                return SimulatorDecision(
                    user_message="I only want a safe dry run first.",
                    persona=context.scheduled_target.persona,
                    goal=context.scheduled_target.goal,
                    rationale=(
                        "The response offered fake-node; API_KEY="
                        "abcdefghijklmnopqrstuvwxyz123456"
                    ),
                    target_coverage_ids=(context.scheduled_target.edge_key,),
                )

            runner = DynamicDualAiChaosRunner(
                ChaosRunConfig.linux(root, session_id="contract-session"),
                simulator,
                ledger=ledger,
                schedule=self._schedule(ledger),
                transport=transport,
                event_stream=FakeRuntimeEventStream([
                    self._event(1, "a" * 64, "b" * 64, "opening_next_action"),
                    self._event(
                        2,
                        "b" * 64,
                        "c" * 64,
                        "opening_next_action",
                        "I only want a safe dry run first.",
                    ),
                ]),
                terminal_outcome_stream=FakeTerminalOutcomeStream(
                    turn_count=2,
                    responses=list(transport.responses),
                ),
                revision=REVISION,
                clock_ns=OrderedClock(),
            )

            result = runner.run()

            self.assertEqual(result.execution_status, "complete")
            self.assertEqual(result.diagnostic_paths, ())
            self.assertEqual(transport.submitted, ["I only want a safe dry run first."])
            self.assertEqual(len(seen), 1)
            self.assertTrue(transport.closed)
            schedule_result = json.loads(result.schedule_result_path.read_text(encoding="utf-8"))
            self.assertEqual(schedule_result["execution_status"], "complete")
            self.assertEqual(schedule_result["passed_target_count"], 1)
            artifacts = {
                json.loads(path.read_text(encoding="utf-8"))["evidence_class"]: json.loads(
                    path.read_text(encoding="utf-8")
                )
                for path in result.evidence_paths
            }
            self.assertEqual(set(artifacts), {"dynamic_dual_ai", "real_cli"})
            for artifact in artifacts.values():
                valid, reason = validate_pty_cli_evidence_artifact(
                    artifact,
                    edge=EDGE,
                    revision=REVISION,
                )
                self.assertTrue(valid, reason)
            artifact = artifacts["dynamic_dual_ai"]
            persisted = json.dumps(artifacts, ensure_ascii=False)
            self.assertNotIn("abcdefghijklmnopqrstuvwxyz123456", persisted)
            self.assertIn("***REDACTED***", persisted)
            self.assertTrue(artifact["content_redacted"])
            self.assertEqual(artifact["turn_observation"]["seed"], 17)
            self.assertEqual(
                artifact["turn_observation"]["verified_postcondition"]["verifier_id"],
                "dynamic-declared-target-set-v1",
            )
            self.assertEqual(artifacts["real_cli"]["turn_observation"]["simulator_decision"], {})
            lane_evidence = schedule_result["targets"][0]["lane_evidence"]
            self.assertEqual(set(lane_evidence), {"dynamic_dual_ai", "real_cli"})
            self.assertEqual(
                list((root / ".agent/dynamic-chaos/contract-session/diagnostics").glob("*.json")),
                [],
            )

    def test_complete_response_precedes_short_runtime_event_grace(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ledger = self._ledger()
            order: list[str] = []

            class OrderedTransport(FakeTransport):
                def read_complete_agent_response(inner_self, *, timeout_seconds: float) -> str:
                    order.append("read")
                    return super(OrderedTransport, inner_self).read_complete_agent_response(
                        timeout_seconds=timeout_seconds
                    )

                def submit_bracketed_paste(inner_self, message: str) -> None:
                    order.append("submit")
                    super(OrderedTransport, inner_self).submit_bracketed_paste(message)

            class OrderedEvents(FakeRuntimeEventStream):
                def baseline(inner_self) -> RuntimeTurnEvent:
                    order.append("baseline")
                    return super(OrderedEvents, inner_self).baseline()

                def next_event(inner_self, *, timeout_seconds: float) -> RuntimeTurnEvent:
                    order.append("event")
                    return super(OrderedEvents, inner_self).next_event(
                        timeout_seconds=timeout_seconds
                    )

            transport = OrderedTransport([
                "Agent> Model config: provider=deepseek, model=deepseek-chat, auth=api_key\n"
                "Agent> Choose a mode.",
                "Agent> Which chain?",
            ])
            events = OrderedEvents([
                self._event(1, "a" * 64, "b" * 64, "opening_next_action"),
                self._event(
                    2,
                    "b" * 64,
                    "c" * 64,
                    "opening_next_action",
                    "Use fake-node.",
                ),
            ])
            runner = DynamicDualAiChaosRunner(
                ChaosRunConfig.linux(root, session_id="contract-session"),
                lambda context: SimulatorDecision(
                    user_message="Use fake-node.",
                    persona=context.scheduled_target.persona,
                    goal=context.scheduled_target.goal,
                    rationale="Selected from the complete response.",
                    target_coverage_ids=(context.scheduled_target.edge_key,),
                ),
                ledger=ledger,
                schedule=self._schedule(ledger),
                transport=transport,
                event_stream=events,
                terminal_outcome_stream=FakeTerminalOutcomeStream(
                    turn_count=2,
                    responses=list(transport.responses),
                ),
                revision=REVISION,
            )

            runner.run()

            self.assertEqual(order, ["read", "baseline", "submit", "read", "event"])

    def test_typed_terminal_failures_do_not_wait_for_committed_event(self) -> None:
        cases = {
            "cancelled": ("aborted", "cancelled"),
            "timeout": ("aborted", "timeout"),
            "provider_failure": ("aborted", "provider_failure"),
            "runtime_failure": ("aborted", "unexpected_failure"),
            "reconciliation_required": (
                "reconciliation_required",
                "external_effect_uncertain",
            ),
        }
        for outcome, (terminal_status, failure_category) in cases.items():
            with self.subTest(outcome=outcome):
                response = "Agent> localized presentation may change"
                runtime_stream = FakeRuntimeEventStream([])
                terminal_stream = FakeTerminalOutcomeStream(
                    turn_count=0,
                    outcomes=[_terminal_outcome(
                        terminal_status,
                        failure_category=failure_category,
                    )],
                )
                with self.assertRaises(TerminalTurnFailure) as raised:
                    wait_for_turn_completion(
                        FakeTransport([response]),
                        runtime_stream,
                        terminal_stream,
                        timeout_seconds=1,
                        expectation=_completion_expectation(),
                        event_grace_seconds=0,
                    )
                expected = (
                    "reconciliation_required"
                    if terminal_status == "reconciliation_required"
                    else failure_category
                )
                self.assertEqual(raised.exception.outcome, expected)
                self.assertEqual(raised.exception.response, response)

    def test_v6_completion_joins_complete_product_head_and_presented_frame(self) -> None:
        response = "Agent> 任意语言的完整结果"
        outcome = _seal_terminal_outcome(replace(
            _terminal_outcome(
                "committed",
                product_fingerprint="c" * 64,
            ),
            presentation_hash=presentation_hash(response),
        ))
        event = replace(
            self._event(2, "c" * 64, "c" * 64, "opening_next_action"),
            schema_version=6,
            observation="turn_committed",
            thread_id="test-session",
            session_purpose="chaos",
            product_authority_id="chaos:test-session",
            base_revision=2,
            base_checkpoint_thread_id="head",
            base_checkpoint_id="checkpoint-2",
            product_revision=3,
            product_checkpoint_thread_id=outcome.product_checkpoint_thread_id,
            product_checkpoint_id=outcome.product_checkpoint_id,
            physical_thread_id=outcome.physical_thread_id,
            attempt_checkpoint_id=str(outcome.attempt_checkpoint_id or ""),
            runtime_event_id=outcome.runtime_event_id,
            runtime_event_sequence=outcome.runtime_event_sequence,
            runtime_event_payload_hash=outcome.runtime_event_payload_hash,
            terminal_event_id=outcome.event_id,
            transaction_id=outcome.transaction_id,
            terminal_outcome="committed",
            render_hash=outcome.render_hash,
            pending_transition={
                "before_hash": "7" * 64,
                "after_hash": "8" * 64,
            },
            render_manifest={"fragment_hashes": []},
            turn_receipt_summary={
                "turn_id": "test:2",
                "input_hash": user_input_hash("test input"),
                "admitted_action_ids": [],
                "execution_order": [],
            },
        )
        runtime_stream = FakeRuntimeEventStream([event])
        terminal_stream = FakeTerminalOutcomeStream(
            turn_count=0,
            outcomes=[outcome],
        )

        completion = wait_for_turn_completion(
            FakeTransport([response]),
            runtime_stream,
            terminal_stream,
            timeout_seconds=1,
            expectation=_completion_expectation(),
            event_grace_seconds=0,
        )

        self.assertEqual(completion.event.transaction_id, outcome.transaction_id)
        self.assertEqual(completion.response, response)

    def test_detour_completion_requires_no_runtime_event_and_keeps_head(self) -> None:
        response = "Agent> terminal help"
        detour = _terminal_detour(response)
        runtime_stream = FakeRuntimeEventStream([])
        terminal_stream = FakeTerminalOutcomeStream(
            turn_count=0,
            outcomes=[detour],
        )

        completion = wait_for_turn_completion(
            FakeTransport([response]),
            runtime_stream,
            terminal_stream,
            timeout_seconds=1,
            expectation=_completion_expectation("help"),
            event_grace_seconds=0,
        )

        self.assertIsInstance(completion, TerminalDetourCompletion)
        self.assertEqual(completion.terminal_detour, detour)

    def test_session_termination_is_not_a_command_or_workflow_completion(
        self,
    ) -> None:
        response = "Agent> bye"
        detour = _seal_terminal_detour(replace(
            _terminal_detour(response),
            command_name="exit",
            input_hash=hashlib.sha256(b"exit").hexdigest(),
            result_kind="session_termination",
            termination_reason="exit",
            exit_code=0,
        ))
        runtime_stream = FakeRuntimeEventStream([])
        terminal_stream = FakeTerminalOutcomeStream(
            turn_count=0,
            outcomes=[detour],
        )

        completion = wait_for_turn_completion(
            FakeTransport([response]),
            runtime_stream,
            terminal_stream,
            timeout_seconds=1,
            expectation=_completion_expectation("exit"),
            event_grace_seconds=0,
        )

        self.assertIsInstance(completion, SessionTerminationCompletion)
        self.assertEqual(completion.termination_reason, "exit")
        self.assertEqual(completion.exit_code, 0)

    def test_v6_completion_rejects_cross_transaction_composition(self) -> None:
        response = "Agent> complete"
        outcome = _seal_terminal_outcome(replace(
            _terminal_outcome(
                "committed",
                product_fingerprint="c" * 64,
            ),
            presentation_hash=presentation_hash(response),
        ))
        event = replace(
            self._event(2, "c" * 64, "c" * 64, "opening_next_action"),
            schema_version=6,
            observation="turn_committed",
            thread_id="test-session",
            session_purpose="chaos",
            product_authority_id="chaos:test-session",
            base_revision=2,
            base_checkpoint_thread_id="head",
            base_checkpoint_id="checkpoint-2",
            product_revision=3,
            product_checkpoint_thread_id=outcome.product_checkpoint_thread_id,
            product_checkpoint_id=outcome.product_checkpoint_id,
            physical_thread_id=outcome.physical_thread_id,
            attempt_checkpoint_id=str(outcome.attempt_checkpoint_id or ""),
            runtime_event_id=outcome.runtime_event_id,
            runtime_event_sequence=outcome.runtime_event_sequence,
            runtime_event_payload_hash=outcome.runtime_event_payload_hash,
            terminal_event_id=outcome.event_id,
            transaction_id="00000000-0000-0000-0000-000000000777",
            terminal_outcome="committed",
            render_hash=outcome.render_hash,
            pending_transition={
                "before_hash": "7" * 64,
                "after_hash": "8" * 64,
            },
            render_manifest={"fragment_hashes": []},
        )
        runtime_stream = FakeRuntimeEventStream([event])
        terminal_stream = FakeTerminalOutcomeStream(
            turn_count=0,
            outcomes=[outcome],
        )

        with self.assertRaisesRegex(
            TerminalProtocolViolation,
            "different transactions",
        ):
            wait_for_turn_completion(
                FakeTransport([response]),
                runtime_stream,
                terminal_stream,
                timeout_seconds=1,
                expectation=_completion_expectation(),
                event_grace_seconds=0,
            )

    def test_startup_audit_accepts_live_and_replay_for_one_committed_event(
        self,
    ) -> None:
        outcome = _terminal_outcome(
            "committed",
            product_fingerprint="c" * 64,
            product_authority="dynamic-dual-ai-chaos:test-session",
        )
        event = replace(
            self._event(2, "c" * 64, "c" * 64, "opening_next_action"),
            schema_version=6,
            observation="turn_committed",
            thread_id="test-session",
            session_purpose="dynamic-dual-ai-chaos",
            product_authority_id="dynamic-dual-ai-chaos:test-session",
            base_revision=outcome.base_revision,
            base_checkpoint_thread_id=outcome.base_checkpoint_thread_id,
            base_checkpoint_id=outcome.base_checkpoint_id,
            product_revision=outcome.product_revision,
            product_checkpoint_thread_id=(
                outcome.product_checkpoint_thread_id
            ),
            product_checkpoint_id=outcome.product_checkpoint_id,
            physical_thread_id=outcome.physical_thread_id,
            attempt_checkpoint_id=str(outcome.attempt_checkpoint_id or ""),
            runtime_event_id=outcome.runtime_event_id,
            runtime_event_sequence=outcome.runtime_event_sequence,
            runtime_event_payload_hash=outcome.runtime_event_payload_hash,
            terminal_event_id=outcome.event_id,
            transaction_id=outcome.transaction_id,
            terminal_outcome="committed",
            render_hash=outcome.render_hash,
            pending_transition={
                "before_hash": "7" * 64,
                "after_hash": "8" * 64,
            },
            render_manifest={"fragment_hashes": []},
        )
        replay = replace(
            outcome,
            projection_id="projection-replay",
            delivery_phase="startup_replay",
            presentation_hash="9" * 64,
        )
        session_event = replace(
            _ready_session_event(
                REVISION,
                startup_response="Agent> startup",
                session_id="test-session",
                session_purpose="dynamic-dual-ai-chaos",
                product_revision=3,
                product_fingerprint="c" * 64,
                product_checkpoint_thread_id=(
                    "attempt:00000000-0000-0000-0000-000000000099"
                ),
                product_checkpoint_id="checkpoint-3",
                runtime_event_fence_terminal_event_id=(
                    "terminal-committed-ok"
                ),
                runtime_event_fence_id="runtime-terminal-committed-ok",
            ),
            replayed_projection_ids=(replay.projection_id,),
        )

        validate_startup_terminal_protocol(
            [event],
            [outcome, replay],
            expected_revision=REVISION,
            session_event=session_event,
            baseline_event=event,
        )

    def test_startup_audit_rejects_conflict_missing_event_and_head_advance(
        self,
    ) -> None:
        committed = _terminal_outcome(
            "committed",
            product_fingerprint="c" * 64,
            product_authority="dynamic-dual-ai-chaos:test-session",
        )
        conflicting = replace(
            committed,
            projection_id="projection-conflict",
            product_fingerprint="d" * 64,
        )
        matching_event = replace(
            self._event(2, "c" * 64, "c" * 64, "opening_next_action"),
            schema_version=6,
            observation="turn_committed",
            thread_id="test-session",
            session_purpose="dynamic-dual-ai-chaos",
            product_authority_id="dynamic-dual-ai-chaos:test-session",
            base_revision=committed.base_revision,
            base_checkpoint_thread_id=committed.base_checkpoint_thread_id,
            base_checkpoint_id=committed.base_checkpoint_id,
            product_revision=committed.product_revision,
            product_checkpoint_thread_id=(
                committed.product_checkpoint_thread_id
            ),
            product_checkpoint_id=committed.product_checkpoint_id,
            physical_thread_id=committed.physical_thread_id,
            attempt_checkpoint_id=str(
                committed.attempt_checkpoint_id or ""
            ),
            runtime_event_id=committed.runtime_event_id,
            runtime_event_sequence=committed.runtime_event_sequence,
            runtime_event_payload_hash=(
                committed.runtime_event_payload_hash
            ),
            terminal_event_id=committed.event_id,
            transaction_id=committed.transaction_id,
            terminal_outcome="committed",
            render_hash=committed.render_hash,
            pending_transition={
                "before_hash": "7" * 64,
                "after_hash": "8" * 64,
            },
            render_manifest={"fragment_hashes": []},
        )
        with self.assertRaisesRegex(RuntimeError, "immutable outbox"):
            validate_startup_terminal_protocol(
                [matching_event],
                [committed, conflicting],
                expected_revision=REVISION,
            )
        with self.assertRaisesRegex(RuntimeError, "no matching runtime event"):
            validate_startup_terminal_protocol(
                [],
                [committed],
                expected_revision=REVISION,
            )
        aborted = replace(
            _terminal_outcome(
                "aborted",
                failure_category="timeout",
                product_fingerprint="b" * 64,
            ),
            base_fingerprint="a" * 64,
        )
        with self.assertRaisesRegex(RuntimeError, "advanced the Product Head"):
            validate_startup_terminal_protocol(
                [],
                [aborted],
                expected_revision=REVISION,
            )
        with self.assertRaisesRegex(RuntimeError, "revision is stale"):
            validate_startup_terminal_protocol(
                [],
                [replace(aborted, origin_revision={
                    "commit": "stale",
                    "worktree_hash": "0" * 64,
                })],
                expected_revision=REVISION,
            )

    def test_startup_session_rejects_wrong_identity_and_presentation(self) -> None:
        response = "Agent> 任意启动内容"
        class WrongSessionStream(FakeTerminalOutcomeStream):
            def baseline_session_events(self, **kwargs):
                current = super().baseline_session_events(**kwargs)[0]
                return (replace(current, session_id="another-session"),)

        with self.assertRaisesRegex(RuntimeError, "typed terminal session"):
            validate_startup_session_event(
                WrongSessionStream(turn_count=1),
                expected_revision=REVISION,
                expected_provider="deepseek",
                expected_model="deepseek-chat",
                expected_session_id="contract-session",
                expected_session_purpose="dynamic-dual-ai-chaos",
                startup_response=response,
            )

        class WrongPresentationStream(FakeTerminalOutcomeStream):
            def baseline_session_events(self, **kwargs):
                current = super().baseline_session_events(**kwargs)[0]
                return (replace(current, presentation_hash="0" * 64),)

        with self.assertRaisesRegex(RuntimeError, "typed terminal session"):
            validate_startup_session_event(
                WrongPresentationStream(turn_count=1),
                expected_revision=REVISION,
                expected_provider="deepseek",
                expected_model="deepseek-chat",
                expected_session_id="contract-session",
                expected_session_purpose="dynamic-dual-ai-chaos",
                startup_response=response,
            )

        class WrongModelStream(FakeTerminalOutcomeStream):
            def baseline_session_events(self, **kwargs):
                current = super().baseline_session_events(**kwargs)[0]
                return (replace(current, model="deepseek-v4-pro"),)

        with self.assertRaisesRegex(
            RuntimeError,
            "expected deepseek/deepseek-chat, observed deepseek/deepseek-v4-pro",
        ):
            validate_startup_session_event(
                WrongModelStream(turn_count=1),
                expected_revision=REVISION,
                expected_provider="deepseek",
                expected_model="deepseek-chat",
                expected_session_id="contract-session",
                expected_session_purpose="dynamic-dual-ai-chaos",
                startup_response=response,
            )

    def test_normal_response_without_event_is_protocol_violation(self) -> None:
        runtime_stream = FakeRuntimeEventStream([])
        terminal_stream = FakeTerminalOutcomeStream(
            turn_count=0,
            outcomes=[_terminal_outcome("committed")],
        )
        with self.assertRaises(TerminalProtocolViolation) as raised:
            wait_for_turn_completion(
                FakeTransport(["Agent> Choose the next action."]),
                runtime_stream,
                terminal_stream,
                timeout_seconds=1,
                expectation=_completion_expectation(),
                event_grace_seconds=0,
            )
        self.assertEqual(raised.exception.failure_kind, "response_without_event")

    def test_nonterminal_runtime_event_cannot_complete_a_turn(self) -> None:
        event = replace(
            self._event(2, "b" * 64, "c" * 64, "opening_next_action"),
            event_type="startup_snapshot",
        )
        runtime_stream = FakeRuntimeEventStream([event])
        terminal_stream = FakeTerminalOutcomeStream(turn_count=1)
        with self.assertRaises(TerminalProtocolViolation) as raised:
            wait_for_turn_completion(
                FakeTransport(["Agent> Choose the next action."]),
                runtime_stream,
                terminal_stream,
                timeout_seconds=1,
                expectation=_completion_expectation(),
                event_grace_seconds=0,
            )
        self.assertIs(raised.exception.event, event)

    def test_event_without_complete_response_is_presentation_violation(self) -> None:
        class MissingResponseTransport(FakeTransport):
            def read_complete_agent_response(
                self,
                *,
                timeout_seconds: float,
            ) -> str:
                del timeout_seconds
                raise TimeoutError("partial output: Agent> still rendering")

        event = self._event(2, "b" * 64, "c" * 64, "opening_next_action")
        runtime_stream = FakeRuntimeEventStream([event])
        terminal_stream = FakeTerminalOutcomeStream(turn_count=1)
        with self.assertRaises(TerminalPresentationViolation) as raised:
            wait_for_turn_completion(
                MissingResponseTransport([]),
                runtime_stream,
                terminal_stream,
                timeout_seconds=1,
                expectation=_completion_expectation(),
                event_grace_seconds=0,
            )
        self.assertIs(raised.exception.event, event)
        self.assertIn("partial output", str(raised.exception.response_error))

    def test_typed_failure_preserves_transcript_and_nonqualifying_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ledger = self._ledger()
            runtime = root / "runtime"
            timeout_response = "Agent> " + t(
                'en',
                'turn_timeout',
                timeout=30,
                provider='deepseek',
                model='deepseek-chat',
            )
            transport = FakeTransport([
                "Agent> Model config: provider=deepseek, model=deepseek-chat, auth=api_key\n"
                "Agent> Choose a mode.",
                timeout_response,
            ])
            events = FakeRuntimeEventStream([
                replace(
                    self._event(
                        1,
                        "a" * 64,
                        "b" * 64,
                        "opening_next_action",
                    ),
                    thread_id="typed-failure",
                    product_authority_id=(
                        "dynamic-dual-ai-chaos:typed-failure"
                    ),
                ),
            ])
            terminal_outcomes = FakeTerminalOutcomeStream(
                turn_count=0,
                startup_outcome_count=0,
                outcomes=[_terminal_outcome(
                    "aborted",
                    failure_category="timeout",
                    product_fingerprint="b" * 64,
                )],
            )
            runner = DynamicDualAiChaosRunner(
                ChaosRunConfig.linux(
                    root,
                    session_id="typed-failure",
                    runtime_root=runtime,
                ),
                lambda context: SimulatorDecision(
                    user_message="Use the offered mode.",
                    persona=context.scheduled_target.persona,
                    goal=context.scheduled_target.goal,
                    rationale="Selected from the complete response.",
                    target_coverage_ids=(context.scheduled_target.edge_key,),
                ),
                ledger=ledger,
                schedule=self._schedule(ledger),
                transport=transport,
                event_stream=events,
                terminal_outcome_stream=terminal_outcomes,
                revision=REVISION,
            )

            with self.assertRaises(TerminalTurnFailure):
                runner.run()

            transcript = (runtime / "transcript.txt").read_text(encoding="utf-8")
            self.assertIn("User> Use the offered mode.", transcript)
            self.assertIn("stopped on timeout", transcript)
            diagnostics = list((runtime / "diagnostics").glob("terminal-*.json"))
            self.assertEqual(len(diagnostics), 1)
            payload = json.loads(diagnostics[0].read_text(encoding="utf-8"))
            self.assertFalse(payload["qualifying_evidence"])
            self.assertEqual(payload["failure"]["terminal_outcome"], "timeout")
            self.assertEqual(
                list((runtime / "evidence").glob("*.json")),
                [],
            )

    def test_scheduler_rejects_a_seed_scenario_owned_by_another_edge(self) -> None:
        ledger = self._ledger()
        with self.assertRaisesRegex(ValueError, "not authoritative for edge"):
            build_chaos_schedule(
                ledger,
                revision=REVISION,
                seed=17,
                targets=[{
                    "target_id": "wrong-seed",
                    "edge_key": EDGE["edge_key"],
                    "persona": "operator",
                    "goal": "choose a mode",
                    "scenario_id": "provider_detected",
                }],
            )

    def test_seeded_runner_traverses_resume_before_exposing_target_response(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ledger = self._ledger()
            schedule = build_chaos_schedule(
                ledger,
                revision=REVISION,
                seed=17,
                targets=[{
                    "target_id": "seeded-mode-turn",
                    "edge_key": EDGE["edge_key"],
                    "persona": "returning operator",
                    "goal": "choose fake-node after resuming reviewed state",
                    "scenario_id": "opening",
                }],
            )
            transport = FakeTransport([
                "Agent> Model config: provider=deepseek, model=deepseek-chat, auth=api_key\n"
                "Agent> Previous configuration found. Continue it?",
                "Agent> Choose a mode.\n1. fake-node\n2. real-node",
                "Agent> Which chain do you want to test?",
            ])
            seen: list[SimulatorContext] = []

            def simulator(context: SimulatorContext) -> SimulatorDecision:
                seen.append(context)
                self.assertIn("Choose a mode", context.previous_agent_response)
                self.assertNotIn("Previous configuration", context.previous_agent_response)
                return SimulatorDecision(
                    user_message="Use the simulated node for this check.",
                    persona=context.scheduled_target.persona,
                    goal=context.scheduled_target.goal,
                    rationale="The restored response offers fake-node.",
                    target_coverage_ids=(context.scheduled_target.edge_key,),
                )

            runner = DynamicDualAiChaosRunner(
                ChaosRunConfig.linux(
                    root,
                    session_id="contract-session",
                ),
                simulator,
                ledger=ledger,
                schedule=schedule,
                transport=transport,
                event_stream=FakeRuntimeEventStream([
                    self._event(1, "a" * 64, "b" * 64, "resume_harness_session"),
                    self._event(
                        2,
                        "b" * 64,
                        "c" * 64,
                        "opening_next_action",
                        "continue",
                    ),
                    self._event(
                        3,
                        "c" * 64,
                        "d" * 64,
                        "chain_select",
                        "Use the simulated node for this check.",
                    ),
                ]),
                terminal_outcome_stream=FakeTerminalOutcomeStream(
                    turn_count=3,
                    responses=list(transport.responses),
                ),
                revision=REVISION,
                clock_ns=OrderedClock(),
            )

            with patch(
                "tests.agent_live.runtime_checkpoint.seed_runtime_checkpoint"
            ) as seed_checkpoint:
                result = runner.run()

            self.assertEqual(result.execution_status, "complete")
            self.assertEqual(transport.submitted, [
                "continue",
                "Use the simulated node for this check.",
            ])
            self.assertEqual(len(seen), 1)
            seed_checkpoint.assert_called_once()

    def test_seeded_runner_exposes_resume_when_resume_is_the_scheduled_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            resume_edge = {
                **EDGE,
                "edge_key": "opening::resume_harness_session::variant::natural_language_option::option:3",
                "question_id": "resume_harness_session",
                "contract_hash": contract_variant_hash({
                    "id": "resume_harness_session",
                    "options": [
                        {"id": "1", "value": "continue"},
                        {"id": "2", "value": "modify"},
                        {"id": "3", "value": "reset"},
                    ],
                }),
                "edge_type": "question_option",
                "action_type": "answer_pending",
                "option_id": "3",
                "option_value": "reset",
                "expected_postcondition": {"confirmed_config": {}},
                "executable_scenario_ids": ["resume_qps"],
            }
            ledger = {"revision": REVISION, "edges": [resume_edge]}
            schedule = build_chaos_schedule(
                ledger,
                revision=REVISION,
                seed=18,
                targets=[{
                    "target_id": "resume-reset-turn",
                    "edge_key": resume_edge["edge_key"],
                    "persona": "returning operator",
                    "goal": "discard the partial configuration in natural language",
                    "scenario_id": "resume_qps",
                }],
            )
            transport = FakeTransport([
                "Agent> Model config: provider=deepseek, model=deepseek-chat, auth=api_key\n"
                "Agent> Previous configuration found. Continue, modify, or clear it?",
                "Agent> Previous configuration cleared. Choose a target mode.",
            ])
            seen: list[SimulatorContext] = []

            def simulator(context: SimulatorContext) -> SimulatorDecision:
                seen.append(context)
                self.assertIn("Continue, modify, or clear", context.previous_agent_response)
                return SimulatorDecision(
                    user_message="Discard that partial setup and let me start clean.",
                    persona=context.scheduled_target.persona,
                    goal=context.scheduled_target.goal,
                    rationale="The live startup response offers clearing the saved session.",
                    target_coverage_ids=(context.scheduled_target.edge_key,),
                )

            committed = replace(
                self._event(
                    2,
                    "b" * 64,
                    "c" * 64,
                    "opening_next_action",
                    "Discard that partial setup and let me start clean.",
                ),
                admitted_action_types=("answer_pending",),
                state_diff_hashes={"confirmed_config": {"before": "1" * 64, "after": "2" * 64}},
                after_value_hashes={"confirmed_config": content_hash({})},
                next_result={"kind": "question", "question_id": "opening_next_action"},
            )
            runner = DynamicDualAiChaosRunner(
                ChaosRunConfig.linux(root, session_id="contract-session"),
                simulator,
                ledger=ledger,
                schedule=schedule,
                transport=transport,
                event_stream=FakeRuntimeEventStream([
                    self._event(1, "a" * 64, "b" * 64, "resume_harness_session"),
                    committed,
                ]),
                terminal_outcome_stream=FakeTerminalOutcomeStream(
                    turn_count=2,
                    responses=list(transport.responses),
                ),
                revision=REVISION,
                clock_ns=OrderedClock(),
            )

            with patch("tests.agent_live.runtime_checkpoint.seed_runtime_checkpoint"):
                result = runner.run()

            self.assertEqual(result.execution_status, "complete")
            self.assertEqual(
                transport.submitted,
                ["Discard that partial setup and let me start clean."],
            )
            self.assertEqual(len(seen), 1)

    def test_action_edge_can_interrupt_unrelated_startup_overlay_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            action_edge = {
                **EDGE,
                "edge_key": "@action_only/coordinator::::variant::action_only_transition::action:queue_workflow_goal",
                "edge_type": "action_transition",
                "question_id": "",
                "action_type": "queue_workflow_goal",
                "executable_scenario_ids": ["action_queue_workflow_goal"],
                "expected_postcondition": {},
            }
            ledger = {"revision": REVISION, "edges": [action_edge]}
            schedule = build_chaos_schedule(
                ledger,
                revision=REVISION,
                seed=177,
                targets=[{
                    "target_id": "seeded-action-turn",
                    "edge_key": action_edge["edge_key"],
                    "persona": "operator with a later workflow goal",
                    "goal": "save sync-observe for after the current benchmark",
                    "scenario_id": "action_queue_workflow_goal",
                }],
            )
            transport = FakeTransport([
                "Agent> Model config: provider=deepseek, model=deepseek-chat, auth=api_key\n"
                "Agent> Previous configuration found. Continue it?",
                "Agent> Confirmed values were kept. Choose the typed action to perform.",
                "Agent> Saved sync-observe as a later workflow goal.",
            ])
            seen: list[SimulatorContext] = []

            def simulator(context: SimulatorContext) -> SimulatorDecision:
                seen.append(context)
                self.assertIn("Choose the typed action", context.previous_agent_response)
                self.assertNotIn("Previous configuration", context.previous_agent_response)
                return SimulatorDecision(
                    user_message="Finish this benchmark first, then observe node synchronization.",
                    persona=context.scheduled_target.persona,
                    goal=context.scheduled_target.goal,
                    rationale="The operator explicitly defers sync-observe until later.",
                    target_coverage_ids=(context.scheduled_target.edge_key,),
                )

            startup = self._event(1, "a" * 64, "b" * 64, "resume_harness_session")
            waiting = replace(
                self._event(
                    2,
                    "b" * 64,
                    "c" * 64,
                    "typed_action_intake",
                    "modify",
                ),
                admitted_action_types=("answer_pending",),
                state_diff_hashes={"pending_question": {"before": "a" * 64, "after": "b" * 64}},
                pending_contract={
                    "id": "typed_action_intake",
                    "accepted_action_types": ["answer_pending"],
                },
                next_result={"kind": "question", "question_id": "typed_action_intake"},
            )
            committed = replace(
                self._event(
                    3,
                    "c" * 64,
                    "d" * 64,
                    "",
                    "Finish this benchmark first, then observe node synchronization.",
                ),
                admitted_action_types=("queue_workflow_goal",),
                state_diff_hashes={"workflow_goals": {"before": "a" * 64, "after": "b" * 64}},
                next_result={"kind": "result", "status": "workflow_goal_queued"},
            )
            runner = DynamicDualAiChaosRunner(
                ChaosRunConfig.linux(root, session_id="contract-session"),
                simulator,
                ledger=ledger,
                schedule=schedule,
                transport=transport,
                event_stream=FakeRuntimeEventStream([startup, waiting, committed]),
                terminal_outcome_stream=FakeTerminalOutcomeStream(
                    turn_count=3,
                    responses=list(transport.responses),
                ),
                revision=REVISION,
                clock_ns=OrderedClock(),
            )

            with patch("tests.agent_live.runtime_checkpoint.seed_runtime_checkpoint"):
                result = runner.run()

            self.assertEqual(result.execution_status, "complete")
            self.assertEqual(transport.submitted, [
                "modify",
                "Finish this benchmark first, then observe node synchronization.",
            ])
            self.assertEqual(len(seen), 1)

    def test_missing_provider_identity_fails_closed_without_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ledger = self._ledger()
            runner = DynamicDualAiChaosRunner(
                ChaosRunConfig.linux(root, session_id="contract-session"),
                lambda context: None,
                ledger=ledger,
                schedule=self._schedule(ledger),
                transport=FakeTransport(["Agent> Ready."]),
                event_stream=FakeRuntimeEventStream([]),
                terminal_outcome_stream=FakeTerminalOutcomeStream(
                    turn_count=0,
                    publish_session_event=False,
                ),
                revision=REVISION,
            )
            with self.assertRaisesRegex(RuntimeError, "terminal session"):
                runner.run()
            result = json.loads(
                (root / ".agent/dynamic-chaos/contract-session/schedule-result.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(result["execution_status"], "incomplete")
            self.assertEqual(result["passed_target_count"], 0)

    def test_runtime_event_from_different_revision_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ledger = self._ledger()
            stale = replace(
                self._event(1, "a" * 64, "b" * 64, "opening_next_action"),
                revision={"commit": "stale", "worktree_hash": "b" * 64},
            )
            runner = DynamicDualAiChaosRunner(
                ChaosRunConfig.linux(root, session_id="contract-session"),
                lambda context: None,
                ledger=ledger,
                schedule=self._schedule(ledger),
                transport=FakeTransport([
                    "Agent> Model config: provider=deepseek, model=deepseek-chat, auth=api_key\n"
                    "Agent> Choose a mode."
                ]),
                event_stream=FakeRuntimeEventStream([stale]),
                terminal_outcome_stream=FakeTerminalOutcomeStream(
                    turn_count=1,
                ),
                revision=REVISION,
            )

            with self.assertRaisesRegex(RuntimeError, "revision"):
                runner.run()

    def test_missing_committed_event_fails_closed_after_transport_return(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ledger = self._ledger()
            transport = FakeTransport([
                "Agent> Model config: provider=deepseek, model=deepseek-chat, auth=api_key\nReady.",
                "Agent> Returned, but no runtime event exists.",
            ])
            runner = DynamicDualAiChaosRunner(
                ChaosRunConfig.linux(root, session_id="contract-session"),
                lambda context: SimulatorDecision(
                    user_message="fake node",
                    persona=context.scheduled_target.persona,
                    goal=context.scheduled_target.goal,
                    rationale="Response offered a mode.",
                    target_coverage_ids=(context.scheduled_target.edge_key,),
                ),
                ledger=ledger,
                schedule=self._schedule(ledger),
                transport=transport,
                event_stream=FakeRuntimeEventStream([
                    self._event(1, "a" * 64, "b" * 64, "opening_next_action"),
                ]),
                terminal_outcome_stream=FakeTerminalOutcomeStream(
                    turn_count=2,
                    responses=list(transport.responses),
                ),
                revision=REVISION,
            )
            with self.assertRaises(TerminalProtocolViolation):
                runner.run()
            self.assertEqual(list((root / ".agent/dynamic-chaos/contract-session/evidence").glob("*.json")), [])
            diagnostics = list(
                (root / ".agent/dynamic-chaos/contract-session/diagnostics").glob("*.json")
            )
            self.assertEqual(len(diagnostics), 1)
            diagnostic = json.loads(diagnostics[0].read_text(encoding="utf-8"))
            self.assertEqual(
                diagnostic["artifact_type"],
                "terminal_completion_diagnostic",
            )
            self.assertFalse(diagnostic["qualifying_evidence"])
            self.assertEqual(
                diagnostic["failure"]["failure_kind"],
                "response_without_event",
            )
            self.assertEqual(diagnostic["last_complete_event"]["turn_index"], 1)

    def test_failed_postcondition_fails_closed_after_event_advancement(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ledger = self._ledger()
            transport = FakeTransport([
                "Agent> Model config: provider=deepseek, model=deepseek-chat, auth=api_key\nReady.",
                "Agent> Authorization: Bearer super-secret-token. Next question.",
            ])

            runner = DynamicDualAiChaosRunner(
                ChaosRunConfig.linux(root, session_id="contract-session"),
                lambda context: SimulatorDecision(
                    user_message="https://rpc.example/abcdefghijklmnopqrstuvwxyz123456",
                    persona=context.scheduled_target.persona,
                    goal=context.scheduled_target.goal,
                    rationale="Response offered a mode.",
                    target_coverage_ids=(context.scheduled_target.edge_key,),
                ),
                ledger=ledger,
                schedule=self._schedule(ledger),
                transport=transport,
                event_stream=FakeRuntimeEventStream([
                    self._event(1, "a" * 64, "b" * 64, "opening_next_action"),
                    replace(
                        self._event(
                            2,
                            "b" * 64,
                            "c" * 64,
                            "chain_select",
                            "https://rpc.example/abcdefghijklmnopqrstuvwxyz123456",
                        ),
                        admitted_action_types=(),
                    ),
                ]),
                terminal_outcome_stream=FakeTerminalOutcomeStream(
                    turn_count=2,
                    responses=list(transport.responses),
                ),
                revision=REVISION,
            )
            observed_pending: list[dict] = []

            def verify_after_durable_boundary(*args: object) -> object:
                diagnostics = list(
                    (root / ".agent/dynamic-chaos/contract-session/diagnostics").glob("*.json")
                )
                self.assertEqual(len(diagnostics), 1)
                pending = json.loads(diagnostics[0].read_text(encoding="utf-8"))
                self.assertEqual(pending["verification_status"], "verification_pending")
                observed_pending.append(pending)
                return verify_runtime_postcondition(*args)  # type: ignore[arg-type]

            with patch(
                "tests.agent_live.dynamic_dual_ai_chaos.verify_runtime_postcondition",
                side_effect=verify_after_durable_boundary,
            ):
                with self.assertRaisesRegex(ValueError, "postcondition did not pass"):
                    runner.run()
            result = json.loads(
                (root / ".agent/dynamic-chaos/contract-session/schedule-result.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(result["targets"][0]["status"], "failed")
            self.assertEqual(result["passed_target_count"], 0)
            self.assertEqual(len(observed_pending), 1)
            diagnostics = list(
                (root / ".agent/dynamic-chaos/contract-session/diagnostics").glob("*.json")
            )
            self.assertEqual(len(diagnostics), 1)
            diagnostic = json.loads(diagnostics[0].read_text(encoding="utf-8"))
            valid, reason = validate_pty_diagnostic_artifact(diagnostic)
            self.assertTrue(valid, reason)
            self.assertEqual(diagnostic["diagnostic_kind"], "failed_attempt")
            self.assertEqual(diagnostic["verification_status"], "postcondition_failed")
            self.assertFalse(diagnostic["qualifying_evidence"])
            boundary = diagnostic["last_complete_boundary"]
            self.assertEqual(
                boundary["completed_turn"]["user_message"],
                "https://rpc.example/***REDACTED***",
            )
            self.assertIn("Bearer ***REDACTED***", boundary["completed_turn"]["agent_response"])
            self.assertNotIn("super-secret-token", diagnostics[0].read_text(encoding="utf-8"))
            transcript = (
                root / ".agent/dynamic-chaos/contract-session/transcript.txt"
            ).read_text(encoding="utf-8")
            self.assertNotIn("super-secret-token", transcript)
            self.assertNotIn("abcdefghijklmnopqrstuvwxyz123456", transcript)
            self.assertIn("Bearer ***REDACTED***", transcript)
            self.assertIn("https://rpc.example/***REDACTED***", transcript)
            self.assertFalse(boundary["verified_postcondition"]["passed"])
            qualifying, rejection = load_valid_evidence_reference(
                str(diagnostics[0]), edge=EDGE, revision=REVISION
            )
            self.assertIsNone(qualifying)
            self.assertIn("never qualify", rejection)
            self.assertEqual(
                list((root / ".agent/dynamic-chaos/contract-session/evidence").glob("*.json")),
                [],
            )

    def test_runner_executes_response_driven_deferred_continuation_before_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            edge = {
                **EDGE,
                "edge_key": "@action_only/coordinator::::variant::action_only_transition::action:change_group:sync_observe",
                "edge_type": "action_transition",
                "question_id": "",
                "action_type": "change_group",
                "expected_postcondition": {
                    "target_group": "sync_observe",
                    "navigation_transition": {
                        "immediate_group": "sync_observe",
                        "prerequisite_deferred_groups": ["target_mode"],
                    },
                },
                "deferred_transition_contract": {
                    "requires_linked_journey": True,
                    "max_continuation_turns": 3,
                    "entry_prerequisite_groups": ["target_mode"],
                    "terminal_group": "sync_observe",
                    "terminal_postcondition": {"target_group": "sync_observe"},
                    "linkage": [
                        "same_thread",
                        "same_revision",
                        "contiguous_turn_index",
                        "contiguous_fingerprint_chain",
                    ],
                },
                "executable_scenario_ids": [],
            }
            ledger = {"revision": REVISION, "edges": [edge]}
            schedule = build_chaos_schedule(
                ledger,
                revision=REVISION,
                seed=276,
                targets=[{
                    "target_id": "deferred-sync-observe",
                    "edge_key": edge["edge_key"],
                    "persona": "operator",
                    "goal": "switch to sync-observe and satisfy its live prerequisite",
                }],
            )
            baseline = replace(
                self._event(1, "a" * 64, "b" * 64, ""),
                active_group="opening",
                pending_contract={},
                admitted_action_types=(),
                state_diff_hashes={},
            )
            deferred = replace(
                self._event(
                    2,
                    "b" * 64,
                    "c" * 64,
                    "target_mode_select",
                    "Switch this workflow to sync-observe.",
                ),
                active_group="target_mode",
                admitted_action_types=("change_group",),
                admitted_action_targets=({"type": "change_group", "group": "sync_observe"},),
                state_diff_hashes={
                    "control.deferred_group": {"before": "", "after": "d" * 64},
                    "active_group": {"before": "1" * 64, "after": "2" * 64},
                },
                after_value_hashes={
                    "control.deferred_group": content_hash("sync_observe"),
                },
                next_result={
                    "kind": "question",
                    "question_id": "target_mode_select",
                    "group": "target_mode",
                },
            )
            terminal = replace(
                self._event(
                    3,
                    "c" * 64,
                    "d" * 64,
                    "sync_observe_source",
                    "Use real-node so sync-observe can continue.",
                ),
                active_group="sync_observe",
                admitted_action_types=("choose_target_mode",),
                state_diff_hashes={
                    "target_mode": {"before": "", "after": "3" * 64},
                    "control.deferred_group": {"before": "d" * 64, "after": ""},
                    "active_group": {"before": "2" * 64, "after": "4" * 64},
                },
                after_value_hashes={},
                next_result={
                    "kind": "question",
                    "question_id": "sync_observe_source",
                    "group": "sync_observe",
                },
            )
            transport = FakeTransport([
                "Agent> Model config: provider=deepseek, model=deepseek-chat, auth=api_key\n"
                "Agent> What would you like to change?",
                "Agent> Choose fake-node, real-node, or sync-observe prerequisites.",
                "Agent> sync-observe selected. Choose the observation source.",
            ])
            seen: list[SimulatorContext] = []

            def simulator(context: SimulatorContext) -> SimulatorDecision:
                seen.append(context)
                if len(seen) == 1:
                    self.assertNotIn("execution_phase", context.coverage_contract)
                    message = "Switch this workflow to sync-observe."
                else:
                    self.assertEqual(
                        context.coverage_contract["execution_phase"],
                        "linked_prerequisite_continuation",
                    )
                    self.assertIn("Choose fake-node", context.previous_agent_response)
                    self.assertEqual(
                        list((root / ".agent/dynamic-chaos/contract-session/evidence").glob("*.json")),
                        [],
                    )
                    message = "Use real-node so sync-observe can continue."
                return SimulatorDecision(
                    user_message=message,
                    persona=context.scheduled_target.persona,
                    goal=context.scheduled_target.goal,
                    rationale="Selected only after reading the current complete CLI response.",
                    target_coverage_ids=(context.scheduled_target.edge_key,),
                )

            runner = DynamicDualAiChaosRunner(
                ChaosRunConfig.linux(root, session_id="contract-session"),
                simulator,
                ledger=ledger,
                schedule=schedule,
                transport=transport,
                event_stream=FakeRuntimeEventStream([baseline, deferred, terminal]),
                terminal_outcome_stream=FakeTerminalOutcomeStream(
                    turn_count=3,
                    responses=list(transport.responses),
                ),
                revision=REVISION,
                clock_ns=OrderedClock(),
            )

            result = runner.run()

            self.assertEqual(result.execution_status, "complete")
            self.assertEqual(len(seen), 2)
            self.assertEqual(transport.submitted, [
                "Switch this workflow to sync-observe.",
                "Use real-node so sync-observe can continue.",
            ])
            self.assertEqual(len(result.turns), 2)
            self.assertEqual(len(result.evidence_paths), 2)
            artifact = json.loads(result.evidence_paths[0].read_text(encoding="utf-8"))
            valid, reason = validate_pty_cli_evidence_artifact(
                artifact,
                edge=edge,
                revision=REVISION,
            )
            self.assertTrue(valid, reason)
            observation = artifact["turn_observation"]
            self.assertEqual(len(observation["runtime_events"]), 3)
            self.assertEqual(len(observation["continuation_turns"]), 1)
            self.assertEqual(
                observation["verified_postcondition"]["details"]
                ["declared_target_results"][edge["edge_key"]]["journey_status"],
                "terminal_postcondition_observed",
            )

    def test_later_transport_failure_preserves_only_the_prior_complete_turn(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ledger = self._ledger()
            schedule = build_chaos_schedule(
                ledger,
                revision=REVISION,
                seed=23,
                targets=[
                    {
                        "target_id": "first-complete",
                        "edge_key": EDGE["edge_key"],
                        "persona": "operator",
                        "goal": "complete one observed turn",
                    },
                    {
                        "target_id": "second-interrupted",
                        "edge_key": EDGE["edge_key"],
                        "persona": "operator",
                        "goal": "exercise a later transport interruption",
                    },
                ],
            )
            transport = FakeTransport([
                "Agent> Model config: provider=deepseek, model=deepseek-chat, auth=api_key\nReady.",
                "Agent> First complete response.",
                "Agent> Second response without a committed event.",
            ])

            def simulator(context: SimulatorContext) -> SimulatorDecision:
                return SimulatorDecision(
                    user_message=f"turn-{context.turn_index}",
                    persona=context.scheduled_target.persona,
                    goal=context.scheduled_target.goal,
                    rationale="The current response determines this live turn.",
                    target_coverage_ids=(context.scheduled_target.edge_key,),
                )

            runner = DynamicDualAiChaosRunner(
                ChaosRunConfig.linux(root, session_id="contract-session"),
                simulator,
                ledger=ledger,
                schedule=schedule,
                transport=transport,
                event_stream=FakeRuntimeEventStream([
                    self._event(1, "a" * 64, "b" * 64, "opening_next_action"),
                    self._event(
                        2,
                        "b" * 64,
                        "c" * 64,
                        "opening_next_action",
                        "turn-2",
                    ),
                ]),
                terminal_outcome_stream=FakeTerminalOutcomeStream(
                    turn_count=2,
                    responses=list(transport.responses),
                ),
                revision=REVISION,
                clock_ns=OrderedClock(),
            )
            with self.assertRaises(TerminalProtocolViolation):
                runner.run()

            diagnostics = list(
                (root / ".agent/dynamic-chaos/contract-session/diagnostics").glob("*.json")
            )
            self.assertEqual(len(diagnostics), 1)
            diagnostic = json.loads(diagnostics[0].read_text(encoding="utf-8"))
            self.assertEqual(
                diagnostic["failure"]["failure_kind"],
                "response_without_event",
            )
            self.assertEqual(diagnostic["last_complete_event"]["turn_index"], 2)
            transcript = (
                root / ".agent/dynamic-chaos/contract-session/transcript.txt"
            ).read_text(encoding="utf-8")
            self.assertIn("User> turn-3", transcript)
            self.assertIn("Second response without a committed event", transcript)
            schedule_result = json.loads(
                (root / ".agent/dynamic-chaos/contract-session/schedule-result.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(schedule_result["passed_target_count"], 1)
            self.assertEqual(schedule_result["targets"][1]["status"], "failed")

    def test_action_edge_rejects_an_unrelated_admitted_action(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            action_edge = {
                **EDGE,
                "edge_key": "@action_only/coordinator::::variant::action_only_transition::action:change_group",
                "edge_type": "action_transition",
                "action_type": "change_group",
                "expected_postcondition": {},
            }
            ledger = {"revision": REVISION, "edges": [action_edge]}
            schedule = build_chaos_schedule(
                ledger,
                revision=REVISION,
                seed=19,
                targets=[{
                    "target_id": "wrong-action",
                    "edge_key": action_edge["edge_key"],
                    "persona": "operator",
                    "goal": "change workflow group",
                }],
            )
            runner = DynamicDualAiChaosRunner(
                ChaosRunConfig.linux(root, session_id="contract-session"),
                lambda context: SimulatorDecision(
                    user_message="Take me to workload settings.",
                    persona=context.scheduled_target.persona,
                    goal=context.scheduled_target.goal,
                    rationale="The operator requested a group change.",
                    target_coverage_ids=(context.scheduled_target.edge_key,),
                ),
                ledger=ledger,
                schedule=schedule,
                transport=FakeTransport([
                    "Agent> Model config: provider=deepseek, model=deepseek-chat, auth=api_key\n"
                    "Agent> Which region?",
                    "Agent> Which chain?",
                ]),
                event_stream=FakeRuntimeEventStream([
                    self._event(1, "a" * 64, "b" * 64, ""),
                    self._event(
                        2,
                        "b" * 64,
                        "c" * 64,
                        "chain_select",
                        "Take me to workload settings.",
                    ),
                ]),
                terminal_outcome_stream=FakeTerminalOutcomeStream(
                    turn_count=2,
                    responses=[
                        (
                            "Agent> Model config: provider=deepseek, "
                            "model=deepseek-chat, auth=api_key\n"
                            "Agent> Which region?"
                        ),
                        "Agent> Which chain?",
                    ],
                ),
                revision=REVISION,
            )
            with self.assertRaisesRegex(ValueError, "scheduled action was not admitted"):
                runner.run()

    def test_manual_input_edge_verifies_target_path_without_treating_metadata_as_values(self) -> None:
        edge = {
            **EDGE,
            "edge_key": "provider_deployment::CLOUD_REGION::variant::free_text::action:answer_pending",
            "question_id": "CLOUD_REGION",
            "edge_type": "manual_input",
            "action_type": "answer_pending",
            "expected_postcondition": {
                "field": "CLOUD_REGION",
                "path": "confirmed_config.CLOUD_REGION",
            },
        }
        baseline = replace(
            self._event(1, "a" * 64, "b" * 64, "CLOUD_REGION"),
            pending_contract={
                "id": "CLOUD_REGION",
                "accepted_action_types": ["answer_pending"],
            },
        )
        committed = replace(
            self._event(2, "b" * 64, "c" * 64, "CLOUD_ZONE"),
            admitted_action_types=("answer_pending",),
            state_diff_hashes={
                "confirmed_config.CLOUD_REGION": {
                    "before": "",
                    "after": "d" * 64,
                }
            },
            after_value_hashes={"confirmed_config.CLOUD_REGION": "d" * 64},
        )

        verified = verify_runtime_postcondition(edge, baseline, committed, None)  # type: ignore[arg-type]

        self.assertTrue(verified.passed, verified.details)
        self.assertEqual(
            verified.details["expected_postcondition_paths"],
            ["confirmed_config.CLOUD_REGION"],
        )

    def test_manual_input_edge_requires_a_declared_successor_question(self) -> None:
        edge = {
            **EDGE,
            "edge_key": "endpoint_process::custom_rpc_method::variant::free_text::action:answer_pending",
            "question_id": "custom_rpc_method",
            "edge_type": "manual_input",
            "action_type": "answer_pending",
            "expected_postcondition": {
                "field": "custom_rpc_method",
                "path": "custom_rpc.catalog.draft.method",
                "next_question_ids": [
                    "custom_rpc_parameter_confirm",
                    "custom_rpc_schema_evidence",
                    "custom_rpc_schema_confirm",
                    "custom_rpc_response_confirm",
                ],
            },
        }
        baseline = replace(
            self._event(1, "a" * 64, "b" * 64, "custom_rpc_method"),
            pending_contract={
                "id": "custom_rpc_method",
                "accepted_action_types": ["answer_pending"],
            },
            after_value_hashes={"custom_rpc.catalog.draft.method": "1" * 64},
        )
        committed = replace(
            self._event(2, "b" * 64, "c" * 64, "benchmark_mode"),
            admitted_action_types=("answer_pending",),
            state_diff_hashes={
                "custom_rpc.catalog.draft.method": {
                    "before": "1" * 64,
                    "after": "2" * 64,
                }
            },
            after_value_hashes={"custom_rpc.catalog.draft.method": "2" * 64},
        )

        verified = verify_runtime_postcondition(edge, baseline, committed, None)  # type: ignore[arg-type]

        self.assertFalse(verified.passed)
        self.assertIn("unexpected next question", " ".join(verified.details["errors"]))

    def test_structured_review_edge_rejects_silently_lost_source_keys(self) -> None:
        edge = {
            **EDGE,
            "edge_key": "provider_deployment::CLOUD_REGION::structured::action:propose_config_values",
            "question_id": "CLOUD_REGION",
            "edge_type": "manual_input",
            "action_type": "propose_config_values",
            "expected_postcondition": {},
        }
        baseline = replace(
            self._event(1, "a" * 64, "b" * 64, "CLOUD_REGION"),
            pending_contract={
                "id": "CLOUD_REGION",
                "accepted_action_types": ["answer_pending", "propose_config_values"],
            },
        )
        committed = replace(
            self._event(2, "b" * 64, "c" * 64, "inferred_config_review"),
            admitted_action_types=("propose_config_values",),
            state_diff_hashes={
                "inferred_config.pending_review": {"before": "", "after": "d" * 64},
            },
            after_value_hashes={
                "inferred_config.pending_review.config_values.CLOUD_REGION": "d" * 64,
            },
        )
        turn = SimpleNamespace(
            user_message="CLOUD_REGION=us-central1\nowner_ticket=INC-4821",
            agent_response="Apply CLOUD_REGION?",
        )

        verified = verify_runtime_postcondition(edge, baseline, committed, turn)  # type: ignore[arg-type]

        self.assertFalse(verified.passed)
        self.assertIn(
            "structured review silently lost source key: OWNER_TICKET",
            verified.details["errors"],
        )
        self.assertEqual(
            verified.details["structured_review_keys"],
            ["CLOUD_REGION", "OWNER_TICKET"],
        )

        complete = replace(
            committed,
            after_value_hashes={
                **committed.after_value_hashes,
                "inferred_config.pending_review.unmapped_values.OWNER_TICKET": "e" * 64,
            },
        )
        accepted = verify_runtime_postcondition(edge, baseline, complete, turn)  # type: ignore[arg-type]
        self.assertTrue(accepted.passed, accepted.details)

    def test_resume_edge_verifies_saved_context_relation_for_any_group(self) -> None:
        relation = {
            "kind": "prefix_equals_before_prefix",
            "after_prefix": "pending_question",
            "before_prefix": "resume_context.pending_question",
            "ignored_suffixes": ["created_turn_index", "resume_action_queue"],
        }
        edge = {
            **EDGE,
            "edge_type": "question_option",
            "action_type": "answer_pending",
            "expected_postcondition": {},
            "expected_state_relations": [
                {
                    "kind": "path_equals_before_path",
                    "after_path": "active_group",
                    "before_path": "resume_context.active_group",
                },
                relation,
                {
                    "kind": "path_equals_before_path",
                    "after_path": "action_queue",
                    "before_path": "action_queue",
                },
            ],
        }
        baseline = replace(
            self._event(1, "a" * 64, "b" * 64, "resume_harness_session"),
            after_value_hashes={
                "resume_context.active_group": "1" * 64,
                "resume_context.pending_question.id": "2" * 64,
                "resume_context.pending_question.group": "1" * 64,
                "action_queue": "7" * 64,
            },
        )
        committed = replace(
            self._event(2, "b" * 64, "c" * 64, "qps_profile_confirm"),
            active_group="qps_profile",
            admitted_action_types=("answer_pending",),
            state_diff_hashes={"resume_context": {"before": "3" * 64, "after": "4" * 64}},
            after_value_hashes={
                "active_group": "1" * 64,
                "pending_question.id": "2" * 64,
                "pending_question.group": "1" * 64,
                "pending_question.created_turn_index": "5" * 64,
                "action_queue": "7" * 64,
            },
        )

        verified = verify_runtime_postcondition(edge, baseline, committed, None)  # type: ignore[arg-type]
        self.assertTrue(verified.passed, verified.details)

        mismatched = replace(
            committed,
            after_value_hashes={**committed.after_value_hashes, "pending_question.id": "9" * 64},
        )
        rejected = verify_runtime_postcondition(edge, baseline, mismatched, None)  # type: ignore[arg-type]
        self.assertFalse(rejected.passed)
        self.assertIn("state prefix relation was not observed", " ".join(rejected.details["errors"]))

        missing_before = replace(baseline, after_value_hashes={})
        missing_rejected = verify_runtime_postcondition(edge, missing_before, committed, None)  # type: ignore[arg-type]
        self.assertFalse(missing_rejected.passed)
        self.assertIn("state relation was not observed", " ".join(missing_rejected.details["errors"]))

    def test_manual_edge_rejects_action_only_transition_without_field_mutation(self) -> None:
        edge = {
            **EDGE,
            "edge_key": "provider_deployment::CLOUD_REGION::manual",
            "question_id": "CLOUD_REGION",
            "edge_type": "manual_input",
            "action_type": "answer_pending",
            "expected_postcondition": {
                "field": "CLOUD_REGION",
                "path": "confirmed_config.CLOUD_REGION",
            },
        }
        baseline = replace(
            self._event(1, "a" * 64, "b" * 64, "CLOUD_REGION"),
            pending_contract={
                "id": "CLOUD_REGION",
                "accepted_action_types": ["answer_pending"],
            },
        )
        committed = replace(
            self._event(2, "b" * 64, "c" * 64, "CLOUD_REGION"),
            admitted_action_types=("answer_pending",),
            state_diff_hashes={"applied_action_ids": {"before": "", "after": "d" * 64}},
            after_value_hashes={},
        )

        verified = verify_runtime_postcondition(edge, baseline, committed, None)  # type: ignore[arg-type]

        self.assertFalse(verified.passed)
        self.assertIn(
            "manual-input postcondition was not observed",
            " ".join(verified.details["errors"]),
        )

    def test_manual_rejection_requires_unchanged_field_and_pending_contract(self) -> None:
        edge = {
            **EDGE,
            "edge_key": "provider_deployment::CLOUD_REGION::empty",
            "question_id": "CLOUD_REGION",
            "edge_type": "manual_input",
            "action_type": "answer_pending",
            "expected_admitted": False,
            "expected_postcondition": {
                "field": "CLOUD_REGION",
                "path": "confirmed_config.CLOUD_REGION",
            },
        }
        contract = {"id": "CLOUD_REGION", "accepted_action_types": ["answer_pending"]}
        baseline = replace(
            self._event(1, "a" * 64, "b" * 64, "CLOUD_REGION"),
            pending_contract=contract,
            after_value_hashes={},
        )
        committed = replace(
            self._event(2, "b" * 64, "c" * 64, "CLOUD_REGION"),
            pending_contract=contract,
            admitted_action_types=(),
            state_diff_hashes={"visible_response": {"before": "", "after": "d" * 64}},
            after_value_hashes={},
        )

        verified = verify_runtime_postcondition(edge, baseline, committed, None)  # type: ignore[arg-type]

        self.assertTrue(verified.passed, verified.details)
        self.assertTrue(verified.details["rejection_observed"])
        self.assertEqual(verified.admitted_typed_actions, ())

        changed = replace(
            committed,
            after_value_hashes={"confirmed_config.CLOUD_REGION": "e" * 64},
        )
        rejected = verify_runtime_postcondition(edge, baseline, changed, None)  # type: ignore[arg-type]
        self.assertFalse(rejected.passed)
        self.assertIn("changed the destination field", " ".join(rejected.details["errors"]))

        advanced = replace(
            committed,
            pending_question_id="CLOUD_ZONE",
            pending_contract={"id": "CLOUD_ZONE", "accepted_action_types": ["answer_pending"]},
        )
        rejected = verify_runtime_postcondition(edge, baseline, advanced, None)  # type: ignore[arg-type]
        self.assertFalse(rejected.passed)
        self.assertIn("preserve the pending question", " ".join(rejected.details["errors"]))

    def test_manual_rejection_can_record_question_declared_negative_evidence(self) -> None:
        edge = {
            **EDGE,
            "edge_key": "endpoint_process::SYNC_OBSERVE_RPC_URL::unreachable",
            "question_id": "SYNC_OBSERVE_RPC_URL",
            "edge_type": "manual_input",
            "action_type": "answer_pending",
            "expected_admitted": False,
            "expected_postcondition": {
                "field": "SYNC_OBSERVE_RPC_URL",
                "path": "endpoint_evidence.sync_rpc_url_ready",
                "rejection_value": False,
            },
        }
        contract = {
            "id": "SYNC_OBSERVE_RPC_URL",
            "accepted_action_types": ["answer_pending"],
            "evidence_path": "endpoint_evidence.sync_rpc_url_ready",
            "rejection_evidence_value": False,
        }
        baseline = replace(
            self._event(1, "a" * 64, "b" * 64, "SYNC_OBSERVE_RPC_URL"),
            pending_contract=contract,
            after_value_hashes={},
        )
        committed = replace(
            self._event(2, "b" * 64, "c" * 64, "SYNC_OBSERVE_RPC_URL"),
            pending_contract=contract,
            admitted_action_types=(),
            state_diff_hashes={
                "endpoint_evidence.sync_rpc_url_ready": {
                    "before": "",
                    "after": "d" * 64,
                }
            },
            after_value_hashes={
                "endpoint_evidence.sync_rpc_url_ready": content_hash(False),
            },
        )

        verified = verify_runtime_postcondition(edge, baseline, committed, None)  # type: ignore[arg-type]
        self.assertTrue(verified.passed, verified.details)

        wrong_value = replace(
            committed,
            after_value_hashes={
                "endpoint_evidence.sync_rpc_url_ready": content_hash(True),
            },
        )
        rejected = verify_runtime_postcondition(edge, baseline, wrong_value, None)  # type: ignore[arg-type]
        self.assertFalse(rejected.passed)
        self.assertIn("declared evidence", " ".join(rejected.details["errors"]))

    def test_manual_chain_edge_verifies_chain_owner_state(self) -> None:
        edge = {
            **EDGE,
            "edge_key": "chain_identity::chain::manual",
            "question_id": "chain",
            "edge_type": "manual_input",
            "action_type": "answer_pending",
            "expected_postcondition": {
                "field": "chain",
                "path": "chain_identity.canonical",
            },
        }
        baseline = replace(
            self._event(1, "a" * 64, "b" * 64, "chain"),
            pending_contract={"id": "chain", "accepted_action_types": ["answer_pending"]},
        )
        committed = replace(
            self._event(2, "b" * 64, "c" * 64, "CLOUD_REGION"),
            admitted_action_types=("answer_pending",),
            state_diff_hashes={
                "chain_identity.canonical": {"before": "", "after": "d" * 64},
            },
            after_value_hashes={"chain_identity.canonical": "d" * 64},
        )

        verified = verify_runtime_postcondition(edge, baseline, committed, None)  # type: ignore[arg-type]

        self.assertTrue(verified.passed, verified.details)

    def test_change_group_edge_rejects_queued_only_navigation(self) -> None:
        edge = {
            **EDGE,
            "edge_key": "@action_only/coordinator::change_group",
            "question_id": "",
            "edge_type": "action_transition",
            "action_type": "change_group",
            "expected_postcondition": {"target_group": ""},
        }
        baseline = self._event(1, "a" * 64, "b" * 64, "inferred_config_review")
        committed = replace(
            self._event(2, "b" * 64, "c" * 64, "inferred_config_review"),
            active_group="chain_identity",
            admitted_action_types=("change_group",),
            admitted_action_targets=({"type": "change_group", "group": "qps_profile"},),
            action_queue_types=("change_group",),
            state_diff_hashes={"action_queue": {"before": "", "after": "d" * 64}},
            next_result={
                "kind": "question",
                "question_id": "inferred_config_review",
                "group": "chain_identity",
            },
        )

        verified = verify_runtime_postcondition(edge, baseline, committed, None)  # type: ignore[arg-type]

        self.assertFalse(verified.passed)
        self.assertIn("destination was not observed", " ".join(verified.details["errors"]))

    def test_change_group_edge_accepts_observed_destination(self) -> None:
        edge = {
            **EDGE,
            "edge_key": "@action_only/coordinator::change_group",
            "question_id": "",
            "edge_type": "action_transition",
            "action_type": "change_group",
            "expected_postcondition": {"target_group": ""},
        }
        baseline = self._event(1, "a" * 64, "b" * 64, "inferred_config_review")
        committed = replace(
            self._event(2, "b" * 64, "c" * 64, "benchmark_mode"),
            active_group="qps_profile",
            admitted_action_types=("change_group",),
            admitted_action_targets=({"type": "change_group", "group": "qps_profile"},),
            state_diff_hashes={"active_group": {"before": "", "after": "d" * 64}},
            next_result={
                "kind": "question",
                "question_id": "benchmark_mode",
                "group": "qps_profile",
            },
        )

        verified = verify_runtime_postcondition(edge, baseline, committed, None)  # type: ignore[arg-type]

        self.assertTrue(verified.passed, verified.details)

    def test_change_group_edge_accepts_declared_prerequisite_deferred_destination(self) -> None:
        edge = {
            **EDGE,
            "edge_key": "@action_only/coordinator::change_group::sync-observe-deferred",
            "question_id": "",
            "edge_type": "action_transition",
            "action_type": "change_group",
            "expected_postcondition": {
                "target_group": "sync_observe",
                "navigation_transition": {
                    "immediate_group": "sync_observe",
                    "prerequisite_deferred_groups": [
                        "target_mode",
                        "chain_identity",
                        "endpoint_process",
                    ],
                },
            },
        }
        baseline = self._event(1, "a" * 64, "b" * 64, "")
        committed = replace(
            self._event(2, "b" * 64, "c" * 64, "target_mode_select"),
            active_group="target_mode",
            admitted_action_types=("change_group",),
            admitted_action_targets=({"type": "change_group", "group": "sync_observe"},),
            state_diff_hashes={
                "control.deferred_group": {"before": "", "after": "d" * 64},
                "active_group": {"before": "", "after": "e" * 64},
            },
            after_value_hashes={
                "control.deferred_group": content_hash("sync_observe"),
            },
            next_result={
                "kind": "question",
                "question_id": "target_mode_select",
                "group": "target_mode",
            },
        )

        intermediate = verify_runtime_postcondition(edge, baseline, committed, None)  # type: ignore[arg-type]

        self.assertFalse(intermediate.passed)
        self.assertEqual(intermediate.observed_coverage_ids, ())
        self.assertEqual(
            intermediate.details["journey_status"],
            "awaiting_terminal_postcondition",
        )
        self.assertIn(
            "requires linked multi-turn terminal postcondition evidence",
            " ".join(intermediate.details["errors"]),
        )

        terminal = replace(
            self._event(3, "c" * 64, "d" * 64, "sync_observe_source"),
            active_group="sync_observe",
            admitted_action_types=("answer_pending",),
            state_diff_hashes={
                "active_group": {"before": "e" * 64, "after": "f" * 64},
                "control.deferred_group": {"before": "d" * 64, "after": ""},
            },
            after_value_hashes={},
            next_result={
                "kind": "question",
                "question_id": "sync_observe_source",
                "group": "sync_observe",
            },
        )
        verified = verify_runtime_postcondition(
            edge,
            baseline,
            committed,
            None,  # type: ignore[arg-type]
            continuation_events=(terminal,),
        )

        self.assertTrue(verified.passed, verified.details)
        self.assertEqual(verified.details["transition_outcome"], "prerequisite_deferred")
        self.assertEqual(
            verified.details["journey_status"],
            "terminal_postcondition_observed",
        )

        broken_link = replace(terminal, before_fingerprint="9" * 64)
        rejected = verify_runtime_postcondition(
            edge,
            baseline,
            committed,
            None,  # type: ignore[arg-type]
            continuation_events=(broken_link,),
        )
        self.assertFalse(rejected.passed)
        self.assertIn("broke the fingerprint chain", " ".join(rejected.details["errors"]))

    def test_natural_language_option_accepts_declared_deferred_owner_action(self) -> None:
        edge = {
            **EDGE,
            "edge_key": "observability::observability_mode::owner-deferred",
            "question_id": "observability_mode",
            "edge_type": "question_option",
            "action_type": "answer_pending",
            "expected_admitted_action_types": ["answer_pending", "set_observability"],
            "prerequisite_deferred_actions": {
                "set_observability": ["target_mode"],
            },
            "expected_postcondition": {"observability.mode": "exporter"},
        }
        baseline = replace(
            self._event(1, "a" * 64, "b" * 64, "observability_mode"),
            active_group="observability",
            pending_contract={
                "id": "observability_mode",
                "accepted_action_types": ["answer_pending", "set_observability"],
            },
            after_value_hashes={},
        )
        committed = replace(
            self._event(2, "b" * 64, "c" * 64, "target_mode_select"),
            active_group="target_mode",
            pending_contract={
                "id": "target_mode_select",
                "resume_action_queue": True,
            },
            admitted_action_types=("set_observability", "request_target_mode_selection"),
            action_queue_types=("set_observability",),
            state_diff_hashes={
                "action_queue": {"before": "", "after": "d" * 64},
                "active_group": {"before": "", "after": "e" * 64},
            },
            after_value_hashes={},
            next_result={
                "kind": "question",
                "question_id": "target_mode_select",
                "group": "target_mode",
            },
        )

        intermediate = verify_runtime_postcondition(edge, baseline, committed, None)  # type: ignore[arg-type]

        self.assertFalse(intermediate.passed)
        self.assertEqual(intermediate.observed_coverage_ids, ())
        self.assertEqual(
            intermediate.details["journey_status"],
            "awaiting_terminal_postcondition",
        )

        terminal = replace(
            self._event(3, "c" * 64, "d" * 64, "CLOUD_REGION"),
            active_group="provider_deployment",
            admitted_action_types=("choose_target_mode", "set_observability"),
            action_queue_types=(),
            state_diff_hashes={
                "target_mode": {"before": "", "after": "e" * 64},
                "observability.mode": {"before": "", "after": "f" * 64},
            },
            after_value_hashes={
                "observability.mode": content_hash("exporter"),
            },
            next_result={
                "kind": "question",
                "question_id": "CLOUD_REGION",
                "group": "provider_deployment",
            },
        )
        verified = verify_runtime_postcondition(
            edge,
            baseline,
            committed,
            None,  # type: ignore[arg-type]
            continuation_events=(terminal,),
        )

        self.assertTrue(verified.passed, verified.details)
        self.assertEqual(verified.details["transition_outcome"], "prerequisite_deferred")
        self.assertEqual(verified.details["deferred_action"], "set_observability")
        self.assertEqual(
            verified.details["journey_status"],
            "terminal_postcondition_observed",
        )

        missing_terminal_value = replace(terminal, after_value_hashes={})
        rejected_terminal = verify_runtime_postcondition(
            edge,
            baseline,
            committed,
            None,  # type: ignore[arg-type]
            continuation_events=(missing_terminal_value,),
        )
        self.assertFalse(rejected_terminal.passed)
        self.assertIn(
            "did not reach terminal postcondition",
            " ".join(rejected_terminal.details["errors"]),
        )

        not_deferred = replace(committed, action_queue_types=())
        rejected = verify_runtime_postcondition(edge, baseline, not_deferred, None)  # type: ignore[arg-type]
        self.assertFalse(rejected.passed)
        self.assertIn(
            "expected postcondition was not observed",
            " ".join(rejected.details["errors"]),
        )

        undeclared = replace(
            baseline,
            pending_contract={
                "id": "observability_mode",
                "accepted_action_types": ["answer_pending"],
            },
        )
        undeclared_result = verify_runtime_postcondition(
            edge,
            undeclared,
            committed,
            None,  # type: ignore[arg-type]
            continuation_events=(terminal,),
        )
        self.assertFalse(undeclared_result.passed)
        self.assertIn(
            "not declared by the pending contract",
            " ".join(undeclared_result.details["errors"]),
        )

    def test_unknown_or_stale_scheduler_target_is_rejected_before_cli_start(self) -> None:
        ledger = self._ledger()
        with self.assertRaisesRegex(ValueError, "unknown authoritative ledger edge"):
            build_chaos_schedule(
                ledger,
                revision=REVISION,
                seed=1,
                targets=[{
                    "edge_key": "invented-edge",
                    "persona": "operator",
                    "goal": "invent evidence",
                }],
            )
        with self.assertRaisesRegex(ValueError, "revision"):
            build_chaos_schedule(
                ledger,
                revision={"commit": "other", "worktree_hash": "b" * 64},
                seed=1,
                targets=[{
                    "edge_key": EDGE["edge_key"],
                    "persona": "operator",
                    "goal": "stale run",
                }],
            )

    def test_runner_sets_independent_docker_paths(self) -> None:
        root = Path("/tmp/anychain-chaos-contract")
        config = ChaosRunConfig.docker(
            root,
            session_id="isolated-session",
            execution_id="chaos-isolated-session",
            service="bench",
        )
        command_text = " ".join(config.command)
        self.assertIn("ANYCHAIN_CHAOS_EXECUTION_ID=chaos-isolated-session", command_text)
        self.assertIn(
            "ANYCHAIN_CHAOS_INNER_CLEANUP_RECEIPT_DIR=/workspace/.agent/dynamic-chaos/isolated-session/container-cleanup-receipts",
            command_text,
        )
        self.assertIn("ANYCHAIN_AGENT_CHECKPOINT_PATH=/workspace/.agent/dynamic-chaos/isolated-session/checkpoints.sqlite", command_text)
        self.assertIn("ANYCHAIN_AGENT_JOBS_DIR=/workspace/.agent/dynamic-chaos/isolated-session/jobs", command_text)
        self.assertEqual(config.transport_kind, "container_pty_bridge")
        self.assertIn("tests.agent_live.container_pty_bridge", config.command)
        self.assertEqual(config.command[-6:], (
            "--",
            "./bin/anychain-agent",
            "--state-file",
            "/workspace/.agent/dynamic-chaos/isolated-session/terminal-session.json",
            "--language",
            "en",
        ))
        self.assertIsInstance(transport_for_config(config), ContainerPtyBridgeTransport)

    def test_runner_sets_independent_linux_terminal_state(self) -> None:
        root = Path("/tmp/anychain-chaos-contract")
        config = ChaosRunConfig.linux(root, session_id="isolated-session")

        self.assertEqual(config.transport_kind, "container_pty_bridge")
        self.assertIn("tests.agent_live.container_pty_bridge", config.command)
        self.assertEqual(config.command[-4:], (
            "--state-file",
            "/tmp/anychain-chaos-contract/.agent/dynamic-chaos/isolated-session/terminal-session.json",
            "--language",
            "en",
        ))

    def test_linux_runtime_accepts_an_explicit_evidence_root(self) -> None:
        root = Path("/tmp/anychain-chaos-contract")
        runtime = root / ".agent" / "evidence" / "phase8" / "g3" / "exact"
        config = ChaosRunConfig.linux(
            root,
            session_id="retained-exact",
            runtime_root=runtime,
            runtime_root_in_process=runtime,
        )

        self.assertEqual(config.runtime_root, runtime)
        self.assertEqual(config.runtime_root_in_process, runtime)
        self.assertEqual(
            config.command[-4:],
            (
                "--state-file",
                str(runtime / "terminal-session.json"),
                "--language",
                "en",
            ),
        )

    def test_runtime_identity_follows_explicit_process_configuration(self) -> None:
        root = Path("/tmp/anychain-chaos-contract")
        with patch(
            "tests.agent_live.dynamic_dual_ai_chaos.load_llm_config",
            return_value=SimpleNamespace(
                provider="deepseek",
                model="configured-runtime-model",
            ),
        ):
            linux = ChaosRunConfig.linux(root, session_id="configured-linux")
            docker = ChaosRunConfig.docker(root, session_id="configured-docker")

        self.assertEqual(
            (linux.provider, linux.model),
            ("deepseek", "configured-runtime-model"),
        )
        self.assertEqual(
            (docker.provider, docker.model),
            ("deepseek", "configured-runtime-model"),
        )

    def test_runtime_identity_follows_persistent_agent_configuration(self) -> None:
        root = Path("/tmp/anychain-chaos-contract")
        configured = SimpleNamespace(
            provider="deepseek",
            model="deepseek-v4-pro",
        )
        with patch(
            "tests.agent_live.dynamic_dual_ai_chaos.load_llm_config",
            return_value=configured,
        ):
            direct = ChaosRunConfig(
                repo_root=root,
                command=("agent",),
                session_id="persistent-direct",
            )
            linux = ChaosRunConfig.linux(root, session_id="persistent-linux")
            docker = ChaosRunConfig.docker(root, session_id="persistent-docker")

        self.assertEqual(
            (direct.provider, direct.model),
            ("deepseek", "deepseek-v4-pro"),
        )
        self.assertEqual(
            (linux.provider, linux.model),
            ("deepseek", "deepseek-v4-pro"),
        )
        self.assertEqual(
            (docker.provider, docker.model),
            ("deepseek", "deepseek-v4-pro"),
        )
        command_text = " ".join(docker.command)
        self.assertIn("LLM_PROVIDER=deepseek", command_text)
        self.assertIn("LLM_MODEL=deepseek-v4-pro", command_text)
        self.assertIn(
            "AGENT_CONFIG_LOCAL=/dev/null",
            command_text,
        )

    def test_runtime_identity_rejects_partial_explicit_configuration(self) -> None:
        with self.assertRaisesRegex(ValueError, "supplied together"):
            ChaosRunConfig(
                repo_root=Path("/tmp/anychain-chaos-contract"),
                command=("agent",),
                provider="deepseek",
            )

    def test_explicit_runtime_identity_overrides_process_configuration(self) -> None:
        root = Path("/tmp/anychain-chaos-contract")
        with patch.dict(
            os.environ,
            {
                "LLM_PROVIDER": "deepseek",
                "LLM_MODEL": "configured-runtime-model",
            },
        ):
            config = ChaosRunConfig.linux(
                root,
                session_id="explicit-runtime",
                provider="openai",
                model="explicit-model",
            )

        self.assertEqual((config.provider, config.model), ("openai", "explicit-model"))

    def test_bracketed_paste_preserves_multiline_unicode_as_one_submission(self) -> None:
        message = "请分析：\n```json\n{\"method\":\"eth_call\",\"params\":[]}\n```"
        encoded = encode_bracketed_paste(message)
        self.assertTrue(encoded.startswith(b"\x1b[200~"))
        self.assertTrue(encoded.endswith(b"\x1b[201~"))
        self.assertEqual(encoded.count(b"\x1b[200~"), 1)
        self.assertEqual(encoded.count(b"\x1b[201~"), 1)
        self.assertFalse(encoded.endswith(b"\r"))

    def test_real_prompt_toolkit_submits_single_and_multiline_paste_once(self) -> None:
        program = (
            "from agent.terminal.io import TerminalIO; "
            "print('Agent> ready', flush=True); "
            "value=TerminalIO().input('en'); "
            "print('Agent> VALUE='+repr(value), flush=True); "
            "print('User> ', end='', flush=True)"
        )
        for message in ("single value", "first line\n第二行\nthird line"):
            with self.subTest(message=message):
                transport = SubprocessPtyTransport(
                    (sys.executable, "-c", program),
                    cwd=Path.cwd(),
                    poll_interval_seconds=0.05,
                )
                transport.start(env=dict(os.environ))
                try:
                    transport.read_complete_agent_response(timeout_seconds=5)
                    transport.submit_bracketed_paste(message)
                    response = transport.read_complete_agent_response(timeout_seconds=5)
                finally:
                    transport.close()
                self.assertIn(f"Agent> VALUE={message!r}", response)
                self.assertEqual(response.count("Agent> VALUE="), 1)

    def test_container_bridge_owns_one_pty_for_multiline_unicode(self) -> None:
        program = (
            "from agent.terminal.io import TerminalIO; "
            "print('Agent> ready', flush=True); "
            "value=TerminalIO().input('en'); "
            "print('Agent> VALUE='+repr(value), flush=True); "
            "print('User> ', end='', flush=True)"
        )
        message = "first line\n第二行\nthird line"
        command = (
            sys.executable,
            "-m",
            "tests.agent_live.container_pty_bridge",
            "--cwd",
            str(Path.cwd()),
            "--",
            sys.executable,
            "-c",
            program,
        )
        transport = ContainerPtyBridgeTransport(command, cwd=Path.cwd())
        transport.start(env=dict(os.environ))
        process = transport._process
        try:
            transport.read_complete_agent_response(timeout_seconds=5)
            transport.submit_bracketed_paste(message)
            response = transport.read_complete_agent_response(timeout_seconds=5)
        finally:
            transport.close()

        self.assertIn(f"Agent> VALUE={message!r}", response)
        self.assertEqual(response.count("Agent> VALUE="), 1)
        self.assertIsNotNone(process)
        self.assertIsNotNone(process.poll())
        self.assertIsNotNone(transport.cleanup_receipt)
        self.assertTrue(bool(transport.cleanup_receipt["cleaned"]))

    def test_container_bridge_forwards_ctrl_c_and_reaps_product_cli(self) -> None:
        program = (
            "from agent.terminal.io import TerminalIO; "
            "print('Agent> ready', flush=True); "
            "io=TerminalIO(); "
            "\ntry: io.input('en')\n"
            "except KeyboardInterrupt: print('Agent> cancelled', flush=True)\n"
            "print('User> ', end='', flush=True)"
        )
        command = (
            sys.executable,
            "-m",
            "tests.agent_live.container_pty_bridge",
            "--cwd",
            str(Path.cwd()),
            "--",
            sys.executable,
            "-c",
            program,
        )
        transport = ContainerPtyBridgeTransport(command, cwd=Path.cwd())
        transport.start(env=dict(os.environ))
        process = transport._process
        try:
            transport.read_complete_agent_response(timeout_seconds=5)
            transport.send_interrupt()
            response = transport.read_complete_agent_response(timeout_seconds=5)
        finally:
            transport.close()

        self.assertIn("Agent> cancelled", response)
        self.assertIsNotNone(process)
        self.assertIsNotNone(process.poll())
        self.assertIsNotNone(transport.cleanup_receipt)
        self.assertTrue(bool(transport.cleanup_receipt["cleaned"]))

    def test_container_bridge_drains_large_stderr_and_reads_fragmented_frame(self) -> None:
        program = (
            "import json,sys,time\n"
            "for line in sys.stdin:\n"
            " request=json.loads(line)\n"
            " sys.stderr.write('x'*200000); sys.stderr.flush()\n"
            " payload=json.dumps({'ok': True, 'response': 'Agent> complete'})+'\\n'\n"
            " sys.stdout.write(payload[:7]); sys.stdout.flush(); time.sleep(0.05)\n"
            " sys.stdout.write(payload[7:]); sys.stdout.flush()\n"
            " if request.get('op') == 'close': break\n"
        )
        transport = ContainerPtyBridgeTransport(
            (sys.executable, "-c", program),
            cwd=Path.cwd(),
            poll_interval_seconds=0.01,
        )
        transport.start(env=dict(os.environ))
        try:
            response = transport.read_complete_agent_response(timeout_seconds=3)
        finally:
            transport.close()

        self.assertEqual(response, "Agent> complete")
        self.assertLessEqual(len(transport._stderr_buffer), transport._stderr_cap_bytes)


if __name__ == "__main__":
    unittest.main()

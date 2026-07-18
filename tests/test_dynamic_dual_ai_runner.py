"""Contract tests for the response-driven PTY chaos runner (no real API)."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Mapping

from tests.agent_live.chaos_scheduler import build_chaos_schedule
from tests.agent_live.coverage_evidence import (
    RuntimeTurnEvent,
    validate_pty_cli_evidence_artifact,
    verify_runtime_postcondition,
)
from tests.agent_live.dynamic_dual_ai_chaos import (
    ChaosRunConfig,
    DynamicDualAiChaosRunner,
    SimulatorContext,
    SimulatorDecision,
    _complete_agent_response,
    encode_bracketed_paste,
)


EDGE = {
    "edge_key": "opening::opening_next_action::fake_node::option",
    "contract_hash": "contract-hash",
    "contract_variant_hash": "variant-hash",
    "question_id": "opening_next_action",
    "applicable": True,
    "real_execution_required": False,
    "evidence": {
        "dynamic_dual_ai": {
            "required": True,
            "applicability_reason": "response-driven semantic test fixture",
        }
    },
}
REVISION = {"commit": "test-commit", "worktree_hash": "a" * 64}


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


class FakeEventStream:
    def __init__(self, events: list[RuntimeTurnEvent]) -> None:
        self.events = list(events)

    def baseline(self) -> RuntimeTurnEvent:
        return self.events.pop(0)

    def next_event(self, *, timeout_seconds: float) -> RuntimeTurnEvent:
        del timeout_seconds
        if not self.events:
            raise RuntimeError("CLI returned without a new committed runtime turn event")
        return self.events.pop(0)


class OrderedClock:
    def __init__(self) -> None:
        self.value = 100

    def __call__(self) -> int:
        self.value += 10
        return self.value


class DynamicDualAiRunnerTest(unittest.TestCase):
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
    ) -> RuntimeTurnEvent:
        return RuntimeTurnEvent(
            schema_version=2,
            event_type="turn_committed",
            thread_id="contract-session",
            session_purpose="dynamic-dual-ai-chaos",
            before_fingerprint=before,
            after_fingerprint=after,
            turn_index=turn_index,
            active_group="opening",
            pending_question_id=pending,
            action_queue_types=(),
            pending_contract={"id": pending} if pending else {},
            revision=REVISION,
            admitted_action_types=("choose_target_mode",) if turn_index > 1 else (),
            state_diff_hashes=(
                {"target_mode": {"before": "", "after": "f" * 64}}
                if turn_index > 1 else {}
            ),
            next_result={"kind": "question", "question_id": pending},
        )

    def test_each_turn_is_selected_from_schedule_after_full_previous_response(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ledger = self._ledger()
            transport = FakeTransport([
                "Agent> Model config: provider=deepseek, model=deepseek-chat, auth=api_key\n"
                "Agent> Choose a mode.\n1. fake-node\n2. real-node",
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
                    rationale="The immediately preceding response offered fake-node.",
                    target_coverage_ids=(context.scheduled_target.edge_key,),
                )

            runner = DynamicDualAiChaosRunner(
                ChaosRunConfig.linux(root, session_id="contract-session"),
                simulator,
                ledger=ledger,
                schedule=self._schedule(ledger),
                transport=transport,
                event_stream=FakeEventStream([
                    self._event(1, "a" * 64, "b" * 64, "opening_next_action"),
                    self._event(2, "b" * 64, "c" * 64, "chain_select"),
                ]),
                revision=REVISION,
                clock_ns=OrderedClock(),
            )

            result = runner.run()

            self.assertEqual(result.execution_status, "complete")
            self.assertEqual(transport.submitted, ["I only want a safe dry run first."])
            self.assertEqual(len(seen), 1)
            self.assertTrue(transport.closed)
            schedule_result = json.loads(result.schedule_result_path.read_text(encoding="utf-8"))
            self.assertEqual(schedule_result["execution_status"], "complete")
            self.assertEqual(schedule_result["passed_target_count"], 1)
            artifact = json.loads(result.evidence_paths[0].read_text(encoding="utf-8"))
            valid, reason = validate_pty_cli_evidence_artifact(
                artifact,
                edge=EDGE,
                revision=REVISION,
            )
            self.assertTrue(valid, reason)
            self.assertEqual(artifact["turn_observation"]["seed"], 17)
            self.assertEqual(
                artifact["turn_observation"]["verified_postcondition"]["verifier_id"],
                "anychain.runtime-transition-proof.v1",
            )

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
                event_stream=FakeEventStream([]),
                revision=REVISION,
            )
            with self.assertRaisesRegex(RuntimeError, "provider/model"):
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
                event_stream=FakeEventStream([stale]),
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
                event_stream=FakeEventStream([
                    self._event(1, "a" * 64, "b" * 64, "opening_next_action"),
                ]),
                revision=REVISION,
            )
            with self.assertRaisesRegex(RuntimeError, "without a new committed"):
                runner.run()
            self.assertEqual(list((root / ".agent/dynamic-chaos/contract-session/evidence").glob("*.json")), [])

    def test_failed_postcondition_fails_closed_after_event_advancement(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ledger = self._ledger()
            transport = FakeTransport([
                "Agent> Model config: provider=deepseek, model=deepseek-chat, auth=api_key\nReady.",
                "Agent> Next question.",
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
                event_stream=FakeEventStream([
                    self._event(1, "a" * 64, "b" * 64, "opening_next_action"),
                    replace(
                        self._event(2, "b" * 64, "c" * 64, "chain_select"),
                        admitted_action_types=(),
                    ),
                ]),
                revision=REVISION,
            )
            with self.assertRaisesRegex(ValueError, "postcondition did not pass"):
                runner.run()
            result = json.loads(
                (root / ".agent/dynamic-chaos/contract-session/schedule-result.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(result["targets"][0]["status"], "failed")
            self.assertEqual(result["passed_target_count"], 0)

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
                event_stream=FakeEventStream([
                    self._event(1, "a" * 64, "b" * 64, "CLOUD_REGION"),
                    self._event(2, "b" * 64, "c" * 64, "chain_select"),
                ]),
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
        config = ChaosRunConfig.docker(root, session_id="isolated-session", service="bench")
        command_text = " ".join(config.command)
        self.assertIn("ANYCHAIN_AGENT_CHECKPOINT_PATH=/workspace/.agent/dynamic-chaos/isolated-session/checkpoints.sqlite", command_text)
        self.assertIn("ANYCHAIN_AGENT_JOBS_DIR=/workspace/.agent/dynamic-chaos/isolated-session/jobs", command_text)
        self.assertEqual(config.command[-2:], ("bench", "./bin/anychain-agent"))

    def test_bracketed_paste_preserves_multiline_unicode_as_one_submission(self) -> None:
        message = "请分析：\n```json\n{\"method\":\"eth_call\",\"params\":[]}\n```"
        encoded = encode_bracketed_paste(message)
        self.assertTrue(encoded.startswith(b"\x1b[200~"))
        self.assertTrue(encoded.endswith(b"\x1b[201~\r"))
        self.assertEqual(encoded.count(b"\x1b[200~"), 1)
        self.assertEqual(encoded.count(b"\x1b[201~"), 1)


if __name__ == "__main__":
    unittest.main()

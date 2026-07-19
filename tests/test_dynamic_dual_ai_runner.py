"""Contract tests for the response-driven PTY chaos runner (no real API)."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Mapping
from unittest.mock import patch

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
            self.assertEqual(artifact["turn_observation"]["seed"], 17)
            self.assertEqual(
                artifact["turn_observation"]["verified_postcondition"]["verifier_id"],
                "anychain.runtime-transition-proof.v1",
            )
            self.assertEqual(artifacts["real_cli"]["turn_observation"]["simulator_decision"], {})
            lane_evidence = schedule_result["targets"][0]["lane_evidence"]
            self.assertEqual(set(lane_evidence), {"dynamic_dual_ai", "real_cli"})

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
                event_stream=FakeEventStream([
                    self._event(1, "a" * 64, "b" * 64, "resume_harness_session"),
                    self._event(2, "b" * 64, "c" * 64, "opening_next_action"),
                    self._event(3, "c" * 64, "d" * 64, "chain_select"),
                ]),
                revision=REVISION,
                clock_ns=OrderedClock(),
            )

            with patch(
                "tests.agent_live.runtime_checkpoint.seed_runtime_checkpoint"
            ) as seed_checkpoint:
                result = runner.run()

            self.assertEqual(result.execution_status, "complete")
            self.assertEqual(transport.submitted, ["1", "Use the simulated node for this check."])
            self.assertEqual(len(seen), 1)
            seed_checkpoint.assert_called_once()

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
        self.assertEqual(config.command[-6:], (
            "bench",
            "./bin/anychain-agent",
            "--state-file",
            "/workspace/.agent/dynamic-chaos/isolated-session/terminal-session.json",
            "--language",
            "en",
        ))

    def test_runner_sets_independent_linux_terminal_state(self) -> None:
        root = Path("/tmp/anychain-chaos-contract")
        config = ChaosRunConfig.linux(root, session_id="isolated-session")

        self.assertEqual(config.command[-4:], (
            "--state-file",
            "/tmp/anychain-chaos-contract/.agent/dynamic-chaos/isolated-session/terminal-session.json",
            "--language",
            "en",
        ))

    def test_bracketed_paste_preserves_multiline_unicode_as_one_submission(self) -> None:
        message = "请分析：\n```json\n{\"method\":\"eth_call\",\"params\":[]}\n```"
        encoded = encode_bracketed_paste(message)
        self.assertTrue(encoded.startswith(b"\x1b[200~"))
        self.assertTrue(encoded.endswith(b"\x1b[201~\r"))
        self.assertEqual(encoded.count(b"\x1b[200~"), 1)
        self.assertEqual(encoded.count(b"\x1b[201~"), 1)


if __name__ == "__main__":
    unittest.main()

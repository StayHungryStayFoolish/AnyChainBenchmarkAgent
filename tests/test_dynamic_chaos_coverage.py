"""Tests for auditable real-CLI and response-driven dual-AI evidence."""

from __future__ import annotations

import unittest

from tests.agent_live.coverage_evidence import (
    COMPILED_GRAPH_RUNNER,
    PTY_DYNAMIC_DUAL_AI_RUNNER,
    PTY_REAL_CLI_RUNNER,
    DynamicTurnSelection,
    PtyCliTurnRecord,
    RuntimeTurnEvent,
    TurnObservation,
    VerifiedPostcondition,
    build_evidence_artifact,
    build_pty_cli_evidence_artifact,
    content_hash,
    pty_transcript_hash,
    validate_pty_cli_evidence_artifact,
)
from tests.agent_live.generate_harness_coverage_ledger import contract_variant_hash


class DynamicChaosCoverageTest(unittest.TestCase):
    def setUp(self) -> None:
        pending_contract = {"id": "opening_next_action"}
        self.edge = {
            "edge_key": "opening::opening_next_action::start_fake::option",
            "edge_type": "question_option",
            "question_id": "opening_next_action",
            "contract_hash": contract_variant_hash(pending_contract),
            "contract_variant_hash": "variant",
            "evidence": {
                "real_cli": {"required": True, "applicability_reason": "PTY contract"},
                "dynamic_dual_ai": {"required": True, "applicability_reason": "semantic contract"},
                "real_execution": {"required": False, "applicability_reason": "no side effect"},
            },
        }
        self.revision = {"commit": "commit", "worktree_hash": "a" * 64}

    def _turn(self) -> PtyCliTurnRecord:
        transcript = {
            "session_id": "session-1",
            "turn_index": 2,
            "previous_agent_response": "What would you like to test?",
            "user_message": "I want to try fake-node first.",
            "agent_response": "Which chain do you want to test?",
        }
        return PtyCliTurnRecord(
            **transcript,
            provider="deepseek",
            model="deepseek-chat",
            before_fingerprint="b" * 64,
            after_fingerprint="c" * 64,
            transcript_hash=pty_transcript_hash(**transcript),
            previous_response_received_at_ns=100,
            user_message_submitted_at_ns=120,
            agent_response_received_at_ns=160,
        )

    def _selection(self, **changes: object) -> DynamicTurnSelection:
        values: dict[str, object] = {
            "persona": "impatient first-time operator",
            "goal": "start a low-risk closed-loop run",
            "selected_message": self._turn().user_message,
            "rationale": "The prior response asked for a goal, so choose a mode indirectly.",
            "target_coverage_ids": (self.edge["edge_key"], "persona:impatient"),
            "selected_at_ns": 110,
        }
        values.update(changes)
        return DynamicTurnSelection(**values)  # type: ignore[arg-type]

    def _event(self, *, turn_index: int, before: str, after: str, pending: str) -> RuntimeTurnEvent:
        return RuntimeTurnEvent(
            schema_version=2,
            event_type="turn_committed",
            thread_id="session-1",
            session_purpose="dynamic-dual-ai-chaos",
            before_fingerprint=before,
            after_fingerprint=after,
            turn_index=turn_index,
            active_group="opening",
            pending_question_id=pending,
            action_queue_types=(),
            pending_contract={"id": pending} if pending else {},
            revision=self.revision,
        )

    def _observation(self, *, dynamic: bool, **changes: object) -> TurnObservation:
        baseline = self._event(
            turn_index=1,
            before="a" * 64,
            after="b" * 64,
            pending="opening_next_action",
        )
        committed = self._event(
            turn_index=2,
            before="b" * 64,
            after="c" * 64,
            pending="chain_select",
        )
        values: dict[str, object] = {
            "seed": 17,
            "revision": self.revision,
            "target_edge_key": self.edge["edge_key"],
            "target_contract_hash": self.edge["contract_hash"],
            "target_variant_hash": self.edge["contract_variant_hash"],
            "prior_agent_response": self._turn().previous_agent_response,
            "simulator_decision": ({
                "persona": self._selection().persona,
                "goal": self._selection().goal,
                "selected_message": self._selection().selected_message,
                "rationale": self._selection().rationale,
                "target_coverage_ids": list(self._selection().target_coverage_ids),
                "selected_at_ns": self._selection().selected_at_ns,
                "simulator": "codex",
                "selection_mode": "response_driven",
            } if dynamic else {}),
            "exact_user_turn": self._turn().user_message,
            "provider": "deepseek",
            "model": "deepseek-chat",
            "before_turn_index": 1,
            "after_turn_index": 2,
            "before_state_fingerprint": "b" * 64,
            "after_state_fingerprint": "c" * 64,
            "pending_contract": dict(baseline.pending_contract),
            "runtime_events": (baseline, committed),
            "verified_postcondition": VerifiedPostcondition(
                verifier_id="tests.checkpoint-verifier",
                passed=True,
                observed_coverage_ids=(self.edge["edge_key"], "group:opening"),
                admitted_typed_actions=("choose_target_mode",),
                state_diff={"target_mode": {"before": None, "after": "fake"}},
                next_question_or_result={"question_id": "chain_select"},
                details={"target_mode": "fake"},
            ),
        }
        values.update(changes)
        return TurnObservation(**values)  # type: ignore[arg-type]

    def test_real_cli_and_dynamic_dual_ai_have_distinct_runner_types(self) -> None:
        real = build_pty_cli_evidence_artifact(
            edge=self.edge,
            evidence_class="real_cli",
            revision=self.revision,
            turn=self._turn(),
            observation=self._observation(dynamic=False),
        )
        dynamic = build_pty_cli_evidence_artifact(
            edge=self.edge,
            evidence_class="dynamic_dual_ai",
            revision=self.revision,
            turn=self._turn(),
            observation=self._observation(dynamic=True),
            dynamic_selection=self._selection(),
        )

        self.assertEqual(real["runner_type"], PTY_REAL_CLI_RUNNER)
        self.assertEqual(dynamic["runner_type"], PTY_DYNAMIC_DUAL_AI_RUNNER)
        self.assertNotEqual(PTY_REAL_CLI_RUNNER, PTY_DYNAMIC_DUAL_AI_RUNNER)
        self.assertEqual(
            dynamic["dynamic_selection"]["previous_response_hash"],
            content_hash(self._turn().previous_agent_response),
        )
        self.assertEqual(dynamic["turn"]["provider"], "deepseek")
        self.assertEqual(dynamic["turn"]["model"], "deepseek-chat")
        self.assertEqual(dynamic["turn"]["before_fingerprint"], "b" * 64)
        self.assertEqual(dynamic["turn"]["after_fingerprint"], "c" * 64)

        for artifact in (real, dynamic):
            valid, reason = validate_pty_cli_evidence_artifact(
                artifact,
                edge=self.edge,
                revision=self.revision,
            )
            self.assertTrue(valid, reason)

    def test_fixed_scripted_prompt_cannot_be_labeled_dynamic(self) -> None:
        scripted = self._selection(selection_mode="scripted")
        with self.assertRaisesRegex(ValueError, "scripted prompts cannot be labeled"):
            build_pty_cli_evidence_artifact(
                edge=self.edge,
                evidence_class="dynamic_dual_ai",
                revision=self.revision,
                turn=self._turn(),
                observation=self._observation(dynamic=True),
                dynamic_selection=scripted,
            )

    def test_compiled_scripted_evidence_cannot_claim_dynamic_class(self) -> None:
        with self.assertRaisesRegex(ValueError, "only supports deterministic"):
            build_evidence_artifact(
                edge=self.edge,
                evidence_class="dynamic_dual_ai",
                scenario_id="fixed-prompts",
                runner_type=COMPILED_GRAPH_RUNNER,
                revision=self.revision,
                input_value="1",
                seed_state={},
                before_state={"last_user_input": "1"},
                after_state={},
                events=(),
                exit_status=0,
                outcome="passed",
            )

    def test_dynamic_selection_must_follow_response_and_match_pty_message(self) -> None:
        with self.assertRaisesRegex(ValueError, "selected after the previous response"):
            build_pty_cli_evidence_artifact(
                edge=self.edge,
                evidence_class="dynamic_dual_ai",
                revision=self.revision,
                turn=self._turn(),
                observation=self._observation(dynamic=True),
                dynamic_selection=self._selection(selected_at_ns=99),
            )
        with self.assertRaisesRegex(ValueError, "differs from the PTY user message"):
            build_pty_cli_evidence_artifact(
                edge=self.edge,
                evidence_class="dynamic_dual_ai",
                revision=self.revision,
                turn=self._turn(),
                observation=self._observation(dynamic=True),
                dynamic_selection=self._selection(selected_message="fixed prompt from a file"),
            )

    def test_tampering_with_audited_turn_is_rejected(self) -> None:
        artifact = build_pty_cli_evidence_artifact(
            edge=self.edge,
            evidence_class="dynamic_dual_ai",
            revision=self.revision,
            turn=self._turn(),
            observation=self._observation(dynamic=True),
            dynamic_selection=self._selection(),
        )
        artifact["turn"]["provider"] = "other"
        valid, reason = validate_pty_cli_evidence_artifact(
            artifact,
            edge=self.edge,
            revision=self.revision,
        )
        self.assertFalse(valid)
        self.assertIn("turn hash mismatch", reason)

    def test_transport_return_cannot_qualify_without_advanced_events_and_postcondition(self) -> None:
        stale = self._observation(
            dynamic=True,
            after_turn_index=1,
        )
        with self.assertRaisesRegex(ValueError, "complete event chain"):
            build_pty_cli_evidence_artifact(
                edge=self.edge,
                evidence_class="dynamic_dual_ai",
                revision=self.revision,
                turn=self._turn(),
                observation=stale,
                dynamic_selection=self._selection(),
            )

        failed_postcondition = self._observation(
            dynamic=True,
            verified_postcondition=VerifiedPostcondition(
                verifier_id="tests.checkpoint-verifier",
                passed=False,
                observed_coverage_ids=(self.edge["edge_key"],),
                admitted_typed_actions=("choose_target_mode",),
                state_diff={"target_mode": {"before": None, "after": "fake"}},
                next_question_or_result={"question_id": "chain_select"},
                details={"mismatch": True},
            ),
        )
        with self.assertRaisesRegex(ValueError, "postcondition did not pass"):
            build_pty_cli_evidence_artifact(
                edge=self.edge,
                evidence_class="dynamic_dual_ai",
                revision=self.revision,
                turn=self._turn(),
                observation=failed_postcondition,
                dynamic_selection=self._selection(),
            )

    def test_real_execution_requires_hashed_job_artifact(self) -> None:
        real_edge = {
            **self.edge,
            "real_execution_required": True,
            "evidence": {
                **self.edge["evidence"],
                "real_execution": {"required": True, "applicability_reason": "side effect"},
            },
        }
        with self.assertRaisesRegex(ValueError, "no verified job artifact"):
            build_pty_cli_evidence_artifact(
                edge=real_edge,
                evidence_class="dynamic_dual_ai",
                revision=self.revision,
                turn=self._turn(),
                observation=self._observation(dynamic=True),
                dynamic_selection=self._selection(),
            )


if __name__ == "__main__":
    unittest.main()

"""Contracts for reviewed execution cases and checkpoint provenance."""

from __future__ import annotations

import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

from tests.agent_live.coverage_evidence import (
    PtyCliTurnRecord,
    RuntimeTurnEvent,
    TurnObservation,
    VerifiedPostcondition,
    _validate_real_cli_provenance,
    content_hash,
    pty_transcript_hash,
)
from tests.agent_live.generate_harness_coverage_ledger import build_ledger
from tests.agent_live.harness_contract_scenarios import canonical_scenario_state
from tests.agent_live.reviewed_execution_cases import (
    REACHABLE_RPC_URL_ENV,
    reviewed_execution_case,
)
from tests.agent_live.runtime_checkpoint import reviewed_scenario, seed_runtime_checkpoint


class ReviewedExecutionCaseTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ledger = build_ledger(
            revision={"commit": "test", "worktree_hash": "a" * 64}
        )

    def test_every_fixed_real_cli_row_has_one_non_null_reviewed_case(self) -> None:
        rows = [
            edge for edge in self.ledger["edges"]
            if edge["evidence"]["real_cli"]["required"]
            and not edge["evidence"]["dynamic_dual_ai"]["required"]
        ]
        self.assertTrue(rows)
        for edge in rows:
            self.assertEqual(len(edge["execution_case_ids"]), 1, edge["edge_key"])
            resolved = reviewed_execution_case(edge)
            self.assertIsNotNone(resolved, edge["edge_key"])
            _scenario, case = resolved  # type: ignore[misc]
            self.assertIs(type(case.expected_admitted), bool)
            self.assertEqual(case.descriptor_hash, edge["execution_case_hash"])

    def test_canonical_scenario_hash_excludes_only_runtime_identity(self) -> None:
        scenario = reviewed_scenario("runtime_real_node_smoke")
        first = deepcopy(dict(scenario.seed_state))
        second = deepcopy(first)
        second["session"]["created_at"] = "2099-01-01T00:00:00Z"
        second["session"]["updated_at"] = "2099-01-01T00:00:01Z"
        second["pending_question"]["execution_request_id"] = "another-runtime-id"
        self.assertEqual(
            content_hash(canonical_scenario_state(first)),
            content_hash(canonical_scenario_state(second)),
        )
        second["target_mode"] = "fake-node"
        self.assertNotEqual(
            content_hash(canonical_scenario_state(first)),
            content_hash(canonical_scenario_state(second)),
        )
    def test_unreviewed_invalid_cross_product_is_not_applicable(self) -> None:
        for question_id in ("chain", "case3_protocol_evidence"):
            edge = next(
                item for item in self.ledger["edges"]
                if item["question_id"] == question_id
                and item["input_class"] == "invalid_literal"
            )
            self.assertFalse(edge["applicable"])
            self.assertFalse(edge["evidence"]["real_cli"]["required"])
            self.assertEqual(edge["execution_case_ids"], [])

    def test_url_case_uses_controlled_environment_provider_without_storing_secret(self) -> None:
        edge = next(
            item for item in self.ledger["edges"]
            if item["question_id"] == "LOCAL_RPC_URL"
            and item["input_class"] == "reachable_url"
        )
        _scenario, case = reviewed_execution_case(edge)  # type: ignore[misc]
        endpoint = "http://local-jsonrpc:8545"
        self.assertEqual(case.resolve_input({REACHABLE_RPC_URL_ENV: endpoint}), endpoint)
        self.assertTrue(case.admits_recorded_input(endpoint))
        self.assertTrue(case.admits_recorded_input("http://another-local-service:9545"))
        self.assertFalse(case.admits_recorded_input("https://public.example/v1?key=secret"))
        self.assertFalse(case.admits_recorded_input("http://10.0.0.4:8545"))
        self.assertNotIn(endpoint, str(case.descriptor))
        with self.assertRaisesRegex(RuntimeError, REACHABLE_RPC_URL_ENV):
            case.resolve_input({})

        trimmed_edge = next(
            item for item in self.ledger["edges"]
            if item["question_id"] == "LOCAL_RPC_URL"
            and item["input_class"] == "trimmed_whitespace_punctuation"
        )
        _scenario, trimmed_case = reviewed_execution_case(trimmed_edge)  # type: ignore[misc]
        wrapped = f"  {endpoint},  "
        self.assertEqual(
            trimmed_case.resolve_input({REACHABLE_RPC_URL_ENV: endpoint}),
            wrapped,
        )
        self.assertTrue(trimmed_case.admits_recorded_input(wrapped))
        self.assertFalse(trimmed_case.admits_recorded_input(f"use {endpoint}"))

    def test_checkpoint_seed_receipt_binds_scenario_session_and_contract(self) -> None:
        scenario = reviewed_scenario("opening")
        with TemporaryDirectory() as tmpdir:
            receipt = seed_runtime_checkpoint(
                scenario.seed_state,
                checkpoint_path=Path(tmpdir) / "checkpoint.sqlite",
                session_id="receipt-session",
                session_purpose="real-cli-coverage",
                scenario_id=scenario.scenario_id,
                scenario_state_fingerprint=scenario.state_fingerprint,
            )
        self.assertEqual(receipt.scenario_id, "opening")
        self.assertEqual(receipt.session_id, "receipt-session")
        self.assertEqual(receipt.pending_question_id, scenario.question["id"])
        self.assertEqual(len(receipt.checkpoint_sha256), 64)
        self.assertEqual(len(receipt.receipt_hash), 64)

    def test_real_cli_provenance_rejects_wrong_case_scenario_and_receipt_hash(self) -> None:
        edge = next(
            item for item in self.ledger["edges"]
            if item["question_id"] == "opening_next_action"
            and item["input_class"] == "exact_option"
            and item["option_id"] == "1"
        )
        scenario, case = reviewed_execution_case(edge)  # type: ignore[misc]
        session_id = "provenance-session"
        revision = self.ledger["revision"]
        baseline = RuntimeTurnEvent(
            schema_version=2,
            event_type="startup_snapshot",
            thread_id=session_id,
            session_purpose="real-cli-coverage",
            before_fingerprint="a" * 64,
            after_fingerprint="b" * 64,
            turn_index=0,
            active_group="opening",
            pending_question_id=str(scenario.question["id"]),
            action_queue_types=(),
            pending_contract=dict(scenario.question),
            revision=revision,
        )
        committed = RuntimeTurnEvent(
            schema_version=2,
            event_type="turn_committed",
            thread_id=session_id,
            session_purpose="real-cli-coverage",
            before_fingerprint="b" * 64,
            after_fingerprint="c" * 64,
            turn_index=1,
            active_group="chain_identity",
            pending_question_id="chain",
            action_queue_types=(),
            pending_contract={"id": "chain"},
            revision=revision,
        )
        transcript = {
            "session_id": session_id,
            "turn_index": 1,
            "previous_agent_response": "Choose a target mode.",
            "user_message": case.resolve_input(),
            "agent_response": "Choose a chain.",
        }
        turn = PtyCliTurnRecord(
            **transcript,
            provider="deepseek",
            model="deepseek-chat",
            before_fingerprint="b" * 64,
            after_fingerprint="c" * 64,
            transcript_hash=pty_transcript_hash(**transcript),
            previous_response_received_at_ns=1,
            user_message_submitted_at_ns=2,
            agent_response_received_at_ns=3,
        )
        observation = TurnObservation(
            seed=0,
            revision=revision,
            target_edge_key=str(edge["edge_key"]),
            target_contract_hash=str(edge["contract_hash"]),
            target_variant_hash=str(edge["contract_variant_hash"]),
            prior_agent_response=turn.previous_agent_response,
            simulator_decision={},
            exact_user_turn=turn.user_message,
            provider=turn.provider,
            model=turn.model,
            before_turn_index=0,
            after_turn_index=1,
            before_state_fingerprint="b" * 64,
            after_state_fingerprint="c" * 64,
            pending_contract=dict(scenario.question),
            runtime_events=(baseline, committed),
            verified_postcondition=VerifiedPostcondition(
                verifier_id="tests.provenance",
                passed=True,
                observed_coverage_ids=(str(edge["edge_key"]),),
                admitted_typed_actions=("choose_target_mode",),
                state_diff={"target_mode": {"before": None, "after": "fake-node"}},
                next_question_or_result={"question_id": "chain"},
                details={"target_mode": "fake-node"},
            ),
        )
        receipt = {
            "scenario_id": scenario.scenario_id,
            "scenario_state_fingerprint": scenario.state_fingerprint,
            "seed_state_hash": content_hash(canonical_scenario_state(scenario.seed_state or {})),
            "projected_state_hash": "d" * 64,
            "checkpoint_sha256": "e" * 64,
            "checkpoint_path": "/tmp/provenance.sqlite",
            "session_id": session_id,
            "session_purpose": "real-cli-coverage",
            "pending_question_id": str(scenario.question["id"]),
            "pending_contract_hash": content_hash(scenario.question),
        }
        receipt["receipt_hash"] = content_hash(receipt)

        _validate_real_cli_provenance(
            edge=edge,
            turn=turn,
            observation=observation,
            execution_case=case.descriptor,
            seed_receipt=receipt,
        )
        with self.assertRaisesRegex(ValueError, "authoritative reviewed descriptor"):
            _validate_real_cli_provenance(
                edge=edge,
                turn=turn,
                observation=observation,
                execution_case={**case.descriptor, "case_id": "wrong"},
                seed_receipt=receipt,
            )
        wrong_scenario = {**receipt, "scenario_id": "wrong"}
        wrong_scenario.pop("receipt_hash")
        wrong_scenario["receipt_hash"] = content_hash(wrong_scenario)
        with self.assertRaisesRegex(ValueError, "does not match reviewed scenario"):
            _validate_real_cli_provenance(
                edge=edge,
                turn=turn,
                observation=observation,
                execution_case=case.descriptor,
                seed_receipt=wrong_scenario,
            )
        with self.assertRaisesRegex(ValueError, "self-hash"):
            _validate_real_cli_provenance(
                edge=edge,
                turn=turn,
                observation=observation,
                execution_case=case.descriptor,
                seed_receipt={**receipt, "receipt_hash": "f" * 64},
            )


if __name__ == "__main__":
    unittest.main()

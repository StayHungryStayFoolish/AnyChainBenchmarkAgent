"""Tests for auditable real-CLI and response-driven dual-AI evidence."""

from __future__ import annotations

import json
import concurrent.futures
import multiprocessing
import os
import shutil
import sys
import tempfile
import unittest
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

from tests.agent_live import coverage_evidence as coverage_evidence_module
from tests.agent_live.coverage_evidence import (
    COMPILED_GRAPH_RUNNER,
    PTY_DYNAMIC_DUAL_AI_RUNNER,
    PTY_REAL_CLI_RUNNER,
    DynamicTurnSelection,
    PtyDiagnosticRecord,
    PtyCliTurnRecord,
    RuntimeTurnEvent,
    TurnObservation,
    VerifiedPostcondition,
    admit_pty_artifact_bundle,
    admit_pty_artifact_pair,
    build_evidence_artifact,
    build_pty_authority_receipt,
    build_pty_diagnostic_artifact,
    build_pty_cli_evidence_artifact,
    content_hash,
    create_pty_authority_signer,
    load_pty_artifact_bundle,
    load_pty_authority_receipt,
    load_valid_evidence_reference,
    pty_artifact_bundle_digest,
    pty_transcript_hash,
    recover_pty_artifact_bundle,
    recover_pty_artifact_pair,
    remove_pty_artifact_bundle,
    remove_pty_artifact_pair,
    rollback_pty_artifact_bundle,
    validate_pty_artifact_bundle,
    validate_pty_authority_receipt,
    validate_pty_cli_candidate_artifact,
    validate_pty_cli_evidence_artifact,
    validate_pty_diagnostic_artifact,
    write_evidence_artifact,
)
from tests.agent_live.generate_harness_coverage_ledger import contract_variant_hash
from tests.agent_live.critical_sequences import (
    observed_coverage_turn_from_artifact,
)


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
        self.signer = create_pty_authority_signer()

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
        event = RuntimeTurnEvent(
            schema_version=6,
            event_type="turn_committed",
            observation="committed test turn",
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
            pending_transition={
                "before_hash": before,
                "after_hash": after,
            },
            runtime_event_id=f"runtime-{turn_index}",
            runtime_event_sequence=turn_index,
            terminal_event_id=f"terminal-{turn_index}",
            transaction_id=f"transaction-{turn_index}",
            terminal_outcome="committed",
            render_hash="e" * 64,
            base_revision=turn_index - 1,
            base_checkpoint_thread_id="physical-session-1",
            base_checkpoint_id=f"checkpoint-{turn_index - 1}",
            product_revision=turn_index,
            product_checkpoint_thread_id="physical-session-1",
            product_checkpoint_id=f"checkpoint-{turn_index}",
            product_authority_id="dynamic-dual-ai-chaos:session-1",
            physical_thread_id="physical-session-1",
            attempt_checkpoint_id=f"checkpoint-{turn_index}",
        )
        return self._seal_event(event)

    @staticmethod
    def _seal_event(event: RuntimeTurnEvent) -> RuntimeTurnEvent:
        unsigned = asdict(event)
        unsigned.pop("runtime_event_payload_hash", None)
        return replace(event, runtime_event_payload_hash=content_hash(unsigned))

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

    @staticmethod
    def _rehash_pty_artifact(artifact: dict[str, object]) -> None:
        artifact.pop("artifact_hash", None)
        artifact.pop("evidence_id", None)
        artifact["evidence_id"] = content_hash(artifact)
        artifact["artifact_hash"] = content_hash(artifact)

    @staticmethod
    def _rehash_diagnostic(artifact: dict[str, object]) -> None:
        artifact.pop("artifact_hash", None)
        artifact.pop("diagnostic_id", None)
        artifact["diagnostic_id"] = content_hash(artifact)
        artifact["artifact_hash"] = content_hash(artifact)

    def _bundle_candidate_paths(self, directory: str) -> tuple[Path, ...]:
        first = build_pty_cli_evidence_artifact(
            edge=self.edge,
            evidence_class="dynamic_dual_ai",
            revision=self.revision,
            turn=self._turn(),
            observation=self._observation(dynamic=True),
            dynamic_selection=self._selection(),
        )
        second = deepcopy(first)
        second["created_at"] = "2099-01-01T00:00:00Z"
        self._rehash_pty_artifact(second)
        return (
            write_evidence_artifact(first, directory),
            write_evidence_artifact(second, directory),
        )

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
            valid, reason = validate_pty_cli_candidate_artifact(
                artifact,
                edge=self.edge,
                revision=self.revision,
            )
            self.assertTrue(valid, reason)

    def test_sensitive_pending_projects_short_secret_and_reference_capability(
        self,
    ) -> None:
        pending = {"id": "RPC_API_KEY", "sensitive_input": True}
        edge = {
            **self.edge,
            "edge_key": "chain_auxiliary_endpoints::RPC_API_KEY::manual",
            "edge_type": "manual_input",
            "question_id": "RPC_API_KEY",
            "contract_hash": contract_variant_hash(pending),
        }
        raw_secret = "short-secret"
        base_turn = self._turn()
        turn = replace(
            base_turn,
            user_message=raw_secret,
            agent_response=f"rejected {raw_secret} for this field",
            transcript_hash=pty_transcript_hash(
                session_id=base_turn.session_id,
                turn_index=base_turn.turn_index,
                previous_agent_response=base_turn.previous_agent_response,
                user_message=raw_secret,
                agent_response=f"rejected {raw_secret} for this field",
            ),
        )
        selection = replace(
            self._selection(),
            selected_message=raw_secret,
            target_coverage_ids=(edge["edge_key"],),
        )
        baseline = self._seal_event(replace(
            self._event(
                turn_index=1,
                before="a" * 64,
                after="b" * 64,
                pending="RPC_API_KEY",
            ),
            pending_contract=pending,
        ))
        committed = self._event(
            turn_index=2,
            before="b" * 64,
            after="c" * 64,
            pending="chain_select",
        )
        reference = "semantic-secret:opaque-capability"
        observation = TurnObservation(
            seed=17,
            revision=self.revision,
            target_edge_key=edge["edge_key"],
            target_contract_hash=edge["contract_hash"],
            target_variant_hash=edge["contract_variant_hash"],
            prior_agent_response=turn.previous_agent_response,
            simulator_decision={
                "persona": selection.persona,
                "goal": selection.goal,
                "selected_message": raw_secret,
                "rationale": selection.rationale,
                "target_coverage_ids": [edge["edge_key"]],
                "selected_at_ns": selection.selected_at_ns,
                "simulator": selection.simulator,
                "selection_mode": selection.selection_mode,
            },
            exact_user_turn=raw_secret,
            provider=turn.provider,
            model=turn.model,
            before_turn_index=1,
            after_turn_index=2,
            before_state_fingerprint="b" * 64,
            after_state_fingerprint="c" * 64,
            pending_contract=pending,
            runtime_events=(baseline, committed),
            verified_postcondition=VerifiedPostcondition(
                verifier_id="tests.secret-projection",
                passed=True,
                observed_coverage_ids=(edge["edge_key"],),
                admitted_typed_actions=("answer_pending",),
                state_diff={
                    "confirmed_config.RPC_API_KEY": {
                        "after": reference,
                    },
                },
                next_question_or_result={"question_id": "chain_select"},
                details={
                    "error": f"invalid value: {raw_secret}",
                    raw_secret: "tainted mapping key",
                    "secret_bindings": [{
                        "reference": reference,
                        "scope_id": "turn:1",
                        "atom_id": "input-secret-1",
                        "value_hash": "a" * 64,
                    }],
                },
            ),
        )

        artifact = build_pty_cli_evidence_artifact(
            edge=edge,
            evidence_class="dynamic_dual_ai",
            revision=self.revision,
            turn=turn,
            observation=observation,
            dynamic_selection=selection,
        )
        serialized = json.dumps(artifact, ensure_ascii=False)

        self.assertNotIn(raw_secret, serialized)
        self.assertNotIn(reference, serialized)
        self.assertNotEqual(
            artifact["turn"]["source_user_message_hash"],
            content_hash(raw_secret),
        )
        self.assertEqual(
            artifact["turn"]["user_message"],
            "***SENSITIVE_INPUT***",
        )
        valid, reason = validate_pty_cli_candidate_artifact(
            artifact,
            edge=edge,
            revision=self.revision,
        )
        self.assertTrue(valid, reason)

    def test_sensitive_diagnostic_projects_echo_and_reference_capability(
        self,
    ) -> None:
        raw_secret = "short-secret"
        reference = "semantic-secret:diagnostic-capability"
        base_turn = self._turn()
        response = f"Accepted credential {raw_secret}"
        turn = replace(
            base_turn,
            user_message=raw_secret,
            agent_response=response,
            transcript_hash=pty_transcript_hash(
                session_id=base_turn.session_id,
                turn_index=base_turn.turn_index,
                previous_agent_response=base_turn.previous_agent_response,
                user_message=raw_secret,
                agent_response=response,
            ),
        )
        sensitive_baseline = self._seal_event(replace(
            self._event(
                turn_index=1,
                before="a" * 64,
                after="b" * 64,
                pending="RPC_API_KEY",
            ),
            pending_contract={
                "id": "RPC_API_KEY",
                "sensitive_input": True,
            },
        ))
        record = PtyDiagnosticRecord(
            diagnostic_kind="failed_attempt",
            verification_status="postcondition_failed",
            target_id="sensitive-diagnostic",
            target_edge_key=self.edge["edge_key"],
            revision=self.revision,
            session_id=turn.session_id,
            reason=f"invalid value: {raw_secret}",
            last_complete_response=response,
            input_baseline_event=sensitive_baseline,
            last_complete_event=self._event(
                turn_index=2,
                before="b" * 64,
                after="c" * 64,
                pending="chain_select",
            ),
            completed_turn=turn,
            dynamic_selection=replace(
                self._selection(),
                selected_message=raw_secret,
            ),
            verified_postcondition=VerifiedPostcondition(
                verifier_id="tests.secret-diagnostic",
                passed=False,
                observed_coverage_ids=(self.edge["edge_key"],),
                admitted_typed_actions=(),
                state_diff={"target_mode": {"after": "fake"}},
                next_question_or_result={},
                details={
                    "error": f"rejected {raw_secret}",
                    "secret_binding": {
                        "reference": reference,
                        "scope_id": "turn:1",
                        "atom_id": "input-secret-1",
                        "value_hash": "a" * 64,
                    },
                },
            ),
        )

        artifact = build_pty_diagnostic_artifact(record)
        serialized = json.dumps(artifact, ensure_ascii=False)

        self.assertNotIn(raw_secret, serialized)
        self.assertNotIn(reference, serialized)
        self.assertTrue(
            artifact["last_complete_boundary"]["input_sensitive"]
        )
        valid, reason = validate_pty_diagnostic_artifact(
            artifact,
            authority=build_pty_authority_receipt(artifact, signer=self.signer),
            trusted_public_key_b64=self.signer.public_key_b64,
        )
        self.assertTrue(valid, reason)

    def test_continuation_uses_its_own_sensitive_pending_contract(self) -> None:
        raw_secret = "continuation-secret"
        root_turn = self._turn()
        selection = self._selection()
        baseline = self._event(
            turn_index=1,
            before="a" * 64,
            after="b" * 64,
            pending="opening_next_action",
        )
        committed = self._seal_event(replace(
            self._event(
                turn_index=2,
                before="b" * 64,
                after="c" * 64,
                pending="RPC_API_KEY",
            ),
            pending_contract={
                "id": "RPC_API_KEY",
                "sensitive_input": True,
            },
        ))
        continuation_event = self._event(
            turn_index=3,
            before="c" * 64,
            after="d" * 64,
            pending="chain_select",
        )
        continuation_transcript = {
            "session_id": root_turn.session_id,
            "turn_index": 3,
            "previous_agent_response": root_turn.agent_response,
            "user_message": raw_secret,
            "agent_response": f"Accepted {raw_secret}",
        }
        continuation_turn = PtyCliTurnRecord(
            **continuation_transcript,
            provider=root_turn.provider,
            model=root_turn.model,
            before_fingerprint="c" * 64,
            after_fingerprint="d" * 64,
            transcript_hash=pty_transcript_hash(**continuation_transcript),
            previous_response_received_at_ns=170,
            user_message_submitted_at_ns=180,
            agent_response_received_at_ns=200,
        )
        continuation_selection = {
            "persona": selection.persona,
            "goal": selection.goal,
            "selected_message": raw_secret,
            "rationale": "Answer the sensitive follow-up.",
            "target_coverage_ids": list(selection.target_coverage_ids),
            "selected_at_ns": 175,
            "simulator": "codex",
            "selection_mode": "response_driven",
        }
        observation = TurnObservation(
            seed=17,
            revision=self.revision,
            target_edge_key=self.edge["edge_key"],
            target_contract_hash=self.edge["contract_hash"],
            target_variant_hash=self.edge["contract_variant_hash"],
            prior_agent_response=root_turn.previous_agent_response,
            simulator_decision={
                **selection.__dict__,
                "target_coverage_ids": list(selection.target_coverage_ids),
            },
            exact_user_turn=root_turn.user_message,
            provider=root_turn.provider,
            model=root_turn.model,
            before_turn_index=1,
            after_turn_index=3,
            before_state_fingerprint="b" * 64,
            after_state_fingerprint="d" * 64,
            pending_contract=baseline.pending_contract,
            runtime_events=(baseline, committed, continuation_event),
            verified_postcondition=VerifiedPostcondition(
                verifier_id="tests.sensitive-continuation",
                passed=True,
                observed_coverage_ids=(self.edge["edge_key"],),
                admitted_typed_actions=("choose_target_mode",),
                state_diff={"target_mode": {"after": "fake"}},
                next_question_or_result={"question_id": "chain_select"},
                details={"target_mode": "fake"},
            ),
            continuation_turns=(continuation_turn,),
            continuation_simulator_decisions=(continuation_selection,),
        )

        artifact = build_pty_cli_evidence_artifact(
            edge=self.edge,
            evidence_class="dynamic_dual_ai",
            revision=self.revision,
            turn=root_turn,
            observation=observation,
            dynamic_selection=selection,
        )
        serialized = json.dumps(artifact, ensure_ascii=False)

        self.assertNotIn(raw_secret, serialized)
        self.assertEqual(
            artifact["turn_observation"]["continuation_turns"][0][
                "user_message"
            ],
            "***SENSITIVE_INPUT***",
        )
        valid, reason = validate_pty_cli_candidate_artifact(
            artifact,
            edge=self.edge,
            revision=self.revision,
        )
        self.assertTrue(valid, reason)

        authority = build_pty_authority_receipt(artifact, signer=self.signer)
        forged_observation = artifact["turn_observation"]
        forged_preceding = forged_observation["runtime_events"][1]
        forged_preceding["pending_contract"] = {"id": "RPC_API_KEY"}
        forged_preceding_unsigned = dict(forged_preceding)
        forged_preceding_unsigned.pop("runtime_event_payload_hash", None)
        forged_preceding["runtime_event_payload_hash"] = content_hash(
            forged_preceding_unsigned
        )
        forged_turn = forged_observation["continuation_turns"][0]
        forged_turn["user_message"] = raw_secret
        forged_turn["agent_response"] = f"Accepted {raw_secret}"
        forged_turn["transcript_hash"] = pty_transcript_hash(
            session_id=forged_turn["session_id"],
            turn_index=forged_turn["turn_index"],
            previous_agent_response=forged_turn["previous_agent_response"],
            user_message=forged_turn["user_message"],
            agent_response=forged_turn["agent_response"],
        )
        forged_observation["continuation_simulator_decisions"][0][
            "selected_message"
        ] = raw_secret
        artifact["turn_observation_hash"] = content_hash(forged_observation)
        self._rehash_pty_artifact(artifact)
        valid, reason = validate_pty_authority_receipt(
            artifact,
            authority,
            trusted_public_key_b64=self.signer.public_key_b64,
        )
        self.assertFalse(valid)
        self.assertIn("authority receipt", reason)

    def test_validator_rejects_rehashed_forged_source_hash(self) -> None:
        turn = self._turn()
        selection = self._selection()
        artifact = build_pty_cli_evidence_artifact(
            edge=self.edge,
            evidence_class="dynamic_dual_ai",
            revision=self.revision,
            turn=turn,
            observation=self._observation(dynamic=True),
            dynamic_selection=selection,
        )
        authority = build_pty_authority_receipt(artifact, signer=self.signer)
        artifact["turn"]["source_user_message_hash"] = "f" * 64
        artifact["turn_hash"] = content_hash(artifact["turn"])
        self._rehash_pty_artifact(artifact)

        valid, reason = validate_pty_cli_evidence_artifact(
            artifact,
            edge=self.edge,
            revision=self.revision,
            authority=authority,
            trusted_public_key_b64=self.signer.public_key_b64,
        )
        self.assertFalse(valid)
        self.assertIn("authority receipt", reason)

    def test_worker_cannot_reissue_authority_with_an_untrusted_key(self) -> None:
        artifact = build_pty_cli_evidence_artifact(
            edge=self.edge,
            evidence_class="dynamic_dual_ai",
            revision=self.revision,
            turn=self._turn(),
            observation=self._observation(dynamic=True),
            dynamic_selection=self._selection(),
        )
        attacker = create_pty_authority_signer()
        artifact["turn"]["agent_response"] = "forged response"
        artifact["turn_hash"] = content_hash(artifact["turn"])
        self._rehash_pty_artifact(artifact)
        valid, reason = validate_pty_authority_receipt(
            artifact,
            build_pty_authority_receipt(
                artifact,
                signer=attacker,
            ),
            trusted_public_key_b64=self.signer.public_key_b64,
        )
        self.assertFalse(valid)
        self.assertIn("signed authority receipt", reason)

    def test_pair_admission_failure_never_leaves_qualifying_receipt(self) -> None:
        artifact = build_pty_cli_evidence_artifact(
            edge=self.edge,
            evidence_class="dynamic_dual_ai",
            revision=self.revision,
            turn=self._turn(),
            observation=self._observation(dynamic=True),
            dynamic_selection=self._selection(),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = write_evidence_artifact(artifact, directory)
            with patch(
                "tests.agent_live.coverage_evidence._rename_directory_noreplace",
                side_effect=OSError("injected authority publish failure"),
            ):
                with self.assertRaisesRegex(OSError, "injected"):
                    admit_pty_artifact_pair(path, signer=self.signer)
            self.assertFalse(path.with_suffix(".admitted").exists())
            with self.assertRaisesRegex(ValueError, "admitted bundle"):
                load_pty_authority_receipt(path)

    def test_pair_lifecycle_removes_committed_and_orphaned_state_idempotently(
        self,
    ) -> None:
        artifact = build_pty_cli_evidence_artifact(
            edge=self.edge,
            evidence_class="dynamic_dual_ai",
            revision=self.revision,
            turn=self._turn(),
            observation=self._observation(dynamic=True),
            dynamic_selection=self._selection(),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = write_evidence_artifact(artifact, directory)
            bundle = admit_pty_artifact_pair(path, signer=self.signer)
            self.assertTrue(bundle.is_dir())
            self.assertEqual(
                load_pty_authority_receipt(path)["artifact_hash"],
                artifact["artifact_hash"],
            )
            orphan = path.with_suffix(".admitted.tmp.orphan")
            orphan.mkdir()
            (orphan / "partial.json").write_text(
                "{}\n",
                encoding="utf-8",
            )

            remove_pty_artifact_pair(path)

            self.assertFalse(path.exists())
            self.assertFalse(bundle.exists())
            self.assertFalse(orphan.exists())
            remove_pty_artifact_pair(path)

    @unittest.skipUnless(
        sys.platform.startswith("linux"),
        "process-crash artifact recovery is Linux-only",
    )
    def test_pair_admission_recovers_staging_left_by_process_crash(self) -> None:
        artifact = build_pty_cli_evidence_artifact(
            edge=self.edge,
            evidence_class="dynamic_dual_ai",
            revision=self.revision,
            turn=self._turn(),
            observation=self._observation(dynamic=True),
            dynamic_selection=self._selection(),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = write_evidence_artifact(artifact, directory)

            def crash_before_publish() -> None:
                with patch(
                    "tests.agent_live.coverage_evidence."
                    "_rename_directory_noreplace",
                    side_effect=lambda *_args: os._exit(91),
                ):
                    admit_pty_artifact_pair(path, signer=self.signer)

            process = multiprocessing.get_context("fork").Process(
                target=crash_before_publish
            )
            process.start()
            process.join(timeout=5)
            self.assertFalse(process.is_alive())
            self.assertEqual(process.exitcode, 91)
            self.assertFalse(path.with_suffix(".admitted").exists())
            self.assertTrue(tuple(
                path.parent.glob(f"{path.stem}.admitted.tmp.*")
            ))

            bundle = admit_pty_artifact_pair(path, signer=self.signer)

            self.assertTrue(bundle.is_dir())
            self.assertFalse(tuple(
                path.parent.glob(f"{path.stem}.admitted.tmp.*")
            ))
            self.assertEqual(
                load_pty_authority_receipt(path)["artifact_hash"],
                artifact["artifact_hash"],
            )

    def test_pair_post_publish_fsync_failures_require_reconciliation(
        self,
    ) -> None:
        artifact = build_pty_cli_evidence_artifact(
            edge=self.edge,
            evidence_class="dynamic_dual_ai",
            revision=self.revision,
            turn=self._turn(),
            observation=self._observation(dynamic=True),
            dynamic_selection=self._selection(),
        )
        for failure_target in ("bundle", "parent"):
            with self.subTest(failure_target=failure_target):
                with tempfile.TemporaryDirectory() as directory:
                    path = write_evidence_artifact(artifact, directory)
                    persisted_artifact = json.loads(
                        path.read_text(encoding="utf-8")
                    )
                    bundle_path = path.with_suffix(".admitted")
                    original_fsync = coverage_evidence_module._fsync_directory

                    def fail_after_publish(candidate: object) -> None:
                        target = (
                            bundle_path
                            if failure_target == "bundle"
                            else path.parent
                        )
                        if bundle_path.exists() and candidate == target:
                            raise OSError(
                                "injected post-publish fsync failure"
                            )
                        original_fsync(candidate)  # type: ignore[arg-type]

                    with patch(
                        "tests.agent_live.coverage_evidence._fsync_directory",
                        side_effect=fail_after_publish,
                    ):
                        with self.assertRaisesRegex(
                            RuntimeError,
                            "durability is uncertain",
                        ):
                            admit_pty_artifact_pair(
                                path,
                                signer=self.signer,
                            )

                    self.assertEqual(
                        load_pty_authority_receipt(path)["artifact_hash"],
                        artifact["artifact_hash"],
                    )
                    self.assertEqual(
                        recover_pty_artifact_pair(
                            path,
                            expected_artifact=persisted_artifact,
                            signer=self.signer,
                        ),
                        bundle_path,
                    )

    @unittest.skipUnless(
        sys.platform.startswith("linux"),
        "atomic shard admission requires Linux renameat2",
    )
    def test_bundle_crashes_leave_only_non_qualifying_staging_then_retry(
        self,
    ) -> None:
        original_stage = coverage_evidence_module._stage_pty_artifact_bundle_item
        for crash_point in ("after_first", "before_second"):
            with self.subTest(crash_point=crash_point):
                with tempfile.TemporaryDirectory() as directory:
                    candidate_paths = self._bundle_candidate_paths(directory)
                    bundle_path = Path(directory) / "pty-shard.admitted"

                    def crash_during_staging() -> None:
                        def stage_or_crash(*args: object, **kwargs: object) -> object:
                            index = int(kwargs["index"])
                            if crash_point == "before_second" and index == 1:
                                os._exit(92)
                            result = original_stage(*args, **kwargs)
                            if crash_point == "after_first" and index == 0:
                                os._exit(91)
                            return result

                        with patch(
                            "tests.agent_live.coverage_evidence."
                            "_stage_pty_artifact_bundle_item",
                            side_effect=stage_or_crash,
                        ):
                            admit_pty_artifact_bundle(
                                bundle_path,
                                candidate_paths,
                                signer=self.signer,
                            )

                    process = multiprocessing.get_context("fork").Process(
                        target=crash_during_staging
                    )
                    process.start()
                    process.join(timeout=5)
                    self.assertFalse(process.is_alive())
                    self.assertIn(process.exitcode, {91, 92})
                    self.assertFalse(bundle_path.exists())
                    staging_paths = tuple(
                        bundle_path.parent.glob(f"{bundle_path.name}.tmp.*")
                    )
                    self.assertEqual(len(staging_paths), 1)
                    self.assertFalse(
                        (staging_paths[0] / "COMMIT.json").exists()
                    )
                    staged_items = tuple(
                        (staging_paths[0] / "items").iterdir()
                    )
                    self.assertEqual(len(staged_items), 1)
                    self.assertEqual(
                        {entry.name for entry in staged_items[0].iterdir()},
                        {"artifact.json", "authority.json", "COMMIT.json"},
                    )
                    with self.assertRaisesRegex(ValueError, "not committed"):
                        load_pty_artifact_bundle(bundle_path)

                    committed = admit_pty_artifact_bundle(
                        bundle_path,
                        candidate_paths,
                        signer=self.signer,
                    )

                    self.assertEqual(committed, bundle_path)
                    self.assertFalse(tuple(
                        bundle_path.parent.glob(f"{bundle_path.name}.tmp.*")
                    ))
                    items, digest = load_pty_artifact_bundle(bundle_path)
                    self.assertEqual(len(items), 2)
                    self.assertEqual(
                        digest,
                        pty_artifact_bundle_digest(bundle_path),
                    )
                    valid, reason = validate_pty_artifact_bundle(
                        bundle_path,
                        trusted_public_key_b64=self.signer.public_key_b64,
                        expected_digest=digest,
                    )
                    self.assertTrue(valid, reason)

    @unittest.skipUnless(
        sys.platform.startswith("linux"),
        "atomic shard admission requires Linux renameat2",
    )
    def test_bundle_rejects_reordered_or_deleted_items(self) -> None:
        for mutation in ("reorder", "delete"):
            with self.subTest(mutation=mutation):
                with tempfile.TemporaryDirectory() as directory:
                    candidate_paths = self._bundle_candidate_paths(directory)
                    bundle_path = Path(directory) / "pty-shard.admitted"
                    admit_pty_artifact_bundle(
                        bundle_path,
                        candidate_paths,
                        signer=self.signer,
                    )
                    bundle_path.chmod(0o700)
                    items_path = bundle_path / "items"
                    items_path.chmod(0o700)
                    if mutation == "reorder":
                        commit_path = bundle_path / "COMMIT.json"
                        commit_path.chmod(0o600)
                        commit = json.loads(
                            commit_path.read_text(encoding="utf-8")
                        )
                        commit["items"].reverse()
                        commit_path.write_text(
                            json.dumps(commit, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8",
                        )
                    else:
                        removed = sorted(items_path.iterdir())[0]
                        removed.chmod(0o700)
                        shutil.rmtree(removed)
                    for child in items_path.iterdir():
                        if child.is_dir():
                            child.chmod(0o500)
                            for member in child.iterdir():
                                member.chmod(0o400)
                    (bundle_path / "COMMIT.json").chmod(0o400)
                    items_path.chmod(0o500)
                    bundle_path.chmod(0o500)

                    valid, reason = validate_pty_artifact_bundle(
                        bundle_path,
                        trusted_public_key_b64=self.signer.public_key_b64,
                    )

                    self.assertFalse(valid)
                    self.assertIn("membership", reason)

    @unittest.skipUnless(
        sys.platform.startswith("linux"),
        "atomic shard admission requires Linux renameat2",
    )
    def test_bundle_lifecycle_loads_digest_rolls_back_and_removes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            candidate_paths = self._bundle_candidate_paths(directory)
            bundle_path = Path(directory) / "pty-shard.admitted"
            admit_pty_artifact_bundle(
                bundle_path,
                candidate_paths,
                signer=self.signer,
            )
            _items, digest = load_pty_artifact_bundle(bundle_path)

            self.assertEqual(
                recover_pty_artifact_bundle(
                    bundle_path,
                    expected_artifacts=[
                        json.loads(path.read_text(encoding="utf-8"))
                        for path in candidate_paths
                    ],
                    signer=self.signer,
                ),
                bundle_path,
            )
            valid, reason = validate_pty_artifact_bundle(
                bundle_path,
                trusted_public_key_b64=self.signer.public_key_b64,
                expected_digest=digest,
            )
            self.assertTrue(valid, reason)
            valid, reason = validate_pty_artifact_bundle(
                bundle_path,
                trusted_public_key_b64=self.signer.public_key_b64,
                expected_digest="0" * 64,
            )
            self.assertFalse(valid)
            self.assertIn("digest", reason)

            rollback_pty_artifact_bundle(bundle_path)
            self.assertFalse(bundle_path.exists())
            self.assertTrue(all(path.exists() for path in candidate_paths))
            admit_pty_artifact_bundle(
                bundle_path,
                candidate_paths,
                signer=self.signer,
            )
            remove_pty_artifact_bundle(bundle_path, candidate_paths)
            self.assertFalse(bundle_path.exists())
            self.assertTrue(all(not path.exists() for path in candidate_paths))
            remove_pty_artifact_bundle(bundle_path, candidate_paths)

    @unittest.skipUnless(
        sys.platform.startswith("linux"),
        "atomic shard admission requires Linux renameat2",
    )
    def test_bundle_post_publish_fsync_failure_requires_reconciliation(
        self,
    ) -> None:
        for failure_target in ("bundle", "parent"):
            with self.subTest(failure_target=failure_target):
                with tempfile.TemporaryDirectory() as directory:
                    candidate_paths = self._bundle_candidate_paths(directory)
                    bundle_path = Path(directory) / "pty-shard.admitted"
                    original_fsync = coverage_evidence_module._fsync_directory

                    def fail_after_publish(candidate: object) -> None:
                        target = (
                            bundle_path
                            if failure_target == "bundle"
                            else bundle_path.parent
                        )
                        if bundle_path.exists() and candidate == target:
                            raise OSError(
                                "injected post-publish fsync failure"
                            )
                        original_fsync(candidate)  # type: ignore[arg-type]

                    with patch(
                        "tests.agent_live.coverage_evidence._fsync_directory",
                        side_effect=fail_after_publish,
                    ):
                        with self.assertRaisesRegex(
                            RuntimeError,
                            "durability is uncertain",
                        ):
                            admit_pty_artifact_bundle(
                                bundle_path,
                                candidate_paths,
                                signer=self.signer,
                            )

                    valid, reason = validate_pty_artifact_bundle(
                        bundle_path,
                        trusted_public_key_b64=self.signer.public_key_b64,
                    )
                    self.assertFalse(valid)
                    self.assertIn("durability reconciliation", reason)
                    with self.assertRaisesRegex(
                        ValueError,
                        "durability reconciliation",
                    ):
                        load_pty_artifact_bundle(bundle_path)
                    with self.assertRaisesRegex(
                        ValueError,
                        "durability reconciliation",
                    ):
                        pty_artifact_bundle_digest(bundle_path)
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "explicit durability recovery",
                    ):
                        admit_pty_artifact_bundle(
                            bundle_path,
                            candidate_paths,
                            signer=self.signer,
                        )
                    self.assertEqual(
                        recover_pty_artifact_bundle(
                            bundle_path,
                            signer=self.signer,
                        ),
                        bundle_path,
                    )
                    valid, reason = validate_pty_artifact_bundle(
                        bundle_path,
                        trusted_public_key_b64=self.signer.public_key_b64,
                    )
                    self.assertTrue(valid, reason)

    @unittest.skipUnless(
        sys.platform.startswith("linux"),
        "shard admission locking requires Linux flock",
    )
    def test_concurrent_bundle_publishers_share_one_commit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            candidate_paths = self._bundle_candidate_paths(directory)
            bundle_path = Path(directory) / "pty-shard.admitted"
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=2
            ) as executor:
                results = tuple(executor.map(
                    lambda _index: admit_pty_artifact_bundle(
                        bundle_path,
                        candidate_paths,
                        signer=self.signer,
                    ),
                    range(2),
                ))

            self.assertEqual(results, (bundle_path, bundle_path))
            valid, reason = validate_pty_artifact_bundle(
                bundle_path,
                trusted_public_key_b64=self.signer.public_key_b64,
            )
            self.assertTrue(valid, reason)
            self.assertFalse(tuple(
                bundle_path.parent.glob(f"{bundle_path.name}.tmp.*")
            ))

    @unittest.skipUnless(
        sys.platform.startswith("linux"),
        "cross-process shard admission requires Linux fork and flock",
    )
    def test_cross_process_bundle_publishers_share_one_commit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            candidate_paths = self._bundle_candidate_paths(directory)
            bundle_path = Path(directory) / "pty-shard.admitted"
            context = multiprocessing.get_context("fork")
            start = context.Event()

            def publish() -> None:
                start.wait()
                admit_pty_artifact_bundle(
                    bundle_path,
                    candidate_paths,
                    signer=self.signer,
                )

            processes = tuple(
                context.Process(target=publish)
                for _index in range(2)
            )
            for process in processes:
                process.start()
            start.set()
            for process in processes:
                process.join(timeout=10)
                self.assertFalse(process.is_alive())
                self.assertEqual(process.exitcode, 0)

            valid, reason = validate_pty_artifact_bundle(
                bundle_path,
                trusted_public_key_b64=self.signer.public_key_b64,
            )
            self.assertTrue(valid, reason)
            self.assertFalse(tuple(
                bundle_path.parent.glob(f"{bundle_path.name}.tmp.*")
            ))

    @unittest.skipUnless(
        sys.platform.startswith("linux"),
        "writer crash recovery requires Linux fork",
    )
    def test_post_rename_writer_crash_requires_explicit_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            candidate_paths = self._bundle_candidate_paths(directory)
            bundle_path = Path(directory) / "pty-shard.admitted"

            def crash_after_rename() -> None:
                original_fsync = coverage_evidence_module._fsync_directory

                def fsync_or_crash(candidate: Path) -> None:
                    if candidate == bundle_path and bundle_path.exists():
                        os._exit(93)
                    original_fsync(candidate)

                with patch(
                    "tests.agent_live.coverage_evidence._fsync_directory",
                    side_effect=fsync_or_crash,
                ):
                    admit_pty_artifact_bundle(
                        bundle_path,
                        candidate_paths,
                        signer=self.signer,
                    )

            process = multiprocessing.get_context("fork").Process(
                target=crash_after_rename
            )
            process.start()
            process.join(timeout=10)
            self.assertFalse(process.is_alive())
            self.assertEqual(process.exitcode, 93)
            self.assertTrue(bundle_path.exists())

            valid, reason = validate_pty_artifact_bundle(
                bundle_path,
                trusted_public_key_b64=self.signer.public_key_b64,
            )
            self.assertFalse(valid)
            self.assertIn("durability reconciliation", reason)
            with self.assertRaisesRegex(
                ValueError,
                "durability reconciliation",
            ):
                load_pty_artifact_bundle(bundle_path)

            recovered = recover_pty_artifact_bundle(
                bundle_path,
                expected_artifacts=[
                    json.loads(path.read_text(encoding="utf-8"))
                    for path in candidate_paths
                ],
                signer=self.signer,
            )

            self.assertEqual(recovered, bundle_path)
            valid, reason = validate_pty_artifact_bundle(
                bundle_path,
                trusted_public_key_b64=self.signer.public_key_b64,
            )
            self.assertTrue(valid, reason)

    def test_bundle_validator_rejects_writable_publication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            candidate_paths = self._bundle_candidate_paths(directory)
            bundle_path = Path(directory) / "pty-shard.admitted"
            admit_pty_artifact_bundle(
                bundle_path,
                candidate_paths,
                signer=self.signer,
            )
            bundle_path.chmod(0o700)

            valid, reason = validate_pty_artifact_bundle(
                bundle_path,
                trusted_public_key_b64=self.signer.public_key_b64,
            )

            self.assertFalse(valid)
            self.assertIn("immutable", reason)

    def test_legacy_per_item_pair_cannot_qualify_as_shard_evidence(
        self,
    ) -> None:
        artifact = build_pty_cli_evidence_artifact(
            edge=self.edge,
            evidence_class="dynamic_dual_ai",
            revision=self.revision,
            turn=self._turn(),
            observation=self._observation(dynamic=True),
            dynamic_selection=self._selection(),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = write_evidence_artifact(artifact, directory)
            admit_pty_artifact_pair(path, signer=self.signer)

            admitted, reason = load_valid_evidence_reference(
                str(path),
                edge=self.edge,
                revision=self.revision,
                trusted_public_key_b64=self.signer.public_key_b64,
            )

            self.assertIsNone(admitted)
            self.assertIn("shard-bundle member", reason)
            with self.assertRaisesRegex(
                ValueError,
                "invalid PTY shard bundle",
            ):
                observed_coverage_turn_from_artifact(
                    bundle_path=path.with_suffix(".admitted"),
                    artifact_id=str(artifact["evidence_id"]),
                    edge=self.edge,
                    revision=self.revision,
                    trusted_public_key_b64=self.signer.public_key_b64,
                )

    def test_pty_builder_rejects_owning_secret_capability(self) -> None:
        secret_edge_key = "semantic-secret:forbidden-edge-capability"
        edge = {**self.edge, "edge_key": secret_edge_key}
        selection = self._selection(
            target_coverage_ids=(secret_edge_key,),
        )
        decision = asdict(selection)
        decision["target_coverage_ids"] = list(
            selection.target_coverage_ids
        )
        base_observation = self._observation(dynamic=True)
        observation = self._observation(
            dynamic=True,
            target_edge_key=secret_edge_key,
            simulator_decision=decision,
            verified_postcondition=replace(
                base_observation.verified_postcondition,
                observed_coverage_ids=(
                    secret_edge_key,
                    "group:opening",
                ),
            ),
        )

        with self.assertRaisesRegex(
            ValueError,
            "owning secret capability",
        ):
            build_pty_cli_evidence_artifact(
                edge=edge,
                evidence_class="dynamic_dual_ai",
                revision=self.revision,
                turn=self._turn(),
                observation=observation,
                dynamic_selection=selection,
            )

    def test_validator_rejects_rehashed_pending_sensitivity_forgery(
        self,
    ) -> None:
        turn = self._turn()
        selection = self._selection()
        artifact = build_pty_cli_evidence_artifact(
            edge=self.edge,
            evidence_class="dynamic_dual_ai",
            revision=self.revision,
            turn=turn,
            observation=self._observation(dynamic=True),
            dynamic_selection=selection,
        )
        authority = build_pty_authority_receipt(artifact, signer=self.signer)
        event = artifact["turn_observation"]["runtime_events"][0]
        event["pending_contract"] = {
            "id": "opening_next_action",
            "sensitive_input": True,
        }
        artifact["turn_observation"]["pending_contract"] = dict(
            event["pending_contract"]
        )
        artifact["turn_observation_hash"] = content_hash(
            artifact["turn_observation"]
        )
        self._rehash_pty_artifact(artifact)

        valid, reason = validate_pty_cli_evidence_artifact(
            artifact,
            edge=self.edge,
            revision=self.revision,
            authority=authority,
            trusted_public_key_b64=self.signer.public_key_b64,
        )
        self.assertFalse(valid)
        self.assertIn("authority receipt", reason)

    def test_diagnostic_validator_rejects_rehashed_provenance_forgery(
        self,
    ) -> None:
        turn = self._turn()
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
        record = PtyDiagnosticRecord(
            diagnostic_kind="interruption",
            verification_status="interrupted",
            target_id="opening-target",
            target_edge_key=self.edge["edge_key"],
            revision=self.revision,
            session_id=turn.session_id,
            reason="transport interrupted",
            last_complete_response=turn.agent_response,
            input_baseline_event=baseline,
            last_complete_event=committed,
            completed_turn=turn,
            dynamic_selection=self._selection(),
        )
        artifact = build_pty_diagnostic_artifact(record)
        authority = build_pty_authority_receipt(artifact, signer=self.signer)
        boundary = artifact["last_complete_boundary"]
        baseline_payload = boundary["input_baseline_event"]
        baseline_payload["pending_contract"] = {
            "id": "opening_next_action",
            "sensitive_input": True,
        }
        baseline_unsigned = dict(baseline_payload)
        baseline_unsigned.pop("runtime_event_payload_hash", None)
        baseline_payload["runtime_event_payload_hash"] = content_hash(
            baseline_unsigned
        )
        boundary["input_sensitive"] = True
        boundary["input_baseline_event_hash"] = content_hash(
            baseline_payload
        )
        self._rehash_diagnostic(artifact)
        valid, reason = validate_pty_diagnostic_artifact(
            artifact,
            authority=authority,
            trusted_public_key_b64=self.signer.public_key_b64,
        )
        self.assertFalse(valid)
        self.assertIn("authority receipt", reason)

        artifact = build_pty_diagnostic_artifact(record)
        authority = build_pty_authority_receipt(artifact, signer=self.signer)
        artifact["target_edge_key"] = "unrelated::edge"
        artifact["last_complete_boundary"]["dynamic_selection"][
            "target_coverage_ids"
        ] = ["unrelated::edge"]
        artifact["last_complete_boundary"]["dynamic_selection_hash"] = (
            content_hash(
                artifact["last_complete_boundary"]["dynamic_selection"]
            )
        )
        self._rehash_diagnostic(artifact)
        valid, reason = validate_pty_diagnostic_artifact(
            artifact,
            authority=authority,
            trusted_public_key_b64=self.signer.public_key_b64,
        )
        self.assertFalse(valid)
        self.assertIn("authority receipt", reason)

        artifact = build_pty_diagnostic_artifact(record)
        artifact["reason"] = "semantic-secret:forged-capability"
        self._rehash_diagnostic(artifact)
        authority = build_pty_authority_receipt(artifact, signer=self.signer)
        valid, reason = validate_pty_diagnostic_artifact(
            artifact,
            authority=authority,
            trusted_public_key_b64=self.signer.public_key_b64,
        )
        self.assertFalse(valid)
        self.assertIn("owning secret capability", reason)

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
        valid, reason = validate_pty_cli_candidate_artifact(
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

"""Contracts for the G4 obligation-to-Journey provider and evidence adapter."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agent.harness.runtime_identity import repository_revision
from tests.agent_live.batch_orchestrator import (
    JourneyControllerAuthoritySnapshot,
    TimeoutPolicy,
    validate_frozen_manifest,
)
from tests.agent_live.chaos_scheduler import (
    build_journey_schedule,
    journey_schedule_payload,
)
from tests.agent_live.coverage_evidence import content_hash
from tests.agent_live.codex_simulator_bridge import build_simulator_attestation
from tests.agent_live.container_process_guard import (
    ContainerProcessGuard,
    validate_cleanup_receipt_artifact,
)
from tests.agent_live.completed_journey_batch import CompletedJourneySource
from tests.agent_live.product_chaos_journey_provider import (
    build_product_chaos_journey_definition,
    build_product_chaos_journey_manifest,
    build_product_chaos_target_payloads,
    convert_completed_journey_to_product_evidence as convert_authorized_journey_to_product_evidence,
    _convert_journey_runtime_to_product_evidence as _convert_completed_journey_to_product_evidence,
    freeze_product_chaos_batch,
    load_frozen_product_chaos_catalog,
    main,
    write_product_chaos_target_set,
)
from tests.agent_live.dynamic_dual_ai_chaos import (
    journey_outcome_verifier_registry_payload,
)
from tests.agent_live.formal_journey_catalog import (
    FORMAL_JOURNEY_VERIFIER_REGISTRY,
)
from tests.agent_live.product_chaos_obligations import (
    build_product_chaos_obligations,
    product_chaos_obligation_report,
)
from tests.agent_live.product_obligation_evidence import (
    admit_product_obligation_evidence,
)
from tests.agent_live.runtime_checkpoint import (
    reviewed_scenario,
    seed_runtime_checkpoint,
)


REVISION = {"commit": "abc123", "worktree_hash": "frozen-tree"}


def convert_completed_journey_to_product_evidence(*args, **kwargs):
    kwargs.setdefault("round_id", "round-1")
    kwargs.setdefault("provider", "deepseek")
    kwargs.setdefault("model", "deepseek-chat")
    return _convert_completed_journey_to_product_evidence(*args, **kwargs)


class ProductChaosJourneyProviderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.obligations = build_product_chaos_obligations(revision=REVISION)

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.obligation = self.obligations[0]

    def tearDown(self) -> None:
        for path in sorted(self.root.rglob("*"), reverse=True):
            path.chmod(0o700 if path.is_dir() else 0o600)
        self.temp.cleanup()

    def test_product_conversion_rejects_forged_completed_batch_source(
        self,
    ) -> None:
        source = CompletedJourneySource(
            runtime_root=self.root / "missing-runtime",
            obligation_id=str(self.obligation["obligation_id"]),
            shard_id="shard-1",
            execution_id="execution-1",
            controller_snapshot=JourneyControllerAuthoritySnapshot(
                candidate_bytes=b"{}",
                authority_bytes=b"{}",
                bundle_digest="a" * 64,
            ),
            runtime_artifacts=(),
            _authority=object(),
        )

        with self.assertRaisesRegex(
            ValueError,
            "completed-batch authority",
        ):
            convert_authorized_journey_to_product_evidence(
                self.obligation,
                revision=REVISION,
                round_id="round-1",
                source=source,
                evidence_path=self.root / "evidence.json",
            )

    def _write_catalog(self) -> Path:
        path = self.root / "catalog.json"
        path.write_text(
            json.dumps({
                "revision_binding": REVISION,
                "obligations": self.obligations,
            }),
            encoding="utf-8",
        )
        return path

    def _runtime(
        self,
        *,
        classification: str = "passed",
        provider: str = "deepseek",
        model: str = "deepseek-chat",
    ) -> Path:
        runtime = self.root / f"runtime-{classification}"
        runtime.mkdir()
        definition = build_product_chaos_journey_definition(
            self.obligation,
            revision=REVISION,
        )
        schedule = build_journey_schedule(
            revision=REVISION,
            seed=self.obligation["seed"],
            journey=definition["journey"],
        )
        schedule_payload = journey_schedule_payload(schedule)
        (runtime / "journey-schedule.json").write_text(
            json.dumps(schedule_payload),
            encoding="utf-8",
        )
        risk_factor_ids = [
            f"{name}:{value}"
            for name, value in self.obligation["factors"].items()
        ]
        messages = (
            "request:\n{\"method\":\"eth_chainId\",\"params\":[]}",
            "jump to the RPC group and record the response evidence",
        )
        responses = ("Agent> evidence recorded", "Agent> RPC catalog updated")
        (runtime / "transcript.txt").write_text(
            "\n".join((
                "Agent> start",
                f"User> {messages[0]}",
                responses[0],
                f"User> {messages[1]}",
                responses[1],
                "",
            )),
            encoding="utf-8",
        )
        session_id = "g4-session"
        session_purpose = "dynamic-dual-ai-chaos"
        scenario = reviewed_scenario(schedule.start_scenario)
        seed_receipt = seed_runtime_checkpoint(
            scenario.seed_state,
            checkpoint_path=runtime / "checkpoints.sqlite",
            session_id=session_id,
            session_purpose=session_purpose,
            scenario_id=scenario.scenario_id,
            scenario_state_fingerprint=scenario.state_fingerprint,
        )
        baseline = {
            "schema_version": 2,
            "event_type": "startup_snapshot",
            "thread_id": session_id,
            "session_purpose": session_purpose,
            "before_fingerprint": "0" * 64,
            "after_fingerprint": "a" * 64,
            "turn_index": 0,
            "active_group": "chain_rpc",
            "pending_question_id": "custom_schema_evidence",
            "pending_contract": {
                "id": "custom_schema_evidence",
                "kind": "evidence",
                "accepted_action_types": ["rpc_catalog_command", "answer_pending"],
            },
            "revision": REVISION,
            "action_queue_types": [],
            "admitted_action_types": ["prepare_session_entry"],
            "admitted_action_targets": [],
            "state_diff_hashes": {},
            "material_state_diff_hashes": {},
            "after_value_hashes": {
                "target_mode": content_hash("fake-node"),
                "workflow_mode": content_hash("rpc_benchmark"),
                "language": content_hash("en"),
                "rpc_mode": content_hash("single"),
                "chain_identity.name": content_hash("solana"),
                "custom_rpc.method": content_hash("eth_chainId"),
                "group_states.chain_rpc.status": content_hash("completed"),
                "input_shape": content_hash("exact"),
            },
            "next_result": {
                "kind": "question",
                "question_id": "custom_schema_evidence",
            },
        }
        first_event = {
            **baseline,
            "event_type": "turn_committed",
            "before_fingerprint": baseline["after_fingerprint"],
            "after_fingerprint": "b" * 64,
            "turn_index": 1,
            "pending_question_id": "custom_schema_response",
            "pending_contract": {
                "id": "custom_schema_response",
                "kind": "evidence",
                "accepted_action_types": ["rpc_catalog_command", "answer_pending"],
            },
            "admitted_action_types": ["answer_pending"],
            "state_diff_hashes": {
                "evidence_buffer.request": {
                    "before": content_hash(""),
                    "after": content_hash("request"),
                },
                "pending_question.id": {
                    "before": content_hash("custom_schema_evidence"),
                    "after": content_hash("custom_schema_response"),
                },
            },
            "material_state_diff_hashes": {
                "evidence_buffer.request": {
                    "before": content_hash(""),
                    "after": content_hash("request"),
                },
            },
            "after_value_hashes": {
                **baseline["after_value_hashes"],
                "input_shape": content_hash("multiline"),
                "evidence_buffer.request": content_hash("request"),
            },
            "next_result": {
                "kind": "question",
                "question_id": "custom_schema_response",
            },
        }
        event = {
            **first_event,
            "before_fingerprint": first_event["after_fingerprint"],
            "after_fingerprint": "c" * 64,
            "turn_index": 2,
            "active_group": "workload_rpc",
            "pending_question_id": "custom_continue",
            "pending_contract": {
                "id": "custom_continue",
                "kind": "numbered_choice",
                "accepted_action_types": ["answer_pending", "rpc_catalog_command"],
            },
            "admitted_action_types": ["change_group", "rpc_catalog_command"],
            "admitted_action_targets": [
                {"type": "change_group", "group": "workload_rpc"},
            ],
            "state_diff_hashes": {
                "active_group": {
                    "before": content_hash("chain_rpc"),
                    "after": content_hash("workload_rpc"),
                },
                "custom_rpc.method": {
                    "before": content_hash(""),
                    "after": content_hash("eth_chainId"),
                },
                "evidence_buffer.response": {
                    "before": content_hash(""),
                    "after": content_hash("response"),
                },
                "pending_question.id": {
                    "before": content_hash("custom_schema_response"),
                    "after": content_hash("custom_continue"),
                },
            },
            "material_state_diff_hashes": {
                "active_group": {
                    "before": content_hash("chain_rpc"),
                    "after": content_hash("workload_rpc"),
                },
                "custom_rpc.method": {
                    "before": content_hash(""),
                    "after": content_hash("eth_chainId"),
                },
                "evidence_buffer.response": {
                    "before": content_hash(""),
                    "after": content_hash("response"),
                },
            },
            "after_value_hashes": {
                **first_event["after_value_hashes"],
                "active_group": content_hash("workload_rpc"),
                "evidence_buffer.response": content_hash("response"),
            },
            "next_result": {
                "kind": "question",
                "question_id": "custom_continue",
            },
        }
        (runtime / "turn-events.jsonl").write_text(
            "\n".join(json.dumps(item) for item in (baseline, first_event, event)) + "\n",
            encoding="utf-8",
        )

        required = self.obligation["verifier_contract"]["required_postcondition_ids"]
        forbidden = self.obligation["verifier_contract"]["forbidden_postcondition_ids"]
        terminal_postconditions = [
            {
                "postcondition_id": item,
                "satisfied": classification == "passed",
                "details": {"checked": True},
                "verifier": {"verifier_id": item, "verifier_version": 2},
            }
            for item in required
        ]
        forbidden_outcomes = [
            {
                "outcome_id": f"forbidden-{item}",
                "satisfied": False,
                "postconditions": [{
                    "postcondition_id": item,
                    "satisfied": False,
                    "details": {"checked": True},
                    "verifier": {"verifier_id": item, "verifier_version": 2},
                }],
            }
            for item in forbidden
        ]
        def turn_payload(index: int, message: str, response: str, committed: dict):
            previous_response = "Agent> start" if index == 1 else responses[index - 2]
            previous_hash = content_hash(previous_response)
            message_hash = content_hash(message)
            return {
            "turn_index": index,
            "selected_at_ns": (index * 3 - 1) * 1_000_000_000,
            "turn_identity": {
                "transcript_hash": content_hash(
                    [previous_response, message, response]
                ),
                "before_fingerprint": committed["before_fingerprint"],
                "after_fingerprint": committed["after_fingerprint"],
                "previous_response_received_at_ns": (index * 3 - 2) * 1_000_000_000,
                "user_message_submitted_at_ns": (index * 3 - 1) * 1_000_000_000,
                "agent_response_received_at_ns": index * 3 * 1_000_000_000,
            },
            "decision": {
                "user_message": message,
                "persona": definition["journey"]["persona"],
                "mission": definition["journey"]["mission"],
                "rationale": "response driven",
                "risk_factor_ids": risk_factor_ids,
            },
            "decision_provenance": {
                "previous_response_hash": previous_hash,
                "user_message_hash": message_hash,
                "selected_at_ns": (index * 3 - 1) * 1_000_000_000,
                "submitted_at_ns": (index * 3 - 1) * 1_000_000_000,
            },
            "observed_edges": [],
            "terminal_outcome": {
                "outcome_id": definition["journey"]["terminal_outcome"]["outcome_id"],
                "satisfied": classification == "passed",
                "postconditions": terminal_postconditions,
            },
            "forbidden_outcomes": forbidden_outcomes,
            }
        turns = [
            turn_payload(1, messages[0], responses[0], first_event),
            turn_payload(2, messages[1], responses[1], event),
        ]
        execution_id = "g4-execution"
        execution_proof = {}
        if classification == "passed":
            receipt_dir = runtime / "container-cleanup-receipts"
            process_env = os.environ.copy()
            process_env["ANYCHAIN_CHAOS_EXECUTION_ID"] = execution_id
            process = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(60)"],
                env=process_env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
            guard = ContainerProcessGuard(
                execution_id,
                receipt_dir=receipt_dir,
                term_grace_seconds=0.1,
                kill_grace_seconds=0.1,
                scan_interval_seconds=0.01,
            )
            guard.register_pid(process.pid, role="agent_process_group_leader")
            try:
                artifact = guard.cleanup(reapers={process.pid: process.poll})
            finally:
                for stream in (process.stdout, process.stderr):
                    if stream is not None:
                        stream.close()
            proof = validate_cleanup_receipt_artifact(
                artifact.path,
                execution_id=execution_id,
                required_roles=(
                    "container_bridge",
                    "agent_process_group_leader",
                ),
                allowed_roots=(receipt_dir,),
            )
            execution_proof = {
                "proof_type": "container_pty_process_guard",
                "transport_kind": "container_pty_bridge",
                **proof,
            }
        for source_turn in turns:
            provenance = source_turn["decision_provenance"]
            request_id = f"request-{source_turn['turn_index']}"
            context_binding = {
                "session_id": "g4-session",
                "turn_index": source_turn["turn_index"],
                "previous_response_hash": provenance[
                    "previous_response_hash"
                ],
                "previous_response_received_at_ns": source_turn[
                    "turn_identity"
                ]["previous_response_received_at_ns"],
                "control_identity": {
                    "lane": "journey",
                    "schedule_id": schedule.schedule_id,
                },
                "observed_edge_keys": [],
            }
            decision = {
                "previous_response_hash": provenance[
                    "previous_response_hash"
                ],
                **source_turn["decision"],
                "broker_request_id": request_id,
            }
            attestation = build_simulator_attestation(
                actor_kind="codex",
                task_id="codex-task-g4-test",
                model="gpt-test",
                request_id=request_id,
                previous_response_hash=decision[
                    "previous_response_hash"
                ],
                context_hash=content_hash(context_binding),
                decision_hash=content_hash(decision),
                user_message_hash=provenance["user_message_hash"],
                turn_index=source_turn["turn_index"],
                declared_at_ns=source_turn["selected_at_ns"],
            )
            provenance.update({
                "execution_id": execution_id,
                "obligation_id": self.obligation["obligation_id"],
                "broker_request_id": request_id,
                "simulator_context_binding": context_binding,
                "simulator_attestation": attestation,
                "variant_attestation": {},
            })
        turn = turns[-1]
        qualifying = classification == "passed"
        journey_payload = {
            "schema_version": 4,
            "artifact_type": "dynamic_dual_ai_journey_evidence",
            "runner_type": "dynamic_dual_ai_journey",
            "schedule_id": schedule.schedule_id,
            "schedule_hash": content_hash(schedule_payload),
            "journey_id": self.obligation["obligation_id"],
            "revision": REVISION,
            "verifier_registry": journey_outcome_verifier_registry_payload(
                FORMAL_JOURNEY_VERIFIER_REGISTRY
            ),
            "session_id": session_id,
            "session_purpose": session_purpose,
            "seed_receipt": seed_receipt.artifact,
            "execution_id": execution_id,
            "provider": provider,
            "model": model,
            "terminal_classification": classification,
            "qualifying_evidence": qualifying,
            "qualification_reason": (
                "trusted_container_pty_execution"
                if qualifying
                else "execution_failed"
            ),
            "execution_proof": execution_proof,
            "terminal_outcome_id": (
                definition["journey"]["terminal_outcome"]["outcome_id"]
                if qualifying else ""
            ),
            "max_turns": definition["journey"]["max_turns"],
            "completed_turn_count": 2,
            "observed_edge_keys": [],
            "failure_reason": "" if qualifying else classification,
            "transcript_hash": "f" * 64,
            "source_transcript_hash": "1" * 64,
            "content_redacted": True,
            "initial_verification": {
                "terminal_outcome": {
                    "outcome_id": definition["journey"]["terminal_outcome"]["outcome_id"],
                    "satisfied": False,
                    "postconditions": [
                        {
                            "postcondition_id": item,
                            "satisfied": False,
                            "details": {},
                            "verifier": {"verifier_id": item},
                        }
                        for item in required
                    ],
                },
                "forbidden_outcomes": forbidden_outcomes,
            },
            "turns": turns,
        }
        journey_evidence_id = content_hash(journey_payload)
        journey_evidence = {**journey_payload, "evidence_id": journey_evidence_id}
        journey_evidence["artifact_hash"] = content_hash(journey_evidence)
        evidence_dir = runtime / "journey-controller-admission"
        evidence_dir.mkdir()
        source_evidence_path = evidence_dir / "candidate.json"
        source_evidence_path.write_text(
            json.dumps(journey_evidence),
            encoding="utf-8",
        )
        source_evidence_path.chmod(0o400)
        evidence_dir.chmod(0o500)
        result = {
            "schema_version": 1,
            "schedule_id": schedule.schedule_id,
            "journey_id": self.obligation["obligation_id"],
            "revision": REVISION,
            "execution_status": classification,
            "terminal_classification": classification,
            "terminal_outcome_id": journey_payload["terminal_outcome_id"],
            "max_turns": definition["journey"]["max_turns"],
            "completed_turn_count": 2,
            "observed_edge_keys": [],
            "failure_reason": journey_payload["failure_reason"],
            "evidence_id": journey_evidence_id,
            "evidence_path": str(source_evidence_path),
            "qualifying_evidence": qualifying,
            "verifier_registry_id": self.obligation["verifier_contract"]["registry_id"],
            "initial_verification": journey_payload["initial_verification"],
            "turns": turns,
        }
        (runtime / "journey-result.json").write_text(
            json.dumps(result),
            encoding="utf-8",
        )
        return runtime

    def _mutate_source_evidence(self, runtime: Path, mutate) -> None:
        result_path = runtime / "journey-result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        evidence_path = Path(result["evidence_path"])
        evidence_path.parent.chmod(0o700)
        evidence_path.chmod(0o600)
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        mutate(evidence)
        unsigned = {
            key: value
            for key, value in evidence.items()
            if key not in {"evidence_id", "artifact_hash"}
        }
        evidence_id = content_hash(unsigned)
        evidence = {**unsigned, "evidence_id": evidence_id}
        evidence["artifact_hash"] = content_hash(evidence)
        evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
        evidence_path.chmod(0o400)
        evidence_path.parent.chmod(0o500)
        result["evidence_id"] = evidence_id
        result["turns"] = evidence["turns"]
        result_path.write_text(json.dumps(result), encoding="utf-8")

    def test_all_definitions_are_batch_compatible_and_contain_no_future_turns(self) -> None:
        manifest = build_product_chaos_journey_manifest(
            self.obligations,
            revision=REVISION,
        )
        self.assertEqual(
            manifest["definition_count"],
            len(self.obligations),
        )
        self.assertFalse(manifest["generation_is_execution"])
        self.assertFalse(manifest["prewritten_future_turns"])
        for row in manifest["definitions"]:
            with self.subTest(obligation_id=row["obligation_id"]):
                definition = row["definition"]
                self.assertEqual(set(definition), {"lane", "verifier_registry", "journey"})
                self.assertEqual(definition["lane"], "journey")
                self.assertEqual(
                    definition["journey"]["journey_id"],
                    row["obligation_id"],
                )
                schedule = build_journey_schedule(
                    revision=REVISION,
                    seed=row["schedule"]["seed"],
                    journey=definition["journey"],
                )
                self.assertEqual(journey_schedule_payload(schedule), row["schedule"])
                serialized = json.dumps(definition)
                self.assertNotIn('"turns"', serialized)
                self.assertNotIn('"user_message"', serialized)

    def test_single_manifest_and_cli_are_revision_and_contract_bound(self) -> None:
        catalog = self._write_catalog()
        output = self.root / "single-manifest.json"
        self.assertEqual(main([
            "definitions",
            "--catalog", str(catalog),
            "--output", str(output),
            "--obligation-id", self.obligation["obligation_id"],
        ]), 0)
        payload = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(payload["definition_count"], 1)
        row = payload["definitions"][0]
        self.assertEqual(row["obligation_contract_hash"], self.obligation["contract_hash"])
        self.assertEqual(row["revision_binding"], REVISION)
        self.assertEqual(
            payload["manifest_hash"],
            content_hash({key: value for key, value in payload.items() if key != "manifest_hash"}),
        )
        with self.assertRaises(FileExistsError):
            main([
                "definitions",
                "--catalog", str(catalog),
                "--output", str(output),
            ])

    def test_definitions_to_targets_preserves_each_obligation_schedule(self) -> None:
        manifest = build_product_chaos_journey_manifest(
            self.obligations,
            revision=REVISION,
        )
        payloads = build_product_chaos_target_payloads(manifest)
        self.assertEqual(len(payloads), len(self.obligations))
        for definition_row, target in zip(
            manifest["definitions"], payloads, strict=True
        ):
            frozen = target["frozen_execution"]
            self.assertEqual(frozen["obligation_id"], definition_row["obligation_id"])
            self.assertEqual(frozen["seed"], definition_row["schedule"]["seed"])
            self.assertEqual(
                frozen["schedule_id"], definition_row["schedule"]["schedule_id"]
            )
            self.assertTrue(target["simulator_attestation_contract"]["required"])
            self.assertNotIn('"user_message"', json.dumps(target))

        target_dir = self.root / "targets"
        target_manifest_path = write_product_chaos_target_set(
            manifest, target_dir
        )
        target_manifest = json.loads(
            target_manifest_path.read_text(encoding="utf-8")
        )
        self.assertEqual(
            target_manifest["target_count"],
            len(self.obligations),
        )
        self.assertTrue(target_manifest["reachability_preflight_complete"])
        self.assertGreater(target_manifest["scenario_preflight_count"], 0)
        self.assertEqual(
            target_manifest["coverage_denominators"],
            {
                "generated": len(self.obligations),
                "executable": len(self.obligations),
                "qualifying": 0,
            },
        )
        self.assertEqual(
            [row["seed"] for row in target_manifest["targets"]],
            [row["schedule"]["seed"] for row in manifest["definitions"]],
        )
        self.assertTrue(all(
            row["applicability_receipt"]["applicable"] is True
            and row["applicability_receipt_hash"]
            == content_hash(row["applicability_receipt"])
            for row in target_manifest["targets"]
        ))
        with patch(
            "tests.agent_live.product_chaos_journey_provider.freeze_batch_manifest"
        ) as freeze:
            freeze.return_value = object()
            freeze_product_chaos_batch(
                repo_root=self.root,
                targets_dir=target_dir,
                manifest_path=self.root / "batch.json",
                runtime_base=self.root / "runtime",
                environment={"DEEPSEEK_API_KEY": "unit-test-secret"},
            )
        self.assertEqual(
            freeze.call_args.kwargs["shard_count"],
            len(self.obligations),
        )
        self.assertEqual(freeze.call_args.kwargs["expected_revision"], REVISION)
        self.assertEqual(
            freeze.call_args.kwargs["required_env_names"],
            ("DEEPSEEK_API_KEY",),
        )
        self.assertNotIn(
            "unit-test-secret",
            repr(freeze.call_args.kwargs),
        )

        definitions_path = self.root / "definitions.json"
        definitions_path.write_text(json.dumps(manifest), encoding="utf-8")
        cli_target_dir = self.root / "cli-targets"
        self.assertEqual(main([
            "targets",
            "--definitions", str(definitions_path),
            "--output-dir", str(cli_target_dir),
        ]), 0)

        with (
            patch.dict(os.environ, {"DEEPSEEK_API_KEY": "cli-test-secret"}),
            patch(
                "tests.agent_live.product_chaos_journey_provider.freeze_batch_manifest"
            ) as freeze,
        ):
            freeze.return_value = SimpleNamespace(
                manifest_id="frozen-manifest",
                pty_authority_trust_root_id="frozen-trust-root",
            )
            self.assertEqual(main([
                "batch",
                "--repo-root", str(self.root),
                "--targets-dir", str(cli_target_dir),
                "--output", str(self.root / "cli-batch.json"),
                "--runtime-base", str(self.root / "cli-runtime"),
                "--worker-runtime", "docker",
                "--max-concurrency", "7",
                "--shard-timeout-seconds", "901",
                "--decision-timeout-seconds", "241",
                "--cleanup-timeout-seconds", "6",
            ]), 0)
        self.assertEqual(
            freeze.call_args.kwargs["shard_count"],
            len(self.obligations),
        )
        self.assertEqual(freeze.call_args.kwargs["worker_runtime"], "docker")
        self.assertEqual(freeze.call_args.kwargs["max_concurrency"], 7)
        self.assertEqual(
            freeze.call_args.kwargs["timeout_policy"],
            TimeoutPolicy(
                shard_seconds=901.0,
                decision_seconds=241.0,
                cleanup_seconds=6.0,
            ),
        )
        self.assertNotIn("cli-test-secret", repr(freeze.call_args.kwargs))

    def test_batch_recomputes_target_applicability_instead_of_trusting_manifest(
        self,
    ) -> None:
        manifest = build_product_chaos_journey_manifest(
            self.obligations,
            revision=REVISION,
        )
        target_dir = self.root / "tampered-applicability-targets"
        target_manifest_path = write_product_chaos_target_set(
            manifest,
            target_dir,
        )
        target_manifest = json.loads(
            target_manifest_path.read_text(encoding="utf-8")
        )
        row = target_manifest["targets"][0]
        row["applicability_receipt"]["applicable"] = False
        row["applicability_receipt_hash"] = content_hash(
            row["applicability_receipt"]
        )
        scenario = next(
            item
            for item in target_manifest["scenario_preflights"]
            if item["scenario_id"] == row["scenario_id"]
        )
        row["preflight_id"] = content_hash({
            "obligation_id": row["obligation_id"],
            "schedule_id": row["schedule_id"],
            "subject_group": row["subject_group"],
            "scenario_id": row["scenario_id"],
            "seed_receipt_hash": scenario["seed_receipt_hash"],
            "applicability_receipt_hash": (
                row["applicability_receipt_hash"]
            ),
        })
        unsigned = {
            key: value
            for key, value in target_manifest.items()
            if key != "manifest_hash"
        }
        target_manifest["manifest_hash"] = content_hash(unsigned)
        target_manifest_path.write_text(
            json.dumps(target_manifest),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(
            ValueError,
            "matching reachability preflight",
        ):
            freeze_product_chaos_batch(
                repo_root=self.root,
                targets_dir=target_dir,
                manifest_path=self.root / "tampered-batch.json",
                runtime_base=self.root / "tampered-runtime",
                environment={"DEEPSEEK_API_KEY": "unit-test-secret"},
            )

    def test_batch_fails_fast_when_required_provider_environment_is_missing(self) -> None:
        manifest = build_product_chaos_journey_manifest(
            self.obligations,
            revision=REVISION,
        )
        target_dir = self.root / "missing-env-targets"
        write_product_chaos_target_set(manifest, target_dir)

        with (
            patch(
                "tests.agent_live.product_chaos_journey_provider.freeze_batch_manifest"
            ) as freeze,
            self.assertRaisesRegex(RuntimeError, "DEEPSEEK_API_KEY"),
        ):
            freeze_product_chaos_batch(
                repo_root=self.root,
                targets_dir=target_dir,
                manifest_path=self.root / "missing-env-batch.json",
                runtime_base=self.root / "missing-env-runtime",
                environment={"DEEPSEEK_API_KEY": "   "},
            )
        freeze.assert_not_called()

    def test_catalog_report_is_the_authority_for_g4_denominator(self) -> None:
        report = product_chaos_obligation_report(
            self.obligations,
            revision=REVISION,
        )
        self.assertEqual(report["required_denominator"], 535)
        self.assertEqual(
            report["by_model"],
            {
                "anychain-agent-product-chaos": 163,
                "anychain-agent-product-chaos-state-control": 372,
            },
        )

    def test_full_authoritative_catalog_freezes_and_reloads_without_mocking(self) -> None:
        repo_root = Path.cwd().resolve()
        revision = repository_revision(repo_root)
        obligations = build_product_chaos_obligations(revision=revision)
        definitions = build_product_chaos_journey_manifest(
            obligations,
            revision=revision,
        )
        target_dir = self.root / "full-targets"
        write_product_chaos_target_set(definitions, target_dir)
        frozen = freeze_product_chaos_batch(
            repo_root=repo_root,
            targets_dir=target_dir,
            manifest_path=self.root / "full-batch.json",
            runtime_base=self.root / "full-runtime",
            worker_runtime="linux",
            environment={"DEEPSEEK_API_KEY": "unit-test-secret"},
        )
        validate_frozen_manifest(frozen)
        self.assertNotIn(
            "unit-test-secret",
            Path(frozen.manifest_path).read_text(encoding="utf-8"),
        )
        self.assertEqual(len(frozen.shards), len(obligations))
        self.assertEqual(
            len({row.obligation_id for row in frozen.shards}),
            len(obligations),
        )
        self.assertTrue(all(row.subject_group for row in frozen.shards))
        self.assertTrue(
            all(row.simulator_attestation_required for row in frozen.shards)
        )

    def test_batch_rejects_rehashed_incomplete_or_cross_bound_target_sets(self) -> None:
        manifest = build_product_chaos_journey_manifest(
            self.obligations,
            revision=REVISION,
        )

        incomplete_dir = self.root / "incomplete-targets"
        incomplete_manifest_path = write_product_chaos_target_set(
            manifest,
            incomplete_dir,
        )
        incomplete = json.loads(
            incomplete_manifest_path.read_text(encoding="utf-8")
        )
        incomplete["targets"].pop()
        incomplete["target_count"] = len(incomplete["targets"])
        obligation_set = [
            {
                "obligation_id": row["obligation_id"],
                "schedule_id": row["schedule_id"],
                "seed": row["seed"],
                "subject_group": row["subject_group"],
            }
            for row in sorted(
                incomplete["targets"],
                key=lambda item: (
                    item["obligation_id"],
                    item["schedule_id"],
                ),
            )
        ]
        incomplete["expected_obligation_set_hash"] = content_hash(obligation_set)
        unsigned = {
            key: value
            for key, value in incomplete.items()
            if key != "manifest_hash"
        }
        incomplete["manifest_hash"] = content_hash(unsigned)
        incomplete_manifest_path.write_text(
            json.dumps(incomplete),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "authoritative catalog"):
            freeze_product_chaos_batch(
                repo_root=self.root,
                targets_dir=incomplete_dir,
                manifest_path=self.root / "incomplete-batch.json",
                runtime_base=self.root / "incomplete-runtime",
                environment={"DEEPSEEK_API_KEY": "unit-test-secret"},
            )

        cross_bound_dir = self.root / "cross-bound-targets"
        cross_bound_manifest_path = write_product_chaos_target_set(
            manifest,
            cross_bound_dir,
        )
        cross_bound = json.loads(
            cross_bound_manifest_path.read_text(encoding="utf-8")
        )
        cross_bound["targets"][0]["subject_group"] = "execution"
        unsigned = {
            key: value
            for key, value in cross_bound.items()
            if key != "manifest_hash"
        }
        cross_bound["manifest_hash"] = content_hash(unsigned)
        cross_bound_manifest_path.write_text(
            json.dumps(cross_bound),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "frozen execution"):
            freeze_product_chaos_batch(
                repo_root=self.root,
                targets_dir=cross_bound_dir,
                manifest_path=self.root / "cross-bound-batch.json",
                runtime_base=self.root / "cross-bound-runtime",
                environment={"DEEPSEEK_API_KEY": "unit-test-secret"},
            )

        stale_preflight_dir = self.root / "stale-preflight-targets"
        stale_preflight_manifest_path = write_product_chaos_target_set(
            manifest,
            stale_preflight_dir,
        )
        stale_preflight = json.loads(
            stale_preflight_manifest_path.read_text(encoding="utf-8")
        )
        checkpoint = (
            stale_preflight_dir
            / stale_preflight["scenario_preflights"][0]["checkpoint_path"]
        )
        checkpoint.write_bytes(checkpoint.read_bytes() + b"tampered")
        with self.assertRaisesRegex(ValueError, "reachability receipt"):
            freeze_product_chaos_batch(
                repo_root=self.root,
                targets_dir=stale_preflight_dir,
                manifest_path=self.root / "stale-preflight-batch.json",
                runtime_base=self.root / "stale-preflight-runtime",
                environment={"DEEPSEEK_API_KEY": "unit-test-secret"},
            )

        drifted_payload_dir = self.root / "drifted-payload-targets"
        drifted_manifest_path = write_product_chaos_target_set(
            manifest,
            drifted_payload_dir,
        )
        drifted_manifest = json.loads(
            drifted_manifest_path.read_text(encoding="utf-8")
        )
        first_target_row = drifted_manifest["targets"][0]
        first_target_path = drifted_payload_dir / first_target_row["path"]
        first_target = json.loads(first_target_path.read_text(encoding="utf-8"))
        first_target["verifier_registry"] = (
            "tests.agent_live.retained_regression_runner:"
            "RETAINED_REGRESSION_JOURNEY_VERIFIER_REGISTRY"
        )
        first_target_path.write_text(
            json.dumps(first_target),
            encoding="utf-8",
        )
        first_target_row["sha256"] = hashlib.sha256(
            first_target_path.read_bytes()
        ).hexdigest()
        unsigned = {
            key: value
            for key, value in drifted_manifest.items()
            if key != "manifest_hash"
        }
        drifted_manifest["manifest_hash"] = content_hash(unsigned)
        drifted_manifest_path.write_text(
            json.dumps(drifted_manifest),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "target payload"):
            freeze_product_chaos_batch(
                repo_root=self.root,
                targets_dir=drifted_payload_dir,
                manifest_path=self.root / "drifted-payload-batch.json",
                runtime_base=self.root / "drifted-payload-runtime",
                environment={"DEEPSEEK_API_KEY": "unit-test-secret"},
            )

    def test_catalog_rejects_tampering_and_stale_revision(self) -> None:
        catalog = self._write_catalog()
        rows, revision = load_frozen_product_chaos_catalog(catalog)
        self.assertEqual(
            (len(rows), revision),
            (len(self.obligations), REVISION),
        )
        payload = json.loads(catalog.read_text(encoding="utf-8"))
        payload["obligations"][0]["contract_hash"] = "0" * 64
        catalog.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "contract hash"):
            load_frozen_product_chaos_catalog(catalog)

    def test_single_obligation_api_rejects_rehashed_noncanonical_row(self) -> None:
        forged = copy.deepcopy(self.obligation)
        forged["factors"]["subject_group"] = "execution"
        unsigned = dict(forged)
        unsigned.pop("contract_hash")
        forged["contract_hash"] = content_hash(unsigned)
        with self.assertRaisesRegex(ValueError, "authoritative catalog"):
            build_product_chaos_journey_definition(
                forged,
                revision=REVISION,
            )

    def test_passed_journey_is_blocked_without_exact_factor_receipts(self) -> None:
        runtime = self._runtime()
        output = self.root / "product-evidence" / "evidence.json"
        with self.assertRaisesRegex(ValueError, "missing product receipts"):
            convert_completed_journey_to_product_evidence(
                self.obligation,
                revision=REVISION,
                runtime_root=runtime,
                evidence_path=output,
            )
        self.assertFalse(output.exists())

    def test_conversion_uses_persistent_nondefault_model_identity(self) -> None:
        runtime = self._runtime(model="deepseek-v4-pro")
        output = self.root / "configured-model-evidence" / "evidence.json"
        with (
            patch(
                "tests.agent_live.product_chaos_journey_provider.load_llm_config",
                return_value=SimpleNamespace(
                    provider="deepseek",
                    model="deepseek-v4-pro",
                ),
            ),
            self.assertRaisesRegex(ValueError, "missing product receipts"),
        ):
            _convert_completed_journey_to_product_evidence(
                self.obligation,
                revision=REVISION,
                round_id="round-configured-model",
                runtime_root=runtime,
                evidence_path=output,
            )

    def test_conversion_rejects_shared_evidence_and_checkpoint_directory(self) -> None:
        runtime = self._runtime()
        shared = self.root / "shared-output"

        with self.assertRaisesRegex(ValueError, "separate directories"):
            convert_completed_journey_to_product_evidence(
                self.obligation,
                revision=REVISION,
                runtime_root=runtime,
                evidence_path=shared / "evidence.json",
                checkpoint_diff_path=shared / "checkpoint-diff.json",
            )

    def test_failed_and_blocked_journeys_are_mapped_honestly(self) -> None:
        cases = (
            ("product_failed", "failed"),
            ("simulator_invalid", "failed"),
            ("infrastructure_interrupted", "failed"),
            ("externally_blocked", "externally_blocked"),
        )
        for index, (classification, expected) in enumerate(cases):
            with self.subTest(classification=classification):
                runtime = self._runtime(classification=classification)
                output = self.root / "product-evidence" / f"evidence-{index}.json"
                convert_completed_journey_to_product_evidence(
                    self.obligation,
                    revision=REVISION,
                    runtime_root=runtime,
                    evidence_path=output,
                )
                payload = json.loads(output.read_text(encoding="utf-8"))
                self.assertEqual(payload["outcome"], expected)
                statuses = {row["status"] for row in payload["verifier_results"]}
                self.assertIn(expected, statuses)
                summary = admit_product_obligation_evidence(
                    obligations=[self.obligation],
                    evidence_paths=[output],
                    revision=REVISION,
                )
                self.assertFalse(summary["complete"])

    def test_conversion_fails_closed_on_provider_or_runtime_tampering(self) -> None:
        runtime = self._runtime(
            classification="infrastructure_interrupted",
            provider="openai",
        )
        with self.assertRaisesRegex(ValueError, "provider/model"):
            convert_completed_journey_to_product_evidence(
                self.obligation,
                revision=REVISION,
                runtime_root=runtime,
                evidence_path=self.root / "product-evidence" / "wrong-provider.json",
            )
        with self.assertRaisesRegex(ValueError, "explicit DeepSeek"):
            convert_completed_journey_to_product_evidence(
                self.obligation,
                revision=REVISION,
                runtime_root=runtime,
                evidence_path=self.root / "product-evidence" / "forged-provider-boundary.json",
                provider="openai",
                model="gpt",
            )

        runtime = self._runtime()
        events = runtime / "turn-events.jsonl"
        rows = [
            json.loads(line)
            for line in events.read_text(encoding="utf-8").splitlines()
        ]
        rows[-1]["after_fingerprint"] = "9" * 64
        events.write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "fingerprints differ"):
            convert_completed_journey_to_product_evidence(
                self.obligation,
                revision=REVISION,
                runtime_root=runtime,
                evidence_path=self.root / "product-evidence" / "tampered-runtime.json",
            )

    def test_conversion_rejects_missing_or_tampered_process_proof(self) -> None:
        runtime = self._runtime()
        self._mutate_source_evidence(
            runtime,
            lambda evidence: evidence.update(execution_proof={}),
        )
        with self.assertRaisesRegex(ValueError, "execution-bound PTY proof"):
            convert_completed_journey_to_product_evidence(
                self.obligation,
                revision=REVISION,
                runtime_root=runtime,
                evidence_path=self.root / "product-evidence" / "missing-proof.json",
            )

        shutil.rmtree(runtime)
        runtime = self._runtime()
        self._mutate_source_evidence(
            runtime,
            lambda evidence: evidence["execution_proof"].update(
                sha256="0" * 64
            ),
        )
        with self.assertRaisesRegex(ValueError, "differs from its receipt"):
            convert_completed_journey_to_product_evidence(
                self.obligation,
                revision=REVISION,
                runtime_root=runtime,
                evidence_path=self.root / "product-evidence" / "tampered-proof.json",
            )

        shutil.rmtree(runtime)
        runtime = self._runtime()
        result = json.loads(
            (runtime / "journey-result.json").read_text(encoding="utf-8")
        )
        evidence_path = Path(result["evidence_path"])
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        original_receipt = Path(evidence["execution_proof"]["path"])
        replay_root = self.root / "foreign-runtime"
        replay_root.mkdir()
        replay_receipt = replay_root / original_receipt.name
        shutil.copy2(original_receipt, replay_receipt)
        self._mutate_source_evidence(
            runtime,
            lambda payload: payload["execution_proof"].update(
                path=str(replay_receipt)
            ),
        )
        with self.assertRaisesRegex(ValueError, "outside its runtime root"):
            convert_completed_journey_to_product_evidence(
                self.obligation,
                revision=REVISION,
                runtime_root=runtime,
                evidence_path=self.root / "product-evidence" / "replayed-proof.json",
            )

    def test_conversion_rejects_registry_payload_with_matching_id(self) -> None:
        runtime = self._runtime()

        def drift_registry(evidence):
            evidence["verifier_registry"]["definitions"].pop()

        self._mutate_source_evidence(runtime, drift_registry)
        with self.assertRaisesRegex(ValueError, "verifier registry"):
            convert_completed_journey_to_product_evidence(
                self.obligation,
                revision=REVISION,
                runtime_root=runtime,
                evidence_path=self.root / "product-evidence" / "registry-drift.json",
            )

    def test_conversion_rejects_missing_scripted_or_stale_attestation(self) -> None:
        runtime = self._runtime()
        output = self.root / "product-evidence" / "missing.json"
        self._mutate_source_evidence(
            runtime,
            lambda evidence: evidence["turns"][0][
                "decision_provenance"
            ].pop("simulator_attestation"),
        )
        with self.assertRaisesRegex(ValueError, "attestation"):
            convert_completed_journey_to_product_evidence(
                self.obligation,
                revision=REVISION,
                runtime_root=runtime,
                evidence_path=output,
            )

        shutil.rmtree(runtime)
        runtime = self._runtime()

        def make_scripted(evidence):
            attestation = evidence["turns"][0]["decision_provenance"][
                "simulator_attestation"
            ]
            attestation["actor"]["actor_kind"] = "script"
            unsigned = {
                key: value
                for key, value in attestation.items()
                if key != "attestation_id"
            }
            attestation["attestation_id"] = content_hash(unsigned)

        self._mutate_source_evidence(runtime, make_scripted)
        with self.assertRaisesRegex(ValueError, "scripted"):
            convert_completed_journey_to_product_evidence(
                self.obligation,
                revision=REVISION,
                runtime_root=runtime,
                evidence_path=self.root / "product-evidence" / "scripted.json",
            )

    def test_conversion_independently_reruns_factor_verifiers(self) -> None:
        runtime = self._runtime()
        events = runtime / "turn-events.jsonl"
        rows = [
            json.loads(line)
            for line in events.read_text(encoding="utf-8").splitlines()
        ]
        for row in rows:
            row["after_value_hashes"]["language"] = content_hash("zh")
        events.write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "independent verifier rerun"):
            convert_completed_journey_to_product_evidence(
                self.obligation,
                revision=REVISION,
                runtime_root=runtime,
                evidence_path=self.root / "product-evidence" / "factor-drift.json",
            )

    def test_conversion_does_not_accept_catalog_generation_as_execution(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing or empty"):
            convert_completed_journey_to_product_evidence(
                self.obligation,
                revision=REVISION,
                runtime_root=self.root / "not-run",
                evidence_path=self.root / "product-evidence" / "not-run.json",
            )

        forged = copy.deepcopy(self.obligation)
        forged["status"] = "passed"
        unsigned = dict(forged)
        unsigned.pop("contract_hash")
        forged["contract_hash"] = content_hash(unsigned)
        with self.assertRaisesRegex(ValueError, "cannot claim execution"):
            build_product_chaos_journey_definition(forged, revision=REVISION)


if __name__ == "__main__":
    unittest.main()

"""Tests for tamper-evident Harness coverage artifacts."""

from __future__ import annotations

import hashlib
import json
import struct
import tempfile
import unittest
import zlib
from pathlib import Path

from tests.agent_live import coverage_evidence as coverage_evidence_module
from tests.agent_live.coverage_events import (
    capture_coverage_events,
    emit_coverage_event,
    observe_compiled_graph_turn,
)
from agent.harness.domains.environment import question_for_environment
from agent.harness.state import new_state
from agent.runners.execution_scenarios import EXECUTION_SCENARIOS
from agent.runners.benchmark_pipeline import (
    _real_node_smoke_plan,
    _smoke_execution_root,
)
from agent.runners.materialize import materialize_runtime_env
from agent.runners.runtime_env_projection import validate_runtime_env_projection
from tests.agent_live.coverage_evidence import (
    COMPILED_GRAPH_RUNNER,
    G5_RUNTIME_CONTRACT,
    RuntimeTurnEvent,
    TurnObservation,
    VerifiedPostcondition,
    _html_report_error,
    _rpc_benchmark_content_error,
    _sync_observe_content_error,
    _validate_runtime_event,
    _redacted_turn_observation,
    build_evidence_artifact,
    build_real_execution_evidence_artifact,
    content_hash,
    g5_scenario_admission,
    validate_evidence_artifact,
    validate_real_execution_ledger_artifacts,
    validate_real_execution_predecessor,
    validate_real_execution_evidence_artifact,
)
from tests.agent_live.graph_turn import invoke_product_graph_turn


class CoverageEvidenceTest(unittest.TestCase):
    @staticmethod
    def _write_summary(path: Path, *, mode: str, max_qps: int) -> None:
        path.write_text(
            json.dumps({
                "run_id": "run-test",
                "benchmark_mode": mode,
                "start_time": "2026-07-24T00:00:00Z",
                "end_time": "2026-07-24T00:00:10Z",
                "max_successful_qps": max_qps,
                "test_parameters": {
                    "initial_qps": max_qps,
                    "max_qps": max_qps,
                    "qps_step": max_qps,
                    "duration_per_level": 10 if max_qps else 0,
                },
            }),
            encoding="utf-8",
        )

    @staticmethod
    def _write_rgba_png(path: Path, *, width: int = 2, height: int = 2) -> None:
        def chunk(kind: bytes, payload: bytes) -> bytes:
            return (
                struct.pack(">I", len(payload))
                + kind
                + payload
                + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
            )

        raw = b"".join(b"\x00" + (b"\x00" * width * 4) for _ in range(height))
        path.write_bytes(
            b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw))
            + chunk(b"IEND", b"")
        )

    def test_rpc_content_contract_rejects_empty_load_and_inconsistent_results(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            performance = root / "performance.csv"
            proxy = root / "proxy.csv"
            vegeta = root / "vegeta.json"
            summary = root / "summary.json"
            artifacts = {
                "performance_csv": str(performance),
                "proxy_method_csv": str(proxy),
                "vegeta_json": str(vegeta),
                "summary_json": str(summary),
            }
            self._write_summary(summary, mode="quick", max_qps=1)
            performance.write_text(
                "timestamp,current_qps,rpc_latency_ms,qps_data_available,cpu_usage,mem_usage\n"
                "2026-07-24T00:00:00Z,1,2,true,10,20\n",
                encoding="utf-8",
            )
            proxy.write_text(
                "timestamp_ns,method_name,status_code,latency_ms\n"
                "1784851200000000000,eth_blockNumber,200,2\n",
                encoding="utf-8",
            )
            vegeta.write_text(json.dumps({
                "requests": 1,
                "success": 1.0,
                "throughput": 1.0,
                "status_codes": {"200": 1},
            }), encoding="utf-8")
            plan = {
                "rpc_mode": "single",
                "chain_template_requirements": {"single_method": "eth_blockNumber"},
            }
            self.assertEqual(
                _rpc_benchmark_content_error(artifacts, plan=plan),
                "",
            )

            performance.write_text(
                "timestamp,current_qps,rpc_latency_ms,qps_data_available,cpu_usage,mem_usage\n"
                "2026-07-24T00:00:00Z,0,0,false,10,20\n",
                encoding="utf-8",
            )
            self.assertIn(
                "positive-QPS",
                _rpc_benchmark_content_error(artifacts, plan=plan),
            )
            performance.write_text(
                "timestamp,current_qps,rpc_latency_ms,qps_data_available,cpu_usage,mem_usage\n"
                "2026-07-24T00:00:00Z,1,2,true,10,20\n",
                encoding="utf-8",
            )
            vegeta.write_text(json.dumps({
                "requests": 2,
                "success": 1.0,
                "throughput": 1.0,
                "status_codes": {"200": 1},
            }), encoding="utf-8")
            self.assertIn(
                "status counts",
                _rpc_benchmark_content_error(artifacts, plan=plan),
            )
            vegeta.write_text(json.dumps({
                "requests": 1,
                "success": 1.0,
                "throughput": 1.0,
                "status_codes": {"200": 1},
            }), encoding="utf-8")
            proxy.write_text(
                "timestamp_ns,method_name,status_code,latency_ms\n"
                "1784851200000000000,eth_getBalance,200,2\n",
                encoding="utf-8",
            )
            self.assertIn(
                "workload count",
                _rpc_benchmark_content_error(artifacts, plan=plan),
            )
            proxy.write_text(
                "timestamp_ns,method_name,status_code,latency_ms\n"
                "1784851211000000000,eth_blockNumber,200,2\n",
                encoding="utf-8",
            )
            self.assertIn(
                "outside the summary",
                _rpc_benchmark_content_error(artifacts, plan=plan),
            )

    def test_sync_content_contract_requires_healthy_probe_and_real_png(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            performance = root / "performance.csv"
            health = root / "health.csv"
            chart = root / "timeline.png"
            summary = root / "summary.json"
            artifacts = {
                "performance_csv": str(performance),
                "sync_health_csv": str(health),
                "sync_timeline_chart": str(chart),
                "summary_json": str(summary),
            }
            self._write_summary(summary, mode="sync_observe", max_qps=0)
            header = (
                "timestamp,cpu_usage,mem_usage,net_total_mbps,local_block_height,"
                "local_health,sync_status,probe_error,execution_mgas_per_sec,"
                "execution_gas_per_sec,execution_metric_source,"
                "execution_metric_status,current_qps,qps_data_available,"
                "data_vda_total_iops,data_vda_avg_await\n"
            )
            performance.write_text(
                header
                + "2026-07-24T00:00:00Z,10,20,1,100,1,healthy,,2,2000000,"
                "chain_mgasps,available,0,false,1,0.5\n",
                encoding="utf-8",
            )
            health.write_text(
                "timestamp,local_block_height,sync_status,probe_error\n"
                "2026-07-24T00:00:00Z,100,healthy,\n",
                encoding="utf-8",
            )
            self._write_rgba_png(chart)
            self.assertEqual(_sync_observe_content_error(artifacts), "")

            performance.write_text(
                header
                + "2026-07-24T00:00:00Z,10,20,1,null,0,unhealthy,"
                "probe_failed,0,0,chain_mgasps,available,0,false,1,0.5\n",
                encoding="utf-8",
            )
            self.assertIn(
                "healthy observed node",
                _sync_observe_content_error(artifacts),
            )
            performance.write_text(
                header
                + "2026-07-24T00:00:00Z,10,20,1,100,1,healthy,,2,2000000,"
                "chain_mgasps,available,0,false,1,0.5\n",
                encoding="utf-8",
            )
            self._write_rgba_png(chart)
            corrupted = bytearray(chart.read_bytes())
            corrupted[-5] ^= 0xFF
            chart.write_bytes(corrupted)
            self.assertIn("invalid", _sync_observe_content_error(artifacts))

    def test_html_content_contract_rejects_placeholder_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            report = Path(tmpdir) / "report.html"
            report.write_text("evidence\n", encoding="utf-8")
            self.assertIn("complete report", _html_report_error(report, label="report"))

    def test_runtime_event_rejects_tampered_control_receipts_and_material_hashes(
        self,
    ) -> None:
        receipt = {
            "receipt_type": "semantic_planner",
            "turn_index": 1,
            "input_hash": "1" * 64,
            "pending_contract_hash": "2" * 64,
            "resolver_invoked": True,
            "result_reason_hash": "3" * 64,
            "planned_action_types": [],
            "semantic_units": [],
            "planner_metrics": {},
        }
        receipt["receipt_id"] = content_hash(receipt)
        event = RuntimeTurnEvent(
            schema_version=3,
            event_type="turn_committed",
            thread_id="runtime-test",
            session_purpose="chaos",
            before_fingerprint="a" * 64,
            after_fingerprint="b" * 64,
            turn_index=1,
            active_group="opening",
            pending_question_id="",
            action_queue_types=(),
            revision={"commit": "commit", "worktree_hash": "c" * 64},
            turn_receipt_summary={
                "turn_id": "turn-1",
                "input_hash": "d" * 64,
                "admitted_action_ids": [],
                "execution_order": [],
            },
            pending_transition={
                "before_hash": "e" * 64,
                "after_hash": "f" * 64,
            },
            control_receipts=(receipt,),
            material_state_diff_hashes={
                "active_group": {"before": "1" * 64, "after": "2" * 64}
            },
        )
        _validate_runtime_event(event)

        tampered_receipt = dict(receipt)
        tampered_receipt["resolver_invoked"] = False
        with self.assertRaisesRegex(ValueError, "receipt hash is stale"):
            _validate_runtime_event(
                event.__class__(
                    **{
                        **event.__dict__,
                        "control_receipts": (tampered_receipt,),
                    }
                )
            )

        with self.assertRaisesRegex(ValueError, "material state diff"):
            _validate_runtime_event(
                event.__class__(
                    **{
                        **event.__dict__,
                        "material_state_diff_hashes": {
                            "active_group": {
                                "before": "not-a-hash",
                                "after": "2" * 64,
                            }
                        },
                    }
                )
            )

        unknown_receipt = {
            "receipt_type": "unowned_observation",
            "turn_index": 1,
        }
        unknown_receipt["receipt_id"] = content_hash(unknown_receipt)
        with self.assertRaisesRegex(
            ValueError,
            "unregistered persisted control receipt type",
        ):
            _validate_runtime_event(
                event.__class__(
                    **{
                        **event.__dict__,
                        "control_receipts": (unknown_receipt,),
                    }
                )
            )

    def test_runtime_event_rejects_rehashed_semantically_invalid_coordinator_receipt(
        self,
    ) -> None:
        receipt = {
            "receipt_type": "pending_resolution",
            "turn_index": 1,
            "pending_id": "CLOUD_REGION",
            "pending_group": "provider_deployment",
            "pending_contract_hash": "1" * 64,
            "resolution_path": "exact_contract",
            "selected_option_id": "",
            "selected_value_hash": "2" * 64,
            "resolved_action_id": "action-1",
            "input_hash": "3" * 64,
            "normalizer": "exact_contract",
            "verdict": "accepted",
        }
        receipt["receipt_id"] = content_hash(receipt)
        receipt["verdict"] = "rejected"
        receipt["receipt_id"] = content_hash(
            {key: value for key, value in receipt.items() if key != "receipt_id"}
        )
        event = RuntimeTurnEvent(
            schema_version=3,
            event_type="turn_committed",
            thread_id="runtime-test",
            session_purpose="chaos",
            before_fingerprint="a" * 64,
            after_fingerprint="b" * 64,
            turn_index=1,
            active_group="provider_deployment",
            pending_question_id="CLOUD_REGION",
            action_queue_types=(),
            revision={"commit": "commit", "worktree_hash": "c" * 64},
            turn_receipt_summary={
                "turn_id": "turn-1",
                "input_hash": "d" * 64,
                "admitted_action_ids": [],
                "execution_order": [],
            },
            pending_transition={
                "before_hash": "e" * 64,
                "after_hash": "f" * 64,
            },
            control_receipts=(receipt,),
        )

        with self.assertRaisesRegex(
            ValueError,
            "pending-resolution receipt semantics are invalid",
        ):
            _validate_runtime_event(event)

    def test_runtime_event_rejects_rehashed_semantically_invalid_domain_receipt(
        self,
    ) -> None:
        from agent.harness.domains.rpc_receipts import emit_endpoint_role_receipt

        state = new_state("runtime-domain-receipt")
        state["turn_index"] = 1
        state["turn_context"] = {"text": "endpoint"}
        emit_endpoint_role_receipt(
            state,
            role="validation",
            case="custom_rpc",
            endpoint="https://example.invalid/private-token",
            ready=True,
            probe_status="ok",
            chain="bsc",
            adapter_family="jsonrpc",
        )
        receipt = dict(state["turn_context"]["control_receipts"][-1])
        receipt["owner"] = "analysis"
        receipt["receipt_id"] = content_hash(
            {key: value for key, value in receipt.items() if key != "receipt_id"}
        )
        event = RuntimeTurnEvent(
            schema_version=3,
            event_type="turn_committed",
            thread_id="runtime-test",
            session_purpose="chaos",
            before_fingerprint="a" * 64,
            after_fingerprint="b" * 64,
            turn_index=1,
            active_group="endpoint_process",
            pending_question_id="",
            action_queue_types=(),
            revision={"commit": "commit", "worktree_hash": "c" * 64},
            turn_receipt_summary={
                "turn_id": "turn-1",
                "input_hash": "d" * 64,
                "admitted_action_ids": [],
                "execution_order": [],
            },
            pending_transition={
                "before_hash": "e" * 64,
                "after_hash": "f" * 64,
            },
            control_receipts=(receipt,),
        )

        with self.assertRaisesRegex(
            ValueError,
            "runtime domain control receipt is invalid",
        ):
            _validate_runtime_event(event)

    def test_redaction_preserves_coverage_identity_that_names_secret_fields(self) -> None:
        edge_key = "chain_auxiliary_endpoints::RPC_API_KEY::contract-hash"
        secret = "runtime-secret-4821"
        postcondition = VerifiedPostcondition(
            verifier_id="unit",
            passed=True,
            observed_coverage_ids=(edge_key,),
            admitted_typed_actions=("answer_pending",),
            state_diff={"confirmed_config.RPC_API_KEY": secret},
            next_question_or_result={},
            details={
                "credential": f"RPC_API_KEY={secret}",
                "declared_target_results": {
                    edge_key: {"credential": f"RPC_API_KEY={secret}"},
                },
            },
        )
        observation = TurnObservation(
            seed=1,
            revision=self.revision,
            target_edge_key=edge_key,
            target_contract_hash="contract",
            target_variant_hash="variant",
            prior_agent_response="Enter the credential.",
            simulator_decision={
                "target_coverage_ids": [edge_key],
                "user_message": f"RPC_API_KEY={secret}",
            },
            exact_user_turn=f"RPC_API_KEY={secret}",
            provider="deepseek",
            model="deepseek-chat",
            before_turn_index=1,
            after_turn_index=2,
            before_state_fingerprint="a" * 64,
            after_state_fingerprint="b" * 64,
            pending_contract={},
            runtime_events=(),
            verified_postcondition=postcondition,
        )

        safe = _redacted_turn_observation(observation)
        serialized = json.dumps(safe, default=lambda value: value.__dict__)

        self.assertEqual(safe.target_edge_key, edge_key)
        self.assertEqual(safe.simulator_decision["target_coverage_ids"], [edge_key])
        self.assertEqual(safe.verified_postcondition.observed_coverage_ids, (edge_key,))
        self.assertIn(
            edge_key,
            safe.verified_postcondition.details["declared_target_results"],
        )
        self.assertNotIn(secret, serialized)
        self.assertIn("***REDACTED***", serialized)

    def setUp(self) -> None:
        self.edge = {
            "edge_key": "provider_deployment::MACHINE_TYPE::variant::valid_literal::manual",
            "contract_hash": "contract",
            "contract_variant_hash": "variant",
            "edge_type": "manual_input",
            "question_id": "MACHINE_TYPE",
            "expected_admitted": True,
            "evidence": {
                "deterministic": {
                    "required": True,
                    "applicability_reason": "unit-test deterministic contract",
                }
            },
        }
        self.revision = {"commit": "commit", "worktree_hash": "a" * 64}

    def _observed_turn(self) -> tuple[dict, dict, object, list]:
        seed = new_state("coverage-evidence", language="en", session_purpose="coverage")
        seed["confirmed_config"] = {"CLOUD_REGION": "test-region", "CLOUD_ZONE": "test-zone"}
        question = question_for_environment(seed, "provider_deployment")
        assert question is not None
        before = {
            **seed,
            "active_group": "provider_deployment",
            "pending_question": question,
            "last_user_input": "n2-standard-16",
        }
        with capture_coverage_events() as events:
            observation = observe_compiled_graph_turn(
                invoke_product_graph_turn,
                before,
                edge_key=self.edge["edge_key"],
                input_value="n2-standard-16",
            )
        return seed, before, observation, [event.as_dict() for event in events]

    def _artifact(self) -> dict:
        seed, before, observation, events = self._observed_turn()
        return build_evidence_artifact(
            edge=self.edge,
            evidence_class="deterministic",
            scenario_id="scenario",
            runner_type=COMPILED_GRAPH_RUNNER,
            revision=self.revision,
            input_value="n2-standard-16",
            seed_state=seed,
            before_state=before,
            after_state=observation.after,
            events=events,
            exit_status=0,
            outcome="passed",
        )

    def test_valid_artifact_requires_revision_exit_status_and_graph_observation(self) -> None:
        valid, reason = validate_evidence_artifact(self._artifact(), edge=self.edge, revision=self.revision)
        self.assertTrue(valid, reason)

    def test_deterministic_artifact_redacts_nested_state_before_hashing(self) -> None:
        secret = "abcdefghijklmnopqrstuvwxyz123456"
        seed = new_state("coverage-secret", language="en", session_purpose="coverage")
        seed["confirmed_config"] = {
            "CLOUD_REGION": "test-region",
            "CLOUD_ZONE": "test-zone",
            "LOCAL_RPC_URL": f"https://rpc.example/{secret}",
        }
        seed["runtime_secret"] = {
            "nested": (f"Authorization: Bearer {secret}",),
        }
        question = question_for_environment(seed, "provider_deployment")
        assert question is not None
        before = {
            **seed,
            "active_group": "provider_deployment",
            "pending_question": question,
            "last_user_input": "n2-standard-16",
        }
        with capture_coverage_events() as captured:
            observation = observe_compiled_graph_turn(
                invoke_product_graph_turn,
                before,
                edge_key=self.edge["edge_key"],
                input_value="n2-standard-16",
            )
        events = [event.as_dict() for event in captured]
        artifact = build_evidence_artifact(
            edge=self.edge,
            evidence_class="deterministic",
            scenario_id="redaction-scenario",
            runner_type=COMPILED_GRAPH_RUNNER,
            revision=self.revision,
            input_value="n2-standard-16",
            seed_state=seed,
            before_state=before,
            after_state=observation.after,
            events=events,
            exit_status=0,
            outcome="passed",
        )
        serialized = json.dumps(artifact, ensure_ascii=False)
        self.assertNotIn(secret, serialized)
        self.assertIn("***REDACTED***", serialized)
        self.assertTrue(artifact["content_redacted"])
        valid, reason = validate_evidence_artifact(
            artifact, edge=self.edge, revision=self.revision
        )
        self.assertTrue(valid, reason)

    def test_tampering_or_revision_change_invalidates_artifact(self) -> None:
        artifact = self._artifact()
        artifact["after_state_hash"] = "tampered"
        valid, _ = validate_evidence_artifact(artifact, edge=self.edge, revision=self.revision)
        self.assertFalse(valid)

        artifact = self._artifact()
        valid, _ = validate_evidence_artifact(
            artifact,
            edge=self.edge,
            revision={"commit": "other", "worktree_hash": "a" * 64},
        )
        self.assertFalse(valid)

    def test_empty_identity_or_malformed_exit_status_is_rejected(self) -> None:
        artifact = self._artifact()
        artifact["revision"] = {"commit": "", "worktree_hash": "a" * 64}
        valid, _ = validate_evidence_artifact(artifact, edge=self.edge, revision=artifact["revision"])
        self.assertFalse(valid)

        artifact = self._artifact()
        artifact["exit_status"] = "not-an-integer"
        valid, _ = validate_evidence_artifact(artifact, edge=self.edge, revision=self.revision)
        self.assertFalse(valid)

    def test_failed_outcome_requires_nonzero_exit_and_failure_event(self) -> None:
        seed, before, observation, events = self._observed_turn()
        artifact = build_evidence_artifact(
            edge=self.edge,
            evidence_class="deterministic",
            scenario_id="failed-scenario",
            runner_type=COMPILED_GRAPH_RUNNER,
            revision=self.revision,
            input_value="n2-standard-16",
            seed_state=seed,
            before_state=before,
            after_state=observation.after,
            events=events,
            exit_status=1,
            outcome="failed",
            error="AssertionError: reviewed postcondition failed",
        )
        valid, reason = validate_evidence_artifact(artifact, edge=self.edge, revision=self.revision)
        self.assertTrue(valid, reason)

    def test_observer_rejects_a_helper_that_only_claims_to_be_the_product_graph(self) -> None:
        with self.assertRaisesRegex(ValueError, "helper mismatch"):
            observe_compiled_graph_turn(
                lambda state: state,
                {"last_user_input": "x"},
                edge_key=self.edge["edge_key"],
                input_value="x",
            )

    def test_event_capture_is_scoped_and_does_not_mutate_runtime_state(self) -> None:
        state = {"confirmed_config": {"BLOCKCHAIN_NODE": "solana"}}
        with capture_coverage_events() as events:
            emit_coverage_event(
                "sample_observation",
                edge_key=self.edge["edge_key"],
                component="test",
                details={"state": state},
            )
        emit_coverage_event("outside_capture", edge_key="ignored", component="test")

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, "sample_observation")
        self.assertEqual(state, {"confirmed_config": {"BLOCKCHAIN_NODE": "solana"}})

    def test_real_execution_artifact_is_independent_hashed_job_evidence(self) -> None:
        scenario = next(
            item for item in EXECUTION_SCENARIOS
            if item.scenario_id == "rpc_real_node_smoke"
        )
        edge = {
            "edge_key": "@action_only/execution::::variant::action_only_transition::approve",
            "contract_hash": "execution-contract",
            "contract_variant_hash": "execution-variant",
            "action_type": "approve_preflight_smoke",
            "evidence": {
                "real_execution": {
                    "required": True,
                    "applicability_reason": "side-effect obligation",
                }
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            original_repo_root = coverage_evidence_module.REPO_ROOT
            coverage_evidence_module.REPO_ROOT = root
            self.addCleanup(
                setattr,
                coverage_evidence_module,
                "REPO_ROOT",
                original_repo_root,
            )
            jobs_dir = root / "jobs"
            job_dir = jobs_dir / "job-1"
            job_dir.mkdir(parents=True)
            approved_file = root / "approved-plan.json"
            admission_dir = jobs_dir / ".g5-admission"
            admission_dir.mkdir()
            envelope_file = admission_dir / "02-rpc_real_node_smoke.json"
            job_file = job_dir / "job.json"
            plan_file = job_dir / "plan.json"
            runtime_env_file = job_dir / "runtime.env"
            log_file = job_dir / "benchmark.log"
            approved_plan = {
                "plan_id": "plan-1",
                "chain": "bsc",
                "benchmark_mode": "quick",
                "rpc_mode": "single",
                "workflow_type": "rpc_benchmark",
                "use_fake_node": False,
                "chain_template_requirements": {
                    "single_method": "eth_blockNumber",
                    "mixed_weighted": [],
                },
                "execution": {
                    "command": ["./blockchain_node_benchmark.sh", "--quick", "--single"],
                    "environment": {
                        "LOCAL_RPC_URL": G5_RUNTIME_CONTRACT.rpc_url,
                        "NODE_PROMETHEUS_METRICS_URL": G5_RUNTIME_CONTRACT.metrics_url,
                        "QUICK_INITIAL_QPS": "1",
                        "QUICK_MAX_QPS": "1",
                        "QUICK_QPS_STEP": "1",
                        "QUICK_DURATION": "10",
                    },
                },
            }
            approved_file.write_text(json.dumps(approved_plan), encoding="utf-8")
            smoke_root = _smoke_execution_root(
                approved_file,
                jobs_dir,
                kind="real_node_smoke",
            )
            data_dir = smoke_root / "benchmark-data"
            data_dir.mkdir(parents=True)
            summary_file = data_dir / "summary.json"
            performance_file = data_dir / "performance.csv"
            html_file = data_dir / "report.html"
            proxy_file = data_dir / "proxy.csv"
            vegeta_file = data_dir / "vegeta.json"
            approved_hash = hashlib.sha256(approved_file.read_bytes()).hexdigest()
            endpoint_hash = hashlib.sha256(
                approved_plan["execution"]["environment"]["LOCAL_RPC_URL"].encode()
            ).hexdigest()
            metrics_hash = hashlib.sha256(
                G5_RUNTIME_CONTRACT.metrics_url.encode()
            ).hexdigest()
            envelope = {
                "schema_version": 1,
                "repository_revision": self.revision,
                "scenario_id": "rpc_real_node_smoke",
                "sequence_index": 2,
                "approved_plan_file": str(approved_file.resolve()),
                "approved_plan_sha256": approved_hash,
                "endpoint_identity_contract": {
                    "chain": "bsc",
                    "env_var": "LOCAL_RPC_URL",
                    "endpoint_sha256": endpoint_hash,
                    "probe_method": "eth_chainId",
                    "params_sha256": content_hash([]),
                    "expected_identity": "0x539",
                },
                "container_metrics_requirements": {
                    "compose_service": "geth-dev",
                    "image_digest": G5_RUNTIME_CONTRACT.image_digest,
                    "image_reference": G5_RUNTIME_CONTRACT.image_reference,
                    "rpc_url_sha256": endpoint_hash,
                    "metrics_env_var": "NODE_PROMETHEUS_METRICS_URL",
                    "metrics_url_sha256": metrics_hash,
                    "metrics_required": True,
                },
                "created_at": "2026-07-24T00:00:00Z",
                "g5_runtime_contract": G5_RUNTIME_CONTRACT.to_dict(),
                "g5_runtime_contract_sha256": content_hash(
                    G5_RUNTIME_CONTRACT.to_dict()
                ),
            }
            envelope_file.write_text(json.dumps(envelope), encoding="utf-8")
            envelope_hash = hashlib.sha256(envelope_file.read_bytes()).hexdigest()
            base_plan = {
                **approved_plan,
                "execution": {
                    **approved_plan["execution"],
                    "idempotency_key": "real_node_smoke:test",
                },
                "execution_provenance": {
                    "scenario_id": "rpc_real_node_smoke",
                    "operation": "real_node_smoke",
                    "approved_plan_file": str(approved_file.resolve()),
                    "approved_plan_sha256": approved_hash,
                    "runtime_overrides": [],
                },
            }
            job_plan = _real_node_smoke_plan(
                approved_file,
                smoke_root,
                plan=base_plan,
            )
            job_plan["execution_provenance"].update({
                "job_id": "job-1",
                "job_plan_file": str(plan_file.resolve()),
            })
            job_file.write_text(json.dumps({
                "job_id": "job-1",
                "status": "completed",
                "exit_code": 0,
                "error": "",
                "created_at": "2026-07-24T00:00:01Z",
            }), encoding="utf-8")
            plan_file.write_text(json.dumps(job_plan), encoding="utf-8")
            materialize_runtime_env(job_plan, job_dir)
            log_file.write_text("benchmark completed\n", encoding="utf-8")
            required_files = {
                "summary_json": summary_file,
                "performance_csv": performance_file,
                "html_report": html_file,
                "proxy_method_csv": proxy_file,
                "vegeta_json": vegeta_file,
            }
            summary_file.write_text(json.dumps({
                "run_id": "run-1",
                "benchmark_mode": "quick",
                "start_time": "2026-07-24T00:00:01Z",
                "end_time": "2026-07-24T00:00:10Z",
                "max_successful_qps": 1,
                "test_parameters": {
                    "initial_qps": 1,
                    "max_qps": 1,
                    "qps_step": 1,
                    "duration_per_level": 10,
                },
            }), encoding="utf-8")
            performance_file.write_text(
                "timestamp,current_qps,rpc_latency_ms,qps_data_available,cpu_usage,mem_usage\n"
                "2026-07-24T00:00:01Z,1,2.5,true,10,20\n",
                encoding="utf-8",
            )
            html_file.write_text(
                "<!doctype html><html><head><title>Benchmark report</title></head>"
                f"<body>{'observed benchmark data ' * 16}</body></html>",
                encoding="utf-8",
            )
            proxy_file.write_text(
                "timestamp_ns,method_name,status_code,latency_ms\n"
                + "".join(
                    f"{1784851201000000000 + index},eth_blockNumber,200,2.5\n"
                    for index in range(10)
                ),
                encoding="utf-8",
            )
            vegeta_file.write_text(json.dumps({
                "requests": 10,
                "success": 1.0,
                "throughput": 1.0,
                "status_codes": {"200": 10},
            }), encoding="utf-8")
            required_records = [
                {
                    "name": name,
                    "job_id": "job-1",
                    "owner": "benchmark_data",
                    "owner_root": str(data_dir.resolve()),
                    "relative_path": str(path.relative_to(data_dir)),
                    "path": str(path),
                    "resolved_path": str(path.resolve()),
                    "link_target": "",
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "size_bytes": path.stat().st_size,
                }
                for name, path in required_files.items()
            ]
            manifest_file = job_dir / "required-artifacts.sha256.json"
            manifest_file.write_text(json.dumps({
                "schema_version": 1,
                "job_id": "job-1",
                "scenario_id": "rpc_real_node_smoke",
                "job_plan_sha256": hashlib.sha256(plan_file.read_bytes()).hexdigest(),
                "owner_roots": {
                    "benchmark_data": str(data_dir.resolve()),
                    "job_control": str(job_dir.resolve()),
                },
                "owner_roots_sha256": content_hash({
                    "benchmark_data": str(data_dir.resolve()),
                    "job_control": str(job_dir.resolve()),
                }),
                "artifacts": required_records,
            }), encoding="utf-8")
            hashed = [
                {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                for path in (job_file, plan_file, manifest_file)
            ]
            log = {"path": str(log_file), "sha256": hashlib.sha256(log_file.read_bytes()).hexdigest()}
            request_contract = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "eth_chainId",
                "params_sha256": content_hash([]),
            }
            response_contract = {
                "jsonrpc": "2.0",
                "id": 1,
                "result_path": "$.result",
                "result_type": "hex_quantity",
                "result": "0x539",
            }
            endpoint_probe = {
                "chain": "bsc",
                "endpoint_env_var": "LOCAL_RPC_URL",
                "endpoint_sha256": endpoint_hash,
                "probe_method": "eth_chainId",
                "expected_identity": "0x539",
                "observed_identity": "0x539",
                "request_contract": request_contract,
                "request_contract_sha256": content_hash(request_contract),
                "response_contract": response_contract,
                "response_contract_sha256": content_hash(response_contract),
                "response_sha256": content_hash(response_contract),
                "http_status": 200,
                "verified": True,
                "probed_at": "2026-07-24T00:00:00Z",
            }
            container = {
                "container_id": "a" * 64,
                "image_digest": G5_RUNTIME_CONTRACT.image_digest,
                "image_reference": G5_RUNTIME_CONTRACT.image_reference,
                "compose_service": "geth-dev",
                "compose_project": "benchmark",
                "network_addresses": ["172.18.0.4"],
                "network_aliases": ["geth-dev"],
            }
            host_attestation = {
                "schema_version": 1,
                "artifact_type": "real_execution_host_attestation",
                "assurance": {
                    "kind": "docker_runtime_observation",
                    "cryptographic_identity": False,
                    "claim": "No cryptographic actor or host identity is asserted.",
                },
                "repository_revision": self.revision,
                "compose": {
                    "project": "benchmark",
                    "worker_service": "bench",
                    "target_service": "geth-dev",
                },
                "worker_container": {
                    "compose_project": "benchmark",
                    "compose_service": "bench",
                    "container_id": "b" * 64,
                    "container_name": "benchmark-bench-1",
                    "image_digest": "sha256:" + "c" * 64,
                    "image_reference": "benchmark:test",
                    "networks": {
                        "benchmark_default": {
                            "network_id": "d" * 64,
                            "endpoint_id": "e" * 64,
                            "gateway": "172.18.0.1",
                            "ip_address": "172.18.0.2",
                            "global_ipv6_address": "",
                            "aliases": ["bench"],
                        }
                    },
                },
                "target_container": {
                    "compose_project": "benchmark",
                    "compose_service": "geth-dev",
                    "container_id": container["container_id"],
                    "container_name": "benchmark-geth-dev-1",
                    "image_digest": container["image_digest"],
                    "image_reference": container["image_reference"],
                    "networks": {
                        "benchmark_default": {
                            "network_id": "d" * 64,
                            "endpoint_id": "f" * 64,
                            "gateway": "172.18.0.1",
                            "ip_address": "172.18.0.4",
                            "global_ipv6_address": "",
                            "aliases": ["geth-dev"],
                        }
                    },
                },
                "docker_inspect_sha256": {
                    "bench": "2" * 64,
                    "geth-dev": "3" * 64,
                },
                "worker": {
                    "argv": [
                        "/workspace/.venv-adk/bin/python",
                        "-m",
                        "tests.agent_live.execute_real_execution_ledger",
                    ],
                    "execution_boundary": "docker_compose_exec_linux_worker",
                },
                "attested_at": "2026-07-23T23:59:59Z",
            }
            host_attestation["worker"]["argv_sha256"] = hashlib.sha256(
                (
                    json.dumps(
                        {"argv": host_attestation["worker"]["argv"]},
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                ).encode()
            ).hexdigest()
            host_serialized = (
                json.dumps(
                    host_attestation,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode()
            host_hash = hashlib.sha256(host_serialized).hexdigest()
            host_file = root / (
                f"real-execution-host-attestation-{host_hash}.json"
            )
            host_file.write_bytes(host_serialized)
            runtime_attestation = {
                "schema_version": 1,
                "repository_revision": self.revision,
                "container": container,
                "container_contract_sha256": content_hash(container),
                "docker_inspect_sha256": "3" * 64,
                "host_attestation_file": str(host_file),
                "host_attestation_sha256": host_hash,
                "rpc_endpoint_sha256": endpoint_hash,
                "rpc_chain_id": "0x539",
                "rpc_response_sha256": content_hash(response_contract),
                "rpc_route": {
                    "scheme": "http",
                    "hostname": "geth-dev",
                    "port": 8545,
                    "path": "/",
                    "resolved_addresses": ["172.18.0.4"],
                },
                "metrics_route": {
                    "scheme": "http",
                    "hostname": "geth-dev",
                    "port": 6060,
                    "path": "/debug/metrics/prometheus",
                    "resolved_addresses": ["172.18.0.4"],
                },
                "metrics_probe": {
                    "metrics_url_sha256": metrics_hash,
                    "http_status": 200,
                    "content_type": "text/plain",
                    "body_sha256": "4" * 64,
                    "body_size_bytes": 128,
                    "non_comment_sample_count": 3,
                    "metric_family_count": 3,
                    "parser": "prometheus_text_v0.0.4",
                    "probed_at": "2026-07-24T00:00:00Z",
                },
                "attested_at": "2026-07-24T00:00:00Z",
            }
            runtime_attestation["attestation_sha256"] = content_hash(
                runtime_attestation
            )
            runtime_projection = validate_runtime_env_projection({
                "job_id": "job-1",
                "run_dir": str(job_dir.resolve()),
                "plan_file": str(plan_file.resolve()),
                "runtime_env_file": str(runtime_env_file.resolve()),
            })
            artifact = build_real_execution_evidence_artifact(
                edge=edge,
                revision=self.revision,
                scenario_id="rpc_real_node_smoke",
                operation_kind="preflight_smoke",
                request={
                    "scenario_id": "rpc_real_node_smoke",
                    "service_operation": "real_node_smoke",
                    "approved_plan_id": "plan-1",
                    "approved_plan_file": str(approved_file.resolve()),
                    "approved_plan_sha256": approved_hash,
                    "approved_plan_revision": self.revision,
                    "admission_envelope_file": str(envelope_file.resolve()),
                    "admission_envelope_sha256": envelope_hash,
                    "jobs_dir": str(jobs_dir.resolve()),
                    "jobs_root_admitted_at": "2026-07-24T00:00:00Z",
                    "required_artifact_hashes": required_records,
                    "endpoint_probe": endpoint_probe,
                    "runtime_attestation": runtime_attestation,
                    "runtime_env_projection": runtime_projection,
                    "ledger_sequence": 2,
                    "predecessor_scenario_id": "rpc_fake_node_smoke",
                },
                result={
                    "status": "completed",
                    "Authorization": "Bearer abcdefghijklmnopqrstuvwxyz123456",
                    "observed_job": {
                        "job_id": "job-1",
                        "run_dir": str(job_dir.resolve()),
                        "plan_file": str(plan_file.resolve()),
                        "runtime_env_file": str(runtime_env_file.resolve()),
                        "created_at": "2026-07-24T00:00:01Z",
                        "status": "completed",
                        "exit_code": 0,
                        "error": "",
                        "artifacts": {
                            name: str(path) for name, path in required_files.items()
                        },
                    },
                },
                job_id="job-1",
                job_artifacts=hashed,
                log_artifacts=(log,),
                started_at="2026-07-24T00:00:00Z",
                finished_at="2026-07-24T00:00:02Z",
            )
            serialized = json.dumps(artifact, ensure_ascii=False)
            self.assertNotIn("abcdefghijklmnopqrstuvwxyz123456", serialized)
            self.assertIn("***REDACTED***", serialized)
            self.assertTrue(artifact["content_redacted"])
            valid, reason = validate_real_execution_evidence_artifact(
                artifact,
                edge=edge,
                revision=self.revision,
            )
            self.assertTrue(valid, reason)

            original_summary = summary_file.read_bytes()
            summary_file.write_text("tampered\n", encoding="utf-8")
            valid, reason = validate_real_execution_evidence_artifact(
                artifact,
                edge=edge,
                revision=self.revision,
            )
            self.assertFalse(valid)
            self.assertIn("required-artifact hash mismatch", reason)
            summary_file.write_bytes(original_summary)

            original_manifest = manifest_file.read_bytes()
            manifest_file.write_text("{}\n", encoding="utf-8")
            valid, reason = validate_real_execution_evidence_artifact(
                artifact,
                edge=edge,
                revision=self.revision,
            )
            self.assertFalse(valid)
            self.assertIn("hashed artifact content mismatch", reason)
            manifest_file.write_bytes(original_manifest)

            original_approved = approved_file.read_bytes()
            approved_file.write_text("{}\n", encoding="utf-8")
            valid, reason = validate_real_execution_evidence_artifact(
                artifact,
                edge=edge,
                revision=self.revision,
            )
            self.assertFalse(valid)
            self.assertIn("approved plan hash mismatch", reason)
            approved_file.write_bytes(original_approved)

            semantic_mismatch = json.loads(json.dumps(artifact))
            semantic_mismatch["request"]["endpoint_probe"]["response_contract"]["result"] = "0x1"
            semantic_mismatch["request_hash"] = content_hash(semantic_mismatch["request"])
            valid, reason = validate_real_execution_evidence_artifact(
                semantic_mismatch,
                edge=edge,
                revision=self.revision,
            )
            self.assertFalse(valid)
            self.assertIn("response contract hash mismatch", reason)

            actual_identity_mismatch = json.loads(json.dumps(artifact))
            mismatched_probe = actual_identity_mismatch["request"]["endpoint_probe"]
            mismatched_probe["response_contract"]["result"] = "0x1"
            mismatched_probe["response_contract_sha256"] = content_hash(
                mismatched_probe["response_contract"]
            )
            mismatched_probe["response_sha256"] = content_hash(
                mismatched_probe["response_contract"]
            )
            mismatched_probe["observed_identity"] = "0x1"
            actual_identity_mismatch["request_hash"] = content_hash(
                actual_identity_mismatch["request"]
            )
            valid, reason = validate_real_execution_evidence_artifact(
                actual_identity_mismatch,
                edge=edge,
                revision=self.revision,
            )
            self.assertFalse(valid)
            self.assertIn("does not match approved identity", reason)

            original_envelope = envelope_file.read_bytes()
            envelope_file.write_text("{}\n", encoding="utf-8")
            valid, reason = validate_real_execution_evidence_artifact(
                artifact,
                edge=edge,
                revision=self.revision,
            )
            self.assertFalse(valid)
            self.assertIn("envelope hash mismatch", reason)
            envelope_file.write_bytes(original_envelope)

            runtime_mismatch = json.loads(json.dumps(artifact))
            runtime_mismatch["request"]["runtime_attestation"]["container"][
                "image_digest"
            ] = "sha256:" + "9" * 64
            runtime_mismatch["request_hash"] = content_hash(runtime_mismatch["request"])
            valid, reason = validate_real_execution_evidence_artifact(
                runtime_mismatch,
                edge=edge,
                revision=self.revision,
            )
            self.assertFalse(valid)
            self.assertIn("attestation hash mismatch", reason)

            secret_error = "benchmark exited 1: Bearer abcdefghijklmnopqrstuvwxyz123456"
            original_job = job_file.read_bytes()
            job_file.write_text(json.dumps({
                "job_id": "job-1",
                "status": "failed",
                "exit_code": 1,
                "error": secret_error,
                "created_at": "2026-07-24T00:00:01Z",
            }), encoding="utf-8")
            failure_hashed = [
                {
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                for path in (job_file, plan_file)
            ]
            observed_failure = build_real_execution_evidence_artifact(
                edge=edge,
                revision=self.revision,
                scenario_id="rpc_real_node_smoke",
                operation_kind="preflight_smoke",
                request={
                    **artifact["request"],
                },
                result={
                    **artifact["result"],
                    "status": "failed",
                    "observed_job": {
                        **artifact["result"]["observed_job"],
                        "status": "failed",
                        "exit_code": 1,
                        "error": secret_error,
                    },
                },
                job_id="job-1",
                job_artifacts=failure_hashed,
                log_artifacts=(log,),
                outcome="observed-fail",
                exit_status=1,
                error=secret_error,
                started_at="2026-07-24T00:00:00Z",
                finished_at="2026-07-24T00:00:02Z",
            )
            valid, reason = validate_real_execution_evidence_artifact(
                observed_failure,
                edge=edge,
                revision=self.revision,
            )
            self.assertFalse(valid)
            self.assertIn("does not qualify", reason)
            self.assertNotIn("abcdefghijklmnopqrstuvwxyz123456", json.dumps(observed_failure))
            valid, reason = validate_real_execution_evidence_artifact(
                observed_failure,
                edge=edge,
                revision=self.revision,
                allow_observed_failure=True,
            )
            self.assertTrue(valid, reason)
            job_file.write_bytes(original_job)

            artifact["result"]["status"] = "failed"
            valid, reason = validate_real_execution_evidence_artifact(
                artifact,
                edge=edge,
                revision=self.revision,
            )
            self.assertFalse(valid)
            self.assertIn("result hash mismatch", reason)

    def test_real_execution_ledger_requires_unique_serial_scenarios(self) -> None:
        revision = self.revision
        scenario_ids = (
            "rpc_fake_node_smoke",
            "rpc_real_node_smoke",
            "rpc_real_node_final",
            "sync_observe_bounded",
        )
        artifacts = []
        for index, scenario_id in enumerate(scenario_ids, start=1):
            scenario = next(
                item for item in EXECUTION_SCENARIOS
                if item.scenario_id == scenario_id
            )
            artifacts.append({
                "scenario_id": scenario_id,
                "job_id": f"job-{index}",
                "revision": revision,
                "outcome": "passed",
                "started_at": f"2026-07-24T00:00:0{index}Z",
                "finished_at": f"2026-07-24T00:00:0{index}Z",
                "request": {
                    "ledger_sequence": index,
                    "predecessor_scenario_id": g5_scenario_admission(
                        scenario_id
                    ).predecessor_scenario_id,
                    "endpoint_probe": {
                        "endpoint_sha256": "endpoint-a"
                        if scenario_id in {
                            "rpc_real_node_smoke",
                            "rpc_real_node_final",
                        }
                        else f"endpoint-{index}",
                    },
                    "runtime_attestation": (
                        {
                            "container": {
                                "container_id": "a" * 64,
                                "image_digest": "sha256:" + "b" * 64,
                            },
                            "host_attestation_file": "/workspace/.agent/evidence/"
                            "real-execution-host-attestation-" + "c" * 64 + ".json",
                            "host_attestation_sha256": "c" * 64,
                            "docker_inspect_sha256": "d" * 64,
                        }
                        if scenario_id != "rpc_fake_node_smoke"
                        else {}
                    ),
                    "runtime_env_projection": {
                        "execution_profile": {
                            "minimum_request_seconds": (
                                10
                                if scenario_id == "rpc_real_node_smoke"
                                else 30
                                if scenario_id == "rpc_real_node_final"
                                else 0
                            ),
                            "output_root_sha256": f"output-{index}",
                        },
                    },
                },
            })
        valid, reason = validate_real_execution_ledger_artifacts(
            artifacts,
            revision=revision,
        )
        self.assertTrue(valid, reason)

        duplicate = json.loads(json.dumps(artifacts))
        duplicate[2]["job_id"] = duplicate[1]["job_id"]
        valid, reason = validate_real_execution_ledger_artifacts(
            duplicate,
            revision=revision,
        )
        self.assertFalse(valid)
        self.assertIn("not unique", reason)

        final_failure = json.loads(json.dumps(artifacts))
        final_failure[-1]["outcome"] = "observed-fail"
        valid, reason = validate_real_execution_ledger_artifacts(
            final_failure,
            revision=revision,
        )
        self.assertFalse(valid)
        self.assertIn("four passing", reason)

        out_of_order = json.loads(json.dumps(artifacts))
        out_of_order[2]["started_at"] = "2026-07-23T23:59:59Z"
        valid, reason = validate_real_execution_ledger_artifacts(
            out_of_order,
            revision=revision,
        )
        self.assertFalse(valid)
        self.assertIn("fully serial", reason)

        final_scenario = next(
            scenario for scenario in EXECUTION_SCENARIOS
            if scenario.scenario_id == "rpc_real_node_final"
        )
        valid, reason = validate_real_execution_predecessor(
            artifacts[:2],
            scenario=final_scenario,
            started_at=artifacts[2]["started_at"],
            endpoint_probe=artifacts[2]["request"]["endpoint_probe"],
            runtime_attestation=artifacts[2]["request"]["runtime_attestation"],
        )
        self.assertTrue(valid, reason)
        valid, reason = validate_real_execution_predecessor(
            artifacts[:1],
            scenario=final_scenario,
            started_at=artifacts[2]["started_at"],
            endpoint_probe=artifacts[2]["request"]["endpoint_probe"],
            runtime_attestation=artifacts[2]["request"]["runtime_attestation"],
        )
        self.assertFalse(valid)
        self.assertIn("predecessor evidence is missing", reason)

    def test_real_execution_builder_rejects_missing_or_forged_files(self) -> None:
        edge = {
            "edge_key": "@action_only/execution::::variant::action_only_transition::approve",
            "contract_hash": "execution-contract",
            "contract_variant_hash": "execution-variant",
            "action_type": "approve_preflight_smoke",
            "evidence": {"real_execution": {"required": True, "applicability_reason": "side effect"}},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "job-forged"
            run_dir.mkdir()
            missing = {
                "path": str(run_dir / "job.json"),
                "sha256": "b" * 64,
            }
            with self.assertRaisesRegex(ValueError, "does not exist"):
                build_real_execution_evidence_artifact(
                    edge=edge,
                    revision=self.revision,
                    scenario_id="rpc_real_node_smoke",
                    operation_kind="preflight_smoke",
                    request={"approved_plan_id": "plan-1"},
                    result={
                        "status": "completed",
                        "observed_job": {"run_dir": str(run_dir)},
                    },
                    job_id="job-forged",
                    job_artifacts=(missing,),
                    log_artifacts=(missing,),
                )

    def test_artifact_builder_rejects_non_required_lane(self) -> None:
        edge = {
            **self.edge,
            "evidence": {
                "deterministic": {
                    "required": False,
                    "applicability_reason": "semantic-only row",
                }
            },
        }
        with self.assertRaisesRegex(ValueError, "not required for deterministic"):
            build_evidence_artifact(
                edge=edge,
                evidence_class="deterministic",
                scenario_id="scenario",
                runner_type=COMPILED_GRAPH_RUNNER,
                revision=self.revision,
                input_value="x",
                seed_state={},
                before_state={"last_user_input": "x"},
                after_state={},
                events=(),
                exit_status=0,
                outcome="passed",
            )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agent.runners.execution_scenarios import (
    EXECUTION_SCENARIOS,
    RPC_BENCHMARK_WORKFLOW,
    SYNC_OBSERVE_WORKFLOW,
)
from tests.agent_live.execute_real_execution_ledger import (
    G5ExecutionStageError,
    _admit_fresh_jobs_root,
    _attest_geth_dev_runtime,
    _build_admission_envelope,
    _edge_by_action,
    _job_evidence_files,
    _plan_file_for_scenario,
    _probe_endpoint_identity,
    _required_artifact_manifest,
    _stage_call,
    _validate_approved_plan_binding,
    _validate_fresh_job,
    _validate_source_plan,
    _write_g5_failure_artifact,
    execute_required_edges,
    validate_g5_failure_artifact,
)
from tests.agent_live.coverage_evidence import G5_RUNTIME_CONTRACT


def _runtime_host_attestation(
    *,
    revision: dict[str, str],
    container_ip: str = "172.18.0.4",
) -> dict:
    return {
        "repository_revision": revision,
        "target_container": {
            "container_id": "a" * 64,
            "image_digest": G5_RUNTIME_CONTRACT.image_digest,
            "image_reference": G5_RUNTIME_CONTRACT.image_reference,
            "compose_service": "geth-dev",
            "compose_project": "benchmark",
            "networks": {
                "benchmark": {
                    "ip_address": container_ip,
                    "aliases": ["geth-dev"],
                },
            },
        },
        "docker_inspect_sha256": {"geth-dev": "d" * 64},
        "attestation_file": "/workspace/.agent/evidence/host.json",
        "attestation_file_sha256": "e" * 64,
    }


def _artifact_job_fixture(
    root: Path,
    *,
    job_id: str,
    artifacts: dict[str, str],
) -> dict:
    jobs_root = root / "jobs"
    run_dir = jobs_root / job_id
    data_root = jobs_root / "outputs" / job_id / "benchmark-data"
    run_dir.mkdir(parents=True)
    data_root.mkdir(parents=True)
    plan = {
        "execution": {
            "environment": {
                "BLOCKCHAIN_BENCHMARK_DATA_DIR": str(data_root),
            },
        },
        "materialized_config": {
            "BLOCKCHAIN_BENCHMARK_DATA_DIR": str(data_root),
        },
    }
    (run_dir / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    (run_dir / "runtime.env").write_text(
        f"export BLOCKCHAIN_BENCHMARK_DATA_DIR='{data_root}'\n",
        encoding="utf-8",
    )
    return {
        "job_id": job_id,
        "run_dir": str(run_dir),
        "plan_file": str(run_dir / "plan.json"),
        "runtime_env_file": str(run_dir / "runtime.env"),
        "artifacts": artifacts,
        "_data_root": str(data_root),
    }


class RealExecutionLedgerRunnerTest(unittest.TestCase):
    def test_stage_failure_is_typed_and_written_as_content_addressed_evidence(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            plans = []
            for name in ("fake", "rpc", "sync"):
                path = root / f"{name}.json"
                path.write_text(json.dumps({"name": name}), encoding="utf-8")
                plans.append(path)
            failure = _write_g5_failure_artifact(
                evidence_dir=root / "evidence",
                revision={"commit": "a" * 40, "worktree_hash": "b" * 64},
                scenario_id="rpc_real_node_smoke",
                stage="endpoint_probe",
                error="Bearer abcdefghijklmnopqrstuvwxyz123456",
                plan_files=plans,
                host_attestation={
                    "attestation_file": str(root / "host.json"),
                    "attestation_file_sha256": "c" * 64,
                },
            )
            payload = json.loads(failure.read_text(encoding="utf-8"))
            observed_hash = hashlib.sha256(failure.read_bytes()).hexdigest()
            self.assertEqual(
                failure.name,
                f"g5-real-execution-failure-{observed_hash}.json",
            )
            self.assertEqual(payload["stage"], "endpoint_probe")
            self.assertEqual(payload["scenario_id"], "rpc_real_node_smoke")
            self.assertNotIn(
                "abcdefghijklmnopqrstuvwxyz123456",
                failure.read_text(encoding="utf-8"),
            )
            self.assertIn("***REDACTED***", payload["error"])
            valid, reason = validate_g5_failure_artifact(
                failure,
                revision={"commit": "a" * 40, "worktree_hash": "b" * 64},
            )
            self.assertTrue(valid, reason)
            failure.write_bytes(failure.read_bytes() + b" ")
            valid, reason = validate_g5_failure_artifact(
                failure,
                revision={"commit": "a" * 40, "worktree_hash": "b" * 64},
            )
            self.assertFalse(valid)
            self.assertIn("content-addressed", reason)

    def test_stage_call_preserves_failure_owner(self) -> None:
        with self.assertRaises(G5ExecutionStageError) as observed:
            _stage_call(
                "sync_observe_bounded",
                "runtime_attestation",
                lambda: (_ for _ in ()).throw(ValueError("invalid runtime")),
            )
        self.assertEqual(observed.exception.scenario_id, "sync_observe_bounded")
        self.assertEqual(observed.exception.stage, "runtime_attestation")

    def test_direct_script_entrypoint_loads_repository_modules(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [
                sys.executable,
                str(repo_root / "tests/agent_live/execute_real_execution_ledger.py"),
                "--help",
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--fake-plan", result.stdout)
        self.assertIn("--rpc-plan", result.stdout)
        self.assertIn("--sync-plan", result.stdout)
        self.assertNotIn("--expected-chain-id", result.stdout)

    def test_fake_node_is_a_first_class_real_execution_scenario(self) -> None:
        scenarios = {scenario.scenario_id: scenario for scenario in EXECUTION_SCENARIOS}
        self.assertEqual(set(scenarios), {
            "rpc_fake_node_smoke",
            "rpc_real_node_smoke",
            "rpc_real_node_final",
            "sync_observe_bounded",
        })
        fake = scenarios["rpc_fake_node_smoke"]
        self.assertTrue(fake.real_evidence_required)
        self.assertEqual(fake.operation_kind, "preflight_smoke")
        self.assertIn("--fake-node", fake.required_command_tokens)
        self.assertIn("--fake-node", scenarios["rpc_real_node_smoke"].forbidden_command_tokens)
        self.assertIn("--fake-node", scenarios["rpc_real_node_final"].forbidden_command_tokens)

    def test_product_scenarios_contain_no_g5_runtime_identity(self) -> None:
        serialized = json.dumps(
            [asdict(scenario) for scenario in EXECUTION_SCENARIOS],
            sort_keys=True,
            default=lambda value: sorted(value)
            if isinstance(value, (set, frozenset))
            else str(value),
        )
        forbidden_fields = {
            "trusted_chain_id",
            "trusted_image_digest",
            "trusted_image_reference",
            "trusted_rpc_url",
            "trusted_metrics_url",
            "trusted_compose_service",
            "g5_runtime_contract",
            "sequence_index",
            "predecessor_scenario_id",
            "endpoint_env_var",
        }
        self.assertTrue(all(
            forbidden_fields.isdisjoint(asdict(scenario))
            for scenario in EXECUTION_SCENARIOS
        ))
        self.assertNotIn("geth-dev", serialized)
        self.assertNotIn(G5_RUNTIME_CONTRACT.image_digest, serialized)
        self.assertNotIn(G5_RUNTIME_CONTRACT.chain_id, serialized)

    def test_scenarios_select_distinct_approved_plan_classes(self) -> None:
        scenarios = {scenario.scenario_id: scenario for scenario in EXECUTION_SCENARIOS}
        fake = Path("/plans/fake.json")
        rpc = Path("/plans/rpc.json")
        sync = Path("/plans/sync.json")
        self.assertEqual(
            _plan_file_for_scenario(
                scenarios["rpc_fake_node_smoke"],
                fake_plan_file=fake,
                rpc_plan_file=rpc,
                sync_plan_file=sync,
            ),
            fake,
        )
        self.assertEqual(
            _plan_file_for_scenario(
                scenarios["rpc_real_node_smoke"],
                fake_plan_file=fake,
                rpc_plan_file=rpc,
                sync_plan_file=sync,
            ),
            rpc,
        )
        self.assertEqual(
            _plan_file_for_scenario(
                scenarios["sync_observe_bounded"],
                fake_plan_file=fake,
                rpc_plan_file=rpc,
                sync_plan_file=sync,
            ),
            sync,
        )

    def test_source_plan_validation_rejects_fake_real_aliasing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            plan_file = Path(tmpdir) / "plan.json"
            plan_file.write_text(json.dumps({
                "workflow_type": RPC_BENCHMARK_WORKFLOW,
                "use_fake_node": True,
            }), encoding="utf-8")
            _validate_source_plan(
                plan_file,
                workflow_type=RPC_BENCHMARK_WORKFLOW,
                use_fake_node=True,
            )
            with self.assertRaisesRegex(RuntimeError, "use_fake_node=false"):
                _validate_source_plan(
                    plan_file,
                    workflow_type=RPC_BENCHMARK_WORKFLOW,
                    use_fake_node=False,
                )
            with self.assertRaisesRegex(RuntimeError, "expected sync_observe"):
                _validate_source_plan(
                    plan_file,
                    workflow_type=SYNC_OBSERVE_WORKFLOW,
                    use_fake_node=False,
                )

    def test_approved_plan_keeps_business_schema_and_requires_runtime_endpoints(self) -> None:
        scenario = next(
            item for item in EXECUTION_SCENARIOS
            if item.scenario_id == "rpc_real_node_smoke"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            plan_file = Path(tmpdir) / "approved.json"
            plan_file.write_text(json.dumps({
                "chain": "bsc",
                "workflow_type": RPC_BENCHMARK_WORKFLOW,
                "use_fake_node": False,
                "execution": {
                    "environment": {
                        "LOCAL_RPC_URL": G5_RUNTIME_CONTRACT.rpc_url,
                        "NODE_PROMETHEUS_METRICS_URL": G5_RUNTIME_CONTRACT.metrics_url,
                    },
                },
            }), encoding="utf-8")
            _validate_approved_plan_binding(
                plan_file,
                scenario=scenario,
            )
            payload = json.loads(plan_file.read_text(encoding="utf-8"))
            self.assertNotIn("execution_acceptance", payload)
            payload["execution"]["environment"].pop("NODE_PROMETHEUS_METRICS_URL")
            plan_file.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "metrics endpoint"):
                _validate_approved_plan_binding(
                    plan_file,
                    scenario=scenario,
                )

    def test_g5_envelope_remains_external_to_the_business_plan(self) -> None:
        scenario = next(
            item for item in EXECUTION_SCENARIOS
            if item.scenario_id == "rpc_real_node_smoke"
        )
        revision = {"commit": "a" * 40, "worktree_hash": "b" * 64}
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            jobs_dir = root / "jobs"
            jobs_dir.mkdir()
            plan_file = root / "approved.json"
            plan = {
                "chain": "bsc",
                "workflow_type": RPC_BENCHMARK_WORKFLOW,
                "use_fake_node": False,
                "execution": {
                    "environment": {
                        "LOCAL_RPC_URL": G5_RUNTIME_CONTRACT.rpc_url,
                        "NODE_PROMETHEUS_METRICS_URL": G5_RUNTIME_CONTRACT.metrics_url,
                    },
                },
            }
            plan_file.write_text(json.dumps(plan), encoding="utf-8")
            original = plan_file.read_bytes()
            envelope_file, envelope, envelope_hash = _build_admission_envelope(
                plan_file,
                plan,
                scenario,
                revision=revision,
                jobs_dir=jobs_dir,
            )
            self.assertEqual(plan_file.read_bytes(), original)
            self.assertNotIn("execution_acceptance", plan)
            self.assertEqual(
                envelope["approved_plan_sha256"],
                hashlib.sha256(original).hexdigest(),
            )
            self.assertEqual(
                envelope["endpoint_identity_contract"]["expected_identity"],
                G5_RUNTIME_CONTRACT.chain_id,
            )
            with self.assertRaises(FileExistsError):
                _build_admission_envelope(
                    plan_file,
                    plan,
                    scenario,
                    revision=revision,
                    jobs_dir=jobs_dir,
                )

    def test_endpoint_probe_binds_endpoint_hash_and_observed_identity(self) -> None:
        scenario = next(
            item for item in EXECUTION_SCENARIOS
            if item.scenario_id == "rpc_real_node_smoke"
        )
        plan = {
            "chain": "bsc",
            "execution": {"environment": {"LOCAL_RPC_URL": G5_RUNTIME_CONTRACT.rpc_url}},
        }
        envelope = {
            "endpoint_identity_contract": {
                "chain": "bsc",
                "env_var": "LOCAL_RPC_URL",
                "probe_method": "eth_chainId",
                "expected_identity": G5_RUNTIME_CONTRACT.chain_id,
            },
        }

        class Response:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            @staticmethod
            def read():
                return b'{"jsonrpc":"2.0","id":1,"result":"0x539"}'

        with patch(
            "tests.agent_live.execute_real_execution_ledger.urllib.request.urlopen",
            return_value=Response(),
        ):
            proof = _probe_endpoint_identity(plan, scenario, envelope)
        self.assertTrue(proof["verified"])
        self.assertEqual(proof["observed_identity"], "0x539")
        self.assertEqual(
            proof["endpoint_sha256"],
            hashlib.sha256(G5_RUNTIME_CONTRACT.rpc_url.encode()).hexdigest(),
        )
        self.assertEqual(proof["response_contract"]["result"], "0x539")
        self.assertEqual(
            proof["response_contract_sha256"],
            hashlib.sha256(
                json.dumps(
                    proof["response_contract"],
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
        )

        class ErrorResponse(Response):
            @staticmethod
            def read():
                return b'{"jsonrpc":"2.0","id":1,"error":{"code":-1,"message":"no"}}'

        with (
            patch(
                "tests.agent_live.execute_real_execution_ledger.urllib.request.urlopen",
                return_value=ErrorResponse(),
            ),
            self.assertRaisesRegex(RuntimeError, "unexpected shape"),
        ):
            _probe_endpoint_identity(plan, scenario, envelope)

    def test_geth_runtime_attestation_uses_host_identity_and_worker_probes(self) -> None:
        scenario = next(
            item for item in EXECUTION_SCENARIOS
            if item.scenario_id == "rpc_real_node_smoke"
        )
        revision = {"commit": "a" * 40, "worktree_hash": "b" * 64}
        plan = {
            "execution": {
                "environment": {
                    "LOCAL_RPC_URL": G5_RUNTIME_CONTRACT.rpc_url,
                    "NODE_PROMETHEUS_METRICS_URL": G5_RUNTIME_CONTRACT.metrics_url,
                },
            },
        }
        endpoint_probe = {
            "endpoint_sha256": "1" * 64,
            "observed_identity": "0x539",
            "response_sha256": "2" * 64,
        }
        envelope = {
            "container_metrics_requirements": {
                "compose_service": "geth-dev",
                "metrics_env_var": "NODE_PROMETHEUS_METRICS_URL",
                "image_digest": G5_RUNTIME_CONTRACT.image_digest,
                "image_reference": G5_RUNTIME_CONTRACT.image_reference,
            },
            "endpoint_identity_contract": {"env_var": "LOCAL_RPC_URL"},
        }
        metrics = {
            "metrics_url_sha256": hashlib.sha256(
                G5_RUNTIME_CONTRACT.metrics_url.encode()
            ).hexdigest(),
            "http_status": 200,
            "content_type": "text/plain",
            "body_sha256": "c" * 64,
            "body_size_bytes": 42,
            "non_comment_sample_count": 2,
            "metric_family_count": 2,
            "parser": "prometheus_text_v0.0.4",
            "probed_at": "2026-07-24T00:00:00Z",
        }
        attestation = _attest_geth_dev_runtime(
            plan,
            endpoint_probe,
            envelope,
            revision=revision,
            host_attestation=_runtime_host_attestation(revision=revision),
            metrics_probe=lambda *_args, **_kwargs: metrics,
            route_resolver=lambda endpoint: {
                "scheme": "http",
                "hostname": "geth-dev",
                "port": 8545 if endpoint == G5_RUNTIME_CONTRACT.rpc_url else 6060,
                "path": "/" if endpoint == G5_RUNTIME_CONTRACT.rpc_url else "/debug/metrics/prometheus",
                "resolved_addresses": ["172.18.0.4"],
            },
        )
        self.assertEqual(attestation["container"]["container_id"], "a" * 64)
        self.assertEqual(
            attestation["container"]["image_digest"],
            G5_RUNTIME_CONTRACT.image_digest,
        )
        self.assertEqual(attestation["metrics_probe"], metrics)

    def test_fresh_jobs_root_and_job_creation_time_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            jobs_dir = Path(tmpdir) / "jobs"
            admitted_at = _admit_fresh_jobs_root(jobs_dir)
            with self.assertRaisesRegex(RuntimeError, "must not already exist"):
                _admit_fresh_jobs_root(jobs_dir)
            job_dir = jobs_dir / "job-new"
            job_dir.mkdir()
            _validate_fresh_job(
                {
                    "job_id": "job-new",
                    "run_dir": str(job_dir),
                    "created_at": admitted_at,
                },
                jobs_dir=jobs_dir,
                admitted_at=admitted_at,
            )
            with self.assertRaisesRegex(RuntimeError, "predates"):
                _validate_fresh_job(
                    {
                        "job_id": "job-new",
                        "run_dir": str(job_dir),
                        "created_at": "2000-01-01T00:00:00Z",
                    },
                    jobs_dir=jobs_dir,
                    admitted_at=admitted_at,
                )

    def test_edge_lookup_requires_one_applicable_execution_edge(self) -> None:
        edge = {
            "action_type": "approve_preflight_smoke",
            "evidence": {"real_execution": {"required": True}},
        }
        self.assertIs(_edge_by_action({"edges": [edge]}, "approve_preflight_smoke"), edge)
        with self.assertRaisesRegex(RuntimeError, "found 0"):
            _edge_by_action({"edges": []}, "approve_preflight_smoke")

    def test_job_evidence_files_are_hashed_and_job_owned(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            job_id = "job_contract"
            run_dir = Path(tmpdir) / job_id
            run_dir.mkdir()
            for name, content in (
                ("job.json", json.dumps({"job_id": job_id})),
                ("plan.json", json.dumps({"plan_id": "plan"})),
                ("runtime.env", "BLOCKCHAIN_NODE=ethereum\n"),
                ("artifact-index.json", json.dumps({"job_id": job_id})),
                ("benchmark.log", "completed\n"),
            ):
                (run_dir / name).write_text(content, encoding="utf-8")
            job_artifacts, log_artifacts = _job_evidence_files({
                "job_id": job_id,
                "run_dir": str(run_dir),
                "runtime_env_file": str(run_dir / "runtime.env"),
                "artifact_index": str(run_dir / "artifact-index.json"),
            })
            self.assertEqual(len(job_artifacts), 4)
            self.assertEqual(len(log_artifacts), 1)
            self.assertTrue(all(len(item["sha256"]) == 64 for item in (*job_artifacts, *log_artifacts)))

    def test_job_evidence_files_require_runtime_env_and_artifact_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            job_id = "job_incomplete"
            run_dir = Path(tmpdir) / job_id
            run_dir.mkdir()
            (run_dir / "job.json").write_text("{}\n", encoding="utf-8")
            (run_dir / "plan.json").write_text("{}\n", encoding="utf-8")
            (run_dir / "benchmark.log").write_text("completed\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "runtime.env"):
                _job_evidence_files({
                    "job_id": job_id,
                    "run_dir": str(run_dir),
                })

    def test_job_evidence_rejects_symlink_and_path_escape(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            job_id = "job_symlink"
            run_dir = root / job_id
            run_dir.mkdir()
            outside = root / "outside.json"
            outside.write_text("{}\n", encoding="utf-8")
            for name in ("job.json", "plan.json", "artifact-index.json"):
                (run_dir / name).write_text("{}\n", encoding="utf-8")
            (run_dir / "runtime.env").symlink_to(outside)
            (run_dir / "benchmark.log").write_text("failed\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "symlink|escapes"):
                _job_evidence_files({
                    "job_id": job_id,
                    "run_dir": str(run_dir),
                    "runtime_env_file": str(run_dir / "runtime.env"),
                    "artifact_index": str(run_dir / "artifact-index.json"),
                })

    def test_required_artifact_manifest_hashes_csv_html_and_all_scenario_outputs(self) -> None:
        scenario = next(
            item for item in EXECUTION_SCENARIOS
            if item.scenario_id == "rpc_fake_node_smoke"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            job_id = "job_fake_contract"
            root = Path(tmpdir)
            job = _artifact_job_fixture(root, job_id=job_id, artifacts={})
            run_dir = Path(job["run_dir"])
            output_dir = Path(job["_data_root"])
            artifacts: dict[str, str] = {}
            for index, name in enumerate(scenario.required_artifacts):
                suffix = ".csv" if name.endswith("_csv") else ".html" if name == "html_report" else ".json"
                path = output_dir / f"{name}{suffix}"
                path.write_text(f"{name}-{index}\n", encoding="utf-8")
                artifacts[name] = str(path)
            job["artifacts"] = artifacts
            result = _required_artifact_manifest(job, scenario)
            self.assertEqual(
                {record["name"] for record in result["records"]},
                set(scenario.required_artifacts),
            )
            self.assertTrue(all(len(record["sha256"]) == 64 for record in result["records"]))
            self.assertTrue(
                all(record["owner"] == "benchmark_data" for record in result["records"])
            )
            self.assertEqual(result["manifest"]["job_id"], job_id)
            manifest_path = Path(result["manifest"]["path"])
            self.assertEqual(manifest_path.parent, run_dir)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["scenario_id"], scenario.scenario_id)

    def test_required_artifact_manifest_fails_closed_on_missing_output(self) -> None:
        scenario = next(
            item for item in EXECUTION_SCENARIOS
            if item.scenario_id == "rpc_fake_node_smoke"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            job_id = "job_fake_missing"
            job = _artifact_job_fixture(
                Path(tmpdir),
                job_id=job_id,
                artifacts={},
            )
            with self.assertRaisesRegex(RuntimeError, "summary_json"):
                _required_artifact_manifest(job, scenario)

    def test_required_artifact_manifest_rejects_wrong_owner(self) -> None:
        scenario = next(
            item for item in EXECUTION_SCENARIOS
            if item.scenario_id == "rpc_fake_node_smoke"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            job_id = "job_escape"
            job = _artifact_job_fixture(root, job_id=job_id, artifacts={})
            run_dir = Path(job["run_dir"])
            artifacts = {}
            for name in scenario.required_artifacts:
                path = run_dir / f"{name}.txt"
                path.write_text("evidence\n", encoding="utf-8")
                artifacts[name] = str(path)
            job["artifacts"] = artifacts
            with self.assertRaisesRegex(RuntimeError, "expected owner"):
                _required_artifact_manifest(job, scenario)

    def test_execute_required_edges_submits_all_four_first_class_scenarios(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            revision = {"commit": "a" * 40, "worktree_hash": "b" * 64}
            fake_plan = root / "fake.json"
            rpc_plan = root / "rpc.json"
            sync_plan = root / "sync.json"
            fake_plan.write_text(json.dumps({
                "chain": "bsc",
                "workflow_type": RPC_BENCHMARK_WORKFLOW,
                "use_fake_node": True,
                "execution": {"environment": {}},
            }), encoding="utf-8")
            rpc_plan.write_text(json.dumps({
                "chain": "bsc",
                "workflow_type": RPC_BENCHMARK_WORKFLOW,
                "use_fake_node": False,
                "execution": {
                    "environment": {
                        "LOCAL_RPC_URL": "http://geth-dev:8545",
                        "NODE_PROMETHEUS_METRICS_URL": "http://geth-dev:6060/debug/metrics/prometheus",
                    },
                },
            }), encoding="utf-8")
            sync_plan.write_text(json.dumps({
                "chain": "bsc",
                "workflow_type": SYNC_OBSERVE_WORKFLOW,
                "use_fake_node": False,
                "execution": {
                    "environment": {
                        "SYNC_OBSERVE_RPC_URL": "http://geth-dev:8545",
                        "NODE_PROMETHEUS_METRICS_URL": "http://geth-dev:6060/debug/metrics/prometheus",
                    },
                },
            }), encoding="utf-8")
            jobs_dir = root / "jobs"
            evidence_dir = root / "evidence"
            requests = []

            class Service:
                def execute(self, request):
                    requests.append(request)
                    job_id = f"job_{request.operation.value}"
                    return SimpleNamespace(
                        succeeded=True,
                        reused=False,
                        to_dict=lambda: {
                            "status": "ok",
                            "data": {"job": {"job_id": job_id}},
                        },
                    )

            edges = [
                {
                    "edge_key": "preflight",
                    "contract_hash": "preflight-contract",
                    "contract_variant_hash": "preflight-variant",
                    "action_type": "approve_preflight_smoke",
                    "evidence": {"real_execution": {"required": True}},
                },
                {
                    "edge_key": "final",
                    "contract_hash": "final-contract",
                    "contract_variant_hash": "final-variant",
                    "action_type": "approve_final_benchmark",
                    "evidence": {"real_execution": {"required": True}},
                },
            ]
            written_paths = iter(evidence_dir / f"{index}.json" for index in range(4))

            with (
                patch(
                    "tests.agent_live.execute_real_execution_ledger.BenchmarkExecutionService",
                    return_value=Service(),
                ),
                patch(
                    "tests.agent_live.execute_real_execution_ledger.repository_revision",
                    return_value=revision,
                ),
                patch(
                    "tests.agent_live.execute_real_execution_ledger.build_ledger",
                    return_value={"edges": edges},
                ),
                patch(
                    "tests.agent_live.execute_real_execution_ledger._require_current_revision",
                ),
                patch(
                    "tests.agent_live.execute_real_execution_ledger._probe_endpoint_identity",
                    return_value={},
                ),
                patch(
                    "tests.agent_live.execute_real_execution_ledger._attest_geth_dev_runtime",
                    return_value={},
                ),
                patch(
                    "tests.agent_live.execute_real_execution_ledger._wait_for_job",
                    side_effect=lambda job_id, **_: {
                        "job_id": job_id,
                        "run_dir": str(jobs_dir / job_id),
                        "created_at": "2026-07-24T00:00:00Z",
                        "status": "completed",
                        "exit_code": 0,
                        "artifacts": {},
                    },
                ),
                patch(
                    "tests.agent_live.execute_real_execution_ledger._validate_fresh_job",
                ),
                patch(
                    "tests.agent_live.execute_real_execution_ledger._required_artifact_manifest",
                    return_value={
                        "manifest": {
                            "job_id": "job",
                            "path": "/tmp/manifest",
                            "sha256": "c" * 64,
                            "size_bytes": 1,
                        },
                        "records": [{"name": "html_report", "sha256": "d" * 64}],
                    },
                ),
                patch(
                    "tests.agent_live.execute_real_execution_ledger._job_evidence_files",
                    return_value=(
                        [{"path": str(root / "mock-job.json")}],
                        [{"path": str(root / "mock-log.txt")}],
                    ),
                ),
                patch(
                    "tests.agent_live.execute_real_execution_ledger.validate_runtime_env_projection",
                    return_value={"projection_receipt_id": "runtime"},
                ),
                patch(
                    "tests.agent_live.execute_real_execution_ledger.build_real_execution_evidence_artifact",
                    side_effect=lambda **kwargs: {"scenario_id": kwargs["scenario_id"]},
                ),
                patch(
                    "tests.agent_live.execute_real_execution_ledger.validate_real_execution_evidence_artifact",
                    return_value=(True, ""),
                ),
                patch(
                    "tests.agent_live.execute_real_execution_ledger.validate_real_execution_predecessor",
                    return_value=(True, ""),
                ),
                patch(
                    "tests.agent_live.execute_real_execution_ledger.validate_real_execution_ledger_artifacts",
                    return_value=(True, ""),
                ),
                patch(
                    "tests.agent_live.execute_real_execution_ledger.write_evidence_artifact",
                    side_effect=lambda *_: next(written_paths),
                ),
            ):
                written = execute_required_edges(
                    fake_plan_file=fake_plan,
                    rpc_plan_file=rpc_plan,
                    sync_plan_file=sync_plan,
                    jobs_dir=jobs_dir,
                    evidence_dir=evidence_dir,
                    timeout_seconds=1,
                    host_attestation={},
                )

            self.assertEqual(len(written), 4)
            self.assertEqual(
                [request.operation.value for request in requests],
                [
                    "fake_node_smoke",
                    "real_node_smoke",
                    "final_benchmark",
                    "sync_observe",
                ],
            )
            self.assertEqual(
                [Path(request.plan_file) for request in requests],
                [fake_plan, rpc_plan, rpc_plan, sync_plan],
            )
            self.assertNotIn("execution_acceptance", json.loads(rpc_plan.read_text()))

    def test_failed_job_writes_observed_fail_and_stops_the_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            revision = {"commit": "a" * 40, "worktree_hash": "b" * 64}
            fake_plan = root / "fake.json"
            rpc_plan = root / "rpc.json"
            sync_plan = root / "sync.json"
            fake_plan.write_text(json.dumps({
                "chain": "bsc",
                "workflow_type": RPC_BENCHMARK_WORKFLOW,
                "use_fake_node": True,
                "execution": {"environment": {}},
            }), encoding="utf-8")
            rpc_plan.write_text(json.dumps({
                "chain": "bsc",
                "workflow_type": RPC_BENCHMARK_WORKFLOW,
                "use_fake_node": False,
                "execution": {
                    "environment": {
                        "LOCAL_RPC_URL": "http://geth-dev:8545",
                        "NODE_PROMETHEUS_METRICS_URL": "http://geth-dev:6060/debug/metrics/prometheus",
                    },
                },
            }), encoding="utf-8")
            sync_plan.write_text(json.dumps({
                "chain": "bsc",
                "workflow_type": SYNC_OBSERVE_WORKFLOW,
                "use_fake_node": False,
                "execution": {
                    "environment": {
                        "SYNC_OBSERVE_RPC_URL": "http://geth-dev:8545",
                        "NODE_PROMETHEUS_METRICS_URL": "http://geth-dev:6060/debug/metrics/prometheus",
                    },
                },
            }), encoding="utf-8")
            jobs_dir = root / "jobs"
            evidence_dir = root / "evidence"
            edge = {
                "edge_key": "preflight",
                "contract_hash": "contract",
                "contract_variant_hash": "variant",
                "action_type": "approve_preflight_smoke",
                "evidence": {"real_execution": {"required": True}},
            }
            service_result = SimpleNamespace(
                succeeded=True,
                reused=False,
                to_dict=lambda: {
                    "status": "ok",
                    "data": {"job": {"job_id": "job-failed"}},
                },
            )
            build = "tests.agent_live.execute_real_execution_ledger.build_real_execution_evidence_artifact"
            with (
                patch(
                    "tests.agent_live.execute_real_execution_ledger.repository_revision",
                    return_value=revision,
                ),
                patch(
                    "tests.agent_live.execute_real_execution_ledger._require_current_revision",
                ),
                patch(
                    "tests.agent_live.execute_real_execution_ledger.build_ledger",
                    return_value={"edges": [edge]},
                ),
                patch(
                    "tests.agent_live.execute_real_execution_ledger.BenchmarkExecutionService",
                ) as service,
                patch(
                    "tests.agent_live.execute_real_execution_ledger._wait_for_job",
                    return_value={
                        "job_id": "job-failed",
                        "run_dir": str(jobs_dir / "job-failed"),
                        "created_at": "2026-07-24T00:00:00Z",
                        "status": "failed",
                        "exit_code": 2,
                        "error": "runner failed",
                    },
                ),
                patch(
                    "tests.agent_live.execute_real_execution_ledger._validate_fresh_job",
                ),
                patch(
                    "tests.agent_live.execute_real_execution_ledger._job_evidence_files",
                    return_value=(
                        [{"path": str(root / "mock-job.json")}],
                        [{"path": str(root / "mock-log.txt")}],
                    ),
                ),
                patch(
                    "tests.agent_live.execute_real_execution_ledger.validate_runtime_env_projection",
                    return_value={"projection_receipt_id": "runtime"},
                ),
                patch(build, return_value={"outcome": "observed-fail"}) as builder,
                patch(
                    "tests.agent_live.execute_real_execution_ledger.validate_real_execution_evidence_artifact",
                    return_value=(True, ""),
                ),
                patch(
                    "tests.agent_live.execute_real_execution_ledger.write_evidence_artifact",
                    return_value=evidence_dir / "observed-fail.json",
                ),
            ):
                service.return_value.execute.return_value = service_result
                with self.assertRaisesRegex(RuntimeError, "observed-fail evidence"):
                    execute_required_edges(
                        fake_plan_file=fake_plan,
                        rpc_plan_file=rpc_plan,
                        sync_plan_file=sync_plan,
                        jobs_dir=jobs_dir,
                        evidence_dir=evidence_dir,
                        timeout_seconds=1,
                        host_attestation={},
                    )
            self.assertEqual(builder.call_args.kwargs["outcome"], "observed-fail")
            self.assertEqual(builder.call_args.kwargs["exit_status"], 2)


if __name__ == "__main__":
    unittest.main()

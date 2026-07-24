from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agent.runners.execution_scenarios import (
    EXECUTION_SCENARIOS,
    RPC_BENCHMARK_WORKFLOW,
    SYNC_OBSERVE_WORKFLOW,
)
from tests.agent_live.execute_real_execution_ledger import (
    _edge_by_action,
    _job_evidence_files,
    _plan_file_for_scenario,
    _required_artifact_manifest,
    _validate_source_plan,
    execute_required_edges,
)


class RealExecutionLedgerRunnerTest(unittest.TestCase):
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

    def test_required_artifact_manifest_hashes_csv_html_and_all_scenario_outputs(self) -> None:
        scenario = next(
            item for item in EXECUTION_SCENARIOS
            if item.scenario_id == "rpc_fake_node_smoke"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            job_id = "job_fake_contract"
            run_dir = Path(tmpdir) / job_id
            output_dir = Path(tmpdir) / "isolated-output"
            run_dir.mkdir()
            output_dir.mkdir()
            artifacts: dict[str, str] = {}
            for index, name in enumerate(scenario.required_artifacts):
                suffix = ".csv" if name.endswith("_csv") else ".html" if name == "html_report" else ".json"
                path = output_dir / f"{name}{suffix}"
                path.write_text(f"{name}-{index}\n", encoding="utf-8")
                artifacts[name] = str(path)
            result = _required_artifact_manifest({
                "job_id": job_id,
                "run_dir": str(run_dir),
                "artifacts": artifacts,
            }, scenario)
            self.assertEqual(
                {record["name"] for record in result["records"]},
                set(scenario.required_artifacts),
            )
            self.assertTrue(all(len(record["sha256"]) == 64 for record in result["records"]))
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
            run_dir = Path(tmpdir) / job_id
            run_dir.mkdir()
            with self.assertRaisesRegex(RuntimeError, "summary_json"):
                _required_artifact_manifest({
                    "job_id": job_id,
                    "run_dir": str(run_dir),
                    "artifacts": {},
                }, scenario)

    def test_execute_required_edges_submits_all_four_first_class_scenarios(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fake_plan = root / "fake.json"
            rpc_plan = root / "rpc.json"
            sync_plan = root / "sync.json"
            fake_plan.write_text(json.dumps({
                "workflow_type": RPC_BENCHMARK_WORKFLOW,
                "use_fake_node": True,
            }), encoding="utf-8")
            rpc_plan.write_text(json.dumps({
                "workflow_type": RPC_BENCHMARK_WORKFLOW,
                "use_fake_node": False,
            }), encoding="utf-8")
            sync_plan.write_text(json.dumps({
                "workflow_type": SYNC_OBSERVE_WORKFLOW,
                "use_fake_node": False,
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
            revision = {"commit": "a" * 40, "worktree_hash": "b" * 64}
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
                    "tests.agent_live.execute_real_execution_ledger._wait_for_job",
                    side_effect=lambda job_id, **_: {
                        "job_id": job_id,
                        "run_dir": str(jobs_dir / job_id),
                        "status": "completed",
                        "exit_code": 0,
                        "artifacts": {},
                    },
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
                    return_value=([{"path": "/tmp/job"}], [{"path": "/tmp/log"}]),
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


if __name__ == "__main__":
    unittest.main()

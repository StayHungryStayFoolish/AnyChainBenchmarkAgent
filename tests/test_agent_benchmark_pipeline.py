"""Regression tests for Phase 5: fixing the Harness -> ADK dependency

inversion (architecture audit Finding E). `agent.runners.benchmark_pipeline`
owns the prepare/smoke/submit business logic that both
`agent.harness.domains.execution_runtime` and `agent.tools.executor` (the CLI
tool-call/tool-schema surface) call into directly, instead of either caller
reaching through a duplicate ADK wrapper layer (retired; see
`agent/README.md`'s "Retired files must not return").
"""

from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path


class BenchmarkPipelineTest(unittest.TestCase):
    def test_sync_observe_contract_materializes_rpc_and_metrics_end_to_end(self) -> None:
        from agent.planners.strategy_planner import generate_plan
        from agent.runners.benchmark_pipeline import _structured_request

        request = _structured_request(
            source_prompt="sync observation",
            chain="ethereum",
            goal="baseline",
            rpc_mode="",
            use_fake_node=None,
            deployment_type="container",
            cloud_provider="other",
            target_rpc_url="",
            mainnet_rpc_url="",
            mainnet_rpc_url_reviewed=True,
            ledger_device="vda",
            accounts_device="",
            blockchain_process_names=[],
            cloud_region="test-region",
            cloud_zone="test-zone",
            machine_type="test-machine",
            data_vol_type="ssd",
            data_vol_size="100",
            data_vol_max_iops="3000",
            data_vol_max_throughput="125",
            accounts_vol_type="",
            accounts_vol_size="",
            accounts_vol_max_iops="",
            accounts_vol_max_throughput="",
            network_interface="eth0",
            network_max_bandwidth_gbps="10",
            qps_initial=None,
            qps_max=None,
            qps_step=None,
            duration_seconds=None,
            rpc_methods=None,
            mixed_weights=None,
            observability_enabled=False,
            observability_mode="disabled",
            observability_auto_stop=True,
            workflow_type="sync_observe",
            sync_observe_rpc_url="http://geth-dev:8545",
            sync_observe_stop_condition="duration",
            sync_observe_duration_seconds=15,
            sync_observe_source="endpoint_only",
            node_prometheus_metrics_url="http://geth-dev:6060/debug/metrics/prometheus",
            exporter_port="9108",
            prometheus_port="9091",
            grafana_port="3001",
            confirmations=["sync_observe_stop_condition"],
            assumed_values=None,
            assumed_for_smoke=False,
        )
        plan = generate_plan(request, discovery={"deployment": {"type": "container"}, "cloud": {"provider": "other"}})

        env = plan["execution"]["environment"]
        materialized = plan["materialized_config"]
        self.assertEqual(request["local_rpc_url"], "http://geth-dev:8545")
        self.assertEqual(env["LOCAL_RPC_URL"], "http://geth-dev:8545")
        self.assertEqual(env["SYNC_OBSERVE_RPC_URL"], "http://geth-dev:8545")
        self.assertEqual(env["NODE_PROMETHEUS_METRICS_URL"], "http://geth-dev:6060/debug/metrics/prometheus")
        self.assertEqual(materialized["SYNC_OBSERVE_RPC_URL"], "http://geth-dev:8545")
        self.assertEqual(materialized["NODE_PROMETHEUS_METRICS_URL"], "http://geth-dev:6060/debug/metrics/prometheus")
        self.assertEqual(plan["execution"]["command"], ["./blockchain_node_benchmark.sh", "--sync-observe", "--duration", "15"])

    def test_running_job_is_never_graded_from_unrelated_archive_evidence(self) -> None:
        from agent.analyzers.result_analyzer import analyze_job

        result = analyze_job({
            "job_id": "job_running_test",
            "status": "running",
            "artifacts": {},
        })

        self.assertEqual(result["grade"], "IN_PROGRESS")
        self.assertEqual(result["status"], "running")
        self.assertNotEqual(result["grade"], "PASS")

    def test_running_job_render_marks_artifacts_pending_instead_of_missing(self) -> None:
        from agent.harness.domains.analysis import _render_persisted_job_facts

        rendered = _render_persisted_job_facts({
            "job_id": "job_running_test",
            "status": "running",
            "grade": "IN_PROGRESS",
            "chain": "bsc",
            "target_mode": "fake-node",
            "rpc_mode": "single",
            "methods": ["eth_getBalance"],
            "benchmark_mode": "quick",
            "initial_qps": 1,
            "max_qps": 1,
            "duration": 10,
            "artifacts": {},
            "run_dir": "/tmp/job_running_test",
        }, "en")

        self.assertIn("generated after completion", rendered)
        self.assertNotIn("<not found>", rendered)

    def test_completed_status_without_required_artifacts_does_not_claim_closed_loop(self) -> None:
        from agent.harness.domains.analysis import _render_persisted_job_facts

        rendered = _render_persisted_job_facts({
            "job_id": "job_incomplete_artifacts",
            "status": "completed",
            "grade": "INCONCLUSIVE",
            "chain": "bsc",
            "target_mode": "fake-node",
            "rpc_mode": "single",
            "methods": ["eth_getBalance"],
            "benchmark_mode": "quick",
            "initial_qps": 1,
            "max_qps": 1,
            "duration": 10,
            "artifacts": {},
            "run_dir": "/tmp/job_incomplete_artifacts",
        }, "en")

        self.assertNotIn("Closed-loop result", rendered)

    def test_preflight_blocks_job_local_custom_method_without_fake_node_fixture(self) -> None:
        from agent.planners.preflight import run_preflight

        plan = {
            "chain": "bsc",
            "use_fake_node": True,
            "rpc_mode": "single",
            "required_inputs": [],
            "configuration_checklist": {"missing_blockers": []},
            "chain_config_override": {
                "_meta": {"adapter_family": "jsonrpc"},
                "rpc_methods": {"single": "eth_accounts"},
            },
            "materialized_config": {},
            "discovery": {},
        }
        result = run_preflight(plan)
        fixture_check = next(
            item for item in result["checks"] if item["name"] == "effective_workload_fixtures_available"
        )
        self.assertFalse(fixture_check["passed"])
        self.assertIn("eth_accounts", fixture_check["detail"])

    def test_preflight_accepts_job_local_method_with_existing_fake_node_fixture(self) -> None:
        from agent.planners.preflight import run_preflight

        plan = {
            "chain": "bsc",
            "use_fake_node": True,
            "rpc_mode": "single",
            "required_inputs": [],
            "configuration_checklist": {"missing_blockers": []},
            "chain_config_override": {
                "_meta": {"adapter_family": "jsonrpc"},
                "rpc_methods": {"single": "eth_blockNumber"},
            },
            "materialized_config": {},
            "discovery": {},
        }
        result = run_preflight(plan)
        fixture_check = next(
            item for item in result["checks"] if item["name"] == "effective_workload_fixtures_available"
        )
        self.assertTrue(fixture_check["passed"])

    def test_preflight_checks_every_enabled_method_in_custom_mixed_workload(self) -> None:
        from agent.planners.preflight import run_preflight

        plan = {
            "chain": "bsc",
            "use_fake_node": True,
            "rpc_mode": "mixed",
            "required_inputs": [],
            "configuration_checklist": {"missing_blockers": []},
            "chain_config_override": {
                "_meta": {"adapter_family": "jsonrpc"},
                "rpc_methods": {
                    "mixed_weighted": [
                        {"method": "eth_blockNumber", "weight": 50},
                        {"method": "eth_accounts", "weight": 50},
                    ],
                },
            },
            "chain_template_requirements": {
                "mixed_weighted": [
                    {"method": "eth_blockNumber", "weight": 50},
                    {"method": "eth_accounts", "weight": 50},
                ],
            },
            "materialized_config": {},
            "discovery": {},
        }
        result = run_preflight(plan)
        fixture_check = next(
            item for item in result["checks"] if item["name"] == "effective_workload_fixtures_available"
        )
        self.assertFalse(fixture_check["passed"])
        self.assertIn("eth_accounts", fixture_check["detail"])
        self.assertNotIn("eth_blockNumber ->", fixture_check["detail"])

    def test_zero_success_vegeta_and_missing_archive_fail_business_result(self) -> None:
        from agent.runners.result_status import classify_benchmark_result

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for name in ("performance.csv", "report.html", "proxy.csv"):
                (root / name).write_text("evidence\n", encoding="utf-8")
            vegeta = root / "vegeta.json"
            vegeta.write_text(json.dumps({"requests": 10, "success": 0, "status_codes": {"404": 10}}))
            result = classify_benchmark_result(
                {"workflow_type": "rpc_benchmark"},
                0,
                {
                    "performance_csv": str(root / "performance.csv"),
                    "html_report": str(root / "report.html"),
                    "proxy_method_csv": str(root / "proxy.csv"),
                    "vegeta_json": str(vegeta),
                },
            )
        self.assertEqual(result["status"], "failed")
        self.assertTrue(result["executed"])
        self.assertFalse(result["workload_passed"])
        self.assertFalse(result["artifacts_complete"])
        self.assertTrue(any("zero successful" in item for item in result["failures"]))
        self.assertTrue(any("archive_dir" in item for item in result["failures"]))

    def test_successful_workload_with_missing_report_is_partial_not_completed(self) -> None:
        from agent.runners.result_status import classify_benchmark_result

        with tempfile.TemporaryDirectory() as tmpdir:
            vegeta = Path(tmpdir) / "vegeta.json"
            vegeta.write_text(json.dumps({"requests": 10, "success": 1, "status_codes": {"200": 10}}))
            result = classify_benchmark_result(
                {"workflow_type": "rpc_benchmark"},
                0,
                {"vegeta_json": str(vegeta)},
            )
        self.assertEqual(result["status"], "partial")
        self.assertTrue(result["executed"])
        self.assertTrue(result["workload_passed"])
        self.assertFalse(result["artifacts_complete"])

    def test_complete_rpc_result_requires_successful_workload_and_all_artifacts(self) -> None:
        from agent.runners.result_status import classify_benchmark_result

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = {}
            for key, name in {
                "summary_json": "summary.json",
                "performance_csv": "performance.csv",
                "html_report": "report.html",
                "proxy_method_csv": "proxy.csv",
            }.items():
                path = root / name
                path.write_text("evidence\n", encoding="utf-8")
                paths[key] = str(path)
            vegeta = root / "vegeta.json"
            vegeta.write_text(json.dumps({"requests": 10, "success": 1, "status_codes": {"200": 10}}))
            paths.update({"archive_dir": str(root), "vegeta_json": str(vegeta)})
            result = classify_benchmark_result({"workflow_type": "rpc_benchmark"}, 0, paths)
        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["passed"])

    def test_sync_observe_result_requires_observed_metrics_without_rpc_artifacts(self) -> None:
        from agent.runners.result_status import classify_benchmark_result

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            performance = root / "performance.csv"
            performance.write_text(
                "timestamp,cpu_usage,mem_usage,net_total_mbps,local_block_height,"
                "sync_status,execution_mgas_per_sec,execution_metric_source,"
                "execution_metric_status,current_qps,qps_data_available,"
                "data_vda_total_iops,data_vda_avg_await\n"
                "1,10,20,1,100,syncing,0,chain_mgasps,available,0,false,3,0.5\n"
                "2,11,21,2,101,syncing,0,chain_mgasps,available,0,false,4,0.6\n",
                encoding="utf-8",
            )
            report = root / "report.html"
            report_en = root / "report_en.html"
            report_zh = root / "report_zh.html"
            chart = root / "sync_execution_timeline.png"
            summary = root / "test_summary.json"
            health = root / "block_height.csv"
            report.write_text("<html>sync observe</html>", encoding="utf-8")
            report_en.write_text("<html>sync observe</html>", encoding="utf-8")
            report_zh.write_text("<html>sync observe</html>", encoding="utf-8")
            chart.write_bytes(b"png")
            summary.write_text("{}", encoding="utf-8")
            health.write_text("timestamp,local_block_height\n1,100\n", encoding="utf-8")
            artifacts = {
                "summary_json": str(summary),
                "performance_csv": str(performance),
                "html_report": str(report),
                "html_report_en": str(report_en),
                "html_report_zh": str(report_zh),
                "sync_timeline_chart": str(chart),
                "sync_health_csv": str(health),
            }
            complete = classify_benchmark_result(
                {"workflow_type": "sync_observe"},
                0,
                artifacts,
            )
            vegeta = root / "vegeta.json"
            vegeta.write_text("{}", encoding="utf-8")
            contaminated = classify_benchmark_result(
                {"workflow_type": "sync_observe"},
                0,
                {**artifacts, "vegeta_json": str(vegeta)},
            )

        self.assertEqual(complete["status"], "completed")
        self.assertEqual(contaminated["status"], "partial")
        self.assertTrue(
            any("forbidden artifact" in item for item in contaminated["artifact_failures"])
        )

    def test_execution_approval_survives_preflight_result_merge(self) -> None:
        from unittest.mock import patch

        from agent.harness.domains.execution_runtime import execute_approved_preflight_and_smoke
        from agent.harness.invariants import apply_state_delta
        from agent.harness.state import new_state
        from agent.runners.application_service import ExecutionOperation, ExecutionResult, ExecutionStatus

        state = new_state("approval-merge")
        state.update({
            "target_mode": "fake-node",
            "workflow_mode": "rpc_benchmark",
            "preflight": {"approved": True},
        })
        prepared = ExecutionResult(
            operation=ExecutionOperation.PREPARE,
            status=ExecutionStatus.BLOCKED,
            data={
                "plan": {},
                "plan_file": "plan.json",
                "preflight": {"passed": False, "blockers": ["fixture missing"]},
            },
        )
        with patch("agent.harness.domains.execution_runtime.execution_service.execute", return_value=prepared):
            result = execute_approved_preflight_and_smoke(state)
        updated = apply_state_delta(state, result.delta, owner="execution")
        self.assertTrue(updated["preflight"]["approved"])
        self.assertEqual(updated["preflight"]["status"], "blocked")

    def test_execution_handler_reports_blocked_when_preflight_blocks(self) -> None:
        from unittest.mock import patch

        from agent.harness.domains.execution import apply_execution_answer
        from agent.harness.state import new_state

        state = new_state("blocked-execution")
        from agent.harness.contracts import HandlerResult, StateDelta

        blocked_state = dict(state)
        blocked_state["preflight"] = {"approved": True, "status": "blocked"}
        with patch(
            "agent.harness.domains.execution.execute_approved_preflight_and_smoke",
            return_value=HandlerResult(
                delta=StateDelta.between(state, blocked_state),
                completion="blocked",
                blocker="preflight blocked",
            ),
        ):
            result = apply_execution_answer(state, True)
        self.assertEqual(result.completion, "blocked")
        self.assertEqual(result.blocker, "preflight blocked")

    def test_custom_single_fake_node_routes_to_fixture_gate_when_fixture_is_missing(self) -> None:
        from agent.harness.domains.chain_rpc import question_for_chain_rpc
        from agent.harness.domains.chain_rpc_support import _chain_rpc_draft
        from agent.harness.domains.rpc_workload import _set_single_workload
        from agent.harness.routing import next_group_and_reason
        from agent.harness.state import new_state

        state = new_state("custom-fixture-gate")
        state.update({
            "target_mode": "fake-node",
            "workflow_mode": "rpc_benchmark",
            "chain_identity": {"canonical": "bsc", "status": "confirmed", "case": "known"},
            "custom_rpc": {"status": "needs_scope", "method": "eth_accounts"},
        })
        state = _chain_rpc_draft(state)
        _set_single_workload(state, "eth_accounts", "custom_rpc")
        self.assertEqual(state["fixture_evidence"]["status"], "missing")
        group, _reason = next_group_and_reason(state)
        self.assertEqual(group, "provider_deployment")

        # The fixture gate becomes the next workload blocker once the earlier
        # environment groups are complete; its visible options all have a
        # registered transition.
        state["confirmed_config"] = {
            "CLOUD_REGION": "test",
            "CLOUD_ZONE": "test-a",
            "MACHINE_TYPE": "test",
            "LEDGER_DEVICE": "vda",
            "DATA_VOL_TYPE": "ssd",
            "DATA_VOL_SIZE": "100",
            "DATA_VOL_MAX_IOPS": "1000",
            "DATA_VOL_MAX_THROUGHPUT": "100",
            "has_accounts_device": False,
            "NETWORK_INTERFACE": "eth0",
            "NETWORK_MAX_BANDWIDTH_GBPS": "10",
        }
        group, _reason = next_group_and_reason(state)
        self.assertEqual(group, "target_samples_fixtures")
        question = question_for_chain_rpc(state, group)
        self.assertEqual(question["id"], "custom_rpc_fixture_choice")
        self.assertEqual(
            {option["value"] for option in question["options"]},
            {"use_template_defaults", "switch_real_node", "generate_fixture_handoff"},
        )

    def test_custom_single_with_existing_fixture_does_not_require_fixture_choice(self) -> None:
        from agent.harness.domains.chain_rpc_support import _chain_rpc_draft
        from agent.harness.domains.rpc_workload import _set_single_workload
        from agent.harness.state import new_state

        state = new_state("custom-existing-fixture")
        state.update({
            "target_mode": "fake-node",
            "workflow_mode": "rpc_benchmark",
            "chain_identity": {"canonical": "bsc", "status": "confirmed", "case": "known"},
            "custom_rpc": {"status": "needs_scope", "method": "eth_blockNumber"},
        })
        state = _chain_rpc_draft(state)
        _set_single_workload(state, "eth_blockNumber", "custom_rpc")
        self.assertEqual(state["fixture_evidence"]["status"], "validated")
        self.assertEqual(state["active_group"], "workload_rpc")

    def test_job_submission_reuses_same_execution_key(self) -> None:
        from agent.runners.job_manager import submit_job

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            plan = root / "plan.json"
            plan.write_text(json.dumps({
                "plan_id": "same-logical-run",
                "execution": {"idempotency_key": "execution:test", "runner_mode": "foreground"},
            }))
            first = submit_job(plan, jobs_dir=root / "jobs", mock=True, approved=True)
            second = submit_job(plan, jobs_dir=root / "jobs", mock=True, approved=True)
        self.assertEqual(first["job_id"], second["job_id"])
        self.assertTrue(second["submission_reused"])

    def test_legacy_completed_job_is_reclassified_from_business_evidence(self) -> None:
        from agent.runners.job_manager import get_job

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "jobs" / "job_legacy"
            run_dir.mkdir(parents=True)
            plan_file = run_dir / "plan.json"
            plan_file.write_text(json.dumps({"workflow_type": "rpc_benchmark"}))
            vegeta = run_dir / "vegeta.json"
            vegeta.write_text(json.dumps({"requests": 10, "success": 0, "status_codes": {"404": 10}}))
            (run_dir / "job.json").write_text(json.dumps({
                "job_id": "job_legacy",
                "plan_id": "legacy",
                "status": "completed",
                "plan_file": str(plan_file),
                "artifacts": {"vegeta_json": str(vegeta)},
            }))
            result = get_job("job_legacy", jobs_dir=root / "jobs")
        self.assertEqual(result["legacy_status"], "completed")
        self.assertEqual(result["legacy_exit_code_inferred"], 0)
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["result_validation"]["workload_passed"])

    def test_result_classifier_resolves_persisted_workspace_paths(self) -> None:
        from unittest.mock import patch

        from agent.runners import result_status

        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            vegeta = repo / "evidence" / "vegeta.json"
            vegeta.parent.mkdir()
            vegeta.write_text(json.dumps({"requests": 10, "success": 0, "status_codes": {"404": 10}}))
            with patch.object(result_status, "REPO_ROOT", repo):
                result = result_status.classify_benchmark_result(
                    {"workflow_type": "rpc_benchmark"},
                    0,
                    {"vegeta_json": "/workspace/evidence/vegeta.json"},
                )
        self.assertEqual(result["status"], "failed")
        self.assertIn("status_codes={'404': 10}", " ".join(result["execution_failures"]))

    def test_execution_question_passes_request_identity_without_mutating_approved_plan(self) -> None:
        from unittest.mock import patch

        from agent.harness.domains.execution import apply_execution_answer, question_for_execution
        from agent.harness.invariants import apply_state_delta
        from agent.harness.state import new_state
        from agent.runners.application_service import ExecutionOperation, ExecutionResult, ExecutionStatus

        with tempfile.TemporaryDirectory() as tmpdir:
            plan_file = Path(tmpdir) / "plan.json"
            plan_file.write_text(json.dumps({"plan_id": "generated", "execution": {}}))
            state = new_state("persisted-approval")
            state.update({"target_mode": "fake-node", "workflow_mode": "rpc_benchmark"})
            with patch(
                "agent.harness.domains.execution.next_group_and_reason",
                return_value=("preflight_smoke_execution", "ready"),
            ):
                state["pending_question"] = question_for_execution(
                    state,
                    "preflight_smoke_execution",
                ) or {}
            request_id = state["pending_question"]["execution_request_id"]
            prepared = ExecutionResult(
                operation=ExecutionOperation.PREPARE,
                status=ExecutionStatus.OK,
                data={
                    "plan": {"plan_id": "generated", "execution": {}},
                    "plan_file": str(plan_file),
                    "preflight": {"passed": True, "blockers": []},
                },
            )
            smoke = ExecutionResult(
                operation=ExecutionOperation.FAKE_NODE_SMOKE,
                status=ExecutionStatus.OK,
                data={"job": {"job_id": "job-smoke", "status": "running"}},
            )
            with patch(
                "agent.harness.domains.execution_runtime.execution_service.execute",
                side_effect=[prepared, smoke],
            ) as execute:
                result = apply_execution_answer(state, True)
            updated = apply_state_delta(state, result.delta, owner="execution")
            saved = json.loads(plan_file.read_text())
        self.assertEqual(updated["preflight"]["execution_request_id"], request_id)
        self.assertEqual(execute.call_args_list[1].args[0].idempotency_key, f"harness:{request_id}")
        self.assertEqual(saved["execution"], {})

    def test_prepare_benchmark_run_returns_tool_result_envelope(self) -> None:
        from agent.runners.benchmark_pipeline import prepare_benchmark_run

        with tempfile.TemporaryDirectory() as tmpdir:
            result = prepare_benchmark_run(
                chain="bsc",
                goal="smoke",
                use_fake_node=True,
                output_dir=tmpdir,
            )
            for key in ("status", "data", "evidence_paths", "warnings", "next_actions"):
                self.assertIn(key, result)
            self.assertIn("plan", result["data"])
            self.assertTrue(Path(result["data"]["plan_file"]).is_file())

    def test_executor_prepare_benchmark_run_uses_application_service(self) -> None:
        from unittest.mock import patch

        from agent.runners.application_service import ExecutionOperation, ExecutionResult, ExecutionStatus
        from agent.tools.executor import execute_tool

        typed = ExecutionResult(
            operation=ExecutionOperation.PREPARE,
            status=ExecutionStatus.OK,
            data={"plan_file": "shared.json"},
        )
        with patch("agent.tools.executor.execution_service.execute", return_value=typed) as execute:
            result = execute_tool("prepare_benchmark_run", {"chain": "solana"})
        self.assertEqual(result["data"]["plan_file"], "shared.json")
        request = execute.call_args.args[0]
        self.assertEqual(request.operation, ExecutionOperation.PREPARE)
        self.assertEqual(request.prepare_kwargs, {"chain": "solana"})

    def test_run_fake_node_smoke_benchmark_blocks_when_plan_file_missing(self) -> None:
        from agent.runners.benchmark_pipeline import run_fake_node_smoke_benchmark

        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_fake_node_smoke_benchmark(
                str(Path(tmpdir) / "does_not_exist.json"), jobs_dir=tmpdir
            )
            self.assertEqual(result["status"], "blocked")

    def test_submit_benchmark_job_blocks_when_plan_file_missing(self) -> None:
        from agent.runners.benchmark_pipeline import submit_benchmark_job

        with tempfile.TemporaryDirectory() as tmpdir:
            result = submit_benchmark_job(str(Path(tmpdir) / "does_not_exist.json"), jobs_dir=tmpdir)
            self.assertEqual(result["status"], "blocked")

    def test_executor_gates_run_fake_node_smoke_benchmark_and_install_dependencies_on_approved(self) -> None:
        from agent.tools.executor import execute_tool

        smoke = execute_tool("run_fake_node_smoke_benchmark", {"plan_file": "unused.json"})
        install = execute_tool("install_dependencies", {})
        self.assertEqual(smoke["status"], "blocked")
        self.assertEqual(smoke["failure"]["code"], "approval_required")
        self.assertEqual(install["status"], "needs_confirmation")
        self.assertTrue(smoke["requires_user_confirmation"])
        self.assertTrue(install["requires_user_confirmation"])

    def test_smoke_message_renders_actual_terminal_commands_not_dict_keys(self) -> None:
        """Regression test for a code-review finding: `_smoke_message` used to

        iterate the `terminal_commands` dict directly (yielding its keys, e.g.
        the literal word "status") instead of its values (the actual runnable
        command text), because `_job_terminal_commands` always returns a dict,
        never the list `_smoke_message` defaulted to.
        """

        from agent.harness.domains.execution_runtime import _smoke_message

        smoke = {
            "status": "ok",
            "data": {
                "job": {"job_id": "job-123"},
                "terminal_commands": {
                    "status": "status job-123",
                    "logs": "logs job-123",
                    "follow": "follow job-123",
                    "analyze": "analyze latest job",
                },
            },
        }
        message = _smoke_message(smoke)
        self.assertIn("status job-123", message)
        self.assertIn("logs job-123", message)
        self.assertNotIn("Use: status; logs; follow; analyze", message)


class RealNodeExecutionStateMachineTest(unittest.TestCase):
    def test_final_execution_plan_has_stable_key_and_owned_artifact_root(self) -> None:
        from agent.runners.benchmark_pipeline import _isolated_execution_plan

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            plan_file = root / "plan.json"
            plan_file.write_text(json.dumps({
                "plan_id": "final-plan",
                "execution": {"environment": {"LOCAL_RPC_URL": "http://node:8545"}},
            }), encoding="utf-8")
            first_plan = _isolated_execution_plan(plan_file, root / "jobs")
            second_plan = _isolated_execution_plan(plan_file, root / "jobs")

        self.assertEqual(first_plan, second_plan)
        self.assertEqual(first_plan["execution"]["idempotency_key"], second_plan["execution"]["idempotency_key"])
        self.assertIn("final_benchmark", first_plan["execution"]["environment"]["BLOCKCHAIN_BENCHMARK_DATA_DIR"])
        self.assertEqual(first_plan["execution"]["environment"]["LOCAL_RPC_URL"], "http://node:8545")
        self.assertFalse((root / "jobs" / "final_benchmark").exists())

    def test_real_node_smoke_plan_preserves_endpoint_and_workload_but_isolates_qps(self) -> None:
        from agent.runners.benchmark_pipeline import _real_node_smoke_plan

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            plan_file = root / "plan.json"
            plan_file.write_text(json.dumps({
                "plan_id": "final-plan",
                "chain": "ethereum",
                "rpc_mode": "single",
                "use_fake_node": False,
                "execution": {
                    "command": ["./blockchain_node_benchmark.sh", "--standard", "--single"],
                    "idempotency_key": "harness:req-1",
                    "environment": {
                        "LOCAL_RPC_URL": "http://geth-dev:8545",
                        "RPC_SINGLE_METHOD": "eth_chainId",
                        "STANDARD_INITIAL_QPS": "100",
                    },
                },
            }), encoding="utf-8")
            smoke = _real_node_smoke_plan(plan_file, root / "smoke")

        env = smoke["execution"]["environment"]
        self.assertEqual(env["LOCAL_RPC_URL"], "http://geth-dev:8545")
        self.assertEqual(env["RPC_SINGLE_METHOD"], "eth_chainId")
        self.assertEqual(env["QUICK_INITIAL_QPS"], "1")
        self.assertEqual(env["QUICK_MAX_QPS"], "1")
        self.assertEqual(env["QUICK_DURATION"], "10")
        self.assertEqual(smoke["execution"]["idempotency_key"], "harness:req-1:real-node-smoke")
        self.assertNotIn("--fake-node", smoke["execution"]["command"])

    def test_real_node_preflight_submits_only_isolated_smoke(self) -> None:
        from unittest.mock import patch

        from agent.harness.domains.execution_runtime import execute_approved_preflight_and_smoke
        from agent.harness.invariants import apply_state_delta
        from agent.harness.state import new_state
        from agent.runners.application_service import ExecutionOperation, ExecutionResult, ExecutionStatus

        with tempfile.TemporaryDirectory() as tmpdir:
            plan_file = Path(tmpdir) / "final.json"
            plan_file.write_text(json.dumps({"plan_id": "final", "execution": {}}), encoding="utf-8")
            state = new_state("real-smoke")
            state.update({
                "target_mode": "real-node",
                "workflow_mode": "rpc_benchmark",
                "preflight": {"approved": True, "execution_request_id": "req"},
            })
            prepared = ExecutionResult(
                operation=ExecutionOperation.PREPARE,
                status=ExecutionStatus.OK,
                data={"plan": {}, "plan_file": str(plan_file), "preflight": {"passed": True}},
            )
            smoke_result = ExecutionResult(
                operation=ExecutionOperation.REAL_NODE_SMOKE,
                status=ExecutionStatus.OK,
                data={"job": {"job_id": "job-smoke", "status": "running"}},
            )
            with patch(
                "agent.harness.domains.execution_runtime.execution_service.execute",
                side_effect=[prepared, smoke_result],
            ) as execute:
                result = execute_approved_preflight_and_smoke(state)
            updated = apply_state_delta(state, result.delta, owner="execution")

        self.assertEqual(execute.call_count, 2)
        request = execute.call_args_list[1].args[0]
        self.assertEqual(request.operation, ExecutionOperation.REAL_NODE_SMOKE)
        self.assertEqual(Path(request.plan_file), plan_file)
        self.assertEqual(updated["smoke"]["purpose"], "real_node_isolated_smoke")
        self.assertEqual(updated["smoke"]["job_id"], "job-smoke")

    def test_completed_smoke_requires_distinct_final_approval(self) -> None:
        from unittest.mock import patch

        from agent.harness.domains.execution import (
            apply_execution_answer,
            question_for_execution,
            reconcile_execution_state,
        )
        from agent.harness.contracts import HandlerResult, StateDelta
        from agent.harness.invariants import apply_state_delta
        from agent.harness.state import new_state

        state = new_state("real-final")
        state.update({
            "target_mode": "real-node",
            "workflow_mode": "rpc_benchmark",
            "plan_file": "final.json",
            "preflight": {"approved": True, "status": "passed"},
            "smoke": {"purpose": "real_node_isolated_smoke", "status": "running", "job_id": "job-smoke"},
            "job": {"job_id": "job-smoke", "status": "running"},
        })
        persisted = {"job_id": "job-smoke", "status": "completed", "artifacts": {}}
        with patch("agent.runners.job_manager.get_job", return_value=persisted):
            reconciled = reconcile_execution_state(state)
        state = apply_state_delta(state, reconciled.delta, owner="execution")
        question = question_for_execution(state, "job_monitoring")
        self.assertEqual(question["id"], "real_node_final_benchmark_confirm")
        self.assertIn("preserve the successful smoke evidence", question["options"][1]["completion_effect"])

        final_state = dict(state)
        final_state.update({
            "job": {"job_id": "job-final", "status": "running"},
            "final_benchmark": {"approved": True, "status": "running", "job_id": "job-final"},
            "visible_response": ["submitted"],
        })
        with patch(
            "agent.harness.domains.execution.execute_approved_final_benchmark",
            return_value=HandlerResult(delta=StateDelta.between(state, final_state)),
        ) as submit:
            result = apply_execution_answer(state, True, question)
        submit.assert_called_once()
        updated = apply_state_delta(state, result.delta, owner="execution")
        self.assertEqual(updated["final_benchmark"]["job_id"], "job-final")

    def test_declined_final_approval_never_submits_job(self) -> None:
        from unittest.mock import patch

        from agent.harness.domains.execution import apply_execution_answer
        from agent.harness.state import new_state

        state = new_state("real-decline")
        state.update({
            "target_mode": "real-node",
            "smoke": {"purpose": "real_node_isolated_smoke", "status": "completed", "job_id": "job-smoke"},
        })
        question = {"id": "real_node_final_benchmark_confirm"}
        with patch("agent.harness.domains.execution.execute_approved_final_benchmark") as submit:
            result = apply_execution_answer(state, False, question)
        submit.assert_not_called()
        from agent.harness.invariants import apply_state_delta

        updated = apply_state_delta(state, result.delta, owner="execution")
        self.assertEqual(updated["final_benchmark"]["decision"], "declined")


class BenchmarkSubprocessEnvTest(unittest.TestCase):
    """C.4 (known-issues.md validation gap): real benchmark execution never

    completed in this environment. Root cause once `pandas`/`matplotlib`/etc.
    and a Go toolchain were installed: the benchmark pipeline's own analysis/
    report scripts call bare `python3`, not the Agent's own venv -- and
    `scripts/install_deps.sh` documents a separate repo-root `.venv` as the
    framework's own Python environment, which the Agent process does not
    necessarily have on `PATH` when it spawns the benchmark subprocess. The
    subprocess then failed partway through (well after Vegeta itself had
    already run) with confusing per-script failure messages, not a clear
    "missing Python package" error. Verified live end to end: a real
    fake-node run with `.venv` on `PATH` completed fully (real Vegeta load,
    real analysis scripts, real bilingual HTML report, real archiving) where
    it previously failed at the report-generation stage.
    """

    def test_prepends_project_venv_bin_to_path_when_it_exists(self) -> None:
        import os
        import tempfile
        from unittest.mock import patch

        from agent.runners.materialize import benchmark_subprocess_env

        with tempfile.TemporaryDirectory() as tmp_repo:
            venv_bin = Path(tmp_repo) / ".venv" / "bin"
            venv_bin.mkdir(parents=True)
            with patch("agent.runners.materialize._REPO_ROOT", Path(tmp_repo)):
                env = benchmark_subprocess_env({"FOO": "bar"})
            self.assertEqual(env["FOO"], "bar")
            self.assertTrue(env["PATH"].startswith(str(venv_bin) + os.pathsep))
            # The rest of the original PATH must still be present, not replaced.
            self.assertIn(os.environ.get("PATH", ""), env["PATH"])

    def test_leaves_path_unchanged_when_no_project_venv_exists(self) -> None:
        import os
        import tempfile
        from unittest.mock import patch

        from agent.runners.materialize import benchmark_subprocess_env

        with tempfile.TemporaryDirectory() as tmp_repo:
            with patch("agent.runners.materialize._REPO_ROOT", Path(tmp_repo)):
                env = benchmark_subprocess_env({})
            self.assertEqual(env.get("PATH", ""), os.environ.get("PATH", ""))


if __name__ == "__main__":
    unittest.main()

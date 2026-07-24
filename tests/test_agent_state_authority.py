from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class TerminalPersistenceAuthorityTest(unittest.TestCase):
    def test_terminal_store_persists_only_shell_ui_and_consent(self) -> None:
        from agent.terminal.repl import TerminalSession, TerminalSessionStore

        with tempfile.TemporaryDirectory() as tmpdir:
            session_file = Path(tmpdir) / "session.json"
            store = TerminalSessionStore(session_file)
            state = TerminalSession(
                language="zh",
                current_question_id="install_dependencies",
                latest_job_id="job_business_fact",
                discovery={"cloud": {"provider": "test-provider"}},
                framework_summary={"chain_count": 42},
                pending_missing_dependencies=["vegeta"],
            )

            store.save(state)
            persisted = json.loads(session_file.read_text(encoding="utf-8"))
            loaded = store.load()

        self.assertEqual(
            persisted,
            {
                "current_question_id": "install_dependencies",
                "language": "zh",
                "pending_missing_dependencies": ["vegeta"],
            },
        )
        self.assertEqual(loaded.latest_job_id, "")
        self.assertEqual(loaded.discovery, {})
        self.assertEqual(loaded.framework_summary, {})

    def test_terminal_store_drops_legacy_business_state_and_non_shell_question(self) -> None:
        from agent.terminal.repl import TerminalSession, TerminalSessionStore

        with tempfile.TemporaryDirectory() as tmpdir:
            session_file = Path(tmpdir) / "session.json"
            session_file.write_text(
                json.dumps(
                    {
                        "language": "en",
                        "current_question_id": "benchmark_approval",
                        "latest_job_id": "job_legacy",
                        "discovery": {"cpu": 8},
                        "framework_summary": {"chain_count": 1},
                    }
                ),
                encoding="utf-8",
            )
            store = TerminalSessionStore(session_file)
            loaded = store.load()
            store.save(TerminalSession(current_question_id="benchmark_approval"))
            rewritten = json.loads(session_file.read_text(encoding="utf-8"))

        self.assertEqual(loaded.current_question_id, "")
        self.assertEqual(loaded.latest_job_id, "")
        self.assertEqual(loaded.discovery, {})
        self.assertEqual(rewritten["current_question_id"], "")


class InvocationContextAuthorityTest(unittest.TestCase):
    def test_current_checkpoint_never_invokes_v12_action_adapter(self) -> None:
        from agent.harness.state import migrate_state, new_state

        current = new_state("current-no-compat", language="en")
        with patch(
            "agent.harness.checkpoint_migrations.compile_v12_custom_rpc_action"
        ) as legacy_adapter:
            migrated = migrate_state(
                current,
                thread_id="current-no-compat",
                language="en",
                session_purpose="user",
            )

        legacy_adapter.assert_not_called()
        self.assertEqual(migrated["schema_version"], current["schema_version"])

    def test_v12_checkpoint_is_persisted_as_v13_once(self) -> None:
        from agent.harness.graph import AnyChainGraphRuntime
        from agent.harness.state import STATE_SCHEMA_VERSION, new_state

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "migration.sqlite"
            runtime = AnyChainGraphRuntime(
                thread_id="persist-migration",
                checkpoint_path=path,
            )
            old = new_state("persist-migration", language="en")
            old["schema_version"] = 12
            runtime._persist_state(old)
            first = runtime.snapshot()
            second = runtime.snapshot()
            runtime.close()

        self.assertEqual(first["schema_version"], STATE_SCHEMA_VERSION)
        self.assertEqual(second["schema_version"], STATE_SCHEMA_VERSION)
        migration_events = [
            event
            for event in second.get("audit_events") or []
            if event.get("event") == "checkpoint_schema_migrated"
        ]
        self.assertEqual(len(migration_events), 1)

    def test_invariant_recovery_projects_external_context_before_raw_checkpoint_write(self) -> None:
        from agent.harness.graph import AnyChainGraphRuntime

        class InvalidTurnGraph:
            def invoke(self, state, **_kwargs):
                state["active_group"] = "not-a-product-group"
                return state

        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = AnyChainGraphRuntime(
                "external-context-recovery",
                checkpoint_path=Path(tmpdir) / "checkpoints.sqlite3",
            )
            try:
                with patch.object(
                    runtime.graph,
                    "invoke",
                    side_effect=InvalidTurnGraph().invoke,
                ):
                    runtime.invoke(
                        "trigger recovery",
                        context={
                            "discovery": {"cloud": {"provider": "gcp"}},
                            "framework_summary": {"chain_count": 36},
                            "web_research": {"available": True},
                            "latest_job_id": "job_external",
                        },
                    )
                raw = dict(runtime.graph.get_state(
                    {"configurable": {"thread_id": "external-context-recovery"}}
                ).values or {})
            finally:
                runtime.close()

        self.assertEqual(raw["discovery"], {})
        self.assertEqual(raw["framework_summary"], {})
        self.assertEqual(raw["web_research"], {})
        self.assertNotIn("latest_job_id", raw)
        self.assertEqual(raw["active_group"], "failure_recovery")

    def test_legacy_checkpoint_migration_drops_external_read_model_copies(self) -> None:
        from agent.harness.state import migrate_state

        migrated = migrate_state(
            {
                "schema_version": 7,
                "discovery": {"cloud": {"provider": "stale"}},
                "framework_summary": {"chain_count": 1},
                "web_research": {"available": True},
                "latest_job_id": "job_stale",
                "confirmed_config": {"CLOUD_REGION": "asia-east1"},
            },
            thread_id="external-context-migration",
            language="en",
            session_purpose="user",
        )

        self.assertEqual(migrated["discovery"], {})
        self.assertEqual(migrated["framework_summary"], {})
        self.assertEqual(migrated["web_research"], {})
        self.assertNotIn("latest_job_id", migrated)
        self.assertEqual(migrated["confirmed_config"], {})
        self.assertEqual(
            migrated["inferred_config"]["pending_review"]["config_values"],
            {"CLOUD_REGION": "asia-east1"},
        )

    def test_graph_invocation_uses_but_never_checkpoints_external_context(self) -> None:
        from agent.harness.graph import AnyChainGraphRuntime

        class EchoTurnGraph:
            observed = {}

            def invoke(self, state, **kwargs):
                context = kwargs.get("context") or {}
                self.observed = {
                    "discovery": context.get("discovery"),
                    "framework_summary": context.get("framework_summary"),
                }
                state["visible_response"] = ["context consumed"]
                state["turn_index"] = int(state.get("turn_index") or 0) + 1
                return state

        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = AnyChainGraphRuntime(
                "external-context-runtime",
                checkpoint_path=Path(tmpdir) / "checkpoints.sqlite3",
            )
            try:
                turn_graph = EchoTurnGraph()
                with patch.object(
                    runtime.graph,
                    "invoke",
                    side_effect=turn_graph.invoke,
                ):
                    runtime.invoke(
                        "show current environment",
                        context={
                            "discovery": {"cloud": {"provider": "gcp"}},
                            "framework_summary": {"chain_count": 36},
                        },
                    )
                persisted = runtime.snapshot()
            finally:
                runtime.close()

        self.assertEqual(turn_graph.observed["framework_summary"], {"chain_count": 36})
        self.assertEqual(persisted["discovery"], {})
        self.assertEqual(persisted["framework_summary"], {})
        self.assertNotIn("latest_job_id", persisted)


class SyncObserveOwnershipAuthorityTest(unittest.TestCase):
    def test_environment_proposes_sync_options_without_writing_sync_state(self) -> None:
        from agent.harness.domains.environment import apply_direct_config_assignments
        from agent.harness.state import new_state

        result = apply_direct_config_assignments(
            new_state("sync-options", language="en"),
            {
                "SYNC_OBSERVE_STOP_CONDITION": "duration",
                "SYNC_OBSERVE_DURATION_SECONDS": "60",
            },
        )

        self.assertFalse(any(write.path[0] == "sync_observe" for write in result.delta.writes))
        self.assertEqual(
            result.followup_actions,
            ({
                "type": "set_sync_observe_options",
                "sync_observe_stop_condition": "duration",
                "sync_observe_duration_seconds": 60,
            },),
        )

    def test_only_sync_domain_may_write_sync_observe_state(self) -> None:
        from agent.harness.contracts import StateDelta
        from agent.harness.invariants import StateInvariantError, validate_delta_owner

        delta = StateDelta.set_values({"sync_observe": {"stop_condition": "duration"}})
        for owner in ("environment", "chain_rpc"):
            with self.subTest(owner=owner), self.assertRaises(StateInvariantError):
                validate_delta_owner(delta, owner)
        validate_delta_owner(delta, "sync_observe")


class JobEvidenceAuthorityTest(unittest.TestCase):
    def _write_legacy_job(self, root: Path) -> tuple[Path, Path]:
        run_dir = root / "jobs" / "job_legacy"
        run_dir.mkdir(parents=True)
        plan_file = run_dir / "plan.json"
        plan_file.write_text(json.dumps({"workflow_type": "rpc_benchmark"}), encoding="utf-8")
        vegeta = run_dir / "vegeta.json"
        vegeta.write_text(
            json.dumps({"requests": 10, "success": 0, "status_codes": {"404": 10}}),
            encoding="utf-8",
        )
        job_file = run_dir / "job.json"
        job_file.write_text(
            json.dumps(
                {
                    "job_id": "job_legacy",
                    "plan_id": "legacy",
                    "status": "completed",
                    "plan_file": str(plan_file),
                    "run_dir": str(run_dir),
                    "artifacts": {"vegeta_json": str(vegeta)},
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return job_file, root / "jobs"

    def test_job_reads_and_startup_discovery_do_not_rewrite_legacy_evidence(self) -> None:
        from agent.runners.job_manager import get_job, list_jobs
        from agent.terminal.startup_state import load_startup_state

        with tempfile.TemporaryDirectory() as tmpdir:
            job_file, jobs_dir = self._write_legacy_job(Path(tmpdir))
            original = job_file.read_bytes()

            derived = get_job("job_legacy", jobs_dir=jobs_dir)
            listed = list_jobs(jobs_dir=jobs_dir)
            startup = load_startup_state(jobs_dir=jobs_dir)

            self.assertEqual(job_file.read_bytes(), original)

        self.assertEqual(derived["legacy_status"], "completed")
        self.assertEqual(derived["status"], "failed")
        self.assertEqual(listed[0]["status"], "failed")
        self.assertEqual(startup["latest_job"]["status"], "failed")

    def test_legacy_job_migration_requires_explicit_command(self) -> None:
        from agent.runners.job_manager import migrate_legacy_job_result

        with tempfile.TemporaryDirectory() as tmpdir:
            job_file, jobs_dir = self._write_legacy_job(Path(tmpdir))
            original = job_file.read_bytes()
            migrated = migrate_legacy_job_result("job_legacy", jobs_dir=jobs_dir)
            migrated_bytes = job_file.read_bytes()
            persisted = json.loads(job_file.read_text(encoding="utf-8"))
            backup_file = Path(persisted["migration_provenance"]["backup_file"])
            backup_bytes = backup_file.read_bytes()

        self.assertNotEqual(migrated_bytes, original)
        self.assertEqual(backup_bytes, original)
        self.assertEqual(persisted["status"], "failed")
        self.assertEqual(persisted["result_validation"], migrated["result_validation"])
        self.assertEqual(persisted["migration_provenance"]["operation"], "legacy_result_migration")
        self.assertEqual(
            persisted["migration_provenance"]["source_sha256"],
            hashlib.sha256(original).hexdigest(),
        )


class ImmutableExecutionPlanTest(unittest.TestCase):
    def test_submission_rejects_full_plan_replacement_even_with_valid_approved_hash(self) -> None:
        from agent.runners.application_service import (
            BenchmarkExecutionService,
            ExecutionFailureCode,
            ExecutionOperation,
            ExecutionRequest,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            approved_plan = root / "approved.json"
            approved_payload = {
                "plan_id": "approved-plan",
                "chain": "bsc",
                "strategy": "smoke",
                "execution": {"command": ["false"], "environment": {}, "runner_mode": "foreground"},
            }
            approved_plan.write_text(json.dumps(approved_payload, sort_keys=True), encoding="utf-8")
            malicious = {
                **approved_payload,
                "execution": {"command": ["true"], "environment": {}, "runner_mode": "foreground"},
                "execution_provenance": {
                    "approved_plan_file": str(approved_plan),
                    "approved_plan_sha256": hashlib.sha256(approved_plan.read_bytes()).hexdigest(),
                },
            }
            result = BenchmarkExecutionService().execute(
                ExecutionRequest(
                    operation=ExecutionOperation.FINAL_BENCHMARK,
                    approved=True,
                    plan_file=approved_plan,
                    plan=malicious,
                    jobs_dir=root / "jobs",
                    mock=True,
                    runtime_override_sources=("untrusted.full_plan",),
                )
            )

        self.assertEqual(result.failure.code, ExecutionFailureCode.INVALID_REQUEST)
        self.assertIn("execution.command", result.failure.message)
        self.assertIn("execution_provenance", result.failure.message)

    def test_idempotency_and_custom_rpc_overrides_live_in_job_plan_with_provenance(self) -> None:
        from agent.runners.application_service import (
            BenchmarkExecutionService,
            ExecutionOperation,
            ExecutionRequest,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            approved_plan = root / "approved.json"
            approved_payload = {
                "plan_id": "approved-plan",
                "chain": "bsc",
                "strategy": "smoke",
                "execution": {"command": ["true"], "environment": {}, "runner_mode": "foreground"},
            }
            approved_plan.write_text(json.dumps(approved_payload, sort_keys=True), encoding="utf-8")
            approved_bytes = approved_plan.read_bytes()
            runtime_plan = {
                **approved_payload,
                "chain_config_override": {"rpc_methods": {"single": "eth_accounts"}},
            }
            request = ExecutionRequest(
                operation=ExecutionOperation.FINAL_BENCHMARK,
                approved=True,
                idempotency_key="harness:request-40",
                plan_file=approved_plan,
                plan=runtime_plan,
                jobs_dir=root / "jobs",
                mock=True,
                runtime_override_sources=("harness.custom_rpc",),
            )

            first = BenchmarkExecutionService().execute(request)
            second = BenchmarkExecutionService().execute(request)
            job = first.data["job"]
            job_plan = json.loads(Path(job["plan_file"]).read_text(encoding="utf-8"))

            self.assertEqual(approved_plan.read_bytes(), approved_bytes)

        provenance = job_plan["execution_provenance"]
        self.assertEqual(job_plan["execution"]["idempotency_key"], "harness:request-40")
        self.assertEqual(job_plan["chain_config_override"], runtime_plan["chain_config_override"])
        self.assertEqual(provenance["approved_plan_file"], str(approved_plan.resolve()))
        self.assertEqual(provenance["approved_plan_sha256"], hashlib.sha256(approved_bytes).hexdigest())
        self.assertEqual(provenance["job_id"], job["job_id"])
        self.assertEqual(provenance["job_plan_file"], job["plan_file"])
        self.assertIn("harness.custom_rpc", provenance["runtime_overrides"])
        self.assertIn("execution.idempotency_key", provenance["runtime_overrides"])
        self.assertIn("custom_rpc.chain_config_override", provenance["runtime_overrides"])
        self.assertTrue(second.reused)
        self.assertEqual(second.data["job"]["job_id"], job["job_id"])

    def test_custom_rpc_closure_does_not_write_the_prepared_plan(self) -> None:
        from agent.harness.domains.execution_runtime import _prepare_benchmark_with_runtime_contract
        from agent.runners.application_service import (
            ExecutionOperation,
            ExecutionResult,
            ExecutionStatus,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            plan_file = Path(tmpdir) / "approved.json"
            base_plan = {"plan_id": "prepared", "chain": "bsc", "execution": {}}
            plan_file.write_text(json.dumps(base_plan, sort_keys=True), encoding="utf-8")
            approved_bytes = plan_file.read_bytes()
            prepared = ExecutionResult(
                operation=ExecutionOperation.PREPARE,
                status=ExecutionStatus.OK,
                data={
                    "plan": dict(base_plan),
                    "plan_file": str(plan_file),
                    "preflight": {"passed": True, "checks": [], "blockers": []},
                },
            )
            preflight = ExecutionResult(
                operation=ExecutionOperation.PREFLIGHT,
                status=ExecutionStatus.OK,
                data={"preflight": {"passed": True, "checks": [], "blockers": []}},
            )
            state = {
                "rpc_mode": "single",
                "chain_identity": {"canonical": "bsc", "adapter_family": "jsonrpc"},
                "workload": {
                    "job_local_override": True,
                    "methods": ["eth_accounts"],
                    "confirmed": True,
                },
                "custom_rpc": {
                    "catalog": {
                        "contract_version": 1,
                        "revision": 1,
                        "chain": "bsc",
                        "methods": [{"method": "eth_accounts", "params": []}],
                        "finished": True,
                    },
                },
            }

            with patch(
                "agent.harness.domains.execution_runtime.execution_service.execute",
                side_effect=[prepared, preflight],
            ):
                result = _prepare_benchmark_with_runtime_contract(state)

            self.assertEqual(plan_file.read_bytes(), approved_bytes)

        self.assertIn("chain_config_override", result["data"]["plan"])


class HistoricalAnalysisAuthorityTest(unittest.TestCase):
    def test_historical_analysis_does_not_replace_latest_execution_state(self) -> None:
        from agent.harness.domains.analysis import report_artifact_entry_result

        state = {
            "language": "en",
            "job": {"job_id": "job_current", "status": "running"},
            "report_context": {"requested_job_id": "job_historical"},
        }
        historical = {"job_id": "job_historical", "status": "completed"}
        with patch(
            "agent.harness.domains.analysis._report_artifact_entry",
            return_value=("historical analysis", historical),
        ):
            result = report_artifact_entry_result(state)

        writes = {write.path: write.value for write in result.delta.writes}
        self.assertNotIn(("job",), writes)
        self.assertFalse(any(path[0] == "job" for path in writes))
        self.assertEqual(writes[("report_context", "analyzed_job_id")], "job_historical")
        self.assertEqual(writes[("report_context", "analyzed_job_status")], "completed")


if __name__ == "__main__":
    unittest.main()

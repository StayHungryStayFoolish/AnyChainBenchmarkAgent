"""Product contracts for structured benchmark failure recovery."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class FailureRecoveryTest(unittest.TestCase):
    @staticmethod
    def _commit_domain_delta(state: dict, result: object, *, owner: str) -> dict:
        """Commit a domain result through the production ownership boundary."""

        from agent.harness.coordinator import _apply_handler_result

        return _apply_handler_result(state, result, owner=owner)

    def test_current_state_summarizes_pending_contract_without_repeating_prompt(self) -> None:
        from agent.harness.oracle import format_current_state
        from agent.harness.questions import manual_question
        from agent.harness.state import new_state

        for question_id, field, prompt in (
            ("CLOUD_REGION", "CLOUD_REGION", "Enter the cloud region with a long explanation."),
            ("custom_rpc_endpoint", "custom_rpc_endpoint", "Provide the complete endpoint validation instructions."),
        ):
            with self.subTest(question_id=question_id):
                state = new_state(f"pending-{question_id}", language="en")
                state["active_group"] = "provider_deployment" if question_id == "CLOUD_REGION" else "endpoint_process"
                state["pending_question"] = manual_question(
                    state["active_group"], question_id, prompt, field=field,
                )
                rendered = format_current_state(state, "en")
                self.assertNotIn(prompt, rendered)
                self.assertIn(f"confirm `{field}`", rendered)

    def test_target_mode_prompt_distinguishes_initial_selection_from_replacement(self) -> None:
        from agent.harness.domains.chain_rpc import question_for_chain_rpc
        from agent.harness.state import new_state

        initial = new_state("initial-target", language="en")
        initial_question = question_for_chain_rpc(initial, "target_mode") or {}
        self.assertEqual(initial_question.get("prompt"), "Choose the target mode.")

        replacement = new_state("replacement-target", language="en")
        replacement["target_mode"] = "fake-node"
        replacement_question = question_for_chain_rpc(replacement, "target_mode") or {}
        self.assertIn("new target mode", str(replacement_question.get("prompt")))

    def test_current_state_localizes_target_mode_blocker(self) -> None:
        from agent.harness.domains.chain_rpc import question_for_chain_rpc
        from agent.harness.oracle import format_current_state
        from agent.harness.state import new_state

        state = new_state("target-mode-label", language="zh")
        state["active_group"] = "target_mode"
        state["pending_question"] = question_for_chain_rpc(state, "target_mode") or {}
        rendered = format_current_state(state, "zh")
        self.assertIn("测试模式", rendered)
        self.assertIn("选择 fake-node、real-node 或 sync-observe", rendered)
        self.assertNotIn("继续确认target_mode", rendered)

    def test_resume_summary_exposes_deferred_configuration_requests(self) -> None:
        from agent.harness.domains.orientation import resume_summary
        from agent.harness.state import new_state

        state = new_state("resume-queue", language="en")
        state["action_queue"] = [{"type": "set_qps_mode", "qps_mode": "quick"}]
        self.assertIn("deferred requests: QPS=quick", resume_summary(state))

    def test_current_state_names_effective_workload_and_weights(self) -> None:
        from agent.harness.oracle import format_current_state
        from agent.harness.state import new_state

        state = new_state("auditable-workload", language="en")
        state.update({
            "target_mode": "real-node",
            "workflow_mode": "rpc_benchmark",
            "chain_identity": {"canonical": "bsc", "status": "confirmed"},
            "rpc_mode": "mixed",
            "workload": {
                "confirmed": True,
                "methods": ["eth_blockNumber", "eth_accounts"],
                "weights": {"eth_blockNumber": 70, "eth_accounts": 30},
            },
        })

        rendered = format_current_state(state, "en")
        self.assertIn("eth_blockNumber, eth_accounts", rendered)
        self.assertIn("eth_blockNumber=70", rendered)
        self.assertIn("eth_accounts=30", rendered)

    def test_fixture_choice_explains_preserved_configuration(self) -> None:
        from agent.harness.domains.orientation import pending_context_response
        from agent.harness.state import new_state

        state = new_state("fixture-preservation", language="en")
        state["workload"] = {"methods": ["eth_accounts"]}
        state["confirmed_config"] = {"LEDGER_DEVICE": "vda", "CLOUD_REGION": "test"}
        rendered = pending_context_response(state, {"id": "custom_rpc_fixture_choice"})

        self.assertIn("eth_accounts", rendered)
        self.assertIn("LEDGER_DEVICE", rendered)
        self.assertIn("preserved", rendered)

    def test_required_failure_codes_all_have_policies(self) -> None:
        from agent.harness.failures import RECOVERY_POLICIES

        self.assertEqual(set(RECOVERY_POLICIES), {
            "DOMAIN_ACTION_BLOCKED",
            "PREFLIGHT_CHECK_FAILED",
            "FIXTURE_MISSING",
            "ENDPOINT_UNREACHABLE",
            "RPC_METHOD_OR_SCHEMA_INVALID",
            "WORKLOAD_PROCESS_FAILED",
            "WORKLOAD_NO_REQUESTS",
            "WORKLOAD_ZERO_SUCCESS",
            "ARTIFACT_INCOMPLETE",
            "HARNESS_INVARIANT_FAILED",
            "MODEL_PROVIDER_UNAVAILABLE",
        })

    def test_preflight_fixture_failure_has_stable_policy(self) -> None:
        from agent.harness.failures import failure_record_from_preflight

        record = failure_record_from_preflight({
            "checks": [{
                "name": "effective_workload_fixtures_available",
                "passed": False,
                "detail": "fixture missing for eth_accounts",
            }],
            "evidence_paths": ["/workspace/evidence/preflight.json"],
        }, confirmed_config={"BLOCKCHAIN_NODE": "bsc", "CLOUD_REGION": "test"})
        self.assertEqual(record["code"], "FIXTURE_MISSING")
        self.assertEqual(record["affected_group"], "target_samples_fixtures")
        self.assertIn("correct_failure", record["allowed_actions"])
        self.assertEqual(record["preserved_config_keys"], ["BLOCKCHAIN_NODE", "CLOUD_REGION"])

    def test_workload_failure_precedes_missing_report_artifacts(self) -> None:
        from agent.harness.failures import failure_record_from_job

        record = failure_record_from_job({
            "job_id": "job_test",
            "status": "failed",
            "result_validation": {"failure_facts": [
                {"code": "ARTIFACT_INCOMPLETE", "source": "artifact", "artifact": "html_report"},
                {"code": "WORKLOAD_ZERO_SUCCESS", "source": "workload", "status_codes": {"404": 10}},
            ]},
        })
        self.assertEqual(record["code"], "WORKLOAD_ZERO_SUCCESS")
        self.assertEqual(record["affected_group"], "workload_rpc")

    def test_failure_record_redacts_secrets_before_persistence(self) -> None:
        from agent.harness.failures import build_failure_record

        secret = "a0528eaafd611ffa9045736ddf744eb59b0eff75"
        record = build_failure_record(
            "ENDPOINT_UNREACHABLE",
            source="endpoint",
            severity="blocking",
            facts=[{"detail": f"Authorization: Bearer {secret}"}],
            evidence_paths=[f"https://rpc.example/{secret}/probe.json"],
        )
        serialized = json.dumps(record)
        self.assertNotIn(secret, serialized)
        self.assertIn("REDACTED", serialized)

    def test_recovery_question_options_have_registered_handlers_and_postconditions(self) -> None:
        from agent.harness.action_registry import ACTION_BY_TYPE
        from agent.harness.domains.recovery import question_for_recovery
        from agent.harness.failures import build_failure_record
        from agent.harness.state import new_state

        state = new_state("recovery-options", language="en")
        state["failure_recovery"] = {"status": "pending", "record": build_failure_record(
            "FIXTURE_MISSING", source="preflight", severity="blocking", facts=[{"detail": "missing"}],
        )}
        question = question_for_recovery(state, "failure_recovery") or {}
        self.assertEqual(question.get("id"), "failure_recovery_action")
        for option in question.get("options") or []:
            action_type = str((option.get("action") or {}).get("type") or "")
            self.assertIn(action_type, ACTION_BY_TYPE)
            self.assertEqual(ACTION_BY_TYPE[action_type].owner, "recovery")
            self.assertTrue(option.get("expected_patch"))

    def test_correction_preserves_unrelated_config_and_reopens_only_policy_scope(self) -> None:
        from agent.harness.contracts import ActionProposal
        from agent.harness.domains.recovery import apply_recovery_action
        from agent.harness.failures import build_failure_record
        from agent.harness.state import new_state
        from agent.harness.coordinator import _apply_handler_result

        state = new_state("recovery-correction")
        state["confirmed_config"] = {"CLOUD_REGION": "asia-east1", "LEDGER_DEVICE": "vda"}
        state["workload"] = {"confirmed": True, "methods": ["eth_accounts"]}
        state["job"] = {"job_id": "job_failed"}
        state["failure_recovery"] = {"status": "pending", "record": build_failure_record(
            "WORKLOAD_ZERO_SUCCESS", source="workload", severity="blocking", facts=[{"detail": "0 success"}],
            confirmed_config=state["confirmed_config"], job_id="job_failed",
        )}
        result = apply_recovery_action(state, ActionProposal("a1", "correct_failure"))
        repaired = _apply_handler_result(state, result, owner="recovery")
        self.assertEqual(repaired["failure_recovery"]["status"], "correcting")
        self.assertEqual(result.next_group, "workload_rpc")
        self.assertEqual(repaired["workload"], {})
        self.assertEqual(repaired["confirmed_config"], state["confirmed_config"])
        self.assertEqual(repaired["job"], {})
        self.assertEqual(set(result.invalidated_groups), {
            "workload_rpc", "target_samples_fixtures", "preflight_smoke_execution", "job_monitoring",
        })

    def test_inspection_uses_llm_only_as_advisory_and_keeps_state(self) -> None:
        from agent.harness.contracts import ActionProposal
        from agent.harness.coordinator import _apply_handler_result
        from agent.harness.domains.recovery import apply_recovery_action
        from agent.harness.failures import build_failure_record
        from agent.harness.state import new_state

        state = new_state("recovery-advisory")
        state["confirmed_config"] = {"CLOUD_REGION": "asia-east1"}
        state["failure_recovery"] = {"status": "pending", "record": build_failure_record(
            "WORKLOAD_PROCESS_FAILED", source="workload", severity="blocking", facts=[{"detail": "exit 2"}],
        )}
        with patch("agent.harness.domains.recovery.analyze_evidence_with_model", return_value="advisory") as analyze:
            result = apply_recovery_action(state, ActionProposal("inspect", "inspect_failure"))
        inspected = _apply_handler_result(state, result, owner="recovery")
        analyze.assert_called_once()
        self.assertEqual(inspected["confirmed_config"], state["confirmed_config"])
        self.assertEqual(inspected["failure_recovery"]["status"], "pending")
        self.assertEqual(result.visible_results[-1], "advisory")
        final_response = "\n".join(inspected["visible_response"])
        self.assertEqual(final_response.count("Execution recovery:"), 1)
        self.assertEqual(final_response.count("exit 2"), 1)
        self.assertIn("advisory", final_response)
        self.assertIn("Inspect failure evidence and diagnostics", final_response)

    def test_inspection_action_queue_keeps_domain_pending_contract_authoritative(self) -> None:
        from agent.harness.coordinator import _finalize_turn_response, _process_action_queue
        from agent.harness.domains.recovery import question_for_recovery
        from agent.harness.failures import build_failure_record
        from agent.harness.state import new_state

        state = new_state("recovery-presentation-owner", language="en")
        state["active_group"] = "failure_recovery"
        state["failure_recovery"] = {"status": "pending", "record": build_failure_record(
            "ENDPOINT_UNREACHABLE",
            source="endpoint",
            severity="blocking",
            facts=[{"detail": "connection refused"}],
        )}
        state["pending_question"] = question_for_recovery(state, "failure_recovery") or {}
        state["visible_response"] = []

        with patch("agent.harness.domains.recovery.analyze_evidence_with_model", return_value="advisory"):
            inspected = _process_action_queue(
                state,
                [{"type": "inspect_failure", "action_id": "inspect"}],
                "inspect the evidence",
                max_actions=1,
            )
        self.assertIsNotNone(inspected)
        final = _finalize_turn_response(inspected or state)
        response = "\n".join(final["visible_response"])

        self.assertEqual(response.count("Execution recovery:"), 1)
        self.assertEqual(response.count("connection refused"), 1)
        self.assertEqual(response.count("Choose the next step."), 1)
        self.assertIn("advisory", response)
        self.assertEqual(final["pending_question"]["id"], "failure_recovery_action")

    def test_cancel_preserves_failure_and_configuration_without_rerun(self) -> None:
        from agent.harness.contracts import ActionProposal
        from agent.harness.domains.recovery import apply_recovery_action
        from agent.harness.failures import build_failure_record
        from agent.harness.state import new_state

        state = new_state("recovery-cancel")
        state["confirmed_config"] = {"LEDGER_DEVICE": "vda"}
        state["job"] = {"job_id": "job_failed"}
        state["failure_recovery"] = {"status": "pending", "record": build_failure_record(
            "WORKLOAD_PROCESS_FAILED", source="workload", severity="blocking", facts=[{"detail": "exit 2"}],
        )}
        result = apply_recovery_action(state, ActionProposal("cancel", "cancel_failure_recovery"))
        cancelled = self._commit_domain_delta(state, result, owner="recovery")
        self.assertEqual(cancelled["failure_recovery"]["status"], "cancelled")
        self.assertEqual(cancelled["confirmed_config"], state["confirmed_config"])
        self.assertEqual(cancelled["job"], state["job"])

    def test_old_checkpoint_migrates_with_empty_recovery(self) -> None:
        from agent.harness.state import migrate_state

        state = migrate_state(
            {"schema_version": 2, "target_mode": "fake-node"},
            thread_id="old-checkpoint",
            language="en",
            session_purpose="user",
        )
        self.assertEqual(state["schema_version"], 9)
        self.assertEqual(state["failure_recovery"], {})

    def test_blocked_approved_preflight_activates_recovery(self) -> None:
        from agent.harness.domains.execution_runtime import execute_approved_preflight_and_smoke
        from agent.runners.application_service import ExecutionOperation, ExecutionResult, ExecutionStatus
        from agent.harness.state import new_state

        state = new_state("blocked-preflight")
        state.update({"target_mode": "fake-node", "workflow_mode": "rpc_benchmark", "preflight": {"approved": True}})
        prepared = ExecutionResult(
            operation=ExecutionOperation.PREPARE,
            status=ExecutionStatus.BLOCKED,
            data={"plan": {}, "plan_file": "", "preflight": {"passed": False, "checks": [
                {"name": "effective_workload_fixtures_available", "passed": False, "detail": "missing fixture"},
            ]}},
            evidence_paths=("/workspace/evidence/preflight.json",),
        )
        with patch("agent.harness.domains.execution_runtime.execution_service.execute", return_value=prepared) as execute:
            result = execute_approved_preflight_and_smoke(state)
        recovered = self._commit_domain_delta(state, result, owner="execution")
        self.assertEqual(execute.call_args.args[0].operation, ExecutionOperation.PREPARE)
        self.assertEqual(result.recovery_command.operation, "activate")
        self.assertNotIn(("failure_recovery",), {write.path for write in result.delta.writes})
        self.assertEqual(recovered["failure_recovery"]["record"]["code"], "FIXTURE_MISSING")
        self.assertEqual(recovered["pending_question"]["id"], "failure_recovery_action")

    def test_only_current_job_receipt_can_activate_recovery(self) -> None:
        from agent.harness.domains.execution import reconcile_execution_state
        from agent.harness.state import new_state

        state = new_state("current-job")
        with patch("agent.runners.job_manager.get_job") as get_job:
            result = reconcile_execution_state(state)
        get_job.assert_not_called()
        state = self._commit_domain_delta(state, result, owner="execution")
        self.assertEqual(state["failure_recovery"], {})

        state["job"] = {"job_id": "job_current_failed"}
        persisted = {"job_id": "job_current_failed", "status": "failed", "error": "exit 2", "artifacts": {}}
        with patch("agent.runners.job_manager.get_job", return_value=persisted):
            result = reconcile_execution_state(state)
        state = self._commit_domain_delta(state, result, owner="execution")
        self.assertEqual(state["failure_recovery"]["status"], "pending")
        self.assertEqual(state["failure_recovery"]["record"]["job_id"], "job_current_failed")

    def test_result_classifier_emits_structured_zero_success_fact(self) -> None:
        from agent.runners.result_status import classify_benchmark_result

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifacts = {}
            for key in ("summary_json", "performance_csv", "html_report", "proxy_method_csv"):
                path = root / key
                path.write_text("evidence", encoding="utf-8")
                artifacts[key] = str(path)
            artifacts["archive_dir"] = str(root)
            vegeta = root / "vegeta.json"
            vegeta.write_text(json.dumps({"requests": 10, "success": 0, "status_codes": {"404": 10}}))
            artifacts["vegeta_json"] = str(vegeta)
            result = classify_benchmark_result({"workflow_type": "rpc_benchmark"}, 0, artifacts)
        self.assertEqual(result["status"], "failed")
        self.assertIn("WORKLOAD_ZERO_SUCCESS", {item["code"] for item in result["failure_facts"]})

    def test_invariant_failure_is_quarantined_as_recovery_state(self) -> None:
        from agent.harness.graph import AnyChainGraphRuntime
        from agent.harness.invariants import StateInvariantError
        from agent.harness.state import new_state

        runtime = object.__new__(AnyChainGraphRuntime)
        runtime.thread_id = "invariant"
        runtime.session_purpose = "chaos"
        runtime._persist_state = lambda state: state
        state = new_state("invariant")
        state["active_group"] = "not-a-real-group"
        recovered = runtime._recover_invariant_failure(state, StateInvariantError("unknown active group"))
        self.assertEqual(recovered["active_group"], "failure_recovery")
        self.assertEqual(recovered["failure_recovery"]["record"]["code"], "HARNESS_INVARIANT_FAILED")
        self.assertEqual(recovered["pending_question"]["id"], "failure_recovery_action")

    def test_complete_single_line_failure_question_is_analyzed_without_paste_mode(self) -> None:
        from agent.harness.domains.analysis import analyze_inline_evidence_result
        from agent.harness.state import new_state

        state = new_state("inline-evidence")
        text = "My custom RPC returned only HTTP 404. Explain what failed and what to fix next."
        with patch("agent.harness.domains.analysis.analyze_evidence_with_model", return_value="analysis"):
            result = analyze_inline_evidence_result(state, text)
        self.assertEqual(result.visible_result, "analysis")
        analyzed = self._commit_domain_delta(state, result, owner="analysis")
        self.assertEqual(analyzed["evidence_buffer"][-1]["text"], text)
        self.assertFalse(result.pending_question)

    def test_current_state_consultation_exposes_domain_failure_and_correction_field(self) -> None:
        from agent.harness.domains.orientation import answer_consultation
        from agent.harness.failures import build_failure_record
        from agent.harness.state import new_state

        state = new_state("endpoint-failure", language="en")
        state["confirmed_config"] = {"CLOUD_REGION": "us-east1", "LEDGER_DEVICE": "vda"}
        state["pending_question"] = {
            "id": "custom_rpc_endpoint",
            "field": "custom_rpc_endpoint",
            "group": "endpoint_process",
        }
        state["endpoint_evidence"] = {
            "last_failure_record": build_failure_record(
                "ENDPOINT_UNREACHABLE",
                source="endpoint",
                severity="blocking",
                facts=[{"detail": "connection refused"}],
                evidence_paths=[".agent/evidence/probe.json"],
                confirmed_config=state["confirmed_config"],
            )
        }

        response = answer_consultation(state, {"topic": "current_config"})

        self.assertIn("ENDPOINT_UNREACHABLE", response)
        self.assertIn("connection refused", response)
        self.assertIn("custom_rpc_endpoint", response)
        self.assertIn("CLOUD_REGION", response)
        self.assertIn("LEDGER_DEVICE", response)

    def test_successful_endpoint_revalidation_clears_stale_domain_failure(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn
        from agent.harness.domains.chain_rpc import question_for_chain_rpc
        from agent.harness.failures import build_failure_record
        from agent.harness.state import new_state

        state = new_state("endpoint-revalidated", language="en")
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "status": "confirmed"}
        state["custom_rpc"] = {"status": "needs_endpoint", "method": "eth_accounts"}
        state["active_group"] = "endpoint_process"
        state["endpoint_evidence"] = {
            "last_failure_record": build_failure_record(
                "ENDPOINT_UNREACHABLE",
                source="endpoint",
                severity="blocking",
                facts=[{"detail": "connection refused"}],
            )
        }
        probe = {"ready": True, "status": "ok", "evidence_file": ".agent/evidence/probe-ok.json"}
        state["pending_question"] = question_for_chain_rpc(state, "endpoint_process") or {}
        state["last_user_input"] = "https://rpc.example"

        with patch("agent.harness.domains.rpc_endpoint.validate_rpc_endpoint", return_value=probe):
            state = invoke_product_graph_turn(state)

        self.assertNotIn("last_failure_record", state["endpoint_evidence"])
        self.assertTrue(state["custom_rpc"]["endpoint_ready"])

    def test_failed_custom_endpoint_keeps_correction_question_ahead_of_deferred_groups(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn
        from agent.harness.domains.chain_rpc import question_for_chain_rpc
        from agent.harness.state import new_state

        state = new_state("endpoint-correction-order", language="en")
        state.update({
            "turn_index": 3,
            "target_mode": "fake-node",
            "workflow_mode": "rpc_benchmark",
            "chain_identity": {"raw": "bsc", "canonical": "bsc", "status": "confirmed", "adapter_family": "jsonrpc"},
            "custom_rpc": {"status": "needs_endpoint"},
            "active_group": "endpoint_process",
            "action_queue": [{"type": "set_qps_mode", "qps_mode": "quick", "confidence": "high"}],
        })
        state["pending_question"] = question_for_chain_rpc(state, "endpoint_process") or {}
        state["pending_question"]["resume_action_queue"] = True
        state["last_user_input"] = "http://127.0.0.1:9"
        probe = {
            "ready": False,
            "status": "needs_valid_endpoint",
            "error": "connection refused",
            "evidence_file": ".agent/evidence/probe-failed.json",
        }

        with patch("agent.harness.domains.rpc_endpoint.validate_rpc_endpoint", return_value=probe):
            result = invoke_product_graph_turn(state)

        self.assertEqual((result.get("pending_question") or {}).get("id"), "custom_rpc_endpoint")
        self.assertTrue((result.get("pending_question") or {}).get("queue_barrier"))
        self.assertEqual([item.get("type") for item in result.get("action_queue") or []], ["set_qps_mode"])
        self.assertEqual((result.get("custom_rpc") or {}).get("status"), "probe_failed")
        prompt = str((result.get("pending_question") or {}).get("prompt") or "")
        self.assertEqual(sum(prompt in item for item in result.get("visible_response") or []), 1)

    def test_successful_custom_endpoint_shows_method_question_without_consuming_deferred_qps(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn
        from agent.harness.domains.chain_rpc import question_for_chain_rpc
        from agent.harness.state import new_state

        state = new_state("endpoint-success-order", language="en")
        state.update({
            "turn_index": 3,
            "target_mode": "fake-node",
            "workflow_mode": "rpc_benchmark",
            "chain_identity": {"raw": "bsc", "canonical": "bsc", "status": "confirmed", "adapter_family": "jsonrpc"},
            "custom_rpc": {"status": "needs_endpoint"},
            "active_group": "endpoint_process",
            "action_queue": [{"type": "set_qps_mode", "qps_mode": "quick", "confidence": "high"}],
        })
        state["pending_question"] = question_for_chain_rpc(state, "endpoint_process") or {}
        state["pending_question"]["resume_action_queue"] = True
        state["last_user_input"] = "https://rpc.example"
        probe = {
            "ready": True,
            "status": "ok",
            "evidence_file": ".agent/evidence/probe-ok.json",
        }

        with patch("agent.harness.domains.rpc_endpoint.validate_rpc_endpoint", return_value=probe):
            result = invoke_product_graph_turn(state)

        self.assertEqual((result.get("pending_question") or {}).get("id"), "custom_rpc_method")
        self.assertEqual([item.get("type") for item in result.get("action_queue") or []], ["set_qps_mode"])
        prompt = str((result.get("pending_question") or {}).get("prompt") or "")
        self.assertEqual(sum(prompt in item for item in result.get("visible_response") or []), 1)


if __name__ == "__main__":
    unittest.main()

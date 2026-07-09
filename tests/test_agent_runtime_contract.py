"""Runtime contract tests for the LangGraph Harness and ADK boundary."""

from __future__ import annotations

import unittest
from pathlib import Path


class AgentRuntimeContractTest(unittest.TestCase):
    def test_retired_runtime_files_are_absent(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        retired = [
            "agent/workflows/conversation_state.py",
            "agent/workflows/transition_executor.py",
            "agent/adk_app/callbacks.py",
            "agent/adk_app/agents/domain.py",
            "agent/adk_app/tools/workflow_state.py",
            "agent/adk_app/workflow/product_context.py",
            "agent/terminal/input_classifier.py",
            "agent/terminal/pending_answers.py",
        ]
        existing = [item for item in retired if (repo / item).exists()]
        self.assertEqual(existing, [])

    def test_adk_tools_do_not_expose_workflow_mutation_tools(self) -> None:
        from agent.adk_app.tools.registry import get_adk_tools

        names = {getattr(item, "__name__", str(item)) for item in get_adk_tools()}
        self.assertFalse({"load_workflow_state", "update_workflow_state", "answer_pending_question"} & names)

    def test_harness_group_registry_is_authoritative_for_product_flow(self) -> None:
        from agent.harness.state import DEFAULT_GROUP_ORDER

        expected_groups = {
            "opening",
            "target_mode",
            "chain_identity",
            "provider_deployment",
            "ledger_disk",
            "accounts_disk",
            "network",
            "endpoint_process",
            "workload_rpc",
            "qps_profile",
            "sync_observe",
            "observability",
            "preflight_smoke_execution",
            "job_monitoring",
        }
        self.assertTrue(expected_groups.issubset(set(DEFAULT_GROUP_ORDER)))


if __name__ == "__main__":
    unittest.main()

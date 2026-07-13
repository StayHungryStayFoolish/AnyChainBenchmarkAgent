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
            "agent/terminal/input_classifier.py",
            "agent/terminal/pending_answers.py",
            "agent/adk_app",
        ]
        existing = [item for item in retired if (repo / item).exists()]
        self.assertEqual(existing, [])

    def test_tool_dispatch_does_not_expose_workflow_mutation_tools(self) -> None:
        from agent.tools.schema import tool_schema

        names = {tool["function"]["name"] for tool in tool_schema()["tools"]}
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

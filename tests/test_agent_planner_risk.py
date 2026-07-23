"""Workflow-aware risk-scoring contracts."""

from __future__ import annotations

import unittest

from agent.planners.risk import score_plan_risk


class PlannerRiskTest(unittest.TestCase):
    def test_sync_observe_uses_its_observation_endpoint(self) -> None:
        risk = score_plan_risk({
            "workflow_type": "sync_observe",
            "use_fake_node": False,
            "execution": {
                "environment": {
                    "LOCAL_RPC_URL": "",
                    "SYNC_OBSERVE_RPC_URL": "http://geth-dev:8545",
                    "OBSERVABILITY_STACK_ENABLED": "false",
                },
            },
        })

        messages = [str(item["message"]) for item in risk["findings"]]
        self.assertNotIn("rpc_benchmark selected without LOCAL_RPC_URL.", messages)
        self.assertNotIn(
            "sync_observe selected without SYNC_OBSERVE_RPC_URL.",
            messages,
        )

    def test_sync_observe_missing_endpoint_has_workflow_specific_guidance(self) -> None:
        risk = score_plan_risk({
            "workflow_type": "sync_observe",
            "use_fake_node": False,
            "execution": {
                "environment": {
                    "LOCAL_RPC_URL": "http://unrelated-rpc-benchmark:8545",
                    "SYNC_OBSERVE_RPC_URL": "",
                    "OBSERVABILITY_STACK_ENABLED": "true",
                },
            },
        })

        self.assertEqual(risk["risk_level"], "medium")
        self.assertIn(
            "sync_observe selected without SYNC_OBSERVE_RPC_URL.",
            [str(item["message"]) for item in risk["findings"]],
        )
        self.assertEqual(
            risk["recommendations"],
            ["Provide a validated SYNC_OBSERVE_RPC_URL for the real node being observed."],
        )

    def test_rpc_benchmark_still_requires_local_rpc_url(self) -> None:
        risk = score_plan_risk({
            "workflow_type": "rpc_benchmark",
            "use_fake_node": False,
            "execution": {
                "environment": {
                    "LOCAL_RPC_URL": "",
                    "SYNC_OBSERVE_RPC_URL": "http://unrelated-observer:8545",
                    "OBSERVABILITY_STACK_ENABLED": "true",
                },
            },
        })

        self.assertIn(
            "rpc_benchmark selected without LOCAL_RPC_URL.",
            [str(item["message"]) for item in risk["findings"]],
        )
        self.assertEqual(
            risk["recommendations"],
            ["Provide LOCAL_RPC_URL or switch to fake-node for local closed-loop validation."],
        )


if __name__ == "__main__":
    unittest.main()

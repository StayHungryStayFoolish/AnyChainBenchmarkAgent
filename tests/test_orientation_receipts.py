"""Contracts for read-only orientation response receipts."""

from __future__ import annotations

import unittest

from agent.harness.contracts import ActionProposal
from agent.harness.domains.orientation import apply_orientation_action
from agent.harness.domains.orientation_receipts import (
    validate_orientation_receipt,
)
from agent.harness.state import new_state
from tests.agent_live.coverage_evidence import content_hash


class OrientationReceiptTest(unittest.TestCase):
    def test_current_state_answer_emits_secret_free_snapshot_receipt(self) -> None:
        state = new_state("orientation-receipt", language="en")
        state["turn_index"] = 4
        state["target_mode"] = "real-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"canonical": "bsc", "status": "confirmed"}
        state["confirmed_config"] = {
            "CLOUD_REGION": "test-region",
            "LOCAL_RPC_URL": "https://node.invalid/private-token",
        }
        action = ActionProposal(
            "orientation-1",
            "answer_opening_question",
            {"topic": "current_config"},
            "high",
        )

        result = apply_orientation_action(state, action)

        self.assertEqual(len(result.control_receipts), 1)
        receipt = dict(result.control_receipts[0])
        valid, reason = validate_orientation_receipt(receipt)
        self.assertTrue(valid, reason)
        self.assertEqual(receipt["topic"], "current_config")
        self.assertTrue(receipt["read_only"])
        self.assertNotIn("private-token", str(receipt))
        self.assertNotIn("test-region", str(receipt))
        self.assertIn("confirmed_fields", receipt["projection_fields"])

    def test_rehashed_orientation_receipt_with_mutating_claim_is_rejected(
        self,
    ) -> None:
        state = new_state("orientation-receipt", language="en")
        state["turn_index"] = 1
        action = ActionProposal(
            "orientation-2",
            "answer_opening_question",
            {"topic": "identity"},
            "high",
        )
        receipt = dict(
            apply_orientation_action(state, action).control_receipts[0]
        )
        receipt["read_only"] = False
        receipt["receipt_id"] = content_hash(
            {key: value for key, value in receipt.items() if key != "receipt_id"}
        )

        valid, _ = validate_orientation_receipt(receipt)

        self.assertFalse(valid)


if __name__ == "__main__":
    unittest.main()

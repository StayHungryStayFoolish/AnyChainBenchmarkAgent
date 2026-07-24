"""Contracts for unknown-chain identity research receipts."""

from __future__ import annotations

import unittest

from agent.harness.domains.chain_identity_receipts import (
    emit_chain_identity_resolution_receipt,
    validate_chain_identity_receipt,
)
from agent.harness.state import new_state
from tests.agent_live.coverage_evidence import content_hash


class ChainIdentityReceiptTest(unittest.TestCase):
    def test_identity_receipt_records_model_and_search_provenance_without_text(
        self,
    ) -> None:
        state = new_state("identity-receipt")
        state["turn_index"] = 3
        candidate = "private-chain-candidate"
        summary = "private official documentation summary"

        receipt = emit_chain_identity_resolution_receipt(
            state,
            candidate=candidate,
            resolution={
                "chain_exists": True,
                "canonical_chain_name": "candidate-mainnet",
                "adapter_family": "jsonrpc",
                "confidence": "high",
                "search_result": {
                    "available": True,
                    "text_summary": summary,
                },
            },
            resolver_source="llm",
            confirmation_required=True,
        )

        valid, reason = validate_chain_identity_receipt(receipt)
        self.assertTrue(valid, reason)
        self.assertTrue(receipt["google_search_invoked"])
        self.assertTrue(receipt["google_search_available"])
        self.assertNotIn(candidate, str(receipt))
        self.assertNotIn(summary, str(receipt))

    def test_rehashed_identity_receipt_cannot_skip_user_confirmation(self) -> None:
        state = new_state("identity-receipt")
        receipt = emit_chain_identity_resolution_receipt(
            state,
            candidate="candidate",
            resolution={"chain_exists": None, "adapter_family": "unknown"},
            resolver_source="llm",
            confirmation_required=True,
        )
        receipt["confirmation_required"] = False
        receipt["receipt_id"] = content_hash(
            {key: value for key, value in receipt.items() if key != "receipt_id"}
        )

        valid, _ = validate_chain_identity_receipt(receipt)

        self.assertFalse(valid)


if __name__ == "__main__":
    unittest.main()

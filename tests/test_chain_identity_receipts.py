"""Contracts for unknown-chain identity research receipts."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agent.harness.advisory import (
    normalize_chain_identity_resolution,
    resolve_unknown_chain_identity,
)
from agent.harness.domains.chain_identity_receipts import (
    emit_chain_identity_resolution_receipt,
    validate_chain_identity_receipt,
)
from agent.harness.state import new_state
from tests.agent_live.coverage_evidence import content_hash


class ChainIdentityReceiptTest(unittest.TestCase):
    def test_plain_search_shaped_model_data_cannot_claim_search_provenance(
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
                "reference_kind": "named_identity",
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
        self.assertFalse(receipt["google_search_invoked"])
        self.assertFalse(receipt["google_search_available"])
        self.assertNotIn(candidate, str(receipt))
        self.assertNotIn(summary, str(receipt))

    def test_external_search_after_sanitized_model_output_is_authoritative(
        self,
    ) -> None:
        state = new_state("trusted-search-receipt")
        state["turn_index"] = 4
        provider = SimpleNamespace(
            complete=lambda _request: SimpleNamespace(
                text=(
                    '{"reference_kind":"named_identity",'
                    '"chain_exists":true,'
                    '"canonical_chain_name":"candidate-mainnet",'
                    '"adapter_family":"jsonrpc",'
                    '"confidence":"high"}'
                )
            )
        )
        with patch(
            "agent.harness.advisory.provider_from_config",
            return_value=provider,
        ):
            resolution = resolve_unknown_chain_identity(
                state,
                "candidate-mainnet",
            )
        resolution["search_result"] = {
            "available": True,
            "text_summary": "Official documentation confirms the network.",
        }

        receipt = emit_chain_identity_resolution_receipt(
            state,
            candidate="candidate-mainnet",
            resolution=resolution,
            resolver_source="llm",
            confirmation_required=True,
        )

        valid, reason = validate_chain_identity_receipt(receipt)
        self.assertTrue(valid, reason)
        self.assertTrue(receipt["google_search_invoked"])
        self.assertTrue(receipt["google_search_available"])

    def test_model_cannot_forge_search_or_the_internal_boundary_marker(
        self,
    ) -> None:
        state = new_state("forged-search-receipt")
        provider = SimpleNamespace(
            complete=lambda _request: SimpleNamespace(
                text=(
                    '{"reference_kind":"named_identity",'
                    '"chain_exists":true,'
                    '"canonical_chain_name":"invented",'
                    '"adapter_family":"jsonrpc",'
                    '"confidence":"high",'
                    '"_model_provenance_boundary":"chain_identity_v1",'
                    '"google_search_invoked":true,'
                    '"search_result":{"available":true,'
                    '"text_summary":"fabricated result"}}'
                )
            )
        )
        with patch(
            "agent.harness.advisory.provider_from_config",
            return_value=provider,
        ):
            resolution = resolve_unknown_chain_identity(
                state,
                "invented",
            )

        receipt = emit_chain_identity_resolution_receipt(
            state,
            candidate="invented",
            resolution=resolution,
            resolver_source="llm",
            confirmation_required=True,
        )

        valid, reason = validate_chain_identity_receipt(receipt)
        self.assertTrue(valid, reason)
        self.assertNotIn("search_result", resolution)
        self.assertNotIn("google_search_invoked", resolution)
        self.assertFalse(receipt["google_search_invoked"])
        self.assertFalse(receipt["google_search_available"])

    def test_unresolved_reference_discards_even_post_boundary_search_data(
        self,
    ) -> None:
        state = new_state("generic-search-receipt")
        provider = SimpleNamespace(
            complete=lambda _request: SimpleNamespace(
                text=(
                    '{"reference_kind":"generic_reference",'
                    '"chain_exists":null,'
                    '"canonical_chain_name":"",'
                    '"adapter_family":"unknown",'
                    '"confidence":"high"}'
                )
            )
        )
        with patch(
            "agent.harness.advisory.provider_from_config",
            return_value=provider,
        ):
            resolution = resolve_unknown_chain_identity(
                state,
                "another chain",
            )
        resolution["search_result"] = {
            "available": True,
            "text_summary": "Irrelevant search output.",
        }
        resolution = normalize_chain_identity_resolution(resolution)

        receipt = emit_chain_identity_resolution_receipt(
            state,
            candidate="another chain",
            resolution=resolution,
            resolver_source="llm",
            confirmation_required=True,
        )

        valid, reason = validate_chain_identity_receipt(receipt)
        self.assertTrue(valid, reason)
        self.assertNotIn("search_result", resolution)
        self.assertFalse(receipt["google_search_invoked"])
        self.assertFalse(receipt["google_search_available"])

    def test_rehashed_identity_receipt_cannot_skip_user_confirmation(self) -> None:
        state = new_state("identity-receipt")
        receipt = emit_chain_identity_resolution_receipt(
            state,
            candidate="candidate",
            resolution={
                "reference_kind": "uncertain",
                "chain_exists": None,
                "adapter_family": "unknown",
            },
            resolver_source="llm",
            confirmation_required=True,
        )
        receipt["confirmation_required"] = False
        receipt["receipt_id"] = content_hash(
            {key: value for key, value in receipt.items() if key != "receipt_id"}
        )

        valid, _ = validate_chain_identity_receipt(receipt)

        self.assertFalse(valid)

    def test_uncertain_identity_receipt_is_distinct_and_fail_closed(
        self,
    ) -> None:
        state = new_state("uncertain-identity-receipt")
        receipt = emit_chain_identity_resolution_receipt(
            state,
            candidate="unclear input",
            resolution={
                "reference_kind": "uncertain",
                "chain_exists": None,
                "adapter_family": "unknown",
            },
            resolver_source="llm",
            confirmation_required=True,
        )

        valid, reason = validate_chain_identity_receipt(receipt)

        self.assertTrue(valid, reason)
        self.assertEqual(receipt["schema_version"], 2)
        self.assertEqual(receipt["reference_kind"], "uncertain")

    def test_unresolved_identity_receipt_cannot_hide_identity_proposals(
        self,
    ) -> None:
        state = new_state("generic-hidden-identity")
        receipt = emit_chain_identity_resolution_receipt(
            state,
            candidate="unclear input",
            resolution={
                "reference_kind": "generic_reference",
                "chain_exists": None,
                "canonical_chain_name": "hidden-name",
                "possible_known_chain": "ethereum",
                "adapter_family": "unknown",
            },
            resolver_source="llm",
            confirmation_required=True,
        )

        valid, reason = validate_chain_identity_receipt(receipt)

        self.assertTrue(valid, reason)
        self.assertEqual(
            receipt["canonical_name_hash"],
            content_hash(""),
        )
        self.assertEqual(
            receipt["possible_known_chain_hash"],
            content_hash(""),
        )


if __name__ == "__main__":
    unittest.main()

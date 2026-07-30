"""Inference contracts for read-only model-backed advisory services."""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agent.harness.advisory import (
    analyze_evidence_with_model,
    extract_chain_mention,
    extract_rpc_schema_from_evidence,
    resolve_unknown_chain_identity,
)
from agent.harness.state import new_state


class _RecordingProvider:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        return SimpleNamespace(
            text=self.responses.pop(0),
            provider="deepseek",
            model="deepseek-v4-pro",
        )


class AdvisoryInferenceContractTest(unittest.TestCase):
    def test_structured_advisories_share_strict_json_inference_contract(
        self,
    ) -> None:
        provider = _RecordingProvider([
            json.dumps({
                "reference_kind": "named_identity",
                "chain_exists": True,
                "canonical_chain_name": "candidate",
                "adapter_family": "jsonrpc",
                "confidence": "medium",
            }),
            json.dumps({
                "found": True,
                "chain_text": "candidate",
                "confidence": "medium",
            }),
            json.dumps({
                "status": "draft",
                "method": "eth_chainId",
                "params": [],
                "response_json_type": "string",
            }),
        ])
        state = new_state("structured-advisory-contract")

        with patch(
            "agent.harness.advisory.provider_from_config",
            return_value=provider,
        ):
            resolve_unknown_chain_identity(state, "candidate")
            extract_chain_mention(state, "switch to candidate")
            extract_rpc_schema_from_evidence(
                state,
                '{"method":"eth_chainId","params":[]}',
                method_hint="eth_chainId",
            )

        self.assertEqual(len(provider.requests), 3)
        self.assertTrue(all(
            request.reasoning_mode == "disabled"
            for request in provider.requests
        ))
        self.assertTrue(all(
            request.replay_safety == "side_effect_free"
            for request in provider.requests
        ))

    def test_freeform_evidence_analysis_keeps_provider_reasoning_available(
        self,
    ) -> None:
        provider = _RecordingProvider(["Observed evidence only."])
        state = new_state("freeform-advisory-contract")

        with patch(
            "agent.harness.advisory.provider_from_config",
            return_value=provider,
        ):
            answer = analyze_evidence_with_model(
                state,
                "ERROR timeout",
                "What failed?",
            )

        self.assertEqual(answer, "Observed evidence only.")
        self.assertEqual(
            provider.requests[0].reasoning_mode,
            "provider_default",
        )


if __name__ == "__main__":
    unittest.main()

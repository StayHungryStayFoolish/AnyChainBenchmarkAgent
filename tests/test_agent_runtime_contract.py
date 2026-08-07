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

    def test_each_workflow_group_has_one_domain_owner(self) -> None:
        from agent.harness.domains.registry import GROUP_OWNER, validate_domain_ownership
        from agent.harness.state import DEFAULT_GROUP_ORDER

        validate_domain_ownership()
        self.assertEqual(set(GROUP_OWNER), set(DEFAULT_GROUP_ORDER))

    def test_visible_choice_requires_action_and_postcondition(self) -> None:
        from agent.harness.contracts import (
            ActionProposal,
            OptionContract,
            QuestionContract,
            TextRef,
        )

        with self.assertRaises(ValueError):
            QuestionContract(
                question_id="broken",
                group="opening",
                owner="orientation",
                kind="numbered_choice",
                prompt=TextRef("question.instruction.option"),
                options=(
                    OptionContract(
                        option_id="1",
                        value="noop",
                        action=ActionProposal(action_id="a1", action_type="answer_pending"),
                        expected_patch={},
                        label=TextRef("question.common.option.yes"),
                    ),
                ),
            )

    def test_environment_disk_choice_is_typed_and_invalidates_only_downstream(self) -> None:
        from agent.harness.domains.environment import apply_environment_answer, question_for_environment
        from agent.harness.invariants import validate_state, verify_expected_patch
        from agent.harness.questions import exact_answer, expected_patch_for_value
        from agent.harness.state import new_state

        state = new_state("environment-contract")
        state["active_group"] = "ledger_disk"
        state["invalidated_groups"] = ["ledger_disk"]
        state["discovery"] = {
            "disks": {
                "candidates": [
                    {"name": "vda", "size": "926.3G", "type": "disk"},
                    {"name": "nbd0", "size": "0B", "type": "disk"},
                ]
            }
        }
        question = question_for_environment(state, "ledger_disk")
        self.assertIsNotNone(question)
        self.assertEqual(question["options"][0]["value"], "vda")
        self.assertTrue(question["options"][0]["action"])
        self.assertTrue(question["options"][0]["expected_patch"])

        matched, value = exact_answer("1", question)
        self.assertTrue(matched)
        handler_result = apply_environment_answer(state, question, value)
        from agent.harness.invariants import apply_state_delta

        self.assertFalse(handler_result.delta.is_empty())
        result = apply_state_delta(state, handler_result.delta, owner="environment")
        expected = expected_patch_for_value(question, value)
        verify_expected_patch(result, expected)
        self.assertEqual(result["confirmed_config"]["LEDGER_DEVICE"], "vda")
        self.assertEqual(handler_result.reconfigured_groups, ("ledger_disk",))
        self.assertIn("preflight_smoke_execution", handler_result.invalidated_groups)
        result["invalidated_groups"] = sorted(
            (set(result["invalidated_groups"]) - set(handler_result.reconfigured_groups))
            | set(handler_result.invalidated_groups)
        )
        result["pending_question"] = question_for_environment(result, "ledger_disk") or {}
        validate_state(result)

    def test_environment_scalar_normalization_is_not_transcript_specific(self) -> None:
        from agent.harness.questions import normalize_scalar

        self.assertEqual(normalize_scalar("  hyperdisk-balanced,  "), "hyperdisk-balanced")
        self.assertEqual(normalize_scalar("eth0；"), "eth0")

    def test_endpoint_extraction_removes_terminal_copy_punctuation(self) -> None:
        from agent.harness.input_values import (
            extract_url_candidate,
            extract_url_candidates,
        )

        self.assertEqual(
            extract_url_candidate("  http://geth-dev:8545,  "),
            "http://geth-dev:8545",
        )
        self.assertEqual(
            extract_url_candidate("endpoint: https://rpc.example/v1。"),
            "https://rpc.example/v1",
        )
        reference = "semantic-secret:60FomrgVQ8d470RPHw5_NKbB-xyh-0PS"
        text = f"Use {reference} instead."
        self.assertEqual(extract_url_candidate(text), "")
        self.assertEqual(extract_url_candidates(text), ())

    def test_orientation_consultation_cannot_steal_pending_workflow_control(self) -> None:
        from agent.harness.contracts import ActionProposal
        from agent.harness.domains.orientation import apply_orientation_action
        from agent.harness.state import new_state

        state = new_state("orientation-contract", language="zh")
        state["active_group"] = "provider_deployment"
        state["pending_question"] = {
            "id": "CLOUD_REGION",
            "group": "provider_deployment",
            "kind": "manual_value",
            "prompt": "请输入 CLOUD_REGION",
        }
        result = apply_orientation_action(
            state,
            ActionProposal(
                action_id="consult-1",
                action_type="answer_opening_question",
                arguments={"topic": "identity"},
                confidence="high",
            ),
        )
        self.assertTrue(result.delta.is_empty())
        self.assertEqual(state["active_group"], "provider_deployment")
        self.assertEqual(state["pending_question"]["id"], "CLOUD_REGION")
        from agent.harness.response_catalog import render_fragment

        self.assertIn(
            "AnyChain Benchmark Agent",
            render_fragment(result.response_fragments[0], "en").text,
        )


if __name__ == "__main__":
    unittest.main()

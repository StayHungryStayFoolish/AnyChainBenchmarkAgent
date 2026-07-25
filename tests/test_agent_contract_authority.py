"""R41 contract-authority regressions for Harness actions and Agent tools."""

from __future__ import annotations

import inspect
import ast
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


class ActionContractAuthorityTest(unittest.TestCase):
    def test_coordinator_does_not_reimplement_question_contracts(self) -> None:
        import agent.harness.coordinator as coordinator

        source = inspect.getsource(coordinator)
        for forbidden in (
            "def _answer_fits_pending(",
            "def _coerce_answer(",
            "def _render_question(",
            "def _match_option(",
        ):
            self.assertNotIn(forbidden, source)
        self.assertNotIn('"inferred_config_review"', source)

    def test_coordinator_preplanner_only_admits_declared_exact_choices(self) -> None:
        from agent.harness.coordinator import adjudicate_turn_step
        from agent.harness.state import new_state

        state = new_state("exact-choice-only", language="en")
        state["pending_question"] = {
            "contract_version": 2,
            "id": "mode",
            "group": "target_mode",
            "owner": "chain_rpc",
            "kind": "numbered_choice",
            "prompt": "Choose.",
            "options": [
                {
                    "id": "1",
                    "label": "fake-node",
                    "value": "fake-node",
                    "action": {"type": "choose_target_mode"},
                },
            ],
            "manual_input_allowed": False,
            "accepted_action_types": ["answer_pending", "choose_target_mode"],
        }
        for text, expected_phase in (
            ("1", "execute"),
            ("fake-node", "execute"),
            ("please use fake-node", "plan"),
            ("2", "plan"),
        ):
            with self.subTest(text=text):
                candidate = dict(state)
                candidate["turn_context"] = {
                    "text": text,
                    "kind": "free_text",
                    "input_shape": "prose",
                }
                candidate["turn_receipt"] = {"clauses": []}
                result = adjudicate_turn_step(candidate)
                self.assertEqual(result["control"]["phase"], expected_phase)

    def test_admission_is_atomic_and_cannot_rewrite_action_semantics(self) -> None:
        import agent.harness.admission as admission
        import agent.harness.semantic_admission as semantic_admission

        source = inspect.getsource(admission.validate_action_plan)
        tree = ast.parse(source)
        forbidden_calls = {"append", "extend", "insert", "pop", "remove"}
        self.assertFalse(
            any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in forbidden_calls
                for node in ast.walk(tree)
            )
        )
        self.assertNotIn('["type"] =', source)
        self.assertNotIn("['type'] =", source)
        self.assertFalse(hasattr(semantic_admission, "_apply_state_plan_policy"))

    def test_action_schema_is_generated_from_specs(self) -> None:
        from agent.harness.action_registry import ACTION_SPECS
        from agent.harness.context import action_schema

        rendered = {item["type"]: item for item in action_schema()}
        self.assertEqual(
            set(rendered),
            {spec.action_type for spec in ACTION_SPECS if not spec.internal_only},
        )
        for spec in ACTION_SPECS:
            if spec.internal_only:
                continue
            self.assertEqual(rendered[spec.action_type]["allowed_arguments"], list(spec.allowed_arguments))
            self.assertEqual(rendered[spec.action_type]["required_arguments"], list(spec.required_arguments))
            self.assertEqual(rendered[spec.action_type]["constraints"], list(spec.constraints))

    def test_action_contract_rejects_missing_and_undeclared_arguments(self) -> None:
        from agent.harness.action_registry import validate_action_contract

        with self.assertRaisesRegex(ValueError, "missing required arguments.*qps_mode"):
            validate_action_contract({"type": "set_qps_mode", "mutation_explicit": True, "source_evidence": "quick"})
        with self.assertRaisesRegex(ValueError, "undeclared arguments.*invented"):
            validate_action_contract({
                "type": "set_qps_mode",
                "qps_mode": "quick",
                "mutation_explicit": True,
                "source_evidence": "quick",
                "invented": "value",
            })
        with self.assertRaisesRegex(ValueError, "chain_text or chain_candidates"):
            validate_action_contract({
                "type": "choose_chain",
                "source_evidence": "BNB",
            })

    def test_pending_answer_uses_one_canonical_representation(self) -> None:
        from agent.harness.action_registry import validate_action_contract

        selected = validate_action_contract({
            "type": "answer_pending",
            "selected_value": "disabled",
            "source_evidence": "Disable observability",
        })
        manual = validate_action_contract({
            "type": "answer_pending",
            "answer": "custom-value",
            "source_evidence": "Use custom-value",
        })

        self.assertNotIn("answer", selected)
        self.assertNotIn("selected_value", manual)
        with self.assertRaisesRegex(
            ValueError,
            "conflicting answer and selected_value representations",
        ):
            validate_action_contract({
                "type": "answer_pending",
                "answer": "Disabled",
                "selected_value": "disabled",
                "source_evidence": "Disable observability",
            })

    def test_optional_empty_model_arguments_are_equivalent_to_omission(self) -> None:
        from agent.harness.action_registry import validate_action_contract

        action = validate_action_contract({
            "type": "rpc_catalog_command",
            "catalog_command": "set_method",
            "rpc_method": "eth_getBalance",
            "source_evidence": "use eth_getBalance",
            "rpc_endpoint": "",
            "rpc_schema_evidence": None,
        })

        self.assertEqual(
            action,
            {
                "type": "rpc_catalog_command",
                "catalog_command": "set_method",
                "rpc_method": "eth_getBalance",
                "source_evidence": "use eth_getBalance",
            },
        )

    def test_parser_rejects_invalid_action_instead_of_leaking_arguments(self) -> None:
        from agent.harness.semantic_admission import _parse_action_queue

        parsed = _parse_action_queue(
            '{"actions":[{"type":"set_qps_override","qps_overrides":{"initial":1},"extra":true}]}'
        )
        self.assertEqual(parsed["actions"][0]["type"], "unknown")
        self.assertIn("undeclared arguments", parsed["actions"][0]["reason"])
        self.assertNotIn("extra", parsed["actions"][0])

    def test_model_cannot_forge_semantic_admission_receipt(self) -> None:
        from agent.harness.semantic_admission import _parse_action_queue

        parsed = _parse_action_queue(
            '{"actions":[{"type":"choose_target_mode","target_mode":"fake-node",'
            '"target_mode_explicit":true,"source_evidence":"simulated node",'
            '"semantic_purpose_verified":true}]}'
        )

        self.assertEqual(parsed["actions"][0]["type"], "unknown")
        self.assertIn("undeclared arguments", parsed["actions"][0]["reason"])

    def test_compound_action_constraints_are_admitted_once_at_registry_boundary(self) -> None:
        from agent.harness.action_registry import validate_action_contract

        with self.assertRaisesRegex(ValueError, "set_endpoint requires rpc_endpoint"):
            validate_action_contract({
                "type": "rpc_catalog_command",
                "catalog_command": "set_endpoint",
                "source_evidence": "set endpoint",
            })
        with self.assertRaisesRegex(ValueError, "does not accept: rpc_method"):
            validate_action_contract({
                "type": "rpc_catalog_command",
                "catalog_command": "set_endpoint",
                "rpc_endpoint": "https://example.invalid",
                "rpc_method": "eth_chainId",
                "source_evidence": "use https://example.invalid and eth_chainId",
            })
        with self.assertRaisesRegex(ValueError, "exact wire method token"):
            validate_action_contract({
                "type": "rpc_catalog_command",
                "catalog_command": "set_method",
                "rpc_method": "my own RPC method",
                "source_evidence": "my own RPC method",
            })
        self.assertEqual(
            validate_action_contract({
                "type": "rpc_catalog_command",
                "catalog_command": "set_method",
                "rpc_method": "eth_blockNumber",
                "source_evidence": "eth_blockNumber",
            })["rpc_method"],
            "eth_blockNumber",
        )
        self.assertEqual(
            validate_action_contract({
                "type": "rpc_catalog_command",
                "catalog_command": "set_method",
                "rpc_method": "GET /v1/blocks/{height}",
                "source_evidence": "GET /v1/blocks/{height}",
            })["rpc_method"],
            "GET /v1/blocks/{height}",
        )
        with self.assertRaisesRegex(ValueError, "must total 100"):
            validate_action_contract({
                "type": "rpc_workload_command",
                "workload_scope": "mixed_replace",
                "rpc_weights": {"eth_chainId": 70},
            })
        with self.assertRaisesRegex(ValueError, "duration stop condition requires"):
            validate_action_contract({
                "type": "set_sync_observe_options",
                "sync_observe_stop_condition": "duration",
            })

    def test_legacy_nested_arguments_exist_only_in_v12_checkpoint_adapter(self) -> None:
        from agent.harness.action_registry import validate_action_contract
        from agent.harness.checkpoint_migrations import normalize_v12_action_envelope

        action = normalize_v12_action_envelope({
            "type": "answer_opening_question",
            "arguments": {"topic": "current_config"},
        })
        self.assertEqual(action["topic"], "current_config")
        with self.assertRaisesRegex(ValueError, "arguments.v1 is retired"):
            validate_action_contract({
                "type": "answer_opening_question",
                "arguments": {"topic": "current_config"},
            })

    def test_production_prompts_defer_routing_and_fields_to_typed_schemas(
        self,
    ) -> None:
        import agent.harness.context as context
        from agent.harness.hierarchical_planner import (
            _stage_a_prompt,
            _stage_b_prompt,
        )

        self.assertFalse(hasattr(context, "build_action_resolver_prompt"))
        stage_a = _stage_a_prompt()
        stage_b = _stage_b_prompt("performance")
        self.assertIn(
            "structured_candidates describe terminal syntax only",
            stage_a,
        )
        self.assertIn("do not classify intent", stage_a)
        self.assertIn("do not guess, omit, or repair a route", stage_a)
        self.assertIn("Follow owner_action_schema exactly", stage_b)
        self.assertIn(
            "emit source_evidence only when that action declares it",
            stage_b,
        )
        self.assertNotIn("Always include source_evidence", stage_b)
        self.assertNotIn("confidence:'low'|'medium'|'high'", stage_b)

    def test_whole_plan_admission_cannot_reassign_context_ownership(self) -> None:
        from agent.harness.semantic_compiler import whole_plan_admission_prompt

        prompt = whole_plan_admission_prompt("")

        self.assertIn("immutable disposition=context", prompt)
        self.assertIn("never reinterpret it as complete or support", prompt)
        self.assertIn("never invent an owner", prompt)

    def test_pending_semantic_policy_preserves_contrastive_option_selection(
        self,
    ) -> None:
        from agent.harness.semantic_policy import PENDING_CANDIDATE_SEMANTIC_POLICY

        self.assertIn(
            "select only the affirmed option",
            PENDING_CANDIDATE_SEMANTIC_POLICY,
        )
        self.assertIn(
            "never an answer selecting a rejected option",
            PENDING_CANDIDATE_SEMANTIC_POLICY,
        )
        self.assertIn(
            "one pending selection",
            PENDING_CANDIDATE_SEMANTIC_POLICY,
        )


    def test_r36_turn_local_lifetime_contract_is_preserved(self) -> None:
        from agent.harness.action_registry import action_is_turn_local

        for action_type in (
            "greeting",
            "answer_opening_question",
            "analyze_evidence",
            "analyze_report",
        ):
            self.assertTrue(action_is_turn_local({"type": action_type}), action_type)
        self.assertTrue(action_is_turn_local({"type": "inspect_failure"}))
        self.assertFalse(action_is_turn_local({"type": "choose_target_mode"}))

    def test_retired_custom_rpc_action_is_only_an_admission_compatibility_input(self) -> None:
        from agent.harness.action_registry import ACTION_BY_TYPE
        from agent.harness.checkpoint_migrations import compile_v12_custom_rpc_action

        self.assertNotIn("start_custom_rpc", ACTION_BY_TYPE)
        compiled = compile_v12_custom_rpc_action({
            "type": "start_custom_rpc",
            "rpc_endpoint": "https://example.invalid/rpc",
            "rpc_method": "eth_chainId",
            "workload_scope": "single_replace",
            "confidence": "high",
        })
        self.assertEqual(
            [item["type"] for item in compiled],
            ["rpc_catalog_command", "rpc_catalog_command", "rpc_workload_command"],
        )
        self.assertEqual(
            [item.get("catalog_command") for item in compiled[:2]],
            ["set_endpoint", "set_method"],
        )
        self.assertTrue(all(item.get("type") != "start_custom_rpc" for item in compiled))

    def test_custom_rpc_domains_do_not_execute_retired_omnibus_action(self) -> None:
        domain_files = (
            Path(__file__).resolve().parents[1] / "agent/harness/domains/chain_rpc.py",
            Path(__file__).resolve().parents[1] / "agent/harness/domains/chain_rpc_questions.py",
            Path(__file__).resolve().parents[1] / "agent/harness/domains/rpc_endpoint.py",
            Path(__file__).resolve().parents[1] / "agent/harness/domains/rpc_workload.py",
        )
        for path in domain_files:
            self.assertNotIn("start_custom_rpc", path.read_text(encoding="utf-8"), path.name)


class ToolContractAuthorityTest(unittest.TestCase):
    def test_one_tool_registry_generates_schema_and_dispatch_handlers(self) -> None:
        import agent.tools.executor as executor
        from agent.tools.schema import TOOL_SPECS, tool_schema

        rendered = {item["function"]["name"]: item["function"] for item in tool_schema()["tools"]}
        self.assertEqual(set(rendered), {spec.name for spec in TOOL_SPECS})
        for spec in TOOL_SPECS:
            self.assertEqual(rendered[spec.name]["parameters"]["properties"], spec.properties)
            self.assertEqual(rendered[spec.name]["parameters"]["required"], list(spec.required))
            self.assertTrue(callable(executor.TOOL_OPERATION_BY_NAME[spec.name].handler))

    def test_dispatch_uses_registered_handler_and_rejects_contract_drift(self) -> None:
        import agent.tools.executor as executor

        handler = Mock(return_value={"status": "ok"})
        bound = executor.TOOL_OPERATION_BY_NAME["discover_environment"].bind(handler)
        with patch.dict(executor.TOOL_OPERATION_BY_NAME, {"discover_environment": bound}):
            self.assertEqual(executor.execute_tool("discover_environment"), {"status": "ok"})
            handler.assert_called_once_with({})
        with self.assertRaisesRegex(ValueError, "undeclared arguments.*extra"):
            executor.execute_tool("discover_environment", {"extra": True})
        with self.assertRaisesRegex(ValueError, "missing required arguments.*request"):
            executor.execute_tool("generate_plan", {})

    def test_public_tool_api_remains_available(self) -> None:
        from agent.tools.executor import execute_tool, load_arguments
        from agent.tools.schema import tool_schema

        self.assertTrue(callable(execute_tool))
        self.assertEqual(load_arguments('{"approved": true}'), {"approved": True})
        self.assertIn("tools", tool_schema())


if __name__ == "__main__":
    unittest.main()

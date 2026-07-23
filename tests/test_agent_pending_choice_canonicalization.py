"""Focused contracts for canonical semantic pending-choice admission."""

from __future__ import annotations

import json
import unittest
from copy import deepcopy
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch


def _state(**updates: Any) -> dict[str, Any]:
    from agent.harness.state import new_state

    state = new_state("pending-choice-canonical", language="en")
    state.update(deepcopy(updates))
    return state


def _admission_response(request: Any) -> SimpleNamespace:
    review = json.loads(request.messages[1].content)
    units = {str(row["unit_id"]): row for row in review["semantic_units"]}
    return SimpleNamespace(text=json.dumps({
        "plan_hash": review["plan_hash"],
        "action_verdicts": [
            {
                "action_id": action["action_id"],
                "verdict": "admit",
                "unit_ids": list(action["unit_ids"]),
                "evidence": [
                    {
                        "unit_id": unit_id,
                        "quote": str(units[unit_id]["source_text"]),
                        "relation": "direct",
                        "support_relation": "",
                    }
                    for unit_id in action["unit_ids"]
                ],
                "grounded_arguments": [
                    {
                        "argument_name": argument,
                        "evidence_quote": str(units[action["unit_ids"][0]]["source_text"]),
                    }
                    for argument in action.get("required_value_grounding_arguments") or []
                ],
                "pending_answer_argument": (
                    action["pending_value_candidates"][0]["candidate_id"]
                    if len(action.get("pending_value_candidates") or []) == 1
                    else ""
                ),
                "reason": "the immutable action preserves all mapped source units",
            }
            for action in review["actions"]
        ],
        "unit_verdicts": [
            {
                "unit_id": unit["unit_id"],
                "verdict": "unresolved" if unit["disposition"] == "unresolved" else "complete",
                "owner_action_ids": list(unit["owner_action_ids"]),
                "evidence_quote": str(unit["source_text"]),
                "omitted_action_type": "",
                "reason": "the immutable unit retains its declared disposition",
            }
            for unit in review["semantic_units"]
        ],
        "reason": "the immutable plan is admitted",
    }, ensure_ascii=False, sort_keys=True))


def _provider(documents: list[dict[str, Any]]) -> Mock:
    provider = Mock()
    compiler_documents = iter(documents)

    def complete(request: Any) -> SimpleNamespace:
        payload = json.loads(request.messages[1].content)
        if "plan_hash" in payload and "immutable_document" in payload:
            return _admission_response(request)
        return SimpleNamespace(text=json.dumps(next(compiler_documents), ensure_ascii=False, sort_keys=True))

    provider.complete.side_effect = complete
    return provider


def _document(action: dict[str, Any], text: str, *, disposition: str = "action") -> dict[str, Any]:
    return {
        "actions": [action],
        "semantic_units": [{
            "unit_id": "unit-1",
            "clause_id": "clause-1",
            "source_text": text,
            "disposition": disposition,
            "action_indexes": [0] if disposition == "action" else [],
            "reason": "focused pending-choice fixture",
        }],
        "reason": "focused pending-choice fixture",
    }


def _recovery_state() -> dict[str, Any]:
    from agent.harness.domains.recovery import question_for_recovery

    state = _state(
        active_group="failure_recovery",
        failure_recovery={
            "status": "pending",
            "record": {
                "code": "endpoint_unreachable",
                "summary": "endpoint failed",
                "allowed_actions": ["cancel_failure_recovery"],
            },
        },
    )
    state["pending_question"] = question_for_recovery(state, "failure_recovery") or {}
    return state


def _performance_manual_state(*, advanced: bool = False) -> dict[str, Any]:
    from agent.harness.domains.performance import question_for_performance

    if advanced:
        state = _state(
            active_group="advanced_tuning",
            target_mode="fake-node",
            workflow_mode="rpc_benchmark",
            advanced_tuning={
                "default_decision_made": True,
                "adjust_field": "MONITOR_INTERVAL",
            },
        )
        group = "advanced_tuning"
    else:
        state = _state(
            active_group="qps_profile",
            target_mode="fake-node",
            workflow_mode="rpc_benchmark",
            qps_profile={
                "mode": "quick",
                "default_decision_made": True,
                "adjust_field": "INITIAL_QPS",
            },
        )
        group = "qps_profile"
    state["pending_question"] = question_for_performance(state, group) or {}
    return state


class CanonicalPendingChoiceTests(unittest.TestCase):
    def test_declared_manual_owner_is_canonicalized_before_admission(self) -> None:
        from agent.harness.intent import resolve_action_queue

        state = _state(active_group="endpoint_process")
        state["pending_question"] = {
            "id": "endpoint",
            "group": "endpoint_process",
            "kind": "url",
            "manual_input_allowed": True,
            "options": [],
            "manual_action": {
                "type": "rpc_catalog_command",
                "catalog_command": "set_endpoint",
                "value_argument": "rpc_endpoint",
            },
            "validation": {"value_type": "url"},
        }
        text = "Use this endpoint.\nhttp://fake-node:8545"
        document = {
            "actions": [{
                "type": "rpc_catalog_command",
                "catalog_command": "set_endpoint",
                "rpc_endpoint": "http://fake-node:8545",
                "source_evidence": "http://fake-node:8545",
            }],
            "semantic_units": [
                {
                    "unit_id": "unit-1",
                    "clause_id": "clause-1",
                    "source_text": "Use this endpoint.",
                    "disposition": "action",
                    "action_indexes": [0],
                    "reason": "endpoint framing",
                },
                {
                    "unit_id": "unit-2",
                    "clause_id": "clause-2",
                    "source_text": "http://fake-node:8545",
                    "disposition": "action",
                    "action_indexes": [0],
                    "reason": "endpoint value",
                },
            ],
        }
        resolved = _document({
            "type": "answer_pending",
            "answer": "http://fake-node:8545",
            "source_evidence": "http://fake-node:8545",
        }, text)
        provider = _provider([document, resolved])

        with patch("agent.harness.intent.provider_from_config", return_value=provider):
            result = resolve_action_queue(state, text)

        self.assertEqual(provider.complete.call_count, 2)
        self.assertEqual(result["actions"][0]["type"], "answer_pending")
        self.assertEqual(result["actions"][0]["answer"], "http://fake-node:8545")

    def test_abstract_mutation_is_canonicalized_to_matching_intake_option(self) -> None:
        from agent.harness.domains.chain_rpc_questions import _target_change_scope_question
        from agent.harness.intent import resolve_action_queue

        state = _state(
            active_group="workload_rpc",
            target_mode="fake-node",
            workflow_mode="rpc_benchmark",
        )
        state["pending_question"] = _target_change_scope_question(state)
        text = "Switch the target mode and keep current values until replacements are confirmed."
        provider = _provider([_document({
            "type": "choose_target_mode",
            "target_mode": "fake-node",
            "target_mode_explicit": False,
            "source_evidence": "Switch the target mode",
        }, text)])

        with patch("agent.harness.intent.provider_from_config", return_value=provider):
            result = resolve_action_queue(state, text)

        self.assertEqual(provider.complete.call_count, 2)
        self.assertEqual(result["actions"][0]["type"], "answer_pending")
        self.assertEqual(result["actions"][0]["selected_value"], "target_mode")

    def test_semantic_review_derives_incomplete_intake_from_registry(self) -> None:
        from agent.harness.domains.chain_rpc_questions import _target_change_scope_question
        from agent.harness.intent import (
            _freeze_bounded_semantic_plan,
            _semantic_fulfillment_prompt,
        )
        from agent.harness.plan_coverage import segment_user_turn

        state = _state(
            active_group="workload_rpc",
            target_mode="fake-node",
            workflow_mode="rpc_benchmark",
        )
        state["pending_question"] = _target_change_scope_question(state)
        text = "I need to change the fake-node / real-node / sync-observe target mode."
        clauses = tuple(segment_user_turn(text))
        candidate = json.dumps({
            "actions": [{
                "type": "request_target_mode_selection",
                "source_evidence": text,
            }],
            "semantic_units": [{
                "unit_id": "unit-1",
                "clause_id": clauses[0].clause_id,
                "source_text": text,
                "disposition": "action",
                "action_indexes": [0],
                "reason": "the source requests target-mode replacement intake",
            }],
        })

        plan = _freeze_bounded_semantic_plan(candidate, state, clauses)
        record = plan.request_payload()["actions"][0]
        policy = _semantic_fulfillment_prompt()

        self.assertIs(record["registry_incomplete_mutation_intake"], True)
        self.assertIs(record["registry_pending_option_admission"], True)
        self.assertIn("initial selection or replacement", record["declared_purpose"])
        self.assertIn("registry_incomplete_mutation_intake=true", policy)
        self.assertNotIn("A target-mode-intake purpose", policy)

    def test_non_intake_action_is_not_marked_as_incomplete_intake(self) -> None:
        from agent.harness.intent import _freeze_bounded_semantic_plan
        from agent.harness.plan_coverage import segment_user_turn

        state = _state(active_group="opening")
        text = "Explain what this Agent can do."
        clauses = tuple(segment_user_turn(text))
        candidate = json.dumps({
            "actions": [{
                "type": "answer_opening_question",
                "topic": "capabilities",
                "source_evidence": text,
            }],
            "semantic_units": [{
                "unit_id": "unit-1",
                "clause_id": clauses[0].clause_id,
                "source_text": text,
                "disposition": "action",
                "action_indexes": [0],
                "reason": "read-only consultation",
            }],
        })

        plan = _freeze_bounded_semantic_plan(candidate, state, clauses)

        self.assertIs(
            plan.request_payload()["actions"][0]["registry_incomplete_mutation_intake"],
            False,
        )

    def test_reviewer_candidate_cannot_replace_a_domain_action(self) -> None:
        from agent.harness.intent import resolve_action_queue

        state = _performance_manual_state()
        text = "Start this profile at 75 QPS."
        bypass = _document({
            "type": "set_qps_override",
            "qps_overrides": {"INITIAL_QPS": 75},
        }, text)
        provider = _provider([bypass, bypass])

        with patch("agent.harness.intent.provider_from_config", return_value=provider):
            result = resolve_action_queue(state, text)

        self.assertGreaterEqual(provider.complete.call_count, 2)
        self.assertLessEqual(provider.complete.call_count, 4)
        self.assertEqual(result["actions"][0]["type"], "clarify_unresolved")

    def test_cross_group_mutation_is_readjudicated_for_manual_pending(self) -> None:
        from agent.harness.intent import resolve_action_queue
        from agent.harness.domains.chain_rpc_questions import _weights_question

        state = _state(
            active_group="endpoint_process",
            target_mode="fake-node",
            workflow_mode="rpc_benchmark",
            chain_identity={"canonical": "bsc", "status": "confirmed", "case": "case2"},
            custom_rpc={"status": "needs_weights", "method": "eth_blockNumber"},
        )
        state["pending_question"] = _weights_question(
            state,
            "new_chain",
        )
        text = (
            "Give eth_blockNumber forty-five percent.\n"
            "Use the remaining fifty-five percent for eth_gasPrice."
        )
        units = [
            {
                "unit_id": f"unit-{index}",
                "clause_id": f"clause-{index}",
                "source_text": clause,
                "disposition": "action",
                "action_indexes": [0],
                "reason": "one part of the requested weight map",
            }
            for index, clause in enumerate(text.splitlines(), start=1)
        ]
        first = {
            "actions": [{
                "type": "rpc_workload_command",
                "workload_scope": "mixed_replace",
                "rpc_weights": {"eth_blockNumber": 45, "eth_gasPrice": 55},
            }],
            "semantic_units": units,
        }
        resolved = {
            "actions": [{
                "type": "answer_pending",
                "answer": "eth_blockNumber=45,eth_gasPrice=55",
                "source_evidence": text.splitlines()[0],
            }],
            "semantic_units": units,
        }
        provider = _provider([first, resolved])

        with patch("agent.harness.intent.provider_from_config", return_value=provider):
            result = resolve_action_queue(state, text)

        self.assertGreaterEqual(provider.complete.call_count, 2)
        self.assertLessEqual(provider.complete.call_count, 4)
        self.assertEqual([item["type"] for item in result["actions"]], ["answer_pending"])

    def test_typed_url_candidates_are_exposed_without_guessing(self) -> None:
        from agent.harness.intent import _action_queue_payload

        state = _state(active_group="endpoint_process")
        state["pending_question"] = {
            "id": "endpoint",
            "group": "endpoint_process",
            "kind": "url",
            "manual_input_allowed": True,
            "validation": {"value_type": "url"},
        }
        one = _action_queue_payload(
            state,
            "Use this endpoint only to validate the method.\nhttp://fake-node:8545",
        )
        many = _action_queue_payload(
            state,
            "Compare http://fake-node:8545 with https://example.invalid/rpc.",
        )

        self.assertEqual(one["pending_typed_candidates"], ["http://fake-node:8545"])
        self.assertEqual(
            many["pending_typed_candidates"],
            ["http://fake-node:8545", "https://example.invalid/rpc"],
        )

    def test_same_unit_explicit_mutation_invalidates_the_old_pending_answer(self) -> None:
        from agent.harness.coordinator import _validate_action_plan

        text = "Switch the target mode; keep current values until replacements are confirmed."
        state = _state(
            active_group="workload_rpc",
            target_mode="fake-node",
            workflow_mode="rpc_benchmark",
            last_user_input=text,
            pending_question={
                "id": "target_change_scope",
                "group": "workload_rpc",
                "kind": "numbered_choice",
                "options": [{"id": "target_mode", "value": "target_mode"}],
            },
        )
        state["turn_context"] = {
            "pending_choice_contracts": [{
                "admission_action_id": "answer-1",
                "question": {"id": "target_change_scope", "group": "workload_rpc"},
                "option": {"selected_value": "target_mode"},
                "semantic_units": [{"unit_id": "unit-1", "source_text": text}],
            }]
        }
        actions = [
            {
                "type": "choose_target_mode",
                "target_mode": "real-node",
                "target_mode_explicit": True,
                "source_evidence": "Switch the target mode",
                "target_mode_semantic_verified": True,
            },
            {
                "type": "answer_pending",
                "answer": "target_mode",
                "selected_value": "target_mode",
                "source_evidence": text,
                "pending_option_semantic_verified": True,
                "_admission_action_id": "answer-1",
            },
        ]

        validated = _validate_action_plan(state, actions)

        self.assertNotIn("answer_pending", [item["type"] for item in validated])
        self.assertEqual([item["type"] for item in validated], ["choose_target_mode"])

    def test_unrelated_cross_group_mutation_does_not_consume_manual_pending(self) -> None:
        from agent.harness.intent import resolve_action_queue

        state = _state(
            active_group="endpoint_process",
            target_mode="real-node",
            workflow_mode="rpc_benchmark",
            pending_question={
                "id": "custom_rpc_endpoint",
                "group": "endpoint_process",
                "kind": "url",
                "manual_input_allowed": True,
                "validation": {"value_type": "url"},
            },
        )
        text = "Switch the QPS profile to quick."
        document = _document({
            "type": "set_qps_mode",
            "qps_mode": "quick",
            "mutation_explicit": True,
            "source_evidence": text,
        }, text)
        provider = _provider([document])

        with patch("agent.harness.intent.provider_from_config", return_value=provider):
            result = resolve_action_queue(state, text)

        self.assertEqual(provider.complete.call_count, 2)
        self.assertEqual([item["type"] for item in result["actions"]], ["set_qps_mode"])

    def test_distinct_unit_mutation_invalidates_pending_answer(self) -> None:
        from agent.harness.coordinator import _validate_action_plan

        choice = "Switch the target-mode setting."
        mutation = "Use real-node now."
        state = _state(
            active_group="workload_rpc",
            target_mode="fake-node",
            workflow_mode="rpc_benchmark",
            last_user_input=f"{choice} {mutation}",
            pending_question={
                "id": "target_change_scope",
                "group": "workload_rpc",
                "kind": "numbered_choice",
                "options": [{"id": "target_mode", "value": "target_mode"}],
            },
        )
        state["turn_context"] = {
            "pending_choice_contracts": [{
                "admission_action_id": "answer-1",
                "question": {"id": "target_change_scope", "group": "workload_rpc"},
                "option": {"selected_value": "target_mode"},
                "semantic_units": [{"unit_id": "unit-1", "source_text": choice}],
            }]
        }
        actions = [
            {
                "type": "answer_pending",
                "answer": "target_mode",
                "selected_value": "target_mode",
                "source_evidence": choice,
                "pending_option_semantic_verified": True,
                "_admission_action_id": "answer-1",
            },
            {
                "type": "choose_target_mode",
                "target_mode": "real-node",
                "target_mode_explicit": True,
                "source_evidence": mutation,
                "target_mode_semantic_verified": True,
            },
        ]

        validated = _validate_action_plan(state, actions)

        self.assertNotIn("answer_pending", [item["type"] for item in validated])
        self.assertIn("choose_target_mode", [item["type"] for item in validated])

    def test_same_group_domain_mutation_is_readjudicated_as_manual_pending_answer(self) -> None:
        from agent.harness.intent import resolve_action_queue

        state = _performance_manual_state()
        text = "Start this profile at 75 QPS."
        first = _document({
            "type": "set_qps_override",
            "qps_overrides": {"INITIAL_QPS": 75},
        }, text)
        resolved = _document({
            "type": "answer_pending",
            "answer": "75",
            "source_evidence": text,
        }, text)
        provider = _provider([first, resolved])

        with patch("agent.harness.intent.provider_from_config", return_value=provider):
            result = resolve_action_queue(state, text)

        self.assertEqual(provider.complete.call_count, 4)
        self.assertEqual([item["type"] for item in result["actions"]], ["answer_pending"])
        self.assertEqual(result["actions"][0]["answer"], "75")
        self.assertEqual(result.get("pending_choice_contracts"), [])

    def test_duplicate_equivalent_manual_answers_coalesce_before_final_admission(self) -> None:
        from agent.harness.intent import resolve_action_queue
        from tests.agent_live.harness_contract_scenarios import question_scenarios

        scenario = next(
            item for item in question_scenarios("en")
            if item.scenario_id == "new_chain_existing_family_needs_weights"
        )
        state = deepcopy(dict(scenario.seed_state))
        first_text = "Give eth_blockNumber forty-five percent."
        second_text = "Use the remaining fifty-five percent for eth_gasPrice."
        text = f"{first_text}\n{second_text}"
        value = {"eth_blockNumber": 45, "eth_gasPrice": 55}
        first = {
            "actions": [{
                "type": "rpc_workload_command",
                "workload_scope": "mixed_replace",
                "rpc_weights": value,
            }],
            "semantic_units": [
                {
                    "unit_id": "unit-1",
                    "clause_id": "clause-1",
                    "source_text": first_text,
                    "disposition": "action",
                    "action_indexes": [0],
                },
                {
                    "unit_id": "unit-2",
                    "clause_id": "clause-2",
                    "source_text": second_text,
                    "disposition": "action",
                    "action_indexes": [0],
                },
            ],
        }
        repaired = {
            "actions": [
                {"type": "answer_pending", "answer": value, "source_evidence": first_text},
                {"type": "answer_pending", "answer": value, "source_evidence": second_text},
            ],
            "semantic_units": [
                {
                    "unit_id": "unit-1",
                    "clause_id": "clause-1",
                    "source_text": first_text,
                    "disposition": "action",
                    "action_indexes": [0, 1],
                },
                {
                    "unit_id": "unit-2",
                    "clause_id": "clause-2",
                    "source_text": second_text,
                    "disposition": "action",
                    "action_indexes": [0, 1],
                },
            ],
        }
        provider = _provider([first, repaired])

        with patch("agent.harness.intent.provider_from_config", return_value=provider):
            result = resolve_action_queue(state, text)

        self.assertEqual(provider.complete.call_count, 4)
        self.assertEqual(len(result["actions"]), 1)
        self.assertEqual(result["actions"][0]["type"], "answer_pending")
        self.assertEqual(result["actions"][0]["answer"], value)

    def test_typed_manual_answer_reaches_domain_without_string_coercion(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn
        from tests.agent_live.harness_contract_scenarios import question_scenarios

        scenario = next(
            item for item in question_scenarios("en")
            if item.scenario_id == "new_chain_existing_family_needs_weights"
        )
        state = deepcopy(dict(scenario.seed_state))
        text = "Use 45 for eth_blockNumber and 55 for eth_gasPrice."
        state["last_user_input"] = text
        weights = {"eth_blockNumber": 45, "eth_gasPrice": 55}
        with patch(
            "agent.harness.coordinator.resolve_action_queue",
            return_value={"actions": [{
                "type": "answer_pending",
                "answer": weights,
                "source_evidence": text,
                "pending_option_semantic_verified": True,
                "semantic_purpose_verified": True,
            }]},
        ):
            result = invoke_product_graph_turn(state)

        self.assertEqual((result.get("chain_identity") or {}).get("weights"), weights)
        self.assertNotEqual(
            (result.get("pending_question") or {}).get("id"),
            "new_chain_custom_weights",
        )

    def test_typed_rpc_method_reaches_domain_without_prose_reinterpretation(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn
        from tests.agent_live.harness_contract_scenarios import question_scenarios

        scenario = next(
            item for item in question_scenarios("en")
            if item.scenario_id == "custom_needs_method"
        )
        state = deepcopy(dict(scenario.seed_state))
        text = "The custom JSON-RPC method is eth_accounts."
        state["last_user_input"] = text
        with patch(
            "agent.harness.coordinator.resolve_action_queue",
            return_value={"actions": [{
                "type": "answer_pending",
                "answer": "eth_accounts",
                "source_evidence": text,
                "pending_option_semantic_verified": True,
                "semantic_purpose_verified": True,
            }]},
        ):
            result = invoke_product_graph_turn(state)

        catalog = (result.get("custom_rpc") or {}).get("catalog") or {}
        draft = catalog.get("draft") or {}
        self.assertEqual(draft.get("method"), "eth_accounts")
        self.assertNotEqual(
            (result.get("pending_question") or {}).get("id"),
            "custom_rpc_method",
        )

    def test_typed_rpc_endpoint_reaches_domain_without_multi_url_reinterpretation(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn
        from tests.agent_live.harness_contract_scenarios import question_scenarios

        scenario = next(
            item for item in question_scenarios("en")
            if item.scenario_id == "custom_needs_endpoint"
        )
        state = deepcopy(dict(scenario.seed_state))
        endpoint = "http://fake-node:8545"
        text = (
            "Docs: https://docs.example.invalid/rpc. "
            f"Use {endpoint} as the validation endpoint."
        )
        state["last_user_input"] = text
        with patch(
            "agent.harness.coordinator.resolve_action_queue",
            return_value={"actions": [{
                "type": "answer_pending",
                "answer": endpoint,
                "source_evidence": endpoint,
                "pending_option_semantic_verified": True,
                "semantic_purpose_verified": True,
            }]},
        ), patch(
            "agent.harness.domains.rpc_endpoint.validate_rpc_endpoint",
            return_value={
                "ready": True,
                "evidence_file": "endpoint.json",
                "safe_method": "eth_blockNumber",
                "checks": [],
            },
        ) as probe:
            result = invoke_product_graph_turn(state)

        self.assertEqual((result.get("custom_rpc") or {}).get("endpoint"), endpoint)
        self.assertEqual(probe.call_args.kwargs["endpoint"], endpoint)

    def test_semantic_weight_mapping_runs_resolver_to_domain_as_one_typed_value(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn
        from tests.agent_live.harness_contract_scenarios import question_scenarios

        scenario = next(
            item for item in question_scenarios("en")
            if item.scenario_id == "new_chain_existing_family_needs_weights"
        )
        state = deepcopy(dict(scenario.seed_state))
        text = "Use 45 for eth_blockNumber and 55 for eth_gasPrice."
        state["last_user_input"] = text
        weights = {"eth_blockNumber": 45, "eth_gasPrice": 55}
        candidate = _document({
            "type": "rpc_workload_command",
            "workload_scope": "mixed_replace",
            "rpc_weights": weights,
        }, text)
        resolved = _document({
            "type": "answer_pending",
            "answer": weights,
            "source_evidence": text,
        }, text)
        provider = _provider([candidate, resolved])

        with patch("agent.harness.intent.provider_from_config", return_value=provider):
            result = invoke_product_graph_turn(state, allow_semantic_resolver=True)

        self.assertEqual(provider.complete.call_count, 4)
        self.assertEqual((result.get("chain_identity") or {}).get("weights"), weights)
        self.assertNotEqual(
            (result.get("pending_question") or {}).get("id"),
            "new_chain_custom_weights",
        )

    def test_deterministic_weight_mapping_runs_complete_product_graph(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn
        from tests.agent_live.harness_contract_scenarios import question_scenarios

        scenario = next(
            item for item in question_scenarios("en")
            if item.scenario_id == "new_chain_existing_family_needs_weights"
        )
        state = deepcopy(dict(scenario.seed_state))
        state["last_user_input"] = "eth_blockNumber=45,eth_gasPrice=55"

        result = invoke_product_graph_turn(state)

        self.assertEqual(
            (result.get("chain_identity") or {}).get("weights"),
            {"eth_blockNumber": 45, "eth_gasPrice": 55},
        )

    def test_legacy_manual_owner_metadata_is_not_an_action_argument(self) -> None:
        from agent.harness.intent import _matching_pending_manual_value
        from tests.agent_live.harness_contract_scenarios import question_scenarios

        scenario = next(
            item for item in question_scenarios("en")
            if item.scenario_id == "custom_needs_schema_evidence"
        )
        pending = deepcopy(dict(scenario.seed_state["pending_question"]))
        pending["manual_action"]["use_complete_turn"] = True
        action = {
            "type": "rpc_catalog_command",
            "catalog_command": "append_evidence",
            "rpc_schema_evidence": "[]",
        }

        self.assertEqual(_matching_pending_manual_value(action, pending), "[]")

    def test_exact_manual_answer_preserves_declared_typed_value(self) -> None:
        from agent.harness.questions import exact_answer
        from tests.agent_live.harness_contract_scenarios import question_scenarios

        scenario = next(
            item for item in question_scenarios("en")
            if item.scenario_id == "new_chain_existing_family_needs_weights"
        )
        matched, value = exact_answer(
            "eth_blockNumber=45,eth_gasPrice=55",
            dict(scenario.seed_state["pending_question"]),
        )

        self.assertTrue(matched)
        self.assertEqual(value, {"eth_blockNumber": 45, "eth_gasPrice": 55})

    def test_natural_language_weights_require_semantic_ownership(self) -> None:
        from agent.harness.questions import (
            answer_fits_pending,
            value_satisfies_pending_contract,
        )
        from tests.agent_live.harness_contract_scenarios import question_scenarios

        scenario = next(
            item for item in question_scenarios("en")
            if item.scenario_id == "new_chain_existing_family_needs_weights"
        )

        self.assertFalse(answer_fits_pending(
            "Use 45 for eth_blockNumber and 55 for eth_gasPrice.",
            dict(scenario.seed_state["pending_question"]),
        ))
        non_manual = deepcopy(dict(scenario.seed_state["pending_question"]))
        non_manual["manual_input_allowed"] = False
        self.assertFalse(value_satisfies_pending_contract(
            {"eth_blockNumber": 45, "eth_gasPrice": 55},
            non_manual,
        ))

    def test_unresolved_neighboring_manual_number_uses_the_same_focused_contract(self) -> None:
        from agent.harness.intent import resolve_action_queue

        state = _performance_manual_state(advanced=True)
        text = "Use a twelve second interval here."
        first = _document(
            {"type": "clarify_unresolved", "clauses": [text]},
            text,
        )
        resolved = _document({
            "type": "answer_pending",
            "answer": "12",
            "source_evidence": text,
        }, text)
        provider = _provider([first, resolved])

        with patch("agent.harness.intent.provider_from_config", return_value=provider):
            result = resolve_action_queue(state, text)

        self.assertEqual(provider.complete.call_count, 4)
        self.assertEqual(result["actions"][0]["type"], "answer_pending")
        self.assertEqual(result["actions"][0]["answer"], "12")

    def test_manual_pending_adjudication_preserves_independent_consultation(self) -> None:
        from agent.harness.intent import resolve_action_queue

        state = _performance_manual_state()
        text = "Start at 75 QPS.\nAlso explain what the quick profile means."
        first = {
            "actions": [
                {"type": "set_qps_override", "qps_overrides": {"INITIAL_QPS": 75}},
                {
                    "type": "answer_opening_question",
                    "topic": "config_explanation",
                    "subject": "quick QPS profile",
                    "source_evidence": "Also explain what the quick profile means.",
                },
            ],
            "semantic_units": [
                {
                    "unit_id": "unit-1",
                    "clause_id": "clause-1",
                    "source_text": "Start at 75 QPS.",
                    "disposition": "action",
                    "action_indexes": [0],
                    "reason": "QPS value",
                },
                {
                    "unit_id": "unit-2",
                    "clause_id": "clause-2",
                    "source_text": "Also explain what the quick profile means.",
                    "disposition": "action",
                    "action_indexes": [1],
                    "reason": "independent consultation",
                },
            ],
        }
        resolved = deepcopy(first)
        resolved["actions"][0] = {
            "type": "answer_pending",
            "answer": "75",
            "source_evidence": "Start at 75 QPS.",
        }
        provider = _provider([first, resolved])

        with patch("agent.harness.intent.provider_from_config", return_value=provider):
            result = resolve_action_queue(state, text)

        self.assertEqual(
            [item["type"] for item in result["actions"]],
            ["answer_pending", "answer_opening_question"],
        )

    def test_unrelated_consultation_does_not_readjudicate_manual_pending(self) -> None:
        from agent.harness.intent import resolve_action_queue

        state = _performance_manual_state()
        text = "What can this Agent do?"
        provider = _provider([_document({
            "type": "answer_opening_question",
            "topic": "capabilities",
            "source_evidence": text,
        }, text)])

        with patch("agent.harness.intent.provider_from_config", return_value=provider):
            result = resolve_action_queue(state, text)

        self.assertEqual(provider.complete.call_count, 2)
        self.assertEqual(result["actions"][0]["type"], "answer_opening_question")

    def test_admitted_clarify_plan_enters_one_focused_adjudication(self) -> None:
        from agent.harness.intent import resolve_action_queue

        state = _recovery_state()
        text = "I want the option that preserves the evidence without correcting anything now."
        first = _document(
            {"type": "clarify_unresolved", "clauses": [text]},
            text,
        )
        resolved = _document(
            {"type": "cancel_failure_recovery", "confidence": "high"},
            text,
        )
        provider = _provider([first, resolved])

        with patch("agent.harness.intent.provider_from_config", return_value=provider):
            result = resolve_action_queue(state, text)

        self.assertEqual(provider.complete.call_count, 4)
        compiler_prompts = [
            request.messages[0].content
            for request in (call.args[0] for call in provider.complete.call_args_list)
            if "plan_hash" not in json.loads(request.messages[1].content)
        ]
        self.assertIn("only focused adjudication", compiler_prompts[1])
        self.assertEqual(result["actions"][0]["type"], "answer_pending")
        self.assertEqual(result["actions"][0]["selected_value"], "cancel")
        self.assertEqual(result["pending_choice_contracts"][0]["question"]["id"], "failure_recovery_action")

    def test_invalid_answer_pending_cannot_bypass_focused_typed_adjudication(self) -> None:
        from agent.harness.intent import resolve_action_queue

        text = "The custom JSON-RPC method is eth_accounts."
        state = _state(active_group="endpoint_process")
        state["pending_question"] = {
            "id": "custom_rpc_method",
            "group": "endpoint_process",
            "kind": "manual_value",
            "field": "custom_rpc_method",
            "manual_input_allowed": True,
            "options": [],
            "validation": {"input_mode": "rpc_method_or_schema_evidence"},
        }
        invalid = _document({
            "type": "answer_pending",
            "answer": text,
            "source_evidence": text,
        }, text)
        resolved = _document({
            "type": "answer_pending",
            "answer": "eth_accounts",
            "source_evidence": "eth_accounts",
        }, text)
        provider = _provider([invalid, resolved])

        with patch("agent.harness.intent.provider_from_config", return_value=provider):
            result = resolve_action_queue(state, text)

        self.assertEqual(provider.complete.call_count, 4)
        self.assertEqual(result["actions"][0]["type"], "answer_pending")
        self.assertEqual(result["actions"][0]["answer"], "eth_accounts")

    def test_admitted_unique_typed_candidate_gets_canonical_reviewer_receipt(self) -> None:
        from agent.harness.intent import resolve_action_queue

        text = "The custom JSON-RPC method is eth_accounts."
        state = _state(active_group="endpoint_process")
        state["pending_question"] = {
            "id": "custom_rpc_method",
            "group": "endpoint_process",
            "kind": "manual_value",
            "field": "custom_rpc_method",
            "manual_input_allowed": True,
            "options": [],
            "validation": {"input_mode": "rpc_method_or_schema_evidence"},
        }
        document = _document({
            "type": "answer_pending",
            "answer": "eth_accounts",
            "source_evidence": "eth_accounts",
        }, text)
        provider = Mock()
        compiler_documents = iter([document, document])

        def complete(request: Any) -> SimpleNamespace:
            payload = json.loads(request.messages[1].content)
            if "plan_hash" not in payload:
                return SimpleNamespace(text=json.dumps(next(compiler_documents)))
            response = json.loads(_admission_response(request).text)
            response["action_verdicts"][0]["pending_answer_argument"] = ""
            return SimpleNamespace(text=json.dumps(response))

        provider.complete.side_effect = complete
        with patch("agent.harness.intent.provider_from_config", return_value=provider):
            result = resolve_action_queue(state, text)

        self.assertEqual(provider.complete.call_count, 2)
        self.assertEqual(result["actions"][0]["type"], "answer_pending")
        self.assertEqual(result["actions"][0]["answer"], "eth_accounts")

    def test_admitted_semantic_grounding_uses_unique_exact_direct_receipt(self) -> None:
        from agent.harness.intent import resolve_action_queue

        text = "Disable observability for this run."
        state = _state(active_group="observability", target_mode="fake-node")
        state["pending_question"] = {
            "id": "observability_mode",
            "group": "observability",
            "kind": "numbered_choice",
            "field": "observability_mode",
            "manual_input_allowed": False,
            "options": [
                {
                    "id": "1",
                    "label": "Disabled",
                    "value": "disabled",
                    "action": {"type": "answer_pending"},
                }
            ],
            "accepted_action_types": ["answer_pending", "set_observability"],
            "validation": {},
        }
        document = _document({
            "type": "set_observability",
            "observability_mode": "disabled",
            "mutation_explicit": True,
            "source_evidence": text,
        }, text)
        provider = Mock()

        def complete(request: Any) -> SimpleNamespace:
            payload = json.loads(request.messages[1].content)
            if "plan_hash" not in payload:
                return SimpleNamespace(text=json.dumps(document))
            response = json.loads(_admission_response(request).text)
            response["action_verdicts"][0]["grounded_arguments"][0][
                "evidence_quote"
            ] = "disabled"
            return SimpleNamespace(text=json.dumps(response))

        provider.complete.side_effect = complete
        with patch("agent.harness.intent.provider_from_config", return_value=provider):
            result = resolve_action_queue(state, text)

        self.assertEqual(provider.complete.call_count, 2)
        self.assertEqual(result["actions"][0]["type"], "set_observability")
        self.assertEqual(result["actions"][0]["observability_mode"], "disabled")

    def test_exact_source_grounding_is_not_repaired_from_semantic_evidence(self) -> None:
        from agent.harness.intent import resolve_action_queue

        text = "Validate the eth_accounts RPC method."
        state = _state(active_group="endpoint_process")
        document = _document({
            "type": "rpc_catalog_command",
            "catalog_command": "set_method",
            "rpc_method": "eth_accounts",
            "source_evidence": text,
        }, text)
        provider = Mock()

        def complete(request: Any) -> SimpleNamespace:
            payload = json.loads(request.messages[1].content)
            if "plan_hash" not in payload:
                return SimpleNamespace(text=json.dumps(document))
            response = json.loads(_admission_response(request).text)
            response["action_verdicts"][0]["grounded_arguments"][0][
                "evidence_quote"
            ] = "invented method evidence"
            return SimpleNamespace(text=json.dumps(response))

        provider.complete.side_effect = complete
        with patch("agent.harness.intent.provider_from_config", return_value=provider):
            result = resolve_action_queue(state, text)

        self.assertEqual(provider.complete.call_count, 4)
        self.assertEqual(result["actions"][0]["type"], "clarify_unresolved")

    def test_structured_pending_config_uses_review_proposal_not_direct_answer(self) -> None:
        from agent.harness.intent import resolve_action_queue

        text = "NETWORK_MAX_BANDWIDTH_GBPS: 25"
        state = _state(active_group="network")
        state["pending_question"] = {
            "id": "NETWORK_MAX_BANDWIDTH_GBPS",
            "group": "network",
            "kind": "manual_value",
            "field": "NETWORK_MAX_BANDWIDTH_GBPS",
            "manual_input_allowed": True,
            "options": [],
            "validation": {"value_type": "positive_number"},
        }
        direct = _document({
            "type": "answer_pending",
            "answer": "25",
            "source_evidence": text,
        }, text)
        proposal = _document({
            "type": "propose_config_values",
            "config_values": {"NETWORK_MAX_BANDWIDTH_GBPS": "25"},
            "unmapped_values": {},
            "source_format": "yaml",
            "source_evidence": text,
        }, text)
        provider = _provider([direct, proposal])

        with patch("agent.harness.intent.provider_from_config", return_value=provider):
            result = resolve_action_queue(state, text)

        self.assertEqual(provider.complete.call_count, 2)
        self.assertEqual(result["actions"][0]["type"], "propose_config_values")
        self.assertEqual(
            result["actions"][0]["config_values"],
            {"NETWORK_MAX_BANDWIDTH_GBPS": "25"},
        )

    def test_structured_pending_field_survives_compiler_clarification(self) -> None:
        from agent.harness.intent import resolve_action_queue

        text = "ACCOUNTS_VOL_TYPE: hyperdisk-balanced"
        state = _state(active_group="accounts_disk")
        state["pending_question"] = {
            "id": "ACCOUNTS_VOL_TYPE",
            "group": "accounts_disk",
            "kind": "manual_value",
            "field": "ACCOUNTS_VOL_TYPE",
            "manual_input_allowed": True,
            "options": [],
            "manual_action": {"type": "answer_pending", "value_argument": "answer"},
            "validation": {"value_type": "scalar_token"},
        }
        clarification = _document({
            "type": "clarify_unresolved",
            "clauses": [text],
        }, text)
        provider = _provider([clarification])

        with patch("agent.harness.intent.provider_from_config", return_value=provider):
            result = resolve_action_queue(state, text)

        self.assertEqual(
            result["actions"][0]["config_values"],
            {"ACCOUNTS_VOL_TYPE": "hyperdisk-balanced"},
        )
        self.assertEqual(result["actions"][0]["type"], "propose_config_values")
        self.assertEqual(provider.complete.call_count, 2)

    def test_structured_pending_candidate_does_not_steal_consultation_owner(self) -> None:
        from agent.harness.intent import resolve_action_queue

        text = "Example only, do not apply: ACCOUNTS_VOL_TYPE: hyperdisk-balanced"
        state = _state(active_group="accounts_disk")
        state["pending_question"] = {
            "id": "ACCOUNTS_VOL_TYPE",
            "group": "accounts_disk",
            "kind": "manual_value",
            "field": "ACCOUNTS_VOL_TYPE",
            "manual_input_allowed": True,
            "options": [],
            "manual_action": {"type": "answer_pending", "value_argument": "answer"},
            "validation": {"value_type": "scalar_token"},
        }
        consultation = _document({
            "type": "answer_opening_question",
            "topic": "config_explanation",
            "subject": "ACCOUNTS_VOL_TYPE",
            "source_evidence": text,
        }, text)
        provider = _provider([consultation])

        with patch("agent.harness.intent.provider_from_config", return_value=provider):
            result = resolve_action_queue(state, text)

        self.assertEqual(provider.complete.call_count, 2)
        self.assertEqual(
            [action["type"] for action in result["actions"]],
            ["answer_opening_question"],
        )

    def test_pending_url_candidate_does_not_steal_consultation_owner(self) -> None:
        from agent.harness.intent import resolve_action_queue

        state = _state(active_group="endpoint_process")
        state["pending_question"] = {
            "id": "endpoint",
            "group": "endpoint_process",
            "kind": "url",
            "manual_input_allowed": True,
            "options": [],
            "manual_action": {
                "type": "rpc_catalog_command",
                "catalog_command": "set_endpoint",
                "value_argument": "rpc_endpoint",
            },
            "validation": {"value_type": "url"},
        }
        text = "Can this Agent analyze http://node-a:8545 without selecting it as the endpoint?"
        consultation = _document({
            "type": "answer_opening_question",
            "topic": "capabilities",
            "subject": "endpoint analysis without selection",
            "source_evidence": text,
        }, text)
        provider = _provider([consultation])

        with patch("agent.harness.intent.provider_from_config", return_value=provider):
            result = resolve_action_queue(state, text)

        self.assertEqual(provider.complete.call_count, 2)
        self.assertEqual(
            [action["type"] for action in result["actions"]],
            ["answer_opening_question"],
        )

    def test_structured_pending_review_preserves_the_complete_atomic_block(self) -> None:
        from agent.harness.intent import resolve_action_queue

        text = (
            "ACCOUNTS_VOL_TYPE: hyperdisk-balanced\n"
            "ACCOUNTS_VOL_SIZE: 2048\n"
            "TEAM_NOTE: verify-with-storage-owner"
        )
        state = _state(active_group="accounts_disk")
        state["pending_question"] = {
            "id": "ACCOUNTS_VOL_TYPE",
            "group": "accounts_disk",
            "kind": "manual_value",
            "field": "ACCOUNTS_VOL_TYPE",
            "manual_input_allowed": True,
            "options": [],
            "manual_action": {"type": "answer_pending", "value_argument": "answer"},
            "validation": {"value_type": "scalar_token"},
        }
        clarification = _document({
            "type": "clarify_unresolved",
            "clauses": [text],
        }, text)
        provider = _provider([clarification])

        with patch("agent.harness.intent.provider_from_config", return_value=provider):
            result = resolve_action_queue(state, text)

        proposal = result["actions"][0]
        self.assertEqual(proposal["config_values"]["ACCOUNTS_VOL_TYPE"], "hyperdisk-balanced")
        self.assertEqual(proposal["config_values"]["ACCOUNTS_VOL_SIZE"], "2048")
        self.assertEqual(proposal["unmapped_values"]["TEAM_NOTE"], "verify-with-storage-owner")
        self.assertEqual(proposal["type"], "propose_config_values")

    def test_structured_pending_fast_path_does_not_steal_workflow_values(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn

        text = "ACCOUNTS_VOL_TYPE: hyperdisk-balanced\nRPC_MODE: mixed"
        state = _state(active_group="accounts_disk")
        state["pending_question"] = {
            "id": "ACCOUNTS_VOL_TYPE",
            "group": "accounts_disk",
            "kind": "manual_value",
            "field": "ACCOUNTS_VOL_TYPE",
            "manual_input_allowed": True,
            "options": [],
            "manual_action": {"type": "answer_pending", "value_argument": "answer"},
            "validation": {"value_type": "scalar_token"},
        }
        state["last_user_input"] = text
        resolver = Mock(return_value={
            "actions": [{"type": "clarify_unresolved", "clauses": [text]}],
        })

        with patch("agent.harness.coordinator.resolve_action_queue", resolver):
            result = invoke_product_graph_turn(state)

        resolver.assert_called_once()
        self.assertNotIn("pending_review", result.get("inferred_config") or {})
        self.assertEqual(result["pending_question"]["id"], "ACCOUNTS_VOL_TYPE")

    def test_multiline_pending_url_survives_compiler_clarification(self) -> None:
        from agent.harness.intent import _materialize_pending_contract_candidates
        from agent.harness.plan_coverage import segment_user_turn

        state = _state(active_group="endpoint_process")
        state["chain_identity"] = {
            "status": "confirmed",
            "canonical": "solana",
            "adapter_family": "jsonrpc",
        }
        state["pending_question"] = {
            "id": "custom_rpc_endpoint",
            "group": "endpoint_process",
            "kind": "url",
            "field": "custom_rpc_endpoint",
            "manual_input_allowed": True,
            "options": [],
            "manual_action": {
                "type": "rpc_catalog_command",
                "catalog_command": "set_endpoint",
                "value_argument": "rpc_endpoint",
            },
            "validation": {},
        }
        text = (
            "Validate the custom method against this endpoint:\n"
            "http://fake-node:19000\n"
            "Do not use a documentation URL as the endpoint."
        )
        clauses = [
            "Validate the custom method against this endpoint:",
            "http://fake-node:19000",
            "Do not use a documentation URL as the endpoint.",
        ]
        compiler_clarification = {
            "actions": [{"type": "clarify_unresolved", "clauses": clauses}],
            "semantic_units": [
                {
                    "unit_id": f"unit-{index}",
                    "clause_id": f"clause-{index}",
                    "source_text": clause,
                    "disposition": "action",
                    "action_indexes": [0],
                    "reason": "compiler requested clarification",
                }
                for index, clause in enumerate(clauses, start=1)
            ],
        }
        materialized = json.loads(_materialize_pending_contract_candidates(
            json.dumps(compiler_clarification),
            state,
            segment_user_turn(text),
        ))

        self.assertEqual(
            [action["type"] for action in materialized["actions"]],
            ["clarify_unresolved", "rpc_catalog_command"],
        )
        self.assertEqual(
            materialized["actions"][1]["rpc_endpoint"],
            "http://fake-node:19000",
        )
        self.assertEqual(
            materialized["semantic_units"][1]["action_indexes"],
            [1],
        )
        self.assertEqual(
            materialized["semantic_units"][0]["action_indexes"],
            [0],
        )
        self.assertEqual(
            materialized["semantic_units"][2]["action_indexes"],
            [0],
        )

    def test_existing_pending_answer_absorbs_exact_candidate_without_duplication(self) -> None:
        from agent.harness.intent import resolve_action_queue

        state = _state(active_group="endpoint_process")
        state["pending_question"] = {
            "id": "custom_rpc_endpoint",
            "group": "endpoint_process",
            "kind": "url",
            "field": "custom_rpc_endpoint",
            "manual_input_allowed": True,
            "options": [],
            "manual_action": {
                "type": "rpc_catalog_command",
                "catalog_command": "set_endpoint",
                "value_argument": "rpc_endpoint",
            },
            "validation": {"value_type": "url"},
        }
        text = (
            "Validate against this endpoint:\n"
            "http://fake-node:19000\n"
            "Do not select a documentation URL."
        )
        document = {
            "actions": [
                {
                    "type": "answer_pending",
                    "answer": "http://fake-node:19000",
                    "source_evidence": "http://fake-node:19000",
                },
                {
                    "type": "clarify_unresolved",
                    "clauses": ["http://fake-node:19000"],
                },
            ],
            "semantic_units": [
                {
                    "unit_id": "unit-1",
                    "clause_id": "clause-1",
                    "source_text": "Validate against this endpoint:",
                    "disposition": "action",
                    "action_indexes": [0],
                    "reason": "operation framing",
                },
                {
                    "unit_id": "unit-2",
                    "clause_id": "clause-2",
                    "source_text": "http://fake-node:19000",
                    "disposition": "action",
                    "action_indexes": [1],
                    "reason": "compiler left the exact candidate unresolved",
                },
                {
                    "unit_id": "unit-3",
                    "clause_id": "clause-3",
                    "source_text": "Do not select a documentation URL.",
                    "disposition": "action",
                    "action_indexes": [0],
                    "reason": "alternative-category exclusion",
                },
            ],
        }
        provider = _provider([document])

        with patch("agent.harness.intent.provider_from_config", return_value=provider):
            result = resolve_action_queue(state, text)

        self.assertEqual(provider.complete.call_count, 2)
        self.assertEqual(
            [action["type"] for action in result["actions"]],
            ["answer_pending"],
        )
        self.assertEqual(result["actions"][0]["answer"], "http://fake-node:19000")

    def test_manual_answer_and_selected_value_forms_coalesce_to_one_owner(self) -> None:
        from agent.harness.intent import resolve_action_queue

        state = _state(active_group="endpoint_process")
        state["pending_question"] = {
            "id": "custom_rpc_endpoint",
            "group": "endpoint_process",
            "kind": "url",
            "field": "custom_rpc_endpoint",
            "manual_input_allowed": True,
            "options": [],
            "manual_action": {
                "type": "rpc_catalog_command",
                "catalog_command": "set_endpoint",
                "value_argument": "rpc_endpoint",
            },
            "validation": {"value_type": "url"},
        }
        selected = "http://fake-node:19000"
        text = f"Use {selected} as the validation endpoint."
        document = {
            "actions": [
                {
                    "type": "answer_pending",
                    "answer": selected,
                    "source_evidence": text,
                },
                {
                    "type": "answer_pending",
                    "selected_value": selected,
                    "source_evidence": text,
                },
            ],
            "semantic_units": [{
                "unit_id": "unit-1",
                "clause_id": "clause-1",
                "source_text": text,
                "disposition": "action",
                "action_indexes": [0, 1],
                "reason": "two model representations claim one typed value",
            }],
        }
        provider = _provider([document])

        with patch("agent.harness.intent.provider_from_config", return_value=provider):
            result = resolve_action_queue(state, text)

        self.assertEqual(provider.complete.call_count, 2)
        self.assertEqual(len(result["actions"]), 1)
        self.assertEqual(result["actions"][0]["type"], "answer_pending")
        self.assertEqual(result["actions"][0]["answer"], selected)
        self.assertNotIn("selected_value", result["actions"][0])

    def test_conflicting_manual_answer_representations_do_not_select_by_field_order(self) -> None:
        from agent.harness.intent import resolve_action_queue

        state = _state(active_group="endpoint_process")
        state["pending_question"] = {
            "id": "custom_rpc_endpoint",
            "group": "endpoint_process",
            "kind": "url",
            "field": "custom_rpc_endpoint",
            "manual_input_allowed": True,
            "options": [],
            "manual_action": {
                "type": "rpc_catalog_command",
                "catalog_command": "set_endpoint",
                "value_argument": "rpc_endpoint",
            },
            "validation": {"value_type": "url"},
        }
        first = "http://fake-node:19000"
        second = "https://example.invalid/rpc"
        text = f"Use either {first} or {second}."
        unresolved = {
            "actions": [{
                "type": "answer_pending",
                "answer": first,
                "selected_value": second,
                "source_evidence": text,
            }],
            "semantic_units": [{
                "unit_id": "unit-1",
                "clause_id": "clause-1",
                "source_text": text,
                "disposition": "action",
                "action_indexes": [0],
                "reason": "two conflicting manual values",
            }],
        }
        clarification = _document(
            {"type": "clarify_unresolved", "clauses": [text]},
            text,
        )
        provider = _provider([unresolved, clarification])

        with patch("agent.harness.intent.provider_from_config", return_value=provider):
            result = resolve_action_queue(state, text)

        self.assertGreaterEqual(provider.complete.call_count, 2)
        self.assertLessEqual(provider.complete.call_count, 4)
        self.assertEqual(
            [action["type"] for action in result["actions"]],
            ["clarify_unresolved"],
        )

    def test_invalid_nonempty_second_manual_representation_is_still_a_conflict(self) -> None:
        from agent.harness.intent import resolve_action_queue

        state = _state(active_group="endpoint_process")
        state["pending_question"] = {
            "id": "custom_rpc_endpoint",
            "group": "endpoint_process",
            "kind": "url",
            "field": "custom_rpc_endpoint",
            "manual_input_allowed": True,
            "options": [],
            "manual_action": {
                "type": "rpc_catalog_command",
                "catalog_command": "set_endpoint",
                "value_argument": "rpc_endpoint",
            },
            "validation": {"value_type": "url"},
        }
        selected = "http://fake-node:19000"
        text = f"Use {selected}; the other model field is wrong."
        conflicting = {
            "actions": [{
                "type": "answer_pending",
                "answer": selected,
                "selected_value": "not-a-url",
                "source_evidence": text,
            }],
            "semantic_units": [{
                "unit_id": "unit-1",
                "clause_id": "clause-1",
                "source_text": text,
                "disposition": "action",
                "action_indexes": [0],
                "reason": "the model emitted conflicting representations",
            }],
        }
        clarification = _document(
            {"type": "clarify_unresolved", "clauses": [text]},
            text,
        )
        provider = _provider([conflicting, clarification])

        with patch("agent.harness.intent.provider_from_config", return_value=provider):
            result = resolve_action_queue(state, text)

        self.assertGreaterEqual(provider.complete.call_count, 2)
        self.assertLessEqual(provider.complete.call_count, 4)
        self.assertEqual(
            [action["type"] for action in result["actions"]],
            ["clarify_unresolved"],
        )

    def test_yes_no_conflict_is_rejected_before_option_canonicalization(self) -> None:
        from agent.harness.intent import resolve_action_queue

        state = _state(active_group="provider_deployment")
        state["pending_question"] = {
            "id": "confirm_value",
            "group": "provider_deployment",
            "kind": "yes_no",
            "field": "confirm_value",
            "manual_input_allowed": False,
            "options": [
                {"id": "yes", "label": "Y", "value": True},
                {"id": "no", "label": "N", "value": False},
            ],
        }
        text = "Y"
        conflicting = _document({
            "type": "answer_pending",
            "answer": "different-nonempty",
            "selected_value": True,
            "source_evidence": text,
        }, text)
        clarification = _document(
            {"type": "clarify_unresolved", "clauses": [text]},
            text,
        )
        provider = _provider([conflicting, clarification])

        with patch("agent.harness.intent.provider_from_config", return_value=provider):
            result = resolve_action_queue(state, text)

        self.assertEqual(
            [action["type"] for action in result["actions"]],
            ["clarify_unresolved"],
        )

    def test_numbered_choice_conflict_is_rejected_before_option_canonicalization(self) -> None:
        from agent.harness.intent import resolve_action_queue

        state = _state(active_group="qps_profile")
        state["pending_question"] = {
            "id": "qps_mode",
            "group": "qps_profile",
            "kind": "numbered_choice",
            "field": "qps_mode",
            "manual_input_allowed": False,
            "options": [
                {"id": "quick", "label": "quick", "value": "quick"},
                {"id": "standard", "label": "standard", "value": "standard"},
            ],
        }
        text = "1"
        conflicting = _document({
            "type": "answer_pending",
            "answer": "different-nonempty",
            "selected_value": "quick",
            "source_evidence": text,
        }, text)
        clarification = _document(
            {"type": "clarify_unresolved", "clauses": [text]},
            text,
        )
        provider = _provider([conflicting, clarification])

        with patch("agent.harness.intent.provider_from_config", return_value=provider):
            result = resolve_action_queue(state, text)

        self.assertEqual(
            [action["type"] for action in result["actions"]],
            ["clarify_unresolved"],
        )

    def test_nested_envelope_conflicts_are_rejected_in_both_directions(self) -> None:
        from agent.harness.intent import resolve_action_queue

        state = _state(active_group="provider_deployment")
        state["pending_question"] = {
            "id": "confirm_value",
            "group": "provider_deployment",
            "kind": "yes_no",
            "field": "confirm_value",
            "manual_input_allowed": False,
            "options": [
                {"id": "yes", "label": "Y", "value": True},
                {"id": "no", "label": "N", "value": False},
            ],
        }
        text = "Y"
        variants = (
            {
                "type": "answer_pending",
                "selected_value": True,
                "arguments": {"answer": "different-nonempty"},
                "source_evidence": text,
            },
            {
                "type": "answer_pending",
                "answer": "different-nonempty",
                "arguments": {"selected_value": True},
                "source_evidence": text,
            },
        )
        for action in variants:
            with self.subTest(action=action):
                conflicting = _document(action, text)
                clarification = _document(
                    {"type": "clarify_unresolved", "clauses": [text]},
                    text,
                )
                provider = _provider([conflicting, clarification])
                with patch("agent.harness.intent.provider_from_config", return_value=provider):
                    result = resolve_action_queue(state, text)
                self.assertEqual(
                    [item["type"] for item in result["actions"]],
                    ["clarify_unresolved"],
                )

    def test_nested_envelope_equal_representations_share_one_option_owner(self) -> None:
        from agent.harness.intent import resolve_action_queue

        state = _state(active_group="provider_deployment")
        state["pending_question"] = {
            "id": "confirm_value",
            "group": "provider_deployment",
            "kind": "yes_no",
            "field": "confirm_value",
            "manual_input_allowed": False,
            "options": [
                {"id": "yes", "label": "Y", "value": True},
                {"id": "no", "label": "N", "value": False},
            ],
        }
        text = "Y"
        document = _document({
            "type": "answer_pending",
            "selected_value": True,
            "arguments": {"answer": True},
            "source_evidence": text,
        }, text)
        provider = _provider([document])

        with patch("agent.harness.intent.provider_from_config", return_value=provider):
            result = resolve_action_queue(state, text)

        self.assertEqual(len(result["actions"]), 1)
        self.assertEqual(result["actions"][0]["type"], "answer_pending")
        self.assertIs(result["actions"][0]["answer"], True)
        self.assertIs(result["actions"][0]["selected_value"], True)

    def test_unresolved_sibling_prevents_partial_pending_commit_in_product_graph(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn

        state = _state(active_group="endpoint_process")
        state["pending_question"] = {
            "id": "custom_rpc_endpoint",
            "group": "endpoint_process",
            "kind": "url",
            "field": "custom_rpc_endpoint",
            "manual_input_allowed": True,
            "options": [],
            "manual_action": {
                "type": "rpc_catalog_command",
                "catalog_command": "set_endpoint",
                "value_argument": "rpc_endpoint",
            },
            "validation": {"value_type": "url"},
        }
        text = (
            "Use this endpoint:\n"
            "http://fake-node:19000\n"
            "Also apply the other thing I mentioned."
        )
        state["last_user_input"] = text
        clauses = [
            "Use this endpoint:",
            "http://fake-node:19000",
            "Also apply the other thing I mentioned.",
        ]
        clarification = {
            "actions": [{"type": "clarify_unresolved", "clauses": clauses}],
            "semantic_units": [
                {
                    "unit_id": f"unit-{index}",
                    "clause_id": f"clause-{index}",
                    "source_text": clause,
                    "disposition": "action",
                    "action_indexes": [0],
                    "reason": "one sibling remains unresolved",
                }
                for index, clause in enumerate(clauses, start=1)
            ],
        }
        provider = _provider([clarification, clarification])

        with patch("agent.harness.intent.provider_from_config", return_value=provider):
            result = invoke_product_graph_turn(state, allow_semantic_resolver=True)

        self.assertEqual(result["pending_question"]["id"], "custom_rpc_endpoint")
        self.assertEqual(result.get("endpoint_evidence") or {}, {})
        self.assertEqual(result.get("custom_rpc") or {}, {})
        self.assertEqual(result.get("action_queue") or [], [])
        self.assertTrue(result.get("visible_response"))

    def test_multiple_pending_urls_do_not_create_a_manual_candidate(self) -> None:
        from agent.harness.intent import resolve_action_queue

        state = _state(active_group="endpoint_process")
        state["pending_question"] = {
            "id": "endpoint",
            "group": "endpoint_process",
            "kind": "url",
            "manual_input_allowed": True,
            "options": [],
            "manual_action": {
                "type": "rpc_catalog_command",
                "catalog_command": "set_endpoint",
                "value_argument": "rpc_endpoint",
            },
            "validation": {},
        }
        text = "Use http://node-a:8545 or http://node-b:8545."
        clarification = _document({
            "type": "clarify_unresolved",
            "clauses": [text],
        }, text)
        provider = _provider([clarification, clarification])

        with patch("agent.harness.intent.provider_from_config", return_value=provider):
            result = resolve_action_queue(state, text)

        self.assertEqual(provider.complete.call_count, 4)
        self.assertEqual(result["actions"][0]["type"], "clarify_unresolved")

    def test_focused_typed_answer_cannot_hide_an_unresolved_compound_unit(self) -> None:
        from agent.harness.intent import resolve_action_queue

        state = _performance_manual_state(advanced=True)
        first_text = "Use a twelve second interval here."
        second_text = "Also change the setting I mentioned yesterday."
        text = f"{first_text}\n{second_text}"
        first = {
            "actions": [
                {"type": "clarify_unresolved", "clauses": [first_text]},
                {
                    "type": "answer_opening_question",
                    "topic": "config_explanation",
                    "subject": "the setting mentioned yesterday",
                    "source_evidence": second_text,
                },
            ],
            "semantic_units": [
                {
                    "unit_id": "unit-1",
                    "clause_id": "clause-1",
                    "source_text": first_text,
                    "disposition": "action",
                    "action_indexes": [0],
                    "reason": "pending answer unresolved",
                },
                {
                    "unit_id": "unit-2",
                    "clause_id": "clause-2",
                    "source_text": second_text,
                    "disposition": "action",
                    "action_indexes": [1],
                    "reason": "independent consultation",
                },
            ],
        }
        focused = {
            "actions": [{
                "type": "answer_pending",
                "answer": "12",
                "source_evidence": first_text,
            }],
            "semantic_units": [
                {
                    "unit_id": "unit-1",
                    "clause_id": "clause-1",
                    "source_text": first_text,
                    "disposition": "action",
                    "action_indexes": [0],
                    "reason": "normalized typed answer",
                },
                {
                    "unit_id": "unit-2",
                    "clause_id": "clause-2",
                    "source_text": second_text,
                    "disposition": "unresolved",
                    "action_indexes": [],
                    "reason": "missing referent",
                },
            ],
        }
        provider = _provider([first, focused])

        with patch("agent.harness.intent.provider_from_config", return_value=provider):
            result = resolve_action_queue(state, text)

        self.assertEqual(provider.complete.call_count, 2)
        self.assertEqual(result["actions"][0]["type"], "clarify_unresolved")
        self.assertIn(second_text, result["actions"][0]["clauses"])

    def test_focused_adjudication_preserves_compound_interruption(self) -> None:
        from agent.harness.intent import resolve_action_queue

        state = _recovery_state()
        text = "Pause recovery and retain the evidence.\nAlso explain what this Agent can do."
        first = {
            "actions": [
                {"type": "clarify_unresolved", "clauses": ["Pause recovery and retain the evidence."]},
                {
                    "type": "answer_opening_question",
                    "topic": "capabilities",
                    "source_evidence": "Also explain what this Agent can do.",
                },
            ],
            "semantic_units": [
                {
                    "unit_id": "unit-1",
                    "clause_id": "clause-1",
                    "source_text": "Pause recovery and retain the evidence.",
                    "disposition": "action",
                    "action_indexes": [0],
                    "reason": "active choice is unresolved",
                },
                {
                    "unit_id": "unit-2",
                    "clause_id": "clause-2",
                    "source_text": "Also explain what this Agent can do.",
                    "disposition": "action",
                    "action_indexes": [1],
                    "reason": "independent consultation",
                },
            ],
        }
        resolved = deepcopy(first)
        resolved["actions"][0] = {"type": "cancel_failure_recovery", "confidence": "high"}
        resolved["semantic_units"][0]["reason"] = "declared pending option owner"
        provider = _provider([first, resolved])

        with patch("agent.harness.intent.provider_from_config", return_value=provider):
            result = resolve_action_queue(state, text)

        self.assertEqual(provider.complete.call_count, 3)
        self.assertEqual(
            [action["type"] for action in result["actions"]],
            ["answer_pending", "answer_opening_question"],
        )
        self.assertEqual(
            result["pending_choice_contracts"][0]["semantic_units"][0]["unit_id"],
            "unit-1",
        )

    def test_fully_represented_unrelated_consultation_does_not_trigger_adjudication(self) -> None:
        from agent.harness.intent import resolve_action_queue

        state = _recovery_state()
        text = "What capabilities does this Agent provide?"
        provider = _provider([_document({
            "type": "answer_opening_question",
            "topic": "capabilities",
            "source_evidence": text,
        }, text)])

        with patch("agent.harness.intent.provider_from_config", return_value=provider):
            result = resolve_action_queue(state, text)

        self.assertEqual(provider.complete.call_count, 2)
        self.assertEqual(result["actions"][0]["type"], "answer_opening_question")
        self.assertEqual(result.get("pending_choice_contracts"), [])

    def test_owner_action_without_source_argument_uses_pending_contract_after_admission(self) -> None:
        from agent.harness.coordinator import _dispatch_pending_action, _validate_action_plan
        from agent.harness.intent import resolve_action_queue

        state = _recovery_state()
        text = "Pause this recovery and retain its evidence."
        provider = _provider([_document({
            "type": "cancel_failure_recovery",
            "confidence": "high",
        }, text)])

        with patch("agent.harness.intent.provider_from_config", return_value=provider):
            result = resolve_action_queue(state, text)

        action = result["actions"][0]
        self.assertEqual(action["type"], "answer_pending")
        self.assertEqual(action["selected_value"], "cancel")
        state["last_user_input"] = text
        state.setdefault("turn_context", {})["pending_choice_contracts"] = result["pending_choice_contracts"]
        admitted = _validate_action_plan(state, result["actions"])
        self.assertEqual([item["type"] for item in admitted], ["answer_pending"])
        self.assertIs(admitted[0]["selection_contract_verified"], True)

        executed = _dispatch_pending_action(state, admitted[0])
        self.assertEqual((executed.get("failure_recovery") or {}).get("status"), "cancelled")
        self.assertEqual(
            ((executed.get("turn_context") or {}).get("admitted_actions") or [{}])[-1].get("type"),
            "cancel_failure_recovery",
        )

    def test_multi_unit_provenance_is_preserved_on_one_canonical_choice(self) -> None:
        from agent.harness.intent import _canonicalize_pending_choice_actions, _parse_json_object

        state = _recovery_state()
        payload = {
            "actions": [{"type": "cancel_failure_recovery", "confidence": "high"}],
            "semantic_units": [
                {
                    "unit_id": "unit-1",
                    "clause_id": "clause-1",
                    "source_text": "Pause recovery.",
                    "disposition": "action",
                    "action_indexes": [0],
                    "reason": "first semantic unit",
                },
                {
                    "unit_id": "unit-2",
                    "clause_id": "clause-2",
                    "source_text": "Keep the evidence.",
                    "disposition": "action",
                    "action_indexes": [0],
                    "reason": "second semantic unit",
                },
            ],
        }

        normalized = _parse_json_object(_canonicalize_pending_choice_actions(
            json.dumps(payload),
            state,
        ))

        self.assertEqual(normalized["actions"][0]["type"], "answer_pending")
        self.assertEqual(
            [unit["unit_id"] for unit in normalized["pending_choice_contracts"][0]["semantic_units"]],
            ["unit-1", "unit-2"],
        )

    def test_pending_typed_candidates_expose_exact_method_and_partial_wire_evidence(self) -> None:
        from agent.harness.questions import typed_pending_value_candidates

        method_question = {
            "kind": "manual_value",
            "manual_input_allowed": True,
            "validation": {"input_mode": "rpc_method_or_schema_evidence"},
        }
        self.assertEqual(
            typed_pending_value_candidates(
                "The custom JSON-RPC method is eth_accounts.",
                method_question,
            ),
            ("eth_accounts",),
        )
        evidence = (
            "I only have the request example so far:\n"
            '{"jsonrpc":"2.0","id":1,"method":"getLatestBlock","params":[]}'
        )
        evidence_question = {
            "kind": "evidence",
            "manual_input_allowed": True,
            "validation": {
                "value_type": "evidence_contribution",
                "max_length": 65536,
            },
        }
        self.assertEqual(
            typed_pending_value_candidates(evidence, evidence_question),
            (evidence,),
        )


if __name__ == "__main__":
    unittest.main()

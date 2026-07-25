"""Focused contracts for canonical semantic pending-choice admission."""

from __future__ import annotations

import json
import unittest
from copy import deepcopy
from types import SimpleNamespace
from typing import Any, Callable
from unittest.mock import Mock, patch


def _state(**updates: Any) -> dict[str, Any]:
    from agent.harness.state import new_state

    state = new_state("pending-choice-canonical", language="en")
    state.update(deepcopy(updates))
    return state


CandidateSelector = Callable[[dict[str, Any], dict[str, dict[str, Any]]], str]


def _admission_response(
    request: Any,
    *,
    candidate_selector: CandidateSelector | None = None,
) -> SimpleNamespace:
    review = json.loads(request.messages[1].content)
    units = {str(row["unit_id"]): row for row in review["semantic_units"]}
    selected_identities = {
        str(action["action_id"]): (
            candidate_selector(action, units)
            if candidate_selector is not None
            else ""
        )
        for action in review["actions"]
    }
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
                    next(
                        (
                            str(candidate["candidate_id"])
                            for candidate in action.get("pending_value_candidates") or []
                            if str(candidate.get("identity") or "")
                            == selected_identities[str(action["action_id"])]
                        ),
                        "",
                    )
                ),
                "turn_candidate_verdicts": _turn_candidate_verdicts(
                    action,
                    units,
                    selected_identity=selected_identities[
                        str(action["action_id"])
                    ],
                ),
                "reason": "the immutable action preserves all mapped source units",
            }
            for action in review["actions"]
        ],
        "unit_verdicts": [
            {
                "unit_id": unit["unit_id"],
                "verdict": (
                    "unresolved"
                    if unit["disposition"] == "unresolved"
                    else "complete"
                ),
                "owner_action_ids": list(unit["owner_action_ids"]),
                "evidence_quote": str(unit["source_text"]),
                "omitted_action_type": "",
                "reason": "the immutable unit retains its declared disposition",
            }
            for unit in review["semantic_units"]
        ],
        "reason": "the immutable plan is admitted",
    }, ensure_ascii=False, sort_keys=True))


def _turn_candidate_verdicts(
    action: dict[str, Any],
    units: dict[str, dict[str, Any]],
    *,
    selected_identity: str = "",
) -> list[dict[str, Any]]:
    pending = list(action.get("pending_value_candidates") or [])
    candidates = list(action.get("turn_pending_value_candidates") or [])
    if not pending or not candidates:
        return []
    sources = [
        str(units[unit_id]["source_text"])
        for unit_id in action.get("unit_ids") or []
    ]
    return [
        {
            "candidate_id": str(candidate["candidate_id"]),
            "verdict": (
                "selected"
                if (
                    selected_identity
                    and candidate.get("identity") == selected_identity
                )
                else "not_selected"
            ),
            "evidence_quote": next(
                (
                    str(units[unit_id]["source_text"])
                    for unit_id in candidate.get("source_unit_ids") or []
                    if unit_id in units
                ),
                sources[0],
            ),
            "reason": "the fixture preserves the candidate's source role",
        }
        for candidate in candidates
    ]


def _select_unique_operation_candidate(
    action: dict[str, Any],
    _units: dict[str, dict[str, Any]],
) -> str:
    """Explicit positive-fixture reviewer decision."""

    candidates = list(action.get("pending_value_candidates") or [])
    return (
        str(candidates[0].get("identity") or "")
        if len(candidates) == 1
        else ""
    )


def _provider(
    documents: list[dict[str, Any]],
    *,
    candidate_selector: CandidateSelector | None = None,
) -> Mock:
    provider = Mock()
    compiler_documents = iter(documents)

    def complete(request: Any) -> SimpleNamespace:
        payload = json.loads(request.messages[1].content)
        if "plan_hash" in payload and "immutable_document" in payload:
            return _admission_response(
                request,
                candidate_selector=candidate_selector,
            )
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


    def test_declared_option_id_is_canonicalized_to_its_value(self) -> None:
        from agent.harness.questions import choice_question, question_text
        from agent.harness.semantic_admission import _canonicalize_pending_choice_actions
        from agent.harness.plan_coverage import segment_user_turn

        state = _state(active_group="chain_auxiliary_endpoints")
        state["pending_question"] = choice_question(
            "chain_auxiliary_endpoints",
            "OPTIONAL_FIELD",
            question_text(
                "question.chain_rpc.auxiliary_endpoint.prompt",
                chain="ethereum",
                field="OPTIONAL_FIELD",
            ),
            owner="chain_rpc",
            field="OPTIONAL_FIELD",
            kind="manual_value",
            manual_input_allowed=True,
            options=[{
                "id": "skip",
                "label": question_text(
                    "question.chain_rpc.option.skip_unconfigured"
                ),
                "value": "none",
                "action": {"type": "answer_pending"},
            }],
            validation={"value_type": "scalar_token"},
        )
        text = "Skip it."
        clause = segment_user_turn(text)[0]
        candidate = json.dumps({
            "actions": [{
                "type": "answer_pending",
                "answer": "skip",
                "source_evidence": "Skip",
            }],
            "semantic_units": [{
                "unit_id": "unit-1",
                "clause_id": clause.clause_id,
                "source_text": clause.text,
                "disposition": "action",
                "action_indexes": [0],
                "reason": "declared option id",
            }],
        })

        canonical = json.loads(_canonicalize_pending_choice_actions(candidate, state))

        self.assertNotIn("answer", canonical["actions"][0])
        self.assertEqual(canonical["actions"][0]["selected_value"], "none")
        self.assertEqual(
            canonical["pending_choice_contracts"][0]["option"]["selected_value"],
            "none",
        )


    def test_semantic_review_derives_incomplete_intake_from_registry(self) -> None:
        from agent.harness.domains.chain_rpc_questions import _target_change_scope_question
        from agent.harness.semantic_admission import (
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
        from agent.harness.semantic_admission import _freeze_bounded_semantic_plan
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


    def test_same_unit_explicit_mutation_invalidates_the_old_pending_answer(self) -> None:
        from agent.harness.admission import validate_action_plan

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

        result = validate_action_plan(state, actions)

        self.assertEqual(result.status, "rejected")
        self.assertFalse(result.actions)


    def test_distinct_unit_mutation_invalidates_pending_answer(self) -> None:
        from agent.harness.admission import validate_action_plan

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

        result = validate_action_plan(state, actions)

        self.assertEqual(result.status, "rejected")
        self.assertFalse(result.actions)


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
            "tests.agent_live.graph_turn.TEST_SEMANTIC_PLANNER",
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
            "tests.agent_live.graph_turn.TEST_SEMANTIC_PLANNER",
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
            "tests.agent_live.graph_turn.TEST_SEMANTIC_PLANNER",
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

    def test_admitted_semantic_weight_mapping_reaches_domain_as_one_typed_value(self) -> None:
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
            "tests.agent_live.graph_turn.TEST_SEMANTIC_PLANNER",
            return_value={"actions": [{
                "type": "answer_pending",
                "answer": weights,
                "source_evidence": text,
                "pending_option_semantic_verified": True,
                "semantic_purpose_verified": True,
            }]},
        ):
            result = invoke_product_graph_turn(state, allow_semantic_resolver=True)

        self.assertEqual((result.get("chain_identity") or {}).get("weights"), weights)
        self.assertNotEqual(
            (result.get("pending_question") or {}).get("id"),
            "new_chain_custom_weights",
        )

    def test_deterministic_weight_mapping_runs_complete_product_graph(self) -> None:
        from tests.agent_live.graph_turn import (
            answer_pending,
        )
        from tests.agent_live.harness_contract_scenarios import question_scenarios

        scenario = next(
            item for item in question_scenarios("en")
            if item.scenario_id == "new_chain_existing_family_needs_weights"
        )
        state = deepcopy(dict(scenario.seed_state))
        text = "eth_blockNumber=45,eth_gasPrice=55"
        state["last_user_input"] = text
        weights = {"eth_blockNumber": 45, "eth_gasPrice": 55}

        result = answer_pending(
            state,
            text,
            manual_value=weights,
        )
        self.assertEqual(
            (result.get("chain_identity") or {}).get("weights"),
            weights,
        )

    def test_legacy_manual_owner_metadata_is_not_an_action_argument(self) -> None:
        from agent.harness.semantic_admission import _matching_pending_manual_value
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


    def test_equal_candidate_identities_are_scoped_per_action_owner(self) -> None:
        from agent.harness.questions import manual_question, question_text
        from agent.harness.semantic_admission import _freeze_bounded_semantic_plan
        from agent.harness.plan_coverage import segment_user_turn

        text = "eth0\neth0"
        state = _state(active_group="network")
        state["pending_question"] = manual_question(
            "network",
            "network_interface",
            question_text("question.environment.network_interface.prompt"),
            owner="environment",
            field="NETWORK_INTERFACE",
            kind="device",
            validation={"value_type": "scalar_token"},
        )
        document = {
            "actions": [
                {
                    "type": "answer_pending",
                    "answer": "eth0",
                    "source_evidence": "eth0",
                },
                {
                    "type": "answer_pending",
                    "answer": "eth0",
                    "source_evidence": "eth0",
                },
            ],
            "semantic_units": [
                {
                    "unit_id": "unit-1",
                    "clause_id": "clause-1",
                    "source_text": "eth0",
                    "disposition": "action",
                    "action_indexes": [0],
                },
                {
                    "unit_id": "unit-2",
                    "clause_id": "clause-2",
                    "source_text": "eth0",
                    "disposition": "action",
                    "action_indexes": [1],
                },
            ],
        }

        frozen = _freeze_bounded_semantic_plan(
            json.dumps(document),
            state,
            segment_user_turn(text),
        ).request_payload()

        self.assertEqual(
            frozen["actions"][0]["turn_pending_value_candidates"][0][
                "source_unit_ids"
            ],
            ["unit-1"],
        )
        self.assertEqual(
            frozen["actions"][1]["turn_pending_value_candidates"][0][
                "source_unit_ids"
            ],
            ["unit-2"],
        )

    def test_lowercase_structured_key_requires_explicit_contract_binding(self) -> None:
        from agent.harness.semantic_admission import _freeze_bounded_semantic_plan
        from agent.harness.questions import manual_question, question_text
        from agent.harness.plan_coverage import segment_user_turn

        text = "customField: eth0"
        state = _state(active_group="network")
        state["pending_question"] = manual_question(
            "network",
            "custom_field",
            question_text(
                "question.environment.reconfiguration_value.prompt",
                field="customField",
            ),
            owner="environment",
            field="customField",
            structured_config_key="customField",
        )
        document = _document({
            "type": "propose_config_values",
            "config_values": {"customField": "eth0"},
            "unmapped_values": {},
            "source_format": "yaml",
            "source_evidence": text,
        }, text)

        frozen = _freeze_bounded_semantic_plan(
            json.dumps(document),
            state,
            segment_user_turn(text),
        ).request_payload()

        self.assertEqual(
            frozen["actions"][0]["pending_value_candidates"][0]["value"],
            "eth0",
        )

    def test_candidate_binding_contract_rejects_unregistered_sources(self) -> None:
        from agent.harness.questions import manual_question, question_text

        prompt = question_text(
            "question.environment.reconfiguration_value.prompt",
            field="value",
        )

        with self.assertRaisesRegex(ValueError, "unknown candidate binding"):
            manual_question(
                "network",
                "bad_binding",
                prompt,
                owner="environment",
                field="value",
                candidate_bindings=({
                    "type": "not_registered",
                    "value_argument": "value",
                },),
            )
        with self.assertRaisesRegex(ValueError, "is not registered as a business value"):
            manual_question(
                "network",
                "metadata_binding",
                prompt,
                owner="environment",
                field="value",
                candidate_bindings=({
                    "type": "set_qps_mode",
                    "value_argument": "source_evidence",
                },),
            )
        with self.assertRaisesRegex(ValueError, "cannot declare mapping_key"):
            manual_question(
                "network",
                "scalar_mapping_binding",
                prompt,
                owner="environment",
                field="value",
                candidate_bindings=({
                    "type": "set_sync_observe_options",
                    "value_argument": "sync_observe_duration_seconds",
                    "mapping_key": "duration",
                },),
            )
        with self.assertRaisesRegex(ValueError, "is not registered as a business value"):
            manual_question(
                "network",
                "control_argument_binding",
                prompt,
                owner="environment",
                field="value",
                candidate_bindings=({
                    "type": "choose_target_mode",
                    "value_argument": "target_mode_explicit",
                },),
            )
        with self.assertRaisesRegex(ValueError, "invalid mapping_key"):
            manual_question(
                "qps_profile",
                "invalid_qps_key",
                prompt,
                owner="performance",
                field="value",
                candidate_bindings=({
                    "type": "set_qps_override",
                    "value_argument": "qps_overrides",
                    "mapping_key": "NOT_A_QPS_FIELD",
                },),
            )
        with self.assertRaisesRegex(ValueError, "requires mapping_key"):
            manual_question(
                "qps_profile",
                "missing_qps_key",
                prompt,
                owner="performance",
                field="value",
                candidate_bindings=({
                    "type": "set_qps_override",
                    "value_argument": "qps_overrides",
                },),
            )

    def test_persisted_unknown_candidate_binding_is_quarantined(self) -> None:
        from agent.harness.state import STATE_SCHEMA_VERSION, migrate_state

        migrated = migrate_state(
            {
                "schema_version": STATE_SCHEMA_VERSION,
                "pending_question": {
                    "contract_version": 2,
                    "id": "stale_question",
                    "group": "qps_profile",
                    "owner": "performance",
                    "kind": "manual_value",
                    "field": "stale_question",
                    "manual_input_allowed": True,
                    "accepted_action_types": [
                        "answer_pending",
                        "removed_action",
                    ],
                    "candidate_bindings": [{
                        "type": "removed_action",
                        "value_argument": "value",
                    }],
                    "options": [],
                    "validation": {},
                },
            },
            thread_id="candidate-binding-migration",
            language="en",
            session_purpose="user",
        )

        self.assertEqual(migrated["pending_question"], {})
        self.assertTrue(any(
            event.get("event") == "checkpoint_pending_actions_quarantined"
            and "contract_version 3" in str(event.get("contract_error") or "")
            for event in migrated.get("audit_events") or []
        ))

    def test_persisted_invalid_known_candidate_binding_is_quarantined(self) -> None:
        from agent.harness.state import STATE_SCHEMA_VERSION, migrate_state

        migrated = migrate_state(
            {
                "schema_version": STATE_SCHEMA_VERSION,
                "pending_question": {
                    "contract_version": 2,
                    "id": "invalid_qps_binding",
                    "group": "qps_profile",
                    "owner": "performance",
                    "kind": "manual_value",
                    "field": "invalid_qps_binding",
                    "manual_input_allowed": True,
                    "accepted_action_types": [
                        "answer_pending",
                        "set_qps_override",
                    ],
                    "candidate_bindings": [{
                        "type": "set_qps_override",
                        "value_argument": "qps_overrides",
                        "mapping_key": "NOT_A_QPS_FIELD",
                    }],
                    "options": [],
                    "validation": {},
                },
            },
            thread_id="candidate-binding-known-invalid",
            language="en",
            session_purpose="user",
        )

        self.assertEqual(migrated["pending_question"], {})
        self.assertTrue(any(
            event.get("event") == "checkpoint_pending_actions_quarantined"
            and "contract_version 3" in str(event.get("contract_error") or "")
            for event in migrated.get("audit_events") or []
        ))

    def test_pre_v10_pending_contract_is_regenerated(self) -> None:
        from agent.harness.state import migrate_state

        migrated = migrate_state(
            {
                "schema_version": 9,
                "pending_question": {
                    "contract_version": 1,
                    "id": "qps_adjust_value",
                    "group": "qps_profile",
                    "kind": "manual_value",
                    "field": "qps_adjust_value",
                    "manual_input_allowed": True,
                    "accepted_action_types": ["answer_pending"],
                    "options": [],
                    "validation": {"value_type": "positive_number"},
                },
            },
            thread_id="candidate-binding-v10-migration",
            language="en",
            session_purpose="user",
        )

        self.assertEqual(migrated["pending_question"], {})
        self.assertTrue(any(
            event.get("event")
            == "checkpoint_legacy_quarantined"
            and event.get("from_schema_version") == 9
            for event in migrated.get("audit_events") or []
        ))

    def test_current_schema_rejects_contract_v1_pending_question(self) -> None:
        from agent.harness.state import STATE_SCHEMA_VERSION, migrate_state

        migrated = migrate_state(
            {
                "schema_version": STATE_SCHEMA_VERSION,
                "pending_question": {
                    "contract_version": 1,
                    "id": "stale_v1",
                    "manual_input_allowed": True,
                    "accepted_action_types": ["answer_pending"],
                },
            },
            thread_id="candidate-binding-current-v1",
            language="en",
            session_purpose="user",
        )

        self.assertEqual(migrated["pending_question"], {})
        self.assertEqual(
            migrated["checkpoint_recovery"]["status"],
            "quarantined",
        )


    def test_structured_pending_fast_path_does_not_steal_workflow_values(self) -> None:
        from agent.harness.questions import manual_question, question_text
        from tests.agent_live.graph_turn import invoke_product_graph_turn

        text = "ACCOUNTS_VOL_TYPE: hyperdisk-balanced\nRPC_MODE: mixed"
        state = _state(active_group="accounts_disk")
        state["pending_question"] = manual_question(
            "accounts_disk",
            "ACCOUNTS_VOL_TYPE",
            question_text("question.environment.accounts_vol_type.prompt"),
            owner="environment",
            field="ACCOUNTS_VOL_TYPE",
            structured_config_key="ACCOUNTS_VOL_TYPE",
            validation={"value_type": "scalar_token"},
        )
        state["last_user_input"] = text
        resolver = Mock(return_value={
            "actions": [{"type": "clarify_unresolved", "clauses": [text]}],
        })

        with patch("tests.agent_live.graph_turn.TEST_SEMANTIC_PLANNER", resolver):
            result = invoke_product_graph_turn(state)

        resolver.assert_called_once()
        self.assertNotIn("pending_review", result.get("inferred_config") or {})
        self.assertEqual(result["pending_question"]["id"], "ACCOUNTS_VOL_TYPE")


    def test_unresolved_sibling_prevents_partial_pending_commit_in_product_graph(self) -> None:
        from agent.harness.questions import manual_question, question_text
        from tests.agent_live.graph_turn import invoke_product_graph_turn

        state = _state(active_group="endpoint_process")
        state["pending_question"] = manual_question(
            "endpoint_process",
            "custom_rpc_endpoint",
            question_text("question.chain_rpc.custom_endpoint.prompt"),
            owner="chain_rpc",
            field="custom_rpc_endpoint",
            kind="url",
            manual_action={
                "type": "rpc_catalog_command",
                "catalog_command": "set_endpoint",
                "value_argument": "rpc_endpoint",
            },
            validation={"value_type": "url"},
            domain_context={"rpc_case": "custom_rpc"},
        )
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
        with patch(
            "tests.agent_live.graph_turn.TEST_SEMANTIC_PLANNER",
            return_value={
                **clarification,
                "actions": [{
                    "type": "unknown",
                    "reason": "one sibling remains unresolved",
                    "confidence": "low",
                }],
            },
        ):
            result = invoke_product_graph_turn(state, allow_semantic_resolver=True)

        self.assertEqual(result["pending_question"]["id"], "custom_rpc_endpoint")
        self.assertEqual(result.get("endpoint_evidence") or {}, {})
        self.assertEqual(result.get("custom_rpc") or {}, {})
        self.assertEqual(result.get("action_queue") or [], [])
        self.assertTrue(result.get("visible_response"))


    def test_multi_unit_provenance_is_preserved_on_one_canonical_choice(self) -> None:
        from agent.harness.semantic_admission import _canonicalize_pending_choice_actions, _parse_json_object

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
        scalar_question = {
            "kind": "device",
            "manual_input_allowed": True,
            "validation": {"value_type": "scalar_token"},
        }
        self.assertEqual(
            typed_pending_value_candidates("eth0", scalar_question),
            ("eth0",),
        )
        self.assertEqual(
            typed_pending_value_candidates(
                "Use this interface for the benchmark:",
                scalar_question,
            ),
            (),
        )


if __name__ == "__main__":
    unittest.main()

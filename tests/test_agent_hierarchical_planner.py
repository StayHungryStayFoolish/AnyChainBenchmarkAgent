from __future__ import annotations

import json
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch


class HierarchicalPlannerContractTest(unittest.TestCase):
    def test_open_chain_pending_rejects_registered_closed_domain_value(self) -> None:
        from agent.harness.hierarchical_planner import _cross_domain_pending_errors

        state = {
            "pending_question": {
                "group": "chain_identity",
                "value_domain": "researched_identity",
            },
        }
        partition = [{
            "unit_id": "unit-1",
            "source_text": "fake-node test",
            "operation": "pending_answer",
        }]

        errors = _cross_domain_pending_errors(partition, state)

        self.assertTrue(errors)
        self.assertIn("choose_target_mode", errors[0])

    def test_open_chain_pending_allows_unregistered_identity_candidate(self) -> None:
        from agent.harness.hierarchical_planner import _cross_domain_pending_errors

        state = {
            "pending_question": {
                "group": "chain_identity",
                "value_domain": "researched_identity",
            },
        }
        partition = [{
            "unit_id": "unit-1",
            "source_text": "AlphaChain",
            "operation": "pending_answer",
        }]

        self.assertEqual(_cross_domain_pending_errors(partition, state), ())

    def test_final_admission_rejects_closed_domain_value_as_chain_identity(self) -> None:
        from agent.harness.domains.chain_rpc_questions import _chain_question
        from agent.harness.plan_coverage import segment_user_turn
        from agent.harness.semantic_admission import prepare_hierarchical_candidate
        from agent.harness.state import new_state

        text = "fake-node"
        clauses = segment_user_turn(text)
        state = new_state("closed-domain-final-admission", language="en")
        state["pending_question"] = _chain_question(state)
        candidate = {
            "actions": [{
                "type": "choose_chain",
                "chain_text": text,
                "source_evidence": text,
            }],
            "semantic_units": [{
                "unit_id": "unit-1",
                "clause_id": "clause-1",
                "source_text": text,
                "disposition": "action",
                "action_indexes": [0],
                "reason": "claimed chain answer",
            }],
        }

        _prepared, validation = prepare_hierarchical_candidate(
            json.dumps(candidate),
            state,
            clauses,
            pending_choice_unit_ids=frozenset({"unit-1"}),
        )

        self.assertFalse(validation.valid)
        self.assertTrue(
            any("registered closed-domain value" in error for error in validation.errors),
            validation.errors,
        )

    def test_final_admission_allows_open_chain_identity_candidate(self) -> None:
        from agent.harness.domains.chain_rpc_questions import _chain_question
        from agent.harness.plan_coverage import segment_user_turn
        from agent.harness.semantic_admission import prepare_hierarchical_candidate
        from agent.harness.state import new_state

        text = "AlphaChain"
        clauses = segment_user_turn(text)
        state = new_state("open-domain-final-admission", language="en")
        state["pending_question"] = _chain_question(state)
        candidate = {
            "actions": [{
                "type": "choose_chain",
                "chain_text": text,
                "source_evidence": text,
            }],
            "semantic_units": [{
                "unit_id": "unit-1",
                "clause_id": "clause-1",
                "source_text": text,
                "disposition": "action",
                "action_indexes": [0],
                "reason": "open chain identity",
            }],
        }

        prepared, validation = prepare_hierarchical_candidate(
            json.dumps(candidate),
            state,
            clauses,
            pending_choice_unit_ids=frozenset({"unit-1"}),
        )

        self.assertTrue(validation.valid, validation.errors)
        self.assertEqual(json.loads(prepared)["actions"][0]["answer"], text)

    def test_stage_b_pending_answer_contract_exposes_complete_typed_options(
        self,
    ) -> None:
        from agent.harness.hierarchical_planner import (
            _stage_b_payload,
            _stage_b_prompt,
        )
        from agent.harness.state import new_state

        state = new_state("stage-b-pending-options", language="en")
        state["pending_question"] = {
            "id": "generic_choice",
            "group": "opening",
            "kind": "numbered_choice",
            "prompt": "Choose one.",
            "field": "generic_choice",
            "manual_input_allowed": False,
            "options": [
                {"id": "first", "label": "First", "value": "alpha"},
                {"id": "other", "label": "None / unsure", "value": "unknown"},
            ],
            "accepted_action_types": ["answer_pending"],
        }
        partition = [{
            "unit_id": "unit-1",
            "clause_id": "clause-1",
            "source_text": "None of these fit and I am not sure.",
            "operation": "pending_answer",
            "owner_routes": [{"owner": "coordinator", "group": ""}],
            "reason": "semantic option selection",
        }]

        payload = _stage_b_payload(
            state,
            "coordinator",
            frozenset(),
            partition,
            ("unit-1",),
        )
        prompt = _stage_b_prompt("coordinator")

        self.assertEqual(
            payload["owner_state"]["pending_question"]["options"],
            state["pending_question"]["options"],
        )
        self.assertIn(
            "answer_pending",
            {row["type"] for row in payload["owner_action_schema"]},
        )
        self.assertIn("owner_state.pending_question", prompt)
        self.assertIn("exactly one option is semantically selected", prompt)
        self.assertIn("zero or several options", prompt)
        self.assertIn("emit the coordinator-owned answer_pending action", prompt)
        self.assertIn(
            "selected_value equal to that option's exact value, omit answer",
            prompt,
        )
        self.assertIn(
            "For valid manual input, emit answer and omit selected_value",
            prompt,
        )
        self.assertIn(
            "Never emit that specialized option.action directly",
            prompt,
        )

    def test_stage_a_contract_keeps_pending_selection_rationale_atomic(
        self,
    ) -> None:
        from agent.harness.hierarchical_planner import (
            _stage_a_admission_prompt,
            _stage_a_prompt,
        )

        partition_prompt = _stage_a_prompt()
        admission_prompt = _stage_a_admission_prompt()

        self.assertIn("form one pending_answer operation", partition_prompt)
        self.assertIn(
            "active pending_question.group as that route's group",
            partition_prompt,
        )
        self.assertIn("referential application", partition_prompt)
        self.assertIn("declared completion effect", partition_prompt)
        self.assertIn(
            "independently asks for research, explanation, navigation, or a mutation",
            partition_prompt,
        )
        self.assertIn(
            "do not approve a fictitious second demand",
            admission_prompt,
        )
        self.assertIn(
            "form one pending_answer for the affirmed option",
            partition_prompt,
        )
        self.assertIn(
            "rejection of one declared option as selecting that rejected option",
            admission_prompt,
        )
        self.assertIn("one parser-proven manual value", partition_prompt)
        self.assertIn("pending_typed_candidates proves one candidate", admission_prompt)

    def test_stage_a_redundancy_preserves_source_coverage_but_not_execution(
        self,
    ) -> None:
        from agent.harness.hierarchical_planner import (
            _partition_after_stage_a_admission,
        )

        partition = [
            {
                "unit_id": "unit-1",
                "clause_id": "clause-1",
                "source_text": "Disable observability for this run;",
                "operation": "pending_answer",
                "owner_routes": [{"owner": "coordinator", "group": ""}],
                "reason": "selects the disabled option",
            },
            {
                "unit_id": "unit-2",
                "clause_id": "clause-2",
                "source_text": "I don't want any monitoring enabled.",
                "operation": "domain_request",
                "owner_routes": [{
                    "owner": "performance",
                    "group": "observability",
                }],
                "reason": "restates the selected option",
            },
        ]

        source, compilation = _partition_after_stage_a_admission(
            partition,
            frozenset({"unit-2"}),
        )

        self.assertEqual([row["unit_id"] for row in source], ["unit-1", "unit-2"])
        self.assertEqual([row["unit_id"] for row in compilation], ["unit-1"])
        self.assertEqual(source[1]["operation"], "context")
        self.assertEqual(source[1]["owner_routes"], [])
        self.assertIn("non-executable duplicate", source[1]["reason"])
        self.assertEqual(partition[1]["operation"], "domain_request")

    def test_active_pending_lane_cannot_bypass_global_partition(self) -> None:
        from tests.agent_live.graph_turn import resolve_product_action_queue_for_test as resolve_product_action_queue
        from agent.harness.state import new_state

        state = new_state("pending-route-first", language="en")
        state["pending_question"] = {
            "id": "generic_choice",
            "group": "opening",
            "kind": "numbered_choice",
            "field": "target_mode",
            "manual_input_allowed": False,
            "options": [{"id": "one", "label": "First", "value": "alpha"}],
            "accepted_action_types": ["answer_pending"],
        }

        with (
            patch(
                "agent.harness.hierarchical_planner.provider_from_config",
                return_value=object(),
            ),
            patch(
                "agent.harness.hierarchical_planner.request_semantic_compilation",
                return_value="{}",
            ) as global_compiler,
        ):
            result = resolve_product_action_queue(
                state,
                "Switch to a different workflow instead.",
            )

        self.assertEqual(global_compiler.call_count, 2)
        self.assertEqual(result["planner_metrics"]["stage_a_calls"], 2)
        self.assertEqual(result["planner_metrics"]["stage_b_calls"], 0)
        self.assertTrue(result["actions"])

    def test_stage_a_partition_does_not_return_to_focused_pending_authority(
        self,
    ) -> None:
        from tests.agent_live.graph_turn import resolve_product_action_queue_for_test as resolve_product_action_queue
        from agent.harness.state import new_state

        state = new_state("stage-a-before-focused", language="en")
        state["pending_question"] = {
            "id": "generic_choice",
            "group": "opening",
            "kind": "numbered_choice",
            "field": "target_mode",
            "manual_input_allowed": False,
            "options": [{"id": "one", "label": "First", "value": "alpha"}],
            "accepted_action_types": ["answer_pending"],
        }
        partition = [
            {
                "unit_id": "focused-unit-1",
                "clause_id": "clause-1",
                "source_text": "Use the first choice.",
                "operation": "pending_answer",
                "owner_routes": [{"owner": "coordinator", "group": "opening"}],
                "reason": "pending selection",
            },
            {
                "unit_id": "focused-unit-2",
                "clause_id": "clause-2",
                "source_text": "Set the QPS profile to quick.",
                "operation": "domain_request",
                "owner_routes": [{
                    "owner": "performance",
                    "group": "qps_profile",
                }],
                "reason": "independent sibling",
            },
        ]
        with (
            patch(
                "agent.harness.hierarchical_planner.provider_from_config",
                return_value=object(),
            ),
            patch(
                "agent.harness.hierarchical_planner.request_semantic_compilation",
                return_value=json.dumps({"semantic_units": partition}),
            ) as stage_a,
            patch(
                "agent.harness.hierarchical_planner._compile_owner_document",
            ) as owner_compiler,
        ):
            result = resolve_product_action_queue(
                state,
                "Use the first choice. Set the QPS profile to quick.",
            )

        self.assertEqual(result["planner_metrics"]["stage_a_calls"], 1)
        stage_a.assert_called_once()
        owner_compiler.assert_called()

    def test_stage_a_keeps_typed_evidence_contribution_atomic(
        self,
    ) -> None:
        from agent.harness.hierarchical_planner import (
            _canonicalize_atomic_evidence_partition,
        )
        from agent.harness.plan_coverage import TurnClause
        from agent.harness.state import new_state

        state = new_state("focused-evidence", language="en")
        state["pending_question"] = {
            "id": "schema_evidence",
            "group": "endpoint_process",
            "validation": {"value_type": "evidence_contribution"},
        }
        source, compilation = _canonicalize_atomic_evidence_partition(
            state,
            (
                TurnClause(
                    "clause-1",
                    "request JSON\nresponse is not available yet",
                    "structured",
                ),
            ),
            [
                {
                    "unit_id": "unit-1",
                    "clause_id": "clause-1",
                    "operation": "pending_answer",
                    "source_text": "request JSON",
                    "owner_routes": [{
                        "owner": "coordinator",
                        "group": "endpoint_process",
                    }],
                },
                {
                    "unit_id": "unit-2",
                    "clause_id": "clause-1",
                    "operation": "context",
                    "source_text": "response is not available yet",
                    "owner_routes": [],
                },
            ],
            [{
                "unit_id": "unit-1",
                "clause_id": "clause-1",
                "operation": "pending_answer",
                "source_text": "request JSON",
                "owner_routes": [{
                    "owner": "coordinator",
                    "group": "endpoint_process",
                }],
            }],
        )

        self.assertEqual(source, compilation)
        self.assertEqual(len(source), 1)
        self.assertEqual(
            source[0]["source_text"],
            "request JSON\nresponse is not available yet",
        )
        self.assertEqual(source[0]["operation"], "pending_answer")

    def test_stage_a_preserves_valid_structured_route(self) -> None:
        from agent.harness.hierarchical_planner import (
            _validate_partition_document,
        )
        from agent.harness.plan_coverage import TurnClause

        clauses = (
            TurnClause("clause-1", "QPS_MODE: quick", "structured"),
        )
        unit = {
            "unit_id": "unit-1",
            "clause_id": "clause-1",
            "source_text": "QPS_MODE: quick",
            "operation": "domain_request",
            "owner_routes": [{
                "owner": "performance",
                "group": "qps_profile",
            }],
            "reason": "structured workflow value",
        }
        partition, errors = _validate_partition_document(
            json.dumps({"semantic_units": [unit], "reason": "complete"}),
            clauses,
        )

        self.assertEqual(
            partition[0]["owner_routes"],
            [{"owner": "performance", "group": "qps_profile"}],
        )
        self.assertEqual(errors, ())

    def test_stage_a_rejects_invalid_structured_route_without_reassigning_owner(
        self,
    ) -> None:
        from agent.harness.hierarchical_planner import (
            _validate_partition_document,
        )
        from agent.harness.plan_coverage import TurnClause

        unit = {
            "unit_id": "unit-1",
            "clause_id": "clause-1",
            "source_text": "QPS_MODE: quick",
            "operation": "domain_request",
            "owner_routes": [{
                "owner": "environment",
                "group": "qps_profile",
            }],
            "reason": "invalid owner claim",
        }
        partition, errors = _validate_partition_document(
            json.dumps({"semantic_units": [unit], "reason": "invalid"}),
            (TurnClause("clause-1", "QPS_MODE: quick", "structured"),),
        )

        self.assertEqual(
            partition[0]["owner_routes"],
            [{"owner": "environment", "group": "qps_profile"}],
        )
        self.assertTrue(
            any("owner/group mismatch" in error for error in errors),
            errors,
        )

    def test_stage_a_rejects_incomplete_structured_route(self) -> None:
        from agent.harness.hierarchical_planner import (
            _validate_partition_document,
        )
        from agent.harness.plan_coverage import TurnClause

        unit = {
            "unit_id": "unit-1",
            "clause_id": "clause-1",
            "source_text": "QPS_MODE: quick",
            "operation": "domain_request",
            "owner_routes": [],
            "reason": "missing route",
        }
        _partition, errors = _validate_partition_document(
            json.dumps({"semantic_units": [unit], "reason": "invalid"}),
            (TurnClause("clause-1", "QPS_MODE: quick", "structured"),),
        )

        self.assertTrue(
            any("actionable unit has no route" in error for error in errors),
            errors,
        )

    def test_unique_manual_candidate_keeps_wrapper_as_source_context(self) -> None:
        from agent.harness.hierarchical_planner import (
            _canonicalize_unique_manual_pending_partition,
        )
        from agent.harness.plan_coverage import segment_user_turn
        from agent.harness.state import new_state

        state = new_state("unique-manual-wrapper", language="en")
        state["active_group"] = "endpoint_process"
        state["pending_question"] = {
            "id": "generic_endpoint",
            "group": "endpoint_process",
            "kind": "url",
            "manual_input_allowed": True,
            "options": [],
            "manual_action": {
                "type": "answer_pending",
                "value_argument": "answer",
            },
            "validation": {},
        }
        clauses = segment_user_turn(
            "Use only this validation value:\n"
            "http://127.0.0.1:8545\n"
            "Do not apply it to the final target."
        )
        partition = [
            {
                "unit_id": f"unit-{index}",
                "clause_id": clause.clause_id,
                "source_text": clause.text,
                "operation": "pending_answer",
                "owner_routes": [{
                    "owner": "coordinator",
                    "group": "endpoint_process",
                }],
                "reason": "one manual transaction",
            }
            for index, clause in enumerate(clauses, start=1)
        ]

        source, compilation = _canonicalize_unique_manual_pending_partition(
            state,
            clauses,
            partition,
            partition,
        )

        self.assertEqual(
            [unit["operation"] for unit in source],
            ["context", "pending_answer", "context"],
        )
        self.assertEqual(len(compilation), 1)
        self.assertEqual(
            compilation[0]["source_text"],
            "http://127.0.0.1:8545",
        )

    def test_stage_a_retains_unclaimed_prose_for_independent_review(self) -> None:
        from agent.harness.hierarchical_planner import (
            _retain_unclaimed_prose_clauses,
        )
        from agent.harness.plan_coverage import TurnClause

        clauses = (
            TurnClause("clause-1", "Choose disabled;", "prose"),
            TurnClause("clause-2", "I do not need metrics.", "prose"),
        )
        units = [{
            "unit_id": "unit-1",
            "clause_id": "clause-1",
            "source_text": "Choose disabled;",
            "operation": "pending_answer",
            "owner_routes": [{
                "owner": "coordinator",
                "group": "observability",
            }],
            "reason": "selects one option",
        }]

        retained = _retain_unclaimed_prose_clauses(units, clauses)

        self.assertEqual(len(retained), 2)
        self.assertEqual(retained[1]["clause_id"], "clause-2")
        self.assertEqual(retained[1]["source_text"], clauses[1].text)
        self.assertEqual(retained[1]["operation"], "context")
        self.assertEqual(retained[1]["owner_routes"], [])

    def test_stage_a_does_not_synthesize_context_for_structured_clause(self) -> None:
        from agent.harness.hierarchical_planner import (
            _retain_unclaimed_prose_clauses,
        )
        from agent.harness.plan_coverage import TurnClause

        clauses = (
            TurnClause("clause-1", "Choose disabled.", "prose"),
            TurnClause("clause-2", "MODE: local", "structured"),
        )
        units = [{
            "unit_id": "unit-1",
            "clause_id": "clause-1",
            "source_text": "Choose disabled.",
            "operation": "pending_answer",
            "owner_routes": [{
                "owner": "coordinator",
                "group": "observability",
            }],
            "reason": "selects one option",
        }]

        retained = _retain_unclaimed_prose_clauses(units, clauses)

        self.assertEqual(retained, units)

    def test_whole_plan_contract_separates_options_from_manual_candidates(
        self,
    ) -> None:
        from agent.harness.semantic_compiler import whole_plan_admission_prompt

        prompt = whole_plan_admission_prompt("semantic policy")

        self.assertIn(
            "pending_answer_argument is an opaque manual-candidate id",
            prompt,
        )
        self.assertIn(
            "Declared options are reviewed through the immutable action and "
            "pending_choice_contracts",
            prompt,
        )
        self.assertIn(
            "supplies an empty pending_value_candidates list, "
            "pending_answer_argument must be exactly the empty string",
            prompt,
        )
        self.assertNotIn("adapter_family_confirm", prompt)

    def test_product_resolver_end_to_end_contract_without_internal_boundary_mocks(
        self,
    ) -> None:
        from tests.agent_live.graph_turn import resolve_product_action_queue_for_test as resolve_product_action_queue
        from agent.harness.state import new_state

        class ContractProvider:
            def complete(self, request):
                system = request.messages[0].content
                payload = json.loads(request.messages[1].content)
                if "Stage A semantic partition" in system:
                    response = {
                        "semantic_units": [{
                            "unit_id": "unit-1",
                            "clause_id": "clause-1",
                            "source_text": "Use fake-node with BSC and quick.",
                            "operation": "domain_request",
                            "owner_routes": [
                                {"owner": "chain_rpc", "group": "target_mode"},
                                {"owner": "chain_rpc", "group": "chain_identity"},
                                {"owner": "performance", "group": "qps_profile"},
                            ],
                            "reason": "three explicit settings",
                        }],
                        "reason": "complete partition",
                    }
                elif "Stage A coverage authority" in system:
                    response = {
                        "unit_verdicts": [{
                            "unit_id": "unit-1",
                            "verdict": "complete",
                            "supports_unit_id": "",
                            "reason": "all settings are routed",
                        }],
                        "clause_verdicts": [{
                            "clause_id": "clause-1",
                            "verdict": "complete",
                            "omitted_owner_routes": [],
                            "reason": "all settings are represented",
                        }],
                        "reason": "complete",
                    }
                elif "Stage B command compiler" in system:
                    owner = payload["owner"]
                    if owner == "chain_rpc":
                        response = {
                            "actions": [
                                {
                                    "type": "choose_target_mode",
                                    "target_mode": "fake-node",
                                    "target_mode_explicit": True,
                                    "source_evidence": "fake-node",
                                },
                                {
                                    "type": "choose_chain",
                                    "chain_text": "BSC",
                                    "source_evidence": "BSC",
                                },
                            ],
                            "bindings": [
                                {
                                    "unit_id": "__harness_route_1",
                                    "action_indexes": [0],
                                    "disposition": "action",
                                    "reason": "target mode",
                                },
                                {
                                    "unit_id": "__harness_route_2",
                                    "action_indexes": [1],
                                    "disposition": "action",
                                    "reason": "chain",
                                },
                            ],
                            "reason": "compiled",
                        }
                    else:
                        response = {
                            "actions": [{
                                "type": "set_qps_mode",
                                "qps_mode": "quick",
                                "mutation_explicit": True,
                                "source_evidence": "quick",
                            }],
                            "bindings": [{
                                "unit_id": "__harness_route_3",
                                "action_indexes": [0],
                                "disposition": "action",
                                "reason": "qps",
                            }],
                            "reason": "compiled",
                        }
                elif "independent admission authority" in system:
                    actions = payload["actions"]
                    units = payload["semantic_units"]
                    unit_by_id = {
                        row["unit_id"]: row for row in units
                    }
                    response = {
                        "plan_hash": payload["plan_hash"],
                        "action_verdicts": [
                            {
                                "action_id": row["action_id"],
                                "verdict": "admit",
                                "unit_ids": row["unit_ids"],
                                "evidence": [
                                    {
                                        "unit_id": unit_id,
                                        "quote": unit_by_id[unit_id]["source_text"],
                                        "relation": "direct",
                                        "support_relation": "",
                                    }
                                    for unit_id in row["unit_ids"]
                                ],
                                "grounded_arguments": [
                                    {
                                        "argument_name": argument,
                                        "evidence_quote": row["action"].get(
                                            "source_evidence",
                                            unit_by_id[row["unit_ids"][0]][
                                                "source_text"
                                            ],
                                        ),
                                    }
                                    for argument in row[
                                        "required_value_grounding_arguments"
                                    ]
                                ],
                                "pending_answer_argument": "",
                                "turn_candidate_verdicts": [],
                                "reason": "directly grounded",
                            }
                            for row in actions
                        ],
                        "unit_verdicts": [
                            {
                                "unit_id": row["unit_id"],
                                "verdict": "complete",
                                "owner_action_ids": row["owner_action_ids"],
                                "evidence_quote": row["source_text"],
                                "omitted_action_type": "",
                                "reason": "all demands have owners",
                            }
                            for row in units
                        ],
                        "reason": "complete and grounded",
                    }
                else:
                    raise AssertionError(system)
                return SimpleNamespace(text=json.dumps(response))

        state = new_state("hierarchical-e2e", language="en")
        with patch(
            "agent.harness.hierarchical_planner.provider_from_config",
            return_value=ContractProvider(),
        ):
            result = resolve_product_action_queue(
                state,
                "Use fake-node with BSC and quick.",
            )

        self.assertFalse(result.get("planner_errors"), result)
        self.assertEqual(
            [action["type"] for action in result["actions"]],
            ["choose_target_mode", "choose_chain", "set_qps_mode"],
        )
        self.assertEqual(result["planner_metrics"]["model_calls"], 4)

    def test_stage_a_partition_is_lossless_and_owner_scoped(self) -> None:
        from agent.harness.hierarchical_planner import _validate_partition_document
        from agent.harness.plan_coverage import segment_user_turn

        clauses = segment_user_turn("Switch to bsc. Use quick.")
        document = {
            "semantic_units": [
                {
                    "unit_id": "unit-1",
                    "clause_id": clauses[0].clause_id,
                    "source_text": "Switch to bsc.",
                    "operation": "domain_request",
                    "owner_routes": [{"owner": "chain_rpc", "group": "chain_identity"}],
                    "reason": "chain mutation",
                },
                {
                    "unit_id": "unit-2",
                    "clause_id": clauses[1].clause_id,
                    "source_text": "Use quick.",
                    "operation": "domain_request",
                    "owner_routes": [{"owner": "performance", "group": "qps_profile"}],
                    "reason": "QPS mode mutation",
                },
            ],
            "reason": "two independent requests",
        }
        units, errors = _validate_partition_document(
            json.dumps(document),
            clauses,
        )

        self.assertFalse(errors)
        self.assertEqual([unit["unit_id"] for unit in units], ["unit-1", "unit-2"])

    def test_stage_a_canonicalizes_numeric_model_ids_before_owner_routing(
        self,
    ) -> None:
        from agent.harness.hierarchical_planner import (
            _owner_requests,
            _validate_partition_document,
        )
        from agent.harness.plan_coverage import segment_user_turn

        clauses = segment_user_turn("Use quick.")
        document = {
            "semantic_units": [{
                "unit_id": 1,
                "clause_id": "clause-1",
                "source_text": "Use quick.",
                "operation": "domain_request",
                "owner_routes": [{
                    "owner": "performance",
                    "group": "qps_profile",
                }],
                "reason": "qps",
            }],
        }

        units, errors = _validate_partition_document(
            json.dumps(document),
            clauses,
        )

        self.assertFalse(errors)
        self.assertEqual(units[0]["unit_id"], "1")
        self.assertEqual(_owner_requests(units), {"performance": ("1",)})

    def test_stage_a_rejects_owner_group_mismatch(self) -> None:
        from agent.harness.hierarchical_planner import _validate_partition_document
        from agent.harness.plan_coverage import segment_user_turn

        clauses = segment_user_turn("Use quick.")
        document = {
            "semantic_units": [{
                "unit_id": "unit-1",
                "clause_id": clauses[0].clause_id,
                "source_text": clauses[0].text,
                "operation": "domain_request",
                "owner_routes": [{"owner": "chain_rpc", "group": "qps_profile"}],
                "reason": "wrong owner",
            }],
        }
        _units, errors = _validate_partition_document(
            json.dumps(document),
            clauses,
        )

        self.assertTrue(any("owner/group mismatch" in error for error in errors))

    def test_stage_a_rejects_cross_clause_reordering(self) -> None:
        from agent.harness.hierarchical_planner import _validate_partition_document
        from agent.harness.plan_coverage import segment_user_turn

        clauses = segment_user_turn("Use bsc. Use quick.")
        document = {
            "semantic_units": [
                {
                    "unit_id": "unit-2",
                    "clause_id": clauses[1].clause_id,
                    "source_text": clauses[1].text,
                    "operation": "domain_request",
                    "owner_routes": [
                        {"owner": "performance", "group": "qps_profile"},
                    ],
                    "reason": "qps",
                },
                {
                    "unit_id": "unit-1",
                    "clause_id": clauses[0].clause_id,
                    "source_text": clauses[0].text,
                    "operation": "domain_request",
                    "owner_routes": [
                        {"owner": "chain_rpc", "group": "chain_identity"},
                    ],
                    "reason": "chain",
                },
            ],
            "reason": "reordered",
        }

        _units, errors = _validate_partition_document(
            json.dumps(document),
            clauses,
        )

        self.assertTrue(any("reorders clauses" in error for error in errors))

    def test_stage_a_rejects_duplicate_owner_group_routes(self) -> None:
        from agent.harness.hierarchical_planner import _validate_partition_document
        from agent.harness.plan_coverage import segment_user_turn

        clauses = segment_user_turn("Use quick.")
        document = {
            "semantic_units": [{
                "unit_id": "unit-1",
                "clause_id": clauses[0].clause_id,
                "source_text": clauses[0].text,
                "operation": "domain_request",
                "owner_routes": [
                    {"owner": "performance", "group": "qps_profile"},
                    {"owner": "performance", "group": "qps_profile"},
                ],
                "reason": "duplicated route",
            }],
        }

        _units, errors = _validate_partition_document(
            json.dumps(document),
            clauses,
        )

        self.assertTrue(any("route is duplicated" in error for error in errors))

    def test_internal_route_ids_cannot_collide_with_model_unit_ids(self) -> None:
        from agent.harness.hierarchical_planner import _expand_partition_routes

        partition = [
            {
                "unit_id": "u",
                "owner_routes": [
                    {"owner": "chain_rpc", "group": "target_mode"},
                    {"owner": "chain_rpc", "group": "chain_identity"},
                ],
            },
            {
                "unit_id": "__harness_route_1",
                "owner_routes": [{
                    "owner": "performance",
                    "group": "qps_profile",
                }],
            },
        ]

        expanded = _expand_partition_routes(partition)
        unit_ids = [str(unit["unit_id"]) for unit in expanded]

        self.assertEqual(len(unit_ids), len(set(unit_ids)))
        self.assertEqual(
            unit_ids,
            ["__harness_route_2", "__harness_route_3", "__harness_route_1"],
        )

    def test_stage_b_payload_exposes_only_one_owner_contract(self) -> None:
        from agent.harness.hierarchical_planner import _stage_b_payload

        payload = _stage_b_payload(
            {
                "language": "en",
                "active_group": "qps_profile",
                "pending_question": {},
                "qps_profile": {"mode": "quick"},
                "custom_rpc": {"method": "must-not-leak"},
                "group_states": {},
            },
            "performance",
            frozenset({"qps_profile"}),
            [{
                "unit_id": "unit-1",
                "clause_id": "clause-1",
                "source_text": "Use quick.",
                "operation": "domain_request",
                "owner_routes": [
                    {"owner": "performance", "group": "qps_profile"},
                ],
                "reason": "QPS selection",
            }],
            ("unit-1",),
        )

        self.assertEqual(
            {row["owner"] for row in payload["owner_action_schema"]},
            {"performance"},
        )
        self.assertEqual(
            {row["target_group"] for row in payload["owner_action_schema"]},
            {"qps_profile"},
        )
        self.assertNotIn(
            "set_observability",
            {row["type"] for row in payload["owner_action_schema"]},
        )
        self.assertNotIn("custom_rpc", payload["owner_state"])
        self.assertEqual(payload["owner_state"]["qps_profile"], {"mode": "quick"})
        self.assertEqual(
            {row["name"] for row in payload["owner_group_schema"]},
            {"qps_profile"},
        )

    def test_stage_b_rejects_nested_current_action_envelopes(self) -> None:
        from agent.harness.hierarchical_planner import _validate_owner_document

        document = {
            "actions": [{
                "type": "set_qps_mode",
                "arguments": {"qps_mode": "quick"},
            }],
            "bindings": [{
                "unit_id": "unit-1",
                "action_indexes": [0],
                "disposition": "action",
                "reason": "invalid nested action",
            }],
        }

        _payload, errors = _validate_owner_document(
            json.dumps(document),
            "performance",
            ("unit-1",),
        )

        self.assertTrue(any("arguments.v1 is retired" in error for error in errors))

    def test_stage_b_rejects_selected_value_absent_from_pending_options(
        self,
    ) -> None:
        from agent.harness.hierarchical_planner import _validate_owner_document

        document = {
            "actions": [{
                "type": "answer_pending",
                "selected_value": "[]",
                "source_evidence": '"params":[]',
            }],
            "bindings": [{
                "unit_id": "unit-1",
                "action_indexes": [0],
                "disposition": "action",
                "reason": "mistook a wire literal for an option",
            }],
        }

        _payload, errors = _validate_owner_document(
            json.dumps(document),
            "coordinator",
            ("unit-1",),
            pending_question={
                "id": "schema_evidence",
                "options": [],
                "manual_input_allowed": True,
            },
        )

        self.assertTrue(
            any("absent from the active pending options" in error for error in errors)
        )

    def test_stage_b_rejects_action_outside_the_unit_group(self) -> None:
        from agent.harness.hierarchical_planner import _validate_owner_document

        document = {
            "actions": [{
                "type": "set_observability",
                "observability_mode": "disabled",
                "mutation_explicit": True,
                "source_evidence": "disable observability",
            }],
            "bindings": [{
                "unit_id": "unit-1",
                "action_indexes": [0],
                "disposition": "action",
                "reason": "wrong group",
            }],
        }

        _payload, errors = _validate_owner_document(
            json.dumps(document),
            "performance",
            ("unit-1",),
            expected_groups={"unit-1": frozenset({"qps_profile"})},
            expected_operations={"unit-1": "domain_request"},
        )

        self.assertTrue(any("outside unit route" in error for error in errors))

    def test_stage_b_rejects_targetless_actions_outside_declared_compiler_group(
        self,
    ) -> None:
        from agent.harness.hierarchical_planner import _validate_owner_document

        probes = (
            (
                "execution",
                "preflight_smoke_execution",
                {"type": "approve_final_benchmark"},
            ),
            (
                "chain_rpc",
                "chain_identity",
                {"type": "use_default_workload"},
            ),
            (
                "analysis",
                "report_artifact_analysis",
                {"type": "analyze_evidence"},
            ),
        )
        for owner, group, action in probes:
            with self.subTest(owner=owner, group=group, action=action["type"]):
                document = {
                    "actions": [action],
                    "bindings": [{
                        "unit_id": "unit-1",
                        "action_indexes": [0],
                        "disposition": "action",
                        "reason": "wrong group",
                    }],
                }
                _payload, errors = _validate_owner_document(
                    json.dumps(document),
                    owner,
                    ("unit-1",),
                    expected_groups={"unit-1": frozenset({group})},
                    expected_operations={"unit-1": "domain_request"},
                )

                self.assertTrue(
                    any("outside unit route" in error for error in errors),
                    errors,
                )

    def test_stage_a_admission_rejects_an_omitted_sibling_demand(self) -> None:
        from agent.harness.hierarchical_planner import _review_stage_a_partition

        payload = {
            "user_text": "Use BSC and quick.",
            "clauses": [{
                "clause_id": "clause-1",
                "text": "Use BSC and quick.",
                "input_shape": "prose",
            }],
            "groups": [
                {"name": "chain_identity", "owner": "chain_rpc"},
                {"name": "qps_profile", "owner": "performance"},
            ],
            "universal_operations": ["domain_request"],
        }
        partition = [{
            "unit_id": "unit-1",
            "clause_id": "clause-1",
            "source_text": "Use BSC and quick.",
            "operation": "domain_request",
            "owner_routes": [{"owner": "chain_rpc", "group": "chain_identity"}],
            "reason": "chain only",
        }]
        response = {
            "unit_verdicts": [{
                "unit_id": "unit-1",
                "verdict": "complete",
                "supports_unit_id": "",
                "reason": "chain is represented",
            }],
            "clause_verdicts": [{
                "clause_id": "clause-1",
                "verdict": "omitted",
                "omitted_owner_routes": [{
                    "owner": "performance",
                    "group": "qps_profile",
                }],
                "reason": "quick is missing",
            }],
            "reason": "one sibling demand is omitted",
        }

        with patch(
            "agent.harness.hierarchical_planner.request_semantic_compilation",
            return_value=json.dumps(response),
        ):
            errors, _sizes, _redundant = _review_stage_a_partition(
                object(),
                payload,
                partition,
            )

        self.assertTrue(any("omitted demand" in error for error in errors))

    def test_stage_a_admission_repairs_only_malformed_contract_output(
        self,
    ) -> None:
        from agent.harness.hierarchical_planner import (
            _review_stage_a_partition,
        )

        payload = {
            "user_text": "Keep monitoring disabled.",
            "clauses": [{
                "clause_id": "clause-1",
                "text": "Keep monitoring disabled.",
            }],
            "groups": [{"name": "observability", "owner": "performance"}],
            "universal_operations": ["pending_answer"],
        }
        partition = [{
            "unit_id": "unit-1",
            "clause_id": "clause-1",
            "source_text": "Keep monitoring disabled.",
            "operation": "pending_answer",
            "owner_routes": [{
                "owner": "coordinator",
                "group": "observability",
            }],
            "reason": "selects the disabled option",
        }]
        valid = {
            "unit_verdicts": [{
                "unit_id": "unit-1",
                "verdict": "complete",
                "supports_unit_id": "",
                "reason": "the selection is represented",
            }],
            "clause_verdicts": [{
                "clause_id": "clause-1",
                "verdict": "complete",
                "omitted_owner_routes": [],
                "reason": "the complete demand is represented",
            }],
            "reason": "the partition is complete",
        }

        with patch(
            "agent.harness.hierarchical_planner.request_semantic_compilation",
            side_effect=["not-json", json.dumps(valid)],
        ) as compiler:
            errors, sizes, _redundant = _review_stage_a_partition(
                object(),
                payload,
                partition,
            )

        self.assertEqual(errors, ())
        self.assertEqual(len(sizes), 2)
        self.assertEqual(compiler.call_count, 2)
        repair_payload = compiler.call_args_list[1].kwargs["request_payload"]
        self.assertIn("contract_repair", repair_payload)

    def test_stage_a_admission_does_not_retry_valid_semantic_rejection(
        self,
    ) -> None:
        from agent.harness.hierarchical_planner import (
            _review_stage_a_partition,
        )

        payload = {
            "user_text": "Use BSC and quick.",
            "clauses": [{
                "clause_id": "clause-1",
                "text": "Use BSC and quick.",
            }],
            "groups": [
                {"name": "chain_identity", "owner": "chain_rpc"},
                {"name": "qps_profile", "owner": "performance"},
            ],
            "universal_operations": ["domain_request"],
        }
        partition = [{
            "unit_id": "unit-1",
            "clause_id": "clause-1",
            "source_text": "Use BSC and quick.",
            "operation": "domain_request",
            "owner_routes": [{
                "owner": "chain_rpc",
                "group": "chain_identity",
            }],
            "reason": "chain only",
        }]
        rejected = {
            "unit_verdicts": [{
                "unit_id": "unit-1",
                "verdict": "complete",
                "supports_unit_id": "",
                "reason": "chain is represented",
            }],
            "clause_verdicts": [{
                "clause_id": "clause-1",
                "verdict": "omitted",
                "omitted_owner_routes": [{
                    "owner": "performance",
                    "group": "qps_profile",
                }],
                "reason": "quick is missing",
            }],
            "reason": "one sibling demand is omitted",
        }

        with patch(
            "agent.harness.hierarchical_planner.request_semantic_compilation",
            return_value=json.dumps(rejected),
        ) as compiler:
            errors, sizes, _redundant = _review_stage_a_partition(
                object(),
                payload,
                partition,
            )

        self.assertTrue(any("omitted demand" in error for error in errors))
        self.assertEqual(len(sizes), 1)
        self.assertEqual(compiler.call_count, 1)

    def test_stage_a_context_unit_defers_omission_authority_to_clause(self) -> None:
        from agent.harness.hierarchical_planner import (
            _review_stage_a_partition,
        )

        payload = {
            "user_text": "Proceed, using the settings already reviewed.",
            "clauses": [{
                "clause_id": "clause-1",
                "text": "Proceed, using the settings already reviewed.",
            }],
            "groups": [{
                "name": "preflight_smoke_execution",
                "owner": "execution",
            }],
            "universal_operations": ["pending_answer", "context"],
        }
        partition = [
            {
                "unit_id": "unit-1",
                "clause_id": "clause-1",
                "source_text": "Proceed",
                "operation": "pending_answer",
                "owner_routes": [{
                    "owner": "coordinator",
                    "group": "preflight_smoke_execution",
                }],
                "reason": "selects one option",
            },
            {
                "unit_id": "unit-2",
                "clause_id": "clause-1",
                "source_text": "using the settings already reviewed.",
                "operation": "context",
                "owner_routes": [],
                "reason": "restates the selected option scope",
            },
        ]
        reviewed = {
            "unit_verdicts": [
                {
                    "unit_id": "unit-1",
                    "verdict": "complete",
                    "supports_unit_id": "",
                    "reason": "the selection is represented",
                },
                {
                    "unit_id": "unit-2",
                    "verdict": "unresolved",
                    "supports_unit_id": "",
                    "reason": "the context has no action owner",
                },
            ],
            "clause_verdicts": [{
                "clause_id": "clause-1",
                "verdict": "complete",
                "omitted_owner_routes": [],
                "reason": "the clause contains no omitted demand",
            }],
            "reason": "complete",
        }

        with patch(
            "agent.harness.hierarchical_planner.request_semantic_compilation",
            return_value=json.dumps(reviewed),
        ) as compiler:
            errors, sizes, _redundant = _review_stage_a_partition(
                object(),
                payload,
                partition,
            )

        self.assertEqual(errors, ())
        self.assertEqual(len(sizes), 1)
        self.assertEqual(compiler.call_count, 1)

    def test_stage_a_admission_identifies_redundant_sibling_unit(self) -> None:
        from agent.harness.hierarchical_planner import (
            _review_stage_a_partition,
        )

        payload = {
            "user_text": "Proceed and perform the declared effect.",
            "clauses": [{
                "clause_id": "clause-1",
                "text": "Proceed",
            }, {
                "clause_id": "clause-2",
                "text": "perform the declared effect.",
            }],
            "groups": [{
                "name": "preflight_smoke_execution",
                "owner": "execution",
            }],
            "universal_operations": ["pending_answer", "domain_request"],
            "pending_question": {
                "id": "generic_specialized_choice",
                "options": [{
                    "value": True,
                    "completion_effect": "Perform the declared effect.",
                    "action": {"type": "approve_preflight_smoke"},
                }],
            },
        }
        partition = [
            {
                "unit_id": "unit-1",
                "clause_id": "clause-1",
                "source_text": "Proceed",
                "operation": "pending_answer",
                "owner_routes": [{
                    "owner": "coordinator",
                    "group": "preflight_smoke_execution",
                }],
                "reason": "selects one option",
            },
            {
                "unit_id": "unit-2",
                "clause_id": "clause-2",
                "source_text": "perform the declared effect.",
                "operation": "domain_request",
                "owner_routes": [{
                    "owner": "execution",
                    "group": "preflight_smoke_execution",
                }],
                "reason": "duplicates the selected option effect",
            },
        ]
        reviewed = {
            "unit_verdicts": [
                {
                    "unit_id": "unit-1",
                    "verdict": "complete",
                    "supports_unit_id": "",
                    "reason": "the selection owns the demand",
                },
                {
                    "unit_id": "unit-2",
                    "verdict": "redundant",
                    "supports_unit_id": "unit-1",
                    "reason": "this only restates the selected option effect",
                },
            ],
            "clause_verdicts": [
                {
                    "clause_id": "clause-1",
                    "verdict": "complete",
                    "omitted_owner_routes": [],
                    "reason": "the selection represents the complete demand",
                },
                {
                    "clause_id": "clause-2",
                    "verdict": "complete",
                    "omitted_owner_routes": [],
                    "reason": "the later clause only restates the effect",
                },
            ],
            "reason": "one redundant unit",
        }

        with patch(
            "agent.harness.hierarchical_planner.request_semantic_compilation",
            return_value=json.dumps(reviewed),
        ) as compiler:
            errors, _sizes, redundant = _review_stage_a_partition(
                object(),
                payload,
                partition,
            )

        self.assertEqual(errors, ())
        self.assertEqual(redundant, frozenset({"unit-2"}))
        self.assertEqual(
            compiler.call_args.kwargs["request_payload"]["pending_question"],
            payload["pending_question"],
        )

    def test_stage_a_admission_can_bind_manual_value_scope_to_candidate(
        self,
    ) -> None:
        from agent.harness.hierarchical_planner import (
            _review_stage_a_partition,
        )

        payload = {
            "user_text": (
                "Use only this value for validation:\n"
                "https://example.invalid/rpc\n"
                "Do not apply it to the final target."
            ),
            "clauses": [
                {"clause_id": "clause-1", "text": "Use only this value for validation:"},
                {"clause_id": "clause-2", "text": "https://example.invalid/rpc"},
                {
                    "clause_id": "clause-3",
                    "text": "Do not apply it to the final target.",
                },
            ],
            "groups": [{
                "name": "endpoint_process",
                "owner": "chain_rpc",
            }],
            "universal_operations": ["pending_answer", "domain_request"],
            "pending_question": {
                "id": "generic_manual_value",
                "group": "endpoint_process",
                "manual_input_allowed": True,
                "options": [],
            },
            "pending_typed_candidates": ["https://example.invalid/rpc"],
        }
        partition = [
            {
                "unit_id": "unit-1",
                "clause_id": "clause-1",
                "source_text": "Use only this value for validation:",
                "operation": "domain_request",
                "owner_routes": [{
                    "owner": "chain_rpc",
                    "group": "endpoint_process",
                }],
                "reason": "scope wrapper",
            },
            {
                "unit_id": "unit-2",
                "clause_id": "clause-2",
                "source_text": "https://example.invalid/rpc",
                "operation": "pending_answer",
                "owner_routes": [{
                    "owner": "coordinator",
                    "group": "endpoint_process",
                }],
                "reason": "typed candidate",
            },
            {
                "unit_id": "unit-3",
                "clause_id": "clause-3",
                "source_text": "Do not apply it to the final target.",
                "operation": "domain_request",
                "owner_routes": [{
                    "owner": "chain_rpc",
                    "group": "endpoint_process",
                }],
                "reason": "exclusion wrapper",
            },
        ]
        reviewed = {
            "unit_verdicts": [
                {
                    "unit_id": "unit-1",
                    "verdict": "redundant",
                    "supports_unit_id": "unit-2",
                    "reason": "purpose scope for the manual candidate",
                },
                {
                    "unit_id": "unit-2",
                    "verdict": "complete",
                    "supports_unit_id": "",
                    "reason": "the typed candidate owns the transaction",
                },
                {
                    "unit_id": "unit-3",
                    "verdict": "redundant",
                    "supports_unit_id": "unit-2",
                    "reason": "application exclusion for the candidate",
                },
            ],
            "clause_verdicts": [
                {
                    "clause_id": f"clause-{index}",
                    "verdict": "complete",
                    "omitted_owner_routes": [],
                    "reason": "fully represented by the one transaction",
                }
                for index in range(1, 4)
            ],
            "reason": "one manual-value transaction with scope context",
        }

        with patch(
            "agent.harness.hierarchical_planner.request_semantic_compilation",
            return_value=json.dumps(reviewed),
        ) as compiler:
            errors, _sizes, redundant = _review_stage_a_partition(
                object(),
                payload,
                partition,
            )

        self.assertEqual(errors, ())
        self.assertEqual(redundant, frozenset({"unit-1", "unit-3"}))
        self.assertEqual(
            compiler.call_args.kwargs["request_payload"][
                "pending_typed_candidates"
            ],
            ["https://example.invalid/rpc"],
        )

    def test_stage_a_admission_allows_contrast_before_selected_unit(self) -> None:
        from agent.harness.hierarchical_planner import (
            _review_stage_a_partition,
        )

        payload = {
            "user_text": "Do not keep defaults. Change the target instead.",
            "clauses": [
                {"clause_id": "clause-1", "text": "Do not keep defaults."},
                {"clause_id": "clause-2", "text": "Change the target instead."},
            ],
            "groups": [{
                "name": "workload_rpc",
                "owner": "chain_rpc",
            }],
            "universal_operations": ["pending_answer", "context"],
            "pending_question": {
                "id": "generic_choice",
                "options": [
                    {"value": "default", "action": {"type": "use_default_workload"}},
                    {"value": "change", "action": {"type": "request_target_change"}},
                ],
            },
        }
        partition = [
            {
                "unit_id": "unit-1",
                "clause_id": "clause-1",
                "source_text": "Do not keep defaults.",
                "operation": "context",
                "owner_routes": [],
                "reason": "contrast for the selected option",
            },
            {
                "unit_id": "unit-2",
                "clause_id": "clause-2",
                "source_text": "Change the target instead.",
                "operation": "pending_answer",
                "owner_routes": [{
                    "owner": "coordinator",
                    "group": "workload_rpc",
                }],
                "reason": "selects one declared option",
            },
        ]
        reviewed = {
            "unit_verdicts": [
                {
                    "unit_id": "unit-1",
                    "verdict": "redundant",
                    "supports_unit_id": "unit-2",
                    "reason": "contrastive evidence for the later selection",
                },
                {
                    "unit_id": "unit-2",
                    "verdict": "complete",
                    "supports_unit_id": "",
                    "reason": "the selected option owns the demand",
                },
            ],
            "clause_verdicts": [
                {
                    "clause_id": "clause-1",
                    "verdict": "complete",
                    "omitted_owner_routes": [],
                    "reason": "contains contrast only",
                },
                {
                    "clause_id": "clause-2",
                    "verdict": "complete",
                    "omitted_owner_routes": [],
                    "reason": "contains the complete selection",
                },
            ],
            "reason": "one contrastive redundant unit",
        }

        with patch(
            "agent.harness.hierarchical_planner.request_semantic_compilation",
            return_value=json.dumps(reviewed),
        ):
            errors, _sizes, redundant = _review_stage_a_partition(
                object(),
                payload,
                partition,
            )

        self.assertEqual(errors, ())
        self.assertEqual(redundant, frozenset({"unit-1"}))

    def test_stage_b_repairs_specialized_option_schema_once(self) -> None:
        from agent.harness.hierarchical_planner import _compile_owner_document
        from agent.harness.state import new_state

        state = new_state("stage-b-structural-repair", language="en")
        state["pending_question"] = {
            "id": "generic_specialized_choice",
            "group": "preflight_smoke_execution",
            "kind": "yes_no",
            "prompt": "Proceed?",
            "field": "generic_specialized_choice",
            "manual_input_allowed": False,
            "options": [{
                "id": "proceed",
                "label": "Proceed",
                "value": "proceed",
                "action": {"type": "approve_preflight_smoke"},
            }],
            "accepted_action_types": ["approve_preflight_smoke"],
        }
        partition = [{
            "unit_id": "unit-1",
            "clause_id": "clause-1",
            "source_text": "Proceed with it.",
            "operation": "domain_request",
            "owner_routes": [{
                "owner": "execution",
                "group": "preflight_smoke_execution",
            }],
            "reason": "selects the declared specialized action",
        }]
        malformed = {
            "actions": [{
                "type": "approve_preflight_smoke",
                "source_evidence": "Proceed with it.",
            }],
            "bindings": [{
                "unit_id": "unit-1",
                "action_indexes": [0],
                "disposition": "action",
                "reason": "selected",
            }],
            "reason": "compiled",
        }
        repaired = {
            "actions": [{"type": "approve_preflight_smoke"}],
            "bindings": [{
                "unit_id": "unit-1",
                "action_indexes": [0],
                "disposition": "action",
                "reason": "selected",
            }],
            "reason": "repaired",
        }

        with (
            patch(
                "agent.harness.hierarchical_planner.provider_from_config",
                return_value=object(),
            ),
            patch(
                "agent.harness.hierarchical_planner.request_semantic_compilation",
                side_effect=[json.dumps(malformed), json.dumps(repaired)],
            ) as compiler,
        ):
            document, errors, sizes = _compile_owner_document(
                state,
                "execution",
                frozenset({"preflight_smoke_execution"}),
                partition,
                ("unit-1",),
            )

        self.assertEqual(errors, ())
        self.assertEqual(
            document["actions"],
            [{"type": "approve_preflight_smoke"}],
        )
        self.assertEqual(len(sizes), 2)
        self.assertEqual(compiler.call_count, 2)
        repair_payload = compiler.call_args_list[1].kwargs["request_payload"]
        self.assertIn("contract_repair", repair_payload)
        self.assertIn(
            "undeclared arguments",
            " ".join(repair_payload["contract_repair"]["validator_errors"]),
        )

    def test_stage_b_repairs_conflicting_pending_answer_representation_once(
        self,
    ) -> None:
        from agent.harness.hierarchical_planner import _compile_owner_document
        from agent.harness.state import new_state

        state = new_state("stage-b-pending-representation-repair", language="en")
        state["pending_question"] = {
            "id": "observability_mode",
            "group": "observability",
            "kind": "numbered_choice",
            "prompt": "Choose observability.",
            "field": "observability_mode",
            "manual_input_allowed": False,
            "options": [{
                "id": "disabled",
                "label": "Disabled",
                "value": "disabled",
                "action": {
                    "type": "set_observability",
                    "observability_mode": "disabled",
                },
            }],
            "accepted_action_types": ["answer_pending"],
        }
        partition = [{
            "unit_id": "unit-1",
            "clause_id": "clause-1",
            "source_text": "Disable observability for this run.",
            "operation": "pending_answer",
            "owner_routes": [{"owner": "coordinator", "group": ""}],
            "reason": "selects one declared option",
        }]
        conflicting = {
            "actions": [{
                "type": "answer_pending",
                "answer": "Disabled",
                "selected_value": "disabled",
                "source_evidence": "Disable observability for this run.",
            }],
            "bindings": [{
                "unit_id": "unit-1",
                "action_indexes": [0],
                "disposition": "action",
                "reason": "selected",
            }],
            "reason": "compiled",
        }
        repaired = {
            "actions": [{
                "type": "answer_pending",
                "selected_value": "disabled",
                "source_evidence": "Disable observability for this run.",
            }],
            "bindings": [{
                "unit_id": "unit-1",
                "action_indexes": [0],
                "disposition": "action",
                "reason": "selected",
            }],
            "reason": "repaired",
        }

        with (
            patch(
                "agent.harness.hierarchical_planner.provider_from_config",
                return_value=object(),
            ),
            patch(
                "agent.harness.hierarchical_planner.request_semantic_compilation",
                side_effect=[json.dumps(conflicting), json.dumps(repaired)],
            ) as compiler,
        ):
            document, errors, sizes = _compile_owner_document(
                state,
                "coordinator",
                frozenset(),
                partition,
                ("unit-1",),
            )

        self.assertEqual(errors, ())
        self.assertEqual(document["actions"], repaired["actions"])
        self.assertEqual(len(sizes), 2)
        self.assertEqual(compiler.call_count, 2)
        repair = compiler.call_args_list[1].kwargs["request_payload"][
            "contract_repair"
        ]
        self.assertIn(
            "conflicting answer and selected_value representations",
            " ".join(repair["validator_errors"]),
        )

    def test_stage_b_does_not_retry_valid_unresolved_binding(self) -> None:
        from agent.harness.hierarchical_planner import _compile_owner_document
        from agent.harness.state import new_state

        state = new_state("stage-b-semantic-unresolved", language="en")
        partition = [{
            "unit_id": "unit-1",
            "clause_id": "clause-1",
            "source_text": "The request is ambiguous.",
            "operation": "domain_request",
            "owner_routes": [{
                "owner": "execution",
                "group": "preflight_smoke_execution",
            }],
            "reason": "ambiguous selection",
        }]
        unresolved = {
            "actions": [],
            "bindings": [{
                "unit_id": "unit-1",
                "action_indexes": [],
                "disposition": "unresolved",
                "reason": "more information is required",
            }],
            "reason": "unresolved",
        }

        with (
            patch(
                "agent.harness.hierarchical_planner.provider_from_config",
                return_value=object(),
            ),
            patch(
                "agent.harness.hierarchical_planner.request_semantic_compilation",
                return_value=json.dumps(unresolved),
            ) as compiler,
        ):
            document, errors, sizes = _compile_owner_document(
                state,
                "execution",
                frozenset({"preflight_smoke_execution"}),
                partition,
                ("unit-1",),
            )

        self.assertEqual(errors, ())
        self.assertEqual(document["actions"], [])
        self.assertEqual(len(sizes), 1)
        self.assertEqual(compiler.call_count, 1)

    def test_merge_preserves_semantic_order_across_owners(self) -> None:
        from agent.harness.hierarchical_planner import _merge_owner_documents

        partition = [
            {
                "unit_id": "unit-1",
                "clause_id": "clause-1",
                "source_text": "Use bsc.",
                "operation": "domain_request",
                "owner_routes": [{"owner": "chain_rpc", "group": "chain_identity"}],
                "reason": "chain",
            },
            {
                "unit_id": "unit-2",
                "clause_id": "clause-2",
                "source_text": "Use quick.",
                "operation": "domain_request",
                "owner_routes": [{"owner": "performance", "group": "qps_profile"}],
                "reason": "qps",
            },
        ]
        documents = {
            "performance": {
                "actions": [{"type": "set_qps_mode", "qps_mode": "quick"}],
                "bindings": [{
                    "unit_id": "unit-2",
                    "action_indexes": [0],
                    "disposition": "action",
                    "reason": "compiled",
                }],
            },
            "chain_rpc": {
                "actions": [{"type": "choose_chain", "chain": "bsc"}],
                "bindings": [{
                    "unit_id": "unit-1",
                    "action_indexes": [0],
                    "disposition": "action",
                    "reason": "compiled",
                }],
            },
        }

        merged = _merge_owner_documents(partition, partition, documents)

        self.assertEqual(
            [action["type"] for action in merged["actions"]],
            ["choose_chain", "set_qps_mode"],
        )
        self.assertEqual(
            [unit["action_indexes"] for unit in merged["semantic_units"]],
            [[0], [1]],
        )

    def test_multi_route_compilation_merges_back_to_one_source_unit(self) -> None:
        from agent.harness.hierarchical_planner import (
            _expand_partition_routes,
            _merge_owner_documents,
        )
        from agent.harness.plan_coverage import (
            segment_user_turn,
            validate_plan_coverage,
        )

        text = "Use BSC with quick."
        source_partition = [{
            "unit_id": "unit-1",
            "clause_id": "clause-1",
            "source_text": text,
            "operation": "domain_request",
            "owner_routes": [
                {"owner": "chain_rpc", "group": "chain_identity"},
                {"owner": "performance", "group": "qps_profile"},
            ],
            "reason": "one source phrase supplies two independent settings",
        }]
        routed_partition = _expand_partition_routes(source_partition)
        documents = {
            "chain_rpc": {
                "actions": [{
                    "type": "choose_chain",
                    "chain_text": "BSC",
                    "source_evidence": "BSC",
                }],
                "bindings": [{
                    "unit_id": "__harness_route_1",
                    "action_indexes": [0],
                    "disposition": "action",
                    "reason": "chain",
                }],
            },
            "performance": {
                "actions": [{
                    "type": "set_qps_mode",
                    "qps_mode": "quick",
                    "mutation_explicit": True,
                    "source_evidence": "quick",
                }],
                "bindings": [{
                    "unit_id": "__harness_route_2",
                    "action_indexes": [0],
                    "disposition": "action",
                    "reason": "qps",
                }],
            },
        }

        merged = _merge_owner_documents(
            source_partition,
            routed_partition,
            documents,
        )
        coverage = validate_plan_coverage(merged, segment_user_turn(text))

        self.assertTrue(coverage.valid, coverage.errors)
        self.assertEqual(len(merged["semantic_units"]), 1)
        self.assertEqual(merged["semantic_units"][0]["action_indexes"], [0, 1])

    def test_pending_answer_and_independent_consultation_remain_separate(self) -> None:
        from agent.harness.hierarchical_planner import _merge_owner_documents

        partition = [
            {
                "unit_id": "unit-1",
                "clause_id": "clause-1",
                "source_text": "Y.",
                "operation": "pending_answer",
                "owner_routes": [{"owner": "coordinator", "group": ""}],
                "reason": "answer",
            },
            {
                "unit_id": "unit-2",
                "clause_id": "clause-2",
                "source_text": "What happens next?",
                "operation": "consultation",
                "owner_routes": [{"owner": "orientation", "group": "opening"}],
                "reason": "independent question",
            },
        ]
        documents = {
            "coordinator": {
                "actions": [{"type": "answer_pending", "selected_value": "yes"}],
                "bindings": [{
                    "unit_id": "unit-1",
                    "action_indexes": [0],
                    "disposition": "action",
                    "reason": "compiled",
                }],
            },
            "orientation": {
                "actions": [{"type": "answer_opening_question", "topic": "next_action"}],
                "bindings": [{
                    "unit_id": "unit-2",
                    "action_indexes": [0],
                    "disposition": "action",
                    "reason": "compiled",
                }],
            },
        }

        merged = _merge_owner_documents(partition, partition, documents)

        self.assertEqual(len(merged["actions"]), 2)
        self.assertEqual(
            [unit["action_indexes"] for unit in merged["semantic_units"]],
            [[0], [1]],
        )

    def test_consultation_cannot_claim_a_pending_option_by_action_type(self) -> None:
        from agent.harness.domains.orientation import opening_question
        from agent.harness.semantic_admission import (
            prepare_hierarchical_candidate,
        )
        from agent.harness.plan_coverage import segment_user_turn
        from agent.harness.state import new_state

        text = "Explain the supported capabilities."
        clauses = segment_user_turn(text)
        state = new_state("consultation-pending", language="en")
        state["pending_question"] = opening_question(state)
        candidate = {
            "actions": [{
                "type": "answer_opening_question",
                "topic": "capabilities",
                "source_evidence": text,
            }],
            "semantic_units": [{
                "unit_id": "unit-1",
                "clause_id": "clause-1",
                "source_text": text,
                "disposition": "action",
                "action_indexes": [0],
                "reason": "independent consultation",
            }],
        }

        prepared, validation = prepare_hierarchical_candidate(
            json.dumps(candidate),
            state,
            clauses,
            pending_choice_unit_ids=frozenset(),
        )

        self.assertTrue(validation.valid, validation.errors)
        payload = json.loads(prepared)
        self.assertEqual(payload["actions"][0]["type"], "answer_opening_question")
        self.assertNotIn("pending_choice_contracts", payload)
        self.assertNotIn("pending_answer_admissions", payload)

    def test_domain_request_cannot_claim_a_pending_option(self) -> None:
        from agent.harness.domains.orientation import opening_question
        from agent.harness.semantic_admission import (
            prepare_hierarchical_candidate,
        )
        from agent.harness.plan_coverage import segment_user_turn
        from agent.harness.state import new_state

        text = "Use fake-node as an additional configuration request."
        clauses = segment_user_turn(text)
        state = new_state("domain-sibling-pending", language="en")
        state["pending_question"] = opening_question(state)
        candidate = {
            "actions": [{
                "type": "choose_target_mode",
                "target_mode": "fake-node",
                "target_mode_explicit": True,
                "source_evidence": "fake-node",
            }],
            "semantic_units": [{
                "unit_id": "unit-1",
                "clause_id": "clause-1",
                "source_text": text,
                "disposition": "action",
                "action_indexes": [0],
                "reason": "independent domain request",
            }],
        }

        prepared, validation = prepare_hierarchical_candidate(
            json.dumps(candidate),
            state,
            clauses,
            pending_choice_unit_ids=frozenset(),
        )

        self.assertTrue(validation.valid, validation.errors)
        payload = json.loads(prepared)
        self.assertEqual(payload["actions"][0]["type"], "choose_target_mode")
        self.assertNotIn("pending_choice_contracts", payload)
        self.assertNotIn("pending_answer_admissions", payload)

    def test_product_resolver_runs_partition_then_owner_compilers(self) -> None:
        from tests.agent_live.graph_turn import resolve_product_action_queue_for_test as resolve_product_action_queue
        from agent.harness.plan_coverage import PlanCoverageResult

        text = "Switch to bsc. Use quick."
        stage_a = {
            "semantic_units": [
                {
                    "unit_id": "unit-1",
                    "clause_id": "clause-1",
                    "source_text": "Switch to bsc.",
                    "operation": "domain_request",
                    "owner_routes": [{"owner": "chain_rpc", "group": "chain_identity"}],
                    "reason": "chain",
                },
                {
                    "unit_id": "unit-2",
                    "clause_id": "clause-2",
                    "source_text": "Use quick.",
                    "operation": "domain_request",
                    "owner_routes": [{"owner": "performance", "group": "qps_profile"}],
                    "reason": "qps",
                },
            ]
        }
        chain = {
            "actions": [{
                "type": "choose_chain",
                "chain_text": "bsc",
                "source_evidence": "bsc",
            }],
            "bindings": [{
                "unit_id": "unit-1",
                "action_indexes": [0],
                "disposition": "action",
                "reason": "compiled",
            }],
        }
        performance = {
            "actions": [{
                "type": "set_qps_mode",
                "qps_mode": "quick",
                "mutation_explicit": True,
                "source_evidence": "quick",
            }],
            "bindings": [{
                "unit_id": "unit-2",
                "action_indexes": [0],
                "disposition": "action",
                "reason": "compiled",
            }],
        }
        captured: dict[str, object] = {}

        def prepare(candidate, _state, clauses, **_kwargs):
            captured["candidate"] = json.loads(candidate)
            return candidate, PlanCoverageResult(True, (), ())

        def review(_provider, candidate, _validation, _state, _clauses, **kwargs):
            captured["allowed"] = kwargs["allowed_action_types"]
            captured["review_candidate"] = json.loads(candidate)
            return SimpleNamespace(request_json="{}"), SimpleNamespace(valid=True, errors=()), ()

        with (
            patch(
                "agent.harness.hierarchical_planner.provider_from_config",
                return_value=object(),
            ),
            patch(
                "agent.harness.hierarchical_planner.request_semantic_compilation",
                side_effect=[
                    json.dumps(stage_a),
                    json.dumps(chain),
                    json.dumps(performance),
                ],
            ) as compiler,
            patch(
                "agent.harness.hierarchical_planner.prepare_hierarchical_candidate",
                side_effect=prepare,
            ),
            patch(
                "agent.harness.hierarchical_planner._review_bounded_semantic_candidate",
                side_effect=review,
            ),
            patch(
                "agent.harness.hierarchical_planner._admitted_action_queue",
                return_value={"actions": captured},
            ),
        ):
            result = resolve_product_action_queue({}, text)

        self.assertEqual(compiler.call_count, 3)
        candidate = captured["candidate"]
        self.assertEqual(
            [action["type"] for action in candidate["actions"]],
            ["choose_chain", "set_qps_mode"],
        )
        self.assertEqual(
            captured["allowed"],
            frozenset({
                "change_chain",
                "choose_adapter_family",
                "choose_chain",
                "request_chain_selection",
                "request_qps_customization",
                "secondary_handoff_command",
                "set_qps_mode",
                "set_qps_override",
            }),
        )
        self.assertEqual(result["planner_metrics"]["stage_a_calls"], 1)
        self.assertEqual(result["planner_metrics"]["stage_b_calls"], 2)
        self.assertEqual(result["planner_metrics"]["admission_calls"], 1)

    def test_parallel_owner_compilers_inherit_one_absolute_turn_deadline(
        self,
    ) -> None:
        from tests.agent_live.graph_turn import resolve_product_action_queue_for_test as resolve_product_action_queue
        from agent.harness.plan_coverage import PlanCoverageResult
        from agent.llm.types import llm_turn_scope, remaining_turn_seconds

        text = "Switch to bsc. Use quick."
        stage_a = {
            "semantic_units": [
                {
                    "unit_id": "unit-1",
                    "clause_id": "clause-1",
                    "source_text": "Switch to bsc.",
                    "operation": "domain_request",
                    "owner_routes": [{
                        "owner": "chain_rpc",
                        "group": "chain_identity",
                    }],
                    "reason": "chain",
                },
                {
                    "unit_id": "unit-2",
                    "clause_id": "clause-2",
                    "source_text": "Use quick.",
                    "operation": "domain_request",
                    "owner_routes": [{
                        "owner": "performance",
                        "group": "qps_profile",
                    }],
                    "reason": "qps",
                },
            ]
        }
        documents = {
            "chain_rpc": {
                "actions": [{
                    "type": "choose_chain",
                    "chain_text": "bsc",
                    "source_evidence": "bsc",
                }],
                "bindings": [{
                    "unit_id": "unit-1",
                    "action_indexes": [0],
                    "disposition": "action",
                    "reason": "compiled",
                }],
            },
            "performance": {
                "actions": [{
                    "type": "set_qps_mode",
                    "qps_mode": "quick",
                    "mutation_explicit": True,
                    "source_evidence": "quick",
                }],
                "bindings": [{
                    "unit_id": "unit-2",
                    "action_indexes": [0],
                    "disposition": "action",
                    "reason": "compiled",
                }],
            },
        }
        observed: list[float] = []
        observed_lock = threading.Lock()

        def compile_owner(
            _state,
            owner,
            _groups,
            _partition,
            _unit_ids,
        ):
            remaining = remaining_turn_seconds()
            with observed_lock:
                observed.append(remaining)
            return documents[owner], (), (100,)

        with (
            patch(
                "agent.harness.hierarchical_planner.provider_from_config",
                return_value=object(),
            ),
            patch(
                "agent.harness.hierarchical_planner.request_semantic_compilation",
                return_value=json.dumps(stage_a),
            ),
            patch(
                "agent.harness.hierarchical_planner._compile_owner_document",
                side_effect=compile_owner,
            ),
            patch(
                "agent.harness.hierarchical_planner.prepare_hierarchical_candidate",
                side_effect=lambda candidate, *_args, **_kwargs: (
                    candidate,
                    PlanCoverageResult(True, (), ()),
                ),
            ),
            patch(
                "agent.harness.hierarchical_planner._review_bounded_semantic_candidate",
                return_value=(
                    SimpleNamespace(request_json="{}"),
                    SimpleNamespace(valid=True, errors=()),
                    (),
                ),
            ),
            patch(
                "agent.harness.hierarchical_planner._admitted_action_queue",
                return_value={"actions": [{"type": "compiled"}]},
            ),
        ):
            with llm_turn_scope(0.5):
                time.sleep(0.04)
                result = resolve_product_action_queue({}, text)

        self.assertEqual(result["actions"], [{"type": "compiled"}])
        self.assertEqual(len(observed), 2)
        self.assertTrue(all(0 < remaining < 0.48 for remaining in observed))
        self.assertLess(max(observed) - min(observed), 0.05)

    def test_provider_failure_is_not_converted_to_user_clarification(self) -> None:
        from tests.agent_live.graph_turn import resolve_product_action_queue_for_test as resolve_product_action_queue
        from agent.llm.types import LLMProviderError

        provider_error = LLMProviderError(
            "unsupported model",
            provider="deepseek",
            model="invalid-model",
            category="configuration",
            status_code=400,
        )
        with patch(
            "agent.harness.hierarchical_planner.provider_from_config",
            side_effect=provider_error,
        ), self.assertRaises(LLMProviderError) as raised:
            resolve_product_action_queue({}, "Use the first option because it is safer.")

        self.assertIs(raised.exception, provider_error)

    def test_parallel_owner_response_failure_is_not_converted_to_clarification(
        self,
    ) -> None:
        from tests.agent_live.graph_turn import resolve_product_action_queue_for_test as resolve_product_action_queue
        from agent.llm.types import LLMProviderError, llm_turn_scope

        stage_a = {
            "semantic_units": [{
                "unit_id": "unit-1",
                "clause_id": "clause-1",
                "source_text": "Switch to bsc.",
                "operation": "domain_request",
                "owner_routes": [{
                    "owner": "chain_rpc",
                    "group": "chain_identity",
                }],
                "reason": "chain",
            }]
        }
        provider_error = LLMProviderError(
            "malformed successful response",
            provider="deepseek",
            model="deepseek-chat",
            category="response",
            stage="provider_response",
        )
        with (
            patch(
                "agent.harness.hierarchical_planner.provider_from_config",
                return_value=object(),
            ),
            patch(
                "agent.harness.hierarchical_planner.request_semantic_compilation",
                return_value=json.dumps(stage_a),
            ),
            patch(
                "agent.harness.hierarchical_planner._compile_owner_document",
                side_effect=provider_error,
            ),
            llm_turn_scope(1),
            self.assertRaises(LLMProviderError) as raised,
        ):
            resolve_product_action_queue({}, "Switch to bsc.")

        self.assertIs(raised.exception, provider_error)

    def test_coordinator_product_path_uses_hierarchical_resolver(self) -> None:
        import agent.harness.coordinator as coordinator
        from agent.harness import hierarchical_planner

        self.assertFalse(hasattr(coordinator, "resolve_action_queue"))
        self.assertIs(
            coordinator.hierarchical_planner,
            hierarchical_planner,
        )


if __name__ == "__main__":
    unittest.main()

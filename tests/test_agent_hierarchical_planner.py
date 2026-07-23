from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch


class HierarchicalPlannerContractTest(unittest.TestCase):
    def test_product_resolver_end_to_end_contract_without_internal_boundary_mocks(
        self,
    ) -> None:
        from agent.harness.hierarchical_planner import resolve_product_action_queue
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
        self.assertEqual(result["planner_metrics"]["model_calls"], 5)

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
            errors, _size = _review_stage_a_partition(
                object(),
                payload,
                partition,
            )

        self.assertTrue(any("omitted demand" in error for error in errors))

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
        from agent.harness.intent import prepare_hierarchical_candidate
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
        from agent.harness.intent import prepare_hierarchical_candidate
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
        from agent.harness.hierarchical_planner import resolve_product_action_queue
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
                "agent.harness.hierarchical_planner._review_stage_a_partition",
                return_value=((), 100),
            ),
            patch(
                "agent.harness.hierarchical_planner.prepare_hierarchical_candidate",
                side_effect=prepare,
            ),
            patch(
                "agent.harness.hierarchical_planner._review_bounded_semantic_candidate",
                side_effect=review,
            ),
            patch(
                "agent.harness.hierarchical_planner._admitted_plan_requires_pending_contract_adjudication",
                return_value=False,
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
        from agent.harness.action_registry import ACTION_BY_TYPE

        self.assertEqual(captured["allowed"], frozenset(ACTION_BY_TYPE))
        self.assertEqual(result["planner_metrics"]["stage_a_calls"], 2)
        self.assertEqual(result["planner_metrics"]["stage_b_calls"], 2)
        self.assertEqual(result["planner_metrics"]["admission_calls"], 1)

    def test_coordinator_product_path_uses_hierarchical_resolver(self) -> None:
        import agent.harness.coordinator as coordinator
        from agent.harness.hierarchical_planner import resolve_product_action_queue

        self.assertIs(coordinator.resolve_action_queue, resolve_product_action_queue)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import threading
import time
import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch


def _closed_enum_review_response(payload: dict) -> dict:
    verdicts = []
    for grounding in payload["groundings"]:
        selected = str(grounding["selected_value"])
        sources = [
            str(source)
            for unit in grounding["source_units"]
            for source in unit["evidence_sources"]
        ]
        quote = next(
            (
                source[
                    source.casefold().index(selected.casefold()):
                    source.casefold().index(selected.casefold()) + len(selected)
                ]
                for source in sources
                if selected.casefold() in source.casefold()
            ),
            "",
        )
        verdicts.append({
            "action_id": grounding["action_id"],
            "argument_name": grounding["argument_name"],
            "selected_value": grounding["selected_value"],
            "status": "selected",
            "evidence_quote": quote,
            "reason": "the exact source affirmatively selects this enum value",
        })
    return {
        "verdicts": verdicts,
        "reason": "all closed-enum values are affirmatively selected",
    }


class HierarchicalPlannerContractTest(unittest.TestCase):
    def test_upstream_mutation_detaches_manual_answer_from_stale_pending(
        self,
    ) -> None:
        import json

        from agent.harness.semantic_admission import (
            _canonicalize_pending_choice_actions,
        )
        from agent.harness.state import new_state

        text = "Change the protocol family, then validate method_a."
        state = new_state("pending-superseded", language="en")
        state["pending_question"] = {
            "id": "new_chain_method",
            "group": "endpoint_process",
            "manual_input_allowed": True,
            "value_domain": "typed_value",
            "validation": {"input_mode": "rpc_method_or_schema_evidence"},
            "manual_action": {
                "type": "rpc_catalog_command",
                "catalog_command": "set_method",
                "value_argument": "rpc_method",
            },
        }
        payload = {
            "actions": [
                {
                    "type": "choose_adapter_family",
                    "adapter_family": "jsonrpc",
                },
                {
                    "type": "answer_pending",
                    "answer": "method_a",
                    "source_evidence": "method_a",
                    "confidence": "high",
                },
            ],
            "semantic_units": [
                {
                    "unit_id": "unit-family",
                    "source_text": "Change the protocol family, ",
                    "action_indexes": [0],
                },
                {
                    "unit_id": "unit-method",
                    "source_text": "then validate method_a.",
                    "action_indexes": [1],
                },
            ],
        }

        canonical = json.loads(
            _canonicalize_pending_choice_actions(
                json.dumps(payload),
                state,
            )
        )

        self.assertEqual(
            [action["type"] for action in canonical["actions"]],
            ["choose_adapter_family", "rpc_catalog_command"],
        )
        self.assertEqual(canonical["actions"][1]["catalog_command"], "set_method")
        self.assertEqual(canonical["actions"][1]["rpc_method"], "method_a")

    def test_registered_cross_group_mutation_requires_independent_proposal(
        self,
    ) -> None:
        from agent.harness.hierarchical_planner import (
            _partition_requires_independent_proposal,
        )

        partition = [
            {
                "unit_id": "unit-1",
                "operation": "domain_request",
                "owner_routes": [{
                    "owner": "chain_rpc",
                    "group": "target_mode",
                }],
            },
            {
                "unit_id": "unit-2",
                "operation": "domain_request",
                "owner_routes": [{
                    "owner": "chain_rpc",
                    "group": "chain_identity",
                }],
            },
        ]
        payload = {
            "pending_question": {"group": "chain_identity"},
            "registered_value_mentions": [{
                "target_group": "target_mode",
                "value": "fake-node",
            }],
        }

        self.assertTrue(
            _partition_requires_independent_proposal(partition, payload)
        )

    def test_registered_value_consultation_does_not_require_mutation_consensus(
        self,
    ) -> None:
        from agent.harness.hierarchical_planner import (
            _partition_requires_independent_proposal,
        )

        payload = {
            "pending_question": {"group": "chain_identity"},
            "registered_value_mentions": [{
                "target_group": "target_mode",
                "value": "fake-node",
            }],
        }
        consultation = [{
            "unit_id": "unit-1",
            "operation": "consultation",
            "owner_routes": [{"owner": "orientation", "group": "target_mode"}],
        }]
        same_group_mutation = [{
            "unit_id": "unit-2",
            "operation": "domain_request",
            "owner_routes": [{"owner": "chain_rpc", "group": "chain_identity"}],
        }]

        self.assertFalse(
            _partition_requires_independent_proposal(consultation, payload)
        )
        self.assertFalse(
            _partition_requires_independent_proposal(same_group_mutation, payload)
        )

    def test_stage_a_convergence_receives_registered_value_authority(
        self,
    ) -> None:
        from agent.harness.hierarchical_planner import _select_stage_a_proposal

        primary = [{
            "unit_id": "unit-1",
            "clause_id": "clause-1",
            "source_text": "fake-node",
            "operation": "domain_request",
            "owner_routes": [{"owner": "chain_rpc", "group": "target_mode"}],
            "reason": "registered mode request",
        }]
        secondary = [{
            **primary[0],
            "reason": "independent registered mode request",
        }]
        stage_a_payload = {
            "user_text": "fake-node test",
            "clauses": [{"clause_id": "clause-1", "text": "fake-node test"}],
            "pending_question": {"group": "chain_identity"},
            "pending_barrier_contract": {},
            "pending_typed_candidates": [],
            "contract_proven_pending_prefixes": [],
            "registered_semantic_value_domains": [{
                "target_group": "target_mode",
                "value": "fake-node",
            }],
            "registered_value_mentions": [{
                "target_group": "target_mode",
                "value": "fake-node",
            }],
            "groups": [],
            "routing_purposes": [{
                "action_type": "choose_target_mode",
                "owner": "chain_rpc",
                "purpose": "choose a target mode",
                "semantic_operations": ["domain_request"],
                "route_groups": ["target_mode"],
            }],
            "universal_operations": ["domain_request"],
            "universal_operation_purposes": {"domain_request": "mutate"},
            "universal_owner_action_purposes": {},
        }
        captured: dict = {}

        def converge(_provider, *, request_payload, **_kwargs):
            captured.update(request_payload)
            return json.dumps({
                "selected_proposal": "primary",
                "primary_hash": request_payload["primary"]["hash"],
                "secondary_hash": request_payload["secondary"]["hash"],
                "reason": "the primary preserves registered value ownership",
            })

        with patch(
            "agent.harness.hierarchical_planner.request_semantic_compilation",
            side_effect=converge,
        ):
            selected, errors, _sizes, receipt = _select_stage_a_proposal(
                object(), stage_a_payload, primary, secondary
            )

        self.assertEqual(selected, primary)
        self.assertEqual(errors, ())
        self.assertTrue(receipt["valid"])
        self.assertEqual(
            captured["registered_value_mentions"],
            stage_a_payload["registered_value_mentions"],
        )
        self.assertEqual(
            captured["registered_semantic_value_domains"],
            stage_a_payload["registered_semantic_value_domains"],
        )
        self.assertEqual(
            captured["routing_purposes"],
            stage_a_payload["routing_purposes"],
        )

    def test_cross_group_registered_value_arbitrates_competing_open_identity(
        self,
    ) -> None:
        from agent.harness.hierarchical_planner import begin_semantic_partition

        text = "fake-node test"
        wrong = {
            "semantic_units": [
                {
                    "unit_id": "unit-1",
                    "clause_id": "clause-1",
                    "source_text": "fake-node ",
                    "operation": "domain_request",
                    "owner_routes": [{
                        "owner": "chain_rpc",
                        "group": "target_mode",
                    }],
                    "reason": "registered mode request",
                },
                {
                    "unit_id": "unit-2",
                    "clause_id": "clause-1",
                    "source_text": "test",
                    "operation": "domain_request",
                    "owner_routes": [{
                        "owner": "chain_rpc",
                        "group": "chain_identity",
                    }],
                    "reason": "incorrect open identity",
                },
            ],
            "reason": "competing interpretation",
        }
        def admission(units, *, support: bool):
            return {
                "unit_verdicts": [
                    {
                        "unit_id": "unit-1",
                        "verdict": "complete",
                        "supports_unit_id": "",
                        "reason": "registered mode request is complete",
                    },
                    {
                        "unit_id": "unit-2",
                        "verdict": "redundant" if support else "complete",
                        "supports_unit_id": "unit-1" if support else "",
                        "reason": "framing support" if support else "named identity",
                    },
                ],
                "clause_verdicts": [{
                    "clause_id": "clause-1",
                    "verdict": "complete",
                    "omitted_owner_routes": [],
                    "reason": "the complete clause is represented",
                }],
                "reason": "reviewed",
            }

        outputs = iter((
            json.dumps(wrong),
            json.dumps(admission(wrong["semantic_units"], support=False)),
        ))
        compiler_payloads = []

        def compile_semantics(_provider, *, system_prompt, request_payload, **_kwargs):
            compiler_payloads.append(request_payload)
            return next(outputs)

        state = {
            "active_group": "chain_identity",
            "pending_question": {
                "id": "chain",
                "group": "chain_identity",
                "owner": "chain_rpc",
                "kind": "chain",
                "manual_input_allowed": True,
            },
        }
        with (
            patch(
                "agent.harness.hierarchical_planner.provider_from_config",
                return_value=object(),
            ),
            patch(
                "agent.harness.hierarchical_planner.request_semantic_compilation",
                side_effect=compile_semantics,
            ) as compiler,
            patch(
                "agent.harness.hierarchical_planner.request_open_identity_relation_jury",
                return_value=SimpleNamespace(
                    decisions=({
                        "unit_id": "unit-2",
                        "relation": "supports_unit",
                        "supports_unit_id": "unit-1",
                        "quorum_reached": True,
                    },),
                    errors=(),
                    request_sizes=(100, 100, 100),
                    receipt={
                        "proposal_hash": "1" * 64,
                        "candidate_unit_ids": ["unit-2"],
                        "possible_support_unit_ids": {"unit-2": ["unit-1"]},
                        "member_response_hashes": ["2" * 64, "3" * 64, "4" * 64],
                        "member_validity": [True, True, True],
                        "request_count": 3,
                        "request_sizes": [100, 100, 100],
                        "decisions": [{
                            "unit_id": "unit-2",
                            "relation": "supports_unit",
                            "supports_unit_id": "unit-1",
                            "quorum_reached": True,
                        }],
                        "valid": True,
                    },
                ),
            ),
        ):
            result = begin_semantic_partition(state, text)

        self.assertEqual(result["status"], "compile_owner")
        self.assertEqual(result["stage_a_calls"], 1)
        self.assertEqual(result["admission_calls"], 4)
        self.assertEqual(compiler.call_count, 2)
        self.assertEqual(
            compiler_payloads[1]["semantic_units"][1]["operation"],
            "context",
        )
        self.assertEqual(
            compiler_payloads[1]["semantic_units"][1]["owner_routes"],
            [],
        )
        self.assertEqual(
            result["source_partition"][1]["operation"],
            "context",
        )
        self.assertIs(
            result["source_partition"][1]["_admission_support"],
            True,
        )
        self.assertEqual(
            result["stage_a_relation_reviews"][0]["decisions"],
            [{
                "unit_id": "unit-2",
                "relation": "supports_unit",
                "supports_unit_id": "unit-1",
                "quorum_reached": True,
            }],
        )
        self.assertEqual(
            [row["groups"] for row in result["owner_requests"]],
            [["target_mode"]],
        )

    def test_open_identity_relation_jury_preserves_a_named_sibling(self) -> None:
        from agent.harness.hierarchical_planner import (
            _review_competing_open_identity_relations,
        )

        partition = [
            {
                "unit_id": "unit-1",
                "clause_id": "clause-1",
                "source_text": "fake-node ",
                "operation": "domain_request",
                "owner_routes": [{"owner": "chain_rpc", "group": "target_mode"}],
            },
            {
                "unit_id": "unit-2",
                "clause_id": "clause-1",
                "source_text": "BNB",
                "operation": "domain_request",
                "owner_routes": [{"owner": "chain_rpc", "group": "chain_identity"}],
            },
        ]
        payload = {
            "user_text": "fake-node BNB",
            "clauses": [{"clause_id": "clause-1", "text": "fake-node BNB"}],
            "pending_question": {"group": "chain_identity"},
            "registered_value_mentions": [{
                "value": "fake-node",
                "target_group": "target_mode",
            }],
        }
        verdict = json.dumps({
            "verdicts": [{
                "unit_id": "unit-2",
                "relation": "named_identity",
                "supports_unit_id": "",
                "evidence_quote": "BNB",
                "reason": "the source names a distinct chain identity",
            }],
        })
        with patch(
            "agent.harness.semantic_compiler.request_semantic_compilation_result",
            side_effect=[
                SimpleNamespace(text=verdict, response_hash=str(index) * 64)
                for index in (1, 2, 3)
            ],
        ):
            normalized, errors, sizes, receipt = (
                _review_competing_open_identity_relations(
                    object(), partition, payload
                )
            )

        self.assertEqual(errors, ())
        self.assertEqual(len(sizes), 3)
        self.assertEqual(normalized, partition)
        self.assertEqual(receipt["member_validity"], [True, True, True])

    def test_open_identity_relation_jury_fails_closed_without_quorum(self) -> None:
        from agent.harness.hierarchical_planner import (
            _review_competing_open_identity_relations,
        )

        partition = [
            {
                "unit_id": "unit-1",
                "clause_id": "clause-1",
                "source_text": "fake-node ",
                "operation": "domain_request",
                "owner_routes": [{"owner": "chain_rpc", "group": "target_mode"}],
            },
            {
                "unit_id": "unit-2",
                "clause_id": "clause-1",
                "source_text": "test",
                "operation": "domain_request",
                "owner_routes": [{"owner": "chain_rpc", "group": "chain_identity"}],
            },
        ]
        payload = {
            "user_text": "fake-node test",
            "clauses": [{"clause_id": "clause-1", "text": "fake-node test"}],
            "pending_question": {"group": "chain_identity"},
            "registered_value_mentions": [{
                "value": "fake-node",
                "target_group": "target_mode",
            }],
        }
        forged = json.dumps({
            "verdicts": [{
                "unit_id": "forged",
                "relation": "supports_unit",
                "supports_unit_id": "unit-1",
                "evidence_quote": "test",
                "reason": "forged unit",
            }],
        })
        named = json.dumps({
            "verdicts": [{
                "unit_id": "unit-2",
                "relation": "named_identity",
                "supports_unit_id": "",
                "evidence_quote": "test",
                "reason": "one minority vote",
            }],
        })
        with patch(
            "agent.harness.semantic_compiler.request_semantic_compilation_result",
            side_effect=[
                SimpleNamespace(text=forged, response_hash="1" * 64),
                SimpleNamespace(text=named, response_hash="2" * 64),
                SimpleNamespace(text="not-json", response_hash="3" * 64),
            ],
        ):
            normalized, errors, _sizes, receipt = (
                _review_competing_open_identity_relations(
                    object(), partition, payload
                )
            )

        self.assertTrue(errors)
        self.assertEqual(normalized[1]["operation"], "unresolved")
        self.assertEqual(normalized[1]["owner_routes"], [])
        self.assertEqual(receipt["member_validity"], [False, True, False])

    def test_open_identity_relation_jury_accepts_support_without_duplicate_reason(
        self,
    ) -> None:
        from agent.harness.semantic_compiler import (
            request_open_identity_relation_jury,
        )

        payload = {
            "relations": [{
                "unit": {
                    "unit_id": "unit-2",
                    "source_text": "run",
                },
                "possible_support_units": [{"unit_id": "unit-1"}],
            }],
        }
        verdict = json.dumps({
            "verdicts": [{
                "unit_id": "unit-2",
                "relation": "supports_unit",
                "supports_unit_id": "unit-1",
                "evidence_quote": "run",
                "reason": "generic request framing supports the registered value",
            }],
        })
        with patch(
            "agent.harness.semantic_compiler.request_semantic_compilation_result",
            side_effect=[
                SimpleNamespace(text=verdict, response_hash=str(index) * 64)
                for index in (1, 2, 3)
            ],
        ):
            review = request_open_identity_relation_jury(
                object(),
                proposal_hash="4" * 64,
                request_payload=payload,
                max_tokens=900,
            )

        self.assertEqual(review.errors, ())
        self.assertEqual(review.receipt["member_validity"], [True, True, True])
        self.assertEqual(review.decisions[0]["relation"], "supports_unit")
        self.assertEqual(review.decisions[0]["supports_unit_id"], "unit-1")

    def test_open_identity_relation_jury_runs_concurrently_in_member_order(
        self,
    ) -> None:
        from agent.harness.semantic_compiler import (
            request_open_identity_relation_jury,
        )
        from agent.llm.types import llm_turn_scope

        payload = {
            "relations": [{
                "unit": {
                    "unit_id": "unit-2",
                    "source_text": "run",
                },
                "possible_support_units": [{"unit_id": "unit-1"}],
            }],
        }
        verdict = json.dumps({
            "verdicts": [{
                "unit_id": "unit-2",
                "relation": "supports_unit",
                "supports_unit_id": "unit-1",
                "evidence_quote": "run",
                "reason": "the source supports the registered value",
            }],
        })
        barrier = threading.Barrier(3)
        result_lock = threading.Lock()
        result_index = 0

        def review(*_args, **_kwargs):
            nonlocal result_index
            with result_lock:
                index = result_index
                result_index += 1
            barrier.wait(timeout=1)
            return SimpleNamespace(
                text=verdict,
                response_hash=str(index + 1) * 64,
            )

        with (
            llm_turn_scope(1),
            patch(
                "agent.harness.semantic_compiler.request_semantic_compilation_result",
                side_effect=review,
            ),
        ):
            relation_review = request_open_identity_relation_jury(
                object(),
                proposal_hash="4" * 64,
                request_payload=payload,
                max_tokens=900,
            )

        self.assertEqual(relation_review.errors, ())
        self.assertEqual(
            set(relation_review.receipt["member_response_hashes"]),
            {"1" * 64, "2" * 64, "3" * 64},
        )
        self.assertEqual(
            relation_review.receipt["member_validity"],
            [True, True, True],
        )

    def test_open_identity_relation_jury_preserves_explicit_unresolved_quorum(self) -> None:
        from agent.harness.hierarchical_planner import (
            _review_competing_open_identity_relations,
        )

        partition = [
            {
                "unit_id": "unit-1", "clause_id": "clause-1",
                "source_text": "fake-node ", "operation": "domain_request",
                "owner_routes": [{"owner": "chain_rpc", "group": "target_mode"}],
            },
            {
                "unit_id": "unit-2", "clause_id": "clause-1",
                "source_text": "ambiguous-name", "operation": "domain_request",
                "owner_routes": [{"owner": "chain_rpc", "group": "chain_identity"}],
            },
        ]
        payload = {
            "user_text": "fake-node ambiguous-name",
            "clauses": [{"clause_id": "clause-1", "text": "fake-node ambiguous-name"}],
            "pending_question": {"group": "chain_identity"},
            "registered_value_mentions": [{"value": "fake-node", "target_group": "target_mode"}],
        }
        verdict = json.dumps({
            "verdicts": [{
                "unit_id": "unit-2", "relation": "unresolved",
                "supports_unit_id": "", "evidence_quote": "ambiguous-name",
                "reason": "the source does not establish either relation",
            }],
        })
        with patch(
            "agent.harness.semantic_compiler.request_semantic_compilation_result",
            side_effect=[
                SimpleNamespace(text=verdict, response_hash=str(index) * 64)
                for index in (1, 2, 3)
            ],
        ):
            normalized, errors, _sizes, receipt = (
                _review_competing_open_identity_relations(
                    object(), partition, payload
                )
            )

        self.assertEqual(errors, ())
        self.assertEqual(normalized[1]["operation"], "unresolved")
        self.assertTrue(receipt["valid"])
        self.assertTrue(receipt["decisions"][0]["quorum_reached"])

    def test_open_identity_relation_jury_does_not_join_unrelated_clauses(self) -> None:
        from agent.harness.hierarchical_planner import (
            _review_competing_open_identity_relations,
        )

        partition = [
            {
                "unit_id": "unit-1", "clause_id": "clause-1",
                "source_text": "fake-node", "operation": "domain_request",
                "owner_routes": [{"owner": "chain_rpc", "group": "target_mode"}],
            },
            {
                "unit_id": "unit-2", "clause_id": "clause-2",
                "source_text": "BNB", "operation": "domain_request",
                "owner_routes": [{"owner": "chain_rpc", "group": "chain_identity"}],
            },
        ]
        payload = {
            "user_text": "fake-node\nBNB",
            "clauses": [
                {"clause_id": "clause-1", "text": "fake-node"},
                {"clause_id": "clause-2", "text": "BNB"},
            ],
            "pending_question": {"group": "chain_identity"},
            "registered_value_mentions": [{"value": "fake-node", "target_group": "target_mode"}],
        }
        with patch(
            "agent.harness.hierarchical_planner.request_open_identity_relation_jury"
        ) as reviewer:
            normalized, errors, sizes, receipt = (
                _review_competing_open_identity_relations(
                    object(), partition, payload
                )
            )

        reviewer.assert_not_called()
        self.assertEqual(normalized, partition)
        self.assertEqual((errors, sizes, receipt), ((), (), {}))

    def test_stage_a_output_budget_scales_with_structured_atom_count(self) -> None:
        from agent.harness.hierarchical_planner import (
            _stage_a_output_token_budget,
            _stage_a_review_output_token_budget,
            _stage_b_output_token_budget,
            _stage_b_review_output_token_budget,
        )

        self.assertEqual(_stage_a_output_token_budget({}), 2600)
        self.assertEqual(
            _stage_a_output_token_budget({
                "structured_candidates": [{
                    "field_candidates": [{} for _item in range(9)],
                }],
            }),
            10300,
        )
        self.assertEqual(
            _stage_a_output_token_budget({
                "structured_candidates": [{
                    "field_candidates": [{} for _item in range(100)],
                }],
            }),
            12000,
        )
        self.assertEqual(_stage_a_review_output_token_budget([], []), 2200)
        self.assertEqual(
            _stage_a_review_output_token_budget([{} for _ in range(5)], [{}]),
            5800,
        )
        self.assertEqual(
            _stage_a_review_output_token_budget([{} for _ in range(100)], [{}]),
            12000,
        )
        self.assertEqual(_stage_b_review_output_token_budget({}, {}), 1800)
        self.assertEqual(_stage_b_output_token_budget({}), 3200)
        self.assertEqual(
            _stage_b_output_token_budget({
                "semantic_units": [{} for _ in range(4)],
            }),
            5800,
        )
        self.assertEqual(
            _stage_b_review_output_token_budget(
                {"semantic_units": [{} for _ in range(4)]},
                {"actions": [{} for _ in range(4)]},
            ),
            6200,
        )

    def test_registered_composite_structured_intake_owns_its_subtree(self) -> None:
        from agent.harness.hierarchical_planner import (
            _semantic_structured_input_candidates,
        )

        candidates = _semantic_structured_input_candidates(json.dumps({
            "chain": "Flow-EVM",
            "protocol_family": "jsonrpc",
            "validation_endpoint": "http://127.0.0.1:8545",
            "rpc_request": {
                "jsonrpc": "2.0",
                "method": "eth_chainId",
                "params": [],
                "id": 1,
            },
        }))

        self.assertEqual(
            [
                candidate["source_path"]
                for candidate in candidates["field_candidates"]
            ],
            [
                "chain",
                "protocol_family",
                "validation_endpoint",
                "rpc_request",
            ],
        )
        self.assertIsInstance(
            candidates["field_candidates"][-1]["raw_value"],
            dict,
        )
        self.assertEqual(
            candidates["field_candidates"][-1]["source_evidence"],
            '{"jsonrpc": "2.0", "method": "eth_chainId", '
            '"params": [], "id": 1}',
        )

    def test_structured_grounding_compares_objects_by_json_value(self) -> None:
        from agent.harness.semantic_compiler import (
            _grounding_quote_contains_value,
        )

        value = {
            "jsonrpc": "2.0",
            "method": "eth_chainId",
            "params": [],
            "id": 1,
        }
        self.assertTrue(_grounding_quote_contains_value(
            value,
            '{\n  "jsonrpc": "2.0", "method": "eth_chainId", '
            '"params": [], "id": 1\n}',
        ))
        self.assertFalse(_grounding_quote_contains_value(
            value,
            '{"jsonrpc":"2.0","method":"eth_blockNumber",'
            '"params":[],"id":1}',
        ))

    def test_structured_protocol_family_reaches_its_registered_owner(self) -> None:
        from agent.harness.hierarchical_planner import (
            _stage_b_payload,
            _validate_partition_document,
        )
        from agent.harness.plan_coverage import TurnClause
        from agent.harness.state import new_state

        source = 'protocol_family: "jsonrpc"'
        clauses = (TurnClause("clause-1", source, "structured"),)
        unit = {
            "unit_id": "__harness_structured_clause-1_1",
            "operation": "domain_request",
            "owner_routes": [{
                "owner": "chain_rpc",
                "group": "chain_identity",
            }],
            "reason": "structured adapter family",
        }

        partition, errors = _validate_partition_document(
            json.dumps({"semantic_units": [unit], "reason": "complete"}),
            clauses,
        )

        self.assertEqual(errors, ())
        self.assertEqual(partition[0]["registered_intake"], {
            "action_type": "choose_adapter_family",
            "owner": "chain_rpc",
            "target_group": "chain_identity",
            "alias": "protocol_family",
            "fixed_arguments": {},
            "value_semantics": "direct_value",
            "value_argument": "adapter_family",
        })
        payload = _stage_b_payload(
            new_state("structured-protocol-family", language="en"),
            "chain_rpc",
            frozenset({"chain_identity"}),
            partition,
            (partition[0]["unit_id"],),
        )
        self.assertEqual(
            payload["semantic_units"][0]["semantic_value"],
            "jsonrpc",
        )

    def test_stage_a_rebinds_colliding_prose_labels_without_dropping_units(
        self,
    ) -> None:
        from agent.harness.hierarchical_planner import _validate_partition_document
        from agent.harness.plan_coverage import TurnClause

        clauses = (TurnClause("clause-1", "fake-node 测试", "prose"),)
        document = {
            "semantic_units": [
                {
                    "unit_id": "clause-1",
                    "clause_id": "clause-1",
                    "source_text": "fake-node",
                    "operation": "domain_request",
                    "owner_routes": [{
                        "owner": "chain_rpc",
                        "group": "target_mode",
                    }],
                    "reason": "registered target mode",
                },
                {
                    "unit_id": "clause-1",
                    "clause_id": "clause-1",
                    "source_text": "测试",
                    "operation": "context",
                    "owner_routes": [],
                    "reason": "operation framing",
                },
            ],
            "reason": "two semantic units",
        }

        first, first_errors = _validate_partition_document(
            json.dumps(document, ensure_ascii=False),
            clauses,
        )
        second, second_errors = _validate_partition_document(
            json.dumps(document, ensure_ascii=False),
            clauses,
        )

        self.assertEqual(first_errors, ())
        self.assertEqual(second_errors, ())
        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)
        self.assertEqual(
            [unit["unit_id"] for unit in first],
            ["__harness_prose_clause-1_1", "__harness_prose_clause-1_2"],
        )
        self.assertEqual(
            [unit["source_text"] for unit in first],
            ["fake-node ", "测试"],
        )

    def test_stage_a_rebinds_missing_prose_label_and_preserves_unique_label(
        self,
    ) -> None:
        from agent.harness.hierarchical_planner import (
            _bind_ambiguous_prose_partition_identities,
        )
        from agent.harness.plan_coverage import TurnClause

        clauses = (TurnClause("clause-1", "configure mode", "prose"),)
        rebound = _bind_ambiguous_prose_partition_identities(
            [
                {"unit_id": "", "clause_id": "clause-1"},
                {"unit_id": "unit-2", "clause_id": "clause-1"},
            ],
            clauses,
        )

        self.assertEqual(rebound[0]["unit_id"], "__harness_prose_clause-1_1")
        self.assertEqual(rebound[1]["unit_id"], "unit-2")

    def test_stage_a_does_not_rebind_structured_atom_collisions(self) -> None:
        from agent.harness.hierarchical_planner import (
            _bind_ambiguous_prose_partition_identities,
        )
        from agent.harness.plan_coverage import TurnClause

        atom_id = "__harness_structured_clause-1_1"
        units = [
            {"unit_id": atom_id, "clause_id": "clause-1"},
            {"unit_id": atom_id, "clause_id": "clause-1"},
        ]

        self.assertEqual(
            _bind_ambiguous_prose_partition_identities(
                units,
                (TurnClause("clause-1", '{"chain":"bsc"}', "structured"),),
            ),
            units,
        )

    def test_receipt_attachment_rejection_becomes_typed_unresolved_work(
        self,
    ) -> None:
        from agent.harness.hierarchical_planner import review_semantic_plan
        from agent.harness.plan_coverage import PlanCoverageResult
        from agent.harness.state import new_state

        text = (
            "Switch to network; use eth0, but do not use eth0; "
            "show current network evidence."
        )
        unit = {
            "unit_id": "network-unit",
            "clause_id": "network-clause",
            "source_text": text,
            "operation": "domain_request",
            "owner_routes": [{
                "owner": "environment",
                "group": "network",
            }],
            "reason": "contradictory network proposal",
        }
        document = {
            "status": "review_plan",
            "clauses": [{
                "clause_id": "network-clause",
                "text": text,
                "input_shape": "prose",
            }],
            "source_partition": [unit],
            "routed_partition": [unit],
            "owner_documents": {
                "environment": {
                    "actions": [{
                        "type": "set_config_value",
                        "key": "NETWORK_INTERFACE",
                        "value": "eth0",
                    }],
                    "bindings": [{
                        "unit_id": "network-unit",
                        "action_indexes": [0],
                        "disposition": "action",
                        "reason": "compiled",
                    }],
                },
            },
        }
        state = new_state("receipt-rejection", language="en")
        state["semantic_plan_draft"] = {
            "status": "ready_for_review",
            "draft_id": "preserve-product-head",
        }

        with (
            patch(
                "agent.harness.hierarchical_planner.prepare_hierarchical_candidate",
                side_effect=lambda candidate, *_args, **_kwargs: (
                    candidate,
                    PlanCoverageResult(True, (), ()),
                ),
            ),
            patch(
                "agent.harness.hierarchical_planner.provider_from_config",
                return_value=object(),
            ),
            patch(
                "agent.harness.hierarchical_planner._review_bounded_semantic_candidate",
                return_value=(
                    SimpleNamespace(request_json="{}"),
                    SimpleNamespace(
                        valid=True,
                        errors=(),
                        request_count=1,
                        request_sizes=(),
                    ),
                    (),
                ),
            ),
            patch(
                "agent.harness.hierarchical_planner._admitted_action_queue",
                side_effect=ValueError(
                    "configuration proposal lacks current-turn provenance: "
                    "NETWORK_INTERFACE"
                ),
            ),
        ):
            result = review_semantic_plan(state, document)

        self.assertEqual(
            [action["type"] for action in result["actions"]],
            ["clarify_unresolved"],
        )
        self.assertEqual(
            result["semantic_units"][0]["disposition"],
            "unresolved",
        )
        self.assertIn(
            "lacks current-turn provenance",
            result["coverage_errors"][0],
        )
        self.assertEqual(
            state["semantic_plan_draft"]["draft_id"],
            "preserve-product-head",
        )

    def test_owner_compile_failure_is_preserved_as_typed_unresolved_work(
        self,
    ) -> None:
        from agent.harness.hierarchical_planner import compile_next_owner
        from agent.harness.state import new_state

        document = {
            "status": "compile_owner",
            "owner_cursor": 0,
            "owner_requests": [{
                "owner": "chain_rpc",
                "unit_ids": ["unit-chain"],
                "groups": ["chain_identity"],
            }],
            "routed_partition": [{
                "unit_id": "unit-chain",
                "clause_id": "clause-1",
                "source_text": "switch to BNB",
                "operation": "domain_request",
                "owner_routes": [{
                    "owner": "chain_rpc",
                    "group": "chain_identity",
                }],
            }],
        }

        with patch(
            "agent.harness.hierarchical_planner._compile_owner_document",
            return_value=(
                {},
                ("Stage B chain_rpc did not return strict JSON",),
                (101, 103),
            ),
        ):
            result = compile_next_owner(new_state("owner-failure"), document)

        self.assertEqual(result["status"], "review_plan")
        self.assertEqual(result["owner_cursor"], 1)
        self.assertEqual(
            result["owner_documents"]["chain_rpc"]["bindings"],
            [{
                "unit_id": "unit-chain",
                "action_indexes": [],
                "disposition": "unresolved",
                "reason": (
                    "The owning compiler could not produce a valid typed "
                    "action after bounded repair."
                ),
            }],
        )
        self.assertEqual(
            result["owner_failures"][0]["unit_ids"],
            ["unit-chain"],
        )

    def test_finalization_does_not_recursively_create_semantic_draft(
        self,
    ) -> None:
        from agent.harness.hierarchical_planner import review_semantic_plan
        from agent.harness.plan_coverage import PlanCoverageResult
        from agent.harness.state import new_state

        state = new_state("finalization-no-recursion", language="en")
        state["semantic_plan_draft"] = {
            "status": "ready_for_review",
            "draft_id": "ready-draft",
            "revision": 2,
        }
        state["turn_context"] = {
            "semantic_draft_finalization": {
                "draft_id": "ready-draft",
                "revision": 2,
            },
        }
        unresolved = {
            "unit_id": "unit-workload",
            "clause_id": "clause-1",
            "source_text": "Do not retain the previous custom workload.",
            "operation": "domain_request",
            "owner_routes": [{
                "owner": "chain_rpc",
                "group": "workload_rpc",
            }],
            "disposition": "unresolved",
            "action_indexes": [],
            "reason": "no registered action represents the clarified demand",
        }
        document = {
            "status": "review_plan",
            "clauses": [{
                "clause_id": "clause-1",
                "text": unresolved["source_text"],
                "input_shape": "prose",
            }],
            "source_partition": [unresolved],
            "routed_partition": [unresolved],
            "owner_documents": {
                "chain_rpc": {
                    "actions": [],
                    "bindings": [{
                        "unit_id": "unit-workload",
                        "action_indexes": [],
                        "disposition": "unresolved",
                        "reason": unresolved["reason"],
                    }],
                },
            },
        }
        coverage = PlanCoverageResult(
            valid=False,
            errors=(),
            unresolved_clauses=(unresolved["source_text"],),
            unresolved_units=(unresolved,),
        )

        with patch(
            "agent.harness.hierarchical_planner.prepare_hierarchical_candidate",
            return_value=("{}", coverage),
        ):
            result = review_semantic_plan(state, document)

        self.assertNotIn("semantic_draft", result)
        self.assertEqual(
            [action["type"] for action in result["actions"]],
            ["clarify_unresolved"],
        )
        self.assertEqual(
            result["unresolved_clauses"],
            [unresolved["source_text"]],
        )
        self.assertIn(
            "finalization remained unresolved",
            result["coverage_errors"][0],
        )

    def test_owner_batch_compiles_concurrently_and_merges_request_order(
        self,
    ) -> None:
        from agent.harness.hierarchical_planner import compile_next_owner
        from agent.harness.state import new_state
        from agent.llm.types import llm_turn_scope

        owners = ("orientation", "performance", "chain_rpc")
        barrier = threading.Barrier(len(owners))

        def compile_owner(_state, owner, _groups, _partition, unit_ids):
            barrier.wait(timeout=1)
            return ({
                "actions": [],
                "bindings": [{
                    "unit_id": unit_ids[0],
                    "action_indexes": [],
                    "disposition": "context",
                    "reason": owner,
                }],
            }, (), (100 + owners.index(owner),))

        document = {
            "status": "compile_owner",
            "owner_cursor": 0,
            "owner_requests": [
                {
                    "owner": owner,
                    "unit_ids": [f"unit-{index}"],
                    "groups": [],
                }
                for index, owner in enumerate(owners)
            ],
            "routed_partition": [],
            "request_sizes": [77],
            "stage_b_calls": 0,
        }
        with (
            llm_turn_scope(1),
            patch(
                "agent.harness.hierarchical_planner._compile_owner_document",
                side_effect=compile_owner,
            ),
        ):
            result = compile_next_owner(new_state("owner-batch"), document)

        self.assertEqual(result["status"], "review_plan")
        self.assertEqual(result["owner_cursor"], 3)
        self.assertEqual(list(result["owner_documents"]), list(owners))
        self.assertEqual(result["request_sizes"], [77, 100, 101, 102])
        self.assertEqual(result["stage_b_calls"], 3)

    def test_stage_a_structured_demand_atoms_can_use_distinct_operations(
        self,
    ) -> None:
        from agent.harness.hierarchical_planner import (
            _stage_b_payload,
            _validate_partition_document,
        )
        from agent.harness.plan_coverage import segment_user_turn
        from agent.harness.state import new_state

        text = '{"CHAIN":"Flow","CLOUD_REGION":"us-1"}'
        clauses = segment_user_turn(text)
        document = {
            "semantic_units": [
                {
                    "unit_id": "__harness_structured_clause-1_1",
                    "clause_id": clauses[0].clause_id,
                    "source_text": text,
                    "source_path": "CHAIN",
                    "operation": "pending_answer",
                    "owner_routes": [{
                        "owner": "coordinator",
                        "group": "chain_identity",
                    }],
                    "reason": "answers the active chain question",
                },
                {
                    "unit_id": "__harness_structured_clause-1_2",
                    "clause_id": clauses[0].clause_id,
                    "source_text": text,
                    "source_path": "CLOUD_REGION",
                    "operation": "domain_request",
                    "owner_routes": [{
                        "owner": "environment",
                        "group": "provider_deployment",
                    }],
                    "reason": "supplies an environment value",
                },
            ],
            "reason": "structured demands",
        }

        partition, errors = _validate_partition_document(
            json.dumps(document),
            clauses,
        )

        self.assertFalse(errors, errors)
        self.assertEqual(
            [unit["source_path"] for unit in partition],
            ["CHAIN", "CLOUD_REGION"],
        )
        payload = _stage_b_payload(
            new_state("structured-demand-atoms", language="en"),
            "coordinator",
            frozenset({"chain_identity"}),
            partition,
            ("__harness_structured_clause-1_1",),
        )
        self.assertEqual(
            payload["semantic_units"][0]["semantic_source"],
            '"Flow"',
        )
        self.assertEqual(
            payload["semantic_units"][0]["semantic_value"],
            "Flow",
        )
        self.assertEqual(
            payload["semantic_units"][0]["source_text"],
            text,
        )

    def test_stage_a_structured_demand_atoms_must_cover_every_field_path(
        self,
    ) -> None:
        from agent.harness.hierarchical_planner import (
            _validate_partition_document,
        )
        from agent.harness.plan_coverage import segment_user_turn

        text = '{"CHAIN":"Flow","CLOUD_REGION":"us-1"}'
        clauses = segment_user_turn(text)
        document = {
            "semantic_units": [{
                "unit_id": "chain-atom",
                "clause_id": clauses[0].clause_id,
                "source_text": text,
                "source_path": "CHAIN",
                "operation": "domain_request",
                "owner_routes": [{
                    "owner": "chain_rpc",
                    "group": "chain_identity",
                }],
                "reason": "chain demand",
            }],
            "reason": "incomplete structured demands",
        }

        _partition, errors = _validate_partition_document(
            json.dumps(document),
            clauses,
        )

        self.assertTrue(
            any("omit source paths" in error for error in errors),
            errors,
        )

    def test_cross_domain_validation_scopes_registered_values_to_demand_atom(
        self,
    ) -> None:
        from agent.harness.hierarchical_planner import (
            _cross_domain_pending_errors,
        )
        from agent.harness.questions import manual_question, question_text
        from agent.harness.state import new_state

        text = '{"CHAIN":"solana","RPC_MODE":"single"}'
        state = new_state("structured-cross-domain", language="en")
        state["pending_question"] = manual_question(
            "chain_identity",
            "chain",
            question_text("question.chain_rpc.chain.prompt"),
            owner="chain_rpc",
            field="chain",
            kind="scalar_token",
        )
        partition = [
            {
                "unit_id": "chain-atom",
                "clause_id": "clause-1",
                "source_text": text,
                "source_path": "CHAIN",
                "operation": "pending_answer",
                "owner_routes": [{
                    "owner": "coordinator",
                    "group": "chain_identity",
                }],
                "reason": "chain answer",
            },
            {
                "unit_id": "rpc-atom",
                "clause_id": "clause-1",
                "source_text": text,
                "source_path": "RPC_MODE",
                "operation": "domain_request",
                "owner_routes": [{
                    "owner": "chain_rpc",
                    "group": "workload_rpc",
                }],
                "reason": "RPC mode request",
            },
        ]

        self.assertEqual(_cross_domain_pending_errors(partition, state), ())

    def test_universal_operation_owners_are_derived_from_reachable_actions(
        self,
    ) -> None:
        from agent.harness.action_registry import (
            universal_semantic_operation_owners,
        )

        owners = universal_semantic_operation_owners()

        self.assertEqual(owners["pending_answer"], ("coordinator",))
        self.assertEqual(owners["consultation"], ("orientation",))
        self.assertEqual(owners["navigation"], ("coordinator",))
        self.assertEqual(
            owners["administrative"],
            ("coordinator", "orientation"),
        )
        self.assertEqual(owners["evidence_analysis"], ("analysis",))
        self.assertEqual(owners["report_analysis"], ("analysis",))
        self.assertNotIn("recovery", owners["administrative"])

    def test_registered_semantic_domains_include_chain_aliases(self) -> None:
        from agent.harness.action_registry import (
            ACTION_BY_TYPE,
            PENDING_BARRIER_POLICIES,
            SEMANTIC_OPERATION_PURPOSES,
            SEMANTIC_OPERATIONS,
            SEMANTIC_VALUE_DOMAIN_POLICY,
            action_registry_contract_hash,
            pending_barrier_semantics,
            registered_semantic_value_domains,
        )

        records = registered_semantic_value_domains()
        chain_spec = ACTION_BY_TYPE["choose_chain"]

        self.assertTrue(any(
            record.get("target_group") == "chain_identity"
            and record.get("value") == "bnb"
            and record.get("canonical_value") == "bsc"
            and record.get("action_type") == chain_spec.action_type
            and record.get("target_group") == chain_spec.target_group
            for record in records
        ))
        self.assertTrue(any(
            record.get("target_group") == "target_mode"
            and record.get("value") == "fake-node"
            for record in records
        ))
        fake_node_records = [
            record
            for record in records
            if record.get("value") == "fake-node"
        ]
        self.assertEqual(len(fake_node_records), 1)
        self.assertEqual(
            fake_node_records[0]["semantic_owner"],
            "target_mode",
        )
        self.assertEqual(
            fake_node_records[0]["target_group"],
            "target_mode",
        )
        self.assertEqual(
            fake_node_records[0]["action_type"],
            "choose_target_mode",
        )
        self.assertEqual(SEMANTIC_VALUE_DOMAIN_POLICY["schema_version"], 2)
        self.assertEqual(
            PENDING_BARRIER_POLICIES["exclusive_owner"][
                "registered_cross_group_values"
            ],
            "route_to_registered_owner",
        )
        self.assertEqual(
            pending_barrier_semantics({
                "queue_barrier": True,
                "barrier_policy": "exclusive_owner",
                "group": "chain_identity",
                "owner": "chain_rpc",
            })["pending_group"],
            "chain_identity",
        )
        self.assertEqual(len(action_registry_contract_hash()), 64)
        self.assertEqual(
            set(SEMANTIC_OPERATION_PURPOSES),
            set(SEMANTIC_OPERATIONS),
        )
        self.assertIn(
            "Hypothetical",
            SEMANTIC_OPERATION_PURPOSES["pending_answer"],
        )
        self.assertIn(
            "before the evidence is pasted",
            SEMANTIC_OPERATION_PURPOSES["evidence_analysis"],
        )

    def test_semantic_domain_representative_is_action_order_independent(
        self,
    ) -> None:
        from agent.harness import action_registry

        expected = [
            row
            for row in action_registry.registered_semantic_value_domains()
            if row.get("value") in {"fake-node", "bnb"}
        ]
        with patch.object(
            action_registry,
            "ACTION_SPECS",
            tuple(reversed(action_registry.ACTION_SPECS)),
        ):
            observed = [
                row
                for row in action_registry.registered_semantic_value_domains()
                if row.get("value") in {"fake-node", "bnb"}
            ]

        self.assertEqual(observed, expected)
        self.assertEqual(
            {
                row["value"]: row["action_type"]
                for row in observed
            },
            {
                "fake-node": "choose_target_mode",
                "bnb": "choose_chain",
            },
        )

    def test_semantic_domain_representative_invariants_fail_closed(self) -> None:
        from dataclasses import replace

        from agent.harness import action_registry

        duplicate_target_mode = tuple(
            replace(spec, semantic_value_representative=True)
            if spec.action_type == "queue_workflow_goal"
            else spec
            for spec in action_registry.ACTION_SPECS
        )
        with (
            patch.object(
                action_registry,
                "ACTION_SPECS",
                duplicate_target_mode,
            ),
            self.assertRaisesRegex(
                RuntimeError,
                "exactly one canonical representative",
            ),
        ):
            action_registry.registered_semantic_value_domains()

        wrong_target_mode_owner = tuple(
            replace(
                spec,
                semantic_value_representative=(
                    spec.action_type == "queue_workflow_goal"
                ),
            )
            if spec.action_type in {
                "choose_target_mode",
                "queue_workflow_goal",
            }
            else spec
            for spec in action_registry.ACTION_SPECS
        )
        with (
            patch.object(
                action_registry,
                "ACTION_SPECS",
                wrong_target_mode_owner,
            ),
            self.assertRaisesRegex(
                RuntimeError,
                "does not own the canonical workflow group",
            ),
        ):
            action_registry.registered_semantic_value_domains()

        missing_chain_representative = tuple(
            replace(spec, semantic_value_representative=False)
            if spec.action_type == "choose_chain"
            else spec
            for spec in action_registry.ACTION_SPECS
        )
        with (
            patch.object(
                action_registry,
                "ACTION_SPECS",
                missing_chain_representative,
            ),
            self.assertRaisesRegex(
                RuntimeError,
                "known chain identities require exactly one",
            ),
        ):
            action_registry.registered_semantic_value_domains()

    def test_pending_barrier_rejects_unknown_policy(self) -> None:
        from agent.harness.action_registry import pending_barrier_semantics

        with self.assertRaisesRegex(ValueError, "unknown pending barrier policy"):
            pending_barrier_semantics({
                "queue_barrier": True,
                "barrier_policy": "model_decides",
            })

    def test_registered_cross_group_value_cannot_be_left_unresolved(self) -> None:
        from agent.harness.hierarchical_planner import (
            _cross_domain_pending_errors,
        )

        state = {
            "pending_question": {
                "group": "chain_identity",
                "value_domain": "researched_identity",
                "queue_barrier": True,
                "barrier_policy": "exclusive_owner",
            },
        }
        unresolved = [{
            "unit_id": "__harness_structured_clause-1_1",
            "source_text": "standard",
            "operation": "unresolved",
            "owner_routes": [],
        }]
        correctly_routed = [{
            "unit_id": "unit-1",
            "source_text": "standard",
            "operation": "domain_request",
            "owner_routes": [{
                "owner": "performance",
                "group": "qps_profile",
            }],
        }]
        incorrectly_routed = [{
            "unit_id": "unit-1",
            "source_text": "standard",
            "operation": "domain_request",
            "owner_routes": [{
                "owner": "chain_rpc",
                "group": "chain_identity",
            }],
        }]

        self.assertTrue(_cross_domain_pending_errors(unresolved, state))
        self.assertEqual(
            _cross_domain_pending_errors(correctly_routed, state),
            (),
        )
        self.assertTrue(_cross_domain_pending_errors(incorrectly_routed, state))

    def test_registered_value_consultation_remains_read_only(self) -> None:
        from agent.harness.hierarchical_planner import (
            _cross_domain_pending_errors,
        )

        state = {
            "pending_question": {
                "group": "chain_identity",
                "value_domain": "researched_identity",
                "queue_barrier": True,
                "barrier_policy": "exclusive_owner",
            },
        }
        consultation = [{
            "unit_id": "unit-1",
            "source_text": "What does the standard profile mean?",
            "operation": "consultation",
            "owner_routes": [{
                "owner": "orientation",
                "group": "qps_profile",
            }],
        }]

        self.assertEqual(
            _cross_domain_pending_errors(consultation, state),
            (),
        )

    def test_stage_a_repairs_registered_cross_group_unresolved_unit(self) -> None:
        from agent.harness.hierarchical_planner import begin_semantic_partition
        from agent.harness.state import new_state

        state = new_state("cross-group-stage-a-repair", language="en")
        state["turn_index"] = 4
        state["active_group"] = "chain_identity"
        state["pending_question"] = {
            "id": "chain",
            "group": "chain_identity",
            "owner": "chain_rpc",
            "kind": "chain",
            "field": "chain",
            "manual_input_allowed": True,
            "options": [],
            "queue_barrier": True,
            "barrier_policy": "exclusive_owner",
            "value_domain": "researched_identity",
        }
        first = {
            "semantic_units": [{
                "unit_id": "unit-1",
                "clause_id": "clause-1",
                "source_text": "standard",
                "operation": "unresolved",
                "owner_routes": [],
                "reason": "another group has an exclusive pending question",
            }],
            "reason": "unresolved",
        }
        repaired = {
            "semantic_units": [{
                "unit_id": "unit-1",
                "clause_id": "clause-1",
                "source_text": "standard",
                "operation": "domain_request",
                "owner_routes": [{
                    "owner": "performance",
                    "group": "qps_profile",
                }],
                "reason": "registered QPS profile request",
            }],
            "reason": "routed",
        }

        with (
            patch(
                "agent.harness.hierarchical_planner.provider_from_config",
                return_value=object(),
            ),
            patch(
                "agent.harness.hierarchical_planner.request_semantic_compilation",
                side_effect=(
                    json.dumps(first),
                    json.dumps(repaired),
                ),
            ) as compiler,
            patch(
                "agent.harness.hierarchical_planner._review_stage_a_partition",
                return_value=((), (777,), frozenset()),
            ) as admission,
        ):
            result = begin_semantic_partition(state, "standard")

        self.assertEqual(result["status"], "compile_owner")
        self.assertEqual(result["stage_a_calls"], 2)
        self.assertEqual(result["admission_calls"], 1)
        admission.assert_called_once()
        self.assertTrue(all(
            call.kwargs["reasoning_mode"] == "disabled"
            for call in compiler.call_args_list
        ))
        self.assertEqual(
            result["owner_requests"],
            [{
                "owner": "performance",
                "unit_ids": ["unit-1"],
                "groups": ["qps_profile"],
            }],
        )
        repair_payload = compiler.call_args_list[1].kwargs["request_payload"]
        self.assertTrue(any(
            "registered cross-group semantic value" in error
            for error in repair_payload["contract_repair"]["validation_errors"]
        ))
        self.assertEqual(
            repair_payload["pending_barrier_contract"][
                "registered_cross_group_values"
            ],
            "route_to_registered_owner",
        )

    def test_local_pending_path_uses_registered_semantic_domains(self) -> None:
        from agent.harness.questions import answer_fits_pending

        region_question = {
            "group": "provider_deployment",
            "kind": "manual_value",
            "manual_input_allowed": True,
            "options": [],
            "validation": {"value_type": "scalar_token"},
        }
        endpoint_question = {
            "group": "endpoint_process",
            "kind": "url",
            "manual_input_allowed": True,
            "options": [],
            "validation": {"value_type": "url"},
        }

        for value in ("bnb", "ethereum", "quick", "real-node"):
            self.assertFalse(answer_fits_pending(value, region_question), value)
        self.assertTrue(answer_fits_pending("asia-east1", region_question))
        self.assertTrue(
            answer_fits_pending(
                "https://ethereum.example/rpc",
                endpoint_question,
            )
        )
        self.assertFalse(answer_fits_pending("en", region_question))

    def test_specialized_rpc_pending_requires_its_declared_input_shape(self) -> None:
        from agent.harness.domains.chain_rpc_questions import (
            _endpoint_validation_question,
        )
        from agent.harness.questions import answer_fits_pending
        from agent.harness.state import new_state

        state = new_state("specialized-pending-shape", language="en")
        state["custom_rpc"] = {
            "status": "needs_endpoint",
            "endpoint_ready": False,
        }
        endpoint_question = _endpoint_validation_question(state)
        self.assertIsNotNone(endpoint_question)
        self.assertFalse(
            answer_fits_pending("I need to switch to BNB", endpoint_question)
        )
        self.assertTrue(
            answer_fits_pending(
                "https://bsc.example/rpc",
                endpoint_question,
            )
        )
        self.assertFalse(answer_fits_pending(
            "switch to BNB and use https://bsc.example/rpc",
            endpoint_question,
        ))

        state["custom_rpc"] = {
            "status": "needs_method",
            "endpoint_ready": True,
        }
        method_question = _endpoint_validation_question(state)
        self.assertIsNotNone(method_question)
        self.assertFalse(answer_fits_pending("bnb", method_question))
        self.assertFalse(answer_fits_pending("quick", method_question))
        self.assertTrue(answer_fits_pending("eth_chainId", method_question))

        state["custom_rpc"] = {
            "status": "needs_schema_evidence",
            "endpoint_ready": True,
            "candidate_method": "eth_chainId",
        }
        evidence_question = _endpoint_validation_question(state)
        self.assertIsNotNone(evidence_question)
        self.assertFalse(
            answer_fits_pending("I need to switch to BNB", evidence_question)
        )
        self.assertTrue(answer_fits_pending(
            '{"jsonrpc":"2.0","method":"eth_chainId","params":[],"id":1}',
            evidence_question,
        ))
        self.assertTrue(answer_fits_pending(
            "method: eth_chainId\nparams:\n  - bnb",
            evidence_question,
        ))
        self.assertTrue(answer_fits_pending(
            "```yaml\nmethod: eth_chainId\nparams:\n  - bnb\n```",
            evidence_question,
        ))
        self.assertFalse(answer_fits_pending(
            '{"note":"switch to BNB"}',
            evidence_question,
        ))
        self.assertFalse(answer_fits_pending(
            'switch to BNB; evidence={"jsonrpc":"2.0","method":"eth_chainId","params":[],"id":1}',
            evidence_question,
        ))

    def test_final_admission_rejects_chain_request_as_rpc_endpoint(self) -> None:
        from agent.harness.domains.chain_rpc_questions import (
            _endpoint_validation_question,
        )
        from agent.harness.plan_coverage import segment_user_turn
        from agent.harness.semantic_admission import prepare_hierarchical_candidate
        from agent.harness.state import new_state

        text = "I need to switch to BNB"
        clauses = segment_user_turn(text)
        state = new_state("endpoint-cross-domain-final-admission", language="en")
        state["custom_rpc"] = {
            "status": "needs_endpoint",
            "endpoint_ready": False,
        }
        state["pending_question"] = _endpoint_validation_question(state)
        candidate = {
            "actions": [{
                "type": "rpc_catalog_command",
                "catalog_command": "set_endpoint",
                "rpc_endpoint": text,
                "source_evidence": text,
            }],
            "semantic_units": [{
                "unit_id": "unit-1",
                "clause_id": "clause-1",
                "source_text": text,
                "disposition": "action",
                "action_indexes": [0],
                "reason": "incorrect endpoint ownership",
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
            any("registered semantic value" in error for error in validation.errors),
            validation.errors,
        )

    def test_structured_pending_atom_does_not_inherit_sibling_config_review(
        self,
    ) -> None:
        from agent.harness.plan_coverage import segment_user_turn
        from agent.harness.semantic_admission import prepare_hierarchical_candidate
        from agent.harness.state import new_state

        text = '{"CHAIN":"solana","CLOUD_REGION":"us-1"}'
        clauses = segment_user_turn(text)
        state = new_state("structured-pending-sibling-config", language="en")
        state["active_group"] = "chain_identity"
        state["pending_question"] = {
            "id": "chain",
            "group": "chain_identity",
            "owner": "chain_rpc",
            "manual_input_allowed": True,
            "options": [],
            "value_domain": "researched_identity",
        }
        candidate = {
            "actions": [
                {
                    "type": "answer_pending",
                    "answer": "solana",
                    "source_evidence": "solana",
                },
                {
                    "type": "propose_config_values",
                    "source_format": "json",
                    "config_values": {"CLOUD_REGION": "us-1"},
                    "source_evidence": "us-1",
                },
            ],
            "semantic_units": [
                {
                    "unit_id": "chain-atom",
                    "clause_id": clauses[0].clause_id,
                    "start": 0,
                    "end": len(text),
                    "source_text": text,
                    "source_path": "CHAIN",
                    "disposition": "action",
                    "action_indexes": [0],
                    "reason": "answers the open chain question",
                },
                {
                    "unit_id": "region-atom",
                    "clause_id": clauses[0].clause_id,
                    "start": 0,
                    "end": len(text),
                    "source_text": text,
                    "source_path": "CLOUD_REGION",
                    "disposition": "action",
                    "action_indexes": [1],
                    "reason": "routes configuration through review",
                },
            ],
            "reason": "structured DemandAtoms preserve independent ownership",
        }

        _prepared, validation = prepare_hierarchical_candidate(
            json.dumps(candidate),
            state,
            clauses,
            pending_choice_unit_ids=frozenset({"chain-atom"}),
        )

        self.assertFalse(
            any(
                "bypasses structured configuration review" in error
                for error in validation.errors
            ),
            validation.errors,
        )

    def test_multi_unit_action_accepts_exact_ordered_source_span(self) -> None:
        from agent.harness.plan_coverage import segment_user_turn
        from agent.harness.semantic_admission import prepare_hierarchical_candidate
        from agent.harness.state import new_state

        text = "region 是 us-1，zone 是 us-1-z；"
        region_source = "region 是 us-1，"
        zone_source = "zone 是 us-1-z；"
        candidate = {
            "actions": [{
                "type": "propose_config_values",
                "source_format": "prose",
                "config_values": {
                    "CLOUD_REGION": "us-1",
                    "CLOUD_ZONE": "us-1-z",
                },
                "source_evidence": text,
            }],
            "semantic_units": [
                {
                    "unit_id": "region-unit",
                    "clause_id": "clause-1",
                    "start": 0,
                    "end": len(region_source),
                    "source_text": region_source,
                    "disposition": "action",
                    "action_indexes": [0],
                    "reason": "region value",
                },
                {
                    "unit_id": "zone-unit",
                    "clause_id": "clause-1",
                    "start": len(region_source),
                    "end": len(text),
                    "source_text": zone_source,
                    "disposition": "action",
                    "action_indexes": [0],
                    "reason": "zone value",
                },
            ],
        }

        _prepared, validation = prepare_hierarchical_candidate(
            json.dumps(candidate, ensure_ascii=False),
            new_state("multi-unit-source-span", language="zh"),
            segment_user_turn(text),
            pending_choice_unit_ids=frozenset(),
        )

        self.assertTrue(validation.valid, validation.errors)

    def test_multi_unit_source_span_rejects_foreign_or_reordered_text(self) -> None:
        from agent.harness.semantic_admission import _source_evidence_is_grounded

        units = [
            {"unit_id": "one", "source_text": "alpha;"},
            {"unit_id": "two", "source_text": "beta"},
        ]

        self.assertTrue(_source_evidence_is_grounded("alpha;beta", units))
        self.assertFalse(
            _source_evidence_is_grounded("alpha;foreign;beta", units)
        )
        self.assertFalse(_source_evidence_is_grounded("betaalpha;", units))
        self.assertFalse(_source_evidence_is_grounded("alpha;", units[1:]))

    def test_final_admission_allows_structured_rpc_endpoint_and_evidence(self) -> None:
        from agent.harness.domains.chain_rpc_questions import (
            _endpoint_validation_question,
        )
        from agent.harness.hierarchical_planner import _cross_domain_pending_errors
        from agent.harness.plan_coverage import segment_user_turn
        from agent.harness.semantic_admission import prepare_hierarchical_candidate
        from agent.harness.state import new_state

        for status, command, argument, text in (
            (
                "needs_endpoint",
                "set_endpoint",
                "rpc_endpoint",
                "https://bsc.example/rpc",
            ),
            (
                "needs_method",
                "set_method",
                "rpc_method",
                "eth_chainId",
            ),
            (
                "needs_schema_evidence",
                "append_evidence",
                "rpc_schema_evidence",
                '{"jsonrpc":"2.0","method":"eth_chainId","params":[],"id":1}',
            ),
            (
                "needs_schema_evidence",
                "append_evidence",
                "rpc_schema_evidence",
                "method: eth_chainId\nparams:\n  - bnb",
            ),
            (
                "needs_schema_evidence",
                "append_evidence",
                "rpc_schema_evidence",
                "```yaml\nmethod: eth_chainId\nparams:\n  - bnb\n```",
            ),
        ):
            with self.subTest(status=status):
                state = new_state(f"structured-{status}", language="en")
                state["custom_rpc"] = {
                    "status": status,
                    "endpoint_ready": status != "needs_endpoint",
                    "candidate_method": "eth_chainId",
                }
                state["pending_question"] = _endpoint_validation_question(state)
                clauses = segment_user_turn(text)
                self.assertEqual(
                    _cross_domain_pending_errors(
                        [{
                            "unit_id": "unit-1",
                            "source_text": text,
                            "operation": "pending_answer",
                        }],
                        state,
                    ),
                    (),
                )
                candidate = {
                    "actions": [{
                        "type": "rpc_catalog_command",
                        "catalog_command": command,
                        argument: text,
                        "source_evidence": text,
                    }],
                    "semantic_units": [{
                        "unit_id": "unit-1",
                        "clause_id": "clause-1",
                        "source_text": text,
                        "disposition": "action",
                        "action_indexes": [0],
                        "reason": "typed RPC input",
                    }],
                }

                _prepared, validation = prepare_hierarchical_candidate(
                    json.dumps(candidate),
                    state,
                    clauses,
                    pending_choice_unit_ids=frozenset({"unit-1"}),
                )

                self.assertTrue(validation.valid, validation.errors)

    def test_rpc_evidence_admission_rejects_non_wire_json_domain_detour(self) -> None:
        from agent.harness.domains.chain_rpc_questions import (
            _endpoint_validation_question,
        )
        from agent.harness.hierarchical_planner import _cross_domain_pending_errors
        from agent.harness.plan_coverage import segment_user_turn
        from agent.harness.semantic_admission import prepare_hierarchical_candidate
        from agent.harness.state import new_state

        text = '{"note":"switch to BNB"}'
        state = new_state("non-wire-json-evidence", language="en")
        state["custom_rpc"] = {
            "status": "needs_schema_evidence",
            "endpoint_ready": True,
            "candidate_method": "eth_chainId",
        }
        state["pending_question"] = _endpoint_validation_question(state)
        partition = [{
            "unit_id": "unit-1",
            "source_text": text,
            "operation": "pending_answer",
        }]
        self.assertTrue(_cross_domain_pending_errors(partition, state))

        candidate = {
            "actions": [{
                "type": "rpc_catalog_command",
                "catalog_command": "append_evidence",
                "rpc_schema_evidence": text,
                "source_evidence": text,
            }],
            "semantic_units": [{
                "unit_id": "unit-1",
                "clause_id": "clause-1",
                "source_text": text,
                "disposition": "action",
                "action_indexes": [0],
                "reason": "incorrect evidence ownership",
            }],
        }
        _prepared, validation = prepare_hierarchical_candidate(
            json.dumps(candidate),
            state,
            segment_user_turn(text),
            pending_choice_unit_ids=frozenset({"unit-1"}),
        )
        self.assertFalse(validation.valid)
        self.assertTrue(
            any("registered semantic value" in error for error in validation.errors),
            validation.errors,
        )

    def test_language_domain_cannot_be_consumed_by_environment_pending(self) -> None:
        from agent.harness.hierarchical_planner import _cross_domain_pending_errors
        from agent.harness.plan_coverage import segment_user_turn
        from agent.harness.semantic_admission import prepare_hierarchical_candidate
        from agent.harness.state import new_state

        text = "en"
        state = new_state("language-domain-ownership", language="zh")
        state["pending_question"] = {
            "id": "CLOUD_REGION",
            "group": "provider_deployment",
            "kind": "manual_value",
            "field": "CLOUD_REGION",
            "manual_input_allowed": True,
            "options": [],
            "accepted_action_types": ["answer_pending"],
            "validation": {"value_type": "scalar_token"},
        }
        wrong_partition = [{
            "unit_id": "unit-1",
            "source_text": text,
            "operation": "pending_answer",
        }]
        self.assertTrue(_cross_domain_pending_errors(wrong_partition, state))

        clauses = segment_user_turn(text)
        wrong_candidate = {
            "actions": [{
                "type": "answer_pending",
                "answer": text,
                "source_evidence": text,
            }],
            "semantic_units": [{
                "unit_id": "unit-1",
                "clause_id": "clause-1",
                "source_text": text,
                "disposition": "action",
                "action_indexes": [0],
                "reason": "incorrect environment ownership",
            }],
        }
        _prepared, validation = prepare_hierarchical_candidate(
            json.dumps(wrong_candidate),
            state,
            clauses,
            pending_choice_unit_ids=frozenset({"unit-1"}),
        )
        self.assertFalse(validation.valid)

        language_candidate = {
            "actions": [{
                "type": "set_response_language",
                "language": "en",
                "source_evidence": "en",
            }],
            "semantic_units": [{
                "unit_id": "unit-1",
                "clause_id": "clause-1",
                "source_text": text,
                "disposition": "action",
                "action_indexes": [0],
                "reason": "explicit language detour",
            }],
        }
        _prepared, validation = prepare_hierarchical_candidate(
            json.dumps(language_candidate),
            state,
            clauses,
            pending_choice_unit_ids=frozenset(),
        )
        self.assertTrue(validation.valid, validation.errors)

    def test_stage_a_repairs_chain_request_misassigned_to_environment_pending(
        self,
    ) -> None:
        from agent.harness.plan_coverage import PlanCoverageResult
        from agent.harness.state import new_state
        from tests.agent_live.graph_turn import (
            resolve_product_action_queue_for_test as resolve_product_action_queue,
        )

        text = "我需要换成 BNB"
        state = new_state("stage-a-cross-domain-repair", language="zh")
        state["pending_question"] = {
            "id": "CLOUD_REGION",
            "group": "provider_deployment",
            "kind": "manual_value",
            "field": "CLOUD_REGION",
            "manual_input_allowed": True,
            "options": [],
            "accepted_action_types": ["answer_pending"],
            "validation": {"value_type": "scalar_token"},
        }
        wrong_partition = {
            "semantic_units": [{
                "unit_id": "unit-1",
                "clause_id": "clause-1",
                "source_text": text,
                "operation": "pending_answer",
                "owner_routes": [{
                    "owner": "coordinator",
                    "group": "provider_deployment",
                }],
                "reason": "incorrectly assigned to active pending",
            }],
        }
        repaired_partition = {
            "semantic_units": [{
                "unit_id": "unit-1",
                "clause_id": "clause-1",
                "source_text": text,
                "operation": "domain_request",
                "owner_routes": [{
                    "owner": "chain_rpc",
                    "group": "chain_identity",
                }],
                "reason": "chain replacement request",
            }],
        }
        owner_document = {
            "actions": [{
                "type": "choose_chain",
                "chain_text": "BNB",
                "source_evidence": "BNB",
            }],
            "bindings": [{
                "unit_id": "unit-1",
                "action_indexes": [0],
                "disposition": "action",
                "reason": "compiled chain replacement",
            }],
        }

        with (
            patch(
                "agent.harness.hierarchical_planner.provider_from_config",
                return_value=object(),
            ),
            patch(
                "agent.harness.hierarchical_planner.request_semantic_compilation",
                side_effect=[
                    json.dumps(wrong_partition, ensure_ascii=False),
                    json.dumps(repaired_partition, ensure_ascii=False),
                ],
            ) as stage_a,
            patch(
                "agent.harness.hierarchical_planner._compile_owner_document",
                return_value=(owner_document, (), (100,)),
            ) as owner_compiler,
            patch(
                "agent.harness.hierarchical_planner._review_stage_a_partition",
                return_value=((), (333,), frozenset()),
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
                return_value={"actions": [{"type": "choose_chain"}]},
            ),
        ):
            result = resolve_product_action_queue(state, text)

        self.assertEqual(stage_a.call_count, 2)
        repair_payload = stage_a.call_args_list[1].kwargs["request_payload"]
        self.assertTrue(repair_payload.get("contract_repair"))
        self.assertTrue(any(
            "registered semantic value" in error
            for error in repair_payload["contract_repair"]["validation_errors"]
        ))
        owner_compiler.assert_called_once()
        self.assertEqual(
            owner_compiler.call_args.args[1:3],
            ("chain_rpc", frozenset({"chain_identity"})),
        )
        self.assertEqual(result["actions"], [{"type": "choose_chain"}])
        self.assertEqual(result["planner_metrics"]["stage_a_calls"], 2)

    def test_environment_pending_rejects_known_chain_identity_prose(self) -> None:
        from agent.harness.hierarchical_planner import _cross_domain_pending_errors

        state = {
            "pending_question": {
                "group": "provider_deployment",
                "value_domain": "environment_value",
            },
        }
        partition = [{
            "unit_id": "unit-1",
            "source_text": "我需要换成 BNB",
            "operation": "pending_answer",
        }]

        errors = _cross_domain_pending_errors(partition, state)

        self.assertTrue(errors)
        self.assertIn("chain_identity", errors[0])
        self.assertIn("bnb", errors[0].lower())

    def test_chain_pending_owns_known_chain_identity(self) -> None:
        from agent.harness.hierarchical_planner import _cross_domain_pending_errors

        state = {
            "pending_question": {
                "group": "chain_identity",
                "value_domain": "researched_identity",
            },
        }
        partition = [{
            "unit_id": "unit-1",
            "source_text": "我需要换成 BNB",
            "operation": "pending_answer",
        }]

        self.assertEqual(_cross_domain_pending_errors(partition, state), ())

    def test_final_admission_rejects_chain_prose_as_environment_value(self) -> None:
        from agent.harness.plan_coverage import segment_user_turn
        from agent.harness.semantic_admission import prepare_hierarchical_candidate
        from agent.harness.state import new_state

        text = "我需要换成 BNB"
        clauses = segment_user_turn(text)
        state = new_state("cross-domain-final-admission", language="zh")
        state["pending_question"] = {
            "id": "CLOUD_REGION",
            "group": "provider_deployment",
            "kind": "text",
            "field": "CLOUD_REGION",
            "manual_input_allowed": True,
            "options": [],
            "accepted_action_types": ["answer_pending"],
            "validation": {
                "value_type": "bounded_text",
                "max_length": 128,
            },
        }
        candidate = {
            "actions": [{
                "type": "answer_pending",
                "answer": text,
                "source_evidence": text,
            }],
            "semantic_units": [{
                "unit_id": "unit-1",
                "clause_id": "clause-1",
                "source_text": text,
                "disposition": "action",
                "action_indexes": [0],
                "reason": "incorrect environment ownership",
            }],
        }

        _prepared, validation = prepare_hierarchical_candidate(
            json.dumps(candidate, ensure_ascii=False),
            state,
            clauses,
            pending_choice_unit_ids=frozenset({"unit-1"}),
        )

        self.assertFalse(validation.valid)
        self.assertTrue(
            any("registered semantic value" in error for error in validation.errors),
            validation.errors,
        )

    def test_final_admission_allows_environment_value_without_domain_conflict(self) -> None:
        from agent.harness.plan_coverage import segment_user_turn
        from agent.harness.semantic_admission import prepare_hierarchical_candidate
        from agent.harness.state import new_state

        text = "asia-east1"
        clauses = segment_user_turn(text)
        state = new_state("environment-value-final-admission", language="en")
        state["pending_question"] = {
            "id": "CLOUD_REGION",
            "group": "provider_deployment",
            "kind": "text",
            "field": "CLOUD_REGION",
            "manual_input_allowed": True,
            "options": [],
            "accepted_action_types": ["answer_pending"],
            "validation": {
                "value_type": "bounded_text",
                "max_length": 128,
            },
        }
        candidate = {
            "actions": [{
                "type": "answer_pending",
                "answer": text,
                "source_evidence": text,
            }],
            "semantic_units": [{
                "unit_id": "unit-1",
                "clause_id": "clause-1",
                "source_text": text,
                "disposition": "action",
                "action_indexes": [0],
                "reason": "valid environment ownership",
            }],
        }

        _prepared, validation = prepare_hierarchical_candidate(
            json.dumps(candidate),
            state,
            clauses,
            pending_choice_unit_ids=frozenset({"unit-1"}),
        )

        self.assertTrue(validation.valid, validation.errors)

    def test_endpoint_pending_allows_chain_name_inside_typed_url(self) -> None:
        from agent.harness.hierarchical_planner import _cross_domain_pending_errors
        from agent.harness.plan_coverage import segment_user_turn
        from agent.harness.semantic_admission import prepare_hierarchical_candidate
        from agent.harness.state import new_state

        text = "https://ethereum.example/rpc"
        clauses = segment_user_turn(text)
        state = new_state("typed-url-domain-value", language="en")
        state["pending_question"] = {
            "id": "LOCAL_RPC_URL",
            "group": "endpoint_process",
            "kind": "url",
            "field": "LOCAL_RPC_URL",
            "manual_input_allowed": True,
            "options": [],
            "accepted_action_types": ["answer_pending"],
            "validation": {"value_type": "url"},
        }
        partition = [{
            "unit_id": "unit-1",
            "source_text": text,
            "operation": "pending_answer",
        }]
        candidate = {
            "actions": [{
                "type": "answer_pending",
                "answer": text,
                "source_evidence": text,
            }],
            "semantic_units": [{
                "unit_id": "unit-1",
                "clause_id": "clause-1",
                "source_text": text,
                "disposition": "action",
                "action_indexes": [0],
                "reason": "typed endpoint",
            }],
        }

        self.assertEqual(_cross_domain_pending_errors(partition, state), ())
        _prepared, validation = prepare_hierarchical_candidate(
            json.dumps(candidate),
            state,
            clauses,
            pending_choice_unit_ids=frozenset({"unit-1"}),
        )
        self.assertTrue(validation.valid, validation.errors)

    def test_pending_option_owns_registered_value_for_its_declared_action(self) -> None:
        from agent.harness.domains.orientation import opening_question
        from agent.harness.hierarchical_planner import _cross_domain_pending_errors
        from agent.harness.state import new_state

        state = new_state("declared-option-domain-value", language="en")
        state["pending_question"] = opening_question(state)
        partition = [{
            "unit_id": "unit-1",
            "source_text": "fake-node",
            "operation": "pending_answer",
        }]

        self.assertEqual(_cross_domain_pending_errors(partition, state), ())

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
            any("registered semantic value" in error for error in validation.errors),
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
            "owner_routes": [{"owner": "orientation", "group": ""}],
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

        semantic_options = payload["owner_state"]["pending_question"]["options"]
        self.assertEqual(
            [option["value"] for option in semantic_options],
            ["alpha", "unknown"],
        )
        self.assertEqual(
            [option["semantic_labels"] for option in semantic_options],
            [["First"], ["None / unsure"]],
        )
        self.assertTrue(all(
            "semantic_labels" not in option
            for option in state["pending_question"]["options"]
        ))
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

    def test_administrative_reset_route_reaches_orientation_schema(self) -> None:
        from agent.harness.hierarchical_planner import (
            _stage_b_payload,
            _validate_partition_document,
        )
        from agent.harness.plan_coverage import segment_user_turn
        from agent.harness.state import new_state

        text = "Clear the configuration and start over."
        clauses = segment_user_turn(text)
        partition = {
            "semantic_units": [{
                "unit_id": "unit-1",
                "clause_id": clauses[0].clause_id,
                "source_text": text,
                "operation": "administrative",
                "owner_routes": [{
                    "owner": "orientation",
                    "group": "opening",
                }],
                "reason": "explicit session reset",
            }],
            "reason": "reset",
        }

        validated, errors = _validate_partition_document(
            json.dumps(partition),
            clauses,
        )
        self.assertFalse(errors, errors)
        payload = _stage_b_payload(
            new_state("administrative-reset", language="en"),
            "orientation",
            frozenset({"opening"}),
            validated,
            ("unit-1",),
        )
        self.assertIn(
            "request_session_reset",
            {row["type"] for row in payload["owner_action_schema"]},
        )

    def test_administrative_queue_route_reaches_coordinator_schema(self) -> None:
        from agent.harness.hierarchical_planner import (
            _stage_b_payload,
            _validate_partition_document,
        )
        from agent.harness.plan_coverage import segment_user_turn
        from agent.harness.state import new_state

        text = "Queue a real-node benchmark for later."
        clauses = segment_user_turn(text)
        partition = {
            "semantic_units": [{
                "unit_id": "unit-1",
                "clause_id": clauses[0].clause_id,
                "source_text": text,
                "operation": "administrative",
                "owner_routes": [{
                    "owner": "coordinator",
                    "group": "",
                }],
                "reason": "queue a later workflow goal",
            }],
            "reason": "queue",
        }

        validated, errors = _validate_partition_document(
            json.dumps(partition),
            clauses,
        )
        self.assertFalse(errors, errors)
        payload = _stage_b_payload(
            new_state("administrative-queue", language="en"),
            "coordinator",
            frozenset(),
            validated,
            ("unit-1",),
        )
        self.assertIn(
            "queue_workflow_goal",
            {row["type"] for row in payload["owner_action_schema"]},
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
        self.assertIn("present idempotent domain_request", partition_prompt)
        self.assertIn("will be supplied later is temporal context", partition_prompt)
        self.assertIn("requirements, format, meaning, or validity", partition_prompt)
        self.assertIn("equality with current state", admission_prompt)
        self.assertIn("A unit has exactly one operation", partition_prompt)
        self.assertIn("operation-specific units", admission_prompt)
        self.assertIn("each demand has its own exact substring", partition_prompt)
        self.assertIn("Distinct exact substrings", admission_prompt)
        self.assertIn("contract_proven_pending_prefixes", partition_prompt)
        self.assertIn("Reject a planner reason that invents", admission_prompt)
        self.assertIn(
            "hypothetical, counterfactual, consequence",
            partition_prompt,
        )
        self.assertIn(
            "contains no present authorization",
            admission_prompt,
        )
        self.assertIn(
            "even when the evidence will be pasted later",
            admission_prompt,
        )

    def test_stage_b_contract_compiles_explicit_existing_value_idempotently(
        self,
    ) -> None:
        from agent.harness.hierarchical_planner import _stage_b_prompt

        prompt = _stage_b_prompt("chain_rpc")

        self.assertIn("keep, reuse, or reconfirm", prompt)
        self.assertIn("idempotent proposal", prompt)
        self.assertIn("would not change state", prompt)
        self.assertIn("promise to supply a value later", prompt)

    def test_stage_a_preserves_route_without_compiler_as_unresolved_atom(
        self,
    ) -> None:
        from agent.harness.hierarchical_planner import (
            _validate_partition_document,
        )
        from agent.harness.plan_coverage import segment_user_turn

        text = "Tell me what the endpoint must support."
        clauses = segment_user_turn(text)
        document = {
            "semantic_units": [{
                "unit_id": "unit-1",
                "clause_id": clauses[0].clause_id,
                "source_text": text,
                "operation": "domain_request",
                "owner_routes": [{
                    "owner": "orientation",
                    "group": "endpoint_process",
                }],
                "reason": "incorrectly treated consultation as mutation",
            }],
            "reason": "invalid route",
        }

        partition, errors = _validate_partition_document(
            json.dumps(document),
            clauses,
        )

        self.assertEqual(errors, ())
        self.assertEqual(partition[0]["operation"], "unresolved")
        self.assertEqual(partition[0]["owner_routes"], [])

    def test_stage_a_payload_projects_authoritative_operation_purposes(
        self,
    ) -> None:
        from agent.harness.action_registry import SEMANTIC_OPERATION_PURPOSES
        from agent.harness.hierarchical_planner import _stage_a_payload
        from agent.harness.plan_coverage import segment_user_turn

        payload = _stage_a_payload(
            {},
            "Can you analyze a log if I paste it next?",
            segment_user_turn("Can you analyze a log if I paste it next?"),
        )

        self.assertEqual(
            payload["universal_operation_purposes"],
            dict(SEMANTIC_OPERATION_PURPOSES),
        )

    def test_stage_a_payload_normalizes_complete_routing_catalog(self) -> None:
        from agent.harness.context import action_schema
        from agent.harness.hierarchical_planner import _stage_a_payload
        from agent.harness.plan_coverage import segment_user_turn

        text = "Configure several independent groups and explain one mode."
        payload = _stage_a_payload({}, text, segment_user_turn(text))
        expected = [
            {
                "action_type": action["type"],
                "owner": action["owner"],
                "purpose": action["purpose"],
                "semantic_operations": action["semantic_operations"],
                "route_groups": action["route_groups"],
            }
            for action in action_schema()
        ]

        self.assertEqual(payload["routing_purposes"], expected)
        self.assertTrue(payload["groups"])
        self.assertTrue(
            all("routing_purposes" not in group for group in payload["groups"])
        )
        self.assertLess(
            len(json.dumps(payload, ensure_ascii=False).encode("utf-8")),
            60_000,
        )

    def test_stage_a_projects_closed_consultation_topic_purposes(self) -> None:
        from agent.harness.action_registry import CONSULTATION_TOPIC_PURPOSES
        from agent.harness.hierarchical_planner import _stage_a_payload
        from agent.harness.plan_coverage import segment_user_turn

        text = "What is the current template workload?"
        payload = _stage_a_payload({}, text, segment_user_turn(text))
        orientation = payload["universal_owner_action_purposes"][
            "consultation"
        ]["orientation"]
        answer = next(
            row
            for row in orientation
            if row["action_type"] == "answer_opening_question"
        )
        self.assertEqual(
            answer["consultation_topic_purposes"],
            dict(CONSULTATION_TOPIC_PURPOSES),
        )

    def test_stage_b_projects_consultation_topics_without_consuming_pending(
        self,
    ) -> None:
        from agent.harness.action_registry import CONSULTATION_TOPIC_PURPOSES
        from agent.harness.hierarchical_planner import _stage_b_payload

        pending = {
            "id": "rpc_mode",
            "group": "workload_rpc",
            "owner": "chain_rpc",
            "field": "rpc_mode",
            "kind": "choice",
            "options": [
                {"id": "1", "label": "single", "value": "single"},
                {"id": "2", "label": "mixed", "value": "mixed"},
            ],
        }
        state = {
            "language": "en",
            "active_group": "workload_rpc",
            "pending_question": pending,
            "chain_identity": {"canonical": "bsc", "status": "confirmed"},
        }
        partition = [{
            "unit_id": "unit-1",
            "clause_id": "clause-1",
            "source_text": "What is the current template workload?",
            "operation": "consultation",
            "owner_routes": [{
                "owner": "orientation",
                "group": "workload_rpc",
            }],
            "reason": "read-only workload question",
        }]

        payload = _stage_b_payload(
            state,
            "orientation",
            frozenset({"workload_rpc"}),
            partition,
            ("unit-1",),
        )
        answer = next(
            row
            for row in payload["owner_action_schema"]
            if row["type"] == "answer_opening_question"
        )

        self.assertEqual(answer["topic_purposes"], dict(CONSULTATION_TOPIC_PURPOSES))
        self.assertEqual(payload["pending_question"], pending)
        self.assertEqual(payload["semantic_units"], partition)

    def test_production_partition_rechecks_when_stage_a_admission_rejects(
        self,
    ) -> None:
        from agent.harness.hierarchical_planner import begin_semantic_partition

        stage_a = {
            "semantic_units": [{
                "unit_id": "unit-1",
                "clause_id": "clause-1",
                "source_text": "If I reset, what would be cleared?",
                "operation": "pending_answer",
                "owner_routes": [{
                    "owner": "coordinator",
                    "group": "opening",
                }],
                "reason": "incorrectly treats a hypothetical as reset approval",
            }],
            "reason": "candidate partition",
        }
        rejected = {
            "unit_verdicts": [{
                "unit_id": "unit-1",
                "verdict": "unresolved",
                "supports_unit_id": "",
                "reason": (
                    "a hypothetical consequence question is not a present "
                    "reset authorization"
                ),
            }],
            "clause_verdicts": [{
                "clause_id": "clause-1",
                "verdict": "unresolved",
                "omitted_owner_routes": [],
                "reason": (
                    "the source asks what would happen and does not authorize "
                    "the pending reset"
                ),
            }],
            "reason": "the candidate partition is not semantically admissible",
        }
        independent = {
            "semantic_units": [{
                "unit_id": "unit-1",
                "clause_id": "clause-1",
                "source_text": "If I reset, what would be cleared?",
                "operation": "consultation",
                "owner_routes": [{
                    "owner": "orientation",
                    "group": "opening",
                }],
                "reason": "read-only reset consequence question",
            }],
            "reason": "independent consultation partition",
        }
        admitted = {
            "unit_verdicts": [{
                "unit_id": "unit-1",
                "verdict": "complete",
                "supports_unit_id": "",
                "reason": "the consultation preserves the complete question",
            }],
            "clause_verdicts": [{
                "clause_id": "clause-1",
                "verdict": "complete",
                "omitted_owner_routes": [],
                "reason": "the complete question has a safe read-only route",
            }],
            "reason": "independent proposal is admissible",
        }
        state = {
            "language": "en",
            "active_group": "opening",
            "pending_question": {
                "id": "resume",
                "group": "opening",
                "owner": "coordinator",
                "kind": "choice",
                "options": [{"value": "reset", "label": "Reset"}],
                "manual_input_allowed": False,
            },
        }

        with (
            patch(
                "agent.harness.hierarchical_planner.provider_from_config",
                return_value=object(),
            ),
            patch(
                "agent.harness.hierarchical_planner.request_semantic_compilation",
                side_effect=[
                    json.dumps(stage_a),
                    json.dumps(rejected),
                    json.dumps(independent),
                    json.dumps(admitted),
                ],
            ) as compiler,
        ):
            result = begin_semantic_partition(
                state,
                "If I reset, what would be cleared?",
            )

        self.assertEqual(compiler.call_count, 4)
        self.assertIn(
            "Stage A coverage authority",
            compiler.call_args_list[1].kwargs["system_prompt"],
        )
        self.assertEqual(result["status"], "compile_owner")
        self.assertEqual(result["admission_calls"], 2)
        self.assertEqual(result["source_partition"][0]["operation"], "consultation")

    def test_inconsistent_wrapper_verdict_gets_one_admission_repair(
        self,
    ) -> None:
        from agent.harness.hierarchical_planner import begin_semantic_partition

        cases = (
            (
                "fake-node 测试",
                "fake-node",
                "测试",
                "chain_rpc",
                "target_mode",
            ),
            (
                "Use quick profile",
                "quick",
                "profile",
                "performance",
                "qps_profile",
            ),
        )
        state = {
            "language": "zh",
            "active_group": "chain_identity",
            "target_mode": "real-node",
            "pending_question": {
                "id": "chain",
                "group": "chain_identity",
                "owner": "chain_rpc",
                "kind": "chain",
                "manual_input_allowed": True,
                "queue_barrier": True,
                "barrier_policy": "exclusive_owner",
                "value_domain": "researched_identity",
            },
        }
        for text, value, wrapper, owner, group in cases:
            with self.subTest(text=text):
                initial_partition = {
                    "semantic_units": [
                        {
                            "unit_id": "unit-1",
                            "clause_id": "clause-1",
                            "source_text": value,
                            "operation": "domain_request",
                            "owner_routes": [{
                                "owner": owner,
                                "group": group,
                            }],
                            "reason": "registered value",
                        },
                        {
                            "unit_id": "unit-2",
                            "clause_id": "clause-1",
                            "source_text": wrapper,
                            "operation": "unresolved",
                            "owner_routes": [],
                            "reason": "wrapper was split incorrectly",
                        },
                    ],
                    "reason": "candidate partition",
                }
                rejected = {
                    "unit_verdicts": [
                        {
                            "unit_id": "unit-1",
                            "verdict": "complete",
                            "supports_unit_id": "",
                            "reason": "registered value is represented",
                        },
                        {
                            "unit_id": "unit-2",
                            "verdict": "unresolved",
                            "supports_unit_id": "",
                            "reason": "wrapper is not independently actionable",
                        },
                    ],
                    "clause_verdicts": [{
                        "clause_id": "clause-1",
                        "verdict": "complete",
                        "omitted_owner_routes": [],
                        "reason": "no sibling demand is omitted",
                    }],
                    "reason": "one wrapper unit remains unresolved",
                }
                accepted = {
                    "unit_verdicts": [
                        {
                            "unit_id": "unit-1",
                            "verdict": "complete",
                            "supports_unit_id": "",
                            "reason": "the complete demand is represented",
                        },
                        {
                            "unit_id": "unit-2",
                            "verdict": "redundant",
                            "supports_unit_id": "unit-1",
                            "reason": "the wrapper adds no independent demand",
                        },
                    ],
                    "clause_verdicts": [{
                        "clause_id": "clause-1",
                        "verdict": "complete",
                        "omitted_owner_routes": [],
                        "reason": "no demand is omitted",
                    }],
                    "reason": "partition is complete",
                }

                with (
                    patch(
                        "agent.harness.hierarchical_planner.provider_from_config",
                        return_value=object(),
                    ),
                    patch(
                        "agent.harness.hierarchical_planner.request_semantic_compilation",
                        side_effect=[
                            json.dumps(initial_partition),
                            json.dumps(rejected),
                            json.dumps(accepted),
                        ],
                    ) as compiler,
                ):
                    result = begin_semantic_partition(state, text)

                self.assertEqual(result["status"], "compile_owner")
                self.assertEqual(result["stage_a_calls"], 1)
                self.assertEqual(result["admission_calls"], 2)
                contract_repair = (
                    compiler.call_args_list[2]
                    .kwargs["request_payload"]["contract_repair"]
                )
                self.assertTrue(any(
                    "internally inconsistent" in error
                    for error in contract_repair["validation_errors"]
                ))
                source_by_id = {
                    unit["unit_id"]: unit
                    for unit in result["source_partition"]
                }
                self.assertEqual(
                    source_by_id["unit-2"]["operation"],
                    "context",
                )
                self.assertNotIn(
                    "unit-2",
                    {
                        unit["unit_id"]
                        for unit in result["routed_partition"]
                    },
                )

    def test_registered_value_consultation_remains_read_only(
        self,
    ) -> None:
        from agent.harness.hierarchical_planner import begin_semantic_partition

        text = "What does fake-node mean?"
        partition = {
            "semantic_units": [{
                "unit_id": "unit-1",
                "clause_id": "clause-1",
                "source_text": text,
                "operation": "consultation",
                "owner_routes": [{
                    "owner": "orientation",
                    "group": "target_mode",
                }],
                "reason": "read-only explanation request",
            }],
            "reason": "consultation partition",
        }
        accepted = {
            "unit_verdicts": [{
                "unit_id": "unit-1",
                "verdict": "complete",
                "supports_unit_id": "",
                "reason": "the question is represented",
            }],
            "clause_verdicts": [{
                "clause_id": "clause-1",
                "verdict": "complete",
                "omitted_owner_routes": [],
                "reason": "no demand is omitted",
            }],
            "reason": "partition is complete",
        }
        state = {
            "active_group": "chain_identity",
            "pending_question": {
                "id": "chain",
                "group": "chain_identity",
                "owner": "chain_rpc",
                "kind": "chain",
                "manual_input_allowed": True,
                "queue_barrier": True,
                "barrier_policy": "exclusive_owner",
            },
        }

        with (
            patch(
                "agent.harness.hierarchical_planner.provider_from_config",
                return_value=object(),
            ),
            patch(
                "agent.harness.hierarchical_planner.request_semantic_compilation",
                side_effect=[json.dumps(partition), json.dumps(accepted)],
            ) as compiler,
        ):
            result = begin_semantic_partition(state, text)

        self.assertEqual(result["status"], "compile_owner")
        self.assertEqual(result["stage_a_calls"], 1)
        self.assertEqual(result["admission_calls"], 1)
        self.assertEqual(compiler.call_count, 2)

    def test_rejected_consultation_fails_closed(
        self,
    ) -> None:
        from agent.harness.hierarchical_planner import begin_semantic_partition

        text = "What does fake-node mean?"
        partition = {
            "semantic_units": [{
                "unit_id": "unit-1",
                "clause_id": "clause-1",
                "source_text": text,
                "operation": "consultation",
                "owner_routes": [{
                    "owner": "orientation",
                    "group": "target_mode",
                }],
                "reason": "read-only explanation request",
            }],
            "reason": "consultation partition",
        }
        rejected = {
            "unit_verdicts": [{
                "unit_id": "unit-1",
                "verdict": "unresolved",
                "supports_unit_id": "",
                "reason": "reviewer rejected the consultation",
            }],
            "clause_verdicts": [{
                "clause_id": "clause-1",
                "verdict": "unresolved",
                "omitted_owner_routes": [],
                "reason": "the consultation meaning is unresolved",
            }],
            "reason": "semantic rejection",
        }
        admitted = {
            "unit_verdicts": [{
                "unit_id": "unit-1",
                "verdict": "complete",
                "supports_unit_id": "",
                "reason": "the source is a complete read-only question",
            }],
            "clause_verdicts": [{
                "clause_id": "clause-1",
                "verdict": "complete",
                "omitted_owner_routes": [],
                "reason": "the consultation route preserves the clause",
            }],
            "reason": "independent semantic review admits the route",
        }
        state = {
            "active_group": "chain_identity",
            "pending_question": {
                "id": "chain",
                "group": "chain_identity",
                "owner": "chain_rpc",
                "kind": "chain",
                "manual_input_allowed": True,
            },
        }

        with (
            patch(
                "agent.harness.hierarchical_planner.provider_from_config",
                return_value=object(),
            ),
            patch(
                "agent.harness.hierarchical_planner.request_semantic_compilation",
                side_effect=[
                    json.dumps(partition),
                    json.dumps(rejected),
                    json.dumps(partition),
                    json.dumps(admitted),
                ],
            ) as compiler,
        ):
            result = begin_semantic_partition(state, text)

        self.assertEqual(result["status"], "compile_owner")
        self.assertEqual(result["stage_a_calls"], 2)
        self.assertEqual(result["admission_calls"], 2)
        self.assertEqual(compiler.call_count, 4)

    def test_independent_unresolved_demand_reaches_draft_compilation(
        self,
    ) -> None:
        from agent.harness.hierarchical_planner import begin_semantic_partition

        text = "Use quick and diagnose the last failed job"
        partition = {
            "semantic_units": [
                {
                    "unit_id": "unit-1",
                    "clause_id": "clause-1",
                    "source_text": "quick",
                    "operation": "domain_request",
                    "owner_routes": [{
                        "owner": "performance",
                        "group": "qps_profile",
                    }],
                    "reason": "registered value",
                },
                {
                    "unit_id": "unit-2",
                    "clause_id": "clause-1",
                    "source_text": "and diagnose the last failed job",
                    "operation": "unresolved",
                    "owner_routes": [],
                    "reason": "independent demand lacks an owner",
                },
            ],
            "reason": "second demand is unresolved",
        }
        rejected = {
            "unit_verdicts": [
                {
                    "unit_id": "unit-1",
                    "verdict": "complete",
                    "supports_unit_id": "",
                    "reason": "mode selection is represented",
                },
                {
                    "unit_id": "unit-2",
                    "verdict": "unresolved",
                    "supports_unit_id": "",
                    "reason": "independent diagnosis demand is unresolved",
                },
            ],
            "clause_verdicts": [{
                "clause_id": "clause-1",
                "verdict": "unresolved",
                "omitted_owner_routes": [],
                "reason": "an independent demand has no owner",
            }],
            "reason": "partition is incomplete",
        }
        state = {
            "active_group": "chain_identity",
            "pending_question": {
                "id": "chain",
                "group": "chain_identity",
                "owner": "chain_rpc",
                "kind": "chain",
                "manual_input_allowed": True,
            },
        }

        with (
            patch(
                "agent.harness.hierarchical_planner.provider_from_config",
                return_value=object(),
            ),
            patch(
                "agent.harness.hierarchical_planner.request_semantic_compilation",
                side_effect=[json.dumps(partition), json.dumps(rejected)],
            ) as compiler,
            patch(
                "agent.harness.hierarchical_planner."
                "_partition_requires_independent_proposal",
                return_value=False,
            ),
        ):
            result = begin_semantic_partition(state, text)

        self.assertEqual(result["status"], "compile_owner")
        self.assertEqual(result["stage_a_calls"], 1)
        self.assertEqual(result["admission_calls"], 1)
        self.assertEqual(compiler.call_count, 2)

    def test_unresolved_partition_converges_to_independent_routed_proposal(
        self,
    ) -> None:
        from agent.harness.hierarchical_planner import begin_semantic_partition

        text = "Return to the previous step.\nChange the RPC endpoint."
        primary = {
            "semantic_units": [
                {
                    "unit_id": "primary-1",
                    "clause_id": "clause-1",
                    "source_text": "Return to the previous step.",
                    "operation": "unresolved",
                    "owner_routes": [],
                    "reason": "the operation was not resolved",
                },
                {
                    "unit_id": "primary-2",
                    "clause_id": "clause-2",
                    "source_text": "Change the RPC endpoint.",
                    "operation": "unresolved",
                    "owner_routes": [],
                    "reason": "the operation was not resolved",
                },
            ],
            "reason": "incomplete first proposal",
        }
        secondary = {
            "semantic_units": [
                {
                    "unit_id": "secondary-1",
                    "clause_id": "clause-1",
                    "source_text": "Return to the previous step.",
                    "operation": "navigation",
                    "owner_routes": [{
                        "owner": "coordinator",
                        "group": "sync_observe",
                    }],
                    "reason": "explicit navigation",
                },
                {
                    "unit_id": "secondary-2",
                    "clause_id": "clause-2",
                    "source_text": "Change the RPC endpoint.",
                    "operation": "domain_request",
                    "owner_routes": [{
                        "owner": "chain_rpc",
                        "group": "endpoint_process",
                    }],
                    "reason": "explicit endpoint reconfiguration",
                },
            ],
            "reason": "complete independent proposal",
        }
        proposal_calls = 0

        def semantic_response(*_args, **kwargs):
            nonlocal proposal_calls
            system = kwargs["system_prompt"]
            payload = kwargs["request_payload"]
            if "Stage A semantic partition" in system:
                proposal_calls += 1
                return json.dumps(primary if proposal_calls == 1 else secondary)
            if "Stage A coverage authority" in system:
                units = payload["semantic_units"]
                unresolved = any(
                    unit["operation"] == "unresolved" for unit in units
                )
                return json.dumps({
                    "unit_verdicts": [
                        {
                            "unit_id": unit["unit_id"],
                            "verdict": (
                                "unresolved" if unresolved else "complete"
                            ),
                            "supports_unit_id": "",
                            "reason": "reviewed against the complete source",
                        }
                        for unit in units
                    ],
                    "clause_verdicts": [
                        {
                            "clause_id": clause["clause_id"],
                            "verdict": (
                                "unresolved" if unresolved else "complete"
                            ),
                            "omitted_owner_routes": [],
                            "reason": "reviewed against the complete source",
                        }
                        for clause in payload["clauses"]
                    ],
                    "reason": "coverage review complete",
                })
            raise AssertionError(system)

        state = {
            "active_group": "sync_observe",
            "pending_question": {
                "id": "duration",
                "group": "sync_observe",
                "owner": "performance",
                "kind": "freeform",
                "manual_input_allowed": True,
            },
        }
        with (
            patch(
                "agent.harness.hierarchical_planner.provider_from_config",
                return_value=object(),
            ),
            patch(
                "agent.harness.hierarchical_planner.request_semantic_compilation",
                side_effect=semantic_response,
            ) as compiler,
        ):
            result = begin_semantic_partition(state, text)

        self.assertEqual(result["status"], "compile_owner", result)
        self.assertEqual(result["stage_a_calls"], 2)
        self.assertEqual(result["admission_calls"], 2)
        self.assertEqual(compiler.call_count, 4)
        self.assertEqual(
            [unit["operation"] for unit in result["routed_partition"]],
            ["navigation", "domain_request"],
        )
        self.assertEqual(
            result["stage_a_convergence"]["selected_proposal"],
            "secondary",
        )
        self.assertEqual(
            result["stage_a_convergence"]["request_count"],
            0,
        )

    def test_stage_a_convergence_rejects_selector_hash_substitution(self) -> None:
        from agent.harness.hierarchical_planner import _select_stage_a_proposal

        primary = [{
            "unit_id": "primary",
            "clause_id": "clause-1",
            "source_text": "change it",
            "operation": "navigation",
            "owner_routes": [{"owner": "coordinator", "group": "opening"}],
            "reason": "navigation",
        }]
        secondary = [{
            "unit_id": "secondary",
            "clause_id": "clause-1",
            "source_text": "change it",
            "operation": "navigation",
            "owner_routes": [{"owner": "coordinator", "group": "opening"}],
            "reason": "navigation",
        }]
        payload = {
            "user_text": "change it",
            "clauses": [{
                "clause_id": "clause-1",
                "text": "change it",
                "input_shape": "prose",
            }],
            "pending_question": {},
            "groups": [],
            "universal_operation_purposes": {},
        }
        forged = json.dumps({
            "selected_proposal": "secondary",
            "primary_hash": "0" * 64,
            "secondary_hash": "1" * 64,
            "reason": "forged selection",
        })
        with patch(
            "agent.harness.hierarchical_planner.request_semantic_compilation",
            return_value=forged,
        ):
            _selected, errors, _sizes, receipt = _select_stage_a_proposal(
                object(), payload, primary, secondary
            )

        self.assertTrue(any("primary proposal hash" in error for error in errors))
        self.assertTrue(any("secondary proposal hash" in error for error in errors))
        self.assertEqual(receipt["selected_proposal"], "secondary")

    def test_stage_a_convergence_selects_the_only_eligible_proposal(
        self,
    ) -> None:
        from agent.harness.hierarchical_planner import _select_stage_a_proposal

        primary = [{
            "unit_id": "primary",
            "clause_id": "clause-1",
            "source_text": "change it",
            "operation": "unresolved",
            "owner_routes": [],
            "reason": "uncertain",
        }]
        secondary = [{
            "unit_id": "secondary",
            "clause_id": "clause-1",
            "source_text": "change it",
            "operation": "navigation",
            "owner_routes": [{"owner": "coordinator", "group": "opening"}],
            "reason": "navigation",
        }]
        payload = {
            "user_text": "change it",
            "clauses": [{
                "clause_id": "clause-1",
                "text": "change it",
                "input_shape": "prose",
            }],
            "pending_question": {},
            "groups": [],
            "universal_operation_purposes": {},
        }

        with patch(
            "agent.harness.hierarchical_planner.request_semantic_compilation",
        ) as compiler:
            selected, errors, sizes, receipt = _select_stage_a_proposal(
                object(), payload, primary, secondary
            )

        compiler.assert_not_called()
        self.assertEqual(selected, secondary)
        self.assertEqual(errors, ())
        self.assertEqual(sizes, ())
        self.assertTrue(receipt["valid"])
        self.assertEqual(receipt["selected_proposal"], "secondary")
        self.assertEqual(receipt["selection_authority"], "harness_eligibility")
        self.assertFalse(receipt["primary_eligible"])
        self.assertTrue(receipt["secondary_eligible"])
        self.assertEqual(receipt["request_count"], 0)

    def test_stage_a_compilation_preflight_rejects_wrong_owner_before_selection(
        self,
    ) -> None:
        from agent.harness.hierarchical_planner import (
            _proposal_compilation_eligible,
            _select_stage_a_proposal,
        )

        source = "May I retest a different value?"
        coordinator = [{
            "unit_id": "primary",
            "clause_id": "clause-1",
            "source_text": source,
            "operation": "navigation",
            "owner_routes": [{
                "owner": "coordinator",
                "group": "chain_identity",
            }],
            "reason": "structurally valid but owned by the wrong compiler",
        }]
        chain = [{
            "unit_id": "secondary",
            "clause_id": "clause-1",
            "source_text": source,
            "operation": "domain_request",
            "owner_routes": [{"owner": "chain_rpc", "group": "chain_identity"}],
            "reason": "present chain mutation request",
        }]

        def compile_owner(_state, owner, _groups, _partition, unit_ids):
            unit_id = unit_ids[0]
            if owner == "coordinator":
                return ({
                    "actions": [],
                    "bindings": [{
                        "unit_id": unit_id,
                        "action_indexes": [],
                        "disposition": "unresolved",
                        "reason": "no coordinator action serves this request",
                    }],
                }, (), (101,))
            return ({
                "actions": [{"type": "chain_change_input"}],
                "bindings": [{
                    "unit_id": unit_id,
                    "action_indexes": [0],
                    "disposition": "action",
                    "reason": "registered chain intake",
                }],
            }, (), (103,))

        with patch(
            "agent.harness.hierarchical_planner._compile_owner_document",
            side_effect=compile_owner,
        ):
            primary_eligible, primary_sizes = (
                _proposal_compilation_eligible({}, coordinator)
            )
            secondary_eligible, secondary_sizes = (
                _proposal_compilation_eligible({}, chain)
            )

        payload = {
            "user_text": source,
            "clauses": [{
                "clause_id": "clause-1",
                "text": source,
                "input_shape": "prose",
            }],
            "pending_question": {},
            "groups": [],
            "universal_operation_purposes": {},
        }
        with patch(
            "agent.harness.hierarchical_planner.request_semantic_compilation",
        ) as convergence:
            selected, errors, sizes, receipt = _select_stage_a_proposal(
                object(),
                payload,
                coordinator,
                chain,
                primary_eligible=primary_eligible,
                secondary_eligible=secondary_eligible,
            )

        convergence.assert_not_called()
        self.assertFalse(primary_eligible)
        self.assertTrue(secondary_eligible)
        self.assertEqual(primary_sizes, (101,))
        self.assertEqual(secondary_sizes, (103,))
        self.assertEqual(selected, chain)
        self.assertEqual(errors, ())
        self.assertEqual(sizes, ())
        self.assertEqual(receipt["selection_authority"], "harness_eligibility")

    def test_stage_a_partition_uses_compilation_preflight_for_competing_routes(
        self,
    ) -> None:
        from agent.harness.hierarchical_planner import begin_semantic_partition

        text = "May I retest a different value?"
        proposals = [
            {
                "semantic_units": [{
                    "unit_id": "primary",
                    "clause_id": "clause-1",
                    "source_text": text,
                    "operation": "navigation",
                    "owner_routes": [{
                        "owner": "coordinator",
                        "group": "chain_identity",
                    }],
                    "reason": "wrong owner",
                }],
                "reason": "primary proposal",
            },
            {
                "semantic_units": [{
                    "unit_id": "secondary",
                    "clause_id": "clause-1",
                    "source_text": text,
                    "operation": "domain_request",
                    "owner_routes": [{
                        "owner": "chain_rpc",
                        "group": "chain_identity",
                    }],
                    "reason": "registered owner",
                }],
                "reason": "secondary proposal",
            },
        ]
        proposal_index = 0

        def semantic_response(*_args, **kwargs):
            nonlocal proposal_index
            system = kwargs["system_prompt"]
            payload = kwargs["request_payload"]
            if "Stage A semantic partition" in system:
                result = proposals[proposal_index]
                proposal_index += 1
                return json.dumps(result)
            raise AssertionError(system)

        with (
            patch(
                "agent.harness.hierarchical_planner.provider_from_config",
                return_value=object(),
            ),
            patch(
                "agent.harness.hierarchical_planner.request_semantic_compilation",
                side_effect=semantic_response,
            ) as compiler,
            patch(
                "agent.harness.hierarchical_planner."
                "_partition_requires_independent_proposal",
                return_value=True,
            ),
            patch(
                "agent.harness.hierarchical_planner."
                "_review_competing_open_identity_relations",
                side_effect=lambda _provider, partition, _payload: (
                    [dict(unit) for unit in partition],
                    (),
                    (),
                    {},
                ),
            ),
            patch(
                "agent.harness.hierarchical_planner._review_stage_a_candidate",
                return_value=((), (), frozenset(), True, ()),
            ),
            patch(
                "agent.harness.hierarchical_planner."
                "_proposal_compilation_eligible",
                side_effect=[(False, (101,)), (True, (103,))],
            ) as compilation_preflight,
        ):
            result = begin_semantic_partition({}, text)

        self.assertEqual(result["status"], "compile_owner", result)
        self.assertEqual(result["stage_a_calls"], 2)
        self.assertEqual(result["admission_calls"], 0)
        self.assertEqual(result["stage_b_calls"], 2)
        self.assertEqual(compiler.call_count, 2)
        self.assertEqual(compilation_preflight.call_count, 2)
        self.assertEqual(
            result["stage_a_convergence"]["selected_proposal"],
            "secondary",
        )
        self.assertEqual(
            result["stage_a_convergence"]["selection_authority"],
            "harness_eligibility",
        )
        self.assertEqual(
            result["owner_requests"],
            [{
                "owner": "chain_rpc",
                "unit_ids": ["secondary"],
                "groups": ["chain_identity"],
            }],
        )

    def test_stage_a_compilation_preflight_is_generic_across_domain_owners(
        self,
    ) -> None:
        from agent.harness.hierarchical_planner import (
            _proposal_compilation_eligible,
        )

        routes = (
            ("chain_rpc", "chain_identity"),
            ("chain_rpc", "endpoint_process"),
            ("chain_rpc", "target_mode"),
            ("performance", "qps_profile"),
            ("orientation", "framework_summary"),
        )

        def compile_owner(_state, _owner, _groups, _partition, unit_ids):
            return ({
                "actions": [{"type": "registered-action"}],
                "bindings": [{
                    "unit_id": unit_ids[0],
                    "action_indexes": [0],
                    "disposition": "action",
                    "reason": "compiled by the registered owner",
                }],
            }, (), (107,))

        with patch(
            "agent.harness.hierarchical_planner._compile_owner_document",
            side_effect=compile_owner,
        ):
            for index, (owner, group) in enumerate(routes, start=1):
                with self.subTest(owner=owner, group=group):
                    eligible, sizes = _proposal_compilation_eligible({}, [{
                        "unit_id": f"unit-{index}",
                        "clause_id": "clause-1",
                        "source_text": "one present request",
                        "operation": (
                            "consultation"
                            if owner == "orientation"
                            else "domain_request"
                        ),
                        "owner_routes": [{"owner": owner, "group": group}],
                        "reason": "registered route",
                    }])
                    self.assertTrue(eligible)
                    self.assertEqual(sizes, (107,))

    def test_stage_a_compilation_preflight_fails_closed_for_unbound_unit(
        self,
    ) -> None:
        from agent.harness.hierarchical_planner import (
            _proposal_compilation_eligible,
        )

        partition = [{
            "unit_id": "unit-1",
            "clause_id": "clause-1",
            "source_text": "change the current setting",
            "operation": "domain_request",
            "owner_routes": [{"owner": "performance", "group": "qps_profile"}],
            "reason": "present request",
        }]
        with patch(
            "agent.harness.hierarchical_planner._compile_owner_document",
            return_value=({
                "actions": [],
                "bindings": [{
                    "unit_id": "unit-1",
                    "action_indexes": [],
                    "disposition": "unresolved",
                    "reason": "insufficient owner compilation",
                }],
            }, (), (109,)),
        ):
            eligible, sizes = _proposal_compilation_eligible({}, partition)

        self.assertFalse(eligible)
        self.assertEqual(sizes, (109,))

    def test_stage_a_convergence_rejects_two_uncompileable_proposals(
        self,
    ) -> None:
        from agent.harness.hierarchical_planner import _select_stage_a_proposal

        primary = [{
            "unit_id": "primary",
            "clause_id": "clause-1",
            "source_text": "change the current setting",
            "operation": "navigation",
            "owner_routes": [{"owner": "coordinator", "group": "qps_profile"}],
            "reason": "first uncompileable route",
        }]
        secondary = [{
            "unit_id": "secondary",
            "clause_id": "clause-1",
            "source_text": "change the current setting",
            "operation": "domain_request",
            "owner_routes": [{"owner": "performance", "group": "qps_profile"}],
            "reason": "second uncompileable route",
        }]
        payload = {
            "user_text": "change the current setting",
            "clauses": [{
                "clause_id": "clause-1",
                "text": "change the current setting",
                "input_shape": "prose",
            }],
            "pending_question": {},
            "groups": [],
            "universal_operation_purposes": {},
        }
        verdict = json.dumps({
            "selected_proposal": "primary",
            "primary_hash": "",
            "secondary_hash": "",
            "reason": "select one despite failed owner compilation",
        })

        def converge(*_args, **kwargs):
            request = kwargs["request_payload"]
            document = json.loads(verdict)
            document["primary_hash"] = request["primary"]["hash"]
            document["secondary_hash"] = request["secondary"]["hash"]
            return json.dumps(document)

        with patch(
            "agent.harness.hierarchical_planner.request_semantic_compilation",
            side_effect=converge,
        ):
            _selected, errors, _sizes, receipt = _select_stage_a_proposal(
                object(),
                payload,
                primary,
                secondary,
                primary_eligible=False,
                secondary_eligible=False,
            )

        self.assertIn(
            "Stage A convergence selected an incomplete proposal",
            errors,
        )
        self.assertFalse(receipt["valid"])

    def test_stage_a_convergence_still_reviews_two_compileable_proposals(
        self,
    ) -> None:
        from agent.harness.hierarchical_planner import _select_stage_a_proposal

        primary = [{
            "unit_id": "primary",
            "clause_id": "clause-1",
            "source_text": "change the current setting",
            "operation": "navigation",
            "owner_routes": [{"owner": "coordinator", "group": "qps_profile"}],
            "reason": "registered navigation",
        }]
        secondary = [{
            "unit_id": "secondary",
            "clause_id": "clause-1",
            "source_text": "change the current setting",
            "operation": "domain_request",
            "owner_routes": [{"owner": "performance", "group": "qps_profile"}],
            "reason": "registered mutation intake",
        }]
        payload = {
            "user_text": "change the current setting",
            "clauses": [{
                "clause_id": "clause-1",
                "text": "change the current setting",
                "input_shape": "prose",
            }],
            "pending_question": {},
            "groups": [],
            "universal_operation_purposes": {},
        }

        def converge(*_args, **kwargs):
            request = kwargs["request_payload"]
            return json.dumps({
                "selected_proposal": "secondary",
                "primary_hash": request["primary"]["hash"],
                "secondary_hash": request["secondary"]["hash"],
                "reason": "the mutation route best matches the present request",
            })

        with patch(
            "agent.harness.hierarchical_planner.request_semantic_compilation",
            side_effect=converge,
        ) as compiler:
            selected, errors, _sizes, receipt = _select_stage_a_proposal(
                object(),
                payload,
                primary,
                secondary,
                primary_eligible=True,
                secondary_eligible=True,
            )

        compiler.assert_called_once()
        self.assertEqual(selected, secondary)
        self.assertEqual(errors, ())
        self.assertTrue(receipt["valid"])
        self.assertEqual(receipt["selection_authority"], "model_convergence")

    def test_malformed_independent_stage_a_proposal_fails_closed(self) -> None:
        from agent.harness.hierarchical_planner import begin_semantic_partition

        text = "Reset this session and explain the previous failure."
        primary = {
            "semantic_units": [{
                "unit_id": "primary",
                "clause_id": "clause-1",
                "source_text": text,
                "operation": "unresolved",
                "owner_routes": [],
                "reason": "unresolved compound request",
            }],
        }
        malformed = "not-json"
        calls = 0

        def semantic_response(*_args, **kwargs):
            nonlocal calls
            system = kwargs["system_prompt"]
            if "Stage A semantic partition" in system:
                calls += 1
                return json.dumps(primary) if calls == 1 else malformed
            if "Stage A coverage authority" in system:
                return json.dumps({
                    "unit_verdicts": [{
                        "unit_id": "primary",
                        "verdict": "unresolved",
                        "supports_unit_id": "",
                        "reason": "unresolved",
                    }],
                    "clause_verdicts": [{
                        "clause_id": "clause-1",
                        "verdict": "unresolved",
                        "omitted_owner_routes": [],
                        "reason": "unresolved",
                    }],
                    "reason": "reviewed",
                })
            raise AssertionError(system)

        with (
            patch(
                "agent.harness.hierarchical_planner.provider_from_config",
                return_value=object(),
            ),
            patch(
                "agent.harness.hierarchical_planner.request_semantic_compilation",
                side_effect=semantic_response,
            ),
        ):
            result = begin_semantic_partition({}, text)

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["stage_a_calls"], 3)
        self.assertTrue(result["errors"])

    def test_stage_b_semantic_rejection_recompiles_complete_owner_document(
        self,
    ) -> None:
        from agent.harness.hierarchical_planner import _compile_owner_document

        text = "Clear this configuration and start over."
        partition = [{
            "unit_id": "reset-unit",
            "clause_id": "clause-1",
            "source_text": text,
            "operation": "administrative",
            "owner_routes": [{"owner": "orientation", "group": ""}],
            "reason": "session reset",
        }]
        wrong = {
            "actions": [{
                "type": "set_response_language",
                "language": "en",
                "source_evidence": text,
            }],
            "bindings": [{
                "unit_id": "reset-unit",
                "action_indexes": [0],
                "disposition": "action",
                "reason": "wrong registered action",
            }],
            "reason": "wrong first proposal",
        }
        corrected = {
            "actions": [{"type": "request_session_reset"}],
            "bindings": [{
                "unit_id": "reset-unit",
                "action_indexes": [0],
                "disposition": "action",
                "reason": "exact reset action",
            }],
            "reason": "corrected proposal",
        }
        compiler_calls = 0
        reviewer_calls = 0

        def response(*_args, **kwargs):
            nonlocal compiler_calls, reviewer_calls
            system = kwargs["system_prompt"]
            if "Stage B semantic review authority" in system:
                reviewer_calls += 1
                payload = kwargs["request_payload"]
                verdict = "reject"
                return json.dumps({
                    "proposal_hash": payload["proposal_hash"],
                    "unit_verdicts": [{
                        "unit_id": "reset-unit",
                        "verdict": verdict,
                        "reason": (
                            "the action purpose does not clear a session"
                            if verdict == "reject"
                            else "the action clears the requested session"
                        ),
                    }],
                    "action_verdicts": [{
                        "action_index": 0,
                        "verdict": verdict,
                        "reason": (
                            "the language selection is absent from the source"
                            if verdict == "reject"
                            else "the reset action is source authorized"
                        ),
                    }],
                    "reason": (
                        "candidate is not source authorized"
                        if verdict == "reject"
                        else "candidate is source authorized"
                    ),
                })
            compiler_calls += 1
            return json.dumps(wrong if compiler_calls == 1 else corrected)

        with (
            patch(
                "agent.harness.hierarchical_planner.provider_from_config",
                return_value=object(),
            ),
            patch(
                "agent.harness.hierarchical_planner.request_semantic_compilation",
                side_effect=response,
            ) as compiler,
        ):
            document, errors, sizes = _compile_owner_document(
                {}, "orientation", frozenset(), partition, ("reset-unit",)
            )

        self.assertEqual(errors, ())
        self.assertEqual(
            document["actions"],
            [{"type": "request_session_reset"}],
        )
        self.assertEqual(len(document["semantic_review_receipts"]), 1)
        self.assertEqual(len(sizes), 5)
        self.assertEqual(compiler.call_count, 5)
        self.assertEqual(reviewer_calls, 3)

    def test_stage_b_semantic_reviewer_malformed_twice_fails_closed(self) -> None:
        from agent.harness.hierarchical_planner import _compile_owner_document

        text = "Clear this configuration and start over."
        partition = [{
            "unit_id": "reset-unit",
            "clause_id": "clause-1",
            "source_text": text,
            "operation": "administrative",
            "owner_routes": [{"owner": "orientation", "group": ""}],
            "reason": "session reset",
        }]
        wrong = json.dumps({
            "actions": [{
                "type": "set_response_language",
                "language": "en",
                "source_evidence": text,
            }],
            "bindings": [{
                "unit_id": "reset-unit",
                "action_indexes": [0],
                "disposition": "action",
                "reason": "wrong registered action",
            }],
            "reason": "wrong proposal",
        })

        def response(*_args, **kwargs):
            if "Stage B semantic review authority" in kwargs["system_prompt"]:
                return "not-json"
            return wrong

        with (
            patch(
                "agent.harness.hierarchical_planner.provider_from_config",
                return_value=object(),
            ),
            patch(
                "agent.harness.hierarchical_planner.request_semantic_compilation",
                side_effect=response,
            ),
        ):
            document, errors, sizes = _compile_owner_document(
                {}, "orientation", frozenset(), partition, ("reset-unit",)
            )

        self.assertTrue(errors)
        self.assertEqual(len(document["semantic_review_receipts"]), 2)
        self.assertEqual(len(sizes), 8)

    def test_stage_b_semantic_second_rejection_fails_closed(self) -> None:
        from agent.harness.hierarchical_planner import _compile_owner_document

        text = "Clear this configuration and start over."
        partition = [{
            "unit_id": "reset-unit",
            "clause_id": "clause-1",
            "source_text": text,
            "operation": "administrative",
            "owner_routes": [{"owner": "orientation", "group": ""}],
            "reason": "session reset",
        }]
        wrong = {
            "actions": [{
                "type": "set_response_language",
                "language": "en",
                "source_evidence": text,
            }],
            "bindings": [{
                "unit_id": "reset-unit",
                "action_indexes": [0],
                "disposition": "action",
                "reason": "wrong registered action",
            }],
            "reason": "wrong proposal",
        }

        def response(*_args, **kwargs):
            payload = kwargs["request_payload"]
            if "Stage B semantic review authority" in kwargs["system_prompt"]:
                return json.dumps({
                    "proposal_hash": payload["proposal_hash"],
                    "unit_verdicts": [{
                        "unit_id": "reset-unit",
                        "verdict": "reject",
                        "reason": "wrong purpose",
                    }],
                    "action_verdicts": [{
                        "action_index": 0,
                        "verdict": "reject",
                        "reason": "wrong action",
                    }],
                    "reason": "rejected",
                })
            return json.dumps(wrong)

        with (
            patch(
                "agent.harness.hierarchical_planner.provider_from_config",
                return_value=object(),
            ),
            patch(
                "agent.harness.hierarchical_planner.request_semantic_compilation",
                side_effect=response,
            ),
        ):
            document, errors, sizes = _compile_owner_document(
                {}, "orientation", frozenset(), partition, ("reset-unit",)
            )

        self.assertTrue(any("rejected action" in error for error in errors))
        self.assertEqual(len(document["semantic_review_receipts"]), 2)
        self.assertEqual(len(sizes), 8)

    def test_stage_b_semantic_review_rejects_hash_substitution(self) -> None:
        from agent.harness.hierarchical_planner import (
            _review_owner_document_semantics,
        )

        payload = {
            "owner": "coordinator",
            "semantic_units": [{
                "unit_id": "unit-1",
                "source_text": "clear it",
            }],
            "owner_action_schema": [],
        }
        document = {
            "actions": [{"type": "reset_session"}],
            "bindings": [],
        }
        forged = json.dumps({
            "proposal_hash": "0" * 64,
            "unit_verdicts": [{
                "unit_id": "unit-1",
                "verdict": "admit",
                "reason": "accepted",
            }],
            "action_verdicts": [{
                "action_index": 0,
                "verdict": "admit",
                "reason": "accepted",
            }],
            "reason": "accepted",
        })

        with patch(
            "agent.harness.hierarchical_planner.request_semantic_compilation",
            return_value=forged,
        ):
            errors, _size, receipt = _review_owner_document_semantics(
                object(), payload, document
            )

        self.assertIn("changed proposal hash", "; ".join(errors))
        self.assertFalse(receipt["valid"])

    def test_stage_b_semantic_review_excludes_model_authored_rationales(
        self,
    ) -> None:
        from agent.harness.hierarchical_planner import (
            _review_owner_document_semantics,
        )

        payload = {
            "owner": "chain_rpc",
            "semantic_units": [{
                "unit_id": "unit-1",
                "source_text": "do not use alpha",
                "operation": "domain_request",
                "owner_routes": [{
                    "owner": "chain_rpc",
                    "group": "target_mode",
                }],
                "reason": "incorrectly claims choose_target_mode",
            }],
            "owner_action_schema": [{
                "type": "request_target_mode_selection",
                "purpose": "registry-owned typed intake",
            }],
        }
        document = {
            "actions": [{
                "type": "request_target_mode_selection",
                "source_evidence": "do not use alpha",
            }],
            "bindings": [{
                "unit_id": "unit-1",
                "action_indexes": [0],
                "disposition": "action",
                "reason": "untrusted binding rationale",
            }],
            "reason": "untrusted document rationale",
        }
        captured: dict[str, object] = {}

        def review(*_args, **kwargs):
            captured.update(kwargs["request_payload"])
            return json.dumps({
                "proposal_hash": kwargs["request_payload"]["proposal_hash"],
                "unit_verdicts": [{
                    "unit_id": "unit-1",
                    "verdict": "admit",
                    "reason": "typed source and route authorize the intake",
                }],
                "action_verdicts": [{
                    "action_index": 0,
                    "verdict": "admit",
                    "reason": "typed candidate matches the registry purpose",
                }],
                "reason": "admitted",
            })

        with patch(
            "agent.harness.hierarchical_planner.request_semantic_compilation",
            side_effect=review,
        ):
            errors, _size, receipt = _review_owner_document_semantics(
                object(), payload, document
            )

        self.assertEqual(errors, ())
        self.assertTrue(receipt["valid"])
        self.assertNotIn("reason", captured["semantic_units"][0])
        self.assertEqual(
            captured["semantic_units"][0]["source_text"],
            "do not use alpha",
        )
        self.assertEqual(
            captured["semantic_units"][0]["owner_routes"],
            payload["semantic_units"][0]["owner_routes"],
        )
        candidate = captured["candidate"]
        self.assertNotIn("reason", candidate)
        self.assertNotIn("reason", candidate["bindings"][0])
        self.assertEqual(candidate["actions"], document["actions"])
        self.assertEqual(candidate["bindings"][0]["action_indexes"], [0])
        self.assertEqual(
            captured["owner_action_schema"],
            payload["owner_action_schema"],
        )
        self.assertEqual(len(captured["proposal_hash"]), 64)

    def test_stage_b_semantic_jury_is_order_independent(self) -> None:
        from agent.harness.hierarchical_planner import (
            _review_owner_document_semantics,
        )

        payload = {
            "owner": "chain_rpc",
            "semantic_units": [{
                "unit_id": "unit-1",
                "source_text": "choose another mode",
            }],
            "owner_action_schema": [{
                "type": "request_target_mode_selection",
                "purpose": "open typed selection",
            }],
        }
        document = {
            "actions": [{
                "type": "request_target_mode_selection",
                "source_evidence": "choose another mode",
            }],
            "bindings": [{
                "unit_id": "unit-1",
                "action_indexes": [0],
                "disposition": "action",
                "reason": "compiled",
            }],
            "reason": "compiled",
        }

        def response(verdict: str, proposal_hash: str) -> str:
            return json.dumps({
                "proposal_hash": proposal_hash,
                "unit_verdicts": [{
                    "unit_id": "unit-1",
                    "verdict": verdict,
                    "reason": "member verdict",
                }],
                "action_verdicts": [{
                    "action_index": 0,
                    "verdict": verdict,
                    "reason": "member verdict",
                }],
                "reason": "reviewed",
            })

        verdicts = iter(("admit", "reject", "admit"))

        def review(*_args, **kwargs):
            return response(
                next(verdicts),
                kwargs["request_payload"]["proposal_hash"],
            )

        with patch(
            "agent.harness.hierarchical_planner.request_semantic_compilation",
            side_effect=review,
        ) as reviewer:
            errors, sizes, receipt = _review_owner_document_semantics(
                object(), payload, document
            )

        self.assertEqual(errors, ())
        self.assertEqual(reviewer.call_count, 3)
        self.assertEqual(len(sizes), 3)
        self.assertEqual(receipt["request_count"], 3)
        self.assertEqual(receipt["member_validity"], [True, False, True])
        self.assertEqual(len(receipt["review_hashes"]), 3)
        self.assertTrue(receipt["valid"])

    def test_stage_b_semantic_jury_runs_concurrently_in_member_order(
        self,
    ) -> None:
        from agent.harness.hierarchical_planner import (
            _review_owner_document_semantics,
        )
        from agent.llm.types import llm_turn_scope

        payload = {
            "owner": "chain_rpc",
            "semantic_units": [{
                "unit_id": "unit-1",
                "source_text": "choose another mode",
            }],
            "owner_action_schema": [{
                "type": "request_target_mode_selection",
                "purpose": "open typed selection",
            }],
        }
        document = {
            "actions": [{
                "type": "request_target_mode_selection",
                "source_evidence": "choose another mode",
            }],
            "bindings": [{
                "unit_id": "unit-1",
                "action_indexes": [0],
                "disposition": "action",
                "reason": "compiled",
            }],
            "reason": "compiled",
        }
        barrier = threading.Barrier(3)

        def review(*_args, **kwargs):
            barrier.wait(timeout=1)
            return json.dumps({
                "proposal_hash": kwargs["request_payload"]["proposal_hash"],
                "unit_verdicts": [{
                    "unit_id": "unit-1",
                    "verdict": "admit",
                    "reason": "member verdict",
                }],
                "action_verdicts": [{
                    "action_index": 0,
                    "verdict": "admit",
                    "reason": "member verdict",
                }],
                "reason": "reviewed",
            })

        with (
            llm_turn_scope(1),
            patch(
                "agent.harness.hierarchical_planner.request_semantic_compilation",
                side_effect=review,
            ),
        ):
            errors, sizes, receipt = _review_owner_document_semantics(
                object(), payload, document
            )

        self.assertEqual(errors, ())
        self.assertEqual(len(sizes), 3)
        self.assertEqual(receipt["request_count"], 3)
        self.assertEqual(receipt["member_validity"], [True, True, True])
        self.assertEqual(len(receipt["review_hashes"]), 3)

    def test_stage_b_semantic_jury_requires_two_admissions(self) -> None:
        from agent.harness.hierarchical_planner import (
            _review_owner_document_semantics,
        )

        payload = {
            "owner": "orientation",
            "semantic_units": [{
                "unit_id": "unit-1",
                "source_text": "clear it",
            }],
            "owner_action_schema": [{
                "type": "request_session_reset",
                "purpose": "clear session",
            }],
        }
        document = {
            "actions": [{"type": "request_session_reset"}],
            "bindings": [{
                "unit_id": "unit-1",
                "action_indexes": [0],
                "disposition": "action",
                "reason": "compiled",
            }],
        }
        verdicts = iter(("reject", "admit", "reject"))

        def review(*_args, **kwargs):
            verdict = next(verdicts)
            return json.dumps({
                "proposal_hash": kwargs["request_payload"]["proposal_hash"],
                "unit_verdicts": [{
                    "unit_id": "unit-1",
                    "verdict": verdict,
                    "reason": "member verdict",
                }],
                "action_verdicts": [{
                    "action_index": 0,
                    "verdict": verdict,
                    "reason": "member verdict",
                }],
                "reason": "reviewed",
            })

        with patch(
            "agent.harness.hierarchical_planner.request_semantic_compilation",
            side_effect=review,
        ) as reviewer:
            errors, sizes, receipt = _review_owner_document_semantics(
                object(), payload, document
            )

        self.assertEqual(reviewer.call_count, 3)
        self.assertEqual(len(sizes), 3)
        self.assertIn("quorum was not reached", "; ".join(errors))
        self.assertEqual(receipt["member_validity"], [False, True, False])
        self.assertFalse(receipt["valid"])

    def test_single_reachable_stage_b_action_needs_no_extra_review(self) -> None:
        from agent.harness.hierarchical_planner import (
            _owner_document_requires_semantic_review,
        )

        payload = {
            "owner_action_schema": [{"type": "queue_workflow_goal"}],
            "semantic_units": [{
                "unit_id": "unit-1",
                "source_text": "save it for later",
            }],
        }
        document = {
            "actions": [{
                "type": "queue_workflow_goal",
                "target_mode": "real-node",
            }],
            "bindings": [{
                "unit_id": "unit-1",
                "action_indexes": [0],
            }],
        }

        self.assertFalse(
            _owner_document_requires_semantic_review(payload, document)
        )

    def test_stage_b_risk_projects_array_grounding_values(self) -> None:
        from agent.harness.hierarchical_planner import (
            _owner_document_requires_semantic_review,
        )

        payload = {
            "owner_action_schema": [
                {"type": "choose_chain"},
                {"type": "request_chain_selection"},
            ],
            "semantic_units": [{
                "unit_id": "unit-1",
                "source_text": "switch to eth",
            }],
        }

        def document(candidates):
            return {
                "actions": [{
                    "type": "choose_chain",
                    "chain_text": "eth",
                    "chain_candidates": candidates,
                    "source_evidence": "eth",
                    "confidence": "high",
                }],
                "bindings": [{
                    "unit_id": "unit-1",
                    "action_indexes": [0],
                }],
            }

        self.assertFalse(
            _owner_document_requires_semantic_review(payload, document(["eth"]))
        )
        self.assertTrue(
            _owner_document_requires_semantic_review(
                payload,
                document(["eth", "bsc"]),
            )
        )

    def test_stage_b_risk_preserves_scalar_and_structured_grounding(self) -> None:
        from agent.harness.action_registry import (
            semantic_grounding_value_projection,
        )
        from agent.harness.hierarchical_planner import (
            _owner_document_requires_semantic_review,
            _source_contains_semantic_grounding_value,
        )

        scalar_payload = {
            "owner_action_schema": [
                {"type": "set_qps_mode"},
                {"type": "request_qps_mode_selection"},
            ],
            "semantic_units": [{
                "unit_id": "unit-1",
                "source_text": "use quick",
            }],
        }
        scalar_document = {
            "actions": [{
                "type": "set_qps_mode",
                "qps_mode": "quick",
            }],
            "bindings": [{"unit_id": "unit-1", "action_indexes": [0]}],
        }
        self.assertFalse(
            _owner_document_requires_semantic_review(
                scalar_payload,
                scalar_document,
            )
        )
        structured = {"chain_choice": "ethereum"}
        self.assertEqual(
            semantic_grounding_value_projection(structured, {"type": "object"}),
            (structured,),
        )
        self.assertTrue(
            _source_contains_semantic_grounding_value(
                structured,
                '{"chain_choice": "ethereum"}',
            )
        )
        self.assertFalse(
            _source_contains_semantic_grounding_value(
                structured,
                '{"chain_choice": "bsc"}',
            )
        )

    def test_admission_repair_preserves_independent_consultation(
        self,
    ) -> None:
        from agent.harness.hierarchical_planner import begin_semantic_partition

        text = "Use quick and explain what it changes"
        partition = {
            "semantic_units": [
                {
                    "unit_id": "unit-1",
                    "clause_id": "clause-1",
                    "source_text": "quick",
                    "operation": "domain_request",
                    "owner_routes": [{
                        "owner": "performance",
                        "group": "qps_profile",
                    }],
                    "reason": "registered value",
                },
                {
                    "unit_id": "unit-2",
                    "clause_id": "clause-1",
                    "source_text": "and",
                    "operation": "unresolved",
                    "owner_routes": [],
                    "reason": "wrapper",
                },
                {
                    "unit_id": "unit-3",
                    "clause_id": "clause-1",
                    "source_text": "explain what it changes",
                    "operation": "consultation",
                    "owner_routes": [{
                        "owner": "orientation",
                        "group": "qps_profile",
                    }],
                    "reason": "read-only sibling demand",
                },
            ],
            "reason": "split partition",
        }
        rejected = {
            "unit_verdicts": [
                {
                    "unit_id": "unit-1",
                    "verdict": "complete",
                    "supports_unit_id": "",
                    "reason": "mode selection is represented",
                },
                {
                    "unit_id": "unit-2",
                    "verdict": "unresolved",
                    "supports_unit_id": "",
                    "reason": "wrapper remains unresolved",
                },
                {
                    "unit_id": "unit-3",
                    "verdict": "complete",
                    "supports_unit_id": "",
                    "reason": "consultation is represented",
                },
            ],
            "clause_verdicts": [{
                "clause_id": "clause-1",
                "verdict": "complete",
                "omitted_owner_routes": [],
                "reason": "all demands have owners",
            }],
            "reason": "wrapper defect",
        }
        repaired = {
            "unit_verdicts": [
                {
                    "unit_id": "unit-1",
                    "verdict": "complete",
                    "supports_unit_id": "",
                    "reason": "mode selection is represented",
                },
                {
                    "unit_id": "unit-2",
                    "verdict": "redundant",
                    "supports_unit_id": "unit-1",
                    "reason": "connective supports the selection",
                },
                {
                    "unit_id": "unit-3",
                    "verdict": "complete",
                    "supports_unit_id": "",
                    "reason": "consultation remains independent",
                },
            ],
            "clause_verdicts": [{
                "clause_id": "clause-1",
                "verdict": "complete",
                "omitted_owner_routes": [],
                "reason": "both independent demands are represented",
            }],
            "reason": "coherent repaired verdict",
        }
        state = {
            "active_group": "chain_identity",
            "pending_question": {
                "id": "chain",
                "group": "chain_identity",
                "owner": "chain_rpc",
                "kind": "chain",
                "manual_input_allowed": True,
            },
        }

        with (
            patch(
                "agent.harness.hierarchical_planner.provider_from_config",
                return_value=object(),
            ),
            patch(
                "agent.harness.hierarchical_planner.request_semantic_compilation",
                side_effect=[
                    json.dumps(partition),
                    json.dumps(rejected),
                    json.dumps(repaired),
                ],
            ) as compiler,
        ):
            result = begin_semantic_partition(state, text)

        self.assertEqual(result["status"], "compile_owner")
        self.assertEqual(result["stage_a_calls"], 1)
        self.assertEqual(result["admission_calls"], 2)
        self.assertEqual(compiler.call_count, 3)
        self.assertEqual(
            {
                unit["operation"]
                for unit in result["routed_partition"]
            },
            {"domain_request", "consultation"},
        )

    def test_repeated_inconsistent_admission_fails_closed(self) -> None:
        from agent.harness.hierarchical_planner import begin_semantic_partition

        text = "Use quick profile"
        partition = {
            "semantic_units": [
                {
                    "unit_id": "unit-1",
                    "clause_id": "clause-1",
                    "source_text": "quick",
                    "operation": "domain_request",
                    "owner_routes": [{
                        "owner": "performance",
                        "group": "qps_profile",
                    }],
                    "reason": "registered value",
                },
                {
                    "unit_id": "unit-2",
                    "clause_id": "clause-1",
                    "source_text": "profile",
                    "operation": "unresolved",
                    "owner_routes": [],
                    "reason": "wrapper was split incorrectly",
                },
            ],
            "reason": "candidate partition",
        }
        rejected = {
            "unit_verdicts": [
                {
                    "unit_id": "unit-1",
                    "verdict": "complete",
                    "supports_unit_id": "",
                    "reason": "mode selection is represented",
                },
                {
                    "unit_id": "unit-2",
                    "verdict": "unresolved",
                    "supports_unit_id": "",
                    "reason": "wrapper remains unresolved",
                },
            ],
            "clause_verdicts": [{
                "clause_id": "clause-1",
                "verdict": "complete",
                "omitted_owner_routes": [],
                "reason": "no independent demand is omitted",
            }],
            "reason": "wrapper defect",
        }
        state = {
            "active_group": "chain_identity",
            "pending_question": {
                "id": "chain",
                "group": "chain_identity",
                "owner": "chain_rpc",
                "kind": "chain",
                "manual_input_allowed": True,
            },
        }

        with (
            patch(
                "agent.harness.hierarchical_planner.provider_from_config",
                return_value=object(),
            ),
            patch(
                "agent.harness.hierarchical_planner.request_semantic_compilation",
                side_effect=[
                    json.dumps(partition),
                    json.dumps(rejected),
                    json.dumps(rejected),
                    json.dumps(partition),
                    json.dumps(rejected),
                    json.dumps(rejected),
                ],
            ) as compiler,
        ):
            result = begin_semantic_partition(state, text)

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["stage_a_calls"], 2)
        self.assertEqual(result["admission_calls"], 4)
        self.assertEqual(compiler.call_count, 6)
        self.assertTrue(
            any(
                "internally inconsistent" in error
                for error in result["errors"]
            ),
            result["errors"],
        )

    def test_coherent_unresolved_admission_repair_reaches_draft_compilation(self) -> None:
        from agent.harness.hierarchical_planner import begin_semantic_partition

        text = "Use quick profile"
        partition = {
            "semantic_units": [
                {
                    "unit_id": "unit-1",
                    "clause_id": "clause-1",
                    "source_text": "quick",
                    "operation": "domain_request",
                    "owner_routes": [{
                        "owner": "performance",
                        "group": "qps_profile",
                    }],
                    "reason": "registered value",
                },
                {
                    "unit_id": "unit-2",
                    "clause_id": "clause-1",
                    "source_text": "profile",
                    "operation": "unresolved",
                    "owner_routes": [],
                    "reason": "unresolved wrapper",
                },
            ],
            "reason": "incomplete partition",
        }
        rejected = {
            "unit_verdicts": [
                {
                    "unit_id": "unit-1",
                    "verdict": "complete",
                    "supports_unit_id": "",
                    "reason": "registered value is represented",
                },
                {
                    "unit_id": "unit-2",
                    "verdict": "unresolved",
                    "supports_unit_id": "",
                    "reason": "wrapper remains unresolved",
                },
            ],
            "clause_verdicts": [{
                "clause_id": "clause-1",
                "verdict": "complete",
                "omitted_owner_routes": [],
                "reason": "no omitted sibling",
            }],
            "reason": "semantic coverage remains incomplete",
        }
        repaired = {
            **rejected,
            "clause_verdicts": [{
                "clause_id": "clause-1",
                "verdict": "unresolved",
                "omitted_owner_routes": [],
                "reason": "the second span may contain an independent demand",
            }],
            "reason": "coherent unresolved verdict",
        }
        state = {
            "active_group": "chain_identity",
            "pending_question": {
                "id": "chain",
                "group": "chain_identity",
                "owner": "chain_rpc",
                "kind": "chain",
                "manual_input_allowed": True,
            },
        }

        with (
            patch(
                "agent.harness.hierarchical_planner.provider_from_config",
                return_value=object(),
            ),
            patch(
                "agent.harness.hierarchical_planner.request_semantic_compilation",
                side_effect=[
                    json.dumps(partition),
                    json.dumps(rejected),
                    json.dumps(repaired),
                ],
            ) as compiler,
            patch(
                "agent.harness.hierarchical_planner."
                "_partition_requires_independent_proposal",
                return_value=False,
            ),
        ):
            result = begin_semantic_partition(state, text)

        self.assertEqual(result["status"], "compile_owner")
        self.assertEqual(result["stage_a_calls"], 1)
        self.assertEqual(result["admission_calls"], 2)
        self.assertEqual(compiler.call_count, 3)
        self.assertEqual(result["errors"], [])

    def test_product_graph_routes_hypothetical_reset_through_consultation_authority(
        self,
    ) -> None:
        from agent.harness.domains.orientation import resume_question
        from agent.harness.state import new_state
        from tests.agent_live.graph_turn import invoke_product_graph_turn

        text = "If I reset, what would be cleared?"

        class ContractProvider:
            def complete(self, request):
                system = request.messages[0].content
                payload = json.loads(request.messages[1].content)
                if "Stage A semantic partition" in system:
                    response = {
                        "semantic_units": [{
                            "unit_id": "unit-1",
                            "clause_id": "clause-1",
                            "source_text": text,
                            "operation": "consultation",
                            "owner_routes": [{
                                "owner": "orientation",
                                "group": "opening",
                            }],
                            "reason": "hypothetical reset consequence question",
                        }],
                        "reason": "one read-only consultation",
                    }
                elif "Stage A coverage authority" in system:
                    response = {
                        "unit_verdicts": [{
                            "unit_id": "unit-1",
                            "verdict": "complete",
                            "supports_unit_id": "",
                            "reason": "the consultation is represented",
                        }],
                        "clause_verdicts": [{
                            "clause_id": "clause-1",
                            "verdict": "complete",
                            "omitted_owner_routes": [],
                            "reason": "the complete question is represented",
                        }],
                        "reason": "complete",
                    }
                elif "Stage B command compiler" in system:
                    unit_id = payload["semantic_units"][0]["unit_id"]
                    response = {
                        "actions": [{
                            "type": "answer_opening_question",
                            "topic": "reset_help",
                            "source_evidence": text,
                        }],
                        "bindings": [{
                            "unit_id": unit_id,
                            "action_indexes": [0],
                            "disposition": "action",
                            "reason": "compile read-only reset guidance",
                        }],
                        "reason": "compiled",
                    }
                elif "independent admission authority" in system:
                    unit_by_id = {
                        row["unit_id"]: row
                        for row in payload["semantic_units"]
                    }
                    response = {
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
                                        "evidence_quote": text,
                                    }
                                    for argument in row[
                                        "required_value_grounding_arguments"
                                    ]
                                ],
                                "pending_answer_argument": "",
                                "turn_candidate_verdicts": [],
                                "reason": "read-only consultation is grounded",
                            }
                            for row in payload["actions"]
                        ],
                        "unit_verdicts": [
                            {
                                "unit_id": row["unit_id"],
                                "verdict": "complete",
                                "owner_action_ids": row["owner_action_ids"],
                                "evidence_quote": row["source_text"],
                                "omitted_action_type": "",
                                "reason": "the question has one read-only owner",
                            }
                            for row in payload["semantic_units"]
                        ],
                        "reason": "complete and grounded",
                    }
                else:
                    raise AssertionError(system)
                return SimpleNamespace(text=json.dumps(response))

        state = new_state("hypothetical-reset-product-graph", language="en")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {
            "raw": "bsc",
            "canonical": "bsc",
            "status": "confirmed",
        }
        state["confirmed_config"] = {
            "BLOCKCHAIN_NODE": "bsc",
            "CLOUD_REGION": "us-1",
        }
        state["pending_question"] = resume_question(state)
        state["last_user_input"] = text
        before = {
            "target_mode": state["target_mode"],
            "workflow_mode": state["workflow_mode"],
            "chain_identity": dict(state["chain_identity"]),
            "confirmed_config": dict(state["confirmed_config"]),
        }
        provider = ContractProvider()

        with (
            patch(
                "agent.harness.bounded_semantic_lane.provider_from_config",
                side_effect=AssertionError(
                    "contextual prose must not enter the bounded semantic lane"
                ),
            ),
            patch(
                "agent.harness.hierarchical_planner.provider_from_config",
                return_value=provider,
            ),
        ):
            result = invoke_product_graph_turn(
                state,
                allow_semantic_resolver=True,
            )

        self.assertEqual(result["target_mode"], before["target_mode"])
        self.assertEqual(result["workflow_mode"], before["workflow_mode"])
        self.assertEqual(result["chain_identity"], before["chain_identity"])
        self.assertEqual(result["confirmed_config"], before["confirmed_config"])
        self.assertEqual(
            result["pending_question"]["id"],
            "resume_harness_session",
        )
        self.assertIn(
            "harness.orientation.consultation.reset_help",
            {
                item["message_id"]
                for item in (
                    result.get("turn_context") or {}
                ).get("response_manifest") or []
            },
        )

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
                "agent.harness.hierarchical_planner._review_stage_a_partition",
                return_value=((), (333,), frozenset()),
            ),
            patch(
                "agent.harness.hierarchical_planner."
                "_review_stage_a_pending_entailment",
                return_value=((), (101, 102, 103)),
            ),
            patch(
                "agent.harness.hierarchical_planner."
                "_partition_requires_independent_proposal",
                return_value=False,
            ),
            patch(
                "agent.harness.hierarchical_planner._compile_owner_document",
                return_value=({"actions": [], "bindings": []}, (), (211,)),
            ) as owner_compiler,
            patch(
                "agent.harness.hierarchical_planner.build_semantic_plan_draft",
                return_value={"status": "awaiting_clarification"},
            ),
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

    @patch(
        "agent.harness.hierarchical_planner.request_semantic_compilation",
    )
    def test_stage_a_canonicalizes_fenced_request_evidence_before_field_validation(
        self,
        semantic_compilation: Mock,
    ) -> None:
        from agent.harness.hierarchical_planner import (
            _request_stage_a_proposal,
            _stage_a_payload,
            _turn_clauses,
        )
        from agent.harness.state import new_state

        source = (
            "Request first:\n"
            "```json\n"
            '{"jsonrpc":"2.0","method":"eth_blockNumber",'
            '"params":[],"id":1}\n'
            "```"
        )
        state = new_state("fenced-request-evidence", language="en")
        state["active_group"] = "endpoint_process"
        state["pending_question"] = {
            "id": "new_chain_schema_evidence",
            "group": "endpoint_process",
            "manual_input_allowed": True,
            "validation": {
                "value_type": "evidence_contribution",
                "max_length": 65536,
            },
        }
        clauses = _turn_clauses(state, source)
        semantic_compilation.return_value = json.dumps({
            "semantic_units": [
                {
                    "unit_id": f"model-field-{index}",
                    "operation": "pending_answer",
                    "owner_routes": [{
                        "owner": "coordinator",
                        "group": "endpoint_process",
                    }],
                    "reason": "field belongs to the pending evidence",
                }
                for index in range(1, 5)
            ],
            "reason": "request evidence",
        })

        partition, errors, _sizes = _request_stage_a_proposal(
            object(),
            _stage_a_payload(state, source, clauses),
            clauses,
            state,
            "test prompt",
        )

        self.assertEqual(errors, ())
        self.assertEqual(len(partition), 1)
        self.assertEqual(
            partition[0]["unit_id"],
            "__harness_atomic_evidence_clause-1",
        )
        self.assertEqual(partition[0]["source_text"], source)
        self.assertNotIn("source_path", partition[0])
        semantic_compilation.assert_called_once()

    def test_stage_a_atomic_evidence_does_not_swallow_independent_operation(
        self,
    ) -> None:
        from agent.harness.hierarchical_planner import (
            _canonicalize_atomic_evidence_partition,
        )
        from agent.harness.plan_coverage import TurnClause
        from agent.harness.state import new_state

        state = new_state("evidence-with-navigation", language="en")
        state["pending_question"] = {
            "id": "schema_evidence",
            "group": "endpoint_process",
            "validation": {"value_type": "evidence_contribution"},
        }
        units = [
            {
                "unit_id": "evidence",
                "clause_id": "clause-1",
                "operation": "pending_answer",
                "source_text": '{"method":"eth_blockNumber"}',
                "owner_routes": [{
                    "owner": "coordinator",
                    "group": "endpoint_process",
                }],
            },
            {
                "unit_id": "navigation",
                "clause_id": "clause-1",
                "operation": "navigation",
                "source_text": "go back",
                "owner_routes": [{"owner": "orientation", "group": ""}],
            },
        ]

        source, compilation = _canonicalize_atomic_evidence_partition(
            state,
            (TurnClause(
                "clause-1",
                '{"method":"eth_blockNumber"}; go back',
                "structured",
            ),),
            units,
            units,
        )

        self.assertEqual(source, units)
        self.assertEqual(compilation, units)

    def test_stage_a_preserves_valid_structured_route(self) -> None:
        from agent.harness.hierarchical_planner import (
            _validate_partition_document,
        )
        from agent.harness.plan_coverage import TurnClause

        clauses = (
            TurnClause("clause-1", "QPS_MODE: quick", "structured"),
        )
        unit = {
            "unit_id": "__harness_structured_clause-1_1",
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

    def test_stage_a_binds_omitted_structured_provenance_from_atom_id(self) -> None:
        from agent.harness.hierarchical_planner import _validate_partition_document
        from agent.harness.plan_coverage import TurnClause

        source = "QPS_MODE: quick"
        unit = {
            "unit_id": "__harness_structured_clause-1_1",
            "operation": "domain_request",
            "owner_routes": [{
                "owner": "performance",
                "group": "qps_profile",
            }],
            "reason": "structured workflow value",
        }

        partition, errors = _validate_partition_document(
            json.dumps({"semantic_units": [unit], "reason": "complete"}),
            (TurnClause("clause-1", source, "structured"),),
        )

        self.assertEqual(errors, ())
        self.assertEqual(partition[0]["clause_id"], "clause-1")
        self.assertEqual(partition[0]["source_text"], source)
        self.assertEqual(partition[0]["source_path"], "QPS_MODE")

    def test_structured_rpc_intake_reaches_stage_b_as_registry_authority(
        self,
    ) -> None:
        from agent.harness.hierarchical_planner import (
            _stage_b_payload,
            _validate_partition_document,
        )
        from agent.harness.plan_coverage import TurnClause
        from agent.harness.state import new_state

        source = 'validation_endpoint: "http://127.0.0.1:8545"'
        clauses = (TurnClause("clause-1", source, "structured"),)
        unit = {
            "unit_id": "__harness_structured_clause-1_1",
            "clause_id": "clause-1",
            "source_text": source,
            "operation": "domain_request",
            "owner_routes": [{
                "owner": "chain_rpc",
                "group": "endpoint_process",
            }],
            "reason": "structured RPC endpoint",
        }

        partition, errors = _validate_partition_document(
            json.dumps({"semantic_units": [unit], "reason": "complete"}),
            clauses,
        )

        self.assertEqual(errors, ())
        self.assertEqual(
            partition[0]["registered_intake"],
            {
                "action_type": "rpc_catalog_command",
                "owner": "chain_rpc",
                "target_group": "endpoint_process",
                "alias": "validation_endpoint",
                "fixed_arguments": {"catalog_command": "set_endpoint"},
                "value_semantics": "direct_value",
                "value_argument": "rpc_endpoint",
            },
        )
        payload = _stage_b_payload(
            new_state("structured-rpc-intake", language="en"),
            "chain_rpc",
            frozenset({"endpoint_process"}),
            partition,
            (partition[0]["unit_id"],),
        )
        semantic_unit = payload["semantic_units"][0]
        self.assertEqual(
            semantic_unit["registered_intake"],
            partition[0]["registered_intake"],
        )
        self.assertEqual(
            semantic_unit["semantic_value"],
            "http://127.0.0.1:8545",
        )

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

    def test_stage_a_derives_pending_answer_group_from_active_contract(
        self,
    ) -> None:
        from agent.harness.hierarchical_planner import (
            _validate_partition_document,
        )
        from agent.harness.plan_coverage import TurnClause
        from agent.harness.questions import manual_question, question_text
        from agent.harness.state import new_state

        state = new_state("pending-route-contract", language="en")
        state["pending_question"] = manual_question(
            "chain_identity",
            "chain",
            question_text("question.chain_rpc.chain.prompt"),
            owner="chain_rpc",
            field="chain",
        )
        unit = {
            "unit_id": "unit-1",
            "clause_id": "clause-1",
            "source_text": "arbitrary wrapper",
            "operation": "pending_answer",
            "owner_routes": [{
                "owner": "coordinator",
                "group": "coordinator",
            }],
            "reason": "model supplied an owner name as the group",
        }

        partition, errors = _validate_partition_document(
            json.dumps({"semantic_units": [unit], "reason": "answer"}),
            (TurnClause("clause-1", "arbitrary wrapper", "prose"),),
            state=state,
        )

        self.assertEqual(errors, ())
        self.assertEqual(
            partition[0]["owner_routes"],
            [{"owner": "coordinator", "group": "chain_identity"}],
        )

    def test_stage_a_does_not_invent_ambiguous_pending_routes(self) -> None:
        from agent.harness.hierarchical_planner import (
            _validate_partition_document,
        )
        from agent.harness.plan_coverage import TurnClause
        from agent.harness.questions import manual_question, question_text
        from agent.harness.state import new_state

        state = new_state("pending-route-negatives", language="en")
        state["pending_question"] = manual_question(
            "chain_identity",
            "chain",
            question_text("question.chain_rpc.chain.prompt"),
            owner="chain_rpc",
            field="chain",
        )
        base = {
            "unit_id": "unit-1",
            "clause_id": "clause-1",
            "source_text": "arbitrary wrapper",
            "operation": "pending_answer",
            "owner_routes": [{
                "owner": "coordinator",
                "group": "coordinator",
            }],
            "reason": "invalid route",
        }
        cases = {
            "missing_pending": ({}, base),
            "wrong_owner": (
                state,
                {
                    **base,
                    "owner_routes": [{
                        "owner": "orientation",
                        "group": "coordinator",
                    }],
                },
            ),
            "multiple_routes": (
                state,
                {
                    **base,
                    "owner_routes": [
                        {"owner": "coordinator", "group": "coordinator"},
                        {"owner": "coordinator", "group": "chain_identity"},
                    ],
                },
            ),
            "non_pending_operation": (
                state,
                {**base, "operation": "domain_request"},
            ),
        }
        for name, (case_state, case_unit) in cases.items():
            with self.subTest(name=name):
                partition, errors = _validate_partition_document(
                    json.dumps({
                        "semantic_units": [case_unit],
                        "reason": "invalid",
                    }),
                    (TurnClause(
                        "clause-1",
                        "arbitrary wrapper",
                        "prose",
                    ),),
                    state=case_state,
                )
                self.assertTrue(errors)
                self.assertNotEqual(
                    partition[0].get("owner_routes"),
                    [{"owner": "coordinator", "group": "chain_identity"}],
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

    def test_declared_option_matching_tolerates_terminal_punctuation(self) -> None:
        from agent.harness.questions import (
            exact_option_answer,
            exact_option_prefix_answer,
        )

        question = {
            "kind": "yes_no",
            "options": [
                {"id": "accept", "value": True},
                {"id": "decline", "value": False},
            ],
        }

        self.assertEqual(exact_option_answer("N。", question), (True, False))
        self.assertEqual(exact_option_answer("yes!", question), (True, True))
        self.assertEqual(
            exact_option_prefix_answer(
                "N, keep the current chain and change target mode.",
                question,
            ),
            (True, False, 3),
        )
        self.assertEqual(
            exact_option_prefix_answer("not yet", question),
            (False, None, 0),
        )

    def test_unique_option_prefix_preserves_unresolved_sibling_span(self) -> None:
        from agent.harness.hierarchical_planner import (
            _canonicalize_unique_option_pending_partition,
        )
        from agent.harness.plan_coverage import segment_user_turn
        from agent.harness.state import new_state

        state = new_state("option-prefix-sibling", language="en")
        state["active_group"] = "opening"
        state["pending_question"] = {
            "id": "accept_recommendation",
            "group": "opening",
            "kind": "yes_no",
            "manual_input_allowed": False,
            "options": [
                {"id": "accept", "value": True},
                {"id": "decline", "value": False},
            ],
        }
        clauses = segment_user_turn(
            "N, keep Solana and configure a real-node benchmark."
        )
        partition = [{
            "unit_id": "unit-unresolved",
            "clause_id": clauses[0].clause_id,
            "start": 0,
            "end": len(clauses[0].text),
            "source_text": clauses[0].text,
            "operation": "unresolved",
            "owner_routes": [],
            "reason": "model did not resolve the compound turn",
        }]

        source, compilation = _canonicalize_unique_option_pending_partition(
            state,
            clauses,
            partition,
            partition,
        )

        self.assertEqual(
            [unit["operation"] for unit in source],
            ["pending_answer", "unresolved"],
        )
        self.assertEqual(source, compilation)
        self.assertEqual(source[0]["source_text"], "N, ")
        self.assertEqual(
            source[1]["source_text"],
            "keep Solana and configure a real-node benchmark.",
        )
        self.assertEqual(source[0]["end"], source[1]["start"])

    def test_uncompilable_sibling_route_does_not_erase_contract_option_unit(
        self,
    ) -> None:
        from agent.harness.hierarchical_planner import _validate_partition_document
        from agent.harness.plan_coverage import segment_user_turn
        from agent.harness.state import new_state

        text = "N, because I want to tune the CPU bottleneck threshold."
        clauses = segment_user_turn(text)
        state = new_state("uncompilable-option-sibling", language="en")
        state["active_group"] = "advanced_tuning"
        state["pending_question"] = {
            "id": "advanced_tuning_confirm",
            "group": "advanced_tuning",
            "owner": "performance",
            "kind": "yes_no",
            "manual_input_allowed": False,
            "options": [
                {"id": "yes", "value": True},
                {"id": "no", "value": False},
            ],
        }
        document = json.dumps({
            "semantic_units": [{
                "unit_id": "stage-a-whole-clause",
                "clause_id": clauses[0].clause_id,
                "start": 0,
                "end": len(text),
                "source_text": text,
                "operation": "domain_request",
                "owner_routes": [{
                    "owner": "performance",
                    "group": "advanced_tuning",
                }],
                "reason": "The user wants to tune an advanced threshold.",
            }],
            "reason": "one compound pending-answer turn",
        })

        partition, errors = _validate_partition_document(
            document,
            clauses,
            state=state,
        )

        self.assertEqual(errors, ())
        self.assertEqual(
            [unit["operation"] for unit in partition],
            ["pending_answer", "unresolved"],
        )
        self.assertEqual(partition[0]["source_text"], "N, ")
        self.assertEqual(
            partition[1]["source_text"],
            "because I want to tune the CPU bottleneck threshold.",
        )

    def test_option_prefix_never_silently_converts_absorbed_sibling_to_context(
        self,
    ) -> None:
        from agent.harness.hierarchical_planner import (
            _canonicalize_unique_option_pending_partition,
        )
        from agent.harness.plan_coverage import segment_user_turn
        from agent.harness.state import new_state

        text = "2, keep Solana and switch to real-node."
        clauses = segment_user_turn(text)
        state = new_state("option-prefix-no-silent-loss", language="en")
        state["active_group"] = "opening"
        state["pending_question"] = {
            "id": "accept_recommendation",
            "group": "opening",
            "kind": "numbered_choice",
            "options": [
                {"id": "1", "value": True},
                {"id": "2", "value": False},
            ],
        }
        partition = [{
            "unit_id": "unit-whole",
            "clause_id": clauses[0].clause_id,
            "start": 0,
            "end": len(text),
            "source_text": text,
            "operation": "pending_answer",
            "owner_routes": [{"owner": "coordinator", "group": "opening"}],
            "reason": "model incorrectly absorbed the complete clause",
        }]

        source, compilation = _canonicalize_unique_option_pending_partition(
            state,
            clauses,
            partition,
            partition,
        )

        self.assertEqual(
            [unit["operation"] for unit in source],
            ["pending_answer", "unresolved"],
        )
        self.assertEqual(source, compilation)
        self.assertEqual(
            source[1]["source_text"],
            "keep Solana and switch to real-node.",
        )

    def test_shared_source_routes_preserve_candidate_and_unresolved_atom(
        self,
    ) -> None:
        from agent.harness.hierarchical_planner import (
            _expand_partition_routes,
            _merge_owner_documents,
        )
        from agent.harness.plan_coverage import segment_user_turn
        from agent.harness.semantic_admission import (
            prepare_hierarchical_candidate,
        )
        from agent.harness.state import new_state

        text = "Use real-node and keep the current QPS goal."
        clauses = segment_user_turn(text)
        source = [{
            "unit_id": "unit-shared",
            "clause_id": clauses[0].clause_id,
            "start": 0,
            "end": len(text),
            "source_text": text,
            "operation": "domain_request",
            "owner_routes": [
                {"owner": "chain_rpc", "group": "target_mode"},
                {"owner": "performance", "group": "qps_profile"},
            ],
            "reason": "one source span has two independently owned demands",
        }]
        expanded = _expand_partition_routes(source)
        candidate = _merge_owner_documents(
            expanded,
            expanded,
            {
                "chain_rpc": {
                    "actions": [{
                        "type": "choose_target_mode",
                        "target_mode": "real-node",
                        "target_mode_explicit": True,
                        "source_evidence": "real-node",
                    }],
                    "bindings": [{
                        "unit_id": expanded[0]["unit_id"],
                        "action_indexes": [0],
                        "disposition": "action",
                        "reason": "the target mode is explicit",
                    }],
                },
                "performance": {
                    "actions": [],
                    "bindings": [{
                        "unit_id": expanded[1]["unit_id"],
                        "action_indexes": [],
                        "disposition": "unresolved",
                        "reason": "the QPS goal has no concrete profile",
                    }],
                },
            },
        )
        state = new_state("shared-source-partial-draft", language="en")

        _prepared, validation = prepare_hierarchical_candidate(
            json.dumps(candidate),
            state,
            clauses,
            pending_choice_unit_ids=frozenset(),
        )

        self.assertFalse(validation.valid)
        self.assertEqual(validation.errors, ())
        self.assertEqual(len(validation.unresolved_units), 1)
        self.assertEqual(
            [
                unit["disposition"]
                for unit in candidate["semantic_units"]
            ],
            ["action", "unresolved"],
        )
        self.assertEqual(
            candidate["semantic_units"][0]["parent_unit_id"],
            "unit-shared",
        )
        self.assertEqual(
            candidate["semantic_units"][1]["parent_unit_id"],
            "unit-shared",
        )

    def test_stage_a_reviewer_sees_contract_option_before_unresolved_sibling(
        self,
    ) -> None:
        from types import SimpleNamespace
        from unittest.mock import patch

        from agent.harness.domains.orientation import recommendation_question
        from agent.harness.hierarchical_planner import (
            begin_semantic_partition,
            compile_next_owner,
            review_semantic_plan,
        )
        from agent.harness.semantic_drafts import semantic_hash
        from agent.harness.state import new_state

        class Provider:
            def complete(self, request):
                system = request.messages[0].content
                payload = json.loads(request.messages[1].content)
                if "Stage A semantic partition" in system:
                    units = [
                        {
                            "unit_id": f"unit-{index}",
                            "clause_id": clause["clause_id"],
                            "source_text": clause["text"],
                            "operation": "unresolved",
                            "owner_routes": [],
                            "reason": "the model left this source unresolved",
                        }
                        for index, clause in enumerate(
                            payload["clauses"],
                            start=1,
                        )
                    ]
                    return SimpleNamespace(text=json.dumps({
                        "semantic_units": units,
                        "reason": "unresolved compound turn",
                    }))
                if "Stage A proposal convergence authority" in system:
                    return SimpleNamespace(text=json.dumps({
                        "selected_proposal": "primary",
                        "primary_hash": payload["primary"]["hash"],
                        "secondary_hash": payload["secondary"]["hash"],
                        "reason": "both proposals preserve the unresolved sibling",
                    }))
                if "Stage A coverage authority" in system:
                    units = payload["semantic_units"]
                    return SimpleNamespace(text=json.dumps({
                        "unit_verdicts": [
                            {
                                "unit_id": unit["unit_id"],
                                "verdict": (
                                    "complete"
                                    if unit["operation"] == "pending_answer"
                                    else "unresolved"
                                ),
                                "supports_unit_id": "",
                                "reason": "reviewed from the canonical source",
                            }
                            for unit in units
                        ],
                        "clause_verdicts": [
                            {
                                "clause_id": clause["clause_id"],
                                "verdict": (
                                    "complete"
                                    if index == 0
                                    else "unresolved"
                                ),
                                "omitted_owner_routes": [],
                                "reason": "the sibling still needs semantic work",
                            }
                            for index, clause in enumerate(payload["clauses"])
                        ],
                        "reason": "canonical pending option retained",
                    }))
                if "Stage B command compiler" in system:
                    unit = payload["semantic_units"][0]
                    return SimpleNamespace(text=json.dumps({
                        "actions": [{
                            "type": "answer_pending",
                            "selected_value": False,
                            "source_evidence": unit["source_text"],
                        }],
                        "bindings": [{
                            "unit_id": unit["unit_id"],
                            "action_indexes": [0],
                            "disposition": "action",
                            "reason": "compiled from the signed option contract",
                        }],
                        "reason": "compiled pending selection",
                    }))
                raise AssertionError(system)

        state = new_state("option-before-stage-a-review", language="zh")
        state["active_group"] = "opening"
        state["pending_question"] = recommendation_question(state)
        text = "N。保留刚才的目标，并告诉我下一步。"
        state["turn_index"] = 2
        state["turn_context"] = {
            "text": text,
            "product_head": {
                "product_authority_id": (
                    "authority:option-before-stage-a-review"
                ),
                "revision": 2,
                "checkpoint_thread_id": (
                    "checkpoint-thread:option-before-stage-a-review"
                ),
                "checkpoint_id": (
                    "checkpoint:option-before-stage-a-review"
                ),
                "state_fingerprint": semantic_hash({
                    "thread_id": "option-before-stage-a-review",
                }),
            },
        }
        with patch(
            "agent.harness.hierarchical_planner.provider_from_config",
            return_value=Provider(),
        ):
            document = begin_semantic_partition(state, text)

        self.assertEqual(document["status"], "compile_owner", document)
        self.assertEqual(
            [
                unit["operation"]
                for unit in document["source_partition"]
            ],
            ["pending_answer", "unresolved"],
        )
        self.assertEqual(
            document["owner_requests"],
            [{
                "owner": "coordinator",
                "unit_ids": [
                    document["source_partition"][0]["unit_id"],
                ],
                "groups": ["opening"],
            }],
        )
        with patch(
            "agent.harness.hierarchical_planner.provider_from_config",
            return_value=Provider(),
        ):
            while document["status"] == "compile_owner":
                document = compile_next_owner(state, document)
            result = review_semantic_plan(state, document)

        self.assertEqual(result["actions"], [])
        draft = result["semantic_draft"]
        self.assertEqual(draft["status"], "awaiting_clarification")
        self.assertEqual(
            [row["action"]["type"] for row in draft["candidates"]],
            ["answer_pending"],
        )
        self.assertEqual(len(draft["unresolved_atoms"]), 1)
        self.assertIn(
            "保留刚才的目标",
            draft["unresolved_atoms"][0]["source_text"],
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
        self.assertIn(
            "Request identity is bound by the Harness transport",
            prompt,
        )
        self.assertNotIn(
            "exactly these keys: plan_hash",
            prompt,
        )

    def test_product_resolver_end_to_end_contract_without_internal_boundary_mocks(
        self,
    ) -> None:
        from tests.agent_live.graph_turn import resolve_product_action_queue_for_test as resolve_product_action_queue
        from agent.harness.state import new_state

        whole_plan_attempts: list[bool] = []

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
                    is_contract_repair = bool(
                        payload.get("admission_contract_repair")
                    )
                    whole_plan_attempts.append(is_contract_repair)
                    actions = payload["actions"]
                    units = payload["semantic_units"]
                    unit_by_id = {
                        row["unit_id"]: row for row in units
                    }
                    response = {
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
                                ] + (
                                    [{
                                        "unit_id": row["unit_ids"][0],
                                        "quote": "fake-node",
                                        "relation": "support",
                                        "support_relation": "operation_restatement",
                                    }]
                                    if row["action"]["type"]
                                    == "choose_target_mode"
                                    and not is_contract_repair
                                    else []
                                ),
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
                elif "closed-enum value grounding authority" in system:
                    response = _closed_enum_review_response(payload)
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
        from agent.harness.action_registry import (
            validate_semantic_consensus_receipt,
        )

        for action in result["actions"]:
            validate_semantic_consensus_receipt(
                action,
                thread_id="hierarchical-e2e",
                session_id="hierarchical-e2e",
                submitted_turn_index=0,
            )
            receipt = action["_semantic_consensus_receipt"]
            self.assertEqual(receipt["request_count"], 7)
            self.assertEqual(
                receipt["review_ids"],
                [
                    "jury_1/attempt_1",
                    "jury_1/attempt_2",
                    "jury_2/attempt_1",
                    "jury_2/attempt_2",
                    "jury_3/attempt_1",
                    "jury_3/attempt_2",
                    "closed_enum_grounding",
                ],
            )
            self.assertEqual(
                len(receipt["review_hashes"]),
                receipt["request_count"],
            )
        self.assertEqual(
            whole_plan_attempts,
            [False, True, False, True, False, True],
        )
        self.assertEqual(result["planner_metrics"]["model_calls"], 11)
        self.assertEqual(result["planner_metrics"]["admission_calls"], 8)

    def test_semantic_draft_recompiles_clarification_through_real_admission(
        self,
    ) -> None:
        from agent.harness.contracts import ActionProposal
        from agent.harness.coordinator import (
            _apply_handler_result,
            _consume_planner_queue,
            _prepare_ready_semantic_draft_finalization,
            admit_turn_step,
            apply_coordinator_action,
        )
        from agent.harness.hierarchical_planner import (
            begin_semantic_partition,
            compile_next_owner,
            review_semantic_plan,
        )
        from agent.harness.invariants import validate_state
        from agent.harness.semantic_drafts import semantic_hash
        from agent.harness.state import new_state

        model_calls: list[tuple[str, bool]] = []

        class ContractProvider:
            def complete(self, request):
                system = request.messages[0].content
                payload = json.loads(request.messages[1].content)
                has_resolution = bool(
                    payload.get("semantic_draft_resolutions")
                )
                clarification_turn = bool(
                    (payload.get("pending_question") or {}).get(
                        "semantic_draft_binding"
                    )
                ) and str(payload.get("user_text") or "").strip() == "quick"
                model_calls.append((system, has_resolution))
                if "semantic-draft clarification reviewer" in system:
                    response = {
                        "contract_hash": payload["contract_hash"],
                        "verdict": "clarifies_atom",
                        "evidence_quote": payload["user_text"],
                        "reason": "the complete turn supplies the profile value",
                    }
                elif "Stage A proposal convergence authority" in system:
                    response = {
                        "selected_proposal": "primary",
                        "primary_hash": payload["primary"]["hash"],
                        "secondary_hash": payload["secondary"]["hash"],
                        "reason": (
                            "both proposals preserve the same unresolved "
                            "profile demand"
                        ),
                    }
                elif "Stage A semantic partition" in system:
                    clauses = payload["clauses"]
                    response = {
                        "semantic_units": [{
                            "unit_id": "unit-draft-answer",
                            "clause_id": clauses[0]["clause_id"],
                            "source_text": clauses[0]["text"],
                            "operation": "pending_answer",
                            "owner_routes": [{
                                "owner": "coordinator",
                                "group": str(
                                    payload["pending_question"]["group"]
                                ),
                            }],
                            "reason": "answers the active draft atom",
                        }],
                        "reason": "draft clarification answer",
                    } if clarification_turn else {
                        "semantic_units": [
                            {
                                "unit_id": "unit-chain",
                                "clause_id": clauses[0]["clause_id"],
                                "source_text": clauses[0]["text"],
                                "operation": "domain_request",
                                "owner_routes": [{
                                    "owner": "chain_rpc",
                                    "group": "chain_identity",
                                }],
                                "reason": "explicit chain selection",
                            },
                            {
                                "unit_id": "unit-profile",
                                "clause_id": clauses[1]["clause_id"],
                                "source_text": clauses[1]["text"],
                                "operation": (
                                    "domain_request"
                                    if has_resolution
                                    else "unresolved"
                                ),
                                "owner_routes": (
                                    [{
                                        "owner": "performance",
                                        "group": "qps_profile",
                                    }]
                                    if has_resolution
                                    else []
                                ),
                                "reason": (
                                    "the user clarified the QPS profile"
                                    if has_resolution
                                    else "the QPS profile value is missing"
                                ),
                            },
                        ],
                        "reason": "lossless two-demand partition",
                    }
                elif "Stage A coverage authority" in system:
                    clarification_unit_id = str(
                        payload["semantic_units"][0]["unit_id"]
                    )
                    response = {
                        "unit_verdicts": [{
                            "unit_id": clarification_unit_id,
                            "verdict": "complete",
                            "supports_unit_id": "",
                            "reason": "the pending answer is represented",
                        }],
                        "clause_verdicts": [{
                            "clause_id": payload["clauses"][0]["clause_id"],
                            "verdict": "complete",
                            "omitted_owner_routes": [],
                            "reason": "the clarification clause is complete",
                        }],
                        "reason": "clarification coverage reviewed",
                    } if clarification_turn else {
                        "unit_verdicts": [
                            {
                                "unit_id": "unit-chain",
                                "verdict": "complete",
                                "supports_unit_id": "",
                                "reason": "chain demand is represented",
                            },
                            {
                                "unit_id": "unit-profile",
                                "verdict": (
                                    "complete"
                                    if has_resolution
                                    else "unresolved"
                                ),
                                "supports_unit_id": "",
                                "reason": (
                                    "clarified profile demand is represented"
                                    if has_resolution
                                    else "profile value still needs evidence"
                                ),
                            },
                        ],
                        "clause_verdicts": [
                            {
                                "clause_id": payload["clauses"][0][
                                    "clause_id"
                                ],
                                "verdict": "complete",
                                "omitted_owner_routes": [],
                                "reason": "chain clause is complete",
                            },
                            {
                                "clause_id": payload["clauses"][1][
                                    "clause_id"
                                ],
                                "verdict": (
                                    "complete"
                                    if has_resolution
                                    else "unresolved"
                                ),
                                "omitted_owner_routes": [],
                                "reason": (
                                    "profile clause is complete"
                                    if has_resolution
                                    else "profile clause needs clarification"
                                ),
                            },
                        ],
                        "reason": "coverage reviewed",
                    }
                elif "Stage B command compiler" in system:
                    unit = payload["semantic_units"][0]
                    if payload["owner"] == "coordinator":
                        binding = payload["pending_question"][
                            "semantic_draft_binding"
                        ]
                        action = {
                            "type": "resolve_semantic_draft_atom",
                            "draft_id": binding["draft_id"],
                            "revision": binding["revision"],
                            "atom_id": binding["atom_id"],
                            "resolution": "quick",
                            "source_evidence": "quick",
                        }
                    elif payload["owner"] == "chain_rpc":
                        action = {
                            "type": "choose_chain",
                            "chain_text": "solana",
                            "source_evidence": "solana",
                        }
                    else:
                        self_resolution = str(
                            unit.get("resolution_evidence") or ""
                        )
                        if self_resolution != "quick":
                            raise AssertionError(
                                "Stage B did not receive bound clarification"
                            )
                        action = {
                            "type": "set_qps_mode",
                            "qps_mode": "quick",
                            "mutation_explicit": True,
                            "source_evidence": self_resolution,
                        }
                    response = {
                        "actions": [action],
                        "bindings": [{
                            "unit_id": unit["unit_id"],
                            "action_indexes": [0],
                            "disposition": "action",
                            "reason": "compiled by authoritative owner",
                        }],
                        "reason": "compiled",
                    }
                elif "independent admission authority" in system:
                    unit_by_id = {
                        row["unit_id"]: row
                        for row in payload["semantic_units"]
                    }
                    response = {
                        "action_verdicts": [
                            {
                                "action_id": row["action_id"],
                                "verdict": "admit",
                                "unit_ids": row["unit_ids"],
                                "evidence": [
                                    {
                                        "unit_id": unit_id,
                                        "quote": unit_by_id[unit_id][
                                            "source_text"
                                        ],
                                        "relation": "direct",
                                        "support_relation": "",
                                    }
                                    for unit_id in row["unit_ids"]
                                ],
                                "grounded_arguments": [
                                    {
                                        "argument_name": argument,
                                        "evidence_quote": row["action"][
                                            "source_evidence"
                                        ],
                                    }
                                    for argument in row[
                                        "required_value_grounding_arguments"
                                    ]
                                ],
                                "pending_answer_argument": "",
                                "turn_candidate_verdicts": [],
                                "reason": "all values have exact evidence",
                            }
                            for row in payload["actions"]
                        ],
                        "unit_verdicts": [
                            {
                                "unit_id": row["unit_id"],
                                "verdict": "complete",
                                "owner_action_ids": row[
                                    "owner_action_ids"
                                ],
                                "evidence_quote": row["source_text"],
                                "omitted_action_type": "",
                                "reason": "the complete demand is owned",
                            }
                            for row in payload["semantic_units"]
                        ],
                        "reason": "complete plan admitted",
                    }
                elif "closed-enum value grounding authority" in system:
                    response = _closed_enum_review_response(payload)
                else:
                    raise AssertionError(system)
                return SimpleNamespace(text=json.dumps(response))

        def compile_plan(state, text):
            document = begin_semantic_partition(state, text)
            while document["status"] == "compile_owner":
                document = compile_next_owner(state, document)
            return review_semantic_plan(state, document)

        text = "Use solana.\nConfigure the benchmark profile."
        state = new_state("draft-real-admission", language="en")
        state["turn_index"] = 4
        state["turn_context"] = {
            "text": text,
            "product_head": {
                "product_authority_id": "authority:draft-real-admission",
                "revision": 4,
                "checkpoint_thread_id": (
                    "checkpoint-thread:draft-real-admission"
                ),
                "checkpoint_id": "checkpoint:draft-real-admission",
                "state_fingerprint": semantic_hash({
                    "thread_id": "draft-real-admission",
                }),
            },
        }
        with patch(
            "agent.harness.hierarchical_planner.provider_from_config",
            return_value=ContractProvider(),
        ):
            first_queue = compile_plan(state, text)
            self.assertIn("semantic_draft", first_queue)
            state = _consume_planner_queue(state, first_queue)
            draft = state["semantic_plan_draft"]
            result = apply_coordinator_action(
                state,
                ActionProposal(
                    action_id="resolve-profile",
                    action_type="resolve_semantic_draft_atom",
                    arguments={
                        "draft_id": draft["draft_id"],
                        "revision": draft["revision"],
                        "atom_id": draft["active_atom_id"],
                        "resolution": "quick",
                        "source_evidence": "quick",
                    },
                    confidence="high",
                ),
            )
            state = _apply_handler_result(
                state,
                result,
                owner="coordinator",
            )
            transition = _prepare_ready_semantic_draft_finalization(state)
            self.assertEqual(transition["phase"], "plan")
            final_queue = compile_plan(
                state,
                state["turn_context"]["text"],
            )
            self.assertNotIn("semantic_draft", final_queue)
            self.assertEqual(
                [row["type"] for row in final_queue["actions"]],
                ["choose_chain", "set_qps_mode"],
                final_queue,
            )
            state = _consume_planner_queue(state, final_queue)
            admitted = admit_turn_step(state)

        self.assertEqual(admitted["semantic_plan_draft"], {})
        self.assertEqual(
            [
                row["action_type"]
                for row in admitted["action_queue"]
            ],
            ["choose_chain", "set_qps_mode"],
        )
        receipts = {
            json.dumps(
                row["admission_metadata"][
                    "semantic_draft_finalization_receipt"
                ],
                sort_keys=True,
            )
            for row in admitted["action_queue"]
        }
        self.assertEqual(len(receipts), 1)
        self.assertTrue(any(
            "independent admission authority" in system
            for system, _has_resolution in model_calls
        ))
        validate_state(admitted)
        import tempfile
        from pathlib import Path

        from agent.harness.graph import AnyChainGraphRuntime

        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "agent.harness.hierarchical_planner.provider_from_config",
            return_value=ContractProvider(),
        ):
            checkpoint_path = Path(tmpdir) / "semantic-runtime.sqlite"
            runtime = AnyChainGraphRuntime(
                thread_id="semantic-runtime-e2e",
                checkpoint_path=checkpoint_path,
            )
            first = runtime.invoke(text, language="en")
            from agent.harness.state import STATE_SCHEMA_VERSION

            self.assertEqual(first["schema_version"], STATE_SCHEMA_VERSION)
            self.assertTrue(first["semantic_plan_draft"], first)
            self.assertEqual(
                {
                    unit["unit_id"]
                    for unit in first["turn_receipt"]["semantic_units"]
                    if unit["disposition"] == "unresolved"
                },
                set(first["turn_receipt"]["unresolved_units"]),
            )
            self.assertFalse(first["turn_receipt"]["admitted_action_ids"])
            runtime_draft = deepcopy(first["semantic_plan_draft"])
            self.assertEqual(
                runtime.snapshot()["semantic_plan_draft"]["draft_id"],
                runtime_draft["draft_id"],
            )
            runtime.close()

            resumed = AnyChainGraphRuntime(
                thread_id="semantic-runtime-e2e",
                checkpoint_path=checkpoint_path,
            )
            restored = resumed.snapshot()
            self.assertTrue(
                restored["semantic_plan_draft"],
                {
                    "checkpoint_recovery": restored.get(
                        "checkpoint_recovery"
                    ),
                    "audit_events": restored.get("audit_events"),
                },
            )
            self.assertEqual(
                restored["semantic_plan_draft"]["draft_id"],
                runtime_draft["draft_id"],
            )
            final = resumed.invoke("quick", language="en")
            resumed.close()

        self.assertEqual(final["semantic_plan_draft"], {}, final)
        receipt = next(
            item
            for item in final["audit_events"]
            if item.get("event") == "semantic_draft_finalized"
        )
        self.assertEqual(
            set(receipt["final_action_ids"]),
            {
                item["action_id"]
                for item in final["action_queue"]
            }
            | {
                item["action_id"]
                for item in final["audit_events"]
                if item.get("event")
                == "semantic_draft_finalization_action_applied"
            },
        )
        validate_state(final)

        from agent.harness.contracts import FailureDescriptor, HandlerResult
        from agent.harness.domains.runtime import DOMAIN_RUNTIME, DomainRuntime

        blocked_performance = DomainRuntime(
            apply_action=lambda _state, _action: HandlerResult(
                blocker=FailureDescriptor(
                    code="harness.failure.test.finalization_blocked",
                    source=__name__,
                ),
                completion="blocked",
            ),
        )
        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "agent.harness.hierarchical_planner.provider_from_config",
            return_value=ContractProvider(),
        ):
            checkpoint_path = Path(tmpdir) / "semantic-rollback.sqlite"
            runtime = AnyChainGraphRuntime(
                thread_id="semantic-runtime-rollback",
                checkpoint_path=checkpoint_path,
            )
            awaiting = runtime.invoke(text, language="en")
            self.assertEqual(
                awaiting["semantic_plan_draft"]["status"],
                "awaiting_clarification",
            )
            with patch.dict(
                DOMAIN_RUNTIME,
                {"performance": blocked_performance},
            ):
                recovered = runtime.invoke("quick", language="en")
            runtime.close()

        self.assertFalse(
            recovered["chain_identity"],
            {
                "chain_identity": recovered.get("chain_identity"),
                "failure_recovery": recovered.get("failure_recovery"),
                "pending_question": recovered.get("pending_question"),
                "action_queue": recovered.get("action_queue"),
            },
        )
        self.assertEqual(recovered["failure_recovery"]["status"], "pending")
        self.assertEqual(
            recovered["semantic_plan_draft"]["status"],
            "stale",
        )
        validate_state(recovered)

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
                {
                    "type": "analyze_evidence",
                    "evidence": "RuntimeError: endpoint failed",
                },
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

                expected = (
                    "typed-option-only"
                    if action["type"] == "approve_final_benchmark"
                    else "outside unit route"
                )
                self.assertTrue(any(expected in error for error in errors), errors)

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
            "routing_purposes": [{
                "action_type": "choose_chain",
                "owner": "chain_rpc",
                "purpose": "choose a chain",
                "semantic_operations": ["domain_request"],
                "route_groups": ["chain_identity"],
            }],
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
        ) as compiler:
            errors, _sizes, _redundant = _review_stage_a_partition(
                object(),
                payload,
                partition,
            )

        self.assertTrue(any("omitted demand" in error for error in errors))
        self.assertEqual(
            compiler.call_args.kwargs["request_payload"]["routing_purposes"],
            payload["routing_purposes"],
        )

    def test_pending_entailment_jury_rejects_unrelated_workflow_request(self) -> None:
        from agent.harness.hierarchical_planner import (
            _review_stage_a_pending_entailment,
        )

        source = "I need to benchmark a different chain."
        payload = {
            "pending_question": {
                "id": "reset_confirm",
                "group": "opening",
                "options": [
                    {"id": "yes", "value": True},
                    {"id": "no", "value": False},
                ],
            },
            "contract_proven_pending_prefixes": [],
            "pending_typed_candidates": [],
        }
        partition = [{
            "unit_id": "unit-1",
            "clause_id": "clause-1",
            "source_text": source,
            "operation": "pending_answer",
            "owner_routes": [{"owner": "coordinator", "group": "opening"}],
        }]
        claim_hash = ""

        def response(*_args, **kwargs):
            nonlocal claim_hash
            claim_hash = kwargs["request_payload"]["claim_hash"]
            return json.dumps({
                "claim_hash": claim_hash,
                "verdict": "different_request",
                "selected_value": None,
                "evidence_quote": "different chain",
                "reason": "the source requests another workflow",
            })

        with patch(
            "agent.harness.hierarchical_planner.request_semantic_compilation",
            side_effect=response,
        ) as compiler:
            errors, sizes = _review_stage_a_pending_entailment(
                object(), payload, partition
            )

        self.assertEqual(compiler.call_count, 3)
        self.assertEqual(len(sizes), 3)
        self.assertTrue(any("quorum rejected" in error for error in errors))

    def test_pending_entailment_jury_requires_two_answer_votes(self) -> None:
        from agent.harness.hierarchical_planner import (
            _review_stage_a_pending_entailment,
        )

        source = "Go ahead and clear the workflow."
        payload = {
            "pending_question": {
                "id": "reset_confirm",
                "group": "opening",
                "options": [
                    {"id": "yes", "value": True},
                    {"id": "no", "value": False},
                ],
            },
            "contract_proven_pending_prefixes": [],
            "pending_typed_candidates": [],
        }
        partition = [{
            "unit_id": "unit-1",
            "clause_id": "clause-1",
            "source_text": source,
            "operation": "pending_answer",
            "owner_routes": [{"owner": "coordinator", "group": "opening"}],
        }]
        votes = iter(("answers", "different_request", "answers"))

        def response(*_args, **kwargs):
            verdict = next(votes)
            return json.dumps({
                "claim_hash": kwargs["request_payload"]["claim_hash"],
                "verdict": verdict,
                "selected_value": True if verdict == "answers" else None,
                "evidence_quote": "clear the workflow",
                "reason": "independent entailment verdict",
            })

        with patch(
            "agent.harness.hierarchical_planner.request_semantic_compilation",
            side_effect=response,
        ):
            errors, sizes = _review_stage_a_pending_entailment(
                object(), payload, partition
            )

        self.assertEqual(errors, ())
        self.assertEqual(len(sizes), 3)

    def test_pending_entailment_jury_is_skipped_for_signed_exact_option(self) -> None:
        from agent.harness.hierarchical_planner import (
            _review_stage_a_pending_entailment,
        )

        errors, sizes = _review_stage_a_pending_entailment(
            object(),
            {
                "pending_question": {"id": "confirm"},
                "contract_proven_pending_prefixes": [{"source_text": "Y"}],
                "pending_typed_candidates": [],
            },
            [{"operation": "pending_answer"}],
        )

        self.assertEqual(errors, ())
        self.assertEqual(sizes, ())

    def test_manual_pending_entailment_requires_semantic_quorum(self) -> None:
        from agent.harness.hierarchical_planner import (
            _review_stage_a_pending_entailment,
            _stage_a_admission_prompt,
        )

        def response(*_args, **kwargs):
            return json.dumps({
                "claim_hash": kwargs["request_payload"]["claim_hash"],
                "verdict": "answers",
                "selected_value": "us-1",
                "evidence_quote": "us-1",
                "reason": "one concrete region value is supplied",
            })

        with patch(
            "agent.harness.hierarchical_planner.request_semantic_compilation",
            side_effect=response,
        ) as compiler:
            errors, sizes = _review_stage_a_pending_entailment(
                object(),
                {
                    "pending_question": {
                        "id": "CLOUD_REGION",
                        "group": "cloud_environment",
                        "manual_input_allowed": True,
                        "validation": {"value_type": "scalar_token"},
                    },
                    "contract_proven_pending_prefixes": [],
                    "pending_typed_candidates": [],
                },
                [{
                    "unit_id": "unit-1",
                    "clause_id": "clause-1",
                    "source_text": "region is us-1",
                    "operation": "pending_answer",
                    "owner_routes": [{
                        "owner": "coordinator",
                        "group": "cloud_environment",
                    }],
                }],
            )

        self.assertEqual(errors, ())
        self.assertEqual(len(sizes), 3)
        self.assertEqual(compiler.call_count, 3)
        self.assertIn(
            "a prose pending_answer is complete only when its exact source "
            "supplies one concrete value",
            _stage_a_admission_prompt(),
        )

    def test_researched_identity_accepts_source_exact_multiword_value(self) -> None:
        from agent.harness.hierarchical_planner import (
            _review_stage_a_pending_entailment,
        )
        from agent.harness.questions import (
            answer_fits_pending,
            researched_identity_value_is_valid,
            value_satisfies_pending_contract,
        )

        source = "AuroraEdge Testnet is the final chain name."
        pending = {
            "id": "chain",
            "group": "chain_identity",
            "kind": "chain",
            "manual_input_allowed": True,
            "options": [],
            "value_domain": "researched_identity",
            "validation": {},
        }
        self.assertFalse(answer_fits_pending(source, pending))
        self.assertTrue(
            value_satisfies_pending_contract("AuroraEdge Testnet", pending)
        )
        self.assertFalse(researched_identity_value_is_valid("AuroraEdge\nTestnet"))
        self.assertFalse(researched_identity_value_is_valid("A" * 81))
        self.assertFalse(researched_identity_value_is_valid("AuroraEdge Testnet."))

        def response(*_args, **kwargs):
            return json.dumps({
                "claim_hash": kwargs["request_payload"]["claim_hash"],
                "verdict": "answers",
                "selected_value": "AuroraEdge Testnet",
                "evidence_quote": "AuroraEdge Testnet",
                "reason": "the exact multi-word identity answers the chain question",
            })

        with patch(
            "agent.harness.hierarchical_planner.request_semantic_compilation",
            side_effect=response,
        ):
            errors, sizes = _review_stage_a_pending_entailment(
                object(),
                {
                    "pending_question": pending,
                    "contract_proven_pending_prefixes": [],
                    "pending_typed_candidates": [],
                },
                [{
                    "unit_id": "unit-1",
                    "clause_id": "clause-1",
                    "source_text": source,
                    "operation": "pending_answer",
                    "owner_routes": [{
                        "owner": "coordinator",
                        "group": "chain_identity",
                    }],
                }],
            )

        self.assertEqual(errors, ())
        self.assertEqual(len(sizes), 3)

    def test_pending_entailment_quorum_precedes_generic_stage_a_review(self) -> None:
        from agent.harness.hierarchical_planner import _review_stage_a_candidate

        payload = {
            "pending_question": {
                "id": "chain",
                "group": "chain_identity",
                "manual_input_allowed": True,
                "value_domain": "researched_identity",
            },
        }
        partition = [{
            "unit_id": "unit-1",
            "clause_id": "clause-1",
            "source_text": "Use AuroraEdge Testnet as the final chain name.",
            "operation": "pending_answer",
            "owner_routes": [{
                "owner": "coordinator",
                "group": "chain_identity",
            }],
        }]

        def coverage_review(_provider, reviewed_payload, _partition, **_kwargs):
            self.assertEqual(
                reviewed_payload[
                    "contract_proven_pending_entailment_unit_ids"
                ],
                ["unit-1"],
            )
            return (), (400,), frozenset(), True

        with patch(
            "agent.harness.hierarchical_planner."
            "_review_stage_a_pending_entailment_detailed",
            return_value=((), (100, 200, 300), frozenset({"unit-1"})),
        ), patch(
            "agent.harness.hierarchical_planner._review_stage_a_partition",
            side_effect=coverage_review,
        ):
            errors, sizes, redundant, contract_valid, rejected = (
                _review_stage_a_candidate(object(), payload, partition)
            )

        self.assertEqual(errors, ())
        self.assertEqual(sizes, (100, 200, 300, 400))
        self.assertEqual(redundant, frozenset())
        self.assertTrue(contract_valid)
        self.assertEqual(rejected, ())

    def test_proven_pending_unit_cannot_mask_unresolved_sibling(self) -> None:
        from agent.harness.hierarchical_planner import (
            _validate_stage_a_admission_document,
        )

        clause = {
            "clause_id": "clause-1",
            "text": "Use AuroraEdge Testnet and change QPS.",
            "input_shape": "prose",
        }
        payload = {
            "clauses": [clause],
            "groups": [
                {"owner": "chain_rpc", "name": "chain_identity"},
                {"owner": "performance", "name": "qps_profile"},
            ],
            "contract_proven_pending_entailment_unit_ids": ["unit-1"],
        }
        partition = [
            {
                "unit_id": "unit-1",
                "clause_id": "clause-1",
                "operation": "pending_answer",
            },
            {
                "unit_id": "unit-2",
                "clause_id": "clause-1",
                "operation": "domain_request",
            },
        ]
        response = json.dumps({
            "unit_verdicts": [
                {
                    "unit_id": "unit-1",
                    "verdict": "unresolved",
                    "supports_unit_id": "",
                    "reason": "generic reviewer rejudged the proven answer",
                },
                {
                    "unit_id": "unit-2",
                    "verdict": "unresolved",
                    "supports_unit_id": "",
                    "reason": "the sibling remains unresolved",
                },
            ],
            "clause_verdicts": [{
                "clause_id": "clause-1",
                "verdict": "unresolved",
                "omitted_owner_routes": [],
                "reason": "one sibling remains unresolved",
            }],
            "reason": "reviewed",
        })

        contract_errors, semantic_errors, _redundant = (
            _validate_stage_a_admission_document(
                response,
                payload,
                partition,
            )
        )

        self.assertEqual(contract_errors, ())
        self.assertTrue(
            any("unit-2" in error for error in semantic_errors),
            semantic_errors,
        )
        self.assertTrue(
            any("clause-1" in error for error in semantic_errors),
            semantic_errors,
        )

    def test_generic_stage_a_cannot_veto_proven_pending_entailment(self) -> None:
        from agent.harness.hierarchical_planner import (
            _validate_stage_a_admission_document,
        )

        payload = {
            "clauses": [{
                "clause_id": "clause-1",
                "text": "Use AuroraEdge Testnet as the final chain name.",
                "input_shape": "prose",
            }],
            "groups": [
                {"owner": "chain_rpc", "name": "chain_identity"},
            ],
            "contract_proven_pending_entailment_unit_ids": ["unit-1"],
        }
        partition = [{
            "unit_id": "unit-1",
            "clause_id": "clause-1",
            "operation": "pending_answer",
        }]
        response = json.dumps({
            "unit_verdicts": [{
                "unit_id": "unit-1",
                "verdict": "unresolved",
                "supports_unit_id": "",
                "reason": "generic reviewer rejudged the proven answer",
            }],
            "clause_verdicts": [{
                "clause_id": "clause-1",
                "verdict": "unresolved",
                "omitted_owner_routes": [],
                "reason": "generic reviewer rejudged the proven answer",
            }],
            "reason": "reviewed",
        })

        contract_errors, semantic_errors, _redundant = (
            _validate_stage_a_admission_document(
                response,
                payload,
                partition,
            )
        )

        self.assertEqual(contract_errors, ())
        self.assertEqual(semantic_errors, ())

    def test_researched_identity_rejects_value_absent_from_source(self) -> None:
        from agent.harness.hierarchical_planner import (
            _review_stage_a_pending_entailment,
        )

        source = "AuroraEdge Testnet is the final chain name."
        pending = {
            "id": "chain",
            "group": "chain_identity",
            "kind": "chain",
            "manual_input_allowed": True,
            "options": [],
            "value_domain": "researched_identity",
            "validation": {},
        }

        def response(*_args, **kwargs):
            return json.dumps({
                "claim_hash": kwargs["request_payload"]["claim_hash"],
                "verdict": "answers",
                "selected_value": "AuroraEdge Mainnet",
                "evidence_quote": "AuroraEdge Testnet",
                "reason": "the proposed value is not the source value",
            })

        with patch(
            "agent.harness.hierarchical_planner.request_semantic_compilation",
            side_effect=response,
        ):
            errors, sizes = _review_stage_a_pending_entailment(
                object(),
                {
                    "pending_question": pending,
                    "contract_proven_pending_prefixes": [],
                    "pending_typed_candidates": [],
                },
                [{
                    "unit_id": "unit-1",
                    "clause_id": "clause-1",
                    "source_text": source,
                    "operation": "pending_answer",
                    "owner_routes": [{
                        "owner": "coordinator",
                        "group": "chain_identity",
                    }],
                }],
            )

        self.assertEqual(len(sizes), 3)
        self.assertTrue(any("quorum rejected" in error for error in errors))

    def test_manual_pending_entailment_rejects_relative_change_request(self) -> None:
        from agent.harness.hierarchical_planner import (
            _review_stage_a_pending_entailment,
        )

        source = "I need to benchmark a different chain."
        payload = {
            "pending_question": {
                "id": "BLOCKCHAIN_NODE",
                "group": "chain_identity",
                "manual_input_allowed": True,
                "validation": {"value_type": "bounded_text"},
            },
            "contract_proven_pending_prefixes": [],
            "pending_typed_candidates": [],
        }
        partition = [{
            "unit_id": "unit-1",
            "clause_id": "clause-1",
            "source_text": source,
            "operation": "pending_answer",
            "owner_routes": [{
                "owner": "coordinator",
                "group": "chain_identity",
            }],
        }]

        def response(*_args, **kwargs):
            return json.dumps({
                "claim_hash": kwargs["request_payload"]["claim_hash"],
                "verdict": "different_request",
                "selected_value": None,
                "evidence_quote": "different chain",
                "reason": "this requests replacement, not one chain value",
            })

        with patch(
            "agent.harness.hierarchical_planner.request_semantic_compilation",
            side_effect=response,
        ) as compiler:
            errors, sizes = _review_stage_a_pending_entailment(
                object(), payload, partition
            )

        self.assertEqual(compiler.call_count, 3)
        self.assertEqual(len(sizes), 3)
        self.assertTrue(any("quorum rejected" in error for error in errors))

    def test_manual_pending_entailment_preserves_typed_contract_identities(
        self,
    ) -> None:
        from agent.harness.hierarchical_planner import (
            _review_stage_a_pending_entailment,
        )

        cases = (
            (
                "qps",
                "Set it to 25.",
                "25",
                25,
                {"value_type": "positive_integer"},
            ),
            (
                "endpoint",
                "Use https://node.example/rpc for this run.",
                "https://node.example/rpc",
                "https://node.example/rpc",
                {"value_type": "url"},
            ),
            (
                "machine",
                "The machine type is c4-standard-16.",
                "c4-standard-16",
                "c4-standard-16",
                {"value_type": "scalar_token"},
            ),
        )
        for name, source, evidence, selected, validation in cases:
            payload = {
                "pending_question": {
                    "id": name,
                    "group": name,
                    "manual_input_allowed": True,
                    "validation": validation,
                },
                "contract_proven_pending_prefixes": [],
                "pending_typed_candidates": [],
            }
            partition = [{
                "unit_id": "unit-1",
                "clause_id": "clause-1",
                "source_text": source,
                "operation": "pending_answer",
                "owner_routes": [{"owner": "coordinator", "group": name}],
            }]

            def response(*_args, **kwargs):
                return json.dumps({
                    "claim_hash": kwargs["request_payload"]["claim_hash"],
                    "verdict": "answers",
                    "selected_value": selected,
                    "evidence_quote": evidence,
                    "reason": "one concrete contract value is supplied",
                })

            with self.subTest(name=name), patch(
                "agent.harness.hierarchical_planner.request_semantic_compilation",
                side_effect=response,
            ):
                errors, sizes = _review_stage_a_pending_entailment(
                    object(), payload, partition
                )

            self.assertEqual(errors, ())
            self.assertEqual(len(sizes), 3)

    def test_manual_pending_entailment_rejects_value_not_grounded_by_quote(
        self,
    ) -> None:
        from agent.harness.hierarchical_planner import (
            _review_stage_a_pending_entailment,
        )

        source = "Set the maximum QPS to 25."
        payload = {
            "pending_question": {
                "id": "MAX_QPS",
                "group": "qps_profile",
                "manual_input_allowed": True,
                "validation": {"value_type": "positive_integer"},
            },
            "contract_proven_pending_prefixes": [],
            "pending_typed_candidates": [],
        }
        partition = [{
            "unit_id": "unit-1",
            "clause_id": "clause-1",
            "source_text": source,
            "operation": "pending_answer",
            "owner_routes": [{
                "owner": "coordinator",
                "group": "qps_profile",
            }],
        }]

        def response(*_args, **kwargs):
            return json.dumps({
                "claim_hash": kwargs["request_payload"]["claim_hash"],
                "verdict": "answers",
                "selected_value": 50,
                "evidence_quote": "25",
                "reason": "invented a different numeric value",
            })

        with patch(
            "agent.harness.hierarchical_planner.request_semantic_compilation",
            side_effect=response,
        ):
            errors, sizes = _review_stage_a_pending_entailment(
                object(), payload, partition
            )

        self.assertEqual(len(sizes), 3)
        self.assertTrue(any("quorum rejected" in error for error in errors))

    def test_pending_entailment_jury_remains_for_semantic_options(self) -> None:
        from agent.harness.hierarchical_planner import (
            _review_stage_a_pending_entailment,
        )

        source = "Go ahead and clear the workflow."
        payload = {
            "pending_question": {
                "id": "reset_confirm",
                "group": "opening",
                "options": [
                    {"id": "yes", "value": True},
                    {"id": "no", "value": False},
                ],
            },
            "contract_proven_pending_prefixes": [],
            "pending_typed_candidates": [],
        }
        partition = [{
            "unit_id": "unit-1",
            "clause_id": "clause-1",
            "source_text": source,
            "operation": "pending_answer",
            "owner_routes": [{"owner": "coordinator", "group": "opening"}],
        }]

        def response(*_args, **kwargs):
            return json.dumps({
                "claim_hash": kwargs["request_payload"]["claim_hash"],
                "verdict": "answers",
                "selected_value": True,
                "evidence_quote": "clear the workflow",
                "reason": "the option effect is explicit",
            })

        with patch(
            "agent.harness.hierarchical_planner.request_semantic_compilation",
            side_effect=response,
        ) as compiler:
            errors, sizes = _review_stage_a_pending_entailment(
                object(), payload, partition
            )

        self.assertEqual(errors, ())
        self.assertEqual(compiler.call_count, 3)
        self.assertEqual(len(sizes), 3)

    def test_pending_entailment_jury_admits_paraphrase_for_one_signed_value(self) -> None:
        from agent.harness.hierarchical_planner import (
            _review_stage_a_pending_entailment,
        )

        source = "不再添加 method"
        payload = {
            "pending_question": {
                "id": "new_chain_method_continue",
                "group": "chain_rpc_configuration",
                "options": [
                    {"id": "add", "value": "add_another"},
                    {"id": "finish", "value": "finish"},
                ],
            },
            "contract_proven_pending_prefixes": [],
            "pending_typed_candidates": [],
        }
        partition = [{
            "unit_id": "unit-1",
            "clause_id": "clause-1",
            "source_text": source,
            "operation": "pending_answer",
            "owner_routes": [{
                "owner": "coordinator",
                "group": "chain_rpc_configuration",
            }],
        }]

        def response(*_args, **kwargs):
            return json.dumps({
                "claim_hash": kwargs["request_payload"]["claim_hash"],
                "verdict": "answers",
                "selected_value": "finish",
                "evidence_quote": source,
                "reason": "the user declines another method",
            })

        with patch(
            "agent.harness.hierarchical_planner.request_semantic_compilation",
            side_effect=response,
        ):
            errors, sizes = _review_stage_a_pending_entailment(
                object(), payload, partition
            )

        self.assertEqual(errors, ())
        self.assertEqual(len(sizes), 3)

    def test_pending_entailment_jury_rejects_conflicting_or_invented_values(self) -> None:
        from agent.harness.hierarchical_planner import (
            _review_stage_a_pending_entailment,
        )

        source = "Proceed with one of those choices."
        payload = {
            "pending_question": {
                "id": "choice",
                "group": "opening",
                "options": [
                    {"id": "yes", "value": True},
                    {"id": "one", "value": 1},
                ],
            },
            "contract_proven_pending_prefixes": [],
            "pending_typed_candidates": [],
        }
        partition = [{
            "unit_id": "unit-1",
            "clause_id": "clause-1",
            "source_text": source,
            "operation": "pending_answer",
            "owner_routes": [{"owner": "coordinator", "group": "opening"}],
        }]

        for name, selected_values in (
            ("conflicting", (True, 1, None)),
            ("invented", ("yes", "yes", "yes")),
        ):
            votes = iter(selected_values)

            def response(*_args, **kwargs):
                selected = next(votes)
                return json.dumps({
                    "claim_hash": kwargs["request_payload"]["claim_hash"],
                    "verdict": "uncertain" if selected is None else "answers",
                    "selected_value": selected,
                    "evidence_quote": source,
                    "reason": "independent entailment verdict",
                })

            with self.subTest(name=name), patch(
                "agent.harness.hierarchical_planner.request_semantic_compilation",
                side_effect=response,
            ):
                errors, sizes = _review_stage_a_pending_entailment(
                    object(), payload, partition
                )

            self.assertTrue(any("quorum rejected" in error for error in errors))
            self.assertEqual(len(sizes), 3)

    def test_pending_option_identity_preserves_json_wire_type(self) -> None:
        from agent.harness.questions import pending_option_value_exists

        pending = {"options": [{"id": "one", "value": 1}]}

        self.assertTrue(pending_option_value_exists(1, pending))
        self.assertFalse(pending_option_value_exists(True, pending))

    def test_explicit_scope_jury_rejects_business_retry_as_session_reset(self) -> None:
        from agent.harness.hierarchical_planner import (
            _review_stage_a_explicit_scope_authorization,
        )

        source = "I need to benchmark a different chain."
        partition = [{
            "unit_id": "unit-1",
            "clause_id": "clause-1",
            "source_text": source,
            "operation": "administrative",
            "owner_routes": [{"owner": "orientation", "group": "opening"}],
        }]

        def response(*_args, **kwargs):
            return json.dumps({
                "claim_hash": kwargs["request_payload"]["claim_hash"],
                "verdict": "not_authorized",
                "evidence_quote": "different chain",
                "reason": "one business target is changing, not the session",
            })

        with patch(
            "agent.harness.hierarchical_planner.request_semantic_compilation",
            side_effect=response,
        ) as compiler:
            errors, sizes, rejected = (
                _review_stage_a_explicit_scope_authorization(
                    object(), partition
                )
            )

        self.assertEqual(compiler.call_count, 3)
        self.assertEqual(len(sizes), 3)
        self.assertTrue(any("quorum rejected" in error for error in errors))
        self.assertEqual(rejected[0]["operation"], "administrative")
        self.assertNotIn("protected_action_purposes", rejected[0])

    def test_explicit_scope_jury_admits_explicit_complete_session_reset(self) -> None:
        from agent.harness.hierarchical_planner import (
            _review_stage_a_explicit_scope_authorization,
        )

        source = "Clear the complete workflow session and start over."
        partition = [{
            "unit_id": "unit-1",
            "clause_id": "clause-1",
            "source_text": source,
            "operation": "administrative",
            "owner_routes": [{"owner": "orientation", "group": "opening"}],
        }]
        votes = iter(("authorized", "not_authorized", "authorized"))

        def response(*_args, **kwargs):
            return json.dumps({
                "claim_hash": kwargs["request_payload"]["claim_hash"],
                "verdict": next(votes),
                "evidence_quote": "complete workflow session",
                "reason": "independent complete-scope verdict",
            })

        with patch(
            "agent.harness.hierarchical_planner.request_semantic_compilation",
            side_effect=response,
        ):
            errors, sizes, rejected = (
                _review_stage_a_explicit_scope_authorization(
                    object(), partition
                )
            )

        self.assertEqual(errors, ())
        self.assertEqual(len(sizes), 3)
        self.assertEqual(rejected, ())

    def test_stage_a_replans_after_protected_scope_rejection(self) -> None:
        from agent.harness.hierarchical_planner import begin_semantic_partition
        from agent.harness.state import new_state

        state = new_state("scope-replan", language="en")
        state["pending_question"] = {
            "id": "reset_confirm",
            "group": "opening",
            "kind": "yes_no",
            "manual_input_allowed": False,
            "options": [
                {"id": "yes", "value": True},
                {"id": "no", "value": False},
            ],
        }
        source = "I need to benchmark a different chain."
        primary = [{
            "unit_id": "unit-primary",
            "clause_id": "clause-1",
            "source_text": source,
            "operation": "pending_answer",
            "owner_routes": [{"owner": "coordinator", "group": "opening"}],
        }]
        destructive = [{
            "unit_id": "unit-destructive",
            "clause_id": "clause-1",
            "source_text": source,
            "operation": "administrative",
            "owner_routes": [{"owner": "orientation", "group": "opening"}],
        }]
        replacement = [{
            "unit_id": "unit-replacement",
            "clause_id": "clause-1",
            "source_text": source,
            "operation": "domain_request",
            "owner_routes": [{
                "owner": "chain_rpc",
                "group": "chain_identity",
            }],
        }]
        rejected_claim = ({
            "unit_id": "unit-destructive",
            "clause_id": "clause-1",
            "proposed_source": source,
            "operation": "administrative",
            "owner": "orientation",
            "group": "opening",
        },)

        with (
            patch(
                "agent.harness.hierarchical_planner.provider_from_config",
                return_value=object(),
            ),
            patch(
                "agent.harness.hierarchical_planner._request_stage_a_proposal",
                side_effect=[
                    (primary, (), (101,)),
                    (destructive, (), (102,)),
                    (replacement, (), (103,)),
                ],
            ) as proposals,
            patch(
                "agent.harness.hierarchical_planner._review_stage_a_partition",
                side_effect=[
                    ((), (201,), frozenset(), True),
                    ((), (202,), frozenset(), True),
                    ((), (203,), frozenset(), True),
                ],
            ),
            patch(
                "agent.harness.hierarchical_planner."
                "_review_stage_a_pending_entailment",
                side_effect=[
                    (("pending claim rejected",), (301, 302, 303)),
                    ((), ()),
                    ((), ()),
                ],
            ),
            patch(
                "agent.harness.hierarchical_planner."
                "_review_stage_a_explicit_scope_authorization",
                side_effect=[
                    (("scope rejected",), (401, 402, 403), rejected_claim),
                    ((), (), ()),
                ],
            ),
        ):
            document = begin_semantic_partition(state, source)

        self.assertEqual(document["status"], "compile_owner")
        self.assertEqual(document["source_partition"], replacement)
        self.assertEqual(proposals.call_count, 3)
        self.assertEqual(
            proposals.call_args_list[2].kwargs["rejected_semantic_claims"],
            rejected_claim,
        )
        self.assertEqual(document["owner_requests"], [{
            "owner": "chain_rpc",
            "unit_ids": ["unit-replacement"],
            "groups": ["chain_identity"],
        }])

    def test_stage_a_selects_counterfactual_request_after_pending_claim_rejected(
        self,
    ) -> None:
        from agent.harness.hierarchical_planner import begin_semantic_partition
        from agent.harness.state import new_state

        state = new_state("pending-counterfactual", language="en")
        state["pending_question"] = {
            "id": "reset_confirm",
            "group": "opening",
            "kind": "yes_no",
            "manual_input_allowed": False,
            "options": [
                {"id": "yes", "value": True},
                {"id": "no", "value": False},
            ],
        }
        source = "I need to benchmark a different chain."
        primary = [{
            "unit_id": "unit-primary",
            "clause_id": "clause-1",
            "source_text": source,
            "operation": "pending_answer",
            "owner_routes": [{"owner": "coordinator", "group": "opening"}],
            "reason": "incorrect pending interpretation",
        }]
        secondary = [{
            "unit_id": "unit-secondary",
            "clause_id": "clause-1",
            "source_text": source,
            "operation": "domain_request",
            "owner_routes": [{
                "owner": "chain_rpc",
                "group": "chain_identity",
            }],
            "reason": "standalone chain replacement request",
        }]

        with (
            patch(
                "agent.harness.hierarchical_planner.provider_from_config",
                return_value=object(),
            ),
            patch(
                "agent.harness.hierarchical_planner._request_stage_a_proposal",
                side_effect=[
                    (primary, (), (101,)),
                    (secondary, (), (102,)),
                ],
            ),
            patch(
                "agent.harness.hierarchical_planner._review_stage_a_partition",
                side_effect=[
                    ((), (201,), frozenset(), True),
                    ((), (202,), frozenset(), True),
                ],
            ),
            patch(
                "agent.harness.hierarchical_planner."
                "_review_stage_a_pending_entailment",
                side_effect=[
                    (("pending claim rejected",), (301, 302, 303)),
                    ((), ()),
                ],
            ),
        ):
            document = begin_semantic_partition(state, source)

        self.assertEqual(document["status"], "compile_owner")
        self.assertEqual(document["source_partition"], secondary)
        self.assertEqual(document["owner_requests"], [{
            "owner": "chain_rpc",
            "unit_ids": ["unit-secondary"],
            "groups": ["chain_identity"],
        }])
        self.assertEqual(
            document["stage_a_convergence"]["selected_proposal"],
            "secondary",
        )
        self.assertEqual(
            document["stage_a_convergence"]["selection_authority"],
            "harness_eligibility",
        )

    def test_unresolved_primary_with_malformed_review_uses_independent_proposal(
        self,
    ) -> None:
        from agent.harness.hierarchical_planner import begin_semantic_partition
        from agent.harness.state import new_state

        state = new_state("malformed-primary-independent", language="en")
        source = "I need to benchmark a different chain."
        primary = [{
            "unit_id": "unit-primary",
            "clause_id": "clause-1",
            "source_text": source,
            "operation": "unresolved",
            "owner_routes": [],
            "reason": "no safe route selected",
        }]
        secondary = [{
            "unit_id": "unit-secondary",
            "clause_id": "clause-1",
            "source_text": source,
            "operation": "domain_request",
            "owner_routes": [{
                "owner": "chain_rpc",
                "group": "chain_identity",
            }],
            "reason": "standalone chain replacement request",
        }]

        with (
            patch(
                "agent.harness.hierarchical_planner.provider_from_config",
                return_value=object(),
            ),
            patch(
                "agent.harness.hierarchical_planner._request_stage_a_proposal",
                side_effect=[
                    (primary, (), (101,)),
                    (secondary, (), (102,)),
                ],
            ) as proposals,
            patch(
                "agent.harness.hierarchical_planner._review_stage_a_partition",
                side_effect=[
                    (("invalid omitted route",), (201, 202), frozenset(), False),
                    ((), (203,), frozenset(), True),
                ],
            ),
            patch(
                "agent.harness.hierarchical_planner."
                "_review_stage_a_pending_entailment",
                return_value=((), ()),
            ),
            patch(
                "agent.harness.hierarchical_planner."
                "_review_stage_a_explicit_scope_authorization",
                return_value=((), (), ()),
            ),
        ):
            document = begin_semantic_partition(state, source)

        self.assertEqual(document["status"], "compile_owner")
        self.assertEqual(document["source_partition"], secondary)
        self.assertEqual(proposals.call_count, 2)
        self.assertEqual(
            document["stage_a_convergence"]["selected_proposal"],
            "secondary",
        )
        self.assertEqual(
            document["stage_a_convergence"]["selection_authority"],
            "harness_eligibility",
        )

    def test_complete_primary_with_malformed_review_does_not_speculate(self) -> None:
        from agent.harness.hierarchical_planner import begin_semantic_partition
        from agent.harness.state import new_state

        state = new_state("malformed-complete-primary", language="en")
        source = "Explain the available benchmark modes."
        primary = [{
            "unit_id": "unit-primary",
            "clause_id": "clause-1",
            "source_text": source,
            "operation": "consultation",
            "owner_routes": [{"owner": "orientation", "group": "opening"}],
            "reason": "benchmark mode consultation",
        }]

        with (
            patch(
                "agent.harness.hierarchical_planner.provider_from_config",
                return_value=object(),
            ),
            patch(
                "agent.harness.hierarchical_planner._request_stage_a_proposal",
                return_value=(primary, (), (101,)),
            ) as proposals,
            patch(
                "agent.harness.hierarchical_planner._review_stage_a_partition",
                return_value=(
                    ("malformed review contract",),
                    (201, 202),
                    frozenset(),
                    False,
                ),
            ),
        ):
            document = begin_semantic_partition(state, source)

        self.assertEqual(document["status"], "failed")
        self.assertEqual(proposals.call_count, 1)
        self.assertIn("malformed review contract", document["errors"])

    def test_invalid_independent_proposal_still_fails_closed(self) -> None:
        from agent.harness.hierarchical_planner import begin_semantic_partition
        from agent.harness.state import new_state

        state = new_state("invalid-independent", language="en")
        source = "I need to benchmark a different chain."
        primary = [{
            "unit_id": "unit-primary",
            "clause_id": "clause-1",
            "source_text": source,
            "operation": "unresolved",
            "owner_routes": [],
            "reason": "no safe route selected",
        }]
        secondary = [{
            "unit_id": "unit-secondary",
            "clause_id": "clause-1",
            "source_text": source,
            "operation": "domain_request",
            "owner_routes": [{
                "owner": "chain_rpc",
                "group": "chain_identity",
            }],
            "reason": "candidate chain request",
        }]

        with (
            patch(
                "agent.harness.hierarchical_planner.provider_from_config",
                return_value=object(),
            ),
            patch(
                "agent.harness.hierarchical_planner._request_stage_a_proposal",
                side_effect=[
                    (primary, (), (101,)),
                    (secondary, (), (102,)),
                ],
            ),
            patch(
                "agent.harness.hierarchical_planner._review_stage_a_partition",
                side_effect=[
                    (("invalid primary review",), (201, 202), frozenset(), False),
                    (("invalid secondary review",), (203, 204), frozenset(), False),
                ],
            ),
        ):
            document = begin_semantic_partition(state, source)

        self.assertEqual(document["status"], "failed")
        self.assertIn("invalid primary review", document["errors"])
        self.assertIn("invalid secondary review", document["errors"])

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
        self.assertTrue(all(
            call.kwargs["reasoning_mode"] == "disabled"
            for call in compiler.call_args_list
        ))
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

    def test_whole_plan_admission_repairs_malformed_omission_rejection(
        self,
    ) -> None:
        from agent.harness.semantic_compiler import (
            WholePlanAdmission,
            _is_explicit_semantic_rejection,
        )

        malformed = WholePlanAdmission(
            valid=False,
            errors=(
                "whole-plan omitted demand is not registry expressible: unit-2",
                "whole-plan admission found unresolved or omitted demand",
            ),
            response={
                "action_verdicts": [{
                    "action_id": "action-1",
                    "verdict": "admit",
                }],
                "unit_verdicts": [{
                    "unit_id": "unit-2",
                    "verdict": "omitted",
                    "omitted_action_type": "",
                }],
            },
        )
        valid_rejection = WholePlanAdmission(
            valid=False,
            errors=(
                "whole-plan admission found unresolved or omitted demand",
            ),
            response={
                "action_verdicts": [{
                    "action_id": "action-1",
                    "verdict": "admit",
                }],
                "unit_verdicts": [{
                    "unit_id": "unit-2",
                    "verdict": "omitted",
                    "omitted_action_type": "answer_opening_question",
                }],
            },
        )

        self.assertFalse(_is_explicit_semantic_rejection(malformed))
        self.assertTrue(_is_explicit_semantic_rejection(valid_rejection))

    def test_whole_plan_canonicalizes_only_ownerless_context_omission(
        self,
    ) -> None:
        from agent.harness.semantic_compiler import (
            _canonicalize_admission_receipts,
        )

        def canonicalize(record: dict, row: dict) -> dict:
            payload = {
                "action_verdicts": [],
                "unit_verdicts": [dict(row)],
            }
            return _canonicalize_admission_receipts(
                payload,
                action_records={},
                unit_records={"unit-1": record},
            )["unit_verdicts"][0]

        context_record = {
            "unit_id": "unit-1",
            "disposition": "context",
            "owner_action_ids": [],
            "source_text": "permission framing",
        }
        omitted = {
            "unit_id": "unit-1",
            "verdict": "omitted",
            "omitted_action_type": "",
            "evidence_quote": "permission framing",
            "reason": "non-executable framing",
        }
        self.assertEqual(
            canonicalize(context_record, omitted)["verdict"],
            "context",
        )

        owned_context = {
            **context_record,
            "owner_action_ids": ["action-1"],
        }
        self.assertEqual(
            canonicalize(owned_context, omitted)["verdict"],
            "omitted",
        )
        self.assertEqual(
            canonicalize(
                context_record,
                {
                    **omitted,
                    "omitted_action_type": "answer_opening_question",
                },
            )["verdict"],
            "omitted",
        )
        self.assertEqual(
            canonicalize(
                {
                    **context_record,
                    "disposition": "unresolved",
                },
                omitted,
            )["verdict"],
            "omitted",
        )

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

    def test_stage_a_admission_cannot_complete_unresolved_source_unit(
        self,
    ) -> None:
        from agent.harness.hierarchical_planner import (
            _validate_stage_a_admission_document,
        )

        payload = {
            "user_text": "Use a mode and do something else",
            "clauses": [{
                "clause_id": "clause-1",
                "text": "Use a mode and do something else",
            }],
            "groups": [{
                "name": "qps_profile",
                "owner": "performance",
            }],
            "universal_operations": ["unresolved"],
        }
        partition = [{
            "unit_id": "unit-1",
            "clause_id": "clause-1",
            "source_text": "do something else",
            "operation": "unresolved",
            "owner_routes": [],
            "reason": "no safe owner",
        }]
        reviewed = {
            "unit_verdicts": [{
                "unit_id": "unit-1",
                "verdict": "complete",
                "supports_unit_id": "",
                "reason": "incorrectly declares unresolved source complete",
            }],
            "clause_verdicts": [{
                "clause_id": "clause-1",
                "verdict": "complete",
                "omitted_owner_routes": [],
                "reason": "incorrectly declares the clause complete",
            }],
            "reason": "inconsistent",
        }

        contract_errors, semantic_errors, redundant = (
            _validate_stage_a_admission_document(
                json.dumps(reviewed),
                payload,
                partition,
            )
        )

        self.assertTrue(any(
            "unresolved source operation was declared complete" in error
            for error in contract_errors
        ))
        self.assertEqual(semantic_errors, ())
        self.assertEqual(redundant, frozenset())

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

    def test_stage_a_admission_can_retire_explicitly_corrected_mutation(
        self,
    ) -> None:
        from agent.harness.hierarchical_planner import (
            _review_stage_a_partition,
        )

        payload = {
            "user_text": "Use AuroraEdge. Actually, use AuroraEdge Testnet.",
            "clauses": [{
                "clause_id": "clause-1",
                "text": "Use AuroraEdge.",
            }, {
                "clause_id": "clause-2",
                "text": "Actually, use AuroraEdge Testnet.",
            }],
            "groups": [{"name": "chain_identity", "owner": "chain_rpc"}],
            "universal_operations": ["domain_request"],
        }
        partition = [{
            "unit_id": "original-chain",
            "clause_id": "clause-1",
            "source_text": "Use AuroraEdge.",
            "operation": "domain_request",
            "owner_routes": [{
                "owner": "chain_rpc",
                "group": "chain_identity",
            }],
            "reason": "initial chain selection",
        }, {
            "unit_id": "corrected-chain",
            "clause_id": "clause-2",
            "source_text": "Actually, use AuroraEdge Testnet.",
            "operation": "domain_request",
            "owner_routes": [{
                "owner": "chain_rpc",
                "group": "chain_identity",
            }],
            "reason": "explicit chain correction",
        }]
        reviewed = {
            "unit_verdicts": [{
                "unit_id": "original-chain",
                "verdict": "redundant",
                "supports_unit_id": "corrected-chain",
                "reason": "the later source explicitly supersedes this value",
            }, {
                "unit_id": "corrected-chain",
                "verdict": "complete",
                "supports_unit_id": "",
                "reason": "the correction is the final requested value",
            }],
            "clause_verdicts": [{
                "clause_id": "clause-1",
                "verdict": "complete",
                "omitted_owner_routes": [],
                "reason": "the original value is preserved as superseded context",
            }, {
                "clause_id": "clause-2",
                "verdict": "complete",
                "omitted_owner_routes": [],
                "reason": "the correction owns the final mutation",
            }],
            "reason": "the explicit correction produces one executable value",
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
        self.assertEqual(redundant, frozenset({"original-chain"}))

    def test_unresolved_mutation_conflict_preserves_sibling_actions(self) -> None:
        from agent.harness.hierarchical_planner import (
            _project_unresolved_mutation_conflicts,
        )

        candidate = {
            "actions": [{
                "type": "choose_chain",
                "chain_text": "AuroraEdge",
                "source_evidence": "AuroraEdge",
            }, {
                "type": "choose_chain",
                "chain_text": "AuroraEdge Testnet",
                "source_evidence": "AuroraEdge Testnet",
            }, {
                "type": "set_rpc_mode",
                "rpc_mode": "mixed",
                "mutation_explicit": True,
                "source_evidence": "mixed",
            }],
            "semantic_units": [{
                "unit_id": "chain-a",
                "clause_id": "clause-1",
                "start": 0,
                "end": 10,
                "source_text": "AuroraEdge",
                "owner_routes": [{
                    "owner": "chain_rpc",
                    "group": "chain_identity",
                }],
                "disposition": "action",
                "action_indexes": [0],
                "reason": "first candidate",
            }, {
                "unit_id": "chain-b",
                "clause_id": "clause-2",
                "start": 0,
                "end": 18,
                "source_text": "AuroraEdge Testnet",
                "owner_routes": [{
                    "owner": "chain_rpc",
                    "group": "chain_identity",
                }],
                "disposition": "action",
                "action_indexes": [1],
                "reason": "second candidate",
            }, {
                "unit_id": "rpc-mode",
                "clause_id": "clause-3",
                "start": 0,
                "end": 5,
                "source_text": "mixed",
                "owner_routes": [{
                    "owner": "chain_rpc",
                    "group": "workload_rpc",
                }],
                "disposition": "action",
                "action_indexes": [2],
                "reason": "independent sibling mutation",
            }],
            "semantic_support_unit_ids": ["chain-a"],
        }

        projected = _project_unresolved_mutation_conflicts(candidate)

        self.assertEqual(
            [action["type"] for action in projected["actions"]],
            ["set_rpc_mode"],
        )
        self.assertEqual(
            [unit["disposition"] for unit in projected["semantic_units"]],
            ["unresolved", "unresolved", "action"],
        )
        self.assertEqual(
            projected["semantic_units"][2]["action_indexes"],
            [0],
        )
        self.assertEqual(projected["semantic_support_unit_ids"], [])

    def test_pending_normalization_projects_late_mutation_conflict_to_draft_atoms(
        self,
    ) -> None:
        from agent.harness.hierarchical_planner import (
            _prepare_candidate_with_normalized_conflict_projection,
        )
        from agent.harness.plan_coverage import TurnClause
        from agent.harness.queue import mutation_conflict_action_groups
        from agent.harness.state import new_state

        state = new_state("normalized-chain-conflict", language="en")
        state["active_group"] = "chain_identity"
        state["pending_question"] = {
            "id": "chain",
            "group": "chain_identity",
            "kind": "chain",
            "owner": "chain_rpc",
            "manual_input_allowed": True,
            "value_domain": "researched_identity",
            "manual_action": {
                "type": "choose_chain",
                "value_argument": "chain_text",
            },
            "options": [],
        }
        clauses = (
            TurnClause(
                "clause-1",
                "I need a fake-node benchmark for AuroraEdge.",
                "prose",
            ),
            TurnClause(
                "clause-2",
                "Actually, the final chain name is AuroraEdge Testnet.",
                "prose",
            ),
            TurnClause(
                "clause-3",
                "Use a custom mixed RPC workload.",
                "prose",
            ),
        )
        candidate = {
            "actions": [{
                "type": "choose_target_mode",
                "target_mode": "fake-node",
                "target_mode_explicit": True,
                "source_evidence": "fake-node",
            }, {
                "type": "choose_chain",
                "chain_text": "AuroraEdge",
                "source_evidence": "AuroraEdge",
            }, {
                "type": "answer_pending",
                "answer": "AuroraEdge Testnet",
                "source_evidence": "AuroraEdge Testnet",
            }, {
                "type": "set_rpc_mode",
                "rpc_mode": "mixed",
                "mutation_explicit": True,
                "source_evidence": "mixed",
            }, {
                "type": "rpc_catalog_command",
                "catalog_command": "enter",
                "source_evidence": "custom mixed RPC workload",
            }],
            "semantic_units": [{
                "unit_id": "target-mode",
                "clause_id": "clause-1",
                "start": 0,
                "end": 44,
                "source_text": "I need a fake-node benchmark for AuroraEdge.",
                "parent_unit_id": "mode-and-chain",
                "owner_routes": [{
                    "owner": "chain_rpc",
                    "group": "target_mode",
                }],
                "disposition": "action",
                "action_indexes": [0],
                "reason": "target mode",
            }, {
                "unit_id": "initial-chain",
                "clause_id": "clause-1",
                "start": 0,
                "end": 44,
                "source_text": "I need a fake-node benchmark for AuroraEdge.",
                "parent_unit_id": "mode-and-chain",
                "owner_routes": [{
                    "owner": "chain_rpc",
                    "group": "chain_identity",
                }],
                "disposition": "action",
                "action_indexes": [1],
                "reason": "initial chain",
            }, {
                "unit_id": "final-chain",
                "clause_id": "clause-2",
                "start": 0,
                "end": 54,
                "source_text": (
                    "Actually, the final chain name is AuroraEdge Testnet."
                ),
                "owner_routes": [{
                    "owner": "coordinator",
                    "group": "chain_identity",
                }],
                "disposition": "action",
                "action_indexes": [2],
                "reason": "manual pending answer",
            }, {
                "unit_id": "rpc-mode",
                "clause_id": "clause-3",
                "start": 0,
                "end": 32,
                "source_text": "Use a custom mixed RPC workload.",
                "parent_unit_id": "custom-workload",
                "owner_routes": [{
                    "owner": "chain_rpc",
                    "group": "workload_rpc",
                }],
                "disposition": "action",
                "action_indexes": [3],
                "reason": "RPC mode",
            }, {
                "unit_id": "rpc-catalog",
                "clause_id": "clause-3",
                "start": 0,
                "end": 32,
                "source_text": "Use a custom mixed RPC workload.",
                "parent_unit_id": "custom-workload",
                "owner_routes": [{
                    "owner": "chain_rpc",
                    "group": "endpoint_process",
                }],
                "disposition": "action",
                "action_indexes": [4],
                "reason": "custom RPC catalog",
            }],
        }

        projected, _text, validation = (
            _prepare_candidate_with_normalized_conflict_projection(
                candidate,
                state,
                clauses,
                pending_choice_unit_ids=frozenset({"final-chain"}),
                semantic_support_unit_ids=frozenset(),
            )
        )

        self.assertFalse(validation.valid)
        self.assertEqual(validation.errors, ())
        self.assertEqual(
            [action["type"] for action in projected["actions"]],
            ["choose_target_mode", "set_rpc_mode", "rpc_catalog_command"],
        )
        self.assertEqual(
            mutation_conflict_action_groups(tuple(projected["actions"])),
            (),
        )
        self.assertEqual(
            [unit["unit_id"] for unit in validation.unresolved_units],
            ["initial-chain", "final-chain"],
        )
        self.assertEqual(
            len(projected.get("admission_action_ids") or []),
            len(projected["actions"]),
        )
        self.assertNotIn("pending_choice_contracts", projected)

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

        self.assertTrue(
            any("typed-option-only" in error for error in errors),
            errors,
        )
        self.assertEqual(len(sizes), 2)
        self.assertEqual(compiler.call_count, 2)
        self.assertTrue(all(
            call.kwargs["reasoning_mode"] == "disabled"
            for call in compiler.call_args_list
        ))
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

    def test_stage_b_uses_registered_intake_for_incomplete_closed_enum_replacement(
        self,
    ) -> None:
        from agent.harness.hierarchical_planner import _compile_owner_document
        from agent.harness.state import new_state

        state = new_state("stage-b-closed-enum-intake", language="en")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        partition = [{
            "unit_id": "unit-1",
            "clause_id": "clause-1",
            "source_text": "Change the target mode; do not keep alpha.",
            "operation": "domain_request",
            "owner_routes": [{
                "owner": "chain_rpc",
                "group": "target_mode",
            }],
            "reason": "explicit replacement without one selected remaining value",
        }]
        compiled = {
            "actions": [{
                "type": "request_target_mode_selection",
                "source_evidence": "Change the target mode; do not keep alpha.",
            }],
            "bindings": [{
                "unit_id": "unit-1",
                "action_indexes": [0],
                "disposition": "action",
                "reason": "registered typed intake owns the incomplete replacement",
            }],
            "reason": "compiled without guessing a closed-enum value",
        }

        with (
            patch(
                "agent.harness.hierarchical_planner.provider_from_config",
                return_value=object(),
            ),
            patch(
                "agent.harness.hierarchical_planner.request_semantic_compilation",
                return_value=json.dumps(compiled),
            ) as compiler,
        ):
            document, errors, sizes = _compile_owner_document(
                state,
                "chain_rpc",
                frozenset({"target_mode"}),
                partition,
                ("unit-1",),
            )

        self.assertEqual(errors, ())
        self.assertEqual(document["actions"], compiled["actions"])
        self.assertEqual(len(sizes), 1)
        request = compiler.call_args.kwargs
        self.assertIn(
            "incomplete_mutation_intake=true",
            request["system_prompt"],
        )
        schema = request["request_payload"]["owner_action_schema"]
        intake = next(
            row
            for row in schema
            if row["type"] == "request_target_mode_selection"
        )
        self.assertTrue(intake["incomplete_mutation_intake"])
        self.assertEqual(intake["target_group"], "target_mode")

    def test_stage_b_lowers_missing_evidence_to_registered_read_intake(
        self,
    ) -> None:
        from agent.harness.hierarchical_planner import _validate_owner_document

        document = {
            "actions": [],
            "bindings": [{
                "unit_id": "unit-1",
                "action_indexes": [],
                "disposition": "unresolved",
                "reason": "evidence has not been pasted yet",
            }],
            "reason": "wait for evidence",
        }

        payload, errors = _validate_owner_document(
            json.dumps(document, ensure_ascii=False),
            "analysis",
            ("unit-1",),
            expected_groups={"unit-1": frozenset()},
            expected_operations={"unit-1": "evidence_analysis"},
            expected_sources={
                "unit-1": ("这是日志，你可以帮我分析么？",),
            },
        )

        self.assertEqual(errors, ())
        self.assertEqual(
            payload["actions"],
            [{"type": "request_evidence_analysis"}],
        )
        self.assertEqual(
            payload["bindings"][0]["disposition"],
            "action",
        )
        self.assertEqual(payload["bindings"][0]["action_indexes"], [0])

    def test_stage_b_does_not_lower_unregistered_unresolved_read(self) -> None:
        from agent.harness.hierarchical_planner import _validate_owner_document

        document = {
            "actions": [],
            "bindings": [{
                "unit_id": "unit-1",
                "action_indexes": [],
                "disposition": "unresolved",
                "reason": "subject is ambiguous",
            }],
            "reason": "clarification is required",
        }

        payload, errors = _validate_owner_document(
            json.dumps(document, ensure_ascii=False),
            "orientation",
            ("unit-1",),
            expected_groups={"unit-1": frozenset()},
            expected_operations={"unit-1": "consultation"},
            expected_sources={"unit-1": ("解释一下",)},
        )

        self.assertEqual(errors, ())
        self.assertEqual(payload["actions"], [])
        self.assertEqual(
            payload["bindings"][0]["disposition"],
            "unresolved",
        )

    def test_stage_b_lowers_value_less_entry_to_unique_registered_intake(
        self,
    ) -> None:
        from agent.harness.hierarchical_planner import _validate_owner_document

        document = {
            "actions": [{
                "type": "choose_chain",
                "source_evidence": "重新测试别的链",
            }],
            "bindings": [{
                "unit_id": "unit-1",
                "action_indexes": [0],
                "disposition": "action",
                "reason": "chain replacement requested without a concrete value",
            }],
            "reason": "compile the registered chain intake",
        }

        payload, errors = _validate_owner_document(
            json.dumps(document, ensure_ascii=False),
            "chain_rpc",
            ("unit-1",),
            expected_groups={"unit-1": frozenset({"chain_identity"})},
            expected_operations={"unit-1": "domain_request"},
            expected_sources={"unit-1": ("重新测试别的链",)},
        )

        self.assertEqual(errors, ())
        self.assertEqual(payload["actions"], [{
            "type": "request_chain_selection",
            "source_evidence": "重新测试别的链",
        }])

    def test_stage_b_lowers_unresolved_value_less_mutation_to_registered_intake(
        self,
    ) -> None:
        from agent.harness.hierarchical_planner import _validate_owner_document

        source = "我需要重新测试别的链，可以么"
        document = {
            "actions": [],
            "bindings": [{
                "unit_id": "unit-1",
                "action_indexes": [],
                "disposition": "unresolved",
                "reason": "replacement requested without a concrete value",
            }],
            "reason": "the typed intake must collect the missing value",
        }

        payload, errors = _validate_owner_document(
            json.dumps(document, ensure_ascii=False),
            "chain_rpc",
            ("unit-1",),
            expected_groups={"unit-1": frozenset({"chain_identity"})},
            expected_operations={"unit-1": "domain_request"},
            expected_sources={"unit-1": (source,)},
        )

        self.assertEqual(errors, ())
        self.assertEqual(payload["actions"], [{
            "type": "request_chain_selection",
            "source_evidence": source,
        }])
        self.assertEqual(payload["bindings"], [{
            "unit_id": "unit-1",
            "action_indexes": [0],
            "disposition": "action",
            "reason": "replacement requested without a concrete value",
        }])

    def test_stage_b_binds_exact_source_to_registered_mutation_intake(
        self,
    ) -> None:
        from agent.harness.hierarchical_planner import _validate_owner_document

        source = "I need to benchmark a different chain."
        document = {
            "actions": [{
                "type": "request_chain_selection",
            }],
            "bindings": [{
                "unit_id": "unit-1",
                "action_indexes": [0],
                "disposition": "action",
                "reason": "collect the omitted replacement chain",
            }],
            "reason": "use the registry-owned incomplete mutation intake",
        }

        payload, errors = _validate_owner_document(
            json.dumps(document),
            "chain_rpc",
            ("unit-1",),
            expected_groups={"unit-1": frozenset({"chain_identity"})},
            expected_operations={"unit-1": "domain_request"},
            expected_sources={"unit-1": (source,)},
        )

        self.assertEqual(errors, ())
        self.assertEqual(payload["actions"], [{
            "type": "request_chain_selection",
            "source_evidence": source,
        }])

    def test_stage_b_does_not_guess_intake_evidence_from_multiple_units(
        self,
    ) -> None:
        from agent.harness.hierarchical_planner import _validate_owner_document

        document = {
            "actions": [{
                "type": "request_chain_selection",
            }],
            "bindings": [
                {
                    "unit_id": unit_id,
                    "action_indexes": [0],
                    "disposition": "action",
                    "reason": "shared intake",
                }
                for unit_id in ("unit-1", "unit-2")
            ],
            "reason": "ambiguous ownership must fail closed",
        }

        payload, errors = _validate_owner_document(
            json.dumps(document),
            "chain_rpc",
            ("unit-1", "unit-2"),
            expected_groups={
                "unit-1": frozenset({"chain_identity"}),
                "unit-2": frozenset({"chain_identity"}),
            },
            expected_operations={
                "unit-1": "domain_request",
                "unit-2": "domain_request",
            },
            expected_sources={
                "unit-1": ("change the chain",),
                "unit-2": ("choose another network",),
            },
        )

        self.assertEqual(payload["actions"], [])
        self.assertTrue(
            any("missing required arguments" in error for error in errors),
            errors,
        )

    def test_stage_b_does_not_lower_unrouted_or_ambiguous_mutation_intake(
        self,
    ) -> None:
        from agent.harness.hierarchical_planner import _validate_owner_document

        document = {
            "actions": [],
            "bindings": [{
                "unit_id": "unit-1",
                "action_indexes": [],
                "disposition": "unresolved",
                "reason": "the requested domain is not known",
            }],
            "reason": "clarification required",
        }

        payload, errors = _validate_owner_document(
            json.dumps(document),
            "chain_rpc",
            ("unit-1",),
            expected_groups={"unit-1": frozenset()},
            expected_operations={"unit-1": "domain_request"},
            expected_sources={"unit-1": ("change it",)},
        )

        self.assertEqual(errors, ())
        self.assertEqual(payload["actions"], [])
        self.assertEqual(payload["bindings"][0]["disposition"], "unresolved")

    def test_stage_b_keeps_concrete_and_multi_candidate_chain_entries(
        self,
    ) -> None:
        from agent.harness.hierarchical_planner import _validate_owner_document

        for action in (
            {
                "type": "choose_chain",
                "chain_text": "ethereum",
                "source_evidence": "ethereum",
            },
            {
                "type": "choose_chain",
                "chain_candidates": ["ethereum", "bsc"],
                "source_evidence": "ethereum or bsc",
            },
        ):
            with self.subTest(action=action):
                document = {
                    "actions": [action],
                    "bindings": [{
                        "unit_id": "unit-1",
                        "action_indexes": [0],
                        "disposition": "action",
                        "reason": "concrete chain input",
                    }],
                    "reason": "compile concrete chain input",
                }
                payload, errors = _validate_owner_document(
                    json.dumps(document),
                    "chain_rpc",
                    ("unit-1",),
                    expected_groups={
                        "unit-1": frozenset({"chain_identity"}),
                    },
                    expected_operations={"unit-1": "domain_request"},
                    expected_sources={
                        "unit-1": (str(action["source_evidence"]),),
                    },
                )

                self.assertEqual(errors, ())
                self.assertEqual(payload["actions"], [action])

    def test_stage_b_repairs_closed_enum_guess_to_registered_intake(self) -> None:
        from agent.harness.hierarchical_planner import _compile_owner_document
        from agent.harness.state import new_state

        state = new_state("stage-b-closed-enum-repair", language="en")
        state["target_mode"] = "fake-node"
        partition = [{
            "unit_id": "unit-1",
            "clause_id": "clause-1",
            "source_text": "Do not use fake-node.",
            "operation": "domain_request",
            "owner_routes": [{
                "owner": "chain_rpc",
                "group": "target_mode",
            }],
            "reason": "rejects one value without selecting a remaining value",
        }]
        guessed = {
            "actions": [{
                "type": "choose_target_mode",
                "target_mode": "real-node",
                "target_mode_explicit": True,
                "source_evidence": "fake-node",
            }],
            "bindings": [{
                "unit_id": "unit-1",
                "action_indexes": [0],
                "disposition": "action",
                "reason": "guessed a remaining value",
            }],
            "reason": "invalid guess",
        }
        repaired = {
            "actions": [{
                "type": "request_target_mode_selection",
                "source_evidence": "Do not use fake-node.",
            }],
            "bindings": [{
                "unit_id": "unit-1",
                "action_indexes": [0],
                "disposition": "action",
                "reason": "registered intake preserves the unresolved choice",
            }],
            "reason": "repaired without selecting a value",
        }

        with (
            patch(
                "agent.harness.hierarchical_planner.provider_from_config",
                return_value=object(),
            ),
            patch(
                "agent.harness.hierarchical_planner.request_semantic_compilation",
                side_effect=[json.dumps(guessed), json.dumps(repaired)],
            ) as compiler,
        ):
            document, errors, sizes = _compile_owner_document(
                state,
                "chain_rpc",
                frozenset({"target_mode"}),
                partition,
                ("unit-1",),
            )

        self.assertEqual(errors, ())
        self.assertEqual(document["actions"], repaired["actions"])
        self.assertEqual(len(sizes), 2)
        repair_errors = compiler.call_args_list[1].kwargs["request_payload"][
            "contract_repair"
        ]["validator_errors"]
        self.assertTrue(any(
            "closed-enum grounding quote names only competing values" in error
            for error in repair_errors
        ))

    def test_stage_b_exhausted_enum_repair_preserves_valid_sibling(self) -> None:
        from agent.harness.hierarchical_planner import _compile_owner_document
        from agent.harness.state import new_state

        state = new_state("stage-b-compound-enum-fallback", language="zh")
        state["target_mode"] = "fake-node"
        state["chain_identity"] = {
            "raw": "bsc",
            "canonical": "bsc",
            "status": "confirmed",
            "case": "known",
        }
        partition = [
            {
                "unit_id": "unit-1",
                "clause_id": "clause-1",
                "source_text": "我需要换成 eth，",
                "operation": "domain_request",
                "owner_routes": [{
                    "owner": "chain_rpc",
                    "group": "chain_identity",
                }],
                "reason": "explicit chain replacement",
            },
            {
                "unit_id": "unit-2",
                "clause_id": "clause-1",
                "source_text": "不使用 fake-node 模式",
                "operation": "domain_request",
                "owner_routes": [{
                    "owner": "chain_rpc",
                    "group": "target_mode",
                }],
                "reason": "rejects the current mode without selecting another",
            },
        ]
        guessed = {
            "actions": [
                {
                    "type": "choose_chain",
                    "chain_text": "eth",
                    "source_evidence": "eth",
                },
                {
                    "type": "choose_target_mode",
                    "target_mode": "real-node",
                    "target_mode_explicit": True,
                    "source_evidence": "fake-node",
                },
            ],
            "bindings": [
                {
                    "unit_id": "unit-1",
                    "action_indexes": [0],
                    "disposition": "action",
                    "reason": "grounded chain replacement",
                },
                {
                    "unit_id": "unit-2",
                    "action_indexes": [1],
                    "disposition": "action",
                    "reason": "unguarded remaining-mode guess",
                },
            ],
            "reason": "one valid sibling and one ungrounded enum guess",
        }

        with (
            patch(
                "agent.harness.hierarchical_planner.provider_from_config",
                return_value=object(),
            ),
            patch(
                "agent.harness.hierarchical_planner.request_semantic_compilation",
                side_effect=[json.dumps(guessed), json.dumps(guessed)],
            ),
        ):
            document, errors, sizes = _compile_owner_document(
                state,
                "chain_rpc",
                frozenset({"chain_identity", "target_mode"}),
                partition,
                ("unit-1", "unit-2"),
            )

        self.assertEqual(errors, ())
        self.assertEqual(len(sizes), 2)
        self.assertEqual(document["actions"][0], guessed["actions"][0])
        self.assertEqual(
            document["actions"][1],
            {
                "type": "request_target_mode_selection",
                "source_evidence": "不使用 fake-node 模式",
            },
        )
        self.assertEqual(
            [binding["action_indexes"] for binding in document["bindings"]],
            [[0], [1]],
        )

    def test_stage_b_enum_fallback_refuses_unrelated_validation_error(self) -> None:
        from agent.harness.hierarchical_planner import (
            _repair_stage_b_closed_enum_intakes,
        )

        document = {
            "actions": [{
                "type": "choose_target_mode",
                "target_mode": "real-node",
                "target_mode_explicit": True,
                "source_evidence": "fake-node",
            }],
            "bindings": [{
                "unit_id": "unit-1",
                "action_indexes": [0],
                "disposition": "action",
                "reason": "unguarded guess",
            }],
        }

        repaired = _repair_stage_b_closed_enum_intakes(
            document,
            (
                "Stage B closed-enum grounding quote names only competing "
                "values: unit-1/choose_target_mode/target_mode",
                "Stage B chain_rpc binding order or cardinality mismatch",
            ),
            owner="chain_rpc",
            expected_sources={"unit-1": ("不使用 fake-node 模式",)},
        )

        self.assertIsNone(repaired)

    def test_stage_b_accepts_minimal_natural_language_enum_grounding(self) -> None:
        from agent.harness.hierarchical_planner import _validate_owner_document

        document = {
            "actions": [{
                "type": "choose_target_mode",
                "target_mode": "real-node",
                "target_mode_explicit": True,
                "source_evidence": "真实节点",
            }],
            "bindings": [{
                "unit_id": "unit-1",
                "action_indexes": [0],
                "disposition": "action",
                "reason": "the affirmative span selects the concrete mode",
            }],
            "reason": "compiled",
        }

        payload, errors = _validate_owner_document(
            json.dumps(document, ensure_ascii=False),
            "chain_rpc",
            ("unit-1",),
            expected_groups={"unit-1": frozenset({"target_mode"})},
            expected_operations={"unit-1": "domain_request"},
            expected_sources={
                "unit-1": "不要 fake-node，改用真实节点",
            },
        )

        self.assertEqual(errors, ())
        self.assertEqual(payload["actions"], document["actions"])

    def test_rejected_closed_enum_selection_repairs_to_registered_intake(
        self,
    ) -> None:
        from types import SimpleNamespace

        from agent.harness.hierarchical_planner import (
            _conservative_closed_enum_intake_repair,
        )

        candidate = {
            "actions": [
                {
                    "type": "choose_chain",
                    "chain_text": "eth",
                    "source_evidence": "eth",
                },
                {
                    "type": "choose_target_mode",
                    "target_mode": "fake-node",
                    "target_mode_explicit": True,
                    "source_evidence": "fake-node",
                },
            ],
            "semantic_units": [
                {
                    "unit_id": "unit-1",
                    "source_text": "我需要换成 eth，",
                    "disposition": "action",
                    "action_indexes": [0],
                },
                {
                    "unit_id": "unit-2",
                    "source_text": "不使用 fake-node 模式",
                    "disposition": "action",
                    "action_indexes": [1],
                },
            ],
            "reason": "compiled",
        }
        plan = SimpleNamespace(action_ids=("chain-action", "mode-action"))
        admission = SimpleNamespace(action_verdicts=(
            {"action_id": "chain-action", "verdict": "admit"},
            {"action_id": "mode-action", "verdict": "reject"},
        ))

        repaired = _conservative_closed_enum_intake_repair(
            candidate,
            plan,
            admission,
        )

        self.assertIsNotNone(repaired)
        self.assertEqual(repaired["actions"][0], candidate["actions"][0])
        self.assertEqual(
            repaired["actions"][1],
            {
                "type": "request_target_mode_selection",
                "source_evidence": "不使用 fake-node 模式",
            },
        )

    def test_rejected_non_enum_action_does_not_rewrite_to_intake(self) -> None:
        from types import SimpleNamespace

        from agent.harness.hierarchical_planner import (
            _conservative_closed_enum_intake_repair,
        )

        candidate = {
            "actions": [{
                "type": "choose_chain",
                "chain_text": "eth",
                "source_evidence": "eth",
            }],
            "semantic_units": [{
                "unit_id": "unit-1",
                "source_text": "change to eth",
                "disposition": "action",
                "action_indexes": [0],
            }],
            "reason": "compiled",
        }
        plan = SimpleNamespace(action_ids=("chain-action",))
        admission = SimpleNamespace(action_verdicts=(
            {"action_id": "chain-action", "verdict": "reject"},
        ))

        self.assertIsNone(
            _conservative_closed_enum_intake_repair(
                candidate,
                plan,
                admission,
            )
        )

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
        from agent.harness.admission import validate_action_plan
        from agent.harness.semantic_admission import (
            _admitted_action_queue,
            _freeze_bounded_semantic_plan,
            prepare_hierarchical_candidate,
        )
        from agent.harness.semantic_compiler import WholePlanAdmission
        from agent.harness.plan_coverage import segment_user_turn
        from agent.harness.state import new_state

        for topic, text in (
            ("capabilities", "Explain the supported capabilities."),
            ("recommendation", "Recommend a safe way to start."),
        ):
            with self.subTest(topic=topic):
                clauses = segment_user_turn(text)
                state = new_state(
                    f"consultation-pending-{topic}",
                    language="en",
                )
                state["pending_question"] = opening_question(state)
                candidate = {
                    "actions": [{
                        "type": "answer_opening_question",
                        "topic": topic,
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
                self.assertEqual(
                    payload["actions"][0]["type"],
                    "answer_opening_question",
                )
                self.assertNotIn("pending_choice_contracts", payload)
                self.assertNotIn("pending_answer_admissions", payload)

                plan = _freeze_bounded_semantic_plan(
                    prepared,
                    state,
                    clauses,
                )
                admitted = _admitted_action_queue(
                    plan,
                    WholePlanAdmission(
                        valid=True,
                        errors=(),
                        action_verdicts=({
                            "action_id": plan.action_ids[0],
                            "verdict": "admit",
                            "unit_ids": ["unit-1"],
                        },),
                    ),
                    state,
                )
                state["turn_context"] = {
                    "semantic_units": admitted["semantic_units"],
                }

                self.assertNotIn(
                    "pending_option_semantic_verified",
                    admitted["actions"][0],
                )
                self.assertEqual(
                    validate_action_plan(
                        state,
                        admitted["actions"],
                    ).status,
                    "accepted",
                )
                if topic == "capabilities":
                    tampered = deepcopy(admitted["actions"])
                    tampered[0]["_plan_transaction_hash"] = "f" * 64
                    result = validate_action_plan(state, tampered)
                    self.assertEqual(result.status, "rejected")
                    self.assertEqual(
                        [row.code for row in result.rejections],
                        ["pending_choice_bypass"],
                    )

    def test_forged_trusted_metadata_cannot_bypass_opening_pending(self) -> None:
        from agent.harness.admission import validate_action_plan
        from agent.harness.domains.orientation import opening_question
        from agent.harness.state import new_state

        state = new_state("forged-opening-metadata", language="en")
        state["pending_question"] = opening_question(state)
        for option in state["pending_question"]["options"]:
            action = {
                **dict(option["action"]),
                "semantic_purpose_verified": True,
                "_admission_action_id": "forged-action",
                "_transaction_action_ids": ["forged-action"],
                "_plan_transaction_hash": "f" * 64,
            }
            with self.subTest(action=action["type"]):
                result = validate_action_plan(state, [action])
                self.assertEqual(result.status, "rejected")
                self.assertEqual(
                    [row.code for row in result.rejections],
                    ["pending_choice_bypass"],
                )

    def test_domain_request_cannot_claim_a_pending_option(self) -> None:
        from agent.harness.domains.orientation import opening_question
        from agent.harness.admission import validate_action_plan
        from agent.harness.semantic_admission import (
            _admitted_action_queue,
            _freeze_bounded_semantic_plan,
            prepare_hierarchical_candidate,
        )
        from agent.harness.semantic_compiler import WholePlanAdmission
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

        plan = _freeze_bounded_semantic_plan(prepared, state, clauses)
        admitted = _admitted_action_queue(
            plan,
            WholePlanAdmission(
                valid=True,
                errors=(),
                action_verdicts=({
                    "action_id": plan.action_ids[0],
                    "verdict": "admit",
                    "unit_ids": ["unit-1"],
                },),
            ),
            state,
        )
        state["turn_context"] = {
            "semantic_units": admitted["semantic_units"],
        }
        result = validate_action_plan(state, admitted["actions"])

        self.assertEqual(result.status, "rejected")
        self.assertEqual(
            [row.code for row in result.rejections],
            ["pending_choice_bypass"],
        )

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
                "agent.harness.hierarchical_planner._review_stage_a_partition",
                return_value=((), (640,), frozenset()),
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
        self.assertEqual(result["planner_metrics"]["admission_calls"], 2)

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
                "agent.harness.hierarchical_planner._review_stage_a_partition",
                return_value=((), (333,), frozenset()),
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
                "agent.harness.hierarchical_planner._review_stage_a_partition",
                return_value=((), (333,), frozenset()),
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

    def test_registry_route_projection_is_identical_across_planner_stages(self) -> None:
        from agent.harness.action_registry import (
            ACTION_SPECS,
            action_route_groups,
            project_action_specs,
        )
        from agent.harness.hierarchical_planner import _action_spec_applies

        for spec in ACTION_SPECS:
            if spec.internal_only:
                continue
            for group in action_route_groups(spec):
                projected = project_action_specs(groups=frozenset({group}))
                self.assertIn(spec, projected, (spec.action_type, group))
                for operation in spec.semantic_operations:
                    self.assertTrue(
                        _action_spec_applies(
                            spec,
                            groups=frozenset({group}),
                            operations=frozenset({operation}),
                        ),
                        (spec.action_type, operation, group),
                    )

    def test_only_parser_bound_manual_pending_value_is_checked_in_stage_a(self) -> None:
        from agent.harness.hierarchical_planner import (
            _pending_answer_contract_errors,
        )

        state = {
            "pending_question": {
                "id": "duration",
                "group": "sync_observe",
                "owner": "sync_observe",
                "manual_input_allowed": True,
                "validation": {"value_type": "positive_integer"},
            }
        }
        prose = [{
            "unit_id": "unit-1",
            "source_text": "Do not continue this observation.",
            "operation": "pending_answer",
        }]
        parser_bound = [{
            "unit_id": "unit-1",
            "source_text": json.dumps({"duration": "invalid"}),
            "source_path": "duration",
            "operation": "pending_answer",
        }]

        self.assertEqual(_pending_answer_contract_errors(prose, state), ())
        self.assertTrue(_pending_answer_contract_errors(parser_bound, state))

    def test_structured_literal_validation_uses_atom_value_not_schema_key(self) -> None:
        from agent.harness.plan_coverage import _structured_atom_source

        source = json.dumps({
            "validation_endpoint": "http://127.0.0.1:8545",
            "rpc_request": {
                "jsonrpc": "2.0",
                "method": "eth_chainId",
                "params": [],
                "id": 1,
            },
        })
        endpoint_atom = _structured_atom_source(
            {"source_path": "validation_endpoint"},
            source,
        )
        request_atom = _structured_atom_source(
            {"source_path": "rpc_request.method"},
            source,
        )

        self.assertNotIn("validation_endpoint", endpoint_atom)
        self.assertIn("127.0.0.1:8545", endpoint_atom)
        self.assertNotIn("rpc_request", request_atom)
        self.assertIn("eth_chainId", request_atom)

    def test_hierarchical_support_receipt_survives_candidate_normalization(self) -> None:
        from agent.harness.semantic_admission import (
            _freeze_bounded_semantic_plan,
            prepare_hierarchical_candidate,
        )
        from agent.harness.plan_coverage import segment_user_turn
        from agent.harness.state import new_state

        source = '{"target_mode":"fake-node"}'
        clauses = segment_user_turn(source)
        support_id = "__harness_structured_clause-1_1"
        candidate = {
            "actions": [],
            "semantic_units": [{
                "unit_id": support_id,
                "clause_id": "clause-1",
                "source_text": source,
                "source_path": "target_mode",
                "disposition": "context",
                "action_indexes": [],
                "reason": "duplicate selection support",
            }],
        }

        prepared, validation = prepare_hierarchical_candidate(
            json.dumps(candidate),
            new_state("trusted-structured-support", language="en"),
            clauses,
            pending_choice_unit_ids=frozenset(),
            semantic_support_unit_ids=frozenset({support_id}),
        )

        self.assertEqual(
            json.loads(prepared)["semantic_support_unit_ids"],
            [support_id],
        )
        self.assertTrue(validation.valid, validation.errors)
        frozen = _freeze_bounded_semantic_plan(
            prepared,
            new_state("trusted-structured-support", language="en"),
            clauses,
            authoritative_context_unit_ids=frozenset({support_id}),
        )
        frozen_unit = frozen.request_payload()["semantic_units"][0]
        self.assertEqual(frozen_unit["required_unit_verdict"], "context")

        with self.assertRaisesRegex(ValueError, "unknown authoritative"):
            _freeze_bounded_semantic_plan(
                prepared,
                new_state("forged-structured-support", language="en"),
                clauses,
                authoritative_context_unit_ids=frozenset({"forged-unit"}),
            )

    def test_nested_rpc_evidence_preserves_wire_method_literal(self) -> None:
        from agent.harness.plan_coverage import segment_user_turn, validate_plan_coverage

        source = '{"rpc_request":{"jsonrpc":"2.0","method":"eth_chainId","params":[],"id":1}}'
        clauses = segment_user_turn(source)
        action = {
            "type": "rpc_catalog_command",
            "catalog_command": "append_evidence",
            "rpc_schema_evidence": {
                "jsonrpc": "2.0",
                "method": "eth_chainId",
                "params": [],
                "id": 1,
            },
        }
        result = validate_plan_coverage(
            {
                "actions": [action],
                "semantic_units": [{
                    "unit_id": "rpc-request",
                    "clause_id": "clause-1",
                    "source_text": source,
                    "source_path": "rpc_request.method",
                    "disposition": "action",
                    "action_indexes": [0],
                    "reason": "complete request evidence",
                }],
            },
            clauses,
        )

        self.assertTrue(result.valid, result.errors)

    def test_turn_bound_secret_url_remains_a_typed_pending_candidate(self) -> None:
        from agent.harness.hierarchical_planner import (
            _stage_b_payload,
            _turn_bound_pending_typed_candidates,
        )
        from agent.harness.secret_refs import (
            project_secret_input,
            release_unowned_input_secret_bindings,
        )
        from agent.harness.state import new_state

        endpoint = "https://rpc.example.invalid/private-token"
        reference, bindings = project_secret_input(
            endpoint,
            scope_id="turn-bound-url",
            force_secret=True,
        )
        text = f"Use {reference} instead."
        state = new_state("turn-bound-url", language="en")
        state["active_group"] = "endpoint_process"
        state["pending_question"] = {
            "id": "new_chain_endpoint",
            "group": "endpoint_process",
            "kind": "url",
            "field": "new_chain_endpoint",
            "manual_input_allowed": True,
            "sensitive_input": True,
            "options": [],
            "validation": {"value_type": "url"},
        }
        state["turn_context"] = {
            "text": text,
            "input_secret_bindings": [dict(item) for item in bindings],
        }
        try:
            self.assertEqual(
                _turn_bound_pending_typed_candidates(state, text),
                (reference,),
            )
            payload = _stage_b_payload(
                state,
                "coordinator",
                frozenset({"endpoint_process"}),
                [{
                    "unit_id": "endpoint-unit",
                    "clause_id": "clause-1",
                    "source_text": text,
                    "operation": "pending_answer",
                    "owner_routes": [{
                        "owner": "coordinator",
                        "group": "endpoint_process",
                    }],
                    "reason": "typed pending candidate",
                }],
                ("endpoint-unit",),
            )
            semantic = payload["semantic_units"][0]
            self.assertEqual(semantic["semantic_source"], reference)
            self.assertEqual(semantic["semantic_value"], reference)
            self.assertNotIn(endpoint, json.dumps(payload))
        finally:
            release_unowned_input_secret_bindings(bindings, {})

    def test_unresolved_queue_collapses_route_copies_by_parent_identity(self) -> None:
        from agent.harness.plan_coverage import TurnClause
        from agent.harness.semantic_admission import _unresolved_action_queue

        source = "Configure the chain and workload."
        queue = _unresolved_action_queue(
            (TurnClause("clause-1", source, "prose"),),
            ("whole-plan review rejected the candidate",),
            semantic_units=[
                {
                    "unit_id": "__harness_route_1",
                    "parent_unit_id": "chain-and-workload",
                    "clause_id": "clause-1",
                    "source_text": source,
                    "disposition": "action",
                    "action_indexes": [0],
                },
                {
                    "unit_id": "__harness_route_2",
                    "parent_unit_id": "chain-and-workload",
                    "clause_id": "clause-1",
                    "source_text": source,
                    "disposition": "action",
                    "action_indexes": [1],
                },
                {
                    "unit_id": "genuinely-repeated-demand",
                    "clause_id": "clause-2",
                    "source_text": source,
                    "disposition": "action",
                    "action_indexes": [2],
                },
            ],
        )

        self.assertEqual(
            queue["unresolved_clauses"],
            [source, source],
        )
        self.assertEqual(
            [unit["unit_id"] for unit in queue["semantic_units"]],
            ["chain-and-workload", "genuinely-repeated-demand"],
        )

    def test_semantic_rejection_draft_preserves_safe_siblings(self) -> None:
        from types import SimpleNamespace

        from agent.harness.hierarchical_planner import (
            _semantic_rejection_draft_projection,
        )
        from agent.harness.semantic_compiler import WholePlanAdmission

        source_partition = [{
            "unit_id": "__harness_route_1",
            "parent_unit_id": "chain-change",
            "clause_id": "clause-1",
            "source_text": "Use AuroraEdge Testnet.",
            "operation": "domain_request",
            "owner_routes": [{
                "owner": "chain_rpc",
                "group": "chain_identity",
            }],
            "reason": "chain route",
        }, {
            "unit_id": "__harness_route_2",
            "parent_unit_id": "chain-change",
            "clause_id": "clause-1",
            "source_text": "Use AuroraEdge Testnet.",
            "operation": "domain_request",
            "owner_routes": [{
                "owner": "coordinator",
                "group": "target_mode",
            }],
            "reason": "mode route",
        }, {
            "unit_id": "mixed-workload",
            "clause_id": "clause-2",
            "source_text": "Use a custom mixed workload.",
            "operation": "domain_request",
            "owner_routes": [{
                "owner": "chain_rpc",
                "group": "workload_rpc",
            }],
            "reason": "independent sibling",
        }]
        candidate = {
            "actions": [{
                "type": "choose_chain",
                "chain_text": "AuroraEdge Testnet",
            }, {
                "type": "set_target_mode",
                "target_mode": "fake-node",
            }, {
                "type": "set_rpc_mode",
                "rpc_mode": "mixed",
            }],
            "semantic_units": [{
                **source_partition[0],
                "disposition": "action",
                "action_indexes": [0],
            }, {
                **source_partition[1],
                "disposition": "action",
                "action_indexes": [1],
            }, {
                **source_partition[2],
                "disposition": "action",
                "action_indexes": [2],
            }],
        }
        plan = SimpleNamespace(
            action_ids=("chain-action", "mode-action", "workload-action"),
        )
        admission = WholePlanAdmission(
            valid=False,
            errors=(
                "whole-plan admission rejected one or more immutable actions",
                "whole-plan admission found unresolved or omitted demand",
            ),
            action_verdicts=(
                {"action_id": "chain-action", "verdict": "reject"},
                {"action_id": "mode-action", "verdict": "admit"},
                {"action_id": "workload-action", "verdict": "admit"},
            ),
            unit_verdicts=(
                {"unit_id": "__harness_route_1", "verdict": "unresolved"},
                {"unit_id": "__harness_route_2", "verdict": "complete"},
                {"unit_id": "mixed-workload", "verdict": "complete"},
            ),
            response={
                "action_verdicts": [],
                "unit_verdicts": [],
            },
        )

        partition, units, actions, validation = (
            _semantic_rejection_draft_projection(
                source_partition,
                candidate,
                plan,
                admission,
            )
        )

        self.assertEqual(
            [unit["unit_id"] for unit in partition],
            ["chain-change", "mixed-workload"],
        )
        self.assertEqual(
            [action["type"] for action in actions],
            ["set_rpc_mode"],
        )
        self.assertEqual(
            [unit["unit_id"] for unit in validation.unresolved_units],
            ["chain-change"],
        )
        self.assertEqual(units[0]["disposition"], "unresolved")
        self.assertEqual(units[1]["action_indexes"], [0])

    def test_clarification_never_renders_internal_secret_reference(self) -> None:
        from agent.harness.contracts import ActionProposal
        from agent.harness.domains.orientation import apply_orientation_action
        from agent.harness.response_catalog import render_fragment
        from agent.harness.state import new_state

        reference = "semantic-secret:opaque-reference"
        result = apply_orientation_action(
            new_state("safe-clarification", language="en"),
            ActionProposal(
                "clarify-sensitive",
                "clarify_unresolved",
                {"clauses": [f"Use {reference} instead."]},
                "high",
            ),
        )

        rendered = render_fragment(result.response_fragments[0], "en").text
        self.assertNotIn(reference, rendered)
        self.assertIn("sensitive value", rendered)


if __name__ == "__main__":
    unittest.main()

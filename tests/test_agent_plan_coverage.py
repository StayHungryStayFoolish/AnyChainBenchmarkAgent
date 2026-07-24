"""Transaction-admission tests for complete user-turn clause coverage."""

from __future__ import annotations

import json
import unittest
import sys

from agent.harness.plan_coverage import segment_user_turn, validate_plan_coverage


def _unit(clause, index: int, action_indexes, *, disposition: str = "action", reason: str = "mapped"):
    return {
        "unit_id": f"unit-{index}",
        "clause_id": clause.clause_id,
        "start": 0,
        "end": len(clause.text),
        "source_text": clause.text,
        "disposition": disposition,
        "action_indexes": list(action_indexes),
        "reason": reason,
    }


class PlanCoverageTest(unittest.TestCase):
    def test_untrusted_duplicate_source_spans_are_rejected(self) -> None:
        clauses = segment_user_turn("help")
        payload = {
            "actions": [
                {"type": "greeting", "source_evidence": "help"},
                {"type": "ask_capabilities"},
            ],
            "semantic_units": [
                _unit(clauses[0], 1, [0]),
                _unit(clauses[0], 2, [1]),
            ],
        }

        result = validate_plan_coverage(payload, clauses)

        self.assertFalse(result.valid)
        self.assertTrue(any("overlap" in error for error in result.errors))

    def test_typed_group_entry_excludes_competing_generic_navigation(self) -> None:
        from agent.harness.plan_coverage import TurnClause, validate_plan_coverage

        text = "Add and validate my own RPC method first."
        payload = {
            "actions": [
                {
                    "type": "change_group",
                    "group": "endpoint_process",
                    "navigation_explicit": True,
                    "source_evidence": text,
                },
                {
                    "type": "rpc_catalog_command",
                    "catalog_command": "enter",
                    "source_evidence": text,
                },
            ],
            "semantic_units": [{
                "unit_id": "unit-1",
                "clause_id": "clause-1",
                "source_text": text,
                "disposition": "action",
                "action_indexes": [0, 1],
                "reason": "duplicate entry owners",
            }],
        }

        result = validate_plan_coverage(payload, (TurnClause("clause-1", text),))

        self.assertFalse(result.valid)
        self.assertTrue(any(
            error.startswith("generic navigation competes with registered typed entry: endpoint_process")
            for error in result.errors
        ))



    def test_single_structured_assignment_is_not_downgraded_to_prose(self) -> None:
        clauses = segment_user_turn("CLOUD_ZONE: us-west-2b")

        self.assertEqual(len(clauses), 1)
        self.assertEqual(clauses[0].input_shape, "structured")
        self.assertEqual(clauses[0].text, "CLOUD_ZONE: us-west-2b")

    def test_single_error_label_remains_structured_syntax_without_assigning_intent(self) -> None:
        clauses = segment_user_turn("RuntimeError: connection refused")

        self.assertEqual(len(clauses), 1)
        self.assertEqual(clauses[0].input_shape, "structured")

    def test_labeled_nested_structured_block_preserves_source_indentation(self) -> None:
        source = "accounts disk:\n  ACCOUNTS_VOL_TYPE: hyperdisk-balanced"

        clauses = segment_user_turn(source)

        self.assertEqual(len(clauses), 1)
        self.assertEqual(clauses[0].input_shape, "structured")
        self.assertEqual(clauses[0].text, source)

    def test_standalone_url_is_not_misclassified_as_a_yaml_key(self) -> None:
        clauses = segment_user_turn(
            "Use this endpoint only for schema validation:\nhttp://fake-node:19000"
        )

        self.assertEqual([item.input_shape for item in clauses], ["prose", "prose"])
        self.assertEqual(clauses[-1].text, "http://fake-node:19000")





    def test_detected_value_question_declares_manual_entry_transition(self) -> None:
        from agent.harness.domains.environment import question_for_environment
        from agent.harness.state import new_state

        state = new_state("detected-manual-entry-contract", language="en")
        state["discovery"] = {"cloud": {"region": "test-region"}}
        question = question_for_environment(state, "provider_deployment")

        self.assertIsNotNone(question)
        self.assertFalse(question["options"][0]["manual_entry"])
        self.assertTrue(question["options"][1]["manual_entry"])


    def test_canonical_pending_choice_receipt_authorizes_exact_typed_value(self) -> None:
        from agent.harness.admission import _validate_action_plan
        from agent.harness.state import new_state

        state = new_state("verified-option-reference", language="en")
        state["last_user_input"] = "Yes, keep the detected value."
        state["pending_question"] = {
            "id": "detected-size",
            "group": "ledger_disk",
            "field": "DATA_VOL_SIZE",
            "kind": "yes_no",
            "options": [
                {"id": "1", "label": "Y", "value": "100", "action": {"type": "answer_pending"}},
                {"id": "2", "label": "N", "value": "__manual__", "action": {"type": "answer_pending"}},
            ],
            "manual_input_allowed": True,
            "validation": {"value_type": "positive_number"},
        }
        actions = [{
            "type": "answer_pending",
            "answer": "100",
            "selected_value": "100",
            "source_evidence": state["last_user_input"],
            "pending_option_semantic_verified": True,
            "semantic_purpose_verified": True,
            "_admission_action_id": "admitted-choice-1",
        }]
        state["turn_context"] = {"pending_choice_contracts": [{
            "action_index": 0,
            "admission_action_id": "admitted-choice-1",
            "question": {
                "id": "detected-size",
                "group": "ledger_disk",
                "contract_version": 1,
            },
            "option": {"id": "1", "selected_value": "100"},
            "semantic_units": [{
                "unit_id": "unit-1",
                "clause_id": "clause-1",
                "source_text": state["last_user_input"],
            }],
        }]}

        prepared = _validate_action_plan(state, actions)

        self.assertEqual(prepared[0]["selected_value"], "100")
        self.assertEqual(prepared[0]["answer"], "100")
        self.assertIs(prepared[0]["selection_contract_verified"], True)

    def test_pending_choice_without_exact_canonical_receipt_is_rejected(self) -> None:
        from agent.harness.admission import _validate_action_plan
        from agent.harness.state import new_state

        state = new_state("missing-pending-choice-receipt", language="en")
        state["last_user_input"] = "Use the detected value."
        state["pending_question"] = {
            "id": "detected-size",
            "group": "ledger_disk",
            "field": "DATA_VOL_SIZE",
            "kind": "yes_no",
            "options": [{
                "id": "1",
                "label": "Y",
                "value": "100",
                "action": {"type": "answer_pending"},
            }],
            "manual_input_allowed": False,
        }
        action = {
            "type": "answer_pending",
            "answer": "100",
            "selected_value": "100",
            "source_evidence": state["last_user_input"],
            "pending_option_semantic_verified": True,
            "semantic_purpose_verified": True,
            "_admission_action_id": "admitted-choice-1",
        }
        forged = {
            "action_index": 0,
            "admission_action_id": "admitted-choice-1",
            "question": {"id": "different-question", "group": "ledger_disk"},
            "option": {"id": "1", "selected_value": "100"},
            "semantic_units": [{"unit_id": "unit-1", "source_text": state["last_user_input"]}],
        }

        for contracts in ([], [forged]):
            with self.subTest(contracts=contracts):
                state["turn_context"] = {"pending_choice_contracts": contracts}
                self.assertEqual(_validate_action_plan(state, [action]), [])

    def test_rpc_mode_coverage_scenario_has_required_workflow_prerequisites(self) -> None:
        from tests.agent_live.harness_contract_scenarios import question_scenarios

        scenario = next(row for row in question_scenarios("en") if row.scenario_id == "rpc_mode")

        self.assertEqual(scenario.seed_state["target_mode"], "fake-node")
        self.assertEqual(scenario.seed_state["workflow_mode"], "rpc_benchmark")

    def test_sync_observe_endpoint_question_explains_process_and_rpc_roles(self) -> None:
        from agent.harness.domains.chain_rpc import question_for_chain_rpc
        from agent.harness.state import new_state

        for language, expected in (
            ("en", ("resource attribution", "sync height and health")),
            ("zh", ("资源归因", "同步高度和健康状态")),
        ):
            state = new_state(f"sync-endpoint-{language}", language=language)
            state.update({
                "target_mode": "sync-observe",
                "workflow_mode": "sync_observe",
                "sync_observe": {"source": "existing_local_node"},
                "endpoint_evidence": {},
            })

            question = question_for_chain_rpc(state, "endpoint_process")

            self.assertEqual(question["id"], "SYNC_OBSERVE_RPC_URL")
            for fragment in expected:
                self.assertIn(fragment, question["prompt"])
            self.assertTrue(question["completion_effect"])
            self.assertIn(
                "探测" if language == "zh" else "probe",
                question["completion_effect"],
            )

    def test_every_probe_owned_endpoint_question_declares_completion_effect(self) -> None:
        from tests.agent_live.harness_contract_scenarios import question_scenarios

        expected = {
            "LOCAL_RPC_URL",
            "SYNC_OBSERVE_RPC_URL",
            "custom_rpc_endpoint",
            "new_chain_endpoint",
        }
        questions = {
            str(scenario.question.get("id") or ""): scenario.question
            for scenario in question_scenarios("en")
            if str(scenario.question.get("id") or "") in expected
        }

        self.assertEqual(set(questions), expected)
        for question_id, question in questions.items():
            with self.subTest(question_id=question_id):
                self.assertIn("probe", str(question.get("completion_effect") or ""))











    def test_same_group_navigation_cannot_replace_the_active_pending_answer(self) -> None:
        from agent.harness.intent import _validate_action_document
        from agent.harness.state import new_state

        source = "先测币安智能链，Ethereum 后面再说。"
        clause = segment_user_turn(source)[0]
        state = new_state("same-group-navigation")
        state["active_group"] = "chain_identity"
        state["pending_question"] = {
            "id": "chain_ambiguity_confirm",
            "group": "chain_identity",
            "kind": "numbered_choice",
            "options": [
                {"id": "bsc", "value": {"chain_choice": "bsc"}},
                {"id": "ethereum", "value": {"chain_choice": "ethereum"}},
            ],
        }
        payload = {
            "actions": [{
                "type": "change_group",
                "group": "chain_identity",
                "navigation_explicit": True,
                "source_evidence": source,
            }],
            "semantic_units": [_unit(clause, 1, [0])],
        }

        result = _validate_action_document(json.dumps(payload), (clause,), state)

        self.assertFalse(result.valid)
        self.assertEqual(result.rejected_action_indexes, (0,))
        self.assertIn("without answering it", " ".join(result.errors))








    def test_prose_context_can_coexist_with_an_owned_action(self) -> None:
        clauses = segment_user_turn(
            "Use BNB Smart Chain for this run; I may add another RPC call afterward."
        )
        payload = {
            "actions": [{
                "type": "choose_chain",
                "chain_text": "BNB Smart Chain",
                "chain_candidates": ["BNB Smart Chain"],
                "source_evidence": clauses[0].text,
            }],
            "semantic_units": [
                _unit(clauses[0], 1, [0]),
                _unit(
                    clauses[1],
                    2,
                    [],
                    disposition="context",
                    reason="tentative future possibility",
                ),
            ],
        }

        result = validate_plan_coverage(payload, clauses)

        self.assertTrue(result.valid, result.errors)

    def test_omitted_explicit_deferred_goal_fails_plan_completeness(self) -> None:
        clauses = segment_user_turn(
            "Observe BSC sync now. Benchmark its real RPC capacity later."
        )
        payload = {
            "actions": [{
                "type": "choose_target_mode",
                "target_mode": "sync-observe",
                "source_evidence": clauses[0].text,
            }],
            "semantic_units": [_unit(clauses[0], 1, [0])],
        }

        result = validate_plan_coverage(payload, clauses)

        self.assertFalse(result.valid)
        self.assertIn("missing semantic units for clause-2", "\n".join(result.errors))

    def test_context_requires_reason_and_cannot_own_actions(self) -> None:
        clauses = segment_user_turn("I may add another RPC call afterward.")
        payload = {
            "actions": [{"type": "resume_current_flow"}],
            "semantic_units": [
                _unit(clauses[0], 1, [0], disposition="context", reason=""),
            ],
        }

        result = validate_plan_coverage(payload, clauses)

        self.assertFalse(result.valid)
        self.assertIn("context semantic unit has no reason", "\n".join(result.errors))
        self.assertIn("context semantic unit has action indexes", "\n".join(result.errors))

    def test_structured_clause_cannot_be_context(self) -> None:
        clauses = segment_user_turn('{"CLOUD_REGION":"asia-east1"}')
        payload = {
            "actions": [],
            "semantic_units": [
                _unit(clauses[0], 1, [], disposition="context", reason="background"),
            ],
        }

        result = validate_plan_coverage(payload, clauses)

        self.assertFalse(result.valid)
        self.assertIn("context semantic unit is not prose", "\n".join(result.errors))




    def test_explicit_navigation_suppresses_same_transaction_generic_resume(self) -> None:
        from agent.harness.action_registry import normalize_action_relations

        actions = [
            {
                "type": "change_group",
                "group": "qps_profile",
                "navigation_explicit": True,
                "source_evidence": "configure QPS first",
            },
            {
                "type": "resume_current_flow",
                "source_evidence": "return afterward",
            },
        ]

        self.assertEqual(
            [item["type"] for item in normalize_action_relations(actions)],
            ["change_group"],
        )

    def test_generic_resume_remains_when_no_navigation_is_present(self) -> None:
        from agent.harness.action_registry import normalize_action_relations

        actions = [{
            "type": "resume_current_flow",
            "source_evidence": "resume the current setup",
        }]

        self.assertEqual(normalize_action_relations(actions), actions)

    def test_equivalent_group_navigation_is_deduplicated_before_queueing(self) -> None:
        from agent.harness.action_registry import normalize_action_relations

        actions = [
            {
                "type": "change_group",
                "group": "qps_profile",
                "navigation_explicit": True,
                "source_evidence": "configure QPS first before the rest",
            },
            {
                "type": "change_group",
                "group": "qps_profile",
                "navigation_explicit": True,
                "semantic_purpose_verified": True,
                "source_evidence": "QPS",
            },
        ]

        normalized = normalize_action_relations(actions)

        self.assertEqual(len(normalized), 1)
        self.assertEqual(normalized[0]["group"], "qps_profile")
        self.assertEqual(
            normalized[0]["source_evidence"],
            "configure QPS first before the rest",
        )
        self.assertIs(normalized[0]["semantic_purpose_verified"], True)

    def test_trusted_action_queue_exposes_one_equivalent_navigation_owner(self) -> None:
        import json

        from agent.harness.intent import _parse_action_queue

        result = _parse_action_queue(json.dumps({
            "actions": [
                {
                    "type": "change_group",
                    "group": "accounts_disk",
                    "navigation_explicit": True,
                    "source_evidence": "open accounts disk settings",
                    "group_navigation_semantic_verified": True,
                },
                {
                    "type": "change_group",
                    "group": "accounts_disk",
                    "navigation_explicit": True,
                    "source_evidence": "revise that optional disk",
                    "group_navigation_semantic_verified": True,
                },
            ],
        }), trusted_metadata=True)

        self.assertEqual(len(result["actions"]), 1)
        self.assertEqual(result["actions"][0]["group"], "accounts_disk")
        self.assertIs(result["actions"][0]["group_navigation_semantic_verified"], True)

    def test_distinct_group_navigation_is_preserved(self) -> None:
        from agent.harness.action_registry import normalize_action_relations

        actions = [
            {"type": "change_group", "group": "qps_profile"},
            {"type": "change_group", "group": "observability"},
        ]

        self.assertEqual(normalize_action_relations(actions), actions)

    def test_distinct_same_group_owner_mutations_are_not_navigation_deduplicated(self) -> None:
        from agent.harness.action_registry import normalize_action_relations

        actions = [
            {
                "type": "set_qps_override",
                "qps_overrides": {"INITIAL_QPS": 5},
                "source_evidence": "start at 5",
            },
            {
                "type": "set_qps_override",
                "qps_overrides": {"MAX_QPS": 20},
                "source_evidence": "cap at 20",
            },
        ]

        self.assertEqual(normalize_action_relations(actions), actions)

    def test_pending_preserving_domain_action_owns_same_transaction_return(self) -> None:
        from agent.harness.action_registry import normalize_action_relations

        actions = [
            {
                "type": "request_qps_customization",
                "source_evidence": "adjust QPS first",
            },
            {
                "type": "resume_current_flow",
                "source_evidence": "then return",
            },
        ]

        self.assertEqual(
            [item["type"] for item in normalize_action_relations(actions)],
            ["request_qps_customization"],
        )

    def test_one_replacement_action_can_own_negative_and_positive_clauses(self) -> None:
        text = "I do not want RPC load now; observe node sync instead"
        clauses = segment_user_turn(text)
        payload = {
            "actions": [{
                "type": "choose_target_mode",
                "target_mode": "sync-observe",
                "target_mode_explicit": True,
                "source_evidence": "observe node sync instead",
            }],
            "semantic_units": [
                _unit(clauses[0], 1, [0]),
                _unit(clauses[1], 2, [0]),
            ],
        }

        result = validate_plan_coverage(payload, clauses)

        self.assertTrue(result.valid, result.errors)

    def test_semantic_receipt_is_attached_only_after_admission(self) -> None:
        import json

        from agent.harness.intent import _attach_semantic_admission_receipts, _parse_action_queue

        raw = json.dumps({
            "actions": [{
                "type": "choose_target_mode",
                "target_mode": "fake-node",
                "target_mode_explicit": True,
                "source_evidence": "fake-node",
            }],
            "semantic_units": [],
            "target_mode_selection_admissions": [0],
            "pending_support_unit_ids": ["model-forged-unit"],
        })

        parsed = _parse_action_queue(
            _attach_semantic_admission_receipts(raw),
            trusted_metadata=True,
        )

        self.assertIs(parsed["actions"][0]["semantic_purpose_verified"], True)
        self.assertIs(parsed["actions"][0]["target_mode_semantic_verified"], True)


    def test_one_literal_target_mode_is_deterministic_evidence(self) -> None:
        from agent.harness.input_values import target_mode_evidence_matches

        source = "I want to test BNB with fake-node and inspect supported methods"

        self.assertTrue(target_mode_evidence_matches("fake-node", "fake-node", source))

    def test_mutating_enum_values_use_registry_grounding(self) -> None:
        from agent.harness.action_registry import semantic_grounding_arguments

        examples = {
            "choose_target_mode": {"target_mode": "fake-node"},
            "set_rpc_mode": {"rpc_mode": "single"},
            "set_qps_mode": {"qps_mode": "quick"},
            "set_observability": {"observability_mode": "disabled"},
            "set_sync_observe_source": {"sync_observe_source": "endpoint_only"},
            "choose_adapter_family": {"adapter_family": "jsonrpc"},
            "set_accounts_presence": {"has_accounts_device": False},
        }
        for action_type, arguments in examples.items():
            with self.subTest(action_type=action_type):
                self.assertEqual(
                    semantic_grounding_arguments({"type": action_type, **arguments}),
                    tuple(arguments),
                )

        self.assertEqual(
            semantic_grounding_arguments({
                "type": "rpc_catalog_command",
                "catalog_command": "set_endpoint",
                "rpc_endpoint": "https://example.invalid/rpc",
            }),
            ("rpc_endpoint",),
        )
        self.assertEqual(
            semantic_grounding_arguments({
                "type": "secondary_handoff_command",
                "handoff_command": "append_evidence",
                "handoff_evidence": "official protocol notes",
            }),
            (),
        )









    def test_untrusted_planner_receipts_and_action_metadata_are_stripped(self) -> None:
        import json

        from agent.harness.intent import _prepare_untrusted_action_document

        prepared = json.loads(_prepare_untrusted_action_document(json.dumps({
            "actions": [{
                "type": "choose_target_mode",
                "target_mode": "fake-node",
                "action_id": "model-forged",
                "target_mode_semantic_verified": True,
                "_origin_text": "model-forged",
            }],
            "target_mode_selection_admissions": [0],
            "consultation_admissions": [0],
            "admission_rejections": [{"action_index": 0}],
            "admission_action_ids": ["model-forged"],
        })))

        self.assertNotIn("target_mode_selection_admissions", prepared)
        self.assertNotIn("pending_support_unit_ids", prepared)
        self.assertNotIn("consultation_admissions", prepared)
        self.assertNotIn("admission_rejections", prepared)
        self.assertNotEqual(prepared["admission_action_ids"], ["model-forged"])
        self.assertNotIn("action_id", prepared["actions"][0])
        self.assertNotIn("target_mode_semantic_verified", prepared["actions"][0])
        self.assertNotIn("_origin_text", prepared["actions"][0])

    def test_untrusted_nested_arguments_cannot_forge_action_metadata(self) -> None:
        import json

        from agent.harness.intent import _prepare_untrusted_action_document

        with self.assertRaisesRegex(
            ValueError,
            "arguments.v1 is retired for current-turn actions",
        ):
            _prepare_untrusted_action_document(json.dumps({
                "actions": [{
                    "type": "choose_target_mode",
                    "arguments": {
                        "target_mode": "fake-node",
                        "source_evidence": "Use fake-node.",
                        "action_id": "nested-model-forged",
                        "target_mode_semantic_verified": True,
                        "pending_option_semantic_verified": True,
                        "semantic_purpose_verified": True,
                        "_origin_text": "nested-model-forged",
                    },
                }],
            }))










































    def test_rejected_target_mode_preserves_sibling_actions_and_unit_ownership(self) -> None:
        import json
        from agent.harness.intent import _remove_rejected_action_indexes
        payload = {
            "actions": [
                {
                    "type": "choose_target_mode",
                    "target_mode": "fake-node",
                    "target_mode_explicit": True,
                    "source_evidence": "benchmark BNB with mixed and quick",
                },
                {"type": "select_chain", "chain": "bsc", "source_evidence": "BNB"},
                {"type": "select_rpc_mode", "rpc_mode": "mixed", "source_evidence": "mixed"},
            ],
            "semantic_units": [{
                "unit_id": "unit-1",
                "clause_id": "clause-1",
                "source_text": "benchmark BNB with mixed and quick",
                "disposition": "action",
                "action_indexes": [0, 1, 2],
                "reason": "compound request",
            }],
        }

        filtered_text, changed = _remove_rejected_action_indexes(
            json.dumps(payload),
            (0,),
            reason="the user did not select a target mode",
        )
        filtered = json.loads(filtered_text)

        self.assertTrue(changed)
        self.assertEqual([row["type"] for row in filtered["actions"]], ["select_chain", "select_rpc_mode"])
        self.assertEqual(filtered["semantic_units"][0]["action_indexes"], [0, 1])
        self.assertEqual(filtered["semantic_units"][0]["disposition"], "action")
        self.assertEqual(filtered["admission_rejections"][0]["action_type"], "choose_target_mode")
        self.assertTrue(filtered["admission_rejections"][0]["admission_action_id"])
        self.assertEqual(filtered["admission_rejections"][0]["stage_action_index"], 0)
        self.assertNotIn("action_index", filtered["admission_rejections"][0])

    def test_sequential_rejections_preserve_distinct_admission_action_ids(self) -> None:
        import json

        from agent.harness.intent import _remove_rejected_action_indexes

        payload = json.dumps({
            "actions": [
                {"type": "choose_target_mode", "target_mode": "fake-node"},
                {"type": "select_chain", "chain": "bsc"},
                {"type": "select_rpc_mode", "rpc_mode": "single"},
            ],
        })
        after_first, _ = _remove_rejected_action_indexes(payload, (0,), reason="first gate")
        first = json.loads(after_first)
        retained_id = first["admission_action_ids"][0]
        after_second, _ = _remove_rejected_action_indexes(after_first, (0,), reason="second gate")
        second = json.loads(after_second)

        self.assertNotEqual(
            second["admission_rejections"][0]["admission_action_id"],
            second["admission_rejections"][1]["admission_action_id"],
        )
        self.assertEqual(second["admission_rejections"][1]["admission_action_id"], retained_id)
        self.assertEqual([row["stage_action_index"] for row in second["admission_rejections"]], [0, 0])

    def test_readded_identical_action_does_not_reuse_rejected_admission_id(self) -> None:
        import json

        from agent.harness.intent import _remove_rejected_action_indexes

        action = {"type": "choose_target_mode", "target_mode": "fake-node"}
        first_text, _ = _remove_rejected_action_indexes(
            json.dumps({"actions": [action]}),
            (0,),
            reason="first attempt",
        )
        first = json.loads(first_text)
        first_id = first["admission_rejections"][0]["admission_action_id"]
        first["actions"] = [action]
        second_text, _ = _remove_rejected_action_indexes(
            json.dumps(first),
            (0,),
            reason="second attempt",
        )
        second = json.loads(second_text)

        self.assertNotEqual(
            first_id,
            second["admission_rejections"][1]["admission_action_id"],
        )






    def test_malformed_navigation_receipt_is_not_authoritative(self) -> None:
        from agent.harness.intent import _valid_group_navigation_admissions

        actions = [{
            "type": "change_group",
            "group": "qps_profile",
            "source_evidence": "configure QPS first",
        }]
        payload = {"group_navigation_admissions": [{
            "action_index": 0,
            "group": "observability",
            "source_evidence": "configure QPS first",
            "destination_quote": "QPS",
        }]}
        self.assertEqual(_valid_group_navigation_admissions(payload, actions), {})




















    def test_explicit_selected_value_remains_authoritative_over_answer_fallback(self) -> None:
        from agent.harness.intent import _declared_option_for_pending_answer

        pending = {
            "options": [
                {"id": "bsc", "value": {"chain_choice": "bsc"}},
                {"id": "reenter", "value": "reenter_chain"},
            ],
        }
        option = _declared_option_for_pending_answer({
            "type": "answer_pending",
            "answer": "reenter_chain",
            "selected_value": {"chain_choice": "bsc"},
        }, pending)

        self.assertEqual(option["id"], "bsc")


















    @unittest.skipUnless(sys.version_info >= (3, 10), "Harness runtime requires Python 3.10+")
    def test_target_mode_intake_precedes_tentative_chain_intake(self) -> None:
        from agent.harness.coordinator import _order_action_queue
        from agent.harness.state import new_state

        actions = [
            {
                "type": "request_chain_selection",
                "chain_candidates": ["BNB"],
                "source_evidence": "possibly BNB",
                "_plan_index": 0,
            },
            {
                "type": "request_target_mode_selection",
                "source_evidence": "not sure which mode",
                "_plan_index": 1,
            },
        ]

        ordered = _order_action_queue(new_state("dependency-test"), actions)

        self.assertEqual(
            [action["type"] for action in ordered],
            ["request_target_mode_selection", "request_chain_selection"],
        )

    def test_structured_config_owner_recovers_an_unmapped_assignment(self) -> None:
        import json
        from agent.harness.intent import _reconcile_structured_candidate_ownership
        from agent.harness.plan_coverage import segment_user_turn

        source = (
            "Apply this copied config:\n"
            "CLOUD_REGION=asia-east1\n"
            "unrelated_ticket=INC-12345"
        )
        payload = {
            "actions": [{
                "type": "propose_config_values",
                "config_values": {"CLOUD_REGION": "asia-east1"},
                "unmapped_values": {},
                "source_format": "env",
                "source_evidence": "CLOUD_REGION=asia-east1",
            }],
            "semantic_units": [
                {
                    "unit_id": "unit-1",
                    "clause_id": "clause-1",
                    "source_text": "Apply this copied config:\nCLOUD_REGION=asia-east1\n",
                    "disposition": "action",
                    "action_indexes": [0],
                },
                {
                    "unit_id": "unit-2",
                    "clause_id": "clause-1",
                    "source_text": "unrelated_ticket=INC-12345",
                    "disposition": "unresolved",
                    "action_indexes": [],
                },
            ],
        }

        reconciled = json.loads(_reconcile_structured_candidate_ownership(
            json.dumps(payload),
            segment_user_turn(source),
        ))

        self.assertEqual(
            reconciled["actions"][0]["unmapped_values"],
            {"UNRELATED_TICKET": "INC-12345"},
        )
        self.assertEqual(len(reconciled["semantic_units"]), 2)
        self.assertEqual(reconciled["semantic_units"][0]["disposition"], "action")
        self.assertEqual(reconciled["semantic_units"][0]["action_indexes"], [0])
        self.assertEqual(reconciled["semantic_units"][1]["disposition"], "unresolved")
        self.assertEqual(reconciled["semantic_units"][1]["action_indexes"], [])













    def test_structured_block_without_config_action_is_not_reclassified(self) -> None:
        import json
        from agent.harness.intent import _reconcile_structured_candidate_ownership
        from agent.harness.plan_coverage import segment_user_turn

        source = "Example only:\nCLOUD_REGION=asia-east1"
        payload = {
            "actions": [{"type": "analyze_evidence", "evidence": source}],
            "semantic_units": [{
                "unit_id": "unit-1",
                "clause_id": "clause-1",
                "source_text": source,
                "disposition": "action",
                "action_indexes": [0],
            }],
        }
        original = json.dumps(payload)

        self.assertEqual(
            _reconcile_structured_candidate_ownership(original, segment_user_turn(source)),
            original,
        )

    def test_structured_workflow_value_requires_its_typed_owner(self) -> None:
        import json
        from agent.harness.intent import _reconcile_structured_candidate_ownership
        from agent.harness.plan_coverage import segment_user_turn

        source = "Apply this config:\nCLOUD_REGION=asia-east1\nRPC_MODE=single"
        payload = {
            "actions": [{
                "type": "propose_config_values",
                "config_values": {"CLOUD_REGION": "asia-east1"},
                "unmapped_values": {},
                "source_format": "env",
                "source_evidence": "CLOUD_REGION=asia-east1",
            }],
            "semantic_units": [
                {
                    "unit_id": "unit-1",
                    "clause_id": "clause-1",
                    "source_text": "Apply this config:\nCLOUD_REGION=asia-east1\n",
                    "disposition": "action",
                    "action_indexes": [0],
                },
                {
                    "unit_id": "unit-2",
                    "clause_id": "clause-1",
                    "source_text": "RPC_MODE=single",
                    "disposition": "unresolved",
                    "action_indexes": [],
                },
            ],
        }

        reconciled = json.loads(_reconcile_structured_candidate_ownership(
            json.dumps(payload),
            segment_user_turn(source),
        ))

        self.assertEqual(len(reconciled["semantic_units"]), 2)
        self.assertEqual(reconciled["semantic_units"][1]["disposition"], "unresolved")

    def test_response_language_is_a_typed_admission_action(self) -> None:
        from agent.harness.action_registry import ACTION_BY_TYPE, validate_action_contract

        action = {
            "type": "set_response_language",
            "language": "zh",
            "source_evidence": "请后续用中文回答",
        }

        validate_action_contract(action)
        self.assertEqual(ACTION_BY_TYPE["set_response_language"].owner, "orientation")
        self.assertEqual(ACTION_BY_TYPE["set_response_language"].lifetime, "turn_local")

    def test_target_mode_intake_contract_covers_initial_and_replacement_selection(self) -> None:
        from agent.harness.action_registry import ACTION_BY_TYPE

        purpose = ACTION_BY_TYPE["request_target_mode_selection"].purpose

        self.assertIn("when it is unresolved", purpose)
        self.assertIn("replacement", purpose)
        self.assertNotIn("Ask for a replacement target mode", purpose)

    def test_consultation_and_analysis_actions_require_source_purpose_review(self) -> None:
        from agent.harness.intent import _requires_semantic_fulfillment_review

        for action in (
            {"type": "answer_opening_question", "topic": "config_explanation"},
            {"type": "analyze_evidence", "evidence": "Traceback"},
            {"type": "analyze_report", "subject": "latency"},
        ):
            with self.subTest(action_type=action["type"]):
                self.assertTrue(_requires_semantic_fulfillment_review(action))










    def test_pending_choice_rejects_semantic_sentence_as_answer(self) -> None:
        from agent.harness.intent import _validate_action_document
        from agent.harness.state import new_state

        text = "I want to adjust QPS before choosing workload defaults"
        clauses = segment_user_turn(text)
        state = new_state("unit-thread", language="en")
        state["pending_question"] = {
            "id": "workload_confirm",
            "kind": "numbered_choice",
            "options": [
                {"id": "1", "label": "Use defaults", "value": "default"},
                {"id": "2", "label": "Add custom RPC method", "value": "custom_rpc"},
            ],
        }
        payload = {
            "actions": [{"type": "answer_pending", "answer": text, "source_evidence": text}],
            "semantic_units": [_unit(clauses[0], 1, [0])],
        }

        result = _validate_action_document(__import__("json").dumps(payload), clauses, state)

        self.assertFalse(result.valid)
        self.assertTrue(any("does not select an exact declared option" in error for error in result.errors))

    def test_pending_choice_accepts_exact_declared_label(self) -> None:
        from agent.harness.intent import (
            _canonicalize_pending_choice_actions,
            _validate_action_document,
        )
        from agent.harness.state import new_state

        text = "Use defaults"
        clauses = segment_user_turn(text)
        state = new_state("unit-thread", language="en")
        state["pending_question"] = {
            "id": "workload_confirm",
            "kind": "numbered_choice",
            "options": [{"id": "1", "label": "Use defaults", "value": "default"}],
        }
        payload = {
            "actions": [{"type": "answer_pending", "answer": text, "source_evidence": text}],
            "semantic_units": [_unit(clauses[0], 1, [0])],
        }

        candidate = _canonicalize_pending_choice_actions(
            __import__("json").dumps(payload),
            state,
        )
        result = _validate_action_document(candidate, clauses, state)

        self.assertTrue(result.valid, result.errors)

    def test_action_contract_error_is_not_reported_as_missing_semantic_units(self) -> None:
        from agent.harness.intent import _validate_action_document

        clauses = segment_user_turn("use quick")
        payload = {
            "actions": [{"type": "set_qps_mode", "qps_mode": "invalid"}],
            "semantic_units": [_unit(clauses[0], 1, [0])],
        }

        result = _validate_action_document(__import__("json").dumps(payload), clauses)

        self.assertFalse(result.valid)
        self.assertTrue(any("action 0 contract invalid" in error for error in result.errors))
        self.assertFalse(any("missing semantic units" in error for error in result.errors))

    def test_compound_turn_rejects_silently_dropped_endpoint_clause(self) -> None:
        clauses = segment_user_turn(
            "Use eth0 at 100 Gbps. The node RPC is http://geth-dev:8545. "
            "The process is geth; I do not have a Prometheus URL."
        )
        result = validate_plan_coverage(
            {
                "actions": [{"type": "propose_config_values", "NETWORK_INTERFACE": "eth0"}],
                "semantic_units": [_unit(clauses[0], 1, [0], reason="network interface")],
            },
            clauses,
        )
        self.assertFalse(result.valid)
        self.assertTrue(any("missing semantic units" in error for error in result.errors))

    def test_complete_compound_mapping_is_admitted(self) -> None:
        clauses = segment_user_turn(
            "Use eth0 at 100 Gbps. The node RPC is http://geth-dev:8545."
        )
        actions = [
            {"type": "propose_config_values", "NETWORK_INTERFACE": "eth0", "NETWORK_MAX_BANDWIDTH_GBPS": 100},
            {"type": "propose_config_values", "LOCAL_RPC_URL": "http://geth-dev:8545"},
        ]
        result = validate_plan_coverage(
            {
                "actions": actions,
                "semantic_units": [
                    _unit(clauses[0], 1, [0], reason="network"),
                    _unit(clauses[1], 2, [1], reason="endpoint"),
                ],
            },
            clauses,
        )
        self.assertTrue(result.valid, result.errors)

    def test_unresolved_clause_blocks_partial_commit(self) -> None:
        clauses = segment_user_turn("Use BSC. Also apply the mystery setting from my other machine.")
        result = validate_plan_coverage(
            {
                "actions": [{"type": "choose_chain", "chain": "bsc"}],
                "semantic_units": [
                    _unit(clauses[0], 1, [0], reason="chain"),
                    _unit(clauses[1], 2, [], disposition="unresolved", reason="setting not supplied"),
                ],
            },
            clauses,
        )
        self.assertFalse(result.valid)
        self.assertEqual(result.unresolved_clauses, (clauses[1].text,))

    def test_multiline_json_is_one_atomic_clause(self) -> None:
        clauses = segment_user_turn('{\n  "LOCAL_RPC_URL": "http://geth-dev:8545",\n  "BLOCKCHAIN_NODE": "geth"\n}')
        self.assertEqual(len(clauses), 1)
        self.assertEqual(clauses[0].input_shape, "structured")

    def test_prose_and_multiline_json_are_separate_clauses(self) -> None:
        clauses = segment_user_turn(
            '直接测试 BNB，用 fake-node 和 quick。这是机器信息：\n'
            '{\n  "CLOUD_REGION": "asia-east1",\n  "LEDGER_DEVICE": "vda"\n}'
        )

        self.assertEqual([item.input_shape for item in clauses], ["prose", "structured"])
        self.assertEqual(
            clauses[-1].text,
            '这是机器信息：\n{\n  "CLOUD_REGION": "asia-east1",\n  "LEDGER_DEVICE": "vda"\n}',
        )

    def test_structured_block_with_trailing_prose_preserves_both_shapes(self) -> None:
        clauses = segment_user_turn(
            '{\n  "CLOUD_REGION": "asia-east1"\n}\n请先让我确认推断结果。'
        )

        self.assertEqual([item.input_shape for item in clauses], ["structured", "prose"])
        self.assertEqual(clauses[1].text, "请先让我确认推断结果。")

    def test_multiline_env_and_yaml_runs_remain_atomic_beside_prose(self) -> None:
        clauses = segment_user_turn(
            "Use these values:\nCLOUD_REGION=asia-east1\nCLOUD_ZONE=asia-east1-c\n"
            "and verify this inventory:\ndevice: vda\nsize_gib: 926"
        )

        self.assertEqual(
            [item.input_shape for item in clauses],
            ["structured", "structured"],
        )
        self.assertEqual(
            clauses[0].text,
            "Use these values:\nCLOUD_REGION=asia-east1\nCLOUD_ZONE=asia-east1-c",
        )
        self.assertEqual(
            clauses[1].text,
            "and verify this inventory:\ndevice: vda\nsize_gib: 926",
        )

    def test_structured_url_must_be_present_in_mapped_action(self) -> None:
        clauses = segment_user_turn('{"LOCAL_RPC_URL":"http://geth-dev:8545"}')
        result = validate_plan_coverage(
            {
                "actions": [{"type": "propose_config_values", "LOCAL_RPC_URL": "http://wrong:8545"}],
                "semantic_units": [_unit(clauses[0], 1, [0], reason="endpoint")],
            },
            clauses,
        )
        self.assertFalse(result.valid)
        self.assertTrue(any("URL" in error for error in result.errors))

    def test_structured_wire_method_must_be_owned_by_mapped_action(self) -> None:
        clauses = segment_user_turn(
            '{"jsonrpc":"2.0","id":1,"method":"eth_getBalance","params":[]}'
        )
        actions = [{"type": "rpc_catalog_command", "catalog_command": "enter"}]

        result = validate_plan_coverage(
            {"actions": actions, "semantic_units": [_unit(clauses[0], 1, [0])]},
            clauses,
        )

        self.assertFalse(result.valid)
        self.assertTrue(any("eth_getBalance" in error for error in result.errors))

    def test_prose_wire_literal_role_is_deferred_to_semantic_admission(self) -> None:
        clauses = segment_user_turn(
            "Use my own RPC method instead of eth_getBalance."
        )
        actions = [{"type": "rpc_catalog_command", "catalog_command": "enter"}]

        result = validate_plan_coverage(
            {"actions": actions, "semantic_units": [_unit(clauses[0], 1, [0])]},
            clauses,
        )

        self.assertTrue(result.valid, result.errors)



    def test_environment_yaml_keys_are_not_rpc_wire_methods(self) -> None:
        clauses = segment_user_turn(
            "storage:\n  ledger_device: vda\n  data_vol_type: hyperdisk-balanced\n"
            "network:\n  bandwidth_gbps: 100"
        )
        actions = [{
            "type": "propose_config_values",
            "config_values": {
                "LEDGER_DEVICE": "vda",
                "DATA_VOL_TYPE": "hyperdisk-balanced",
                "NETWORK_MAX_BANDWIDTH_GBPS": "100",
            },
        }]

        result = validate_plan_coverage(
            {"actions": actions, "semantic_units": [_unit(clauses[0], 1, [0])]},
            clauses,
        )

        self.assertTrue(result.valid, result.errors)

    def test_full_clause_can_map_to_dependency_ordered_actions_without_losing_method(self) -> None:
        clauses = segment_user_turn(
            "Set target mode to fake-node, chain to BSC, and start custom RPC setup for eth_getBalance."
        )
        actions = [
            {"type": "choose_target_mode", "target_mode": "fake-node"},
            {"type": "choose_chain", "chain_text": "BSC"},
            {"type": "rpc_catalog_command", "catalog_command": "enter"},
            {"type": "rpc_catalog_command", "catalog_command": "set_method", "rpc_method": "eth_getBalance"},
        ]

        result = validate_plan_coverage(
            {"actions": actions, "semantic_units": [_unit(clauses[0], 1, [0, 1, 2, 3])]},
            clauses,
        )

        self.assertTrue(result.valid, result.errors)

    def test_punctuation_free_compound_turn_requires_atomic_semantic_units(self) -> None:
        clauses = segment_user_turn("先配置 QPS quick 然后关闭 Grafana 再回磁盘")
        self.assertEqual(len(clauses), 1)
        text = clauses[0].text
        boundaries = [0, text.index("然后"), text.index("再回"), len(text)]
        actions = [
            {"type": "set_qps_mode", "qps_mode": "quick"},
            {"type": "set_observability", "observability_mode": "disabled"},
            {"type": "change_group", "group": "hardware_storage"},
        ]
        units = []
        for index, (start, end) in enumerate(zip(boundaries, boundaries[1:]), start=1):
            units.append({
                "unit_id": f"unit-{index}",
                "clause_id": clauses[0].clause_id,
                "start": start,
                "end": end,
                "source_text": text[start:end],
                "disposition": "action",
                "action_indexes": [index - 1],
                "reason": "independent request",
            })
        result = validate_plan_coverage({"actions": actions, "semantic_units": units}, clauses)
        self.assertTrue(result.valid, result.errors)

    def test_prose_anchor_partition_keeps_unclaimed_text_visible_to_reviewer(self) -> None:
        from agent.harness.plan_coverage import _canonicalize_semantic_units

        clauses = segment_user_turn("configure quick then disable metrics")
        text = clauses[0].text
        units, errors = _canonicalize_semantic_units([{
            "unit_id": "unit-1",
            "clause_id": clauses[0].clause_id,
            "source_text": "configure quick",
            "disposition": "action",
            "action_indexes": [0],
            "reason": "qps",
        }], {clauses[0].clause_id: clauses[0]})

        self.assertEqual(errors, [])
        self.assertEqual(units[0]["source_text"], text)
        self.assertIn("then disable metrics", units[0]["source_text"])

    def test_compound_prose_conjunctions_do_not_discard_registered_actions(self) -> None:
        clauses = segment_user_turn(
            "Use fake-node for BSC, choose the quick QPS profile, and disable observability."
        )
        actions = [
            {"type": "choose_target_mode", "target_mode": "fake-node"},
            {"type": "choose_chain", "chain_text": "BSC"},
            {"type": "set_qps_mode", "qps_mode": "quick"},
            {"type": "set_observability", "observability_mode": "disabled"},
        ]
        units = [
            {
                "unit_id": "unit-1",
                "clause_id": clauses[0].clause_id,
                "source_text": "Use fake-node for BSC",
                "disposition": "action",
                "action_indexes": [0, 1],
                "reason": "mode and chain",
            },
            {
                "unit_id": "unit-2",
                "clause_id": clauses[0].clause_id,
                "source_text": "choose the quick QPS profile",
                "disposition": "action",
                "action_indexes": [2],
                "reason": "qps",
            },
            {
                "unit_id": "unit-3",
                "clause_id": clauses[0].clause_id,
                "source_text": "disable observability",
                "disposition": "action",
                "action_indexes": [3],
                "reason": "observability",
            },
        ]

        result = validate_plan_coverage(
            {"actions": actions, "semantic_units": units}, clauses
        )

        self.assertTrue(result.valid, result.errors)

    def test_harness_derives_unicode_spans_for_compound_consultation(self) -> None:
        clauses = segment_user_turn(
            "我想先知道你能做什么，以及如果要压测一个真实的 BNB 节点，我需要准备哪些东西？"
        )
        result = validate_plan_coverage(
            {
                "actions": [
                    {"type": "answer_opening_question", "topic": "capabilities"},
                    {"type": "answer_opening_question", "topic": "requirements"},
                ],
                "semantic_units": [
                    {
                        "unit_id": "unit-1",
                        "clause_id": "clause-1",
                        "start": 0,
                        "end": 999,
                        "source_text": "我想先知道你能做什么",
                        "disposition": "action",
                        "action_indexes": [0],
                        "reason": "capabilities",
                    },
                    {
                        "unit_id": "unit-2",
                        "clause_id": "clause-1",
                        "start": -1,
                        "end": -1,
                        "source_text": "以及如果要压测一个真实的 BNB 节点，我需要准备哪些东西？",
                        "disposition": "action",
                        "action_indexes": [1],
                        "reason": "requirements",
                    },
                ],
            },
            clauses,
        )
        self.assertTrue(result.valid, result.errors)

    def test_repeated_ambiguous_source_anchors_are_rejected(self) -> None:
        clauses = segment_user_turn("status status")
        result = validate_plan_coverage(
            {
                "actions": [{"type": "answer_opening_question", "topic": "current_context"}],
                "semantic_units": [{
                    "unit_id": "unit-1",
                    "clause_id": "clause-1",
                    "source_text": "status",
                    "disposition": "action",
                    "action_indexes": [0],
                    "reason": "status",
                }],
            },
            clauses,
        )
        self.assertFalse(result.valid)
        self.assertTrue(any("ambiguous" in error or "omitted prose" in error for error in result.errors))

    def test_structured_clause_cannot_be_reconstructed_from_partial_anchor(self) -> None:
        clauses = segment_user_turn('{"LOCAL_RPC_URL":"http://geth-dev:8545"}')
        result = validate_plan_coverage(
            {
                "actions": [{"type": "propose_config_values", "LOCAL_RPC_URL": "http://geth-dev:8545"}],
                "semantic_units": [{
                    "unit_id": "unit-1",
                    "clause_id": "clause-1",
                    "source_text": "http://geth-dev:8545",
                    "disposition": "action",
                    "action_indexes": [0],
                    "reason": "endpoint",
                }],
            },
            clauses,
        )
        self.assertFalse(result.valid)
        self.assertTrue(any("structured source anchors omit content" in error for error in result.errors))

    def test_atomic_structured_clause_can_map_to_multiple_owner_actions(self) -> None:
        clauses = segment_user_turn("Use values:\nCLOUD_REGION=asia-east1\nRPC_MODE=mixed")
        actions = [
            {
                "type": "propose_config_values",
                "config_values": {"CLOUD_REGION": "asia-east1"},
                "source_evidence": clauses[0].text,
            },
            {
                "type": "set_rpc_mode",
                "rpc_mode": "mixed",
                "source_evidence": "RPC_MODE=mixed",
            },
        ]
        result = validate_plan_coverage(
            {
                "actions": actions,
                "semantic_units": [
                    {
                        "unit_id": "unit-1",
                        "clause_id": "clause-1",
                        "source_text": clauses[0].text,
                        "disposition": "action",
                        "action_indexes": [0],
                        "reason": "environment values",
                    },
                    {
                        "unit_id": "unit-2",
                        "clause_id": "clause-1",
                        "source_text": clauses[0].text,
                        "disposition": "action",
                        "action_indexes": [1],
                        "reason": "workflow-owned RPC mode",
                    },
                ],
            },
            clauses,
        )

        self.assertTrue(result.valid, result.errors)

    def test_atomic_structured_clause_rejects_action_and_unresolved_mix(self) -> None:
        clauses = segment_user_turn("Use values:\nCLOUD_REGION=asia-east1\nMYSTERY=value")
        known_anchor = "Use values:\nCLOUD_REGION=asia-east1"
        unknown_anchor = "MYSTERY=value"
        result = validate_plan_coverage(
            {
                "actions": [{"type": "propose_config_values", "config_values": {"CLOUD_REGION": "asia-east1"}}],
                "semantic_units": [
                    {
                        "unit_id": "unit-1",
                        "clause_id": "clause-1",
                        "source_text": known_anchor,
                        "disposition": "action",
                        "action_indexes": [0],
                        "reason": "known value",
                    },
                    {
                        "unit_id": "unit-2",
                        "clause_id": "clause-1",
                        "source_text": unknown_anchor,
                        "disposition": "unresolved",
                        "action_indexes": [],
                        "reason": "unknown value",
                    },
                ],
            },
            clauses,
        )

        self.assertFalse(result.valid)
        self.assertEqual(result.unresolved_clauses, (unknown_anchor,))

    def test_consultation_scope_accepts_only_read_only_effects(self) -> None:
        clauses = segment_user_turn("I am only asking and do not change configuration")
        unit = {
            "unit_id": "unit-1",
            "clause_id": "clause-1",
            "source_text": clauses[0].text,
            "disposition": "action",
            "action_indexes": [0],
            "scope_constraint": "consultation_only",
            "reason": "read-only scope",
        }
        accepted = validate_plan_coverage(
            {
                "actions": [{"type": "answer_opening_question", "topic": "current_config"}],
                "semantic_units": [unit],
            },
            clauses,
        )
        rejected = validate_plan_coverage(
            {
                "actions": [{"type": "set_qps_mode", "qps_mode": "quick"}],
                "semantic_units": [unit],
            },
            clauses,
        )
        self.assertTrue(accepted.valid, accepted.errors)
        self.assertFalse(rejected.valid)
        self.assertTrue(any("configuration_mutation action" in error for error in rejected.errors))

    def test_no_configuration_mutation_scope_accepts_durable_navigation(self) -> None:
        clauses = segment_user_turn("Jump to QPS first without changing its values")
        unit = {
            "unit_id": "unit-1",
            "clause_id": "clause-1",
            "source_text": clauses[0].text,
            "disposition": "action",
            "action_indexes": [0],
            "scope_constraint": "no_configuration_mutation",
            "reason": "navigation without configuration mutation",
        }
        accepted = validate_plan_coverage(
            {
                "actions": [{
                    "type": "change_group",
                    "group": "qps_profile",
                    "navigation_explicit": True,
                    "source_evidence": "Jump to QPS first",
                }],
                "semantic_units": [unit],
            },
            clauses,
        )
        rejected = validate_plan_coverage(
            {
                "actions": [{"type": "set_qps_mode", "qps_mode": "quick"}],
                "semantic_units": [unit],
            },
            clauses,
        )
        self.assertTrue(accepted.valid, accepted.errors)
        self.assertFalse(rejected.valid)
        self.assertTrue(any("configuration_mutation action" in error for error in rejected.errors))

    def test_consultation_scope_does_not_authorize_navigation(self) -> None:
        clauses = segment_user_turn("Only explain the current QPS profile")
        result = validate_plan_coverage(
            {
                "actions": [{
                    "type": "change_group",
                    "group": "qps_profile",
                    "navigation_explicit": True,
                    "source_evidence": clauses[0].text,
                }],
                "semantic_units": [{
                    "unit_id": "unit-1",
                    "clause_id": "clause-1",
                    "source_text": clauses[0].text,
                    "disposition": "action",
                    "action_indexes": [0],
                    "scope_constraint": "consultation_only",
                    "reason": "read-only consultation",
                }],
            },
            clauses,
        )
        self.assertFalse(result.valid)
        self.assertTrue(any("workflow_navigation action" in error for error in result.errors))

    def test_no_execution_scope_rejects_execution_effect_only(self) -> None:
        clauses = segment_user_turn("Configure quick but do not run anything")
        unit = {
            "unit_id": "unit-1",
            "clause_id": "clause-1",
            "source_text": clauses[0].text,
            "disposition": "action",
            "action_indexes": [0],
            "scope_constraint": "no_execution",
            "reason": "configuration without execution",
        }
        accepted = validate_plan_coverage(
            {
                "actions": [{"type": "set_qps_mode", "qps_mode": "quick"}],
                "semantic_units": [unit],
            },
            clauses,
        )
        rejected = validate_plan_coverage(
            {
                "actions": [{"type": "approve_preflight_smoke"}],
                "semantic_units": [unit],
            },
            clauses,
        )
        self.assertTrue(accepted.valid, accepted.errors)
        self.assertFalse(rejected.valid)
        self.assertTrue(any("execution action" in error for error in rejected.errors))

    def test_empty_consultation_scope_mapping_binds_all_read_only_actions(self) -> None:
        clauses = segment_user_turn("I am only consulting and have not selected a test")
        result = validate_plan_coverage(
            {
                "actions": [
                    {"type": "answer_opening_question", "topic": "current_config"},
                    {"type": "answer_opening_question", "topic": "requirements"},
                ],
                "semantic_units": [{
                    "unit_id": "unit-1",
                    "clause_id": "clause-1",
                    "source_text": clauses[0].text,
                    "disposition": "action",
                    "action_indexes": [],
                    "scope_constraint": "consultation_only",
                    "reason": "read-only scope",
                }],
            },
            clauses,
        )
        self.assertTrue(result.valid, result.errors)






class UnresolvedSemanticInventoryTest(unittest.TestCase):
    @staticmethod
    def _unresolved_plan(clause):
        return {
            "actions": [],
            "semantic_units": [{
                "unit_id": "compound-unit",
                "clause_id": clause.clause_id,
                "source_text": clause.text,
                "disposition": "unresolved",
                "action_indexes": [],
                "reason": "initial planner omitted the demands",
            }],
        }











class RegistryBoundedSemanticRecoveryTest(unittest.TestCase):

    def test_absent_unit_table_is_restored_for_independent_admission(self) -> None:
        from agent.harness.intent import _materialize_absent_semantic_units

        clauses = tuple(segment_user_turn(
            "Take me to observability; I want to decide among its options."
        ))
        candidate = _materialize_absent_semantic_units(
            json.dumps({
                "actions": [{
                    "type": "change_group",
                    "group": "observability",
                    "navigation_explicit": True,
                    "source_evidence": "Take me to observability",
                }]
            }),
            clauses,
        )
        payload = json.loads(candidate)

        self.assertEqual(
            [row["source_text"] for row in payload["semantic_units"]],
            [clause.text for clause in clauses],
        )
        self.assertTrue(all(
            row["action_indexes"] == [0] for row in payload["semantic_units"]
        ))

    def test_group_navigation_contract_owns_value_less_revision_purpose(self) -> None:
        from agent.harness.action_registry import ACTION_BY_TYPE
        from agent.harness.intent import (
            GROUP_NAVIGATION_SEMANTIC_POLICY,
            _action_queue_prompt,
            _semantic_fulfillment_prompt,
        )

        self.assertIn("value-less statement", GROUP_NAVIGATION_SEMANTIC_POLICY)
        self.assertIn("without a concrete setting or value", GROUP_NAVIGATION_SEMANTIC_POLICY)
        self.assertIn("decide among that group's later typed options", GROUP_NAVIGATION_SEMANTIC_POLICY)
        self.assertIn(GROUP_NAVIGATION_SEMANTIC_POLICY, _action_queue_prompt())
        self.assertIn(
            GROUP_NAVIGATION_SEMANTIC_POLICY,
            _semantic_fulfillment_prompt(review_kind="units"),
        )
        self.assertIn(
            "operation_restatement",
            ACTION_BY_TYPE["change_group"].semantic_support_relations,
        )





    @staticmethod
    def _provider(decisions):
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        provider = Mock()
        provider.complete.return_value = SimpleNamespace(
            text=json.dumps({"decisions": decisions})
        )
        return provider

    @staticmethod
    def _unit(clause, unit_id, action_indexes, *, disposition="action"):
        return {
            "unit_id": unit_id,
            "clause_id": clause.clause_id,
            "source_text": clause.text,
            "disposition": disposition,
            "action_indexes": list(action_indexes),
            "reason": "planner mapping",
        }






    def test_lifecycle_registry_declares_shared_restatement_contract(self) -> None:
        from agent.harness.action_registry import ACTION_BY_TYPE

        for action_type in (
            "queue_workflow_goal",
            "activate_next_workflow_goal",
            "discard_next_workflow_goal",
        ):
            self.assertIn(
                "operation_restatement",
                ACTION_BY_TYPE[action_type].semantic_support_relations,
            )









    def test_empty_config_proposal_cannot_steal_direct_pending_answer(self) -> None:
        from agent.harness.intent import _remove_empty_config_proposals

        payload = {
            "actions": [
                {
                    "type": "answer_pending",
                    "answer": "run-secret-4821",
                    "source_evidence": "run-secret-4821",
                },
                {
                    "type": "propose_config_values",
                    "config_values": {},
                    "unmapped_values": {},
                    "conflicts": [],
                    "source_evidence": "Use it only for this run.",
                },
            ],
            "semantic_units": [
                {
                    "unit_id": "clause-1",
                    "clause_id": "clause-1",
                    "source_text": "The credential is run-secret-4821.",
                    "disposition": "action",
                    "action_indexes": [0],
                    "reason": "answers the pending field",
                },
                {
                    "unit_id": "clause-2",
                    "clause_id": "clause-2",
                    "source_text": "Use it only for this run.",
                    "disposition": "action",
                    "action_indexes": [1],
                    "reason": "planner attached an empty proposal",
                },
            ],
        }

        normalized = json.loads(_remove_empty_config_proposals(json.dumps(payload)))

        self.assertEqual(normalized["actions"], [payload["actions"][0]])
        self.assertEqual(normalized["semantic_units"][0]["action_indexes"], [0])
        self.assertEqual(normalized["semantic_units"][1]["action_indexes"], [])
        self.assertEqual(normalized["semantic_units"][1]["disposition"], "unresolved")

    def test_nonempty_config_proposal_is_not_removed(self) -> None:
        from agent.harness.intent import _remove_empty_config_proposals

        payload = {
            "actions": [{
                "type": "propose_config_values",
                "config_values": {"NETWORK_INTERFACE": "eth0"},
                "unmapped_values": {},
                "conflicts": [],
            }],
            "semantic_units": [],
        }

        self.assertEqual(
            json.loads(_remove_empty_config_proposals(json.dumps(payload))),
            payload,
        )





















    def test_read_only_consultation_registry_contract_supports_pending_selection_scope(self) -> None:
        from agent.harness.action_registry import ACTION_BY_TYPE

        spec = ACTION_BY_TYPE["answer_opening_question"]
        self.assertTrue(spec.pending_option_admission)
        self.assertIn("non_mutation_scope", spec.semantic_support_relations)

    def test_untrusted_scope_schema_row_is_canonicalized_only_by_exact_registry_identity(self) -> None:
        from agent.harness.action_registry import semantic_scope_schema
        from agent.harness.intent import _prepare_untrusted_action_document

        registered = next(
            row for row in semantic_scope_schema() if row["name"] == "consultation_only"
        )
        payload = {
            "actions": [],
            "semantic_units": [{
                "unit_id": "unit-1",
                "clause_id": "clause-1",
                "source_text": "Show capabilities",
                "disposition": "action",
                "action_indexes": [],
                "scope_constraint": registered,
                "reason": "read-only scope",
            }],
        }

        canonical = json.loads(_prepare_untrusted_action_document(json.dumps(payload)))
        self.assertEqual(
            canonical["semantic_units"][0]["scope_constraint"],
            "consultation_only",
        )

        altered = dict(registered)
        altered["description"] = "untrusted replacement"
        payload["semantic_units"][0]["scope_constraint"] = altered
        preserved = json.loads(_prepare_untrusted_action_document(json.dumps(payload)))
        self.assertEqual(preserved["semantic_units"][0]["scope_constraint"], altered)













    def test_pending_owner_receipt_can_bind_structured_support_context(self) -> None:
        source = (
            "deployment_notes:\n"
            "  node_process: geth\n"
            "  location: this_machine\n"
            "Use the already running local real-node process as the sync-observe source."
        )
        clauses = segment_user_turn(source)
        self.assertEqual([clause.input_shape for clause in clauses], ["structured", "prose"])
        payload = {
            "actions": [{
                "type": "answer_pending",
                "selected_value": "existing_local_node",
                "source_evidence": clauses[1].text,
            }],
            "pending_answer_admissions": [0],
            "pending_support_unit_ids": ["unit-context"],
            "semantic_units": [
                {
                    **_unit(clauses[0], 1, [], disposition="context", reason="pending support"),
                    "unit_id": "unit-context",
                },
                {
                    **_unit(clauses[1], 2, [0]),
                    "unit_id": "unit-choice",
                },
            ],
        }

        admitted = validate_plan_coverage(payload, clauses)
        unowned = validate_plan_coverage(
            {key: value for key, value in payload.items() if key != "pending_support_unit_ids"},
            clauses,
        )

        self.assertTrue(admitted.valid, admitted.errors)
        self.assertFalse(unowned.valid)
        self.assertIn("context semantic unit is not prose", "\n".join(unowned.errors))

















    def test_structured_parser_does_not_replace_compiler_answer_action(self) -> None:
        import json
        from agent.harness.intent import _reconcile_structured_candidate_ownership

        source = '{"ACCOUNTS_DEVICE":"/dev/nvme1n1"}'
        clause = segment_user_turn(source)[0]
        payload = {
            "actions": [{
                "type": "answer_pending",
                "answer": "/dev/nvme1n1",
                "source_evidence": source,
            }],
            "semantic_units": [_unit(clause, 1, [0])],
        }

        reconciled = json.loads(_reconcile_structured_candidate_ownership(
            json.dumps(payload),
            (clause,),
        ))

        self.assertEqual(reconciled, payload)

    def test_structured_parser_hydrates_only_declared_proposal_owner(self) -> None:
        import json
        from agent.harness.intent import _reconcile_structured_candidate_ownership

        source = "CLOUD_REGION=asia-east1\nunrelated_ticket=INC-12345"
        clause = segment_user_turn(source)[0]
        payload = {
            "actions": [{
                "type": "propose_config_values",
                "config_values": {},
                "unmapped_values": {},
                "source_format": "env",
                "source_evidence": source,
            }],
            "semantic_units": [_unit(clause, 1, [0])],
        }

        reconciled = json.loads(_reconcile_structured_candidate_ownership(
            json.dumps(payload),
            (clause,),
        ))

        self.assertEqual(
            reconciled["actions"][0]["config_values"],
            {"CLOUD_REGION": "asia-east1"},
        )
        self.assertEqual(
            reconciled["actions"][0]["unmapped_values"],
            {"UNRELATED_TICKET": "INC-12345"},
        )
        self.assertEqual(reconciled["semantic_units"], payload["semantic_units"])

    def test_cross_clause_structured_assignment_without_pending_intent_is_not_claimed(self) -> None:
        from agent.harness.intent import _reconcile_structured_candidate_ownership
        from agent.harness.state import new_state

        source = "```env\nCLOUD_REGION=asia-east1\n```\nWhat does this setting mean?"
        clauses = segment_user_turn(source)
        payload = {
            "actions": [{
                "type": "answer_capability_question",
                "topic": "configuration",
                "source_evidence": clauses[1].text,
            }],
            "semantic_units": [
                _unit(clauses[0], 1, [], disposition="unresolved"),
                _unit(clauses[1], 2, [0]),
            ],
        }
        state = new_state("structured-pending-consultation", language="en")
        state["pending_question"] = {
            "id": "CLOUD_REGION",
            "group": "provider_deployment",
            "field": "CLOUD_REGION",
            "kind": "manual_value",
            "manual_input_allowed": True,
            "validation": {"value_type": "scalar_token"},
            "options": [],
        }

        reconciled = json.loads(_reconcile_structured_candidate_ownership(
            json.dumps(payload),
            clauses,
            state,
        ))

        self.assertEqual(reconciled, payload)

    def test_cross_clause_structured_assignment_does_not_override_a_different_pending_value(self) -> None:
        from agent.harness.intent import _reconcile_structured_candidate_ownership
        from agent.harness.state import new_state

        source = "```env\nNETWORK_MAX_BANDWIDTH_GBPS=25\n```\nActually use 30 instead."
        clauses = segment_user_turn(source)
        payload = {
            "actions": [{
                "type": "answer_pending",
                "answer": "30",
                "selected_value": "30",
                "source_evidence": clauses[1].text,
            }],
            "semantic_units": [
                _unit(clauses[0], 1, [], disposition="unresolved"),
                _unit(clauses[1], 2, [0]),
            ],
        }
        state = new_state("structured-pending-conflict", language="en")
        state["pending_question"] = {
            "id": "NETWORK_MAX_BANDWIDTH_GBPS",
            "group": "network",
            "field": "NETWORK_MAX_BANDWIDTH_GBPS",
            "kind": "manual_value",
            "manual_input_allowed": True,
            "validation": {"value_type": "positive_number"},
            "options": [],
        }

        reconciled = json.loads(_reconcile_structured_candidate_ownership(
            json.dumps(payload),
            clauses,
            state,
        ))

        self.assertEqual(reconciled, payload)

    def test_exact_structured_pending_assignment_stays_out_of_direct_pending_candidates(self) -> None:
        from agent.harness.intent import _action_queue_payload
        from agent.harness.state import new_state

        source = "NETWORK_MAX_BANDWIDTH_GBPS: 25"
        state = new_state("structured-pending-clarification", language="en")
        state["pending_question"] = {
            "id": "NETWORK_MAX_BANDWIDTH_GBPS",
            "group": "network",
            "field": "NETWORK_MAX_BANDWIDTH_GBPS",
            "kind": "manual_value",
            "manual_input_allowed": True,
            "validation": {"value_type": "positive_number"},
            "options": [],
        }

        request = _action_queue_payload(state, source)

        self.assertEqual(request["pending_typed_candidates"], [])
        self.assertEqual(
            request["structured_candidates"][0]["config_values"],
            {"NETWORK_MAX_BANDWIDTH_GBPS": "25"},
        )
        self.assertEqual(
            request["structured_candidates"][0]["config_values"],
            {"NETWORK_MAX_BANDWIDTH_GBPS": "25"},
        )

    def test_structured_candidate_for_another_group_does_not_answer_pending_field(self) -> None:
        from agent.harness.intent import _action_queue_payload
        from agent.harness.state import new_state

        state = new_state("structured-cross-group", language="en")
        state["pending_question"] = {
            "id": "CLOUD_ZONE",
            "group": "provider_deployment",
            "field": "CLOUD_ZONE",
            "kind": "manual_value",
            "manual_input_allowed": True,
            "validation": {"value_type": "scalar_token"},
            "options": [],
        }

        request = _action_queue_payload(
            state,
            "NETWORK_MAX_BANDWIDTH_GBPS: 25\nThen tell me what remains.",
        )

        self.assertEqual(request["pending_typed_candidates"], [])
        self.assertEqual(
            request["structured_candidates"][0]["config_values"],
            {"NETWORK_MAX_BANDWIDTH_GBPS": "25"},
        )
        self.assertEqual(len(request["clauses"]), 2)

    def test_case3_policy_does_not_reassign_removed_demands_to_handoff_owner(self) -> None:
        from agent.harness.intent import _apply_state_plan_policy
        from agent.harness.state import new_state

        state = new_state("case3-evidence-policy", language="en")
        state["chain_identity"] = {
            "case": "case3",
            "adapter_family": "unsupported",
        }
        payload = {
            "actions": [
                {"type": "rpc_catalog_command", "catalog_command": "enter"},
                {
                    "type": "rpc_catalog_command",
                    "catalog_command": "set_method",
                    "rpc_method": "getLatestBlock",
                },
                {
                    "type": "secondary_handoff_command",
                    "handoff_command": "append_evidence",
                    "handoff_evidence": (
                        'POST /rpc\n{"jsonrpc":"2.0","method":"getLatestBlock","params":[]}'
                    ),
                    "source_evidence": (
                        'POST /rpc\n{"jsonrpc":"2.0","method":"getLatestBlock","params":[]}'
                    ),
                },
            ],
            "semantic_units": [
                {
                    "unit_id": "unit-1",
                    "clause_id": "clause-1",
                    "source_text": "I only have the request example so far:",
                    "disposition": "action",
                    "action_indexes": [0],
                    "reason": "request framing",
                },
                {
                    "unit_id": "unit-2",
                    "clause_id": "clause-2",
                    "source_text": "POST /rpc",
                    "disposition": "action",
                    "action_indexes": [0],
                    "reason": "request transport",
                },
                {
                    "unit_id": "unit-3",
                    "clause_id": "clause-3",
                    "source_text": '{"jsonrpc":"2.0","method":"getLatestBlock","params":[]}',
                    "disposition": "action",
                    "action_indexes": [1, 2],
                    "reason": "request payload",
                },
            ],
        }

        normalized = json.loads(_apply_state_plan_policy(json.dumps(payload), state))

        self.assertEqual(
            [action["type"] for action in normalized["actions"]],
            ["secondary_handoff_command"],
        )
        self.assertEqual(normalized["semantic_units"][0]["action_indexes"], [])
        self.assertEqual(normalized["semantic_units"][0]["disposition"], "unresolved")
        self.assertEqual(normalized["semantic_units"][1]["action_indexes"], [])
        self.assertEqual(normalized["semantic_units"][1]["disposition"], "unresolved")
        self.assertEqual(normalized["semantic_units"][2]["action_indexes"], [0])
        self.assertEqual(normalized["semantic_units"][2]["disposition"], "action")
        self.assertEqual(
            normalized["actions"][0]["source_evidence"],
            'POST /rpc\n{"jsonrpc":"2.0","method":"getLatestBlock","params":[]}',
        )




if __name__ == "__main__":
    unittest.main()

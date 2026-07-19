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

    def test_pending_arbitration_preserves_an_atomic_config_review_proposal(self) -> None:
        from unittest.mock import Mock

        from agent.harness.intent import _adjudicate_pending_action_ownership
        from agent.harness.state import new_state

        source = "Review CLOUD_REGION=us-central1 and DATA_VOL_SIZE=1000 before applying them."
        clause = segment_user_turn(source)[0]
        state = new_state("pending-config-review")
        state["pending_question"] = {
            "id": "CLOUD_REGION",
            "group": "provider_deployment",
            "field": "CLOUD_REGION",
            "kind": "manual_value",
            "manual_input_allowed": True,
            "validation": {"value_type": "scalar_token"},
        }
        payload = {
            "actions": [{
                "type": "propose_config_values",
                "config_values": {"CLOUD_REGION": "us-central1", "DATA_VOL_SIZE": 1000},
                "unmapped_values": {},
                "source_format": "natural_language",
                "source_evidence": source,
            }],
            "semantic_units": [_unit(clause, 1, [0])],
        }
        provider = Mock()

        result_text, changed = _adjudicate_pending_action_ownership(
            provider,
            json.dumps(payload),
            state,
            source,
        )

        self.assertFalse(changed)
        self.assertEqual(json.loads(result_text), payload)
        provider.complete.assert_not_called()

    def test_literal_grounded_manual_answer_cannot_be_vetoed_by_reviewer(self) -> None:
        import json
        from unittest.mock import Mock

        from agent.harness.intent import _adjudicate_manual_pending_answers
        from agent.harness.state import new_state

        source = "Use 1000 GiB instead."
        clause = segment_user_turn(source)[0]
        state = new_state("literal-manual-admission")
        state["pending_question"] = {
            "id": "DATA_VOL_SIZE",
            "group": "ledger_disk",
            "field": "DATA_VOL_SIZE",
            "kind": "yes_no",
            "manual_input_allowed": True,
            "validation": {"value_type": "positive_number"},
            "options": [{"label": "Y", "value": "926"}, {"label": "N", "value": "__manual__"}],
        }
        payload = {
            "actions": [{
                "type": "answer_pending",
                "answer": "1000",
                "selected_value": None,
                "source_evidence": source,
            }],
            "semantic_units": [_unit(clause, 1, [0])],
        }
        provider = Mock()

        result, changed = _adjudicate_manual_pending_answers(
            provider,
            json.dumps(payload),
            state,
            source,
        )

        provider.complete.assert_not_called()
        self.assertFalse(changed)
        self.assertEqual(json.loads(result)["pending_answer_admissions"], [0])

    def test_literal_grounding_cannot_turn_a_question_into_a_manual_url_value(self) -> None:
        import json
        from unittest.mock import Mock

        from agent.harness.intent import _adjudicate_pending_action_ownership
        from agent.harness.state import new_state

        source = "What kind of synchronization endpoint do I need here?"
        state = new_state("manual-url-question", language="en")
        state["pending_question"] = {
            "id": "SYNC_OBSERVE_RPC_URL",
            "group": "sync_observe",
            "kind": "manual_value",
            "field": "SYNC_OBSERVE_RPC_URL",
            "prompt": "Enter the synchronization RPC endpoint.",
            "manual_input_allowed": True,
            "options": [],
            "validation": {"value_type": "url"},
        }
        payload = json.dumps({
            "actions": [{
                "type": "answer_pending",
                "answer": source,
                "source_evidence": source,
            }],
            "semantic_units": [{
                "unit_id": "unit-1",
                "source_text": source,
                "disposition": "action",
                "action_indexes": [0],
            }],
        })
        provider = Mock()
        provider.complete.return_value = type("Response", (), {"text": json.dumps({
            "reviews": [{
                "action_index": 0,
                "decision": "reject",
                "evidence_quote": "",
                "reason": "the source is a question, not a URL value",
            }],
        })})()

        result, changed = _adjudicate_pending_action_ownership(
            provider,
            payload,
            state,
            source,
        )

        self.assertTrue(changed)
        self.assertEqual(json.loads(result)["actions"], [])
        provider.complete.assert_called_once()

    def test_existing_target_returns_field_question_to_registered_consultation(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _adjudicate_target_mode_selection_requests
        from agent.harness.state import new_state

        source = "What kind of synchronization endpoint do I need here?"
        state = new_state("target-intake-reject", language="en")
        state["target_mode"] = "sync-observe"
        state["pending_question"] = {
            "id": "SYNC_OBSERVE_RPC_URL",
            "group": "sync_observe",
            "field": "SYNC_OBSERVE_RPC_URL",
            "prompt": "Enter the synchronization RPC endpoint.",
        }
        payload = json.dumps({
            "actions": [{
                "type": "request_target_mode_selection",
                "source_evidence": source,
            }],
            "semantic_units": [{
                "unit_id": "unit-1",
                "source_text": source,
                "disposition": "action",
                "action_indexes": [0],
            }],
        })
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "reviews": [{
                "action_index": 0,
                "decision": "consultation",
                "consultation_topic": "config_explanation",
                "consultation_subject": "SYNC_OBSERVE_RPC_URL",
                "evidence_quote": source,
                "reason": "the source asks about the active endpoint field",
            }],
        }))

        result, changed = _adjudicate_target_mode_selection_requests(
            provider,
            payload,
            state,
            source,
        )

        self.assertTrue(changed)
        self.assertEqual(json.loads(result)["actions"], [{
            "type": "answer_opening_question",
            "topic": "config_explanation",
            "subject": "SYNC_OBSERVE_RPC_URL",
            "source_evidence": source,
        }])
        self.assertEqual(json.loads(result)["consultation_admissions"], [0])

    def test_existing_target_rejects_unrelated_mode_intake_without_registered_replacement(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _adjudicate_target_mode_selection_requests
        from agent.harness.state import new_state

        source = "This sentence is unrelated."
        state = new_state("target-intake-unrelated", language="en")
        state["target_mode"] = "sync-observe"
        payload = json.dumps({
            "actions": [{
                "type": "request_target_mode_selection",
                "source_evidence": source,
            }],
        })
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "reviews": [{
                "action_index": 0,
                "decision": "reject",
                "consultation_topic": "",
                "consultation_subject": "",
                "evidence_quote": "",
                "reason": "unrelated content",
            }],
        }))

        result, changed = _adjudicate_target_mode_selection_requests(
            provider,
            payload,
            state,
            source,
        )

        self.assertTrue(changed)
        self.assertEqual(json.loads(result)["actions"], [])

    def test_existing_target_allows_explicit_replacement_mode_intake(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _adjudicate_target_mode_selection_requests
        from agent.harness.state import new_state

        source = "I want to reconsider sync-observe and choose another target mode."
        state = new_state("target-intake-replace", language="en")
        state["target_mode"] = "sync-observe"
        payload = json.dumps({
            "actions": [{
                "type": "request_target_mode_selection",
                "source_evidence": source,
            }],
        })
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "reviews": [{
                "action_index": 0,
                "decision": "admit",
                "consultation_topic": "",
                "consultation_subject": "",
                "evidence_quote": "reconsider sync-observe and choose another target mode",
                "reason": "the source explicitly requests replacement mode intake",
            }],
        }))

        result, changed = _adjudicate_target_mode_selection_requests(
            provider,
            payload,
            state,
            source,
        )

        self.assertFalse(changed)
        self.assertEqual(json.loads(result)["target_mode_selection_admissions"], [0])

    def test_manual_answer_not_present_in_source_still_requires_review(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _adjudicate_manual_pending_answers
        from agent.harness.state import new_state

        source = "Use the larger value instead."
        clause = segment_user_turn(source)[0]
        state = new_state("semantic-manual-review")
        state["pending_question"] = {
            "id": "DATA_VOL_SIZE",
            "group": "ledger_disk",
            "field": "DATA_VOL_SIZE",
            "kind": "yes_no",
            "manual_input_allowed": True,
            "validation": {"value_type": "positive_number"},
            "options": [{"label": "Y", "value": "926"}, {"label": "N", "value": "__manual__"}],
        }
        payload = {
            "actions": [{
                "type": "answer_pending",
                "answer": "1000",
                "selected_value": None,
                "source_evidence": source,
            }],
            "semantic_units": [_unit(clause, 1, [0])],
        }
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "reviews": [{
                "action_index": 0,
                "decision": "reject",
                "evidence_quote": "",
                "reason": "the normalized value is not source-grounded",
            }]
        }))

        result, _ = _adjudicate_manual_pending_answers(provider, json.dumps(payload), state, source)

        provider.complete.assert_called_once()
        self.assertEqual(json.loads(result)["semantic_units"][0]["disposition"], "unresolved")

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

    def test_context_requires_independent_admission_and_preserves_chain_action(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _validate_semantic_fulfillment
        from agent.harness.state import new_state

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
            "chain_selection_admissions": [0],
        }
        provider = Mock()
        provider.complete.side_effect = [
            SimpleNamespace(text=json.dumps({
                "unit_reviews": [{
                    "unit_id": "unit-1",
                    "complete": True,
                    "reason": "the chain request is preserved",
                }],
            })),
            SimpleNamespace(text=json.dumps({
                "context_reviews": [{
                    "unit_id": "unit-2",
                    "context_only": True,
                    "reason": "no current operation requested",
                }],
            })),
        ]
        state = new_state("context-admission", language="en")
        state["pending_question"] = {
            "id": "chain",
            "kind": "manual_value",
            "field": "BLOCKCHAIN_NODE",
            "options": [],
        }

        result = _validate_semantic_fulfillment(provider, json.dumps(payload), clauses, state)

        self.assertTrue(result.valid, result.errors)
        request_payload = json.loads(provider.complete.call_args_list[1].args[0].messages[1].content)
        self.assertEqual(request_payload["pending_question"]["id"], "chain")
        self.assertEqual(request_payload["context_reviews"][0]["unit_id"], "unit-2")

    def test_actionable_text_cannot_be_admitted_as_context(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _validate_semantic_fulfillment
        from agent.harness.state import new_state

        clauses = segment_user_turn("add eth_chainId now.")
        payload = {
            "actions": [],
            "semantic_units": [
                _unit(clauses[0], 1, [], disposition="context", reason="background"),
            ],
        }
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "context_reviews": [{
                "unit_id": "unit-1",
                "context_only": False,
                "reason": "this is a current custom RPC request",
            }],
        }))

        result = _validate_semantic_fulfillment(
            provider,
            json.dumps(payload),
            clauses,
            new_state("context-rejection", language="en"),
        )

        self.assertFalse(result.valid)
        self.assertIn("context semantic unit unit-1 admission failed", "\n".join(result.errors))

    def test_registry_recovery_can_preserve_context_without_synthesizing_an_action(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _recover_registry_bounded_semantic_actions
        from agent.harness.plan_coverage import PlanCoverageResult
        from agent.harness.state import new_state

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
                    disposition="unresolved",
                    reason="no current action",
                ),
            ],
        }
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "decisions": [{
                "unit_id": "unit-2",
                "disposition": "context",
                "group": "",
                "existing_action_index": None,
                "target_mode": "",
                "consultation_topic": "",
                "turn_local_action_type": "",
                "evidence_quote": "I may add another RPC call afterward.",
                "reason": "tentative future possibility",
            }],
        }))

        recovered_text, changed = _recover_registry_bounded_semantic_actions(
            provider,
            json.dumps(payload),
            clauses,
            new_state("context-recovery", language="en"),
            "; ".join(clause.text for clause in clauses),
            PlanCoverageResult(
                valid=False,
                errors=(),
                unresolved_clauses=(clauses[1].text,),
            ),
        )

        recovered = json.loads(recovered_text)
        self.assertTrue(changed)
        self.assertEqual(len(recovered["actions"]), 1)
        self.assertEqual(recovered["semantic_units"][1]["disposition"], "context")
        self.assertEqual(recovered["semantic_units"][1]["action_indexes"], [])

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
                "source_evidence": "simulated node",
            }],
            "semantic_units": [],
            "target_mode_selection_admissions": [0],
        })

        parsed = _parse_action_queue(
            _attach_semantic_admission_receipts(raw),
            trusted_metadata=True,
        )

        self.assertIs(parsed["actions"][0]["semantic_purpose_verified"], True)
        self.assertIs(parsed["actions"][0]["target_mode_semantic_verified"], True)

    def test_generic_benchmark_goal_fails_target_mode_explicitness(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _adjudicate_target_mode_actions

        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "reviews": [{
                "action_index": 0,
                "explicit": False,
                "evidence_quote": "",
                "unresolved_selection": False,
                "unresolved_quote": "",
                "reason": "benchmark settings do not select a target mode",
            }],
        }))
        text = "测试 BNB，用 mixed，QPS quick，并开启本地 Grafana"
        payload = json.dumps({
            "actions": [{
                "type": "choose_target_mode",
                "target_mode": "fake-node",
                "target_mode_explicit": True,
                "source_evidence": text,
            }],
        })

        result, changed = _adjudicate_target_mode_actions(provider, payload)

        self.assertTrue(changed)
        self.assertEqual(json.loads(result)["actions"], [])

    def test_one_literal_target_mode_is_deterministic_evidence(self) -> None:
        from agent.harness.input_values import target_mode_evidence_matches

        source = "I want to test BNB with fake-node and inspect supported methods"

        self.assertTrue(target_mode_evidence_matches("fake-node", "fake-node", source))

    def test_queued_workflow_goal_requires_its_own_target_mode_evidence(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _adjudicate_target_mode_actions

        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "reviews": [{
                "action_index": 0,
                "explicit": False,
                "evidence_quote": "",
                "unresolved_selection": False,
                "unresolved_quote": "",
                "reason": "returning to the review does not select a later target mode",
            }],
        }))
        payload = json.dumps({
            "actions": [{
                "type": "queue_workflow_goal",
                "target_mode": "fake-node",
                "goal": "return to candidate review",
                "source_evidence": "then return to confirm these values",
            }],
        })

        result, changed = _adjudicate_target_mode_actions(provider, payload)

        self.assertTrue(changed)
        self.assertEqual(json.loads(result)["actions"], [])

    def test_queued_workflow_goal_accepts_semantic_later_mode_evidence(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _adjudicate_target_mode_actions

        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "reviews": [{
                "action_index": 0,
                "explicit": True,
                "evidence_quote": "benchmark its real RPC capacity later",
                "unresolved_selection": False,
                "unresolved_quote": "",
                "reason": "the later real RPC benchmark explicitly selects real-node",
            }],
        }))
        payload = json.dumps({
            "actions": [{
                "type": "queue_workflow_goal",
                "target_mode": "real-node",
                "goal": "benchmark real RPC capacity later",
                "source_evidence": "benchmark its real RPC capacity later",
            }],
        })

        result, changed = _adjudicate_target_mode_actions(provider, payload)

        self.assertFalse(changed)
        self.assertEqual(json.loads(result)["target_mode_selection_admissions"], [0])

    def test_read_only_consultation_admits_direct_indirect_and_declarative_requests(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _adjudicate_consultation_actions

        sources = [
            "Which chains are supported?",
            "Could you tell me which chains are supported",
            "I want to see which chains are supported",
        ]
        payload = {
            "actions": [
                {
                    "type": "answer_opening_question",
                    "topic": "supported_chains",
                    "source_evidence": source,
                }
                for source in sources
            ],
            "semantic_units": [
                {
                    "unit_id": f"unit-{index}",
                    "source_text": source,
                    "disposition": "action",
                    "action_indexes": [index],
                }
                for index, source in enumerate(sources)
            ],
        }
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "reviews": [
                {
                    "action_index": index,
                    "present_consultation": True,
                    "evidence_quote": source,
                    "reason": "present read-only information request",
                }
                for index, source in enumerate(sources)
            ],
        }))

        result, changed = _adjudicate_consultation_actions(provider, json.dumps(payload))

        self.assertFalse(changed)
        self.assertEqual(json.loads(result)["consultation_admissions"], [0, 1, 2])

    def test_consultation_admission_repairs_wrong_topic_to_active_field_owner(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _adjudicate_consultation_actions
        from agent.harness.state import new_state

        source = "What kind of synchronization endpoint do I need here?"
        state = new_state("consultation-topic-owner", language="en")
        state["pending_question"] = {
            "id": "SYNC_OBSERVE_RPC_URL",
            "field": "SYNC_OBSERVE_RPC_URL",
            "prompt": "Enter the synchronization RPC endpoint.",
        }
        payload = json.dumps({
            "actions": [{
                "type": "answer_opening_question",
                "topic": "current_config",
                "source_evidence": source,
            }],
            "semantic_units": [{
                "unit_id": "unit-1",
                "source_text": source,
                "disposition": "action",
                "action_indexes": [0],
            }],
        })
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "reviews": [{
                "action_index": 0,
                "present_consultation": True,
                "topic_matches": False,
                "replacement_topic": "config_explanation",
                "replacement_subject": "SYNC_OBSERVE_RPC_URL",
                "evidence_quote": source,
                "reason": "the source asks what the active endpoint field requires",
            }],
        }))

        result, changed = _adjudicate_consultation_actions(provider, payload, state)

        self.assertTrue(changed)
        self.assertEqual(json.loads(result)["actions"], [{
            "type": "answer_opening_question",
            "topic": "config_explanation",
            "subject": "SYNC_OBSERVE_RPC_URL",
            "source_evidence": source,
        }])
        self.assertEqual(json.loads(result)["consultation_admissions"], [0])

    def test_future_only_consultation_candidate_is_removed_without_dropping_sibling_mode_intake(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _adjudicate_consultation_actions

        payload = {
            "actions": [
                {
                    "type": "answer_opening_question",
                    "topic": "supported_chains",
                    "source_evidence": "I may ask about supported chains later",
                },
                {
                    "type": "request_target_mode_selection",
                    "source_evidence": "I am not sure whether to use fake-node or real-node",
                },
            ],
            "semantic_units": [{
                "unit_id": "unit-1",
                "source_text": "I may ask about supported chains later",
                "disposition": "action",
                "action_indexes": [0],
            }],
            "target_mode_selection_admissions": [1],
        }
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "reviews": [{
                "action_index": 0,
                "present_consultation": False,
                "evidence_quote": "",
                "reason": "future-only possibility",
            }],
        }))

        result, changed = _adjudicate_consultation_actions(provider, json.dumps(payload))
        result_payload = json.loads(result)

        self.assertTrue(changed)
        self.assertEqual(
            [action["type"] for action in result_payload["actions"]],
            ["request_target_mode_selection"],
        )
        self.assertEqual(result_payload["target_mode_selection_admissions"], [0])

    def test_malformed_consultation_receipt_fails_closed_for_only_that_action(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _adjudicate_consultation_actions

        payload = {
            "actions": [{
                "type": "answer_opening_question",
                "topic": "supported_chains",
                "source_evidence": "show supported chains",
            }],
            "semantic_units": [{
                "unit_id": "unit-1",
                "source_text": "show supported chains",
                "disposition": "action",
                "action_indexes": [0],
            }],
        }
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text="{}")

        result, changed = _adjudicate_consultation_actions(provider, json.dumps(payload))

        self.assertTrue(changed)
        self.assertEqual(json.loads(result)["actions"], [])

    def test_mutually_exclusive_target_modes_require_semantic_adjudication(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.input_values import target_mode_evidence_matches
        from agent.harness.intent import _adjudicate_target_mode_actions

        source = "I am not sure whether to use fake-node or real-node"
        self.assertFalse(target_mode_evidence_matches("fake-node", "fake-node", source))

        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "reviews": [{
                "action_index": 0,
                "explicit": False,
                "evidence_quote": "",
                "unresolved_selection": True,
                "unresolved_quote": source,
                "reason": "the source explicitly leaves the target mode unresolved",
            }],
        }))
        payload = json.dumps({
            "actions": [{
                "type": "choose_target_mode",
                "target_mode": "fake-node",
                "target_mode_explicit": True,
                "source_evidence": "fake-node",
            }],
            "semantic_units": [{
                "unit_id": "unit-1",
                "clause_id": "clause-1",
                "source_text": source,
                "disposition": "action",
                "action_indexes": [0],
                "reason": "planner proposed a target mode",
            }],
        })

        result, changed = _adjudicate_target_mode_actions(provider, payload)

        self.assertTrue(changed)
        self.assertEqual(
            json.loads(result)["actions"],
            [{"type": "request_target_mode_selection", "source_evidence": source}],
        )
        provider.complete.assert_called_once()

    def test_holistic_target_mode_contract_rejects_ambiguous_current_and_future_split(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _adjudicate_target_mode_actions

        source = "Maybe fake-node or sync-observe; I am not sure which one."
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "decision": "unresolved",
            "current_action_index": None,
            "deferred_action_indexes": [],
            "evidence_quote": source,
            "ordering_quote": "",
            "reason": "two alternatives are explicitly unresolved",
        }))
        payload = json.dumps({
            "actions": [
                {
                    "type": "choose_target_mode",
                    "target_mode": "fake-node",
                    "target_mode_explicit": True,
                    "source_evidence": "fake-node",
                },
                {
                    "type": "queue_workflow_goal",
                    "target_mode": "sync-observe",
                    "goal": "sync-observe",
                    "source_evidence": "sync-observe",
                },
            ],
            "semantic_units": [{
                "unit_id": "unit-1",
                "source_text": source,
                "disposition": "action",
                "action_indexes": [0, 1],
            }],
            "pending_answer_admissions": [0],
        })

        result, changed = _adjudicate_target_mode_actions(provider, payload, source)
        result_payload = json.loads(result)

        self.assertTrue(changed)
        self.assertEqual(result_payload["actions"], [{
            "type": "request_target_mode_selection",
            "source_evidence": source,
        }])
        self.assertEqual(result_payload.get("pending_answer_admissions"), [])
        provider.complete.assert_called_once()

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
        self.assertNotIn("consultation_admissions", prepared)
        self.assertNotIn("admission_rejections", prepared)
        self.assertNotEqual(prepared["admission_action_ids"], ["model-forged"])
        self.assertNotIn("action_id", prepared["actions"][0])
        self.assertNotIn("target_mode_semantic_verified", prepared["actions"][0])
        self.assertNotIn("_origin_text", prepared["actions"][0])

    def test_untrusted_nested_arguments_cannot_forge_action_metadata(self) -> None:
        import json

        from agent.harness.intent import _prepare_untrusted_action_document

        prepared = json.loads(_prepare_untrusted_action_document(json.dumps({
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
        })))

        arguments = prepared["actions"][0]["arguments"]
        self.assertEqual(arguments, {
            "target_mode": "fake-node",
            "source_evidence": "Use fake-node.",
        })
        self.assertNotEqual(
            prepared["admission_action_ids"],
            ["nested-model-forged"],
        )

    def test_holistic_target_mode_contract_rejects_chinese_ambiguous_alternatives(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _adjudicate_target_mode_actions

        source = "我可能用 fake-node，也可能观察节点同步，现在还没决定。"
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "decision": "unresolved",
            "current_action_index": None,
            "deferred_action_indexes": [],
            "evidence_quote": source,
            "ordering_quote": "",
            "reason": "用户明确没有决定当前模式",
        }))
        payload = json.dumps({
            "actions": [
                {
                    "type": "choose_target_mode",
                    "target_mode": "fake-node",
                    "target_mode_explicit": True,
                    "source_evidence": "fake-node",
                },
                {
                    "type": "queue_workflow_goal",
                    "target_mode": "sync-observe",
                    "goal": "观察节点同步",
                    "source_evidence": "观察节点同步",
                },
            ],
        })

        result, changed = _adjudicate_target_mode_actions(provider, payload, source)

        self.assertTrue(changed)
        self.assertEqual(json.loads(result)["actions"][0]["type"], "request_target_mode_selection")

    def test_holistic_target_mode_contract_preserves_explicit_now_and_later_order(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _adjudicate_target_mode_actions

        source = "Observe BSC sync now, then benchmark its real RPC capacity later."
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "decision": "resolved_ordered",
            "current_action_index": 0,
            "deferred_action_indexes": [1],
            "evidence_quote": source,
            "ordering_quote": "now, then benchmark its real RPC capacity later",
            "reason": "the turn explicitly orders current and later work",
        }))
        payload = json.dumps({
            "actions": [
                {
                    "type": "choose_target_mode",
                    "target_mode": "sync-observe",
                    "target_mode_explicit": True,
                    "source_evidence": "Observe BSC sync now",
                },
                {
                    "type": "queue_workflow_goal",
                    "target_mode": "real-node",
                    "goal": "benchmark its real RPC capacity later",
                    "source_evidence": "benchmark its real RPC capacity later",
                },
            ],
        })

        result, changed = _adjudicate_target_mode_actions(provider, payload, source)

        self.assertFalse(changed)
        self.assertEqual(
            [item["type"] for item in json.loads(result)["actions"]],
            ["choose_target_mode", "queue_workflow_goal"],
        )
        self.assertEqual(json.loads(result)["target_mode_selection_admissions"], [0, 1])
        provider.complete.assert_called_once()

    def test_holistic_target_mode_contract_does_not_queue_a_comparison(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _adjudicate_target_mode_actions

        source = "Use real-node; I only mentioned fake-node for comparison."
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "decision": "resolved_ordered",
            "current_action_index": 0,
            "deferred_action_indexes": [],
            "evidence_quote": source,
            "ordering_quote": "",
            "reason": "real-node is selected and fake-node is only compared",
        }))
        payload = json.dumps({
            "actions": [
                {
                    "type": "choose_target_mode",
                    "target_mode": "real-node",
                    "target_mode_explicit": True,
                    "source_evidence": "Use real-node",
                },
                {
                    "type": "queue_workflow_goal",
                    "target_mode": "fake-node",
                    "goal": "fake-node",
                    "source_evidence": "fake-node for comparison",
                },
            ],
        })

        result, changed = _adjudicate_target_mode_actions(provider, payload, source)

        self.assertTrue(changed)
        self.assertEqual(
            [item["type"] for item in json.loads(result)["actions"]],
            ["choose_target_mode"],
        )
        self.assertEqual(json.loads(result)["target_mode_selection_admissions"], [0])
        provider.complete.assert_called_once()

    def test_holistic_target_mode_contract_invalid_verdict_fails_closed_without_clarification(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _adjudicate_target_mode_actions

        source = "Maybe fake-node or sync-observe; I am not sure which one."
        payload = json.dumps({
            "actions": [
                {
                    "type": "choose_target_mode",
                    "target_mode": "fake-node",
                    "source_evidence": "fake-node",
                },
                {
                    "type": "queue_workflow_goal",
                    "target_mode": "sync-observe",
                    "goal": "sync-observe",
                    "source_evidence": "sync-observe",
                },
            ],
            "semantic_units": [{
                "unit_id": "unit-1",
                "source_text": source,
                "disposition": "action",
                "action_indexes": [0, 1],
            }],
        })
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "decision": "unresolved",
            "current_action_index": None,
            "deferred_action_indexes": [],
            "evidence_quote": "not an exact quote",
            "ordering_quote": "",
        }))

        result, changed = _adjudicate_target_mode_actions(provider, payload, source)
        parsed = json.loads(result)

        self.assertTrue(changed)
        self.assertEqual(parsed["actions"], [])
        self.assertEqual(parsed["semantic_units"][0]["disposition"], "unresolved")
        self.assertNotIn("request_target_mode_selection", json.dumps(parsed))

    def test_holistic_target_mode_contract_missing_verdict_fails_closed(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _adjudicate_target_mode_actions

        source = "Use real-node now and fake-node later."
        payload = json.dumps({
            "actions": [
                {"type": "choose_target_mode", "target_mode": "real-node", "source_evidence": "real-node now"},
                {"type": "queue_workflow_goal", "target_mode": "fake-node", "goal": "later", "source_evidence": "fake-node later"},
            ],
        })
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text="{}")

        result, changed = _adjudicate_target_mode_actions(provider, payload, source)

        self.assertTrue(changed)
        self.assertEqual(json.loads(result)["actions"], [])

    def test_unresolved_semantic_answer_to_target_menu_becomes_mode_intake(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.domains.orientation import opening_question
        from agent.harness.intent import _adjudicate_pending_answer_actions
        from agent.harness.state import new_state

        source = "I am not sure whether to use fake-node or real-node"
        state = new_state("unit-thread", language="en")
        state["pending_question"] = opening_question(state)
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "reviews": [{
                "action_index": 0,
                "decision": "unresolved_target_mode",
                "evidence_quote": source,
                "reason": "the user explicitly leaves target mode unresolved",
            }],
        }))
        payload = json.dumps({
            "actions": [{
                "type": "answer_pending",
                "selected_value": None,
                "source_evidence": source,
            }],
            "semantic_units": [{
                "unit_id": "unit-1",
                "clause_id": "clause-1",
                "source_text": source,
                "disposition": "action",
                "action_indexes": [0],
                "reason": "semantic pending response",
            }],
        })

        result, changed = _adjudicate_pending_answer_actions(provider, payload, state, source)

        self.assertTrue(changed)
        self.assertEqual(
            json.loads(result)["actions"],
            [{"type": "request_target_mode_selection", "source_evidence": source}],
        )

    def test_semantic_pending_selection_uses_one_finite_review_decision(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _adjudicate_pending_answer_actions
        from agent.harness.state import new_state

        source = "Yes, use the proposed Solana template."
        state = new_state("unit-thread", language="en")
        state["pending_question"] = {
            "id": "unknown_chain_identity_confirm",
            "group": "chain_identity",
            "kind": "numbered_choice",
            "prompt": "Use the proposed chain?",
            "options": [
                {
                    "label": "Use `solana`",
                    "value": "confirm_known_chain",
                    "action": {"type": "answer_pending", "answer": "confirm_known_chain"},
                },
                {
                    "label": "No, re-enter chain name",
                    "value": "reenter_chain",
                    "action": {"type": "answer_pending", "answer": "reenter_chain"},
                },
                {
                    "label": "This is another real chain; choose protocol",
                    "value": "choose_protocol",
                    "action": {"type": "answer_pending", "answer": "choose_protocol"},
                },
            ],
        }
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "reviews": [{
                "action_index": 0,
                "decision": "select_option",
                "evidence_quote": source,
                "reason": "explicit confirmation",
            }],
        }))
        payload = json.dumps({
            "actions": [{
                "type": "answer_pending",
                "answer": "confirm_known_chain",
                "selected_value": "confirm_known_chain",
                "source_evidence": source,
            }],
            "semantic_units": [
                {
                    "unit_id": "unit-1",
                    "clause_id": "clause-1",
                    "source_text": "Yes, use the",
                    "disposition": "action",
                    "action_indexes": [0],
                },
                {
                    "unit_id": "unit-2",
                    "clause_id": "clause-1",
                    "source_text": "proposed Solana template.",
                    "disposition": "action",
                    "action_indexes": [0],
                },
            ],
        })

        result, changed = _adjudicate_pending_answer_actions(provider, payload, state, source)

        document = json.loads(result)
        self.assertTrue(changed)
        self.assertEqual(document["pending_answer_admissions"], [0])
        self.assertEqual(document["actions"][0]["selected_value"], "confirm_known_chain")

    def test_negative_pending_review_rechecks_natural_language_option_semantics(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _adjudicate_pending_answer_actions
        from agent.harness.state import new_state

        source = "Keep this profile's displayed defaults."
        state = new_state("unit-thread-natural-pending", language="en")
        state["pending_question"] = {
            "id": "qps_profile_confirm",
            "group": "qps_profile",
            "kind": "yes_no",
            "prompt": "Use the displayed intensive profile defaults?",
            "options": [
                {"label": "Y", "value": True, "action": {"type": "answer_pending"}},
                {"label": "N", "value": False, "action": {"type": "answer_pending"}},
            ],
        }
        provider = Mock()
        provider.complete.side_effect = [
            SimpleNamespace(text=json.dumps({
                "reviews": [{
                    "action_index": 0,
                    "decision": "reject",
                    "evidence_quote": "",
                    "reason": "the source does not literally say Y",
                }],
            })),
            SimpleNamespace(text=json.dumps({
                "reviews": [{
                    "action_index": 0,
                    "decision": "select_option",
                    "evidence_quote": source,
                    "reason": "the complete natural-language answer accepts the displayed defaults",
                }],
            })),
        ]
        payload = json.dumps({
            "actions": [{
                "type": "answer_pending",
                "answer": True,
                "selected_value": True,
                "source_evidence": source,
            }],
            "semantic_units": [{
                "unit_id": "unit-1",
                "clause_id": "clause-1",
                "source_text": source,
                "disposition": "action",
                "action_indexes": [0],
            }],
        })

        result, changed = _adjudicate_pending_answer_actions(provider, payload, state, source)

        self.assertTrue(changed)
        self.assertEqual(json.loads(result)["pending_answer_admissions"], [0])
        self.assertEqual(provider.complete.call_count, 2)

    def test_malformed_negative_pending_readjudication_remains_rejected(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _adjudicate_pending_answer_actions
        from agent.harness.state import new_state

        source = "Can you explain what keeping these defaults changes?"
        state = new_state("unit-thread-malformed-readjudication", language="en")
        state["pending_question"] = {
            "id": "qps_profile_confirm",
            "group": "qps_profile",
            "kind": "yes_no",
            "prompt": "Use the displayed defaults?",
            "options": [{"label": "Y", "value": True, "action": {"type": "answer_pending"}}],
        }
        provider = Mock()
        provider.complete.side_effect = [
            SimpleNamespace(text=json.dumps({
                "reviews": [{"action_index": 0, "decision": "reject", "evidence_quote": "", "reason": "help"}],
            })),
            SimpleNamespace(text=json.dumps({
                "reviews": [{"action_index": 0, "decision": "select_option", "evidence_quote": "not in source", "reason": "bad quote"}],
            })),
        ]
        payload = json.dumps({
            "actions": [{
                "type": "answer_pending",
                "answer": True,
                "selected_value": True,
                "source_evidence": source,
            }],
            "semantic_units": [{
                "unit_id": "unit-1",
                "clause_id": "clause-1",
                "source_text": source,
                "disposition": "action",
                "action_indexes": [0],
            }],
        })

        result, changed = _adjudicate_pending_answer_actions(provider, payload, state, source)

        self.assertTrue(changed)
        self.assertEqual(json.loads(result)["actions"], [])

    def test_semantic_pending_selection_materializes_registered_non_answer_effect(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _adjudicate_pending_answer_actions
        from agent.harness.state import new_state

        source = "I may just want to see which chains are supported."
        state = new_state("unit-thread-materialize-option", language="en")
        state["pending_question"] = {
            "id": "opening_next_action",
            "group": "opening",
            "kind": "numbered_choice",
            "prompt": "What would you like me to help with?",
            "options": [{
                "label": "Learn supported chains",
                "value": "info",
                "action": {"type": "answer_opening_question", "topic": "capabilities"},
            }],
        }
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "reviews": [{
                "action_index": 0,
                "decision": "select_option",
                "evidence_quote": "see which chains are supported",
                "reason": "read-only capability request",
            }],
        }))
        payload = json.dumps({
            "actions": [{
                "type": "answer_pending",
                "answer": "info",
                "selected_value": "info",
                "source_evidence": source,
            }],
            "semantic_units": [{
                "unit_id": "unit-1",
                "clause_id": "clause-1",
                "source_text": source,
                "disposition": "action",
                "action_indexes": [0],
            }],
        })

        result, changed = _adjudicate_pending_answer_actions(provider, payload, state, source)

        self.assertTrue(changed)
        self.assertEqual(json.loads(result)["actions"], [{
            "type": "answer_opening_question",
            "topic": "capabilities",
            "source_evidence": "see which chains are supported",
        }])

    def test_invalid_registered_pending_effect_remains_fail_closed(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _adjudicate_pending_answer_actions
        from agent.harness.state import new_state

        source = "Use the broken option."
        state = new_state("unit-thread-invalid-option-effect", language="en")
        state["pending_question"] = {
            "id": "invalid_option",
            "group": "opening",
            "kind": "numbered_choice",
            "prompt": "Choose.",
            "options": [{
                "label": "Broken",
                "value": "broken",
                "action": {"type": "choose_target_mode", "target_mode": "not-a-mode"},
            }],
        }
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "reviews": [{
                "action_index": 0,
                "decision": "select_option",
                "evidence_quote": source,
                "reason": "selected",
            }],
        }))
        payload = json.dumps({
            "actions": [{
                "type": "answer_pending",
                "answer": "broken",
                "selected_value": "broken",
                "source_evidence": source,
            }],
            "semantic_units": [{
                "unit_id": "unit-1",
                "clause_id": "clause-1",
                "source_text": source,
                "disposition": "action",
                "action_indexes": [0],
            }],
        })

        result, changed = _adjudicate_pending_answer_actions(provider, payload, state, source)

        self.assertTrue(changed)
        self.assertEqual(json.loads(result)["actions"], [])

    def test_malformed_pending_review_is_retried_against_the_same_contract(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _adjudicate_pending_answer_actions
        from agent.harness.state import new_state

        source = "No, let me type the chain again."
        state = new_state("unit-thread", language="en")
        state["pending_question"] = {
            "id": "unknown_chain_identity_confirm",
            "group": "chain_identity",
            "kind": "numbered_choice",
            "prompt": "Use the proposed chain?",
            "options": [{
                "label": "No, re-enter chain name",
                "value": "reenter_chain",
                "action": {"type": "answer_pending", "answer": "reenter_chain"},
            }],
        }
        provider = Mock()
        provider.complete.side_effect = [
            SimpleNamespace(text=json.dumps({
                "reviews": [{"action_index": 0, "decision": "select_option", "evidence_quote": ""}],
            })),
            SimpleNamespace(text=json.dumps({
                "reviews": [{
                    "action_index": 0,
                    "decision": "select_option",
                    "evidence_quote": source,
                    "reason": "explicit rejection of the proposed chain",
                }],
            })),
        ]
        payload = json.dumps({
            "actions": [{
                "type": "answer_pending",
                "answer": "reenter_chain",
                "selected_value": "reenter_chain",
                "source_evidence": source,
            }],
            "semantic_units": [{
                "unit_id": "unit-1",
                "clause_id": "clause-1",
                "source_text": source,
                "disposition": "action",
                "action_indexes": [0],
            }],
        })

        result, changed = _adjudicate_pending_answer_actions(provider, payload, state, source)

        self.assertTrue(changed)
        self.assertEqual(json.loads(result)["pending_answer_admissions"], [0])
        self.assertEqual(provider.complete.call_count, 2)

    def test_ambiguous_pending_reply_does_not_consume_registered_option(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _adjudicate_pending_answer_actions
        from agent.harness.state import new_state

        source = "Can you explain what this choice changes?"
        state = new_state("unit-thread", language="en")
        state["pending_question"] = {
            "id": "unknown_chain_identity_confirm",
            "group": "chain_identity",
            "kind": "numbered_choice",
            "prompt": "Use the proposed chain?",
            "options": [{
                "label": "Use `solana`",
                "value": "confirm_known_chain",
                "action": {"type": "answer_pending", "answer": "confirm_known_chain"},
            }],
        }
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "reviews": [{
                "action_index": 0,
                "decision": "reject",
                "evidence_quote": "",
                "reason": "read-only explanation request",
            }],
        }))
        payload = json.dumps({
            "actions": [{
                "type": "answer_pending",
                "answer": "confirm_known_chain",
                "selected_value": "confirm_known_chain",
                "source_evidence": source,
            }],
            "semantic_units": [{
                "unit_id": "unit-1",
                "clause_id": "clause-1",
                "source_text": source,
                "disposition": "action",
                "action_indexes": [0],
            }],
        })

        result, changed = _adjudicate_pending_answer_actions(provider, payload, state, source)

        self.assertTrue(changed)
        self.assertEqual(json.loads(result)["actions"], [])

    def test_same_group_owner_restatement_can_bind_to_displayed_pending_option(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _reconcile_pending_owner_mutations
        from agent.harness.state import new_state

        source = "Those quick-profile defaults are fine; keep all four values as shown."
        state = new_state("unit-thread", language="en")
        state["pending_question"] = {
            "id": "profile-defaults",
            "group": "qps_profile",
            "kind": "yes_no",
            "prompt": "Use the displayed profile defaults?",
            "accepted_action_types": ["answer_pending"],
            "options": [
                {"label": "Keep defaults", "value": True},
                {"label": "Adjust values", "value": False},
            ],
        }
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "reviews": [{
                "action_index": 0,
                "decision": "select_pending_option",
                "selected_option_value": True,
                "evidence_quote": source,
                "reason": "the source accepts the displayed defaults",
            }],
        }))
        payload = json.dumps({
            "actions": [{
                "type": "set_qps_mode",
                "qps_mode": "quick",
                "mutation_explicit": True,
                "source_evidence": source,
            }],
            "semantic_units": [{
                "unit_id": "unit-1",
                "clause_id": "clause-1",
                "source_text": source,
                "disposition": "action",
                "action_indexes": [0],
            }],
        })

        result, changed = _reconcile_pending_owner_mutations(provider, payload, state, source)

        self.assertTrue(changed)
        self.assertEqual(json.loads(result)["actions"], [{
            "type": "answer_pending",
            "answer": True,
            "selected_value": True,
            "source_evidence": source,
        }])

    def test_explicit_same_group_owner_change_is_not_rebound_to_pending_answer(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _reconcile_pending_owner_mutations
        from agent.harness.state import new_state

        source = "Switch from quick to standard instead."
        state = new_state("unit-thread", language="en")
        state["pending_question"] = {
            "id": "profile-defaults",
            "group": "qps_profile",
            "kind": "yes_no",
            "prompt": "Use the displayed profile defaults?",
            "accepted_action_types": ["answer_pending"],
            "options": [
                {"label": "Keep defaults", "value": True},
                {"label": "Adjust values", "value": False},
            ],
        }
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "reviews": [{
                "action_index": 0,
                "decision": "keep_owner_mutation",
                "selected_option_value": None,
                "evidence_quote": source,
                "reason": "the source requests a different profile",
            }],
        }))
        action = {
            "type": "set_qps_mode",
            "qps_mode": "standard",
            "mutation_explicit": True,
            "source_evidence": source,
        }
        payload = json.dumps({
            "actions": [action],
            "semantic_units": [{
                "unit_id": "unit-1",
                "clause_id": "clause-1",
                "source_text": source,
                "disposition": "action",
                "action_indexes": [0],
            }],
        })

        result, changed = _reconcile_pending_owner_mutations(provider, payload, state, source)

        self.assertFalse(changed)
        self.assertEqual(json.loads(result)["actions"], [action])

    def test_pending_owner_reconciliation_preserves_explicit_other_group_detour(self) -> None:
        import json
        from unittest.mock import Mock

        from agent.harness.intent import _reconcile_pending_owner_mutations
        from agent.harness.state import new_state

        source = "Disable observability."
        state = new_state("unit-thread", language="en")
        state["pending_question"] = {
            "id": "profile-defaults",
            "group": "qps_profile",
            "kind": "yes_no",
            "prompt": "Use the displayed profile defaults?",
            "accepted_action_types": ["answer_pending"],
            "options": [
                {"label": "Keep defaults", "value": True},
                {"label": "Adjust values", "value": False},
            ],
        }
        provider = Mock()
        payload = json.dumps({
            "actions": [{
                "type": "set_observability",
                "observability": "disabled",
                "mutation_explicit": True,
                "source_evidence": source,
            }],
            "semantic_units": [],
        })

        result, changed = _reconcile_pending_owner_mutations(provider, payload, state, source)

        self.assertFalse(changed)
        self.assertEqual(result, payload)
        provider.complete.assert_not_called()

    def test_pending_owner_reconciliation_preserves_independent_compound_demands(self) -> None:
        import json
        from unittest.mock import Mock

        from agent.harness.intent import _reconcile_pending_owner_mutations
        from agent.harness.state import new_state

        source = "Use fake-node for BNB with mixed, quick, and local observability."
        state = new_state("unit-thread", language="en")
        state["pending_question"] = {
            "id": "opening_next_action",
            "group": "opening",
            "kind": "numbered_choice",
            "prompt": "Choose what to do.",
            "accepted_action_types": ["choose_target_mode", "answer_opening_question"],
            "options": [{
                "id": "fake",
                "label": "Start fake-node benchmark",
                "value": "fake-node",
                "action": {
                    "type": "choose_target_mode",
                    "target_mode": "fake-node",
                    "target_mode_explicit": True,
                },
            }],
        }
        actions = [
            {
                "type": "choose_target_mode",
                "target_mode": "fake-node",
                "target_mode_explicit": True,
                "source_evidence": "fake-node",
            },
            {
                "type": "choose_chain",
                "chain_text": "BNB",
                "source_evidence": "BNB",
            },
            {
                "type": "set_rpc_mode",
                "rpc_mode": "mixed",
                "mutation_explicit": True,
                "source_evidence": "mixed",
            },
            {
                "type": "set_qps_mode",
                "qps_mode": "quick",
                "mutation_explicit": True,
                "source_evidence": "quick",
            },
            {
                "type": "set_observability",
                "observability_mode": "local",
                "mutation_explicit": True,
                "source_evidence": "local observability",
            },
        ]
        payload = json.dumps({"actions": actions, "semantic_units": []})
        from types import SimpleNamespace

        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "reviews": [{
                "action_index": 0,
                "decision": "select_pending_option",
                "selected_option_value": "fake-node",
                "evidence_quote": "fake-node",
                "reason": "selects the displayed target mode",
            }],
        }))

        result, changed = _reconcile_pending_owner_mutations(provider, payload, state, source)

        self.assertTrue(changed)
        self.assertEqual(json.loads(result)["actions"], actions)
        self.assertEqual(json.loads(result)["pending_answer_admissions"], [0])

    def test_pending_owner_rechecks_negative_exact_declared_effect(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _reconcile_pending_owner_mutations
        from agent.harness.state import new_state

        source = (
            "Plans changed: don't generate benchmark traffic. "
            "I only want to watch an already-running real node catch up to chain head."
        )
        state = new_state("unit-thread", language="en")
        state["pending_question"] = {
            "id": "target_mode_select",
            "group": "target_mode",
            "kind": "numbered_choice",
            "prompt": "Choose the target mode.",
            "accepted_action_types": ["choose_target_mode"],
            "options": [{
                "id": "sync-observe",
                "label": "sync-observe",
                "value": "sync-observe",
                "action": {
                    "type": "choose_target_mode",
                    "target_mode": "sync-observe",
                    "target_mode_explicit": True,
                },
            }],
        }
        action = {
            "type": "choose_target_mode",
            "target_mode": "sync-observe",
            "target_mode_explicit": True,
            "source_evidence": "watch an already-running real node catch up to chain head",
        }
        payload = json.dumps({
            "actions": [action],
            "semantic_units": [{
                "unit_id": "unit-1",
                "clause_id": "clause-1",
                "source_text": source,
                "disposition": "action",
                "action_indexes": [0],
            }],
        })
        provider = Mock()
        provider.complete.side_effect = [
            SimpleNamespace(text=json.dumps({
                "reviews": [{
                    "action_index": 0,
                    "decision": "reject",
                    "selected_option_value": None,
                    "evidence_quote": "",
                    "reason": "first reviewer missed the natural-language meaning",
                }],
            })),
            SimpleNamespace(text=json.dumps({
                "reviews": [{
                    "action_index": 0,
                    "decision": "select_pending_option",
                    "selected_option_value": "sync-observe",
                    "evidence_quote": "watch an already-running real node catch up to chain head",
                    "reason": "the requested no-load catch-up observation selects sync-observe",
                }],
            })),
        ]

        result, changed = _reconcile_pending_owner_mutations(provider, payload, state, source)

        self.assertTrue(changed)
        self.assertEqual(json.loads(result)["actions"], [action])
        self.assertEqual(json.loads(result)["pending_answer_admissions"], [0])
        self.assertEqual(provider.complete.call_count, 2)

    def test_pending_owner_second_review_still_rejects_unrelated_exact_effect(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _reconcile_pending_owner_mutations
        from agent.harness.state import new_state

        source = "Before I decide, explain whether this mode changes the report format."
        state = new_state("unit-thread", language="en")
        state["pending_question"] = {
            "id": "target_mode_select",
            "group": "target_mode",
            "kind": "numbered_choice",
            "prompt": "Choose the target mode.",
            "accepted_action_types": ["choose_target_mode"],
            "options": [{
                "id": "sync-observe",
                "label": "sync-observe",
                "value": "sync-observe",
                "action": {
                    "type": "choose_target_mode",
                    "target_mode": "sync-observe",
                    "target_mode_explicit": True,
                },
            }],
        }
        payload = json.dumps({
            "actions": [{
                "type": "choose_target_mode",
                "target_mode": "sync-observe",
                "target_mode_explicit": True,
                "source_evidence": source,
            }],
            "semantic_units": [{
                "unit_id": "unit-1",
                "clause_id": "clause-1",
                "source_text": source,
                "disposition": "action",
                "action_indexes": [0],
            }],
        })
        rejection = SimpleNamespace(text=json.dumps({
            "reviews": [{
                "action_index": 0,
                "decision": "reject",
                "selected_option_value": None,
                "evidence_quote": "",
                "reason": "the source asks a question and makes no selection",
            }],
        }))
        provider = Mock()
        provider.complete.side_effect = [rejection, rejection]

        result, changed = _reconcile_pending_owner_mutations(provider, payload, state, source)

        self.assertTrue(changed)
        self.assertEqual(json.loads(result)["actions"], [])
        self.assertNotIn("pending_answer_admissions", json.loads(result))
        self.assertEqual(provider.complete.call_count, 2)

    def test_pending_option_contract_replaces_similar_cross_group_navigation(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _reconcile_pending_owner_mutations
        from agent.harness.state import new_state

        source = "Take me to the choice for changing the chain or target mode."
        state = new_state("unit-thread", language="en")
        state["pending_question"] = {
            "id": "workload_confirm",
            "group": "workload_rpc",
            "kind": "numbered_choice",
            "prompt": "Review the current workload.",
            "accepted_action_types": [
                "answer_pending",
                "request_target_change",
                "rpc_catalog_command",
                "use_default_workload",
            ],
            "options": [
                {
                    "id": "change_target",
                    "label": "Change chain or target mode",
                    "value": "change_target",
                    "action": {"type": "request_target_change"},
                },
            ],
        }
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "reviews": [{
                "action_index": 0,
                "decision": "select_pending_option",
                "selected_option_value": "change_target",
                "evidence_quote": source,
                "reason": "the source selects the displayed typed option",
            }],
        }))
        payload = json.dumps({
            "actions": [{
                "type": "change_group",
                "group": "target_mode",
                "navigation_explicit": True,
                "source_evidence": source,
            }],
            "semantic_units": [{
                "unit_id": "unit-1",
                "clause_id": "clause-1",
                "source_text": source,
                "disposition": "action",
                "action_indexes": [0],
            }],
        })

        result, changed = _reconcile_pending_owner_mutations(provider, payload, state, source)

        self.assertTrue(changed)
        self.assertEqual(json.loads(result)["actions"], [{
            "type": "request_target_change",
        }])

    def test_pending_option_contract_preserves_explicit_cross_group_detour(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _reconcile_pending_owner_mutations
        from agent.harness.state import new_state

        source = "Do not choose that menu option; open target-mode settings directly."
        state = new_state("unit-thread", language="en")
        state["pending_question"] = {
            "id": "workload_confirm",
            "group": "workload_rpc",
            "kind": "numbered_choice",
            "prompt": "Review the current workload.",
            "accepted_action_types": ["answer_pending", "request_target_change"],
            "options": [{
                "id": "change_target",
                "label": "Change chain or target mode",
                "value": "change_target",
                "action": {"type": "request_target_change"},
            }],
        }
        action = {
            "type": "change_group",
            "group": "target_mode",
            "navigation_explicit": True,
            "source_evidence": source,
        }
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "reviews": [{
                "action_index": 0,
                "decision": "keep_owner_mutation",
                "selected_option_value": None,
                "evidence_quote": source,
                "reason": "the source rejects the option and requests a direct detour",
            }],
        }))
        payload = json.dumps({
            "actions": [action],
            "semantic_units": [{
                "unit_id": "unit-1",
                "clause_id": "clause-1",
                "source_text": source,
                "disposition": "action",
                "action_indexes": [0],
            }],
        })

        result, changed = _reconcile_pending_owner_mutations(provider, payload, state, source)

        self.assertFalse(changed)
        self.assertEqual(json.loads(result)["actions"], [action])

    def test_declared_typed_option_effect_receives_pending_admission(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _adjudicate_pending_answer_actions
        from agent.harness.state import new_state

        source = "Take me to the choice for changing the chain or target mode."
        state = new_state("unit-thread", language="en")
        state["pending_question"] = {
            "id": "workload_confirm",
            "group": "workload_rpc",
            "kind": "numbered_choice",
            "prompt": "Review the current workload.",
            "accepted_action_types": ["answer_pending", "request_target_change"],
            "options": [{
                "id": "change_target",
                "label": "Change chain or target mode",
                "value": "change_target",
                "action": {"type": "request_target_change"},
            }],
        }
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "reviews": [{
                "action_index": 0,
                "decision": "select_option",
                "evidence_quote": source,
                "reason": "the source selects the declared option",
            }],
        }))
        payload = json.dumps({
            "actions": [{
                "type": "request_target_change",
                "source_evidence": source,
            }],
            "semantic_units": [{
                "unit_id": "unit-1",
                "clause_id": "clause-1",
                "source_text": source,
                "disposition": "action",
                "action_indexes": [0],
            }],
        })

        result, changed = _adjudicate_pending_answer_actions(provider, payload, state, source)

        self.assertTrue(changed)
        document = json.loads(result)
        self.assertEqual(document["actions"], [{"type": "request_target_change"}])
        self.assertEqual(document["pending_answer_admissions"], [0])

    def test_consultation_option_remains_owned_by_consultation_admission(self) -> None:
        import json
        from unittest.mock import Mock

        from agent.harness.intent import _adjudicate_pending_answer_actions
        from agent.harness.state import new_state

        source = "I need to know which chains and RPC methods are supported first."
        state = new_state("unit-thread", language="en")
        state["pending_question"] = {
            "id": "opening_next_action",
            "group": "opening",
            "kind": "numbered_choice",
            "prompt": "What would you like me to help with?",
            "options": [{
                "id": "info",
                "label": "Learn supported chains and RPC methods",
                "value": "info",
                "action": {"type": "answer_opening_question", "topic": "capabilities"},
            }],
        }
        payload = json.dumps({
            "actions": [{
                "type": "answer_opening_question",
                "topic": "capabilities",
                "source_evidence": source,
            }],
            "semantic_units": [],
        })
        provider = Mock()

        result, changed = _adjudicate_pending_answer_actions(provider, payload, state, source)

        self.assertFalse(changed)
        self.assertEqual(result, payload)
        provider.complete.assert_not_called()

    def test_unsupported_same_group_mutation_is_removed_fail_closed(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _reconcile_pending_owner_mutations
        from agent.harness.state import new_state

        source = "What do those defaults mean?"
        state = new_state("unit-thread", language="en")
        state["pending_question"] = {
            "id": "profile-defaults",
            "group": "qps_profile",
            "kind": "yes_no",
            "prompt": "Use the displayed profile defaults?",
            "accepted_action_types": ["answer_pending"],
            "options": [
                {"label": "Keep defaults", "value": True},
                {"label": "Adjust values", "value": False},
            ],
        }
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "reviews": [{
                "action_index": 0,
                "decision": "reject",
                "selected_option_value": None,
                "evidence_quote": "",
                "reason": "the source asks for an explanation",
            }],
        }))
        payload = json.dumps({
            "actions": [{
                "type": "set_qps_mode",
                "qps_mode": "quick",
                "mutation_explicit": True,
                "source_evidence": source,
            }],
            "semantic_units": [{
                "unit_id": "unit-1",
                "clause_id": "clause-1",
                "source_text": source,
                "disposition": "action",
                "action_indexes": [0],
            }],
        })

        result, changed = _reconcile_pending_owner_mutations(provider, payload, state, source)

        self.assertTrue(changed)
        document = json.loads(result)
        self.assertEqual(document["actions"], [])
        self.assertEqual(document["semantic_units"][0]["disposition"], "unresolved")

    def test_manual_pending_direct_prose_answer_is_admitted(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _adjudicate_manual_pending_answers
        from agent.harness.state import new_state

        source = "Use asia-east1 for this run."
        state = new_state("unit-thread", language="en")
        state["pending_question"] = {
            "id": "CLOUD_REGION",
            "group": "provider_deployment",
            "kind": "text",
            "field": "CLOUD_REGION",
            "prompt": "Enter CLOUD_REGION.",
            "manual_input_allowed": True,
            "options": [],
        }
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "reviews": [{
                "action_index": 0,
                "decision": "direct_answer",
                "evidence_quote": "asia-east1",
                "reason": "direct value for the displayed field",
            }],
        }))
        payload = json.dumps({
            "actions": [{"type": "answer_pending", "answer": "asia-east1"}],
            "semantic_units": [{
                "unit_id": "unit-1",
                "clause_id": "clause-1",
                "source_text": source,
                "disposition": "action",
                "action_indexes": [0],
            }],
        })

        result, _changed = _adjudicate_manual_pending_answers(provider, payload, state, source)

        self.assertEqual(json.loads(result)["pending_answer_admissions"], [0])

    def test_manual_pending_navigation_prose_becomes_generic_resume(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _adjudicate_manual_pending_answers
        from agent.harness.state import new_state

        source = "回到 benchmark 设置"
        state = new_state("unit-thread", language="zh")
        state["pending_question"] = {
            "id": "chain",
            "group": "chain_identity",
            "kind": "chain",
            "field": "chain",
            "prompt": "你想测试哪条链？",
            "manual_input_allowed": True,
            "options": [],
        }
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "reviews": [{
                "action_index": 0,
                "decision": "generic_resume",
                "evidence_quote": source,
                "reason": "resume request rather than a chain value",
            }],
        }, ensure_ascii=False))
        payload = json.dumps({
            "actions": [{"type": "answer_pending", "answer": "fake-node"}],
            "semantic_units": [{
                "unit_id": "unit-1",
                "clause_id": "clause-1",
                "source_text": source,
                "disposition": "action",
                "action_indexes": [0],
            }],
        }, ensure_ascii=False)

        result, changed = _adjudicate_manual_pending_answers(provider, payload, state, source)

        self.assertTrue(changed)
        self.assertEqual(json.loads(result)["actions"], [{
            "type": "resume_current_flow",
            "source_evidence": source,
        }])

    def test_manual_pending_question_is_rejected_as_a_field_answer(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _adjudicate_manual_pending_answers
        from agent.harness.state import new_state

        source = "What does a cloud region change?"
        state = new_state("unit-thread", language="en")
        state["pending_question"] = {
            "id": "CLOUD_REGION",
            "group": "provider_deployment",
            "kind": "text",
            "field": "CLOUD_REGION",
            "prompt": "Enter CLOUD_REGION.",
            "manual_input_allowed": True,
            "options": [],
        }
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "reviews": [{
                "action_index": 0,
                "decision": "reject",
                "evidence_quote": "",
                "reason": "read-only question",
            }],
        }))
        payload = json.dumps({
            "actions": [{"type": "answer_pending", "answer": "us-east-1"}],
            "semantic_units": [{
                "unit_id": "unit-1",
                "clause_id": "clause-1",
                "source_text": source,
                "disposition": "action",
                "action_indexes": [0],
            }],
        })

        result, changed = _adjudicate_manual_pending_answers(provider, payload, state, source)

        self.assertTrue(changed)
        self.assertEqual(json.loads(result)["actions"], [])

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

    def test_owner_receipt_does_not_skip_compound_unit_inventory(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _challenge_registry_action_inventory

        cases = (
            (
                "consultation_admissions",
                {"type": "answer_opening_question", "topic": "capabilities"},
            ),
            (
                "pending_answer_admissions",
                {"type": "answer_pending", "answer": "bsc", "selected_value": "bsc"},
            ),
        )
        for receipt_key, action in cases:
            with self.subTest(receipt_key=receipt_key):
                provider = Mock()
                provider.complete.return_value = SimpleNamespace(text="{}")
                payload = json.dumps({
                    "actions": [action],
                    receipt_key: [0],
                    "semantic_units": [{
                        "unit_id": "unit-compound",
                        "source_text": "Use BSC, and also change the QPS profile.",
                        "disposition": "action",
                        "action_indexes": [0],
                    }],
                })

                result, changed, incomplete = _challenge_registry_action_inventory(
                    provider,
                    payload,
                )

                self.assertFalse(changed)
                self.assertEqual(result, payload)
                self.assertEqual(incomplete, ("unit-compound",))
                self.assertEqual(provider.complete.call_count, 2)

    def test_generic_group_navigation_becomes_resume_current_flow(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _adjudicate_group_navigation_actions

        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "reviews": [{
                "action_index": 0,
                "destination_named": False,
                "destination_quote": "",
                "generic_resume": True,
                "resume_quote": "回到 benchmark 设置",
                "reason": "generic workflow resume",
            }],
        }))
        payload = {
            "actions": [{
                "type": "change_group",
                "group": "qps_profile",
                "navigation_explicit": True,
                "source_evidence": "回到 benchmark 设置",
            }],
            "semantic_units": [{
                "unit_id": "unit-1",
                "clause_id": "clause-1",
                "source_text": "回到 benchmark 设置",
                "disposition": "action",
                "action_indexes": [0],
            }],
        }

        result_text, changed = _adjudicate_group_navigation_actions(
            provider,
            json.dumps(payload),
            {"pending_question": {"id": "chain"}},
        )
        result = json.loads(result_text)

        self.assertTrue(changed)
        self.assertEqual(result["actions"], [{
            "type": "resume_current_flow",
            "source_evidence": "回到 benchmark 设置",
        }])
        self.assertEqual(result["semantic_units"][0]["action_indexes"], [0])

    def test_navigation_receipt_suppresses_only_same_scope_inventory_reinterpretation(self) -> None:
        from agent.harness.intent import _duplicates_admitted_group_navigation

        represented = [{
            "group_navigation_admission": {
                "group": "qps_profile",
                "source_evidence": "configure QPS first before the rest",
                "destination_quote": "QPS",
                "specific_change_requested": False,
            },
        }]
        self.assertTrue(_duplicates_admitted_group_navigation({
            "proposed_group": "qps_profile",
            "proposed_evidence_quote": "configure QPS",
            "represented_actions": represented,
        }))
        self.assertFalse(_duplicates_admitted_group_navigation({
            "proposed_group": "observability",
            "proposed_evidence_quote": "before the rest",
            "represented_actions": represented,
        }))

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

    def test_navigation_receipt_bypasses_duplicate_generic_purpose_review(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _validate_semantic_fulfillment
        from agent.harness.plan_coverage import segment_user_turn
        from agent.harness.state import new_state

        text = "configure QPS first"
        clauses = segment_user_turn(text)
        payload = {
            "actions": [{
                "type": "change_group",
                "group": "qps_profile",
                "navigation_explicit": True,
                "source_evidence": text,
            }],
            "semantic_units": [_unit(clauses[0], 1, [0])],
            "group_navigation_admissions": [{
                "action_index": 0,
                "group": "qps_profile",
                "source_evidence": text,
                "destination_quote": "QPS",
            }],
        }
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "unit_reviews": [{
                "unit_id": "unit-1",
                "complete": True,
                "missing_demand_quote": "",
                "reason": "the source requests only navigation",
            }],
        }))

        result = _validate_semantic_fulfillment(
            provider,
            json.dumps(payload),
            clauses,
            new_state("navigation-receipt", language="en"),
        )

        self.assertTrue(result.valid, result.errors)
        provider.complete.assert_called_once()

    def test_navigation_receipt_does_not_hide_a_concrete_destination_value(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _validate_semantic_fulfillment
        from agent.harness.plan_coverage import segment_user_turn
        from agent.harness.state import new_state

        text = "Pause this paste and let me configure quick QPS first."
        clauses = segment_user_turn(text)
        payload = {
            "actions": [{
                "type": "change_group",
                "group": "qps_profile",
                "navigation_explicit": True,
                "source_evidence": text,
            }],
            "semantic_units": [_unit(clauses[0], 1, [0])],
            "group_navigation_admissions": [{
                "action_index": 0,
                "group": "qps_profile",
                "source_evidence": text,
                "destination_quote": "QPS",
            }],
        }
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "unit_reviews": [{
                "unit_id": "unit-1",
                "complete": False,
                "missing_demand_quote": "quick",
                "reason": "the concrete QPS mode is not represented",
            }],
        }))

        result = _validate_semantic_fulfillment(
            provider,
            json.dumps(payload),
            clauses,
            new_state("navigation-value-omission", language="en"),
        )

        self.assertFalse(result.valid)
        self.assertIn("semantic unit unit-1 fulfilment failed", "\n".join(result.errors))

    def test_explicit_group_navigation_is_retained(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _adjudicate_group_navigation_actions

        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "reviews": [{
                "action_index": 0,
                "destination_named": True,
                "destination_quote": "QPS settings",
                "generic_resume": False,
                "resume_quote": "",
                "reason": "source names the QPS area",
            }],
        }))
        payload = {
            "actions": [{
                "type": "change_group",
                "group": "qps_profile",
                "navigation_explicit": True,
                "source_evidence": "Return to QPS settings",
            }],
            "semantic_units": [],
        }

        result_text, changed = _adjudicate_group_navigation_actions(
            provider,
            json.dumps(payload),
            {"pending_question": {"id": "chain"}},
        )

        self.assertFalse(changed)
        result = json.loads(result_text)
        self.assertEqual(result["actions"], payload["actions"])
        self.assertEqual(result["semantic_units"], payload["semantic_units"])
        self.assertEqual(result["group_navigation_admissions"], [{
            "action_index": 0,
            "group": "qps_profile",
            "source_evidence": "Return to QPS settings",
            "destination_quote": "QPS settings",
        }])

    def test_navigation_only_owner_intake_normalizes_to_registered_group_navigation(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _adjudicate_group_navigation_actions

        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "reviews": [{
                "action_index": 0,
                "destination_named": True,
                "destination_quote": "configure QPS first",
                "generic_resume": False,
                "resume_quote": "",
                "specific_change_requested": False,
                "specific_change_quote": "",
                "reason": "the source names an area but no value/default change",
            }],
        }))
        payload = {
            "actions": [{
                "type": "request_qps_customization",
                "source_evidence": "configure QPS first before the rest",
            }],
            "semantic_units": [{
                "unit_id": "unit-1",
                "clause_id": "clause-1",
                "source_text": "configure QPS first before the rest",
                "disposition": "action",
                "action_indexes": [0],
            }],
        }

        result_text, changed = _adjudicate_group_navigation_actions(
            provider,
            json.dumps(payload),
            {"pending_question": {"id": "chain"}},
        )
        result = json.loads(result_text)

        self.assertTrue(changed)
        self.assertEqual(result["actions"], [{
            "type": "change_group",
            "group": "qps_profile",
            "navigation_explicit": True,
            "source_evidence": "configure QPS first",
        }])
        self.assertEqual(result["semantic_units"][0]["action_indexes"], [0])
        self.assertEqual(result["group_navigation_admissions"], [{
            "action_index": 0,
            "group": "qps_profile",
            "source_evidence": "configure QPS first",
            "destination_quote": "configure QPS first",
        }])

    def test_false_positive_specific_change_is_independently_rejected(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _adjudicate_group_navigation_actions

        provider = Mock()
        provider.complete.side_effect = [
            SimpleNamespace(text=json.dumps({"reviews": [{
                "action_index": 0,
                "destination_named": True,
                "destination_quote": "configure QPS first",
                "generic_resume": False,
                "resume_quote": "",
                "specific_change_requested": True,
                "specific_change_quote": "configure QPS",
                "reason": "first reviewer over-interpreted configure",
            }]})),
            SimpleNamespace(text=json.dumps({"reviews": [{
                "action_index": 0,
                "specific_change_requested": False,
                "specific_change_quote": "",
                "reason": "no value or default change is requested",
            }]})),
        ]
        payload = {
            "actions": [{
                "type": "request_qps_customization",
                "source_evidence": "configure QPS first before the rest",
            }],
            "semantic_units": [],
        }

        result_text, changed = _adjudicate_group_navigation_actions(
            provider,
            json.dumps(payload),
            {"pending_question": {"id": "chain"}},
        )
        result = json.loads(result_text)

        self.assertTrue(changed)
        self.assertEqual(result["actions"][0]["type"], "change_group")
        self.assertEqual(result["actions"][0]["group"], "qps_profile")
        self.assertEqual(provider.complete.call_count, 2)

    def test_malformed_specific_change_re_adjudication_fails_closed(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _adjudicate_group_navigation_actions

        provider = Mock()
        provider.complete.side_effect = [
            SimpleNamespace(text=json.dumps({"reviews": [{
                "action_index": 0,
                "destination_named": True,
                "destination_quote": "QPS",
                "generic_resume": False,
                "resume_quote": "",
                "specific_change_requested": True,
                "specific_change_quote": "adjust QPS",
                "reason": "specific change",
            }]})),
            SimpleNamespace(text=json.dumps({"reviews": [{
                "action_index": 0,
                "specific_change_requested": True,
                "specific_change_quote": "not in source",
                "reason": "invalid quote",
            }]})),
        ]
        payload = {
            "actions": [{
                "type": "request_qps_customization",
                "source_evidence": "adjust QPS",
            }],
            "semantic_units": [],
        }

        result_text, changed = _adjudicate_group_navigation_actions(
            provider,
            json.dumps(payload),
            {"pending_question": {}},
        )
        result = json.loads(result_text)

        self.assertTrue(changed)
        self.assertEqual(result["actions"], [])
        self.assertEqual(result["admission_rejections"][0]["action_type"], "request_qps_customization")

    def test_explicit_owner_change_remains_owner_intake(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _adjudicate_group_navigation_actions

        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "reviews": [{
                "action_index": 0,
                "destination_named": True,
                "destination_quote": "QPS defaults",
                "generic_resume": False,
                "resume_quote": "",
                "specific_change_requested": True,
                "specific_change_quote": "change the QPS defaults",
                "reason": "explicit owner change",
            }],
        }))
        payload = {
            "actions": [{
                "type": "request_qps_customization",
                "source_evidence": "change the QPS defaults",
            }],
            "semantic_units": [],
        }

        result_text, changed = _adjudicate_group_navigation_actions(
            provider,
            json.dumps(payload),
            {"pending_question": {}},
        )

        self.assertFalse(changed)
        self.assertEqual(json.loads(result_text), payload)

    def test_duplicate_group_control_review_fails_closed(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _adjudicate_group_navigation_actions

        provider = Mock()
        row = {
            "action_index": 0,
            "destination_named": True,
            "destination_quote": "QPS",
            "generic_resume": False,
            "resume_quote": "",
            "specific_change_requested": False,
            "specific_change_quote": "",
            "reason": "duplicate",
        }
        provider.complete.return_value = SimpleNamespace(text=json.dumps({"reviews": [row, row]}))
        payload = {
            "actions": [{
                "type": "request_qps_customization",
                "source_evidence": "visit QPS",
            }],
            "semantic_units": [],
        }

        result_text, changed = _adjudicate_group_navigation_actions(
            provider,
            json.dumps(payload),
            {"pending_question": {}},
        )
        result = json.loads(result_text)

        self.assertTrue(changed)
        self.assertEqual(result["actions"], [])
        self.assertEqual(result["admission_rejections"][0]["action_type"], "request_qps_customization")

    def test_rejected_group_navigation_preserves_sibling_action(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _adjudicate_group_navigation_actions

        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "reviews": [{
                "action_index": 0,
                "destination_named": False,
                "destination_quote": "",
                "generic_resume": False,
                "resume_quote": "",
                "reason": "the source asks a question rather than navigating",
            }],
        }))
        payload = {
            "actions": [
                {
                    "type": "change_group",
                    "group": "qps_profile",
                    "navigation_explicit": True,
                    "source_evidence": "What does QPS mean?",
                },
                {
                    "type": "answer_opening_question",
                    "topic": "config_explanation",
                    "subject": "QPS",
                },
            ],
            "semantic_units": [{
                "unit_id": "unit-1",
                "clause_id": "clause-1",
                "source_text": "What does QPS mean?",
                "disposition": "action",
                "action_indexes": [0, 1],
            }],
        }

        result_text, changed = _adjudicate_group_navigation_actions(
            provider,
            json.dumps(payload),
            {"pending_question": {}},
        )
        result = json.loads(result_text)

        self.assertTrue(changed)
        self.assertEqual([action["type"] for action in result["actions"]], ["answer_opening_question"])
        self.assertEqual(result["semantic_units"][0]["action_indexes"], [0])
        self.assertEqual(result["admission_rejections"][0]["action_type"], "change_group")

    def test_uncertain_chain_mention_is_not_an_actionable_selection(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _adjudicate_chain_selection_actions

        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "reviews": [{
                "action_index": 0,
                "committed_selection": False,
                "commitment_quote": "",
                "uncertain_candidate": True,
                "uncertainty_quote": "可能测 BNB",
                "informational_only": False,
                "information_quote": "",
                "finite_candidate_request": False,
                "candidate_quote": "",
                "reason": "the chain is only a possible future target",
            }],
        }))
        payload = {
            "actions": [{
                "type": "choose_chain",
                "chain_text": "BNB",
                "chain_candidates": ["BNB"],
                "source_evidence": "我这边可能测 BNB",
            }],
        }

        result_text, changed = _adjudicate_chain_selection_actions(provider, json.dumps(payload))
        result = json.loads(result_text)

        self.assertTrue(changed)
        self.assertEqual(result["actions"], [{
            "type": "request_chain_selection",
            "chain_candidates": ["BNB"],
            "source_evidence": "可能测 BNB",
        }])

    def test_explicit_chain_selection_is_actionable(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _adjudicate_chain_selection_actions

        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "reviews": [{
                "action_index": 0,
                "committed_selection": True,
                "commitment_quote": "测试 BNB",
                "uncertain_candidate": False,
                "uncertainty_quote": "",
                "informational_only": False,
                "information_quote": "",
                "finite_candidate_request": False,
                "candidate_quote": "",
                "reason": "explicit benchmark target",
            }],
        }))
        payload = {
            "actions": [{
                "type": "choose_chain",
                "chain_text": "BNB",
                "chain_candidates": ["BNB"],
                "source_evidence": "我要测试 BNB",
            }],
        }

        result_text, changed = _adjudicate_chain_selection_actions(provider, json.dumps(payload))

        self.assertFalse(changed)
        result = json.loads(result_text)
        self.assertEqual(result["actions"], payload["actions"])
        self.assertEqual(result["chain_selection_admissions"], [0])

    def test_active_chain_intake_is_part_of_chain_selection_adjudication(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _adjudicate_chain_selection_actions
        from agent.harness.state import new_state

        source = "Call the chain Sola."
        state = new_state("unit-thread", language="en")
        state["pending_question"] = {
            "id": "chain",
            "group": "chain_identity",
            "kind": "chain",
            "field": "BLOCKCHAIN_NODE",
            "manual_input_allowed": True,
            "options": [],
        }
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "reviews": [{
                "action_index": 0,
                "committed_selection": True,
                "commitment_quote": source,
                "uncertain_candidate": False,
                "uncertainty_quote": "",
                "informational_only": False,
                "information_quote": "",
                "finite_candidate_request": False,
                "candidate_quote": "",
                "reason": "direct answer to the active typed chain intake",
            }],
        }))
        payload = {
            "actions": [{
                "type": "choose_chain",
                "chain_text": "Sola",
                "source_evidence": source,
            }],
        }

        result_text, changed = _adjudicate_chain_selection_actions(
            provider,
            json.dumps(payload),
            state,
        )

        self.assertFalse(changed)
        self.assertEqual(json.loads(result_text)["chain_selection_admissions"], [0])
        request_payload = json.loads(provider.complete.call_args.args[0].messages[-1].content)
        self.assertIs(request_payload["reviews"][0]["active_chain_intake"], True)

    def test_explicit_finite_chain_candidate_request_is_actionable(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _adjudicate_chain_selection_actions

        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "reviews": [{
                "action_index": 0,
                "committed_selection": False,
                "commitment_quote": "",
                "uncertain_candidate": True,
                "uncertainty_quote": "either BNB or Ethereum",
                "informational_only": False,
                "information_quote": "",
                "finite_candidate_request": True,
                "candidate_quote": "help me choose between those two",
                "reason": "explicit finite candidate request",
            }],
        }))
        source = "Benchmark either BNB or Ethereum; help me choose between those two"
        payload = {
            "actions": [{
                "type": "choose_chain",
                "chain_candidates": ["BNB", "Ethereum"],
                "source_evidence": source,
            }],
        }

        result_text, changed = _adjudicate_chain_selection_actions(provider, json.dumps(payload))

        self.assertFalse(changed)
        result = json.loads(result_text)
        self.assertEqual(result["actions"], payload["actions"])
        self.assertEqual(result["chain_selection_admissions"], [0])

    def test_general_semantic_reviewer_cannot_overrule_chain_admission(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _validate_semantic_fulfillment
        from agent.harness.state import new_state

        source = "Test BNB and tell me which RPC methods are supported"
        clauses = segment_user_turn(source)
        payload = {
            "actions": [
                {"type": "choose_chain", "chain_text": "BNB", "source_evidence": "Test BNB"},
                {
                    "type": "answer_opening_question",
                    "topic": "supported_rpc_methods",
                    "source_evidence": "tell me which RPC methods are supported",
                },
            ],
            "semantic_units": [{
                "unit_id": "unit-1",
                "clause_id": clauses[0].clause_id,
                "source_text": source,
                "disposition": "action",
                "action_indexes": [0, 1],
            }],
            "chain_selection_admissions": [0],
        }
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "reviews": [{"action_index": 1, "supported": True, "reason": "supported"}],
            "unit_reviews": [{"unit_id": "unit-1", "complete": True, "reason": "complete"}],
        }))

        result = _validate_semantic_fulfillment(
            provider,
            json.dumps(payload),
            clauses,
            new_state("unit-thread", language="en"),
        )

        self.assertTrue(result.valid, result.errors)
        review_payload = json.loads(provider.complete.call_args_list[0].args[0].messages[1].content)
        self.assertEqual([row["action_index"] for row in review_payload["reviews"]], [1])

    def test_single_durable_action_unit_is_audited_for_omitted_demands(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _validate_semantic_fulfillment
        from agent.harness.state import new_state

        source = "Test BNB with mixed workload, quick QPS, and local Grafana"
        clauses = segment_user_turn(source)
        payload = {
            "actions": [{
                "type": "choose_chain",
                "chain_text": "BNB",
                "source_evidence": source,
            }],
            "semantic_units": [_unit(clauses[0], 1, [0])],
            "chain_selection_admissions": [0],
        }
        provider = Mock()
        negative = {
            "unit_reviews": [{
                "unit_id": "unit-1",
                "complete": False,
                "missing_demand_quote": "mixed workload",
                "reason": "the workload, QPS, and observability demands are omitted",
            }],
        }
        provider.complete.side_effect = [
            SimpleNamespace(text=json.dumps(negative)),
            SimpleNamespace(text=json.dumps(negative)),
        ]

        result = _validate_semantic_fulfillment(
            provider,
            json.dumps(payload),
            clauses,
            new_state("single-unit-completeness", language="en"),
        )

        self.assertFalse(result.valid)
        self.assertEqual(result.incomplete_unit_ids, ("unit-1",))
        first_request = json.loads(provider.complete.call_args_list[0].args[0].messages[1].content)
        self.assertEqual(first_request["unit_reviews"][0]["mapped_actions"][0]["operation_index"], 0)

    def test_single_durable_action_unit_passes_when_it_preserves_the_whole_request(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _validate_semantic_fulfillment
        from agent.harness.state import new_state

        source = "Test BNB"
        clauses = segment_user_turn(source)
        payload = {
            "actions": [{
                "type": "choose_chain",
                "chain_text": "BNB",
                "source_evidence": source,
            }],
            "semantic_units": [_unit(clauses[0], 1, [0])],
            "chain_selection_admissions": [0],
        }
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "unit_reviews": [{
                "unit_id": "unit-1",
                "complete": True,
                "missing_demand_quote": "",
                "reason": "the chain selection preserves the complete request",
            }],
        }))

        result = _validate_semantic_fulfillment(
            provider,
            json.dumps(payload),
            clauses,
            new_state("single-unit-complete", language="en"),
        )

        self.assertTrue(result.valid, result.errors)
        provider.complete.assert_called_once()

    def test_negative_unit_verdict_is_re_adjudicated_under_quote_contract(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _adjudicate_incomplete_unit_reviews

        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "unit_reviews": [{
                "unit_id": "unit-1",
                "complete": True,
                "missing_demand_quote": "",
                "reason": "the mapped operations preserve the related request",
            }],
        }))
        reviews = [{
            "unit_id": "unit-1",
            "source_text": "这是部署工单里复制出来的部分信息。",
            "related_source_units": [{"unit_id": "unit-2", "source_text": "使用 Ethereum"}],
            "mapped_actions": [{"operation_index": 0, "declared_purpose": "select Ethereum"}],
        }]

        rows = _adjudicate_incomplete_unit_reviews(
            provider,
            unit_reviews=reviews,
            first_rows=[{
                "unit_id": "unit-1",
                "complete": False,
                "reason": "this framing is complete",
            }],
        )

        self.assertIs(rows[0]["complete"], True)
        self.assertEqual(provider.complete.call_count, 1)

    def test_negative_unit_adjudication_requires_an_exact_omitted_demand_quote(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _adjudicate_incomplete_unit_reviews

        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "unit_reviews": [{
                "unit_id": "unit-1",
                "complete": False,
                "missing_demand_quote": "a demand that is not in the source",
                "reason": "incomplete",
            }],
        }))
        first = {"unit_id": "unit-1", "complete": False, "reason": "incomplete"}

        rows = _adjudicate_incomplete_unit_reviews(
            provider,
            unit_reviews=[{
                "unit_id": "unit-1",
                "source_text": "Use Ethereum and also explain the current report.",
                "related_source_units": [],
                "mapped_actions": [{"operation_index": 0, "declared_purpose": "select Ethereum"}],
            }],
            first_rows=[first],
        )

        self.assertEqual(rows, [first])

    def test_registry_inventory_recovers_an_embedded_owner_demand(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _challenge_registry_action_inventory

        source = "Use fake-node for BNB with mixed RPC, quick QPS, and local Grafana."
        clauses = segment_user_turn(source)
        payload = {
            "actions": [{"type": "set_rpc_mode", "rpc_mode": "mixed", "source_evidence": "mixed RPC"}],
            "semantic_units": [_unit(clauses[0], 1, [0])],
        }
        provider = Mock()
        provider.complete.side_effect = [
            SimpleNamespace(text=json.dumps({
                "findings": [{
                    "unit_id": "unit-1",
                    "status": "missing",
                    "missing_demands": [{
                        "group": "chain_identity",
                        "evidence_quote": "BNB",
                        "reason": "the chain selection is not represented",
                    }],
                    "reason": "one registered owner demand is missing",
                }],
            })),
            SimpleNamespace(text=json.dumps({
                "reviews": [{
                    "demand_id": "unit-1:0",
                    "direct_unrepresented": True,
                    "evidence_quote": "BNB",
                    "reason": "the source directly selects BNB and no action preserves it",
                }],
            })),
            SimpleNamespace(text=json.dumps({
                "action": {
                    "type": "choose_chain",
                    "chain_text": "BNB",
                    "source_evidence": "BNB",
                },
                "reason": "the chain owner can represent the quoted demand",
            })),
        ]

        recovered_text, changed, incomplete = _challenge_registry_action_inventory(
            provider,
            json.dumps(payload),
        )
        recovered = json.loads(recovered_text)

        self.assertTrue(changed)
        self.assertEqual(incomplete, ())
        self.assertEqual([action["type"] for action in recovered["actions"]], ["set_rpc_mode", "choose_chain"])
        self.assertEqual(recovered["semantic_units"][0]["action_indexes"], [0, 1])

    def test_registry_inventory_audits_complete_units_without_rejudging_pending_action(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _challenge_registry_action_inventory
        from agent.harness.state import new_state

        source = (
            "Plans changed: don't generate benchmark traffic. "
            "I only want to watch an already-running real node catch up to chain head."
        )
        clauses = segment_user_turn(source)
        payload = {
            "actions": [{
                "type": "choose_target_mode",
                "target_mode": "sync-observe",
                "target_mode_explicit": True,
                "source_evidence": clauses[1].text,
            }],
            "pending_answer_admissions": [0],
            "semantic_units": [
                _unit(clauses[0], 1, [0]),
                _unit(clauses[1], 2, [0]),
            ],
        }
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "findings": [
                {
                    "unit_id": "unit-1",
                    "status": "complete",
                    "missing_demands": [],
                    "reason": "the admitted mode also preserves the no-traffic constraint",
                },
                {
                    "unit_id": "unit-2",
                    "status": "complete",
                    "missing_demands": [],
                    "reason": "the admitted mode preserves the observation goal",
                },
            ],
        }))
        state = new_state("inventory-pending-owner", language="en")
        state["pending_question"] = {
            "id": "target_mode_select",
            "prompt": "Choose the target mode.",
            "options": [{
                "id": "sync-observe",
                "label": "Observe a real node catching up without benchmark traffic",
                "value": "sync-observe",
                "action": {
                    "type": "choose_target_mode",
                    "target_mode": "sync-observe",
                    "target_mode_explicit": True,
                },
                "expected_patch": {
                    "target_mode": "sync-observe",
                    "workflow_mode": "sync_observe",
                },
            }],
        }

        result_text, changed, incomplete = _challenge_registry_action_inventory(
            provider,
            json.dumps(payload),
            state,
        )

        self.assertFalse(changed)
        self.assertEqual(incomplete, ())
        self.assertEqual(json.loads(result_text), payload)
        provider.complete.assert_called_once()
        request = json.loads(provider.complete.call_args.args[0].messages[1].content)
        represented = request["units"][0]["represented_actions"][0]
        self.assertEqual(represented["owner_admission"], "pending_option")
        self.assertEqual(
            represented["declared_pending_option"]["declared_action"]["type"],
            "choose_target_mode",
        )
        self.assertEqual(
            represented["declared_pending_option"]["expected_patch"]["workflow_mode"],
            "sync_observe",
        )

    def test_registry_inventory_exposes_failure_recovery_option_effect(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _challenge_registry_action_inventory
        from agent.harness.state import new_state

        source = "我先修正出错的 endpoint 配置，然后重新验证。"
        clauses = segment_user_turn(source)
        payload = {
            "actions": [{"type": "correct_failure"}],
            "pending_answer_admissions": [0],
            "semantic_units": [_unit(clauses[0], 1, [0])],
        }
        state = new_state("inventory-recovery-owner", language="zh")
        state["pending_question"] = {
            "id": "failure_recovery_action",
            "prompt": "请选择下一步。",
            "options": [{
                "id": "1",
                "label": "修正受影响的配置并重新验证",
                "value": "correct",
                "action": {"type": "correct_failure"},
                "expected_patch": {"failure_recovery.status": "correcting"},
            }],
        }
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "findings": [{
                "unit_id": "unit-1",
                "status": "complete",
                "missing_demands": [],
                "reason": "the declared recovery option covers correction and revalidation",
            }],
        }))

        result_text, changed, incomplete = _challenge_registry_action_inventory(
            provider,
            json.dumps(payload, ensure_ascii=False),
            state,
        )

        self.assertFalse(changed)
        self.assertEqual(incomplete, ())
        self.assertEqual(json.loads(result_text), payload)
        request = json.loads(provider.complete.call_args.args[0].messages[1].content)
        represented = request["units"][0]["represented_actions"][0]
        self.assertEqual(represented["owner_admission"], "pending_option")
        self.assertEqual(represented["declared_pending_option"], {
            "declared_action": {"type": "correct_failure"},
            "expected_patch": {"failure_recovery.status": "correcting"},
            "label": "修正受影响的配置并重新验证",
            "question_id": "failure_recovery_action",
            "question_prompt": "请选择下一步。",
            "value": "correct",
        })

    def test_registry_inventory_recovers_demand_outside_pending_option_effect(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _challenge_registry_action_inventory
        from agent.harness.state import new_state

        source = "Use BSC for this benchmark, and set the QPS mode to quick."
        clauses = segment_user_turn(source)
        payload = {
            "actions": [{
                "type": "answer_pending",
                "answer": "bsc",
                "selected_value": "bsc",
                "source_evidence": "Use BSC",
            }],
            "pending_answer_admissions": [0],
            "semantic_units": [_unit(clauses[0], 1, [0])],
        }
        state = new_state("inventory-owner-independent-demand", language="en")
        state["pending_question"] = {
            "id": "chain_ambiguity_confirm",
            "prompt": "Choose the chain.",
            "options": [{
                "id": "bsc",
                "label": "BNB Smart Chain",
                "value": "bsc",
                "action": {"type": "answer_pending", "answer": "bsc"},
                "expected_patch": {"chain_identity.canonical": "bsc"},
            }],
        }
        provider = Mock()
        provider.complete.side_effect = [
            SimpleNamespace(text=json.dumps({
                "findings": [{
                    "unit_id": "unit-1",
                    "status": "missing",
                    "missing_demands": [{
                        "group": "qps_profile",
                        "evidence_quote": "set the QPS mode to quick",
                        "reason": "the QPS request is independent of the selected chain option",
                    }],
                    "reason": "one independent demand is not represented",
                }],
            })),
            SimpleNamespace(text=json.dumps({
                "reviews": [{
                    "demand_id": "unit-1:0",
                    "direct_unrepresented": True,
                    "evidence_quote": "set the QPS mode to quick",
                    "reason": "the source directly requests quick QPS and no action preserves it",
                }],
            })),
            SimpleNamespace(text=json.dumps({
                "action": {
                    "type": "set_qps_mode",
                    "qps_mode": "quick",
                    "mutation_explicit": True,
                    "source_evidence": "set the QPS mode to quick",
                },
                "reason": "the QPS owner can represent the quoted demand",
            })),
        ]

        recovered_text, changed, incomplete = _challenge_registry_action_inventory(
            provider,
            json.dumps(payload),
            state,
        )
        recovered = json.loads(recovered_text)

        self.assertTrue(changed)
        self.assertEqual(incomplete, ())
        self.assertEqual(
            [action["type"] for action in recovered["actions"]],
            ["answer_pending", "set_qps_mode"],
        )
        self.assertEqual(recovered["semantic_units"][0]["action_indexes"], [0, 1])

    def test_registry_inventory_still_audits_independent_sibling_of_pending_answer(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _challenge_registry_action_inventory

        source = "Watch the node catch up. Also switch the QPS profile to quick."
        clauses = segment_user_turn(source)
        payload = {
            "actions": [{
                "type": "choose_target_mode",
                "target_mode": "sync-observe",
                "target_mode_explicit": True,
                "source_evidence": clauses[0].text,
            }, {
                "type": "set_qps_mode",
                "qps_mode": "quick",
                "mutation_explicit": True,
                "source_evidence": clauses[1].text,
            }],
            "pending_answer_admissions": [0],
            "semantic_units": [
                _unit(clauses[0], 1, [0]),
                _unit(clauses[1], 2, [1]),
            ],
        }
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "findings": [
                {
                    "unit_id": "unit-1",
                    "status": "complete",
                    "missing_demands": [],
                    "reason": "the admitted observation goal is represented",
                },
                {
                    "unit_id": "unit-2",
                    "status": "complete",
                    "missing_demands": [],
                    "reason": "the independent QPS request is represented",
                },
            ],
        }))

        result_text, changed, incomplete = _challenge_registry_action_inventory(
            provider,
            json.dumps(payload),
        )

        self.assertFalse(changed)
        self.assertEqual(incomplete, ())
        self.assertEqual(json.loads(result_text), payload)
        request = json.loads(provider.complete.call_args.args[0].messages[1].content)
        self.assertEqual([row["unit_id"] for row in request["units"]], ["unit-1", "unit-2"])

    def test_registry_inventory_does_not_duplicate_an_existing_typed_action(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _challenge_registry_action_inventory

        source = "Use BNB."
        clauses = segment_user_turn(source)
        action = {"type": "choose_chain", "chain_text": "BNB", "source_evidence": "Use BNB"}
        payload = {"actions": [action], "semantic_units": [_unit(clauses[0], 1, [0])]}
        provider = Mock()
        provider.complete.side_effect = [
            SimpleNamespace(text=json.dumps({"findings": [{
                "unit_id": "unit-1",
                "status": "missing",
                "missing_demands": [{"group": "chain_identity", "evidence_quote": "BNB", "reason": "duplicate"}],
                "reason": "duplicate proposal",
            }]})),
            SimpleNamespace(text=json.dumps({"reviews": [{
                "demand_id": "unit-1:0",
                "direct_unrepresented": False,
                "evidence_quote": "BNB",
                "reason": "the represented chain action already preserves this demand",
            }]})),
        ]

        recovered_text, changed, incomplete = _challenge_registry_action_inventory(provider, json.dumps(payload))

        self.assertFalse(changed)
        self.assertEqual(incomplete, ())
        self.assertEqual(len(json.loads(recovered_text)["actions"]), 1)

    def test_registry_inventory_rejects_an_invented_workflow_prerequisite(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _challenge_registry_action_inventory

        clauses = segment_user_turn("change to BNB")
        action = {"type": "change_chain", "chain_text": "BNB", "source_evidence": "change to BNB"}
        payload = {"actions": [action], "semantic_units": [_unit(clauses[0], 1, [0])]}
        provider = Mock()
        provider.complete.side_effect = [
            SimpleNamespace(text=json.dumps({"findings": [{
                "unit_id": "unit-1",
                "status": "missing",
                "missing_demands": [{
                    "group": "opening",
                    "evidence_quote": "change to BNB",
                    "reason": "changing a chain allegedly requires an opening selection",
                }],
                "reason": "invented prerequisite",
            }]})),
            SimpleNamespace(text=json.dumps({"reviews": [{
                "demand_id": "unit-1:0",
                "direct_unrepresented": False,
                "evidence_quote": "change to BNB",
                "reason": "opening is an internal prerequisite, not a direct source demand",
            }]})),
        ]

        result_text, changed, incomplete = _challenge_registry_action_inventory(provider, json.dumps(payload))

        self.assertFalse(changed)
        self.assertEqual(incomplete, ())
        self.assertEqual(json.loads(result_text), payload)
        self.assertEqual(provider.complete.call_count, 2)

    def test_registry_inventory_fails_closed_when_direct_demand_verification_is_malformed(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _challenge_registry_action_inventory

        clauses = segment_user_turn("Use mixed RPC for BNB")
        payload = {
            "actions": [{"type": "set_rpc_mode", "rpc_mode": "mixed", "source_evidence": "mixed RPC"}],
            "semantic_units": [_unit(clauses[0], 1, [0])],
        }
        provider = Mock()
        provider.complete.side_effect = [
            SimpleNamespace(text=json.dumps({"findings": [{
                "unit_id": "unit-1",
                "status": "missing",
                "missing_demands": [{
                    "group": "chain_identity",
                    "evidence_quote": "BNB",
                    "reason": "chain is omitted",
                }],
                "reason": "one missing demand",
            }]})),
            SimpleNamespace(text="{}"),
        ]

        result_text, changed, incomplete = _challenge_registry_action_inventory(provider, json.dumps(payload))

        self.assertFalse(changed)
        self.assertEqual(incomplete, ("unit-1",))
        self.assertEqual(json.loads(result_text), payload)

    def test_inventory_wrapper_preserves_a_prevalidated_plan_when_challenger_is_incomplete(self) -> None:
        import json
        from unittest.mock import Mock, patch

        from agent.harness.intent import _challenge_and_validate_registry_inventory
        from agent.harness.plan_coverage import PlanCoverageResult
        from agent.harness.state import new_state

        plan = json.dumps({
            "actions": [{
                "type": "request_target_mode_selection",
                "source_evidence": "not sure whether to use fake-node or real-node",
            }],
            "semantic_units": [],
        })
        validation = PlanCoverageResult(True, (), ())
        with patch(
            "agent.harness.intent._challenge_registry_action_inventory",
            return_value=(plan, False, ("unit-1",)),
        ):
            result_plan, result_validation = _challenge_and_validate_registry_inventory(
                Mock(),
                plan,
                (),
                new_state("inventory-wrapper"),
                "not sure whether to use fake-node or real-node",
                validation,
            )

        self.assertEqual(result_plan, plan)
        self.assertIs(result_validation, validation)

    def test_inventory_wrapper_fails_closed_for_incomplete_owner_admitted_unit(self) -> None:
        import json
        from unittest.mock import Mock, patch

        from agent.harness.intent import _challenge_and_validate_registry_inventory
        from agent.harness.plan_coverage import PlanCoverageResult
        from agent.harness.state import new_state

        plan = json.dumps({
            "actions": [{
                "type": "answer_pending",
                "answer": "bsc",
                "selected_value": "bsc",
                "source_evidence": "Use BSC and change QPS too",
            }],
            "pending_answer_admissions": [0],
            "semantic_units": [{
                "unit_id": "unit-owner",
                "source_text": "Use BSC and change QPS too",
                "disposition": "action",
                "action_indexes": [0],
            }],
        })
        validation = PlanCoverageResult(True, (), ())
        with patch(
            "agent.harness.intent._challenge_registry_action_inventory",
            return_value=(plan, False, ("unit-owner",)),
        ):
            result_plan, result_validation = _challenge_and_validate_registry_inventory(
                Mock(),
                plan,
                (),
                new_state("inventory-owner-fail-closed"),
                "Use BSC and change QPS too",
                validation,
            )

        self.assertEqual(result_plan, plan)
        self.assertFalse(result_validation.valid)
        self.assertEqual(result_validation.incomplete_unit_ids, ("unit-owner",))

    def test_inventory_wrapper_never_bypasses_an_initially_invalid_plan(self) -> None:
        from unittest.mock import Mock, patch

        from agent.harness.intent import _challenge_and_validate_registry_inventory
        from agent.harness.plan_coverage import PlanCoverageResult
        from agent.harness.state import new_state

        validation = PlanCoverageResult(False, ("invalid original plan",), ())
        challenger = Mock()
        with patch("agent.harness.intent._challenge_registry_action_inventory", challenger):
            result_plan, result_validation = _challenge_and_validate_registry_inventory(
                Mock(),
                "{}",
                (),
                new_state("inventory-invalid"),
                "input",
                validation,
            )

        challenger.assert_not_called()
        self.assertEqual(result_plan, "{}")
        self.assertIs(result_validation, validation)

    def test_registry_inventory_recovers_only_verified_demands_from_a_mixed_finding(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock, patch

        from agent.harness.intent import _challenge_registry_action_inventory

        clauses = segment_user_turn("Run BNB against fake-node")
        payload = {
            "actions": [{
                "type": "choose_target_mode",
                "target_mode": "fake-node",
                "source_evidence": "fake-node",
            }],
            "semantic_units": [_unit(clauses[0], 1, [0])],
        }
        provider = Mock()
        provider.complete.side_effect = [
            SimpleNamespace(text=json.dumps({"findings": [{
                "unit_id": "unit-1",
                "status": "missing",
                "missing_demands": [
                    {
                        "group": "opening",
                        "evidence_quote": "Run BNB against fake-node",
                        "reason": "invented opening prerequisite",
                    },
                    {
                        "group": "chain_identity",
                        "evidence_quote": "BNB",
                        "reason": "direct chain demand is unrepresented",
                    },
                ],
                "reason": "one false and one real candidate",
            }]})),
            SimpleNamespace(text=json.dumps({"reviews": [
                {
                    "demand_id": "unit-1:0",
                    "direct_unrepresented": False,
                    "evidence_quote": "Run BNB against fake-node",
                    "reason": "opening is not a source demand",
                },
                {
                    "demand_id": "unit-1:1",
                    "direct_unrepresented": True,
                    "evidence_quote": "BNB",
                    "reason": "BNB is directly selected and unrepresented",
                },
            ]})),
        ]
        chain_action = {"type": "choose_chain", "chain_text": "BNB", "source_evidence": "BNB"}

        with patch("agent.harness.intent._resolve_owned_group_mutation", return_value=chain_action) as resolver:
            result_text, changed, incomplete = _challenge_registry_action_inventory(provider, json.dumps(payload))
        result = json.loads(result_text)

        self.assertTrue(changed)
        self.assertEqual(incomplete, ())
        self.assertEqual([action["type"] for action in result["actions"]], ["choose_target_mode", "choose_chain"])
        resolver.assert_called_once_with(provider, "chain_identity", "BNB")

    def test_direct_inventory_verification_uses_bounded_batches(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _verify_direct_inventory_demands

        candidates = [
            {
                "demand_id": f"unit-1:{index}",
                "unit_id": "unit-1",
                "source_text": f"source demand {index}",
                "represented_actions": [],
                "proposed_group": "chain_identity",
                "proposed_evidence_quote": f"demand {index}",
                "challenger_reason": "untrusted",
            }
            for index in range(5)
        ]
        provider = Mock()
        provider.complete.side_effect = [
            SimpleNamespace(text=json.dumps({"reviews": [
                {
                    "demand_id": f"unit-1:{index}",
                    "direct_unrepresented": False,
                    "evidence_quote": f"demand {index}",
                    "reason": "not direct",
                }
                for index in range(4)
            ]})),
            SimpleNamespace(text=json.dumps({"reviews": [{
                "demand_id": "unit-1:4",
                "direct_unrepresented": True,
                "evidence_quote": "demand 4",
                "reason": "direct and missing",
            }]})),
        ]

        verified, valid = _verify_direct_inventory_demands(provider, candidates)

        self.assertTrue(valid)
        self.assertEqual(verified, {"unit-1:4"})
        self.assertEqual(provider.complete.call_count, 2)
        first_payload = json.loads(provider.complete.call_args_list[0].args[0].messages[1].content)
        second_payload = json.loads(provider.complete.call_args_list[1].args[0].messages[1].content)
        self.assertEqual(len(first_payload["candidates"]), 4)
        self.assertEqual(len(second_payload["candidates"]), 1)

    def test_navigation_semantics_are_shared_by_all_inventory_boundaries(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import (
            GROUP_NAVIGATION_SEMANTIC_POLICY,
            _challenge_registry_action_inventory,
            _resolve_owned_group_mutation,
            _semantic_fulfillment_prompt,
            _verify_direct_inventory_demands,
        )

        clauses = segment_user_turn("Configure the QPS area before continuing.")
        payload = {
            "actions": [{
                "type": "change_group",
                "group": "qps_profile",
                "navigation_explicit": True,
                "source_evidence": clauses[0].text,
            }],
            "semantic_units": [_unit(clauses[0], 1, [0])],
        }
        challenge_provider = Mock()
        challenge_provider.complete.return_value = SimpleNamespace(text=json.dumps({"findings": [{
            "unit_id": "unit-1",
            "status": "complete",
            "missing_demands": [],
            "reason": "navigation opens later typed questions",
        }]}))
        _challenge_registry_action_inventory(challenge_provider, json.dumps(payload))

        verify_provider = Mock()
        verify_provider.complete.return_value = SimpleNamespace(text=json.dumps({"reviews": [{
            "demand_id": "unit-1:0",
            "direct_unrepresented": False,
            "evidence_quote": "Configure the QPS area",
            "reason": "already represented by navigation",
        }]}))
        _verify_direct_inventory_demands(verify_provider, [{
            "demand_id": "unit-1:0",
            "unit_id": "unit-1",
            "source_text": clauses[0].text,
            "represented_actions": [],
            "proposed_group": "qps_profile",
            "proposed_evidence_quote": "Configure the QPS area",
            "challenger_reason": "untrusted",
        }])

        owner_provider = Mock()
        owner_provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "action": None,
            "reason": "navigation is not an owner mutation",
        }))
        _resolve_owned_group_mutation(owner_provider, "qps_profile", clauses[0].text)

        self.assertIn(GROUP_NAVIGATION_SEMANTIC_POLICY, _semantic_fulfillment_prompt(review_kind="actions"))
        self.assertIn(
            GROUP_NAVIGATION_SEMANTIC_POLICY,
            challenge_provider.complete.call_args.args[0].messages[0].content,
        )
        self.assertIn(
            GROUP_NAVIGATION_SEMANTIC_POLICY,
            verify_provider.complete.call_args.args[0].messages[0].content,
        )
        self.assertIn(
            GROUP_NAVIGATION_SEMANTIC_POLICY,
            owner_provider.complete.call_args.args[0].messages[0].content,
        )

    def test_registry_inventory_fails_closed_for_an_unsupported_group(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _challenge_registry_action_inventory

        source = "Use fake-node and invent another control plane."
        clauses = segment_user_turn(source)
        payload = {
            "actions": [{"type": "choose_target_mode", "target_mode": "fake-node", "source_evidence": "fake-node"}],
            "semantic_units": [_unit(clauses[0], 1, [0])],
        }
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({"findings": [{
            "unit_id": "unit-1",
            "status": "missing",
            "missing_demands": [{"group": "not_registered", "evidence_quote": "another control plane", "reason": "unsupported"}],
            "reason": "unsupported proposal",
        }]}))

        _, changed, incomplete = _challenge_registry_action_inventory(provider, json.dumps(payload))

        self.assertFalse(changed)
        self.assertEqual(incomplete, ("unit-1",))

    def test_registry_inventory_skips_consultation_only_units(self) -> None:
        import json
        from unittest.mock import Mock

        from agent.harness.intent import _challenge_registry_action_inventory

        clauses = segment_user_turn("What does fake-node do?")
        payload = {
            "actions": [{"type": "answer_capability_question", "topic": "target_modes"}],
            "semantic_units": [_unit(clauses[0], 1, [0])],
        }
        provider = Mock()

        recovered_text, changed, incomplete = _challenge_registry_action_inventory(provider, json.dumps(payload))

        self.assertFalse(changed)
        self.assertEqual(incomplete, ())
        self.assertEqual(json.loads(recovered_text), payload)
        provider.complete.assert_not_called()

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
        self.assertEqual(len(reconciled["semantic_units"]), 1)
        self.assertEqual(reconciled["semantic_units"][0]["source_text"], source)
        self.assertEqual(reconciled["semantic_units"][0]["disposition"], "action")
        self.assertEqual(reconciled["semantic_units"][0]["action_indexes"], [0])

    def test_missing_semantic_units_are_reconstructed_without_changing_actions(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _reconstruct_missing_semantic_units

        source = "Test BNB with mixed and quick"
        clauses = segment_user_turn(source)
        actions = [
            {"type": "choose_chain", "chain_text": "BNB", "source_evidence": source},
            {
                "type": "set_rpc_mode",
                "rpc_mode": "mixed",
                "mutation_explicit": True,
                "source_evidence": "mixed",
            },
            {
                "type": "set_qps_mode",
                "qps_mode": "quick",
                "mutation_explicit": True,
                "source_evidence": "quick",
            },
        ]
        provider = Mock()
        provider.complete.side_effect = [
            SimpleNamespace(text='{"semantic_units": ['),
            SimpleNamespace(text=json.dumps({
                "semantic_units": [{
                    "unit_id": "unit-1",
                    "clause_id": "clause-1",
                    "source_text": source,
                    "disposition": "action",
                    "action_indexes": [0, 1, 2],
                    "reason": "all explicit demands are represented",
                }],
            })),
        ]

        recovered = json.loads(_reconstruct_missing_semantic_units(
            provider,
            json.dumps({"actions": actions}),
            clauses,
        ))

        self.assertEqual(recovered["actions"], actions)
        self.assertEqual(recovered["semantic_units"][0]["source_text"], source)
        self.assertEqual(provider.complete.call_count, 2)

    def test_lossy_semantic_anchor_partition_is_reconstructed_without_changing_actions(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _reconstruct_missing_semantic_units

        source = "Keep Grafana local, use quick, make BNB mixed, and run against fake-node."
        clauses = segment_user_turn(source)
        actions = [
            {"type": "set_observability", "observability_mode": "local", "source_evidence": "Grafana local"},
            {"type": "set_qps_mode", "qps_mode": "quick", "source_evidence": "quick"},
            {"type": "choose_chain", "chain_text": "BNB", "source_evidence": "BNB"},
            {"type": "set_rpc_mode", "rpc_mode": "mixed", "source_evidence": "mixed"},
            {"type": "choose_target_mode", "target_mode": "fake-node", "source_evidence": "fake-node"},
        ]
        lossy_units = [
            {
                "unit_id": f"unit-{index}",
                "clause_id": clauses[0].clause_id,
                "source_text": anchor,
                "disposition": "action",
                "action_indexes": [index - 1],
                "reason": "partial exact anchor",
            }
            for index, anchor in enumerate(("Grafana local", "quick", "BNB", "mixed", "fake-node"), start=1)
        ]
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "semantic_units": [{
                "unit_id": "unit-1",
                "clause_id": clauses[0].clause_id,
                "source_text": source,
                "disposition": "action",
                "action_indexes": [0, 1, 2, 3, 4],
                "reason": "one lossless full-clause partition",
            }],
        }))

        recovered = json.loads(_reconstruct_missing_semantic_units(
            provider,
            json.dumps({"actions": actions, "semantic_units": lossy_units}),
            clauses,
        ))

        self.assertEqual(recovered["actions"], actions)
        self.assertEqual(recovered["semantic_units"][0]["source_text"], source)
        self.assertEqual(recovered["semantic_units"][0]["action_indexes"], [0, 1, 2, 3, 4])
        provider.complete.assert_called_once()

    def test_reconstructed_partition_cannot_orphan_an_immutable_action(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _reconstruct_missing_semantic_units

        source = "Use BNB with mixed RPC."
        clauses = segment_user_turn(source)
        actions = [
            {"type": "choose_chain", "chain_text": "BNB", "source_evidence": "BNB"},
            {"type": "set_rpc_mode", "rpc_mode": "mixed", "source_evidence": "mixed RPC"},
        ]
        original = {
            "actions": actions,
            "semantic_units": [{
                "unit_id": "unit-1",
                "clause_id": clauses[0].clause_id,
                "source_text": "BNB",
                "disposition": "action",
                "action_indexes": [0],
                "reason": "lossy partition",
            }],
        }
        invalid_reconstruction = SimpleNamespace(text=json.dumps({
            "semantic_units": [{
                "unit_id": "unit-1",
                "clause_id": clauses[0].clause_id,
                "source_text": source,
                "disposition": "action",
                "action_indexes": [0],
                "reason": "incorrectly drops the mixed action",
            }],
        }))
        provider = Mock()
        provider.complete.side_effect = [invalid_reconstruction, invalid_reconstruction]

        recovered_text = _reconstruct_missing_semantic_units(provider, json.dumps(original), clauses)

        self.assertEqual(json.loads(recovered_text), original)
        self.assertEqual(provider.complete.call_count, 2)

    def test_reconstructed_partition_rejects_non_source_text(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _reconstruct_missing_semantic_units

        source = "Use BNB with mixed RPC."
        clauses = segment_user_turn(source)
        action = {"type": "choose_chain", "chain_text": "BNB", "source_evidence": "BNB"}
        original = {
            "actions": [action],
            "semantic_units": [{
                "unit_id": "unit-1",
                "clause_id": clauses[0].clause_id,
                "source_text": "BNB",
                "disposition": "action",
                "action_indexes": [0],
                "reason": "lossy partition",
            }],
        }
        invalid_reconstruction = SimpleNamespace(text=json.dumps({
            "semantic_units": [{
                "unit_id": "unit-1",
                "clause_id": clauses[0].clause_id,
                "source_text": "Use another chain.",
                "disposition": "action",
                "action_indexes": [0],
                "reason": "not exact source text",
            }],
        }))
        provider = Mock()
        provider.complete.side_effect = [invalid_reconstruction, invalid_reconstruction]

        recovered_text = _reconstruct_missing_semantic_units(provider, json.dumps(original), clauses)

        self.assertEqual(json.loads(recovered_text), original)
        self.assertEqual(provider.complete.call_count, 2)

    def test_existing_semantic_units_are_never_reconstructed(self) -> None:
        import json
        from unittest.mock import Mock

        from agent.harness.intent import _reconstruct_missing_semantic_units

        clauses = segment_user_turn("hello")
        payload = {
            "actions": [{"type": "greeting", "source_evidence": "hello"}],
            "semantic_units": [_unit(clauses[0], 1, [0])],
        }
        provider = Mock()
        original = json.dumps(payload)

        self.assertEqual(
            _reconstruct_missing_semantic_units(provider, original, clauses),
            original,
        )
        provider.complete.assert_not_called()

    def test_malformed_semantic_unit_metadata_is_reconstructed_without_changing_actions(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _reconstruct_missing_semantic_units

        source = "CLOUD_REGION=asia-east1\nAsk for the remaining required values."
        clauses = segment_user_turn(source)
        actions = [{
            "type": "propose_config_values",
            "config_values": {"CLOUD_REGION": "asia-east1"},
            "source_format": "env",
            "source_evidence": "CLOUD_REGION=asia-east1",
        }]
        malformed = {
            "actions": actions,
            "semantic_units": [{
                "clause_id": clauses[0].clause_id,
                "source_text": clauses[0].text,
                "disposition": "action",
                "action_indexes": [0],
                "reason": "missing unit identity",
            }],
        }
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "semantic_units": [
                {
                    "unit_id": f"unit-{index}",
                    "clause_id": clause.clause_id,
                    "source_text": clause.text,
                    "disposition": "action",
                    "action_indexes": [0],
                    "reason": "proposal plus continuation scope",
                }
                for index, clause in enumerate(clauses, start=1)
            ],
        }))

        recovered = json.loads(_reconstruct_missing_semantic_units(
            provider,
            json.dumps(malformed),
            clauses,
        ))

        self.assertEqual(recovered["actions"], actions)
        self.assertTrue(all(unit.get("unit_id") for unit in recovered["semantic_units"]))
        provider.complete.assert_called_once()

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

    def test_explicit_group_navigation_does_not_require_destination_values(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _validate_semantic_fulfillment
        from agent.harness.state import new_state

        text = "I want to configure QPS before the environment values"
        clauses = segment_user_turn(text)
        payload = {
            "actions": [{
                "type": "change_group",
                "group": "qps_profile",
                "navigation_explicit": True,
                "source_evidence": text,
            }],
            "semantic_units": [_unit(clauses[0], 1, [0])],
        }
        provider = Mock()
        provider.complete.side_effect = [
            SimpleNamespace(text=json.dumps({
                "reviews": [{"action_index": 0, "supported": True, "reason": "explicit navigation"}],
            })),
            SimpleNamespace(text=json.dumps({
                "unit_reviews": [{
                    "unit_id": "unit-1",
                    "complete": True,
                    "missing_demand_quote": "",
                    "reason": "navigation intentionally opens later typed intake",
                }],
            })),
        ]
        state = new_state("unit-thread", language="en")
        state["pending_question"] = {
            "id": "CLOUD_REGION",
            "kind": "manual_value",
            "field": "CLOUD_REGION",
            "options": [],
        }

        result = _validate_semantic_fulfillment(provider, json.dumps(payload), clauses, state)

        self.assertTrue(result.valid, result.errors)
        self.assertEqual(provider.complete.call_count, 2)

    def test_specific_qps_adjustment_cannot_be_laundered_as_group_navigation(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _validate_semantic_fulfillment
        from agent.harness.state import new_state

        text = "I want to adjust QPS before choosing workload defaults"
        clauses = segment_user_turn(text)
        payload = {
            "actions": [{
                "type": "change_group",
                "group": "qps_profile",
                "navigation_explicit": True,
                "source_evidence": text,
            }],
            "semantic_units": [_unit(clauses[0], 1, [0])],
        }
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "reviews": [{
                "action_index": 0,
                "supported": False,
                "reason": "specific QPS customization needs its domain intake action",
            }],
            "unit_reviews": [],
        }))

        result = _validate_semantic_fulfillment(
            provider,
            json.dumps(payload),
            clauses,
            new_state("unit-thread", language="en"),
        )

        self.assertFalse(result.valid)
        self.assertIn("semantic fulfilment failed", "\n".join(result.errors))

    def test_unit_review_cannot_treat_unrelated_pending_question_as_required(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _validate_semantic_fulfillment
        from agent.harness.state import new_state

        text = "use real-node instead"
        clauses = segment_user_turn(text)
        payload = {
            "actions": [{
                "type": "choose_target_mode",
                "target_mode": "real-node",
                "target_mode_explicit": True,
                "source_evidence": text,
            }],
            "semantic_units": [_unit(clauses[0], 1, [0])],
        }
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "reviews": [{"action_index": 0, "supported": True, "reason": "explicit mode change"}],
            "unit_reviews": [{"unit_id": "unit-1", "complete": True, "reason": "complete"}],
        }))
        state = new_state("unit-thread", language="en")
        state["pending_question"] = {
            "id": "CLOUD_REGION",
            "kind": "manual_value",
            "field": "CLOUD_REGION",
            "options": [],
        }

        result = _validate_semantic_fulfillment(provider, json.dumps(payload), clauses, state)

        self.assertTrue(result.valid, result.errors)
        review_payload = json.loads(provider.complete.call_args_list[0].args[0].messages[1].content)
        self.assertEqual(review_payload["pending_question"], {})

    def test_negative_action_purpose_review_can_reverse_a_false_negative(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _adjudicate_rejected_action_reviews

        review = {
            "action_index": 0,
            "operation_arguments": {"group": "qps_profile"},
            "declared_purpose": "Temporarily route to a named workflow group only for an explicit navigation request.",
            "source_units": ["I want to configure QPS first before the rest"],
        }
        first = [{
            "action_index": 0,
            "supported": False,
            "reason": "configure must mean value customization",
        }]
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "reviews": [{
                "action_index": 0,
                "supported": True,
                "reason": "the source names the area but supplies no specific value mutation",
            }],
        }))

        rows = _adjudicate_rejected_action_reviews(
            provider,
            action_reviews=[review],
            first_rows=first,
        )

        self.assertTrue(rows[0]["supported"])

    def test_negative_action_purpose_readjudication_preserves_genuine_rejection(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _adjudicate_rejected_action_reviews

        review = {
            "action_index": 0,
            "operation_arguments": {"group": "qps_profile"},
            "declared_purpose": "Temporarily route to a named workflow group only for an explicit navigation request.",
            "source_units": ["Set INITIAL_QPS to 500"],
        }
        first = [{"action_index": 0, "supported": False, "reason": "specific mutation"}]
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "reviews": [{"action_index": 0, "supported": False, "reason": "specific mutation"}],
        }))

        rows = _adjudicate_rejected_action_reviews(
            provider,
            action_reviews=[review],
            first_rows=first,
        )

        self.assertFalse(rows[0]["supported"])

    def test_duplicate_action_purpose_readjudication_remains_fail_closed(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _adjudicate_rejected_action_reviews

        review = {
            "action_index": 0,
            "operation_arguments": {"group": "qps_profile"},
            "declared_purpose": "Navigate to QPS without changing values.",
            "source_units": ["Visit QPS first"],
        }
        first = [{"action_index": 0, "supported": False, "reason": "first rejection"}]
        duplicate = {"action_index": 0, "supported": True, "reason": "duplicate"}
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "reviews": [duplicate, duplicate],
        }))

        rows = _adjudicate_rejected_action_reviews(
            provider,
            action_reviews=[review],
            first_rows=first,
        )

        self.assertEqual(rows, first)

    def test_answer_pending_review_receives_typed_pending_contract(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _validate_semantic_fulfillment
        from agent.harness.state import new_state

        text = "Y"
        clauses = segment_user_turn(text)
        payload = {
            "actions": [{"type": "answer_pending", "answer": "Y", "source_evidence": text}],
            "semantic_units": [_unit(clauses[0], 1, [0])],
        }
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "reviews": [{"action_index": 0, "supported": True, "reason": "exact answer"}],
            "unit_reviews": [{"unit_id": "unit-1", "complete": True, "reason": "complete"}],
        }))
        state = new_state("unit-thread", language="en")
        state["pending_question"] = {
            "id": "target_mode_change_confirm",
            "kind": "yes_no",
            "field": "",
            "options": [
                {"id": "yes", "label": "Y", "value": "confirm"},
                {"id": "no", "label": "N", "value": "cancel"},
            ],
        }

        result = _validate_semantic_fulfillment(provider, json.dumps(payload), clauses, state)

        self.assertTrue(result.valid, result.errors)
        review_payload = json.loads(provider.complete.call_args_list[0].args[0].messages[1].content)
        self.assertEqual(review_payload["pending_question"]["id"], "target_mode_change_confirm")
        self.assertEqual(review_payload["pending_question"]["options"][0]["value"], "confirm")

    def test_specialized_chain_extractor_recovers_compound_plan_omission(self) -> None:
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _recover_omitted_chain_selection, _parse_json_object
        from agent.harness.state import new_state

        source = "I want to test BNB with fake-node"
        payload = {
            "actions": [{
                "type": "choose_target_mode",
                "target_mode": "fake-node",
                "target_mode_explicit": True,
                "source_evidence": source,
            }],
            "semantic_units": [{
                "unit_id": "unit-1",
                "clause_id": "clause-1",
                "source_text": source,
                "disposition": "action",
                "action_indexes": [0],
            }],
        }
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=(
            '{"found":true,"chain_text":"BNB","confidence":"high",'
            '"reason":"explicit benchmark target"}'
        ))

        recovered = _parse_json_object(_recover_omitted_chain_selection(
            provider,
            __import__("json").dumps(payload),
            new_state("unit-thread", language="en"),
        ))

        self.assertEqual([item["type"] for item in recovered["actions"]], ["choose_target_mode", "choose_chain"])
        self.assertEqual(recovered["actions"][1]["chain_text"], "BNB")
        self.assertEqual(recovered["semantic_units"][0]["action_indexes"], [0, 1])

    def test_chain_extractor_recovers_after_semantic_pending_mode_selection(self) -> None:
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _parse_json_object, _recover_omitted_chain_selection
        from agent.harness.state import new_state

        source = "I want to test BNB with fake-node"
        payload = {
            "actions": [{
                "type": "answer_pending",
                "selected_value": "fake-node",
                "source_evidence": source,
            }],
            "semantic_units": [{
                "unit_id": "unit-1",
                "clause_id": "clause-1",
                "source_text": source,
                "disposition": "action",
                "action_indexes": [0],
            }],
        }
        state = new_state("unit-thread", language="en")
        state["pending_question"] = {
            "id": "opening_next_action",
            "kind": "numbered_choice",
            "options": [{
                "id": "1",
                "value": "fake-node",
                "action": {
                    "type": "choose_target_mode",
                    "target_mode": "fake-node",
                    "target_mode_explicit": True,
                },
            }],
        }
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=(
            '{"found":true,"chain_text":"BNB","confidence":"high",'
            '"reason":"explicit benchmark target"}'
        ))

        recovered = _parse_json_object(_recover_omitted_chain_selection(
            provider,
            __import__("json").dumps(payload),
            state,
        ))

        self.assertEqual([item["type"] for item in recovered["actions"]], ["answer_pending", "choose_chain"])
        self.assertEqual(recovered["actions"][1]["chain_text"], "BNB")
        self.assertEqual(recovered["semantic_units"][0]["action_indexes"], [0, 1])

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
        from agent.harness.intent import _validate_action_document
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

        result = _validate_action_document(__import__("json").dumps(payload), clauses, state)

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

    def test_semantic_admission_rejects_named_method_demand_laundered_as_entry(self) -> None:
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _validate_semantic_fulfillment
        from agent.harness.state import new_state

        clauses = segment_user_turn("Add eth_accounts as my custom RPC method.")
        payload = {
            "actions": [{
                "type": "rpc_catalog_command",
                "catalog_command": "enter",
                "source_evidence": clauses[0].text,
            }],
            "semantic_units": [_unit(clauses[0], 1, [0])],
        }
        provider = Mock()
        provider.complete.side_effect = [
            SimpleNamespace(text=json.dumps({
                "reviews": [{
                    "action_index": 0,
                    "supported": True,
                    "reason": "the source requests custom-RPC intake",
                }],
            })),
            SimpleNamespace(text=json.dumps({
                "unit_reviews": [{
                    "unit_id": "unit-1",
                    "complete": False,
                    "missing_demand_quote": "eth_accounts",
                    "reason": "the named wire method is not preserved",
                }],
            })),
            SimpleNamespace(text=json.dumps({
                "unit_reviews": [{
                    "unit_id": "unit-1",
                    "complete": False,
                    "missing_demand_quote": "eth_accounts",
                    "reason": "independent review confirms the omission",
                }],
            })),
        ]

        result = _validate_semantic_fulfillment(
            provider,
            json.dumps(payload),
            clauses,
            new_state("named-method-demand", language="en"),
        )

        self.assertFalse(result.valid)
        self.assertTrue(any("independent review confirms the omission" in error for error in result.errors))

    def test_semantic_admission_accepts_replaced_default_as_contextual_literal(self) -> None:
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _validate_semantic_fulfillment
        from agent.harness.state import new_state

        clauses = segment_user_turn(
            "Keep this single, but use my own RPC method instead of eth_getBalance."
        )
        payload = {
            "actions": [{
                "type": "rpc_catalog_command",
                "catalog_command": "enter",
                "source_evidence": clauses[0].text,
            }],
            "semantic_units": [_unit(clauses[0], 1, [0])],
        }
        provider = Mock()
        provider.complete.side_effect = [
            SimpleNamespace(text=json.dumps({
                "reviews": [{
                    "action_index": 0,
                    "supported": True,
                    "reason": "the source selects custom-RPC intake",
                }],
            })),
            SimpleNamespace(text=json.dumps({
                "unit_reviews": [{
                    "unit_id": "unit-1",
                    "complete": True,
                    "missing_demand_quote": "",
                    "reason": "the old default is comparison context, not a selected method",
                }],
            })),
        ]

        result = _validate_semantic_fulfillment(
            provider,
            json.dumps(payload),
            clauses,
            new_state("replaced-default-literal", language="en"),
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

    def test_semantic_unit_partition_rejects_a_source_gap(self) -> None:
        clauses = segment_user_turn("configure quick then disable metrics")
        text = clauses[0].text
        result = validate_plan_coverage(
            {
                "actions": [{"type": "set_qps_mode", "qps_mode": "quick"}],
                "semantic_units": [{
                    "unit_id": "unit-1",
                    "clause_id": clauses[0].clause_id,
                    "start": 0,
                    "end": text.index(" then"),
                    "source_text": text[:text.index(" then")],
                    "disposition": "action",
                    "action_indexes": [0],
                    "reason": "qps",
                }],
            },
            clauses,
        )
        self.assertFalse(result.valid)
        self.assertTrue(any("omitted prose" in error for error in result.errors))

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

    def test_consultation_scope_accepts_only_turn_local_actions(self) -> None:
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
        self.assertTrue(any("durable action" in error for error in rejected.errors))

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


    def test_compound_unit_review_receives_shared_action_relationship_context(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _validate_semantic_fulfillment
        from agent.harness.state import new_state

        text = (
            "These are deployment notes. "
            "Use Ethereum with CLOUD_REGION=asia-east1."
        )
        clauses = segment_user_turn(text)
        payload = {
            "actions": [
                {
                    "type": "choose_chain",
                    "chain_text": "ethereum",
                    "chain_candidates": ["ethereum"],
                    "source_evidence": clauses[1].text,
                },
                {
                    "type": "propose_config_values",
                    "config_values": {"CLOUD_REGION": "asia-east1"},
                    "unmapped_values": {},
                    "source_format": "env",
                    "source_evidence": clauses[1].text,
                },
            ],
            "semantic_units": [
                _unit(clauses[0], 1, [0, 1]),
                _unit(clauses[1], 2, [0, 1]),
            ],
            "chain_selection_admissions": [0],
        }
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "reviews": [{
                "action_index": 1,
                "supported": True,
                "reason": "the mapped value is source grounded",
            }],
            "unit_reviews": [
                {"unit_id": "unit-1", "complete": True, "reason": "framing"},
                {"unit_id": "unit-2", "complete": True, "reason": "facts preserved"},
            ],
        }))

        result = _validate_semantic_fulfillment(
            provider,
            json.dumps(payload),
            clauses,
            new_state("related-unit-context", language="en"),
        )

        self.assertTrue(result.valid, result.errors)
        request = provider.complete.call_args.args[0]
        review_payload = json.loads(request.messages[1].content)
        by_id = {row["unit_id"]: row for row in review_payload["unit_reviews"]}
        self.assertEqual(
            by_id["unit-1"]["related_source_units"][0]["source_text"],
            clauses[1].text,
        )
        self.assertEqual(
            by_id["unit-2"]["related_source_units"][0]["source_text"],
            clauses[0].text,
        )
        self.assertEqual(provider.complete.call_count, 2)
        action_payload = json.loads(provider.complete.call_args_list[0].args[0].messages[1].content)
        unit_payload = json.loads(provider.complete.call_args_list[1].args[0].messages[1].content)
        self.assertIn("reviews", action_payload)
        self.assertNotIn("unit_reviews", action_payload)
        self.assertIn("unit_reviews", unit_payload)
        self.assertNotIn("reviews", unit_payload)

    def test_malformed_compound_unit_review_does_not_erase_valid_action_reviews(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _validate_semantic_fulfillment
        from agent.harness.state import new_state

        text = "Use Ethereum with CLOUD_REGION=asia-east1."
        clauses = segment_user_turn(text)
        payload = {
            "actions": [
                {
                    "type": "choose_chain",
                    "chain_text": "ethereum",
                    "source_evidence": text,
                },
                {
                    "type": "propose_config_values",
                    "config_values": {"CLOUD_REGION": "asia-east1"},
                    "unmapped_values": {},
                    "source_format": "prose",
                    "source_evidence": text,
                },
            ],
            "semantic_units": [_unit(clauses[0], 1, [0, 1])],
            "chain_selection_admissions": [0],
        }
        provider = Mock()
        provider.complete.side_effect = [
            SimpleNamespace(text=json.dumps({
                "reviews": [{
                    "action_index": 1,
                    "supported": True,
                    "reason": "source-grounded value",
                }],
            })),
            SimpleNamespace(text='{"unit_reviews": ['),
            SimpleNamespace(text='{"unit_reviews": ['),
        ]

        result = _validate_semantic_fulfillment(
            provider,
            json.dumps(payload),
            clauses,
            new_state("independent-semantic-gates", language="en"),
        )

        self.assertFalse(result.valid)
        self.assertTrue(any("semantic unit fulfilment review missing" in row for row in result.errors))
        self.assertFalse(any("semantic fulfilment review missing for action" in row for row in result.errors))

    def test_bounded_action_review_retries_only_missing_or_duplicate_ids(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _request_bounded_semantic_rows

        rows = [
            {"action_index": 2, "declared_purpose": "first", "source_units": ["a"]},
            {"action_index": 7, "declared_purpose": "second", "source_units": ["b"]},
        ]
        provider = Mock()
        provider.complete.side_effect = [
            SimpleNamespace(text=json.dumps({
                "reviews": [
                    {"action_index": 2, "supported": True, "reason": "valid"},
                    {"action_index": 7, "supported": True, "reason": "duplicate one"},
                    {"action_index": 7, "supported": False, "reason": "duplicate two"},
                ],
            })),
            SimpleNamespace(text=json.dumps({
                "reviews": [
                    {"action_index": 7, "supported": True, "reason": "valid retry"},
                ],
            })),
        ]

        result = _request_bounded_semantic_rows(
            provider,
            review_kind="actions",
            rows=rows,
            pending_question={},
        )

        self.assertEqual([row["action_index"] for row in result], [2, 7])
        retry_payload = json.loads(provider.complete.call_args_list[1].args[0].messages[1].content)
        self.assertEqual(
            [row["action_index"] for row in retry_payload["reviews"]],
            [7],
        )


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

    def test_compound_unresolved_unit_is_partitioned_without_routing(self) -> None:
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _decompose_unresolved_semantic_units

        text = "我要测试 BNB，用 mixed，QPS quick，并开启本地 Grafana"
        clauses = segment_user_turn(text)
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "partitions": [{
                "unit_id": "compound-unit",
                "source_units": ["我要测试 BNB", "用 mixed", "QPS quick", "并开启本地 Grafana"],
                "reason": "four independent configuration demands",
            }],
        }))

        result_text, changed = _decompose_unresolved_semantic_units(
            provider,
            json.dumps(self._unresolved_plan(clauses[0])),
            clauses,
        )
        result = json.loads(result_text)

        self.assertTrue(changed)
        self.assertEqual(result["actions"], [])
        self.assertEqual(
            [unit["source_text"] for unit in result["semantic_units"]],
            ["我要测试 BNB", "用 mixed", "QPS quick", "并开启本地 Grafana"],
        )
        self.assertTrue(all(unit["disposition"] == "unresolved" for unit in result["semantic_units"]))

    def test_empty_planner_document_preserves_every_authoritative_clause_as_unresolved(self) -> None:
        from unittest.mock import Mock

        from agent.harness.intent import _reconstruct_missing_semantic_units

        text = "benchmark BNB and enable local Grafana\nQPS quick"
        clauses = segment_user_turn(text)
        provider = Mock()

        result = json.loads(_reconstruct_missing_semantic_units(provider, "{}", clauses))

        self.assertEqual(result["actions"], [])
        self.assertEqual(
            [unit["source_text"] for unit in result["semantic_units"]],
            [clause.text for clause in clauses],
        )
        self.assertTrue(all(unit["disposition"] == "unresolved" for unit in result["semantic_units"]))
        provider.complete.assert_not_called()

    def test_malformed_empty_planner_output_preserves_source_without_inventing_actions(self) -> None:
        from unittest.mock import Mock

        from agent.harness.intent import _reconstruct_missing_semantic_units

        clauses = segment_user_turn("我要测试 BNB，用 mixed")
        provider = Mock()

        result = json.loads(_reconstruct_missing_semantic_units(provider, "not-json", clauses))

        self.assertEqual(result["actions"], [])
        self.assertEqual(result["semantic_units"][0]["source_text"], clauses[0].text)
        self.assertEqual(result["semantic_units"][0]["action_indexes"], [])
        provider.complete.assert_not_called()

    def test_reordered_english_demands_use_the_same_partition_contract(self) -> None:
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _decompose_unresolved_semantic_units

        text = "Enable local Grafana, benchmark BNB, keep QPS quick, and use mixed RPC mode"
        clauses = segment_user_turn(text)
        anchors = ["Enable local Grafana", "benchmark BNB", "keep QPS quick", "and use mixed RPC mode"]
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "partitions": [{
                "unit_id": "compound-unit",
                "source_units": anchors,
                "reason": "independent reordered demands",
            }],
        }))

        result_text, changed = _decompose_unresolved_semantic_units(
            provider,
            json.dumps(self._unresolved_plan(clauses[0])),
            clauses,
        )

        self.assertTrue(changed)
        self.assertEqual(
            [unit["source_text"] for unit in json.loads(result_text)["semantic_units"]],
            anchors,
        )

    def test_incomplete_partition_is_rejected_without_changing_the_plan(self) -> None:
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _decompose_unresolved_semantic_units

        text = "benchmark BNB and enable local Grafana"
        clauses = segment_user_turn(text)
        original = json.dumps(self._unresolved_plan(clauses[0]))
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "partitions": [{
                "unit_id": "compound-unit",
                "source_units": ["benchmark BNB", "local Grafana"],
                "reason": "incorrectly omitted the enable instruction",
            }],
        }))

        result_text, changed = _decompose_unresolved_semantic_units(provider, original, clauses)

        self.assertFalse(changed)
        self.assertEqual(json.loads(result_text), json.loads(original))
        self.assertEqual(provider.complete.call_count, 2)

    def test_invalid_first_partition_retries_before_owner_recovery(self) -> None:
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _decompose_unresolved_semantic_units

        text = "benchmark BNB and enable local Grafana"
        clauses = segment_user_turn(text)
        provider = Mock()
        provider.complete.side_effect = [
            SimpleNamespace(text=json.dumps({
                "partitions": [{
                    "unit_id": "compound-unit",
                    "source_units": ["benchmark BNB", "local Grafana"],
                    "reason": "incomplete first attempt",
                }],
            })),
            SimpleNamespace(text=json.dumps({
                "partitions": [{
                    "unit_id": "compound-unit",
                    "source_units": ["benchmark BNB", "and enable local Grafana"],
                    "reason": "complete second attempt",
                }],
            })),
        ]

        result_text, changed = _decompose_unresolved_semantic_units(
            provider,
            json.dumps(self._unresolved_plan(clauses[0])),
            clauses,
        )

        self.assertTrue(changed)
        self.assertEqual(provider.complete.call_count, 2)
        self.assertEqual(
            [unit["source_text"] for unit in json.loads(result_text)["semantic_units"]],
            ["benchmark BNB", "and enable local Grafana"],
        )

    def test_all_invalid_partition_attempts_leave_original_plan_unchanged(self) -> None:
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _decompose_unresolved_semantic_units

        text = "benchmark BNB and enable local Grafana"
        clauses = segment_user_turn(text)
        original = json.dumps(self._unresolved_plan(clauses[0]))
        provider = Mock()
        provider.complete.side_effect = [
            SimpleNamespace(text="{}"),
            SimpleNamespace(text=json.dumps({
                "partitions": [{
                    "unit_id": "compound-unit",
                    "source_units": ["benchmark BNB", "local Grafana"],
                    "reason": "still incomplete",
                }],
            })),
        ]

        result_text, changed = _decompose_unresolved_semantic_units(provider, original, clauses)

        self.assertFalse(changed)
        self.assertEqual(provider.complete.call_count, 2)
        self.assertEqual(json.loads(result_text), json.loads(original))

    def test_partitioned_demands_recover_through_four_registered_owners(self) -> None:
        from types import SimpleNamespace
        from unittest.mock import Mock, patch

        from agent.harness.intent import (
            _decompose_unresolved_semantic_units,
            _recover_registry_bounded_semantic_actions,
        )
        from agent.harness.state import new_state

        text = "我要测试 BNB，用 mixed，QPS quick，并开启本地 Grafana"
        clauses = segment_user_turn(text)
        partition_provider = Mock()
        partition_provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "partitions": [{
                "unit_id": "compound-unit",
                "source_units": ["我要测试 BNB", "用 mixed", "QPS quick", "并开启本地 Grafana"],
                "reason": "four independent demands",
            }],
        }))
        decomposed, changed = _decompose_unresolved_semantic_units(
            partition_provider,
            json.dumps(self._unresolved_plan(clauses[0])),
            clauses,
        )
        self.assertTrue(changed)

        decisions = [
            {
                "unit_id": f"compound-unit.part-{index}",
                "disposition": "owner_mutation",
                "group": group,
                "consultation_topic": "",
                "target_mode": "",
                "turn_local_action_type": "",
                "existing_action_index": None,
                "evidence_quote": quote,
                "reason": "registered owner demand",
            }
            for index, (group, quote) in enumerate(
                [
                    ("chain_identity", "BNB"),
                    ("workload_rpc", "mixed"),
                    ("qps_profile", "QPS quick"),
                    ("observability", "本地 Grafana"),
                ],
                start=1,
            )
        ]
        recovery_provider = Mock()
        recovery_provider.complete.return_value = SimpleNamespace(text=json.dumps({"decisions": decisions}))

        def owned_action(_provider, group, source):
            return {
                "chain_identity": {"type": "choose_chain", "chain_text": "BNB", "source_evidence": source},
                "workload_rpc": {"type": "set_rpc_mode", "rpc_mode": "mixed", "mutation_explicit": True, "source_evidence": source},
                "qps_profile": {"type": "set_qps_mode", "qps_mode": "quick", "mutation_explicit": True, "source_evidence": source},
                "observability": {"type": "set_observability", "observability_mode": "local", "mutation_explicit": True, "source_evidence": source},
            }[group]

        with patch("agent.harness.intent._resolve_owned_group_mutation", side_effect=owned_action):
            recovered_text, recovered = _recover_registry_bounded_semantic_actions(
                recovery_provider,
                decomposed,
                clauses,
                new_state("compound-recovery", language="zh"),
                text,
            )
        payload = json.loads(recovered_text)

        self.assertTrue(recovered)
        self.assertEqual(
            [action["type"] for action in payload["actions"]],
            ["choose_chain", "set_rpc_mode", "set_qps_mode", "set_observability"],
        )
        self.assertTrue(all(unit["disposition"] == "action" for unit in payload["semantic_units"]))

    def test_unsplit_compound_unit_recovers_multiple_owner_actions_and_mode_barrier(self) -> None:
        from types import SimpleNamespace
        from unittest.mock import Mock, patch

        from agent.harness.intent import _recover_registry_bounded_semantic_actions
        from agent.harness.state import new_state

        text = "我要测试 BNB，用 mixed，QPS quick，并开启本地 Grafana"
        clauses = segment_user_turn(text)
        decisions = [{
            "unit_id": "compound-unit",
            "disposition": "unresolved_target_mode",
            "group": "",
            "consultation_topic": "",
            "target_mode": "",
            "turn_local_action_type": "",
            "existing_action_index": None,
            "evidence_quote": "我要测试 BNB",
            "reason": "benchmark requested without selecting its target mode",
        }]
        decisions.extend({
            "unit_id": "compound-unit",
            "disposition": "owner_mutation",
            "group": group,
            "consultation_topic": "",
            "target_mode": "",
            "turn_local_action_type": "",
            "existing_action_index": None,
            "evidence_quote": quote,
            "reason": "independent registered owner demand",
        } for group, quote in [
            ("chain_identity", "BNB"),
            ("workload_rpc", "mixed"),
            ("qps_profile", "QPS quick"),
            ("observability", "本地 Grafana"),
        ])
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(
            text=json.dumps({"decisions": decisions})
        )

        def owned_action(_provider, group, source):
            return {
                "chain_identity": {"type": "choose_chain", "chain_text": "BNB", "source_evidence": source},
                "workload_rpc": {"type": "set_rpc_mode", "rpc_mode": "mixed", "mutation_explicit": True, "source_evidence": source},
                "qps_profile": {"type": "set_qps_mode", "qps_mode": "quick", "mutation_explicit": True, "source_evidence": source},
                "observability": {"type": "set_observability", "observability_mode": "local", "mutation_explicit": True, "source_evidence": source},
            }[group]

        with patch("agent.harness.intent._resolve_owned_group_mutation", side_effect=owned_action):
            recovered_text, recovered = _recover_registry_bounded_semantic_actions(
                provider,
                json.dumps(self._unresolved_plan(clauses[0])),
                clauses,
                new_state("unsplit-compound-recovery", language="zh"),
                text,
            )
        payload = json.loads(recovered_text)

        self.assertTrue(recovered)
        self.assertEqual(
            [action["type"] for action in payload["actions"]],
            [
                "request_target_mode_selection",
                "choose_chain",
                "set_rpc_mode",
                "set_qps_mode",
                "set_observability",
            ],
        )
        self.assertEqual(payload["semantic_units"][0]["action_indexes"], [0, 1, 2, 3, 4])
        self.assertEqual(payload["semantic_units"][0]["disposition"], "action")


class RegistryBoundedSemanticRecoveryTest(unittest.TestCase):
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

    def test_invalid_consultation_is_recovered_as_one_shared_workload_detour(self) -> None:
        import json

        from agent.harness.intent import _recover_registry_bounded_semantic_actions
        from agent.harness.state import new_state

        user_text = (
            "Before filling in the cloud region, take me to the RPC workload configuration; "
            "I want to decide the request mode first."
        )
        clauses = segment_user_turn(user_text)
        payload = {
            "actions": [{
                "type": "answer_opening_question",
                "topic": "workload",
                "source_evidence": user_text,
            }],
            "semantic_units": [
                self._unit(clauses[0], "unit-1", [0]),
                self._unit(clauses[1], "unit-2", [0]),
            ],
        }
        provider = self._provider([
            {
                "unit_id": "unit-1",
                "disposition": "navigation",
                "group": "workload_rpc",
                "consultation_topic": "",
                "evidence_quote": "RPC workload configuration",
                "reason": "explicit destination",
            },
            {
                "unit_id": "unit-2",
                "disposition": "navigation",
                "group": "workload_rpc",
                "consultation_topic": "",
                "evidence_quote": "request mode",
                "reason": "same explicit destination",
            },
        ])
        state = new_state("recovery-workload", language="en")

        recovered_text, changed = _recover_registry_bounded_semantic_actions(
            provider,
            json.dumps(payload),
            clauses,
            state,
            user_text,
        )
        recovered = json.loads(recovered_text)

        self.assertTrue(changed)
        self.assertEqual(len(recovered["actions"]), 1)
        self.assertEqual(recovered["actions"][0]["type"], "change_group")
        self.assertEqual(recovered["actions"][0]["group"], "workload_rpc")
        self.assertEqual(recovered["actions"][0]["source_evidence"], user_text)
        self.assertEqual(
            [unit["action_indexes"] for unit in recovered["semantic_units"]],
            [[0], [0]],
        )

    def test_rejected_multiline_error_recovers_through_registered_turn_local_action(self) -> None:
        import json

        from agent.harness.intent import _recover_registry_bounded_semantic_actions
        from agent.harness.plan_coverage import PlanCoverageResult
        from agent.harness.state import new_state

        user_text = (
            "Traceback (most recent call last):\n"
            "  File \"blockchain_node_benchmark.sh\", line 42\n"
            "RuntimeError: endpoint timeout while probing eth_blockNumber"
        )
        clauses = segment_user_turn(user_text)
        payload = {
            "actions": [{
                "type": "answer_opening_question",
                "topic": "config_explanation",
                "source_evidence": user_text,
            }],
            "semantic_units": [
                self._unit(clause, f"unit-{index}", [0])
                for index, clause in enumerate(clauses, start=1)
            ],
        }
        provider = self._provider([
            {
                "unit_id": f"unit-{index}",
                "disposition": "turn_local_action",
                "group": "",
                "existing_action_index": None,
                "target_mode": "",
                "consultation_topic": "",
                "turn_local_action_type": "analyze_evidence",
                "evidence_quote": clause.text,
                "reason": "the source is execution evidence",
            }
            for index, clause in enumerate(clauses, start=1)
        ])
        validation = PlanCoverageResult(
            False,
            ("consultation purpose rejected",),
            (),
            rejected_action_indexes=(0,),
            incomplete_unit_ids=tuple(f"unit-{index}" for index in range(1, len(clauses) + 1)),
        )

        recovered_text, changed = _recover_registry_bounded_semantic_actions(
            provider,
            json.dumps(payload),
            clauses,
            new_state("recovery-evidence", language="en"),
            user_text,
            validation,
        )
        recovered = json.loads(recovered_text)

        self.assertTrue(changed)
        self.assertEqual(recovered["actions"], [{"type": "analyze_evidence", "evidence": user_text}])
        self.assertTrue(all(unit["action_indexes"] == [0] for unit in recovered["semantic_units"]))
        request_payload = json.loads(provider.complete.call_args.args[0].messages[1].content)
        self.assertEqual(
            request_payload["recoverable_turn_local_actions"],
            [{
                "type": "analyze_evidence",
                "purpose": "Analyze pasted or previously collected logs/errors/evidence without becoming a deferred workflow command.",
                "source_argument": "evidence",
            }],
        )

    def test_read_only_workload_question_stays_consultation(self) -> None:
        import json

        from agent.harness.intent import _recover_registry_bounded_semantic_actions
        from agent.harness.state import new_state

        user_text = "What is the current default RPC workload?"
        clauses = segment_user_turn(user_text)
        payload = {
            "actions": [{
                "type": "answer_opening_question",
                "topic": "workload",
                "source_evidence": user_text,
            }],
            "semantic_units": [self._unit(clauses[0], "unit-1", [0])],
        }
        provider = self._provider([{
            "unit_id": "unit-1",
            "disposition": "consultation",
            "group": "",
            "consultation_topic": "workload_config",
            "evidence_quote": "default RPC workload",
            "reason": "read-only question",
        }])

        recovered_text, changed = _recover_registry_bounded_semantic_actions(
            provider,
            json.dumps(payload),
            clauses,
            new_state("recovery-consultation", language="en"),
            user_text,
        )
        recovered = json.loads(recovered_text)

        self.assertTrue(changed)
        self.assertEqual(
            recovered["actions"],
            [{
                "type": "answer_opening_question",
                "topic": "workload_config",
                "source_evidence": user_text,
            }],
        )

    def test_unresolved_mode_goal_coexists_with_supported_chain_consultation(self) -> None:
        import json

        from agent.harness.intent import _recover_registry_bounded_semantic_actions
        from agent.harness.plan_coverage import PlanCoverageResult
        from agent.harness.state import new_state

        user_text = (
            "I want to run the framework, but I am not sure whether to use fake-node or real-node; "
            "I may test BNB, or I may just inspect the supported chains."
        )
        clauses = segment_user_turn(user_text)
        payload = {
            "actions": [
                {
                    "type": "answer_pending",
                    "answer": "info",
                    "selected_value": "info",
                    "source_evidence": clauses[0].text,
                },
                {
                    "type": "answer_opening_question",
                    "topic": "supported_chains",
                    "source_evidence": clauses[1].text,
                },
            ],
            "semantic_units": [
                self._unit(clauses[0], "unit-1", [0]),
                self._unit(clauses[1], "unit-2", [1]),
            ],
        }
        validation = PlanCoverageResult(
            False,
            ("action 0 is not a declared pending option",),
            (),
            rejected_action_indexes=(0,),
        )
        provider = self._provider([{
            "unit_id": "unit-1",
            "disposition": "unresolved_target_mode",
            "group": "",
            "consultation_topic": "",
            "evidence_quote": "not sure whether to use fake-node or real-node",
            "reason": "explicit benchmark goal leaves target mode undecided",
        }])

        recovered_text, changed = _recover_registry_bounded_semantic_actions(
            provider,
            json.dumps(payload),
            clauses,
            new_state("recovery-unresolved-mode", language="en"),
            user_text,
            validation,
        )
        recovered = json.loads(recovered_text)

        self.assertTrue(changed)
        self.assertEqual(
            [action["type"] for action in recovered["actions"]],
            ["answer_opening_question", "request_target_mode_selection"],
        )
        self.assertEqual(recovered["actions"][0]["topic"], "supported_chains")
        self.assertNotIn("chain_text", recovered["actions"][1])

    def test_explicit_target_mode_is_recovered_from_a_rejected_intake(self) -> None:
        import json

        from agent.harness.intent import _recover_registry_bounded_semantic_actions
        from agent.harness.plan_coverage import PlanCoverageResult
        from agent.harness.state import new_state

        user_text = "Set up a benchmark against a real node for me."
        clauses = segment_user_turn(user_text)
        payload = {
            "actions": [{
                "type": "request_target_mode_selection",
                "source_evidence": user_text,
            }],
            "semantic_units": [self._unit(clauses[0], "unit-1", [0])],
        }
        validation = PlanCoverageResult(
            False,
            ("action 0 semantic fulfilment failed",),
            (),
            rejected_action_indexes=(0,),
        )
        provider = self._provider([{
            "unit_id": "unit-1",
            "disposition": "explicit_target_mode",
            "group": "",
            "target_mode": "real-node",
            "consultation_topic": "",
            "evidence_quote": "real node",
            "reason": "the source explicitly selects real-node",
        }])

        recovered_text, changed = _recover_registry_bounded_semantic_actions(
            provider,
            json.dumps(payload),
            clauses,
            new_state("recovery-explicit-mode", language="en"),
            user_text,
            validation,
        )

        self.assertTrue(changed)
        self.assertEqual(json.loads(recovered_text)["actions"], [{
            "type": "choose_target_mode",
            "target_mode": "real-node",
            "target_mode_explicit": True,
            "source_evidence": "real node",
        }])

    def test_conflicting_recovered_target_modes_fail_closed(self) -> None:
        import json

        from agent.harness.intent import _recover_registry_bounded_semantic_actions
        from agent.harness.state import new_state

        user_text = "Use fake-node; actually use real-node."
        clauses = segment_user_turn(user_text)
        payload = {
            "actions": [],
            "semantic_units": [
                self._unit(clauses[0], "unit-1", [], disposition="unresolved"),
                self._unit(clauses[1], "unit-2", [], disposition="unresolved"),
            ],
        }
        provider = self._provider([
            {
                "unit_id": "unit-1",
                "disposition": "explicit_target_mode",
                "group": "",
                "target_mode": "fake-node",
                "consultation_topic": "",
                "evidence_quote": "fake-node",
                "reason": "explicit selection",
            },
            {
                "unit_id": "unit-2",
                "disposition": "explicit_target_mode",
                "group": "",
                "target_mode": "real-node",
                "consultation_topic": "",
                "evidence_quote": "real-node",
                "reason": "conflicting explicit selection",
            },
        ])
        original = json.dumps(payload)

        recovered_text, changed = _recover_registry_bounded_semantic_actions(
            provider,
            original,
            clauses,
            new_state("recovery-conflicting-modes", language="en"),
            user_text,
        )

        self.assertFalse(changed)
        self.assertEqual(recovered_text, original)

    def test_explicit_and_unresolved_target_mode_recovery_fail_closed(self) -> None:
        import json

        from agent.harness.intent import _recover_registry_bounded_semantic_actions
        from agent.harness.state import new_state

        user_text = "Run fake-node, although the target mode is still undecided."
        clauses = segment_user_turn(user_text)
        payload = {
            "actions": [],
            "semantic_units": [
                self._unit(clauses[0], "unit-1", [], disposition="unresolved")
            ],
        }
        provider = self._provider([
            {
                "unit_id": "unit-1",
                "disposition": "explicit_target_mode",
                "group": "",
                "target_mode": "fake-node",
                "consultation_topic": "",
                "evidence_quote": "fake-node",
                "reason": "explicit selection",
            },
            {
                "unit_id": "unit-1",
                "disposition": "unresolved_target_mode",
                "group": "",
                "target_mode": "",
                "consultation_topic": "",
                "evidence_quote": "target mode is still undecided",
                "reason": "contradictory unresolved conclusion",
            },
        ])
        original = json.dumps(payload)

        recovered_text, changed = _recover_registry_bounded_semantic_actions(
            provider,
            original,
            clauses,
            new_state("recovery-explicit-unresolved-mode", language="en"),
            user_text,
        )

        self.assertFalse(changed)
        self.assertEqual(recovered_text, original)

    def test_mode_recommendation_without_execution_goal_stays_consultation(self) -> None:
        import json

        from agent.harness.intent import _recover_registry_bounded_semantic_actions
        from agent.harness.state import new_state

        user_text = "What is the difference between fake-node and real-node?"
        clauses = segment_user_turn(user_text)
        payload = {
            "actions": [],
            "semantic_units": [
                self._unit(clauses[0], "unit-1", [], disposition="unresolved")
            ],
        }
        provider = self._provider([{
            "unit_id": "unit-1",
            "disposition": "consultation",
            "group": "",
            "consultation_topic": "mode_comparison",
            "evidence_quote": "difference between fake-node and real-node",
            "reason": "read-only comparison without an execution goal",
        }])

        recovered_text, changed = _recover_registry_bounded_semantic_actions(
            provider,
            json.dumps(payload),
            clauses,
            new_state("recovery-mode-comparison", language="en"),
            user_text,
        )

        self.assertTrue(changed)
        self.assertEqual(
            json.loads(recovered_text)["actions"],
            [{
                "type": "answer_opening_question",
                "topic": "mode_comparison",
                "source_evidence": user_text,
            }],
        )

    def test_generic_resume_restores_flow_only_with_pending_question(self) -> None:
        import json

        from agent.harness.intent import _recover_registry_bounded_semantic_actions
        from agent.harness.state import new_state

        user_text = "Return to the benchmark setup."
        clauses = segment_user_turn(user_text)
        payload = {
            "actions": [],
            "semantic_units": [
                self._unit(clauses[0], "unit-1", [], disposition="unresolved")
            ],
        }
        provider = self._provider([{
            "unit_id": "unit-1",
            "disposition": "generic_resume",
            "group": "",
            "consultation_topic": "",
            "evidence_quote": "benchmark setup",
            "reason": "generic resume",
        }])
        state = new_state("recovery-resume", language="en")
        state["pending_question"] = {"id": "CLOUD_REGION", "group": "provider_deployment"}

        recovered_text, changed = _recover_registry_bounded_semantic_actions(
            provider, json.dumps(payload), clauses, state, user_text
        )

        self.assertTrue(changed)
        self.assertEqual(json.loads(recovered_text)["actions"][0]["type"], "resume_current_flow")

    def test_declared_pending_semantic_recovers_negative_preservation_resume(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _recover_declared_pending_option_semantics
        from agent.harness.state import new_state

        user_text = "Continue the saved setup; do not clear the BNB, mixed, quick, or local Grafana choices."
        clauses = segment_user_turn(user_text)
        payload = {
            "actions": [],
            "semantic_units": [self._unit(clauses[0], "unit-1", [], disposition="unresolved")],
        }
        state = new_state("pending-semantic-negative-resume", language="en")
        state["pending_question"] = {
            "id": "resume_harness_session",
            "prompt": "Resume the saved configuration?",
            "options": [
                {"value": "continue", "semantic_action": "continue_current_flow"},
                {"value": "modify"},
                {"value": "reset"},
            ],
        }
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "matches": [{
                "unit_id": "unit-1",
                "semantic_action": "continue_current_flow",
                "evidence_quote": "Continue the saved setup",
                "reason": "selects the declared continuation option",
            }],
        }))

        recovered_text, changed = _recover_declared_pending_option_semantics(
            provider,
            json.dumps(payload),
            state,
        )
        recovered = json.loads(recovered_text)

        self.assertTrue(changed)
        self.assertEqual(recovered["actions"], [{
            "type": "resume_current_flow",
            "source_evidence": "Continue the saved setup",
        }])
        self.assertEqual(recovered["semantic_units"][0]["action_indexes"], [0])

    def test_declared_pending_semantic_rejects_unregistered_match(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _recover_declared_pending_option_semantics
        from agent.harness.state import new_state

        user_text = "Continue or reset it; I am not sure."
        clauses = segment_user_turn(user_text)
        original = json.dumps({
            "actions": [],
            "semantic_units": [self._unit(clauses[0], "unit-1", [], disposition="unresolved")],
        })
        state = new_state("pending-semantic-conflict", language="en")
        state["pending_question"] = {
            "id": "semantic-choice",
            "options": [
                {"value": "continue", "semantic_action": "continue_current_flow"},
                {"value": "other", "semantic_action": "other_semantic"},
            ],
        }
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "matches": [{
                "unit_id": "unit-1",
                "semantic_action": "other_semantic",
                "evidence_quote": "reset",
                "reason": "not declared by a registered action",
            }],
        }))

        recovered_text, changed = _recover_declared_pending_option_semantics(
            provider,
            original,
            state,
        )

        self.assertFalse(changed)
        self.assertEqual(recovered_text, original)

    def test_declared_pending_option_recovers_registered_domain_effect(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _recover_declared_pending_option_semantics
        from agent.harness.state import new_state

        user_text = "Keep single mode, but use my own RPC method instead of the default."
        clauses = segment_user_turn(user_text)
        payload = {
            "actions": [],
            "semantic_units": [self._unit(clauses[0], "unit-1", [], disposition="unresolved")],
        }
        state = new_state("pending-domain-option", language="en")
        state["pending_question"] = {
            "id": "workload_confirm",
            "prompt": "Choose the default workload or add a custom RPC method.",
            "options": [
                {
                    "id": "default",
                    "label": "Use defaults",
                    "value": "default",
                    "action": {"type": "use_default_workload"},
                },
                {
                    "id": "custom_rpc",
                    "label": "Add custom RPC method",
                    "value": "custom_rpc",
                    "action": {"type": "rpc_catalog_command", "catalog_command": "enter"},
                },
            ],
        }
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "matches": [{
                "unit_id": "unit-1",
                "option_id": "custom_rpc",
                "evidence_quote": "use my own RPC method instead of the default",
                "reason": "selects the declared custom RPC option",
            }],
        }))

        recovered_text, changed = _recover_declared_pending_option_semantics(
            provider,
            json.dumps(payload),
            state,
        )
        recovered = json.loads(recovered_text)

        self.assertTrue(changed)
        self.assertEqual(recovered["actions"], [{
            "type": "rpc_catalog_command",
            "catalog_command": "enter",
            "source_evidence": "use my own RPC method instead of the default",
        }])
        self.assertEqual(recovered["semantic_units"][0]["action_indexes"], [0])

    def test_declared_pending_option_reuses_existing_equivalent_domain_effect(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _recover_declared_pending_option_semantics
        from agent.harness.plan_coverage import PlanCoverageResult
        from agent.harness.state import new_state

        user_text = "Use my own RPC method instead of the displayed default."
        clauses = segment_user_turn(user_text)
        payload = {
            "actions": [{
                "type": "rpc_catalog_command",
                "catalog_command": "enter",
                "source_evidence": "Use my own RPC method",
            }],
            "semantic_units": [self._unit(clauses[0], "unit-1", [0])],
        }
        state = new_state("pending-domain-option-idempotent", language="en")
        state["pending_question"] = {
            "id": "workload_confirm",
            "options": [{
                "id": "custom_rpc",
                "label": "Add custom RPC method",
                "value": "custom_rpc",
                "action": {"type": "rpc_catalog_command", "catalog_command": "enter"},
            }],
        }
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "matches": [{
                "unit_id": "unit-1",
                "option_id": "custom_rpc",
                "evidence_quote": "Use my own RPC method",
                "reason": "selects the declared custom RPC option",
            }],
        }))

        recovered_text, changed = _recover_declared_pending_option_semantics(
            provider,
            json.dumps(payload),
            state,
            PlanCoverageResult(
                valid=False,
                errors=("semantic unit incomplete",),
                unresolved_clauses=(),
                incomplete_unit_ids=("unit-1",),
            ),
        )
        recovered = json.loads(recovered_text)

        self.assertTrue(changed)
        self.assertEqual(len(recovered["actions"]), 1)
        self.assertEqual(recovered["semantic_units"][0]["action_indexes"], [0])

    def test_declared_pending_option_replaces_rejected_planner_action(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _recover_declared_pending_option_semantics
        from agent.harness.plan_coverage import PlanCoverageResult
        from agent.harness.state import new_state

        source = "这两个都不是，我重新把链名输一遍。"
        clauses = segment_user_turn(source)
        payload = {
            "actions": [{
                "type": "request_chain_selection",
                "chain_candidates": ["bsc", "ethereum"],
                "source_evidence": source,
            }],
            "semantic_units": [self._unit(clauses[0], "unit-1", [0])],
        }
        state = new_state("pending-replace-invalid", language="zh")
        state["pending_question"] = {
            "id": "chain_ambiguity_confirm",
            "prompt": "请选择链。",
            "options": [
                {
                    "id": "bsc",
                    "label": "bsc",
                    "value": "bsc",
                    "action": {"type": "answer_pending", "answer": "bsc"},
                },
                {
                    "id": "reenter",
                    "label": "重新输入",
                    "value": "reenter",
                    "action": {"type": "answer_pending", "answer": "reenter"},
                },
            ],
        }
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "matches": [{
                "unit_id": "unit-1",
                "option_id": "reenter",
                "evidence_quote": "重新把链名输一遍",
                "reason": "selects the declared re-entry option",
            }],
        }))

        recovered_text, changed = _recover_declared_pending_option_semantics(
            provider,
            json.dumps(payload),
            state,
            PlanCoverageResult(
                valid=False,
                errors=("wrong operation",),
                unresolved_clauses=(),
                rejected_action_indexes=(0,),
                incomplete_unit_ids=("unit-1",),
            ),
        )
        recovered = json.loads(recovered_text)

        self.assertTrue(changed)
        self.assertEqual(recovered["actions"], [{
            "type": "answer_pending",
            "answer": "reenter",
            "selected_value": "reenter",
            "source_evidence": "重新把链名输一遍",
        }])
        self.assertEqual(recovered["semantic_units"][0]["action_indexes"], [0])

    def test_declared_pending_contract_recovers_manual_value_from_prose(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _recover_declared_pending_option_semantics
        from agent.harness.plan_coverage import PlanCoverageResult
        from agent.harness.state import new_state

        source = "Use http://geth-dev:8545 as the real endpoint for this observation."
        clauses = segment_user_turn(source)
        payload = {
            "actions": [{
                "type": "rpc_catalog_command",
                "catalog_command": "set_endpoint",
                "rpc_endpoint": "http://geth-dev:8545",
                "source_evidence": source,
            }],
            "semantic_units": [self._unit(clauses[0], "unit-1", [0])],
        }
        state = new_state("pending-manual-recovery", language="en")
        state["pending_question"] = {
            "id": "SYNC_OBSERVE_RPC_URL",
            "group": "endpoint_process",
            "prompt": "Provide the real sync-observe endpoint.",
            "field": "SYNC_OBSERVE_RPC_URL",
            "kind": "url",
            "manual_input_allowed": True,
            "validation": {},
            "options": [],
        }
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "manual_matches": [{
                "unit_id": "unit-1",
                "answer": "http://geth-dev:8545",
                "evidence_quote": "http://geth-dev:8545",
                "reason": "directly supplies the requested URL",
            }],
        }))

        recovered_text, changed = _recover_declared_pending_option_semantics(
            provider,
            json.dumps(payload),
            state,
            PlanCoverageResult(
                valid=False,
                errors=("wrong pending owner",),
                unresolved_clauses=(),
                rejected_action_indexes=(0,),
                incomplete_unit_ids=("unit-1",),
            ),
        )
        recovered = json.loads(recovered_text)

        self.assertTrue(changed)
        self.assertEqual(recovered["actions"], [{
            "type": "answer_pending",
            "answer": "http://geth-dev:8545",
            "source_evidence": "http://geth-dev:8545",
        }])

    def test_declared_pending_contract_can_bind_supporting_context(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _recover_declared_pending_option_semantics
        from agent.harness.plan_coverage import PlanCoverageResult
        from agent.harness.state import new_state

        source = "Plans changed: don't generate benchmark traffic. I only want to watch the node catch up."
        clauses = segment_user_turn(source)
        payload = {
            "actions": [],
            "semantic_units": [
                self._unit(clauses[0], "unit-context", [], disposition="unresolved"),
                self._unit(clauses[1], "unit-choice", [], disposition="unresolved"),
            ],
        }
        state = new_state("pending-context-recovery", language="en")
        state["pending_question"] = {
            "id": "target_mode_select",
            "prompt": "Choose the target mode.",
            "options": [{
                "id": "sync",
                "label": "sync-observe",
                "value": "sync-observe",
                "action": {
                    "type": "choose_target_mode",
                    "target_mode": "sync-observe",
                    "target_mode_explicit": True,
                },
            }],
        }
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "matches": [{
                "unit_id": "unit-choice",
                "option_id": "sync",
                "evidence_quote": "watch the node catch up",
                "reason": "selects sync observation",
            }],
            "contexts": [{
                "unit_id": "unit-context",
                "supports_unit_id": "unit-choice",
                "evidence_quote": "don't generate benchmark traffic",
                "reason": "rules out load generation in support of sync observation",
            }],
        }))

        recovered_text, changed = _recover_declared_pending_option_semantics(
            provider,
            json.dumps(payload),
            state,
            PlanCoverageResult(
                valid=False,
                errors=("unresolved clauses",),
                unresolved_clauses=tuple(clause.text for clause in clauses),
                incomplete_unit_ids=("unit-context", "unit-choice"),
            ),
        )
        recovered = json.loads(recovered_text)

        self.assertTrue(changed)
        self.assertEqual(recovered["actions"][0]["type"], "choose_target_mode")
        self.assertEqual(recovered["semantic_units"][0]["disposition"], "context")
        self.assertEqual(recovered["semantic_units"][1]["action_indexes"], [0])

    def test_declared_pending_contract_binds_context_to_admitted_sibling(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _recover_declared_pending_option_semantics
        from agent.harness.plan_coverage import PlanCoverageResult
        from agent.harness.state import new_state

        source = "Plans changed: don't generate benchmark traffic. I only want to watch the node catch up."
        clauses = segment_user_turn(source)
        payload = {
            "actions": [{
                "type": "choose_target_mode",
                "target_mode": "sync-observe",
                "target_mode_explicit": True,
                "source_evidence": "watch the node catch up",
            }],
            "target_mode_selection_admissions": [0],
            "semantic_units": [
                self._unit(clauses[0], "unit-context", [], disposition="unresolved"),
                self._unit(clauses[1], "unit-choice", [0]),
            ],
        }
        state = new_state("pending-context-anchor", language="en")
        state["pending_question"] = {
            "id": "target_mode_select",
            "prompt": "Choose the target mode.",
            "options": [{
                "id": "sync-observe",
                "label": "sync-observe",
                "value": "sync-observe",
                "action": {
                    "type": "choose_target_mode",
                    "target_mode": "sync-observe",
                    "target_mode_explicit": True,
                },
            }],
        }
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "contexts": [{
                "unit_id": "unit-context",
                "supports_unit_id": "unit-choice",
                "evidence_quote": "don't generate benchmark traffic",
                "reason": "rules out load generation for the admitted observation mode",
            }],
        }))

        recovered_text, changed = _recover_declared_pending_option_semantics(
            provider,
            json.dumps(payload),
            state,
            PlanCoverageResult(
                valid=False,
                errors=("unresolved clause",),
                unresolved_clauses=(clauses[0].text,),
                incomplete_unit_ids=("unit-context",),
            ),
        )
        recovered = json.loads(recovered_text)

        self.assertTrue(changed)
        self.assertEqual(recovered["actions"], payload["actions"])
        self.assertEqual(recovered["semantic_units"][0]["disposition"], "context")
        self.assertEqual(recovered["semantic_units"][1]["action_indexes"], [0])

    def test_declared_pending_contract_does_not_bind_unrelated_text_to_anchor(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _recover_declared_pending_option_semantics
        from agent.harness.plan_coverage import PlanCoverageResult
        from agent.harness.state import new_state

        source = "Watch the node catch up. Also explain the previous report."
        clauses = segment_user_turn(source)
        payload = {
            "actions": [{
                "type": "choose_target_mode",
                "target_mode": "sync-observe",
                "target_mode_explicit": True,
                "source_evidence": "Watch the node catch up",
            }],
            "target_mode_selection_admissions": [0],
            "semantic_units": [
                self._unit(clauses[0], "unit-choice", [0]),
                self._unit(clauses[1], "unit-independent", [], disposition="unresolved"),
            ],
        }
        state = new_state("pending-context-anchor-negative", language="en")
        state["pending_question"] = {
            "id": "target_mode_select",
            "prompt": "Choose the target mode.",
            "options": [{
                "id": "sync-observe",
                "label": "sync-observe",
                "value": "sync-observe",
                "action": {
                    "type": "choose_target_mode",
                    "target_mode": "sync-observe",
                    "target_mode_explicit": True,
                },
            }],
        }
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({}))
        original = json.dumps(payload)

        recovered_text, changed = _recover_declared_pending_option_semantics(
            provider,
            original,
            state,
            PlanCoverageResult(
                valid=False,
                errors=("unresolved clause",),
                unresolved_clauses=(clauses[1].text,),
                incomplete_unit_ids=("unit-independent",),
            ),
        )

        self.assertFalse(changed)
        self.assertEqual(recovered_text, original)

    def test_declared_pending_contract_reconciles_deferred_conflicting_option(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _recover_declared_pending_option_semantics
        from agent.harness.plan_coverage import PlanCoverageResult
        from agent.harness.state import new_state

        source = "Test BSC first. Leave Ethereum for later."
        clauses = segment_user_turn(source)
        payload = {
            "actions": [
                {
                    "type": "answer_pending",
                    "answer": "bsc",
                    "source_evidence": "Test BSC first",
                },
                {
                    "type": "answer_pending",
                    "answer": "ethereum",
                    "source_evidence": "Ethereum",
                },
            ],
            "semantic_units": [
                self._unit(clauses[0], "unit-bsc", [0]),
                self._unit(clauses[1], "unit-ethereum", [1]),
            ],
        }
        state = new_state("pending-conflict-recovery", language="en")
        state["pending_question"] = {
            "id": "chain_ambiguity_confirm",
            "prompt": "Choose the chain to test first.",
            "options": [
                {
                    "id": "bsc",
                    "label": "bsc",
                    "value": "bsc",
                    "action": {"type": "answer_pending", "answer": "bsc"},
                },
                {
                    "id": "ethereum",
                    "label": "ethereum",
                    "value": "ethereum",
                    "action": {"type": "answer_pending", "answer": "ethereum"},
                },
            ],
        }
        provider = Mock()
        provider.complete.side_effect = [
            SimpleNamespace(text=json.dumps({
                "matches": [
                    {
                        "unit_id": "unit-bsc",
                        "option_id": "bsc",
                        "evidence_quote": "BSC",
                    },
                    {
                        "unit_id": "unit-ethereum",
                        "option_id": "ethereum",
                        "evidence_quote": "Ethereum",
                    },
                ],
            })),
            SimpleNamespace(text=json.dumps({
                "selected_unit_id": "unit-bsc",
                "selected_option_id": "bsc",
                "evidence_quote": "BSC",
                "contexts": [{
                    "unit_id": "unit-ethereum",
                    "evidence_quote": "Leave Ethereum for later",
                    "reason": "defers the unselected alternative",
                }],
                "ambiguous": False,
            })),
        ]

        recovered_text, changed = _recover_declared_pending_option_semantics(
            provider,
            json.dumps(payload),
            state,
            PlanCoverageResult(
                valid=False,
                errors=("conflicting options",),
                unresolved_clauses=(),
                rejected_action_indexes=(0, 1),
                incomplete_unit_ids=("unit-bsc", "unit-ethereum"),
            ),
        )
        recovered = json.loads(recovered_text)

        self.assertFalse(changed)
        self.assertEqual(len(recovered["actions"]), 2)
        self.assertEqual(recovered["semantic_units"][0]["action_indexes"], [0])
        self.assertEqual(recovered["semantic_units"][1]["disposition"], "action")
        self.assertEqual(recovered["semantic_units"][1]["action_indexes"], [1])

    def test_ambiguous_unit_fails_closed_without_rewriting_siblings(self) -> None:
        import json

        from agent.harness.intent import _recover_registry_bounded_semantic_actions
        from agent.harness.state import new_state

        user_text = "Hello; move it somewhere else."
        clauses = segment_user_turn(user_text)
        payload = {
            "actions": [
                {"type": "greeting", "source_evidence": "Hello"},
                {
                    "type": "answer_opening_question",
                    "topic": "somewhere",
                    "source_evidence": "move it somewhere else",
                },
            ],
            "semantic_units": [
                self._unit(clauses[0], "unit-1", [0]),
                self._unit(clauses[1], "unit-2", [1]),
            ],
        }
        provider = self._provider([{
            "unit_id": "unit-2",
            "disposition": "not_group_control",
            "group": "",
            "consultation_topic": "",
            "evidence_quote": "",
            "reason": "no destination",
        }])
        original = json.dumps(payload)

        recovered_text, changed = _recover_registry_bounded_semantic_actions(
            provider,
            original,
            clauses,
            new_state("recovery-ambiguous", language="en"),
            user_text,
        )

        self.assertFalse(changed)
        self.assertEqual(recovered_text, original)

    def test_registered_adjacent_group_detours_are_not_phrase_specific(self) -> None:
        import json

        from agent.harness.intent import _recover_registry_bounded_semantic_actions
        from agent.harness.state import new_state

        cases = (
            ("Take me to QPS tuning.", "qps_profile", "QPS tuning"),
            ("Open observability configuration.", "observability", "observability configuration"),
            ("Go configure the ledger disk.", "ledger_disk", "ledger disk"),
            ("Return to chain selection.", "chain_identity", "chain selection"),
            ("Visit endpoint and process settings.", "endpoint_process", "endpoint and process settings"),
        )
        for index, (user_text, group, quote) in enumerate(cases):
            with self.subTest(group=group):
                clauses = segment_user_turn(user_text)
                payload = {
                    "actions": [],
                    "semantic_units": [
                        self._unit(clauses[0], "unit-1", [], disposition="unresolved")
                    ],
                }
                provider = self._provider([{
                    "unit_id": "unit-1",
                    "disposition": "navigation",
                    "group": group,
                    "consultation_topic": "",
                    "evidence_quote": quote,
                    "reason": "registered destination",
                }])
                recovered_text, changed = _recover_registry_bounded_semantic_actions(
                    provider,
                    json.dumps(payload),
                    clauses,
                    new_state(f"recovery-adjacent-{index}", language="en"),
                    user_text,
                )
                self.assertTrue(changed)
                self.assertEqual(json.loads(recovered_text)["actions"][0]["group"], group)

    def test_semantically_rejected_sibling_is_replaced_without_losing_valid_navigation(self) -> None:
        import json

        from agent.harness.intent import _recover_registry_bounded_semantic_actions
        from agent.harness.plan_coverage import PlanCoverageResult
        from agent.harness.state import new_state

        user_text = (
            "Before filling in the cloud region, take me to the RPC workload configuration; "
            "I want to decide the request mode first."
        )
        clauses = segment_user_turn(user_text)
        payload = {
            "actions": [
                {
                    "type": "change_group",
                    "group": "workload_rpc",
                    "navigation_explicit": True,
                    "source_evidence": clauses[0].text,
                },
                {
                    "type": "set_rpc_mode",
                    "rpc_mode": "single",
                    "mutation_explicit": True,
                    "source_evidence": clauses[1].text,
                },
            ],
            "semantic_units": [
                self._unit(clauses[0], "unit-1", [0]),
                self._unit(clauses[1], "unit-2", [1]),
            ],
        }
        validation = PlanCoverageResult(
            False,
            ("action 1 semantic fulfilment failed",),
            (),
            rejected_action_indexes=(1,),
        )
        provider = self._provider([{
            "unit_id": "unit-2",
            "disposition": "navigation",
            "group": "workload_rpc",
            "consultation_topic": "",
            "evidence_quote": "request mode",
            "reason": "requests the workload choice, not a concrete mode mutation",
        }])

        recovered_text, changed = _recover_registry_bounded_semantic_actions(
            provider,
            json.dumps(payload),
            clauses,
            new_state("recovery-semantic-sibling", language="en"),
            user_text,
            validation,
        )
        recovered = json.loads(recovered_text)

        self.assertTrue(changed)
        self.assertEqual(len(recovered["actions"]), 1)
        self.assertEqual(recovered["actions"][0]["type"], "change_group")
        self.assertEqual(recovered["actions"][0]["source_evidence"], clauses[0].text)
        self.assertEqual(
            [unit["action_indexes"] for unit in recovered["semantic_units"]],
            [[0], [0]],
        )

    def test_temporal_framing_can_share_one_existing_valid_navigation(self) -> None:
        import json

        from agent.harness.intent import _recover_registry_bounded_semantic_actions
        from agent.harness.state import new_state

        user_text = (
            "Hold the current review for a moment. "
            "Configure the QPS profile first, then return to the review."
        )
        clauses = segment_user_turn(user_text)
        payload = {
            "actions": [{
                "type": "change_group",
                "group": "qps_profile",
                "navigation_explicit": True,
                "source_evidence": clauses[1].text,
            }],
            "semantic_units": [
                self._unit(clauses[0], "unit-1", [], disposition="unresolved"),
                self._unit(clauses[1], "unit-2", [0]),
            ],
        }
        provider = self._provider([{
            "unit_id": "unit-1",
            "disposition": "shared_navigation",
            "group": "",
            "existing_action_index": 0,
            "consultation_topic": "",
            "evidence_quote": "Hold the current review for a moment",
            "reason": "temporal framing for the existing QPS detour",
        }])

        recovered_text, changed = _recover_registry_bounded_semantic_actions(
            provider,
            json.dumps(payload),
            clauses,
            new_state("recovery-shared-navigation", language="en"),
            user_text,
        )
        recovered = json.loads(recovered_text)

        self.assertTrue(changed)
        self.assertEqual(len(recovered["actions"]), 1)
        self.assertEqual(recovered["actions"][0]["type"], "change_group")
        self.assertEqual(
            [unit["action_indexes"] for unit in recovered["semantic_units"]],
            [[0], [0]],
        )

    def test_shared_navigation_rejects_missing_or_non_navigation_sibling(self) -> None:
        import json

        from agent.harness.intent import _recover_registry_bounded_semantic_actions
        from agent.harness.state import new_state

        user_text = "Hold this for now. Show the current configuration."
        clauses = segment_user_turn(user_text)
        payload = {
            "actions": [{
                "type": "answer_opening_question",
                "topic": "current_config",
                "source_evidence": clauses[1].text,
            }],
            "semantic_units": [
                self._unit(clauses[0], "unit-1", [], disposition="unresolved"),
                self._unit(clauses[1], "unit-2", [0]),
            ],
        }
        provider = self._provider([{
            "unit_id": "unit-1",
            "disposition": "shared_navigation",
            "group": "",
            "existing_action_index": 0,
            "consultation_topic": "",
            "evidence_quote": "Hold this for now",
            "reason": "invalid attempt to share a consultation",
        }])
        original = json.dumps(payload)

        recovered_text, changed = _recover_registry_bounded_semantic_actions(
            provider,
            original,
            clauses,
            new_state("recovery-no-navigation-sibling", language="en"),
            user_text,
        )

        self.assertFalse(changed)
        self.assertEqual(recovered_text, original)

    def test_owned_mutation_resolver_is_scoped_to_the_exact_group(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _resolve_owned_group_mutation

        provider = Mock()
        provider.complete.return_value = SimpleNamespace(
            text=json.dumps({"action": None, "reason": "no concrete value"})
        )

        self.assertIsNone(
            _resolve_owned_group_mutation(provider, "workload_rpc", "change the request mode")
        )
        request = provider.complete.call_args.args[0]
        payload = json.loads(request.messages[1].content)
        candidate_types = {row["type"] for row in payload["candidate_actions"]}

        self.assertIn("set_rpc_mode", candidate_types)
        self.assertNotIn("choose_target_mode", candidate_types)
        self.assertNotIn("request_target_mode_selection", candidate_types)


if __name__ == "__main__":
    unittest.main()

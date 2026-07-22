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
    def test_natural_boolean_entailment_rejects_unrelated_mutation(self) -> None:
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _recover_declared_pending_option_semantics
        from agent.harness.state import new_state

        source = "I want to benchmark Solana with fake-node."
        clauses = segment_user_turn(source)
        state = new_state("pending-boolean-unrelated", language="en")
        state["pending_question"] = {
            "id": "inferred_config_review",
            "group": "provider_deployment",
            "field": "inferred_config_review",
            "kind": "yes_no",
            "manual_input_allowed": False,
            "prompt": "Apply the inferred configuration values?",
            "options": [
                {"id": "yes", "label": "Y", "value": True, "action": {"type": "answer_pending"}},
                {"id": "no", "label": "N", "value": False, "action": {"type": "answer_pending"}},
            ],
        }
        payload = {
            "actions": [{"type": "clarify_unresolved", "clauses": [source]}],
            "semantic_units": [_unit(clauses[0], 1, [0], disposition="unresolved")],
        }
        false_selection = SimpleNamespace(text=json.dumps({
            "decision": "select_option",
            "option_id": "yes",
            "answer": "",
            "evidence_quote": source,
            "supporting_unit_ids": ["unit-1"],
            "independent_unit_ids": [],
            "reason": "incorrectly inferred agreement from an unrelated request",
        }))
        provider = Mock()
        provider.complete.side_effect = [
            false_selection,
            false_selection,
            SimpleNamespace(text=json.dumps({
                "entailed": False,
                "contradicted": False,
                "evidence_quote": "",
                "reason": "the turn requests another mutation and does not answer the proposition",
            })),
        ]

        recovered, changed = _recover_declared_pending_option_semantics(
            provider,
            json.dumps(payload),
            state,
            clauses=clauses,
            user_text=source,
        )

        self.assertFalse(changed)
        self.assertEqual(recovered, json.dumps(payload))
        self.assertEqual(provider.complete.call_count, 3)

    def test_pending_consensus_preserves_disputed_support_as_independent(self) -> None:
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _recover_declared_pending_option_semantics
        from agent.harness.state import new_state

        source = "Use quick now. Run standard later."
        clauses = segment_user_turn(source)
        state = new_state("pending-partition-consensus", language="en")
        state["pending_question"] = {
            "id": "benchmark_mode",
            "group": "qps_profile",
            "field": "benchmark_mode",
            "kind": "numbered_choice",
            "manual_input_allowed": False,
            "options": [
                {"id": "quick", "label": "quick", "value": "quick", "action": {"type": "answer_pending"}},
                {"id": "standard", "label": "standard", "value": "standard", "action": {"type": "answer_pending"}},
            ],
        }
        payload = {
            "actions": [{"type": "clarify_unresolved", "clauses": [source]}],
            "semantic_units": [
                _unit(clause, index, [0], disposition="unresolved")
                for index, clause in enumerate(clauses, start=1)
            ],
        }
        provider = Mock()
        provider.complete.side_effect = [
            SimpleNamespace(text=json.dumps({
                "decision": "select_option",
                "option_id": "quick",
                "answer": "",
                "evidence_quote": "Use quick now",
                "supporting_unit_ids": ["unit-1", "unit-2"],
                "independent_unit_ids": [],
                "reason": "first adjudicator treats the later run as support",
            })),
            SimpleNamespace(text=json.dumps({
                "decision": "select_option",
                "option_id": "quick",
                "answer": "",
                "evidence_quote": "Use quick now",
                "supporting_unit_ids": ["unit-1"],
                "independent_unit_ids": ["unit-2"],
                "reason": "second adjudicator preserves the later run request",
            })),
        ]

        recovered, changed = _recover_declared_pending_option_semantics(
            provider,
            json.dumps(payload),
            state,
            clauses=clauses,
            user_text=source,
        )
        document = json.loads(recovered)

        self.assertTrue(changed)
        self.assertEqual(provider.complete.call_count, 2)
        self.assertEqual(document["semantic_units"][0]["disposition"], "action")
        self.assertEqual(document["semantic_units"][1]["disposition"], "unresolved")
        self.assertNotIn("pending_context_admissions", document)

    def test_single_structured_assignment_is_not_downgraded_to_prose(self) -> None:
        clauses = segment_user_turn("CLOUD_ZONE: us-west-2b")

        self.assertEqual(len(clauses), 1)
        self.assertEqual(clauses[0].input_shape, "structured")
        self.assertEqual(clauses[0].text, "CLOUD_ZONE: us-west-2b")

    def test_single_error_label_remains_structured_syntax_without_assigning_intent(self) -> None:
        clauses = segment_user_turn("RuntimeError: connection refused")

        self.assertEqual(len(clauses), 1)
        self.assertEqual(clauses[0].input_shape, "structured")

    def test_pending_option_conflicting_verdicts_fail_closed(self) -> None:
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _recover_declared_pending_option_semantics
        from agent.harness.state import new_state

        source = "Yes, keep the detected 100 GiB value; it matches this disk."
        clauses = segment_user_turn(source)
        state = new_state("pending-option-challenge", language="en")
        state["pending_question"] = {
            "id": "detected-size",
            "group": "ledger_disk",
            "field": "DATA_VOL_SIZE",
            "kind": "yes_no",
            "options": [
                {"id": "1", "label": "Y", "value": True, "action": {"type": "answer_pending"}},
                {"id": "2", "label": "N", "value": False, "action": {"type": "answer_pending"}},
            ],
            "manual_input_allowed": False,
        }
        payload = {
            "actions": [{"type": "clarify_unresolved", "clauses": [source]}],
            "semantic_units": [
                _unit(clause, index, [0], disposition="unresolved")
                for index, clause in enumerate(clauses, start=1)
            ],
        }
        provider = Mock()
        provider.complete.side_effect = [
            SimpleNamespace(text=json.dumps({
                "decision": "no_selection",
                "option_id": "",
                "answer": "",
                "evidence_quote": "",
                "supporting_unit_ids": [],
                "independent_unit_ids": ["unit-1"],
                "reason": "first verdict missed the wrapped answer",
            })),
            SimpleNamespace(text=json.dumps({
                "decision": "select_option",
                "option_id": "1",
                "answer": "",
                "evidence_quote": "Yes",
                "supporting_unit_ids": [
                    f"unit-{index}" for index in range(1, len(clauses) + 1)
                ],
                "independent_unit_ids": [],
                "reason": "the explanation supports the declared yes option",
            })),
        ]

        recovered, changed = _recover_declared_pending_option_semantics(
            provider,
            json.dumps(payload),
            state,
            clauses=clauses,
            user_text=source,
        )
        self.assertFalse(changed)
        self.assertEqual(recovered, json.dumps(payload))
        self.assertEqual(provider.complete.call_count, 2)

    def test_pending_manual_value_conflicting_verdicts_fail_closed(self) -> None:
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _recover_declared_pending_option_semantics
        from agent.harness.state import new_state

        source = "The requirement changed.\nReplace bsc with ethereum.\nKeep the current target mode for now."
        clauses = segment_user_turn(source)
        state = new_state("pending-manual-challenge", language="en")
        state["pending_question"] = {
            "id": "replacement",
            "group": "chain_identity",
            "field": "chain_change_input",
            "kind": "manual_value",
            "options": [],
            "manual_input_allowed": True,
            "validation": {"value_type": "scalar_token", "max_length": 180},
        }
        payload = {
            "actions": [{"type": "clarify_unresolved", "clauses": [source]}],
            "semantic_units": [
                _unit(clause, index, [0], disposition="unresolved")
                for index, clause in enumerate(clauses, start=1)
            ],
        }
        provider = Mock()
        provider.complete.side_effect = [
            SimpleNamespace(text=json.dumps({
                "decision": "no_selection",
                "option_id": "",
                "answer": "",
                "evidence_quote": "",
                "supporting_unit_ids": [],
                "independent_unit_ids": [
                    f"unit-{index}" for index in range(1, len(clauses) + 1)
                ],
                "reason": "first verdict missed the embedded scalar",
            })),
            SimpleNamespace(text=json.dumps({
                "decision": "manual_value",
                "option_id": "",
                "answer": "ethereum",
                "evidence_quote": "ethereum",
                "supporting_unit_ids": [
                    f"unit-{index}" for index in range(1, len(clauses) + 1)
                ],
                "independent_unit_ids": [],
                "reason": "one source-grounded replacement value plus preservation context",
            })),
        ]

        recovered, changed = _recover_declared_pending_option_semantics(
            provider,
            json.dumps(payload),
            state,
            clauses=clauses,
            user_text=source,
        )
        self.assertFalse(changed)
        self.assertEqual(recovered, json.dumps(payload))
        self.assertEqual(provider.complete.call_count, 2)

    def test_terminal_manual_value_supersedes_declared_manual_entry_option(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.coordinator import _dispatch_pending_action
        from agent.harness.intent import _recover_declared_pending_option_semantics
        from agent.harness.state import new_state

        source = (
            "Do not trust test-region; that came from the test harness.\n"
            "The deployment ticket assigns this node to asia-east1, so use that as CLOUD_REGION."
        )
        clauses = segment_user_turn(source)
        state = new_state("pending-manual-entry-precedence", language="en")
        state["pending_question"] = {
            "id": "CLOUD_REGION",
            "group": "provider_deployment",
            "field": "CLOUD_REGION",
            "kind": "yes_no",
            "manual_input_allowed": True,
            "validation": {"value_type": "scalar_token"},
            "options": [
                {"id": "1", "label": "Y", "value": "test-region", "action": {"type": "answer_pending"}},
                {
                    "id": "2",
                    "label": "N",
                    "value": "__manual__",
                    "manual_entry": True,
                    "action": {"type": "answer_pending"},
                },
            ],
        }
        payload = {
            "actions": [{"type": "clarify_unresolved", "clauses": [source]}],
            "semantic_units": [
                _unit(clause, index, [0], disposition="unresolved")
                for index, clause in enumerate(clauses, start=1)
            ],
        }
        unit_ids = [f"unit-{index}" for index in range(1, len(clauses) + 1)]
        provider = Mock()
        provider.complete.side_effect = [
            SimpleNamespace(text=json.dumps({
                "decision": "select_option",
                "option_id": "2",
                "answer": "",
                "evidence_quote": "Do not trust test-region",
                "supporting_unit_ids": unit_ids,
                "independent_unit_ids": [],
                "reason": "rejects the detected value and enters manual input",
            })),
            SimpleNamespace(text=json.dumps({
                "decision": "manual_value",
                "option_id": "",
                "answer": "asia-east1",
                "evidence_quote": "asia-east1",
                "supporting_unit_ids": unit_ids,
                "independent_unit_ids": [],
                "reason": "supplies the terminal replacement value",
            })),
        ]

        recovered_text, changed = _recover_declared_pending_option_semantics(
            provider,
            json.dumps(payload),
            state,
            clauses=clauses,
            user_text=source,
        )
        recovered = json.loads(recovered_text)

        self.assertTrue(changed)
        self.assertEqual(recovered["actions"], [{
            "type": "answer_pending",
            "answer": "asia-east1",
            "source_evidence": "asia-east1",
        }])
        state["last_user_input"] = source
        committed = _dispatch_pending_action(state, recovered["actions"][0])
        self.assertEqual(committed["confirmed_config"]["CLOUD_REGION"], "asia-east1")
        self.assertEqual(committed["pending_question"], {})

    def test_terminal_manual_value_does_not_override_a_non_manual_option_conflict(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _recover_declared_pending_option_semantics
        from agent.harness.state import new_state

        source = "Use the detected region; asia-east1 is only an example."
        clauses = segment_user_turn(source)
        state = new_state("pending-manual-entry-conflict", language="en")
        state["pending_question"] = {
            "id": "CLOUD_REGION",
            "group": "provider_deployment",
            "field": "CLOUD_REGION",
            "kind": "yes_no",
            "manual_input_allowed": True,
            "validation": {"value_type": "scalar_token"},
            "options": [
                {"id": "1", "label": "Y", "value": "test-region", "action": {"type": "answer_pending"}},
                {
                    "id": "2",
                    "label": "N",
                    "value": "__manual__",
                    "manual_entry": True,
                    "action": {"type": "answer_pending"},
                },
            ],
        }
        payload = {
            "actions": [{"type": "clarify_unresolved", "clauses": [source]}],
            "semantic_units": [
                _unit(clause, index, [0], disposition="unresolved")
                for index, clause in enumerate(clauses, start=1)
            ],
        }
        unit_ids = [f"unit-{index}" for index in range(1, len(clauses) + 1)]
        provider = Mock()
        provider.complete.side_effect = [
            SimpleNamespace(text=json.dumps({
                "decision": "select_option",
                "option_id": "1",
                "answer": "",
                "evidence_quote": "Use the detected region",
                "supporting_unit_ids": unit_ids,
                "independent_unit_ids": [],
                "reason": "selects the detected value",
            })),
            SimpleNamespace(text=json.dumps({
                "decision": "manual_value",
                "option_id": "",
                "answer": "asia-east1",
                "evidence_quote": "asia-east1",
                "supporting_unit_ids": unit_ids,
                "independent_unit_ids": [],
                "reason": "incorrectly treats the example as a replacement",
            })),
        ]

        recovered_text, changed = _recover_declared_pending_option_semantics(
            provider,
            json.dumps(payload),
            state,
            clauses=clauses,
            user_text=source,
        )

        self.assertFalse(changed)
        self.assertEqual(recovered_text, json.dumps(payload))

    def test_detected_value_question_declares_manual_entry_transition(self) -> None:
        from agent.harness.domains.environment import question_for_environment
        from agent.harness.state import new_state

        state = new_state("detected-manual-entry-contract", language="en")
        state["discovery"] = {"cloud": {"region": "test-region"}}
        question = question_for_environment(state, "provider_deployment")

        self.assertIsNotNone(question)
        self.assertFalse(question["options"][0]["manual_entry"])
        self.assertTrue(question["options"][1]["manual_entry"])

    def test_pending_owner_binds_untyped_natural_answer_to_declared_option_value(self) -> None:
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _adjudicate_pending_answer_actions
        from agent.harness.state import new_state

        source = "Yes, keep the detected 100 GiB value; it matches this disk."
        clauses = segment_user_turn(source)
        state = new_state("pending-owner-option-binding", language="en")
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
        payload = {
            "actions": [{
                "type": "answer_pending",
                "answer": "Yes",
                "source_evidence": "Yes",
            }],
            "semantic_units": [_unit(clauses[0], 1, [0])],
        }
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "reviews": [{
                "action_index": 0,
                "decision": "select_option",
                "selected_option_id": "1",
                "evidence_quote": "Yes",
                "reason": "the answer accepts the detected value",
            }],
        }))

        reviewed, changed = _adjudicate_pending_answer_actions(
            provider,
            json.dumps(payload),
            state,
            source,
        )
        document = json.loads(reviewed)

        self.assertTrue(changed)
        self.assertEqual(document["actions"][0]["selected_value"], "100")
        self.assertEqual(document["pending_answer_admissions"], [0])

    def test_verified_option_reference_is_bound_to_typed_contract_value_before_dispatch(self) -> None:
        from agent.harness.coordinator import _validate_action_plan
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
            "selected_value": "1",
            "source_evidence": state["last_user_input"],
            "pending_option_semantic_verified": True,
            "semantic_purpose_verified": True,
        }]

        prepared = _validate_action_plan(state, actions)

        self.assertEqual(prepared[0]["selected_value"], "100")
        self.assertEqual(prepared[0]["answer"], "100")
        self.assertIs(prepared[0]["selection_contract_verified"], True)

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

    def test_pending_completion_effect_is_available_to_context_admission(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _validate_semantic_fulfillment
        from agent.harness.state import new_state

        source = (
            "The syncing node is reachable inside Docker.\n"
            "Use http://geth-dev:8545 as its observation RPC endpoint.\n"
            "Please validate it before continuing."
        )
        clauses = segment_user_turn(source)
        payload = {
            "actions": [{
                "type": "answer_pending",
                "answer": "http://geth-dev:8545",
                "source_evidence": "http://geth-dev:8545",
            }],
            "semantic_units": [
                _unit(clauses[0], 1, [], disposition="context", reason="endpoint provenance"),
                _unit(clauses[1], 2, [0]),
                _unit(clauses[2], 3, [], disposition="context", reason="declared completion scope"),
            ],
            "pending_answer_admissions": [0],
        }
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "context_reviews": [
                {"unit_id": "unit-1", "context_only": True, "reason": "provenance"},
                {"unit_id": "unit-3", "context_only": True, "reason": "the probe is the declared completion effect"},
            ],
        }))
        state = new_state("pending-completion-context", language="en")
        state["pending_question"] = {
            "id": "SYNC_OBSERVE_RPC_URL",
            "field": "SYNC_OBSERVE_RPC_URL",
            "kind": "url",
            "manual_input_allowed": True,
            "options": [],
            "completion_effect": "Probe this endpoint and record validation evidence before continuing.",
        }

        result = _validate_semantic_fulfillment(provider, json.dumps(payload), clauses, state)

        self.assertTrue(result.valid, result.errors)
        request = json.loads(provider.complete.call_args.args[0].messages[1].content)
        self.assertEqual(
            request["pending_question"]["completion_effect"],
            state["pending_question"]["completion_effect"],
        )

    def test_unrelated_request_remains_outside_pending_completion_effect(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _validate_semantic_fulfillment
        from agent.harness.state import new_state

        clauses = segment_user_turn("Use http://geth-dev:8545. Also switch the benchmark to fake-node.")
        payload = {
            "actions": [{
                "type": "answer_pending",
                "answer": "http://geth-dev:8545",
                "source_evidence": "http://geth-dev:8545",
            }],
            "semantic_units": [
                _unit(clauses[0], 1, [0]),
                _unit(clauses[1], 2, [], disposition="context", reason="claimed completion scope"),
            ],
            "pending_answer_admissions": [0],
        }
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "context_reviews": [{
                "unit_id": "unit-2",
                "context_only": False,
                "reason": "target-mode change is not the endpoint probe completion effect",
            }],
        }))
        state = new_state("pending-completion-reject", language="en")
        state["pending_question"] = {
            "id": "SYNC_OBSERVE_RPC_URL",
            "field": "SYNC_OBSERVE_RPC_URL",
            "kind": "url",
            "manual_input_allowed": True,
            "options": [],
            "completion_effect": "Probe this endpoint and record validation evidence before continuing.",
        }

        result = _validate_semantic_fulfillment(provider, json.dumps(payload), clauses, state)

        self.assertFalse(result.valid)
        self.assertIn("context semantic unit unit-2 admission failed", "\n".join(result.errors))

    def test_pending_owner_receipt_closes_duplicate_context_review(self) -> None:
        import json
        from unittest.mock import Mock

        from agent.harness.intent import _validate_semantic_fulfillment
        from agent.harness.state import new_state

        clauses = segment_user_turn(
            "Use the fake-node option for this run. "
            "I want to validate the complete framework loop first."
        )
        payload = {
            "actions": [{
                "type": "choose_target_mode",
                "target_mode": "fake-node",
                "source_evidence": clauses[0].text,
            }],
            "semantic_units": [
                _unit(clauses[0], 1, [0]),
                _unit(
                    clauses[1],
                    2,
                    [],
                    disposition="context",
                    reason="purpose supporting the pending selection",
                ),
            ],
            "pending_answer_admissions": [0],
            "pending_support_unit_ids": ["unit-2"],
        }
        provider = Mock()
        provider.complete.side_effect = AssertionError(
            "receipt-covered pending support must not be reviewed again"
        )
        state = new_state("pending-owner-receipt", language="en")
        state["pending_question"] = {
            "id": "target_mode_select",
            "kind": "choice",
            "options": [
                {"id": "fake-node", "label": "fake-node", "value": "fake-node"},
                {"id": "real-node", "label": "real-node", "value": "real-node"},
            ],
        }

        result = _validate_semantic_fulfillment(provider, json.dumps(payload), clauses, state)

        self.assertTrue(result.valid, result.errors)
        provider.complete.assert_not_called()

    def test_same_clause_manual_field_is_folded_into_atomic_config_review(self) -> None:
        from agent.harness.intent import _merge_pending_manual_answer_into_config_proposal
        from agent.harness.state import new_state

        source = "interface: eth0\nlink_speed_gbps: 100\nUse 100 Gbps as the maximum bandwidth."
        clause = segment_user_turn(source)[0]
        state = new_state("atomic-pending-config")
        state["pending_question"] = {
            "id": "NETWORK_MAX_BANDWIDTH_GBPS",
            "group": "network",
            "field": "NETWORK_MAX_BANDWIDTH_GBPS",
            "kind": "manual_value",
            "manual_input_allowed": True,
            "validation": {"value_type": "positive_number"},
        }
        payload = {
            "actions": [
                {
                    "type": "propose_config_values",
                    "config_values": {"NETWORK_INTERFACE": "eth0"},
                    "unmapped_values": {"LINK_SPEED_GBPS": 100},
                    "source_format": "structured",
                    "source_evidence": source,
                },
                {
                    "type": "answer_pending",
                    "answer": "100",
                    "source_evidence": "100 Gbps",
                },
            ],
            "semantic_units": [_unit(clause, 1, [0, 1])],
            "pending_answer_admissions": [1],
        }

        result_text, changed = _merge_pending_manual_answer_into_config_proposal(
            json.dumps(payload),
            state,
        )
        result = json.loads(result_text)

        self.assertTrue(changed)
        self.assertEqual(len(result["actions"]), 1)
        self.assertEqual(
            result["actions"][0]["config_values"],
            {"NETWORK_INTERFACE": "eth0", "NETWORK_MAX_BANDWIDTH_GBPS": "100"},
        )
        self.assertEqual(result["semantic_units"][0]["action_indexes"], [0])
        self.assertEqual(result["pending_answer_admissions"], [])
        self.assertNotIn("admission_rejections", result)

    def test_separate_clause_manual_field_is_folded_into_the_only_config_review(self) -> None:
        from agent.harness.intent import _merge_pending_manual_answer_into_config_proposal
        from agent.harness.state import new_state

        source = "interface: eth0\nUse 100 Gbps as the maximum bandwidth."
        clauses = segment_user_turn(source)
        state = new_state("atomic-pending-config-separate-clauses")
        state["pending_question"] = {
            "id": "NETWORK_MAX_BANDWIDTH_GBPS",
            "group": "network",
            "field": "NETWORK_MAX_BANDWIDTH_GBPS",
            "kind": "manual_value",
            "manual_input_allowed": True,
            "validation": {"value_type": "positive_number"},
        }
        payload = {
            "actions": [
                {
                    "type": "propose_config_values",
                    "config_values": {"NETWORK_INTERFACE": "eth0"},
                    "unmapped_values": {},
                    "source_format": "mixed",
                    "source_evidence": clauses[0].text,
                },
                {
                    "type": "answer_pending",
                    "answer": "100",
                    "source_evidence": clauses[1].text,
                },
            ],
            "semantic_units": [
                _unit(clauses[0], 1, [0]),
                _unit(clauses[1], 2, [1]),
            ],
            "pending_answer_admissions": [1],
        }

        result_text, changed = _merge_pending_manual_answer_into_config_proposal(
            json.dumps(payload),
            state,
        )
        result = json.loads(result_text)

        self.assertTrue(changed)
        self.assertEqual(len(result["actions"]), 1)
        self.assertEqual(result["actions"][0]["config_values"], {
            "NETWORK_INTERFACE": "eth0",
            "NETWORK_MAX_BANDWIDTH_GBPS": "100",
        })
        self.assertEqual(
            [unit["action_indexes"] for unit in result["semantic_units"]],
            [[0], [0]],
        )

    def test_manual_field_is_not_folded_when_config_transaction_is_ambiguous(self) -> None:
        from agent.harness.intent import _merge_pending_manual_answer_into_config_proposal
        from agent.harness.state import new_state

        source = "CLOUD_REGION=us-east1\nNETWORK_INTERFACE=eth0\nUse 100 Gbps."
        clauses = segment_user_turn(source)
        state = new_state("ambiguous-config-transaction")
        state["pending_question"] = {
            "id": "NETWORK_MAX_BANDWIDTH_GBPS",
            "field": "NETWORK_MAX_BANDWIDTH_GBPS",
            "kind": "manual_value",
            "manual_input_allowed": True,
            "validation": {"value_type": "positive_number"},
        }
        payload = {
            "actions": [
                {"type": "propose_config_values", "config_values": {"CLOUD_REGION": "us-east1"}},
                {"type": "propose_config_values", "config_values": {"NETWORK_INTERFACE": "eth0"}},
                {"type": "answer_pending", "answer": "100", "source_evidence": "100 Gbps"},
            ],
            "semantic_units": [_unit(clauses[0], 1, [0, 1, 2])],
            "pending_answer_admissions": [2],
        }

        result_text, changed = _merge_pending_manual_answer_into_config_proposal(
            json.dumps(payload),
            state,
        )

        self.assertFalse(changed)
        self.assertEqual(json.loads(result_text), payload)

    def test_atomic_config_review_does_not_hide_a_conflicting_value(self) -> None:
        from agent.harness.intent import _merge_pending_manual_answer_into_config_proposal
        from agent.harness.state import new_state

        source = "NETWORK_MAX_BANDWIDTH_GBPS=50\nUse 100 Gbps instead."
        clause = segment_user_turn(source)[0]
        state = new_state("conflicting-pending-config")
        state["pending_question"] = {
            "id": "NETWORK_MAX_BANDWIDTH_GBPS",
            "group": "network",
            "field": "NETWORK_MAX_BANDWIDTH_GBPS",
            "kind": "manual_value",
            "manual_input_allowed": True,
            "validation": {"value_type": "positive_number"},
        }
        payload = {
            "actions": [
                {
                    "type": "propose_config_values",
                    "config_values": {"NETWORK_MAX_BANDWIDTH_GBPS": "50"},
                    "unmapped_values": {},
                    "source_format": "structured",
                    "source_evidence": source,
                },
                {"type": "answer_pending", "answer": "100", "source_evidence": "100 Gbps"},
            ],
            "semantic_units": [_unit(clause, 1, [0, 1])],
            "pending_answer_admissions": [1],
        }

        result_text, changed = _merge_pending_manual_answer_into_config_proposal(
            json.dumps(payload),
            state,
        )

        self.assertFalse(changed)
        self.assertEqual(json.loads(result_text), payload)

    def test_manual_value_scope_constraint_is_owned_by_the_pending_contract(self) -> None:
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _recover_declared_pending_option_semantics
        from agent.harness.state import new_state

        source = "The API key is chaos-test-key-4821. Apply it only to this run without changing the chain template."
        clauses = segment_user_turn(source)
        state = new_state("manual-value-scope")
        state["pending_question"] = {
            "id": "RPC_API_KEY",
            "group": "chain_auxiliary_endpoints",
            "field": "RPC_API_KEY",
            "kind": "manual_value",
            "manual_input_allowed": True,
            "validation": {"value_type": "scalar_token"},
        }
        payload = {
            "actions": [],
            "semantic_units": [
                _unit(clauses[0], 1, [], disposition="unresolved", reason="unresolved"),
                _unit(clauses[1], 2, [], disposition="unresolved", reason="unresolved"),
            ],
        }
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "decision": "manual_value",
            "option_id": "",
            "answer": "chaos-test-key-4821",
            "evidence_quote": "chaos-test-key-4821",
            "supporting_unit_ids": ["unit-1", "unit-2"],
            "independent_unit_ids": [],
            "reason": "the second sentence only scopes the supplied value",
        }))

        result_text, changed = _recover_declared_pending_option_semantics(
            provider,
            json.dumps(payload),
            state,
            clauses=clauses,
            user_text=source,
        )
        result = json.loads(result_text)

        self.assertTrue(changed)
        self.assertEqual(result["actions"], [{
            "type": "answer_pending",
            "answer": "chaos-test-key-4821",
            "source_evidence": "chaos-test-key-4821",
        }])
        self.assertEqual(result["semantic_units"][1]["disposition"], "context")

    def test_manual_url_answer_owns_contextual_mode_mention(self) -> None:
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _recover_declared_pending_option_semantics
        from agent.harness.state import new_state

        source = (
            "Use http://host.docker.internal:8545 for sync and health checks. "
            "Keep fake-node only as the existing local process attribution context."
        )
        clauses = segment_user_turn(source)
        state = new_state("manual-url-context")
        state["pending_question"] = {
            "id": "SYNC_OBSERVE_RPC_URL",
            "group": "endpoint_process",
            "field": "SYNC_OBSERVE_RPC_URL",
            "kind": "url",
            "manual_input_allowed": True,
        }
        payload = {
            "actions": [{
                "type": "choose_target_mode",
                "target_mode": "fake-node",
                "target_mode_explicit": True,
                "source_evidence": "fake-node",
            }],
            "semantic_units": [
                _unit(clauses[0], 1, [0]),
                _unit(clauses[1], 2, [0]),
            ],
        }
        verdict = {
            "decision": "manual_value",
            "option_id": "",
            "answer": "http://host.docker.internal:8545",
            "evidence_quote": "http://host.docker.internal:8545",
            "supporting_unit_ids": ["unit-1", "unit-2"],
            "independent_unit_ids": [],
            "reason": "the mode mention describes the existing attribution role",
        }
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps(verdict))

        result_text, changed = _recover_declared_pending_option_semantics(
            provider,
            json.dumps(payload),
            state,
            clauses=clauses,
            user_text=source,
        )
        result = json.loads(result_text)

        self.assertTrue(changed)
        self.assertEqual(result["actions"], [{
            "type": "answer_pending",
            "answer": "http://host.docker.internal:8545",
            "source_evidence": "http://host.docker.internal:8545",
        }])
        self.assertEqual(
            [unit["disposition"] for unit in result["semantic_units"]],
            ["action", "context"],
        )

    def test_manual_url_answer_preserves_distinct_explicit_mode_switch(self) -> None:
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _recover_declared_pending_option_semantics
        from agent.harness.state import new_state

        source = (
            "Use http://host.docker.internal:8545 for this endpoint. "
            "After recording it, switch this workflow to real-node."
        )
        clauses = segment_user_turn(source)
        state = new_state("manual-url-independent-jump")
        state["pending_question"] = {
            "id": "SYNC_OBSERVE_RPC_URL",
            "group": "endpoint_process",
            "field": "SYNC_OBSERVE_RPC_URL",
            "kind": "url",
            "manual_input_allowed": True,
        }
        payload = {
            "actions": [{
                "type": "choose_target_mode",
                "target_mode": "real-node",
                "target_mode_explicit": True,
                "source_evidence": "switch this workflow to real-node",
            }],
            "semantic_units": [
                _unit(clauses[0], 1, [0]),
                _unit(clauses[1], 2, [0]),
            ],
        }
        verdict = {
            "decision": "manual_value",
            "option_id": "",
            "answer": "http://host.docker.internal:8545",
            "evidence_quote": "http://host.docker.internal:8545",
            "supporting_unit_ids": ["unit-1"],
            "independent_unit_ids": ["unit-2"],
            "reason": "the endpoint answer and later mode switch are independent",
        }
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps(verdict))

        result_text, changed = _recover_declared_pending_option_semantics(
            provider,
            json.dumps(payload),
            state,
            clauses=clauses,
            user_text=source,
        )
        result = json.loads(result_text)

        self.assertTrue(changed)
        self.assertEqual(
            [action["type"] for action in result["actions"]],
            ["choose_target_mode", "answer_pending"],
        )
        self.assertEqual(result["semantic_units"][0]["action_indexes"], [1])
        self.assertEqual(result["semantic_units"][1]["action_indexes"], [0])

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
        self.assertTrue(changed)
        document = json.loads(result)
        self.assertEqual(document["pending_answer_admissions"], [0])
        self.assertEqual(document["actions"][0]["answer"], "1000")
        self.assertNotIn("selected_value", document["actions"][0])
        self.assertEqual(document["actions"][0]["source_evidence"], "1000")

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
            "pending_support_unit_ids": ["model-forged-unit"],
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

    def test_pending_review_rejects_local_no_when_complete_turn_selects_other_option(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _adjudicate_pending_answer_actions
        from agent.harness.state import new_state

        source = (
            "No. Sola is a different real blockchain, not Solana. "
            "Take the path where we identify its protocol family."
        )
        clauses = segment_user_turn(source)
        state = new_state("pending-complete-turn-conflict", language="en")
        state["pending_question"] = {
            "id": "unknown_chain_identity_confirm",
            "group": "chain_identity",
            "kind": "numbered_choice",
            "prompt": "Use the suggested chain?",
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
        payload = json.dumps({
            "actions": [{
                "type": "answer_pending",
                "answer": "reenter_chain",
                "selected_value": "reenter_chain",
                "source_evidence": "No.",
            }],
            "semantic_units": [
                {
                    "unit_id": f"unit-{index}",
                    "clause_id": clause.clause_id,
                    "source_text": clause.text,
                    "disposition": "action" if index == 1 else "context",
                    "action_indexes": [0] if index == 1 else [],
                    "reason": "planner classification",
                }
                for index, clause in enumerate(clauses, start=1)
            ],
        })
        provider = Mock()

        def reject(request):
            request_payload = json.loads(request.messages[-1].content)
            review = request_payload["reviews"][0]
            self.assertEqual(review["source_units"], ["No."])
            self.assertEqual(review["complete_source"], source)
            return SimpleNamespace(text=json.dumps({
                "reviews": [{
                    "action_index": 0,
                    "decision": "reject",
                    "evidence_quote": "",
                    "reason": "the complete turn selects the different protocol option",
                }],
            }))

        provider.complete.side_effect = reject
        result, changed = _adjudicate_pending_answer_actions(provider, payload, state, source)
        document = json.loads(result)

        self.assertTrue(changed)
        self.assertEqual(document["actions"], [])
        self.assertEqual(document.get("pending_answer_admissions", []), [])
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

    def test_consultation_option_is_admitted_by_the_active_pending_owner(self) -> None:
        import json
        from types import SimpleNamespace
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
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "reviews": [{
                "action_index": 0,
                "decision": "select_option",
                "selected_option_id": "info",
                "evidence_quote": source,
                "reason": "the complete source selects the declared read-only option",
            }],
        }))

        result, changed = _adjudicate_pending_answer_actions(provider, payload, state, source)

        self.assertTrue(changed)
        document = json.loads(result)
        self.assertEqual(document["actions"], [{
            "type": "answer_opening_question",
            "topic": "capabilities",
            "source_evidence": source,
        }])
        self.assertEqual(document["pending_answer_admissions"], [0])
        provider.complete.assert_called()

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
            "validation": {"value_type": "scalar_token"},
            "options": [],
        }
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "reviews": [{
                "action_index": 0,
                "decision": "direct_answer",
                "selected_value": "asia-east1",
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

        document = json.loads(result)
        self.assertEqual(document["pending_answer_admissions"], [0])
        self.assertEqual(document["actions"][0]["answer"], "asia-east1")
        self.assertEqual(document["actions"][0]["selected_value"], "asia-east1")
        self.assertEqual(document["actions"][0]["source_evidence"], "asia-east1")

    def test_prose_wrapped_rpc_method_compiles_through_declared_owner(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _adjudicate_pending_action_ownership
        from agent.harness.state import new_state

        source = "The method name is `eth_chainId`. I do not have its request or response sample with me yet."
        clauses = segment_user_turn(source)
        state = new_state("manual-rpc-method", language="en")
        state["pending_question"] = {
            "id": "custom_rpc_method",
            "group": "endpoint_process",
            "kind": "manual_value",
            "field": "custom_rpc_method",
            "prompt": "Enter the custom RPC method name to validate.",
            "manual_input_allowed": True,
            "accepted_action_types": ["rpc_catalog_command"],
            "manual_action": {
                "type": "rpc_catalog_command",
                "catalog_command": "set_method",
                "value_argument": "rpc_method",
            },
            "validation": {"input_mode": "rpc_method_or_schema_evidence"},
            "options": [],
        }
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "reviews": [{
                "action_index": 0,
                "decision": "direct_answer",
                "selected_value": "eth_chainId",
                "evidence_quote": "eth_chainId",
                "reason": "the source directly names the requested RPC method",
            }],
        }))
        payload = {
            "actions": [{
                "type": "answer_pending",
                "answer": source,
                "source_evidence": source,
            }],
            "semantic_units": [_unit(clauses[0], 1, [0])],
        }

        result, changed = _adjudicate_pending_action_ownership(
            provider, json.dumps(payload), state, source
        )
        document = json.loads(result)

        self.assertTrue(changed)
        self.assertEqual(document["actions"], [{
            "type": "rpc_catalog_command",
            "catalog_command": "set_method",
            "rpc_method": "eth_chainId",
            "source_evidence": "eth_chainId",
        }])
        self.assertEqual(document["pending_answer_admissions"], [0])

    def test_structured_rpc_request_recovers_omitted_method_owner_symmetrically(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _adjudicate_pending_action_ownership
        from agent.harness.state import new_state

        source = (
            "Here is the request I found; I do not have its response yet:\n"
            '{"jsonrpc":"2.0","id":9,"method":"eth_getBalance",'
            '"params":["0x0000000000000000000000000000000000000000","latest"]}'
        )
        clauses = segment_user_turn(source)
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "reviews": [{
                "action_index": 0,
                "decision": "direct_answer",
                "selected_value": "eth_getBalance",
                "evidence_quote": "eth_getBalance",
                "reason": "the structured request supplies the requested wire method",
            }],
        }))

        for question_id in ("custom_rpc_method", "new_chain_method"):
            state = new_state(f"structured-owner-{question_id}", language="en")
            state["pending_question"] = {
                "id": question_id,
                "group": "endpoint_process",
                "kind": "manual_value",
                "field": question_id,
                "prompt": "Enter the RPC method name to validate.",
                "manual_input_allowed": True,
                "accepted_action_types": ["rpc_catalog_command"],
                "manual_action": {
                    "type": "rpc_catalog_command",
                    "catalog_command": "set_method",
                    "value_argument": "rpc_method",
                },
                "validation": {"input_mode": "rpc_method_or_schema_evidence"},
                "options": [],
            }
            payload = {
                "actions": [{
                    "type": "clarify_unresolved",
                    "clauses": [clauses[-1].text],
                }],
                "semantic_units": [
                    _unit(clauses[0], 1, [], disposition="context"),
                    _unit(clauses[-1], 2, [0]),
                ],
            }

            result, changed = _adjudicate_pending_action_ownership(
                provider, json.dumps(payload), state, source
            )
            document = json.loads(result)

            self.assertTrue(changed)
            self.assertEqual(document["actions"], [{
                "type": "rpc_catalog_command",
                "catalog_command": "set_method",
                "rpc_method": "eth_getBalance",
                "source_evidence": "eth_getBalance",
            }])
            self.assertEqual(document["pending_answer_admissions"], [0])

        from agent.harness.action_registry import ACTION_BY_TYPE

        self.assertIn(
            "evidence_completeness",
            ACTION_BY_TYPE["rpc_catalog_command"].semantic_support_relations,
        )

    def test_complete_turn_evidence_owner_recovers_partial_wire_evidence(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _adjudicate_pending_action_ownership
        from agent.harness.state import new_state

        cases = (
            (
                "I only have the request so far; no response was captured:\n"
                '{"jsonrpc":"2.0","id":9,"method":"chain_status","params":[]}',
                '{"jsonrpc":"2.0","id":9,"method":"chain_status","params":[]}',
            ),
            (
                "目前只拿到了响应，没有请求：\n"
                '{"jsonrpc":"2.0","id":9,"result":{"height":"42"}}',
                '{"jsonrpc":"2.0","id":9,"result":{"height":"42"}}',
            ),
        )
        for source, evidence_quote in cases:
            with self.subTest(source=source):
                clauses = segment_user_turn(source)
                state = new_state("structured-handoff-evidence", language="en")
                state["pending_question"] = {
                    "id": "opaque-evidence-question",
                    "group": "chain_identity",
                    "kind": "evidence",
                    "field": "protocol_evidence",
                    "prompt": "Provide protocol evidence.",
                    "manual_input_allowed": True,
                    "accepted_action_types": ["secondary_handoff_command"],
                    "manual_action": {
                        "type": "secondary_handoff_command",
                        "handoff_command": "append_evidence",
                        "value_argument": "handoff_evidence",
                        "use_complete_turn": True,
                    },
                    "validation": {
                        "value_type": "evidence_contribution",
                        "max_length": 65536,
                    },
                    "structured_input_owner": True,
                    "options": [],
                }
                payload = {
                    "actions": [{
                        "type": "clarify_unresolved",
                        "clauses": [clauses[-1].text],
                    }],
                    "semantic_units": [
                        _unit(clauses[0], 1, [], disposition="context"),
                        _unit(clauses[-1], 2, [0]),
                    ],
                }
                provider = Mock()
                provider.complete.return_value = SimpleNamespace(text=json.dumps({
                    "reviews": [{
                        "action_index": 0,
                        "decision": "direct_answer",
                        "selected_value": evidence_quote,
                        "evidence_quote": evidence_quote,
                        "reason": "the structured fragment is attributable wire evidence",
                    }],
                }))

                result, changed = _adjudicate_pending_action_ownership(
                    provider,
                    json.dumps(payload),
                    state,
                    source,
                )
                document = json.loads(result)

                self.assertTrue(changed)
                self.assertEqual(document["actions"], [{
                    "type": "secondary_handoff_command",
                    "handoff_command": "append_evidence",
                    "handoff_evidence": source,
                    "source_evidence": source,
                }])
                self.assertEqual(document["pending_answer_admissions"], [0])
                review_payload = json.loads(provider.complete.call_args.args[0].messages[1].content)
                self.assertEqual(
                    review_payload["reviews"][0]["validation"]["value_type"],
                    "evidence_contribution",
                )
                self.assertTrue(review_payload["reviews"][0]["structured_input_owner"])
                self.assertTrue(review_payload["reviews"][0]["use_complete_turn"])
                self.assertIn(
                    "protocol-development evidence",
                    review_payload["reviews"][0]["declared_owner_purpose"],
                )

        from agent.harness.action_registry import ACTION_BY_TYPE

        self.assertIn(
            "evidence_completeness",
            ACTION_BY_TYPE["secondary_handoff_command"].semantic_support_relations,
        )

    def test_structured_evidence_owner_does_not_claim_unrelated_json(self) -> None:
        import json

        from agent.harness.intent import _seed_structured_manual_pending_candidates

        source = '{"CLOUD_REGION":"asia-east1","INITIAL_QPS":10}'
        clauses = segment_user_turn(source)
        payload = {
            "actions": [{"type": "clarify_unresolved", "clauses": [source]}],
            "semantic_units": [_unit(clauses[0], 1, [0])],
        }
        pending = {
            "structured_input_owner": True,
            "manual_action": {
                "type": "secondary_handoff_command",
                "handoff_command": "append_evidence",
                "value_argument": "handoff_evidence",
                "use_complete_turn": True,
            },
            "validation": {"value_type": "evidence_contribution"},
        }

        seeded = _seed_structured_manual_pending_candidates(payload, pending, source)

        self.assertEqual(seeded, set())
        self.assertEqual(payload["actions"], [{
            "type": "clarify_unresolved",
            "clauses": [source],
        }])

    def test_structured_evidence_owner_supersedes_generic_wire_analysis_only(self) -> None:
        from agent.harness.intent import _seed_structured_manual_pending_candidates

        source = (
            "I only have the request so far; no response was captured:\n"
            '{"jsonrpc":"2.0","id":9,"method":"chain_status","params":[]}'
        )
        clauses = segment_user_turn(source)
        self.assertGreaterEqual(len(clauses), 2)
        payload = {
            "actions": [{
                "type": "analyze_evidence",
                "evidence": clauses[-1].text,
                "question": "",
            }],
            "semantic_units": [
                _unit(clauses[0], 1, [], disposition="unresolved"),
                _unit(clauses[-1], 2, [0]),
            ],
        }
        pending = {
            "structured_input_owner": True,
            "manual_action": {
                "type": "secondary_handoff_command",
                "handoff_command": "append_evidence",
                "value_argument": "handoff_evidence",
                "use_complete_turn": True,
            },
            "validation": {"value_type": "evidence_contribution"},
        }

        seeded = _seed_structured_manual_pending_candidates(payload, pending, source)

        self.assertEqual(seeded, {0})
        self.assertEqual(payload["actions"], [{
            "type": "answer_pending",
            "answer": source,
            "source_evidence": source,
        }])
        self.assertEqual(payload["semantic_units"][0]["disposition"], "context")
        self.assertEqual(payload["semantic_units"][0]["action_indexes"], [])
        self.assertEqual(payload["semantic_units"][-1]["action_indexes"], [0])

    def test_structured_evidence_owner_preserves_separate_analysis_demand(self) -> None:
        from agent.harness.intent import _seed_structured_manual_pending_candidates

        source = (
            "Analyze this request before saving it;\n"
            '{"jsonrpc":"2.0","id":9,"method":"chain_status","params":[]}'
        )
        clauses = segment_user_turn(source)
        self.assertGreaterEqual(len(clauses), 2)
        payload = {
            "actions": [{
                "type": "analyze_evidence",
                "evidence": source,
                "question": clauses[0].text,
            }],
            "semantic_units": [
                _unit(clauses[0], 1, [0]),
                _unit(clauses[-1], 2, [0]),
            ],
        }
        pending = {
            "structured_input_owner": True,
            "manual_action": {
                "type": "secondary_handoff_command",
                "handoff_command": "append_evidence",
                "value_argument": "handoff_evidence",
                "use_complete_turn": True,
            },
            "validation": {"value_type": "evidence_contribution"},
        }

        seeded = _seed_structured_manual_pending_candidates(payload, pending, source)

        self.assertEqual(seeded, {1})
        self.assertEqual(payload["actions"][0]["type"], "analyze_evidence")
        self.assertEqual(payload["actions"][1]["type"], "answer_pending")
        self.assertEqual(payload["semantic_units"][0]["action_indexes"], [0])
        self.assertEqual(payload["semantic_units"][-1]["action_indexes"], [0, 1])

    def test_structured_rpc_request_does_not_bypass_semantic_rejection(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _adjudicate_pending_action_ownership
        from agent.harness.state import new_state

        source = (
            "Do not use this documentation example:\n"
            '{"jsonrpc":"2.0","id":1,"method":"eth_chainId","params":[]}'
        )
        clauses = segment_user_turn(source)
        state = new_state("structured-owner-rejection", language="en")
        state["pending_question"] = {
            "id": "opaque-method-question",
            "group": "endpoint_process",
            "kind": "manual_value",
            "field": "custom_rpc_method",
            "prompt": "Enter the RPC method name to validate.",
            "manual_input_allowed": True,
            "accepted_action_types": ["rpc_catalog_command"],
            "manual_action": {
                "type": "rpc_catalog_command",
                "catalog_command": "set_method",
                "value_argument": "rpc_method",
            },
            "validation": {"input_mode": "rpc_method_or_schema_evidence"},
            "options": [],
        }
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "reviews": [{
                "action_index": 0,
                "decision": "reject",
                "selected_value": None,
                "evidence_quote": "",
                "reason": "the source explicitly rejects the example",
            }],
        }))
        payload = {
            "actions": [{"type": "clarify_unresolved", "clauses": [source]}],
            "semantic_units": [_unit(clauses[-1], 1, [0])],
        }

        result, _changed = _adjudicate_pending_action_ownership(
            provider, json.dumps(payload), state, source
        )
        document = json.loads(result)

        self.assertEqual(document["actions"], [])
        self.assertEqual(document["semantic_units"][0]["disposition"], "unresolved")

    def test_case3_multiline_evidence_compiles_through_declared_owner(self) -> None:
        import json

        from agent.harness.domains.chain_rpc_questions import _case3_evidence_question
        from agent.harness.intent import _materialize_pending_manual_owner_actions
        from agent.harness.state import new_state

        source = (
            "The protocol documentation says this chain exposes its own peer-to-peer RPC transport.\n"
            "Its request and response schema is not compatible with the currently supported adapter families."
        )
        state = new_state("case3-evidence-owner", language="en")
        state["chain_identity"] = {
            "raw": "WeirdP2PChain",
            "canonical": "WeirdP2PChain",
            "adapter_family": "unsupported",
            "status": "case3_collecting_evidence",
            "case": "case3",
        }
        state["secondary_handoff"] = {
            "status": "collecting_evidence",
            "kind": "case3_protocol_adapter_implementation",
            "evidence": [],
        }
        state["pending_question"] = _case3_evidence_question(state)
        payload = {
            "actions": [{
                "type": "answer_pending",
                "answer": source,
                "selected_value": source,
                "source_evidence": source,
            }],
            "semantic_units": [{
                "unit_id": "unit-1",
                "clause_id": "clause-1",
                "source_text": source,
                "disposition": "action",
                "action_indexes": [0],
            }],
            "pending_answer_admissions": [0],
        }

        result, changed = _materialize_pending_manual_owner_actions(
            json.dumps(payload), state, source
        )
        document = json.loads(result)

        self.assertTrue(changed)
        self.assertEqual(document["actions"], [{
            "type": "secondary_handoff_command",
            "handoff_command": "append_evidence",
            "handoff_evidence": source,
            "source_evidence": source,
        }])
        self.assertEqual(document["pending_answer_admissions"], [0])
        self.assertEqual(document["semantic_units"][0]["action_indexes"], [0])

    def test_manual_pending_review_extracts_typed_values_generically(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _adjudicate_manual_pending_answers
        from agent.harness.state import new_state

        cases = (
            ("url", "Use https://rpc.example.test for validation.", "https://rpc.example.test", {}),
            ("region", "Set this run to us-central1 please.", "us-central1", {"value_type": "scalar_token"}),
            ("device", "Use /dev/nvme1n1 for the ledger.", "/dev/nvme1n1", {"value_type": "scalar_token"}),
            ("number", "The limit should be 20000 IOPS.", "20000", {"value_type": "positive_number"}),
        )
        for kind, source, selected, validation in cases:
            with self.subTest(kind=kind):
                state = new_state(f"manual-{kind}", language="en")
                state["pending_question"] = {
                    "id": kind,
                    "group": "test_group",
                    "kind": "url" if kind == "url" else ("device" if kind == "device" else "manual_value"),
                    "field": kind,
                    "prompt": f"Enter {kind}.",
                    "manual_input_allowed": True,
                    "validation": validation,
                    "options": [],
                }
                provider = Mock()
                provider.complete.return_value = SimpleNamespace(text=json.dumps({
                    "reviews": [{
                        "action_index": 0,
                        "decision": "direct_answer",
                        "selected_value": selected,
                        "evidence_quote": selected,
                        "reason": "direct typed value",
                    }],
                }))
                payload = json.dumps({
                    "actions": [{"type": "answer_pending", "answer": source}],
                    "semantic_units": [{
                        "unit_id": "unit-1",
                        "clause_id": "clause-1",
                        "source_text": source,
                        "disposition": "action",
                        "action_indexes": [0],
                    }],
                })

                result, changed = _adjudicate_manual_pending_answers(
                    provider, payload, state, source
                )
                document = json.loads(result)

                self.assertTrue(changed)
                self.assertEqual(document["pending_answer_admissions"], [0])
                self.assertEqual(document["actions"][0]["selected_value"], selected)
                self.assertEqual(document["actions"][0]["source_evidence"], selected)

    def test_manual_pending_review_rejects_untrusted_extracted_values(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _adjudicate_manual_pending_answers
        from agent.harness.state import new_state

        cases = (
            (
                "invented",
                "Use the endpoint I mentioned earlier.",
                "https://invented.example",
                "the endpoint I mentioned earlier",
                "url",
                {},
            ),
            (
                "invalid",
                "Use not-a-number for the limit.",
                "not-a-number",
                "not-a-number",
                "manual_value",
                {"value_type": "positive_number"},
            ),
            (
                "ambiguous",
                "Use 1000 or 2000 for the limit.",
                "1000 or 2000",
                "1000 or 2000",
                "manual_value",
                {"value_type": "positive_number"},
            ),
        )
        for name, source, selected, quote, question_kind, validation in cases:
            with self.subTest(name=name):
                state = new_state(f"manual-reject-{name}", language="en")
                state["pending_question"] = {
                    "id": "limit",
                    "group": "test_group",
                    "kind": question_kind,
                    "field": "limit",
                    "prompt": "Enter the limit.",
                    "manual_input_allowed": True,
                    "validation": validation,
                    "options": [],
                }
                provider = Mock()
                provider.complete.return_value = SimpleNamespace(text=json.dumps({
                    "reviews": [{
                        "action_index": 0,
                        "decision": "direct_answer",
                        "selected_value": selected,
                        "evidence_quote": quote,
                        "reason": "candidate value",
                    }],
                }))
                payload = json.dumps({
                    "actions": [{"type": "answer_pending", "answer": source}],
                    "semantic_units": [{
                        "unit_id": "unit-1",
                        "clause_id": "clause-1",
                        "source_text": source,
                        "disposition": "action",
                        "action_indexes": [0],
                    }],
                })

                result, changed = _adjudicate_manual_pending_answers(
                    provider, payload, state, source
                )
                document = json.loads(result)

                self.assertTrue(changed)
                self.assertEqual(document["actions"], [])
                self.assertEqual(document["semantic_units"][0]["disposition"], "unresolved")

    def test_prose_wrapped_endpoint_compiles_through_declared_owner(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _adjudicate_pending_action_ownership
        from agent.harness.state import new_state

        source = "Use https://rpc.example.test only to validate this method."
        clauses = segment_user_turn(source)
        state = new_state("manual-rpc-endpoint", language="en")
        state["pending_question"] = {
            "id": "custom_rpc_endpoint",
            "group": "endpoint_process",
            "kind": "url",
            "field": "custom_rpc_endpoint",
            "prompt": "Provide a reachable RPC endpoint.",
            "manual_input_allowed": True,
            "accepted_action_types": ["rpc_catalog_command"],
            "manual_action": {
                "type": "rpc_catalog_command",
                "catalog_command": "set_endpoint",
                "value_argument": "rpc_endpoint",
            },
            "options": [],
        }
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "reviews": [{
                "action_index": 0,
                "decision": "direct_answer",
                "selected_value": "https://rpc.example.test",
                "evidence_quote": "https://rpc.example.test",
                "reason": "the source directly supplies the endpoint",
            }],
        }))
        payload = {
            "actions": [{"type": "answer_pending", "answer": source}],
            "semantic_units": [_unit(clauses[0], 1, [0])],
        }

        result, changed = _adjudicate_pending_action_ownership(
            provider, json.dumps(payload), state, source
        )

        self.assertTrue(changed)
        self.assertEqual(json.loads(result)["actions"], [{
            "type": "rpc_catalog_command",
            "catalog_command": "set_endpoint",
            "rpc_endpoint": "https://rpc.example.test",
            "source_evidence": "https://rpc.example.test",
        }])

    def test_generic_resume_is_not_compiled_by_manual_domain_owner(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _adjudicate_pending_action_ownership
        from agent.harness.state import new_state

        source = "Return to the current benchmark setup."
        clauses = segment_user_turn(source)
        state = new_state("manual-owner-resume", language="en")
        state["pending_question"] = {
            "id": "custom_rpc_endpoint",
            "group": "endpoint_process",
            "kind": "url",
            "field": "custom_rpc_endpoint",
            "prompt": "Provide a reachable RPC endpoint.",
            "manual_input_allowed": True,
            "manual_action": {
                "type": "rpc_catalog_command",
                "catalog_command": "set_endpoint",
                "value_argument": "rpc_endpoint",
            },
            "options": [],
        }
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "reviews": [{
                "action_index": 0,
                "decision": "generic_resume",
                "selected_value": None,
                "evidence_quote": source,
                "reason": "the source requests generic workflow resumption",
            }],
        }))
        payload = {
            "actions": [{"type": "answer_pending", "answer": source}],
            "semantic_units": [_unit(clauses[0], 1, [0])],
        }

        result, changed = _adjudicate_pending_action_ownership(
            provider, json.dumps(payload), state, source
        )

        self.assertTrue(changed)
        self.assertEqual(json.loads(result)["actions"], [{
            "type": "resume_current_flow",
            "source_evidence": source,
        }])

    def test_manual_pending_review_schema_requires_one_selected_value_receipt(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _request_manual_pending_reviews

        provider = Mock()
        provider.complete.side_effect = [
            SimpleNamespace(text=json.dumps({
                "reviews": [{
                    "action_index": 0,
                    "decision": "direct_answer",
                    "evidence_quote": "asia-east1",
                    "reason": "missing selected value",
                }],
            })),
            SimpleNamespace(text=json.dumps({
                "reviews": [
                    {
                        "action_index": 0,
                        "decision": "direct_answer",
                        "selected_value": "asia-east1",
                        "evidence_quote": "asia-east1",
                        "reason": "duplicate one",
                    },
                    {
                        "action_index": 0,
                        "decision": "direct_answer",
                        "selected_value": "asia-east1",
                        "evidence_quote": "asia-east1",
                        "reason": "duplicate two",
                    },
                ],
            })),
        ]

        result = _request_manual_pending_reviews(provider, [{
            "action_index": 0,
            "question": "Enter CLOUD_REGION.",
            "field": "CLOUD_REGION",
            "kind": "manual_value",
            "proposed_answer": "Use asia-east1.",
            "source_units": ["Use asia-east1."],
        }])

        self.assertEqual(result, {"reviews": []})
        self.assertEqual(provider.complete.call_count, 2)

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

    def test_owner_receipt_is_not_reopened_by_compound_unit_inventory(self) -> None:
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
                self.assertEqual(incomplete, ())
                provider.complete.assert_not_called()

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

    def test_backward_navigation_replaces_incorrect_generic_resume(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _adjudicate_group_navigation_actions

        source = "Actually, take me back one step, but keep the values I already confirmed."
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "reviews": [{
                "action_index": 0,
                "destination_named": False,
                "destination_quote": "",
                "backward_navigation": True,
                "backward_quote": "take me back one step",
                "generic_resume": False,
                "resume_quote": "",
                "specific_change_requested": False,
                "specific_change_quote": "",
                "reason": "the source requests the previous workflow step",
            }],
        }))
        payload = {
            "actions": [{
                "type": "resume_current_flow",
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

        result_text, changed = _adjudicate_group_navigation_actions(
            provider,
            json.dumps(payload),
            {"pending_question": {"id": "qps_profile_confirm"}},
        )

        self.assertTrue(changed)
        self.assertEqual(json.loads(result_text)["actions"], [{"type": "go_back"}])

    def test_chinese_backward_navigation_uses_semantic_unit_as_evidence(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _adjudicate_group_navigation_actions

        source = "回到上一步，保留已经确认的配置"
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "reviews": [{
                "action_index": 0,
                "destination_named": False,
                "destination_quote": "",
                "backward_navigation": True,
                "backward_quote": "回到上一步",
                "generic_resume": False,
                "resume_quote": "",
                "specific_change_requested": False,
                "specific_change_quote": "",
                "reason": "previous step requested",
            }],
        }))
        payload = {
            "actions": [{"type": "go_back"}],
            "semantic_units": [{
                "unit_id": "unit-1",
                "clause_id": "clause-1",
                "source_text": source,
                "disposition": "action",
                "action_indexes": [0],
            }],
        }

        result_text, changed = _adjudicate_group_navigation_actions(
            provider,
            json.dumps(payload),
            {"pending_question": {"id": "benchmark_mode"}},
        )

        self.assertFalse(changed)
        self.assertEqual(json.loads(result_text)["actions"], [{"type": "go_back"}])

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

    def test_named_group_change_without_concrete_value_uses_registered_intake(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _adjudicate_group_navigation_actions

        source = (
            "I need to change the blockchain chain first; "
            "take me to the chain identity configuration."
        )
        provider = Mock()
        provider.complete.side_effect = [
            SimpleNamespace(text=json.dumps({"reviews": [{
                "action_index": 0,
                "destination_named": True,
                "destination_quote": "chain identity configuration",
                "generic_resume": False,
                "resume_quote": "",
                "specific_change_requested": True,
                "specific_change_quote": "change the blockchain chain",
                "reason": "the source requests a chain change in the named group",
            }]})),
            SimpleNamespace(text=json.dumps({"reviews": [{
                "action_index": 0,
                "specific_change_requested": True,
                "specific_change_quote": "change the blockchain chain",
                "reason": "the change is explicit but has no replacement value",
            }]})),
            SimpleNamespace(text=json.dumps({
                "action": None,
                "reason": "no concrete replacement chain was supplied",
            })),
        ]
        payload = {
            "actions": [{
                "type": "change_group",
                "group": "chain_identity",
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
        }

        result_text, changed = _adjudicate_group_navigation_actions(
            provider,
            json.dumps(payload),
            {"pending_question": {}},
        )
        result = json.loads(result_text)

        self.assertTrue(changed)
        self.assertEqual(result["actions"], [{
            "type": "request_chain_selection",
            "source_evidence": source,
        }])
        self.assertNotIn("group_navigation_admissions", result)
        self.assertEqual(provider.complete.call_count, 3)

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

    def test_pending_owner_binds_answer_only_to_heterogeneous_declared_option(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _adjudicate_pending_answer_actions
        from agent.harness.state import new_state

        source = "这两个都不是，我重新输入正确的链名。"
        state = new_state("pending-heterogeneous-options", language="zh")
        state["pending_question"] = {
            "id": "chain_ambiguity_confirm",
            "group": "chain_identity",
            "kind": "numbered_choice",
            "prompt": "请选择链。",
            "options": [
                {
                    "id": "bsc",
                    "label": "bsc",
                    "value": {"chain_choice": "bsc"},
                    "action": {
                        "type": "answer_pending",
                        "answer": {"chain_choice": "bsc"},
                    },
                },
                {
                    "id": "reenter",
                    "label": "重新输入链名",
                    "value": "reenter_chain",
                    "action": {"type": "answer_pending", "answer": "reenter_chain"},
                },
            ],
        }
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "reviews": [{
                "action_index": 0,
                "decision": "select_option",
                "evidence_quote": "重新输入正确的链名",
                "reason": "the source selects the declared re-entry option",
            }],
        }))
        payload = json.dumps({
            "actions": [{
                "type": "answer_pending",
                "answer": "reenter_chain",
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
        admitted = json.loads(result)

        self.assertTrue(changed)
        self.assertEqual(admitted["pending_answer_admissions"], [0])
        self.assertEqual(admitted["actions"][0]["answer"], "reenter_chain")
        self.assertEqual(admitted["actions"][0]["selected_value"], "reenter_chain")

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

    def test_registry_inventory_excludes_pending_owner_units(self) -> None:
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
        provider.complete.assert_not_called()

    def test_registry_inventory_excludes_failure_recovery_owner_unit(self) -> None:
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

        result_text, changed, incomplete = _challenge_registry_action_inventory(
            provider,
            json.dumps(payload, ensure_ascii=False),
            state,
        )

        self.assertFalse(changed)
        self.assertEqual(incomplete, ())
        self.assertEqual(json.loads(result_text), payload)
        provider.complete.assert_not_called()

    def test_registry_inventory_does_not_reinterpret_an_owner_partition(self) -> None:
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
        recovered_text, changed, incomplete = _challenge_registry_action_inventory(
            provider,
            json.dumps(payload),
            state,
        )
        recovered = json.loads(recovered_text)

        self.assertFalse(changed)
        self.assertEqual(incomplete, ())
        self.assertEqual(recovered, payload)
        provider.complete.assert_not_called()

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
        self.assertEqual([row["unit_id"] for row in request["units"]], ["unit-2"])

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

    def test_inventory_wrapper_does_not_let_challenger_revoke_owner_receipt(self) -> None:
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
        self.assertIs(result_validation, validation)

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

    def test_final_structured_authority_restores_facts_lost_by_model_recovery(self) -> None:
        import json
        from unittest.mock import Mock, patch

        from agent.harness.intent import _finalize_structured_syntax_authority
        from agent.harness.plan_coverage import PlanCoverageResult, segment_user_turn
        from agent.harness.state import new_state

        source = (
            "CLOUD_PROVIDER=gcp\n"
            "CLOUD_REGION=us-central1\n"
            "CLOUD_ZONE=us-central1-a\n"
            "owner_ticket=INC-4821"
        )
        clause = segment_user_turn(source)[0]
        model_replacement = {
            "actions": [{
                "type": "propose_config_values",
                "config_values": {
                    "CLOUD_REGION": "us-central1",
                    "CLOUD_ZONE": "us-central1-a",
                },
                "unmapped_values": {},
                "source_format": "env",
                "source_evidence": source,
            }],
            "semantic_units": [_unit(clause, 1, [0])],
        }
        valid = PlanCoverageResult(True, (), ())

        with patch("agent.harness.intent._validate_semantic_fulfillment", return_value=valid):
            finalized_text, validation = _finalize_structured_syntax_authority(
                Mock(),
                json.dumps(model_replacement),
                (clause,),
                new_state("final-structured-authority", language="en"),
                source,
                valid,
            )

        finalized = json.loads(finalized_text)
        self.assertTrue(validation.valid, validation.errors)
        self.assertEqual(finalized["actions"][0]["config_values"], {
            "CLOUD_REGION": "us-central1",
            "CLOUD_ZONE": "us-central1-a",
        })
        self.assertEqual(finalized["actions"][0]["unmapped_values"], {
            "CLOUD_PROVIDER": "gcp",
            "OWNER_TICKET": "INC-4821",
        })

    def test_final_input_authority_restores_prose_pending_owner_after_model_recovery(self) -> None:
        import json
        from unittest.mock import Mock, patch

        from agent.harness.intent import _finalize_structured_syntax_authority
        from agent.harness.plan_coverage import PlanCoverageResult, segment_user_turn
        from agent.harness.state import new_state

        source = (
            "The node is exposed at http://geth-dev:8545.\n"
            "Treat that as LOCAL_RPC_URL, but probe it before trust."
        )
        clauses = segment_user_turn(source)
        model_replacement = {
            "actions": [{
                "type": "propose_config_values",
                "config_values": {"LOCAL_RPC_URL": "http://geth-dev:8545"},
                "unmapped_values": {},
                "conflicts": [],
                "source_format": "mixed",
                "source_evidence": source,
            }],
            "semantic_units": [
                _unit(clauses[0], 1, [0]),
                _unit(clauses[1], 2, [0]),
            ],
        }
        state = new_state("final-prose-owner", language="en")
        state["pending_question"] = {
            "id": "LOCAL_RPC_URL",
            "group": "endpoint_process",
            "field": "LOCAL_RPC_URL",
            "kind": "url",
            "manual_input_allowed": True,
            "options": [],
        }
        valid = PlanCoverageResult(True, (), ())

        with patch("agent.harness.intent._validate_semantic_fulfillment", return_value=valid):
            finalized_text, validation = _finalize_structured_syntax_authority(
                Mock(),
                json.dumps(model_replacement),
                clauses,
                state,
                source,
                valid,
            )

        finalized = json.loads(finalized_text)
        self.assertTrue(validation.valid, validation.errors)
        self.assertEqual(finalized["actions"], [{
            "type": "answer_pending",
            "answer": "http://geth-dev:8545",
            "source_evidence": "http://geth-dev:8545",
        }])

    def test_final_input_authority_restores_finite_option_after_model_recovery(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock, patch

        from agent.harness.intent import _finalize_structured_syntax_authority
        from agent.harness.plan_coverage import PlanCoverageResult, segment_user_turn
        from agent.harness.state import new_state

        source = (
            "Do not continue with the template defaults. "
            "I need to add a custom RPC method."
        )
        clauses = segment_user_turn(source)
        model_replacement = {
            "actions": [{
                "type": "clarify_unresolved",
                "clauses": [clause.text for clause in clauses],
                "reason": "late recovery lost the finite option owner",
            }],
            "semantic_units": [
                _unit(clause, index, [0], disposition="unresolved")
                for index, clause in enumerate(clauses, start=1)
            ],
        }
        state = new_state("final-option-owner", language="en")
        state["pending_question"] = {
            "id": "workload_confirm",
            "group": "workload_rpc",
            "field": "workload_choice",
            "kind": "numbered_choice",
            "manual_input_allowed": False,
            "options": [{
                "id": "default",
                "label": "Use defaults",
                "value": "default",
                "action": {"type": "use_default_workload"},
            }, {
                "id": "custom_rpc",
                "label": "Add custom RPC method",
                "value": "custom_rpc",
                "action": {"type": "rpc_catalog_command", "catalog_command": "enter"},
            }],
        }
        invalid = PlanCoverageResult(
            valid=False,
            errors=("late recovery left both clauses unresolved",),
            unresolved_clauses=tuple(clause.text for clause in clauses),
            rejected_action_indexes=(0,),
            incomplete_unit_ids=("unit-1", "unit-2"),
        )
        provider = Mock()
        provider.complete.side_effect = [
            SimpleNamespace(text=json.dumps({
                "decision": "select_option",
                "option_id": "custom_rpc",
                "evidence_quote": "add a custom RPC method",
                "supporting_unit_ids": ["unit-1", "unit-2"],
                "independent_unit_ids": [],
                "reason": "rejecting defaults supports the selected mutually exclusive custom-RPC option",
            })),
            SimpleNamespace(text=json.dumps({
                "decision": "select_option",
                "option_id": "custom_rpc",
                "evidence_quote": "add a custom RPC method",
                "supporting_unit_ids": ["unit-1", "unit-2"],
                "independent_unit_ids": [],
                "reason": "independent confirmation of the same pending selection",
            })),
        ]
        valid = PlanCoverageResult(True, (), ())

        with patch("agent.harness.intent._validate_semantic_fulfillment", return_value=valid):
            finalized_text, validation = _finalize_structured_syntax_authority(
                provider,
                json.dumps(model_replacement),
                clauses,
                state,
                source,
                invalid,
            )

        finalized = json.loads(finalized_text)
        self.assertTrue(validation.valid, validation.errors)
        self.assertEqual(finalized["actions"], [{
            "type": "rpc_catalog_command",
            "catalog_command": "enter",
            "source_evidence": "add a custom RPC method",
        }])
        self.assertEqual(finalized["pending_answer_admissions"], [0])
        self.assertEqual(
            [unit["disposition"] for unit in finalized["semantic_units"]],
            ["context", "action"],
        )
        self.assertEqual(provider.complete.call_count, 2)

    def test_final_input_authority_keeps_multifield_prose_proposal_for_review(self) -> None:
        import json
        from unittest.mock import Mock

        from agent.harness.intent import _finalize_structured_syntax_authority
        from agent.harness.plan_coverage import PlanCoverageResult, segment_user_turn
        from agent.harness.state import new_state

        source = "Use http://geth-dev:8545 and set the region to us-west-2."
        clauses = segment_user_turn(source)
        proposal = {
            "actions": [{
                "type": "propose_config_values",
                "config_values": {
                    "LOCAL_RPC_URL": "http://geth-dev:8545",
                    "CLOUD_REGION": "us-west-2",
                },
                "unmapped_values": {},
                "conflicts": [],
                "source_format": "prose",
                "source_evidence": source,
            }],
            "semantic_units": [_unit(clauses[0], 1, [0])],
        }
        state = new_state("final-multifield-owner", language="en")
        state["pending_question"] = {
            "id": "LOCAL_RPC_URL",
            "group": "endpoint_process",
            "field": "LOCAL_RPC_URL",
            "kind": "url",
            "manual_input_allowed": True,
            "options": [],
        }
        valid = PlanCoverageResult(True, (), ())

        finalized_text, validation = _finalize_structured_syntax_authority(
            Mock(), json.dumps(proposal), clauses, state, source, valid
        )

        self.assertTrue(validation.valid)
        self.assertEqual(json.loads(finalized_text)["actions"], proposal["actions"])

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

    def test_lossy_ordered_action_anchors_fall_back_to_complete_clause_transaction(self) -> None:
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _reconstruct_missing_semantic_units

        source = "Run sync-observe now, and after that keep a real-node benchmark as the next workflow."
        clauses = segment_user_turn(source)
        actions = [
            {
                "type": "choose_target_mode",
                "target_mode": "sync-observe",
                "target_mode_explicit": True,
                "source_evidence": "Run sync-observe now",
            },
            {
                "type": "queue_workflow_goal",
                "target_mode": "real-node",
                "goal": "real-node benchmark",
                "source_evidence": "real-node benchmark as the next workflow",
            },
        ]
        lossy_units = [
            {
                "unit_id": "unit-1",
                "clause_id": clauses[0].clause_id,
                "source_text": "Run sync-observe now",
                "disposition": "action",
                "action_indexes": [0],
                "reason": "current workflow",
            },
            {
                "unit_id": "unit-2",
                "clause_id": clauses[0].clause_id,
                "source_text": "real-node benchmark as the next workflow",
                "disposition": "action",
                "action_indexes": [1],
                "reason": "deferred workflow",
            },
        ]
        invalid = SimpleNamespace(text=json.dumps({"semantic_units": lossy_units}))
        provider = Mock()
        provider.complete.side_effect = [invalid, invalid]

        recovered = json.loads(_reconstruct_missing_semantic_units(
            provider,
            json.dumps({"actions": actions, "semantic_units": lossy_units}),
            clauses,
        ))

        self.assertEqual(recovered["actions"], actions)
        self.assertEqual(recovered["semantic_units"], [{
            "unit_id": "clause-1-complete-action-transaction",
            "clause_id": "clause-1",
            "source_text": source,
            "disposition": "action",
            "action_indexes": [0, 1],
            "reason": "complete source clause owns the immutable action transaction",
        }])
        self.assertTrue(validate_plan_coverage(recovered, clauses).valid)
        self.assertEqual(provider.complete.call_count, 2)

    def test_complete_clause_transaction_still_rejects_an_omitted_demand(self) -> None:
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _validate_semantic_fulfillment
        from agent.harness.state import new_state

        source = "Run sync-observe now, benchmark real-node later, and email the result."
        clauses = segment_user_turn(source)
        payload = {
            "actions": [
                {
                    "type": "choose_target_mode",
                    "target_mode": "sync-observe",
                    "target_mode_explicit": True,
                    "source_evidence": "Run sync-observe now",
                },
                {
                    "type": "queue_workflow_goal",
                    "target_mode": "real-node",
                    "goal": "benchmark real-node later",
                    "source_evidence": "benchmark real-node later",
                },
            ],
            "semantic_units": [{
                "unit_id": "clause-1-complete-action-transaction",
                "clause_id": "clause-1",
                "source_text": source,
                "disposition": "action",
                "action_indexes": [0, 1],
                "reason": "complete source clause owns the immutable action transaction",
            }],
            "target_mode_selection_admissions": [0, 1],
        }
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "unit_reviews": [{
                "unit_id": "clause-1-complete-action-transaction",
                "complete": False,
                "missing_demand_quote": "email the result",
                "reason": "the output-delivery demand has no mapped action",
            }],
        }))

        result = _validate_semantic_fulfillment(
            provider,
            json.dumps(payload),
            clauses,
            new_state("complete-clause-omission", language="en"),
        )

        self.assertFalse(result.valid)
        self.assertIn("fulfilment failed", "\n".join(result.errors))

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
        source_reviews = [
            {
                "source_text": clauses[0].text,
                "role": "support",
                "support_relation": "provenance",
                "reason": "deployment-note provenance",
            },
            {
                "source_text": clauses[1].text,
                "role": "direct",
                "support_relation": "",
                "reason": "source-grounded operation",
            },
        ]
        provider.complete.side_effect = [
            SimpleNamespace(text=json.dumps({"reviews": [
                {
                    "action_index": 0,
                    "supported": True,
                    "source_unit_reviews": source_reviews,
                    "reason": "the chain selection is source grounded",
                },
                {
                    "action_index": 1,
                    "supported": True,
                    "source_unit_reviews": source_reviews,
                    "reason": "the mapped value is source grounded",
                },
            ]})),
            SimpleNamespace(text=json.dumps({"unit_reviews": [
                {"unit_id": "unit-1", "complete": True, "reason": "framing"},
                {"unit_id": "unit-2", "complete": True, "reason": "facts preserved"},
            ]})),
        ]

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
        from agent.harness.plan_coverage import PlanCoverageResult
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
    def test_pending_manual_replacement_owns_rejection_and_new_value_as_one_transaction(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _recover_declared_pending_option_semantics
        from agent.harness.plan_coverage import PlanCoverageResult
        from agent.harness.state import new_state

        source = (
            "Do not use the detected capacity for this run.\n"
            "Set the requested capacity to 2048 GiB instead."
        )
        clauses = segment_user_turn(source)
        payload = {
            "actions": [{
                "type": "answer_pending",
                "answer": "2048",
                "source_evidence": clauses[1].text,
            }],
            "semantic_units": [
                _unit(clauses[0], 1, [0]),
                _unit(clauses[1], 2, [0]),
            ],
        }
        state = new_state("pending-manual-replacement-transaction", language="en")
        state["pending_question"] = {
            "id": "generic_capacity",
            "group": "ledger_disk",
            "field": "GENERIC_CAPACITY",
            "manual_input_allowed": True,
            "manual_action": {"type": "answer_pending", "value_argument": "answer"},
            "validation": {
                "value_type": "positive_number",
                "normalization": "semantic_scalar",
            },
            "options": [
                {
                    "id": "keep",
                    "label": "Y",
                    "value": "100",
                    "manual_entry": False,
                    "action": {"type": "answer_pending"},
                },
                {
                    "id": "replace",
                    "label": "N",
                    "value": "__manual__",
                    "manual_entry": True,
                    "action": {"type": "answer_pending"},
                },
            ],
        }
        verdict = SimpleNamespace(text=json.dumps({
            "decision": "manual_value",
            "option_id": "",
            "answer": "2048",
            "evidence_quote": "2048 GiB",
            "supporting_unit_ids": ["unit-1"],
            "independent_unit_ids": [],
            "reason": "one replacement transaction for the displayed field",
        }))
        provider = Mock()
        provider.complete.side_effect = [verdict, verdict]

        recovered_text, changed = _recover_declared_pending_option_semantics(
            provider,
            json.dumps(payload),
            state,
            PlanCoverageResult(
                valid=False,
                errors=("pending owner review required",),
                unresolved_clauses=(),
                incomplete_unit_ids=("unit-1", "unit-2"),
            ),
            clauses,
            source,
        )
        recovered = json.loads(recovered_text)

        self.assertTrue(changed)
        self.assertEqual(len(recovered["actions"]), 1)
        self.assertEqual(recovered["actions"][0]["type"], "answer_pending")
        self.assertEqual(recovered["actions"][0]["answer"], "2048")
        self.assertEqual(
            [unit["disposition"] for unit in recovered["semantic_units"]],
            ["context", "action"],
        )
        contract = provider.complete.call_args_list[0].args[0].messages[0].content
        self.assertIn("complete replacement transaction", contract)
        self.assertIn("same displayed field", contract)

    def test_group_navigation_contract_owns_value_less_revision_purpose(self) -> None:
        from agent.harness.action_registry import ACTION_BY_TYPE
        from agent.harness.intent import (
            GROUP_NAVIGATION_SEMANTIC_POLICY,
            _semantic_fulfillment_prompt,
        )

        self.assertIn("value-less statement", GROUP_NAVIGATION_SEMANTIC_POLICY)
        self.assertIn("without a concrete setting or value", GROUP_NAVIGATION_SEMANTIC_POLICY)
        self.assertIn(
            GROUP_NAVIGATION_SEMANTIC_POLICY,
            _semantic_fulfillment_prompt(review_kind="units"),
        )
        self.assertIn(
            "operation_restatement",
            ACTION_BY_TYPE["change_group"].semantic_support_relations,
        )

    def test_equivalent_owner_intake_and_navigation_share_one_group_transition(self) -> None:
        from types import SimpleNamespace
        from unittest.mock import Mock, patch

        from agent.harness.intent import _recover_registry_bounded_semantic_actions
        from agent.harness.plan_coverage import PlanCoverageResult
        from agent.harness.state import new_state

        source = (
            "I need to change the blockchain chain first; "
            "take me to the chain identity configuration."
        )
        clauses = segment_user_turn(source)
        payload = {
            "actions": [
                {
                    "type": "change_chain",
                    "chain_candidates": [],
                    "chain_text": clauses[0].text,
                    "source_evidence": clauses[0].text,
                },
                {
                    "type": "change_group",
                    "group": "chain_identity",
                    "navigation_explicit": True,
                    "source_evidence": clauses[1].text,
                },
            ],
            "semantic_units": [
                _unit(clauses[0], 1, [0]),
                _unit(clauses[1], 2, [1]),
            ],
        }
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({"decisions": [
            {
                "unit_id": "unit-1",
                "disposition": "owner_mutation",
                "group": "chain_identity",
                "existing_action_index": None,
                "target_mode": "",
                "consultation_topic": "",
                "turn_local_action_type": "",
                "scope_constraint": "",
                "evidence_quote": clauses[0].text,
                "reason": "requests a chain change without a replacement value",
            },
        ]}))

        with patch("agent.harness.intent._resolve_owned_group_mutation", return_value=None):
            recovered_text, changed = _recover_registry_bounded_semantic_actions(
                provider,
                json.dumps(payload),
                clauses,
                new_state("equivalent-chain-navigation", language="en"),
                source,
                PlanCoverageResult(
                    valid=False,
                    errors=("invalid value for change_chain.chain_candidates: too few items",),
                    unresolved_clauses=(clauses[0].text,),
                    rejected_action_indexes=(0,),
                    incomplete_unit_ids=("unit-1",),
                ),
            )

        recovered = json.loads(recovered_text)
        self.assertTrue(changed)
        self.assertEqual(len(recovered["actions"]), 1)
        self.assertEqual(recovered["actions"][0]["type"], "request_chain_selection")
        self.assertEqual(recovered["actions"][0]["source_evidence"], clauses[0].text)
        self.assertEqual(
            [unit["action_indexes"] for unit in recovered["semantic_units"]],
            [[0], [0]],
        )

    def test_manual_pending_recovery_receives_declared_completion_effect(self) -> None:
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _recover_declared_pending_option_semantics
        from agent.harness.plan_coverage import PlanCoverageResult
        from agent.harness.state import new_state

        source = "http://fake-node:19000\nProbe this exact endpoint before trusting it."
        clauses = segment_user_turn(source)
        payload = {
            "actions": [{
                "type": "answer_pending",
                "answer": "http://fake-node:19000",
                "source_evidence": "http://fake-node:19000",
            }],
            "semantic_units": [
                _unit(clauses[0], 1, [0]),
                _unit(clauses[1], 2, [], disposition="unresolved"),
            ],
        }
        state = new_state("manual-completion-scope", language="en")
        state["pending_question"] = {
            "id": "LOCAL_RPC_URL",
            "group": "endpoint_process",
            "kind": "manual_value",
            "field": "LOCAL_RPC_URL",
            "manual_input_allowed": True,
            "validation": {"value_type": "url"},
            "options": [],
            "completion_effect": "Probe this endpoint and record validation evidence before continuing.",
        }
        verdict = SimpleNamespace(text=json.dumps({
            "decision": "manual_value",
            "option_id": "",
            "answer": "http://fake-node:19000",
            "evidence_quote": "http://fake-node:19000",
            "supporting_unit_ids": ["unit-2"],
            "independent_unit_ids": [],
            "reason": "the second line restates the declared completion effect",
        }))
        provider = Mock()
        provider.complete.side_effect = [verdict, verdict]

        _recover_declared_pending_option_semantics(
            provider,
            json.dumps(payload),
            state,
            PlanCoverageResult(False, ("unresolved",), (clauses[1].text,), (), ()),
            clauses,
            source,
        )

        request_payload = json.loads(provider.complete.call_args_list[0].args[0].messages[-1].content)
        self.assertIn("Probe this endpoint", request_payload["pending_question"]["completion_effect"])

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
                "disposition": "registered_action",
                "group": "",
                "existing_action_index": None,
                "target_mode": "",
                "consultation_topic": "",
                "registered_action_type": "analyze_evidence",
                "support_relation": "",
                "scope_constraint": "",
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
        analyze_contract = next(
            item
            for item in request_payload["recoverable_registered_actions"]
            if item["type"] == "analyze_evidence"
        )
        self.assertEqual(
            analyze_contract,
            {
                "type": "analyze_evidence",
                "purpose": "Analyze pasted or previously collected logs/errors/evidence without becoming a deferred workflow command.",
                "source_argument": "evidence",
                "effect": "read_only",
                "support_relations": [
                    "explanatory_context",
                    "operation_restatement",
                    "provenance",
                    "format_scope",
                    "temporal_scope",
                ],
            },
        )

    def test_lifecycle_operation_restatement_shares_one_recovered_action(self) -> None:
        import json

        from agent.harness.intent import _recover_registry_bounded_semantic_actions
        from agent.harness.state import new_state

        cases = (
            (
                "discard_next_workflow_goal",
                "Remove the queued follow-up task; I no longer want the saved sync observation.",
                "Remove the queued follow-up task",
                "I no longer want the saved sync observation",
            ),
            (
                "activate_next_workflow_goal",
                "我仍然需要之前保存的观察任务；现在继续执行那个待办。",
                "现在继续执行那个待办",
                "我仍然需要之前保存的观察任务",
            ),
        )
        for action_type, user_text, direct_quote, support_quote in cases:
            with self.subTest(action_type=action_type):
                clauses = segment_user_turn(user_text)
                payload = {
                    "actions": [],
                    "semantic_units": [
                        self._unit(clause, f"unit-{index}", [], disposition="unresolved")
                        for index, clause in enumerate(clauses, start=1)
                    ],
                }
                direct_unit = next(
                    f"unit-{index}"
                    for index, clause in enumerate(clauses, start=1)
                    if direct_quote in clause.text
                )
                self.assertTrue(any(support_quote in clause.text for clause in clauses))
                decisions = []
                for index, clause in enumerate(clauses, start=1):
                    unit_id = f"unit-{index}"
                    decisions.append({
                        "unit_id": unit_id,
                        "disposition": (
                            "registered_action"
                            if unit_id == direct_unit
                            else "registered_action_support"
                        ),
                        "group": "",
                        "existing_action_index": None,
                        "target_mode": "",
                        "consultation_topic": "",
                        "registered_action_type": action_type,
                        "support_relation": (
                            "" if unit_id == direct_unit else "operation_restatement"
                        ),
                        "scope_constraint": "",
                        "evidence_quote": clause.text,
                        "reason": "direct lifecycle command or same-operation restatement",
                    })
                provider = self._provider(decisions)

                recovered_text, changed = _recover_registry_bounded_semantic_actions(
                    provider,
                    json.dumps(payload),
                    clauses,
                    new_state(f"lifecycle-restatement-{action_type}", language="en"),
                    user_text,
                )
                recovered = json.loads(recovered_text)

                self.assertTrue(changed)
                self.assertEqual(len(recovered["actions"]), 1)
                self.assertEqual(recovered["actions"][0]["type"], action_type)
                self.assertTrue(all(
                    unit["action_indexes"] == [0]
                    for unit in recovered["semantic_units"]
                ))

    def test_operation_restatement_cannot_authorize_lifecycle_action_by_itself(self) -> None:
        import json

        from agent.harness.intent import _recover_registry_bounded_semantic_actions
        from agent.harness.state import new_state

        user_text = "I no longer want the saved sync observation."
        clauses = segment_user_turn(user_text)
        payload = {
            "actions": [],
            "semantic_units": [self._unit(
                clauses[0], "unit-1", [], disposition="unresolved"
            )],
        }
        provider = self._provider([{
            "unit_id": "unit-1",
            "disposition": "registered_action_support",
            "group": "",
            "existing_action_index": None,
            "target_mode": "",
            "consultation_topic": "",
            "registered_action_type": "discard_next_workflow_goal",
            "support_relation": "operation_restatement",
            "scope_constraint": "",
            "evidence_quote": clauses[0].text,
            "reason": "support without a direct operation",
        }])

        recovered_text, changed = _recover_registry_bounded_semantic_actions(
            provider,
            json.dumps(payload),
            clauses,
            new_state("restatement-without-direct", language="en"),
            user_text,
        )

        self.assertFalse(changed)
        self.assertEqual(json.loads(recovered_text), payload)

    def test_independent_duplicate_lifecycle_demands_do_not_merge_partially(self) -> None:
        import json

        from agent.harness.intent import _recover_registry_bounded_semantic_actions
        from agent.harness.state import new_state

        user_text = "Discard the oldest saved goal; then discard the next saved goal too."
        clauses = segment_user_turn(user_text)
        payload = {
            "actions": [],
            "semantic_units": [
                self._unit(clause, f"unit-{index}", [], disposition="unresolved")
                for index, clause in enumerate(clauses, start=1)
            ],
        }
        provider = self._provider([
            {
                "unit_id": f"unit-{index}",
                "disposition": "registered_action",
                "group": "",
                "existing_action_index": None,
                "target_mode": "",
                "consultation_topic": "",
                "registered_action_type": "discard_next_workflow_goal",
                "support_relation": "",
                "scope_constraint": "",
                "evidence_quote": clause.text,
                "reason": "each clause requests a separate queue mutation",
            }
            for index, clause in enumerate(clauses, start=1)
        ])

        recovered_text, changed = _recover_registry_bounded_semantic_actions(
            provider,
            json.dumps(payload),
            clauses,
            new_state("duplicate-discard-demands", language="en"),
            user_text,
        )

        self.assertFalse(changed)
        self.assertEqual(json.loads(recovered_text), payload)

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
            "decision": "select_option",
            "option_id": "1",
            "evidence_quote": "Continue the saved setup",
            "supporting_unit_ids": ["unit-1"],
            "independent_unit_ids": [],
            "reason": "selects the declared continuation option",
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

    def test_single_pending_field_proposal_from_prose_compiles_to_pending_answer(self) -> None:
        from agent.harness.intent import _normalize_prose_pending_field_proposal
        from agent.harness.state import new_state

        source = (
            "The credential is run-secret-4821.\n"
            "Use it only for this run without changing the saved template."
        )
        clauses = segment_user_turn(source)
        state = new_state("prose-pending-proposal")
        state["pending_question"] = {
            "id": "RPC_API_KEY",
            "field": "RPC_API_KEY",
            "kind": "manual_value",
            "manual_input_allowed": True,
            "validation": {"value_type": "scalar_token", "max_length": 180},
        }
        payload = {
            "actions": [{
                "type": "propose_config_values",
                "config_values": {"RPC_API_KEY": "run-secret-4821"},
                "unmapped_values": {},
                "conflicts": [],
                "source_format": "mixed",
                "source_evidence": clauses[-1].text,
            }],
            "semantic_units": [
                _unit(clause, index, [0])
                for index, clause in enumerate(clauses, start=1)
            ],
        }

        normalized = json.loads(_normalize_prose_pending_field_proposal(
            json.dumps(payload),
            state,
            clauses,
            source,
        ))

        self.assertEqual(normalized["actions"], [{
            "type": "answer_pending",
            "answer": "run-secret-4821",
            "source_evidence": "run-secret-4821",
        }])
        self.assertEqual(
            [unit["action_indexes"] for unit in normalized["semantic_units"]],
            [[0], [0]],
        )

    def test_structured_pending_field_proposal_remains_a_review_transaction(self) -> None:
        from agent.harness.intent import _normalize_prose_pending_field_proposal
        from agent.harness.state import new_state

        source = '{"RPC_API_KEY":"run-secret-4821"}'
        clauses = segment_user_turn(source)
        state = new_state("structured-pending-proposal")
        state["pending_question"] = {
            "id": "RPC_API_KEY",
            "field": "RPC_API_KEY",
            "kind": "manual_value",
            "manual_input_allowed": True,
            "validation": {"value_type": "scalar_token", "max_length": 180},
        }
        payload = {
            "actions": [{
                "type": "propose_config_values",
                "config_values": {"RPC_API_KEY": "run-secret-4821"},
                "unmapped_values": {},
                "conflicts": [],
            }],
            "semantic_units": [_unit(clauses[0], 1, [0])],
        }

        normalized = _normalize_prose_pending_field_proposal(
            json.dumps(payload),
            state,
            clauses,
            source,
        )

        self.assertEqual(json.loads(normalized), payload)

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

    def test_declared_pending_semantic_recovers_joint_multi_clause_answer(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _recover_declared_pending_option_semantics
        from agent.harness.plan_coverage import PlanCoverageResult
        from agent.harness.state import new_state

        source = (
            "No, I really do mean a separate chain named sola. "
            "It is not Solana; continue by determining its protocol family."
        )
        clauses = segment_user_turn(source)
        self.assertEqual(len(clauses), 3)
        payload = {
            "actions": [],
            "semantic_units": [
                {
                    "unit_id": "unit-1",
                    "clause_id": clauses[0].clause_id,
                    "source_text": clauses[0].text,
                    "disposition": "unresolved",
                    "action_indexes": [],
                },
                {
                    "unit_id": "unit-2",
                    "clause_id": clauses[1].clause_id,
                    "source_text": clauses[1].text,
                    "disposition": "unresolved",
                    "action_indexes": [],
                },
                {
                    "unit_id": "unit-3",
                    "clause_id": clauses[2].clause_id,
                    "source_text": clauses[2].text,
                    "disposition": "unresolved",
                    "action_indexes": [],
                },
            ],
        }
        state = new_state("pending-complete-split-clause", language="en")
        state["pending_question"] = {
            "id": "unknown_chain_identity_confirm",
            "prompt": "Use the suggested chain?",
            "options": [
                {"id": "known", "value": "confirm_known_chain", "action": {"type": "answer_pending"}},
                {"id": "protocol", "value": "choose_protocol", "action": {"type": "answer_pending"}},
            ],
        }
        provider = Mock()

        def complete(request):
            request_payload = json.loads(request.messages[-1].content)
            contract_payload = request_payload.get("pending_contract", request_payload)
            self.assertEqual(contract_payload["complete_turn"], [clause.as_dict() for clause in clauses])
            self.assertEqual(len(contract_payload["units"]), 3)
            return SimpleNamespace(text=json.dumps({
                "decision": "select_option",
                "option_id": "protocol",
                "evidence_quote": "continue by determining its protocol family",
                "supporting_unit_ids": ["unit-1", "unit-2", "unit-3"],
                "independent_unit_ids": [],
                "reason": "the complete turn selects protocol discovery",
            }))

        provider.complete.side_effect = complete
        recovered_text, changed = _recover_declared_pending_option_semantics(
            provider,
            json.dumps(payload),
            state,
            PlanCoverageResult(
                valid=False,
                errors=("three unresolved clauses",),
                unresolved_clauses=tuple(clause.text for clause in clauses),
            ),
            clauses,
        )
        recovered = json.loads(recovered_text)

        self.assertTrue(changed)
        self.assertEqual(len(recovered["semantic_units"]), 3)
        self.assertEqual(
            [unit["disposition"] for unit in recovered["semantic_units"]],
            ["context", "context", "action"],
        )
        self.assertEqual(recovered["semantic_units"][2]["action_indexes"], [0])
        self.assertEqual(recovered["actions"], [{
            "type": "answer_pending",
            "answer": "choose_protocol",
            "selected_value": "choose_protocol",
            "source_evidence": "continue by determining its protocol family",
        }])

    def test_declared_pending_owner_treats_declared_correction_continuation_as_support(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _recover_declared_pending_option_semantics
        from agent.harness.plan_coverage import PlanCoverageResult
        from agent.harness.state import new_state

        source = (
            "No, that response contract is incomplete. "
            "Let me provide the correct response evidence."
        )
        clauses = segment_user_turn(source)
        payload = {
            "actions": [{
                "type": "clarify_unresolved",
                "clauses": [clause.text for clause in clauses],
                "reason": "planner did not bind the finite option",
            }],
            "semantic_units": [
                self._unit(clause, f"unit-{index}", [0], disposition="unresolved")
                for index, clause in enumerate(clauses, start=1)
            ],
        }
        state = new_state("pending-response-correction", language="en")
        state["pending_question"] = {
            "id": "custom_rpc_response_confirm",
            "group": "endpoint_process",
            "kind": "yes_no",
            "manual_input_allowed": False,
            "options": [
                {
                    "id": "yes",
                    "label": "Y",
                    "value": True,
                    "action": {"type": "answer_pending", "answer": True},
                },
                {
                    "id": "no",
                    "label": "N",
                    "value": False,
                    "action": {"type": "answer_pending", "answer": False},
                },
            ],
        }
        provider = Mock()
        selection = SimpleNamespace(text=json.dumps({
            "decision": "select_option",
            "option_id": "no",
            "evidence_quote": "No, that response contract is incomplete",
            "supporting_unit_ids": ["unit-1", "unit-2"],
            "independent_unit_ids": [],
            "reason": "rejects the observed contract and commits to its correction flow",
        }))
        provider.complete.side_effect = [
            selection,
            selection,
            SimpleNamespace(text=json.dumps({
                "entailed": True,
                "contradicted": False,
                "evidence_quote": "No, that response contract is incomplete",
                "reason": "the user explicitly rejects the observed response contract",
            })),
        ]

        recovered_text, changed = _recover_declared_pending_option_semantics(
            provider,
            json.dumps(payload),
            state,
            PlanCoverageResult(
                valid=False,
                errors=("unresolved finite option",),
                unresolved_clauses=tuple(clause.text for clause in clauses),
                rejected_action_indexes=(0,),
                incomplete_unit_ids=("unit-1", "unit-2"),
            ),
            clauses,
            source,
        )
        recovered = json.loads(recovered_text)

        self.assertTrue(changed)
        self.assertEqual(recovered["actions"], [{
            "type": "answer_pending",
            "answer": False,
            "selected_value": False,
            "source_evidence": "No, that response contract is incomplete",
        }])
        self.assertEqual(recovered["pending_answer_admissions"], [0])
        self.assertEqual(
            [unit["disposition"] for unit in recovered["semantic_units"]],
            ["action", "context"],
        )

    def test_declared_pending_semantic_keeps_ambiguous_split_clause_unresolved(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _recover_declared_pending_option_semantics
        from agent.harness.plan_coverage import PlanCoverageResult
        from agent.harness.state import new_state

        source = "Use the suggested chain. Or maybe treat it as another chain; I am not sure."
        clauses = segment_user_turn(source)
        self.assertGreater(len(clauses), 1)
        original = json.dumps({
            "actions": [],
            "semantic_units": [
                {
                    "unit_id": "unit-1",
                    "clause_id": clauses[0].clause_id,
                    "source_text": clauses[0].text,
                    "disposition": "unresolved",
                    "action_indexes": [],
                },
                {
                    "unit_id": "unit-2",
                    "clause_id": clauses[1].clause_id,
                    "source_text": clauses[1].text,
                    "disposition": "unresolved",
                    "action_indexes": [],
                },
            ],
        })
        state = new_state("pending-ambiguous-split-clause", language="en")
        state["pending_question"] = {
            "id": "unknown_chain_identity_confirm",
            "options": [
                {"id": "known", "value": "confirm_known_chain", "action": {"type": "answer_pending"}},
                {"id": "protocol", "value": "choose_protocol", "action": {"type": "answer_pending"}},
            ],
        }
        provider = Mock()

        recovered_text, changed = _recover_declared_pending_option_semantics(
            provider,
            original,
            state,
            PlanCoverageResult(
                valid=False,
                errors=("ambiguous unresolved units",),
                unresolved_clauses=tuple(clause.text for clause in clauses),
            ),
            clauses,
        )

        self.assertFalse(changed)
        self.assertEqual(recovered_text, original)

    def test_declared_pending_semantic_conflict_remains_unresolved(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _recover_declared_pending_option_semantics
        from agent.harness.state import new_state

        user_text = "Discard that partial setup and let me start clean."
        clauses = segment_user_turn(user_text)
        payload = {
            "actions": [],
            "semantic_units": [self._unit(clauses[0], "unit-1", [], disposition="unresolved")],
        }
        state = new_state("pending-imperative-retry", language="en")
        state["pending_question"] = {
            "id": "resume_harness_session",
            "prompt": "Resume the saved configuration?",
            "options": [
                {"id": "continue", "value": "continue", "semantic_action": "continue_current_flow"},
                {"id": "modify", "value": "modify"},
                {"id": "reset", "label": "Clear and start over", "value": "reset"},
            ],
        }
        provider = Mock()
        provider.complete.side_effect = [
            SimpleNamespace(text=json.dumps({})),
            SimpleNamespace(text=json.dumps({
                "decision": "select_option",
                "option_id": "reset",
                "evidence_quote": "Discard that partial setup and let me start clean",
                "supporting_unit_ids": ["unit-1"],
                "independent_unit_ids": [],
                "reason": "imperative paraphrase selects the declared reset effect",
            })),
        ]

        recovered_text, changed = _recover_declared_pending_option_semantics(
            provider,
            json.dumps(payload),
            state,
            clauses=clauses,
        )
        self.assertFalse(changed)
        self.assertEqual(provider.complete.call_count, 2)
        self.assertEqual(recovered_text, json.dumps(payload))

    def test_declared_pending_semantic_second_empty_adjudication_remains_unresolved(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _recover_declared_pending_option_semantics
        from agent.harness.state import new_state

        user_text = "Maybe continue, or clear it; I have not decided."
        clauses = segment_user_turn(user_text)
        original = json.dumps({
            "actions": [],
            "semantic_units": [self._unit(clauses[0], "unit-1", [], disposition="unresolved")],
        })
        state = new_state("pending-ambiguous-retry", language="en")
        state["pending_question"] = {
            "id": "resume_harness_session",
            "options": [
                {"id": "continue", "value": "continue", "semantic_action": "continue_current_flow"},
                {"id": "reset", "value": "reset"},
            ],
        }
        provider = Mock()
        provider.complete.side_effect = [
            SimpleNamespace(text=json.dumps({})),
            SimpleNamespace(text=json.dumps({"matches": [], "manual_matches": [], "contexts": []})),
        ]

        recovered_text, changed = _recover_declared_pending_option_semantics(
            provider,
            original,
            state,
            clauses=clauses,
        )

        self.assertFalse(changed)
        self.assertEqual(recovered_text, original)
        self.assertEqual(provider.complete.call_count, 2)

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
            "decision": "select_option",
            "option_id": "custom_rpc",
            "evidence_quote": "use my own RPC method instead of the default",
            "supporting_unit_ids": ["unit-1"],
            "independent_unit_ids": [],
            "reason": "selects the declared custom RPC option",
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
            "decision": "select_option",
            "option_id": "custom_rpc",
            "evidence_quote": "Use my own RPC method",
            "supporting_unit_ids": ["unit-1"],
            "independent_unit_ids": [],
            "reason": "selects the declared custom RPC option",
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

    def test_declared_pending_owner_does_not_reopen_valid_planner_effect(self) -> None:
        import json
        from unittest.mock import Mock

        from agent.harness.intent import _recover_declared_pending_option_semantics
        from agent.harness.state import new_state

        source = "Replace the single workload with this custom method."
        clauses = segment_user_turn(source)
        payload = {
            "actions": [{
                "type": "rpc_catalog_command",
                "catalog_command": "enter",
                "source_evidence": source,
            }],
            "semantic_units": [self._unit(clauses[0], "unit-1", [0])],
        }
        state = new_state("pending-owner-early-admission", language="en")
        state["pending_question"] = {
            "id": "custom_rpc_scope",
            "group": "rpc_workload",
            "options": [{
                "id": "single_replace",
                "label": "Replace single workload",
                "value": "single_replace",
                "action": {"type": "rpc_catalog_command", "catalog_command": "enter"},
            }],
        }
        provider = Mock()

        recovered_text, changed = _recover_declared_pending_option_semantics(
            provider,
            json.dumps(payload),
            state,
            clauses=clauses,
        )
        recovered = json.loads(recovered_text)

        self.assertFalse(changed)
        self.assertEqual(provider.complete.call_count, 0)
        self.assertEqual(recovered["actions"][0]["type"], "rpc_catalog_command")

    def test_pending_url_owner_recovers_value_without_erasing_grounded_mode_action(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _recover_declared_pending_option_semantics
        from agent.harness.state import new_state

        source = (
            "The local Geth endpoint is http://geth-dev:8545.\n"
            "Use it for sync observation and verify it first."
        )
        clauses = segment_user_turn(source)
        payload = {
            "actions": [{
                "type": "choose_target_mode",
                "target_mode": "sync-observe",
                "target_mode_explicit": True,
                "source_evidence": "sync observation",
            }],
            "semantic_units": [
                self._unit(clauses[0], "unit-1", [0]),
                self._unit(clauses[1], "unit-2", [0]),
            ],
        }
        state = new_state("pending-url-owner", language="en")
        state["pending_question"] = {
            "id": "SYNC_OBSERVE_RPC_URL",
            "group": "endpoint_process",
            "field": "SYNC_OBSERVE_RPC_URL",
            "kind": "url",
            "manual_input_allowed": True,
            "accepted_action_types": ["answer_pending"],
            "options": [],
        }
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "decision": "manual_value",
            "option_id": "",
            "answer": "http://geth-dev:8545",
            "evidence_quote": "http://geth-dev:8545",
            "supporting_unit_ids": ["unit-1"],
            "independent_unit_ids": ["unit-2"],
            "reason": "the first sentence supplies the requested URL",
        }))

        recovered_text, changed = _recover_declared_pending_option_semantics(
            provider, json.dumps(payload), state, clauses=clauses, user_text=source
        )
        recovered = json.loads(recovered_text)

        self.assertTrue(changed)
        self.assertEqual(
            {item["type"] for item in recovered["actions"]},
            {"answer_pending", "choose_target_mode"},
        )
        pending_answer = next(item for item in recovered["actions"] if item["type"] == "answer_pending")
        self.assertEqual(pending_answer["answer"], "http://geth-dev:8545")

    def test_pending_option_accepts_explanation_for_declared_domain_action(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _recover_declared_pending_option_semantics
        from agent.harness.state import new_state

        source = "The defaults do not include the call I need. Add a custom RPC method."
        clauses = segment_user_turn(source)
        payload = {
            "actions": [{
                "type": "clarify_unresolved",
                "clauses": [clause.text for clause in clauses],
                "reason": "unresolved",
            }],
            "semantic_units": [
                self._unit(clauses[0], "unit-1", [0]),
                self._unit(clauses[1], "unit-2", [0]),
            ],
        }
        state = new_state("pending-option-explanation", language="en")
        state["pending_question"] = {
            "id": "workload_confirm",
            "group": "workload_rpc",
            "field": "workload_choice",
            "manual_input_allowed": False,
            "accepted_action_types": ["answer_pending", "rpc_catalog_command"],
            "options": [{
                "id": "custom_rpc",
                "label": "Add custom RPC method",
                "value": "custom_rpc",
                "action": {"type": "rpc_catalog_command", "catalog_command": "enter"},
            }],
        }
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "decision": "select_option",
            "option_id": "custom_rpc",
            "answer": "",
            "evidence_quote": "Add a custom RPC method",
            "supporting_unit_ids": ["unit-1", "unit-2"],
            "independent_unit_ids": [],
            "reason": "the first sentence explains why the declared option is needed",
        }))

        recovered_text, changed = _recover_declared_pending_option_semantics(
            provider, json.dumps(payload), state, clauses=clauses, user_text=source
        )
        recovered = json.loads(recovered_text)

        self.assertTrue(changed)
        self.assertEqual(recovered["actions"], [{
            "type": "rpc_catalog_command",
            "catalog_command": "enter",
            "source_evidence": "Add a custom RPC method",
        }])

    def test_pending_option_reviews_context_labeled_choice_and_rationale(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _recover_declared_pending_option_semantics
        from agent.harness.state import new_state

        source = (
            "Use the fake-node option for this run. "
            "I want to validate the complete framework loop before connecting a real endpoint."
        )
        clauses = segment_user_turn(source)
        payload = {
            "actions": [],
            "semantic_units": [
                self._unit(clauses[0], "unit-choice", [], disposition="context"),
                self._unit(clauses[1], "unit-purpose", [], disposition="context"),
            ],
        }
        state = new_state("pending-context-choice", language="en")
        state["pending_question"] = {
            "id": "target_mode_select",
            "group": "entry_mode",
            "prompt": "Choose the target mode.",
            "manual_input_allowed": False,
            "accepted_action_types": ["choose_target_mode"],
            "options": [{
                "id": "fake-node",
                "label": "fake-node",
                "value": "fake-node",
                "action": {
                    "type": "choose_target_mode",
                    "target_mode": "fake-node",
                    "target_mode_explicit": True,
                },
            }],
        }
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "decision": "select_option",
            "option_id": "fake-node",
            "answer": "",
            "evidence_quote": "Use the fake-node option",
            "supporting_unit_ids": ["unit-choice", "unit-purpose"],
            "independent_unit_ids": [],
            "reason": "the first unit selects the option and the second explains its purpose",
        }))

        recovered_text, changed = _recover_declared_pending_option_semantics(
            provider, json.dumps(payload), state, clauses=clauses, user_text=source
        )
        recovered = json.loads(recovered_text)

        self.assertTrue(changed)
        self.assertEqual([item["type"] for item in recovered["actions"]], ["choose_target_mode"])
        self.assertEqual(recovered["actions"][0]["target_mode"], "fake-node")
        self.assertEqual(recovered["semantic_units"][0]["disposition"], "action")
        self.assertEqual(recovered["semantic_units"][1]["disposition"], "context")

    def test_pending_option_contract_prioritizes_declared_omission_over_manual_value(self) -> None:
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _recover_declared_pending_option_semantics
        from agent.harness.state import new_state

        source = "Skip this optional credential; leave it unconfigured."
        clauses = segment_user_turn(source)
        payload = {
            "actions": [{
                "type": "answer_pending",
                "answer": "Skip",
                "source_evidence": "Skip",
            }],
            "semantic_units": [
                self._unit(clause, f"unit-{index}", [0], disposition="action")
                for index, clause in enumerate(clauses, start=1)
            ],
        }
        state = new_state("pending-optional-omission", language="en")
        state["pending_question"] = {
            "id": "optional_credential",
            "group": "chain_auxiliary_endpoints",
            "field": "OPTIONAL_CREDENTIAL",
            "kind": "manual_value",
            "prompt": "Supply the optional credential or skip it.",
            "manual_input_allowed": True,
            "validation": {"value_type": "scalar_token", "max_length": 180},
            "options": [{
                "id": "omit",
                "label": "Skip (not configured)",
                "value": "none",
                "action": {"type": "answer_pending"},
                "expected_patch": {"confirmed_config.OPTIONAL_CREDENTIAL": "none"},
            }],
        }
        verdict = SimpleNamespace(text=json.dumps({
            "decision": "select_option",
            "option_id": "omit",
            "answer": "",
            "evidence_quote": "Skip this optional credential",
            "supporting_unit_ids": ["unit-1", "unit-2"],
            "independent_unit_ids": [],
            "reason": "the complete turn selects the declared omission effect",
        }))
        provider = Mock()
        provider.complete.side_effect = [verdict, verdict]

        recovered_text, changed = _recover_declared_pending_option_semantics(
            provider,
            json.dumps(payload),
            state,
            clauses=clauses,
            user_text=source,
        )
        recovered = json.loads(recovered_text)

        self.assertTrue(changed)
        self.assertEqual(recovered["actions"], [{
            "type": "answer_pending",
            "answer": "none",
            "selected_value": "none",
            "source_evidence": "Skip this optional credential",
        }])
        contract = provider.complete.call_args_list[0].args[0].messages[0].content
        self.assertIn("finite declared option owns it", contract)

    def test_pending_read_only_option_owns_non_mutating_scope_context(self) -> None:
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _recover_declared_pending_option_semantics
        from agent.harness.state import new_state

        source = (
            "I am not ready to configure a benchmark. "
            "Show me the supported chains, RPC methods, and extension paths."
        )
        clauses = segment_user_turn(source)
        payload = {
            "actions": [{
                "type": "clarify_unresolved",
                "clauses": [clause.text for clause in clauses],
                "reason": "the turn was split before pending ownership",
            }],
            "semantic_units": [
                self._unit(clause, f"unit-{index}", [0], disposition="unresolved")
                for index, clause in enumerate(clauses, start=1)
            ],
        }
        state = new_state("pending-read-only-scope", language="en")
        state["pending_question"] = {
            "id": "entry_choice",
            "group": "opening",
            "field": "target_mode",
            "kind": "numbered_choice",
            "prompt": "What would you like help with?",
            "manual_input_allowed": False,
            "options": [{
                "id": "guide",
                "label": "Learn supported chains, RPC methods, and extension paths",
                "description": "Read-only product guidance without changing configuration.",
                "value": "info",
                "action": {"type": "answer_opening_question", "topic": "capabilities"},
                "expected_patch": {},
                "return_policy": "stop_after_response",
            }],
        }
        verdict = SimpleNamespace(text=json.dumps({
            "decision": "select_option",
            "option_id": "guide",
            "answer": "",
            "evidence_quote": "Show me the supported chains, RPC methods, and extension paths",
            "supporting_unit_ids": ["unit-1", "unit-2"],
            "independent_unit_ids": [],
            "reason": "the first unit limits mutations and the second requests one read-only option",
        }))
        provider = Mock()
        provider.complete.side_effect = [verdict, verdict]

        recovered_text, changed = _recover_declared_pending_option_semantics(
            provider,
            json.dumps(payload),
            state,
            clauses=clauses,
            user_text=source,
        )
        recovered = json.loads(recovered_text)

        self.assertTrue(changed)
        self.assertEqual(recovered["actions"], [{
            "type": "answer_opening_question",
            "topic": "capabilities",
            "source_evidence": "Show me the supported chains, RPC methods, and extension paths",
        }])
        contract = provider.complete.call_args_list[0].args[0].messages[0].content
        self.assertIn("non-mutating scope context", contract)

    def test_pending_owned_consultation_cannot_be_rewritten_by_generic_owner(self) -> None:
        from unittest.mock import Mock

        from agent.harness.intent import _adjudicate_consultation_actions
        from agent.harness.state import new_state

        payload = {
            "actions": [{
                "type": "answer_opening_question",
                "topic": "capabilities",
                "source_evidence": "Show me the supported chains and extension paths",
            }],
            "pending_answer_admissions": [0],
            "semantic_units": [],
        }
        provider = Mock()

        result, changed = _adjudicate_consultation_actions(
            provider,
            json.dumps(payload),
            new_state("pending-consultation-owner", language="en"),
        )

        self.assertFalse(changed)
        self.assertEqual(json.loads(result), payload)
        provider.complete.assert_not_called()

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

    def test_pending_option_partitions_rationale_sharing_provisional_action(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _recover_declared_pending_option_semantics
        from agent.harness.state import new_state

        source = (
            "Use the fake-node option for this run. "
            "I want to validate the complete framework loop before connecting a real endpoint."
        )
        clauses = segment_user_turn(source)
        payload = {
            "actions": [{
                "type": "choose_target_mode",
                "target_mode": "fake-node",
                "target_mode_explicit": True,
                "source_evidence": clauses[0].text,
            }],
            "semantic_units": [
                self._unit(clauses[0], "unit-choice", [0]),
                self._unit(clauses[1], "unit-purpose", [0]),
            ],
        }
        state = new_state("pending-provisional-anchor", language="en")
        state["pending_question"] = {
            "id": "target_mode_select",
            "group": "entry_mode",
            "prompt": "Choose the target mode.",
            "manual_input_allowed": False,
            "accepted_action_types": ["choose_target_mode"],
            "options": [{
                "id": "fake-node",
                "label": "fake-node",
                "value": "fake-node",
                "action": {
                    "type": "choose_target_mode",
                    "target_mode": "fake-node",
                    "target_mode_explicit": True,
                },
            }],
        }
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "conflict": False,
            "ambiguous": False,
            "supporting_unit_ids": ["unit-purpose"],
            "independent_unit_ids": [],
            "reason": "the remaining unit explains why the immutable selection is appropriate",
        }))

        recovered_text, changed = _recover_declared_pending_option_semantics(
            provider, json.dumps(payload), state, clauses=clauses, user_text=source
        )
        recovered = json.loads(recovered_text)

        self.assertTrue(changed)
        self.assertNotIn("pending_answer_admissions", recovered)
        self.assertEqual(recovered["pending_support_unit_ids"], ["unit-purpose"])
        self.assertEqual(provider.complete.call_count, 2)
        self.assertEqual(recovered["semantic_units"][0]["action_indexes"], [0])
        self.assertEqual(recovered["semantic_units"][1]["action_indexes"], [])
        self.assertEqual(recovered["semantic_units"][1]["disposition"], "context")
        review_payload = json.loads(provider.complete.call_args_list[0].args[0].messages[-1].content)
        self.assertEqual(review_payload["selected_option"], {
            "option_id": "fake-node",
            "label": "fake-node",
            "description": "",
            "manual_entry": False,
            "semantic_action": "",
            "value": "fake-node",
            "declared_action": {
                "type": "choose_target_mode",
                "target_mode": "fake-node",
                "target_mode_explicit": True,
            },
            "expected_patch": {},
            "question_prompt": "Choose the target mode.",
            "completion_effect": "",
        })

    def test_pending_option_anchor_carries_opening_effect_for_explanatory_rationale(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _recover_declared_pending_option_semantics
        from agent.harness.state import new_state

        source = "Start the fake-node benchmark. I need the fastest framework check first."
        clauses = segment_user_turn(source)
        payload = {
            "actions": [
                {
                    "type": "choose_target_mode",
                    "target_mode": "fake-node",
                    "target_mode_explicit": True,
                    "source_evidence": clauses[0].text,
                },
                {
                    "type": "clarify_unresolved",
                    "clauses": [clauses[1].text],
                    "reason": "the second sentence was not mapped",
                },
            ],
            "semantic_units": [
                self._unit(clauses[0], "unit-choice", [0]),
                self._unit(clauses[1], "unit-purpose", [1], disposition="unresolved"),
            ],
        }
        state = new_state("opening-option-rationale", language="en")
        state["pending_question"] = {
            "id": "opening_next_action",
            "group": "opening",
            "field": "target_mode",
            "kind": "numbered_choice",
            "prompt": "What would you like me to help with?",
            "manual_input_allowed": False,
            "accepted_action_types": [
                "answer_opening_question",
                "answer_pending",
                "choose_target_mode",
            ],
            "options": [{
                "id": "1",
                "label": "Start a fake-node benchmark",
                "description": (
                    "Fast, low-risk framework validation with recorded fixtures; "
                    "it does not measure real-node performance."
                ),
                "value": "fake-node",
                "action": {
                    "type": "choose_target_mode",
                    "target_mode": "fake-node",
                    "target_mode_explicit": True,
                },
                "expected_patch": {
                    "target_mode": "fake-node",
                    "workflow_mode": "rpc_benchmark",
                },
            }],
        }
        verdict = SimpleNamespace(text=json.dumps({
            "conflict": False,
            "ambiguous": False,
            "supporting_unit_ids": ["unit-purpose"],
            "independent_unit_ids": [],
            "reason": "the second sentence explains the selected framework-check option",
        }))
        provider = Mock()
        provider.complete.side_effect = [verdict, verdict]

        recovered_text, changed = _recover_declared_pending_option_semantics(
            provider,
            json.dumps(payload),
            state,
            clauses=clauses,
            user_text=source,
        )
        recovered = json.loads(recovered_text)

        self.assertTrue(changed)
        self.assertEqual([action["type"] for action in recovered["actions"]], ["choose_target_mode"])
        self.assertEqual(recovered["pending_support_unit_ids"], ["unit-purpose"])
        review_payload = json.loads(provider.complete.call_args_list[0].args[0].messages[-1].content)
        self.assertEqual(
            review_payload["selected_option"]["expected_patch"],
            {"target_mode": "fake-node", "workflow_mode": "rpc_benchmark"},
        )
        self.assertEqual(
            review_payload["selected_option"]["description"],
            (
                "Fast, low-risk framework validation with recorded fixtures; "
                "it does not measure real-node performance."
            ),
        )

    def test_pending_option_preserves_independent_context_labeled_request(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _recover_declared_pending_option_semantics
        from agent.harness.state import new_state

        source = "Use fake-node for this run. Also change the QPS profile to quick."
        clauses = segment_user_turn(source)
        payload = {
            "actions": [{
                "type": "set_qps_mode",
                "qps_mode": "quick",
                "mutation_explicit": True,
                "source_evidence": "change the QPS profile to quick",
            }],
            "semantic_units": [
                self._unit(clauses[0], "unit-choice", [], disposition="context"),
                self._unit(clauses[1], "unit-qps", [0], disposition="action"),
            ],
        }
        state = new_state("pending-context-sibling", language="en")
        state["pending_question"] = {
            "id": "target_mode_select",
            "group": "entry_mode",
            "prompt": "Choose the target mode.",
            "manual_input_allowed": False,
            "accepted_action_types": ["choose_target_mode"],
            "options": [{
                "id": "fake-node",
                "label": "fake-node",
                "value": "fake-node",
                "action": {
                    "type": "choose_target_mode",
                    "target_mode": "fake-node",
                    "target_mode_explicit": True,
                },
            }],
        }
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "decision": "select_option",
            "option_id": "fake-node",
            "answer": "",
            "evidence_quote": "Use fake-node",
            "supporting_unit_ids": ["unit-choice"],
            "independent_unit_ids": ["unit-qps"],
            "reason": "the mode selection and QPS mutation are independent",
        }))

        recovered_text, changed = _recover_declared_pending_option_semantics(
            provider, json.dumps(payload), state, clauses=clauses, user_text=source
        )
        recovered = json.loads(recovered_text)

        self.assertTrue(changed)
        self.assertEqual(
            [item["type"] for item in recovered["actions"]],
            ["set_qps_mode", "choose_target_mode"],
        )
        self.assertEqual(recovered["semantic_units"][1]["action_indexes"], [0])

    def test_declared_pending_owner_replaces_competing_effect_and_keeps_independent_sibling(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _recover_declared_pending_option_semantics
        from agent.harness.state import new_state

        source = "Use my custom method, and change the QPS profile to quick."
        clauses = segment_user_turn(source)
        payload = {
            "actions": [
                {
                    "type": "change_group",
                    "group": "endpoint_process",
                    "source_evidence": "Use my custom method",
                },
                {
                    "type": "set_qps_mode",
                    "qps_mode": "quick",
                    "mutation_explicit": True,
                    "source_evidence": "change the QPS profile to quick",
                },
            ],
            "semantic_units": [self._unit(clauses[0], "unit-1", [0, 1])],
        }
        state = new_state("pending-owner-preserve-sibling", language="en")
        state["pending_question"] = {
            "id": "custom_rpc_scope",
            "group": "rpc_workload",
            "options": [{
                "id": "single_replace",
                "label": "Replace single workload",
                "value": "single_replace",
                "action": {"type": "rpc_catalog_command", "catalog_command": "enter"},
            }],
        }
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "decision": "select_option",
            "option_id": "single_replace",
            "evidence_quote": "Use my custom method",
            "supporting_unit_ids": ["unit-1"],
            "independent_unit_ids": [],
            "reason": "selects the declared custom method option",
        }))

        recovered_text, changed = _recover_declared_pending_option_semantics(
            provider,
            json.dumps(payload),
            state,
            clauses=clauses,
        )
        recovered = json.loads(recovered_text)

        self.assertTrue(changed)
        self.assertEqual(
            [action["type"] for action in recovered["actions"]],
            ["set_qps_mode", "rpc_catalog_command"],
        )
        self.assertEqual(recovered["pending_answer_admissions"], [1])
        self.assertEqual(recovered["semantic_units"][0]["action_indexes"], [0, 1])

    def test_declared_pending_owner_removes_same_source_consultation(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _recover_declared_pending_option_semantics
        from agent.harness.plan_coverage import PlanCoverageResult
        from agent.harness.state import new_state

        source = "Show me the diagnostic evidence first; I will not change configuration yet."
        clauses = segment_user_turn(source)
        payload = {
            "actions": [
                {
                    "type": "inspect_failure",
                    "source_evidence": "Show me the diagnostic evidence first",
                },
                {
                    "type": "answer_opening_question",
                    "topic": "current_step",
                    "source_evidence": "Show me the diagnostic evidence first",
                },
            ],
            "semantic_units": [self._unit(clauses[0], "unit-1", [0, 1])],
        }
        state = new_state("pending-owner-same-source", language="en")
        state["pending_question"] = {
            "id": "failure_recovery_action",
            "group": "failure_recovery",
            "options": [{
                "id": "inspect",
                "label": "Inspect failure evidence and diagnostics",
                "value": "inspect",
                "action": {"type": "inspect_failure"},
            }],
        }
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "decision": "select_option",
            "option_id": "inspect",
            "evidence_quote": "Show me the diagnostic evidence first",
            "supporting_unit_ids": ["unit-1"],
            "independent_unit_ids": [],
            "reason": "selects the declared evidence-inspection option",
        }))

        recovered_text, changed = _recover_declared_pending_option_semantics(
            provider,
            json.dumps(payload),
            state,
            PlanCoverageResult(
                valid=False,
                errors=("competing same-source consultation",),
                unresolved_clauses=(),
                incomplete_unit_ids=("unit-1",),
            ),
            clauses=clauses,
            user_text=source,
        )
        recovered = json.loads(recovered_text)

        self.assertTrue(changed)
        self.assertEqual([action["type"] for action in recovered["actions"]], ["inspect_failure"])
        self.assertEqual(recovered["pending_answer_admissions"], [0])
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
            "decision": "select_option",
            "option_id": "reenter",
            "evidence_quote": "重新把链名输一遍",
            "supporting_unit_ids": ["unit-1"],
            "independent_unit_ids": [],
            "reason": "selects the declared re-entry option",
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
            "decision": "manual_value",
            "answer": "http://geth-dev:8545",
            "evidence_quote": "http://geth-dev:8545",
            "supporting_unit_ids": ["unit-1"],
            "independent_unit_ids": [],
            "reason": "directly supplies the requested URL",
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

    def test_declared_pending_contract_recovers_semantic_typed_scalar(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _recover_declared_pending_option_semantics
        from agent.harness.plan_coverage import PlanCoverageResult
        from agent.harness.questions import manual_question
        from agent.harness.state import new_state

        source = "Set the unified monitoring interval to five seconds."
        clauses = segment_user_turn(source)
        payload = {
            "actions": [{
                "type": "clarify_unresolved",
                "clause": source,
                "source_evidence": source,
            }],
            "semantic_units": [
                self._unit(clauses[0], "unit-1", [0], disposition="unresolved")
            ],
        }
        state = new_state("pending-semantic-scalar", language="en")
        state["pending_question"] = manual_question(
            "advanced_tuning",
            "advanced_tuning_adjust_value",
            "Enter the value for the selected advanced setting.",
            field="advanced_tuning_adjust_value",
            validation={"value_type": "positive_number"},
        )
        provider = Mock()
        provider.complete.side_effect = [
            SimpleNamespace(text=json.dumps({
                "decision": "manual_value",
                "answer": "5",
                "evidence_quote": source,
                "supporting_unit_ids": [],
                "independent_unit_ids": [],
                "reason": "the typed value is five seconds",
            })),
            SimpleNamespace(text=json.dumps({
                "decision": "manual_value",
                "answer": "5",
                "evidence_quote": "five seconds",
                "supporting_unit_ids": [],
                "independent_unit_ids": [],
                "reason": "the canonical positive number is 5",
            })),
        ]

        recovered_text, changed = _recover_declared_pending_option_semantics(
            provider,
            json.dumps(payload),
            state,
            PlanCoverageResult(
                valid=False,
                errors=("unresolved typed scalar",),
                unresolved_clauses=(source,),
                rejected_action_indexes=(0,),
                incomplete_unit_ids=("unit-1",),
            ),
            clauses=clauses,
            user_text=source,
        )
        recovered = json.loads(recovered_text)

        self.assertTrue(changed)
        self.assertEqual(recovered["actions"], [{
            "type": "answer_pending",
            "answer": "5",
            "source_evidence": source,
        }])
        self.assertEqual(recovered["pending_answer_admissions"], [0])

    def test_semantic_scalar_recovery_is_not_available_to_literal_text_contracts(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _recover_declared_pending_option_semantics
        from agent.harness.plan_coverage import PlanCoverageResult
        from agent.harness.questions import manual_question
        from agent.harness.state import new_state

        source = "Use the value described as five seconds."
        clauses = segment_user_turn(source)
        payload = {
            "actions": [],
            "semantic_units": [
                self._unit(clauses[0], "unit-1", [], disposition="unresolved")
            ],
        }
        state = new_state("pending-literal-text", language="en")
        state["pending_question"] = manual_question(
            "environment_hardware",
            "machine_type",
            "Enter the machine type.",
            field="MACHINE_TYPE",
            validation={"value_type": "bounded_text", "max_length": 80},
        )
        verdict = SimpleNamespace(text=json.dumps({
            "decision": "manual_value",
            "answer": "5",
            "evidence_quote": "five seconds",
            "supporting_unit_ids": [],
            "independent_unit_ids": [],
            "reason": "normalized text",
        }))
        provider = Mock()
        provider.complete.side_effect = [verdict, verdict]

        recovered_text, changed = _recover_declared_pending_option_semantics(
            provider,
            json.dumps(payload),
            state,
            PlanCoverageResult(
                valid=False,
                errors=("unresolved text",),
                unresolved_clauses=(source,),
                incomplete_unit_ids=("unit-1",),
            ),
            clauses=clauses,
            user_text=source,
        )

        self.assertFalse(changed)
        self.assertEqual(json.loads(recovered_text), payload)

    def test_semantic_scalar_recovery_is_contract_generic(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _recover_declared_pending_option_semantics
        from agent.harness.plan_coverage import PlanCoverageResult
        from agent.harness.questions import manual_question
        from agent.harness.state import new_state

        cases = (
            {
                "name": "environment integer",
                "group": "environment_hardware",
                "validation": "positive_integer",
                "source": "Use six worker threads.",
                "answer": "6",
                "quote_1": "Use six worker threads.",
                "quote_2": "six worker threads",
                "supporting": [],
            },
            {
                "name": "performance decimal",
                "group": "performance_mode",
                "validation": "positive_number",
                "source": "Set bandwidth ceiling to 12.5 gigabits.",
                "answer": "12.5",
                "quote_1": "Set bandwidth ceiling to 12.5 gigabits.",
                "quote_2": "12.5 gigabits",
                "supporting": [],
            },
            {
                "name": "sync observe multiline",
                "group": "sync_observe",
                "validation": "positive_number",
                "source": "For the sync observer:\nset the interval to seven seconds.",
                "answer": "7",
                "quote_1": "set the interval to seven seconds.",
                "quote_2": "seven seconds",
                "supporting": ["unit-1"],
            },
        )
        for case in cases:
            with self.subTest(case["name"]):
                clauses = segment_user_turn(case["source"])
                units = [
                    self._unit(
                        clause,
                        f"unit-{index}",
                        [],
                        disposition="unresolved",
                    )
                    for index, clause in enumerate(clauses, start=1)
                ]
                payload = {"actions": [], "semantic_units": units}
                state = new_state(f"semantic-{case['name']}", language="en")
                state["pending_question"] = manual_question(
                    case["group"],
                    "generic_typed_value",
                    "Enter the requested value.",
                    field="generic_typed_value",
                    validation={"value_type": case["validation"]},
                )
                verdicts = [
                    SimpleNamespace(text=json.dumps({
                        "decision": "manual_value",
                        "answer": case["answer"],
                        "evidence_quote": quote,
                        "supporting_unit_ids": case["supporting"],
                        "independent_unit_ids": [],
                        "reason": "one source-grounded typed scalar",
                    }))
                    for quote in (case["quote_1"], case["quote_2"])
                ]
                provider = Mock()
                provider.complete.side_effect = verdicts

                recovered_text, changed = _recover_declared_pending_option_semantics(
                    provider,
                    json.dumps(payload),
                    state,
                    PlanCoverageResult(
                        valid=False,
                        errors=("unresolved typed scalar",),
                        unresolved_clauses=tuple(clause.text for clause in clauses),
                        incomplete_unit_ids=tuple(unit["unit_id"] for unit in units),
                    ),
                    clauses=clauses,
                    user_text=case["source"],
                )

                self.assertTrue(changed)
                self.assertEqual(
                    json.loads(recovered_text)["actions"][-1]["answer"],
                    case["answer"],
                )

    def test_semantic_scalar_recovery_fails_closed(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _recover_declared_pending_option_semantics
        from agent.harness.plan_coverage import PlanCoverageResult
        from agent.harness.questions import manual_question
        from agent.harness.state import new_state

        source = "Set the interval to five or six seconds."
        clauses = segment_user_turn(source)
        payload = {
            "actions": [],
            "semantic_units": [
                self._unit(clauses[0], "unit-1", [], disposition="unresolved")
            ],
        }
        state = new_state("semantic-scalar-fail-closed", language="en")
        state["pending_question"] = manual_question(
            "sync_observe",
            "generic_typed_value",
            "Enter the requested value.",
            field="generic_typed_value",
            validation={"value_type": "positive_number"},
        )
        coverage = PlanCoverageResult(
            valid=False,
            errors=("unresolved typed scalar",),
            unresolved_clauses=(source,),
            incomplete_unit_ids=("unit-1",),
        )

        cases = (
            (
                "adjudicator disagreement",
                {
                    "decision": "manual_value",
                    "answer": "5",
                    "evidence_quote": "five",
                },
                {
                    "decision": "manual_value",
                    "answer": "6",
                    "evidence_quote": "six",
                },
            ),
            (
                "conflicting values",
                {"decision": "ambiguous", "answer": "", "evidence_quote": source},
                {"decision": "ambiguous", "answer": "", "evidence_quote": source},
            ),
            (
                "invalid zero",
                {"decision": "manual_value", "answer": "0", "evidence_quote": source},
                {"decision": "manual_value", "answer": "0", "evidence_quote": source},
            ),
            (
                "missing exact evidence",
                {"decision": "manual_value", "answer": "5", "evidence_quote": "not present"},
                {"decision": "manual_value", "answer": "5", "evidence_quote": "not present"},
            ),
        )
        for name, first, second in cases:
            with self.subTest(name):
                provider = Mock()
                provider.complete.side_effect = [
                    SimpleNamespace(text=json.dumps({
                        **verdict,
                        "option_id": "",
                        "supporting_unit_ids": [],
                        "independent_unit_ids": [],
                        "reason": name,
                    }))
                    for verdict in (first, second)
                ]

                recovered_text, changed = _recover_declared_pending_option_semantics(
                    provider,
                    json.dumps(payload),
                    state,
                    coverage,
                    clauses=clauses,
                    user_text=source,
                )

                self.assertFalse(changed)
                self.assertEqual(json.loads(recovered_text), payload)

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
            "decision": "select_option",
            "option_id": "sync",
            "evidence_quote": "watch the node catch up",
            "supporting_unit_ids": ["unit-context", "unit-choice"],
            "independent_unit_ids": [],
            "reason": "selects sync observation and rules out load generation",
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

    def test_pending_option_scope_owns_restatement_of_visible_execution_contract(self) -> None:
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _recover_declared_pending_option_semantics
        from agent.harness.plan_coverage import PlanCoverageResult
        from agent.harness.state import new_state

        prompt = (
            "Preflight passed, but no isolated real-node smoke has run. "
            "Revalidate and submit the safe low-traffic smoke now?"
        )
        variants = (
            (
                "en",
                "Proceed with the isolated safe low-traffic real-node smoke now; "
                "revalidate the custom RPC method before submission.",
            ),
            (
                "zh",
                "现在执行隔离的安全小流量 real-node smoke；"
                "提交前重新校验已经确认的自定义 RPC method。",
            ),
        )
        for language, source in variants:
            with self.subTest(language=language):
                clauses = segment_user_turn(source)
                payload = {
                    "actions": [
                        {
                            "type": "approve_preflight_smoke",
                            "source_evidence": clauses[0].text,
                        },
                        {
                            "type": "answer_pending",
                            "answer": True,
                            "selected_value": True,
                            "source_evidence": clauses[0].text,
                        },
                        {
                            "type": "analyze_evidence",
                            "evidence": clauses[1].text,
                            "question": clauses[1].text,
                            "source_evidence": clauses[1].text,
                        },
                    ],
                    "semantic_units": [
                        _unit(clauses[0], 1, [0, 1]),
                        _unit(clauses[1], 2, [2]),
                    ],
                }
                state = new_state(f"pending-execution-scope-{language}", language=language)
                state["pending_question"] = {
                    "id": "real_node_smoke_confirm",
                    "group": "job_monitoring",
                    "kind": "yes_no",
                    "prompt": prompt,
                    "options": [{
                        "id": "1",
                        "label": "Y",
                        "value": True,
                        "action": {"type": "approve_preflight_smoke"},
                        "expected_patch": {"preflight.approved": True},
                        "completion_effect": (
                            "Revalidate the confirmed custom RPC method and submit the isolated smoke."
                        ),
                    }],
                }
                verdict = SimpleNamespace(text=json.dumps({
                    "conflict": False,
                    "ambiguous": False,
                    "supporting_unit_ids": ["unit-2"],
                    "independent_unit_ids": [],
                    "reason": "restates work already declared by the selected option",
                }))
                provider = Mock()
                provider.complete.side_effect = [verdict, verdict]

                recovered_text, changed = _recover_declared_pending_option_semantics(
                    provider,
                    json.dumps(payload),
                    state,
                    PlanCoverageResult(
                        valid=False,
                        errors=("action 1 semantic fulfilment failed",),
                        unresolved_clauses=(),
                        rejected_action_indexes=(2,),
                    ),
                    clauses,
                    source,
                )
                recovered = json.loads(recovered_text)
                adjudication_payload = json.loads(
                    provider.complete.call_args_list[0].args[0].messages[-1].content
                )

                self.assertTrue(changed)
                self.assertEqual(
                    [action["type"] for action in recovered["actions"]],
                    ["approve_preflight_smoke", "answer_pending"],
                )
                self.assertEqual(recovered["semantic_units"][1]["disposition"], "context")
                self.assertEqual(recovered["pending_support_unit_ids"], ["unit-2"])
                self.assertEqual(adjudication_payload["selected_option"]["question_prompt"], prompt)
                self.assertIn(
                    "Revalidate the confirmed custom RPC method",
                    adjudication_payload["selected_option"]["completion_effect"],
                )

    def test_pending_option_scope_does_not_absorb_independent_endpoint_mutation(self) -> None:
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _recover_declared_pending_option_semantics
        from agent.harness.plan_coverage import PlanCoverageResult
        from agent.harness.state import new_state

        source = (
            "Proceed with the isolated safe low-traffic real-node smoke now; "
            "switch the endpoint to http://other-node:8545 before submission."
        )
        clauses = segment_user_turn(source)
        payload = {
            "actions": [
                {
                    "type": "approve_preflight_smoke",
                    "source_evidence": clauses[0].text,
                },
                {
                    "type": "answer_pending",
                    "answer": True,
                    "selected_value": True,
                    "source_evidence": clauses[0].text,
                },
                {
                    "type": "analyze_evidence",
                    "evidence": clauses[1].text,
                    "question": clauses[1].text,
                    "source_evidence": clauses[1].text,
                },
            ],
            "semantic_units": [
                _unit(clauses[0], 1, [0, 1]),
                _unit(clauses[1], 2, [2]),
            ],
        }
        state = new_state("pending-execution-independent-endpoint", language="en")
        state["pending_question"] = {
            "id": "real_node_smoke_confirm",
            "group": "job_monitoring",
            "kind": "yes_no",
            "prompt": "Revalidate and submit the safe low-traffic smoke now?",
            "options": [{
                "id": "1",
                "label": "Y",
                "value": True,
                "action": {"type": "approve_preflight_smoke"},
                "completion_effect": (
                    "Revalidate the already confirmed endpoint and custom RPC method, "
                    "then submit the isolated smoke."
                ),
            }],
        }
        verdict = SimpleNamespace(text=json.dumps({
            "conflict": False,
            "ambiguous": False,
            "supporting_unit_ids": [],
            "independent_unit_ids": ["unit-2"],
            "reason": "changing the endpoint is a separate configuration mutation",
        }))
        provider = Mock()
        provider.complete.side_effect = [verdict, verdict]

        recovered_text, changed = _recover_declared_pending_option_semantics(
            provider,
            json.dumps(payload),
            state,
            PlanCoverageResult(
                valid=False,
                errors=("action 1 semantic fulfilment failed",),
                unresolved_clauses=(),
                rejected_action_indexes=(2,),
            ),
            clauses,
            source,
        )
        recovered = json.loads(recovered_text)

        self.assertFalse(changed)
        self.assertEqual(recovered, payload)
        self.assertNotIn("pending_support_unit_ids", recovered)
        self.assertEqual(
            [action["type"] for action in recovered["actions"]],
            ["approve_preflight_smoke", "answer_pending", "analyze_evidence"],
        )

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
            "conflict": False,
            "ambiguous": False,
            "supporting_unit_ids": ["unit-context"],
            "independent_unit_ids": [],
            "reason": "supports the already admitted observation mode",
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
            SimpleNamespace(text=json.dumps({})),
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

    def test_declared_pending_semantic_replaces_valid_clarification_with_manual_answer(self) -> None:
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _recover_declared_pending_option_semantics
        from agent.harness.plan_coverage import PlanCoverageResult
        from agent.harness.state import new_state

        source = "Use /dev/nvme1n1 as the dedicated accounts and state disk."
        clause = segment_user_turn(source)[0]
        payload = {
            "actions": [{
                "type": "clarify_unresolved",
                "clauses": [source],
                "reason": "planner did not bind the value",
            }],
            "semantic_units": [_unit(clause, 1, [0])],
        }
        state = new_state("recover-valid-clarification", language="en")
        state["pending_question"] = {
            "id": "ACCOUNTS_DEVICE",
            "group": "accounts_disk",
            "field": "ACCOUNTS_DEVICE",
            "kind": "device",
            "prompt": "Enter ACCOUNTS_DEVICE.",
            "manual_input_allowed": True,
            "validation": {"value_type": "scalar_token"},
            "options": [],
        }
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "decision": "manual_value",
            "answer": "/dev/nvme1n1",
            "evidence_quote": "/dev/nvme1n1",
            "supporting_unit_ids": ["unit-1"],
            "independent_unit_ids": [],
            "reason": "directly supplies the requested device",
        }))

        recovered_text, changed = _recover_declared_pending_option_semantics(
            provider,
            json.dumps(payload),
            state,
            PlanCoverageResult(True, (), ()),
            (clause,),
        )
        recovered = json.loads(recovered_text)

        self.assertTrue(changed)
        self.assertEqual(recovered["actions"], [{
            "type": "answer_pending",
            "answer": "/dev/nvme1n1",
            "source_evidence": "/dev/nvme1n1",
        }])
        self.assertEqual(recovered["semantic_units"][0]["action_indexes"], [0])

    def test_declared_pending_semantic_separates_value_owner_from_multiline_context(self) -> None:
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _recover_declared_pending_option_semantics
        from agent.harness.plan_coverage import PlanCoverageResult
        from agent.harness.state import new_state

        source = (
            "Use the dedicated accounts volume.\n"
            "The device is /dev/nvme1n1.\n"
            "Continue with that value."
        )
        clauses = segment_user_turn(source)
        payload = {
            "actions": [{
                "type": "clarify_unresolved",
                "clauses": [clause.text for clause in clauses],
                "reason": "planner did not bind the value",
            }],
            "semantic_units": [
                _unit(clause, index, [0])
                for index, clause in enumerate(clauses, start=1)
            ],
        }
        state = new_state("recover-multiline-manual-owner", language="en")
        state["pending_question"] = {
            "id": "ACCOUNTS_DEVICE",
            "group": "accounts_disk",
            "field": "ACCOUNTS_DEVICE",
            "kind": "device",
            "prompt": "Enter ACCOUNTS_DEVICE.",
            "manual_input_allowed": True,
            "manual_action": {"type": "answer_pending", "value_argument": "answer"},
            "validation": {"value_type": "scalar_token"},
            "options": [],
        }
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "decision": "manual_value",
            "answer": "/dev/nvme1n1",
            "evidence_quote": "The device is /dev/nvme1n1.",
            "supporting_unit_ids": ["unit-1", "unit-3"],
            "independent_unit_ids": [],
            "reason": "units one and three only support the value in unit two",
        }))

        recovered_text, changed = _recover_declared_pending_option_semantics(
            provider,
            json.dumps(payload),
            state,
            PlanCoverageResult(True, (), ()),
            clauses,
            source,
        )
        recovered = json.loads(recovered_text)

        self.assertTrue(changed)
        self.assertEqual(recovered["actions"], [{
            "type": "answer_pending",
            "answer": "/dev/nvme1n1",
            "source_evidence": "The device is /dev/nvme1n1.",
        }])
        self.assertEqual(
            [unit["disposition"] for unit in recovered["semantic_units"]],
            ["context", "action", "context"],
        )

    def test_declared_pending_semantic_rejects_value_owner_as_independent_demand(self) -> None:
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _recover_declared_pending_option_semantics
        from agent.harness.state import new_state

        source = "Use the dedicated accounts volume.\nThe device is /dev/nvme1n1."
        clauses = segment_user_turn(source)
        payload = {
            "actions": [],
            "semantic_units": [
                _unit(clause, index, [])
                for index, clause in enumerate(clauses, start=1)
            ],
        }
        state = new_state("reject-independent-manual-owner", language="en")
        state["pending_question"] = {
            "id": "ACCOUNTS_DEVICE",
            "group": "accounts_disk",
            "field": "ACCOUNTS_DEVICE",
            "kind": "device",
            "prompt": "Enter ACCOUNTS_DEVICE.",
            "manual_input_allowed": True,
            "manual_action": {"type": "answer_pending", "value_argument": "answer"},
            "validation": {"value_type": "scalar_token"},
            "options": [],
        }
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "decision": "manual_value",
            "answer": "/dev/nvme1n1",
            "evidence_quote": "The device is /dev/nvme1n1.",
            "supporting_unit_ids": ["unit-1"],
            "independent_unit_ids": ["unit-2"],
            "reason": "invalidly classifies the owner as independent",
        }))

        recovered_text, changed = _recover_declared_pending_option_semantics(
            provider,
            json.dumps(payload),
            state,
            clauses=clauses,
            user_text=source,
        )

        self.assertFalse(changed)
        self.assertEqual(json.loads(recovered_text), payload)

    def test_declared_pending_semantic_rejects_duplicate_evidence_owners(self) -> None:
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _recover_declared_pending_option_semantics
        from agent.harness.state import new_state

        source = "Primary device: /dev/nvme1n1.\nFallback device: /dev/nvme1n1."
        clauses = segment_user_turn(source)
        payload = {
            "actions": [],
            "semantic_units": [
                _unit(clause, index, [])
                for index, clause in enumerate(clauses, start=1)
            ],
        }
        state = new_state("reject-duplicate-manual-owners", language="en")
        state["pending_question"] = {
            "id": "ACCOUNTS_DEVICE",
            "group": "accounts_disk",
            "field": "ACCOUNTS_DEVICE",
            "kind": "device",
            "prompt": "Enter ACCOUNTS_DEVICE.",
            "manual_input_allowed": True,
            "manual_action": {"type": "answer_pending", "value_argument": "answer"},
            "validation": {"value_type": "scalar_token"},
            "options": [],
        }
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "decision": "manual_value",
            "answer": "/dev/nvme1n1",
            "evidence_quote": "/dev/nvme1n1",
            "supporting_unit_ids": ["unit-1", "unit-2"],
            "independent_unit_ids": [],
            "reason": "the quote appears in two possible owner units",
        }))

        recovered_text, changed = _recover_declared_pending_option_semantics(
            provider,
            json.dumps(payload),
            state,
            clauses=clauses,
            user_text=source,
        )

        self.assertFalse(changed)
        self.assertEqual(json.loads(recovered_text), payload)

    def test_declared_pending_semantic_attaches_scope_to_admitted_manual_anchor(self) -> None:
        from types import SimpleNamespace
        from unittest.mock import Mock

        from agent.harness.intent import _recover_declared_pending_option_semantics
        from agent.harness.state import new_state

        source = (
            "The credential is run-secret-4821.\n"
            "Use it only for this run without changing the saved template."
        )
        clauses = segment_user_turn(source)
        payload = {
            "actions": [{
                "type": "answer_pending",
                "answer": "run-secret-4821",
                "source_evidence": "run-secret-4821",
            }],
            "pending_answer_admissions": [0],
            "semantic_units": [
                _unit(clauses[0], 1, [0]),
                {
                    **_unit(clauses[1], 2, []),
                    "disposition": "unresolved",
                },
            ],
        }
        state = new_state("manual-anchor-scope", language="en")
        state["pending_question"] = {
            "id": "RUNTIME_CREDENTIAL",
            "group": "chain_auxiliary_endpoints",
            "field": "RUNTIME_CREDENTIAL",
            "kind": "manual_value",
            "prompt": "Enter the runtime credential.",
            "manual_input_allowed": True,
            "validation": {"value_type": "scalar_token", "max_length": 180},
            "options": [],
        }
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(text=json.dumps({
            "decision": "manual_value",
            "answer": "run-secret-4821",
            "evidence_quote": "run-secret-4821",
            "supporting_unit_ids": ["unit-2"],
            "independent_unit_ids": [],
            "reason": "the second unit only scopes the admitted value",
        }))

        recovered_text, changed = _recover_declared_pending_option_semantics(
            provider,
            json.dumps(payload),
            state,
            clauses=clauses,
            user_text=source,
        )
        recovered = json.loads(recovered_text)

        self.assertTrue(changed)
        self.assertEqual(recovered["actions"], payload["actions"])
        self.assertEqual(recovered["pending_answer_admissions"], [0])
        self.assertEqual(recovered["semantic_units"][0]["action_indexes"], [0])
        self.assertEqual(recovered["semantic_units"][1]["disposition"], "context")
        self.assertEqual(recovered["semantic_units"][1]["action_indexes"], [])

    def test_structured_pending_config_assignment_is_owned_by_review_proposal(self) -> None:
        from agent.harness.intent import _reconcile_structured_candidate_ownership
        from agent.harness.state import new_state

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
        state = new_state("structured-pending-review", language="en")
        state["pending_question"] = {
            "id": "ACCOUNTS_DEVICE",
            "group": "accounts_disk",
            "field": "ACCOUNTS_DEVICE",
            "kind": "device",
            "manual_input_allowed": True,
            "validation": {"value_type": "scalar_token"},
            "options": [],
        }

        reconciled = json.loads(_reconcile_structured_candidate_ownership(
            json.dumps(payload),
            (clause,),
            state,
        ))

        self.assertEqual(reconciled["actions"][0]["type"], "propose_config_values")
        self.assertEqual(
            reconciled["actions"][0]["config_values"],
            {"ACCOUNTS_DEVICE": "/dev/nvme1n1"},
        )
        self.assertEqual(reconciled["actions"][0]["source_format"], "json")
        self.assertEqual(reconciled["semantic_units"][0]["action_indexes"], [0])

    def test_structured_pending_assignment_and_trailing_proposal_scope_rejoin_one_review(self) -> None:
        from agent.harness.intent import (
            _reconcile_structured_candidate_ownership,
            _validate_action_document,
        )
        from agent.harness.state import new_state

        source = (
            "The network profile says:\n"
            "```env\nNETWORK_MAX_BANDWIDTH_GBPS=25\n```\n"
            "Use that as the proposed bandwidth value."
        )
        clauses = segment_user_turn(source)
        payload = {
            "actions": [{
                "type": "answer_pending",
                "answer": "25",
                "selected_value": "25",
                "source_evidence": "25",
            }],
            "semantic_units": [
                _unit(clauses[0], 1, [], disposition="unresolved"),
                _unit(clauses[1], 2, [0]),
            ],
            "pending_answer_admissions": [0],
        }
        state = new_state("structured-pending-cross-clause", language="en")
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

        self.assertEqual(reconciled["actions"][0]["type"], "propose_config_values")
        self.assertEqual(
            reconciled["actions"][0]["config_values"],
            {"NETWORK_MAX_BANDWIDTH_GBPS": "25"},
        )
        self.assertEqual(reconciled["actions"][0]["source_format"], "env")
        self.assertEqual(reconciled["pending_answer_admissions"], [])
        self.assertEqual(reconciled["semantic_units"][0]["action_indexes"], [0])
        self.assertEqual(reconciled["semantic_units"][1]["action_indexes"], [0])
        self.assertTrue(
            _validate_action_document(json.dumps(reconciled), clauses, state).valid
        )

    def test_cross_clause_structured_pending_assignment_uses_pending_intent_before_admission(self) -> None:
        from agent.harness.intent import _reconcile_structured_candidate_ownership
        from agent.harness.state import new_state

        source = "```env\nCLOUD_REGION=asia-east1\n```\nUse that proposed value."
        clauses = segment_user_turn(source)
        payload = {
            "actions": [{
                "type": "answer_pending",
                "answer": "asia-east1",
                "selected_value": "asia-east1",
                "source_evidence": "asia-east1",
            }],
            "semantic_units": [
                _unit(clauses[0], 1, [], disposition="unresolved"),
                _unit(clauses[1], 2, [0]),
            ],
        }
        state = new_state("structured-pending-no-admission", language="en")
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

        self.assertEqual(reconciled["actions"][0]["type"], "propose_config_values")
        self.assertEqual(
            reconciled["actions"][0]["config_values"],
            {"CLOUD_REGION": "asia-east1"},
        )
        self.assertEqual(reconciled["actions"][0]["source_format"], "env")

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

    def test_manual_pending_owner_merges_equivalent_domain_action(self) -> None:
        from agent.harness.intent import _materialize_pending_manual_owner_actions
        from agent.harness.state import new_state

        source = (
            "Use this endpoint only for validating the custom method:\n"
            "http://fake-node:19000\n"
            "Also show the current workflow status."
        )
        clauses = segment_user_turn(source)
        state = new_state("manual-owner-merge", language="en")
        state["pending_question"] = {
            "id": "opaque-endpoint-question",
            "group": "endpoint_process",
            "kind": "url",
            "manual_input_allowed": True,
            "manual_action": {
                "type": "rpc_catalog_command",
                "catalog_command": "set_endpoint",
                "value_argument": "rpc_endpoint",
            },
        }
        payload = {
            "actions": [
                {
                    "type": "rpc_catalog_command",
                    "catalog_command": "set_endpoint",
                    "rpc_endpoint": "http://fake-node:19000",
                    "source_evidence": "http://fake-node:19000",
                },
                {
                    "type": "answer_pending",
                    "selected_value": "http://fake-node:19000",
                    "source_evidence": "http://fake-node:19000",
                },
                {"type": "show_current_state"},
            ],
            "semantic_units": [
                _unit(clauses[0], 1, [0, 1]),
                _unit(clauses[0], 2, [2]),
            ],
            "pending_answer_admissions": [1],
        }

        materialized, changed = _materialize_pending_manual_owner_actions(
            json.dumps(payload), state, source
        )
        result = json.loads(materialized)

        self.assertTrue(changed)
        self.assertEqual(
            [action["type"] for action in result["actions"]],
            ["rpc_catalog_command", "show_current_state"],
        )
        self.assertEqual(result["pending_answer_admissions"], [0])
        self.assertEqual(result["semantic_units"][0]["action_indexes"], [0])
        self.assertEqual(result["semantic_units"][1]["action_indexes"], [1])

    def test_structured_pending_evidence_compiles_to_catalog_owner(self) -> None:
        from agent.harness.intent import (
            _materialize_pending_manual_owner_actions,
            _validate_action_document,
        )
        from agent.harness.state import new_state

        source = (
            'request:\n  jsonrpc: "2.0"\n  id: 1\n'
            '  method: eth_blockNumber\n  params: []\n'
            'response:\n  jsonrpc: "2.0"\n  id: 1\n  result: "0x1"'
        )
        clauses = segment_user_turn(source)
        state = new_state("structured-evidence-owner", language="en")
        state["pending_question"] = {
            "id": "opaque-evidence-question",
            "group": "endpoint_process",
            "kind": "evidence",
            "manual_input_allowed": True,
            "manual_action": {
                "type": "rpc_catalog_command",
                "catalog_command": "append_evidence",
                "value_argument": "rpc_schema_evidence",
                "use_complete_turn": True,
            },
        }
        payload = {
            "actions": [{
                "type": "answer_pending",
                "answer": "[]",
                "selected_value": "[]",
                "source_evidence": "params: []",
            }],
            "semantic_units": [_unit(clauses[0], 1, [0])],
            "pending_answer_admissions": [0],
        }

        materialized, changed = _materialize_pending_manual_owner_actions(
            json.dumps(payload), state, source
        )
        result = json.loads(materialized)

        self.assertTrue(changed)
        self.assertEqual(result["actions"][0]["type"], "rpc_catalog_command")
        self.assertEqual(result["actions"][0]["catalog_command"], "append_evidence")
        self.assertEqual(result["actions"][0]["rpc_schema_evidence"], source)
        self.assertTrue(_validate_action_document(materialized, clauses, state).valid)


if __name__ == "__main__":
    unittest.main()

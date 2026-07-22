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


class CanonicalPendingChoiceTests(unittest.TestCase):
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

        self.assertEqual(provider.complete.call_count, 4)
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


if __name__ == "__main__":
    unittest.main()

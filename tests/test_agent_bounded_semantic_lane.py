from __future__ import annotations

import json
import unittest
from types import SimpleNamespace


class _Provider:
    def __init__(self, *documents: dict[str, object]) -> None:
        self.documents = list(documents)
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        return SimpleNamespace(
            text=json.dumps(self.documents.pop(0)),
            provider="test",
            model="test",
            raw={},
        )


def _pending_question(*, manual: bool = False) -> dict[str, object]:
    question: dict[str, object] = {
        "id": "mode-choice",
        "group": "target_mode",
        "owner": "chain_rpc",
        "kind": "choice",
        "options": [
            {"id": "simulated", "value": "fake-node"},
            {"id": "live", "value": "real-node"},
        ],
        "manual_input_allowed": manual,
        "validation": (
            {"value_type": "positive_integer"} if manual else {}
        ),
    }
    return question


def _mapping(
    candidate_id: str,
    quote: str,
    *,
    clauses: tuple[str, ...] = ("clause-1",),
    sibling: str = "none",
) -> dict[str, object]:
    return {
        "decision": "selected",
        "candidate_id": candidate_id,
        "source_quote": quote,
        "clause_verdicts": [
            {"clause_id": clause, "verdict": "direct", "reason": "bound"}
            for clause in clauses
        ],
        "sibling_verdict": sibling,
        "reason": "unique candidate",
    }


def _admission(
    mapping: dict[str, object],
    candidate_id: str,
    *,
    accept: bool = True,
) -> dict[str, object]:
    from agent.harness.bounded_semantic_lane import _content_hash

    return {
        "mapping_hash": _content_hash(mapping),
        "candidate_id": candidate_id,
        "verdict": "accept" if accept else "reject",
        "checks": {
            "candidate_immutable": accept,
            "quote_is_exact": accept,
            "all_clauses_covered": accept,
            "no_sibling_demand": accept,
            "no_ambiguity": accept,
        },
        "reason": "valid" if accept else "rejected",
    }


class BoundedSemanticLaneTest(unittest.TestCase):
    def test_catalog_uses_pending_options_manual_values_and_exact_registry_occurrences(self):
        from agent.harness.bounded_semantic_lane import build_candidate_catalog

        state = {"pending_question": _pending_question(manual=True)}
        catalog = build_candidate_catalog(state, "100 standard")

        self.assertFalse(any(
            row["source_kind"] == "pending_option"
            and row["canonical_value"] == "fake-node"
            for row in catalog
        ))
        self.assertTrue(any(
            row["source_kind"] == "pending_manual"
            and row["canonical_value"] == "100 standard"
            for row in catalog
        ) is False)
        self.assertTrue(any(
            row["source_kind"] == "registered_value"
            and row["canonical_value"] == "standard"
            and row["action_type"] == "set_qps_mode"
            and row["owner"] == "performance"
            and row["group"] == "qps_profile"
            for row in catalog
        ))
        self.assertFalse(any(
            row["action_type"] == "queue_workflow_goal"
            for row in catalog
        ))
        self.assertTrue(all(
            len(row["candidate_id"]) > 20
            and len(row["source_hash"]) == 64
            and len(row["registry_hash"]) == 64
            for row in catalog
        ))

        anchored_catalog = build_candidate_catalog(
            state,
            "Use the simulated option.",
        )
        self.assertTrue(any(
            row["source_kind"] == "pending_option"
            and row["canonical_value"] == "fake-node"
            and row["matched_value"] == "simulated"
            for row in anchored_catalog
        ))

        manual_catalog = build_candidate_catalog(
            {"pending_question": _pending_question(manual=True)},
            "100",
        )
        self.assertTrue(any(
            row["source_kind"] == "pending_manual"
            and row["canonical_value"] == "100"
            for row in manual_catalog
        ))

    def test_pending_option_materializes_canonical_answer_pending(self):
        from agent.harness.bounded_semantic_lane import (
            build_candidate_catalog,
            compile_bounded_semantic_value,
        )

        state = {"pending_question": _pending_question()}
        text = "Use the simulated option."
        candidate = next(
            row
            for row in build_candidate_catalog(state, text)
            if row["canonical_value"] == "fake-node"
        )
        mapper = _mapping(candidate["candidate_id"], "simulated option")
        provider = _Provider(mapper, _admission(mapper, candidate["candidate_id"]))

        document = compile_bounded_semantic_value(
            state,
            text,
            provider=provider,
        )

        self.assertIsNotNone(document)
        self.assertEqual(document["status"], "review_plan")
        self.assertEqual(document["planner_kind"], "bounded_semantic_value")
        action = document["owner_documents"]["coordinator"]["actions"][0]
        self.assertEqual(action, {
            "type": "answer_pending",
            "selected_value": "fake-node",
            "source_evidence": "simulated option",
        })
        self.assertEqual(len(provider.requests), 2)
        self.assertEqual(document["admission_calls"], 1)
        self.assertTrue(all(
            request.reasoning_mode == "disabled"
            for request in provider.requests
        ))
        projection_state = {
            **state,
            "turn_context": {"text": text},
        }
        from agent.harness.bounded_semantic_lane import (
            review_bounded_semantic_plan,
        )

        queue = review_bounded_semantic_plan(projection_state, document)
        self.assertEqual(queue["actions"], [action])
        self.assertEqual(queue["planner_metrics"]["model_calls"], 2)

    def test_manual_typed_value_materializes_answer_pending(self):
        from agent.harness.bounded_semantic_lane import (
            build_candidate_catalog,
            compile_bounded_semantic_value,
        )

        state = {"pending_question": _pending_question(manual=True)}
        candidate = next(
            row
            for row in build_candidate_catalog(state, "100")
            if row["source_kind"] == "pending_manual"
        )
        mapper = _mapping(candidate["candidate_id"], "100")
        provider = _Provider(mapper, _admission(mapper, candidate["candidate_id"]))

        document = compile_bounded_semantic_value(
            state,
            "100",
            provider=provider,
        )

        action = document["owner_documents"]["coordinator"]["actions"][0]
        self.assertEqual(action["type"], "answer_pending")
        self.assertEqual(action["answer"], "100")
        self.assertNotIn("selected_value", action)

    def test_registered_value_materializes_registry_action_and_required_explicitness(self):
        from agent.harness.bounded_semantic_lane import (
            build_candidate_catalog,
            compile_bounded_semantic_value,
        )

        state = {"pending_question": _pending_question()}
        candidate = next(
            row
            for row in build_candidate_catalog(state, "standard")
            if row["action_type"] == "set_qps_mode"
        )
        mapper = _mapping(candidate["candidate_id"], "standard")
        provider = _Provider(mapper, _admission(mapper, candidate["candidate_id"]))

        document = compile_bounded_semantic_value(
            state,
            "standard",
            provider=provider,
        )

        action = document["owner_documents"]["performance"]["actions"][0]
        self.assertEqual(action, {
            "type": "set_qps_mode",
            "qps_mode": "standard",
            "mutation_explicit": True,
            "source_evidence": "standard",
        })
        self.assertEqual(
            document["source_partition"][0]["owner_routes"],
            [{"owner": "performance", "group": "qps_profile"}],
        )
        from agent.harness.hierarchical_planner import _merge_owner_documents
        from agent.harness.plan_coverage import (
            segment_user_turn,
            validate_plan_coverage,
        )

        merged = _merge_owner_documents(
            document["source_partition"],
            document["routed_partition"],
            document["owner_documents"],
        )
        coverage = validate_plan_coverage(
            merged,
            segment_user_turn("standard"),
        )
        self.assertTrue(coverage.valid, coverage.errors)

    def test_bounded_value_projects_only_with_current_immutable_receipt(self):
        from copy import deepcopy

        from agent.harness.bounded_semantic_lane import (
            build_candidate_catalog,
            compile_bounded_semantic_value,
            review_bounded_semantic_plan,
        )

        text = "quick"
        state = {
            "turn_context": {"text": text},
            "pending_question": _pending_question(),
        }
        candidate = next(
            row
            for row in build_candidate_catalog(state, text)
            if row["action_type"] == "set_qps_mode"
        )
        mapper = _mapping(candidate["candidate_id"], text)
        provider = _Provider(mapper, _admission(mapper, candidate["candidate_id"]))
        document = compile_bounded_semantic_value(
            state,
            text,
            provider=provider,
        )

        queue = review_bounded_semantic_plan(state, document)

        self.assertEqual(queue["actions"][0]["type"], "set_qps_mode")
        self.assertEqual(queue["actions"][0]["qps_mode"], "quick")
        self.assertEqual(queue["planner_metrics"]["model_calls"], 2)
        self.assertEqual(len(provider.requests), 2)

        for mutation in ("source", "question", "action", "receipt"):
            with self.subTest(mutation=mutation):
                changed_state = deepcopy(state)
                changed_document = deepcopy(document)
                if mutation == "source":
                    changed_state["turn_context"]["text"] = "standard"
                elif mutation == "question":
                    changed_state["pending_question"]["id"] = "other"
                elif mutation == "action":
                    action = changed_document["owner_documents"]["performance"]["actions"][0]
                    action["qps_mode"] = "standard"
                else:
                    changed_document["bounded_admission_receipt"]["receipt_hash"] = "0" * 64
                with self.assertRaises(ValueError):
                    review_bounded_semantic_plan(changed_state, changed_document)

    def test_ambiguous_pending_prose_cannot_manufacture_an_option_binding(self):
        from agent.harness.bounded_semantic_lane import (
            compile_bounded_semantic_value,
        )

        provider = _Provider()

        self.assertIsNone(compile_bounded_semantic_value(
            {"pending_question": _pending_question()},
            "Please help me decide.",
            provider=provider,
        ))
        self.assertEqual(provider.requests, [])

    def test_multiline_support_clauses_project_one_action_without_cardinality_error(
        self,
    ):
        from agent.harness.bounded_semantic_lane import (
            build_candidate_catalog,
            compile_bounded_semantic_value,
            review_bounded_semantic_plan,
        )

        text = "quick\nplease proceed"
        state = {
            "turn_context": {"text": text},
            "pending_question": _pending_question(),
        }
        candidate = next(
            row
            for row in build_candidate_catalog(state, text)
            if row["action_type"] == "set_qps_mode"
        )
        mapper = _mapping(
            candidate["candidate_id"],
            "quick",
            clauses=("clause-1", "clause-2"),
        )
        mapper["clause_verdicts"][1]["verdict"] = "support"
        provider = _Provider(
            mapper,
            _admission(mapper, candidate["candidate_id"]),
        )

        document = compile_bounded_semantic_value(
            state,
            text,
            provider=provider,
        )
        queue = review_bounded_semantic_plan(state, document)

        self.assertEqual(len(queue["actions"]), 1)
        self.assertEqual(len(queue["semantic_units"]), 2)
        self.assertTrue(all(
            unit["action_indexes"] == [0]
            for unit in queue["semantic_units"]
        ))

    def test_rejected_independent_admission_escalates_without_projection(self):
        from agent.harness.bounded_semantic_lane import (
            build_candidate_catalog,
            compile_bounded_semantic_value,
        )

        state = {"pending_question": _pending_question()}
        text = "standard"
        candidate = next(
            row
            for row in build_candidate_catalog(state, text)
            if row["action_type"] == "set_qps_mode"
        )
        mapper = _mapping(candidate["candidate_id"], text)
        provider = _Provider(
            mapper,
            _admission(mapper, candidate["candidate_id"], accept=False),
        )

        self.assertIsNone(compile_bounded_semantic_value(
            state,
            text,
            provider=provider,
        ))
        self.assertEqual(len(provider.requests), 2)

    def test_mapper_contract_failures_escalate_without_admission(self):
        from agent.harness.bounded_semantic_lane import (
            build_candidate_catalog,
            compile_bounded_semantic_value,
        )

        state = {"pending_question": _pending_question()}
        text = "standard profile"
        candidate = next(
            row
            for row in build_candidate_catalog(state, text)
            if row["action_type"] == "set_qps_mode"
        )
        invalid_documents = (
            _mapping("invented", "standard"),
            _mapping(candidate["candidate_id"], "not a substring"),
            _mapping(candidate["candidate_id"], "profile"),
            _mapping(candidate["candidate_id"], "standard", clauses=()),
            _mapping(candidate["candidate_id"], "standard", sibling="present"),
        )
        for invalid in invalid_documents:
            with self.subTest(invalid=invalid):
                provider = _Provider(invalid)
                self.assertIsNone(compile_bounded_semantic_value(
                    state,
                    text,
                    provider=provider,
                ))
                self.assertEqual(len(provider.requests), 1)

    def test_multiple_registered_values_escalate_before_admission(self):
        from agent.harness.bounded_semantic_lane import (
            build_candidate_catalog,
            compile_bounded_semantic_value,
        )

        state = {}
        text = "quick and standard"
        candidate = next(
            row
            for row in build_candidate_catalog(state, text)
            if row["canonical_value"] == "quick"
        )
        provider = _Provider(_mapping(candidate["candidate_id"], "quick"))

        self.assertIsNone(compile_bounded_semantic_value(
            state,
            text,
            provider=provider,
        ))
        self.assertEqual(len(provider.requests), 1)

    def test_no_catalog_escalates_without_model_call(self):
        from agent.harness.bounded_semantic_lane import (
            compile_bounded_semantic_value,
        )

        provider = _Provider()
        self.assertIsNone(compile_bounded_semantic_value(
            {},
            "value with no registered semantic identity",
            provider=provider,
        ))
        self.assertEqual(provider.requests, [])


if __name__ == "__main__":
    unittest.main()

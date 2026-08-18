"""Architecture and product-contract tests for the rebuilt Agent Harness."""

from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys
import threading
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable
from unittest.mock import Mock, patch

from agent.harness.questions import QUESTION_CONTRACT_VERSION


REPO_ROOT = Path(__file__).resolve().parents[1]
DOMAIN_ROOT = REPO_ROOT / "agent" / "harness" / "domains"


def _imported_modules(path: Path) -> set[str]:
    """Return normalized modules imported by one source file."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    package = ".".join(path.relative_to(REPO_ROOT).with_suffix("").parts[:-1])
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
            continue
        if not isinstance(node, ast.ImportFrom):
            continue
        module = node.module or ""
        if node.level:
            module = importlib.util.resolve_name("." * node.level + module, package)
        if module:
            imported.add(module)
        imported.update(f"{module}.{alias.name}" if module else alias.name for alias in node.names)
    return imported


def _state(language: str = "en", **updates: Any) -> dict[str, Any]:
    from agent.harness.state import new_state

    state = new_state(f"architecture-{language}", language=language)
    state.update(deepcopy(updates))
    return state


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


def _whole_plan_admission_payload(
    request: Any,
    *,
    candidate_selector: Callable[
        [dict[str, Any], dict[str, dict[str, Any]]],
        str,
    ] | None = None,
) -> dict[str, Any]:
    """Build one valid immutable verdict from the actual reviewer request."""

    review = json.loads(request.messages[1].content)
    units = {
        str(row["unit_id"]): row
        for row in review["semantic_units"]
    }
    action_verdicts = []
    for action in review["actions"]:
        selected_identity = (
            candidate_selector(action, units)
            if candidate_selector is not None
            else ""
        )
        evidence = [
            {
                "unit_id": unit_id,
                "quote": str(units[unit_id]["source_text"]),
                "relation": "direct",
                "support_relation": "",
            }
            for unit_id in action["unit_ids"]
        ]
        action_verdicts.append({
            "action_id": action["action_id"],
            "verdict": "admit",
            "unit_ids": list(action["unit_ids"]),
            "evidence": evidence,
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
                        == selected_identity
                    ),
                    "",
                )
            ),
            "turn_candidate_verdicts": _turn_candidate_verdicts(
                action,
                units,
                selected_identity=selected_identity,
            ),
            "reason": "the immutable registered action preserves its source units",
        })
    unit_verdicts = []
    for unit in review["semantic_units"]:
        disposition = str(unit["disposition"])
        unit_verdicts.append({
            "unit_id": unit["unit_id"],
            "verdict": "context" if disposition == "context" else "complete",
            "owner_action_ids": list(unit["owner_action_ids"]),
            "evidence_quote": str(unit["source_text"]),
            "omitted_action_type": "",
            "reason": "the immutable unit has its declared owner",
        })
    return {
        "action_verdicts": action_verdicts,
        "unit_verdicts": unit_verdicts,
        "reason": "the complete immutable plan is admitted",
    }


def _whole_plan_admission_response(
    request: Any,
    *,
    candidate_selector: Callable[
        [dict[str, Any], dict[str, dict[str, Any]]],
        str,
    ] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(text=json.dumps(
        _whole_plan_admission_payload(
            request,
            candidate_selector=candidate_selector,
        ),
        ensure_ascii=False,
        sort_keys=True,
    ))


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


def _bounded_intent_provider(
    compiler_documents: list[str | dict[str, Any]],
    admission_responses: list[Callable[[Any], SimpleNamespace]] | None = None,
    *,
    candidate_selector: Callable[
        [dict[str, Any], dict[str, dict[str, Any]]],
        str,
    ] | None = None,
) -> Mock:
    """Return a provider whose compiler and admission calls share one channel."""

    provider = Mock()
    documents = iter(compiler_documents)
    admissions = iter(admission_responses or [])

    def complete(request: Any) -> SimpleNamespace:
        request_payload = json.loads(request.messages[1].content)
        if "plan_hash" in request_payload and "immutable_document" in request_payload:
            builder = next(admissions, None)
            if builder is not None:
                return builder(request)
            return _whole_plan_admission_response(
                request,
                candidate_selector=candidate_selector,
            )
        document = next(documents)
        text = document if isinstance(document, str) else json.dumps(
            document,
            ensure_ascii=False,
            sort_keys=True,
        )
        return SimpleNamespace(text=text)

    provider.complete.side_effect = complete
    return provider


def _single_action_document(
    action: dict[str, Any],
    source_text: str,
    *,
    disposition: str = "action",
) -> dict[str, Any]:
    return {
        "actions": [action],
        "semantic_units": [{
            "unit_id": "unit-1",
            "clause_id": "clause-1",
            "source_text": source_text,
            "disposition": disposition,
            "action_indexes": [0] if disposition == "action" else [],
            "reason": "test semantic unit",
        }],
        "reason": "test plan",
    }


def _immutable_admission_fixture() -> tuple[Any, dict[str, Any]]:
    from agent.harness.semantic_compiler import freeze_semantic_plan

    actions = [
        {"type": "greeting", "source_evidence": "Hello"},
        {
            "type": "answer_opening_question",
            "topic": "capabilities",
            "source_evidence": "What can you do?",
        },
    ]
    units = [
        {
            "unit_id": "unit-1",
            "clause_id": "clause-1",
            "source_text": "Hello",
            "disposition": "action",
            "action_indexes": [0],
        },
        {
            "unit_id": "unit-2",
            "clause_id": "clause-2",
            "source_text": "What can you do?",
            "disposition": "action",
            "action_indexes": [1],
        },
    ]
    document = {"actions": actions, "semantic_units": units}
    action_records = [
        {
            "action_id": f"action-{index + 1}",
            "action_index": index,
            "action": action,
            "unit_ids": [f"unit-{index + 1}"],
            "allowed_support_relations": [],
        }
        for index, action in enumerate(actions)
    ]
    unit_records = [
        {
            "unit_id": unit["unit_id"],
            "unit_index": index,
            "unit": unit,
            "source_text": unit["source_text"],
            "disposition": "action",
            "owner_action_ids": [f"action-{index + 1}"],
        }
        for index, unit in enumerate(units)
    ]
    plan = freeze_semantic_plan(
        document,
        action_records=action_records,
        unit_records=unit_records,
        review_context={"pending_question": {}},
    )
    request = SimpleNamespace(messages=[None, SimpleNamespace(content=plan.request_json)])
    return plan, _whole_plan_admission_payload(request)


def _grounded_mutation_admission_fixture(
    *,
    source: str = "Use fake-node.",
) -> tuple[Any, dict[str, Any]]:
    from agent.harness.semantic_compiler import freeze_semantic_plan

    action = {
        "type": "choose_target_mode",
        "target_mode": "fake-node",
        "target_mode_explicit": True,
        "source_evidence": "fake-node",
    }
    unit = {
        "unit_id": "unit-1",
        "clause_id": "clause-1",
        "source_text": source,
        "disposition": "action",
        "action_indexes": [0],
    }
    plan = freeze_semantic_plan(
        {"actions": [action], "semantic_units": [unit]},
        action_records=[{
            "action_id": "action-1",
            "action_index": 0,
            "action": action,
            "registry_effect": "configuration_mutation",
            "required_value_grounding_arguments": ["target_mode"],
            "closed_enum_grounding_values": {
                "target_mode": [
                    "fake-node",
                    "real-node",
                    "sync-observe",
                ],
            },
            "operation_arguments": {"target_mode": "fake-node"},
            "unit_ids": ["unit-1"],
            "allowed_support_relations": [],
        }],
        unit_records=[{
            "unit_id": "unit-1",
            "unit_index": 0,
            "unit": unit,
            "source_text": source,
            "evidence_sources": [source],
            "disposition": "action",
            "owner_action_ids": ["action-1"],
        }],
        review_context={"pending_question": {}},
    )
    request = SimpleNamespace(
        messages=[None, SimpleNamespace(content=plan.request_json)]
    )
    return plan, _whole_plan_admission_payload(request)


def _incomplete_intake_admission_fixture(
    *,
    forged: bool = False,
) -> tuple[Any, dict[str, Any]]:
    from agent.harness.action_registry import (
        ACTION_BY_TYPE,
        resolve_action_target_group,
    )
    from agent.harness.semantic_compiler import freeze_semantic_plan

    source = "I need to benchmark a different chain."
    action = {
        "type": "greeting" if forged else "request_chain_selection",
        "source_evidence": source,
    }
    unit = {
        "unit_id": "unit-1",
        "clause_id": "clause-1",
        "source_text": source,
        "disposition": "action",
        "action_indexes": [0],
    }
    spec = ACTION_BY_TYPE[action["type"]]
    plan = freeze_semantic_plan(
        {"actions": [action], "semantic_units": [unit]},
        action_records=[{
            "action_id": "action-1",
            "action_index": 0,
            "action": action,
            "registry_owner": spec.owner,
            "registry_effect": spec.effect,
            "registry_target_group": resolve_action_target_group(action),
            "registry_incomplete_mutation_intake": True,
            "registry_incomplete_read_intake": False,
            "required_value_grounding_arguments": [],
            "unit_ids": ["unit-1"],
            "allowed_support_relations": [],
            "required_evidence_relations": [{
                "unit_id": "unit-1",
                "relation": "direct",
                "support_relation": "",
            }],
        }],
        unit_records=[{
            "unit_id": "unit-1",
            "unit_index": 0,
            "unit": unit,
            "source_text": source,
            "evidence_sources": [source],
            "disposition": "action",
            "owner_action_ids": ["action-1"],
        }],
        review_context={"pending_question": {}},
    )
    request = SimpleNamespace(
        messages=[None, SimpleNamespace(content=plan.request_json)]
    )
    return plan, _whole_plan_admission_payload(request)


def _confirmation_proposal_admission_fixture() -> tuple[Any, dict[str, Any]]:
    from agent.harness.action_registry import ACTION_BY_TYPE
    from agent.harness.semantic_compiler import freeze_semantic_plan

    source = "Change the chain to BNB."
    action = {
        "type": "choose_chain",
        "chain_text": "BNB",
        "source_evidence": "BNB",
    }
    unit = {
        "unit_id": "unit-1",
        "clause_id": "clause-1",
        "source_text": source,
        "disposition": "action",
        "action_indexes": [0],
    }
    spec = ACTION_BY_TYPE["choose_chain"]
    plan = freeze_semantic_plan(
        {"actions": [action], "semantic_units": [unit]},
        action_records=[{
            "action_id": "action-1",
            "action_index": 0,
            "action": action,
            "registry_effect": spec.effect,
            "required_value_grounding_arguments": ["chain_text"],
            "open_identity_grounding_arguments": ["chain_text"],
            "unit_ids": ["unit-1"],
            "allowed_support_relations": [],
        }],
        unit_records=[{
            "unit_id": "unit-1",
            "unit_index": 0,
            "unit": unit,
            "source_text": source,
            "evidence_sources": [source],
            "disposition": "action",
            "owner_action_ids": ["action-1"],
        }],
        review_context={"pending_question": {}},
    )
    request = SimpleNamespace(
        messages=[None, SimpleNamespace(content=plan.request_json)]
    )
    return plan, _whole_plan_admission_payload(request)


def _closed_enum_grounding_payload(
    plan: Any,
    *,
    status: str = "selected",
    evidence_quote: str = "fake-node",
) -> dict[str, Any]:
    from agent.harness.semantic_compiler import _closed_enum_grounding_request

    request = _closed_enum_grounding_request(plan)
    assert request is not None
    return {
        "verdicts": [
            {
                "action_id": row["action_id"],
                "argument_name": row["argument_name"],
                "selected_value": row["selected_value"],
                "status": status,
                "evidence_quote": evidence_quote,
                "reason": f"the exact source classifies the value as {status}",
            }
            for row in request["groundings"]
        ],
        "reason": "independent closed-enum grounding review",
    }


def _immutable_required_relation_fixture() -> tuple[Any, dict[str, Any]]:
    from agent.harness.semantic_compiler import freeze_semantic_plan

    source = "fake-node benchmark"
    action = {"type": "greeting", "source_evidence": "fake-node"}
    unit = {
        "unit_id": "unit-1",
        "clause_id": "clause-1",
        "source_text": source,
        "disposition": "action",
        "action_indexes": [0],
    }
    plan = freeze_semantic_plan(
        {"actions": [action], "semantic_units": [unit]},
        action_records=[{
            "action_id": "action-1",
            "action_index": 0,
            "action": action,
            "unit_ids": ["unit-1"],
            "allowed_support_relations": ["operation_restatement"],
            "required_evidence_relations": [{
                "unit_id": "unit-1",
                "relation": "direct",
                "support_relation": "",
            }],
        }],
        unit_records=[{
            "unit_id": "unit-1",
            "unit_index": 0,
            "unit": unit,
            "source_text": source,
            "disposition": "action",
            "owner_action_ids": ["action-1"],
        }],
        review_context={"pending_question": {}},
    )
    request = SimpleNamespace(
        messages=[None, SimpleNamespace(content=plan.request_json)]
    )
    return plan, _whole_plan_admission_payload(request)


def _immutable_required_context_fixture(
    *, authoritative: bool = True,
) -> tuple[Any, dict[str, Any]]:
    from agent.harness.semantic_compiler import freeze_semantic_plan

    action_source = "fake-node"
    context_source = "benchmark"
    action = {"type": "greeting", "source_evidence": action_source}
    units = [
        {
            "unit_id": "unit-1",
            "clause_id": "clause-1",
            "source_text": action_source,
            "disposition": "action",
            "action_indexes": [0],
        },
        {
            "unit_id": "unit-2",
            "clause_id": "clause-1",
            "source_text": context_source,
            "disposition": "context",
            "action_indexes": [],
        },
    ]
    plan = freeze_semantic_plan(
        {"actions": [action], "semantic_units": units},
        action_records=[{
            "action_id": "action-1",
            "action_index": 0,
            "action": action,
            "unit_ids": ["unit-1"],
            "allowed_support_relations": [],
            "required_evidence_relations": [],
        }],
        unit_records=[
            {
                "unit_id": "unit-1",
                "unit_index": 0,
                "unit": units[0],
                "source_text": action_source,
                "disposition": "action",
                "required_unit_verdict": "",
                "owner_action_ids": ["action-1"],
            },
            {
                "unit_id": "unit-2",
                "unit_index": 1,
                "unit": units[1],
                "source_text": context_source,
                "disposition": "context",
                "required_unit_verdict": "context" if authoritative else "",
                "owner_action_ids": [],
            },
        ],
        review_context={"pending_question": {}},
    )
    request = SimpleNamespace(
        messages=[None, SimpleNamespace(content=plan.request_json)]
    )
    return plan, _whole_plan_admission_payload(request)


class BoundedSemanticAdmissionTest(unittest.TestCase):

    def test_authoritative_context_verdict_cannot_be_reinterpreted(self) -> None:
        from agent.harness.semantic_admission import ALLOWED_ACTION_TYPES
        from agent.harness.semantic_compiler import validate_whole_plan_admission

        plan, valid = _immutable_required_context_fixture()
        self.assertTrue(validate_whole_plan_admission(
            json.dumps(valid),
            plan,
            allowed_action_types=ALLOWED_ACTION_TYPES,
        ).valid)

        omitted = deepcopy(valid)
        omitted["unit_verdicts"][1].update({
            "verdict": "omitted",
            "omitted_action_type": "choose_chain",
            "reason": "reinterpret the support as a pending chain answer",
        })
        result = validate_whole_plan_admission(
            json.dumps(omitted),
            plan,
            allowed_action_types=ALLOWED_ACTION_TYPES,
        )
        self.assertFalse(result.valid)
        self.assertIn(
            "immutable verdict contract",
            "; ".join(result.errors),
        )

    def test_frozen_authoritative_context_must_be_ownerless_context(self) -> None:
        from agent.harness.semantic_compiler import freeze_semantic_plan

        action = {"type": "greeting", "source_evidence": "hello"}
        unit = {
            "unit_id": "unit-1",
            "clause_id": "clause-1",
            "source_text": "hello",
            "disposition": "action",
            "action_indexes": [0],
        }
        with self.assertRaisesRegex(ValueError, "not context-only"):
            freeze_semantic_plan(
                {"actions": [action], "semantic_units": [unit]},
                action_records=[{
                    "action_id": "action-1",
                    "action_index": 0,
                    "action": action,
                    "unit_ids": ["unit-1"],
                    "allowed_support_relations": [],
                    "required_evidence_relations": [],
                }],
                unit_records=[{
                    "unit_id": "unit-1",
                    "unit_index": 0,
                    "unit": unit,
                    "source_text": "hello",
                    "disposition": "action",
                    "required_unit_verdict": "context",
                    "owner_action_ids": ["action-1"],
                }],
                review_context={"pending_question": {}},
            )

    def test_ordinary_context_remains_open_to_omission_review(self) -> None:
        from agent.harness.semantic_admission import ALLOWED_ACTION_TYPES
        from agent.harness.semantic_compiler import validate_whole_plan_admission

        plan, payload = _immutable_required_context_fixture(
            authoritative=False,
        )
        payload["unit_verdicts"][1].update({
            "verdict": "omitted",
            "omitted_action_type": "choose_chain",
            "reason": "the ordinary context contains a separate chain demand",
        })
        result = validate_whole_plan_admission(
            json.dumps(payload),
            plan,
            allowed_action_types=ALLOWED_ACTION_TYPES,
        )
        self.assertFalse(result.valid)
        self.assertNotIn(
            "immutable verdict contract",
            "; ".join(result.errors),
        )
        self.assertIn(
            "unresolved or omitted demand",
            "; ".join(result.errors),
        )
    def test_canonical_chain_selection_uses_mutation_consensus(
        self,
    ) -> None:
        from agent.harness.action_registry import ACTION_BY_TYPE
        from agent.harness.semantic_admission import ALLOWED_ACTION_TYPES
        from agent.harness.semantic_compiler import request_whole_plan_admission

        plan, valid = _confirmation_proposal_admission_fixture()
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(
            text=json.dumps(valid, sort_keys=True),
        )

        admission = request_whole_plan_admission(
            provider,
            plan,
            semantic_policy="preserve the immutable plan",
            allowed_action_types=ALLOWED_ACTION_TYPES,
        )

        self.assertNotIn("change_chain", ACTION_BY_TYPE)
        self.assertEqual(ACTION_BY_TYPE["choose_chain"].effect, "configuration_mutation")
        self.assertTrue(admission.valid, admission.errors)
        self.assertTrue(admission.consensus_required)
        self.assertEqual(provider.complete.call_count, 3)

    def test_mutation_jury_runs_concurrently_with_ordered_receipts(self) -> None:
        from agent.harness.semantic_admission import ALLOWED_ACTION_TYPES
        from agent.harness.semantic_compiler import request_whole_plan_admission
        from agent.llm.types import llm_turn_scope

        plan, valid = _confirmation_proposal_admission_fixture()
        barrier = threading.Barrier(3)

        class Provider:
            def complete(self, _request):
                barrier.wait(timeout=1)
                return SimpleNamespace(text=json.dumps(valid, sort_keys=True))

        with llm_turn_scope(1):
            admission = request_whole_plan_admission(
                Provider(),
                plan,
                semantic_policy="preserve the immutable plan",
                allowed_action_types=ALLOWED_ACTION_TYPES,
            )

        self.assertTrue(admission.valid, admission.errors)
        self.assertEqual(
            admission.review_ids,
            (
                "jury_1/attempt_1",
                "jury_2/attempt_1",
                "jury_3/attempt_1",
            ),
        )

    def test_canonical_chain_selection_purpose_is_lifecycle_neutral(self) -> None:
        from agent.harness.action_registry import ACTION_BY_TYPE
        from agent.harness.semantic_admission import _semantic_action_purpose

        action = {
            "type": "choose_chain",
            "chain_text": "BNB",
            "source_evidence": "BNB",
        }
        spec = ACTION_BY_TYPE["choose_chain"]
        initial_state = {"chain_identity": {}}
        replacement_state = {
            "chain_identity": {
                "raw": "solana",
                "canonical": "solana",
                "status": "confirmed",
            },
        }

        self.assertEqual(
            _semantic_action_purpose(action, spec.purpose, initial_state),
            _semantic_action_purpose(action, spec.purpose, replacement_state),
        )
        self.assertEqual(spec.required_state_path, ())
        self.assertNotIn("change_chain", ACTION_BY_TYPE)

    def test_stage_b_chain_context_does_not_expose_canonical_alias_catalog(
        self,
    ) -> None:
        from agent.harness.hierarchical_planner import _stage_b_payload

        source = "我需要换成 BNB"
        payload = _stage_b_payload(
            {
                "chain_identity": {
                    "raw": "solana",
                    "canonical": "solana",
                    "status": "confirmed",
                },
                "pending_question": {},
            },
            "chain_rpc",
            frozenset({"chain_identity"}),
            ({
                "unit_id": "unit-1",
                "clause_id": "clause-1",
                "source_text": source,
                "operation": "domain_request",
                "owner_routes": [{
                    "owner": "chain_rpc",
                    "group": "chain_identity",
                }],
                "reason": "source-grounded chain selection",
            },),
            ("unit-1",),
        )

        self.assertNotIn("registered_semantic_value_domains", payload)
        self.assertEqual(payload["semantic_units"][0]["source_text"], source)
        choose_chain = next(
            row
            for row in payload["owner_action_schema"]
            if row["type"] == "choose_chain"
        )
        self.assertIn("exact user-supplied raw chain identity", choose_chain["purpose"])
        self.assertEqual(
            choose_chain["open_identity_grounding_arguments"],
            ["chain_text", "chain_candidates"],
        )

    def test_grounded_mutation_requires_three_member_semantic_jury(self) -> None:
        from agent.harness.semantic_admission import ALLOWED_ACTION_TYPES
        from agent.harness.semantic_compiler import request_whole_plan_admission

        plan, valid = _grounded_mutation_admission_fixture()
        provider = Mock()
        provider.complete.side_effect = [
            SimpleNamespace(text=json.dumps(valid, sort_keys=True)),
            SimpleNamespace(text=json.dumps(valid, sort_keys=True)),
            SimpleNamespace(text=json.dumps(valid, sort_keys=True)),
            SimpleNamespace(
                text=json.dumps(
                    _closed_enum_grounding_payload(plan),
                    sort_keys=True,
                )
            ),
        ]

        admission = request_whole_plan_admission(
            provider,
            plan,
            semantic_policy="preserve the immutable plan",
            allowed_action_types=ALLOWED_ACTION_TYPES,
        )

        self.assertTrue(admission.valid, admission.errors)
        self.assertTrue(admission.consensus_required)
        self.assertEqual(provider.complete.call_count, 4)
        self.assertEqual(admission.request_count, 4)
        self.assertEqual(len(admission.review_hashes), 4)

    def test_closed_enum_negation_fails_closed_after_whole_plan_consensus(
        self,
    ) -> None:
        from agent.harness.semantic_admission import ALLOWED_ACTION_TYPES
        from agent.harness.semantic_compiler import request_whole_plan_admission

        source = "不使用 fake-node 模式"
        plan, valid = _grounded_mutation_admission_fixture(source=source)
        rejected = _closed_enum_grounding_payload(
            plan,
            status="rejected",
            evidence_quote=source,
        )
        provider = Mock()
        provider.complete.side_effect = [
            SimpleNamespace(text=json.dumps(valid, sort_keys=True)),
            SimpleNamespace(text=json.dumps(valid, sort_keys=True)),
            SimpleNamespace(text=json.dumps(valid, sort_keys=True)),
            SimpleNamespace(text=json.dumps(rejected, ensure_ascii=False)),
        ]

        admission = request_whole_plan_admission(
            provider,
            plan,
            semantic_policy="preserve the immutable plan",
            allowed_action_types=ALLOWED_ACTION_TYPES,
        )

        self.assertFalse(admission.valid)
        self.assertEqual(provider.complete.call_count, 4)
        self.assertEqual(admission.request_count, 4)
        self.assertEqual(admission.action_verdicts[0]["verdict"], "reject")
        self.assertIn(
            "independent closed-enum grounding rejected",
            "; ".join(admission.errors),
        )

    def test_malformed_closed_enum_grounding_fails_closed(self) -> None:
        from agent.harness.semantic_admission import ALLOWED_ACTION_TYPES
        from agent.harness.semantic_compiler import request_whole_plan_admission

        plan, valid = _grounded_mutation_admission_fixture()
        provider = Mock()
        provider.complete.side_effect = [
            SimpleNamespace(text=json.dumps(valid, sort_keys=True)),
            SimpleNamespace(text=json.dumps(valid, sort_keys=True)),
            SimpleNamespace(text=json.dumps(valid, sort_keys=True)),
            SimpleNamespace(text="{}"),
        ]

        admission = request_whole_plan_admission(
            provider,
            plan,
            semantic_policy="preserve the immutable plan",
            allowed_action_types=ALLOWED_ACTION_TYPES,
        )

        self.assertFalse(admission.valid)
        self.assertEqual(provider.complete.call_count, 4)
        self.assertEqual(admission.action_verdicts[0]["verdict"], "reject")

    def test_grounded_mutation_jury_rejection_is_atomic(self) -> None:
        from agent.harness.semantic_admission import ALLOWED_ACTION_TYPES
        from agent.harness.semantic_compiler import request_whole_plan_admission

        plan, valid = _grounded_mutation_admission_fixture()
        rejected = deepcopy(valid)
        rejected["action_verdicts"][0]["verdict"] = "reject"
        rejected["action_verdicts"][0]["reason"] = (
            "the source rejects the immutable value"
        )
        provider = Mock()
        provider.complete.side_effect = [
            SimpleNamespace(text=json.dumps(valid, sort_keys=True)),
            SimpleNamespace(text=json.dumps(rejected, sort_keys=True)),
            SimpleNamespace(text=json.dumps(rejected, sort_keys=True)),
        ]

        admission = request_whole_plan_admission(
            provider,
            plan,
            semantic_policy="preserve the immutable plan",
            allowed_action_types=ALLOWED_ACTION_TYPES,
        )

        self.assertFalse(admission.valid)
        self.assertTrue(admission.consensus_required)
        self.assertIn(
            "grounded mutation semantic jury did not reach admission quorum",
            "; ".join(admission.errors),
        )
        self.assertEqual(provider.complete.call_count, 3)

    def test_one_grounded_mutation_rejection_cannot_veto_two_admissions(self) -> None:
        from agent.harness.semantic_admission import ALLOWED_ACTION_TYPES
        from agent.harness.semantic_compiler import request_whole_plan_admission

        plan, valid = _grounded_mutation_admission_fixture()
        rejected = deepcopy(valid)
        rejected["action_verdicts"][0]["verdict"] = "reject"
        rejected["action_verdicts"][0]["reason"] = (
            "the immutable value is not affirmatively selected"
        )
        verdicts = (rejected, valid, valid)
        for rejected_index in range(3):
            ordered = list(verdicts[1:])
            ordered.insert(rejected_index, verdicts[0])
            provider = Mock()
            provider.complete.side_effect = [
                SimpleNamespace(text=json.dumps(item, sort_keys=True))
                for item in ordered
            ] + [SimpleNamespace(
                text=json.dumps(
                    _closed_enum_grounding_payload(plan),
                    sort_keys=True,
                )
            )]

            admission = request_whole_plan_admission(
                provider,
                plan,
                semantic_policy="preserve the immutable plan",
                allowed_action_types=ALLOWED_ACTION_TYPES,
            )

            with self.subTest(rejected_index=rejected_index):
                self.assertTrue(admission.valid, admission.errors)
                self.assertTrue(admission.consensus_required)
                self.assertEqual(provider.complete.call_count, 4)
                self.assertEqual(len(admission.review_hashes), 4)

    def test_one_malformed_jury_member_cannot_veto_two_admissions(self) -> None:
        from agent.harness.semantic_admission import ALLOWED_ACTION_TYPES
        from agent.harness.semantic_compiler import request_whole_plan_admission

        plan, valid = _grounded_mutation_admission_fixture()
        provider = Mock()
        provider.complete.side_effect = [
            SimpleNamespace(text=json.dumps(valid, sort_keys=True)),
            SimpleNamespace(text="{}"),
            SimpleNamespace(text=json.dumps(valid, sort_keys=True)),
            SimpleNamespace(
                text=json.dumps(
                    _closed_enum_grounding_payload(plan),
                    sort_keys=True,
                )
            ),
        ]

        admission = request_whole_plan_admission(
            provider,
            plan,
            semantic_policy="preserve the immutable plan",
            allowed_action_types=ALLOWED_ACTION_TYPES,
        )

        self.assertTrue(admission.valid, admission.errors)
        self.assertTrue(admission.consensus_required)
        self.assertEqual(provider.complete.call_count, 4)

    def test_closed_enum_rejection_uses_quorum_admission_verdicts(self) -> None:
        from agent.harness.semantic_admission import ALLOWED_ACTION_TYPES
        from agent.harness.semantic_compiler import request_whole_plan_admission

        plan, valid = _grounded_mutation_admission_fixture()
        rejected_grounding = _closed_enum_grounding_payload(
            plan,
            status="rejected",
            evidence_quote="not fake-node",
        )
        provider = Mock()
        provider.complete.side_effect = [
            SimpleNamespace(text="{}"),
            SimpleNamespace(text=json.dumps(valid, sort_keys=True)),
            SimpleNamespace(text=json.dumps(valid, sort_keys=True)),
            SimpleNamespace(text=json.dumps(rejected_grounding, sort_keys=True)),
        ]

        admission = request_whole_plan_admission(
            provider,
            plan,
            semantic_policy="preserve the immutable plan",
            allowed_action_types=ALLOWED_ACTION_TYPES,
        )

        self.assertFalse(admission.valid)
        self.assertEqual(len(admission.action_verdicts), 1)
        self.assertEqual(admission.action_verdicts[0]["verdict"], "reject")
        self.assertIn(
            "independent closed-enum grounding",
            admission.action_verdicts[0]["reason"],
        )

    def test_legacy_two_reviewer_consensus_receipt_fails_closed(self) -> None:
        from agent.harness.action_registry import (
            build_semantic_consensus_receipt,
            validate_semantic_consensus_receipt,
        )

        receipt = build_semantic_consensus_receipt(
            thread_id="thread-1",
            session_id="session-1",
            submitted_turn_index=3,
            transaction_hash="transaction-1",
            plan_hash="plan-1",
            admission_action_ids=("action-1",),
            review_hashes=("review-1", "review-2", "review-3"),
            review_ids=(
                "jury_1/attempt_1",
                "jury_2/attempt_1",
                "jury_3/attempt_1",
            ),
            request_count=3,
            request_sizes=(10, 10, 10),
        )
        legacy = dict(receipt)
        legacy.update({
            "version": 3,
            "review_hashes": ["review-1", "review-2"],
            "review_ids": ["jury_1/attempt_1", "jury_2/attempt_1"],
            "request_count": 2,
            "request_sizes": [10, 10],
        })
        action = {
            "_semantic_consensus_receipt": legacy,
            "_plan_transaction_hash": "transaction-1",
            "_admission_action_id": "action-1",
        }

        with self.assertRaisesRegex(ValueError, "receipt is incomplete"):
            validate_semantic_consensus_receipt(action)

    def test_tampered_semantic_jury_receipt_fails_closed(self) -> None:
        from agent.harness.action_registry import (
            build_semantic_consensus_receipt,
            validate_semantic_consensus_receipt,
        )

        receipt = build_semantic_consensus_receipt(
            thread_id="thread-1",
            session_id="session-1",
            submitted_turn_index=3,
            transaction_hash="transaction-1",
            plan_hash="plan-1",
            admission_action_ids=("action-1",),
            review_hashes=("review-1", "review-2", "review-3"),
            review_ids=(
                "jury_1/attempt_1",
                "jury_2/attempt_1",
                "jury_3/attempt_1",
            ),
            request_count=3,
            request_sizes=(10, 10, 10),
        )
        tampered = deepcopy(receipt)
        tampered["review_ids"][1] = "primary"
        action = {
            "_semantic_consensus_receipt": tampered,
            "_plan_transaction_hash": "transaction-1",
            "_admission_action_id": "action-1",
        }

        with self.assertRaisesRegex(ValueError, "receipt is incomplete"):
            validate_semantic_consensus_receipt(action)

    def test_two_malformed_jury_members_fail_closed(self) -> None:
        from agent.harness.semantic_admission import ALLOWED_ACTION_TYPES
        from agent.harness.semantic_compiler import request_whole_plan_admission

        plan, valid = _grounded_mutation_admission_fixture()
        provider = Mock()
        provider.complete.side_effect = [
            SimpleNamespace(text=json.dumps(valid, sort_keys=True)),
            SimpleNamespace(text="{}"),
            SimpleNamespace(text="{}"),
        ]

        admission = request_whole_plan_admission(
            provider,
            plan,
            semantic_policy="preserve the immutable plan",
            allowed_action_types=ALLOWED_ACTION_TYPES,
        )

        self.assertFalse(admission.valid)
        self.assertTrue(admission.consensus_required)
        self.assertIn(
            "semantic jury did not reach admission quorum",
            "; ".join(admission.errors),
        )
        self.assertEqual(provider.complete.call_count, 3)

    def test_typed_intake_primary_admission_does_not_escalate(self) -> None:
        from agent.harness.semantic_admission import ALLOWED_ACTION_TYPES
        from agent.harness.semantic_compiler import request_whole_plan_admission

        plan, valid = _incomplete_intake_admission_fixture()
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(
            text=json.dumps(valid, sort_keys=True),
        )

        admission = request_whole_plan_admission(
            provider,
            plan,
            semantic_policy="preserve the immutable plan",
            allowed_action_types=ALLOWED_ACTION_TYPES,
        )

        self.assertTrue(admission.valid, admission.errors)
        self.assertFalse(admission.consensus_required)
        self.assertEqual(provider.complete.call_count, 1)
        self.assertEqual(admission.review_ids, ("primary/attempt_1",))

    def test_typed_intake_primary_rejection_uses_adaptive_quorum(self) -> None:
        from agent.harness.semantic_admission import ALLOWED_ACTION_TYPES
        from agent.harness.semantic_compiler import request_whole_plan_admission

        plan, valid = _incomplete_intake_admission_fixture()
        rejected = deepcopy(valid)
        rejected["action_verdicts"][0]["verdict"] = "reject"
        rejected["action_verdicts"][0]["reason"] = (
            "the source does not request this intake"
        )
        provider = Mock()
        provider.complete.side_effect = [
            SimpleNamespace(text=json.dumps(rejected, sort_keys=True)),
            SimpleNamespace(text=json.dumps(valid, sort_keys=True)),
            SimpleNamespace(text=json.dumps(valid, sort_keys=True)),
        ]

        admission = request_whole_plan_admission(
            provider,
            plan,
            semantic_policy="preserve the immutable plan",
            allowed_action_types=ALLOWED_ACTION_TYPES,
        )

        self.assertTrue(admission.valid, admission.errors)
        self.assertTrue(admission.consensus_required)
        self.assertEqual(provider.complete.call_count, 3)
        self.assertEqual(
            admission.review_ids,
            (
                "jury_1/attempt_1",
                "jury_2/attempt_1",
                "jury_3/attempt_1",
            ),
        )

    def test_typed_intake_malformed_primary_can_reach_quorum(self) -> None:
        from agent.harness.semantic_admission import ALLOWED_ACTION_TYPES
        from agent.harness.semantic_compiler import request_whole_plan_admission

        plan, valid = _incomplete_intake_admission_fixture()
        provider = Mock()
        provider.complete.side_effect = [
            SimpleNamespace(text="{}"),
            SimpleNamespace(text=json.dumps(valid, sort_keys=True)),
            SimpleNamespace(text=json.dumps(valid, sort_keys=True)),
        ]

        admission = request_whole_plan_admission(
            provider,
            plan,
            semantic_policy="preserve the immutable plan",
            allowed_action_types=ALLOWED_ACTION_TYPES,
        )

        self.assertTrue(admission.valid, admission.errors)
        self.assertTrue(admission.consensus_required)
        self.assertEqual(provider.complete.call_count, 3)
        self.assertEqual(len(admission.review_hashes), 3)

    def test_typed_intake_adaptive_quorum_fails_closed(self) -> None:
        from agent.harness.semantic_admission import ALLOWED_ACTION_TYPES
        from agent.harness.semantic_compiler import request_whole_plan_admission

        plan, valid = _incomplete_intake_admission_fixture()
        rejected = deepcopy(valid)
        rejected["action_verdicts"][0]["verdict"] = "reject"
        rejected["action_verdicts"][0]["reason"] = "the intake is not selected"
        provider = Mock()
        provider.complete.side_effect = [
            SimpleNamespace(text=json.dumps(rejected, sort_keys=True)),
            SimpleNamespace(text=json.dumps(valid, sort_keys=True)),
            SimpleNamespace(text=json.dumps(rejected, sort_keys=True)),
        ]

        admission = request_whole_plan_admission(
            provider,
            plan,
            semantic_policy="preserve the immutable plan",
            allowed_action_types=ALLOWED_ACTION_TYPES,
        )

        self.assertFalse(admission.valid)
        self.assertTrue(admission.consensus_required)
        self.assertIn(
            "typed intake semantic jury did not reach admission quorum",
            admission.errors,
        )
        self.assertEqual(provider.complete.call_count, 3)

    def test_typed_intake_malformed_secondary_cannot_form_quorum(self) -> None:
        from agent.harness.semantic_admission import ALLOWED_ACTION_TYPES
        from agent.harness.semantic_compiler import request_whole_plan_admission

        plan, valid = _incomplete_intake_admission_fixture()
        rejected = deepcopy(valid)
        rejected["action_verdicts"][0]["verdict"] = "reject"
        rejected["action_verdicts"][0]["reason"] = "the intake is not selected"
        provider = Mock()
        provider.complete.side_effect = [
            SimpleNamespace(text=json.dumps(rejected, sort_keys=True)),
            SimpleNamespace(text="{}"),
            SimpleNamespace(text=json.dumps(valid, sort_keys=True)),
        ]

        admission = request_whole_plan_admission(
            provider,
            plan,
            semantic_policy="preserve the immutable plan",
            allowed_action_types=ALLOWED_ACTION_TYPES,
        )

        self.assertFalse(admission.valid)
        self.assertTrue(admission.consensus_required)
        self.assertEqual(provider.complete.call_count, 3)

    def test_non_intake_rejection_does_not_escalate(self) -> None:
        from agent.harness.semantic_admission import ALLOWED_ACTION_TYPES
        from agent.harness.semantic_compiler import request_whole_plan_admission

        plan, valid = _immutable_admission_fixture()
        rejected = deepcopy(valid)
        rejected["action_verdicts"][0]["verdict"] = "reject"
        rejected["action_verdicts"][0]["reason"] = "the action is not selected"
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(
            text=json.dumps(rejected, sort_keys=True),
        )

        admission = request_whole_plan_admission(
            provider,
            plan,
            semantic_policy="preserve the immutable plan",
            allowed_action_types=ALLOWED_ACTION_TYPES,
        )

        self.assertFalse(admission.valid)
        self.assertFalse(admission.consensus_required)
        self.assertEqual(provider.complete.call_count, 1)

    def test_forged_incomplete_intake_metadata_does_not_escalate(self) -> None:
        from agent.harness.semantic_admission import ALLOWED_ACTION_TYPES
        from agent.harness.semantic_compiler import request_whole_plan_admission

        plan, valid = _incomplete_intake_admission_fixture(forged=True)
        rejected = deepcopy(valid)
        rejected["action_verdicts"][0]["verdict"] = "reject"
        rejected["action_verdicts"][0]["reason"] = "the action is not selected"
        provider = Mock()
        provider.complete.return_value = SimpleNamespace(
            text=json.dumps(rejected, sort_keys=True),
        )

        admission = request_whole_plan_admission(
            provider,
            plan,
            semantic_policy="preserve the immutable plan",
            allowed_action_types=ALLOWED_ACTION_TYPES,
        )

        self.assertFalse(admission.valid)
        self.assertFalse(admission.consensus_required)
        self.assertEqual(provider.complete.call_count, 1)

    def test_strict_json_compilation_disables_provider_reasoning(self) -> None:
        from agent.harness.semantic_admission import ALLOWED_ACTION_TYPES
        from agent.harness.semantic_compiler import (
            request_semantic_compilation_result,
            request_whole_plan_admission,
        )

        compilation_provider = Mock()
        compilation_provider.complete.return_value = SimpleNamespace(
            text="{}",
            provider="deepseek",
            model="deepseek-v4-pro",
        )
        request_semantic_compilation_result(
            compilation_provider,
            system_prompt="Return one JSON object.",
            request_payload={"value": "test"},
        )

        compilation_request = compilation_provider.complete.call_args.args[0]
        self.assertEqual(compilation_request.reasoning_mode, "disabled")

        plan, valid = _immutable_admission_fixture()
        admission_provider = Mock()
        admission_provider.complete.return_value = SimpleNamespace(
            text=json.dumps(valid, sort_keys=True),
        )
        admission = request_whole_plan_admission(
            admission_provider,
            plan,
            semantic_policy="preserve the immutable plan",
            allowed_action_types=ALLOWED_ACTION_TYPES,
        )

        self.assertTrue(admission.valid, admission.errors)
        self.assertFalse(admission.consensus_required)
        self.assertEqual(admission_provider.complete.call_count, 1)
        admission_request = admission_provider.complete.call_args.args[0]
        self.assertEqual(admission_request.reasoning_mode, "disabled")

    def test_admission_prompt_requires_unit_and_action_relation_consistency(self) -> None:
        from agent.harness.semantic_compiler import whole_plan_admission_prompt

        prompt = whole_plan_admission_prompt("policy")

        self.assertIn("one consistency contract", prompt)
        self.assertIn("Never return complete for a support-cited unit", prompt)
        self.assertIn(
            "excluding one value in a closed enum with multiple remaining values",
            prompt,
        )

    def test_typed_option_can_reopen_completed_intake_without_weakening_lifecycle(self) -> None:
        from agent.harness.action_registry import (
            build_replacement_intake_admission_receipt,
            lifecycle_rejected_action_indexes,
        )
        from agent.harness.state import new_state

        state = new_state("typed-replacement-lifecycle", language="en")
        state["target_mode"] = "fake-node"
        state["chain_identity"] = {
            "canonical": "bsc",
            "status": "confirmed",
        }
        ungrounded = {
            "type": "request_chain_selection",
            "source_evidence": "choose a chain",
        }
        declared_option = {
            **ungrounded,
            "selection_contract_verified": True,
        }
        admitted_replacement = {
            **ungrounded,
            "semantic_purpose_verified": True,
            "_admission_action_id": "replacement-action",
            "_plan_transaction_hash": "1" * 64,
        }
        admitted_replacement["_replacement_intake_receipt"] = (
            build_replacement_intake_admission_receipt(
                thread_id=state["thread_id"],
                session_id=state["session"]["id"],
                submitted_turn_index=state["turn_index"],
                transaction_hash="1" * 64,
                admission_action_id="replacement-action",
                action_type="request_chain_selection",
                source_evidence="choose a chain",
                target_group="chain_identity",
                provides_capabilities=("chain_identity",),
                reviewer_evidence_hash="2" * 64,
            )
        )
        semantic_only = {
            **ungrounded,
            "semantic_purpose_verified": True,
        }

        self.assertEqual(
            lifecycle_rejected_action_indexes(state, [ungrounded]),
            (0,),
        )
        self.assertEqual(
            lifecycle_rejected_action_indexes(state, [declared_option]),
            (),
        )
        self.assertEqual(
            lifecycle_rejected_action_indexes(state, [admitted_replacement]),
            (),
        )
        self.assertEqual(
            lifecycle_rejected_action_indexes(state, [semantic_only]),
            (0,),
        )
        tampered = {
            **admitted_replacement,
            "source_evidence": "unrelated consultation",
        }
        self.assertEqual(
            lifecycle_rejected_action_indexes(state, [tampered]),
            (0,),
        )

    def test_admitted_compound_chain_and_mode_replacement_is_atomic(self) -> None:
        from agent.harness.admission import validate_action_plan
        from agent.harness.action_registry import (
            build_replacement_intake_admission_receipt,
        )
        from agent.harness.state import new_state

        state = new_state("compound-chain-mode-replacement", language="en")
        state["target_mode"] = "fake-node"
        state["chain_identity"] = {
            "raw": "bsc",
            "canonical": "bsc",
            "status": "confirmed",
            "case": "known",
        }
        source = "Switch to ethereum and do not use fake-node."
        actions = [
            {
                "type": "choose_chain",
                "chain_text": "ethereum",
                "source_evidence": "Switch to ethereum",
                "semantic_purpose_verified": True,
            },
            {
                "type": "request_target_mode_selection",
                "source_evidence": "do not use fake-node",
                "semantic_purpose_verified": True,
                "_admission_action_id": "mode-action",
                "_plan_transaction_hash": "3" * 64,
            },
        ]
        actions[1]["_replacement_intake_receipt"] = (
            build_replacement_intake_admission_receipt(
                thread_id=state["thread_id"],
                session_id=state["session"]["id"],
                submitted_turn_index=state["turn_index"],
                transaction_hash="3" * 64,
                admission_action_id="mode-action",
                action_type="request_target_mode_selection",
                source_evidence="do not use fake-node",
                target_group="target_mode",
                provides_capabilities=("target_mode",),
                reviewer_evidence_hash="4" * 64,
            )
        )

        result = validate_action_plan(state, actions)

        self.assertEqual(result.status, "accepted", result.rejections)
        self.assertEqual(
            [action["type"] for action in result.actions],
            ["choose_chain", "request_target_mode_selection"],
        )

    def test_replacement_intake_receipt_survives_durable_queue_round_trip(
        self,
    ) -> None:
        from agent.harness.action_registry import (
            build_replacement_intake_admission_receipt,
            lifecycle_rejected_action_indexes,
        )
        from agent.harness.contracts import action_envelope_to_dict
        from agent.harness.coordinator import _build_action_envelope, _queue_action
        from agent.harness.state import new_state

        state = new_state("replacement-intake-queue-round-trip", language="en")
        state["target_mode"] = "fake-node"
        source = "Do not keep fake-node."
        action = {
            "type": "request_target_mode_selection",
            "action_id": "queued-mode-intake",
            "source_evidence": source,
            "_admission_action_id": "admitted-mode-intake",
            "_plan_transaction_hash": "5" * 64,
            "_transaction_action_ids": ["admitted-mode-intake"],
        }
        action["_replacement_intake_receipt"] = (
            build_replacement_intake_admission_receipt(
                thread_id=state["thread_id"],
                session_id=state["session"]["id"],
                submitted_turn_index=state["turn_index"],
                transaction_hash="5" * 64,
                admission_action_id="admitted-mode-intake",
                action_type="request_target_mode_selection",
                source_evidence=source,
                target_group="target_mode",
                provides_capabilities=("target_mode",),
                reviewer_evidence_hash="6" * 64,
            )
        )

        durable = action_envelope_to_dict(_build_action_envelope(state, action))
        restored = _queue_action(durable)

        self.assertEqual(
            restored["_replacement_intake_receipt"],
            action["_replacement_intake_receipt"],
        )
        self.assertEqual(
            restored["_plan_transaction_hash"],
            action["_plan_transaction_hash"],
        )
        self.assertEqual(lifecycle_rejected_action_indexes(state, [restored]), ())

        copied = deepcopy(state)
        copied["thread_id"] = "another-thread"
        copied["session"] = {
            **copied["session"],
            "id": "another-session",
        }
        copied["action_queue"] = [durable]
        from agent.harness.coordinator import select_action_step

        rejected = select_action_step(copied)
        self.assertEqual(rejected["action_queue"], [])
        self.assertEqual(
            rejected["action_errors"][-1]["error"],
            "durable_admission_metadata_invalid",
        )

    def _validate(self, payload: dict[str, Any] | str):
        from agent.harness.semantic_admission import ALLOWED_ACTION_TYPES
        from agent.harness.semantic_compiler import validate_whole_plan_admission

        plan, _valid = _immutable_admission_fixture()
        text = payload if isinstance(payload, str) else json.dumps(payload, sort_keys=True)
        return validate_whole_plan_admission(
            text,
            plan,
            allowed_action_types=ALLOWED_ACTION_TYPES,
        )

    def test_reviewer_payload_is_strict_json_with_exact_schema(self) -> None:
        _plan, valid = _immutable_admission_fixture()
        self.assertTrue(self._validate(valid).valid)
        self.assertFalse(self._validate("not json").valid)
        malformed = deepcopy(valid)
        malformed["actions"] = []
        result = self._validate(malformed)
        self.assertFalse(result.valid)
        self.assertIn("undeclared top-level keys", "; ".join(result.errors))

    def test_missing_duplicate_and_forged_action_verdicts_fail_closed(self) -> None:
        _plan, valid = _immutable_admission_fixture()
        variants: dict[str, dict[str, Any]] = {}
        missing = deepcopy(valid)
        missing["action_verdicts"].pop()
        variants["missing"] = missing
        duplicate = deepcopy(valid)
        duplicate["action_verdicts"] = [
            duplicate["action_verdicts"][0],
            deepcopy(duplicate["action_verdicts"][0]),
        ]
        variants["duplicate"] = duplicate
        forged = deepcopy(valid)
        forged["action_verdicts"][0]["action_id"] = "forged-action"
        variants["forged"] = forged
        for name, payload in variants.items():
            with self.subTest(name=name):
                self.assertFalse(self._validate(payload).valid)

    def test_missing_duplicate_and_forged_unit_verdicts_fail_closed(self) -> None:
        _plan, valid = _immutable_admission_fixture()
        variants: dict[str, dict[str, Any]] = {}
        missing = deepcopy(valid)
        missing["unit_verdicts"].pop()
        variants["missing"] = missing
        duplicate = deepcopy(valid)
        duplicate["unit_verdicts"] = [
            duplicate["unit_verdicts"][0],
            deepcopy(duplicate["unit_verdicts"][0]),
        ]
        variants["duplicate"] = duplicate
        forged = deepcopy(valid)
        forged["unit_verdicts"][0]["unit_id"] = "forged-unit"
        variants["forged"] = forged
        for name, payload in variants.items():
            with self.subTest(name=name):
                self.assertFalse(self._validate(payload).valid)

    def test_identical_duplicate_action_evidence_is_canonicalized(self) -> None:
        _plan, valid = _immutable_admission_fixture()
        duplicated = deepcopy(valid)
        duplicated["action_verdicts"][0]["evidence"].append(
            deepcopy(duplicated["action_verdicts"][0]["evidence"][0])
        )

        result = self._validate(duplicated)

        self.assertTrue(result.valid, result.errors)
        self.assertEqual(len(result.action_verdicts[0]["evidence"]), 1)

    def test_conflicting_duplicate_action_evidence_fails_closed(self) -> None:
        _plan, valid = _immutable_admission_fixture()
        conflicting = deepcopy(valid)
        second = deepcopy(conflicting["action_verdicts"][0]["evidence"][0])
        second["relation"] = "support"
        second["support_relation"] = "invented"
        conflicting["action_verdicts"][0]["evidence"].append(second)

        result = self._validate(conflicting)

        self.assertFalse(result.valid)
        self.assertIn("duplicates unit id", "; ".join(result.errors))

    def test_required_direct_relation_rejects_reviewer_support_duplicate(
        self,
    ) -> None:
        from agent.harness.semantic_admission import ALLOWED_ACTION_TYPES
        from agent.harness.semantic_compiler import validate_whole_plan_admission

        plan, payload = _immutable_required_relation_fixture()
        payload["action_verdicts"][0]["evidence"].append({
            "unit_id": "unit-1",
            "quote": "benchmark",
            "relation": "support",
            "support_relation": "operation_restatement",
        })

        result = validate_whole_plan_admission(
            json.dumps(payload, sort_keys=True),
            plan,
            allowed_action_types=ALLOWED_ACTION_TYPES,
        )

        self.assertFalse(result.valid)
        self.assertIn("duplicates unit id", "; ".join(result.errors))
        self.assertIn("immutable relation contract", "; ".join(result.errors))

    def test_required_relation_without_matching_receipt_fails_closed(self) -> None:
        from agent.harness.semantic_admission import ALLOWED_ACTION_TYPES
        from agent.harness.semantic_compiler import validate_whole_plan_admission

        plan, payload = _immutable_required_relation_fixture()
        payload["action_verdicts"][0]["evidence"] = [{
            "unit_id": "unit-1",
            "quote": "benchmark",
            "relation": "support",
            "support_relation": "operation_restatement",
        }]
        payload["unit_verdicts"][0]["verdict"] = "support"

        result = validate_whole_plan_admission(
            json.dumps(payload, sort_keys=True),
            plan,
            allowed_action_types=ALLOWED_ACTION_TYPES,
        )

        self.assertFalse(result.valid)
        self.assertIn("immutable relation contract", "; ".join(result.errors))

    def test_required_relation_does_not_hide_inexact_or_forged_receipts(
        self,
    ) -> None:
        from agent.harness.semantic_admission import ALLOWED_ACTION_TYPES
        from agent.harness.semantic_compiler import validate_whole_plan_admission

        for name, extra in {
            "inexact": {
                "unit_id": "unit-1",
                "quote": "invented",
                "relation": "support",
                "support_relation": "operation_restatement",
            },
            "forged": {
                "unit_id": "forged-unit",
                "quote": "fake-node",
                "relation": "direct",
                "support_relation": "",
            },
        }.items():
            with self.subTest(name=name):
                plan, payload = _immutable_required_relation_fixture()
                payload["action_verdicts"][0]["evidence"].append(extra)
                result = validate_whole_plan_admission(
                    json.dumps(payload, sort_keys=True),
                    plan,
                    allowed_action_types=ALLOWED_ACTION_TYPES,
                )
                self.assertFalse(result.valid)

    def test_required_relation_does_not_hide_malformed_or_ambiguous_receipts(
        self,
    ) -> None:
        from agent.harness.semantic_admission import ALLOWED_ACTION_TYPES
        from agent.harness.semantic_compiler import validate_whole_plan_admission

        for name, extra in {
            "invalid_relation": {
                "unit_id": "unit-1",
                "quote": "benchmark",
                "relation": "invalid",
                "support_relation": "",
            },
            "second_direct": {
                "unit_id": "unit-1",
                "quote": "benchmark",
                "relation": "direct",
                "support_relation": "",
            },
            "missing_field": {
                "unit_id": "unit-1",
                "quote": "benchmark",
                "relation": "support",
            },
        }.items():
            with self.subTest(name=name):
                plan, payload = _immutable_required_relation_fixture()
                payload["action_verdicts"][0]["evidence"].append(extra)
                result = validate_whole_plan_admission(
                    json.dumps(payload, sort_keys=True),
                    plan,
                    allowed_action_types=ALLOWED_ACTION_TYPES,
                )
                self.assertFalse(result.valid)

    def test_required_direct_relation_rejects_support_unit_verdict(self) -> None:
        from agent.harness.semantic_admission import ALLOWED_ACTION_TYPES
        from agent.harness.semantic_compiler import validate_whole_plan_admission

        plan, payload = _immutable_required_relation_fixture()
        payload["unit_verdicts"][0]["verdict"] = "support"

        result = validate_whole_plan_admission(
            json.dumps(payload, sort_keys=True),
            plan,
            allowed_action_types=ALLOWED_ACTION_TYPES,
        )

        self.assertFalse(result.valid)
        self.assertIn("immutable direct relation", "; ".join(result.errors))

    def test_duplicate_rejected_or_forged_evidence_is_not_admitted(self) -> None:
        _plan, valid = _immutable_admission_fixture()
        rejected = deepcopy(valid)
        rejected["action_verdicts"][0]["verdict"] = "reject"
        rejected["action_verdicts"][0]["evidence"].append(
            deepcopy(rejected["action_verdicts"][0]["evidence"][0])
        )
        rejected_result = self._validate(rejected)
        self.assertFalse(rejected_result.valid)
        self.assertIn(
            "duplicates unit id",
            "; ".join(rejected_result.errors),
        )

        forged = deepcopy(valid)
        forged["action_verdicts"][0]["evidence"][0]["unit_id"] = "forged-unit"
        forged["action_verdicts"][0]["evidence"].append(
            deepcopy(forged["action_verdicts"][0]["evidence"][0])
        )
        forged_result = self._validate(forged)
        self.assertFalse(forged_result.valid)
        self.assertIn("forged unit id", "; ".join(forged_result.errors))

    def test_reviewer_cannot_reorder_or_modify_the_immutable_plan(self) -> None:
        _plan, valid = _immutable_admission_fixture()
        reordered = deepcopy(valid)
        reordered["action_verdicts"].reverse()
        reordered["unit_verdicts"].reverse()
        result = self._validate(reordered)
        self.assertFalse(result.valid)
        self.assertIn("order does not match", "; ".join(result.errors))

        modified = deepcopy(valid)
        modified["action_verdicts"][0]["replacement_action"] = {
            "type": "reset_session",
        }
        result = self._validate(modified)
        self.assertFalse(result.valid)
        self.assertIn("undeclared keys", "; ".join(result.errors))

    def test_exact_quote_and_transport_bound_plan_identity_fail_closed(self) -> None:
        _plan, valid = _immutable_admission_fixture()
        invented = deepcopy(valid)
        invented["action_verdicts"][0]["evidence"][0]["quote"] = "invented"
        invented["unit_verdicts"][0]["evidence_quote"] = "invented"
        self.assertFalse(self._validate(invented).valid)

        forged_request_identity = deepcopy(valid)
        forged_request_identity["plan_hash"] = "forged"
        result = self._validate(forged_request_identity)
        self.assertFalse(result.valid)
        self.assertIn("undeclared top-level keys", "; ".join(result.errors))

    def test_reviewer_response_uses_local_request_binding(self) -> None:
        plan, valid = _immutable_admission_fixture()

        self.assertNotIn("plan_hash", valid)
        result = self._validate(valid)
        self.assertTrue(result.valid, result.errors)
        self.assertEqual(len(plan.plan_hash), 64)

    def test_registry_expressible_omission_rejects_the_whole_plan(self) -> None:
        _plan, valid = _immutable_admission_fixture()
        omitted = deepcopy(valid)
        omitted["unit_verdicts"][0].update({
            "verdict": "omitted",
            "omitted_action_type": "set_qps_mode",
            "reason": "the source has an omitted registered demand",
        })
        result = self._validate(omitted)
        self.assertFalse(result.valid)
        self.assertIn("neither complete nor support", "; ".join(result.errors))

    def test_registered_support_unit_is_admitted_without_becoming_context(self) -> None:
        from agent.harness.semantic_admission import ALLOWED_ACTION_TYPES
        from agent.harness.semantic_compiler import (
            freeze_semantic_plan,
            validate_whole_plan_admission,
        )

        support_text = "Use this endpoint only to validate the method."
        direct_text = "http://fake-node:8545"
        action = {
            "type": "rpc_catalog_command",
            "catalog_command": "set_endpoint",
            "rpc_endpoint": direct_text,
            "source_evidence": direct_text,
        }
        units = [
            {
                "unit_id": "unit-1",
                "clause_id": "clause-1",
                "source_text": support_text,
                "disposition": "action",
                "action_indexes": [0],
            },
            {
                "unit_id": "unit-2",
                "clause_id": "clause-2",
                "source_text": direct_text,
                "disposition": "action",
                "action_indexes": [0],
            },
        ]
        document = {"actions": [action], "semantic_units": units}
        plan = freeze_semantic_plan(
            document,
            action_records=[{
                "action_id": "action-1",
                "action_index": 0,
                "action": action,
                "unit_ids": ["unit-1", "unit-2"],
                "allowed_support_relations": ["operation_restatement"],
            }],
            unit_records=[
                {
                    "unit_id": unit["unit_id"],
                    "unit_index": index,
                    "unit": unit,
                    "source_text": unit["source_text"],
                    "disposition": "action",
                    "owner_action_ids": ["action-1"],
                }
                for index, unit in enumerate(units)
            ],
            review_context={"pending_question": {}},
        )
        payload = {
            "action_verdicts": [{
                "action_id": "action-1",
                "verdict": "admit",
                "unit_ids": ["unit-1", "unit-2"],
                "evidence": [
                    {
                        "unit_id": "unit-1",
                        "quote": support_text,
                        "relation": "support",
                        "support_relation": "operation_restatement",
                    },
                    {
                        "unit_id": "unit-2",
                        "quote": direct_text,
                        "relation": "direct",
                        "support_relation": "",
                    },
                ],
                "grounded_arguments": [],
                "pending_answer_argument": "",
                "turn_candidate_verdicts": [],
                "reason": "one supporting unit frames one direct endpoint value",
            }],
            "unit_verdicts": [
                {
                    "unit_id": "unit-1",
                    "verdict": "support",
                    "owner_action_ids": ["action-1"],
                    "evidence_quote": support_text,
                    "omitted_action_type": "",
                    "reason": "the unit scopes the direct endpoint operation",
                },
                {
                    "unit_id": "unit-2",
                    "verdict": "complete",
                    "owner_action_ids": ["action-1"],
                    "evidence_quote": direct_text,
                    "omitted_action_type": "",
                    "reason": "the unit directly supplies the endpoint",
                },
            ],
            "reason": "the immutable plan preserves both units",
        }

        result = validate_whole_plan_admission(
            json.dumps(payload),
            plan,
            allowed_action_types=ALLOWED_ACTION_TYPES,
        )
        self.assertTrue(result.valid, result.errors)

        no_owner = deepcopy(payload)
        no_owner["unit_verdicts"][0]["owner_action_ids"] = []
        self.assertFalse(validate_whole_plan_admission(
            json.dumps(no_owner),
            plan,
            allowed_action_types=ALLOWED_ACTION_TYPES,
        ).valid)

        wrong_relation = deepcopy(payload)
        wrong_relation["unit_verdicts"][0]["verdict"] = "complete"
        result = validate_whole_plan_admission(
            json.dumps(wrong_relation),
            plan,
            allowed_action_types=ALLOWED_ACTION_TYPES,
        )
        self.assertFalse(result.valid)
        self.assertIn(
            "complete unit is not direct",
            "; ".join(result.errors),
        )

    def test_required_value_grounding_has_exact_argument_cardinality(self) -> None:
        from agent.harness.semantic_admission import ALLOWED_ACTION_TYPES
        from agent.harness.semantic_compiler import (
            freeze_semantic_plan,
            validate_whole_plan_admission,
        )

        source = "Use the simulated-node workflow."
        action = {
            "type": "choose_target_mode",
            "target_mode": "fake-node",
            "target_mode_explicit": True,
            "source_evidence": source,
        }
        unit = {
            "unit_id": "unit-1",
            "clause_id": "clause-1",
            "source_text": source,
            "disposition": "action",
            "action_indexes": [0],
        }
        plan = freeze_semantic_plan(
            {"actions": [action], "semantic_units": [unit]},
            action_records=[{
                "action_id": "action-1",
                "action_index": 0,
                "action": action,
                "unit_ids": ["unit-1"],
                "allowed_support_relations": [],
                "operation_arguments": {"target_mode": "fake-node"},
                "required_value_grounding_arguments": ["target_mode"],
                "closed_enum_grounding_values": {
                    "target_mode": [
                        "fake-node",
                        "real-node",
                        "sync-observe",
                    ],
                },
            }],
            unit_records=[{
                "unit_id": "unit-1",
                "unit_index": 0,
                "unit": unit,
                "source_text": source,
                "disposition": "action",
                "owner_action_ids": ["action-1"],
            }],
            review_context={"pending_question": {}},
        )
        payload = {
            "action_verdicts": [{
                "action_id": "action-1",
                "verdict": "admit",
                "unit_ids": ["unit-1"],
                "evidence": [{
                    "unit_id": "unit-1",
                    "quote": source,
                    "relation": "direct",
                    "support_relation": "",
                }],
                "grounded_arguments": [{
                    "argument_name": "target_mode",
                    "evidence_quote": "simulated-node workflow",
                }],
                "pending_answer_argument": "",
                "turn_candidate_verdicts": [],
                "reason": "the source selects the concrete target mode",
            }],
            "unit_verdicts": [{
                "unit_id": "unit-1",
                "verdict": "complete",
                "owner_action_ids": ["action-1"],
                "evidence_quote": source,
                "omitted_action_type": "",
                "reason": "the concrete selection is preserved",
            }],
            "reason": "the immutable plan is grounded",
        }
        valid = validate_whole_plan_admission(
            json.dumps(payload),
            plan,
            allowed_action_types=ALLOWED_ACTION_TYPES,
        )
        self.assertTrue(valid.valid, valid.errors)

        missing = deepcopy(payload)
        missing["action_verdicts"][0]["grounded_arguments"] = []
        result = validate_whole_plan_admission(
            json.dumps(missing),
            plan,
            allowed_action_types=ALLOWED_ACTION_TYPES,
        )
        self.assertFalse(result.valid)
        self.assertIn("grounded argument order or cardinality mismatch", "; ".join(result.errors))

        competing_source = "Do not use fake-node mode."
        competing_action = {
            **action,
            "target_mode": "real-node",
            "source_evidence": competing_source,
        }
        competing_unit = {
            **unit,
            "source_text": competing_source,
        }
        competing_plan = freeze_semantic_plan(
            {
                "actions": [competing_action],
                "semantic_units": [competing_unit],
            },
            action_records=[{
                "action_id": "action-1",
                "action_index": 0,
                "action": competing_action,
                "unit_ids": ["unit-1"],
                "allowed_support_relations": [],
                "operation_arguments": {"target_mode": "real-node"},
                "required_value_grounding_arguments": ["target_mode"],
                "closed_enum_grounding_values": {
                    "target_mode": [
                        "fake-node",
                        "real-node",
                        "sync-observe",
                    ],
                },
            }],
            unit_records=[{
                "unit_id": "unit-1",
                "unit_index": 0,
                "unit": competing_unit,
                "source_text": competing_source,
                "disposition": "action",
                "owner_action_ids": ["action-1"],
            }],
            review_context={"pending_question": {}},
        )
        competing_payload = deepcopy(payload)
        competing_payload["action_verdicts"][0]["evidence"][0]["quote"] = (
            competing_source
        )
        competing_payload["action_verdicts"][0]["grounded_arguments"][0][
            "evidence_quote"
        ] = competing_source
        competing_payload["unit_verdicts"][0]["evidence_quote"] = (
            competing_source
        )
        result = validate_whole_plan_admission(
            json.dumps(competing_payload),
            competing_plan,
            allowed_action_types=ALLOWED_ACTION_TYPES,
        )
        self.assertFalse(result.valid)
        self.assertIn(
            "closed-enum grounding quote names only competing values",
            "; ".join(result.errors),
        )


def _rpc_catalog(
    *,
    methods: list[dict[str, Any]] | None = None,
    draft: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "contract_version": 1,
        "revision": 1,
        "methods": deepcopy(methods or []),
        "draft": deepcopy(draft or {}),
        "finished": bool(methods),
    }


def _commit_result(
    state: dict[str, Any],
    result: Any,
    *,
    owner: str,
) -> dict[str, Any]:
    """Commit a typed domain result through the coordinator authority."""

    from agent.harness.coordinator import _apply_handler_result

    return _apply_handler_result(state, result, owner=owner)


def _admitted_proposal_action(
    state: dict[str, Any],
    config_values: dict[str, Any],
    source_text: str,
    *,
    source_format: str = "prose",
) -> dict[str, Any]:
    from agent.harness.semantic_admission import _attach_semantic_admission_receipts

    document = {
        "actions": [{
            "type": "propose_config_values",
            "config_values": config_values,
            "unmapped_values": {},
            "conflicts": [],
            "source_format": source_format,
            "source_evidence": source_text,
            "confidence": "high",
        }],
        "semantic_units": [{
            "unit_id": "proposal-unit-1",
            "clause_id": "proposal-clause-1",
            "source_text": source_text,
            "disposition": "action",
            "action_indexes": [0],
        }],
    }
    return json.loads(
        _attach_semantic_admission_receipts(json.dumps(document), state)
    )["actions"][0]


def _admitted_field_intake_action(
    state: dict[str, Any],
    field: str,
    source_text: str,
) -> dict[str, Any]:
    from agent.harness.action_registry import (
        build_admission_transaction_hash,
        build_field_intake_admission_receipt,
    )

    action = {
        "type": "request_config_field_input",
        "config_field": field,
        "source_evidence": source_text,
        "confidence": "high",
    }
    unit = {
        "unit_id": "field-unit-1",
        "clause_id": "field-clause-1",
        "source_text": source_text,
        "disposition": "action",
        "action_indexes": [0],
    }
    thread_id = str(state.get("thread_id") or "default")
    session_id = str((state.get("session") or {}).get("id") or thread_id)
    # Product admission occurs after the graph prepares and increments the turn.
    turn_index = int(state.get("turn_index") or 0) + 1
    admission_action_id = "field-admission-1"
    transaction_hash = build_admission_transaction_hash(
        thread_id=thread_id,
        session_id=session_id,
        submitted_turn_index=turn_index,
        actions=[action],
        semantic_units=[unit],
        admission_action_ids=[admission_action_id],
    )
    action.update({
        "_admission_action_id": admission_action_id,
        "_transaction_action_ids": [admission_action_id],
        "_plan_transaction_hash": transaction_hash,
        "_semantic_admission_receipt": build_field_intake_admission_receipt(
            thread_id=thread_id,
            session_id=session_id,
            submitted_turn_index=turn_index,
            transaction_hash=transaction_hash,
            admission_action_id=admission_action_id,
            config_field=field,
            source_evidence=source_text,
            source_unit_id=unit["unit_id"],
            source_unit_text=source_text,
            source_quote=source_text,
            semantic_unit_hash="field-semantic-unit-hash",
            reviewer_evidence_hash="field-reviewer-evidence-hash",
        ),
    })
    return action


class HarnessArchitectureTest(unittest.TestCase):
    def test_pending_admission_requires_complete_turn_candidate_receipts(self) -> None:
        from agent.harness.semantic_admission import ALLOWED_ACTION_TYPES
        from agent.harness.semantic_compiler import (
            freeze_semantic_plan,
            validate_whole_plan_admission,
        )

        source = "Do not use eth0. Use eth1 instead."
        action = {
            "type": "answer_pending",
            "answer": "eth1",
            "source_evidence": source,
        }
        unit = {
            "unit_id": "unit-1",
            "clause_id": "clause-1",
            "source_text": source,
            "disposition": "action",
            "action_indexes": [0],
        }
        plan = freeze_semantic_plan(
            {"actions": [action], "semantic_units": [unit]},
            action_records=[{
                "action_id": "action-1",
                "action_index": 0,
                "action": action,
                "operation_arguments": {"answer": "eth1"},
                "unit_ids": ["unit-1"],
                "allowed_support_relations": [],
                "required_value_grounding_arguments": [],
                "exact_source_value_arguments": [],
                "pending_value_candidates": [{
                    "candidate_id": "candidate-0",
                    "argument": "answer",
                    "identity": "scalar:eth1",
                    "path": [],
                    "value": "eth1",
                }],
                "turn_pending_value_candidates": [
                    {
                        "candidate_id": "turn-candidate-0",
                        "identity": "scalar:eth0",
                        "source_unit_ids": ["unit-1"],
                        "value": "eth0",
                    },
                    {
                        "candidate_id": "turn-candidate-1",
                        "identity": "scalar:eth1",
                        "source_unit_ids": ["unit-1"],
                        "value": "eth1",
                    },
                ],
            }],
            unit_records=[{
                "unit_id": "unit-1",
                "unit_index": 0,
                "unit": unit,
                "source_text": source,
                "disposition": "action",
                "owner_action_ids": ["action-1"],
            }],
            review_context={"pending_question": {}},
        )
        payload = {
            "action_verdicts": [{
                "action_id": "action-1",
                "verdict": "admit",
                "unit_ids": ["unit-1"],
                "evidence": [{
                    "unit_id": "unit-1",
                    "quote": source,
                    "relation": "direct",
                    "support_relation": "",
                }],
                "grounded_arguments": [],
                "pending_answer_argument": "candidate-0",
                "turn_candidate_verdicts": [
                    {
                        "candidate_id": "turn-candidate-0",
                        "verdict": "not_selected",
                        "evidence_quote": "Do not use eth0.",
                        "reason": "the source explicitly rejects this value",
                    },
                    {
                        "candidate_id": "turn-candidate-1",
                        "verdict": "selected",
                        "evidence_quote": "Use eth1 instead.",
                        "reason": "the source explicitly selects this value",
                    },
                ],
                "reason": "the correction selects exactly one candidate",
            }],
            "unit_verdicts": [{
                "unit_id": "unit-1",
                "verdict": "complete",
                "owner_action_ids": ["action-1"],
                "evidence_quote": source,
                "omitted_action_type": "",
                "reason": "the pending answer preserves the correction",
            }],
            "reason": "reviewed",
        }
        admitted = validate_whole_plan_admission(
            json.dumps(payload),
            plan,
            allowed_action_types=ALLOWED_ACTION_TYPES,
        )
        self.assertTrue(admitted.valid, admitted.errors)

        forged_candidate_quote = deepcopy(payload)
        forged_candidate_quote["action_verdicts"][0][
            "turn_candidate_verdicts"
        ][1]["evidence_quote"] = "Use"
        rejected = validate_whole_plan_admission(
            json.dumps(forged_candidate_quote),
            plan,
            allowed_action_types=ALLOWED_ACTION_TYPES,
        )
        self.assertFalse(rejected.valid)
        self.assertIn(
            "turn candidate evidence is not exact",
            "; ".join(rejected.errors),
        )

        missing = deepcopy(payload)
        missing["action_verdicts"][0]["turn_candidate_verdicts"].pop()
        rejected = validate_whole_plan_admission(
            json.dumps(missing),
            plan,
            allowed_action_types=ALLOWED_ACTION_TYPES,
        )
        self.assertFalse(rejected.valid)
        self.assertIn(
            "turn candidate order or cardinality mismatch",
            "; ".join(rejected.errors),
        )

        wrong_selected = deepcopy(payload)
        wrong_selected["action_verdicts"][0]["turn_candidate_verdicts"][0]["verdict"] = "selected"
        wrong_selected["action_verdicts"][0]["turn_candidate_verdicts"][1]["verdict"] = "not_selected"
        rejected = validate_whole_plan_admission(
            json.dumps(wrong_selected),
            plan,
            allowed_action_types=ALLOWED_ACTION_TYPES,
        )
        self.assertFalse(rejected.valid)
        self.assertIn(
            "lacks one matching turn candidate",
            "; ".join(rejected.errors),
        )

    def test_evidence_collection_actions_require_their_registered_lifecycle(self) -> None:
        from agent.harness.action_registry import lifecycle_rejected_action_indexes
        from agent.harness.state import new_state

        empty = new_state("evidence-lifecycle-empty", language="en")
        actions = [
            {"type": "pause_evidence_collection", "source_evidence": "pause this setup"},
            {"type": "resume_evidence_collection", "source_evidence": "resume"},
        ]
        self.assertEqual(lifecycle_rejected_action_indexes(empty, actions), (0, 1))

        active = new_state("evidence-lifecycle-active", language="en")
        active["evidence_collection"] = {"status": "active", "lines": ["trace"]}
        self.assertEqual(lifecycle_rejected_action_indexes(active, actions), ())

        paused = new_state("evidence-lifecycle-paused", language="en")
        paused["evidence_collection"] = {"status": "paused", "lines": ["trace"]}
        self.assertEqual(lifecycle_rejected_action_indexes(paused, actions), (0,))

        ordered = [
            {"type": "pause_evidence_collection", "source_evidence": "pause"},
            {
                "type": "append_evidence_collection",
                "evidence": "more",
                "source_evidence": "more",
            },
        ]
        self.assertEqual(lifecycle_rejected_action_indexes(active, ordered), (1,))
        ordered.reverse()
        self.assertEqual(lifecycle_rejected_action_indexes(active, ordered), ())

        completing_append = {
            "type": "append_evidence_collection",
            "evidence": 'response: {"jsonrpc":"2.0","result":"0x1"}',
            "source_evidence": 'response: {"jsonrpc":"2.0","result":"0x1"}',
        }
        completing = deepcopy(active)
        completing["evidence_collection"]["lines"] = [
            'request: {"jsonrpc":"2.0","method":"eth_blockNumber"}',
        ]
        self.assertEqual(
            lifecycle_rejected_action_indexes(
                completing,
                [
                    completing_append,
                    {"type": "pause_evidence_collection", "source_evidence": "pause"},
                ],
            ),
            (1,),
        )

    def test_exact_reviewer_quote_must_bind_the_selected_rpc_value(self) -> None:
        from agent.harness.semantic_compiler import freeze_semantic_plan, validate_whole_plan_admission
        from agent.harness.semantic_admission import ALLOWED_ACTION_TYPES

        source = "Use https://first.example/rpc for reference but test https://second.example/rpc."
        action = {
            "type": "rpc_catalog_command",
            "catalog_command": "set_endpoint",
            "rpc_endpoint": "https://second.example/rpc",
            "source_evidence": source,
        }
        unit = {
            "unit_id": "unit-1",
            "clause_id": "clause-1",
            "source_text": source,
            "disposition": "action",
            "action_indexes": [0],
        }
        plan = freeze_semantic_plan(
            {"actions": [action], "semantic_units": [unit]},
            action_records=[{
                "action_id": "action-1",
                "action_index": 0,
                "action": action,
                "operation_arguments": {
                    "catalog_command": "set_endpoint",
                    "rpc_endpoint": "https://second.example/rpc",
                },
                "unit_ids": ["unit-1"],
                "allowed_support_relations": [],
                "required_value_grounding_arguments": ["rpc_endpoint"],
                "exact_source_value_arguments": ["rpc_endpoint"],
                "pending_value_candidates": [],
            }],
            unit_records=[{
                "unit_id": "unit-1",
                "unit_index": 0,
                "unit": unit,
                "source_text": source,
                "disposition": "action",
                "owner_action_ids": ["action-1"],
            }],
            review_context={"pending_question": {}},
        )
        payload = {
            "action_verdicts": [{
                "action_id": "action-1",
                "verdict": "admit",
                "unit_ids": ["unit-1"],
                "evidence": [{
                    "unit_id": "unit-1",
                    "quote": source,
                    "relation": "direct",
                    "support_relation": "",
                }],
                "grounded_arguments": [{
                    "argument_name": "rpc_endpoint",
                    "evidence_quote": "https://first.example/rpc",
                }],
                "pending_answer_argument": "",
                "turn_candidate_verdicts": [],
                "reason": "endpoint selection",
            }],
            "unit_verdicts": [{
                "unit_id": "unit-1",
                "verdict": "complete",
                "owner_action_ids": ["action-1"],
                "evidence_quote": source,
                "omitted_action_type": "",
                "reason": "complete",
            }],
            "reason": "reviewed",
        }
        rejected = validate_whole_plan_admission(
            json.dumps(payload),
            plan,
            allowed_action_types=ALLOWED_ACTION_TYPES,
        )
        self.assertFalse(rejected.valid)
        self.assertIn("does not contain its immutable value", "; ".join(rejected.errors))

        payload["action_verdicts"][0]["grounded_arguments"][0]["evidence_quote"] = (
            "test https://second.example/rpc"
        )
        accepted = validate_whole_plan_admission(
            json.dumps(payload),
            plan,
            allowed_action_types=ALLOWED_ACTION_TYPES,
        )
        self.assertTrue(accepted.valid, accepted.errors)

    def test_structured_candidate_evidence_rejects_wrong_explicit_values(self) -> None:
        from agent.harness.semantic_compiler import (
            _quote_supports_pending_candidate,
        )

        source = "eth_a=10, eth_b=90"
        self.assertFalse(_quote_supports_pending_candidate(
            source,
            {
                "identity": (
                    'rpc_weights:{"eth_a":70,"eth_b":30}'
                ),
                "value": {"eth_a": 70, "eth_b": 30},
            },
            {},
            [source],
        ))

    def test_short_option_label_does_not_match_inside_natural_language(self) -> None:
        from agent.harness.semantic_compiler import (
            _quote_supports_pending_candidate,
        )

        source = "No, reject that detected size."
        self.assertTrue(_quote_supports_pending_candidate(
            source,
            {
                "identity": 'option:"__manual__"',
                "value": "__manual__",
            },
            {
                "options": [{
                    "id": "2",
                    "label": "N",
                    "value": "__manual__",
                }],
            },
            [source],
        ))

    def test_go_back_cancels_field_reconfiguration_continuation(self) -> None:
        from agent.harness.coordinator import apply_coordinator_action
        from agent.harness.contracts import ActionProposal

        state = _state(
            control={
                "field_reconfiguration_continuation": {
                    "group": "ledger_disk",
                    "config_field": "DATA_VOL_SIZE",
                    "prerequisite_question_id": "LEDGER_DEVICE",
                }
            },
            group_history=["network"],
        )
        result = apply_coordinator_action(
            state,
            ActionProposal("back-1", "go_back", {}, "high"),
        )
        committed = _commit_result(state, result, owner="coordinator")
        self.assertNotIn(
            "field_reconfiguration_continuation",
            committed.get("control") or {},
        )
        self.assertEqual(result.completion, "completed")

    def test_legacy_custom_rpc_migration_quarantines_all_executable_source(
        self,
    ) -> None:
        from agent.harness.state import migrate_state

        base = {
            "schema_version": 12,
            "action_queue": [{
                "type": "start_custom_rpc",
                "rpc_endpoint": "https://second.example/rpc",
                "rpc_method": "eth_chainId",
                "source_evidence": "Use https://first.example/rpc with eth_accounts",
            }],
        }
        rejected = migrate_state(
            base,
            thread_id="legacy-untrusted",
            language="en",
            session_purpose="user",
        )
        self.assertEqual(rejected["action_queue"], [])
        migration_event = next(
            event
            for event in rejected["audit_events"]
            if event.get("event") == "checkpoint_legacy_inflight_quarantined"
        )
        self.assertEqual(migration_event["quarantined"]["action_queue"], 1)

        trusted = deepcopy(base)
        trusted["action_queue"][0]["source_evidence"] = (
            "Use https://second.example/rpc with eth_chainId"
        )
        migrated = migrate_state(
            trusted,
            thread_id="legacy-trusted",
            language="en",
            session_purpose="user",
        )
        self.assertEqual(migrated["action_queue"], [])
        self.assertTrue(any(
            event.get("event") == "checkpoint_legacy_inflight_quarantined"
            for event in migrated["audit_events"]
        ))

    def test_checkpoint_migration_quarantines_retired_pending_action_contracts(self) -> None:
        from agent.harness.state import migrate_state

        pending = {
            "contract_version": 1,
            "id": "legacy-custom-rpc-choice",
            "group": "workload_rpc",
            "kind": "numbered_choice",
            "field": "rpc_workload",
            "manual_input_allowed": False,
            "options": [{
                "id": "1",
                "value": "custom_rpc",
                "action": {"type": "start_custom_rpc"},
            }],
            "accepted_action_types": ["answer_pending", "start_custom_rpc"],
        }
        migrated = migrate_state(
            {
                "schema_version": 12,
                "active_group": "workload_rpc",
                "pending_question": pending,
                "action_queue": [{"type": "unknown_retired_action"}],
            },
            thread_id="legacy-pending-action",
            language="en",
            session_purpose="user",
        )

        self.assertEqual(migrated["pending_question"], {})
        self.assertEqual(migrated["active_group"], "workload_rpc")
        self.assertEqual(migrated["action_queue"], [])
        events = migrated["audit_events"]
        self.assertEqual(events[-1]["event"], "checkpoint_schema_migrated")

    def test_checkpoint_migration_quarantines_pre_v3_pending_action_contracts(self) -> None:
        from agent.harness.state import migrate_state

        pending = {
            "contract_version": 2,
            "id": "current-choice",
            "group": "opening",
            "kind": "numbered_choice",
            "field": "target_mode",
            "manual_input_allowed": False,
            "options": [{
                "id": "1",
                "value": "fake-node",
                "action": {
                    "type": "choose_target_mode",
                    "target_mode": "fake-node",
                    "target_mode_explicit": True,
                },
            }],
            "accepted_action_types": ["answer_pending", "choose_target_mode"],
        }
        migrated = migrate_state(
            {
                "schema_version": 12,
                "active_group": "opening",
                "pending_question": pending,
            },
            thread_id="current-pending-action",
            language="en",
            session_purpose="user",
        )

        self.assertEqual(migrated["pending_question"], {})
        self.assertEqual(
            migrated["checkpoint_recovery"]["status"],
            "quarantined",
        )
        self.assertTrue(any(
            event.get("event") == "checkpoint_legacy_inflight_quarantined"
            and event.get("quarantined", {}).get("pending_question") is True
            for event in migrated["audit_events"]
        ))

    def test_checkpoint_migration_quarantines_retired_resume_context_contract(self) -> None:
        from agent.harness.state import migrate_state

        stale = {
            "contract_version": 1,
            "id": "legacy-resumed-custom-rpc",
            "group": "workload_rpc",
            "kind": "numbered_choice",
            "options": [{
                "id": "1",
                "value": "custom_rpc",
                "action": {"type": "start_custom_rpc"},
            }],
        }
        migrated = migrate_state(
            {
                "schema_version": 12,
                "active_group": "opening",
                "pending_question": {
                    "contract_version": 2,
                    "id": "resume_harness_session",
                    "group": "opening",
                    "kind": "numbered_choice",
                    "field": "resume_harness_session",
                    "manual_input_allowed": False,
                    "options": [{
                        "id": "1",
                        "value": "continue",
                        "action": {"type": "answer_pending", "answer": "continue"},
                    }],
                    "accepted_action_types": ["answer_pending"],
                },
                "resume_context": {
                    "active_group": "workload_rpc",
                    "pending_question": stale,
                },
            },
            thread_id="legacy-resume-context",
            language="en",
            session_purpose="user",
        )

        self.assertEqual(migrated["pending_question"], {})
        self.assertEqual(migrated["resume_context"]["pending_question"], {})
        events = [
            row
            for row in migrated["audit_events"]
            if row.get("event") == "checkpoint_pending_actions_quarantined"
        ]
        self.assertEqual(
            {event["location"] for event in events},
            {"resume_context"},
        )
        self.assertTrue(any(
            event.get("event") == "checkpoint_legacy_inflight_quarantined"
            and event.get("quarantined", {}).get("pending_question") is True
            for event in migrated["audit_events"]
        ))
        self.assertTrue(all(
            f"contract_version {QUESTION_CONTRACT_VERSION}"
            in str(event.get("contract_error") or "")
            for event in events
        ))

    def test_semantic_planner_has_no_state_policy_repair_authority(self) -> None:
        from agent.harness import semantic_admission

        self.assertFalse(hasattr(semantic_admission, "_apply_state_plan_policy"))

    def test_generic_navigation_purpose_excludes_registered_typed_entry(self) -> None:
        from agent.harness.semantic_admission import _semantic_action_purpose

        purpose = _semantic_action_purpose(
            {"type": "change_group", "group": "endpoint_process"},
            "fallback",
        )

        self.assertIn("does not match a registered typed entry purpose", purpose)
        self.assertIn("Enter custom RPC method catalog setup", purpose)

    def test_go_back_purpose_is_bound_to_recoverable_state(self) -> None:
        from agent.harness.semantic_admission import _semantic_action_purpose

        unavailable = _semantic_action_purpose(
            {"type": "go_back"},
            "fallback",
            {"group_history": [], "interruption_stack": []},
        )
        available = _semantic_action_purpose(
            {"type": "go_back"},
            "fallback",
            {"group_history": ["network"], "interruption_stack": []},
        )

        self.assertIn("operation is unavailable", unavailable)
        self.assertIn("['network']", available)

    def test_go_back_without_recoverable_state_fails_plan_validation(self) -> None:
        from agent.harness.semantic_admission import _validate_action_document
        from agent.harness.plan_coverage import TurnClause
        from agent.harness.state import new_state

        text = "Go back to endpoint settings."
        document = json.dumps({
            "actions": [{"type": "go_back", "source_evidence": text}],
            "semantic_units": [{
                "unit_id": "unit-1",
                "clause_id": "clause-1",
                "source_text": text,
                "disposition": "action",
                "action_indexes": [0],
                "reason": "back navigation",
            }],
        })
        state = new_state("no-history", language="en")

        result = _validate_action_document(
            document,
            (TurnClause("clause-1", text),),
            state,
        )

        self.assertFalse(result.valid)
        self.assertIn("go_back has no recoverable", "; ".join(result.errors))

    def test_rpc_catalog_payload_must_exist_in_its_mapped_source_unit(self) -> None:
        from agent.harness.semantic_admission import _validate_action_document
        from agent.harness.plan_coverage import TurnClause
        from agent.harness.state import new_state

        text = "Configure the method shown in another document."
        document = json.dumps({
            "actions": [{
                "type": "rpc_catalog_command",
                "catalog_command": "set_method",
                "rpc_method": "eth_accounts",
                "source_evidence": text,
            }],
            "semantic_units": [{
                "unit_id": "unit-1",
                "clause_id": "clause-1",
                "source_text": text,
                "disposition": "action",
                "action_indexes": [0],
                "reason": "method request",
            }],
        })

        result = _validate_action_document(
            document,
            (TurnClause("clause-1", text),),
            new_state("rpc-grounding", language="en"),
        )

        self.assertFalse(result.valid)
        self.assertIn("rpc_method is not present", "; ".join(result.errors))

    def test_rpc_catalog_wire_payload_requires_exact_source_evidence(self) -> None:
        from agent.harness.semantic_admission import _validate_action_document
        from agent.harness.plan_coverage import TurnClause
        from agent.harness.state import new_state

        text = "Use the selected method from the saved document."
        document = json.dumps({
            "actions": [{
                "type": "rpc_catalog_command",
                "catalog_command": "set_method",
                "rpc_method": "eth_accounts",
                "source_evidence": text,
            }],
            "semantic_units": [{
                "unit_id": "unit-1",
                "clause_id": "clause-1",
                "source_text": text,
                "disposition": "action",
                "action_indexes": [0],
            }],
        })

        result = _validate_action_document(
            document,
            (TurnClause("clause-1", text),),
            new_state("rpc-exact-source", language="en"),
        )

        self.assertFalse(result.valid)
        self.assertIn("exact source_evidence", "; ".join(result.errors))

    def test_scalar_reconfiguration_contract_is_registry_owned(self) -> None:
        from agent.harness.action_registry import validate_action_contract
        from agent.harness.context import action_schema
        from agent.workflows.group_registry import (
            GROUP_SPEC_BY_NAME,
            reconfiguration_question_for_field,
        )

        schema = next(
            item for item in action_schema()
            if item["type"] == "request_config_field_input"
        )
        registered = {
            field
            for group in GROUP_SPEC_BY_NAME.values()
            for field, _question in group.reconfiguration_questions
        }
        self.assertEqual(set(schema["allowed_config_fields"]), registered)
        self.assertEqual(
            reconfiguration_question_for_field("NETWORK_INTERFACE"),
            "network_interface",
        )
        validate_action_contract({
            "type": "request_config_field_input",
            "config_field": "CLOUD_REGION",
            "source_evidence": "I need to change the cloud region.",
        })
        with self.assertRaisesRegex(ValueError, "registered reconfigurable field"):
            validate_action_contract({
                "type": "request_config_field_input",
                "config_field": "qps_profile",
                "source_evidence": "change QPS",
            })

    def test_normal_environment_interruption_cannot_reopen_confirmed_field(
        self,
    ) -> None:
        from agent.harness.domains.environment import (
            reconstruct_environment_question,
        )
        from agent.harness.coordinator import _reconstruct_question
        from agent.harness.state import new_state

        state = new_state("normal-interruption", language="en")
        state["confirmed_config"] = {
            "LEDGER_DEVICE": "vda",
            "DATA_VOL_TYPE": "pd-ssd",
            "DATA_VOL_SIZE": "926",
            "DATA_VOL_MAX_IOPS": "20000",
            "DATA_VOL_MAX_THROUGHPUT": "1000",
        }
        identity = {
            "group": "ledger_disk",
            "owner": "environment",
            "question_id": "DATA_VOL_TYPE",
            "field": "DATA_VOL_TYPE",
            "reason": "propose_config_values_overlay",
        }

        self.assertIsNone(reconstruct_environment_question(state, identity))
        self.assertIsNone(_reconstruct_question(state, identity))

        identity["reconfiguration_target_field"] = "DATA_VOL_TYPE"
        question = reconstruct_environment_question(state, identity)
        self.assertIsNotNone(question)
        self.assertEqual(question["id"], "DATA_VOL_TYPE")
        self.assertEqual(
            question["reconfiguration_target_field"],
            "DATA_VOL_TYPE",
        )
        resumed = _reconstruct_question(state, identity)
        self.assertIsNotNone(resumed)
        self.assertEqual(resumed["id"], "DATA_VOL_TYPE")

    def test_explicit_reconfiguration_is_not_deduplicated_as_normal_interruption(
        self,
    ) -> None:
        from agent.harness.coordinator import _push_interruption_frame
        from agent.harness.state import new_state

        state = new_state("interruption-dedup", language="en")
        normal = {
            "id": "DATA_VOL_TYPE",
            "group": "ledger_disk",
            "owner": "environment",
            "field": "DATA_VOL_TYPE",
        }
        explicit = {
            **normal,
            "reconfiguration_target_field": "DATA_VOL_TYPE",
        }

        _push_interruption_frame(state, normal, reason="explicit_navigation")
        _push_interruption_frame(state, explicit, reason="explicit_navigation")

        self.assertEqual(len(state["interruption_stack"]), 2)
        self.assertNotIn(
            "reconfiguration_target_field",
            state["interruption_stack"][0],
        )
        self.assertEqual(
            state["interruption_stack"][1]["reconfiguration_target_field"],
            "DATA_VOL_TYPE",
        )

    def test_explicit_reconfiguration_frame_expires_after_alternate_confirmation(
        self,
    ) -> None:
        from agent.harness.coordinator import (
            _pop_interruption_question,
            _push_interruption_frame,
        )
        from agent.harness.state import new_state
        from agent.harness.transitions import mark_field_confirmed

        state = new_state("interruption-fulfilled", language="en")
        state["confirmed_config"] = {"DATA_VOL_TYPE": "pd-ssd"}
        pending = {
            "id": "DATA_VOL_TYPE",
            "group": "ledger_disk",
            "owner": "environment",
            "field": "DATA_VOL_TYPE",
            "reconfiguration_target_field": "DATA_VOL_TYPE",
        }
        _push_interruption_frame(state, pending, reason="explicit_navigation")

        mark_field_confirmed(
            state,
            "DATA_VOL_TYPE",
            group="ledger_disk",
        )

        self.assertIsNone(_pop_interruption_question(state))
        self.assertEqual(state["interruption_stack"], [])

    def test_scalar_reconfiguration_opens_exact_question_without_copying_state(self) -> None:
        from tests.agent_live.graph_turn import invoke_action

        cases = (
            ("CLOUD_REGION", "provider_deployment", "CLOUD_REGION", "asia-east1"),
            ("DATA_VOL_MAX_IOPS", "ledger_disk", "DATA_VOL_MAX_IOPS", "20000"),
            ("NETWORK_INTERFACE", "network", "network_interface", "eth0"),
        )
        for field, group, question_id, current_value in cases:
            with self.subTest(field=field):
                state = _state(
                    active_group="opening",
                    confirmed_config={
                        "CLOUD_REGION": "asia-east1",
                        "CLOUD_ZONE": "asia-east1-a",
                        "MACHINE_TYPE": "n2-standard-8",
                        "LEDGER_DEVICE": "vda",
                        "DATA_VOL_TYPE": "hyperdisk-balanced",
                        "DATA_VOL_SIZE": "926",
                        "DATA_VOL_MAX_IOPS": "20000",
                        "DATA_VOL_MAX_THROUGHPUT": "1000",
                        "NETWORK_INTERFACE": "eth0",
                        "NETWORK_MAX_BANDWIDTH_GBPS": "100",
                    },
                )
                source = f"change {field}"
                action = _admitted_field_intake_action(state, field, source)
                action["action_id"] = f"edit-{field}"
                result = invoke_action(state, action, f"change {field}")
                self.assertIsNotNone(result)
                assert result is not None
                self.assertEqual(result["active_group"], group)
                self.assertEqual(result["pending_question"]["id"], question_id)
                self.assertEqual(result["pending_question"]["field"], field)
                self.assertEqual(result["confirmed_config"][field], current_value)
                self.assertNotIn(current_value, result["visible_response"][-1])

    def test_scalar_reconfiguration_respects_environment_prerequisites(self) -> None:
        from tests.agent_live.graph_turn import answer_pending, invoke_action

        state = _state(
            active_group="opening",
            confirmed_config={"has_accounts_device": False},
        )
        source = "change ACCOUNTS_VOL_MAX_IOPS"
        action = _admitted_field_intake_action(state, "ACCOUNTS_VOL_MAX_IOPS", source)
        action["action_id"] = "edit-accounts-iops"
        result = invoke_action(state, action, "change ACCOUNTS_VOL_MAX_IOPS")
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["active_group"], "accounts_disk")
        self.assertEqual(result["pending_question"]["id"], "has_accounts_device")
        self.assertEqual(
            result["pending_question"]["reconfiguration_target_field"],
            "ACCOUNTS_VOL_MAX_IOPS",
        )
        self.assertNotIn("ACCOUNTS_VOL_MAX_IOPS", result["confirmed_config"])
        resumed = answer_pending(
            result,
            "Y",
            dict(result["pending_question"]),
            selected_value=True,
        )
        for question_id, value in (
            ("ACCOUNTS_DEVICE", "vdb"),
            ("ACCOUNTS_VOL_TYPE", "ssd"),
            ("ACCOUNTS_VOL_SIZE", "1000"),
        ):
            self.assertEqual(resumed["pending_question"]["id"], question_id)
            resumed = answer_pending(
                resumed,
                value,
                dict(resumed["pending_question"]),
                manual_value=value,
            )
        self.assertEqual(resumed["active_group"], "accounts_disk")
        self.assertEqual(resumed["pending_question"]["id"], "ACCOUNTS_VOL_MAX_IOPS")
        self.assertEqual(resumed["pending_question"]["field"], "ACCOUNTS_VOL_MAX_IOPS")
        self.assertNotIn(
            "field_reconfiguration_continuation",
            resumed.get("control") or {},
        )


    def test_change_group_schema_and_validator_share_public_navigation_catalog(self) -> None:
        from agent.harness.action_registry import validate_action_contract
        from agent.harness.context import action_schema
        from agent.workflows.group_registry import USER_NAVIGABLE_GROUPS

        change_group = next(item for item in action_schema() if item["type"] == "change_group")
        self.assertEqual(change_group["allowed_groups"], list(USER_NAVIGABLE_GROUPS))
        validate_action_contract({
            "type": "change_group",
            "group": "qps_profile",
            "navigation_explicit": True,
            "source_evidence": "show the QPS profile",
        })
        with self.assertRaisesRegex(ValueError, "user-navigable"):
            validate_action_contract({
                "type": "change_group",
                "group": "job_monitoring",
                "navigation_explicit": True,
                "source_evidence": "go to job monitoring",
            })

    def test_group_schema_exposes_registered_domain_entry_actions(self) -> None:
        from agent.harness.context import group_schema

        endpoint_process = next(
            item for item in group_schema() if item["name"] == "endpoint_process"
        )
        self.assertEqual(endpoint_process["entry_actions"], [{
            "type": "rpc_catalog_command",
            "purpose": (
                "Enter custom RPC method catalog setup and collect its endpoint, "
                "method, and schema evidence."
            ),
            "fixed_arguments": {"catalog_command": "enter"},
            "value_arguments": [],
        }])
        target_mode = next(
            item for item in group_schema() if item["name"] == "target_mode"
        )
        self.assertEqual(
            target_mode["entry_actions"][0]["fixed_arguments"],
            {"target_mode_explicit": True},
        )
        self.assertEqual(
            target_mode["entry_actions"][0]["value_arguments"],
            ["target_mode"],
        )
        self.assertNotIn(
            "target_mode",
            target_mode["entry_actions"][0]["fixed_arguments"],
        )
        chain_identity = next(
            item for item in group_schema() if item["name"] == "chain_identity"
        )
        choose_chain = next(
            item
            for item in chain_identity["entry_actions"]
            if item["type"] == "choose_chain"
        )
        self.assertEqual(
            choose_chain["value_arguments"],
            ["chain_text", "chain_candidates"],
        )

    def test_entry_metadata_participates_in_action_registry_identity(self) -> None:
        from dataclasses import replace
        from unittest.mock import patch

        from agent.harness import action_registry

        original = action_registry.ACTION_SPECS
        choose_chain = next(
            spec for spec in original if spec.action_type == "choose_chain"
        )
        modified = tuple(
            replace(
                spec,
                entry_intake_value_arguments=("chain_text",),
            )
            if spec is choose_chain
            else spec
            for spec in original
        )
        baseline = action_registry.action_registry_contract_hash()
        with patch.object(action_registry, "ACTION_SPECS", modified):
            changed = action_registry.action_registry_contract_hash()
        self.assertNotEqual(baseline, changed)

    def test_semantic_operation_purposes_participate_in_registry_identity(
        self,
    ) -> None:
        from unittest.mock import patch

        from agent.harness import action_registry

        baseline = action_registry.action_registry_contract_hash()
        changed_purposes = dict(action_registry.SEMANTIC_OPERATION_PURPOSES)
        changed_purposes["consultation"] += " Changed contract."
        with patch.object(
            action_registry,
            "SEMANTIC_OPERATION_PURPOSES",
            changed_purposes,
        ):
            changed = action_registry.action_registry_contract_hash()
        self.assertNotEqual(baseline, changed)

    def test_chain_rpc_invalidations_commit_cross_domain_state_once_at_coordinator(self) -> None:
        from agent.harness.contracts import HandlerResult, StateDelta
        from agent.harness.transitions import (
            invalidate_for_chain_change,
            invalidate_for_rpc_mode_change,
            invalidate_for_target_mode,
        )

        execution_state = {
            "plan": {"status": "ready"},
            "plan_file": "/tmp/plan.json",
            "preflight": {"passed": True},
            "smoke": {"passed": True},
            "final_benchmark": {"status": "completed"},
            "job": {"job_id": "job-old", "status": "completed"},
        }
        cases = (
            (
                "chain_change",
                lambda state: invalidate_for_chain_change(state, new_chain="ethereum"),
                {},
            ),
            (
                "target_mode_change",
                lambda state: (
                    state.update(target_mode="sync-observe", workflow_mode="sync_observe"),
                    invalidate_for_target_mode(state, previous_mode="real-node"),
                ),
                {"qps_profile": {}},
            ),
            (
                "rpc_mode_change",
                lambda state: (
                    invalidate_for_rpc_mode_change(state),
                    state.update(rpc_mode="mixed"),
                ),
                {},
            ),
        )
        for name, mutate, expected in cases:
            with self.subTest(case=name):
                original = _state(
                    target_mode="real-node",
                    workflow_mode="rpc_benchmark",
                    chain_identity={"canonical": "bsc", "status": "confirmed"},
                    rpc_mode="single",
                    workload={"confirmed": True, "methods": ["eth_blockNumber"]},
                    qps_profile={"mode": "quick"},
                    sync_observe={"source": "existing_local_node"},
                    confirmed_config={"CLOUD_REGION": "asia-east1"},
                    interruption_stack=[{"group": "network", "reason": "user_jump"}],
                    **execution_state,
                )
                changed = deepcopy(original)
                mutate(changed)

                # Domain transitions declare cross-domain invalidations without
                # writing execution/performance/sync-owned roots themselves.
                for key, value in execution_state.items():
                    self.assertEqual(changed[key], value)
                if name == "target_mode_change":
                    self.assertEqual(changed["qps_profile"], {"mode": "quick"})
                    self.assertEqual(changed["sync_observe"], {"source": "existing_local_node"})

                committed = _commit_result(
                    original,
                    HandlerResult(
                        delta=StateDelta.between(original, changed),
                        invalidated_groups=tuple(sorted(
                            set(changed.get("invalidated_groups") or [])
                            - set(original.get("invalidated_groups") or [])
                        )),
                    ),
                    owner="chain_rpc",
                )

                for key in execution_state:
                    self.assertIn(committed.get(key), ({}, ""))
                for key, value in expected.items():
                    self.assertEqual(committed.get(key), value)
                self.assertEqual(committed["confirmed_config"]["CLOUD_REGION"], "asia-east1")
                self.assertEqual(committed["interruption_stack"], [{"group": "network", "reason": "user_jump"}])

    def test_endpoint_and_qps_changes_use_registry_invalidation_at_commit_boundary(self) -> None:
        from agent.harness.contracts import ActionProposal
        from agent.harness.domains.chain_rpc import (
            apply_chain_rpc_answer,
            question_for_chain_rpc,
        )
        from agent.harness.domains.performance import apply_performance_action

        execution = {
            "plan": {"status": "ready"},
            "plan_file": "/tmp/old-plan.json",
            "preflight": {"passed": True},
            "smoke": {"passed": True},
            "final_benchmark": {"status": "completed"},
            "job": {"job_id": "job-old", "status": "completed"},
        }
        endpoint_state = _state(
            target_mode="real-node",
            workflow_mode="rpc_benchmark",
            chain_identity={"canonical": "bsc", "status": "confirmed", "adapter_family": "jsonrpc"},
            confirmed_config={
                "CLOUD_REGION": "asia-east1",
                "LOCAL_RPC_URL": "http://old.invalid",
            },
            **execution,
        )
        endpoint_question = question_for_chain_rpc(
            endpoint_state,
            "endpoint_process",
        )
        self.assertEqual(
            endpoint_question["domain_context"],
            {
                "contract_type": "rpc_endpoint",
                "endpoint_role": "final_benchmark",
                "rpc_case": "runtime",
                "config_field": "LOCAL_RPC_URL",
            },
        )
        from agent.harness.domains.rpc_receipts import exact_value_hash

        endpoint_state["current_action"] = {
            "action_id": "endpoint-change",
        }
        endpoint_state["turn_context"] = {
            "control_receipts": [],
            "admitted_actions": [{
                "action_id": "endpoint-change",
                "argument_value_hashes": {
                    "selected_value": exact_value_hash(
                        "http://new.invalid"
                    ),
                },
            }],
        }
        probe = {"ready": True, "status": "ready", "evidence_file": "/tmp/probe.json"}
        with patch("agent.harness.domains.rpc_endpoint.validate_rpc_endpoint", return_value=probe):
            endpoint_result = apply_chain_rpc_answer(
                endpoint_state,
                endpoint_question,
                "http://new.invalid",
                "http://new.invalid",
            )
        endpoint_committed = _commit_result(endpoint_state, endpoint_result, owner="chain_rpc")
        self.assertEqual(endpoint_committed["confirmed_config"]["LOCAL_RPC_URL"], "http://new.invalid")
        self.assertEqual(endpoint_committed["confirmed_config"]["CLOUD_REGION"], "asia-east1")
        for key in execution:
            self.assertIn(endpoint_committed.get(key), ({}, ""))

        qps_state = _state(
            target_mode="real-node",
            workflow_mode="rpc_benchmark",
            chain_identity={"canonical": "bsc", "status": "confirmed"},
            qps_profile={"mode": "standard", "confirmed": True},
            confirmed_config={"CLOUD_REGION": "asia-east1"},
            **execution,
        )
        qps_result = apply_performance_action(
            qps_state,
            ActionProposal("qps-change", "set_qps_mode", {"qps_mode": "quick"}, "high"),
        )
        qps_committed = _commit_result(qps_state, qps_result, owner="performance")
        self.assertEqual(qps_committed["qps_profile"]["mode"], "quick")
        self.assertEqual(qps_committed["confirmed_config"]["CLOUD_REGION"], "asia-east1")
        for key in execution:
            self.assertIn(qps_committed.get(key), ({}, ""))

    def test_config_proposal_contract_accepts_explicit_source_evidence(self) -> None:
        from agent.harness.action_registry import validate_action_contract

        validate_action_contract({
            "type": "propose_config_values",
            "config_values": {"CLOUD_REGION": "asia-east1"},
            "source_format": "json",
            "source_evidence": '{"CLOUD_REGION":"asia-east1"}',
        })

    def test_config_proposal_contract_accepts_natural_language_source_format(self) -> None:
        from agent.harness.action_registry import validate_action_contract

        action = validate_action_contract({
            "type": "propose_config_values",
            "config_values": {"CLOUD_REGION": "asia-east1"},
            "source_format": "prose",
            "source_evidence": "Set CLOUD_REGION to asia-east1.",
        })

        self.assertEqual(action["source_format"], "prose")

    def test_rejected_semantic_plan_cannot_commit_structured_config_side_channel(self) -> None:
        from agent.harness.coordinator import _consume_planner_queue

        text = 'use BNB fake-node and {"CLOUD_REGION":"asia-east1"}'
        state = _state(
            last_user_input=text,
            turn_context={"text": text},
        )
        rejected = {
            "actions": [{
                "type": "clarify_unresolved",
                "clauses": [text],
                "confidence": "high",
            }],
            "reason": "plan coverage rejected partial mapping",
        }

        result = _consume_planner_queue(state, rejected)

        self.assertEqual(
            [item.get("type") for item in result.get("proposed_actions") or []],
            ["clarify_unresolved"],
        )

    def test_semantic_planner_receipt_projects_duplicate_action_types_once(
        self,
    ) -> None:
        from agent.harness.control_receipts import (
            validate_persisted_domain_control_receipt,
        )
        from agent.harness.coordinator import _consume_planner_queue

        text = "What can you do, and how should I start?"
        state = _state(
            last_user_input=text,
            turn_context={"text": text},
        )
        queue = {
            "actions": [
                {
                    "type": "clarify_unresolved",
                    "clauses": ["What can you do?"],
                    "confidence": "high",
                },
                {
                    "type": "clarify_unresolved",
                    "clauses": ["How should I start?"],
                    "confidence": "high",
                },
            ],
            "semantic_units": [
                {"unit_id": "unit-1", "disposition": "action"},
                {"unit_id": "unit-2", "disposition": "action"},
            ],
            "planner_metrics": {"unit_count": 2},
            "reason": "two units share one action type",
        }

        result = _consume_planner_queue(state, queue)

        receipt = next(
            item
            for item in result["turn_context"]["control_receipts"]
            if item.get("receipt_type") == "semantic_planner"
        )
        self.assertEqual(
            receipt["planned_action_types"],
            ["clarify_unresolved"],
        )
        valid, reason = validate_persisted_domain_control_receipt(
            receipt,
            turn_index=int(result.get("turn_index") or 0),
        )
        self.assertTrue(valid, reason)

    def test_clarification_is_a_whole_turn_transaction_barrier(self) -> None:
        from agent.harness.action_registry import validate_action_transaction_contract
        from agent.harness.admission import validate_action_plan

        with self.assertRaisesRegex(ValueError, "whole-turn transaction barrier"):
            validate_action_transaction_contract([
                {
                    "type": "clarify_unresolved",
                    "clauses": ["Clarify the remaining request."],
                },
                {
                    "type": "answer_pending",
                    "answer": "http://fake-node:19000",
                    "source_evidence": "http://fake-node:19000",
                },
            ])

        validate_action_transaction_contract([
            {
                "type": "clarify_unresolved",
                "clauses": ["Clarify the first unresolved request."],
            },
            {
                "type": "clarify_unresolved",
                "clauses": ["Clarify the second unresolved request."],
            },
        ])

        result = validate_action_plan(_state(), [
            {
                "type": "clarify_unresolved",
                "clauses": ["Clarify the unresolved request."],
            },
            {
                "type": "answer_pending",
                "answer": "http://fake-node:19000",
                "source_evidence": "http://fake-node:19000",
            },
        ])
        self.assertEqual(result.status, "rejected")
        self.assertEqual(
            [item.code for item in result.rejections],
            ["transaction_invalid"],
        )

    def test_durable_config_proposals_merge_independent_fields_across_turns(self) -> None:
        from agent.harness.coordinator import _merge_durable_action_queue

        first_state = _state(turn_index=7)
        second_state = _state(turn_index=8)
        first_values = {
                "CLOUD_REGION": "asia-east1",
                "CLOUD_ZONE": "asia-east1-c",
                "MACHINE_TYPE": "n2-standard-16",
                "NETWORK_MAX_BANDWIDTH_GBPS": 100,
        }
        first_source = json.dumps(first_values)
        second_values = {"LOCAL_RPC_URL": "http://geth-dev:8545"}
        second_source = json.dumps(second_values)
        existing_action = _admitted_proposal_action(
            first_state, first_values, first_source, source_format="json"
        )
        existing_action["_origin_text"] = first_source
        incoming_action = _admitted_proposal_action(
            second_state, second_values, second_source, source_format="json"
        )
        incoming_action["_origin_text"] = second_source

        merged = _merge_durable_action_queue([existing_action], [incoming_action])

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["config_values"], {
            "CLOUD_REGION": "asia-east1",
            "CLOUD_ZONE": "asia-east1-c",
            "MACHINE_TYPE": "n2-standard-16",
            "NETWORK_MAX_BANDWIDTH_GBPS": "100",
            "LOCAL_RPC_URL": "http://geth-dev:8545",
        })
        self.assertEqual(
            set(merged[0]["_proposal_field_receipts"]),
            set(merged[0]["config_values"]),
        )
        self.assertEqual(len(merged[0]["_proposal_transaction_hashes"]), 2)
        self.assertEqual(merged[0]["_merged_origin_texts"], [first_source, second_source])

    def test_durable_scalar_supersession_does_not_inherit_old_metadata(self) -> None:
        from agent.harness.coordinator import _merge_durable_action_queue

        merged = _merge_durable_action_queue(
            [{
                "type": "set_qps_mode",
                "qps_mode": "quick",
                "source_evidence": "old quick",
                "_plan_transaction_hash": "old-transaction",
            }],
            [{
                "type": "set_qps_mode",
                "qps_mode": "standard",
                "source_evidence": "new standard",
            }],
        )

        self.assertEqual(merged, [{
            "type": "set_qps_mode",
            "qps_mode": "standard",
            "source_evidence": "new standard",
        }])

    def test_upstream_mutation_rejects_answer_bound_to_stale_pending_group(self) -> None:
        from agent.harness.admission import validate_action_plan

        state = _state(
            target_mode="fake-node",
            workflow_mode="rpc_benchmark",
            active_group="endpoint_process",
            pending_question={
                "id": "new_chain_endpoint",
                "group": "endpoint_process",
                "kind": "url",
                "field": "new_chain_endpoint",
                "manual_input_allowed": True,
                "accepted_action_types": ["answer_pending", "rpc_catalog_command"],
                "options": [],
            },
            last_user_input=(
                "Switch to ethereum real-node and use http://geth-dev:8545; "
                "keep the environment values."
            ),
        )
        result = validate_action_plan(state, [
            {
                "type": "answer_pending",
                "answer": "http://geth-dev:8545",
                "source_evidence": "use http://geth-dev:8545",
                "confidence": "high",
            },
            {
                "type": "choose_target_mode",
                "target_mode": "real-node",
                "target_mode_explicit": True,
                "source_evidence": "real-node",
                "confidence": "high",
            },
            {
                "type": "choose_chain",
                "chain_text": "ethereum",
                "source_evidence": "ethereum",
                "confidence": "high",
            },
            {
                "type": "propose_config_values",
                "config_values": {"LOCAL_RPC_URL": "http://geth-dev:8545"},
                "unmapped_values": [],
                "conflicts": [],
                "confidence": "high",
            },
        ])

        self.assertEqual(result.status, "rejected")
        self.assertFalse(result.actions)
        self.assertTrue(result.rejections)

    def test_workload_consultation_is_specific_and_non_mutating(self) -> None:
        from agent.harness.domains.orientation import consultation_fragment
        from agent.harness.response_catalog import render_fragment

        state = _state(
            chain_identity={"canonical": "bsc", "status": "confirmed"},
            rpc_mode="mixed",
        )
        response = render_fragment(
            consultation_fragment(state, {"topic": "workload_config"}),
            "en",
        ).text

        self.assertIn("eth_getBalance=25", response)
        self.assertNotIn("Current state:", response)
        self.assertFalse((state.get("workload") or {}).get("confirmed"))

    def test_twenty_groups_have_exactly_one_of_eight_domain_owners(self) -> None:
        from agent.harness.domains.registry import DOMAIN_GROUPS, GROUP_OWNER
        from agent.harness.state import DEFAULT_GROUP_ORDER

        expected_groups = {
            "opening",
            "target_mode",
            "chain_identity",
            "provider_deployment",
            "ledger_disk",
            "accounts_disk",
            "network",
            "endpoint_process",
            "chain_auxiliary_endpoints",
            "workload_rpc",
            "target_samples_fixtures",
            "qps_profile",
            "sync_observe",
            "observability",
            "advanced_tuning",
            "preflight_smoke_execution",
            "job_monitoring",
            "failure_recovery",
            "error_evidence_analysis",
            "report_artifact_analysis",
        }
        declared = [group for groups in DOMAIN_GROUPS.values() for group in groups]

        self.assertEqual(len(DOMAIN_GROUPS), 8)
        self.assertEqual(len(declared), 20)
        self.assertEqual(len(declared), len(set(declared)), "a group has more than one owner")
        self.assertEqual(set(declared), expected_groups)
        self.assertEqual(set(DEFAULT_GROUP_ORDER), expected_groups)
        self.assertEqual(set(GROUP_OWNER), expected_groups)

    def test_domain_modules_do_not_import_the_coordinator(self) -> None:
        offenders: dict[str, list[str]] = {}
        for path in sorted(DOMAIN_ROOT.rglob("*.py")):
            forbidden = sorted(
                module
                for module in _imported_modules(path)
                if module in {"agent.harness.coordinator", "harness.coordinator"}
                or module.endswith(".harness.coordinator")
            )
            if forbidden:
                offenders[str(path.relative_to(REPO_ROOT))] = forbidden
        self.assertEqual(offenders, {})

    def test_terminal_repl_does_not_import_domain_routing(self) -> None:
        repl = REPO_ROOT / "agent" / "terminal" / "repl.py"
        forbidden_roots = (
            "agent.harness.domains",
            "agent.harness.routing",
            "harness.domains",
            "harness.routing",
        )
        offenders = sorted(
            module
            for module in _imported_modules(repl)
            if any(module == root or module.startswith(root + ".") for root in forbidden_roots)
        )
        self.assertEqual(offenders, [])

    def test_legacy_group_owner_is_absent(self) -> None:
        self.assertFalse((REPO_ROOT / "agent" / "harness" / "groups.py").exists())

    def test_domain_modules_import_under_canonical_package_identity(self) -> None:
        modules = (
            "chain_rpc",
            "chain_rpc_questions",
            "chain_rpc_support",
            "chain_identity",
            "rpc_endpoint",
            "rpc_workload",
            "chain_handoff",
            "recovery",
        )
        script = "\n".join(f"import agent.harness.domains.{name}" for name in modules)
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_harness_has_no_script_style_import_fallbacks(self) -> None:
        offenders: list[str] = []
        harness_root = REPO_ROOT / "agent" / "harness"
        for path in sorted(harness_root.rglob("*.py")):
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
            import_fallback = any(
                isinstance(node, ast.Try)
                and any(
                    isinstance(child, (ast.Import, ast.ImportFrom))
                    for statement in node.body
                    for child in ast.walk(statement)
                )
                and bool(node.handlers)
                for node in ast.walk(tree)
            )
            if (
                "except ImportError" in source
                or "except ModuleNotFoundError" in source
                or import_fallback
            ):
                offenders.append(str(path.relative_to(REPO_ROOT)))
        self.assertEqual(offenders, [])

    def test_harness_exposes_only_checkpointed_semantic_planner_stages(self) -> None:
        import importlib.util

        from agent.harness import hierarchical_planner, semantic_admission

        self.assertTrue(callable(hierarchical_planner.begin_semantic_partition))
        self.assertTrue(callable(hierarchical_planner.compile_next_owner))
        self.assertTrue(callable(hierarchical_planner.review_semantic_plan))
        self.assertFalse(
            hasattr(hierarchical_planner, "resolve_product_action_queue")
        )
        self.assertIsNone(importlib.util.find_spec("agent.harness.intent"))
        self.assertFalse(hasattr(semantic_admission, "resolve_action_queue"))
        self.assertFalse(
            hasattr(
                semantic_admission,
                "adjudicate_active_pending_contract",
            )
        )

    def test_static_call_graph_rejects_retired_planner_authorities(
        self,
    ) -> None:
        import ast

        harness_root = REPO_ROOT / "agent" / "harness"
        planner_definitions: list[tuple[str, str]] = []
        retired_definitions: list[tuple[str, str]] = []
        forbidden_imports: list[tuple[str, str]] = []
        retired_names = {
            "resolve_action_queue",
            "adjudicate_active_pending_contract",
            "_focused_pending_request_payload",
            "_pending_contract_adjudication_prompt",
            "_action_queue_prompt",
            "_action_queue_payload",
        }
        planner_modules = {
            "agent.harness.hierarchical_planner",
            "agent.harness.semantic_admission",
        }
        for path in sorted(harness_root.rglob("*.py")):
            relative = str(path.relative_to(REPO_ROOT))
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(
                    node,
                    (ast.FunctionDef, ast.AsyncFunctionDef),
                ):
                    if node.name in {
                        "begin_semantic_partition",
                        "compile_next_owner",
                        "review_semantic_plan",
                    }:
                        planner_definitions.append((relative, node.name))
                    if node.name in retired_names:
                        retired_definitions.append((relative, node.name))
                if not isinstance(node, ast.ImportFrom):
                    continue
                module = str(node.module or "")
                if (
                    (
                        module in planner_modules
                        or any(
                            module.endswith(name.rsplit(".", 1)[-1])
                            for name in planner_modules
                        )
                    )
                    and (
                        "/terminal/" in f"/{relative}"
                        or "/domains/" in f"/{relative}"
                    )
                ):
                    forbidden_imports.append((relative, module))

        self.assertEqual(
            planner_definitions,
            [
                ("agent/harness/hierarchical_planner.py", "begin_semantic_partition"),
                ("agent/harness/hierarchical_planner.py", "compile_next_owner"),
                ("agent/harness/hierarchical_planner.py", "review_semantic_plan"),
            ],
        )
        self.assertEqual(retired_definitions, [])
        self.assertEqual(forbidden_imports, [])

    def test_coordinator_cannot_bypass_the_compiled_product_graph(self) -> None:
        from agent.harness import coordinator

        self.assertFalse(hasattr(coordinator, "process_turn"))
        self.assertFalse(hasattr(coordinator, "_process_turn"))
        self.assertFalse(hasattr(coordinator, "_route_free_text"))
        self.assertFalse(hasattr(coordinator, "_action_answers_pending_contract"))

    def test_graph_runtime_exposes_only_typed_state_mutations(self) -> None:
        from agent.harness.graph import AnyChainGraphRuntime

        self.assertFalse(hasattr(AnyChainGraphRuntime, "update"))
        self.assertTrue(callable(AnyChainGraphRuntime.reset))
        self.assertTrue(callable(AnyChainGraphRuntime.prepare_resume_offer))
        self.assertTrue(callable(AnyChainGraphRuntime.clear_evidence_collection))

    def test_python_cache_artifacts_are_not_tracked(self) -> None:
        result = subprocess.run(
            ["git", "-c", "safe.directory=*", "ls-files"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        tracked = [
            path
            for path in result.stdout.splitlines()
            if "__pycache__" in Path(path).parts or path.endswith((".pyc", ".pyo"))
        ]
        self.assertEqual(tracked, [])

    def test_every_registered_action_owner_has_executable_dispatch_coverage(self) -> None:
        from agent.harness.action_registry import ACTION_SPECS
        from agent.harness.contracts import ActionProposal, HandlerResult
        from agent.harness.coordinator import COORDINATOR_RUNTIME
        from agent.harness.domains.runtime import DOMAIN_RUNTIME

        registered_owners = {spec.owner for spec in ACTION_SPECS}
        runtimes = {**DOMAIN_RUNTIME, "coordinator": COORDINATOR_RUNTIME}
        dispatch_owners = set(runtimes)
        owner_mismatch = {
            "missing": sorted(registered_owners - dispatch_owners),
            "extra": sorted(dispatch_owners - registered_owners),
        }
        self.assertEqual(
            owner_mismatch,
            {"missing": [], "extra": []},
            "registered action owners need public owner dispatchers",
        )

        arguments = {
            "answer_opening_question": {"topic": "identity"},
            "choose_target_mode": {"target_mode": "fake-node"},
            "choose_chain": {"chain_text": "bsc"},
            "change_group": {"group": "opening"},
            "set_rpc_mode": {"rpc_mode": "single"},
            "set_qps_mode": {"qps_mode": "quick"},
            "set_qps_override": {
                "qps_overrides": {"INITIAL_QPS": 1, "MAX_QPS": 2, "QPS_STEP": 1, "DURATION": 1}
            },
            "set_observability": {"observability_mode": "disabled"},
            "rpc_catalog_command": {"catalog_command": "enter"},
            "rpc_workload_command": {"workload_scope": "single_replace"},
            "set_sync_observe_source": {"sync_observe_source": "existing_local_node"},
            "set_accounts_presence": {"has_accounts_device": False},
            "propose_config_values": {"config_values": {"CLOUD_REGION": "test-region"}},
            "analyze_evidence": {"evidence": "error evidence"},
            "analyze_report": {"subject": "latest job"},
            "answer_pending": {"answer": "value"},
            "unknown": {"reason": "unresolved"},
        }
        with patch(
            "agent.harness.domains.execution_runtime.execution_service.execute"
        ), patch(
            "agent.harness.domains.analysis.analyze_evidence_with_model",
            return_value="reviewed deterministic evidence analysis",
        ):
            for spec in ACTION_SPECS:
                with self.subTest(action_type=spec.action_type, owner=spec.owner):
                    handler = runtimes[spec.owner].apply_action
                    state = _state(
                        workflow_mode="sync_observe",
                        rpc_mode="mixed",
                        chain_identity={"canonical": "bsc", "status": "confirmed"},
                        qps_profile={"mode": "quick"},
                    )
                    result = handler(
                        state,
                        ActionProposal(
                            action_id=f"architecture:{spec.action_type}",
                            action_type=spec.action_type,
                            arguments=arguments.get(spec.action_type, {}),
                            confidence="high",
                        ),
                    )
                    self.assertIsInstance(result, HandlerResult)
                    if result.blocker is not None:
                        self.assertFalse(
                            result.blocker.code.endswith(".unsupported_action"),
                            (
                                f"{spec.action_type} routed to {spec.owner} but "
                                f"returned {result.blocker.code}"
                            ),
                        )

    def test_langgraph_has_explicit_control_plane_nodes(self) -> None:
        from agent.harness import graph as graph_module

        source = (REPO_ROOT / "agent/harness/graph.py").read_text(encoding="utf-8")
        for node in (
            "prepare",
            "adjudicate",
            "partition",
            "compile_owner",
            "review_plan",
            "admit",
            "select_action",
            "commit_action",
            "invoke_effect",
            "perform_effect",
            "commit_receipt",
            "fallback",
            "compose",
            "validate",
        ):
            self.assertIn(f'graph.add_node("{node}"', source)
        self.assertNotIn('graph.add_node("turn"', source)
        self.assertFalse(hasattr(graph_module, "process_turn"))

    def test_hierarchical_planner_contract_keeps_stage_authority_separate(
        self,
    ) -> None:
        from agent.harness.hierarchical_planner import (
            _stage_a_prompt,
            _stage_b_prompt,
        )

        stage_a = _stage_a_prompt()
        stage_b = _stage_b_prompt("orientation")
        self.assertIn("Do not choose product actions", stage_a)
        self.assertIn("every sibling demand separately", stage_a)
        self.assertIn("do not classify intent", stage_a)
        self.assertIn("mark the unit unresolved", stage_a)
        self.assertIn("Use only actions and arguments in owner_action_schema", stage_b)
        self.assertIn("owned by another domain", stage_b)
        self.assertIn("mark that binding unresolved", stage_b)

    def test_group_specs_own_dependencies_and_invalidation_metadata(self) -> None:
        from agent.workflows.group_registry import GROUP_SPEC_BY_NAME

        self.assertNotIn("target_mode", GROUP_SPEC_BY_NAME["chain_identity"].depends_on)
        self.assertIn("target_mode", GROUP_SPEC_BY_NAME["endpoint_process"].depends_on)
        self.assertIn("workload_rpc", GROUP_SPEC_BY_NAME["chain_identity"].invalidates)
        self.assertIn("preflight_smoke_execution", GROUP_SPEC_BY_NAME["qps_profile"].invalidates)

    def test_rpc_catalog_semantic_purpose_describes_the_selected_operation(self) -> None:
        from agent.harness.semantic_admission import _semantic_action_purpose

        purpose = _semantic_action_purpose(
            {
                "type": "rpc_catalog_command",
                "catalog_command": "append_evidence",
            },
            "Apply exactly one catalog transition.",
        )

        self.assertIn("parameter", purpose)
        self.assertIn("response", purpose)
        self.assertNotIn("transition", purpose)

    def test_unresolved_known_chain_proposal_rebuilds_after_group_barrier(self) -> None:
        from agent.harness.domains.chain_rpc import question_for_chain_rpc
        from agent.harness.questions import render_question

        state = _state(
            active_group="chain_identity",
            target_mode="real-node",
            chain_identity={
                "raw": "LocalEvmDemo",
                "canonical": "LocalEvmDemo",
                "proposed_known_chain": "ethereum",
                "status": "needs_known_chain_confirmation",
                "case": "known_candidate",
                "llm_resolution": {"canonical_chain_name": "ethereum"},
            },
        )

        question = question_for_chain_rpc(state, "chain_identity")

        self.assertEqual(question["id"], "unknown_chain_identity_confirm")
        rendered = render_question(question, "en")
        self.assertIn("LocalEvmDemo", rendered)
        self.assertIn("ethereum", rendered)

    def test_structured_configuration_has_no_pre_planner_admission_path(self) -> None:
        from agent.harness.coordinator import adjudicate_turn_step

        inputs = (
            "CLOUD_PROVIDER=gcp\nCLOUD_REGION=us-central1\n"
            "CLOUD_ZONE=us-central1-a\nowner_ticket=INC-4821",
            "CLOUD_REGION=us-central1",
        )
        for text in inputs:
            with self.subTest(text=text):
                state = _state()
                state["turn_context"] = {"kind": "free_text", "text": text}
                state["proposed_actions"] = []

                result = adjudicate_turn_step(state)

                self.assertEqual(result["control"]["phase"], "plan")
                self.assertEqual(result["control"]["reason"], "semantic_input")
                self.assertEqual(result["proposed_actions"], [])


class HarnessQuestionContractTest(unittest.TestCase):
    def test_executable_qps_scenarios_include_rpc_workflow_prerequisites(self) -> None:
        from tests.agent_live.harness_contract_scenarios import question_scenarios

        qps_scenarios = {
            scenario.scenario_id: scenario
            for scenario in question_scenarios("en")
            if scenario.scenario_id in {"qps_confirm", "qps_adjust", "qps_adjust_value"}
        }

        self.assertEqual(set(qps_scenarios), {"qps_confirm", "qps_adjust", "qps_adjust_value"})
        for scenario in qps_scenarios.values():
            self.assertEqual(scenario.seed_state.get("target_mode"), "fake-node")
            self.assertEqual(scenario.seed_state.get("workflow_mode"), "rpc_benchmark")


    def test_scalar_contract_owns_overlimit_tokens_but_not_prose_detours(self) -> None:
        from agent.harness.questions import answer_fits_pending, manual_literal_violation

        question = {
            "id": "RPC_API_KEY",
            "group": "chain_auxiliary_endpoints",
            "kind": "manual_value",
            "field": "RPC_API_KEY",
            "manual_input_allowed": True,
            "validation": {"value_type": "scalar_token", "max_length": 8},
        }

        self.assertTrue(answer_fits_pending("x" * 9, question))
        self.assertEqual(
            manual_literal_violation("x" * 9, question),
            {"code": "max_length", "max_length": 8},
        )
        self.assertFalse(answer_fits_pending("please explain this field", question))
        self.assertEqual(manual_literal_violation("please explain this field", question), {})

    def test_overlimit_scalar_routes_semantic_planning_without_mutating_field(self) -> None:
        from agent.harness.domains.chain_rpc import question_for_chain_rpc
        from agent.harness.state import new_state
        from tests.agent_live.graph_turn import invoke_product_graph_turn

        state = new_state("overlimit-scalar", language="en", session_purpose="coverage")
        state["chain_identity"] = {"canonical": "litecoin", "status": "confirmed"}
        state["active_group"] = "chain_auxiliary_endpoints"
        state["pending_question"] = question_for_chain_rpc(
            state,
            "chain_auxiliary_endpoints",
        )
        state["last_user_input"] = "x" * 181

        with patch(
            "tests.agent_live.graph_turn.TEST_SEMANTIC_PLANNER",
            return_value={"actions": []},
        ) as resolver:
            result = invoke_product_graph_turn(state)

        resolver.assert_called_once()
        self.assertEqual(result["pending_question"]["id"], "RPC_API_KEY")
        self.assertNotIn("RPC_API_KEY", result.get("confirmed_config") or {})
        self.assertNotIn(
            "clarify_unresolved",
            [item.get("type") for item in result.get("proposed_actions") or []],
        )

    def test_manual_numeric_choice_rejects_invalid_replacement_without_planning(self) -> None:
        from agent.harness.questions import (
            choice_question,
            manual_literal_violation,
            question_text,
        )

        question = choice_question(
            "ledger_disk",
            "DATA_VOL_SIZE",
            question_text(
                "question.environment.detected_disk_size.prompt",
                field="DATA_VOL_SIZE",
                value="926",
            ),
            owner="environment",
            field="DATA_VOL_SIZE",
            kind="yes_no",
            manual_input_allowed=True,
            validation={"value_type": "positive_number"},
            options=[
                {
                    "label": question_text("question.common.option.yes"),
                    "value": "926",
                },
                {
                    "label": question_text("question.common.option.no"),
                    "value": "__manual__",
                },
            ],
        )

        self.assertEqual(manual_literal_violation("not-a-number", question), {"code": "positive_number"})
        self.assertEqual(manual_literal_violation("0", question), {"code": "positive_number"})
        self.assertEqual(manual_literal_violation("Y", question), {})
        self.assertEqual(manual_literal_violation("N", question), {})

    def test_positive_integer_contract_owns_invalid_scalars_but_not_prose_detours(self) -> None:
        from agent.harness.questions import answer_fits_pending

        question = {
            "id": "sync_observe_duration_seconds",
            "group": "sync_observe",
            "kind": "positive_integer",
            "field": "sync_observe_duration_seconds",
            "manual_input_allowed": True,
            "validation": {"value_type": "positive_integer"},
        }

        for value in ("60", "0", "-1", "1.5", "unknown"):
            self.assertTrue(answer_fits_pending(value, question), value)
        self.assertFalse(answer_fits_pending("take me back to QPS settings", question))

    def test_rpc_method_question_routes_prose_with_embedded_params_to_planner(self) -> None:
        from agent.harness.questions import answer_fits_pending

        method_question = {
            "id": "new_chain_method",
            "group": "endpoint_process",
            "kind": "manual_value",
            "field": "new_chain_method",
            "manual_input_allowed": True,
            "validation": {"input_mode": "rpc_method_or_schema_evidence"},
        }
        text = (
            'Use eth_getBalance with params '
            '["0x0000000000000000000000000000000000000000", "latest"]. '
            "The first parameter is an address."
        )

        self.assertFalse(answer_fits_pending(text, method_question))

    def test_rpc_schema_question_accepts_standalone_params_but_not_arbitrary_json(self) -> None:
        from agent.harness.questions import answer_fits_pending

        evidence_question = {
            "id": "new_chain_schema_evidence",
            "group": "endpoint_process",
            "kind": "evidence",
            "field": "new_chain_schema_evidence",
            "manual_input_allowed": True,
        }

        self.assertTrue(answer_fits_pending('["0xabc", "latest"]', evidence_question))
        self.assertFalse(answer_fits_pending('{"ticket": "INC-1"}', evidence_question))

    def test_evidence_contract_requires_semantic_ownership_for_multiline_prose(self) -> None:
        from agent.harness.questions import (
            answer_fits_pending,
            manual_question,
            question_text,
            value_satisfies_pending_contract,
        )

        question = manual_question(
            "chain_identity",
            "protocol_evidence",
            question_text("question.chain_rpc.case3_protocol_evidence.prompt"),
            owner="chain_rpc",
            field="protocol_evidence",
            kind="evidence",
        )
        evidence = (
            "The protocol uses a peer-to-peer RPC transport.\n"
            "Its request schema is not compatible with the supported adapters."
        )

        self.assertFalse(answer_fits_pending(evidence, question))
        self.assertTrue(value_satisfies_pending_contract(evidence, question))
        self.assertEqual(
            question["validation"],
            {"value_type": "evidence_contribution", "max_length": 65536},
        )

    def test_turn_finalizer_removes_superseded_barrier_question_only(self) -> None:
        from agent.harness.contracts import (
            ResponseFragment,
            response_fragment_to_dict,
        )
        from agent.harness.coordinator import _finalize_turn_response
        from agent.harness.questions import (
            manual_question,
            question_text,
            render_question,
        )

        endpoint = manual_question(
            "endpoint_process",
            "new_chain_endpoint",
            question_text("question.chain_rpc.new_chain_endpoint.prompt"),
            owner="chain_rpc",
            field="new_chain_endpoint",
            kind="url",
        )
        schema = manual_question(
            "endpoint_process",
            "new_chain_schema_confirm",
            question_text(
                "question.chain_rpc.schema_evidence.prompt",
                method="eth_chainId",
            ),
            owner="chain_rpc",
            field="new_chain_schema_confirm",
            kind="yes_no",
        )
        state = _state("en")
        state["active_group"] = "endpoint_process"
        state["turn_context"] = {
            "installed_questions": [endpoint, schema],
        }
        state["pending_question"] = schema
        state["response_fragments"] = [
            response_fragment_to_dict(
                ResponseFragment(
                    kind="evidence",
                    message_id="analysis.model_document",
                    payload={
                        "text": "Endpoint validation passed. Evidence: probe.json.",
                        "source_kind": "test",
                        "language": "en",
                        "evidence_hash": "a" * 64,
                        "evidence_paths": ["probe.json"],
                    },
                    source=__name__,
                )
            )
        ]

        result = _finalize_turn_response(state)

        rendered = "\n".join(result["visible_response"])
        self.assertIn("Endpoint validation passed. Evidence: probe.json.", rendered)
        self.assertNotIn(render_question(endpoint, "en"), rendered)
        self.assertEqual(rendered.count(render_question(schema, "en")), 1)

    def test_suspended_earlier_question_precedes_later_derived_question(self) -> None:
        from agent.harness.coordinator import _ask_next_blocking_question
        from agent.harness.domains.chain_rpc import question_for_chain_rpc
        from agent.harness.domains.performance import question_for_performance
        from agent.harness.response import finalize_turn_response

        state = _state(
            "en",
            target_mode="fake-node",
            workflow_mode="rpc_benchmark",
            chain_identity={
                "raw": "sola",
                "canonical": "sola",
                "status": "needs_protocol_confirmation",
            },
            qps_profile={"mode": "quick", "confirmed": False},
        )
        adapter_question = question_for_chain_rpc(state, "chain_identity")
        qps_question = question_for_performance(state, "qps_profile")
        self.assertEqual(adapter_question["id"], "adapter_family_confirm")
        self.assertEqual(qps_question["id"], "qps_profile_confirm")
        state["active_group"] = "qps_profile"
        state["pending_question"] = qps_question
        state["interruption_stack"] = [{
            "group": "chain_identity",
            "question_id": "adapter_family_confirm",
            "reason": "same_turn_action_queue",
        }]
        result = _ask_next_blocking_question(state)
        result = finalize_turn_response(result)

        self.assertEqual(result["active_group"], "chain_identity")
        self.assertEqual(result["pending_question"]["id"], "adapter_family_confirm")
        rendered = "\n".join(result["visible_response"])
        self.assertIn("adapter family", rendered)
        self.assertNotIn("Default QPS profile", rendered)

    def test_generic_manual_question_has_bounded_scalar_contract(self) -> None:
        from agent.harness.questions import (
            literal_matches_validation,
            manual_question,
            question_text,
        )

        question = manual_question(
            "endpoint_process",
            "process_name",
            question_text("question.chain_rpc.real_node_process.prompt"),
            owner="chain_rpc",
            field="BLOCKCHAIN_PROCESS_NAMES",
        )

        self.assertTrue(literal_matches_validation("geth", question["validation"]))
        self.assertFalse(literal_matches_validation("geth node\nmore evidence", question["validation"]))

    def test_process_question_separates_raw_turn_routing_from_extracted_value_validation(self) -> None:
        from agent.harness.domains.chain_rpc import question_for_chain_rpc
        from agent.harness.questions import (
            answer_fits_pending,
            value_satisfies_pending_contract,
        )

        state = _state(
            "en",
            target_mode="real-node",
            chain_identity={"canonical": "bsc", "status": "confirmed"},
            endpoint_evidence={"local_rpc_url_ready": True},
        )
        question = question_for_chain_rpc(state, "endpoint_process")

        self.assertEqual(question["id"], "BLOCKCHAIN_PROCESS_NAMES")
        self.assertEqual(question["validation"]["value_type"], "bounded_text")
        self.assertTrue(answer_fits_pending("geth", question))
        self.assertFalse(answer_fits_pending("Match command line: geth --networkid 1337", question))
        self.assertTrue(value_satisfies_pending_contract("geth --networkid 1337", question))
        self.assertFalse(value_satisfies_pending_contract("geth\nsecond command", question))
        self.assertFalse(value_satisfies_pending_contract("geth\x7f--networkid", question))
        self.assertFalse(value_satisfies_pending_contract("x" * 513, question))

    def test_detected_provider_question_declares_manual_override_grammar(self) -> None:
        from agent.harness.domains.environment import question_for_environment
        from agent.harness.questions import value_satisfies_pending_contract

        for detected_key, field in (
            ("region", "CLOUD_REGION"),
            ("zone", "CLOUD_ZONE"),
            ("machine_type", "MACHINE_TYPE"),
        ):
            confirmed = {
                "CLOUD_REGION": "test-region",
                "CLOUD_ZONE": "test-zone",
                "MACHINE_TYPE": "test-machine",
            }
            confirmed.pop(field)
            state = _state(
                "en",
                confirmed_config=confirmed,
                discovery={"cloud": {detected_key: f"detected-{detected_key}"}},
            )
            question = question_for_environment(state, "provider_deployment")
            self.assertEqual(question["id"], field)
            self.assertEqual(question["validation"], {"value_type": "scalar_token"})
            self.assertTrue(value_satisfies_pending_contract("replacement-value", question))
            self.assertFalse(value_satisfies_pending_contract("replacement value", question))

    def test_manual_questions_declare_domain_candidate_sources(self) -> None:
        from agent.harness.domains.chain_rpc_questions import _weights_question
        from agent.harness.domains.environment import question_for_environment
        from agent.harness.domains.performance import question_for_performance
        from agent.harness.domains.sync_observe import question_for_sync_observe

        environment = _state(
            "en",
            confirmed_config={
                "CLOUD_REGION": "test-region",
                "CLOUD_ZONE": "test-zone",
                "MACHINE_TYPE": "test-machine",
            },
            discovery={"network": {"interfaces": ["eth0"]}},
        )
        network_question = question_for_environment(environment, "network")
        self.assertEqual(
            network_question["structured_config_key"],
            "NETWORK_INTERFACE",
        )
        self.assertIn(
            "propose_config_values",
            network_question["accepted_action_types"],
        )

        qps = _state(
            "en",
            target_mode="fake-node",
            qps_profile={
                "mode": "quick",
                "default_decision_made": True,
                "adjust_field": "INITIAL_QPS",
            },
        )
        qps_question = question_for_performance(qps, "qps_profile")
        self.assertEqual(qps_question["candidate_bindings"], [{
            "type": "set_qps_override",
            "value_argument": "qps_overrides",
            "mapping_key": "INITIAL_QPS",
        }])

        weights = _state(
            "en",
            chain_identity={"canonical": "bsc"},
            custom_rpc={
                "status": "needs_weights",
                "method": "eth_blockNumber",
            },
        )
        weight_question = _weights_question(weights, "custom_rpc")
        self.assertEqual(weight_question["candidate_bindings"], [{
            "type": "rpc_workload_command",
            "value_argument": "rpc_weights",
        }])

        sync = _state(
            "en",
            workflow_mode="sync_observe",
            target_mode="sync-observe",
            confirmed_config={
                "SYNC_OBSERVE_RPC_URL": "http://geth-dev:8545",
                "MAINNET_RPC_URL_REVIEWED": True,
            },
            endpoint_evidence={"sync_rpc_url_ready": True},
            sync_observe={
                "source": "endpoint_only",
                "stop_condition": "duration",
            },
        )
        sync_question = question_for_sync_observe(sync)
        self.assertEqual(sync_question["candidate_bindings"], [{
            "type": "set_sync_observe_options",
            "value_argument": "sync_observe_duration_seconds",
        }])

    def test_model_pending_admission_uses_extracted_value_contract(self) -> None:
        from agent.harness.admission import (
            _action_answers_pending_contract,
        )
        from tests.agent_live.graph_turn import invoke_action
        from agent.harness.domains.chain_rpc import question_for_chain_rpc
        from agent.harness.domains.environment import question_for_environment

        process_state = _state(
            "en",
            target_mode="real-node",
            chain_identity={"canonical": "bsc", "status": "confirmed"},
            endpoint_evidence={"local_rpc_url_ready": True},
        )
        process_state["pending_question"] = question_for_chain_rpc(
            process_state,
            "endpoint_process",
        )
        process_state["last_user_input"] = (
            "The node runs under geth.\nMatch command line: geth --networkid 1337"
        )
        process_action = {
            "type": "answer_pending",
            "answer": "geth --networkid 1337",
            "source_evidence": "Match command line: geth --networkid 1337",
            "semantic_purpose_verified": True,
            "pending_option_semantic_verified": True,
        }
        self.assertTrue(_action_answers_pending_contract(process_state, process_action))
        process_result = invoke_action(
            process_state,
            process_action,
            process_state["last_user_input"],
        )
        self.assertEqual(
            process_result["confirmed_config"]["BLOCKCHAIN_PROCESS_NAMES"],
            "geth --networkid 1337",
        )

        region_state = _state(
            "en",
            discovery={"cloud": {"region": "test-region"}},
        )
        region_state["pending_question"] = question_for_environment(
            region_state,
            "provider_deployment",
        )
        region_state["last_user_input"] = (
            "Do not use the detected region.\nSet CLOUD_REGION=us-central1 instead."
        )
        region_action = {
            "type": "answer_pending",
            "answer": "us-central1",
            "source_evidence": "Set CLOUD_REGION=us-central1 instead.",
            "semantic_purpose_verified": True,
            "pending_option_semantic_verified": True,
        }
        self.assertTrue(_action_answers_pending_contract(region_state, region_action))
        region_result = invoke_action(
            region_state,
            region_action,
            region_state["last_user_input"],
        )
        self.assertEqual(region_result["confirmed_config"]["CLOUD_REGION"], "us-central1")

        detected_region_action = {
            "type": "answer_pending",
            "answer": "asia-east1",
            "source_evidence": "set CLOUD_REGION to asia-east1",
            "semantic_purpose_verified": True,
            "pending_option_semantic_verified": True,
        }
        detected_region_result = invoke_action(
            region_state,
            detected_region_action,
            detected_region_action["source_evidence"],
        )
        self.assertEqual(
            detected_region_result["confirmed_config"]["CLOUD_REGION"],
            "asia-east1",
        )
        self.assertEqual(
            detected_region_result["pending_question"]["id"],
            "CLOUD_ZONE",
        )

        normalized_state = _state("en")
        normalized_state["pending_question"] = question_for_environment(
            normalized_state,
            "provider_deployment",
        )
        normalized_action = {
            "type": "answer_pending",
            "selected_value": "us-central1",
            "source_evidence": "us-central1",
            "_origin_text": "Use the following region for this run:\nus-central1",
        }
        normalized_result = invoke_action(
            normalized_state,
            normalized_action,
            normalized_action["_origin_text"],
        )
        self.assertEqual(
            normalized_result["confirmed_config"]["CLOUD_REGION"],
            "us-central1",
        )

    def _question_cases(self) -> list[tuple[str, Callable[[str], dict[str, Any] | None]]]:
        from agent.harness.contracts import ActionProposal
        from agent.harness.domains.chain_rpc import question_for_chain_rpc
        from agent.harness.domains.chain_rpc import apply_chain_rpc_action
        from agent.harness.domains.chain_identity import (
            _chain_ambiguity_question,
        )
        from agent.harness.domains.environment import question_for_environment
        from agent.harness.domains.environment import config_proposal_review_question
        from agent.harness.domains.execution import question_for_execution
        from agent.harness.domains.orientation import (
            opening_question,
            resume_modify_group_question,
            resume_question,
        )
        from agent.harness.domains.performance import question_for_performance
        from agent.harness.domains.recovery import question_for_recovery
        from agent.harness.domains.sync_observe import question_for_sync_observe

        def chain_question(group: str, **updates: Any) -> Callable[[str], dict[str, Any] | None]:
            return lambda language: question_for_chain_rpc(_state(language, **updates), group)

        def performance_question(group: str, **updates: Any) -> Callable[[str], dict[str, Any] | None]:
            return lambda language: question_for_performance(_state(language, **updates), group)

        def pending_from(
            operation: Callable[[dict[str, Any]], Any],
        ) -> Callable[[str], dict[str, Any] | None]:
            from agent.harness.contracts import HandlerResult

            def factory(language: str) -> dict[str, Any] | None:
                state = _state(language)
                result = operation(state)
                if isinstance(result, HandlerResult):
                    pending = result.pending_question
                    return dict(pending) if pending else None
                if isinstance(result, dict) and result.get("id"):
                    return result
                return state.get("pending_question") or None

            return factory

        complete_config = {
            "CLOUD_REGION": "test-region",
            "CLOUD_ZONE": "test-zone",
            "MACHINE_TYPE": "test-machine",
            "LEDGER_DEVICE": "vda",
            "DATA_VOL_TYPE": "test-disk",
            "DATA_VOL_SIZE": "1",
            "DATA_VOL_MAX_IOPS": "1",
            "DATA_VOL_MAX_THROUGHPUT": "1",
            "has_accounts_device": False,
            "NETWORK_INTERFACE": "eth0",
            "NETWORK_MAX_BANDWIDTH_GBPS": "1",
        }

        def execution_question(language: str) -> dict[str, Any] | None:
            state = _state(
                language,
                target_mode="fake-node",
                workflow_mode="rpc_benchmark",
                chain_identity={"canonical": "bsc", "status": "confirmed"},
                confirmed_config=complete_config,
                rpc_mode="single",
                workload={"confirmed": True},
                qps_profile={"mode": "quick", "confirmed": True},
                observability={"mode": "disabled"},
                advanced_tuning={"confirmed": True},
            )
            return question_for_execution(state, "preflight_smoke_execution")

        validated_methods = [{"method": "eth_blockNumber"}, {"method": "eth_gasPrice"}]
        schema_draft = {"method": "eth_blockNumber", "params": []}
        return [
            ("opening", lambda language: opening_question(_state(language))),
            ("resume", lambda language: resume_question(_state(language, target_mode="fake-node"))),
            (
                "resume_modify_group",
                lambda language: resume_modify_group_question(
                    _state(
                        language,
                        target_mode="fake-node",
                        workflow_mode="rpc_benchmark",
                    )
                ),
            ),
            (
                "resume_quarantine",
                lambda language: resume_question(
                    _state(language, checkpoint_recovery={"status": "quarantined", "reason": "partial"})
                ),
            ),
            (
                "provider_detected",
                lambda language: question_for_environment(
                    _state(language, discovery={"cloud": {"region": "test-region"}}),
                    "provider_deployment",
                ),
            ),
            (
                "provider_zone",
                lambda language: question_for_environment(
                    _state(language, confirmed_config={"CLOUD_REGION": "test-region"}),
                    "provider_deployment",
                ),
            ),
            (
                "provider_machine",
                lambda language: question_for_environment(
                    _state(
                        language,
                        confirmed_config={"CLOUD_REGION": "test-region", "CLOUD_ZONE": "test-zone"},
                    ),
                    "provider_deployment",
                ),
            ),
            *(
                (
                    f"inferred_config_{group}",
                    lambda language, group=group: config_proposal_review_question(
                        group,
                        {"config_values": {"CLOUD_REGION": "test-region"}},
                        language=language,
                    ),
                )
                for group in ("provider_deployment", "ledger_disk", "accounts_disk", "network")
            ),
            (
                "accounts_presence",
                lambda language: question_for_environment(_state(language), "accounts_disk"),
            ),
            (
                "ledger_size_inference",
                lambda language: question_for_environment(
                    _state(
                        language,
                        confirmed_config={"LEDGER_DEVICE": "vda", "DATA_VOL_TYPE": "test-disk"},
                        discovery={"disks": {"candidates": [{"name": "vda", "size": "100G", "type": "disk"}]}},
                    ),
                    "ledger_disk",
                ),
            ),
            *(
                (
                    f"ledger_{question_id.lower()}",
                    lambda language, confirmed=confirmed: question_for_environment(
                        _state(language, confirmed_config=confirmed), "ledger_disk"
                    ),
                )
                for question_id, confirmed in (
                    ("LEDGER_DEVICE", {}),
                    ("DATA_VOL_TYPE", {"LEDGER_DEVICE": "vda"}),
                    ("DATA_VOL_MAX_IOPS", {"LEDGER_DEVICE": "vda", "DATA_VOL_TYPE": "test-disk", "DATA_VOL_SIZE": "1"}),
                    ("DATA_VOL_MAX_THROUGHPUT", {"LEDGER_DEVICE": "vda", "DATA_VOL_TYPE": "test-disk", "DATA_VOL_SIZE": "1", "DATA_VOL_MAX_IOPS": "1"}),
                )
            ),
            *(
                (
                    f"accounts_{question_id.lower()}",
                    lambda language, confirmed=confirmed: question_for_environment(
                        _state(language, confirmed_config=confirmed), "accounts_disk"
                    ),
                )
                for question_id, confirmed in (
                    ("ACCOUNTS_DEVICE", {"has_accounts_device": True}),
                    ("ACCOUNTS_VOL_TYPE", {"has_accounts_device": True, "ACCOUNTS_DEVICE": "vdb"}),
                    ("ACCOUNTS_VOL_SIZE", {"has_accounts_device": True, "ACCOUNTS_DEVICE": "vdb", "ACCOUNTS_VOL_TYPE": "test-disk"}),
                    ("ACCOUNTS_VOL_MAX_IOPS", {"has_accounts_device": True, "ACCOUNTS_DEVICE": "vdb", "ACCOUNTS_VOL_TYPE": "test-disk", "ACCOUNTS_VOL_SIZE": "1"}),
                    ("ACCOUNTS_VOL_MAX_THROUGHPUT", {"has_accounts_device": True, "ACCOUNTS_DEVICE": "vdb", "ACCOUNTS_VOL_TYPE": "test-disk", "ACCOUNTS_VOL_SIZE": "1", "ACCOUNTS_VOL_MAX_IOPS": "1"}),
                )
            ),
            (
                "network_interface",
                lambda language: question_for_environment(_state(language), "network"),
            ),
            (
                "network_bandwidth",
                lambda language: question_for_environment(
                    _state(language, confirmed_config={"NETWORK_INTERFACE": "eth0"}), "network"
                ),
            ),
            ("target_mode", chain_question("target_mode")),
            (
                "target_mode_change",
                pending_from(
                    lambda state: apply_chain_rpc_action(
                        {
                            **state,
                            "target_mode": "fake-node",
                            "active_group": "qps_profile",
                        },
                        ActionProposal(
                            "target-mode-change",
                            "choose_target_mode",
                            {"target_mode": "real-node", "target_mode_explicit": True},
                            "high",
                        ),
                    )
                ),
            ),
            (
                "chain_manual",
                chain_question("chain_identity"),
            ),
            (
                "unknown_chain_identity",
                pending_from(
                    lambda state: apply_chain_rpc_action(
                        state,
                        ActionProposal(
                            "unknown-chain",
                            "choose_chain",
                            {
                                "chain_text": "sola",
                                "reference_kind": "named_identity",
                                "chain_exists": False,
                                "possible_known_chain": "solana",
                                "confidence": "high",
                            },
                            "high",
                        ),
                    )
                ),
            ),
            (
                "chain_change",
                pending_from(
                    lambda state: apply_chain_rpc_action(
                        {
                            **state,
                            "chain_identity": {"canonical": "bsc", "status": "confirmed"},
                        },
                        ActionProposal(
                            "chain-change",
                            "choose_chain",
                            {"chain_text": "ethereum"},
                            "high",
                        ),
                    )
                ),
            ),
            (
                "chain_ambiguity",
                pending_from(
                    lambda state: _chain_ambiguity_question(
                        state, "bsc", {"chain_candidates": ["bsc", "ethereum"]}
                    )
                ),
            ),
            (
                "adapter_family",
                chain_question("chain_identity", chain_identity={"status": "needs_adapter_family_confirmation"}),
            ),
            (
                "case3_next",
                chain_question(
                    "chain_identity",
                    chain_identity={"status": "case3_collecting_evidence"},
                    secondary_handoff={"evidence": ["evidence"]},
                ),
            ),
            (
                "case3_evidence",
                chain_question(
                    "chain_identity",
                    chain_identity={"status": "case3_needs_evidence"},
                ),
            ),
            *(
                (
                    f"endpoint_{question_id.lower()}",
                    chain_question("endpoint_process", **updates),
                )
                for question_id, updates in (
                    ("LOCAL_RPC_URL", {"target_mode": "real-node", "chain_identity": {"canonical": "bsc", "status": "confirmed"}}),
                    ("BLOCKCHAIN_PROCESS_NAMES", {"target_mode": "real-node", "chain_identity": {"canonical": "bsc", "status": "confirmed"}, "endpoint_evidence": {"local_rpc_url_ready": True}}),
                    ("SYNC_OBSERVE_RPC_URL", {"target_mode": "sync-observe", "workflow_mode": "sync_observe", "chain_identity": {"canonical": "bsc", "status": "confirmed"}, "sync_observe": {"source": "endpoint_only"}}),
                )
            ),
            (
                "chain_change_input",
                pending_from(
                    lambda state: apply_chain_rpc_action(
                        {
                            **state,
                            "target_mode": "fake-node",
                            "workflow_mode": "rpc_benchmark",
                            "chain_identity": {"canonical": "bsc", "status": "confirmed"},
                        },
                        ActionProposal(
                            "chain-change-input",
                            "request_chain_selection",
                            {},
                            "high",
                        ),
                    )
                ),
            ),
            (
                "mainnet_review",
                chain_question(
                    "endpoint_process",
                    target_mode="real-node",
                    chain_identity={"canonical": "bsc", "status": "confirmed"},
                    endpoint_evidence={"local_rpc_url_ready": True},
                    confirmed_config={"BLOCKCHAIN_PROCESS_NAMES": "node"},
                ),
            ),
            (
                "chain_auxiliary_api_key",
                chain_question(
                    "chain_auxiliary_endpoints",
                    chain_identity={"canonical": "litecoin", "status": "confirmed"},
                ),
            ),
            (
                "custom_adapter",
                chain_question(
                    "endpoint_process",
                    chain_identity={"canonical": "bsc", "status": "confirmed"},
                    custom_rpc={"status": "needs_adapter_family_confirmation"},
                ),
            ),
            *(
                (
                    f"custom_{status}",
                    chain_question(
                        "endpoint_process",
                        chain_identity={"canonical": "bsc", "status": "confirmed"},
                    custom_rpc={"status": status, "catalog": _rpc_catalog(methods=extra.get("validated_methods")), **{key: value for key, value in extra.items() if key != "validated_methods"}},
                    ),
                )
                for status, extra in (
                    ("needs_endpoint", {}),
                    ("needs_method", {"endpoint_ready": True}),
                    ("needs_schema_evidence", {"endpoint_ready": True, "method": "eth_blockNumber"}),
                    ("needs_weights", {"validated_methods": validated_methods}),
                )
            ),
            (
                "custom_schema",
                chain_question(
                    "endpoint_process",
                    chain_identity={"canonical": "bsc", "status": "confirmed"},
                    custom_rpc={"status": "schema_needs_confirmation", "catalog": _rpc_catalog(draft=schema_draft)},
                ),
            ),
            (
                "custom_response",
                chain_question(
                    "endpoint_process",
                    chain_identity={"canonical": "bsc", "status": "confirmed"},
                    custom_rpc={
                        "status": "response_needs_confirmation",
                        "catalog": _rpc_catalog(draft={"observed_response": {"shape_hash": "shape-1", "sample": '{"result":"0x1"}'}}),
                    },
                ),
            ),
            (
                "custom_continue",
                chain_question(
                    "endpoint_process",
                    chain_identity={"canonical": "bsc", "status": "confirmed"},
                    custom_rpc={"status": "method_validated_next"},
                ),
            ),
            (
                "custom_scope",
                chain_question(
                    "endpoint_process",
                    chain_identity={"canonical": "bsc", "status": "confirmed"},
                    custom_rpc={"status": "needs_scope"},
                ),
            ),
            (
                "custom_single_method",
                chain_question(
                    "endpoint_process",
                    chain_identity={"canonical": "bsc", "status": "confirmed"},
                    custom_rpc={"status": "needs_single_method", "catalog": _rpc_catalog(methods=validated_methods)},
                ),
            ),
            (
                "new_chain_schema",
                chain_question(
                    "endpoint_process",
                    chain_identity={
                        "canonical": "new-chain",
                        "status": "existing_family_schema_needs_confirmation",
                    },
                    custom_rpc={"catalog": _rpc_catalog(draft=schema_draft)},
                ),
            ),
            (
                "new_chain_response",
                chain_question(
                    "endpoint_process",
                    chain_identity={
                        "canonical": "new-chain",
                        "status": "existing_family_response_needs_confirmation",
                    },
                    custom_rpc={"catalog": _rpc_catalog(draft={"observed_response": {"shape_hash": "shape-1", "sample": '{"result":"0x1"}'}})},
                ),
            ),
            *(
                (
                    f"new_chain_{status}",
                    chain_question(
                        "endpoint_process",
                        chain_identity={"canonical": "new-chain", "status": status, **extra},
                        custom_rpc={"catalog": _rpc_catalog(methods=extra.get("validated_methods"))},
                        endpoint_evidence={"candidate_endpoint_ready": status != "existing_family_needs_endpoint"},
                    ),
                )
                for status, extra in (
                    ("existing_family_needs_endpoint", {}),
                    ("existing_family_needs_method", {}),
                    ("existing_family_needs_schema_evidence", {"candidate_method": "eth_blockNumber"}),
                    ("existing_family_needs_weights", {"validated_methods": validated_methods}),
                )
            ),
            (
                "new_chain_continue",
                chain_question(
                    "endpoint_process",
                    chain_identity={"canonical": "new-chain", "status": "existing_family_method_validated_next"},
                ),
            ),
            (
                "new_chain_scope",
                chain_question(
                    "endpoint_process",
                    chain_identity={"canonical": "new-chain", "status": "existing_family_needs_workload_scope"},
                ),
            ),
            (
                "new_chain_single_method",
                chain_question(
                    "endpoint_process",
                    chain_identity={
                        "canonical": "new-chain",
                        "status": "existing_family_needs_single_method",
                    },
                    custom_rpc={"catalog": _rpc_catalog(methods=validated_methods)},
                ),
            ),
            (
                "rpc_mode",
                chain_question("workload_rpc", chain_identity={"canonical": "bsc", "status": "confirmed"}),
            ),
            (
                "workload_single",
                chain_question(
                    "workload_rpc",
                    rpc_mode="single",
                    chain_identity={"canonical": "bsc", "status": "confirmed"},
                ),
            ),
            (
                "workload_mixed",
                chain_question(
                    "workload_rpc",
                    rpc_mode="mixed",
                    chain_identity={"canonical": "bsc", "status": "confirmed"},
                ),
            ),
            (
                "target_change_scope",
                pending_from(
                    lambda state: __import__(
                        "agent.harness.domains.chain_rpc_questions", fromlist=["_target_change_scope_question"]
                    )._target_change_scope_question(state)
                ),
            ),
            (
                "new_chain_runtime",
                chain_question(
                    "target_samples_fixtures",
                    target_mode="fake-node",
                    chain_identity={"status": "existing_family_runtime_choice"},
                ),
            ),
            (
                "missing_custom_fixture",
                chain_question(
                    "target_samples_fixtures",
                    chain_identity={"canonical": "bsc", "status": "confirmed"},
                    fixture_evidence={"status": "missing", "missing": [{"method": "eth_accounts"}]},
                ),
            ),
            ("qps_mode", performance_question("qps_profile")),
            ("qps_confirm", performance_question("qps_profile", qps_profile={"mode": "quick"})),
            (
                "qps_adjust",
                performance_question("qps_profile", qps_profile={"mode": "quick", "default_decision_made": True}),
            ),
            (
                "qps_adjust_value",
                performance_question(
                    "qps_profile",
                    qps_profile={"mode": "quick", "default_decision_made": True, "adjust_field": "INITIAL_QPS"},
                ),
            ),
            ("observability", performance_question("observability")),
            ("advanced_confirm", performance_question("advanced_tuning")),
            (
                "advanced_adjust",
                performance_question("advanced_tuning", advanced_tuning={"default_decision_made": True}),
            ),
            (
                "advanced_adjust_value",
                performance_question(
                    "advanced_tuning",
                    advanced_tuning={"default_decision_made": True, "adjust_field": "MONITOR_INTERVAL"},
                ),
            ),
            (
                "sync_source",
                lambda language: question_for_sync_observe(_state(language, workflow_mode="sync_observe")),
            ),
            (
                "sync_after_setup",
                lambda language: question_for_sync_observe(
                    _state(language, workflow_mode="sync_observe", sync_observe={"source": "client_setup"})
                ),
            ),
            (
                "sync_stop",
                lambda language: question_for_sync_observe(
                    _state(
                        language,
                        workflow_mode="sync_observe",
                        sync_observe={"source": "existing_local_node"},
                        confirmed_config={
                            "SYNC_OBSERVE_RPC_URL": "http://node:8545",
                            "BLOCKCHAIN_PROCESS_NAMES": "geth",
                            "MAINNET_RPC_URL_REVIEWED": True,
                        },
                        endpoint_evidence={"sync_rpc_url_ready": True},
                    )
                ),
            ),
            (
                "sync_duration",
                lambda language: question_for_sync_observe(
                    _state(
                        language,
                        workflow_mode="sync_observe",
                        sync_observe={"source": "existing_local_node", "stop_condition": "duration"},
                        confirmed_config={
                            "SYNC_OBSERVE_RPC_URL": "http://node:8545",
                            "BLOCKCHAIN_PROCESS_NAMES": "geth",
                            "MAINNET_RPC_URL_REVIEWED": True,
                        },
                        endpoint_evidence={"sync_rpc_url_ready": True},
                    )
                ),
            ),
            (
                "failure_recovery",
                lambda language: question_for_recovery(
                    _state(
                        language,
                        failure_recovery={
                            "status": "pending",
                            "record": {
                                "code": "endpoint_unreachable",
                                "summary": "endpoint failed",
                                "allowed_actions": ["correct_failure", "inspect_failure", "cancel_failure_recovery"],
                            },
                        },
                    ),
                    "failure_recovery",
                ),
            ),
            ("execution", execution_question),
        ]

    def test_domain_choice_questions_are_typed_executable_and_renderable(self) -> None:
        from agent.harness.action_registry import ACTION_BY_TYPE
        from agent.harness.contracts import text_ref_from_dict
        from agent.harness.questions import render_question
        from agent.harness.response_catalog import render_text_ref

        for case_name, factory in self._question_cases():
            with self.subTest(case=case_name):
                try:
                    questions = {language: factory(language) for language in ("en", "zh")}
                except Exception as exc:
                    self.fail(f"{case_name} question factory raised {type(exc).__name__}: {exc}")
                self.assertIsNotNone(questions["en"], "factory did not expose its expected blocking question")
                self.assertIsNotNone(questions["zh"], "factory did not expose its expected blocking question")
                english = questions["en"]
                chinese = questions["zh"]
                assert english is not None and chinese is not None
                if english.get("kind") not in {"numbered_choice", "yes_no"}:
                    continue
                self.assertEqual(english.get("kind"), chinese.get("kind"))
                self.assertEqual(english.get("id"), chinese.get("id"))
                for language, question in questions.items():
                    assert question is not None
                    self.assertEqual(
                        question.get("contract_version"),
                        QUESTION_CONTRACT_VERSION,
                    )
                    self.assertTrue(question.get("options"))
                    for option in question["options"]:
                        action = option.get("action") or {}
                        self.assertIn(action.get("type"), ACTION_BY_TYPE)
                        self.assertTrue(
                            option.get("expected_patch")
                            or option.get("return_policy") == "stop_after_response"
                            or action.get("type") != "answer_pending",
                            f"{question['id']}/{option.get('id')} has no postcondition",
                        )
                    rendered = render_question(question, language)
                    self.assertTrue(rendered.strip())
                    prompt = render_text_ref(
                        text_ref_from_dict(question["prompt_ref"]),
                        language,
                        kind="question_prompt",
                    )
                    self.assertIn(prompt, rendered)
                    for option in question["options"]:
                        label = render_text_ref(
                            text_ref_from_dict(option["label_ref"]),
                            language,
                            kind="option_label",
                        )
                        self.assertIn(label, rendered)
                    if question.get("manual_input_allowed"):
                        expected_instruction = (
                            "请回复选项编号或名称，也可以直接输入自定义值。"
                            if language == "zh"
                            else "Reply with an option number or name, or enter a custom value."
                        )
                    else:
                        expected_instruction = (
                            "请回复选项编号或名称。"
                            if language == "zh"
                            else "Reply with an option number or name."
                        )
                    self.assertIn(expected_instruction, rendered)
                    if language == "zh":
                        self.assertNotIn("Reply with ", rendered)
                    else:
                        self.assertNotIn("请回复", rendered)
                        self.assertNotIn("你可以回复", rendered)
                self.assertNotEqual(render_question(english, "en"), render_question(chinese, "zh"))

    def test_question_renderer_requires_explicit_language(self) -> None:
        from agent.harness.questions import (
            choice_question,
            question_text,
            render_question,
        )

        question = choice_question(
            "opening",
            "explicit_language_contract",
            question_text("question.orientation.opening.prompt"),
            owner="orientation",
            field="choice",
            options=[{
                "id": "one",
                "label": question_text(
                    "question.orientation.resume.option.continue"
                ),
                "value": "one",
            }],
        )

        with self.assertRaises(TypeError):
            render_question(question)  # type: ignore[call-arg]

    def test_option_description_is_visible_semantic_context_not_an_exact_alias(self) -> None:
        from agent.harness.contracts import text_ref_from_dict
        from agent.harness.questions import (
            choice_question,
            exact_answer,
            question_text,
            render_question,
        )
        from agent.harness.response_catalog import render_text_ref

        question = choice_question(
            "opening",
            "described_option_contract",
            question_text("question.orientation.opening.prompt"),
            owner="orientation",
            field="target_mode",
            options=[{
                "id": "fake-node",
                "label": question_text(
                    "question.orientation.opening.fake_node.label"
                ),
                "description": question_text(
                    "question.orientation.opening.fake_node.description"
                ),
                "value": "fake-node",
                "action": {"type": "choose_target_mode", "target_mode": "fake-node"},
            }],
        )

        description = render_text_ref(
            text_ref_from_dict(question["options"][0]["description_ref"]),
            "en",
            kind="option_description",
        )
        self.assertIn(description, render_question(question, "en"))
        self.assertEqual(exact_answer("fake-node", question), (True, "fake-node"))
        self.assertEqual(exact_answer(description, question), (False, None))

    def test_opening_options_explain_product_effect_in_both_languages(self) -> None:
        from agent.harness.domains.orientation import opening_question
        from agent.harness.contracts import text_ref_from_dict
        from agent.harness.questions import render_question
        from agent.harness.response_catalog import render_text_ref

        for language in ("en", "zh"):
            with self.subTest(language=language):
                question = opening_question(_state(language))
                descriptions = [
                    render_text_ref(
                        text_ref_from_dict(option["description_ref"]),
                        language,
                        kind="option_description",
                    ).strip()
                    for option in question["options"]
                ]
                self.assertEqual(len(descriptions), 4)
                self.assertTrue(all(descriptions))
                rendered = render_question(question, language)
                for description in descriptions:
                    self.assertIn(description, rendered)

    def test_manual_choice_question_has_complete_validation_at_construction(self) -> None:
        from agent.harness.questions import choice_question, question_text

        question = choice_question(
            "chain_auxiliary_endpoints",
            "RPC_API_KEY",
            question_text(
                "question.chain_rpc.auxiliary_endpoint.prompt",
                chain="ethereum",
                field="RPC_API_KEY",
            ),
            owner="chain_rpc",
            field="RPC_API_KEY",
            kind="manual_value",
            manual_input_allowed=True,
            options=[{
                "id": "skip",
                "label": question_text(
                    "question.chain_rpc.option.skip_unconfigured"
                ),
                "value": "none",
            }],
        )

        self.assertEqual(
            question["validation"],
            {"value_type": "scalar_token", "max_length": 180},
        )

    def test_every_registered_question_has_a_runtime_contract_scenario(self) -> None:
        from agent.workflows.group_registry import GROUP_QUESTION_ORDER

        rendered = set()
        for _case_name, factory in self._question_cases():
            question = factory("en")
            if question:
                rendered.add((str(question.get("group") or ""), str(question.get("id") or "")))
        registered = {
            (group, question_id)
            for group, question_ids in GROUP_QUESTION_ORDER
            for question_id in question_ids
        }

        self.assertEqual(
            registered,
            rendered,
            "group question metadata and constructible runtime contracts have drifted",
        )


class HarnessStateInvariantTest(unittest.TestCase):


    def test_harness_replaces_duplicate_model_action_ids(self) -> None:
        from agent.harness.action_registry import assign_action_ids

        actions = assign_action_ids(
            "thread:9",
            "set two values",
            [
                {"action_id": "model-duplicate", "type": "set_qps_mode", "qps_mode": "quick"},
                {"action_id": "model-duplicate", "type": "set_observability", "observability_mode": "disabled"},
            ],
        )

        self.assertNotEqual(actions[0]["action_id"], actions[1]["action_id"])
        self.assertNotIn("model-duplicate", {item["action_id"] for item in actions})
        self.assertEqual(actions, assign_action_ids("thread:9", "set two values", actions))

    def test_model_manual_answer_must_be_grounded_in_the_current_turn(self) -> None:
        from agent.harness.admission import _action_answers_pending_contract

        state = _state(
            last_user_input="I want to configure QPS first",
            active_group="provider_deployment",
            pending_question={
                "id": "CLOUD_ZONE",
                "group": "provider_deployment",
                "kind": "manual_value",
                "field": "CLOUD_ZONE",
                "manual_input_allowed": True,
                "validation": {"value_type": "scalar_token"},
            },
        )
        invented = {
            "type": "answer_pending",
            "answer": "us-1",
            "source_evidence": "I want to configure QPS first",
        }
        grounded = {
            "type": "answer_pending",
            "answer": "asia-east1-c",
            "source_evidence": "asia-east1-c",
        }
        field_name_as_value = {
            "type": "answer_pending",
            "answer": "CLOUD_ZONE",
            "selected_value": "CLOUD_ZONE",
            "source_evidence": "return to CLOUD_ZONE config",
        }

        self.assertFalse(_action_answers_pending_contract(state, invented))
        state["last_user_input"] = "return to CLOUD_ZONE config"
        self.assertFalse(_action_answers_pending_contract(state, field_name_as_value))
        state["last_user_input"] = "my zone is asia-east1-c"
        self.assertTrue(_action_answers_pending_contract(state, grounded))

    def test_model_cannot_infer_fake_node_from_an_unresolved_benchmark_goal(self) -> None:
        from agent.harness.admission import _action_answers_pending_contract
        from agent.harness.domains.orientation import opening_question

        user_text = "我要测试 BNB，用 mixed，QPS quick，并开启本地 Grafana"
        state = _state(last_user_input=user_text)
        state["pending_question"] = opening_question(state)
        invented = {
            "type": "answer_pending",
            "selected_value": "fake-node",
            "source_evidence": user_text,
        }

        self.assertFalse(_action_answers_pending_contract(state, invented))

    def test_config_review_overlay_restores_the_interrupted_typed_question(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn

        pending = {
            "contract_version": 1,
            "id": "custom_rpc_endpoint",
            "group": "endpoint_process",
            "owner": "chain_rpc",
            "kind": "url",
            "prompt": "Provide validation endpoint.",
            "field": "custom_rpc_endpoint",
            "manual_input_allowed": True,
            "options": [],
            "accepted_action_types": ["answer_pending", "rpc_catalog_command"],
            "validation": {"value_type": "scalar_token"},
        }
        source = "Set LOCAL_RPC_URL to https://example.invalid/rpc"
        state = _state(
            active_group="endpoint_process",
            pending_question=pending,
            last_user_input=source,
            custom_rpc={"status": "needs_endpoint"},
            chain_identity={"raw": "bsc", "canonical": "bsc", "status": "confirmed", "case": "known"},
            confirmed_config={"BLOCKCHAIN_NODE": "bsc"},
        )
        def resolved(current_state: dict[str, Any], _text: str) -> dict[str, Any]:
            return {"actions": [_admitted_proposal_action(
                current_state,
                {"LOCAL_RPC_URL": "https://example.invalid/rpc"},
                source,
            )]}

        with patch("tests.agent_live.graph_turn.TEST_SEMANTIC_PLANNER", side_effect=resolved):
            review = process_turn(state)

        self.assertEqual(review["pending_question"]["id"], "inferred_config_review")
        self.assertEqual(review["interruption_stack"][-1]["question_id"], "custom_rpc_endpoint")

        review["last_user_input"] = "N"
        restored = process_turn(review)
        self.assertEqual(restored["pending_question"]["id"], "custom_rpc_endpoint")
        self.assertNotIn("LOCAL_RPC_URL", restored.get("confirmed_config") or {})

    def test_action_dependencies_precede_phase_for_bound_pending_answers(self) -> None:
        from agent.harness.coordinator import _order_action_queue

        state = _state(
            active_group="endpoint_process",
            pending_question={
                "id": "validation_endpoint",
                "group": "endpoint_process",
                "kind": "url",
                "requires_capabilities": ["chain_identity"],
            },
            chain_identity={},
        )
        actions = [
            {"type": "answer_pending", "answer": "http://geth-dev:8545"},
            {"type": "choose_chain", "chain_text": "ethereum"},
        ]

        ordered = _order_action_queue(state, actions)

        self.assertEqual([item["type"] for item in ordered], ["choose_chain", "answer_pending"])

    def test_sync_observe_checkpoint_migration_removes_rpc_only_state(self) -> None:
        from agent.harness.state import migrate_state
        from tests.agent_live.graph_turn import action_types

        migrated = migrate_state(
            {
                "schema_version": 12,
                "workflow_mode": "sync_observe",
                "target_mode": "sync-observe",
                "active_group": "qps_profile",
                "pending_question": {"id": "benchmark_mode", "group": "qps_profile"},
                "rpc_mode": "mixed",
                "workload": {"confirmed": True},
                "qps_profile": {"mode": "quick"},
                "action_queue": [
                    {"type": "set_qps_mode", "qps_mode": "quick"},
                    {"type": "answer_opening_question", "topic": "current_config"},
                ],
            },
            thread_id="sync-migration",
            language="en",
            session_purpose="user",
        )

        self.assertEqual(migrated["rpc_mode"], "")
        self.assertEqual(migrated["workload"], {})
        self.assertEqual(migrated["qps_profile"], {})
        self.assertEqual(migrated["active_group"], "opening")
        self.assertEqual(migrated["pending_question"], {})
        self.assertEqual(action_types(migrated), [])
        self.assertTrue(any(
            event.get("event") == "checkpoint_legacy_inflight_quarantined"
            for event in migrated["audit_events"]
        ))

    def test_sync_observe_invariant_rejects_runtime_qps_contamination(self) -> None:
        from agent.harness.invariants import StateInvariantError, validate_state

        state = _state(
            target_mode="sync-observe",
            workflow_mode="sync_observe",
            qps_profile={"mode": "quick"},
        )

        with self.assertRaisesRegex(StateInvariantError, "RPC benchmark workload or QPS"):
            validate_state(state)

    def test_validate_state_accepts_well_formed_invariant_inputs(self) -> None:
        from agent.harness.invariants import validate_state

        states = [
            _state(),
            _state(
                active_group="provider_deployment",
                pending_question={
                    "id": "region",
                    "group": "provider_deployment",
                    "owner": "environment",
                    "kind": "manual_value",
                },
            ),
            _state(
                active_group="network",
                action_queue=[
                    {
                        "action_id": "one",
                        "action_type": "set_qps_mode",
                        "owner": "performance",
                        "arguments": {"qps_mode": "quick"},
                        "effect_kind": "pure",
                        "status": "admitted",
                    },
                    {
                        "action_id": "two",
                        "action_type": "set_observability",
                        "owner": "performance",
                        "arguments": {"observability_mode": "disabled"},
                        "effect_kind": "pure",
                        "status": "admitted",
                    },
                ],
                group_history=["opening", "provider_deployment"],
            ),
            _state(
                active_group="network",
                pending_question={
                    "id": "interface",
                    "group": "network",
                    "owner": "environment",
                },
                invalidated_groups=["network"],
                group_states={"network": {"status": "reconfiguring"}},
            ),
        ]
        for index, state in enumerate(states):
            with self.subTest(case=index):
                validate_state(state)

    def test_validate_state_rejects_invalid_architecture_inputs(self) -> None:
        from agent.harness.invariants import StateInvariantError, validate_state

        cases = {
            "unknown_active_group": _state(active_group="not-a-group"),
            "pending_owner_mismatch": _state(
                active_group="network",
                pending_question={"id": "region", "group": "provider_deployment"},
            ),
            "pending_invalidated_group": _state(
                active_group="network",
                pending_question={"id": "interface", "group": "network"},
                invalidated_groups=["network"],
            ),
            "duplicate_action_id": _state(
                action_queue=[{"action_id": "same"}, {"action_id": "same"}],
            ),
            "unknown_group_history": _state(group_history=["opening", "not-a-group"]),
            "case3_pending_without_handoff_owner": _state(
                active_group="chain_identity",
                chain_identity={
                    "status": "case3_needs_evidence",
                    "adapter_family": "unsupported",
                    "case": "case3",
                },
                pending_question={
                    "id": "case3_protocol_evidence",
                    "group": "chain_identity",
                },
            ),
        }
        for name, state in cases.items():
            with self.subTest(case=name), self.assertRaises(StateInvariantError):
                validate_state(state)

    def test_validate_state_accepts_complete_case3_evidence_owner(self) -> None:
        from agent.harness.invariants import validate_state

        state = _state(
            active_group="chain_identity",
            chain_identity={
                "raw": "WeirdP2PChain",
                "canonical": "WeirdP2PChain",
                "status": "case3_needs_evidence",
                "adapter_family": "unsupported",
                "case": "case3",
            },
            secondary_handoff={
                "status": "collecting_evidence",
                "kind": "case3_protocol_adapter_implementation",
                "chain": "WeirdP2PChain",
                "adapter_family": "unsupported",
                "evidence": [],
            },
            pending_question={
                "id": "case3_protocol_evidence",
                "group": "chain_identity",
                "owner": "chain_rpc",
            },
        )

        validate_state(state)

    def test_completed_group_cannot_own_a_pending_question(self) -> None:
        from agent.harness.invariants import StateInvariantError, validate_state

        state = _state(
            active_group="provider_deployment",
            pending_question={"id": "region", "group": "provider_deployment"},
            group_states={"provider_deployment": {"status": "completed"}},
        )
        with self.assertRaises(StateInvariantError):
            validate_state(state)

    def test_runtime_workload_overrides_do_not_mutate_template_defaults(self) -> None:
        from agent.harness.contracts import ActionProposal
        from agent.harness.domains.chain_rpc import apply_chain_rpc_action
        from agent.validators.rpc_workload import default_workload

        template_path = REPO_ROOT / "config" / "chains" / "bsc.json"
        file_before = template_path.read_bytes()
        defaults_before = deepcopy(default_workload("bsc"))
        state = _state(
            rpc_mode="mixed",
            chain_identity={"canonical": "bsc", "status": "confirmed"},
        )
        try:
            defaults_result = apply_chain_rpc_action(
                state,
                ActionProposal("defaults", "use_default_workload", {}, "high"),
            )
        except Exception as exc:
            self.fail(f"default workload action raised {type(exc).__name__}: {exc}")
        self.assertFalse(defaults_result.blocker)
        defaults_state = _commit_result(state, defaults_result, owner="chain_rpc")
        override_result = apply_chain_rpc_action(
            defaults_state,
            ActionProposal("override", "configure_workload_weights", {}, "high"),
        )
        self.assertFalse(override_result.blocker)
        override_state = _commit_result(defaults_state, override_result, owner="chain_rpc")
        self.assertTrue((override_state.get("custom_rpc") or {}).get("job_local_override"))
        self.assertEqual(default_workload("bsc"), defaults_before)
        self.assertEqual(template_path.read_bytes(), file_before)

    def test_planner_snapshot_includes_rpc_draft_and_secondary_handoff(self) -> None:
        from agent.harness.context import workflow_snapshot

        state = _state(
            custom_rpc={"status": "needs_schema_evidence", "catalog": {"revision": 3}},
            endpoint_evidence={"candidate_endpoint_ready": True},
            secondary_handoff={"status": "collecting_evidence", "evidence": ["official docs"]},
        )

        snapshot = workflow_snapshot(state)

        self.assertEqual(snapshot["custom_rpc"]["catalog"]["revision"], 3)
        self.assertTrue(snapshot["endpoint_evidence"]["candidate_endpoint_ready"])
        self.assertEqual(snapshot["secondary_handoff"]["evidence"], ["official docs"])

    def test_planner_snapshot_projects_framework_inventory_to_aggregate_facts(self) -> None:
        from agent.harness.context import workflow_snapshot

        state = _state(
            framework_summary={
                "chain_count": 36,
                "family_count": 6,
                "unique_rpc_method_count": 109,
                "families": {"jsonrpc": 16, "rest": 5},
                "chains": [{"chain": "bsc", "methods": ["eth_blockNumber"]}],
                "run_modes": [{"id": "rpc_benchmark", "purpose": "long prose"}],
            },
            web_research={
                "status": "available",
                "provider": "google_search",
                "available": True,
                "results": [{"title": "irrelevant planner evidence"}],
            },
        )

        snapshot = workflow_snapshot(state)

        self.assertEqual(snapshot["framework_summary"], {
            "chain_count": 36,
            "family_count": 6,
            "unique_rpc_method_count": 109,
            "adapter_families": ["jsonrpc", "rest"],
        })
        self.assertNotIn("chains", snapshot["framework_summary"])
        self.assertEqual(snapshot["web_research"], {
            "status": "available",
            "provider": "google_search",
            "available": True,
        })

    def test_case3_domain_rejects_rpc_catalog_mutation(self) -> None:
        from agent.harness.contracts import ActionProposal
        from agent.harness.domains.chain_rpc import apply_chain_rpc_action

        state = _state(chain_identity={
            "canonical": "WeirdP2PChain",
            "adapter_family": "unsupported",
            "case": "case3",
        })
        result = apply_chain_rpc_action(
            state,
            ActionProposal(
                "wrong-domain",
                "rpc_catalog_command",
                {"catalog_command": "set_method", "rpc_method": "method_id"},
                "high",
            ),
        )

        self.assertIsNotNone(result.blocker)
        self.assertEqual(result.blocker.code, "chain_rpc.failure.invalid_operation")
        self.assertEqual(
            result.blocker.arguments,
            {"operation": "rpc_catalog_for_unsupported_family"},
        )
        self.assertEqual(state["chain_identity"]["case"], "case3")

    def test_chain_selection_rejects_status_word_not_present_in_evidence(self) -> None:
        from agent.harness.action_registry import validate_action_contract

        with self.assertRaisesRegex(ValueError, "source_evidence"):
            validate_action_contract({
                "type": "choose_chain",
                "chain_text": "unknown",
                "source_evidence": "called LocalEvmDemo",
                "confidence": "high",
            })

        accepted = validate_action_contract({
            "type": "choose_chain",
            "chain_text": "LocalEvmDemo",
            "source_evidence": "called LocalEvmDemo",
            "confidence": "high",
        })
        self.assertEqual(accepted["chain_text"], "LocalEvmDemo")

    def test_sync_observe_rejects_rpc_catalog_until_ordered_mode_change(self) -> None:
        from agent.harness.action_registry import lifecycle_rejected_action_indexes

        state = {"target_mode": "sync-observe", "workflow_mode": "sync_observe"}
        catalog = {
            "type": "rpc_catalog_command",
            "catalog_command": "set_endpoint",
            "rpc_endpoint": "http://geth-dev:8545",
        }
        self.assertEqual(lifecycle_rejected_action_indexes(state, [catalog]), (0,))

        ordered = [
            {
                "type": "choose_target_mode",
                "target_mode": "real-node",
                "target_mode_explicit": True,
                "source_evidence": "switch to real-node",
            },
            catalog,
        ]
        self.assertEqual(lifecycle_rejected_action_indexes(state, ordered), ())

        reversed_order = [catalog, ordered[0]]
        self.assertEqual(lifecycle_rejected_action_indexes(state, reversed_order), (0,))

    def test_sync_observe_catalog_plan_fails_closed_before_dispatch(self) -> None:
        from agent.harness.admission import validate_action_plan
        from agent.harness.state import new_state

        state = new_state("sync-catalog-boundary", language="en")
        state["target_mode"] = "sync-observe"
        state["workflow_mode"] = "sync_observe"
        state["pending_question"] = {
            "contract_version": 1,
            "id": "SYNC_OBSERVE_RPC_URL",
            "group": "endpoint_process",
            "kind": "url",
            "field": "SYNC_OBSERVE_RPC_URL",
            "manual_input_allowed": True,
            "accepted_action_types": ["answer_pending"],
            "options": [],
            "validation": {},
        }
        result = validate_action_plan(state, [{
                "type": "rpc_catalog_command",
                "catalog_command": "set_endpoint",
                "rpc_endpoint": "http://geth-dev:8545",
                "source_evidence": "http://geth-dev:8545",
        }])
        self.assertEqual(result.status, "rejected")
        self.assertEqual(
            [item.code for item in result.rejections],
            ["lifecycle_incompatible"],
        )

    def test_replayed_confirmed_adapter_family_selection_is_idempotent(self) -> None:
        from agent.harness.contracts import ActionProposal
        from agent.harness.domains.chain_rpc import apply_chain_rpc_action
        from agent.harness.state import new_state

        state = new_state("adapter-family-idempotence", language="en")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["active_group"] = "endpoint_process"
        state["chain_identity"] = {
            "raw": "Flow-EVM",
            "canonical": "Flow-EVM",
            "adapter_family": "jsonrpc",
            "status": "existing_family_needs_endpoint",
            "case": "case2",
            "identity_confirmed": True,
        }

        result = apply_chain_rpc_action(
            state,
            ActionProposal(
                "family-replay",
                "choose_adapter_family",
                {"adapter_family": "jsonrpc"},
                "high",
            ),
        )

        committed = _commit_result(state, result, owner="chain_rpc")
        self.assertEqual(result.completion, "unchanged")
        self.assertEqual(committed["chain_identity"], state["chain_identity"])
        self.assertEqual(committed["active_group"], "endpoint_process")

if __name__ == "__main__":
    unittest.main()

"""Architecture and product-contract tests for the rebuilt Agent Harness."""

from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable
from unittest.mock import Mock, patch


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


def _whole_plan_admission_payload(request: Any) -> dict[str, Any]:
    """Build one valid immutable verdict from the actual reviewer request."""

    review = json.loads(request.messages[1].content)
    units = {
        str(row["unit_id"]): row
        for row in review["semantic_units"]
    }
    action_verdicts = []
    for action in review["actions"]:
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
        "plan_hash": review["plan_hash"],
        "action_verdicts": action_verdicts,
        "unit_verdicts": unit_verdicts,
        "reason": "the complete immutable plan is admitted",
    }


def _whole_plan_admission_response(request: Any) -> SimpleNamespace:
    return SimpleNamespace(text=json.dumps(
        _whole_plan_admission_payload(request),
        ensure_ascii=False,
        sort_keys=True,
    ))


def _bounded_intent_provider(
    compiler_documents: list[str | dict[str, Any]],
    admission_responses: list[Callable[[Any], SimpleNamespace]] | None = None,
) -> Mock:
    """Return a provider whose compiler and admission calls share one channel."""

    provider = Mock()
    documents = iter(compiler_documents)
    admissions = iter(admission_responses or [])

    def complete(request: Any) -> SimpleNamespace:
        request_payload = json.loads(request.messages[1].content)
        if "plan_hash" in request_payload and "immutable_document" in request_payload:
            builder = next(admissions, _whole_plan_admission_response)
            return builder(request)
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


class BoundedSemanticAdmissionTest(unittest.TestCase):
    def _validate(self, payload: dict[str, Any] | str):
        from agent.harness.intent import ALLOWED_ACTION_TYPES
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

    def test_exact_quote_and_plan_hash_grounding_fail_closed(self) -> None:
        _plan, valid = _immutable_admission_fixture()
        invented = deepcopy(valid)
        invented["action_verdicts"][0]["evidence"][0]["quote"] = "invented"
        invented["unit_verdicts"][0]["evidence_quote"] = "invented"
        self.assertFalse(self._validate(invented).valid)

        forged_hash = deepcopy(valid)
        forged_hash["plan_hash"] = "forged"
        result = self._validate(forged_hash)
        self.assertFalse(result.valid)
        self.assertIn("plan_hash mismatch", "; ".join(result.errors))

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
        self.assertIn("not complete", "; ".join(result.errors))


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
    from agent.harness.intent import _attach_semantic_admission_receipts

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
    turn_index = int(state.get("turn_index") or 0)
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

    def test_scalar_reconfiguration_supersedes_same_group_navigation(self) -> None:
        from agent.harness.action_registry import normalize_action_relations

        actions = normalize_action_relations([
            {
                "type": "change_group",
                "group": "provider_deployment",
                "navigation_explicit": True,
                "source_evidence": "go to provider and change the region",
            },
            {
                "type": "request_config_field_input",
                "config_field": "CLOUD_REGION",
                "source_evidence": "change the region",
            },
        ])
        self.assertEqual(
            [item["type"] for item in actions],
            ["request_config_field_input"],
        )


    def test_scalar_reconfiguration_opens_exact_question_without_copying_state(self) -> None:
        from agent.harness.coordinator import _apply_queue_action

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
                result = _apply_queue_action(state, action, f"change {field}")
                self.assertIsNotNone(result)
                assert result is not None
                self.assertEqual(result["active_group"], group)
                self.assertEqual(result["pending_question"]["id"], question_id)
                self.assertEqual(result["pending_question"]["field"], field)
                self.assertEqual(result["confirmed_config"][field], current_value)
                self.assertNotIn(current_value, result["visible_response"][-1])

    def test_scalar_reconfiguration_respects_environment_prerequisites(self) -> None:
        from agent.harness.coordinator import _apply_pending_answer, _apply_queue_action

        state = _state(
            active_group="opening",
            confirmed_config={"has_accounts_device": False},
        )
        source = "change ACCOUNTS_VOL_MAX_IOPS"
        action = _admitted_field_intake_action(state, "ACCOUNTS_VOL_MAX_IOPS", source)
        action["action_id"] = "edit-accounts-iops"
        result = _apply_queue_action(state, action, "change ACCOUNTS_VOL_MAX_IOPS")
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["active_group"], "accounts_disk")
        self.assertEqual(result["pending_question"]["id"], "has_accounts_device")
        self.assertEqual(
            result["pending_question"]["reconfiguration_target_field"],
            "ACCOUNTS_VOL_MAX_IOPS",
        )
        self.assertNotIn("ACCOUNTS_VOL_MAX_IOPS", result["confirmed_config"])
        resumed = _apply_pending_answer(
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
            resumed = _apply_pending_answer(
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

    def test_chain_rpc_invalidations_commit_cross_domain_state_once_at_coordinator(self) -> None:
        from agent.harness.domains.chain_rpc_support import _domain_result
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
                    _domain_result(original, changed),
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
        from agent.harness.domains.chain_rpc import apply_chain_rpc_answer
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
        endpoint_question = {
            "id": "LOCAL_RPC_URL",
            "group": "endpoint_process",
            "field": "LOCAL_RPC_URL",
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
        from agent.harness.coordinator import plan_turn_step

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

        with patch("agent.harness.coordinator.resolve_action_queue", return_value=rejected):
            result = plan_turn_step(state)

        self.assertEqual(
            [item.get("type") for item in result.get("proposed_actions") or []],
            ["clarify_unresolved"],
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

    def test_upstream_mutation_rejects_answer_bound_to_stale_pending_group(self) -> None:
        from agent.harness.coordinator import _validate_action_plan

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
        actions = _validate_action_plan(state, [
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
                "type": "change_chain",
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

        self.assertNotIn("answer_pending", [item.get("type") for item in actions])
        self.assertIn("choose_target_mode", [item.get("type") for item in actions])
        self.assertIn("change_chain", [item.get("type") for item in actions])
        self.assertIn("propose_config_values", [item.get("type") for item in actions])

    def test_model_action_envelope_normalizes_nested_arguments_at_boundary(self) -> None:
        from agent.harness.action_registry import normalize_action_envelope

        normalized = normalize_action_envelope({
            "type": "answer_opening_question",
            "confidence": "high",
            "arguments": {"topic": "current_config", "subject": "workload"},
        })

        self.assertEqual(normalized["topic"], "current_config")
        self.assertEqual(normalized["subject"], "workload")
        self.assertNotIn("arguments", normalized)

    def test_nested_arguments_cannot_override_action_identity(self) -> None:
        from agent.harness.action_registry import normalize_action_envelope

        normalized = normalize_action_envelope({
            "type": "set_qps_mode",
            "arguments": {"type": "reset_session", "qps_mode": "quick"},
        })

        self.assertEqual(normalized["type"], "set_qps_mode")
        self.assertEqual(normalized["qps_mode"], "quick")

    def test_workload_consultation_is_specific_and_non_mutating(self) -> None:
        from agent.harness.domains.orientation import answer_consultation

        state = _state(
            chain_identity={"canonical": "bsc", "status": "confirmed"},
            rpc_mode="mixed",
        )
        response = answer_consultation(state, {"topic": "workload_config"})

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
            if "except ImportError" in source or "except ModuleNotFoundError" in source:
                offenders.append(str(path.relative_to(REPO_ROOT)))
        self.assertEqual(offenders, [])

    def test_free_form_intent_has_one_llm_entry_point(self) -> None:
        from agent.harness import intent

        self.assertTrue(callable(intent.resolve_action_queue))
        self.assertFalse(hasattr(intent, "resolve_pending_choice"))

    def test_coordinator_cannot_bypass_the_compiled_product_graph(self) -> None:
        from agent.harness import coordinator

        self.assertFalse(hasattr(coordinator, "process_turn"))
        self.assertFalse(hasattr(coordinator, "_process_turn"))
        self.assertFalse(hasattr(coordinator, "_route_free_text"))

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
            "change_chain": {"chain_text": "bsc"},
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
        with patch("agent.harness.domains.execution_runtime.execution_service.execute"):
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
                    self.assertNotIn("unsupported", result.blocker.casefold())

    def test_langgraph_has_explicit_control_plane_nodes(self) -> None:
        from agent.harness import graph as graph_module

        source = (REPO_ROOT / "agent/harness/graph.py").read_text(encoding="utf-8")
        for node in ("prepare", "adjudicate", "plan", "admit", "execute", "fallback", "compose", "validate"):
            self.assertIn(f'graph.add_node("{node}"', source)
        self.assertNotIn('graph.add_node("turn"', source)
        self.assertFalse(hasattr(graph_module, "process_turn"))

    def test_single_planner_contract_keeps_consultations_independent_of_pending(self) -> None:
        from agent.harness.context import build_action_resolver_prompt

        prompt = build_action_resolver_prompt()
        self.assertIn("classify every unit independently", prompt)
        self.assertIn("the Harness derives them", prompt)
        self.assertIn("must not omit prose", prompt)
        self.assertIn("consultation_only", prompt)
        self.assertIn("no_configuration_mutation", prompt)
        self.assertIn("action_schema.effect", prompt)
        self.assertIn("A pending question never takes precedence", prompt)
        self.assertIn("current_config/current_context/next_action", prompt)

    def test_group_specs_own_dependencies_and_invalidation_metadata(self) -> None:
        from agent.workflows.group_registry import GROUP_SPEC_BY_NAME

        self.assertIn("target_mode", GROUP_SPEC_BY_NAME["chain_identity"].depends_on)
        self.assertIn("workload_rpc", GROUP_SPEC_BY_NAME["chain_identity"].invalidates)
        self.assertIn("preflight_smoke_execution", GROUP_SPEC_BY_NAME["qps_profile"].invalidates)

    def test_rpc_catalog_semantic_purpose_describes_the_selected_operation(self) -> None:
        from agent.harness.intent import _semantic_action_purpose

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
        self.assertIn("LocalEvmDemo", question["prompt"])
        self.assertIn("ethereum", question["prompt"])

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

    def test_overlimit_scalar_is_rejected_without_semantic_planning(self) -> None:
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
            "agent.harness.coordinator.resolve_action_queue",
            side_effect=AssertionError("semantic planner must not receive an invalid scalar"),
        ):
            result = invoke_product_graph_turn(state)

        self.assertEqual(result["pending_question"]["id"], "RPC_API_KEY")
        self.assertNotIn("RPC_API_KEY", result.get("confirmed_config") or {})
        self.assertNotIn(
            "clarify_unresolved",
            [item.get("type") for item in result.get("proposed_actions") or []],
        )

    def test_manual_numeric_choice_rejects_invalid_replacement_without_planning(self) -> None:
        from agent.harness.questions import choice_question, manual_literal_violation

        question = choice_question(
            "ledger_disk",
            "DATA_VOL_SIZE",
            "Use the detected size?",
            field="DATA_VOL_SIZE",
            kind="yes_no",
            manual_input_allowed=True,
            validation={"value_type": "positive_number"},
            options=[
                {"label": "Y", "value": "926"},
                {"label": "N", "value": "__manual__"},
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
            value_satisfies_pending_contract,
        )

        question = manual_question(
            "chain_identity",
            "protocol_evidence",
            "Provide protocol evidence.",
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
        from agent.harness.coordinator import _finalize_turn_response
        from agent.harness.questions import manual_question, render_question

        endpoint = manual_question(
            "endpoint_process",
            "new_chain_endpoint",
            "Provide a reachable validation endpoint.",
            field="new_chain_endpoint",
            kind="url",
        )
        schema = manual_question(
            "endpoint_process",
            "new_chain_schema_confirm",
            "Confirm the extracted request contract.",
            field="new_chain_schema_confirm",
            kind="yes_no",
        )
        state = _state("en")
        state["active_group"] = "endpoint_process"
        state["turn_context"] = {
            "installed_questions": [endpoint, schema],
        }
        state["pending_question"] = schema
        state["visible_response"] = [
            "Endpoint validation passed. Evidence: probe.json.",
            render_question(endpoint, "en"),
            render_question(schema, "en"),
        ]

        result = _finalize_turn_response(state)

        rendered = result["visible_response"]
        self.assertIn("Endpoint validation passed. Evidence: probe.json.", rendered)
        self.assertNotIn(render_question(endpoint, "en"), rendered)
        self.assertEqual(rendered.count(render_question(schema, "en")), 1)

    def test_suspended_earlier_question_precedes_later_derived_question(self) -> None:
        from agent.harness.coordinator import _ask_next_blocking_question
        from agent.harness.domains.chain_rpc import question_for_chain_rpc
        from agent.harness.domains.performance import question_for_performance

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
        state["visible_response"] = [
            question_for_performance(state, "qps_profile")["prompt"]
            + "\n1. Y\n2. N",
        ]

        result = _ask_next_blocking_question(state)

        self.assertEqual(result["active_group"], "chain_identity")
        self.assertEqual(result["pending_question"]["id"], "adapter_family_confirm")
        rendered = "\n".join(result["visible_response"])
        self.assertIn("adapter family", rendered)
        self.assertNotIn("Default QPS profile", rendered)

    def test_generic_manual_question_has_bounded_scalar_contract(self) -> None:
        from agent.harness.questions import literal_matches_validation, manual_question

        question = manual_question(
            "endpoint_process",
            "process_name",
            "Enter process name.",
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

    def test_model_pending_admission_uses_extracted_value_contract(self) -> None:
        from agent.harness.coordinator import (
            _action_answers_pending_contract,
            _dispatch_pending_action,
        )
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
        process_result = _dispatch_pending_action(process_state, process_action)
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
        region_result = _dispatch_pending_action(region_state, region_action)
        self.assertEqual(region_result["confirmed_config"]["CLOUD_REGION"], "us-central1")

        detected_region_action = {
            "type": "answer_pending",
            "answer": "asia-east1",
            "source_evidence": "set CLOUD_REGION to asia-east1",
            "semantic_purpose_verified": True,
            "pending_option_semantic_verified": True,
        }
        detected_region_result = _dispatch_pending_action(
            region_state,
            detected_region_action,
        )
        self.assertEqual(
            detected_region_result["confirmed_config"]["CLOUD_REGION"],
            "asia-east1",
        )
        self.assertEqual(detected_region_result["pending_question"], {})

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
        normalized_result = _dispatch_pending_action(normalized_state, normalized_action)
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
                                "resolution": {
                                    "chain_exists": False,
                                    "possible_known_chain": "solana",
                                    "confidence": "high",
                                },
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
                            "change_chain",
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
        from agent.harness.questions import render_question

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
                    self.assertEqual(question.get("contract_version"), 1)
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
                    self.assertIn(str(question.get("prompt") or ""), rendered)
                    for option in question["options"]:
                        self.assertIn(str(option.get("label") or option.get("value")), rendered)
                    if question.get("manual_input_allowed"):
                        expected_instruction = (
                            "你可以回复编号，也可以直接输入自定义值。"
                            if language == "zh"
                            else "Reply with a number, or type a custom value directly."
                        )
                    else:
                        expected_instruction = (
                            "请回复选项编号或选项名称。"
                            if language == "zh"
                            else "Reply with an option number or option name."
                        )
                    self.assertIn(expected_instruction, rendered)
                    if language == "zh":
                        self.assertNotIn("Reply with ", rendered)
                    else:
                        self.assertNotIn("请回复", rendered)
                        self.assertNotIn("你可以回复", rendered)
                self.assertNotEqual(render_question(english, "en"), render_question(chinese, "zh"))

    def test_question_renderer_requires_explicit_language(self) -> None:
        from agent.harness.questions import choice_question, render_question

        question = choice_question(
            "opening",
            "explicit_language_contract",
            "Choose one.",
            field="choice",
            options=[{"id": "one", "label": "one", "value": "one"}],
        )

        with self.assertRaises(TypeError):
            render_question(question)  # type: ignore[call-arg]

    def test_option_description_is_visible_semantic_context_not_an_exact_alias(self) -> None:
        from agent.harness.questions import choice_question, exact_answer, render_question

        description = "Fast framework validation with recorded fixtures."
        question = choice_question(
            "opening",
            "described_option_contract",
            "Choose one.",
            field="target_mode",
            options=[{
                "id": "fake-node",
                "label": "Start fake-node",
                "description": description,
                "value": "fake-node",
                "action": {"type": "choose_target_mode", "target_mode": "fake-node"},
            }],
        )

        self.assertEqual(question["options"][0]["description"], description)
        self.assertIn(description, render_question(question, "en"))
        self.assertEqual(exact_answer("fake-node", question), (True, "fake-node"))
        self.assertEqual(exact_answer(description, question), (False, None))

    def test_opening_options_explain_product_effect_in_both_languages(self) -> None:
        from agent.harness.domains.orientation import opening_question
        from agent.harness.questions import render_question

        for language in ("en", "zh"):
            with self.subTest(language=language):
                question = opening_question(_state(language))
                descriptions = [
                    str(option.get("description") or "").strip()
                    for option in question["options"]
                ]
                self.assertEqual(len(descriptions), 4)
                self.assertTrue(all(descriptions))
                rendered = render_question(question, language)
                for description in descriptions:
                    self.assertIn(description, rendered)

    def test_manual_choice_question_has_complete_validation_at_construction(self) -> None:
        from agent.harness.questions import choice_question

        question = choice_question(
            "chain_auxiliary_endpoints",
            "RPC_API_KEY",
            "Provide the API key or skip.",
            field="RPC_API_KEY",
            kind="manual_value",
            manual_input_allowed=True,
            options=[{"id": "skip", "label": "Skip", "value": "none"}],
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
    def test_free_form_planner_uses_exactly_two_provider_calls(self) -> None:
        from agent.harness.intent import resolve_action_queue

        text = "What can you do?"
        provider = _bounded_intent_provider([
            _single_action_document(
                {
                    "type": "answer_opening_question",
                    "topic": "capabilities",
                    "source_evidence": text,
                    "confidence": "high",
                },
                text,
            ),
        ])
        with patch("agent.harness.intent.provider_from_config", return_value=provider):
            result = resolve_action_queue(_state(), text)

        self.assertEqual(provider.complete.call_count, 2)
        self.assertEqual(result["actions"][0]["type"], "answer_opening_question")
        reviewer_request = provider.complete.call_args_list[1].args[0]
        immutable = json.loads(reviewer_request.messages[1].content)
        self.assertEqual(immutable["immutable_document"]["actions"][0]["type"], "answer_opening_question")
        self.assertTrue(immutable["plan_hash"])

    def test_structural_failure_uses_one_repair_and_stays_below_four_calls(self) -> None:
        from agent.harness.intent import resolve_action_queue

        text = "What can you do?"
        provider = _bounded_intent_provider([
            _single_action_document({"type": "not_registered"}, text),
            _single_action_document(
                {
                    "type": "answer_opening_question",
                    "topic": "capabilities",
                    "source_evidence": text,
                },
                text,
            ),
        ])
        with patch("agent.harness.intent.provider_from_config", return_value=provider):
            result = resolve_action_queue(_state(), text)

        self.assertEqual(provider.complete.call_count, 3)
        self.assertLessEqual(provider.complete.call_count, 4)
        system_prompts = [
            call.args[0].messages[0].content
            for call in provider.complete.call_args_list
        ]
        self.assertEqual(
            sum(prompt.startswith("Repair one malformed AnyChain typed action-plan response") for prompt in system_prompts),
            1,
        )
        self.assertEqual(result["actions"][0]["type"], "answer_opening_question")

    def test_rejected_admission_uses_one_repair_and_exactly_four_calls(self) -> None:
        from agent.harness.intent import resolve_action_queue

        text = "Before I paste evidence, summarize which chain and custom method you retained."
        invalid = _single_action_document(
            {
                "type": "rpc_catalog_command",
                "catalog_command": "append_evidence",
                "rpc_schema_evidence": text,
                "source_evidence": text,
            },
            text,
        )
        repaired = _single_action_document(
            {
                "type": "answer_opening_question",
                "topic": "current_config",
                "source_evidence": text,
            },
            text,
        )

        def reject(request: Any) -> SimpleNamespace:
            payload = _whole_plan_admission_payload(request)
            payload["action_verdicts"][0]["verdict"] = "reject"
            payload["action_verdicts"][0]["reason"] = "the immutable mutation does not answer the consultation"
            payload["unit_verdicts"][0].update({
                "verdict": "omitted",
                "omitted_action_type": "answer_opening_question",
                "reason": "a registry-expressible consultation is omitted",
            })
            return SimpleNamespace(text=json.dumps(payload, sort_keys=True))

        provider = _bounded_intent_provider(
            [invalid, repaired],
            [reject, _whole_plan_admission_response],
        )
        with patch("agent.harness.intent.provider_from_config", return_value=provider):
            result = resolve_action_queue(_state(), text)

        self.assertEqual(provider.complete.call_count, 4)
        self.assertEqual(result["actions"][0]["type"], "answer_opening_question")
        self.assertEqual(result["actions"][0]["topic"], "current_config")

    def test_final_malformed_admission_stops_at_four_calls(self) -> None:
        from agent.harness.intent import resolve_action_queue

        text = "Explain the current configuration."
        document = _single_action_document(
            {
                "type": "answer_opening_question",
                "topic": "current_config",
                "source_evidence": text,
            },
            text,
        )

        def reject(request: Any) -> SimpleNamespace:
            payload = _whole_plan_admission_payload(request)
            payload["action_verdicts"][0]["verdict"] = "reject"
            payload["action_verdicts"][0]["reason"] = "force the one bounded repair"
            return SimpleNamespace(text=json.dumps(payload, sort_keys=True))

        provider = _bounded_intent_provider(
            [document, document],
            [reject, lambda _request: SimpleNamespace(text='{"plan_hash":"forged"}')],
        )
        with patch("agent.harness.intent.provider_from_config", return_value=provider):
            result = resolve_action_queue(_state(), text)

        self.assertEqual(provider.complete.call_count, 4)
        self.assertEqual(result["actions"][0]["type"], "clarify_unresolved")

    def test_pending_option_keeps_single_rpc_effect_behind_canonical_contract(self) -> None:
        from agent.harness.domains.chain_rpc import question_for_chain_rpc
        from agent.harness.intent import resolve_action_queue

        state = _state(
            active_group="workload_rpc",
            target_mode="fake-node",
            workflow_mode="rpc_benchmark",
            chain_identity={"raw": "bsc", "canonical": "bsc", "status": "confirmed", "case": "case1"},
            confirmed_config={"BLOCKCHAIN_NODE": "bsc"},
        )
        state["pending_question"] = question_for_chain_rpc(state, "workload_rpc") or {}
        text = "I only need one RPC method."
        provider = _bounded_intent_provider([
            _single_action_document(
                {
                    "type": "answer_pending",
                    "answer": "single",
                    "selected_value": "single",
                    "source_evidence": text,
                },
                text,
            ),
        ])
        with patch("agent.harness.intent.provider_from_config", return_value=provider):
            result = resolve_action_queue(state, text)

        self.assertEqual(provider.complete.call_count, 2)
        self.assertEqual(result["actions"][0]["type"], "answer_pending")
        self.assertEqual(result["actions"][0]["selected_value"], "single")
        self.assertTrue(result["actions"][0]["pending_option_semantic_verified"])
        self.assertEqual(result["pending_choice_contracts"][0]["option"]["selected_value"], "single")

    def test_pending_owner_action_is_normalized_to_canonical_mixed_choice(self) -> None:
        from agent.harness.domains.chain_rpc import question_for_chain_rpc
        from agent.harness.intent import resolve_action_queue

        state = _state(
            active_group="workload_rpc",
            target_mode="fake-node",
            workflow_mode="rpc_benchmark",
            chain_identity={"raw": "bsc", "canonical": "bsc", "status": "confirmed", "case": "case1"},
            confirmed_config={"BLOCKCHAIN_NODE": "bsc"},
        )
        state["pending_question"] = question_for_chain_rpc(state, "workload_rpc") or {}
        text = "Use several weighted RPC methods."
        provider = _bounded_intent_provider([
            _single_action_document(
                {
                    "type": "set_rpc_mode",
                    "rpc_mode": "mixed",
                    "mutation_explicit": True,
                    "source_evidence": text,
                },
                text,
            ),
        ])
        with patch("agent.harness.intent.provider_from_config", return_value=provider):
            result = resolve_action_queue(state, text)

        self.assertEqual(provider.complete.call_count, 2)
        self.assertEqual(result["actions"][0]["type"], "answer_pending")
        self.assertEqual(result["actions"][0]["selected_value"], "mixed")
        self.assertTrue(result["actions"][0]["pending_option_semantic_verified"])
        self.assertEqual(result["pending_choice_contracts"][0]["option"]["selected_value"], "mixed")

    def test_pending_manual_value_is_typed_grounded_and_admitted(self) -> None:
        from agent.harness.intent import resolve_action_queue

        state = _state(
            active_group="provider_deployment",
            pending_question={
                "id": "CLOUD_REGION",
                "group": "provider_deployment",
                "kind": "manual_value",
                "field": "CLOUD_REGION",
                "manual_input_allowed": True,
                "options": [],
                "accepted_action_types": ["answer_pending"],
                "validation": {"value_type": "scalar_token"},
            },
        )
        text = "Use us-east1 for this run."
        provider = _bounded_intent_provider([
            _single_action_document(
                {
                    "type": "answer_pending",
                    "answer": "us-east1",
                    "source_evidence": "us-east1",
                },
                text,
            ),
        ])
        with patch("agent.harness.intent.provider_from_config", return_value=provider):
            result = resolve_action_queue(state, text)

        self.assertEqual(provider.complete.call_count, 2)
        self.assertEqual(result["actions"][0]["type"], "answer_pending")
        self.assertEqual(result["actions"][0]["answer"], "us-east1")
        self.assertTrue(result["actions"][0]["pending_option_semantic_verified"])
        self.assertTrue(result["actions"][0]["semantic_purpose_verified"])

    def test_unresolved_candidate_fails_closed_without_unit_retries(self) -> None:
        from agent.harness.intent import resolve_action_queue

        text = "Choose whichever workload is suitable."
        unresolved = _single_action_document(
            {
                "type": "clarify_unresolved",
                "clauses": [text],
            },
            text,
            disposition="unresolved",
        )
        provider = _bounded_intent_provider([unresolved, unresolved])
        with patch("agent.harness.intent.provider_from_config", return_value=provider):
            result = resolve_action_queue(_state(), text)

        self.assertEqual(provider.complete.call_count, 2)
        self.assertEqual(result["actions"][0]["type"], "clarify_unresolved")

    def test_rpc_mode_comparison_remains_read_only(self) -> None:
        from agent.harness.domains.chain_rpc import question_for_chain_rpc
        from agent.harness.intent import resolve_action_queue

        state = _state(
            active_group="workload_rpc",
            target_mode="fake-node",
            workflow_mode="rpc_benchmark",
            chain_identity={"raw": "bsc", "canonical": "bsc", "status": "confirmed", "case": "case1"},
            confirmed_config={"BLOCKCHAIN_NODE": "bsc"},
        )
        state["pending_question"] = question_for_chain_rpc(state, "workload_rpc") or {}
        text = "How do single and mixed differ?"
        provider = _bounded_intent_provider([
            _single_action_document(
                {
                    "type": "answer_opening_question",
                    "topic": "mode_comparison",
                    "source_evidence": text,
                },
                text,
            ),
        ])
        with patch("agent.harness.intent.provider_from_config", return_value=provider):
            result = resolve_action_queue(state, text)

        self.assertEqual(provider.complete.call_count, 2)
        self.assertEqual(result["actions"][0]["type"], "answer_opening_question")
        self.assertFalse(any(action.get("type") == "set_rpc_mode" for action in result["actions"]))

    def test_payload_free_custom_rpc_entry_uses_two_calls(self) -> None:
        from agent.harness.intent import resolve_action_queue

        text = "I need to supply my own RPC method instead of the template defaults."
        provider = _bounded_intent_provider([
            _single_action_document(
                {
                    "type": "rpc_catalog_command",
                    "catalog_command": "enter",
                },
                text,
            ),
        ])
        with patch("agent.harness.intent.provider_from_config", return_value=provider):
            result = resolve_action_queue(_state(), text)

        self.assertEqual(provider.complete.call_count, 2)
        self.assertEqual(result["actions"][0]["type"], "rpc_catalog_command")
        self.assertEqual(result["actions"][0]["catalog_command"], "enter")

    def test_provider_call_count_is_constant_for_one_through_twenty_units(self) -> None:
        from agent.harness.intent import resolve_action_queue
        from agent.harness.plan_coverage import segment_user_turn

        for count in range(1, 21):
            with self.subTest(unit_count=count):
                text = " ".join(f"Question {index}?" for index in range(1, count + 1))
                clauses = segment_user_turn(text)
                self.assertEqual(len(clauses), count)
                document = {
                    "actions": [{
                        "type": "answer_opening_question",
                        "topic": "capabilities",
                        "source_evidence": clauses[0].text,
                    }],
                    "semantic_units": [
                        {
                            "unit_id": f"unit-{index}",
                            "clause_id": clause.clause_id,
                            "source_text": clause.text,
                            "disposition": "action",
                            "action_indexes": [0],
                            "reason": "one complete consultation",
                        }
                        for index, clause in enumerate(clauses, start=1)
                    ],
                }
                provider = _bounded_intent_provider([document])
                with patch("agent.harness.intent.provider_from_config", return_value=provider):
                    result = resolve_action_queue(_state(), text)
                self.assertEqual(provider.complete.call_count, 2)
                self.assertEqual(result["actions"][0]["type"], "answer_opening_question")

    def test_provider_call_count_is_constant_for_one_through_twenty_actions(self) -> None:
        from agent.harness.intent import resolve_action_queue
        from agent.harness.plan_coverage import segment_user_turn

        for count in range(1, 21):
            with self.subTest(action_count=count):
                text = " ".join(f"Question {index}?" for index in range(1, count + 1))
                clauses = segment_user_turn(text)
                document = {
                    "actions": [
                        {
                            "type": "answer_opening_question",
                            "topic": "capabilities",
                            "subject": f"question-{index}",
                            "source_evidence": clause.text,
                        }
                        for index, clause in enumerate(clauses, start=1)
                    ],
                    "semantic_units": [
                        {
                            "unit_id": f"unit-{index}",
                            "clause_id": clause.clause_id,
                            "source_text": clause.text,
                            "disposition": "action",
                            "action_indexes": [index - 1],
                            "reason": "one independently owned consultation",
                        }
                        for index, clause in enumerate(clauses, start=1)
                    ],
                }
                provider = _bounded_intent_provider([document])
                with patch("agent.harness.intent.provider_from_config", return_value=provider):
                    result = resolve_action_queue(_state(), text)
                self.assertEqual(provider.complete.call_count, 2)
                self.assertEqual(len(result["actions"]), count)

    def test_runtime_resolver_has_no_old_serial_adjudicator_path(self) -> None:
        import inspect

        from agent.harness.intent import resolve_action_queue

        source = inspect.getsource(resolve_action_queue)
        obsolete_runtime_calls = {
            "_recover_omitted_chain_selection",
            "_recover_declared_pending_option_semantics",
            "_adjudicate_pending_action_ownership",
            "_adjudicate_group_navigation_actions",
            "_adjudicate_chain_selection_actions",
            "_adjudicate_target_mode_actions",
            "_adjudicate_consultation_actions",
            "_reconstruct_missing_semantic_units",
            "_validate_semantic_fulfillment",
            "_challenge_and_validate_registry_inventory",
            "_finalize_structured_syntax_authority",
        }
        for name in obsolete_runtime_calls:
            with self.subTest(helper=name):
                self.assertNotIn(name, source)

    def test_multiline_structured_and_prose_units_share_one_whole_plan_review(self) -> None:
        from agent.harness.intent import resolve_action_queue

        text = (
            "Apply these values:\n"
            "CLOUD_REGION=us-east1\n"
            "MACHINE_TYPE=n2-standard-8\n"
            "Then tell me what remains?"
        )
        provider = Mock()

        def complete(request: Any) -> SimpleNamespace:
            payload = json.loads(request.messages[1].content)
            if "plan_hash" in payload:
                return _whole_plan_admission_response(request)
            structured = payload["structured_candidates"][0]
            structured_clause_id = structured["clause_id"]
            actions = [
                {
                    "type": "propose_config_values",
                    "config_values": structured["config_values"],
                    "unmapped_values": structured.get("unmapped_values") or {},
                    "conflicts": [],
                    "source_format": structured["source_format"],
                    "source_evidence": next(
                        clause["text"]
                        for clause in payload["clauses"]
                        if clause["clause_id"] == structured_clause_id
                    ),
                },
                {
                    "type": "answer_opening_question",
                    "topic": "next_action",
                    "source_evidence": "Then tell me what remains?",
                },
            ]
            units = [
                {
                    "unit_id": f"unit-{index}",
                    "clause_id": clause["clause_id"],
                    "source_text": clause["text"],
                    "disposition": "action",
                    "action_indexes": [0] if clause["input_shape"] == "structured" else [1],
                    "reason": "structured configuration or follow-up consultation",
                }
                for index, clause in enumerate(payload["clauses"], start=1)
            ]
            return SimpleNamespace(text=json.dumps({
                "actions": actions,
                "semantic_units": units,
            }, sort_keys=True))

        provider.complete.side_effect = complete
        with patch("agent.harness.intent.provider_from_config", return_value=provider):
            result = resolve_action_queue(_state(), text)

        self.assertEqual(provider.complete.call_count, 2)
        self.assertEqual(
            [action["type"] for action in result["actions"]],
            ["propose_config_values", "answer_opening_question"],
        )
        self.assertEqual(
            result["actions"][0]["config_values"],
            {"CLOUD_REGION": "us-east1", "MACHINE_TYPE": "n2-standard-8"},
        )

    def test_group_jump_and_field_mutation_remain_one_immutable_ordered_plan(self) -> None:
        from agent.harness.intent import resolve_action_queue

        first = "Open the QPS configuration."
        second = "I also need to replace CLOUD_REGION."
        text = f"{first} {second}"
        provider = _bounded_intent_provider([{
            "actions": [
                {
                    "type": "change_group",
                    "group": "qps_profile",
                    "navigation_explicit": True,
                    "source_evidence": first,
                },
                {
                    "type": "request_config_field_input",
                    "config_field": "CLOUD_REGION",
                    "source_evidence": second,
                },
            ],
            "semantic_units": [
                {
                    "unit_id": "unit-1",
                    "clause_id": "clause-1",
                    "source_text": first,
                    "disposition": "action",
                    "action_indexes": [0],
                    "reason": "explicit group jump",
                },
                {
                    "unit_id": "unit-2",
                    "clause_id": "clause-2",
                    "source_text": second,
                    "disposition": "action",
                    "action_indexes": [1],
                    "reason": "registered field mutation intake",
                },
            ],
        }])
        with patch("agent.harness.intent.provider_from_config", return_value=provider):
            result = resolve_action_queue(_state(), text)

        self.assertEqual(provider.complete.call_count, 2)
        self.assertEqual(
            [action["type"] for action in result["actions"]],
            ["change_group", "request_config_field_input"],
        )
        self.assertTrue(result["actions"][0]["group_navigation_semantic_verified"])
        self.assertTrue(result["actions"][1]["semantic_purpose_verified"])


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
        from agent.harness.coordinator import _action_answers_pending_contract

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
        from agent.harness.coordinator import _action_answers_pending_contract
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

    def test_distinct_consultation_topics_are_not_dropped_after_coverage(self) -> None:
        from agent.harness.coordinator import _drop_conflicting_answer_actions

        actions = [
            {"type": "answer_opening_question", "topic": "current_config"},
            {"type": "answer_opening_question", "topic": "requirements"},
            {"type": "answer_opening_question", "topic": "workflow"},
            {"type": "answer_opening_question", "topic": "mode_comparison"},
        ]

        filtered = _drop_conflicting_answer_actions(_state(), actions)
        self.assertEqual(
            [item["topic"] for item in filtered],
            ["current_config", "requirements", "workflow", "mode_comparison"],
        )

    def test_config_review_overlay_restores_the_interrupted_typed_question(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn

        pending = {
            "contract_version": 1,
            "id": "custom_rpc_endpoint",
            "group": "endpoint_process",
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

        with patch("agent.harness.coordinator.resolve_action_queue", side_effect=resolved):
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

        migrated = migrate_state(
            {
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
        self.assertEqual([item["type"] for item in migrated["action_queue"]], ["answer_opening_question"])

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
                    "kind": "manual_value",
                },
            ),
            _state(
                active_group="network",
                action_queue=[{"action_id": "one"}, {"action_id": "two"}],
                group_history=["opening", "provider_deployment"],
            ),
            _state(
                active_group="network",
                pending_question={"id": "interface", "group": "network"},
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

    def test_report_analysis_owns_job_status_and_evidence_help_for_the_turn(self) -> None:
        from agent.harness.coordinator import _drop_conflicting_answer_actions

        actions = [
            {"type": "answer_opening_question", "topic": "current_job"},
            {"type": "answer_opening_question", "topic": "evidence_help"},
            {"type": "answer_opening_question", "topic": "extension"},
            {"type": "analyze_report", "subject": "what ran and what cannot be inferred"},
        ]
        pruned = _drop_conflicting_answer_actions(_state(), actions)
        topics = {str(item.get("topic") or "") for item in pruned}
        self.assertNotIn("current_job", topics)
        self.assertNotIn("evidence_help", topics)
        self.assertIn("extension", topics)
        self.assertTrue(any(item.get("type") == "analyze_report" for item in pruned))

    def test_case3_plan_admission_removes_rpc_catalog_sibling_and_preserves_coverage(self) -> None:
        import json

        from agent.harness.intent import _apply_state_plan_policy

        source = "Official docs: request envelope contains method_id."
        payload = {
            "actions": [
                {
                    "type": "secondary_handoff_command",
                    "handoff_command": "append_evidence",
                    "handoff_evidence": source,
                },
                {
                    "type": "rpc_catalog_command",
                    "catalog_command": "set_method",
                    "rpc_method": "method_id",
                },
            ],
            "semantic_units": [{
                "unit_id": "unit-1",
                "clause_id": "clause-1",
                "source_text": source,
                "disposition": "action",
                "action_indexes": [0, 1],
            }],
        }
        state = _state(chain_identity={
            "canonical": "WeirdP2PChain",
            "adapter_family": "unsupported",
            "case": "case3",
        })

        admitted = json.loads(_apply_state_plan_policy(json.dumps(payload), state))

        self.assertEqual([item["type"] for item in admitted["actions"]], ["secondary_handoff_command"])
        self.assertEqual(admitted["semantic_units"][0]["action_indexes"], [0])
        self.assertEqual(admitted["semantic_units"][0]["disposition"], "action")

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

        self.assertIn("supported adapter family", result.blocker)
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
        from agent.harness.coordinator import _validate_action_plan
        from agent.harness.invariants import StateInvariantError
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
        with self.assertRaisesRegex(StateInvariantError, "target-mode lifecycle"):
            _validate_action_plan(state, [{
                "type": "rpc_catalog_command",
                "catalog_command": "set_endpoint",
                "rpc_endpoint": "http://geth-dev:8545",
                "source_evidence": "http://geth-dev:8545",
            }])

    def test_intent_policy_removes_sync_observe_catalog_mutation_for_repair(self) -> None:
        import json

        from agent.harness.intent import _apply_state_plan_policy
        from agent.harness.state import new_state

        state = new_state("sync-intent-boundary", language="en")
        state["target_mode"] = "sync-observe"
        state["workflow_mode"] = "sync_observe"
        payload = {
            "actions": [{
                "type": "rpc_catalog_command",
                "catalog_command": "set_endpoint",
                "rpc_endpoint": "http://geth-dev:8545",
                "source_evidence": "Use http://geth-dev:8545 for sync observation.",
            }],
            "semantic_units": [{
                "unit_id": "u1",
                "clause_id": "c1",
                "source_text": "Use http://geth-dev:8545 for sync observation.",
                "disposition": "action",
                "action_indexes": [0],
                "reason": "endpoint selection",
            }],
        }

        result = json.loads(_apply_state_plan_policy(json.dumps(payload), state))

        self.assertEqual(result["actions"], [])
        self.assertEqual(result["semantic_units"][0]["disposition"], "unresolved")
        self.assertIn("target-mode lifecycle", result["semantic_units"][0]["reason"])


if __name__ == "__main__":
    unittest.main()

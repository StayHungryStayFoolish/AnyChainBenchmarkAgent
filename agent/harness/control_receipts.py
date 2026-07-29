"""Trusted admission boundary for domain-owned observation receipts."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping

from .domains.analysis_receipts import validate_analysis_receipt
from .domains.chain_identity_receipts import validate_chain_identity_receipt
from .domains.orientation_receipts import validate_orientation_receipt
from .domains.rpc_receipts import validate_rpc_receipt


_HANDLER_OWNER_BY_RECEIPT = {
    "analysis_evidence_block": "analysis",
    "analysis_invocation": "analysis",
    "analysis_report": "analysis",
    "chain_identity_resolution": "chain_rpc",
    "orientation_response": "orientation",
    "rpc_endpoint_role": "chain_rpc",
    "rpc_catalog_transition": "chain_rpc",
    "rpc_schema_provenance": "chain_rpc",
    "rpc_workload_commit": "chain_rpc",
    "rpc_workload_materialization": "execution",
}

_COORDINATOR_RECEIPT_TYPES = frozenset({
    "pending_resolution",
    "semantic_partition",
    "owner_compilation",
    "whole_plan_review",
    "semantic_planner",
    "fallback_selection",
    "response_composition",
    "domain_commit",
    "execution_approval",
})
_SHA256_LENGTH = 64
_EMPTY_DOCUMENT_HASH = hashlib.sha256(b"{}").hexdigest()
_EMPTY_ERRORS_HASH = hashlib.sha256(b"[]").hexdigest()


def _content_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _valid_hash(value: Any, *, allow_empty: bool = False) -> bool:
    text = str(value or "")
    if allow_empty and not text:
        return True
    return (
        len(text) == _SHA256_LENGTH
        and text == text.lower()
        and all(character in "0123456789abcdef" for character in text)
    )


def _exact_fields(receipt: Mapping[str, Any], expected: set[str]) -> bool:
    return set(receipt) == {*expected, "receipt_id"}


def _valid_string_list(value: Any, *, unique: bool = False) -> bool:
    if not isinstance(value, list) or not all(
        isinstance(item, str) for item in value
    ):
        return False
    return not unique or len(value) == len(set(value))


def _valid_response_fragment_manifest(
    value: Any,
    *,
    allow_pending_question: bool,
) -> bool:
    if not isinstance(value, list):
        return False
    allowed_roles = {"message", "status", "evidence", "warning", "error"}
    if allow_pending_question:
        allowed_roles.add("pending_question")
    semantic_hashes: list[str] = []
    for fragment in value:
        if (
            not isinstance(fragment, Mapping)
            or set(fragment)
            != {"semantic_hash", "render_hash", "message_id", "role"}
            or not _valid_hash(fragment.get("semantic_hash"))
            or not _valid_hash(fragment.get("render_hash"))
            or not str(fragment.get("message_id") or "")
            or fragment.get("role") not in allowed_roles
        ):
            return False
        semantic_hashes.append(str(fragment["semantic_hash"]))
    return len(semantic_hashes) == len(set(semantic_hashes))


def _valid_receipt_identity(
    receipt: Mapping[str, Any],
    *,
    receipt_type: str,
    turn_index: int,
) -> tuple[bool, str]:
    if receipt.get("receipt_type") != receipt_type:
        return False, "coordinator receipt type is invalid"
    if (
        not isinstance(receipt.get("turn_index"), int)
        or isinstance(receipt.get("turn_index"), bool)
        or receipt.get("turn_index") != turn_index
    ):
        return False, "coordinator receipt belongs to a different turn"
    recorded = str(receipt.get("receipt_id") or "")
    unsigned = {
        str(key): value
        for key, value in receipt.items()
        if str(key) != "receipt_id"
    }
    if not _valid_hash(recorded) or recorded != _content_hash(unsigned):
        return False, "coordinator receipt identity is stale"
    return True, ""


def _validate_pending_resolution(
    receipt: Mapping[str, Any],
) -> tuple[bool, str]:
    fields = {
        "receipt_type",
        "turn_index",
        "pending_id",
        "pending_group",
        "pending_contract_hash",
        "resolution_path",
        "selected_option_id",
        "selected_value_hash",
        "resolved_action_id",
        "input_hash",
        "normalizer",
        "verdict",
    }
    if not _exact_fields(receipt, fields):
        return False, "pending-resolution receipt shape is invalid"
    if (
        not str(receipt.get("pending_id") or "")
        or not str(receipt.get("pending_group") or "")
        or receipt.get("resolution_path")
        not in {"exact_contract", "typed_manual_value"}
        or not str(receipt.get("normalizer") or "")
        or not str(receipt.get("resolved_action_id") or "")
        or receipt.get("verdict") != "accepted"
        or not _valid_hash(receipt.get("pending_contract_hash"))
        or not _valid_hash(receipt.get("selected_value_hash"))
        or not _valid_hash(receipt.get("input_hash"))
    ):
        return False, "pending-resolution receipt semantics are invalid"
    return True, ""


def _validate_execution_approval(
    receipt: Mapping[str, Any],
) -> tuple[bool, str]:
    fields = {
        "receipt_type",
        "turn_index",
        "approval_question_id",
        "pending_resolution_receipt_id",
        "answer_action_id",
        "approval_action_id",
        "approval_action_type",
        "execution_request_id",
        "side_effect_intent_id",
        "side_effect_intent_hash",
        "side_effect_receipt_id",
        "side_effect_receipt_hash",
        "idempotency_key_hash",
        "request_fingerprint",
        "job_submission_receipt_id",
        "job_submission_receipt_hash",
        "approved_plan_hash",
        "repository_revision",
        "plan_hash",
        "workflow_type",
        "target_mode",
        "job_id",
    }
    if not _exact_fields(receipt, fields):
        return False, "execution-approval receipt shape is invalid"
    revision = receipt.get("repository_revision")
    approval_contracts = {
        "preflight_smoke_confirm": "approve_preflight_smoke",
        "real_node_smoke_confirm": "approve_preflight_smoke",
        "real_node_final_benchmark_confirm": "approve_final_benchmark",
    }
    question_id = str(receipt.get("approval_question_id") or "")
    if (
        approval_contracts.get(question_id)
        != receipt.get("approval_action_type")
        or not _valid_hash(receipt.get("pending_resolution_receipt_id"))
        or not str(receipt.get("answer_action_id") or "")
        or not str(receipt.get("approval_action_id") or "")
        or receipt.get("answer_action_id") == receipt.get("approval_action_id")
        or not str(receipt.get("execution_request_id") or "")
        or not _valid_hash(receipt.get("side_effect_intent_id"))
        or not _valid_hash(receipt.get("side_effect_intent_hash"))
        or not _valid_hash(receipt.get("side_effect_receipt_id"))
        or not _valid_hash(receipt.get("side_effect_receipt_hash"))
        or not _valid_hash(receipt.get("idempotency_key_hash"))
        or not _valid_hash(receipt.get("request_fingerprint"))
        or not _valid_hash(receipt.get("job_submission_receipt_id"))
        or not _valid_hash(receipt.get("job_submission_receipt_hash"))
        or not _valid_hash(receipt.get("approved_plan_hash"))
        or not isinstance(revision, Mapping)
        or set(revision) != {"commit", "worktree_hash"}
        or not _valid_hash(revision.get("worktree_hash"))
        or len(str(revision.get("commit") or "")) != 40
        or not _valid_hash(receipt.get("plan_hash"))
        or receipt.get("workflow_type") not in {"rpc_benchmark", "sync_observe"}
        or receipt.get("target_mode")
        not in {"fake-node", "real-node", "sync-observe"}
        or not str(receipt.get("job_id") or "")
    ):
        return False, "execution-approval receipt semantics are invalid"
    return True, ""


def _validate_semantic_planner(
    receipt: Mapping[str, Any],
) -> tuple[bool, str]:
    fields = {
        "receipt_type",
        "turn_index",
        "input_hash",
        "pending_contract_hash",
        "resolver_invoked",
        "result_reason_hash",
        "planned_action_types",
        "semantic_units",
        "planner_metrics",
    }
    if not _exact_fields(receipt, fields):
        return False, "semantic-planner receipt shape is invalid"
    actions = receipt.get("planned_action_types")
    units = receipt.get("semantic_units")
    metrics = receipt.get("planner_metrics")
    if (
        receipt.get("resolver_invoked") is not True
        or not _valid_hash(receipt.get("input_hash"))
        or not _valid_hash(receipt.get("pending_contract_hash"))
        or not _valid_hash(receipt.get("result_reason_hash"))
        or not _valid_string_list(actions, unique=True)
        or not isinstance(units, list)
        or not isinstance(metrics, Mapping)
    ):
        return False, "semantic-planner receipt semantics are invalid"
    unit_ids: list[str] = []
    for unit in units:
        if (
            not isinstance(unit, Mapping)
            or set(unit) != {"unit_id", "disposition"}
            or not str(unit.get("unit_id") or "")
            or not str(unit.get("disposition") or "")
        ):
            return False, "semantic-planner unit is invalid"
        unit_ids.append(str(unit["unit_id"]))
    if len(unit_ids) != len(set(unit_ids)):
        return False, "semantic-planner units are duplicated"
    for key, value in metrics.items():
        if not isinstance(key, str) or not isinstance(value, (int, float, bool)):
            return False, "semantic-planner metric is invalid"
        if isinstance(value, float) and not math.isfinite(value):
            return False, "semantic-planner metric is not finite"
    return True, ""


def _valid_nonnegative_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _validate_semantic_partition(
    receipt: Mapping[str, Any],
) -> tuple[bool, str]:
    fields = {
        "receipt_type",
        "turn_index",
        "planning_lane",
        "status",
        "unit_count",
        "owner_count",
        "stage_a_calls",
        "errors_hash",
    }
    if not _exact_fields(receipt, fields):
        return False, "semantic-partition receipt shape is invalid"
    if (
        receipt.get("planning_lane")
        not in {"bounded_semantic_value", "hierarchical"}
        or receipt.get("status") not in {"compile_owner", "review_plan", "failed"}
        or not _valid_nonnegative_integer(receipt.get("unit_count"))
        or not _valid_nonnegative_integer(receipt.get("owner_count"))
        or not _valid_nonnegative_integer(receipt.get("stage_a_calls"))
        or not _valid_hash(receipt.get("errors_hash"))
    ):
        return False, "semantic-partition receipt semantics are invalid"
    if (
        receipt.get("status") == "compile_owner"
        and int(receipt["owner_count"]) == 0
    ):
        return False, "semantic-partition compile schedule has no owner"
    return True, ""


def _validate_owner_compilation(
    receipt: Mapping[str, Any],
) -> tuple[bool, str]:
    fields = {
        "receipt_type",
        "turn_index",
        "owner",
        "cursor_before",
        "cursor_after",
        "status",
        "document_hash",
        "errors_hash",
    }
    if not _exact_fields(receipt, fields):
        return False, "owner-compilation receipt shape is invalid"
    status = receipt.get("status")
    cursor_before = receipt.get("cursor_before")
    cursor_after = receipt.get("cursor_after")
    document_hash = receipt.get("document_hash")
    errors_hash = receipt.get("errors_hash")
    if (
        not str(receipt.get("owner") or "")
        or not _valid_nonnegative_integer(cursor_before)
        or not _valid_nonnegative_integer(cursor_after)
        or status not in {"compile_owner", "review_plan", "failed"}
        or not _valid_hash(document_hash)
        or not _valid_hash(errors_hash)
    ):
        return False, "owner-compilation receipt semantics are invalid"
    if status == "failed":
        if (
            int(cursor_after) != int(cursor_before)
            or document_hash != _EMPTY_DOCUMENT_HASH
            or errors_hash == _EMPTY_ERRORS_HASH
        ):
            return False, "failed owner-compilation receipt semantics are invalid"
    elif (
        int(cursor_after) != int(cursor_before) + 1
        or document_hash == _EMPTY_DOCUMENT_HASH
        or errors_hash != _EMPTY_ERRORS_HASH
    ):
        return False, "successful owner-compilation receipt semantics are invalid"
    return True, ""


def _validate_whole_plan_review(
    receipt: Mapping[str, Any],
) -> tuple[bool, str]:
    fields = {
        "receipt_type",
        "turn_index",
        "status",
        "result_hash",
        "admission_calls",
    }
    if not _exact_fields(receipt, fields):
        return False, "whole-plan-review receipt shape is invalid"
    if (
        receipt.get("status") != "reviewed"
        or not _valid_hash(receipt.get("result_hash"))
        or not _valid_nonnegative_integer(receipt.get("admission_calls"))
    ):
        return False, "whole-plan-review receipt semantics are invalid"
    return True, ""


def _validate_fallback_selection(
    receipt: Mapping[str, Any],
) -> tuple[bool, str]:
    fields = {
        "receipt_type",
        "turn_index",
        "selected_group",
        "reason_hash",
        "group_readiness",
        "pending_after_id",
    }
    if not _exact_fields(receipt, fields):
        return False, "fallback-selection receipt shape is invalid"
    readiness = receipt.get("group_readiness")
    if not _valid_hash(receipt.get("reason_hash")) or not isinstance(
        readiness, Mapping
    ):
        return False, "fallback-selection receipt semantics are invalid"
    for group, fact in readiness.items():
        if (
            not isinstance(group, str)
            or not group
            or not isinstance(fact, Mapping)
            or set(fact) != {"ready", "continuation", "reason_hash"}
            or not isinstance(fact.get("ready"), bool)
            or not isinstance(fact.get("continuation"), bool)
            or not _valid_hash(fact.get("reason_hash"))
        ):
            return False, "fallback-selection readiness is invalid"
    selected = str(receipt.get("selected_group") or "")
    if selected and selected not in readiness:
        return False, "fallback-selection group is not in the readiness projection"
    return True, ""


def _validate_response_composition(
    receipt: Mapping[str, Any],
) -> tuple[bool, str]:
    fields = {
        "receipt_type",
        "turn_index",
        "language",
        "active_group",
        "source_action_ids",
        "pending_contract_hash",
        "fragments",
    }
    if not _exact_fields(receipt, fields):
        return False, "response-composition receipt shape is invalid"
    fragments = receipt.get("fragments")
    if (
        not str(receipt.get("language") or "")
        or not _valid_string_list(receipt.get("source_action_ids"), unique=True)
        or not _valid_hash(receipt.get("pending_contract_hash"))
        or not _valid_response_fragment_manifest(
            fragments,
            allow_pending_question=True,
        )
    ):
        return False, "response-composition receipt semantics are invalid"
    return True, ""


def _validate_domain_commit(
    receipt: Mapping[str, Any],
) -> tuple[bool, str]:
    common = {
        "receipt_type",
        "turn_index",
        "owner",
        "completion",
        "group_registry_contract_hash",
        "pending_before_hash",
        "pending_after_hash",
        "consumed_action_ids",
        "invalidated_groups",
        "invalidated_fields",
        "response_fragments",
    }
    rejected = {*common, "blocker_semantic_hash"}
    committed = {
        *common,
        "reconfigured_groups",
        "group_state_transitions",
        "material_delta",
        "navigation_operation",
        "navigation_origin_group",
        "navigation_target_group",
        "pending_after_id",
    }
    completion = str(receipt.get("completion") or "")
    expected = rejected if completion == "rejected" else committed
    if not _exact_fields(receipt, expected):
        return False, "domain-commit receipt shape is invalid"
    if (
        not str(receipt.get("owner") or "")
        or completion
        not in {"rejected", "unchanged", "in_progress", "completed", "blocked"}
        or not _valid_hash(receipt.get("group_registry_contract_hash"))
        or not _valid_hash(receipt.get("pending_before_hash"))
        or not _valid_hash(receipt.get("pending_after_hash"))
    ):
        return False, "domain-commit receipt semantics are invalid"
    for field in (
        "consumed_action_ids",
        "invalidated_groups",
        "invalidated_fields",
    ):
        if not _valid_string_list(receipt.get(field), unique=True):
            return False, f"domain-commit {field} is invalid"
    if not _valid_response_fragment_manifest(
        receipt.get("response_fragments"),
        allow_pending_question=False,
    ):
        return False, "domain-commit response fragments are invalid"
    if completion == "rejected":
        if (
            not _valid_hash(receipt.get("blocker_semantic_hash"))
            or receipt.get("consumed_action_ids")
            or receipt.get("invalidated_groups")
            or receipt.get("invalidated_fields")
        ):
            return False, "rejected domain-commit receipt is contradictory"
        return True, ""
    if not _valid_string_list(receipt.get("reconfigured_groups"), unique=True):
        return False, "domain-commit reconfigured groups are invalid"
    group_state_transitions = receipt.get("group_state_transitions")
    if not isinstance(group_state_transitions, list):
        return False, "domain-commit group-state transitions are invalid"
    seen_groups: set[str] = set()
    allowed_group_states = {
        "",
        "in_progress",
        "reconfiguring",
        "completed",
        "invalidated",
    }
    for item in group_state_transitions:
        if (
            not isinstance(item, Mapping)
            or set(item) != {"group", "before", "after"}
            or not str(item.get("group") or "")
            or str(item.get("before") or "") not in allowed_group_states
            or str(item.get("after") or "") not in allowed_group_states
            or str(item.get("before") or "") == str(item.get("after") or "")
            or str(item["group"]) in seen_groups
        ):
            return False, "domain-commit group-state transition is invalid"
        seen_groups.add(str(item["group"]))
    navigation_operation = str(receipt.get("navigation_operation") or "")
    navigation_origin = str(receipt.get("navigation_origin_group") or "")
    navigation_target = str(receipt.get("navigation_target_group") or "")
    if navigation_operation:
        if (
            navigation_operation not in {"change_group", "go_back"}
            or not navigation_origin
            or not navigation_target
        ):
            return False, "domain-commit navigation binding is invalid"
    elif navigation_origin or navigation_target:
        return False, "domain-commit navigation fields are contradictory"
    delta = receipt.get("material_delta")
    if not isinstance(delta, list):
        return False, "domain-commit material delta is invalid"
    seen_paths: set[tuple[str, str]] = set()
    for item in delta:
        if (
            not isinstance(item, Mapping)
            or set(item) != {"operation", "path", "value_hash"}
            or item.get("operation") not in {"write", "delete"}
            or not str(item.get("path") or "")
        ):
            return False, "domain-commit material delta item is invalid"
        operation = str(item["operation"])
        value_hash = str(item.get("value_hash") or "")
        if (
            operation == "write"
            and not _valid_hash(value_hash)
            or operation == "delete"
            and value_hash
        ):
            return False, "domain-commit material delta hash is invalid"
        identity = (operation, str(item["path"]))
        if identity in seen_paths:
            return False, "domain-commit material delta is duplicated"
        seen_paths.add(identity)
    return True, ""


def validate_coordinator_control_receipt(
    receipt: Mapping[str, Any],
    *,
    turn_index: int,
) -> tuple[bool, str]:
    """Validate one coordinator-owned receipt without trusting its hash."""

    receipt_type = str(receipt.get("receipt_type") or "")
    if receipt_type not in _COORDINATOR_RECEIPT_TYPES:
        return False, "unregistered coordinator receipt type"
    valid, reason = _valid_receipt_identity(
        receipt,
        receipt_type=receipt_type,
        turn_index=turn_index,
    )
    if not valid:
        return valid, reason
    validators = {
        "pending_resolution": _validate_pending_resolution,
        "semantic_partition": _validate_semantic_partition,
        "owner_compilation": _validate_owner_compilation,
        "whole_plan_review": _validate_whole_plan_review,
        "semantic_planner": _validate_semantic_planner,
        "fallback_selection": _validate_fallback_selection,
        "response_composition": _validate_response_composition,
        "domain_commit": _validate_domain_commit,
        "execution_approval": _validate_execution_approval,
    }
    return validators[receipt_type](receipt)


def validate_domain_control_receipt(
    receipt: Mapping[str, Any],
    *,
    handler_owner: str,
    turn_index: int,
) -> tuple[bool, str]:
    """Validate identity, semantic schema, and handler ownership."""

    receipt_type = str(receipt.get("receipt_type") or "")
    if receipt_type in _COORDINATOR_RECEIPT_TYPES:
        return False, "coordinator receipt crossed a domain handler boundary"
    expected_handler = _HANDLER_OWNER_BY_RECEIPT.get(receipt_type)
    if expected_handler is None:
        return False, "unregistered domain receipt type"
    if handler_owner != expected_handler:
        return False, "domain receipt crossed the wrong handler boundary"
    if receipt.get("turn_index") != turn_index:
        return False, "domain receipt belongs to a different turn"
    if receipt_type.startswith("analysis_"):
        return validate_analysis_receipt(receipt)
    if receipt_type == "chain_identity_resolution":
        return validate_chain_identity_receipt(receipt)
    if receipt_type == "orientation_response":
        return validate_orientation_receipt(receipt)
    return validate_rpc_receipt(receipt)


def validate_persisted_domain_control_receipt(
    receipt: Mapping[str, Any],
    *,
    turn_index: int,
) -> tuple[bool, str]:
    """Validate a persisted owner receipt without trusting event metadata."""

    receipt_type = str(receipt.get("receipt_type") or "")
    if receipt_type in _COORDINATOR_RECEIPT_TYPES:
        return validate_coordinator_control_receipt(
            receipt,
            turn_index=turn_index,
        )
    handler_owner = _HANDLER_OWNER_BY_RECEIPT.get(receipt_type)
    if handler_owner is None:
        return False, "unregistered persisted control receipt type"
    return validate_domain_control_receipt(
        receipt,
        handler_owner=handler_owner,
        turn_index=turn_index,
    )

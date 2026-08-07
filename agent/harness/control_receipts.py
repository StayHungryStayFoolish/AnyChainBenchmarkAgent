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
    "rpc_method_probe": "chain_rpc",
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
EXECUTION_APPROVAL_CONTRACTS = {
    "preflight_smoke_confirm": "approve_preflight_smoke",
    "real_node_smoke_confirm": "approve_preflight_smoke",
    "real_node_final_benchmark_confirm": "approve_final_benchmark",
}
SEMANTIC_PARTITION_PLANNING_LANES = frozenset({
    "bounded_semantic_value",
    "hierarchical",
    "semantic_draft_finalization",
})


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


def execution_intent_projection(intent: Mapping[str, Any]) -> dict[str, Any]:
    """Project execution intent lineage without retaining request payloads."""

    return {
        "intent_id": str(intent.get("intent_id") or ""),
        "turn_id": str(intent.get("turn_id") or ""),
        "action_id": str(intent.get("action_id") or ""),
        "operation": str(intent.get("operation") or ""),
        "execution_request_id": str(
            intent.get("execution_request_id") or ""
        ),
        "idempotency_key_hash": _content_hash(
            str(intent.get("idempotency_key") or "")
        ),
        "request_fingerprint": str(
            intent.get("request_fingerprint") or ""
        ),
        "expected_receipt_kind": str(
            intent.get("expected_receipt_kind") or ""
        ),
        "status": str(intent.get("status") or ""),
        "attempt_count": int(intent.get("attempt_count") or 0),
    }


def execution_side_effect_projection(
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Project side-effect receipt lineage without retaining handler results."""

    return {
        "receipt_id": str(receipt.get("receipt_id") or ""),
        "intent_id": str(receipt.get("intent_id") or ""),
        "action_id": str(receipt.get("action_id") or ""),
        "status": str(receipt.get("status") or ""),
        "idempotency_key_hash": _content_hash(
            str(receipt.get("idempotency_key") or "")
        ),
        "job_id": str(receipt.get("job_id") or ""),
        "failure_code": str(receipt.get("failure_code") or ""),
        "retryable": bool(receipt.get("retryable")),
        "result_hash": _content_hash(receipt.get("result") or {}),
    }


def execution_side_effect_receipt_id(receipt: Mapping[str, Any]) -> str:
    """Return the content identity of an observed external side effect."""

    projection = execution_side_effect_projection(receipt)
    projection.pop("receipt_id", None)
    return _content_hash(projection)


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
    question_id = str(receipt.get("approval_question_id") or "")
    if (
        EXECUTION_APPROVAL_CONTRACTS.get(question_id)
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
        receipt.get("planning_lane") not in SEMANTIC_PARTITION_PLANNING_LANES
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
        "authority_chain",
    }
    if not _exact_fields(receipt, fields):
        return False, "whole-plan-review receipt shape is invalid"
    if (
        receipt.get("status") != "reviewed"
        or not _valid_hash(receipt.get("result_hash"))
        or not _valid_nonnegative_integer(receipt.get("admission_calls"))
        or not _valid_planner_authority_chain(receipt.get("authority_chain"))
    ):
        return False, "whole-plan-review receipt semantics are invalid"
    return True, ""


def _valid_request_sizes(value: Any) -> bool:
    return (
        isinstance(value, list)
        and all(
            _valid_nonnegative_integer(item) and int(item) > 0
            for item in value
        )
    )


def _valid_planner_authority_chain(value: Any) -> bool:
    if not isinstance(value, Mapping) or set(value) != {
        "stage_a_convergence",
        "stage_a_relation_reviews",
        "pending_entailment_reviews",
        "stage_b_semantic_reviews",
    }:
        return False
    stage_a = value.get("stage_a_convergence")
    if not isinstance(stage_a, Mapping):
        return False
    if stage_a:
        if set(stage_a) != {
            "primary_hash",
            "primary_eligible",
            "secondary_hash",
            "secondary_eligible",
            "selected_proposal",
            "selection_authority",
            "verdict_hash",
            "request_count",
            "request_sizes",
            "valid",
        }:
            return False
        if (
            not _valid_hash(stage_a.get("primary_hash"))
            or not _valid_hash(stage_a.get("secondary_hash"))
            or not isinstance(stage_a.get("primary_eligible"), bool)
            or not isinstance(stage_a.get("secondary_eligible"), bool)
            or not _valid_hash(stage_a.get("verdict_hash"))
            or stage_a.get("selected_proposal")
            not in {"primary", "secondary", "none", "rejected"}
            or stage_a.get("selection_authority")
            not in {"harness_eligibility", "model_convergence"}
            or not _valid_nonnegative_integer(stage_a.get("request_count"))
            or not _valid_request_sizes(stage_a.get("request_sizes"))
            or int(stage_a["request_count"]) != len(stage_a["request_sizes"])
            or (
                stage_a.get("selection_authority") == "model_convergence"
                and int(stage_a["request_count"]) == 0
            )
            or (
                stage_a.get("selection_authority") == "harness_eligibility"
                and int(stage_a["request_count"]) != 0
            )
            or (
                stage_a.get("selection_authority") == "harness_eligibility"
                and (
                    stage_a["primary_eligible"]
                    == stage_a["secondary_eligible"]
                    or stage_a.get("selected_proposal")
                    != (
                        "primary"
                        if stage_a["primary_eligible"]
                        else "secondary"
                    )
                    or stage_a.get("valid") is not True
                )
            )
            or (
                stage_a.get("selection_authority") == "model_convergence"
                and stage_a["primary_eligible"]
                != stage_a["secondary_eligible"]
            )
            or not isinstance(stage_a.get("valid"), bool)
            or (
                stage_a.get("valid") is True
                and stage_a.get("selected_proposal")
                not in {"primary", "secondary"}
            )
        ):
            return False
    relation_reviews = value.get("stage_a_relation_reviews")
    if not isinstance(relation_reviews, list):
        return False
    for review in relation_reviews:
        if not isinstance(review, Mapping) or set(review) != {
            "proposal_hash",
            "candidate_unit_ids",
            "possible_support_unit_ids",
            "member_response_hashes",
            "member_validity",
            "request_count",
            "request_sizes",
            "decisions",
            "valid",
        }:
            return False
        candidate_ids = review.get("candidate_unit_ids")
        support_ids = review.get("possible_support_unit_ids")
        decisions = review.get("decisions")
        if (
            not _valid_hash(review.get("proposal_hash"))
            or not isinstance(candidate_ids, list)
            or not candidate_ids
            or not all(isinstance(item, str) and item for item in candidate_ids)
            or len(candidate_ids) != len(set(candidate_ids))
            or not isinstance(support_ids, Mapping)
            or set(support_ids) != set(candidate_ids)
            or any(
                not isinstance(values, list)
                or not values
                or not all(isinstance(item, str) and item for item in values)
                or len(values) != len(set(values))
                for values in support_ids.values()
            )
            or not isinstance(review.get("member_response_hashes"), list)
            or len(review["member_response_hashes"]) != 3
            or not all(
                _valid_hash(item)
                for item in review["member_response_hashes"]
            )
            or not isinstance(review.get("member_validity"), list)
            or len(review["member_validity"]) != 3
            or not all(isinstance(item, bool) for item in review["member_validity"])
            or review.get("request_count") != 3
            or not _valid_request_sizes(review.get("request_sizes"))
            or len(review["request_sizes"]) != 3
            or not isinstance(decisions, list)
            or len(decisions) != len(candidate_ids)
            or not isinstance(review.get("valid"), bool)
        ):
            return False
        for index, decision in enumerate(decisions):
            if not isinstance(decision, Mapping) or set(decision) != {
                "unit_id", "relation", "supports_unit_id", "quorum_reached"
            }:
                return False
            unit_id = str(decision.get("unit_id") or "")
            relation = str(decision.get("relation") or "")
            supports_unit_id = str(decision.get("supports_unit_id") or "")
            if (
                unit_id != candidate_ids[index]
                or relation not in {
                    "named_identity", "supports_unit", "unresolved"
                }
                or (
                    relation == "supports_unit"
                    and supports_unit_id not in support_ids[unit_id]
                )
                or (relation != "supports_unit" and supports_unit_id)
                or not isinstance(decision.get("quorum_reached"), bool)
            ):
                return False
        if review["valid"] is not all(
            decision["quorum_reached"] is True
            for decision in decisions
        ):
            return False
    pending_reviews = value.get("pending_entailment_reviews")
    if not isinstance(pending_reviews, list):
        return False
    member_fields = {
        "member_index",
        "response_hash",
        "response_json_valid",
        "response_shape_valid",
        "verdict",
        "selected_value_present",
        "selected_identity_hash",
        "evidence_quote_hash",
        "reason_hash",
        "evidence_source_bound",
        "selected_value_evidence_bound",
        "pending_contract_valid",
        "accepted_vote",
        "rejection_code",
    }
    rejection_codes = {
        "accepted",
        "invalid_json",
        "invalid_shape",
        "invalid_verdict",
        "missing_evidence_quote",
        "evidence_not_source_bound",
        "missing_reason",
        "non_answer_selected_value",
        "semantic_non_answer",
        "missing_selected_identity",
        "researched_identity_contract_rejected",
        "selected_value_not_evidence_bound",
        "manual_identity_mismatch",
    }
    for review in pending_reviews:
        if not isinstance(review, Mapping) or set(review) != {
            "proposal_hash",
            "claim_hash",
            "unit_id",
            "pending_contract_hash",
            "source_hash",
            "request_count",
            "request_sizes",
            "members",
            "identity_vote_counts",
            "quorum_identity_hash",
            "quorum_reached",
        }:
            return False
        members = review.get("members")
        vote_counts = review.get("identity_vote_counts")
        if (
            not all(
                _valid_hash(review.get(field))
                for field in (
                    "proposal_hash",
                    "claim_hash",
                    "pending_contract_hash",
                    "source_hash",
                )
            )
            or not str(review.get("unit_id") or "")
            or review.get("request_count") != 3
            or not _valid_request_sizes(review.get("request_sizes"))
            or len(review["request_sizes"]) != 3
            or not isinstance(members, list)
            or len(members) != 3
            or not isinstance(vote_counts, list)
            or not isinstance(review.get("quorum_reached"), bool)
            or not _valid_hash(
                review.get("quorum_identity_hash"),
                allow_empty=True,
            )
        ):
            return False
        accepted_identity_hashes: list[str] = []
        for index, member in enumerate(members, start=1):
            if not isinstance(member, Mapping) or set(member) != member_fields:
                return False
            boolean_fields = {
                "response_json_valid",
                "response_shape_valid",
                "selected_value_present",
                "evidence_source_bound",
                "selected_value_evidence_bound",
                "pending_contract_valid",
                "accepted_vote",
            }
            if (
                member.get("member_index") != index
                or not _valid_hash(member.get("response_hash"))
                or any(
                    not isinstance(member.get(field), bool)
                    for field in boolean_fields
                )
                or str(member.get("verdict") or "")
                not in {"", "answers", "different_request", "uncertain"}
                or not _valid_hash(
                    member.get("selected_identity_hash"), allow_empty=True
                )
                or not _valid_hash(
                    member.get("evidence_quote_hash"), allow_empty=True
                )
                or not _valid_hash(member.get("reason_hash"), allow_empty=True)
                or member.get("rejection_code") not in rejection_codes
            ):
                return False
            accepted = member["accepted_vote"] is True
            rejection_code = str(member["rejection_code"])
            if not member["response_json_valid"]:
                compatible_rejection_codes = {"invalid_json"}
            elif not member["response_shape_valid"]:
                compatible_rejection_codes = {"invalid_shape"}
            elif not member["verdict"]:
                compatible_rejection_codes = {"invalid_verdict"}
            elif not member["evidence_quote_hash"]:
                compatible_rejection_codes = {"missing_evidence_quote"}
            elif not member["evidence_source_bound"]:
                compatible_rejection_codes = {"evidence_not_source_bound"}
            elif not member["reason_hash"]:
                compatible_rejection_codes = {"missing_reason"}
            elif member["verdict"] != "answers":
                compatible_rejection_codes = {
                    "non_answer_selected_value"
                    if member["selected_value_present"]
                    else "semantic_non_answer"
                }
            elif not member["selected_identity_hash"]:
                compatible_rejection_codes = {"missing_selected_identity"}
            elif not member["pending_contract_valid"]:
                compatible_rejection_codes = {
                    "researched_identity_contract_rejected",
                    "manual_identity_mismatch",
                }
            elif not member["selected_value_evidence_bound"]:
                compatible_rejection_codes = {
                    "selected_value_not_evidence_bound",
                    "manual_identity_mismatch",
                }
            else:
                compatible_rejection_codes = {"accepted"}
            if rejection_code not in compatible_rejection_codes:
                return False
            if accepted:
                if (
                    rejection_code != "accepted"
                    or member.get("verdict") != "answers"
                    or not member.get("response_json_valid")
                    or not member.get("response_shape_valid")
                    or not member.get("evidence_source_bound")
                    or not member.get("selected_value_evidence_bound")
                    or not member.get("pending_contract_valid")
                    or not _valid_hash(member.get("selected_identity_hash"))
                ):
                    return False
                accepted_identity_hashes.append(
                    str(member["selected_identity_hash"])
                )
            elif rejection_code == "accepted":
                return False
        normalized_counts: dict[str, int] = {}
        for row in vote_counts:
            if (
                not isinstance(row, Mapping)
                or set(row) != {"identity_hash", "count"}
                or not _valid_hash(row.get("identity_hash"))
                or not _valid_nonnegative_integer(row.get("count"))
                or int(row["count"]) <= 0
                or int(row["count"]) > 3
                or str(row["identity_hash"]) in normalized_counts
            ):
                return False
            normalized_counts[str(row["identity_hash"])] = int(row["count"])
        observed_counts = {
            identity_hash: accepted_identity_hashes.count(identity_hash)
            for identity_hash in sorted(set(accepted_identity_hashes))
        }
        quorum_hashes = sorted(
            identity_hash
            for identity_hash, count in observed_counts.items()
            if count >= 2
        )
        quorum_reached = len(quorum_hashes) == 1
        if (
            normalized_counts != observed_counts
            or review["quorum_reached"] is not quorum_reached
            or str(review.get("quorum_identity_hash") or "")
            != (quorum_hashes[0] if quorum_reached else "")
        ):
            return False
    reviews = value.get("stage_b_semantic_reviews")
    if not isinstance(reviews, list):
        return False
    for review in reviews:
        if not isinstance(review, Mapping) or set(review) != {
            "owner",
            "proposal_hash",
            "review_hashes",
            "member_validity",
            "request_count",
            "request_sizes",
            "valid",
        }:
            return False
        if (
            not str(review.get("owner") or "")
            or not _valid_hash(review.get("proposal_hash"))
            or not isinstance(review.get("review_hashes"), list)
            or not all(_valid_hash(value) for value in review["review_hashes"])
            or not isinstance(review.get("member_validity"), list)
            or not all(isinstance(value, bool) for value in review["member_validity"])
            or not _valid_nonnegative_integer(review.get("request_count"))
            or not _valid_request_sizes(review.get("request_sizes"))
            or int(review["request_count"]) != len(review["request_sizes"])
            or int(review["request_count"]) != 3
            or len(review["review_hashes"]) != 3
            or len(review["member_validity"]) != 3
            or not isinstance(review.get("valid"), bool)
            or review["valid"] != (sum(review["member_validity"]) >= 2)
        ):
            return False
    return True


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
        "terminal_response_hash",
        "terminal_semantic_hash",
    }
    if not _exact_fields(receipt, fields):
        return False, "response-composition receipt shape is invalid"
    fragments = receipt.get("fragments")
    if (
        not str(receipt.get("language") or "")
        or not _valid_string_list(receipt.get("source_action_ids"), unique=True)
        or not _valid_hash(receipt.get("pending_contract_hash"))
        or not _valid_hash(receipt.get("terminal_response_hash"))
        or not _valid_hash(receipt.get("terminal_semantic_hash"))
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
        "cause_kind",
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
        or receipt.get("cause_kind")
        not in {"admitted_action", "system_reconcile", "validation_rejection"}
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
    if len(receipt.get("consumed_action_ids") or ()) > 1:
        return False, "domain-commit has more than one causal action"
    if not _valid_response_fragment_manifest(
        receipt.get("response_fragments"),
        allow_pending_question=False,
    ):
        return False, "domain-commit response fragments are invalid"
    if completion == "rejected":
        if (
            receipt.get("cause_kind") != "validation_rejection"
            or not _valid_hash(receipt.get("blocker_semantic_hash"))
            or receipt.get("consumed_action_ids")
            or receipt.get("invalidated_groups")
            or receipt.get("invalidated_fields")
        ):
            return False, "rejected domain-commit receipt is contradictory"
        return True, ""
    consumed_action_ids = receipt.get("consumed_action_ids") or []
    cause_kind = receipt.get("cause_kind")
    if (
        cause_kind == "admitted_action"
        and len(consumed_action_ids) != 1
        or cause_kind == "system_reconcile"
        and consumed_action_ids
        or cause_kind == "validation_rejection"
    ):
        return False, "domain-commit cause does not match its causal action"
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
    if cause_kind == "system_reconcile" and (
        delta
        or receipt.get("invalidated_groups")
        or receipt.get("invalidated_fields")
        or receipt.get("reconfigured_groups")
        or group_state_transitions
        or navigation_operation
    ):
        return False, "system reconcile cannot mutate business workflow state"
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
